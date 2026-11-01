from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from cdrc.common.metric.base import BaseMetric, MetricResult


def _normalize_code(value: Any) -> Optional[str]:
    """Best-effort extraction and normalization of a code string."""
    if value is None:
        return None
    if hasattr(value, "code"):
        value = getattr(value, "code")
    elif isinstance(value, dict):
        value = value.get("code")
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _extract_codes_from_output(obj: Any) -> Set[str]:
    """Extract a set of unique procedure codes from an OutputSchema-like object.

    Handles pydantic objects or plain dicts. Returns upper-cased codes for stability.
    """
    # Access procedures list
    if hasattr(obj, "procedure"):
        items = getattr(obj, "procedure") or []
    elif isinstance(obj, dict):
        items = obj.get("procedure") or []
    else:
        items = []

    codes: Set[str] = set()
    for it in items:
        code = _normalize_code(it)
        if code:
            codes.add(code.upper())
    return codes


class PrecRecF1Metric(BaseMetric):
    """Compute precision, recall, and F1 at the HCPCS code level.

    - Instance-level: set comparison between predicted and gold codes per example.
    - Aggregated:
        - Per-code metrics using union label space across the dataset
        - Micro averages (global TP/FP/FN across all codes and examples)
        - Macro averages (mean of per-code precision/recall/F1)
    """

    def __init__(self, codes_to_track: Optional[List[str]] = None) -> None:
        super().__init__(supervised=True)
        # If provided, these codes will always appear in the aggregated output columns
        # even if absent in data (useful for dashboards/deep dives). Defaults to ["G2211"].
        default_codes = ["G2211"] if codes_to_track is None else codes_to_track
        self.codes_to_track: Set[str] = {c.strip().upper() for c in default_codes if str(c).strip()}

    def calculate(self, outputs: List[Any], gold: List[Any]) -> MetricResult:
        # Pre-extract codes and customer names per instance
        pred_codes_per_example: List[Set[str]] = []
        gold_codes_per_example: List[Set[str]] = []
        customers: List[str] = []

        for i in range(len(outputs)):
            pred_codes = _extract_codes_from_output(outputs[i])
            gold_codes = _extract_codes_from_output(gold[i])
            pred_codes_per_example.append(pred_codes)
            gold_codes_per_example.append(gold_codes)

            # Extract customer name from outputs.additional_output.customer_name
            customer_name = None
            out_obj = outputs[i]
            if hasattr(out_obj, "additional_output") and isinstance(getattr(out_obj, "additional_output"), dict):
                customer_name = out_obj.additional_output.get("customer_name")
            elif isinstance(out_obj, dict):
                customer_name = (out_obj.get("additional_output") or {}).get("customer_name")
            customers.append(str(customer_name) if customer_name else "overall")

        # Build instance-level dataframe with tp, fp, fn per instance
        instance_data = []
        for i in range(len(outputs)):
            pred_codes = pred_codes_per_example[i]
            gold_codes = gold_codes_per_example[i]
            tp = len(pred_codes & gold_codes)
            fp = len(pred_codes - gold_codes)
            fn = len(gold_codes - pred_codes)

            instance_data.append(
                {
                    "customer": customers[i],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_codes": sorted(list(pred_codes)),
                    "gold_codes": sorted(list(gold_codes)),
                    "pred_count": len(pred_codes),
                    "gold_count": len(gold_codes),
                }
            )

        df_instances = pd.DataFrame(instance_data)

        # Compute precision, recall, f1 for each instance
        df_instances["precision"] = df_instances.apply(
            lambda row: row["tp"] / (row["tp"] + row["fp"]) if (row["tp"] + row["fp"]) > 0 else 0.0, axis=1
        )
        df_instances["recall"] = df_instances.apply(lambda row: row["tp"] / (row["tp"] + row["fn"]) if (row["tp"] + row["fn"]) > 0 else 0.0, axis=1)
        df_instances["f1"] = df_instances.apply(
            lambda row: (
                (2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])) if (row["precision"] + row["recall"]) > 0 else 0.0
            ),
            axis=1,
        )

        # Instance metrics as list of dicts
        instance_metrics = df_instances.to_dict(orient="records")

        # Collect all codes across dataset and ensure tracked codes are included
        all_codes: Set[str] = set(self.codes_to_track)
        for pred_codes in pred_codes_per_example:
            all_codes.update(pred_codes)
        for gold_codes in gold_codes_per_example:
            all_codes.update(gold_codes)

        # Helper function to compute metrics for a group
        def compute_metrics_for_group(group_df: pd.DataFrame, group_indices: List[int]) -> Dict[str, Any]:
            # Micro metrics (sum tp, fp, fn across instances)
            tp_sum = float(group_df["tp"].sum())
            fp_sum = float(group_df["fp"].sum())
            fn_sum = float(group_df["fn"].sum())

            micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
            micro_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
            micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

            # Per-code metrics for this group
            group_codes: Set[str] = set(self.codes_to_track)
            for i in group_indices:
                group_codes.update(pred_codes_per_example[i])
                group_codes.update(gold_codes_per_example[i])

            per_code_stats = {}
            for code in sorted(group_codes):
                tp_c = sum(1 for i in group_indices if code in pred_codes_per_example[i] and code in gold_codes_per_example[i])
                fp_c = sum(1 for i in group_indices if code in pred_codes_per_example[i] and code not in gold_codes_per_example[i])
                fn_c = sum(1 for i in group_indices if code not in pred_codes_per_example[i] and code in gold_codes_per_example[i])

                prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0

                per_code_stats[code] = {
                    "precision": prec_c,
                    "recall": rec_c,
                    "f1": f1_c,
                    "tp": float(tp_c),
                    "fp": float(fp_c),
                    "fn": float(fn_c),
                }

            # Macro metrics (average over codes)
            if per_code_stats:
                macro_precision = sum(s["precision"] for s in per_code_stats.values()) / len(per_code_stats)
                macro_recall = sum(s["recall"] for s in per_code_stats.values()) / len(per_code_stats)
                macro_f1 = sum(s["f1"] for s in per_code_stats.values()) / len(per_code_stats)
            else:
                macro_precision = macro_recall = macro_f1 = 0.0

            # Flatten per-code stats for CSV
            flattened = {}
            for code in sorted(per_code_stats.keys()):
                stats = per_code_stats[code]
                key_prefix = f"code_{code}"
                flattened[f"{key_prefix}_precision"] = stats["precision"]
                flattened[f"{key_prefix}_recall"] = stats["recall"]
                flattened[f"{key_prefix}_f1"] = stats["f1"]
                flattened[f"{key_prefix}_tp"] = stats["tp"]
                flattened[f"{key_prefix}_fp"] = stats["fp"]
                flattened[f"{key_prefix}_fn"] = stats["fn"]

            return {
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "micro_tp": tp_sum,
                "micro_fp": fp_sum,
                "micro_fn": fn_sum,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "num_examples": len(group_indices),
                "label_space_size": len(per_code_stats),
                **flattened,
            }

        # Compute overall metrics
        overall_metrics = compute_metrics_for_group(df_instances, list(range(len(outputs))))
        overall_metrics = {"type": "overall", **overall_metrics}

        # Compute per-customer metrics using groupby
        aggregated_rows = [overall_metrics]

        unique_customers = sorted([c for c in df_instances["customer"].unique() if c != "overall"])
        for customer in unique_customers:
            customer_indices = df_instances[df_instances["customer"] == customer].index.tolist()
            customer_df = df_instances.iloc[customer_indices]
            customer_metrics = compute_metrics_for_group(customer_df, customer_indices)
            customer_metrics = {"type": customer, **customer_metrics}
            aggregated_rows.append(customer_metrics)

        aggregated_metric = pd.DataFrame(aggregated_rows)

        result: MetricResult = {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }
        return result
