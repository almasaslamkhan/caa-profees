from typing import Any

import numpy as np

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import binarize, get_nested_attribute, get_nested_attribute_list
from cdrc.profees_coding.modifiers.models.utils import get_modifiers_in_scope


class ModifierMetric(BaseMetric):
    """
    Per-modifier metrics (ignores CPT pairing).
    - Labels are just modifier codes (e.g., '24', '25', ...).
    - instance_metrics are computed over the SAME global label space as aggregated_metric.
    """

    def __init__(self, label_key: str, scope):
        """
        Args:
            label_key (str): Dotted path to the modifier codes list in each example,
                             e.g., 'em.modifiers.code'.
        """
        super().__init__(supervised=True)
        self.label_key = label_key
        self.scope = get_modifiers_in_scope(scope)

    @staticmethod
    def _norm_code(x) -> str | None:
        if x is None:
            return None
        s = str(x).strip()
        return s.upper() if s else None

    def _mods_from_example(self, example: dict) -> list[str]:
        """
        Extract and normalize modifier codes from an example using get_nested_attribute.
        Handles both list[str] and list[dict] with dict['code'].
        """
        raw = get_nested_attribute(example, self.label_key) or []
        mods = []
        for m in raw:
            if isinstance(m, dict):
                mods.append(self._norm_code(m.get("code")))
            else:
                mods.append(self._norm_code(m))
        # dedupe and drop empties
        return sorted({m for m in mods if m and m in self.scope})

    def calculate(self, outputs: list[Any], gold: list[Any]) -> MetricResult:
        # 1) Collect per-instance modifier lists
        y_true_lists: list[list[str]] = []
        y_pred_lists: list[list[str]] = []
        for i, output in enumerate(outputs):
            y_true_lists.append(self._mods_from_example(gold[i]))
            y_pred_lists.append(self._mods_from_example(output))

        # 2) Build global mapping and binarize over the UNION label space
        _, mapping = binarize(y_true_lists + y_pred_lists)
        y_true, _ = binarize(y_true_lists, mapping)  # (N, L)
        y_pred, _ = binarize(y_pred_lists, mapping)  # (N, L)

        # 3) Aggregated per-label metrics (unchanged behavior)
        aggregated_metric: dict[str, dict | float] = {}

        # Micro metrics: pool all modifiers and compute global precision/recall/f1
        total_tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        total_fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        total_fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        total_tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        denom_all = total_tp + total_fp + total_fn + total_tn
        micro_accuracy = (total_tp + total_tn) / denom_all if denom_all > 0 else 0.0
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        aggregated_metric.update(
            {
                "micro_accuracy": micro_accuracy,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
            }
        )
        for label, idx in mapping.items():
            true_col = y_true[:, idx]
            pred_col = y_pred[:, idx]

            tp = int(np.sum((true_col == 1) & (pred_col == 1)))
            fp = int(np.sum((true_col == 0) & (pred_col == 1)))
            fn = int(np.sum((true_col == 1) & (pred_col == 0)))
            tn = int(np.sum((true_col == 0) & (pred_col == 0)))

            denom_all = tp + fp + fn + tn
            accuracy = (tp + tn) / denom_all if denom_all > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            aggregated_metric[label] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        # 4) Instance-level metrics over the SAME label space
        instance_metrics: list[dict] = []
        for i in range(y_true.shape[0]):
            true_row = y_true[i]
            pred_row = y_pred[i]

            tp = int(np.sum((true_row == 1) & (pred_row == 1)))
            fp = int(np.sum((true_row == 0) & (pred_row == 1)))
            fn = int(np.sum((true_row == 1) & (pred_row == 0)))
            tn = int(np.sum((true_row == 0) & (pred_row == 0)))

            denom_all = tp + fp + fn + tn
            accuracy = (tp + tn) / denom_all if denom_all > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            instance_metrics.append(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

        return {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }


class CombinedModifierMicroMetric(BaseMetric):
    """
    Micro metrics pooled across both EM and Procedure modifiers.
    Treats all modifiers (both categories) as a single label space and reports micro precision/recall/f1/accuracy.
    """

    def __init__(self, em_label_key: str, procedure_label_key: str, scope):
        super().__init__(supervised=True)
        self.em_label_key = em_label_key
        self.procedure_label_key = procedure_label_key
        self.scope = get_modifiers_in_scope(scope)

    @staticmethod
    def _norm_code(x) -> str | None:
        if x is None:
            return None
        s = str(x).strip()
        return s.upper() if s else None

    def _mods_from_example(self, example: dict, label_key: str) -> list[str]:
        raw = get_nested_attribute(example, label_key) or []
        mods = []
        for m in raw:
            if isinstance(m, dict):
                mods.append(self._norm_code(m.get("code")))
            else:
                mods.append(self._norm_code(m))

        return sorted({m for m in mods if m and m in self.scope})

    def _combined_mods(self, example: dict) -> list[str]:
        em_mods = self._mods_from_example(example, self.em_label_key)
        proc_mods = self._mods_from_example(example, self.procedure_label_key)
        return sorted({*em_mods, *proc_mods})

    def calculate(self, outputs: list[Any], gold: list[Any]) -> MetricResult:
        # Build combined modifier lists per instance
        y_true_lists: list[list[str]] = []
        y_pred_lists: list[list[str]] = []
        for i, output in enumerate(outputs):
            y_true_lists.append(self._combined_mods(gold[i]))
            y_pred_lists.append(self._combined_mods(output))

        # Binarize over union of EM + procedure modifiers
        _, mapping = binarize(y_true_lists + y_pred_lists)
        y_true, _ = binarize(y_true_lists, mapping)
        y_pred, _ = binarize(y_pred_lists, mapping)

        # Micro metrics pooled across all modifiers
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        denom_all = tp + fp + fn + tn
        micro_accuracy = (tp + tn) / denom_all if denom_all > 0 else 0.0
        micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        aggregated_metric: dict[str, float] = {
            "micro_accuracy": micro_accuracy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        }

        # Per-instance metrics over the combined label space
        instance_metrics: list[dict] = []
        for i in range(y_true.shape[0]):
            true_row = y_true[i]
            pred_row = y_pred[i]

            tp_i = int(np.sum((true_row == 1) & (pred_row == 1)))
            fp_i = int(np.sum((true_row == 0) & (pred_row == 1)))
            fn_i = int(np.sum((true_row == 1) & (pred_row == 0)))
            tn_i = int(np.sum((true_row == 0) & (pred_row == 0)))

            denom_i = tp_i + fp_i + fn_i + tn_i
            acc_i = (tp_i + tn_i) / denom_i if denom_i > 0 else 0.0
            prec_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
            rec_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
            f1_i = 2 * prec_i * rec_i / (prec_i + rec_i) if (prec_i + rec_i) > 0 else 0.0

            instance_metrics.append(
                {
                    "accuracy": acc_i,
                    "precision": prec_i,
                    "recall": rec_i,
                    "f1": f1_i,
                }
            )

        return {"instance_metrics": instance_metrics, "aggregated_metric": aggregated_metric}
