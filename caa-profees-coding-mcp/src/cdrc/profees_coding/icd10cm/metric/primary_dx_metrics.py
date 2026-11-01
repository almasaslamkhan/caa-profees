from typing import Any, Iterable, List, Optional

import numpy as np
from pydantic import BaseModel

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import binarize, get_nested_attribute, get_values_for_metrics_calculations


def collect_primary_codes(obj: Any, path: str) -> List[str]:
    """
    Extract primary ICD codes for paths like:
      - "diagnosis"
    Assumes the final list contains ICD10CM objects (or dicts) with fields:
      code: str, is_primary: bool
    """

    def _get(x, k):
        return x.get(k) if isinstance(x, dict) else getattr(x, k, None)

    codes: List[str] = []
    for item in _get(obj, path):
        if _get(item, "is_primary") and _get(item, "code"):
            codes.extend([str(_get(item, "code")).strip().upper()])
            break
    return codes


class PrimaryDXMetrics(BaseMetric):
    """Accuracy over **primary** ICD codes inside linked_diagnosis blocks."""

    def __init__(self, label_key: str | list[str]):
        super().__init__(supervised=True)
        self.label_key = label_key

    def _codes_for_sample(self, obj: Any) -> list[str]:
        if isinstance(self.label_key, (list, tuple)):
            codes: list[str] = []
            for lk in self.label_key:
                codes.extend(collect_primary_codes(obj, lk))
            return codes
        else:
            return collect_primary_codes(obj, self.label_key)

    def calculate(self, outputs: list[Any], gold: list[Any]) -> MetricResult:
        assert len(outputs) == len(gold), f"len(outputs)={len(outputs)} != len(gold)={len(gold)}"

        # Per-sample primary-code sets
        y_true_codes_per_sample = [self._codes_for_sample(g) for g in gold]
        y_pred_codes_per_sample = [self._codes_for_sample(o) for o in outputs]

        # Build label space from union of all codes
        _, mapping = binarize(y_true_codes_per_sample + y_pred_codes_per_sample)
        y_true, _ = binarize(y_true_codes_per_sample, mapping)
        y_pred, _ = binarize(y_pred_codes_per_sample, mapping)

        # handle empty label space (no primary codes anywhere)
        if y_true.size == 0 or y_pred.size == 0:
            n = len(outputs)
            instance_metrics = [{"strict_correct": 1, "macro_correct": 1.0} for _ in range(n)]
            aggregated_metric = {
                "strict_accuracy": 1.0,
                "macro_accuracy": 1.0,
                "micro_accuracy": 1.0,
            }
            return {"instance_metrics": instance_metrics, "aggregated_metric": aggregated_metric}

        # y_true, y_pred are (N_samples, N_labels)
        assert y_true.shape == y_pred.shape, f"Shape mismatch after binarize: y_true={y_true.shape}, y_pred={y_pred.shape}"

        N = y_true.shape[0]

        # Strict: full row equality
        strict_accs = [bool(np.array_equal(y_true[i], y_pred[i])) for i in range(N)]
        strict_accuracy = float(np.mean(strict_accs))

        # Macro: average per-label accuracy
        macro_accuracy = float((y_true == y_pred).mean(axis=0).mean())

        # Micro: accuracy over all (sample, label) pairs
        micro_accuracy = float((y_true == y_pred).mean())

        instance_metrics = [
            {
                "strict_correct": int(np.array_equal(y_true[i], y_pred[i])),
                "macro_correct": float((y_true[i] == y_pred[i]).mean()),
            }
            for i in range(N)
        ]

        aggregated_metric = {
            "strict_accuracy": strict_accuracy,
            "macro_accuracy": macro_accuracy,
            "micro_accuracy": micro_accuracy,
        }
        return {"instance_metrics": instance_metrics, "aggregated_metric": aggregated_metric}
