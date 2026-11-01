from typing import Any, List, Union

import numpy as np

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import binarize


def collect_primary_codes(obj: Any, path: str) -> List[str]:
    """
    Extract primary ICD codes for paths like:
      - "em.linked_diagnosis"
      - "procedure.linked_diagnosis"
    Assumes the final list contains ICD10CM objects (or dicts) with fields:
      code: str, is_primary: bool
    """

    def _get(x, k):
        return x.get(k) if isinstance(x, dict) else getattr(x, k, None)

    parent_key, child_key = path.split(".", 1)  # e.g., ("em", "linked_diagnosis")

    parent_list = _get(obj, parent_key)
    if parent_list is None:
        return []
    if not isinstance(parent_list, list):
        parent_list = [parent_list]  # tolerate single item

    codes: List[str] = []
    for item in parent_list:
        linked = _get(item, child_key) or []
        found_primary = False
        if linked:

            for dx in linked:
                if _get(dx, "is_primary") and _get(dx, "code"):
                    codes.append(str(_get(dx, "code")).strip().upper())
                    found_primary = True
                    break

            if not found_primary:
                codes.append("")
        else:
            codes.append("")

    return codes


class PrimaryDXMetrics(BaseMetric):
    """Accuracy over **primary** ICD codes inside linked_diagnosis blocks."""

    def __init__(self, label_key: Union[str, list[str]]):
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

        # Flatten across all samples into one list of strings
        y_true_codes = [code for g in gold for code in self._codes_for_sample(g)]
        y_pred_codes = [code for o in outputs for code in self._codes_for_sample(o)]

        # binarize multilabel
        _, mapping = binarize(y_true_codes + y_pred_codes)
        y_true, _ = binarize(y_true_codes, mapping)
        y_pred, _ = binarize(y_pred_codes, mapping)

        # handle empty label space
        if y_true.size == 0 or y_pred.size == 0:
            instance_metrics = [{"strict_correct": 1, "macro_correct": 1.0} for _ in range(len(outputs))]
            aggregated_metric = {"strict_accuracy": 1.0, "macro_accuracy": 1.0, "micro_accuracy": 1.0}
            return {"instance_metrics": instance_metrics, "aggregated_metric": aggregated_metric}

        strict_accs = [np.array_equal(y_true[i], y_pred[i]) for i in range(len(y_true))]
        strict_accuracy = float(np.mean(strict_accs))
        macro_accuracy = float((y_true == y_pred).mean(axis=0).mean())
        micro_accuracy = float((y_true == y_pred).mean())

        instance_metrics = [
            {"strict_correct": int(np.array_equal(y_true[i], y_pred[i])), "macro_correct": float((y_true[i] == y_pred[i]).mean())}
            for i in range(len(y_pred))
        ]

        aggregated_metric = {
            "strict_accuracy": strict_accuracy,
            "macro_accuracy": macro_accuracy,
            "micro_accuracy": micro_accuracy,
        }

        return {"instance_metrics": instance_metrics, "aggregated_metric": aggregated_metric}
