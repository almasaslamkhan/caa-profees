from typing import Any

import numpy as np

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import binarize, get_nested_attribute


class MicroMetrics(BaseMetric):
    """Concrete implementation of micro precision, recall, F1, and accuracy."""

    def __init__(self, label_key: str):
        """Initialize with supervised flag.

        Args:
            label_key (str): The label key which is going to be used for comparing the fields.
                             It MUST point to the location of label in the schema.
                             For example, 'cpt.code'.
        """
        super().__init__(supervised=True)
        self.label_key = label_key

    def calculate(self, outputs: list[Any], gold: list[Any]) -> MetricResult:
        y_true = []
        y_pred = []
        for i, output in enumerate(outputs):
            _gold = get_nested_attribute(gold[i], self.label_key)
            _out = get_nested_attribute(output, self.label_key)
            # _gold, _out = get_values_for_metrics_calculations(y_true=gold[i], y_pred=output, label_key=self.label_key)

            y_true.append(_gold)
            y_pred.append(_out)

        # binarize them
        _, mapping = binarize(y_true + y_pred)
        y_true_bin, _ = binarize(y_true, mapping)
        y_pred_bin, _ = binarize(y_pred, mapping)

        # Flatten for micro calculation
        y_true_flat = np.array(y_true_bin).reshape(-1)
        y_pred_flat = np.array(y_pred_bin).reshape(-1)

        TP = np.logical_and(y_true_flat == 1, y_pred_flat == 1).sum()
        FP = np.logical_and(y_true_flat == 0, y_pred_flat == 1).sum()
        FN = np.logical_and(y_true_flat == 1, y_pred_flat == 0).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

        aggregated_metric = {"micro_precision": precision, "micro_recall": recall, "micro_f1": f1, "micro_accuracy": accuracy}

        result: MetricResult = {
            "instance_metrics": [],  # optional: can fill with per-instance TP/FP/FN if needed
            "aggregated_metric": aggregated_metric,
        }
        return result
