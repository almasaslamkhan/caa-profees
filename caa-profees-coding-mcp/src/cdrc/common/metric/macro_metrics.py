from typing import Any

import numpy as np

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import binarize, get_nested_attribute


class MacroMetrics(BaseMetric):
    """Concrete implementation of macro precision, recall, F1, and accuracy."""

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

        y_true_arr = np.array(y_true_bin)
        y_pred_arr = np.array(y_pred_bin)

        num_labels = y_true_arr.shape[1]
        precisions, recalls, f1s, accuracies = [], [], [], []

        # Macro metrics per label
        for i in range(num_labels):
            y_t = y_true_arr[:, i]
            y_p = y_pred_arr[:, i]

            TP = np.logical_and(y_t == 1, y_p == 1).sum()
            FP = np.logical_and(y_t == 0, y_p == 1).sum()
            FN = np.logical_and(y_t == 1, y_p == 0).sum()

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            acc = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            accuracies.append(acc)

        # Instance-level metrics
        instance_metrics = []
        for i in range(len(y_true_arr)):
            y_t = y_true_arr[i]
            y_p = y_pred_arr[i]

            TP = np.logical_and(y_t == 1, y_p == 1).sum()
            FP = np.logical_and(y_t == 0, y_p == 1).sum()
            FN = np.logical_and(y_t == 1, y_p == 0).sum()

            inst_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            inst_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            inst_f1 = (2 * inst_precision * inst_recall / (inst_precision + inst_recall)) if (inst_precision + inst_recall) > 0 else 0.0
            inst_accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

            instance_metrics.append({"precision": inst_precision, "recall": inst_recall, "f1": inst_f1, "accuracy": inst_accuracy})

        aggregated_metric = {
            "macro_precision": np.mean(precisions),
            "macro_recall": np.mean(recalls),
            "macro_f1": np.mean(f1s),
            "macro_accuracy": np.mean(accuracies),
        }

        result: MetricResult = {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }
        return result
