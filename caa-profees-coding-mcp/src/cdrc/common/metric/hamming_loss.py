from typing import Any

import numpy as np

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import binarize, get_nested_attribute


class HammingLoss(BaseMetric):
    """Concrete implementation of Hamming Loss metric."""

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
        """Calculate metrics for outputs comparing to gold data.

        Args:
            outputs (list[Any]): The model outputs.
            gold (Optional[list[Any]]): The gold standard outputs.

        Returns:
            MetricResult: A dictionary containing instance metrics and aggregated metrics.
        """
        y_true = []
        y_pred = []
        for i, output in enumerate(outputs):
            _gold = get_nested_attribute(gold[i], self.label_key)
            _out = get_nested_attribute(output, self.label_key)

            y_true.append(_gold)
            y_pred.append(_out)

        # binarize them
        _, mapping = binarize(y_true + y_pred)
        y_true, _ = binarize(y_true, mapping)
        y_pred, _ = binarize(y_pred, mapping)

        instance_metrics = []
        for i in range(len(y_pred)):
            loss = np.not_equal(y_true[i], y_pred[i]).mean()

            instance_metrics.append({"loss": loss})

        values = [m["loss"] for m in instance_metrics]
        hamming_loss = np.mean(values)
        aggregated_metric = {"hamming_loss": hamming_loss}

        result: MetricResult = {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }
        return result
