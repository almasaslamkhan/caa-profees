from typing import Any

import numpy as np

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import binarize, get_nested_attribute


class Jaccard(BaseMetric):
    """Concrete implementation of Jaccard Score metric."""

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
            # _gold, _out = get_values_for_metrics_calculations(y_true=gold[i], y_pred=output, label_key=self.label_key)

            y_true.append(_gold)
            y_pred.append(_out)

        # binarize them
        _, mapping = binarize(y_true + y_pred)
        y_true, _ = binarize(y_true, mapping)
        y_pred, _ = binarize(y_pred, mapping)

        instance_metrics = []
        for i in range(len(y_pred)):
            inter = np.logical_and(y_true[i], y_pred[i]).sum()
            union = np.logical_or(y_true[i], y_pred[i]).sum()

            score = np.where(union == 0, 1.0, inter / union).mean()

            instance_metrics.append({"score": score})

        values = [m["score"] for m in instance_metrics]
        jaccard = np.mean(values)
        aggregated_metric = {"jaccard": jaccard}

        result: MetricResult = {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }
        return result
