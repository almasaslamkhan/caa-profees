from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import get_nested_attribute


class Latency(BaseMetric):
    """Concrete implementation of accuracy metric."""

    def __init__(self):
        """Initialize with supervised flag."""
        super().__init__(supervised=True)

    def calculate(self, outputs: list[Any], gold: list[Any]) -> MetricResult:
        """Calculate metrics for outputs comparing to gold data.

        Args:
            outputs (list[Any]): The model outputs.

        Returns:
            MetricResult: A dictionary containing instance metrics and aggregated metrics.
        """
        instance_metrics = []
        for i, output in enumerate(outputs):
            instance_metrics.append({"latency": output.additional_output.get("latency", -1)})

        avg_latency = np.mean([m["latency"] for m in instance_metrics])
        aggregated_metric = {"latency": avg_latency}

        result: MetricResult = {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }
        return result
