from abc import ABC, abstractmethod
from typing import Any, Optional

from typing_extensions import NotRequired, Required, TypedDict


class MetricResult(TypedDict, total=False):
    """TypedDict specifying the required structure of metric calculation results."""

    instance_metrics: Required[list[dict[str, Any]]]
    aggregated_metric: Required[dict[str, Any]]
    # Any additional fields are allowed


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(self, supervised: bool = False):
        """Initialize with supervised flag.

        Args:
            supervised (bool): Whether the metric is supervised (requires gold data).
        """
        self.supervised = supervised

    @abstractmethod
    def calculate(self, outputs: list[Any], gold: Optional[list[Any]] = None) -> MetricResult:
        """Abstract method to calculate the metric.

        Args:
            outputs (List[Any]): The model outputs.
            gold (Optional[List[Any]]): The gold standard outputs, if any.

        Returns:
            MetricResult: The calculated metrics.
        """
        pass
