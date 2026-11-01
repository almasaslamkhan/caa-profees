from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.profees_coding.em.data.schema import ComplexityLevel
from cdrc.profees_coding.em.metric.complexity_utils import (
    QUADRATIC_WEIGHTS,
    calculate_coding_errors,
    calculate_coding_rates,
    custom_weighted_kappa,
    extract_complexity_levels,
)


class ComplexityLevelMetrics(BaseMetric):
    """A set of metrics to analyse model performance on predicting complexity levels:
    - Quadratic Weighted Kappa (the implementation can be adjusted to use custom weights in the future)
    - Mean undercoding error
    - Mean overcoding error
    - Mean undercoding rate
    - Mean overcoding rate
    """

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
        """Calculate metrics for outputs comparing to gold data. If the gold data doesn't contain the label_key, the instance is ignored.

        Args:
            outputs (list[Any]): The model outputs.
            gold (Optional[list[Any]]): The gold standard outputs.

        Returns:
            MetricResult: A dictionary containing instance metrics and aggregated metrics.
        """

        gold_levels, output_levels = extract_complexity_levels(gold, outputs, self.label_key)
        if gold_levels is None or output_levels is None:
            return MetricResult(
                instance_metrics=[],
                aggregated_metric={},
            )

        kappa = custom_weighted_kappa(gold_levels, output_levels, QUADRATIC_WEIGHTS)
        coding_errors = calculate_coding_errors(gold_levels, output_levels)
        coding_rates = calculate_coding_rates(gold_levels, output_levels)

        aggregated_metric = {"N": len(gold_levels), "quadratic_weighted_kappa": kappa}

        aggregated_metric.update(coding_errors)
        aggregated_metric.update(coding_rates)

        result: MetricResult = {
            "instance_metrics": [],
            "aggregated_metric": aggregated_metric,
        }

        return result
