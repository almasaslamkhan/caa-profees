from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import pandas as pd
from pydantic import BaseModel


class BaseReportGenerator(ABC):
    """Abstract base class for report generators."""

    def __init__(self, save_path: str):
        """Initialize the report generator."""
        self.save_path = save_path

    @abstractmethod
    def generate_reports(
        self,
        inputs: list[BaseModel],
        outputs: list[Any],
        metrics: list[dict[str, Any]],
        gold: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> Tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Abstract method to generate reports.

        Args:
            inputs (list[BaseModel]): The input data.
            outputs (list[Any]): The model outputs.
            metrics (list[Dict[str, Any]]): The computed per-instance metrics.
            gold (Optional[list[Any]]): The gold standard outputs, if any.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The sample report and summary report DataFrames.
        """
        pass
