import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

from cdrc.common.llm_utils import TokenTracker
from cdrc.common.metric.base import MetricResult
from cdrc.common.report_generation.base import BaseReportGenerator


class CSVReportGenerator(BaseReportGenerator):
    """Concrete implementation of a report generator."""

    def generate_reports(
        self,
        input_data: list[BaseModel],
        output_data: list[Any],
        metrics: dict[str, MetricResult],
        gold_data: Optional[list[Any]] = None,
        get_token_stats_report: Optional[bool] = True,
    ) -> Tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Generate sample report and separate summary reports for each metric.

        Args:
            input_data (list[BaseModel]): The input data.
            output_data (list[Any]): The model outputs.
            metrics (dict[str, Any]]): The computed per-instance metrics.
            gold_data (Optional[list[Any]]): The gold standard outputs, if any.

        Returns:
            Tuple[pd.DataFrame, dict[str, pd.DataFrame]]: The sample report and summary report DataFrames.
        """
        sample_report = self.create_sample_report(input_data, output_data, metrics, gold_data)
        summary_reports = self.create_summary_reports(metrics)

        os.makedirs(self.save_path, exist_ok=True)

        self.save_report(sample_report, os.path.join(self.save_path, "sample_report.csv"))

        token_stats_fields = [
            "Total Tokens",
            "Prompt Tokens",
            "Completion Tokens",
            "Prompt Tokens Cached",
            "Reasoning Tokens",
            "Total Cost",
            "Successful Requests",
            "Requests Time",
            "Rate Limit Delays",
            "Avg Time Per Instance",
            "Avg Wall Time Per Instance",
            "Avg Tokens Per Instance",
            "Avg Cost Per Instance",
        ]
        # Calculate the maximum width needed for all dataframes
        max_width = max(max(report.shape[1] for report in summary_reports.values()), len(token_stats_fields))
        empty_rows = [[np.nan] * max_width] * 3

        all_summary_reports = []
        for metric_name, summary_report in summary_reports.items():
            metric_header = [metric_name] + [np.nan] * (max_width - 1)
            all_summary_reports.append(metric_header)

            all_summary_reports.extend([summary_report.columns.tolist()])
            all_summary_reports.extend(summary_report.values.tolist())
            all_summary_reports.extend(empty_rows)

        summary_report = pd.DataFrame(all_summary_reports)
        self.save_report(summary_report, os.path.join(self.save_path, "summary_report.csv"), False)

        if get_token_stats_report:
            token_stats_report = []
            token_stats = TokenTracker.get_stats("model_usage")

            if token_stats.total_tokens != 0:
                # token_stats_header = ["Token Statistics"] + [np.nan] * (max_width - 1)
                # token_stats_report.append(token_stats_header)

                # Add token stats headers and values as separate rows
                token_stats_categories = token_stats_fields + [np.nan] * (max_width - len(token_stats_fields))
                token_stats_report.append(token_stats_categories)

                token_stats_values = [
                    token_stats.total_tokens,
                    token_stats.prompt_tokens,
                    token_stats.completion_tokens,
                    token_stats.prompt_tokens_cached,
                    token_stats.reasoning_tokens,
                    f"${token_stats.total_cost:.5f}",
                    token_stats.successful_requests,
                    [round(t, 2) for t in token_stats.requests_time_in_seconds],
                    [round(t, 2) for t in token_stats.rate_limit_delays],
                    (sum(token_stats.requests_time_in_seconds) - sum(token_stats.rate_limit_delays)) / len(input_data),
                    (time.time() - token_stats.init_time) / len(input_data),
                    token_stats.total_tokens / len(input_data),
                    token_stats.total_cost / len(input_data),
                ] + [np.nan] * (max_width - len(token_stats_fields))
                token_stats_report.append(token_stats_values)

                token_report = pd.DataFrame(token_stats_report)
                self.save_report(token_report, os.path.join(self.save_path, "token_usage_report.csv"), False)

        return sample_report, summary_reports

    def create_sample_report(
        self,
        input_data: list[BaseModel],
        output_data: list[Any],
        metrics: dict[str, MetricResult],
        gold_data: Optional[list[Any]] = None,
    ) -> pd.DataFrame:
        """Create a detailed report for each data sample.

        Args:
            input_data (list[BaseModel]): The input data.
            output_data (list[Any]): The model outputs.
            metrics (dict[str, Any]]): The computed per-instance metrics.
            gold_data (Optional[list[Any]]): The gold standard outputs, if any.

        Returns:
            pd.DataFrame: The sample report DataFrame.
        """
        num_samples = len(input_data)
        rows = []

        for i in range(num_samples):
            row = {}

            # Expand input fields into separate columns
            input_sample = input_data[i]
            input_dict = json.loads(input_sample.model_dump_json())

            for key, value in input_dict.items():
                row[f"input_{key}"] = value

            # Expand output fields into separate columns
            output_sample = output_data[i]
            if isinstance(output_sample, BaseModel):
                output_dict = json.loads(output_sample.model_dump_json())
            else:
                output_dict = output_sample  # Assuming output_sample is a dict
            for key, value in output_dict.items():
                row[f"output_{key}"] = value

            # Expand gold fields into separate columns
            if gold_data:
                gold_sample = gold_data[i]
                if isinstance(gold_sample, BaseModel):
                    gold_dict = json.loads(gold_sample.model_dump_json())
                else:
                    gold_dict = gold_sample  # Assuming gold_sample is a dict

                for key, value in gold_dict.items():
                    row[f"gold_{key}"] = value

            # Include all sample-level metrics
            for metric_name, metric_result in metrics.items():
                # """
                if (
                    "instance_metrics" in metric_result
                    and isinstance(metric_result["instance_metrics"], list)
                    and i < len(metric_result["instance_metrics"])
                ):
                    # """
                    # if metric_result.get("instance_metrics", []):
                    instance_metric = metric_result["instance_metrics"][i]
                for key, value in instance_metric.items():
                    row[f"{metric_name}_{key}"] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def create_summary_reports(self, metrics: dict[str, MetricResult]) -> dict[str, pd.DataFrame]:
        """Create separate summary reports for each metric.

        Args:
            metrics (dict[str, Any]]): The computed per-instance metrics.

        Returns:
            dict[str, pd.DataFrame]: The summary report DataFrames.
        """
        summary_reports = {}
        for metric_name, metric_result in metrics.items():
            aggregated_metric = metric_result["aggregated_metric"]
            # Allow metrics to directly provide a pandas DataFrame with multiple rows
            if isinstance(aggregated_metric, pd.DataFrame):
                summary_reports[metric_name] = aggregated_metric
            else:
                summary_reports[metric_name] = pd.DataFrame([aggregated_metric])
        return summary_reports

    def save_report(self, report_data: pd.DataFrame, filename: str, header: bool = True) -> None:
        """Save report data to a CSV file.

        Args:
            report_data (pd.DataFrame): The report data to save.
            filename (str): The filename to save the report as.
        """
        report_data.to_csv(filename, index=False, header=header)
