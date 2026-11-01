import os
import pickle
from abc import ABC
from datetime import datetime
from typing import Any, Callable, Optional, Tuple, Union

import pandas as pd
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from tqdm import tqdm

from cdrc.common.callback.latency import ChainLatencyCallback
from cdrc.common.data.reader import BaseDataReader
from cdrc.common.llm_utils import TokenTracker
from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.observability import attach_agentlens_callback, flush_agentlens, log_gold_feedback
from cdrc.common.report_generation.base import BaseReportGenerator
from cdrc.common.utils import load_evaluation_results


class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""

    def __init__(
        self,
        data_reader: BaseDataReader,
        model: CompiledStateGraph,
        model_config: Union[RunnableConfig, list[RunnableConfig]],
        metrics: dict[str, BaseMetric],
        report_generator: BaseReportGenerator,
        batch_size: int = -1,
        debug_mode: bool = False,
        previous_run_path: str = None,
        cache_dir: str = None,
        reprediction: bool = False,
    ):
        """Initialize with data reader, model, metrics, and report generator.

        Args:
            data_reader (BaseDataReader): The data reader.
            model CompiledStateGraph: The langgraph model to evaluate OR the previous state.
            model_config (Union[RunnableConfig, list[RunnableConfig]]): The config(s) that are sent to the model during .batch runs
            metrics (dict[str, BaseMetric]): The metrics to use for evaluation.
            report_generator (BaseReportGenerator): The report generator.
            batch_size (int): The number of batches, if set to -1 would send the entire dataset to the model at once.
            debug_mode (bool): Whether to run in debug mode.
            previous_run_path (str): Path to previous run results if one wishes to only calculate metrics without running the model.
            cache_dir (str): Directory to look for cached outputs (and save partial results) to enable resuming a previous failed run.
            filter_function (Optional[Callable]): Function to apply to the data.
            filter_kwargs (dict[str, Any]): Arguments to pass to the filter function.
            reprediction (bool): Whether we are running reprediction.
        """
        self.data_reader = data_reader
        self.model = model
        self.metrics = metrics
        self.report_generator = report_generator
        self.model_config = model_config
        self.batch_size = batch_size
        self.debug_mode = debug_mode
        self.previous_run_path = previous_run_path
        self.cache_dir = cache_dir
        self.reprediction = reprediction

        if isinstance(self.model_config, dict):
            if "latency" in self.metrics.keys():
                self.latency_callback = ChainLatencyCallback()
                self.model_config["callbacks"] = [self.latency_callback]

    """Concrete implementation of an evaluator."""

    def run(
        self, data_path: str, filter_function: Optional[Callable] = None, **filter_kwargs
    ) -> Tuple[dict[str, MetricResult], pd.DataFrame, dict[str, pd.DataFrame]]:
        """Evaluate the model and generate reports.

        Args:
            data_path (str): The path to the data file.
            filter_function (Optional[Callable]): Function to apply to the data.
            **filter_kwargs: Arguments to pass to the filter function.

        Returns:
            Tuple[dict[str, MetricResult], pd.DataFrame, dict[str, pd.DataFrame]]: The list of per-instance metrics,
            sample report DataFrame, and summary report DataFrame.
        """
        try:
            if isinstance(self.model, CompiledStateGraph):
                return self._run_langgraph_model(data_path=data_path, filter_function=filter_function, **filter_kwargs)
            else:
                return self._run_cached_model(data_path=data_path, filter_function=filter_function, **filter_kwargs)
        finally:
            flush_agentlens(timeout_s=5.0)

    def _run_langgraph_model(
        self, data_path: str, filter_function: Optional[Callable] = None, **filter_kwargs
    ) -> Tuple[dict[str, MetricResult], pd.DataFrame, dict[str, pd.DataFrame]]:
        """Evaluate a langgraph model and generate reports.

        Args:
            data_path (str): The path to the data file.

        Returns:
            Tuple[dict[str, MetricResult], pd.DataFrame, dict[str, pd.DataFrame]]: The list of per-instance metrics,
            sample report DataFrame, and summary report DataFrame.
        """
        input_data, gold_data = self.data_reader.read_data(data_path, filter_function=filter_function, **filter_kwargs)
        cache_filename = None
        callback_handlers = []

        cfg = self.model_config.get("configurable", {})
        base_metadata = self.model_config.get("metadata", {})

        def _build_eval_config(example: Any) -> tuple[dict[str, Any], Any | None]:
            callbacks, handler = attach_agentlens_callback(self.model_config.get("callbacks", []))
            example_id = getattr(example, "example_id", None)
            metadata = dict(base_metadata)
            if example_id is not None:
                metadata["conversation_id"] = str(example_id)
                metadata["example_id"] = str(example_id)

            return (
                {
                    "configurable": {"thread_id": example_id, "max_concurrency": 999, **cfg},
                    "callbacks": callbacks,
                    "metadata": metadata,
                },
                handler,
            )

        if self.batch_size == -1:
            # Predict outputs for all data
            configs = []
            for example in input_data:
                config, handler = _build_eval_config(example)
                configs.append(config)
                callback_handlers.append(handler)
            # configs = [{"configurable": {"thread_id": x.example_id, "max_concurrency": 999, **cfg}} for x in input_data]
            if self.reprediction:
                data = []
                for x in input_data:
                    x = x.model_dump()
                    # x["signed_note"] = x["text"]
                    # del x["text"]
                    data.append(x)
                input_data = data
            all_outputs = self.model.batch(input_data, configs, debug=self.debug_mode)
        else:
            all_outputs = []
            num_samples = len(input_data)

            # Create cache filename based on data path
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                data_file_name = os.path.basename(data_path).split(".")[0]
                cache_filename = os.path.join(self.cache_dir, f"cached_outputs_for_{data_file_name}.pkl")

                # Check if cache exists and load it
                if os.path.exists(cache_filename):
                    try:
                        with open(cache_filename, "rb") as f:
                            cache_data = pickle.load(f)
                            all_outputs = cache_data["outputs"]
                            start_idx = len(all_outputs)
                            print(f"Loaded {start_idx} cached outputs. Resuming from index {start_idx}.")

                            # Restore token tracking stats if available
                            if "token_stats" in cache_data:
                                TokenTracker.restore_stats(cache_data["token_stats"])
                                print("Restored token tracking statistics from cache.")

                            # Check if we've already processed all samples
                            if start_idx > num_samples:
                                raise ValueError(
                                    f"Cache file {cache_filename} contains more samples than the data file "
                                    f"{data_path}. Please delete the cache file and run again."
                                )
                    except Exception as e:
                        print(f"Error loading cache: {e}. Starting from beginning.")
                        all_outputs = []
                        start_idx = 0
                else:
                    print(f"Cache dir provided but no cache found at {cache_filename}. Starting from beginning.")
                    start_idx = 0
            else:
                print("No cache dir provided. Starting from beginning.")
                start_idx = 0

            for i in tqdm(
                range(start_idx, num_samples, self.batch_size),
                desc="Running evaluation",
                initial=(start_idx + self.batch_size - 1) // self.batch_size,
                total=len(range(0, num_samples, self.batch_size)),
            ):
                batch_inputs = input_data[i : min(i + self.batch_size, num_samples)]
                configs = []
                batch_handlers = []
                for example in batch_inputs:
                    config, handler = _build_eval_config(example)
                    configs.append(config)
                    batch_handlers.append(handler)
                # configs = [{"configurable": {"thread_id": x.example_id, "max_concurrency": 999, **cfg}} for x in batch_inputs]
                if self.reprediction:
                    data = []
                    for x in batch_inputs:
                        x = x.model_dump()
                        # x["signed_note"] = x["text"]
                        # del x["text"]
                        data.append(x)
                    batch_inputs = data
                try:
                    result = self.model.batch(batch_inputs, configs, debug=self.debug_mode)
                    all_outputs.extend(result)
                    callback_handlers.extend(batch_handlers)

                    # Save cache after each batch
                    if cache_filename:
                        with open(cache_filename, "wb") as f:
                            pickle.dump(
                                {
                                    "outputs": all_outputs,
                                    "timestamp": datetime.now().isoformat(),
                                    "token_stats": TokenTracker.get_all_stats(),  # Save token stats
                                },
                                f,
                            )
                except Exception as e:
                    print(f"Error processing batch starting at index {i}: {e}")
                    if cache_filename:
                        print(f"You can resume from the last successful sample using the cache at {cache_filename}")
                    raise
        # cast to output schema
        all_outputs = [self.data_reader.output_schema(**output) for output in all_outputs]
        if "latency" in self.metrics.keys():
            for i, latency in enumerate(self.latency_callback.latencies):
                if all_outputs[i].additional_output is None:
                    all_outputs[i].additional_output = {}
                all_outputs[i].additional_output["latency"] = latency

        self._log_agentlens_gold_feedback(
            input_data=input_data,
            gold_data=gold_data,
            callback_handlers=callback_handlers,
            data_path=data_path,
        )

        # Collect metrics
        metrics = {}
        for metric_name, metric in self.metrics.items():
            metrics_result = metric.calculate(all_outputs, gold_data)
            metrics[metric_name] = metrics_result

        # Generate reports
        sample_report, summary_reports = self.report_generator.generate_reports(input_data, all_outputs, metrics, gold_data)

        # Delete cache file if run completed successfully
        if cache_filename and os.path.exists(cache_filename):
            os.remove(cache_filename)
            print(f"Evaluation completed successfully. Cache file {cache_filename} has been deleted.")

        return metrics, sample_report, summary_reports

    @staticmethod
    def _log_agentlens_gold_feedback(
        input_data: list[Any],
        gold_data: list[Any],
        callback_handlers: list[Any],
        data_path: str,
    ) -> None:
        if not gold_data or not callback_handlers:
            return

        data_file = os.path.basename(data_path) if data_path else None

        for input_item, gold_item, handler in zip(input_data, gold_data, callback_handlers):
            trace_id = getattr(handler, "last_trace_id", None) if handler is not None else None
            if not trace_id:
                continue

            example_id = getattr(input_item, "example_id", None)
            metadata = {}
            if example_id is not None:
                metadata["example_id"] = str(example_id)
            if data_file:
                metadata["data_file"] = data_file

            try:
                log_gold_feedback(
                    target_trace_id=trace_id,
                    expected=gold_item,
                    metadata=metadata or None,
                )
            except Exception:
                continue

    def _run_cached_model(
        self, data_path: str, filter_function: Optional[Callable] = None, **filter_kwargs
    ) -> Tuple[dict[str, MetricResult], pd.DataFrame, dict[str, pd.DataFrame]]:
        """Evaluate cached generation and generate reports.

        Args:
            data_path (str): The path to the csv report file.

        Returns:
            Tuple[dict[str, MetricResult], pd.DataFrame, dict[str, pd.DataFrame]]: The list of per-instance metrics,
            sample report DataFrame, and summary report DataFrame.
        """

        print("loading previous run results...", flush=True)
        input_data, all_outputs, gold_data = load_evaluation_results(
            csv_path=self.previous_run_path, input_schema=self.data_reader.input_schema, output_schema=self.data_reader.output_schema
        )

        # if data_path is specified, read the gold data from the specified path
        if data_path:
            print("loading gold data...", flush=True)
            _, gold_data = self.data_reader.read_data(data_path, filter_function=filter_function, **filter_kwargs)

        # Collect metrics
        metrics = {}
        print("computing metrics...", flush=True)
        for metric_name, metric in self.metrics.items():
            metrics_result = metric.calculate(all_outputs, gold_data)
            metrics[metric_name] = metrics_result

        # Generate reports
        print("saving reports...", flush=True)
        sample_report, summary_reports = self.report_generator.generate_reports(
            input_data, all_outputs, metrics, gold_data, get_token_stats_report=False
        )
        return metrics, sample_report, summary_reports
