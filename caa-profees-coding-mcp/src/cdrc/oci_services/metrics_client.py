import inspect
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import wraps
from timeit import default_timer as timer

import oci
from oci.monitoring.models import Datapoint, MetricDataDetails, PostMetricDataDetails

from cdrc.common.logger import logger
from cdrc.config import settings

# constants
CONFIG_KEY_METRICS_BUFFER_INTERVAL = "metrics.metrics_buffer.interval"
CONFIG_KEY_METRICS_FLUSH_THRESHOLD = "metrics.metrics_buffer.flush_threshold"
CONFIG_KEY_METRICS_MAX_RECORDS_PER_POST_CALL = "metrics.metrics_buffer.max_records_per_post_call"
# locks and threading resources
lock = threading.Lock()
monitoring_thread_pool_executor_lock = threading.Lock()
metrics_buffer_lock = threading.Lock()
metrics_buffer = deque()
is_worker_running_lock = threading.Lock()
is_worker_running = True


def on_exit_metric_handler():
    # flushing last batch before exit (in the worker's main thread)
    try:
        logger.debug(f"PId: {os.getpid()}: " f" pushing last batch and closing MetricsClient TPE.")
        flush_metrics_buffer(is_last_batch=True)
        logger.debug(f"PId: {os.getpid()}: pushed last batch")
        MetricsClient.shutdown_thread_pool_executor()
        return 0
    except (RuntimeError, Exception) as exc:
        logger.error(f"Error encountered while cleaning up metrics" f" data and resources for shutdown. : {exc}")
        return -1


def check_buffer_and_push_metrics_at_intervals():
    logger.debug("ENTER: check_buffer_and_push_metrics_at_intervals")

    _interval_in_ms = settings.get(CONFIG_KEY_METRICS_BUFFER_INTERVAL)
    _interval = _interval_in_ms / 1000.0  # seconds

    while True:
        try:
            logger.debug(f"Next run in {_interval} seconds.")
            time.sleep(_interval)
            if not is_worker_running:  # Stop if worker shuts down because cannot schedule new futures after shutdown
                logger.debug(f"is_worker_running = {is_worker_running} " f" : closing main loop")
                return 0
            logger.debug(f"Waited for {_interval} seconds. START: Call to flush_metrics_buffer.")
            flush_metrics_buffer()
            logger.debug("DONE: Call to flush_metrics_buffer.")
        except Exception as exc:
            logger.error(f"Error during flushing metrics buffer: error= {exc}")
            return -1


def _add_record_to_metrics_buffer(metric_name, count=None, value=None, dimensions=None, **kwargs):
    if len(kwargs) > 0:
        raise AssertionError(f"Currently no kwargs are supported, but got these kwargs: {kwargs.keys()}")
    metrics_buffer.append(_build_metric_data_details(metric_name, count=count, value=value, dimensions=dimensions))
    # logger.debug(f"Metrics buffer was appended {metrics_buffer}")


def _build_metric_data_details(metric_name, count=None, value=None, dimensions=None):
    """Builds monitoring MetricDataDetails object (a datapoint)"""

    logger.debug("Building MetricDataDetails for {0} with value {1}".format(metric_name, value))
    dims = dict(settings.get("metrics.metrics_dimensions"))
    if dimensions is not None:
        dims.update(dimensions)
    metric_data_details = MetricDataDetails(
        name=metric_name,
        namespace=settings.get("metrics.metrics_namespace"),
        dimensions=dims,
        resource_group=settings.get("metrics.resource_group"),
        compartment_id=settings.get("metrics.metrics_compartment"),
        datapoints=[
            Datapoint(
                count=count,
                value=value,
                timestamp=datetime.now(timezone.utc).astimezone().isoformat(),
            )
        ],
    )
    return metric_data_details


def _flush_metrics_buffer(is_last_batch):
    flush_threshold = settings.get(CONFIG_KEY_METRICS_FLUSH_THRESHOLD)
    max_records_per_post_call = settings.get(CONFIG_KEY_METRICS_MAX_RECORDS_PER_POST_CALL)
    metric_records = []  # A list of MetricDataDetails
    num_records = None
    _info_str = f"pid={os.getpid()} {threading.current_thread().getName()} "
    with metrics_buffer_lock:
        num_records = len(metrics_buffer)
        logger.debug(
            f"{_info_str}"
            f"num_records = {num_records}, "
            f"flush_threshold = {flush_threshold}, max_records_per_post_call = {max_records_per_post_call}"
        )
        logger.debug(f"Metrics buffer holds: {metrics_buffer}")
        if num_records > flush_threshold or is_last_batch:
            for i in range(num_records):
                metric_records.append(metrics_buffer.popleft())

    if len(metric_records) > 0:
        chunk_sizes = [max_records_per_post_call for i in range(num_records // max_records_per_post_call)]
        if num_records % max_records_per_post_call > 0:
            chunk_sizes.append(num_records % max_records_per_post_call)
        logger.debug(f"{_info_str}" f"Metric records chunk sizes = {chunk_sizes}")
        start_idx = 0
        for chunk_size in chunk_sizes:
            metric_data_details = metric_records[start_idx : (start_idx + chunk_size)]
            try:
                post_metric_data_details = PostMetricDataDetails(metric_data=metric_data_details)
                # Post synchronously from the interval worker to avoid piling up futures.
                MetricsClient.get_monitoring_client().post_metric_data(post_metric_data_details)
                # if is_last_batch:
                #     # use main worker thread if it is the last batch
                #     MetricsClient.get_monitoring_client().post_metric_data(post_metric_data_details)
                # else:
                #     # submit metric to the executor for pushing
                #     # during the next run
                #     MetricsClient.get_monitoring_thread_pool_executor().submit(
                #         MetricsClient.get_monitoring_client().post_metric_data,
                #         post_metric_data_details,
                #     )
            except Exception as exc:
                logger.error(f"{_info_str} " f"Error while submitting metric data details: Error = {exc}")
            start_idx += chunk_size


def flush_metrics_buffer(is_last_batch=False):
    try:
        _flush_metrics_buffer(is_last_batch)
    except Exception as exc:
        logger.error(f"pid={os.getpid()} {threading.current_thread().getName()}" f"Error while flushing metrics buffer : error = {exc}")


def emit(metric_name, count=None, value=None, dimensions=None, **kwargs):
    """Submits/schedules metric with given params to be emitted. Metrics won't be pushed unless
    flush_metric_buffer is called.
    """
    try:
        _add_record_to_metrics_buffer(metric_name, count=count, value=value, dimensions=dimensions, **kwargs)
    except AssertionError as err:
        logger.error(f"Parameter assertion error in emit: {err}")
    except Exception as err:
        logger.error(f"Unknown error in emit: {err}")


def measure_time(metric_name: str):
    def decorator(func):
        """
        Wrapper for the function to calculate the time taken and emit the metric accordingly
        """

        @wraps(func)
        async def _time_it_async(*args, **kwargs):
            """
            Function that calculates the time taken in millisecond for the async method to be executed and then emit the
            metric __qualname__ returns ClassName.MethodName
            """
            start = timer()
            response = await func(*args, **kwargs)
            end = timer()
            end_ = 1000 * (end - start)
            emit("{}.ResourceTime".format(metric_name), value=end_)
            return response

        @wraps(func)
        def _time_it_sync(*args, **kwargs):
            """
            Function that calculates the time taken in millisecond for the sync method to be executed and then emit the
            metric __qualname__ returns ClassName.MethodName
            """
            start = timer()
            response = func(*args, **kwargs)
            end = timer()
            end_ = 1000 * (end - start)
            emit("{}.ResourceTime".format(metric_name), value=end_)
            return response

        if inspect.iscoroutinefunction(func):
            return _time_it_async
        else:
            return _time_it_sync

    return decorator


class MetricsClient:
    """Monitoring Client wrapper."""

    _monitoring_client = None
    _monitoring_thread_pool_executor = None

    def __init__(self):
        raise RuntimeError(" It's a singleton class. Try calling get_monitoring_client() instead")

    @classmethod
    def get_monitoring_client(cls):
        if not cls._monitoring_client:
            with lock:
                if not cls._monitoring_client:
                    logger.debug("Fetching oci monitoring client")
                    cls._monitoring_client = oci.monitoring.MonitoringClient(
                        config=dict(settings.get("metrics.metrics_config")),  # metrics_config is of type ConfigTree
                        signer=oci.auth.signers.InstancePrincipalsSecurityTokenSigner(),
                    )
                    cls._monitoring_client.base_client.endpoint = settings.get("metrics.monitoring_client_endpoint")
        return cls._monitoring_client

    @classmethod
    def _create_tp_executor_for_metrics(cls):
        """To be used only by get/reset _monitoring_thread_pool_executor to initialize
        new thread pool executor for handling metrics emission.
        """
        max_worker_threads = settings.get("metrics.metrics_tpe.max_worker_threads")
        if max_worker_threads < 2:
            raise ValueError(f"max_worker_threads must be greater than 2, but received {max_worker_threads}.")
        logger.debug(f"Settings: metrics.metrics_tpe.max_worker_threads = {max_worker_threads} " f"{type(max_worker_threads)}")
        thread_name_prefix = str(os.getpid())
        cls._monitoring_thread_pool_executor = ThreadPoolExecutor(max_workers=max_worker_threads, thread_name_prefix=thread_name_prefix)
        logger.debug(
            f"Initialized _monitoring_thread_pool_executor with max_workers={max_worker_threads}" f" and thread_name_prefix = {thread_name_prefix}"
        )

    @classmethod
    def get_monitoring_thread_pool_executor(cls):
        if cls._monitoring_thread_pool_executor is None:
            with monitoring_thread_pool_executor_lock:
                if cls._monitoring_thread_pool_executor is None:
                    cls._create_tp_executor_for_metrics()
        return cls._monitoring_thread_pool_executor

    @classmethod
    def reset_monitoring_thread_pool_executor(cls):
        """To be used in just after a worker process has been initialized
        so that it has reference to an active ThreadPoolExecutor
        """
        logger.debug("ENTER: reset_monitoring_thread_pool_executor")
        with monitoring_thread_pool_executor_lock:
            logger.debug("Re-initializing _monitoring_thread_pool_executor")
            cls._create_tp_executor_for_metrics()
        logger.debug("DONE: reset_monitoring_thread_pool_executor")
        return cls._monitoring_thread_pool_executor

    @classmethod
    def activate(cls):
        logger.debug("ENTER:activate - Activating thread to check metrics buffer at regular intervals.")
        tpe = cls.get_monitoring_thread_pool_executor()
        tpe.submit(check_buffer_and_push_metrics_at_intervals)
        logger.debug("EXIT:activate - Activating thread to check metrics buffer at regular intervals.")

    @classmethod
    def shutdown_thread_pool_executor(cls):
        logger.debug("Shutting down Metrics Client TPE")
        with monitoring_thread_pool_executor_lock:
            if cls._monitoring_thread_pool_executor is not None:
                try:
                    cls._monitoring_thread_pool_executor.shutdown(wait=True)
                    logger.debug("Successfully shut down Metrics Client TPE")
                except RuntimeError as exc:
                    logger.error(f"Error while shutting down MetricsClient TPE: {exc}")
            else:
                logger.debug("MetricsClient TPE unavailable. Skipping shutdown.")
