# agent_obs/metrics.py

# Thin wrapper that delegates metric publication to cda_caim_sdk.core.monitoring.

# No local OCI fallback or stdout publishing; relies solely on the SDK publisher.

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Dict, Mapping, Optional

try:
    from cda_caim_sdk.core import monitoring as _mon  # type: ignore
except ModuleNotFoundError:
    _mon = None

try:
    from common_utility import logging_context

    _logger: logging.Logger = logging_context.get_logger(__name__)
except Exception:
    _logger = logging.getLogger(__name__)


METRICS_REPORTING_NAMESPACE = "profees-coding-agent"


def publish_metric(
    name: str,
    value: float,
    dimensions: Optional[Mapping[str, str]] = None,
    unit: Optional[str] = None,
) -> None:
    """
    Publish a single metric using cda_caim_sdk.core.monitoring.
    Notes:
    - unit is carried as a dimension for consistency with prior behavior.

    - timestamp is ignored; upstream publisher determines timestamp semantics.

    """

    dims: Dict[str, str] = dict(dimensions or {})
    if unit and "unit" not in dims:
        dims["unit"] = str(unit)
    try:
        _logger.debug(f"metrics.emit name={name} value={float(value)} dims={dims}")
        if _mon is None:
            _logger.debug("metrics.emit skipped because cda_caim_sdk is unavailable")
            return
        _mon._log_metric(name=name, value=float(value), dimensions=dims)
        _logger.debug("metrics.emit success")
    except Exception:
        _logger.exception(f"metrics.emit failure name={name} dims={dims}")


def counter(name: str, inc: float = 1.0, dimensions: Optional[Mapping[str, str]] = None) -> None:
    publish_metric(name=name, value=float(inc), dimensions=dimensions, unit="count")


def gauge(
    name: str,
    value: float,
    dimensions: Optional[Mapping[str, str]] = None,
    unit: Optional[str] = None,
) -> None:
    publish_metric(name=name, value=float(value), dimensions=dimensions, unit=unit)


@contextmanager
def timeit(name: str, dimensions: Optional[Mapping[str, str]] = None, unit: str = "ms"):
    """
    Measure wall-clock time for the enclosed block and emit a gauge metric in milliseconds.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dur_ms = (time.perf_counter() - t0) * 1000.0
        gauge(name=name, value=dur_ms, dimensions=dimensions, unit=unit)


"""
Usage:
  from agent_obs.metrics import gauge, counter, timeit, publish_metric
  counter("agent.event.received", 1, {"agent": "coding", "event": event_name})
  with timeit("agent.llm.duration_ms", {"agent": "coding", "model": "gpt"}):
      ... call LLM ...
  t0 = time.perf_counter()
  ... call service ...
  gauge("service.si.medicalcoding.duration_ms",
        (time.perf_counter() - t0) * 1000.0, {"agent": "coding"})
This module intentionally delegates entirely to cda_caim_sdk.core.monitoring._log_metric
as requested. No local fallbacks are implemented.
"""
