import argparse
import gc
import os
import time
import tracemalloc
import uuid
from typing import Final

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger

from cdrc.common.llm_utils import TokenStats, TokenTracker
from cdrc.oci_services.metrics_client import MetricsClient, emit, measure_time
from cdrc.profees_coding.deploy.api.model.app_gen1 import ProfeesModel
from cdrc.profees_coding.deploy.api.utils import DeliveryVehicle, InputData, MetricNames

MODES: Final = ("SDS", "LOCAL", "DEV", "PROD")
BYTES_PER_GB: Final = 1024 * 1024 * 1024
SOFT_GC_THRESHOLD_BYTES: Final = int(7 * BYTES_PER_GB)
HARD_GC_THRESHOLD_BYTES: Final = int(7.5 * BYTES_PER_GB)
_PROCESS = psutil.Process(os.getpid())

# curl -X POST http://0.0.0.0:8002/predict -H 'Content-Type: application/json' --data @cdrc/profees_coding/deploy/api/model/payload.json


def resolve_mode() -> str:
    """Resolve MODE_PROFEES with a sensible default and validation."""
    mode = os.getenv("MODE_PROFEES") or ("SDS" if os.path.exists("/sds") else "LOCAL")
    mode = mode.upper()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode}")
    return mode


def resolve_log_level() -> str:
    """Allow LOG_LEVEL override while keeping the default used in test.py."""
    return os.getenv("LOG_LEVEL", "info")


def build_model(log_level: str | None = None) -> ProfeesModel:
    """Factory so tests can swap logging without touching globals."""
    return ProfeesModel(
        delivery_vehicle=DeliveryVehicle.SCRATCHPAD,
        mode=resolve_mode(),
        log_level=log_level or resolve_log_level(),
    )


app = FastAPI(title="ProFees Model API", description="Async wrapper around ProfeesModel.predict", version="1.0.0")
model = build_model()


def _rss_bytes() -> int:
    try:
        return _PROCESS.memory_info().rss
    except Exception as err:
        logger.warning("Failed to read process RSS memory: {}", err)
        return 0


def run_memory_gc_guard() -> str:
    """
    Run tiered GC based on RSS memory:
    - soft GC when RSS >= 7 GB
    - hard GC when RSS >= 7.5 GB after soft GC
    """
    rss_before = _rss_bytes()
    if rss_before < SOFT_GC_THRESHOLD_BYTES:
        return "none"

    rss_before_gb = rss_before / BYTES_PER_GB
    logger.warning(
        "RSS {:.2f} GB crossed soft threshold {:.2f} GB. Running soft gc.collect().",
        rss_before_gb,
        SOFT_GC_THRESHOLD_BYTES / BYTES_PER_GB,
    )
    emit(MetricNames.API_GC_SOFT_TRIGGERED, value=1)
    soft_unreachable = gc.collect()
    rss_after_soft = _rss_bytes()
    rss_after_soft_gb = rss_after_soft / BYTES_PER_GB
    logger.warning(
        "Soft gc.collect() complete. unreachable={}, rss_after_soft={:.2f} GB",
        soft_unreachable,
        rss_after_soft_gb,
    )

    if rss_after_soft < HARD_GC_THRESHOLD_BYTES:
        return "soft"

    logger.error(
        "RSS {:.2f} GB crossed hard threshold {:.2f} GB after soft GC. Running hard gc.collect(2).",
        rss_after_soft_gb,
        HARD_GC_THRESHOLD_BYTES / BYTES_PER_GB,
    )
    emit(MetricNames.API_GC_HARD_TRIGGERED, value=1)
    hard_unreachable = gc.collect(2)
    rss_after_hard = _rss_bytes()
    logger.error(
        "Hard gc.collect(2) complete. unreachable={}, rss_after_hard={:.2f} GB",
        hard_unreachable,
        rss_after_hard / BYTES_PER_GB,
    )
    return "hard"


@app.post("/predict")
async def predict(data: InputData):
    """Run a single prediction mirroring cdrc/.../model/test.py behaviour."""

    request_tracker_id = f"req_{uuid.uuid4()}_worker_{os.getpid()}"
    emit(MetricNames.ORDERS_RECEIVED, value=len(data.orders))
    emit(MetricNames.API_REQUEST_RECEIVED, value=1)
    start_time = time.perf_counter()
    try:
        response, stats = await model.predict(data, request_id=request_tracker_id)
        emit_token_stats(stats)
        emit(MetricNames.API_REQUEST_SUCCESS, value=1)
        return response.model_dump(mode="json")
    except HTTPException as http_exc:
        # Log stats if possible before re-raising
        try:
            error_stats = TokenTracker.get_stats(request_tracker_id)
            logger.error(f"HTTPException during prediction for tracker ID {request_tracker_id}. Stats captured: {error_stats}")
        except Exception as stat_err:
            logger.error(f"Could not retrieve stats during HTTPException handling for {request_tracker_id}: {stat_err}")
        raise http_exc
    except Exception as e:
        # Log stats if possible before raising internal error
        try:
            error_stats = TokenTracker.get_stats(request_tracker_id)
            logger.error(f"Exception during prediction for tracker ID {request_tracker_id}. Stats captured: {error_stats}")
        except Exception as stat_err:
            logger.error(f"Could not retrieve stats during Exception handling for {request_tracker_id}: {stat_err}")
        logger.error("ERROR during prediction for tracker ID {}: {}", request_tracker_id, str(e), exc_info=True)
        emit(MetricNames.API_REQUEST_SERVER_ERROR, value=1)
        # emit(MetricNames.API_REQUEST_ERROR_MESSAGE, value=str(e))
        raise HTTPException(status_code=500, detail="Profees Api error: An internal server error occurred.")
    finally:
        # --- Crucial Cleanup Step to avoid memory leaks ---
        # Ensure the instance associated with this request ID is removed
        TokenTracker.remove_instance(request_tracker_id)
        # Calculate total time and emit the metric
        elapsed_time = time.perf_counter() - start_time
        emit(MetricNames.API_REQUEST_TIME, value=elapsed_time)
        try:
            run_memory_gc_guard()
        except Exception as gc_err:
            logger.error("GC guard failed: {}", gc_err)


@app.get("/health")
async def health():
    return {"status": "ok"}


def emit_token_stats(stats: TokenStats):
    emit(MetricNames.LLM_TOTAL_TOKENS, value=stats.total_tokens)
    emit(MetricNames.LLM_SUCCESSFUL_REQUESTS, value=stats.successful_requests)
    emit(MetricNames.LLM_TOTAL_COST, value=stats.total_cost)
    emit(MetricNames.LLM_PROMPT_TOKENS, value=stats.prompt_tokens)
    emit(MetricNames.LLM_COMPLETION_TOKENS, value=stats.completion_tokens)
    emit(MetricNames.LLM_PROMPT_TOKENS_CACHED, value=stats.prompt_tokens_cached)
    emit(MetricNames.LLM_REASONING_TOKENS, value=stats.reasoning_tokens)
    emit(MetricNames.LLM_REQUEST_TIME, value=sum(stats.requests_time_in_seconds) if stats.requests_time_in_seconds else 0)
    emit(MetricNames.LLM_RATE_LIMIT_DELAYS, value=sum(stats.rate_limit_delays) if stats.rate_limit_delays else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ProFees FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8002, help="Port to run the API server (default: 8002)")
    parser.add_argument(
        "--log-level",
        "--log_level",
        dest="log_level",
        default=None,
        help="Override model log level (default: env LOG_LEVEL or info)",
    )
    cli_args, extra = parser.parse_known_args()

    if extra:
        logger.debug("Ignoring unexpected CLI arguments: %s", extra)

    if cli_args.log_level:
        model = build_model(cli_args.log_level)

    tracemalloc.start()
    MetricsClient.activate()
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)
