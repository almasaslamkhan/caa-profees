from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Sequence

from loguru import logger

try:
    import aad_agentlens as al
    from aad_agentlens.config import get_config as _get_agentlens_runtime_config
    from aad_agentlens.config import is_enabled as _agentlens_is_enabled
    from aad_agentlens.instrumentation.langchain import AgentLensCallbackHandler
    from aad_agentlens.transports.base import BaseTransport
    from aad_agentlens.transports.http import HTTPTransport
except Exception:  # pragma: no cover - optional dependency safety
    al = None
    AgentLensCallbackHandler = None
    BaseTransport = object
    HTTPTransport = None

    def _get_agentlens_runtime_config():
        return None

    def _agentlens_is_enabled() -> bool:
        return False


_INITIALIZED = False
_ACTIVE = False


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class _FailOpenHTTPTransport(BaseTransport):  # type: ignore[misc]
    """Wrap HTTP transport and open a local circuit after repeated send failures."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        timeout_s: float,
        max_attempts: int,
        base_backoff_ms: int,
        max_backoff_ms: int,
        failure_threshold: int,
    ) -> None:
        self._delegate = HTTPTransport(
            endpoint=endpoint,
            api_key=api_key,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            base_backoff_ms=base_backoff_ms,
            max_backoff_ms=max_backoff_ms,
        )
        self._endpoint = endpoint
        self._failure_threshold = max(1, failure_threshold)
        self._consecutive_failures = 0
        self._tripped = False

    def _on_result(self, ok: bool) -> bool:
        if ok:
            self._consecutive_failures = 0
            return True

        self._consecutive_failures += 1
        if not self._tripped and self._consecutive_failures >= self._failure_threshold:
            self._tripped = True
            cfg = _get_agentlens_runtime_config()
            if cfg is not None:
                cfg.enabled = False
            globals()["_ACTIVE"] = False
            logger.warning(
                "AgentLens transport disabled after {} consecutive send failures to {}. " "Fix endpoint/service and restart process to re-enable.",
                self._consecutive_failures,
                self._endpoint,
            )
        return ok

    def send_batch(self, spans: list[dict[str, Any]], sent_at_unix_ms: int) -> bool:
        if self._tripped:
            return True
        return self._on_result(self._delegate.send_batch(spans, sent_at_unix_ms))

    def send_feedback_batch(self, feedback: list[dict[str, Any]], sent_at_unix_ms: int) -> bool:
        if self._tripped:
            return True
        return self._on_result(self._delegate.send_feedback_batch(feedback, sent_at_unix_ms))

    def close(self) -> None:
        self._delegate.close()


def _build_agentlens_transport(endpoint: str, api_key: str):
    if HTTPTransport is None:
        return None

    timeout_s = _float_env("AGENTLENS_REQUEST_TIMEOUT_S", 2.0)
    max_attempts = max(1, _int_env("AGENTLENS_RETRY_MAX_ATTEMPTS", 5))
    base_backoff_ms = max(1, _int_env("AGENTLENS_RETRY_BASE_MS", 100))
    max_backoff_ms = max(base_backoff_ms, _int_env("AGENTLENS_RETRY_MAX_MS", 2000))
    fail_open = _is_truthy(os.getenv("AGENTLENS_FAIL_OPEN"), default=True)

    if not fail_open:
        return HTTPTransport(
            endpoint=endpoint,
            api_key=api_key,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            base_backoff_ms=base_backoff_ms,
            max_backoff_ms=max_backoff_ms,
        )

    failure_threshold = max(1, _int_env("AGENTLENS_FAIL_OPEN_MAX_FAILURES", 3))
    return _FailOpenHTTPTransport(
        endpoint=endpoint,
        api_key=api_key,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
        base_backoff_ms=base_backoff_ms,
        max_backoff_ms=max_backoff_ms,
        failure_threshold=failure_threshold,
    )


def _is_truthy(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_agentlens_config(default_project: str) -> tuple[str, str, str, str, bool, str | None]:
    from cdrc.config import settings

    agentlens_settings = settings.get("agentlens", {}) or {}

    endpoint_default = agentlens_settings.get("endpoint") or "http://127.0.0.1:62341"
    api_key_default = agentlens_settings.get("api_key")
    project_default = agentlens_settings.get("project") or default_project
    env_default = agentlens_settings.get("env") or "dev"
    enabled_default = _is_truthy(agentlens_settings.get("enabled"), default=True)
    run_name_default = agentlens_settings.get("run_name", None)

    endpoint = os.getenv("AGENTLENS_ENDPOINT", endpoint_default)
    api_key = os.getenv("AGENTLENS_API_KEY", "" if api_key_default is None else str(api_key_default))
    project = os.getenv("AGENTLENS_PROJECT", str(project_default))
    env = os.getenv("AGENTLENS_ENV", str(env_default))
    enabled = _is_truthy(os.getenv("AGENTLENS_ENABLED"), default=enabled_default)
    run_name = os.getenv("AGENTLENS_RUN_NAME", None if run_name_default is None else str(run_name_default))
    return endpoint, api_key, project, env, enabled, run_name


def ensure_agentlens_initialized(default_project: str = "clinical-rev-cycle") -> bool:
    global _INITIALIZED, _ACTIVE

    if _INITIALIZED:
        _ACTIVE = _agentlens_is_enabled()
        return _ACTIVE

    _INITIALIZED = True

    if al is None:
        logger.info("AgentLens SDK is not importable; observability hooks are disabled.")
        _ACTIVE = False
        return False

    endpoint, api_key, project, env, enabled, run_name = _get_agentlens_config(default_project)
    transport = _build_agentlens_transport(endpoint, api_key)

    al.init(
        endpoint=endpoint,
        api_key=api_key,
        project=project,
        env=env,
        enabled=enabled,
        transport=transport,
        run_name=run_name,
    )
    _ACTIVE = _agentlens_is_enabled()

    if _ACTIVE:
        logger.info(
            "AgentLens enabled | endpoint={} project={} env={} run_name={}",
            endpoint,
            project,
            env,
            run_name,
        )
    else:
        logger.warning("AgentLens initialized but inactive. Set AGENTLENS_ENDPOINT/AGENTLENS_PROJECT (and API key for non-loopback endpoints).")

    return _ACTIVE


def build_langchain_callbacks(
    existing_callbacks: Sequence[Any] | None = None,
    *,
    trace_intermediate_steps: bool = False,
) -> list[Any]:
    callbacks, _ = attach_agentlens_callback(
        existing_callbacks,
        trace_intermediate_steps=trace_intermediate_steps,
    )
    return callbacks


def attach_agentlens_callback(
    existing_callbacks: Sequence[Any] | None = None,
    *,
    trace_intermediate_steps: bool = False,
) -> tuple[list[Any], Any | None]:
    callbacks = list(existing_callbacks or [])

    if not ensure_agentlens_initialized():
        return callbacks, None

    if AgentLensCallbackHandler is None:
        return callbacks, None

    for callback in callbacks:
        if isinstance(callback, AgentLensCallbackHandler):
            return callbacks, callback

    handler = AgentLensCallbackHandler(
        trace_intermediate_steps=trace_intermediate_steps,
    )
    callbacks.append(handler)
    return callbacks, handler


def start_agentlens_trace(
    name: str,
    *,
    conversation_id: str | None = None,
    correlation: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    if not ensure_agentlens_initialized():
        return nullcontext()

    if al is None or not _agentlens_is_enabled():
        return nullcontext()

    return al.start_trace(
        name=name,
        conversation_id=conversation_id,
        correlation=correlation,
        metadata=metadata,
        tags=tags,
    )


def flush_agentlens(timeout_s: float = 5.0) -> None:
    if al is None:
        return
    if not ensure_agentlens_initialized():
        return
    try:
        al.flush(timeout_s=timeout_s)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("AgentLens flush failed: {}", exc)


def _serialize_feedback_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()

    if isinstance(value, dict):
        return {str(key): _serialize_feedback_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_serialize_feedback_value(item) for item in value]

    if isinstance(value, tuple):
        return [_serialize_feedback_value(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def log_gold_feedback(
    *,
    target_trace_id: str | None = None,
    target_span_id: str | None = None,
    expected: Any,
    metadata: dict[str, Any] | None = None,
    comment: str = "Imported from eval gold set",
    feedback_name: str = "gold_label",
) -> None:
    if al is None:
        return
    if not ensure_agentlens_initialized():
        return
    if not _agentlens_is_enabled():
        return
    if not target_trace_id and not target_span_id:
        return

    al.log_feedback(
        target_trace_id=target_trace_id,
        target_span_id=target_span_id,
        expected=_serialize_feedback_value(expected),
        comment=comment,
        metadata=metadata,
        source="gold",
        feedback_name=feedback_name,
    )
