from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Mapping

import requests

from .schema import RetrievalRequestModel, RetrievalResponseModel
from .service import RetrievalService


class RagClient(ABC):
    @abstractmethod
    def retrieve(self, request: RetrievalRequestModel) -> RetrievalResponseModel:
        raise NotImplementedError


def _normalize_local_config(config: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    merged.setdefault("feature_name", "G2211")
    merged.setdefault("artifact_selection", "latest_registry")
    if merged.get("per_label_limit") is not None:
        merged.setdefault("positive_example_limit", merged["per_label_limit"])
        merged.setdefault("negative_example_limit", merged["per_label_limit"])
    merged.setdefault("retrieval_top_k", 4)
    merged.setdefault("retrieval_candidate_pool_size", max(int(merged["retrieval_top_k"]) * 6, int(merged["retrieval_top_k"])))
    merged.setdefault("positive_example_limit", 2)
    merged.setdefault("negative_example_limit", 2)
    return merged


@lru_cache(maxsize=8)
def _build_cached_local_service(config_blob: str) -> RetrievalService:
    return RetrievalService(_normalize_local_config(json.loads(config_blob)))


class LocalRagClient(RagClient):
    def __init__(self, config: Mapping[str, Any]):
        self._config = dict(config)

    def retrieve(self, request: RetrievalRequestModel) -> RetrievalResponseModel:
        merged_config = {
            **self._config,
            "feature_name": request.feature_name,
            "target_code": request.target_code,
            **dict(request.overrides or {}),
        }
        config_blob = json.dumps(merged_config, sort_keys=True, default=str)
        service = _build_cached_local_service(config_blob)
        return service.retrieve(request)


class HttpRagClient(RagClient):
    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def retrieve(self, request: RetrievalRequestModel) -> RetrievalResponseModel:
        response = requests.post(
            f"{self.base_url}/retrieve",
            json=request.model_dump(mode="json"),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return RetrievalResponseModel.model_validate(response.json())


def create_rag_client(config: Mapping[str, Any] | None = None) -> RagClient:
    cfg = dict(config or {})
    base_url = str(cfg.get("api_url") or os.getenv("HCPCS_RAG_API_URL") or "").strip()
    if base_url:
        timeout_seconds = float(cfg.get("timeout_seconds", 30.0))
        return HttpRagClient(base_url=base_url, timeout_seconds=timeout_seconds)
    return LocalRagClient(config=cfg)


__all__ = [
    "HttpRagClient",
    "LocalRagClient",
    "RagClient",
    "create_rag_client",
]
