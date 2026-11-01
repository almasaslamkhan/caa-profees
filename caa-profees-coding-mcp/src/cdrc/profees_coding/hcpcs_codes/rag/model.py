"""Compatibility wrapper around the HCPCS retrieval service.

The core runtime no longer uses LangGraph. `build_rag_graph` now returns a
small adapter object with an `invoke` method so existing call sites can migrate
without a hard cut-over.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Mapping

from .schema import RagEncounterPayloadModel, RetrievalRequestModel
from .service import RetrievalService


class RetrievalGraphAdapter:
    def __init__(self, config: Mapping[str, Any]):
        self._config = dict(config)
        self._service = RetrievalService(self._config)

    def invoke(self, payload: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        configurable = dict(self._config)
        if config and isinstance(config, Mapping):
            configurable.update(dict(config.get("configurable") or {}))
        feature_name = str(configurable.get("feature_name") or configurable.get("target_code") or configurable.get("candidate_codes") or "G2211")
        request = RetrievalRequestModel(
            feature_name=feature_name,
            encounter=RagEncounterPayloadModel.model_validate(dict(payload)),
            overrides=configurable,
        )
        return self._service.retrieve(request).model_dump()


def build_rag_graph(args: Namespace) -> tuple[RetrievalGraphAdapter, dict[str, Any]]:
    config = vars(args)
    return RetrievalGraphAdapter(config), {"configurable": config}


def build_retrieval_graph(args: Namespace) -> tuple[RetrievalGraphAdapter, dict[str, Any]]:
    return build_rag_graph(args)


__all__ = [
    "RetrievalGraphAdapter",
    "build_rag_graph",
    "build_retrieval_graph",
]
