from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from loguru import logger

from .artifacts import ArtifactSelection, resolve_artifact
from .profiles import RetrievalProfile, resolve_retrieval_profile
from .ranking import build_query_signals, rerank_examples
from .schema import (
    PromptInjectionExampleModel,
    PromptInjectionPayloadModel,
    RagEncounterPayloadModel,
    RetrievalDiagnosticsModel,
    RetrievalRequestModel,
    RetrievalResponseModel,
    RetrievedEncounterModel,
    SummaryBundleModel,
)
from .summaries import build_summary_bundle
from .vector_store import embed_text, load_vector_store, search_vector_store

AMBULATORY_ROUTE = "ambulatory"
FACILITY_ROUTE = "facility"


def route_encounter(encounter_metadata: Mapping[str, Any] | None) -> str:
    encounter_type = ""
    if isinstance(encounter_metadata, Mapping):
        encounter_type = str(encounter_metadata.get("type") or "")
    encounter_type = encounter_type.strip().lower()
    if encounter_type in {"inpatient", "hospital", "ed", "emergency", "emergency department"}:
        return FACILITY_ROUTE
    return AMBULATORY_ROUTE


def _resolve_example_limits(config: Mapping[str, Any], profile: RetrievalProfile) -> tuple[int, int, int]:
    legacy_limit = config.get("per_label_limit")
    positive_limit = config.get("positive_example_limit")
    negative_limit = config.get("negative_example_limit")

    if legacy_limit is not None:
        if positive_limit is None:
            positive_limit = legacy_limit
        if negative_limit is None:
            negative_limit = legacy_limit

    positive = int(positive_limit or profile.positive_example_limit)
    negative = int(negative_limit or profile.negative_example_limit)
    max_examples = int(config.get("retrieval_top_k") or profile.max_examples)
    return max_examples, positive, negative


@lru_cache(maxsize=8)
def _load_cached_store(index_dir: str):
    return load_vector_store(index_dir)


class RetrievalService:
    def __init__(self, config: Mapping[str, Any] | None = None):
        self._config = dict(config or {})

    def _resolve_profile(self, request: RetrievalRequestModel, *, config: Mapping[str, Any]) -> RetrievalProfile:
        return resolve_retrieval_profile(
            feature_name=request.feature_name,
            target_code=request.target_code,
            candidate_codes=config.get("candidate_codes"),
        )

    def _merged_config(self, request: RetrievalRequestModel) -> dict[str, Any]:
        merged = dict(self._config)
        merged.update(dict(request.overrides or {}))
        return merged

    def _build_prompt_payload(
        self,
        *,
        profile: RetrievalProfile,
        retrieved_examples: list[RetrievedEncounterModel],
    ) -> PromptInjectionPayloadModel:
        return PromptInjectionPayloadModel(
            target_code=profile.target_code,
            guidance=profile.prompt_guidance,
            examples=[
                PromptInjectionExampleModel(
                    label=example.label,
                    encounter_summary=example.encounter_summary,
                    teaching_justification=example.teaching_justification,
                    matched_signals=example.matched_signals,
                )
                for example in retrieved_examples
            ],
        )

    def _build_missing_artifact_response(
        self,
        *,
        profile: RetrievalProfile,
        encounter_route: str,
        query_summary: dict[str, Any],
        query_signals: list[str],
    ) -> RetrievalResponseModel:
        return RetrievalResponseModel(
            feature_name=profile.name,
            feature_code=profile.code,
            target_code=profile.target_code,
            encounter_route=encounter_route,
            retrieval_status="missing_artifact",
            retrieval_status_reason=f"No retrieval artifact is available for {profile.target_code}.",
            query_summary=SummaryBundleModel(**query_summary),
            query_signals=query_signals,
            retrieved_encounters=[],
            label_distribution={},
            diagnostics=RetrievalDiagnosticsModel(
                pool_size=0,
                selected_count=0,
                positive_example_limit=profile.positive_example_limit,
                negative_example_limit=profile.negative_example_limit,
            ),
            prompt_payload=self._build_prompt_payload(profile=profile, retrieved_examples=[]),
        )

    def _resolve_query_summary(
        self,
        *,
        request: RetrievalRequestModel,
        artifact: ArtifactSelection | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        cache_path = None
        if artifact and artifact.summary_cache_path:
            cache_path = str(artifact.summary_cache_path)
        elif config.get("summary_cache_path"):
            cache_path = str(config["summary_cache_path"])

        return build_summary_bundle(
            example_id=request.encounter.example_id,
            text=request.encounter.text,
            cache_path=cache_path,
            additional_input=request.encounter.additional_input,
        )

    def _resolve_artifact(self, *, config: dict[str, Any], profile: RetrievalProfile) -> ArtifactSelection | None:
        return resolve_artifact(config, target_code=profile.target_code)

    def _build_candidates(
        self,
        *,
        artifact: ArtifactSelection,
        request: RetrievalRequestModel,
        query_summary: dict[str, Any],
        query_signals: list[str],
        config: dict[str, Any],
        profile: RetrievalProfile,
    ) -> tuple[list[dict[str, Any]], RetrievalDiagnosticsModel]:
        pool_size = int(config.get("retrieval_candidate_pool_size") or 24)
        embedding_proxy = config.get("embedding_proxy")
        _, positive_limit, negative_limit = _resolve_example_limits(config, profile)

        model_name = str(config.get("embedding_model_override") or artifact.manifest.embedding_model)
        if (
            "registry" in artifact.source
            and config.get("embedding_model_override")
            and str(config["embedding_model_override"]) != str(artifact.manifest.embedding_model)
        ):
            raise ValueError(
                "embedding_model_override does not match the registry-selected artifact embedding model "
                f"({config['embedding_model_override']} != {artifact.manifest.embedding_model})."
            )

        store = _load_cached_store(str(artifact.index_dir))
        query_vector = embed_text(
            query_summary.get("summary_compact") or request.encounter.text,
            model_name=model_name,
            proxy=embedding_proxy,
        )
        raw_hits = search_vector_store(store, query_vector, top_k=pool_size + 1)

        query_id = str(request.encounter.example_id or "")
        candidates: list[dict[str, Any]] = []
        for hit in raw_hits:
            metadata = dict(hit.metadata or {})
            encounter_id = str(metadata.get("encounter_id") or metadata.get("example_id") or metadata.get("record_id") or "")
            example_id = str(metadata.get("example_id") or encounter_id or metadata.get("record_id") or "")
            if query_id and query_id in {encounter_id, example_id}:
                continue

            label = profile.normalize_label(metadata.get("label") or metadata.get("g_code_label") or metadata.get("target_code_label"))
            encounter_summary = str(metadata.get("encounter_summary") or metadata.get("summary_compact") or "").strip()
            teaching_justification = str(metadata.get("teaching_justification") or metadata.get("decision_contribution") or "").strip()
            candidates.append(
                {
                    "encounter_id": encounter_id or None,
                    "example_id": example_id or None,
                    "target_code": str(metadata.get("target_code") or metadata.get("g_code") or profile.target_code),
                    "label": label,
                    "encounter_source": metadata.get("encounter_source"),
                    "encounter_summary": encounter_summary,
                    "teaching_justification": teaching_justification,
                    "candidate_signals": list(metadata.get("signals") or []),
                    "metadata": metadata,
                    "dense_score": float(hit.score),
                }
            )
            if len(candidates) >= pool_size:
                break

        diagnostics = RetrievalDiagnosticsModel(
            pool_size=len(candidates),
            dense_score_min=min((candidate["dense_score"] for candidate in candidates), default=None),
            dense_score_max=max((candidate["dense_score"] for candidate in candidates), default=None),
            per_label_limit=max(positive_limit, negative_limit) or None,
            positive_example_limit=positive_limit,
            negative_example_limit=negative_limit,
            artifact_build_id=artifact.build_id,
            artifact_source=artifact.source,
            embedding_model=model_name,
        )
        return candidates, diagnostics

    def retrieve(self, request: RetrievalRequestModel) -> RetrievalResponseModel:
        config = self._merged_config(request)
        profile = self._resolve_profile(request, config=config)
        encounter_route = route_encounter(request.encounter.encounter_metadata)

        if profile.allowed_routes and encounter_route not in profile.allowed_routes:
            return RetrievalResponseModel(
                feature_name=profile.name,
                feature_code=profile.code,
                target_code=profile.target_code,
                encounter_route=encounter_route,
                retrieval_status="not_applicable",
                retrieval_status_reason=(f"{profile.target_code} retrieval is configured only for {', '.join(profile.allowed_routes)} encounters."),
                query_summary=SummaryBundleModel(),
                query_signals=[],
                retrieved_encounters=[],
                label_distribution={},
                diagnostics=RetrievalDiagnosticsModel(
                    pool_size=0,
                    selected_count=0,
                    positive_example_limit=profile.positive_example_limit,
                    negative_example_limit=profile.negative_example_limit,
                ),
                prompt_payload=self._build_prompt_payload(profile=profile, retrieved_examples=[]),
            )

        artifact = self._resolve_artifact(config=config, profile=profile)
        query_summary = self._resolve_query_summary(request=request, artifact=artifact, config=config)
        query_signals = build_query_signals(
            profile=profile,
            query_summary=query_summary,
            encounter=request.encounter,
            encounter_route=encounter_route,
        )

        if artifact is None:
            return self._build_missing_artifact_response(
                profile=profile,
                encounter_route=encounter_route,
                query_summary=query_summary,
                query_signals=query_signals,
            )

        try:
            candidates, diagnostics = self._build_candidates(
                artifact=artifact,
                request=request,
                query_summary=query_summary,
                query_signals=query_signals,
                config=config,
                profile=profile,
            )
            max_examples, positive_limit, negative_limit = _resolve_example_limits(config, profile)
            selected_payload, label_distribution = rerank_examples(
                candidates=candidates,
                query_signals=query_signals,
                profile=profile,
                max_examples=max_examples,
                positive_limit=positive_limit,
                negative_limit=negative_limit,
            )
        except Exception as exc:  # pragma: no cover - requires runtime deps
            logger.error("HCPCS retrieval failed for target_code={} example_id={}: {}", profile.target_code, request.encounter.example_id, exc)
            return RetrievalResponseModel(
                feature_name=profile.name,
                feature_code=profile.code,
                target_code=profile.target_code,
                encounter_route=encounter_route,
                retrieval_status="error",
                retrieval_status_reason=str(exc),
                query_summary=SummaryBundleModel(**query_summary),
                query_signals=query_signals,
                retrieved_encounters=[],
                label_distribution={},
                diagnostics=RetrievalDiagnosticsModel(
                    pool_size=0,
                    selected_count=0,
                    positive_example_limit=_resolve_example_limits(config, profile)[1],
                    negative_example_limit=_resolve_example_limits(config, profile)[2],
                    artifact_build_id=artifact.build_id,
                    artifact_source=artifact.source,
                ),
                prompt_payload=self._build_prompt_payload(profile=profile, retrieved_examples=[]),
            )

        retrieved_encounters = [RetrievedEncounterModel(**payload) for payload in selected_payload]
        diagnostics.selected_count = len(retrieved_encounters)
        _, positive_limit, negative_limit = _resolve_example_limits(config, profile)
        diagnostics.positive_example_limit = positive_limit
        diagnostics.negative_example_limit = negative_limit
        diagnostics.per_label_limit = max(diagnostics.positive_example_limit or 0, diagnostics.negative_example_limit or 0) or None

        return RetrievalResponseModel(
            feature_name=profile.name,
            feature_code=profile.code,
            target_code=profile.target_code,
            encounter_route=encounter_route,
            retrieval_status="ok",
            retrieval_status_reason=None,
            query_summary=SummaryBundleModel(**query_summary),
            query_signals=query_signals,
            retrieved_encounters=retrieved_encounters,
            label_distribution=label_distribution,
            diagnostics=diagnostics,
            prompt_payload=self._build_prompt_payload(
                profile=profile,
                retrieved_examples=retrieved_encounters,
            ),
        )

    def retrieve_payload(self, payload: Mapping[str, Any], *, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        base_feature = (
            (overrides or {}).get("feature_name")
            or (overrides or {}).get("target_code")
            or self._config.get("feature_name")
            or self._config.get("target_code")
            or "G2211"
        )
        request = RetrievalRequestModel(
            feature_name=str(base_feature),
            encounter=RagEncounterPayloadModel.model_validate(dict(payload)),
            overrides=dict(overrides or {}),
        )
        return self.retrieve(request).model_dump()


__all__ = [
    "AMBULATORY_ROUTE",
    "FACILITY_ROUTE",
    "RetrievalService",
    "route_encounter",
]
