from __future__ import annotations

from argparse import Namespace
from typing import Any, Mapping

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.hcpcs_codes.models.common import (
    AMBULATORY_ROUTE,
    FACILITY_ROUTE,
    HCPCSGraphState,
    HCPCSInputGraphState,
    HCPCSOutputGraphState,
    apply_input_ablation_mode,
    is_hcpcs_family_enabled,
    is_hcpcs_family_enabled_for_config,
    merge_results,
    route_encounter,
)
from cdrc.profees_coding.hcpcs_codes.models.utils import CONFIG
from cdrc.profees_coding.hcpcs_codes.rag.clients import RagClient, create_rag_client
from cdrc.profees_coding.hcpcs_codes.rag.schema import RagEncounterPayloadModel, RetrievalRequestModel, RetrievalResponseModel

from .constants import HCPCS_AWV_PROMPT_VERSION, HCPCS_BRANCH_NAME, HCPCS_G2211_PROMPT_VERSION, HCPCS_OTHER_PROMPT_VERSION
from .nodes.awv_eligibility import awv_eligibility_precheck, awv_eligibility_precheck_deterministic
from .nodes.hcpcs_code import hcpcs_code


def _branch_config(
    config: RunnableConfig,
    *,
    branch_name: str,
    prompt_config_key: str,
    default_prompt_version: Any,
) -> RunnableConfig:
    branch_config = dict(config or {})
    configurable = dict(branch_config.get("configurable") or {})
    configurable[CONFIG.PROMPT_VERSION] = configurable.get(prompt_config_key, default_prompt_version)
    configurable[HCPCS_BRANCH_NAME] = branch_name
    branch_config["configurable"] = configurable
    return branch_config


def _run_branch(
    state: Mapping[str, Any],
    config: RunnableConfig,
    *,
    result_key: str,
    branch_name: str,
    prompt_config_key: str,
    default_prompt_version: Any,
    allowed_route: str,
) -> dict[str, Any]:
    if not is_hcpcs_family_enabled_for_config(config, branch_name):
        return {}
    if state.get("encounter_route") != allowed_route:
        return {}

    result = hcpcs_code(
        apply_input_ablation_mode(state, config),
        _branch_config(
            config,
            branch_name=branch_name,
            prompt_config_key=prompt_config_key,
            default_prompt_version=default_prompt_version,
        ),
    )
    return {result_key: result}


def run_g2211_classifier(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
    return _run_branch(
        state,
        config,
        result_key="g2211_result",
        branch_name="g2211",
        prompt_config_key=HCPCS_G2211_PROMPT_VERSION,
        default_prompt_version="v4.4",
        allowed_route=AMBULATORY_ROUTE,
    )


def run_awv_eligibility_precheck(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
    if not is_hcpcs_family_enabled_for_config(config, "awv"):
        return {}
    if state.get("encounter_route") != AMBULATORY_ROUTE:
        return {}
    return awv_eligibility_precheck(state, config)


def run_awv_eligibility_precheck_deterministic(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
    if not is_hcpcs_family_enabled_for_config(config, "awv"):
        return {}
    if state.get("encounter_route") != AMBULATORY_ROUTE:
        return {}
    return awv_eligibility_precheck_deterministic(state, config)


def run_awv_classifier(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
    return _run_branch(
        state,
        config,
        result_key="awv_result",
        branch_name="awv",
        prompt_config_key=HCPCS_AWV_PROMPT_VERSION,
        default_prompt_version={"version": "v5.1", "codes_supported": ["G0402", "G0438", "G0439"]},
        allowed_route=AMBULATORY_ROUTE,
    )


def _get_awv_precheck_mode(config: Mapping[str, Any] | Namespace | None) -> str:
    if isinstance(config, Mapping):
        raw_mode = config.get(CONFIG.AWV_PRECHECK_MODE) or config.get("awv_precheck_mode")
    else:
        raw_mode = getattr(config, CONFIG.AWV_PRECHECK_MODE, None) or getattr(config, "awv_precheck_mode", None)
    mode = str(raw_mode or "deterministic").strip().lower()
    if mode not in {"deterministic", "llm"}:
        logger.warning("Unsupported AWV precheck mode '{}'; falling back to deterministic", mode)
        return "deterministic"
    return mode


def _select_awv_precheck_node(config: Mapping[str, Any] | Namespace | None):
    mode = _get_awv_precheck_mode(config)
    return run_awv_eligibility_precheck if mode == "llm" else run_awv_eligibility_precheck_deterministic


def _route_after_awv_precheck(state: Mapping[str, Any]) -> str:
    if state.get("encounter_route") != AMBULATORY_ROUTE:
        return "merge_results"
    return "awv_classifier" if state.get("awv_precheck_passed", False) else "merge_results"


def run_other_hcpcs_classifier(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
    return _run_branch(
        state,
        config,
        result_key="other_hcpcs_result",
        branch_name="other_hcpcs",
        prompt_config_key=HCPCS_OTHER_PROMPT_VERSION,
        default_prompt_version={"version": "v6.1", "codes_supported": ["Other HCPCS"]},
        allowed_route=FACILITY_ROUTE,
    )


def _default_rag_config() -> dict[str, Any]:
    return {
        "feature_name": "G2211",
        "artifact_selection": "latest_registry",
        "artifact_root": "./cdrc/profees_coding/hcpcs_codes/artifacts/rag/g2211",
        "summary_cache_path": "./cdrc/profees_coding/hcpcs_codes/artifacts/rag/g2211/summaries.jsonl",
        "vdb_dir": "./cdrc/profees_coding/hcpcs_codes/artifacts/rag/g2211/index/oracle-text-embedding-3-large-1-dev-local",
        "retrieval_top_k": 4,
        "retrieval_candidate_pool_size": 24,
        "positive_example_limit": 2,
        "negative_example_limit": 2,
        "prompt_max_examples": 4,
    }


def _normalize_rag_config(args: Namespace) -> dict[str, Any]:
    rag_config = _default_rag_config()
    raw_rag_config = getattr(args, "rag", {}) or {}
    if isinstance(raw_rag_config, Mapping):
        rag_config.update(raw_rag_config)
    legacy_limit = rag_config.get("per_label_limit")
    if legacy_limit is not None:
        rag_config.setdefault("positive_example_limit", legacy_limit)
        rag_config.setdefault("negative_example_limit", legacy_limit)
    if "prompt_max_examples" not in rag_config and rag_config.get("retrieval_top_k") is not None:
        rag_config["prompt_max_examples"] = rag_config["retrieval_top_k"]
    return rag_config


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _format_retrieval_examples_for_prompt(response: RetrievalResponseModel, *, max_examples: int) -> str:
    payload = response.prompt_payload
    if payload is None or not payload.examples:
        return ""

    lines = [
        f"Retrieved analogies for {payload.target_code}:",
        payload.guidance,
    ]
    for index, example in enumerate(payload.examples[:max_examples], start=1):
        lines.append(f"{index}. Label: {example.label or '?'}")
        lines.append(f"   Summary: {example.encounter_summary}")
        lines.append(f"   Teaching justification: {example.teaching_justification}")
        if example.matched_signals:
            lines.append(f"   Matched signals: {', '.join(example.matched_signals)}")
    return "\n".join(lines).strip()


def _augment_state_with_retrieval_examples(
    state: Mapping[str, Any],
    response: RetrievalResponseModel,
    *,
    max_examples: int,
) -> dict[str, Any]:
    augmented_state = dict(state)
    retrieval_prompt_block = _format_retrieval_examples_for_prompt(response, max_examples=max_examples)

    retrieval_payload = {
        "target_code": response.target_code or response.feature_code,
        "query_summary": response.query_summary.summary_compact,
        "query_signals": list(response.query_signals or []),
        "retrieval_examples": [
            example.model_dump() for example in (response.prompt_payload.examples if response.prompt_payload else [])[:max_examples]
        ],
        "retrieval_guidance": response.prompt_payload.guidance if response.prompt_payload else "",
        "retrieval_prompt_block": retrieval_prompt_block,
    }

    additional_input = state.get("additional_input")
    if isinstance(additional_input, Mapping):
        merged_additional_input = dict(additional_input)
    elif additional_input is None:
        merged_additional_input = {}
    else:
        merged_additional_input = {"original_additional_input": additional_input}

    merged_additional_input["retrieval"] = retrieval_payload
    merged_additional_input["retrieval_examples"] = retrieval_payload["retrieval_examples"]
    merged_additional_input["retrieval_prompt_block"] = retrieval_prompt_block
    augmented_state["additional_input"] = merged_additional_input
    return augmented_state


def _rag_request_from_state(state: Mapping[str, Any], rag_config: Mapping[str, Any]) -> RetrievalRequestModel:
    return RetrievalRequestModel(
        feature_name=str(rag_config.get("feature_name") or "G2211"),
        encounter=RagEncounterPayloadModel(
            example_id=state.get("example_id"),
            text=str(state.get("text") or ""),
            patient_sex=state.get("patient_sex"),
            patient_age=state.get("patient_age"),
            orders_display=state.get("orders_display"),
            health_plan=state.get("health_plan"),
            awv_history=state.get("awv_history"),
            encounter_date=state.get("encounter_date"),
            encounter_metadata=state.get("encounter_metadata"),
            all_patient_problems=state.get("all_patient_problems"),
            additional_input=state.get("additional_input"),
        ),
        overrides={},
    )


def _attach_rag_payload(branch_output: dict[str, Any], rag_payload: dict[str, Any]) -> dict[str, Any]:
    if not branch_output:
        return branch_output

    result = dict(branch_output.get("g2211_result") or {})
    additional_output = dict(result.get("additional_output") or {})
    additional_output["rag"] = rag_payload
    result["additional_output"] = additional_output
    return {"g2211_result": result}


def _build_g2211_node(*, rag_client: RagClient | None, retrieval_enabled: bool, rag_config: Mapping[str, Any]):
    max_examples = int(rag_config.get("prompt_max_examples", 4))

    def node(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
        if not is_hcpcs_family_enabled_for_config(config, "g2211"):
            return {}
        if not retrieval_enabled or rag_client is None:
            return run_g2211_classifier(state, config)

        if state.get("encounter_route") != AMBULATORY_ROUTE:
            return {}

        rag_payload: dict[str, Any]
        augmented_state = dict(state)
        try:
            rag_response = rag_client.retrieve(_rag_request_from_state(state, rag_config))
            rag_payload = rag_response.model_dump()
            if rag_response.prompt_payload and rag_response.prompt_payload.examples:
                augmented_state = _augment_state_with_retrieval_examples(
                    state,
                    rag_response,
                    max_examples=max_examples,
                )
        except Exception as exc:  # pragma: no cover - runtime integration failure path
            logger.warning("Helix RAG retrieval failed for example_id={}: {}", state.get("example_id"), exc)
            rag_payload = {
                "feature_name": str(rag_config.get("feature_name") or "G2211"),
                "feature_code": str(rag_config.get("feature_name") or "G2211"),
                "target_code": str(rag_config.get("feature_name") or "G2211"),
                "retrieval_status": "error",
                "retrieval_status_reason": str(exc),
                "query_summary": {"summary_compact": "", "summary_full": "", "source": "helix_error", "sections": {}},
                "query_signals": [],
                "retrieved_encounters": [],
                "label_distribution": {},
                "diagnostics": {},
                "prompt_payload": {"target_code": str(rag_config.get("feature_name") or "G2211"), "guidance": "", "examples": []},
            }

        branch_output = run_g2211_classifier(augmented_state, config)
        return _attach_rag_payload(branch_output, rag_payload)

    return node


def build_helix_graph(args: Namespace) -> tuple[CompiledStateGraph, dict[str, Any]]:
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}}

    retrieval_enabled = _coerce_bool(getattr(args, "retrieval_enabled", False), default=False)
    rag_config = _normalize_rag_config(args)
    g2211_enabled = is_hcpcs_family_enabled(getattr(args, "hcpcs_in_scope", None), "g2211")
    rag_client = create_rag_client(rag_config) if retrieval_enabled and g2211_enabled else None

    workflow = StateGraph(HCPCSGraphState, input=HCPCSInputGraphState, output=HCPCSOutputGraphState)

    workflow.add_node("route_encounter", route_encounter)
    workflow.add_node("g2211_classifier", _build_g2211_node(rag_client=rag_client, retrieval_enabled=retrieval_enabled, rag_config=rag_config))
    workflow.add_node("awv_eligibility_precheck", _select_awv_precheck_node(config))
    workflow.add_node("awv_classifier", run_awv_classifier)
    workflow.add_node("other_hcpcs_classifier", run_other_hcpcs_classifier)
    workflow.add_node("merge_results", merge_results)

    workflow.add_edge(START, "route_encounter")
    workflow.add_edge("route_encounter", "g2211_classifier")
    workflow.add_edge("route_encounter", "awv_eligibility_precheck")
    workflow.add_edge("route_encounter", "other_hcpcs_classifier")
    workflow.add_conditional_edges(
        "awv_eligibility_precheck",
        _route_after_awv_precheck,
        {
            "awv_classifier": "awv_classifier",
            "merge_results": "merge_results",
        },
    )
    workflow.add_edge("g2211_classifier", "merge_results")
    workflow.add_edge("awv_classifier", "merge_results")
    workflow.add_edge("other_hcpcs_classifier", "merge_results")
    workflow.add_edge("merge_results", END)

    return workflow.compile(), model_config


def build_llm_only_graph(args: Namespace) -> tuple[CompiledStateGraph, dict[str, Any]]:
    return build_helix_graph(args)
