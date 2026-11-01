from __future__ import annotations

from argparse import Namespace
from typing import Any, Mapping

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.hcpcs_codes.models.common import (
    HCPCSGraphState,
    HCPCSInputGraphState,
    HCPCSOutputGraphState,
    apply_input_ablation_mode,
    is_hcpcs_family_enabled_for_config,
    merge_results,
)
from cdrc.profees_coding.hcpcs_codes.models.utils import CONFIG

from .constants import HCPCS_AWV_PROMPT_VERSION, HCPCS_BRANCH_NAME, HCPCS_G2211_PROMPT_VERSION, HCPCS_OTHER_PROMPT_VERSION
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
) -> dict[str, Any]:
    if not is_hcpcs_family_enabled_for_config(config, branch_name):
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
    )


def run_awv_classifier(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
    return _run_branch(
        state,
        config,
        result_key="awv_result",
        branch_name="awv",
        prompt_config_key=HCPCS_AWV_PROMPT_VERSION,
        default_prompt_version={"version": "v5.1", "codes_supported": ["G0402", "G0438", "G0439"]},
    )


def run_other_hcpcs_classifier(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
    return _run_branch(
        state,
        config,
        result_key="other_hcpcs_result",
        branch_name="other_hcpcs",
        prompt_config_key=HCPCS_OTHER_PROMPT_VERSION,
        default_prompt_version={"version": "v6.1", "codes_supported": ["Other HCPCS"]},
    )


def build_baseline_llm_graph(args: Namespace) -> tuple[CompiledStateGraph, dict[str, Any]]:
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}}

    workflow = StateGraph(HCPCSGraphState, input=HCPCSInputGraphState, output=HCPCSOutputGraphState)
    workflow.add_node("g2211_classifier", run_g2211_classifier)
    workflow.add_node("awv_classifier", run_awv_classifier)
    workflow.add_node("other_hcpcs_classifier", run_other_hcpcs_classifier)
    workflow.add_node("merge_results", merge_results)

    workflow.add_edge(START, "g2211_classifier")
    workflow.add_edge(START, "awv_classifier")
    workflow.add_edge(START, "other_hcpcs_classifier")
    workflow.add_edge("g2211_classifier", "merge_results")
    workflow.add_edge("awv_classifier", "merge_results")
    workflow.add_edge("other_hcpcs_classifier", "merge_results")
    workflow.add_edge("merge_results", END)

    return workflow.compile(), model_config
