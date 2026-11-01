from argparse import Namespace
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from cdrc.common.data.schema import (
    AnaesthesiaConcurrentStaffInfo,
    AnaesthesiaStaffInfo,
    EncounterMetaData,
    FacilityInfo,
    HealthPlan,
    LabInfo,
    ProviderInfo,
)
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.modifiers.data.schema import ICD10CM, EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.data_classes import CONFIG
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.branch_extraction import modifier_branch_extraction_node
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.modifier_50_lt_rt import build_modifier_50_lt_rt_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.modifier_95 import build_modifier_95_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.modifier_p1_p6 import build_modifier_p1_p6_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.modifier_po_pn import build_modifier_po_pn_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.modifier_qk_qx import build_modifier_qk_qx_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.modifier_qw_90 import build_modifier_qw_90_subgraph
from cdrc.profees_coding.modifiers.models.utils import get_modifiers_in_scope

from .nodes.modifiers import ModifierGraphOutput, ModifierJustification, build_modifiers_subgraph


class ModifiersInputGraphState(TypedDict):
    example_id: Union[int, str, None]
    text: str
    patient_sex: str
    patient_age: int
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    all_patient_problems: Optional[List[Problem]]
    procedure: List[Procedure]
    em: List[EMItem]
    diagnosis: List[ICD10CM]
    payer_name: List[str]
    encounter_metadata: Optional[EncounterMetaData]
    health_plan: Optional[List[HealthPlan]]
    facility_info: Optional[List[FacilityInfo]]
    provider_info: Optional[List[ProviderInfo]]
    lab_info: Optional[List[LabInfo]]
    anaesthesia_staff_info: Optional[List[AnaesthesiaStaffInfo]]
    anaesthesia_concurrent_staff_info: Optional[List[AnaesthesiaConcurrentStaffInfo]]

    modifier_outputs: dict[str, ModifierGraphOutput]
    modifier_branch_plan: Dict[str, List[str]]


class ModifiersOutputGraphState(TypedDict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    unassigned_modifiers: List[ModifierJustification]


class ModifiersGraphState(ModifiersInputGraphState, ModifiersOutputGraphState):
    pass


def get_modifiers_subgraph(args: Namespace, modifier_names: List[str]):
    graph = build_modifiers_subgraph(args, modifier_names=modifier_names)
    return graph if hasattr(graph, "invoke") else graph.compile()


def _norm_mod_code(s: str) -> str:
    return (s or "").strip().upper()


def _dedupe_modifiers_by_code(mods: list[Modifier]) -> list[Modifier]:
    seen = set()
    out = []
    for m in mods:
        code = _norm_mod_code(m.code)
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(m)
    return out


def reconcile_node(state: ModifiersGraphState, config: RunnableConfig) -> ModifiersGraphState:
    def _dump(x):
        return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)

    def _norm_code(cpt) -> str:
        try:
            if isinstance(cpt, str):
                return cpt.strip().upper()
            d = _dump(cpt) or {}
            return str(d.get("code") or d.get("cpt_code") or "").strip().upper()
        except Exception:
            return str(cpt).strip().upper()

    def _to_modifier(value) -> Modifier:
        return value if isinstance(value, Modifier) else Modifier(**(_dump(value) or {}))

    em_mods_by_code: dict[str, list[Modifier]] = {}
    proc_mods_by_code: dict[str, list[Modifier]] = {}
    all_unassigned: list[ModifierJustification] = []

    for mod_output in (state.get("modifier_outputs") or {}).values():
        for em_in in mod_output.get("em") or []:
            d = _dump(em_in) or {}
            code = _norm_code(d)
            for m in d.get("modifiers") or []:
                if not _norm_mod_code(getattr(m, "code", None) or (_dump(m) or {}).get("code")):
                    continue
                em_mods_by_code.setdefault(code, []).append(_to_modifier(m))

        for proc_in in mod_output.get("procedure") or []:
            d = _dump(proc_in) or {}
            code = _norm_code(d)
            for m in d.get("modifiers") or []:
                if not _norm_mod_code(getattr(m, "code", None) or (_dump(m) or {}).get("code")):
                    continue
                proc_mods_by_code.setdefault(code, []).append(_to_modifier(m))

        all_unassigned.extend(mod_output.get("unassigned_modifiers") or [])

    assigned_codes: set[str] = set()
    for em_mods in em_mods_by_code.values():
        for m in em_mods:
            assigned_codes.add(_norm_mod_code(m.code))
    for proc_mods in proc_mods_by_code.values():
        for m in proc_mods:
            assigned_codes.add(_norm_mod_code(m.code))

    deduped_unassigned: list[ModifierJustification] = []
    seen = set()
    for m in all_unassigned:
        d = _dump(m) or {}
        code = _norm_mod_code(d.get("code") or getattr(m, "code", None))
        if not code:
            continue
        key = (code, d.get("evidence"), d.get("justification"))
        if key in seen:
            continue
        seen.add(key)
        deduped_unassigned.append(m)

    output_em = []
    for em_in in state.get("em") or []:
        d = (_dump(em_in) or {}).copy()
        assigned = em_mods_by_code.get(_norm_code(d), [])
        d["modifiers"] = _dedupe_modifiers_by_code(assigned)
        output_em.append(EMItem(**d))

    output_proc = []
    for proc_in in state.get("procedure") or []:
        d = (_dump(proc_in) or {}).copy()
        code = _norm_code(d)
        assigned = proc_mods_by_code.get(code, [])
        d["modifiers"] = _dedupe_modifiers_by_code(assigned)
        d["linked_diagnosis"] = []
        output_proc.append(Procedure(**d))

    return {
        "em": output_em,
        "procedure": output_proc,
        "unassigned_modifiers": deduped_unassigned,
    }


def _normalize_code(raw: Any) -> str:
    return str(raw or "").strip().upper()


def _dedupe_codes(codes: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for code in codes or []:
        normalized = _normalize_code(code)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _process_modifiers_in_scope(scope, grouped=True):
    new_scope = []
    p_modifiers = ["P1", "P2", "P3", "P4", "P5", "P6"]
    facility_modifiers = ["PO", "PN"]
    anesthesia_modifiers = ["QK", "QX"]
    lab_modifiers = ["QW", "90"]
    payers_modifiers = ["GY", "UD"]
    laterality_modifiers = ["50", "LT", "RT"]
    therapy_modifiers = ["GP", "GO", "GN", "CQ", "CO"]
    for m in scope:
        if m in p_modifiers:
            if not any(set(group) & set(p_modifiers) for group in new_scope):
                new_scope.append(p_modifiers)
        elif m in laterality_modifiers:
            if not any(set(group) & set(laterality_modifiers) for group in new_scope):
                new_scope.append(laterality_modifiers)
        elif m in anesthesia_modifiers:
            if not any(set(group) & set(anesthesia_modifiers) for group in new_scope):
                new_scope.append(anesthesia_modifiers)
        elif m in facility_modifiers:
            if not any(set(group) & set(facility_modifiers) for group in new_scope):
                new_scope.append(facility_modifiers)
        elif m in lab_modifiers:
            if not any(set(group) & set(lab_modifiers) for group in new_scope):
                new_scope.append(lab_modifiers)
        elif m in payers_modifiers:
            if not any(set(group) & set(payers_modifiers) for group in new_scope):
                new_scope.append(payers_modifiers)
        elif m in therapy_modifiers:
            if not any(set(group) & set(therapy_modifiers) for group in new_scope):
                new_scope.append(therapy_modifiers)
        else:
            new_scope.append([m])

    return new_scope


def _with_token_tracker(tracker_name: str, graph):
    def _invoke(state: ModifiersGraphState, config: RunnableConfig):
        cfg: RunnableConfig = dict(config or {})
        metadata = dict(cfg.get("metadata", {}) or {})

        base_tracker_id = str(metadata.get("token_tracker_id") or "model_usage")
        metadata["token_tracker_id"] = f"{base_tracker_id}:{tracker_name}"
        cfg["metadata"] = metadata

        if "max_concurrency" not in cfg:
            cfg["max_concurrency"] = 999

        return graph.invoke(state, cfg)

    return RunnableLambda(_invoke)


def _with_node_token_tracker(name: str, graph):
    return _with_token_tracker(f"modifier_{name}", graph)


def build_parallel_modifiers_graph(args: Namespace, grouped=True, track_subgraph=False) -> tuple[CompiledStateGraph, dict]:
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}, "max_concurrency": 999}

    scope = get_modifiers_in_scope(config.get("modifiers_in_scope", "25"))
    scope = _process_modifiers_in_scope(scope, grouped=grouped)
    # print("Scope:", scope)
    subgraphs = {}
    for m in scope:
        subgraph_key = ",".join(m)
        if grouped:
            if "95" in m:
                subgraphs[subgraph_key] = build_modifier_95_subgraph(args, modifier_names=m)
            elif any(code in {"P1", "P2", "P3", "P4", "P5", "P6"} for code in m):
                subgraphs[subgraph_key] = build_modifier_p1_p6_subgraph(args, modifier_names=m)
            elif any(code in {"50", "LT", "RT"} for code in m):
                subgraphs[subgraph_key] = build_modifier_50_lt_rt_subgraph(args, modifier_names=m)
            # elif any(code in {"QW", "90"} for code in m):
            #     subgraphs[subgraph_key] = build_modifier_qw_90_subgraph(args, modifier_names=m)
            elif any(code in {"PO", "PN"} for code in m):
                subgraphs[subgraph_key] = build_modifier_po_pn_subgraph(args, modifier_names=m)
            elif any(code in {"QX", "QK"} for code in m):
                subgraphs[subgraph_key] = build_modifier_qk_qx_subgraph(args, modifier_names=m)
            else:
                subgraphs[subgraph_key] = get_modifiers_subgraph(args, modifier_names=m)
        else:
            subgraphs[subgraph_key] = get_modifiers_subgraph(args, modifier_names=m)
        subgraphs[subgraph_key] = subgraphs[subgraph_key] if hasattr(subgraphs[subgraph_key], "invoke") else subgraphs[subgraph_key].compile()

    if track_subgraph:
        subgraphs = {name: _with_node_token_tracker(name, graph) for name, graph in subgraphs.items()}

    def run_parallel_and_namespace(state: ModifiersGraphState, config: RunnableConfig):
        # runs after modifier_branch_extraction_node
        cfg: RunnableConfig = dict(config or {})
        if "max_concurrency" not in cfg:
            cfg["max_concurrency"] = 999

        branch_plan = state.get("modifier_branch_plan") or {}
        branch_runnables = {}
        for modifier_names in scope:
            branch_name = ",".join(modifier_names)
            eligible_codes = _dedupe_codes(branch_plan.get(branch_name) or [])
            if not eligible_codes:
                continue

            graph = subgraphs[branch_name]
            branch_state = {
                **state,
                "eligible_cpt_codes": eligible_codes,
            }

            def _invoke_branch(
                _state: ModifiersGraphState,
                *,
                graph=graph,
                branch_state=branch_state,
                cfg=cfg,
            ):
                return graph.invoke(branch_state, cfg)

            branch_runnables[branch_name] = RunnableLambda(_invoke_branch)

        if not branch_runnables:
            return {"modifier_outputs": {}}

        outputs = RunnableMap(branch_runnables).invoke(state, cfg)
        return {"modifier_outputs": {f"modifier_{name}": out for name, out in outputs.items()}}

    workflow = StateGraph(ModifiersGraphState, input=ModifiersInputGraphState, output=ModifiersOutputGraphState)
    branch_extraction_runnable = RunnableLambda(
        lambda state, config: modifier_branch_extraction_node(
            state,
            config=config,
            branch_modifier_groups=scope,
        )
    )
    if track_subgraph:
        branch_extraction_runnable = _with_token_tracker("branch_selection", branch_extraction_runnable)

    workflow.add_node("extract_modifier_branches", branch_extraction_runnable)
    workflow.add_node("run_modifiers_parallel", RunnableLambda(run_parallel_and_namespace))
    workflow.add_node("reconcile", reconcile_node)

    workflow.add_edge(START, "extract_modifier_branches")
    workflow.add_edge("extract_modifier_branches", "run_modifiers_parallel")
    workflow.add_edge("run_modifiers_parallel", "reconcile")
    workflow.add_edge("reconcile", END)

    return workflow.compile(), model_config
