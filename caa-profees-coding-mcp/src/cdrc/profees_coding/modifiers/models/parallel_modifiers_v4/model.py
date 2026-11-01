from argparse import Namespace
from typing import List, Optional, TypedDict, Union

from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from cdrc.common.data.schema import AnaesthesiaRecordInfo, AnaesthesiaStaffInfo, EncounterMetaData, FacilityInfo, HealthPlan, LabInfo, ProviderInfo
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.modifiers.data.schema import ICD10CM, EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.nodes.modifier_50_lt_rt import build_modifier_50_lt_rt_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.nodes.modifier_95 import build_modifier_95_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.nodes.modifier_gy import build_modifier_gy_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.nodes.modifier_p1_p6 import build_modifier_p1_p6_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.nodes.modifier_po_pn import build_modifier_po_pn_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.nodes.modifier_qk_qx import build_modifier_qk_qx_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.nodes.modifier_qw import build_modifier_qw_subgraph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.nodes.modifier_qw_90 import build_modifier_qw_90_subgraph
from cdrc.profees_coding.modifiers.models.utils import get_modifiers_in_scope

from .nodes.modifiers import ModifierGraphOutput, build_modifiers_subgraph


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
    anaesthesia_record_info: Optional[List[AnaesthesiaRecordInfo]]

    modifier_outputs: dict[str, ModifierGraphOutput]


class ModifiersOutputGraphState(TypedDict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")


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
    del config

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

    output_em = []
    for em_in in state.get("em") or []:
        d = (_dump(em_in) or {}).copy()
        selected_mods = em_mods_by_code.get(_norm_code(d), [])
        d["modifiers"] = _dedupe_modifiers_by_code(selected_mods)
        output_em.append(EMItem(**d))

    output_proc = []
    for proc_in in state.get("procedure") or []:
        d = (_dump(proc_in) or {}).copy()
        code = _norm_code(d)
        selected_mods = proc_mods_by_code.get(code, [])
        d["modifiers"] = _dedupe_modifiers_by_code(selected_mods)
        d["linked_diagnosis"] = []
        output_proc.append(Procedure(**d))

    return {
        "em": output_em,
        "procedure": output_proc,
    }


def process_modifiers_in_scope(scope):
    new_scope = []
    done_lateral = False
    done_p_modifiers = False
    p_modifiers = {"P1", "P2", "P3", "P4", "P5", "P6"}
    for m in scope:
        if m in p_modifiers:
            if not done_p_modifiers:
                done_p_modifiers = True
                grouped = [mod for mod in scope if mod in p_modifiers]
                new_scope.append(grouped)
        elif m in ["50", "LT", "RT"]:
            if not done_lateral:
                done_lateral = True
                new_scope.append(["50", "LT", "RT"])
        elif m in ["PO", "PN"]:
            if not any(set(group) & {"PO", "PN"} for group in new_scope):
                new_scope.append(["PO", "PN"])
        elif m in ["QK", "QX"]:
            if not any(set(group) & {"QK", "QX"} for group in new_scope):
                new_scope.append(["QK", "QX"])
        elif m in ["QW", "90"]:
            if not any(set(group) & {"QW", "90"} for group in new_scope):
                new_scope.append(["QW", "90"])
        elif m in ["GY", "UD"]:
            if not any(set(group) & {"GY", "UD"} for group in new_scope):
                new_scope.append(["GY", "UD"])
        elif m in ["GP", "GO", "GN", "CQ", "CO"]:
            if not any(set(group) & {"GP", "GO", "GN", "CQ", "CO"} for group in new_scope):
                new_scope.append(["GP", "GO", "GN", "CQ", "CO"])
        else:
            new_scope.append([m])

        # new_scope.append([m])
    return new_scope


def _with_node_token_tracker(name: str, graph):
    def _invoke(state: ModifiersGraphState, config: RunnableConfig):
        cfg: RunnableConfig = dict(config or {})
        metadata = dict(cfg.get("metadata", {}) or {})

        base_tracker_id = str(metadata.get("token_tracker_id") or "model_usage")
        metadata["token_tracker_id"] = f"{base_tracker_id}:modifier_{name}"
        cfg["metadata"] = metadata

        if "max_concurrency" not in cfg:
            cfg["max_concurrency"] = 999

        return graph.invoke(state, cfg)

    return RunnableLambda(_invoke)


def build_parallel_modifiers_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}, "max_concurrency": 999}

    workflow = StateGraph(ModifiersGraphState, input=ModifiersInputGraphState, output=ModifiersOutputGraphState)

    scope = get_modifiers_in_scope(config.get("modifiers_in_scope"))
    # scope = [[x] for x in scope]
    scope = process_modifiers_in_scope(scope)
    subgraphs = {}
    for m in scope:
        # if "GY" in m:
        #     subgraphs[",".join(m)] = build_modifier_gy_subgraph(args, modifier_names=m)
        # elif "QW" in m:
        #     subgraphs[",".join(m)] = build_modifier_qw_subgraph(args, modifier_names=m)
        if "95" in m:
            subgraphs[",".join(m)] = build_modifier_95_subgraph(args, modifier_names=m)
        elif any(code in {"P1", "P2", "P3", "P4", "P5", "P6"} for code in m):
            subgraphs[",".join(m)] = build_modifier_p1_p6_subgraph(args, modifier_names=m)
        elif any(code in {"50", "LT", "RT"} for code in m):
            subgraphs[",".join(m)] = build_modifier_50_lt_rt_subgraph(args, modifier_names=m)
        # elif any(code in {"QW", "90"} for code in m):
        #     subgraphs[",".join(m)] = build_modifier_qw_90_subgraph(args, modifier_names=m)
        elif any(code in {"PO", "PN"} for code in m):
            subgraphs[",".join(m)] = build_modifier_po_pn_subgraph(args, modifier_names=m)
        elif any(code in {"QX", "QK"} for code in m):
            subgraphs[",".join(m)] = build_modifier_qk_qx_subgraph(args, modifier_names=m)
        else:
            subgraphs[",".join(m)] = get_modifiers_subgraph(args, modifier_names=m)
        # subgraphs[",".join(m)] = get_modifiers_subgraph(args, modifier_names=m)
    # subgraphs = {",".join(m): get_modifiers_subgraph(args, modifier_names=m) for m in scope}
    # subgraphs = {
    #     name: _with_node_token_tracker(name, graph)
    #     for name, graph in subgraphs.items()
    # }
    parallel = RunnableMap(subgraphs)

    def run_parallel_and_namespace(state: ModifiersGraphState, config: RunnableConfig):
        cfg: RunnableConfig = dict(config or {})
        if "max_concurrency" not in config:
            cfg["max_concurrency"] = 999

        outputs = parallel.invoke(state, cfg)
        return {"modifier_outputs": {f"modifier_{name}": out for name, out in outputs.items()}}

    workflow.add_node("run_modifiers_parallel", RunnableLambda(run_parallel_and_namespace))
    workflow.add_node("reconcile", reconcile_node)

    workflow.add_edge(START, "run_modifiers_parallel")
    workflow.add_edge("run_modifiers_parallel", "reconcile")
    workflow.add_edge("reconcile", END)

    return workflow.compile(), model_config
