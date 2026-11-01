from argparse import Namespace
from typing import List, Optional, TypedDict, Union

from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.modifiers.data.schema import ICD10CM, EMItem, Modifier, Order, Problem, Procedure
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

    modifier_outputs: dict[str, ModifierGraphOutput]


class ModifiersOutputGraphState(TypedDict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    unassigned_modifiers: List[ModifierJustification]


class ModifiersGraphState(ModifiersInputGraphState, ModifiersOutputGraphState):
    pass


def get_modifiers_subgraph(args: Namespace, modifier_name: str):
    graph = build_modifiers_subgraph(args, modifier_name=modifier_name)
    return graph if hasattr(graph, "invoke") else graph.compile()


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

    def _norm_mod_code(s: str) -> str:
        return (s or "").strip().upper()

    def _to_modifier(value) -> Modifier:
        return value if isinstance(value, Modifier) else Modifier(**(_dump(value) or {}))

    em_mods_by_code: dict[str, list[Modifier]] = {}
    proc_mods_by_code: dict[str, list[Modifier]] = {}
    all_unassigned: list[ModifierJustification] = []

    for mod_output in (state.get("modifier_outputs") or {}).values():
        # print(mod_output.get("em"))
        # print(mod_output.get("unassigned_modifiers"))
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
    # deduped_unassigned = all_unassigned

    output_em = []
    for em_in in state.get("em") or []:
        d = (_dump(em_in) or {}).copy()
        assigned = em_mods_by_code.get(_norm_code(d), [])
        d["modifiers"] = assigned
        output_em.append(EMItem(**d))

    output_proc = []
    for proc_in in state.get("procedure") or []:
        d = (_dump(proc_in) or {}).copy()
        code = _norm_code(d)
        assigned = proc_mods_by_code.get(code, [])
        d["modifiers"] = assigned
        d["linked_diagnosis"] = []
        output_proc.append(Procedure(**d))

    return {
        "em": output_em,
        "procedure": output_proc,
        "unassigned_modifiers": deduped_unassigned,
    }


def build_parallel_modifiers_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}, "max_concurrency": 999}

    workflow = StateGraph(ModifiersGraphState, input=ModifiersInputGraphState, output=ModifiersOutputGraphState)

    scope = get_modifiers_in_scope(config.get("modifiers_in_scope"))
    subgraphs = {m: get_modifiers_subgraph(args, modifier_name=m) for m in scope}
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
