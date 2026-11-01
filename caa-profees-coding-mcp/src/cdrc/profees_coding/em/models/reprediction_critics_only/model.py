import sqlite3
from argparse import Namespace
from typing import List, Optional, TypedDict, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.deploy.api.utils import ProfeesResponse
from cdrc.profees_coding.em.data.schema import ComplexityDetails, EMItem, EMTimeSpent, EncounterType, EncounterTypeDetails, PatientType
from cdrc.profees_coding.em.models.checkpoint import SqliteCheckpointer
from cdrc.profees_coding.em.models.medprompt_detailed_multi.model import mdm_justification_lines
from cdrc.profees_coding.em.models.medprompt_detailed_multi.nodes import build_encounter_type_subgraph, build_time_statement_subgraph
from cdrc.profees_coding.em.models.utils import get_code, get_complexity_from_code

from .nodes import build_mdm_complexity_subgraph
from .preprocessing import MDMResult


# ------------------------ defining the graph state ---------------------------
class CptEmInputGraphState(TypedDict):
    example_id: Union[int, str, None]
    soap_text_previous: str
    text: str
    response_previous: Optional[ProfeesResponse]
    encounter_type: List[EncounterTypeDetails]
    patient_type: PatientType
    patient_age: int
    orders_display: str


class CptEmOutputGraphState(TypedDict):
    em: List[EMItem]
    pc_critic_response: str
    rc_critic_response: str
    dc_critic_response: str
    mdm_results: MDMResult
    problem_complexity: ComplexityDetails
    data_complexity: ComplexityDetails
    risk_of_complications: ComplexityDetails
    encounter_type: List[EncounterTypeDetails]
    time_spent: EMTimeSpent


class CptEmGraphState(CptEmInputGraphState, CptEmOutputGraphState):
    pass


# ------------------------ defining the graph nodes ---------------------------
def get_encounter_type_subgraph(args: Namespace):
    graph = build_encounter_type_subgraph(args)
    return graph


def get_mdm_complexity_subgraph(args: Namespace):
    graph = build_mdm_complexity_subgraph(args)
    return graph


def get_time_statement_subgraph(args: Namespace):
    graph = build_time_statement_subgraph(args)
    return graph


def decision_tree_node(state: CptEmGraphState, config: RunnableConfig) -> CptEmGraphState:
    mdm = state["mdm_results"]

    encounter_types: List[EncounterTypeDetails] = state["encounter_type"]

    # ---- SORT: Office first, then Preventive, then others
    def _sort_key(et: EncounterTypeDetails):
        name = getattr(et, "encounter_type", "")
        n = (name or "").lower()
        if n == getattr(EncounterType, "OUTPATIENT", "Office or Other Outpatient Service").lower() or n == "office or other outpatient service":
            return (0, n)
        if n == getattr(EncounterType, "PREVENTIVE_MEDICINE", "Preventive Medicine").lower() or n == "preventive medicine" or n == "preventative":
            return (1, n)
        return (2, n)

    encounter_types = sorted(encounter_types, key=_sort_key)

    # Convert complexity objects to plain dicts to avoid cross-class issues
    pc_dict = {
        "level": getattr(mdm.problem_complexity.level, "value", mdm.problem_complexity.level),
        "justification": mdm.problem_complexity.justification,
        "detailed_justification": mdm.problem_complexity.detailed_justification,
        "confidence": mdm.problem_complexity.confidence,
    }
    dc_dict = {
        "level": getattr(mdm.data_complexity.level, "value", mdm.data_complexity.level),
        "justification": mdm.data_complexity.justification,
        "detailed_justification": mdm.data_complexity.detailed_justification,
        "confidence": mdm.data_complexity.confidence,
    }
    rc_dict = {
        "level": getattr(mdm.risk_of_complications.level, "value", mdm.risk_of_complications.level),
        "justification": mdm.risk_of_complications.justification,
        "detailed_justification": mdm.risk_of_complications.detailed_justification,
        "confidence": mdm.risk_of_complications.confidence,
    }

    em_items: List[EMItem] = []
    for et in encounter_types:
        et_str = et.encounter_type  # canonical string

        em_code, em_description, source = get_code(
            time_spent_minutes=state["time_spent"].time_spent_minutes,
            patient_type=state["patient_type"],
            encounter_type=et_str,
            problem_complexity_level=mdm.problem_complexity.level,
            data_complexity_level=mdm.data_complexity.level,
            risk_level=mdm.risk_of_complications.level,
            age=state["patient_age"],
        )

        # Build justification:
        # - Always include encounter type
        # - Include MDM lines only if NOT preventive
        et_just = getattr(et, "justification", None) or ""
        mdm_lines = "\n".join(mdm_justification_lines(mdm))
        include_mdm = et_str.lower() not in {"preventive medicine", "preventative"}
        # justification = f"Encounter type: {et_just}"
        # justification = f"Encounter type: {et_str}" + (f". {et_just}" if et_just else "")
        justification = "Encounter type: " + (f"{et_just}" if et_just else "")

        if include_mdm and mdm_lines:
            # justification = justification + "\nMDM:\n" + mdm_lines
            justification = justification + "\n" + mdm_lines
        justification = justification + f"\nTime statement: {state['time_spent'].justification}"

        # mdm_complexity only for OUTPATIENT (and non-preventive)
        mdm_complexity_dict = None
        if include_mdm and et_str == EncounterType.OUTPATIENT:
            lvl = get_complexity_from_code(em_code)
            mdm_complexity_dict = {
                "level": getattr(lvl, "value", lvl),
                "justification": None,
                "confidence": state.get(
                    "confidence",
                ),
            }

        # Conditionally include pc/dc/rc only when NOT preventive
        em_kwargs = {
            "code": em_code,
            "description": em_description,
            "source": source,
            "confidence": state.get(
                "confidence",
            ),
            "system": "CPT",
            "justification": justification,
            "encounter_type": {"encounter_type": et_str, "justification": et_just or None},
            "problem_complexity": pc_dict if include_mdm else None,
            "data_complexity": dc_dict if include_mdm else None,
            "risk_of_complications": rc_dict if include_mdm else None,
            "mdm_complexity": mdm_complexity_dict if include_mdm else None,
            "time_spent": state.get("time_spent"),
            "linked_diagnosis": [],
            "modifiers": [],
        }

        em_items.append(EMItem(**em_kwargs))

    return {
        "em": em_items,
        "encounter_type": encounter_types,
        "problem_complexity": mdm.problem_complexity,
        "data_complexity": mdm.data_complexity,
        "risk_of_complications": mdm.risk_of_complications,
        "mdm_results": mdm,  # to avoid cluttering the output if no critic
    }


# ------------------------ defining the graph builder ---------------------------
def build_reprediction_critics_only_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:
    # set the prompts and model config
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}}

    workflow = StateGraph(CptEmGraphState, input=CptEmInputGraphState, output=CptEmOutputGraphState)

    workflow.add_node("encounter_type_classifier", get_encounter_type_subgraph(args))
    workflow.add_node("mdm_complexity_classifier", get_mdm_complexity_subgraph(args))
    workflow.add_node("time_statement_extractor", get_time_statement_subgraph(args))
    workflow.add_node("decision_tree", decision_tree_node)

    workflow.add_edge(START, "encounter_type_classifier")
    workflow.add_edge(START, "mdm_complexity_classifier")
    workflow.add_edge(START, "time_statement_extractor")

    workflow.add_edge("encounter_type_classifier", "decision_tree")
    workflow.add_edge("time_statement_extractor", "decision_tree")
    workflow.add_edge("mdm_complexity_classifier", "decision_tree")

    workflow.add_edge("decision_tree", END)
    if not args.checkpoint_path:
        return workflow.compile(), model_config
    conn = sqlite3.connect(args.checkpoint_path, check_same_thread=False)
    checkpointer = SqliteCheckpointer(conn, serde=JsonPlusSerializer(pickle_fallback=True))

    return workflow.compile(checkpointer=checkpointer), model_config
