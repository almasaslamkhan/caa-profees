import sqlite3
from argparse import Namespace
from typing import List, TypedDict, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.em.data.schema import ComplexityDetails, EMItem, EMTimeSpent, EncounterType, EncounterTypeDetails, PatientType
from cdrc.profees_coding.em.models.checkpoint import SqliteCheckpointer
from cdrc.profees_coding.em.models.utils import get_code, get_complexity_from_code, get_critical_care_codes, get_em_code_description

from .data_classes import CONFIG
from .nodes import build_encounter_type_subgraph, build_mdm_complexity_subgraph, build_time_statement_subgraph
from .preprocessing import MDMResult


# ------------------------ defining the graph state ---------------------------
class CptEmInputGraphState(TypedDict):
    example_id: Union[int, str, None]
    text: str
    patient_type: PatientType
    patient_age: int
    orders_display: str
    soap_text_previous: str


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
    confidence: float  # default is -1 if not calculated


class CptEmGraphState(CptEmInputGraphState, CptEmOutputGraphState):
    pass


# ------------------------ defining the graph nodes ---------------------------
def get_encounter_type_subgraph(args: Namespace):
    graph = build_encounter_type_subgraph(args)
    return graph


def get_time_statement_subgraph(args: Namespace):
    graph = build_time_statement_subgraph(args)
    return graph


def get_mdm_complexity_subgraph(args: Namespace):
    graph = build_mdm_complexity_subgraph(args)
    return graph


def _mdm_justification_lines(mdm: MDMResult) -> list[str]:
    """
    Format a readable justification summary from the MDMResult.
    Produces lines like:
        1. problem_complexity: Low. Justification: Stable chronic illness.
        2. data_complexity: Minimal. Justification: Only one test reviewed.
        3. risk_of_complications: Moderate. Justification: Antibiotic prescribed.
    """
    parts = []
    mdm_dict = mdm.model_dump()
    for i, key in enumerate(["problem_complexity", "data_complexity", "risk_of_complications"], start=1):
        v = mdm_dict.get(key)
        if not v:
            continue
        level = getattr(v.get("level"), "value", v.get("level", ""))
        justification = v.get("justification", "")
        parts.append(f"{i}. {key}: {level}. Justification: {justification}")
    return parts


def _has_normalized_mdm_details(mdm) -> bool:
    return all(hasattr(getattr(mdm, field, None), "level") for field in ("problem_complexity", "data_complexity", "risk_of_complications"))


def decision_tree_node(state: CptEmGraphState, config: RunnableConfig) -> CptEmGraphState:
    rci_mode = config["configurable"][CONFIG.RCI_MODE]

    # Build MDM triple either from critics or precomputed
    if rci_mode != "none" or not _has_normalized_mdm_details(state.get("mdm_results")):
        mdm = MDMResult(
            problem_complexity=state["problem_complexity"],
            data_complexity=state["data_complexity"],
            risk_of_complications=state["risk_of_complications"],
        )
    else:
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
        et_str = et.encounter_type.value  # canonical string

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
        mdm_lines = "\n".join(_mdm_justification_lines(mdm))
        non_mdm_encounter_types = {
            EncounterType.PREVENTIVE_MEDICINE,
            EncounterType.CRITICAL_CARE,
            EncounterType.NEWBORN_INITIAL,
            EncounterType.NEWBORN_SUBSEQUENT,
            EncounterType.NEWBORN_SAME_DAY,
            EncounterType.NEWBORN_OTHER_SETTING,
        }
        include_mdm = et_str not in non_mdm_encounter_types and et_str.lower() not in {"preventive medicine", "preventative"}
        # justification = f"Encounter type: {et_just}"
        justification = f"Encounter type: {et_str}" + (f". {et_just}" if et_just else "")
        if include_mdm and mdm_lines:
            # justification = justification + "\nMDM:\n" + mdm_lines
            justification = justification + "\n" + mdm_lines
            justification = justification + f"\nTime statement: {state['time_spent'].justification}"
        elif et_str == EncounterType.CRITICAL_CARE:
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
        if source != "NOT_SUPPORTED":
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
                "pc_critic_response": "" if rci_mode == "none" else state.get("pc_critic_response", ""),
                "dc_critic_response": "" if rci_mode == "none" else state.get("dc_critic_response", ""),
                "rc_critic_response": "" if rci_mode == "none" else state.get("rc_critic_response", ""),
                "mdm_complexity": mdm_complexity_dict if include_mdm else None,
                "time_spent": state.get("time_spent"),
                "linked_diagnosis": [],
                "modifiers": [],
            }
            em_items.append(EMItem(**em_kwargs))

            if et_str == EncounterType.CRITICAL_CARE:
                for add_on_code in get_critical_care_codes(state["time_spent"].time_spent_minutes)[1:]:
                    add_on_kwargs = {
                        **em_kwargs,
                        "code": add_on_code,
                        "description": get_em_code_description(add_on_code),
                        "source": "Time",
                        "justification": justification + "\nCritical care add-on code supported by documented time.",
                    }
                    em_items.append(EMItem(**add_on_kwargs))

    return {
        "em": em_items,
        "encounter_type": encounter_types,
        "problem_complexity": mdm.problem_complexity,
        "data_complexity": mdm.data_complexity,
        "risk_of_complications": mdm.risk_of_complications,
        "mdm_results": None if rci_mode == "none" or rci_mode == "simple" else state["mdm_results"],
        "soap_text_previous": state["text"],
    }


# ------------------------ defining the graph builder ---------------------------
def build_medprompt_cpt_em_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:
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

    workflow.add_edge("time_statement_extractor", "decision_tree")
    workflow.add_edge("encounter_type_classifier", "decision_tree")
    workflow.add_edge("mdm_complexity_classifier", "decision_tree")

    workflow.add_edge("decision_tree", END)
    if args.checkpoint_path:
        conn = sqlite3.connect(args.checkpoint_path, check_same_thread=False)
        # checkpointer = SqliteSaver(conn, serde=JsonPlusSerializer(pickle_fallback=True))
        checkpointer = SqliteCheckpointer(conn, serde=JsonPlusSerializer(pickle_fallback=True))

        return workflow.compile(checkpointer=checkpointer), model_config
    else:
        return workflow.compile(), model_config
