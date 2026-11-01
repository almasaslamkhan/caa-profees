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
from cdrc.profees_coding.em.models.utils import get_code, get_complexity_from_code

from .data_classes import CONFIG
from .nodes import build_encounter_type_subgraph, build_mdm_complexity_subgraph, build_patient_type_subgraph, build_time_statement_subgraph
from .preprocessing import MDMResult


# ------------------------ defining the graph state ---------------------------
class CptEmInputGraphState(TypedDict):
    example_id: Union[int, str, None]
    text: str
    patient_type: PatientType
    patient_age: int
    orders_display: str
    soap_text_previous: str
    reason_for_visit: str
    appointment_type: str
    service_date: str


class CptEmOutputGraphState(TypedDict):
    em: List[EMItem]
    patient_type: PatientType
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
    patient_type_prediction: dict
    patient_type_justification: str


class CptEmGraphState(CptEmInputGraphState, CptEmOutputGraphState):
    pass


# ------------------------ defining the graph nodes ---------------------------
def get_encounter_type_subgraph(args: Namespace):
    graph = build_encounter_type_subgraph(args)
    return graph


def get_patient_type_subgraph(args: Namespace):
    graph = build_patient_type_subgraph(args)
    return graph


def get_time_statement_subgraph(args: Namespace):
    graph = build_time_statement_subgraph(args)
    return graph


def get_mdm_complexity_subgraph(args: Namespace):
    graph = build_mdm_complexity_subgraph(args)
    return graph


def is_pt_mode_enabled(args: Namespace) -> bool:
    value = getattr(args, "pt_mode_on_off", True)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"off", "false", "0", "no"}:
            return False
        if normalized in {"on", "true", "1", "yes"}:
            return True
    return bool(value)


def mdm_justification_lines(mdm: MDMResult) -> list[str]:
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
        # parts.append(f"{i}. {key}: {level}. Justification: {justification}")
        key_str = key.replace("_", " ").title()
        parts.append(f"{key_str}: {level}. {justification}")
    return parts


def patient_type_justification_line(patient_type: PatientType, justification: str | None = None) -> str:
    patient_type_value = getattr(patient_type, "value", patient_type)
    justification_text = str(justification or "").strip()
    if not justification_text:
        return f"Patient type: {patient_type_value}. Patient visit history supports the prediction."
    if justification_text[-1] not in ".!?":
        justification_text = f"{justification_text}."
    return f"Patient type: {patient_type_value}. {justification_text}"


def decision_tree_node(state: CptEmGraphState, config: RunnableConfig) -> CptEmGraphState:
    rci_mode = config["configurable"][CONFIG.RCI_MODE]
    default_time = EMTimeSpent(
        time_spent_minutes=0,
        evidence=None,
        justification="Time statement not used for this encounter type.",
    )
    time_spent = state.get("time_spent") or default_time

    # Build MDM triple either from critics or precomputed
    if rci_mode != "none":
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
            time_spent_minutes=time_spent.time_spent_minutes,
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
        mdm_lines = mdm_justification_lines(mdm)
        include_mdm = et_str.lower() not in {"preventive medicine", "preventative"}
        # justification = f"Encounter type: {et_just}"
        # justification = f"Encounter type: {et_str}" + (f". {et_just}" if et_just else "")
        justification_lines = [
            "Encounter type: " + (f"{et_just}" if et_just else ""),
            patient_type_justification_line(state["patient_type"], state.get("patient_type_justification")),
        ]

        if include_mdm and mdm_lines:
            # justification = justification + "\nMDM:\n" + mdm_lines
            justification_lines.extend(mdm_lines)
            if et_str == EncounterType.OUTPATIENT:
                justification_lines.append(f"Time statement: {time_spent.justification}")
        justification = "\n".join(justification_lines)

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
    if not em_items:
        not_supported_encounter = next(
            (
                et
                for et in encounter_types
                if getattr(getattr(et, "encounter_type", None), "value", getattr(et, "encounter_type", None)) == EncounterType.NOT_SUPPORTED.value
            ),
            None,
        )
        not_supported_justification = getattr(not_supported_encounter, "justification", None)
        em_items.append(
            EMItem(
                code="NOT_SUPPORTED",
                description="NOT_SUPPORTED",
                source="NOT_SUPPORTED",
                system="CPT",
                justification=not_supported_justification or "Not supported service.",
                encounter_type={"encounter_type": EncounterType.NOT_SUPPORTED.value, "justification": not_supported_justification or None},
            )
        )
    return {
        "em": em_items,
        "patient_type": state["patient_type"],
        "encounter_type": encounter_types,
        "problem_complexity": mdm.problem_complexity,
        "data_complexity": mdm.data_complexity,
        "risk_of_complications": mdm.risk_of_complications,
        "mdm_results": None if rci_mode == "none" or rci_mode == "simple" else state["mdm_results"],
        "soap_text_previous": state["text"],
        "patient_type_justification": state.get("patient_type_justification", ""),
    }


# ------------------------ defining the graph builder ---------------------------
def build_medprompt_cpt_em_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:
    # set the prompts and model config
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}}
    pt_mode_enabled = is_pt_mode_enabled(args)

    workflow = StateGraph(CptEmGraphState, input=CptEmInputGraphState, output=CptEmOutputGraphState)

    if pt_mode_enabled:
        workflow.add_node("patient_type_classifier", get_patient_type_subgraph(args))
    workflow.add_node("encounter_type_classifier", get_encounter_type_subgraph(args))
    workflow.add_node("mdm_complexity_classifier", get_mdm_complexity_subgraph(args))
    is_ed = str(getattr(args, "site", "")).strip().lower() == "ed"
    if not is_ed:
        workflow.add_node("time_statement_extractor", get_time_statement_subgraph(args))
    workflow.add_node("decision_tree", decision_tree_node)

    if pt_mode_enabled:
        workflow.add_edge(START, "patient_type_classifier")

    workflow.add_edge(START, "encounter_type_classifier")
    workflow.add_edge(START, "mdm_complexity_classifier")

    if not is_ed:
        workflow.add_edge(START, "time_statement_extractor")
        workflow.add_edge("time_statement_extractor", "decision_tree")

    if pt_mode_enabled:
        workflow.add_edge("encounter_type_classifier", "decision_tree")
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
