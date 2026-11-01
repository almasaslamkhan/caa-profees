import copy
import json
import random
import sqlite3
from argparse import Namespace
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from cdrc.common.embeddings_utils import get_embedding_models
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.deploy.api.utils import ProfeesResponse
from cdrc.profees_coding.em.data.schema import (
    ComplexityDetails,
    CptEmCode,
    EMItem,
    EMTimeSpent,
    EncounterType,
    EncounterTypeDetails,
    PatientType,
    Problem,
)
from cdrc.profees_coding.em.models.checkpoint import SqliteCheckpointer
from cdrc.profees_coding.em.models.medprompt_detailed_multi.model import mdm_justification_lines
from cdrc.profees_coding.em.models.medprompt_detailed_multi.nodes import build_encounter_type_subgraph, build_time_statement_subgraph
from cdrc.profees_coding.em.models.utils import create_markdown_diff, get_code, get_complexity_from_code

from .data_classes import CONFIG, METADATA_KEYS, PROMPTS
from .preprocessing import MDMResult, extract_indexable_text

vector_store = None


def get_encounter_type_subgraph(args: Namespace):
    graph = build_encounter_type_subgraph(args)
    return graph


def get_time_statement_subgraph(args: Namespace):
    graph = build_time_statement_subgraph(args)
    return graph


def _init_or_get_vdb(vdb_index_path: str, embedding_model: str) -> VectorStore:
    global vector_store
    if vector_store is None:
        embeddings = get_embedding_models()[embedding_model]()
        vector_store = FAISS.load_local(vdb_index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store


def format_few_shot_examples(
    results: List[Tuple[Any, float]], threshold: float = -1, omit_soap_note: bool = False, icl_mode: str = "standard"
) -> str:
    if icl_mode == "standard":
        few_shot_examples = []
        for i, (res, score) in enumerate(results):
            if score > threshold:
                soap = res.metadata[METADATA_KEYS.SOAP_NOTE] if not omit_soap_note else "<SOAP NOTE OMMITTED REFER TO RESULTS BELOW>"
                cot = res.metadata[METADATA_KEYS.CHAIN_OF_THOUGHT_REASONING]
                few_shot_examples.append(
                    f"EXAMPLE {i + 1}:\n\n**SOAP Note:**\n {soap}\n\n**Results:** {cot}\n\n----------------------------------\n"  # noqa: E501
                )
        # logger.info("\n".join(few_shot_examples))
        return "\n".join(few_shot_examples)

    elif icl_mode == "knowledge_base":
        few_shot_examples = []
        for i, (res, score) in enumerate(results):
            if score > threshold:
                example = copy.deepcopy(res.metadata)
                # remove example_id key
                example.pop("example_id")
                for k, v in example.items():
                    v["justification"] = v["gold_justification"] + "\n\n" + v["model_justification"]
                    # # v["justification"] = v["model_justification"]
                    # v["justification"] = v["gold_justification"]
                    v.pop("gold_justification")
                    v.pop("model_justification")

                few_shot_examples.append(
                    f"EXAMPLE {i + 1}:\n\n**SOAP Note:**\n <SOAP NOTE OMMITTED REFER TO RESULTS BELOW>\n\n**Results:** {example}\n\n----------------------------------\n"  # noqa: E501
                )
        return "\n".join(few_shot_examples)
    else:
        raise ValueError(f"Invalid ICL mode: {icl_mode}, must be one of 'standard' or 'knowledge_base'")


def _get_model_kwargs_and_possibly_change_seed(config: Dict) -> Dict:
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)
    return model_kwargs


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
@track_tokens("model_usage")
def prediction_node(state: CptEmGraphState, config: RunnableConfig) -> CptEmGraphState:
    # print("Signed", state["signed_note"])
    # print("Draft", state["text"])

    num_few_shot_examples = int(config["configurable"][CONFIG.NUM_FEW_SHOT_EXAMPLES])
    icl_mode = config["configurable"].get(CONFIG.ICL_MODE, "standard")
    if "response_previous" in state and state["response_previous"]:
        state["problem_complexity"] = state["response_previous"].em[0].problem_complexity
        state["data_complexity"] = state["response_previous"].em[0].data_complexity
        state["risk_of_complications"] = state["response_previous"].em[0].risk_of_complications
        state["pc_critic_response"] = state["response_previous"].em[0].pc_critic_response
        state["dc_critic_response"] = state["response_previous"].em[0].dc_critic_response
        state["rc_critic_response"] = state["response_previous"].em[0].rc_critic_response
        state["encounter_type"] = [i.encounter_type for i in state["response_previous"].em]
    if num_few_shot_examples > 0:
        vdb_text = config["configurable"].get(CONFIG.VDB_TEXT, "text")
        if vdb_text == "justification":
            draft_problem_complexity = state["problem_complexity"]
            draft_data_complexity = state["data_complexity"]
            draft_risk_of_complications = state["risk_of_complications"]
            draft_pc_critic_response = state["pc_critic_response"]
            draft_dc_critic_response = state["dc_critic_response"]
            draft_rc_critic_response = state["rc_critic_response"]
            text_for_vdb = [
                draft_problem_complexity,
                draft_pc_critic_response,
                draft_data_complexity,
                draft_dc_critic_response,
                draft_risk_of_complications,
                draft_rc_critic_response,
            ]
            text_for_vdb = "\n".join(text_for_vdb)
        elif vdb_text == "text":
            text_for_vdb = state.get("text", state["soap_text_previous"])
        else:
            text_for_vdb = state["soap_text_previous"]
        vector_store = _init_or_get_vdb(config["configurable"][CONFIG.VDB_INDEX_PATH], config["configurable"][CONFIG.EMBEDDING_MODEL])
        # we get an additional example in case the first match is the current soap note itself
        results = vector_store.similarity_search_with_score(extract_indexable_text(text_for_vdb), k=num_few_shot_examples + 1)
        # filter results to ensure the same soap note is not part of the few shot examples
        # In production, "example_id" is missing in state
        if state.get("example_id"):
            results = [
                result
                for result in results
                if (result[0].metadata.get(METADATA_KEYS.SOAP_NOTE) != state["soap_text_previous"])
                and (result[0].metadata.get(METADATA_KEYS.SOAP_NOTE) != state["text"])
                and result[0].metadata.get("example_id") != state["example_id"]
            ]
        if len(results) == num_few_shot_examples:
            logger.debug("The current soap note was part of the few shot examples, and it has been removed to implement the leave-one-out approach.")
        else:
            if icl_mode == "knowledge_base":
                results = results[: int(config["configurable"][CONFIG.NUM_FEW_SHOT_EXAMPLES])]
            else:
                results = results[-int(config["configurable"][CONFIG.NUM_FEW_SHOT_EXAMPLES]) :]
        omit_soap_note = config["configurable"].get(CONFIG.OMIT_SOAP_NOTE, False)
        try:
            few_shot_examples = format_few_shot_examples(results, omit_soap_note=omit_soap_note, icl_mode=icl_mode)
        except KeyError as e:
            raise ValueError(
                f"Error while formatting few shot examples: {e}. "
                f"Make sure icl_mode {icl_mode} is compatible with "
                f"vdb_index_path {config['configurable'][CONFIG.VDB_INDEX_PATH]}."
            )
    else:
        few_shot_examples = ""

    omit_patient_problems = config["configurable"].get(CONFIG.OMIT_PATIENT_PROBLEMS, False)
    additional_info = "Encounter Date: " + datetime.now().strftime("%Y-%m-%d") + "\n"
    if state.get("all_patient_problems") and not omit_patient_problems:
        additional_info += "Problems addressed during the encounter:\n" + "\n".join(
            [f"- {problem.name}".replace("type=None", "") for problem in state.get("all_patient_problems", [])]
        )
    if state.get("orders_display"):
        additional_info += "\n\nOrders for the encounter:\n" + state.get("orders_display")

    if config["configurable"].get("skip_additional_info", False):
        additional_info = "Encounter Date: " + datetime.now().strftime("%Y-%m-%d") + "\n"

    note_diff, _, _ = create_markdown_diff(state["soap_text_previous"], state["text"])

    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION]
    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser = PydanticOutputParserWithRetries(pydantic_object=MDMResult, correction_model=model, num_retries=3)

    prompt = ChatPromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.REPREDICTION][prompt_version]["user"], template_format="jinja2"
    )
    prompt.format(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser
    mdm = chain.invoke(
        {
            "text": state["text"],
            "draft_note": state["soap_text_previous"],
            "note_diff": note_diff,
            "draft_enc_type_out": json.dumps([x.model_dump() for x in state["encounter_type"]], indent=2),
            "draft_pc": state["problem_complexity"].model_dump_json(indent=2),
            "draft_pc_response": state["pc_critic_response"],
            "draft_dc": state["data_complexity"].model_dump_json(indent=2),
            "draft_dc_response": state["dc_critic_response"],
            "draft_rc": state["risk_of_complications"].model_dump_json(indent=2),
            "draft_rc_response": state["rc_critic_response"],
            "few_shot_examples": few_shot_examples,
            "additional_info": additional_info,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return {
        "problem_complexity": mdm.problem_complexity,
        "data_complexity": mdm.data_complexity,
        "risk_of_complications": mdm.risk_of_complications,
        "mdm_results": mdm,
    }


def decision_tree_node(state: CptEmGraphState, config: RunnableConfig) -> CptEmOutputGraphState:
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
        "mdm_results": mdm,
    }


# ------------------------ defining the graph builder ---------------------------
def get_majority_vote(results: List[MDMResult]) -> MDMResult:
    bests = {}
    for field in MDMResult.model_fields:
        counts = Counter([getattr(result, field).level for result in results])
        most_common = counts.most_common(1)[0]
        bests[field] = [getattr(result, field) for result in results if getattr(result, field).level == most_common[0]][0]
        if hasattr(bests[field], "confidence"):
            bests[field].confidence = most_common[1] / len(results)
    return MDMResult(**bests)


def get_majority_with_index(alist: List[str]) -> Tuple[str, int]:
    # Count occurrences of each element
    counts = Counter(alist)
    # Get the most common element and its count
    most_common = counts.most_common(1)[0]
    # Find the first index of the most common element
    majority_index = alist.index(most_common[0])
    return most_common[0], majority_index


def get_aggregate_majority(parallel_results: List[CptEmGraphState]) -> CptEmGraphState:
    mdm_results = [
        MDMResult(
            problem_complexity=result["problem_complexity"],
            data_complexity=result["data_complexity"],
            risk_of_complications=result["risk_of_complications"],
        )
        for result in parallel_results
    ]
    mdm_final = get_majority_vote(mdm_results)

    return {
        "mdm_results": mdm_final,
        "problem_complexity": mdm_final.problem_complexity,
        "data_complexity": mdm_final.data_complexity,
        "risk_of_complications": mdm_final.risk_of_complications,
    }


def build_reprediction_mdm_only_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:
    # set the prompts and model config
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}}

    # workflow = StateGraph(CptEmGraphState)

    # workflow.add_node("classifier", prediction_node)
    # workflow.add_edge(START, "classifier")

    # workflow.add_edge("classifier", END)

    # try:
    #     args.num_repeats = int(args.num_repeats)
    # except Exception:
    #     args.num_repeats = 1

    # def batch_and_aggregate(state: CptEmGraphState, config: RunnableConfig):
    #     single_flow = workflow.compile()
    #     invokation = [state for _ in range(args.num_repeats)]
    #     parallel_res = single_flow.batch(invokation, config, return_exceptions=True)
    #     if all(isinstance(res, Exception) for res in parallel_res):
    #         raise ValueError(f"All parallel executions failed, no valid results to aggregate, {parallel_res}")
    #     parallel_res = [res for res in parallel_res if not isinstance(res, Exception)]
    #     aggregated_result = get_aggregate_majority(parallel_res)
    #     return aggregated_result

    parallel_workflow = StateGraph(CptEmGraphState, input=CptEmInputGraphState, output=CptEmOutputGraphState)

    parallel_workflow.add_node("encounter_type_classifier", get_encounter_type_subgraph(args))
    parallel_workflow.add_node("time_statement_extractor", get_time_statement_subgraph(args))
    parallel_workflow.add_node("classifier", prediction_node)
    parallel_workflow.add_node("decision_tree", decision_tree_node)
    parallel_workflow.add_edge(START, "classifier")
    parallel_workflow.add_edge(START, "time_statement_extractor")
    parallel_workflow.add_edge(START, "encounter_type_classifier")
    parallel_workflow.add_edge("encounter_type_classifier", "decision_tree")
    parallel_workflow.add_edge("time_statement_extractor", "decision_tree")
    parallel_workflow.add_edge("classifier", "decision_tree")
    parallel_workflow.add_edge("decision_tree", END)

    if not args.checkpoint_path:
        return parallel_workflow.compile(), model_config
    conn = sqlite3.connect(args.checkpoint_path, check_same_thread=False)
    checkpointer = SqliteCheckpointer(conn, serde=JsonPlusSerializer(pickle_fallback=True))

    return parallel_workflow.compile(checkpointer=checkpointer), model_config
