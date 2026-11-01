import copy
import json
import random
from argparse import Namespace
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.messages import AnyMessage
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from cdrc.common.embeddings_utils import get_embedding_models
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.deploy.api.utils import ProfeesResponse
from cdrc.profees_coding.em.data.schema import (
    ComplexityDetails,
    ComplexityLevel,
    EMItem,
    EncounterTypeDetails,
    PatientType,
    Problem,
    ProblemComplexityType,
)
from cdrc.profees_coding.em.models.utils import create_markdown_diff

from ..data_classes import CONFIG, METADATA_KEYS, PROMPTS
from ..preprocessing import MDMResult, extract_indexable_text

vector_store = None


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
class MdmComplexityGraphState(TypedDict, total=False):
    example_id: Union[int, str, None]
    soap_text_previous: str
    text: str
    note_diff: str

    orders_display: str
    patient_type: PatientType

    response_previous: Optional[ProfeesResponse]
    # encounter_type: List[EncounterTypeDetails]
    em: List[EMItem]
    pc_critic_response: str
    rc_critic_response: str
    dc_critic_response: str
    problem_complexity: ComplexityDetails
    data_complexity: ComplexityDetails
    risk_of_complications: ComplexityDetails

    draft_pc_critic_response: str
    draft_rc_critic_response: str
    draft_dc_critic_response: str
    draft_problem_complexity: ComplexityDetails
    draft_data_complexity: ComplexityDetails
    draft_risk_of_complications: ComplexityDetails

    mdm_results: MDMResult
    messages: List[AnyMessage]
    additional_info: str
    few_shot_examples: str


# ------------------------ defining the graph nodes ---------------------------
@track_tokens("model_usage")
def init_node(state: MdmComplexityGraphState, config: RunnableConfig) -> MdmComplexityGraphState:
    num_few_shot_examples = int(config["configurable"][CONFIG.NUM_FEW_SHOT_EXAMPLES])
    icl_mode = config["configurable"].get(CONFIG.ICL_MODE, "standard")
    if "response_previous" in state and state["response_previous"]:
        state["problem_complexity"] = state["response_previous"].em[0].problem_complexity
        state["data_complexity"] = state["response_previous"].em[0].data_complexity
        state["risk_of_complications"] = state["response_previous"].em[0].risk_of_complications
        state["pc_critic_response"] = state["response_previous"].em[0].pc_critic_response
        state["dc_critic_response"] = state["response_previous"].em[0].dc_critic_response
        state["rc_critic_response"] = state["response_previous"].em[0].rc_critic_response
        # state["encounter_type"] = [i.encounter_type for i in state["response_previous"].em]
    if num_few_shot_examples > 0:
        vdb_text = config["configurable"].get(CONFIG.VDB_TEXT, "soap_text_previous")
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

    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION]
    system_prompt = SystemMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.SYSTEM][prompt_version], template_format="jinja2"
    )

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

    messages = [
        system_prompt.format(few_shot_examples=few_shot_examples),
    ]

    return {
        "mdm_results": "",  # to keep the before-critic response in the output
        "problem_complexity": "",
        "data_complexity": "",
        "risk_of_complications": "",
        "messages": messages,
        "additional_info": additional_info,
        "few_shot_examples": few_shot_examples,
        "note_diff": note_diff,
        "draft_problem_complexity": state["problem_complexity"],
        "draft_data_complexity": state["data_complexity"],
        "draft_risk_of_complications": state["risk_of_complications"],
        "draft_pc_critic_response": state["pc_critic_response"],
        "draft_dc_critic_response": state["dc_critic_response"],
        "draft_rc_critic_response": state["rc_critic_response"],
        # "encounter_type": state["encounter_type"],
    }


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# multi critic nodes
class ProblemDetails(BaseModel):
    justification: str = Field("", description="The justification for the type of problem complexity based on guidelines and instructions.")
    name: str = Field(..., description="The name of the problem")
    type: ProblemComplexityType = Field(..., description="The type of problem complexity")


class ProblemComplexityList(BaseModel):
    chain_of_thought: str = Field("", description="Your chain of thought reasoning for the criticism based on the instructions provided.")
    problems: List[ProblemDetails] = Field(description="A list of problems with their names and types.")


@track_tokens("model_usage")
def pc_critic_node(state: MdmComplexityGraphState, config: RunnableConfig) -> MdmComplexityGraphState:
    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser = PydanticOutputParserWithRetries(pydantic_object=ProblemComplexityList, correction_model=model, num_retries=3)
    messages = copy.deepcopy(state.get("messages"))
    # append human message with request for criticism
    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION_PC]
    user_prompt = HumanMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.PROBLEM_COMPLEXITY][prompt_version], template_format="jinja2"
    )

    text = state["text"]
    draft_note = state["soap_text_previous"]
    note_diff = state["note_diff"]
    additional_info = state["additional_info"]
    few_shot_examples = state["few_shot_examples"]

    user_prompt = user_prompt.format(
        text=text,
        draft_note=draft_note,
        note_diff=note_diff,
        # draft_enc_type_out=json.dumps([x.model_dump() for x in state["encounter_type"]], indent=2),
        draft_pc=state["draft_problem_complexity"].model_dump_json(indent=2),
        draft_pc_response=state["draft_pc_critic_response"],
        draft_dc=state["draft_data_complexity"].model_dump_json(indent=2),
        draft_dc_response=state["draft_dc_critic_response"],
        draft_rc=state["draft_risk_of_complications"].model_dump_json(indent=2),
        draft_rc_response=state["draft_rc_critic_response"],
        additional_info=additional_info,
        few_shot_examples=few_shot_examples,
        format_instructions=parser.get_format_instructions(),
    )
    messages.append(user_prompt)

    response = model.invoke(messages)
    parsed_response = parser.invoke(response)
    pc = get_problem_complexity_from_problem_list(parsed_response.problems)
    return {"problem_complexity": pc, "pc_critic_response": parsed_response.chain_of_thought}


def get_problem_complexity_from_problem_list(problems: List[ProblemDetails]) -> ComplexityDetails:
    """
    Calculate the overall complexity level based on a list of problems.

    Args:
        problems: List of Problem objects, each with a type attribute of ProblemComplexityType

    Returns:
        ComplexityDetails: The overall complexity level and justification based on the problems
    """
    # Count occurrences of each problem type and track problems by type
    problem_counts = {ptype: 0 for ptype in ProblemComplexityType}
    problems_by_type = {ptype: [] for ptype in ProblemComplexityType}

    for problem in problems:
        if problem.type:  # Only count problems that have a type assigned
            problem_counts[problem.type] += 1
            problems_by_type[problem.type].append(problem.name)

    # Determine complexity level and build justification
    if problem_counts[ProblemComplexityType.CHRONIC_SEVERE] >= 1 or problem_counts[ProblemComplexityType.LIFE_THREATENING] >= 1:
        level = ComplexityLevel.HIGH
    elif (
        problem_counts[ProblemComplexityType.CHRONIC_EXACERBATION] >= 1
        or problem_counts[ProblemComplexityType.STABLE_CHRONIC] >= 2
        or problem_counts[ProblemComplexityType.UNDIAGNOSED_UNCERTAIN] >= 1
        or problem_counts[ProblemComplexityType.ACUTE_SYSTEMIC] >= 1
        or problem_counts[ProblemComplexityType.ACUTE_COMPLICATED] >= 1
    ):
        level = ComplexityLevel.MODERATE
    elif (
        problem_counts[ProblemComplexityType.SELF_LIMITED] >= 2
        or problem_counts[ProblemComplexityType.STABLE_CHRONIC] == 1
        or problem_counts[ProblemComplexityType.ACUTE_UNCOMPLICATED] >= 1
        or problem_counts[ProblemComplexityType.STABLE_ACUTE] >= 1
        or problem_counts[ProblemComplexityType.ACUTE_UNCOMPLICATED_INPATIENT] >= 1
    ):
        level = ComplexityLevel.LOW
    elif problem_counts[ProblemComplexityType.SELF_LIMITED] == 1:
        level = ComplexityLevel.MINIMAL
    else:
        if all(count == 0 for count in problem_counts.values()):
            # raise ValueError("No problems with valid types found")
            return ComplexityDetails(level=ComplexityLevel.MINIMAL, justification="No problems with valid types found")
        else:
            # raise ValueError("No valid problem complexity combination found")
            return ComplexityDetails(level=ComplexityLevel.MINIMAL, justification="No valid problem complexity combination found")

    # Build justification string
    justification_parts = []
    for ptype in ProblemComplexityType:
        if problem_counts[ptype] > 0:
            problem_names = problems_by_type[ptype]
            justification_parts.append(f"{problem_counts[ptype]} {ptype.value}: {', '.join(problem_names)}")

    justification = ". ".join(justification_parts)
    justification += ".\n"
    justification += "\n".join([f"{problem.name}: {problem.justification}" for problem in problems])

    return ComplexityDetails(level=level, justification=justification)


class TandDComplexityCategory(BaseModel):
    reasoning: str = Field("", description="The justification of the count of elements in the category.")
    count: int = Field(0, description="Number of tests and documents reviewed or ordered.")


class OtherDataComplexityCategory(BaseModel):
    reasoning: str = Field("", description="If present, provide a justification for the presence of the element.")
    is_present: bool = Field(False, description="Whether the element is present or not.")


class MDMAuditSheet(BaseModel):
    chain_of_thought: str = Field("", description="The chain of thought reasoning for the data complexity.")
    tests_and_documents: Optional[TandDComplexityCategory] = Field(
        description="Tests & Documents (T&D) - including external notes reviewed, unique tests reviewed, unique tests ordered."
    )
    IHx: Optional[OtherDataComplexityCategory] = Field(description="Assessment requiring an independent historian(s) (IHx)")
    Intpr: Optional[OtherDataComplexityCategory] = Field(description="Independent interpretation of tests (Intpr)")
    DISC: Optional[OtherDataComplexityCategory] = Field(description="Discussion of management or test interpretation (DISC)")

    @model_validator(mode="after")
    def ensure_no_none_values(self) -> "MDMAuditSheet":
        if self.tests_and_documents is None:
            self.tests_and_documents = TandDComplexityCategory()
        if self.IHx is None:
            self.IHx = OtherDataComplexityCategory()
        if self.Intpr is None:
            self.Intpr = OtherDataComplexityCategory()
        if self.DISC is None:
            self.DISC = OtherDataComplexityCategory()
        return self


@track_tokens("model_usage")
def dc_critic_node(state: MdmComplexityGraphState, config: RunnableConfig) -> MdmComplexityGraphState:
    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser = PydanticOutputParserWithRetries(pydantic_object=MDMAuditSheet, correction_model=model, num_retries=3)
    messages = copy.deepcopy(state.get("messages"))
    # append human message with request for criticism
    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION_DC]
    user_prompt = HumanMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.DATA_COMPLEXITY][prompt_version], template_format="jinja2"
    )
    text = state["text"]
    draft_note = state["soap_text_previous"]
    note_diff = state["note_diff"]
    additional_info = state["additional_info"]
    few_shot_examples = state["few_shot_examples"]

    user_prompt = user_prompt.format(
        text=text,
        draft_note=draft_note,
        note_diff=note_diff,
        # draft_enc_type_out=json.dumps([x.model_dump() for x in state["encounter_type"]], indent=2),
        draft_pc=state["draft_problem_complexity"].model_dump_json(indent=2),
        draft_pc_response=state["draft_pc_critic_response"],
        draft_dc=state["draft_data_complexity"].model_dump_json(indent=2),
        draft_dc_response=state["draft_dc_critic_response"],
        draft_rc=state["draft_risk_of_complications"].model_dump_json(indent=2),
        draft_rc_response=state["draft_rc_critic_response"],
        additional_info=additional_info,
        few_shot_examples=few_shot_examples,
        format_instructions=parser.get_format_instructions(),
    )
    messages.append(user_prompt)

    response = model.invoke(messages)
    parsed_response = parser.invoke(response)
    final_response = get_dc(parsed_response)
    return {"data_complexity": final_response, "dc_critic_response": parsed_response.chain_of_thought}


def get_dc(audit_sheet: MDMAuditSheet) -> ComplexityDetails:
    def _format_data_complexity_justification() -> str:
        TnD, IHx, Intpr, DISC = audit_sheet.tests_and_documents, audit_sheet.IHx, audit_sheet.Intpr, audit_sheet.DISC
        justification = ""
        if TnD.count:
            justification += f"Tests & Documents: {TnD.count} elements present. {TnD.reasoning}\n"
        if IHx.is_present:
            justification += f"Assessment requiring an independent historian(s): {IHx.reasoning}\n"
        if Intpr.is_present:
            justification += f"Independent interpretation of tests: {Intpr.reasoning}\n"
        if DISC.is_present:
            justification += f"Discussion of management or test interpretation: {DISC.reasoning}\n"
        return justification.strip()

    TnD, IHx, Intpr, DISC = audit_sheet.tests_and_documents, audit_sheet.IHx, audit_sheet.Intpr, audit_sheet.DISC
    justification = _format_data_complexity_justification()
    category_1 = (TnD.count + IHx.is_present) >= 3
    category_2 = Intpr.is_present
    category_3 = DISC.is_present
    # high case at least 2 of 3 categories
    if (category_1 + category_2 + category_3) >= 2:
        return ComplexityDetails(level=ComplexityLevel.HIGH, justification=f"Extensive Data Elements Present. {justification}")
    # moderate case at least 1 of 3 categories
    if (category_1 + category_2 + category_3) == 1:
        return ComplexityDetails(level=ComplexityLevel.MODERATE, justification=f"Moderate Data Elements Present. {justification}")
    # low case either 2 documents or an independent historian
    if TnD.count == 2 or IHx.is_present:
        return ComplexityDetails(level=ComplexityLevel.LOW, justification=f"Limited Data Elements Present. {justification}")
    # minimal case, less than 2 test/documents and no independent historian, interpretation or discussion
    if TnD.count < 2:
        return ComplexityDetails(level=ComplexityLevel.MINIMAL, justification=f"Minimal Data Elements Present. {justification}")


class RCComplexityCategory(BaseModel):
    chain_of_thought: str = Field("", description="The chain of thought reasoning for the risk of complications.")
    risk_of_complications: ComplexityDetails = Field(..., description="The risk of complications.")


@track_tokens("model_usage")
def rc_critic_node(state: MdmComplexityGraphState, config: RunnableConfig) -> MdmComplexityGraphState:
    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser = PydanticOutputParserWithRetries(pydantic_object=RCComplexityCategory, correction_model=model, num_retries=3)

    # append human message with request for criticism
    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION_RC]
    user_prompt = HumanMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.RISK_OF_COMPLICATIONS][prompt_version], template_format="jinja2"
    )
    messages = copy.deepcopy(state.get("messages"))
    text = state["text"]
    draft_note = state["soap_text_previous"]
    note_diff = (state["note_diff"],)
    additional_info = state["additional_info"]

    user_prompt = user_prompt.format(
        text=text,
        draft_note=draft_note,
        note_diff=note_diff,
        # draft_enc_type_out=json.dumps([x.model_dump() for x in state["encounter_type"]], indent=2),
        draft_pc=state["draft_problem_complexity"].model_dump_json(indent=2),
        draft_pc_response=state["draft_pc_critic_response"],
        draft_dc=state["draft_data_complexity"].model_dump_json(indent=2),
        draft_dc_response=state["draft_dc_critic_response"],
        draft_rc=state["draft_risk_of_complications"].model_dump_json(indent=2),
        draft_rc_response=state["draft_rc_critic_response"],
        additional_info=additional_info,
        format_instructions=parser.get_format_instructions(),
    )
    messages.append(user_prompt)
    response = model.invoke(messages)
    parsed_response = parser.invoke(response)

    return {"risk_of_complications": parsed_response.risk_of_complications, "rc_critic_response": parsed_response.chain_of_thought}


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


def get_aggregate_majority(parallel_results: List[MdmComplexityGraphState]) -> MdmComplexityGraphState:
    mdm_results = [
        MDMResult(
            problem_complexity=result["problem_complexity"],
            data_complexity=result["data_complexity"],
            risk_of_complications=result["risk_of_complications"],
        )
        for result in parallel_results
    ]
    mdm_final = get_majority_vote(mdm_results)
    pc_index = get_majority_with_index([result["problem_complexity"].level for result in parallel_results])[1]
    rc_index = get_majority_with_index([result["risk_of_complications"].level for result in parallel_results])[1]
    dc_index = get_majority_with_index([result["data_complexity"].level for result in parallel_results])[1]
    return {
        "mdm_results": mdm_final,
        "problem_complexity": mdm_final.problem_complexity,
        "data_complexity": mdm_final.data_complexity,
        "risk_of_complications": mdm_final.risk_of_complications,
        "pc_critic_response": parallel_results[pc_index]["pc_critic_response"],
        "rc_critic_response": parallel_results[rc_index]["rc_critic_response"],
        "dc_critic_response": parallel_results[dc_index]["dc_critic_response"],
        # "encounter_type": parallel_results[0]["encounter_type"],
    }


def build_mdm_complexity_subgraph(args: Namespace):
    workflow = StateGraph(MdmComplexityGraphState)

    workflow.add_node("initialise", init_node)
    workflow.add_node("pc_critic", pc_critic_node)
    workflow.add_node("dc_critic", dc_critic_node)
    workflow.add_node("rc_critic", rc_critic_node)
    workflow.add_edge(START, "initialise")
    workflow.add_edge("initialise", "pc_critic")
    workflow.add_edge("initialise", "dc_critic")
    workflow.add_edge("initialise", "rc_critic")
    workflow.add_edge("pc_critic", END)
    workflow.add_edge("dc_critic", END)
    workflow.add_edge("rc_critic", END)

    try:
        args.num_repeats = int(args.num_repeats)
    except Exception:
        args.num_repeats = 1

    def batch_and_aggregate(state: MdmComplexityGraphState, config: RunnableConfig):
        single_flow = workflow.compile()
        invokation = [state for _ in range(args.num_repeats)]
        parallel_res = single_flow.batch(invokation, config, return_exceptions=True)
        if all(isinstance(res, Exception) for res in parallel_res):
            raise ValueError(f"All parallel executions failed, no valid results to aggregate, {parallel_res}")
        parallel_res = [res for res in parallel_res if not isinstance(res, Exception)]
        aggregated_result = get_aggregate_majority(parallel_res)
        return aggregated_result

    parallel_workflow = StateGraph(MdmComplexityGraphState)
    parallel_workflow.add_node("batch_and_aggregate", batch_and_aggregate)
    parallel_workflow.add_edge(START, "batch_and_aggregate")
    parallel_workflow.add_edge("batch_and_aggregate", END)

    return parallel_workflow.compile()
