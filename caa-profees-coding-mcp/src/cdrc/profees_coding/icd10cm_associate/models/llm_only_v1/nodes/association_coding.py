import random
from argparse import Namespace
from typing import List, Literal, Optional, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, LinkedDiagnosis, Order, Problem, Procedure
from cdrc.profees_coding.icd10cm_associate.models.utils import CONFIG, PROMPTS

# --------------------------------------------------------------------------------
# Defining the LangGraph subgraph.
# --------------------------------------------------------------------------------


class InputState(dict):
    example_id: Optional[Union[int, str]]
    text: str
    patient_sex: Optional[str]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    all_patient_problems: Optional[List[Problem]]


class OutputState(dict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


class AssociationCodingStateGraph(InputState, OutputState):
    pass


def build_associate_coding_subgraph(args: Namespace):
    workflow = StateGraph(AssociationCodingStateGraph)
    workflow.add_node("association_classifier", association_coding)
    workflow.add_edge(START, "association_classifier")
    workflow.add_edge("association_classifier", END)
    return workflow.compile()


# --------------------------------------------------------------------------------
# Defining the Pydantic base model used by the LLM.
# --------------------------------------------------------------------------------


class OutputProcedure(BaseModel):
    code: str = Field(..., description="CPT/HCPCS procedure code")
    system: Literal["CPT", "HCPCS"]
    linked_diagnosis: List[LinkedDiagnosis] = Field(default_factory=list)


class OutputEMItem(BaseModel):
    code: str = Field(..., description="E/M CPT code")
    linked_diagnosis: List[LinkedDiagnosis] = Field(default_factory=list)


class OutputModel(BaseModel):
    procedure: List[OutputProcedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[OutputEMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


# --------------------------------------------------------------------------------
# The LangGraph association coding node.
# --------------------------------------------------------------------------------


@track_tokens("model_usage")
def association_coding(state: AssociationCodingStateGraph, config: RunnableConfig) -> dict:

    # Choose and instantiate model based on runtime configuration
    model_kwargs = _get_model_kwargs(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    # Output parser with retry/correction logic
    parser = PydanticOutputParserWithRetries(pydantic_object=OutputModel, correction_model=model, num_retries=3)
    chain = model | parser  # Compose chain of model and parsing logic

    # Build system/user prompts with current templates
    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION]
    system_prompt = SystemMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_ONLY][prompt_version]["system"], template_format="jinja2"
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_ONLY][prompt_version]["user"], template_format="jinja2"
    )

    # Compose the full message sequence including format instructions
    messages = [
        system_prompt.format(),
        user_prompt.format(
            note=state["text"],
            patient_age=state.get("patient_age", ""),
            patient_sex=state.get("patient_sex", ""),
            orders_display=state.get("orders_display", ""),
            em=state["em"],
            procedure=state["procedure"],
            diagnosis=state["diagnosis"],
            format_instructions=parser.get_format_instructions(),
        ),
    ]

    # Since o1-preview and mini models do not support system messages, need to concatenate both into user message.
    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)

    # Call LLM chain and get parsed structured result
    response = chain.invoke(messages)

    # Take the model response and build the association output.
    output = _build_output_payload(response, state)

    return output


# --------------------------------------------------------------------------------
# Utility functions.
# --------------------------------------------------------------------------------


def _get_model_kwargs(config: dict) -> dict:

    # Extract model kwargs, use default seed if not set
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))

    # For reproducibility, randomize the seed when repeating model calls
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)

    return model_kwargs


def _build_output_payload(response: OutputModel, state: AssociationCodingStateGraph):

    def _dump(x):  # Helper: Obtain dict representation from pydantic or regular objects
        return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)

    def _norm_code(x):  # Standardize code string for lookup (used for EM items)
        return str((_dump(x) or {}).get("code", "")).strip().upper()

    def _norm_key(x):  # Standardize (code, system) tuple for CPT/HCPCS lookups
        d = _dump(x) or {}
        return str(d.get("code", "")).strip().upper(), (d.get("system") or "").strip().upper()

    # Build lookup table: map EM code to predicted linked diagnoses
    em_code_to_linked_diagnoses = {}
    for em_item in response.em or []:
        em_dict = _dump(em_item) or {}
        em_code = _norm_code(em_dict)
        if em_code:
            em_code_to_linked_diagnoses[em_code] = [_dump(linked_dx) for linked_dx in (em_dict.get("linked_diagnosis") or [])]

    # Build lookup table: map (procedure code, system) to predicted linked diagnoses
    procedure_key_to_linked_diagnoses = {}
    for procedure_item in response.procedure or []:
        procedure_dict = _dump(procedure_item) or {}
        procedure_key = _norm_key(procedure_dict)
        if all(procedure_key):
            procedure_key_to_linked_diagnoses[procedure_key] = [_dump(linked_dx) for linked_dx in (procedure_dict.get("linked_diagnosis") or [])]

    # Rebuild EM output objects, updating linked diagnoses.
    output_em_items = []
    for input_em_item in state.get("em") or []:
        input_em_dict = (_dump(input_em_item) or {}).copy()
        input_em_dict["linked_diagnosis"] = em_code_to_linked_diagnoses.get(_norm_code(input_em_dict), [])
        output_em_items.append(EMItem(**input_em_dict))

    # Rebuild procedure output objects, updating linked diagnoses.
    output_procedure_items = []
    for input_procedure_item in state.get("procedure") or []:
        input_procedure_dict = (_dump(input_procedure_item) or {}).copy()
        input_procedure_dict["linked_diagnosis"] = procedure_key_to_linked_diagnoses.get(_norm_key(input_procedure_dict), [])
        output_procedure_items.append(Procedure(**input_procedure_dict))

    # Returning the CPT-E/M and procedure outputs with the linked diagnoses.
    return {
        "em": output_em_items,
        "procedure": output_procedure_items,
        "diagnosis": [ICD10CM(**_dump(d)) for d in (response.diagnosis or [])],
    }
