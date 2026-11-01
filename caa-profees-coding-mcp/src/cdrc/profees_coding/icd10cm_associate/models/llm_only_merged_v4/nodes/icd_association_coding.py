import copy
import random
import re
from typing import List, Optional, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.icd10cm_associate.data.schema import EMItem, Order, Problem, Procedure
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


class IntermediateState(dict):
    procedure_associations: Optional[List]
    em_associations: Optional[List]


class OutputState(dict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")


# Working state for this subgraph only. Final diagnosis output is built in post-processing.
class AssociationCodingStateGraph(InputState, IntermediateState, OutputState):
    pass


def build_merged_association_coding_subgraph(args):

    workflow = StateGraph(AssociationCodingStateGraph)

    # Add new normalization node at the start
    workflow.add_node("prepare_state", prepare_state_node)
    workflow.add_node("populate_description", populate_description_node)
    workflow.add_node("model_inference", model_inference_node)

    workflow.add_edge(START, "prepare_state")
    workflow.add_edge("prepare_state", "populate_description")
    workflow.add_edge("populate_description", "model_inference")
    workflow.add_edge("model_inference", END)

    return workflow.compile()


# --------------------------------------------------------------------------------
# State Preprocessing
# --------------------------------------------------------------------------------


def prepare_state_node(state: AssociationCodingStateGraph, config: RunnableConfig):

    # Convert inner state dictionaries in the procedure list to Procedure instances if not already.
    state["procedure"] = [charge if isinstance(charge, Procedure) else Procedure(**charge) for charge in (state.get("procedure") or [])]

    # Convert inner state dictionaries in the em list to EMItem instances if not already.
    state["em"] = [charge if isinstance(charge, EMItem) else EMItem(**charge) for charge in (state.get("em") or [])]

    # Return the correctly typed state.
    return state


# --------------------------------------------------------------------------------
# Model Inference Node
# --------------------------------------------------------------------------------


def populate_description_node(state: AssociationCodingStateGraph, config: RunnableConfig):

    # Early exit if populate description is not enabled.
    if not config["configurable"].get("populate_code_descriptions", False):
        return state

    # Get resources dictionary containing all code-description mappings
    resources = config["configurable"].get("resources", {})

    # Extract individual lookup tables for the respective code types
    cpt_descriptions = resources.get("cpt_descriptions", {})
    hcpcs_descriptions = resources.get("hcpcs_descriptions", {})

    # Populate descriptions for Procedure items
    for procedure in state.get("procedure", []) or []:

        # Always replace description if found in the mapping dictionaries.
        description = cpt_descriptions.get(procedure.code) or hcpcs_descriptions.get(procedure.code)
        procedure.description = description if description else procedure.description

    # Populate descriptions for EM items
    for em in state.get("em", []) or []:

        # Always replace description if found in the mapping dictionaries.
        description = cpt_descriptions.get(em.code)
        em.description = description if description else em.description

    return state


# --------------------------------------------------------------------------------
# Model Inference Node
# --------------------------------------------------------------------------------


class OutputDiagnosis(BaseModel):
    code: str = Field(..., description="ICD-10-CM code.")
    justification: str = Field(..., description="Reason for why an association was made.")

    @model_validator(mode="before")
    def normalize_and_validate_code(cls, values):

        # Ensure model validators only process dictionary-like payloads.
        if not isinstance(values, dict):
            return values

        # Normalize the diagnosis code before parser validation.
        code = str(values.get("code", "")).strip().upper()
        values["code"] = code

        # Validate ICD-10-CM code format to trigger parser correction retries when malformed.
        if not re.match(r"^[A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?$", code):
            raise ValueError(f"Invalid ICD-10-CM code: {code}")

        return values


class OutputProcedure(BaseModel):
    code: str = Field(..., description="CPT/HCPCS procedure code.")
    linked_diagnoses: List[OutputDiagnosis] = Field(default_factory=list)


class OutputCPTEM(BaseModel):
    code: str = Field(..., description="CPT-E/M code.")
    linked_diagnoses: List[OutputDiagnosis] = Field(default_factory=list)


class OutputModel(BaseModel):
    em_associations: List[OutputCPTEM] = Field(default_factory=list, description="List of CPT-E/M codes associations.")
    procedure_associations: List[OutputProcedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures associations.")


@track_tokens("model_usage")
def model_inference_node(state: AssociationCodingStateGraph, config: RunnableConfig):

    # Choose and instantiate model based on runtime configuration
    model_kwargs = _get_model_kwargs(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    # Output parser with retry/correction logic
    parser = PydanticOutputParserWithRetries(pydantic_object=OutputModel, correction_model=model, num_retries=3)

    # Building the prompt used for the LLM call.
    messages = _build_prompt(state, config, parser)

    # Compose chain of model and parsing logic
    chain = model | parser

    # Call LLM chain and get parsed structured result
    response = chain.invoke(messages)

    # Ensuring both the response and state are in dictionary representation.
    response = _dump_to_dictionary(response)

    # Saving the model's response to the graph state.
    state["procedure_associations"] = response["procedure_associations"]
    state["em_associations"] = response["em_associations"]

    # Persist a deep-copied snapshot for report export when debugging is enabled.
    if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
        state["intermediate_raw_associations"] = _build_association_snapshot(
            state.get("em_associations", []),
            state.get("procedure_associations", []),
        )

    return state


def _get_model_kwargs(config: dict) -> dict:

    # Extract model kwargs, use default seed if not set
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))

    # For reproducibility, randomize the seed when repeating model calls
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)

    return model_kwargs


def _build_prompt(state, config, parser):

    # Extracting the system and user prompts.
    prompt_key = config["configurable"].get(CONFIG.PROMPT_KEY, PROMPTS.OUTPATIENT)
    if prompt_key not in (PROMPTS.OUTPATIENT, PROMPTS.ED):
        raise ValueError(f"Invalid prompt_key '{prompt_key}'. Allowed values: {[PROMPTS.OUTPATIENT, PROMPTS.ED]}")

    prompts = config["configurable"][CONFIG.PROMPTS]
    if prompt_key not in prompts:
        raise KeyError(f"Missing prompt key '{prompt_key}' in prompts config. Available keys: {sorted(prompts.keys())}")

    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION]
    prompt_bank = prompts[prompt_key]
    if prompt_version not in prompt_bank:
        raise KeyError(f"Missing prompt version '{prompt_version}' under prompt key '{prompt_key}'. Available versions: {sorted(prompt_bank.keys())}")

    system_prompt = SystemMessagePromptTemplate.from_template(prompt_bank[prompt_version]["system"], template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(prompt_bank[prompt_version]["user"], template_format="jinja2")

    # Constructing the final message list.
    messages = [
        system_prompt.format(),
        user_prompt.format(
            note=state["text"],
            patient_age=state.get("patient_age", ""),
            patient_sex=state.get("patient_sex", ""),
            orders_display=state.get("orders_display", ""),
            em=_format_codes(state.get("em", [])),
            procedure=_format_codes(state.get("procedure", [])),
            format_instructions=parser.get_format_instructions(),
        ),
    ]

    # Since o1-preview and mini models do not support system messages, need to concatenate both into user message.
    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)

    return messages


def _format_codes(code_type, code_field="code", desc_field="description"):
    codes = []

    # Iterating over all codes from a specific field (i.e., em, procedure, diagnosis).
    for item in code_type or []:

        # Extracting the codes and description fields using attribute access.
        code = getattr(item, code_field, "")
        desc = getattr(item, desc_field, "")

        # Formatting to - D36.13 (Benign neoplasm...)
        if code and desc:
            codes.append(f"- {code} ({desc})")
        elif code:
            codes.append(f"- {code}")

    return "\n".join(codes)


def _dump_to_dictionary(x):

    # Helper: Obtain dict representation from pydantic or regular objects.
    return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)


def _build_association_snapshot(em_associations, procedure_associations):

    # Snapshot as plain deep-copied dictionaries so later node mutations don't back-propagate.
    return {
        "em_associations": [copy.deepcopy(_dump_to_dictionary(association)) for association in (em_associations or [])],
        "procedure_associations": [copy.deepcopy(_dump_to_dictionary(association)) for association in (procedure_associations or [])],
    }
