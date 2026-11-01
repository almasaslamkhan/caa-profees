import random
from typing import List, Optional, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Order, Problem, Procedure
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
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


class AssociationCodingStateGraph(InputState, IntermediateState, OutputState):
    pass


def build_merged_association_coding_subgraph(args):

    workflow = StateGraph(AssociationCodingStateGraph)

    # Add new normalization node at the start
    workflow.add_node("prepare_state", prepare_state_node)
    workflow.add_node("populate_description", populate_description_node)
    workflow.add_node("model_inference", model_inference_node)
    workflow.add_node("payload_contruction", payload_contruction_node)

    workflow.add_edge(START, "prepare_state")
    workflow.add_edge("prepare_state", "populate_description")
    workflow.add_edge("populate_description", "model_inference")
    workflow.add_edge("model_inference", "payload_contruction")
    workflow.add_edge("payload_contruction", END)

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


class OutputLinkedDiagnosis(BaseModel):
    code: str = Field(..., description="ICD-10-CM code.")
    justification: str = Field(..., description="Reason for why an association was made.")


class OutputProcedure(BaseModel):
    code: str = Field(..., description="CPT/HCPCS procedure code.")
    linked_diagnoses: List[OutputLinkedDiagnosis] = Field(default_factory=list)


class OutputCPTEM(BaseModel):
    code: str = Field(..., description="CPT-E/M code.")
    linked_diagnoses: List[OutputLinkedDiagnosis] = Field(default_factory=list)


class OutputModel(BaseModel):
    icd_10_cm_codes: List[ICD10CM] = Field(default_factory=list, description="List of ICD-10-CM codes for this note.")
    procedure_associations: List[OutputProcedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures associations.")
    em_associations: List[OutputCPTEM] = Field(default_factory=list, description="List of CPT-E/M codes associations.")


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
    state["diagnosis"] = [ICD10CM(**diagnosis) for diagnosis in response["icd_10_cm_codes"]]
    state["procedure_associations"] = response["procedure_associations"]
    state["em_associations"] = response["em_associations"]

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
    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION]
    system_prompt = SystemMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_ONLY][prompt_version]["system"], template_format="jinja2"
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_ONLY][prompt_version]["user"], template_format="jinja2"
    )

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


# --------------------------------------------------------------------------------
# Payload Construction Node
# --------------------------------------------------------------------------------


@track_tokens("model_usage")
def payload_contruction_node(state: AssociationCodingStateGraph, config: RunnableConfig):

    # Build a lookup dictionary of all the ICD-10 codes and their content.
    diagnoses = _build_diagnosis_mapping(state.get("diagnosis", []))

    # Build a lookup dictionary of all CPT-E/M and procedure charges.
    em_associations = _build_association_mapping(state.get("em_associations", []), diagnoses)
    procedure_associations = _build_association_mapping(state.get("procedure_associations", []), diagnoses)

    # Building the final em list with the new linked diagnoses field.
    em = []
    for em_charge in state.get("em") or []:
        charge = _dump_to_dictionary(em_charge) or {}
        charge_code = str(charge.get("code", "")).strip().upper()
        charge["linked_diagnosis"] = em_associations.get(charge_code, [])
        em.append(EMItem(**charge))

    # Building the final procedure list with the new linked diagnoses field.
    procedure = []
    for procedure_charge in state.get("procedure") or []:
        charge = _dump_to_dictionary(procedure_charge) or {}
        charge_code = str(charge.get("code", "")).strip().upper()
        charge["linked_diagnosis"] = procedure_associations.get(charge_code, [])
        procedure.append(Procedure(**charge))

    # Updating the procedure and em state.
    state["em"] = em
    state["procedure"] = procedure

    return state


def _build_diagnosis_mapping(diagnosis_list):

    # Build a lookup dictionary of all the ICD-10 codes and their content.
    diagnoses = {}

    # Iterating over the diagnosis list.
    for dx in diagnosis_list:

        # Converting the dx object into a dictionary.
        dx_dictionary = _dump_to_dictionary(dx)

        # Extracting the ICD-10 code for the diagnosis dictionary.
        diagnosis_code = str(dx_dictionary.get("code", "")).strip().upper()

        # Mapping from ICD-10 code to content (is_primary, system, description, justification).
        diagnoses[diagnosis_code] = dx_dictionary

    return diagnoses


def _build_association_mapping(charge_list, diagnoses):

    # Dictionary storing the mapping between E/M or procedure code to their linked diagnoses.
    associations = {}

    # Iterating over each CPT-E/M code and processing its associations.
    for charge in charge_list:

        # Converting to dictionary and normalizing codes.
        charge_dictionary = _dump_to_dictionary(charge)

        # Extracting the code from the charge item.
        charge_code = str(charge_dictionary.get("code", "")).strip().upper()

        # Build lookup table which maps each code to their linked diagnoses.
        if charge_code:

            # Storing the updated linked diagnoses.
            linked_diagnoses = []

            # Iterating over each linked diagnosis for the current code.
            for i, linked_dx in enumerate(charge_dictionary.get("linked_diagnoses") or []):

                # Dumping each linked dx to a dictionary.
                linked_diagnosis = _dump_to_dictionary(linked_dx)

                # Extracting the linked diagnose code.
                dx_code = linked_diagnosis.get("code", "")

                # Only include the association if the dx code is in the diagnoses list.
                if dx_code in diagnoses:

                    # If the linked diagnose is the first in the list then set is_primary to true.
                    is_primary = i == 0

                    # Extracting the code system field.
                    system = diagnoses.get(dx_code, {}).get("system", "")

                    # Extracting the diagnosis description field.
                    description = diagnoses.get(dx_code, {}).get("description", "")

                    # Extracting the association justification.
                    justification = linked_diagnosis.get("justification", "")

                    # Appending to the tracking list.
                    linked_diagnoses.append(
                        {"code": dx_code, "is_primary": is_primary, "system": system, "description": description, "justification": justification}
                    )

            # Adding new code -> linked diagnosis mapping.
            associations[charge_code] = linked_diagnoses

    return associations
