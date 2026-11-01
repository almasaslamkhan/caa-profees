import random
from argparse import Namespace
from typing import List, Literal, Optional, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
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
    procedure: List[OutputProcedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures.")
    em: List[OutputCPTEM] = Field(default_factory=list, description="List of CPT-E/M codes.")


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

    # Building the prompt used for the LLM call.
    messages = _build_prompt(state, config, parser)

    # Call LLM chain and get parsed structured result
    response = chain.invoke(messages)

    # Take the model response and build the association output.
    output = _build_output_payload(response, state)

    return output


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
            diagnosis=_format_codes(state.get("diagnosis", [])),
            format_instructions=parser.get_format_instructions(),
        ),
    ]

    # Since o1-preview and mini models do not support system messages, need to concatenate both into user message.
    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)

    return messages


def _build_output_payload(response: OutputModel, state: AssociationCodingStateGraph):

    # Ensuring both the response and state are in dictionary representation.
    response = _dump_to_dictionary(response)
    state = _dump_to_dictionary(state)

    # Build a lookup dictionary of all the ICD-10 codes and their content.
    diagnoses = _build_diagnosis_mapping(state.get("diagnosis", []))

    # Build a lookup dictionary of all CPT-E/M and procedure charges.
    em_associations, em_associated_diagnoses = _build_association_mapping(response.get("em", []), diagnoses)
    procedure_associations, procedure_associated_diagnoses = _build_association_mapping(response.get("procedure", []), diagnoses)

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

    # Combining all refined diagnoses from both em and procedure associations.
    associated_diagnoses = em_associated_diagnoses + procedure_associated_diagnoses

    # Remove duplicates while preserving order
    seen = set()
    deduplicated_diagnoses = []
    for item in associated_diagnoses:
        marker = tuple(sorted(item.items()))  # stable representation, no schema change
        if marker not in seen:
            seen.add(marker)
            deduplicated_diagnoses.append(item)

    # The associated diagnoses list without duplication.
    associated_diagnoses = deduplicated_diagnoses

    # Extracting the diganosis list directly from the state.
    # diagnosis = state.get("diagnosis", [])
    # Returning the CPT-E/M and procedure outputs with the linked diagnoses.
    return {"em": em, "procedure": procedure, "diagnosis": associated_diagnoses}


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


def _format_codes(code_type, code_field="code", desc_field="description"):
    codes = []

    # Iterating over all codes from a specific field (i.e., em, procedure, diagnosis).
    for item in code_type or []:

        # Extracting the dicti
        item_dict = _dump_to_dictionary(item)

        # Extracting the codes and description fields.
        code = item_dict.get(code_field, "")
        desc = item_dict.get(desc_field, "")

        # Formmatting to - D36.13 (Benign neoplasm...)
        if code and desc:
            codes.append(f"- {code} ({desc})")
        elif code:
            codes.append(f"- {code}")

    return "\n".join(codes)


def _dump_to_dictionary(x):

    # Helper: Obtain dict representation from pydantic or regular objects.
    return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)


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

    # List for storing the diagnoses which are associated.
    diagnoses_list = []

    # Dictionary storing the mapping between CPT-E/M code to their linked diagnoses.
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

            # Iterating over each linked diagnose for the current code.
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

                    # Updating the linked diagnoses list.
                    diagnoses_list.append(diagnoses.get(dx_code, {}))

            # Adding new code -> linked diagnosis mapping.
            associations[charge_code] = linked_diagnoses

    return associations, diagnoses_list
