import json
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
from cdrc.profees_coding.icd10cm_associate.models.utils import CONFIG

ICD_CODE_PATTERN = r"^[A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?$"


class InputState(dict):
    example_id: Optional[Union[int, str]]
    text: str
    patient_age: Optional[int]
    patient_sex: Optional[str]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    all_patient_problems: Optional[List[Problem]]
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    procedure_associations: Optional[List] = Field(default_factory=list, description="List of CPT/HCPCS procedure associations.")
    em_associations: Optional[List] = Field(default_factory=list, description="List of CPT-E/M code associations.")


class OutputState(dict):
    procedure_associations: Optional[List] = Field(default_factory=list, description="List of CPT/HCPCS procedure associations.")
    em_associations: Optional[List] = Field(default_factory=list, description="List of CPT-E/M code associations.")


class ZCodeEnhancerStateGraph(InputState, OutputState):
    pass


def build_z_code_enhancer_subgraph(args):

    workflow = StateGraph(ZCodeEnhancerStateGraph)
    workflow.add_node("z_code_enhancer_model_inference", model_inference_node)

    workflow.add_edge(START, "z_code_enhancer_model_inference")
    workflow.add_edge("z_code_enhancer_model_inference", END)

    return workflow.compile()


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
        if not re.match(ICD_CODE_PATTERN, code):
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
def model_inference_node(state: ZCodeEnhancerStateGraph, config: RunnableConfig):

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
    response = _dump_to_dictionary(response)

    # Preserve all non-Z associations from upstream, update only Z-code associations.
    state["em_associations"] = _merge_association_lists(state.get("em_associations", []), response.get("em_associations", []))
    state["procedure_associations"] = _merge_association_lists(state.get("procedure_associations", []), response.get("procedure_associations", []))

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
    prompts = config["configurable"][CONFIG.PROMPTS]["z_code_enhancer"]
    prompt_version = config["configurable"].get(CONFIG.ENHANCER_PROMPT_VERSION, "v1")
    if prompt_version not in prompts:
        raise ValueError(f"Unknown z_code_enhancer prompt version '{prompt_version}'. Available versions: {list(prompts.keys())}")
    system_prompt = SystemMessagePromptTemplate.from_template(prompts[prompt_version]["system"], template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(prompts[prompt_version]["user"], template_format="jinja2")

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
            em_associations=_format_associations(state.get("em_associations", [])),
            procedure_associations=_format_associations(state.get("procedure_associations", [])),
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


def _format_associations(association_list):
    normalized_associations = []

    # Iterating over all association payloads.
    for association in association_list or []:
        association_dictionary = _dump_to_dictionary(association)
        charge_code = _normalize_charge_code(association_dictionary.get("code", ""))
        linked_diagnoses = []

        for linked_dx in association_dictionary.get("linked_diagnoses", []):
            linked_dx_dictionary = _dump_to_dictionary(linked_dx)
            diagnosis_code = _normalize_code(linked_dx_dictionary.get("code", ""))
            if diagnosis_code:
                linked_diagnoses.append({"code": diagnosis_code, "justification": linked_dx_dictionary.get("justification", "")})

        normalized_associations.append({"code": charge_code, "linked_diagnoses": linked_diagnoses})

    return json.dumps(normalized_associations, indent=2)


def _merge_association_lists(upstream_associations, enhancer_associations):
    enhancer_lookup = {}

    # Mapping enhancer associations by charge code.
    for association in enhancer_associations or []:
        association_dictionary = _dump_to_dictionary(association)
        charge_code = _normalize_charge_code(association_dictionary.get("code", ""))
        if charge_code:
            enhancer_lookup[charge_code] = association_dictionary

    merged_associations = []

    # Keep upstream charge order and only apply enhancer changes for matching charges.
    for association in upstream_associations or []:
        association_dictionary = _dump_to_dictionary(association)
        charge_code = _normalize_charge_code(association_dictionary.get("code", ""))

        if charge_code and charge_code in enhancer_lookup:
            merged_associations.append(_merge_single_association(association_dictionary, enhancer_lookup[charge_code]))
        else:
            merged_associations.append(_dedupe_and_normalize_linked_diagnoses(association_dictionary))

    return merged_associations


def _merge_single_association(upstream_association, enhancer_association):
    merged = dict(upstream_association)
    merged["code"] = _normalize_charge_code(merged.get("code", ""))

    # Non-Z diagnoses are source-of-truth from upstream only.
    upstream_non_z = []
    for linked_dx in upstream_association.get("linked_diagnoses", []):
        linked_dx_dictionary = _dump_to_dictionary(linked_dx)
        diagnosis_code = _normalize_code(linked_dx_dictionary.get("code", ""))
        if diagnosis_code and not diagnosis_code.startswith("Z"):
            upstream_non_z.append(linked_dx_dictionary)

    # Z diagnoses are replaced by enhancer output.
    enhancer_z = []
    for linked_dx in enhancer_association.get("linked_diagnoses", []):
        linked_dx_dictionary = _dump_to_dictionary(linked_dx)
        diagnosis_code = _normalize_code(linked_dx_dictionary.get("code", ""))
        if diagnosis_code and diagnosis_code.startswith("Z"):
            enhancer_z.append({"code": diagnosis_code, "justification": linked_dx_dictionary.get("justification", "")})

    merged["linked_diagnoses"] = _dedupe_linked_diagnoses(upstream_non_z + enhancer_z)
    return merged


def _dedupe_and_normalize_linked_diagnoses(association):
    association_dictionary = dict(association)
    association_dictionary["code"] = _normalize_charge_code(association_dictionary.get("code", ""))
    association_dictionary["linked_diagnoses"] = _dedupe_linked_diagnoses(association_dictionary.get("linked_diagnoses", []))
    return association_dictionary


def _dedupe_linked_diagnoses(linked_diagnoses):
    normalized = []
    seen_codes = set()

    # Normalize diagnosis codes and enforce per-charge uniqueness.
    for linked_dx in linked_diagnoses or []:
        linked_dx_dictionary = _dump_to_dictionary(linked_dx)
        diagnosis_code = _normalize_code(linked_dx_dictionary.get("code", ""))

        if not diagnosis_code or diagnosis_code in seen_codes:
            continue
        if not re.match(ICD_CODE_PATTERN, diagnosis_code):
            continue

        normalized_linked_dx = dict(linked_dx_dictionary)
        normalized_linked_dx["code"] = diagnosis_code
        normalized.append(normalized_linked_dx)
        seen_codes.add(diagnosis_code)

    return normalized


def _normalize_code(code: str) -> str:
    return str(code or "").strip().upper()


def _normalize_charge_code(code: str) -> str:
    return str(code or "").strip().upper()


def _dump_to_dictionary(x):

    # Helper: Obtain dict representation from pydantic or regular objects.
    return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)
