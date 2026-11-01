import copy
import json
import random
import re
from typing import List, Optional, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.icd10cm_associate.data.schema import EMItem, Order, Problem, Procedure
from cdrc.profees_coding.icd10cm_associate.models.utils import CONFIG, PROMPTS

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


class TargetEnhancerStateGraph(InputState, OutputState):
    pass


class OutputDiagnosis(BaseModel):
    code: str = Field(..., description="ICD-10-CM code.")
    justification: str = Field(..., description="Reason for why an association was made.")

    @model_validator(mode="before")
    def normalize_and_validate_code(cls, values):

        # Ensure model validators only process dictionary-like payloads.
        if not isinstance(values, dict):
            return values

        # Normalize the diagnosis code before parser validation.
        code = _normalize_code(values.get("code", ""))
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
def target_enhancer_node(state: TargetEnhancerStateGraph, config: RunnableConfig):

    # Resolve which enhancer variant should run for this evaluation.
    enhancer_type = _get_enhancer_type(config)

    # Early exit when the enhancer stage is intentionally disabled.
    if enhancer_type == "none":
        if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
            state["intermediate_enhanced_associations"] = _build_association_snapshot(
                state.get("em_associations", []),
                state.get("procedure_associations", []),
            )
        return state

    # Choose and instantiate model based on runtime configuration.
    model_kwargs = _get_model_kwargs(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    # Output parser with retry/correction logic.
    parser = PydanticOutputParserWithRetries(pydantic_object=OutputModel, correction_model=model, num_retries=3)

    # Build the prompt used for the enhancer LLM call.
    messages = _build_prompt(state, config, parser, enhancer_type)

    # Compose chain of model and parsing logic.
    chain = model | parser

    # Call LLM chain and get parsed structured result.
    response = _dump_to_dictionary(chain.invoke(messages))
    target_prefix = _resolve_target_prefix(enhancer_type)

    # Preserve all non-target diagnoses from upstream, replacing only the target family.
    state["em_associations"] = _merge_association_lists(
        state.get("em_associations", []),
        response.get("em_associations", []),
        target_prefix=target_prefix,
    )
    state["procedure_associations"] = _merge_association_lists(
        state.get("procedure_associations", []),
        response.get("procedure_associations", []),
        target_prefix=target_prefix,
    )

    # Persist a deep-copied snapshot for report export when enabled.
    if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
        state["intermediate_enhanced_associations"] = _build_association_snapshot(
            state.get("em_associations", []),
            state.get("procedure_associations", []),
        )

    return state


def _get_enhancer_type(config: RunnableConfig) -> str:

    # Normalizing the configured enhancer selector and validating supported values.
    enhancer_type = str(config["configurable"].get(CONFIG.ENHANCER_TYPE, "none") or "none").strip().lower()
    if enhancer_type not in {"none", "z", "r"}:
        raise ValueError(f"Unknown enhancer_type '{enhancer_type}'. Allowed values: ['none', 'z', 'r']")
    return enhancer_type


def _resolve_target_prefix(enhancer_type: str) -> str:

    # Map the configured enhancer to the ICD-10 family prefix it is allowed to modify.
    if enhancer_type == "z":
        return "Z"
    if enhancer_type == "r":
        return "R"
    raise ValueError(f"Enhancer type '{enhancer_type}' does not map to a target diagnosis family.")


def _get_model_kwargs(config: RunnableConfig) -> dict:

    # Extract model kwargs, use default seed if not set.
    model_kwargs = dict(config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42}))
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))

    # For reproducibility, randomize the seed when repeating model calls.
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)

    return model_kwargs


def _build_prompt(state: TargetEnhancerStateGraph, config: RunnableConfig, parser, enhancer_type: str):

    # Resolve the prompt bank for the selected enhancer family.
    prompt_bank_key = PROMPTS.Z_CODE_ENHANCER if enhancer_type == "z" else PROMPTS.R_CODE_ENHANCER
    prompt_bank = _get_prompt_bank(config, prompt_bank_key, enhancer_type)
    prompt_version = config["configurable"].get(CONFIG.ENHANCER_PROMPT_VERSION, "v1")
    if prompt_version not in prompt_bank:
        raise KeyError(
            f"Missing prompt version '{prompt_version}' under prompt bank '{prompt_bank_key}'. Available versions: {sorted(prompt_bank.keys())}"
        )

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
            em_associations=_format_associations(state.get("em_associations", [])),
            procedure_associations=_format_associations(state.get("procedure_associations", [])),
            format_instructions=parser.get_format_instructions(),
        ),
    ]

    # Since o1-preview and mini models do not support system messages, concatenate both into one user message.
    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)

    return messages


def _get_prompt_bank(config: RunnableConfig, prompt_bank_key: str, enhancer_type: str):

    # Resolve the prompt bank from V5 prompts, tolerating the legacy V3 R-bank typo during migration.
    prompts = config["configurable"].get(CONFIG.PROMPTS, {})
    prompt_bank = prompts.get(prompt_bank_key)
    if prompt_bank is None and enhancer_type == "r":
        prompt_bank = prompts.get("r_code_enahncer")
    if prompt_bank is None:
        raise KeyError(f"Missing prompt bank '{prompt_bank_key}' for enhancer_type '{enhancer_type}'.")
    return prompt_bank


def _format_codes(code_type, code_field="code", desc_field="description"):
    codes = []

    # Iterating over all codes from a specific field.
    for item in code_type or []:
        item_dictionary = _dump_to_dictionary(item) or {}
        code = _normalize_charge_code(item_dictionary.get(code_field, ""))
        desc = str(item_dictionary.get(desc_field, "") or "").strip()

        # Formatting to "- 99214 (Office visit)" where possible.
        if code and desc:
            codes.append(f"- {code} ({desc})")
        elif code:
            codes.append(f"- {code}")

    return "\n".join(codes)


def _format_associations(association_list):
    normalized_associations = []

    # Iterating over all association payloads.
    for association in association_list or []:
        association_dictionary = _dump_to_dictionary(association) or {}
        charge_code = _normalize_charge_code(association_dictionary.get("code", ""))
        linked_diagnoses = []

        for linked_dx in association_dictionary.get("linked_diagnoses", []):
            linked_dx_dictionary = _dump_to_dictionary(linked_dx) or {}
            diagnosis_code = _normalize_code(linked_dx_dictionary.get("code", ""))
            if diagnosis_code:
                linked_diagnoses.append({"code": diagnosis_code, "justification": linked_dx_dictionary.get("justification", "")})

        normalized_associations.append({"code": charge_code, "linked_diagnoses": linked_diagnoses})

    return json.dumps(normalized_associations, indent=2)


def _merge_association_lists(upstream_associations, enhancer_associations, target_prefix: str):
    enhancer_lookup = {}

    # Mapping enhancer associations by charge code.
    for association in enhancer_associations or []:
        association_dictionary = _dump_to_dictionary(association) or {}
        charge_code = _normalize_charge_code(association_dictionary.get("code", ""))
        if charge_code:
            enhancer_lookup[charge_code] = association_dictionary

    merged_associations = []

    # Keep upstream charge order and only apply enhancer changes for matching charges.
    for association in upstream_associations or []:
        association_dictionary = _dump_to_dictionary(association) or {}
        charge_code = _normalize_charge_code(association_dictionary.get("code", ""))

        if charge_code and charge_code in enhancer_lookup:
            merged_associations.append(_merge_single_association(association_dictionary, enhancer_lookup[charge_code], target_prefix))
        else:
            merged_associations.append(_dedupe_and_normalize_linked_diagnoses(association_dictionary))

    return merged_associations


def _merge_single_association(upstream_association, enhancer_association, target_prefix: str):
    merged = {"code": _normalize_charge_code(upstream_association.get("code", ""))}

    # Non-target diagnoses are source-of-truth from upstream only.
    merged_linked_diagnoses = []
    for linked_dx in upstream_association.get("linked_diagnoses", []):
        linked_dx_dictionary = _dump_to_dictionary(linked_dx) or {}
        diagnosis_code = _normalize_code(linked_dx_dictionary.get("code", ""))
        if diagnosis_code and not diagnosis_code.startswith(target_prefix):
            linked_dx_dictionary["code"] = diagnosis_code
            merged_linked_diagnoses.append(linked_dx_dictionary)

    # Target-family diagnoses are replaced by enhancer output.
    for linked_dx in enhancer_association.get("linked_diagnoses", []):
        linked_dx_dictionary = _dump_to_dictionary(linked_dx) or {}
        diagnosis_code = _normalize_code(linked_dx_dictionary.get("code", ""))
        if diagnosis_code and diagnosis_code.startswith(target_prefix):
            linked_dx_dictionary["code"] = diagnosis_code
            merged_linked_diagnoses.append(linked_dx_dictionary)

    merged["linked_diagnoses"] = merged_linked_diagnoses
    return _dedupe_and_normalize_linked_diagnoses(merged)


def _dedupe_and_normalize_linked_diagnoses(association):
    association_dictionary = dict(_dump_to_dictionary(association) or {})
    association_dictionary["code"] = _normalize_charge_code(association_dictionary.get("code", ""))

    deduped_linked_diagnoses = []
    seen_codes = set()
    for linked_dx in association_dictionary.get("linked_diagnoses", []):
        linked_dx_dictionary = _dump_to_dictionary(linked_dx) or {}
        diagnosis_code = _normalize_code(linked_dx_dictionary.get("code", ""))
        if not diagnosis_code or diagnosis_code in seen_codes:
            continue

        linked_dx_dictionary["code"] = diagnosis_code
        deduped_linked_diagnoses.append(linked_dx_dictionary)
        seen_codes.add(diagnosis_code)

    association_dictionary["linked_diagnoses"] = deduped_linked_diagnoses
    return association_dictionary


def _normalize_code(value) -> str:

    # Normalizing ICD-10-CM diagnosis codes to the uppercase canonical representation.
    return str(value or "").strip().upper()


def _normalize_charge_code(value) -> str:

    # Normalizing CPT/HCPCS/E/M charge codes to a stable uppercase representation.
    return str(value or "").strip().upper()


def _dump_to_dictionary(x):

    # Helper: Obtain dict representation from pydantic or regular objects.
    return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)


def _build_association_snapshot(em_associations, procedure_associations):

    # Snapshot as plain deep-copied dictionaries so later node mutations don't back-propagate.
    return {
        "em_associations": [copy.deepcopy(_dump_to_dictionary(association)) for association in (em_associations or [])],
        "procedure_associations": [copy.deepcopy(_dump_to_dictionary(association)) for association in (procedure_associations or [])],
    }
