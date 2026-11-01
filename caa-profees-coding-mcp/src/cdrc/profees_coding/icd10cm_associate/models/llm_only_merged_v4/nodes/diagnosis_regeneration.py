import copy
from typing import List, Optional

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from cdrc.common.icd10_cm_tools.agent_tools import get_description as tool_get_description
from cdrc.common.icd10_cm_tools.agent_tools import is_billable as tool_is_billable
from cdrc.common.icd10_cm_tools.src.tools.validator import cm_code_validator as icd10cm_validator
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Procedure
from cdrc.profees_coding.icd10cm_associate.models.utils import CONFIG


class IntermediateState(dict):
    procedure_associations: Optional[List] = Field(default_factory=list, description="List of CPT/HCPCS procedure associations.")
    em_associations: Optional[List] = Field(default_factory=list, description="List of CPT-E/M code associations.")
    not_valid_associations: Optional[List] = Field(
        default_factory=list, description="List of dropped (charge_code, diagnosis_code) pairs due to invalid ICD-10-CM codes."
    )
    not_billable_associations: Optional[List] = Field(
        default_factory=list, description="List of dropped (charge_code, diagnosis_code) pairs due to non-billable ICD-10-CM codes."
    )


class OutputState(dict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


class PostProcessingStateGraph(IntermediateState, OutputState):
    pass


# --------------------------------------------------------------------------------
# Valid Diagnosis Regenerator Node
# --------------------------------------------------------------------------------


class ValidReplacement(BaseModel):
    charge_code: str = Field(..., description="CPT-E/M or CPT/HCPCS charge code receiving a replacement.")
    problematic_code: str = Field(..., description="Invalid/non-billable ICD-10-CM code that was dropped.")
    replacement_code: str = Field(..., description="Replacement ICD-10-CM code that is valid and billable.")
    justification: str = Field(..., description="Reason for why an association was made.")


class ValidOutput(BaseModel):
    replacements: List[ValidReplacement] = Field(
        default_factory=list,
        description="List of replacement operations keyed by (charge_code, problematic_code).",
    )


@track_tokens("model_usage")
def diagnosis_regenerator(state: PostProcessingStateGraph, config: RunnableConfig):

    # Skip this node when diagnosis regenerator stage is disabled.
    if not config["configurable"].get(CONFIG.DIAGNOSIS_REGENERATOR, True):
        if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
            state["intermediate_regenerated_associations"] = _build_association_snapshot(
                state.get("em_associations", []),
                state.get("procedure_associations", []),
            )
            state["intermediate_regeneration_recommendations"] = _build_recommendation_snapshot([], {})
        return state

    # Skip this node when there are no invalid/non-billable diagnosis drops to process.
    if len(state.get("not_valid_associations") or []) == 0 and len(state.get("not_billable_associations") or []) == 0:
        if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
            state["intermediate_regenerated_associations"] = _build_association_snapshot(
                state.get("em_associations", []),
                state.get("procedure_associations", []),
            )
            state["intermediate_regeneration_recommendations"] = _build_recommendation_snapshot([], {})
        return state

    # List 1: (charge_code, problematic_diagnosis_code) pairs.
    problematic_associations = _build_invalid_charge_map(state)

    # Map 2: problematic diagnosis code -> valid/billable recommendations.
    recommended_associations = _retrieve_valid_charge_map(problematic_associations)

    # Persist rule-based recommendation candidates for report export when debugging is enabled.
    if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
        state["intermediate_regeneration_recommendations"] = _build_recommendation_snapshot(
            problematic_associations,
            recommended_associations,
        )

    # Choose and instantiate model based on runtime configuration.
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    # Output parser with retry/correction logic.
    parser = PydanticOutputParserWithRetries(pydantic_object=ValidOutput, correction_model=model, num_retries=3)

    # Building the prompt used for the LLM call.
    messages = _build_prompt(state, config, parser, problematic_associations, recommended_associations)

    # Compose chain of model and parsing logic.
    chain = model | parser

    # Call LLM chain and get parsed structured result.
    response = chain.invoke(messages)

    # Ensuring both the response and state are in dictionary representation.
    response = _dump_to_dictionary(response)

    # Apply replacement operations to the intermediate association state.
    _replace_associations(state, problematic_associations, response["replacements"])

    # Persist a deep-copied snapshot for report export when debugging is enabled.
    if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
        state["intermediate_regenerated_associations"] = _build_association_snapshot(
            state.get("em_associations", []),
            state.get("procedure_associations", []),
        )

    return state


# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------


def _dump_to_dictionary(x):
    # Helper: Obtain dict representation from pydantic or regular objects.
    return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)


def _build_association_snapshot(em_associations, procedure_associations):

    # Snapshot as plain deep-copied dictionaries so later node mutations don't back-propagate.
    return {
        "em_associations": [copy.deepcopy(_dump_to_dictionary(association)) for association in (em_associations or [])],
        "procedure_associations": [copy.deepcopy(_dump_to_dictionary(association)) for association in (procedure_associations or [])],
    }


def _build_recommendation_snapshot(problematic_associations, recommended_associations):

    # Snapshot rule-generated candidates so downstream report exports can inspect recommendation context.
    return {
        "problematic_associations": [copy.deepcopy(item) for item in (problematic_associations or [])],
        "recommended_associations": {
            code: [copy.deepcopy(candidate) for candidate in candidates] for code, candidates in (recommended_associations or {}).items()
        },
    }


def _build_invalid_charge_map(state: PostProcessingStateGraph) -> list[tuple[str, str, str]]:

    # Flat list of normalized (charge_code, problematic_diagnosis_code) pairs.
    problematic_code_charge_pairs = []

    # These are the two tracking lists created by validate_code_node.
    for tracking_field in ["not_valid_associations", "not_billable_associations"]:

        # Iterate through each dropped tuple record.
        for dropped_item in state.get(tracking_field) or []:

            # Normalizing the charge code and diagnosis code.
            charge_code = str(dropped_item[0] or "").strip().upper()
            problematic_code = str(dropped_item[1] or "").strip().upper()

            # Fetch the problematic code description once and store it with the dropped pair.
            try:
                problematic_description = tool_get_description(code=problematic_code) or ""
            except Exception:
                problematic_description = ""

            # Append each dropped pair in observed order.
            problematic_code_charge_pairs.append((charge_code, problematic_code, problematic_description))

    return problematic_code_charge_pairs


def _retrieve_valid_charge_map(problematic_code_charge_pairs):

    # Pull all ICD-10-CM codes and keep only valid billable dotted codes.
    all_codes = icd10cm_validator.get_all_codes(with_dots=True) or []
    billable_code_pairs = []
    seen_codes = set()

    # Iterating over all the codes.
    for raw_code in all_codes:

        # Normalize each candidate code once.
        candidate_code = str(raw_code or "").strip().upper()

        # Keep each code only once, and it must be billable.
        if candidate_code in seen_codes or not tool_is_billable(code=candidate_code):
            continue

        # Track the candidate code for lookup.
        seen_codes.add(candidate_code)
        billable_code_pairs.append((candidate_code, candidate_code.replace(".", "")))

    # Resolve one problematic diagnosis code to valid/billable candidates.
    def _resolve_candidates(problematic_code):

        # Normalize the incoming diagnosis code and stop early for empty input.
        normalized_code = str(problematic_code or "").strip().upper()
        if not normalized_code:
            return []

        # Normalize to no-dot format used for prefix matching.
        normalized_no_dot = normalized_code.replace(".", "")

        # ICD-10-CM code body is max 7 chars (excluding the dot).
        normalized_no_dot = normalized_no_dot[:7]
        if len(normalized_no_dot) < 3:
            return []

        # Back off prefix length one char at a time until we reach 3 chars.
        for prefix_length in range(len(normalized_no_dot), 2, -1):

            # Build the prefix used for this pass.
            current_prefix = normalized_no_dot[:prefix_length]

            # Keep only candidates that share the current prefix.
            prefix_matches = []
            for candidate_code, candidate_no_dot in billable_code_pairs:
                if candidate_no_dot.startswith(current_prefix):
                    prefix_matches.append(candidate_code)

            # First non-empty match list wins.
            if prefix_matches:
                # Convert candidate codes to (code, description) tuples.
                recommendation_tuples = []
                for candidate_code in prefix_matches:
                    try:
                        description = tool_get_description(code=candidate_code) or ""
                    except Exception:
                        description = ""
                    recommendation_tuples.append((candidate_code, description))
                return recommendation_tuples

        # No prefix from full length down to 3 produced any candidates.
        return []

    # Build recommendations once per problematic diagnosis code.
    unique_problematic_codes = {
        problematic_item[1]
        for problematic_item in problematic_code_charge_pairs
        if isinstance(problematic_item, (list, tuple)) and len(problematic_item) > 1
    }

    # Dictionary storing one recommendation list per unique problematic diagnosis code.
    invalid_code_recommendation_map = {}
    for problematic_code in unique_problematic_codes:
        invalid_code_recommendation_map[problematic_code] = _resolve_candidates(problematic_code)

    return invalid_code_recommendation_map


def _build_prompt(state, config, parser, problematic_associations, recommended_associations):

    # Accessing the full prompt bank from runtime config.
    prompts = config["configurable"][CONFIG.PROMPTS]

    # Resolving diagnosis regenerator prompt key/version from runtime config.
    regenerator_key = config["configurable"].get(CONFIG.REGENERATOR_KEY, "diagnosis_regenerator")
    regenerator_version = config["configurable"].get(CONFIG.REGENERATOR_VERSION, "v1")

    # Ensure prompt key exists in the prompt bank.
    if regenerator_key not in prompts:
        raise KeyError(f"Missing regenerator prompt key '{regenerator_key}' in prompts config. Available keys: {sorted(prompts.keys())}")

    # Ensure prompt version exists under the configured key.
    prompt_bank = prompts[regenerator_key]
    if regenerator_version not in prompt_bank:
        raise KeyError(
            f"Missing regenerator prompt version '{regenerator_version}' under key '{regenerator_key}'. "
            f"Available versions: {sorted(prompt_bank.keys())}"
        )

    # Extracting system/user prompt templates for the configured key/version.
    system_prompt = SystemMessagePromptTemplate.from_template(prompt_bank[regenerator_version]["system"], template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(prompt_bank[regenerator_version]["user"], template_format="jinja2")

    # Building the E/M section as plain bullet rows:
    # - CODE (DESCRIPTION)
    em_lines = []
    for item in state.get("em", []) or []:
        item_dictionary = _dump_to_dictionary(item) or {}
        code = item_dictionary.get("code", "")
        description = item_dictionary.get("description", "")
        if code:
            em_lines.append(f"- {code} ({description})" if description else f"- {code}")

    # Building the CPT/HCPCS procedure section using the same bullet format.
    procedure_lines = []
    for item in state.get("procedure", []) or []:
        item_dictionary = _dump_to_dictionary(item) or {}
        code = item_dictionary.get("code", "")
        description = item_dictionary.get("description", "")
        if code:
            procedure_lines.append(f"- {code} ({description})" if description else f"- {code}")

    # Problematic rows already come in grouped format:
    # (charge_code, problematic_diagnosis_code, problematic_diagnosis_description).
    problematic_rows = list(problematic_associations or [])

    # Expanding the recommendation map into grouped rows:
    # (charge_code, recommended_diagnosis_code, recommended_diagnosis_description).
    recommended_rows = [
        (charge_code, recommended_code, recommended_description)
        for charge_code, problematic_code, _ in (problematic_associations or [])
        for recommended_code, recommended_description in recommended_associations.get(problematic_code, [])
    ]

    # Constructing the final message list with all user-template variables populated.
    messages = [
        system_prompt.format(),
        user_prompt.format(
            note=state.get("text", ""),
            patient_age=state.get("patient_age", ""),
            patient_sex=state.get("patient_sex", ""),
            em="\n".join(em_lines),
            procedure="\n".join(procedure_lines),
            problematic_associations=_format_charge_rows(problematic_rows),
            recommended_associations=_format_charge_rows(recommended_rows),
            format_instructions=parser.get_format_instructions(),
        ),
    ]

    # Since o1-preview and mini models do not support system messages,
    # concatenate system+user content into a single user message.
    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)

    return messages


def _format_charge_rows(rows):

    # Grouping rows by charge code so each section renders as:
    grouped_rows = {}
    for charge_code, code, description in rows:
        if charge_code and code:
            grouped_rows.setdefault(charge_code, []).append((code, description))

    # Rendering grouped rows into a prompt-friendly multiline string.
    lines = []
    for charge_code, code_items in grouped_rows.items():

        # Inserting a blank line between charge-code groups for readability.
        if lines:
            lines.append("")

        # Adding the charge-code header line.
        lines.append(charge_code)

        # Appending one bullet row per diagnosis code under this charge.
        lines.extend([f"- {code} ({description})" if description else f"- {code}" for code, description in code_items])

    # Returning the final block that is injected into the prompt template.
    return "\n".join(lines)


def _replace_associations(state, problematic_associations, replacements):

    # Convert replacement operations into dictionary format.
    replacement_list = []
    for replacement in replacements:
        replacement_list.append(_dump_to_dictionary(replacement))

    # Convert association payloads into dictionary format so we can update in-place.
    for association_field in ["em_associations", "procedure_associations"]:
        updated_associations = []
        for association in state.get(association_field) or []:
            updated_associations.append(_dump_to_dictionary(association))
        state[association_field] = updated_associations

    # Apply replacements in problematic-pair order to preserve expected append order.
    for charge_code, problematic_code, _ in problematic_associations:

        # Find the first replacement operation matching this (charge_code, problematic_code) pair.
        replacement_payload = None
        for replacement in replacement_list:
            if replacement["charge_code"] == charge_code and replacement["problematic_code"] == problematic_code:
                replacement_payload = replacement
                break

        # If the model did not return a replacement for this pair, skip it.
        if replacement_payload is None:
            continue

        # Append replacement code to matching charge, avoiding duplicates.
        replacement_code = replacement_payload["replacement_code"]
        for association_field in ["em_associations", "procedure_associations"]:
            for association in state.get(association_field) or []:

                # Only process the association row for the matching charge code.
                if association["code"] != charge_code:
                    continue

                # Read current linked diagnoses for this charge.
                linked_diagnoses = association["linked_diagnoses"]

                # Skip append when replacement diagnosis already exists for this charge.
                if replacement_code not in [linked_dx["code"] for linked_dx in linked_diagnoses]:
                    linked_diagnoses.append({"code": replacement_code, "justification": replacement_payload["justification"]})

                # One row per charge code per association field, so stop scanning this field.
                break
