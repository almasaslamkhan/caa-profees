import enum
import json
import pathlib
import typing

import jinja2
import pandas
import pydantic
import tqdm
import yaml

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm


class SuggestionAction(str, enum.Enum):

    # Suggestion labels describe what kind of change should be made to the original prompt.
    ADD = "ADD"
    REMOVE = "REMOVE"
    REVISE = "REVISE"
    NO_CHANGE = "NO_CHANGE"


class AssociationSuggestion(pydantic.BaseModel):

    # This is the structured response expected from the prompt-improvement node.
    error_id: str = pydantic.Field(..., description="Identifier of the eligible association error being improved.")
    action: SuggestionAction = pydantic.Field(..., description="Prompt improvement action: add, remove, revise, or no change.")
    target_location: str = pydantic.Field(..., description="Exact section or subsection of the original prompt where the change should be made.")
    suggestion: str = pydantic.Field(..., description="Concise, generalizable prompt improvement suggestion for this association error pattern.")
    reason: str = pydantic.Field(..., description="Reason this suggestion would help address the error.")


class SuggestionOutput(pydantic.BaseModel):

    # The suggestion node should return exactly one suggestion for each eligible error_id.
    output: typing.List[AssociationSuggestion] = pydantic.Field(
        ..., description="Prompt improvement suggestions for every eligible association error."
    )


def llm_suggestion_node(state: typing.Dict[str, typing.Any]):

    # Run a second LLM pass that turns judge findings into original-prompt improvement suggestions.
    state["suggestion_results"] = _suggestion_analysis(
        _build_model(state),
        state,
        state["encounters"],
        state["analysis_results"],
        state["icd10_descriptions"],
    )

    return state


def _build_model(state: typing.Dict[str, typing.Any]):

    # Model kwargs can be overridden from the CLI as JSON, but otherwise use the main script defaults.
    model_kwargs = json.loads(state["model_kwargs"]) if state.get("model_kwargs") else state["default_model_kwargs"]

    # Instantiate through the shared CDRC LLM registry so model names match the rest of the project.
    return get_llm()[state["model_name"]](**model_kwargs)


def _suggestion_analysis(
    model,
    state: typing.Dict[str, typing.Any],
    encounters: typing.List[typing.Any],
    analysis_results: typing.List[typing.Dict[str, typing.Any]],
    icd10_descriptions: typing.Dict[str, str],
):

    # The suggestion response has its own schema because it targets prompt changes, not error labels.
    parser = PydanticOutputParserWithRetries(pydantic_object=SuggestionOutput, correction_model=model, num_retries=3)

    # Compose model invocation with structured parsing as one reusable chain.
    chain = model | parser

    # Load the suggestion prompt and the original outpatient prompt being critiqued.
    system_prompt, user_prompt = _load_prompt_templates(state)
    original_prompt = _load_original_prompt(state)

    suggestion_results = []

    for encounter, analysis in tqdm.tqdm(zip(encounters, analysis_results), total=len(encounters), desc="Suggestion"):
        eligible_errors = _filter_errors_for_suggestions(
            _get_value(encounter, "association_errors", []),
            analysis,
            state.get("suggestion_error_types", "all"),
        )

        # Only AMBIGUOUS and WRONG judge decisions should receive prompt-improvement suggestions.
        if not eligible_errors:
            suggestion_results.append({"output": []})
            continue

        # Start with the same encounter context used by the judge, then add suggestion-only inputs.
        prompt_variables = _build_encounter_prompt_variables(encounter, icd10_descriptions, parser, eligible_errors)

        # Add the original prompt and judge findings so suggestions are grounded and targeted.
        prompt_variables.update(
            {
                "original_system_prompt": original_prompt["system"],
                "original_user_prompt": original_prompt["user"],
                "judge_decisions": _format_judge_decisions(eligible_errors, analysis),
            }
        )
        rendered_system_prompt = system_prompt.render(prompt_variables)
        rendered_user_prompt = user_prompt.render(prompt_variables)

        # Store one parsed suggestion result per encounter in the same order as encounters.
        suggestion_results.append(_run_llm_with_retries(chain, rendered_system_prompt, rendered_user_prompt))

    return suggestion_results


def _filter_errors_for_suggestions(
    errors: typing.List[typing.Any],
    analysis: typing.Dict[str, typing.Any],
    suggestion_error_types: str = "all",
) -> typing.List[typing.Any]:

    # Normalize the experiment setting once so callers can pass "FP", "fp", etc.
    normalized_suggestion_error_types = _stringify(suggestion_error_types).lower()
    if normalized_suggestion_error_types not in {"all", "fp", "fn"}:
        raise ValueError("suggestion_error_types must be one of: all, fp, fn")

    # Suggestions should only be generated for errors the judge marked AMBIGUOUS or WRONG.
    eligible_classifications = {"AMBIGUOUS", "WRONG"}

    # Index judge decisions so each association error can be filtered by its own classification.
    judgments_by_error_id = {judgment.get("error_id", ""): judgment for judgment in analysis.get("output", [])}
    eligible_errors = []

    for error in errors:
        judgment = judgments_by_error_id.get(_get_value(error, "error_id"), {})
        classification = _stringify(judgment.get("classification")).upper()

        # Skip defensible model behavior and likely gold-label issues.
        if classification not in eligible_classifications:
            continue

        # Apply the optional FP/FN experiment scope after the judge-classification filter.
        error_type = _enum_value(_get_value(error, "error_type"))
        if normalized_suggestion_error_types == "fp" and error_type != "FALSE_POSITIVE":
            continue
        if normalized_suggestion_error_types == "fn" and error_type != "FALSE_NEGATIVE":
            continue

        eligible_errors.append(error)

    return eligible_errors


def _load_prompt_templates(state: typing.Dict[str, typing.Any]):

    # The prompt file and key path are centralized in main.py and passed through graph state.
    prompt_section = _load_yaml_section(state["prompt_file"], state["suggestion_prompt_key_path"])

    # Compile both prompt halves as Jinja templates for encounter-level rendering.
    system_prompt = jinja2.Template(prompt_section["system"])
    user_prompt = jinja2.Template(prompt_section["user"])

    return system_prompt, user_prompt


def _load_original_prompt(state: typing.Dict[str, typing.Any]) -> typing.Dict[str, str]:

    # The original prompt file and key path are centralized in main.py as the improvement target.
    prompt_section = _load_yaml_section(state["original_prompt_file"], state["original_prompt_key_path"])

    return {"system": prompt_section["system"], "user": prompt_section["user"]}


def _load_yaml_section(path: pathlib.Path, key_path: typing.Tuple[str, ...]) -> typing.Dict[str, typing.Any]:

    # Load the whole YAML file once, then walk down to the requested prompt section.
    with open(path, "r", encoding="utf-8") as file:
        section = yaml.safe_load(file)

    # Walk nested prompt keys like ("llm-suggestion", "v1") without duplicating YAML plumbing.
    for key in key_path:
        section = section[key]

    return section


def _build_encounter_prompt_variables(
    encounter: typing.Any,
    icd10_descriptions: typing.Dict[str, str],
    parser,
    eligible_errors: typing.List[typing.Any],
) -> typing.Dict[str, typing.Any]:

    # This shared context mirrors the judge prompt so suggestions see the same encounter evidence.
    return {
        "patient_age": _get_value(encounter, "patient_age"),
        "patient_sex": _get_value(encounter, "patient_sex"),
        "note_text": _get_value(encounter, "note_text"),
        "em_codes": _format_codes(_get_value(encounter, "em_codes", [])),
        "procedure_codes": _format_codes(_get_value(encounter, "procedure_codes", [])),
        "gold_associations": _format_associations(_get_value(encounter, "gold_associations", []), icd10_descriptions),
        "predicted_associations": _format_associations(
            _get_value(encounter, "predicted_associations", []), icd10_descriptions, include_justifications=True
        ),
        "association_errors": _format_association_errors_for_prompt(eligible_errors),
        "guideline_context": _format_guideline_context_for_prompt(eligible_errors),
        "format_instructions": parser.get_format_instructions(),
    }


def _format_codes(codes: typing.Any) -> str:

    # Non-list values are uncommon, but stringify them rather than failing output formatting.
    if not isinstance(codes, list):
        return _stringify(codes)

    formatted_codes = []
    for item in codes:

        # Only dictionary items have the code/description/system structure we expect.
        if not isinstance(item, dict):
            continue

        # Keep code display compact; deeper code metadata stays out of the judge-facing CSV.
        code = _normalize_code(item.get("code"))
        description = _stringify(item.get("description"))
        system = _stringify(item.get("system"))

        label = code
        if description:
            label += f" ({description})"
        if system:
            label += f" [{system}]"
        if label:
            formatted_codes.append(f"- {label}")

    # One bullet per charge makes this render cleanly in spreadsheet cells.
    return "\n".join(formatted_codes)


def _format_associations(
    associations: typing.List[typing.Any],
    icd10_descriptions: typing.Dict[str, str],
    include_justifications: bool = False,
) -> str:
    associations_by_charge = {}
    formatted_associations = []

    # Group diagnosis links under each charge code for easier spreadsheet review.
    for association in associations:
        key = _get_value(association, "key")
        charge_key = (_get_value(key, "charge_type"), _get_value(key, "charge_code"))
        associations_by_charge.setdefault(charge_key, []).append(association)

    # Render each charge as a heading followed by its linked diagnosis bullets.
    for charge_key, charge_associations in associations_by_charge.items():
        charge_code = charge_key[1]
        formatted_associations.append(charge_code)

        for association in charge_associations:
            key = _get_value(association, "key")
            diagnosis_code = _get_value(key, "diagnosis_code")

            # Prefer the central ICD-10 description file, but keep source text as a fallback.
            diagnosis_description = icd10_descriptions.get(diagnosis_code, _get_value(association, "diagnosis_description"))

            if diagnosis_description:
                line = f"- {diagnosis_code} ({diagnosis_description})"
            else:
                line = f"- {diagnosis_code}"

            if include_justifications and _get_value(association, "justification"):
                # Predicted associations include model-generated justifications for judge context.
                line += f" | model justification: {_get_value(association, 'justification')}"

            formatted_associations.append(line)

    return "\n".join(formatted_associations)


def _format_association_errors_for_prompt(errors: typing.List[typing.Any]) -> str:
    formatted_errors = []

    # Keep error IDs in the prompt so the suggestion node can return one suggestion per listed error.
    for error in errors:
        formatted_errors.append(
            f"- {_get_value(error, 'error_id')} | {_enum_value(_get_value(error, 'error_type'))} | {_format_association(_get_value(error, 'association'))}"
        )

    return "\n".join(formatted_errors)


def _format_guideline_context_for_prompt(errors: typing.List[typing.Any]) -> str:

    # Join all per-error guideline blocks into the prompt section.
    return "\n\n".join(_get_value(error, "guideline_context") for error in errors if _get_value(error, "guideline_context"))


def _format_judge_decisions(
    errors: typing.List[typing.Any],
    analysis: typing.Dict[str, typing.Any],
) -> str:

    # Index judge outputs by error_id so we can align them back to grouped charge display.
    judgments_by_error_id = {judgment.get("error_id", ""): judgment for judgment in analysis.get("output", [])}
    formatted_decisions = []

    # Group judge decisions under each charge code to mirror the association error display.
    for charge_code, charge_errors in _group_errors_by_charge(errors).items():
        formatted_decisions.append(charge_code)

        for error in charge_errors:
            diagnosis_code = _get_value(_get_value(_get_value(error, "association"), "key"), "diagnosis_code")

            # Missing judgments are allowed when the LLM failed or returned an incomplete response.
            judgment = judgments_by_error_id.get(_get_value(error, "error_id"), {})
            decision = _stringify(judgment.get("classification"))
            justification = _stringify(judgment.get("justification"))

            line = f"- {decision} | {diagnosis_code}"
            if justification:
                line += f" | {justification}"
            formatted_decisions.append(line)

    return "\n".join(formatted_decisions)


def _group_errors_by_charge(errors: typing.List[typing.Any]) -> typing.Dict[str, typing.List[typing.Any]]:
    errors_by_charge = {}

    # Preserve encounter order while grouping all errors for the same charge together.
    for error in errors:
        charge_code = _get_value(_get_value(_get_value(error, "association"), "key"), "charge_code")
        errors_by_charge.setdefault(charge_code, []).append(error)

    return errors_by_charge


def _format_association(association: typing.Any) -> str:
    key = _get_value(association, "key")
    charge_type = _get_value(key, "charge_type")
    charge_code = _get_value(key, "charge_code")
    diagnosis_code = _get_value(key, "diagnosis_code")

    # This compact representation is used inside prompts and guideline context blocks.
    output = f"{charge_type} {charge_code} -> {diagnosis_code}"

    # Add optional context only when present, avoiding noisy empty labels.
    details = []
    if _get_value(association, "charge_description"):
        details.append(f"service: {_get_value(association, 'charge_description')}")
    if _get_value(association, "diagnosis_description"):
        details.append(f"diagnosis: {_get_value(association, 'diagnosis_description')}")
    if _get_value(association, "justification"):
        details.append(f"source justification: {_get_value(association, 'justification')}")

    if details:
        output += " | " + " | ".join(details)

    return output


def _run_llm_with_retries(chain, rendered_system_prompt: str, rendered_user_prompt: str):

    # LangChain chat models expect role/content dictionaries.
    messages = [{"role": "system", "content": rendered_system_prompt}, {"role": "user", "content": rendered_user_prompt}]

    # Retry the whole chain because failures may be transient model or parser errors.
    for _ in range(5):
        try:
            # Pipe model output through the parser so successful results are already structured.
            output = chain.invoke(messages)
            return {"output": output.model_dump(mode="json")["output"]}
        except Exception:  # noqa: BLE001 - analysis script should preserve failed rows instead of stopping.
            continue

    # Failed encounters still get CSV rows, making gaps visible instead of silently dropping data.
    return {"output": []}


def _get_value(item: typing.Any, field: str, default: typing.Any = ""):

    # Node scripts are standalone, so consume cross-node objects structurally.
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _enum_value(value: typing.Any) -> str:

    # Enums from another node should display as their raw value in prompts.
    return _stringify(getattr(value, "value", value))


def _normalize_code(value: typing.Any) -> str:

    # Normalize codes once so comparisons are case-insensitive and whitespace-insensitive.
    return _stringify(value).upper()


def _stringify(value: typing.Any) -> str:

    # Treat None and pandas NaN as empty strings for cleaner prompts and CSV cells.
    if value is None:
        return ""
    try:
        if pandas.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    # Fall back to normal string conversion for all other scalar values.
    return str(value).strip()
