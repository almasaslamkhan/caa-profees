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

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent
ICD10_DATA_DIR = ANALYSIS_DIR.parent.parent / "data" / "2026"
ICD10_DESCRIPTION_FILE = ICD10_DATA_DIR / "icd10_descriptions_2026.json"


class ErrorClassification(str, enum.Enum):

    # Judge labels describe whether the model-side FP/FN is defensible.
    JUSTIFIED = "JUSTIFIED"
    AMBIGUOUS = "AMBIGUOUS"
    WRONG = "WRONG"
    MISANNOTATED = "MISANNOTATED"


class AssociationJudgment(pydantic.BaseModel):

    # This is the structured response expected from the LLM-as-judge node.
    error_id: str = pydantic.Field(..., description="Identifier of the association error being judged.")
    classification: ErrorClassification = pydantic.Field(
        ..., description="Whether the model-side error was justified, ambiguous, wrong, or caused by a likely gold misannotation."
    )
    justification: str = pydantic.Field(..., description="Concise documentation-based justification for the classification.")


class JudgeOutput(pydantic.BaseModel):

    # The parser expects one judgment object for every error_id shown in the prompt.
    output: typing.List[AssociationJudgment] = pydantic.Field(..., description="Judgments for every association error in the prompt.")


def llm_as_judge_node(state: typing.Dict[str, typing.Any]):

    # Load ICD-10-CM descriptions once so output association lists are compact but readable.
    icd10_descriptions = _load_icd10_descriptions()

    # Save descriptions and judge outputs for both CSV export and suggestion prompting.
    state["icd10_descriptions"] = icd10_descriptions
    state["analysis_results"] = _llm_analysis(_build_model(state), state, state["encounters"], icd10_descriptions)

    return state


def _load_icd10_descriptions() -> typing.Dict[str, str]:

    # The description file is a simple mapping from ICD-10-CM code to display description.
    with open(ICD10_DESCRIPTION_FILE, "r", encoding="utf-8") as file:
        descriptions = json.load(file)

    # Store both dotted and non-dotted forms so lookups work across source formats.
    description_lookup = {}
    for code, description in descriptions.items():
        normalized_code = _normalize_code(code)
        normalized_description = _stringify(description)
        description_lookup[normalized_code] = normalized_description
        description_lookup[normalized_code.replace(".", "")] = normalized_description

    return description_lookup


def _build_model(state: typing.Dict[str, typing.Any]):

    # Model kwargs can be overridden from the CLI as JSON, but otherwise use the main script defaults.
    model_kwargs = json.loads(state["model_kwargs"]) if state.get("model_kwargs") else state["default_model_kwargs"]

    # Instantiate through the shared CDRC LLM registry so model names match the rest of the project.
    return get_llm()[state["model_name"]](**model_kwargs)


def _llm_analysis(model, state: typing.Dict[str, typing.Any], encounters: typing.List[typing.Any], icd10_descriptions: typing.Dict[str, str]):

    # The retrying parser can ask the same model to repair malformed JSON before surfacing a failure.
    parser = PydanticOutputParserWithRetries(pydantic_object=JudgeOutput, correction_model=model, num_retries=3)

    # Compose model invocation with structured parsing as one reusable chain.
    chain = model | parser

    # Load the LLM-as-judge prompt templates from analysis/llm_as_judge/prompt.yaml.
    system_prompt, user_prompt = _load_prompt_templates(state)

    analysis_results = []

    for encounter in tqdm.tqdm(encounters, total=len(encounters), desc="Progression"):

        # Render one prompt per encounter so the judge can evaluate all FP/FN errors together.
        prompt_variables = _build_encounter_prompt_variables(encounter, icd10_descriptions, parser)
        rendered_system_prompt = system_prompt.render(prompt_variables)
        rendered_user_prompt = user_prompt.render(prompt_variables)

        # Store one parsed judge result per encounter in the same order as encounters.
        analysis_results.append(_run_llm_with_retries(chain, rendered_system_prompt, rendered_user_prompt))

    return analysis_results


def _load_prompt_templates(state: typing.Dict[str, typing.Any]):

    # The prompt file and key path are centralized in main.py and passed through graph state.
    prompt_section = _load_yaml_section(state["prompt_file"], state["judge_prompt_key_path"])

    # Compile both prompt halves as Jinja templates for encounter-level rendering.
    system_prompt = jinja2.Template(prompt_section["system"])
    user_prompt = jinja2.Template(prompt_section["user"])

    return system_prompt, user_prompt


def _load_yaml_section(path: pathlib.Path, key_path: typing.Tuple[str, ...]) -> typing.Dict[str, typing.Any]:

    # Load the whole YAML file once, then walk down to the requested prompt section.
    with open(path, "r", encoding="utf-8") as file:
        section = yaml.safe_load(file)

    # Walk nested prompt keys like ("llm-as-judge", "v1") without duplicating YAML plumbing.
    for key in key_path:
        section = section[key]

    return section


def _build_encounter_prompt_variables(encounter: typing.Any, icd10_descriptions: typing.Dict[str, str], parser) -> typing.Dict[str, typing.Any]:

    # This shared context is consumed by both the judge prompt and the suggestion prompt.
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
        "association_errors": _format_association_errors_for_prompt(_get_value(encounter, "association_errors", [])),
        "guideline_context": _format_guideline_context_for_prompt(_get_value(encounter, "association_errors", [])),
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

    # Keep error IDs in the prompt so the judge can return one classification per listed error.
    for error in errors:
        formatted_errors.append(
            f"- {_get_value(error, 'error_id')} | {_enum_value(_get_value(error, 'error_type'))} | {_format_association(_get_value(error, 'association'))}"
        )

    return "\n".join(formatted_errors)


def _format_guideline_context_for_prompt(errors: typing.List[typing.Any]) -> str:

    # Join all per-error guideline blocks into the prompt section.
    return "\n\n".join(_get_value(error, "guideline_context") for error in errors if _get_value(error, "guideline_context"))


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
