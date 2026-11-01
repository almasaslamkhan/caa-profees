import json
import pathlib
import typing

import jinja2
import pydantic
import yaml

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm

ELIGIBLE_ACTIONS = {"ADD", "REMOVE", "REVISE"}


class GeneratedPromptOutput(pydantic.BaseModel):

    # The generator returns the complete replacement prompt content.
    system: str = pydantic.Field(..., description="Updated system prompt text.")
    user: str = pydantic.Field(..., description="Updated user prompt text.")


def prompt_generator_node(state: typing.Dict[str, typing.Any]):

    # Load the original prompt that will be rewritten or copied into the generated YAML.
    original_prompt = _load_original_prompt(state)

    # Gather only actionable ADD/REMOVE/REVISE suggestions from all encounters.
    prompt_suggestions = _collect_prompt_suggestions(state.get("suggestion_results", []))

    # If there are no actionable suggestions, keep the original prompt unchanged.
    if prompt_suggestions:
        generated_prompt = _generate_prompt_with_llm(
            _build_model(state),
            state,
            original_prompt,
            prompt_suggestions,
        )
    else:
        generated_prompt = original_prompt

    # Write the generated prompt into the requested outpatient -> v4 YAML shape.
    _write_generated_prompt(state["generated_prompt_file"], generated_prompt, state["generated_prompt_key_path"])

    # Save a copy in state for tests or downstream inspection.
    state["generated_prompt"] = generated_prompt

    print(f"Saved generated prompt to {state['generated_prompt_file']}")

    return state


def _collect_prompt_suggestions(suggestion_results: typing.List[typing.Dict[str, typing.Any]]) -> typing.List[typing.Dict[str, str]]:
    prompt_suggestions = []

    # Flatten all encounter-level suggestion outputs into one generator input list.
    for suggestion_result in suggestion_results:
        for suggestion in suggestion_result.get("output", []):
            action = _enum_value(suggestion.get("action")).upper()

            # NO_CHANGE and unknown actions should not influence the generated prompt.
            if action not in ELIGIBLE_ACTIONS:
                continue

            prompt_suggestions.append(
                {
                    "action": action,
                    "target_location": _stringify(suggestion.get("target_location")),
                    "suggestion": _stringify(suggestion.get("suggestion")),
                    "reason": _stringify(suggestion.get("reason")),
                }
            )

    return prompt_suggestions


def _generate_prompt_with_llm(
    model,
    state: typing.Dict[str, typing.Any],
    original_prompt: typing.Dict[str, str],
    prompt_suggestions: typing.List[typing.Dict[str, str]],
) -> typing.Dict[str, str]:

    # The retrying parser can ask the same model to repair malformed JSON before surfacing a failure.
    parser = PydanticOutputParserWithRetries(pydantic_object=GeneratedPromptOutput, correction_model=model, num_retries=3)

    # Compose model invocation with structured parsing as one reusable chain.
    chain = model | parser

    # Load the prompt-generator instruction templates.
    system_prompt, user_prompt = _load_prompt_templates(state)

    # Render one prompt-generation request using all actionable suggestions.
    prompt_variables = {
        "original_system_prompt": original_prompt["system"],
        "original_user_prompt": original_prompt["user"],
        "prompt_suggestions": _format_prompt_suggestions(prompt_suggestions),
        "format_instructions": parser.get_format_instructions(),
    }
    rendered_system_prompt = system_prompt.render(prompt_variables)
    rendered_user_prompt = user_prompt.render(prompt_variables)

    return _run_llm_with_retries(chain, rendered_system_prompt, rendered_user_prompt, original_prompt)


def _build_model(state: typing.Dict[str, typing.Any]):

    # Model kwargs can be overridden from the CLI as JSON, but otherwise use the main script defaults.
    model_kwargs = json.loads(state["model_kwargs"]) if state.get("model_kwargs") else state["default_model_kwargs"]

    # Instantiate through the shared CDRC LLM registry so model names match the rest of the project.
    return get_llm()[state["model_name"]](**model_kwargs)


def _load_prompt_templates(state: typing.Dict[str, typing.Any]):

    # The prompt file and key path are centralized in main.py and passed through graph state.
    prompt_section = _load_yaml_section(state["prompt_file"], state["prompt_generator_key_path"])

    # Compile both prompt halves as Jinja templates for prompt-generation rendering.
    system_prompt = jinja2.Template(prompt_section["system"])
    user_prompt = jinja2.Template(prompt_section["user"])

    return system_prompt, user_prompt


def _load_original_prompt(state: typing.Dict[str, typing.Any]) -> typing.Dict[str, str]:

    # The original prompt file and key path are centralized in main.py as the generator source.
    prompt_section = _load_yaml_section(state["original_prompt_file"], state["original_prompt_key_path"])

    return {"system": prompt_section["system"], "user": prompt_section["user"]}


def _load_yaml_section(path: pathlib.Path, key_path: typing.Tuple[str, ...]) -> typing.Dict[str, typing.Any]:

    # Load the whole YAML file once, then walk down to the requested prompt section.
    with open(path, "r", encoding="utf-8") as file:
        section = yaml.safe_load(file)

    # Walk nested prompt keys like ("prompt-generator", "v1") without duplicating YAML plumbing.
    for key in key_path:
        section = section[key]

    return section


def _format_prompt_suggestions(prompt_suggestions: typing.List[typing.Dict[str, str]]) -> str:
    suggestions_by_location = {}
    formatted_suggestions = []

    # Group suggestions by target prompt section so the generator can edit section-by-section.
    for suggestion in prompt_suggestions:
        target_location = suggestion["target_location"] or "Unspecified"
        suggestions_by_location.setdefault(target_location, []).append(suggestion)

    for target_location, suggestions in suggestions_by_location.items():
        formatted_suggestions.append(target_location)

        for suggestion in suggestions:
            line = f"- {suggestion['action']} | {suggestion['suggestion']}"
            if suggestion["reason"]:
                line += f" | reason: {suggestion['reason']}"
            formatted_suggestions.append(line)

    return "\n".join(formatted_suggestions)


def _run_llm_with_retries(
    chain,
    rendered_system_prompt: str,
    rendered_user_prompt: str,
    fallback_prompt: typing.Dict[str, str],
) -> typing.Dict[str, str]:

    # LangChain chat models expect role/content dictionaries.
    messages = [{"role": "system", "content": rendered_system_prompt}, {"role": "user", "content": rendered_user_prompt}]

    # Retry the whole chain because failures may be transient model or parser errors.
    for _ in range(5):
        try:
            # Pipe model output through the parser so successful results are already structured.
            output = chain.invoke(messages)
            return output.model_dump(mode="json")
        except Exception:  # noqa: BLE001 - analysis script should preserve workflow completion instead of stopping.
            continue

    # If generation fails, still write a valid prompt YAML by falling back to the original prompt.
    return fallback_prompt


def _write_generated_prompt(
    path: pathlib.Path,
    generated_prompt: typing.Dict[str, str],
    generated_prompt_key_path: typing.Tuple[str, ...],
):

    # Build the requested outpatient -> v4 YAML structure.
    payload = {
        generated_prompt_key_path[0]: {
            generated_prompt_key_path[1]: {
                "system": generated_prompt["system"],
                "user": generated_prompt["user"],
            }
        }
    }

    # Ensure the output directory exists before writing the generated YAML.
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(payload, file, Dumper=_BlockStringDumper, sort_keys=False, allow_unicode=True)


class _BlockStringDumper(yaml.SafeDumper):
    pass


def _represent_string(dumper, data):

    # Multi-line prompts are easier to review when YAML writes them as block strings.
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BlockStringDumper.add_representer(str, _represent_string)


def _stringify(value: typing.Any) -> str:

    # Treat None as an empty string for cleaner prompt-generation inputs.
    if value is None:
        return ""

    # Fall back to normal string conversion for all other scalar values.
    return str(value).strip()


def _enum_value(value: typing.Any) -> str:

    # Enum-like values should be converted to their raw value before filtering actions.
    return _stringify(getattr(value, "value", value))
