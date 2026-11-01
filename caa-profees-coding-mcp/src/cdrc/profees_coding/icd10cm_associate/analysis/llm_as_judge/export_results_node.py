import typing

import pandas


def export_results_node(state: typing.Dict[str, typing.Any]):

    # Format the judged encounters into the final spreadsheet-friendly CSV.
    # The export function handles all markdown-style grouping and bullet formatting.
    _reformat_dataframe(
        state["encounters"],
        state["analysis_results"],
        state["suggestion_results"],
        state["icd10_descriptions"],
    ).to_csv(
        state["output_file"],
        index=False,
        encoding="utf-8",
    )
    print(f"Saved judge output to {state['output_file']}")

    return state


def _reformat_dataframe(
    encounters: typing.List[typing.Any],
    analysis_results: typing.List[typing.Dict[str, typing.Any]],
    suggestion_results: typing.List[typing.Dict[str, typing.Any]],
    icd10_descriptions: typing.Dict[str, str],
) -> pandas.DataFrame:
    rows = []

    # Each output row corresponds to one encounter with at least one association error.
    for encounter, analysis, suggestions in zip(encounters, analysis_results, suggestion_results):

        # List-valued fields are stored as markdown bullets so the CSV remains readable in spreadsheets.
        rows.append(
            {
                "Sample Report": _get_value(encounter, "source_report"),
                "Encounter ID": _get_value(encounter, "encounter_id"),
                "Patient Age": _get_value(encounter, "patient_age"),
                "Patient Sex": _get_value(encounter, "patient_sex"),
                "Note/Text": _get_value(encounter, "note_text"),
                "Gold Associations": _format_associations(_get_value(encounter, "gold_associations", []), icd10_descriptions),
                "Predicted Associations": _format_associations(
                    _get_value(encounter, "predicted_associations", []), icd10_descriptions, include_justifications=True
                ),
                "Association Errors": _format_association_errors(_get_value(encounter, "association_errors", [])),
                "Judge Decisions": _format_judge_decisions(_get_value(encounter, "association_errors", []), analysis),
                "Suggestion": _format_prompt_improvement_suggestions(_get_value(encounter, "association_errors", []), suggestions),
            }
        )

    # Return a dataframe so the graph node can handle the final write in one obvious place.
    return pandas.DataFrame(rows)


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


def _format_association_errors(errors: typing.List[typing.Any]) -> str:
    formatted_errors = []

    # Group model-side errors under each charge code for easier spreadsheet review.
    for charge_code, charge_errors in _group_errors_by_charge(errors).items():
        formatted_errors.append(charge_code)

        for error in charge_errors:
            diagnosis_code = _get_value(_get_value(_get_value(error, "association"), "key"), "diagnosis_code")

            # Shorten false-positive/false-negative labels for compact CSV display.
            error_type = "FP" if _enum_value(_get_value(error, "error_type")) == "FALSE_POSITIVE" else "FN"
            justification = _get_value(_get_value(error, "association"), "justification")

            # False positives come from predicted links; false negatives come from gold links.
            justification_label = "model justification" if _enum_value(_get_value(error, "error_type")) == "FALSE_POSITIVE" else "gold justification"

            line = f"- {diagnosis_code} ({error_type})"
            if justification:
                line += f" | {justification_label}: {justification}"
            formatted_errors.append(line)

    return "\n".join(formatted_errors)


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


def _format_prompt_improvement_suggestions(
    errors: typing.List[typing.Any],
    suggestions: typing.Dict[str, typing.Any],
) -> str:

    # Index suggestions by error_id so each recommendation lines up with the original error.
    suggestions_by_error_id = {suggestion.get("error_id", ""): suggestion for suggestion in suggestions.get("output", [])}
    formatted_suggestions = []

    # If suggestion generation failed for this encounter, leave the cell blank.
    if not suggestions_by_error_id:
        return ""

    # Group suggestions under charge code to mirror Judge Decisions and Association Errors.
    for charge_code, charge_errors in _group_errors_by_charge(errors).items():
        formatted_suggestions.append(charge_code)

        for error in charge_errors:

            # Skip an individual error if the suggestion output did not include that error_id.
            suggestion = suggestions_by_error_id.get(_get_value(error, "error_id"))
            if not suggestion:
                continue

            diagnosis_code = _get_value(_get_value(_get_value(error, "association"), "key"), "diagnosis_code")
            action = _stringify(suggestion.get("action"))
            target_location = _stringify(suggestion.get("target_location"))
            suggestion_text = _stringify(suggestion.get("suggestion"))
            reason = _stringify(suggestion.get("reason"))

            # Keep the action first so spreadsheet reviewers can scan prompt-change categories quickly.
            line = f"- {action} | {diagnosis_code}"
            if target_location:
                line += f" | location: {target_location}"
            if suggestion_text:
                line += f" | {suggestion_text}"
            if reason:
                line += f" | reason: {reason}"
            formatted_suggestions.append(line)

    return "\n".join(formatted_suggestions)


def _group_errors_by_charge(errors: typing.List[typing.Any]) -> typing.Dict[str, typing.List[typing.Any]]:
    errors_by_charge = {}

    # Preserve encounter order while grouping all errors for the same charge together.
    for error in errors:
        charge_code = _get_value(_get_value(_get_value(error, "association"), "key"), "charge_code")
        errors_by_charge.setdefault(charge_code, []).append(error)

    return errors_by_charge


def _get_value(item: typing.Any, field: str, default: typing.Any = ""):

    # Node scripts are standalone, so consume cross-node objects structurally.
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _enum_value(value: typing.Any) -> str:

    # Enums from another node should display as their raw value in CSV output.
    return _stringify(getattr(value, "value", value))


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
