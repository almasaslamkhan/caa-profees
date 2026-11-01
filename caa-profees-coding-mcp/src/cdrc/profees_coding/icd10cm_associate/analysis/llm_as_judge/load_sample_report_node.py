import ast
import enum
import pathlib
import typing

import pandas
import pydantic


class ErrorType(str, enum.Enum):

    # Error types are always from the model perspective.
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"


class AssociationKey(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    # Frozen keys let us compare gold and predicted association sets directly.
    charge_type: str
    charge_code: str
    diagnosis_code: str


class AssociationDetail(pydantic.BaseModel):

    # Store the key plus human-readable context used in prompts and CSV output.
    key: AssociationKey
    charge_description: str = ""
    diagnosis_description: str = ""
    justification: str = ""


class AssociationError(pydantic.BaseModel):

    # Each error represents one charge-to-diagnosis link that differs between gold and prediction.
    error_id: str
    error_type: ErrorType
    association: AssociationDetail
    guideline_context: str = ""


class EncounterRecord(pydantic.BaseModel):

    # One row from the sample report after extracting only fields needed for judging.
    source_report: str
    encounter_id: str
    patient_age: str
    patient_sex: str
    note_text: str
    em_codes: typing.Any
    procedure_codes: typing.Any
    gold_associations: typing.List[AssociationDetail]
    predicted_associations: typing.List[AssociationDetail]
    association_errors: typing.List[AssociationError]


def load_sample_report_node(state: typing.Dict[str, typing.Any]):

    # Load each raw experiment report and reduce it to encounters with association-level errors.
    analysis_data_paths = _coerce_analysis_data_paths(state["analysis_data"])
    encounters = []
    total_loaded_encounters = 0

    for analysis_data_path in analysis_data_paths:

        # Read the report independently so each extracted encounter can preserve its source file.
        df = pandas.read_csv(analysis_data_path)
        total_loaded_encounters += len(df)
        encounters.extend(_build_error_encounters(df, analysis_data_path.name))

    # Count all individual FP/FN links across the filtered encounter list.
    total_errors = sum(len(encounter.association_errors) for encounter in encounters)

    # Save extracted encounters into graph state for downstream nodes.
    state["encounters"] = encounters

    # Print the same summary counts used for quick non-LLM verification.
    print(f"Loaded {total_loaded_encounters} encounters.")
    print(f"Loaded {len(analysis_data_paths)} sample report(s).")
    print(f"Found {len(encounters)} encounters with association errors.")
    print(f"Found {total_errors} total association-level errors.")

    return state


def _coerce_analysis_data_paths(analysis_data: typing.Any) -> typing.List[pathlib.Path]:

    # The CLI passes a list, but this keeps programmatic single-path calls backward compatible.
    if isinstance(analysis_data, (str, pathlib.Path)):
        return [pathlib.Path(analysis_data)]

    # Convert each provided value to a Path so pandas can read all reports consistently.
    return [pathlib.Path(path) for path in analysis_data]


def _build_error_encounters(df: pandas.DataFrame, source_report: str) -> typing.List[EncounterRecord]:
    encounters = []

    # Build every encounter, but only keep rows with at least one association mismatch.
    for _, row in df.iterrows():
        encounter = _build_encounter(row, source_report)
        if encounter.association_errors:
            encounters.append(encounter)

    return encounters


def _build_encounter(row: pandas.Series, source_report: str) -> EncounterRecord:

    # The sample report stores structured columns as Python-literal strings, so parse them first.
    em_codes = _parse_literal_cell(row.get("input_em"))
    procedure_codes = _parse_literal_cell(row.get("input_procedure"))
    gold_em = _parse_literal_cell(row.get("gold_em"))
    gold_procedure = _parse_literal_cell(row.get("gold_procedure"))
    predicted_em = _parse_literal_cell(row.get("output_em"))
    predicted_procedure = _parse_literal_cell(row.get("output_procedure"))

    # Compare final linked diagnoses because these are the fields used by the evaluation metrics.
    # E/M and procedure links are normalized into one shared AssociationDetail representation.
    gold_associations = _extract_associations(gold_em, "E/M") + _extract_associations(gold_procedure, "Procedure")
    predicted_associations = _extract_associations(predicted_em, "E/M") + _extract_associations(
        predicted_procedure,
        "Procedure",
    )

    # Finding each association-level false positive or false negative.
    association_errors = _find_association_errors(gold_associations, predicted_associations)

    # Store the original encounter context plus the extracted association comparison.
    return EncounterRecord(
        source_report=source_report,
        encounter_id=_stringify(row.get("input_example_id")),
        patient_age=_stringify(row.get("input_patient_age")),
        patient_sex=_stringify(row.get("input_patient_sex")),
        note_text=_stringify(row.get("input_text")),
        em_codes=em_codes,
        procedure_codes=procedure_codes,
        gold_associations=gold_associations,
        predicted_associations=predicted_associations,
        association_errors=association_errors,
    )


def _parse_literal_cell(value: typing.Any) -> typing.Any:

    # Blank CSV cells are read as NaN; treat them as empty structured values.
    if value is None:
        return []
    if isinstance(value, (list, dict)):
        return value
    if pandas.isna(value):
        return []
    if not isinstance(value, str):
        return []

    # Strip before parsing so whitespace-only cells become empty values.
    stripped = value.strip()
    if not stripped:
        return []

    try:
        # ast.literal_eval safely handles the report's single-quoted list/dict strings.
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        # Some fields are plain text; leave those intact instead of failing extraction.
        return stripped


def _extract_associations(charges: typing.Any, charge_type: str) -> typing.List[AssociationDetail]:
    associations = []
    seen_associations: typing.Set[AssociationKey] = set()

    # Non-list payloads cannot contain charge dictionaries, so there is nothing to extract.
    if not isinstance(charges, list):
        return associations

    for charge in charges:

        # Skip malformed entries defensively so one bad charge does not break the report.
        if not isinstance(charge, dict):
            continue

        charge_code = _normalize_code(charge.get("code"))
        if not charge_code:
            continue

        # Preserve descriptions/justifications for human-readable audit output and judge context.
        charge_description = _stringify(charge.get("description"))

        # Final report payloads use linked_diagnosis; intermediate snapshots may use linked_diagnoses.
        linked_diagnoses = charge.get("linked_diagnosis")
        if linked_diagnoses is None:
            linked_diagnoses = charge.get("linked_diagnoses")

        # Normalize a single linked diagnosis dict to the same list format as multiple diagnoses.
        if isinstance(linked_diagnoses, dict):
            linked_diagnoses = [linked_diagnoses]
        if not isinstance(linked_diagnoses, list):
            linked_diagnoses = []

        for diagnosis in linked_diagnoses:

            # Linked diagnosis entries should be dictionaries containing at minimum a code.
            if not isinstance(diagnosis, dict):
                continue

            diagnosis_code = _normalize_code(diagnosis.get("code"))
            if not diagnosis_code:
                continue

            # Dedupe by charge type, charge code, and diagnosis code.
            association_key = AssociationKey(
                charge_type=charge_type,
                charge_code=charge_code,
                diagnosis_code=diagnosis_code,
            )
            if association_key in seen_associations:
                continue
            seen_associations.add(association_key)

            # Keep the original description and source justification with the normalized key.
            association = AssociationDetail(
                key=association_key,
                charge_description=charge_description,
                diagnosis_description=_stringify(diagnosis.get("description")),
                justification=_stringify(diagnosis.get("justification")),
            )
            associations.append(association)

    return associations


def _find_association_errors(
    gold_associations: typing.List[AssociationDetail], predicted_associations: typing.List[AssociationDetail]
) -> typing.List[AssociationError]:

    # Convert to dictionaries so set arithmetic can compare only the normalized association keys.
    gold_by_key = {association.key: association for association in gold_associations}
    predicted_by_key = {association.key: association for association in predicted_associations}

    # Predicted-only links are false positives: the model included an association absent from gold.
    false_positive_keys = sorted(predicted_by_key.keys() - gold_by_key.keys(), key=_sort_association_key)

    # Gold-only links are false negatives: the model omitted an association present in gold.
    false_negative_keys = sorted(gold_by_key.keys() - predicted_by_key.keys(), key=_sort_association_key)

    errors = []
    error_index = 1

    # Number errors locally within each encounter to keep prompts and output easy to scan.
    for key in false_positive_keys:
        errors.append(
            AssociationError(
                error_id=f"ERR-{error_index:03d}",
                error_type=ErrorType.FALSE_POSITIVE,
                association=predicted_by_key[key],
            )
        )
        error_index += 1

    # False negatives use the gold association detail because the prediction omitted the link.
    for key in false_negative_keys:
        errors.append(
            AssociationError(
                error_id=f"ERR-{error_index:03d}",
                error_type=ErrorType.FALSE_NEGATIVE,
                association=gold_by_key[key],
            )
        )
        error_index += 1

    return errors


def _sort_association_key(key: AssociationKey):

    # Stable sorting makes prompts and CSV output deterministic between runs.
    return key.charge_type, key.charge_code, key.diagnosis_code


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
