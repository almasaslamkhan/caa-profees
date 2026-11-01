import json
import pathlib
import typing
import xml.etree.ElementTree

import pandas
import pydantic

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent
ICD10_DATA_DIR = ANALYSIS_DIR.parent.parent / "data" / "2026"
CPT_ICD10_SUPPORTED_FILE = ICD10_DATA_DIR / "cpt_icd10_supported_2026.json"
CPT_ICD10_NONCOVERED_FILE = ICD10_DATA_DIR / "cpt_icd10_noncovered_2026.json"
HCPCS_ICD10_SUPPORTED_FILE = ICD10_DATA_DIR / "hcpcs_icd10_supported_2026.json"
HCPCS_ICD10_NONCOVERED_FILE = ICD10_DATA_DIR / "hcpcs_icd10_noncovered_2026.json"
ICD10_TABULAR_FILE = ICD10_DATA_DIR / "icd10cm_tabular_2026.xml"


class Icd10TabularGuidance(pydantic.BaseModel):

    # Compact representation of the ICD-10-CM tabular context relevant to one diagnosis code.
    code: str = ""
    description: str = ""
    chapter: str = ""
    section: str = ""
    parent_categories: typing.List[str] = pydantic.Field(default_factory=list)
    notes: typing.List[str] = pydantic.Field(default_factory=list)


def guideline_context_node(state: typing.Dict[str, typing.Any]):

    # Add guideline context before prompting so every judged error has code-specific audit support.
    guideline_resources = _load_guideline_resources()

    # Attach context directly to each association error so later prompt formatting stays simple.
    for encounter in state["encounters"]:

        # Procedure rows may say whether the charge is CPT or HCPCS; use that before fallback inference.
        procedure_system_lookup = _build_procedure_system_lookup(_get_value(encounter, "procedure_codes", []))

        for error in _get_value(encounter, "association_errors", []):
            _set_value(error, "guideline_context", _build_guideline_context(error, procedure_system_lookup, guideline_resources))

    return state


def _load_guideline_resources() -> typing.Dict[str, typing.Any]:

    # These resources are intentionally limited to guideline context used by the judge prompt.
    return {
        "cpt_supported": _load_charge_icd10_mapping(CPT_ICD10_SUPPORTED_FILE),
        "cpt_noncovered": _load_charge_icd10_mapping(CPT_ICD10_NONCOVERED_FILE),
        "hcpcs_supported": _load_charge_icd10_mapping(HCPCS_ICD10_SUPPORTED_FILE),
        "hcpcs_noncovered": _load_charge_icd10_mapping(HCPCS_ICD10_NONCOVERED_FILE),
        "icd10_tabular": _load_icd10_tabular_guidance(),
    }


def _load_charge_icd10_mapping(path: pathlib.Path) -> typing.Dict[str, typing.Set[str]]:

    # Coverage files map each CPT/HCPCS code to a list of ICD-10-CM codes.
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    # Convert lists to sets because every later use is a membership check.
    mapping = {}
    for charge_code, diagnosis_codes in payload.items():
        normalized_charge_code = _normalize_code(charge_code)
        mapping[normalized_charge_code] = {_normalize_code(diagnosis_code) for diagnosis_code in diagnosis_codes}

    return mapping


def _load_icd10_tabular_guidance() -> typing.Dict[str, Icd10TabularGuidance]:

    # Parse the ICD-10-CM tabular XML once into a code lookup used by all encounters.
    root = xml.etree.ElementTree.parse(ICD10_TABULAR_FILE).getroot()
    tabular_lookup = {}

    def walk(element, chapter="", section="", parent_categories=None, parent_notes=None):

        # Avoid mutable defaults while carrying context down the recursive traversal.
        parent_categories = parent_categories or []
        parent_notes = parent_notes or []

        # Carry chapter, section, parent categories, and parent notes down the tree.
        if element.tag == "chapter":
            chapter = _stringify(element.findtext("desc"))
        elif element.tag == "section":
            section = _stringify(element.findtext("desc"))
        elif element.tag == "diag":

            # Each diag node may be a billable code or a parent category.
            code = _normalize_code(element.findtext("name"))
            description = _stringify(element.findtext("desc"))
            direct_notes = _format_direct_tabular_notes(element)

            if code:
                # Save both direct notes and inherited parent notes, capped to keep prompts compact.
                guidance = Icd10TabularGuidance(
                    code=code,
                    description=description,
                    chapter=chapter,
                    section=section,
                    parent_categories=parent_categories,
                    notes=[*direct_notes, *parent_notes][:10],
                )
                tabular_lookup[code] = guidance
                tabular_lookup[code.replace(".", "")] = guidance

            # Descendant codes need parent categories and parent notes as context.
            if code and description:
                parent_categories = [*parent_categories, f"{code}: {description}"]
            if code:
                parent_notes = [*parent_notes, *[f"{code} {note}" for note in direct_notes]]

        # Recurse only into nodes that affect chapter, section, or diagnosis context.
        for child in element:
            if child.tag in {"chapter", "section", "diag"}:
                walk(child, chapter, section, parent_categories, parent_notes)

    walk(root)

    return tabular_lookup


def _format_direct_tabular_notes(element) -> typing.List[str]:
    formatted_notes = []

    # Direct notes include items like excludes1, excludes2, codeFirst, and useAdditionalCode.
    for child in element:
        if child.tag in {"name", "desc", "diag"}:
            continue

        for note in child.iter("note"):
            note_text = _stringify(note.text)
            if note_text:
                # Prefix the note with its XML tag so the model knows what rule type it came from.
                formatted_notes.append(f"{child.tag}: {note_text}")

    return formatted_notes


def _build_procedure_system_lookup(procedure_codes: typing.Any) -> typing.Dict[str, str]:
    lookup = {}

    # Procedure input rows may include a system field that identifies CPT vs HCPCS directly.
    for procedure in procedure_codes if isinstance(procedure_codes, list) else []:
        if not isinstance(procedure, dict):
            continue

        # Normalize both code and system so downstream checks are case-insensitive.
        code = _normalize_code(procedure.get("code"))
        system = _normalize_code(procedure.get("system"))
        if code and system:
            lookup[code] = system

    return lookup


def _build_guideline_context(
    error: typing.Any,
    procedure_system_lookup: typing.Dict[str, str],
    guideline_resources: typing.Dict[str, typing.Any],
) -> str:
    association = _get_value(error, "association")
    key = _get_value(association, "key")
    charge_code = _get_value(key, "charge_code")
    diagnosis_code = _get_value(key, "diagnosis_code")

    # Determine whether to use CPT or HCPCS coverage maps for this charge.
    charge_system = _infer_charge_system(key, procedure_system_lookup)
    supported_mapping = guideline_resources[f"{charge_system.lower()}_supported"]
    noncovered_mapping = guideline_resources[f"{charge_system.lower()}_noncovered"]

    # Pull tabular context for the diagnosis code, allowing dotted/non-dotted variants.
    tabular_guidance = _get_tabular_guidance(guideline_resources["icd10_tabular"], diagnosis_code)

    # Keep one compact context block per error_id so the judge can map guidance back to outputs.
    return "\n".join(
        [
            f"{_get_value(error, 'error_id')} | {_enum_value(_get_value(error, 'error_type'))} | {_format_association(association)}",
            "",
            "Coverage guidance:",
            *_format_coverage_guidance(charge_system, charge_code, diagnosis_code, supported_mapping, noncovered_mapping),
            "",
            "ICD-10-CM tabular guidance:",
            *_format_tabular_guidance(tabular_guidance, diagnosis_code),
            "",
            "Audit relevance:",
            f"- Use this context when deciding whether {diagnosis_code} supports medical necessity for {charge_code}.",
        ]
    )


def _infer_charge_system(key: typing.Any, procedure_system_lookup: typing.Dict[str, str]) -> str:
    charge_code = _get_value(key, "charge_code")

    # Prefer the source procedure metadata when available.
    if charge_code in procedure_system_lookup:
        return "HCPCS" if procedure_system_lookup[charge_code] == "HCPCS" else "CPT"

    # Letter-prefixed procedure codes are generally HCPCS; numeric codes are treated as CPT.
    if charge_code[:1].isalpha():
        return "HCPCS"
    return "CPT"


def _get_tabular_guidance(tabular_lookup: typing.Dict[str, Icd10TabularGuidance], diagnosis_code: str):

    # Try dotted and non-dotted ICD-10-CM forms because source files are not always consistent.
    for candidate in _build_icd10_code_candidates(diagnosis_code):
        if candidate in tabular_lookup:
            return tabular_lookup[candidate]
    return None


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


def _format_coverage_guidance(
    charge_system: str,
    charge_code: str,
    diagnosis_code: str,
    supported_mapping: typing.Dict[str, typing.Set[str]],
    noncovered_mapping: typing.Dict[str, typing.Set[str]],
) -> typing.List[str]:

    # Pull the explicit supported and noncovered lists for this charge, if any exist.
    supported_codes = supported_mapping.get(charge_code, set())
    noncovered_codes = noncovered_mapping.get(charge_code, set())
    guidance = []

    # Supported mappings can help the judge understand medical-necessity coverage context.
    if supported_codes:
        guidance.append(f"- {charge_system} {charge_code} has an explicit supported ICD-10-CM list.")
        if _code_in_guideline_set(diagnosis_code, supported_codes):
            guidance.append(f"- {diagnosis_code} is included in the supported ICD-10-CM list for {charge_system} {charge_code}.")
        else:
            guidance.append(f"- {diagnosis_code} is not included in the supported ICD-10-CM list for {charge_system} {charge_code}.")
    else:
        guidance.append(f"- No explicit supported ICD-10-CM list was found for {charge_system} {charge_code}.")

    # Noncovered mappings are stronger negative guidance when the diagnosis appears in the list.
    if noncovered_codes:
        if _code_in_guideline_set(diagnosis_code, noncovered_codes):
            guidance.append(f"- {diagnosis_code} is explicitly noncovered for {charge_system} {charge_code}.")
        else:
            guidance.append(f"- {diagnosis_code} is not in the explicit noncovered list for {charge_system} {charge_code}.")
    else:
        guidance.append(f"- No explicit noncovered guidance was found for {diagnosis_code} with {charge_system} {charge_code}.")

    # Missing mapping data should not be mistaken for a coverage decision.
    if not supported_codes and not noncovered_codes:
        guidance.append("- Do not treat missing coverage guidance as evidence that the association is correct or incorrect.")

    return guidance


def _format_tabular_guidance(guidance: typing.Optional[Icd10TabularGuidance], diagnosis_code: str) -> typing.List[str]:

    # If no XML match exists, say that explicitly so the model does not infer hidden guidance.
    if guidance is None:
        return [f"- No ICD-10-CM tabular guidance was found for {diagnosis_code}."]

    # Start with the matched code and description, then add broader hierarchy context.
    formatted_guidance = [f"- {guidance.code}: {guidance.description}"]

    # Include only the nearest parent categories to avoid making the prompt too noisy.
    for parent_category in guidance.parent_categories[-3:]:
        formatted_guidance.append(f"- Parent category {parent_category}")

    if guidance.chapter:
        formatted_guidance.append(f"- Chapter: {guidance.chapter}")
    if guidance.section:
        formatted_guidance.append(f"- Section: {guidance.section}")

    for note in guidance.notes:
        formatted_guidance.append(f"- {note}")

    return formatted_guidance


def _code_in_guideline_set(diagnosis_code: str, guideline_codes: typing.Set[str]) -> bool:

    # Membership checks use all common ICD-10-CM code spellings for the same diagnosis.
    return any(candidate in guideline_codes for candidate in _build_icd10_code_candidates(diagnosis_code))


def _build_icd10_code_candidates(diagnosis_code: str) -> typing.List[str]:

    # Build a small set of dotted and non-dotted candidates for robust lookups.
    normalized_code = _normalize_code(diagnosis_code)
    no_dot_code = normalized_code.replace(".", "")

    candidates = []
    if normalized_code:
        candidates.append(normalized_code)
    if no_dot_code and no_dot_code not in candidates:
        candidates.append(no_dot_code)

    # Add a dotted variant when the source code was stored without a dot.
    dotted_code = f"{no_dot_code[:3]}.{no_dot_code[3:]}" if len(no_dot_code) > 3 else ""
    if dotted_code and dotted_code not in candidates:
        candidates.append(dotted_code)

    return candidates


def _get_value(item: typing.Any, field: str, default: typing.Any = ""):

    # Node scripts are standalone, so consume cross-node objects structurally.
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _set_value(item: typing.Any, field: str, value: typing.Any):

    # Support both dictionary-style and object-style graph state payloads.
    if isinstance(item, dict):
        item[field] = value
    else:
        setattr(item, field, value)


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
