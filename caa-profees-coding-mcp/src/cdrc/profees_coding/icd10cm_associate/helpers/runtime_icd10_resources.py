"""
Utilities for loading and resolving runtime ICD-10-CM resources from local JSON artifacts.
"""

import json
from pathlib import Path


def normalize_icd10_resource_code(value):

    # Converting ICD-10-CM codes to the standard dotted uppercase representation used by local artifacts.
    code = str(value or "").strip().upper().replace(".", "").replace(" ", "")

    if not code or code == "XX000":
        return ""

    if len(code) <= 3:
        return code

    return f"{code[:3]}.{code[3:]}"


def build_icd10_lookup_candidates(value):

    # Building a small list of equivalent code forms so runtime lookups can tolerate missing dots.
    raw_code = str(value or "").strip().upper().replace(" ", "")
    if not raw_code:
        return []

    candidates = []
    seen_candidates = set()

    def append_candidate(candidate):
        candidate = str(candidate or "").strip().upper()
        if candidate and candidate not in seen_candidates:
            seen_candidates.add(candidate)
            candidates.append(candidate)

    append_candidate(raw_code)
    append_candidate(raw_code.replace(".", ""))
    append_candidate(normalize_icd10_resource_code(raw_code))

    return candidates


def resolve_icd10_resource_code(resource_map, code):

    # Returning the first code representation that exists in the loaded resource map.
    for candidate in build_icd10_lookup_candidates(code):
        if candidate in (resource_map or {}):
            return candidate

    return None


def get_icd10_resource_value(resource_map, code, default=None):

    # Resolving a code against the resource map while tolerating dotted and non-dotted inputs.
    resolved_code = resolve_icd10_resource_code(resource_map, code)
    if resolved_code is None:
        return default

    return resource_map.get(resolved_code, default)


def load_icd10_json_resource(path, resource_name, required):

    # Loading one ICD-10-CM JSON artifact and normalizing its keys to the canonical dotted format.
    file_path = Path(path)

    try:
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as error:
        if required:
            raise RuntimeError(f"Failed to load required {resource_name} resource from '{file_path}': {error}") from error
        return {}

    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected {resource_name} resource at '{file_path}' to contain a JSON object.")

    normalized_payload = {}
    for raw_code, value in payload.items():
        normalized_code = normalize_icd10_resource_code(raw_code)
        if normalized_code:
            normalized_payload[normalized_code] = value

    return normalized_payload
