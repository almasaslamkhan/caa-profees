#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

_DATA_ROOT = Path(__file__).resolve().parents[1] / "assets"
DEFAULT_CPT_ICD_MAPPING = _DATA_ROOT / "cpt_icd10_eligibility.json"
DEFAULT_HCPCS_ICD_MAPPING = _DATA_ROOT / "hcpcs_icd10_eligibility.json"
DEFAULT_CPT_DESCRIPTIONS = _DATA_ROOT / "cpt_descriptions.json"
DEFAULT_HCPCS_DESCRIPTIONS = _DATA_ROOT / "hcpcs_descriptions.json"
DEFAULT_ICD10_DESCRIPTIONS = _DATA_ROOT / "icd10_descriptions.json"


def _read_json(path: str | None):
    if path:
        return json.loads(Path(path).read_text())
    return json.loads(sys.stdin.read())


def _write_json(data, path: str | None):
    output = json.dumps(data, indent=2, sort_keys=False)
    if path:
        Path(path).write_text(output + "\n")
    else:
        sys.stdout.write(output + "\n")


def _load_mapping(path: str):
    try:
        raw = json.loads(Path(path).read_text())
        return {code: set(icds) for code, icds in raw.items()}
    except Exception:
        return {}


def _load_descriptions(path: str):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _filter_linked_diagnoses(items, mapping, system):
    if not items:
        return items
    updated = []
    for item in items:
        item_dict = dict(item)
        if item_dict.get("system") == system:
            code = item_dict.get("code", "")
            linked = item_dict.get("linked_diagnosis", []) or []
            if code in mapping and len(mapping[code]) > 0:
                filtered = [dx for dx in linked if dx.get("code", "") in mapping[code]]
                item_dict["linked_diagnosis"] = filtered
        updated.append(item_dict)
    return updated


def _populate_descriptions(state, cpt_desc, hcpcs_desc, icd10_desc):
    for procedure in state.get("procedure", []) or []:
        code = procedure.get("code", "")
        description = cpt_desc.get(code) or hcpcs_desc.get(code)
        if description:
            procedure["description"] = description
    for em in state.get("em", []) or []:
        code = em.get("code", "")
        description = cpt_desc.get(code)
        if description:
            em["description"] = description
    for dx in state.get("diagnosis", []) or []:
        code = dx.get("code", "")
        description = icd10_desc.get(code)
        if description:
            dx["description"] = description


def _remove_unassociated_diagnoses(state):
    charges = (state.get("em", []) or []) + (state.get("procedure", []) or [])
    associated = set()
    for charge in charges:
        for dx in charge.get("linked_diagnosis", []) or []:
            code = dx.get("code", "")
            if code:
                associated.add(code)
    if not associated:
        state["diagnosis"] = []
        return
    filtered = [dx for dx in (state.get("diagnosis", []) or []) if dx.get("code", "") in associated]
    state["diagnosis"] = filtered


def main():
    parser = argparse.ArgumentParser(description="Apply ICD-10 association post-processing rules.")
    parser.add_argument("--input", help="Input JSON file. Defaults to stdin.")
    parser.add_argument("--output", help="Output JSON file. Defaults to stdout.")
    parser.add_argument("--cpt-filtering", action="store_true", help="Enable CPT eligibility filtering.")
    parser.add_argument("--hcpcs-filtering", action="store_true", help="Enable HCPCS eligibility filtering.")
    parser.add_argument("--populate-descriptions", action="store_true", help="Populate descriptions from mapping files.")
    parser.add_argument("--remove-unassociated", action="store_true", help="Remove diagnoses with no links.")
    parser.add_argument("--cpt-icd-mapping-path", default=DEFAULT_CPT_ICD_MAPPING)
    parser.add_argument("--hcpcs-icd-mapping-path", default=DEFAULT_HCPCS_ICD_MAPPING)
    parser.add_argument("--cpt-description-path", default=DEFAULT_CPT_DESCRIPTIONS)
    parser.add_argument("--hcpcs-description-path", default=DEFAULT_HCPCS_DESCRIPTIONS)
    parser.add_argument("--icd10-description-path", default=DEFAULT_ICD10_DESCRIPTIONS)
    args = parser.parse_args()

    state = _read_json(args.input)

    if args.cpt_filtering:
        cpt_mapping = _load_mapping(args.cpt_icd_mapping_path)
        state["procedure"] = _filter_linked_diagnoses(state.get("procedure", []), cpt_mapping, "CPT")

    if args.hcpcs_filtering:
        hcpcs_mapping = _load_mapping(args.hcpcs_icd_mapping_path)
        state["procedure"] = _filter_linked_diagnoses(state.get("procedure", []), hcpcs_mapping, "HCPCS")

    if args.populate_descriptions:
        cpt_desc = _load_descriptions(args.cpt_description_path)
        hcpcs_desc = _load_descriptions(args.hcpcs_description_path)
        icd10_desc = _load_descriptions(args.icd10_description_path)
        _populate_descriptions(state, cpt_desc, hcpcs_desc, icd10_desc)

    if args.remove_unassociated:
        _remove_unassociated_diagnoses(state)

    _write_json(state, args.output)


if __name__ == "__main__":
    main()
