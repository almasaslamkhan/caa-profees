import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_DATASET_PATH = _ROOT / "data" / "icd10cm" / "fy2026_apr2026"
_METADATA_PATH = _DATASET_PATH / "metadata.json"
_RULE_CONFIG_PATH = _DATASET_PATH / "combo_rules.yaml"

ACTIVE_DATASET = json.loads(_METADATA_PATH.read_text())
ACTIVE_RULE_CONFIG: dict[str, Any]
__version__: str
ACTIVE_RETAIN_CODE_SET: frozenset[str]
ACTIVE_CODE_SET: frozenset[str]


@dataclass(frozen=True)
class ComboRule:
    required: tuple[str, ...]
    output: str
    source: str = "curated"


@dataclass(frozen=True)
class Evidence:
    code: str
    provenance: frozenset[str]
    is_original: bool


@dataclass(frozen=True)
class RuleMatch:
    rule: ComboRule
    provenance: frozenset[str]
    evidence_codes: frozenset[str]


def _to_dotted(code: str) -> str:
    compact = code.strip().upper().replace(".", "")
    if not compact:
        return ""
    if "." in code:
        return code.strip().upper()
    return compact if len(compact) <= 3 else compact[:3] + "." + compact[3:]


def _compact(code: str) -> str:
    return code.replace(".", "")


def _sort_codes(codes: set[str] | frozenset[str]) -> list[str]:
    return sorted(codes, key=lambda code: (_compact(code), code))


def _dataset_file(path_text: str) -> Path:
    return _DATASET_PATH / path_text


def _as_dotted_set(values: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    return {_to_dotted(value) for value in values}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.isdigit():
        return int(value)
    return value.strip('"').strip("'")


def _yaml_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lines.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
    return lines


def _parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    container: Any = [] if lines[index][1].startswith("- ") else {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation in rule file: {content}")

        if isinstance(container, list):
            if not content.startswith("- "):
                break
            item = content[2:].strip()
            if item:
                container.append(_parse_scalar(item))
                index += 1
            else:
                value, index = _parse_yaml_block(lines, index + 1, indent + 2)
                container.append(value)
            continue

        if content.startswith("- "):
            break
        key, separator, value_text = content.partition(":")
        if not separator:
            raise ValueError(f"Invalid mapping line in rule file: {content}")
        key = key.strip()
        value_text = value_text.strip()
        if value_text:
            container[key] = _parse_scalar(value_text)
            index += 1
        else:
            value, index = _parse_yaml_block(lines, index + 1, indent + 2)
            container[key] = value

    return container, index


def _load_rule_config() -> dict[str, Any]:
    lines = _yaml_lines(_RULE_CONFIG_PATH)
    parsed, index = _parse_yaml_block(lines, 0, 0)
    if index != len(lines):
        raise ValueError("Could not parse the complete rule file")
    if not isinstance(parsed, dict):
        raise ValueError("Rule file must be a mapping")
    parsed["_source_file"] = _RULE_CONFIG_PATH.name
    return parsed


ACTIVE_RULE_CONFIG = _load_rule_config()
__version__ = ACTIVE_RULE_CONFIG["rules_version"]


def get_active_dataset_metadata() -> dict[str, Any]:
    return deepcopy(ACTIVE_DATASET)


def _load_rules() -> tuple[list[ComboRule], set[str]]:
    rules: list[ComboRule] = []
    required_detail_codes: set[str] = set()

    for set_name in ACTIVE_RULE_CONFIG.get("rule_set_order", []):
        rule_set = ACTIVE_RULE_CONFIG["rule_sets"][set_name]
        source = rule_set.get("source", set_name)
        for left, right in rule_set.get("combo_codes", {}).items():
            required = tuple(_to_dotted(code) for code in left.split(",") if code.strip())
            rules.append(ComboRule(required=required, output=_to_dotted(right), source=source))

    for values in ACTIVE_RULE_CONFIG.get("required_detail_codes", {}).values():
        required_detail_codes.update(_as_dotted_set(values))

    unique: dict[tuple[str, ...], ComboRule] = {}
    for rule in rules:
        key = tuple(sorted(rule.required, key=_compact))
        unique[key] = ComboRule(required=key, output=rule.output, source=rule.source)
    return list(unique.values()), required_detail_codes


def _build_family_aliases() -> dict[str, dict[str, frozenset[str] | tuple[str, ...]]]:
    aliases: dict[str, dict[str, frozenset[str] | tuple[str, ...]]] = {}
    for required_code, spec in ACTIVE_RULE_CONFIG.get("family_aliases", {}).items():
        codes = _as_dotted_set(spec.get("codes", []))
        prefixes = tuple(prefix.upper() for prefix in spec.get("prefixes", []))
        aliases[_to_dotted(required_code)] = {
            "codes": frozenset(codes),
            "prefixes": prefixes,
        }
    return aliases


def _build_specificity_rankings() -> dict[str, dict[str, int]]:
    rankings: dict[str, dict[str, int]] = {}
    for name, ranking in ACTIVE_RULE_CONFIG.get("specificity_rankings", {}).items():
        rankings[name] = {_to_dotted(code): int(rank) for code, rank in ranking.items()}
    return rankings


_COMBO_RULES, _MANDATORY_DETAIL_CODES = _load_rules()
ACTIVE_RETAIN_CODE_SET = frozenset(_MANDATORY_DETAIL_CODES)
ACTIVE_CODE_SET = ACTIVE_RETAIN_CODE_SET
_FAMILY_ALIASES = _build_family_aliases()
_SPECIFICITY_RANKINGS = _build_specificity_rankings()


def _policy_items(section_name: str) -> list[dict[str, Any]]:
    section = ACTIVE_RULE_CONFIG.get(section_name, {})
    if isinstance(section, dict):
        return [value for value in section.values() if isinstance(value, dict)]
    if isinstance(section, list):
        return [value for value in section if isinstance(value, dict)]
    return []


def _family_aliases(required: str) -> set[str]:
    spec = _FAMILY_ALIASES.get(required)
    if not spec:
        return set()
    return set(spec["codes"])


def _matches_family_alias(code: str, required: str) -> bool:
    spec = _FAMILY_ALIASES.get(required)
    if not spec:
        return False
    return code in spec["codes"] or any(code.startswith(prefix) for prefix in spec["prefixes"])


def _is_prefix_specific(candidate: str, required: str) -> bool:
    candidate_compact = _compact(candidate)
    required_compact = _compact(required)
    return len(candidate_compact) > len(required_compact) and candidate_compact.startswith(required_compact)


def _satisfies(evidence: Evidence, required: str) -> bool:
    if evidence.code == required:
        return True
    if _is_prefix_specific(evidence.code, required):
        return True
    return evidence.is_original and _matches_family_alias(evidence.code, required)


def _evidence_score(evidence: Evidence, required: str, output: str) -> tuple[int, int, str]:
    if evidence.code == required:
        kind = 0
    elif evidence.code == output:
        kind = 1
    elif _is_prefix_specific(evidence.code, required):
        kind = 2
    elif _matches_family_alias(evidence.code, required):
        kind = 3
    else:
        kind = 4
    return (kind, len(evidence.provenance), _compact(evidence.code))


def _match_rule(rule: ComboRule, evidence_by_code: dict[str, Evidence]) -> RuleMatch | None:
    provenance: set[str] = set()
    evidence_codes: set[str] = set()

    for required in rule.required:
        matches = [evidence for evidence in evidence_by_code.values() if _satisfies(evidence, required)]
        if not matches:
            return None
        evidence = min(matches, key=lambda item: _evidence_score(item, required, rule.output))
        provenance.update(evidence.provenance)
        evidence_codes.add(evidence.code)

    return RuleMatch(rule=rule, provenance=frozenset(provenance), evidence_codes=frozenset(evidence_codes))


def _kept_matches(matches: list[RuleMatch]) -> list[RuleMatch]:
    kept: list[RuleMatch] = []
    for idx, match in enumerate(matches):
        suppressed = False
        for other_idx, other in enumerate(matches):
            if idx == other_idx:
                continue
            if match.provenance < other.provenance:
                suppressed = True
                break
            if (
                match.provenance == other.provenance
                and match.rule.output != other.rule.output
                and _is_less_specific_output(match.rule.output, other.rule.output)
            ):
                suppressed = True
                break
        if not suppressed:
            kept.append(match)
    return kept


def _matches_codes_or_prefixes(code: str, spec: dict[str, Any], code_key: str, prefix_key: str) -> bool:
    if code in _as_dotted_set(spec.get(code_key, [])):
        return True
    return any(code.startswith(prefix.upper()) for prefix in spec.get(prefix_key, []))


def _is_less_specific_output(candidate: str, other: str) -> bool:
    if candidate == other:
        return False
    if _compact(other).startswith(_compact(candidate)) and len(_compact(other)) > len(_compact(candidate)):
        return True
    for spec in _policy_items("less_specific_outputs"):
        if _matches_codes_or_prefixes(candidate, spec, "candidate_codes", "candidate_prefixes"):
            if _matches_codes_or_prefixes(other, spec, "other_codes", "other_prefixes"):
                return True
    return False


def _closure(normalized_inputs: list[str]) -> tuple[dict[str, Evidence], list[RuleMatch]]:
    evidence_by_code: dict[str, Evidence] = {}
    for code in normalized_inputs:
        if code not in evidence_by_code:
            evidence_by_code[code] = Evidence(code=code, provenance=frozenset({code}), is_original=True)

    all_kept_matches: list[RuleMatch] = []
    for _ in range(25):
        matches = [match for rule in _COMBO_RULES if (match := _match_rule(rule, evidence_by_code)) is not None]
        kept = _kept_matches(matches)
        all_kept_matches = kept

        changed = False
        for match in kept:
            output = match.rule.output
            current = evidence_by_code.get(output)
            if current is None:
                evidence_by_code[output] = Evidence(
                    code=output,
                    provenance=match.provenance,
                    is_original=False,
                )
                changed = True
            else:
                merged = current.provenance | match.provenance
                if merged != current.provenance:
                    evidence_by_code[output] = Evidence(
                        code=output,
                        provenance=frozenset(merged),
                        is_original=current.is_original,
                    )
                    changed = True

        if not changed:
            return evidence_by_code, all_kept_matches

    raise RuntimeError("ICD-10-CM combo rule closure did not converge")


def _has_higher_ranked_output(code: str, mapped_outputs: set[str]) -> bool:
    for ranking in _SPECIFICITY_RANKINGS.values():
        rank = ranking.get(code)
        if rank is not None and any(ranking.get(output, -1) > rank for output in mapped_outputs):
            return True
    return False


def _matches_retain_rule(code: str, rule: dict[str, Any]) -> bool:
    if code in _as_dotted_set(rule.get("except_codes", [])):
        return False
    if code in _as_dotted_set(rule.get("codes", [])):
        return True
    return any(code.startswith(prefix.upper()) for prefix in rule.get("prefixes", []))


def _should_retain_original(code: str, mapped_outputs: set[str]) -> bool:
    if _has_higher_ranked_output(code, mapped_outputs):
        return False
    if code in ACTIVE_RETAIN_CODE_SET:
        return True
    if not mapped_outputs:
        return True
    return any(_matches_retain_rule(code, rule) for rule in _policy_items("retain_original_rules"))


def _remove_invalid_co_outputs(outputs: set[str]) -> set[str]:
    result = set(outputs)
    for first, second in combinations(outputs, 2):
        if _is_less_specific_output(first, second):
            result.discard(first)
        elif _is_less_specific_output(second, first):
            result.discard(second)

    for exclusion in _policy_items("co_output_exclusions"):
        if any(any(output.startswith(prefix.upper()) for prefix in exclusion.get("if_any_prefixes", [])) for output in result):
            result.difference_update(_as_dotted_set(exclusion.get("remove_codes", [])))
    return result


def _output_code_mapping_many(input_codes: list[str]) -> dict[str, list[str]]:
    normalized_inputs = [_to_dotted(code) for code in input_codes]
    _, kept_matches = _closure(normalized_inputs)

    result: dict[str, set[str]] = {}
    outputs_by_evidence: dict[str, set[str]] = {normalized: set() for normalized in normalized_inputs}

    for match in kept_matches:
        result.setdefault(match.rule.output, set()).update(match.provenance)
        for normalized in match.provenance:
            outputs_by_evidence.setdefault(normalized, set()).add(match.rule.output)

    for normalized in normalized_inputs:
        if _should_retain_original(normalized, outputs_by_evidence.get(normalized, set())):
            result.setdefault(normalized, set()).add(normalized)

    kept_outputs = _remove_invalid_co_outputs(set(result))
    result = {output: evidence_codes for output, evidence_codes in result.items() if output in kept_outputs}

    return {output: _sort_codes(evidence_codes) for output, evidence_codes in sorted(result.items(), key=lambda item: (_compact(item[0]), item[0]))}


def icd10_code_mapping_many(input_codes: list[str]) -> dict[str, list[str]]:
    return _output_code_mapping_many(input_codes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ICD-10-CM combo-code rules.")
    parser.add_argument("codes", nargs="+", help="ICD-10-CM codes to map.")
    args = parser.parse_args()

    print(json.dumps(icd10_code_mapping_many(args.codes), indent=2, sort_keys=True))


__all__ = [
    "ACTIVE_CODE_SET",
    "ACTIVE_RETAIN_CODE_SET",
    "ACTIVE_DATASET",
    "ACTIVE_RULE_CONFIG",
    "__version__",
    "get_active_dataset_metadata",
    "icd10_code_mapping_many",
]


if __name__ == "__main__":
    main()
