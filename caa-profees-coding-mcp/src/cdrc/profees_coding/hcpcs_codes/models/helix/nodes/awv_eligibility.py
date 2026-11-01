import json
import re
from argparse import Namespace
from datetime import date, datetime
from typing import Any, List, Literal, Mapping, Optional, Sequence

from langchain_core.messages import HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from cdrc.common.data.schema import AWVHistory, HealthPlan
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.hcpcs_codes.models.helix.constants import HCPCS_AWV_ELIGIBILITY_PROMPT_VERSION
from cdrc.profees_coding.hcpcs_codes.models.utils import CONFIG, PROMPTS

from .hcpcs_code import _as_bool, _get_customer_name_from_state, _get_model_kwargs_and_possibly_change_seed

AWV_CODES = {"G0438", "G0439"}
IPPE_CODE = "G0402"

MEDICARE_POSITIVE_PATTERNS = (
    "medicare",
    "medicare advantage",
    "part b",
    "a and b",
    "traditional medicare",
    "medigap",
    "medicare supplement",
    "plan g",
    "plan n",
    "aarp",
)
ACTIVE_STATUS_TERMS = {"active", "current", "eligible", "in effect"}
INACTIVE_STATUS_TERMS = {"inactive", "terminated", "expired", "cancelled", "canceled"}
PRECHECK_FAILURE_EXPLANATIONS = {
    "ENCOUNTER_DATE_INVALID": "the encounter date is missing or invalid",
    "HEALTH_PLAN_MISSING": "insurance coverage details were not available",
    "MEDICARE_PART_B_AMBIGUOUS": "health plan is ambiguous for Medicare Part B",
    "MEDICARE_PART_B_NOT_CONFIRMED": "Medicare Part B not confirmed",
    "PLAN_NOT_ACTIVE_ON_ENCOUNTER": "the available Medicare coverage does not appear active for the encounter date",
    "PART_B_EFFECTIVE_DATE_INVALID": "the Medicare Part B effective date is unavailable",
    "IPPE_ALREADY_BILLED": "a prior G0402/IPPE claim is already on file",
    "SUBSEQUENT_AWV_TOO_SOON": "a prior AWV claim is within the last 12 months",
}
CODE_BLOCKER_EXPLANATIONS = {
    "MEDICARE_PART_B_NOT_CONFIRMED": "Medicare Part B coverage is not confirmed",
    "PLAN_NOT_ACTIVE_ON_ENCOUNTER": "the available Medicare coverage does not appear active for the encounter date",
    "PART_B_EFFECTIVE_DATE_INVALID": "the Medicare Part B effective date is unavailable",
    "WITHIN_FIRST_PART_B_YEAR": "the encounter occurred within the first 12 months of Medicare Part B",
    "NOT_WITHIN_FIRST_PART_B_YEAR": "the encounter occurred more than 12 months after Medicare Part B began",
    "PRIOR_IPPE_EXISTS": "a prior G0402/IPPE claim is already on file",
    "IPPE_WITHIN_12_MONTHS": "a prior G0402/IPPE claim is within the last 12 months",
    "PRIOR_AWV_EXISTS": "a prior AWV claim is already on file",
    "NO_PRIOR_AWV_HISTORY": "no prior AWV claim is available in the history",
    "AWV_WITHIN_12_MONTHS": "a prior AWV claim is within the last 12 months",
}
AWV_FALLBACK_DESCRIPTIONS = {
    "G0402": "Initial Preventive Physical Exam / Welcome to Medicare",
    "G0438": "Initial Annual Wellness Visit",
    "G0439": "Subsequent Annual Wellness Visit",
}
CODE_DISPLAY_NAMES = {
    "G0402": "G0402 (IPPE / Welcome to Medicare)",
    "G0438": "G0438 (Initial AWV)",
    "G0439": "G0439 (Subsequent AWV)",
}
IPPE_FALLBACK_PATTERNS = (
    r"\bwelcome\s+to\s+medicare\b",
    r"\bippe\b",
    r"\binitial\s+preventive\s+physical\s+exam\b",
    r"\bwelcome\s+visit\b",
    r"\bwelcome\s+to\s+awv\b",
)
INITIAL_AWV_FALLBACK_PATTERNS = (
    r"\binitial\s+(?:medicare\s+)?annual\s+wellness\s+visit\b",
    r"\binitial\s+awv\b",
    r"\b(?:medicare\s+)?annual\s+wellness\s+visit\s*,\s*initial\b",
    r"\bawv\s*,\s*initial\b",
)
AWV_FALLBACK_PATTERNS = (
    r"\bsubsequent\s+(?:medicare\s+)?annual\s+wellness\s+visit\b",
    r"\bsubsequent\s+awv\b",
    r"\b(?:medicare\s+)?annual\s+wellness\s+visit\b",
    r"\bmedicare\s+wellness\b",
    r"\bawv\b",
)


class PlanInterpretationOutput(BaseModel):
    medicare_part_b_likely: Optional[bool] = Field(None, description="Whether the plan likely represents active Medicare Part B coverage")
    evidence: List[str] = Field(default_factory=list, description="Short evidence snippets from the plan text")


class AWVEligibilityOutput(BaseModel):
    medicare_part_b_likely: Optional[bool]
    plan_active_on_encounter: Optional[bool]
    first_part_b_effective_date: Optional[str]
    months_since_part_b_effective: Optional[int]
    within_first_part_b_year: Optional[bool]
    prior_ippe_exists: bool
    latest_ippe_claim_date: Optional[str]
    months_since_latest_ippe: Optional[int]
    prior_ippe_within_12_months: bool
    prior_awv_exists: bool
    latest_awv_claim_date: Optional[str]
    months_since_latest_awv: Optional[int]
    g0402_eligible: bool
    g0402_blockers: List[str] = Field(default_factory=list)
    g0438_eligible: bool
    g0438_blockers: List[str] = Field(default_factory=list)
    g0439_eligible: bool
    g0439_blockers: List[str] = Field(default_factory=list)
    hard_blocks: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    summary_for_classifier: str

    @property
    def g0402_history_allows(self) -> bool:
        return not self.prior_ippe_exists

    @property
    def g0438_history_allows(self) -> bool:
        return not self.prior_awv_exists

    @property
    def g0439_history_allows(self) -> bool:
        return bool(self.prior_awv_exists and self.months_since_latest_awv is not None and self.months_since_latest_awv >= 12)


def _dedupe_preserving_order(items: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _describe_blockers(blockers: Sequence[str], *, explanation_map: Mapping[str, str]) -> List[str]:
    return [explanation_map.get(blocker, blocker.replace("_", " ").lower()) for blocker in _dedupe_preserving_order(blockers)]


def _join_phrases(items: Sequence[str]) -> str:
    values = [item.strip() for item in items if item and item.strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _context_gap_phrase(state: Mapping[str, Any]) -> str:
    missing_plan = not _coerce_health_plan_entries(state.get("health_plan"))
    missing_history = not _coerce_awv_history_entries(state.get("awv_history"))
    if missing_plan and missing_history:
        return "coverage details and prior AWV/IPPE history were not available for additional verification"
    if missing_plan:
        return "coverage details were not available for additional verification"
    if missing_history:
        return "prior AWV/IPPE history was not available for additional verification"
    return "supporting eligibility details were not available for additional verification"


def _build_fallback_procedure_justification(code: str, state: Mapping[str, Any]) -> str:
    if code == "G0402":
        return "The note documents IPPE or Welcome to Medicare wording; coverage or claim history details were unavailable."
    if code == "G0438":
        return "The note documents an initial AWV; coverage or claim history details were unavailable."
    return "The note documents a Medicare AWV; coverage or claim history details were unavailable."


def _build_fallback_precheck_justification(code: str, state: Mapping[str, Any]) -> str:
    context_gap = _context_gap_phrase(state)
    return f"{CODE_DISPLAY_NAMES[code]} is the best-supported code from the SOAP note, and {context_gap}."


def _build_failure_justification(eligibility: AWVEligibilityOutput) -> str:
    phrases = _describe_blockers(eligibility.hard_blocks, explanation_map=PRECHECK_FAILURE_EXPLANATIONS)

    if phrases:
        return f"AWV precheck failed: a Medicare AWV/IPPE code cannot be supported because {_join_phrases(phrases)}."

    code_summaries: List[str] = []
    for code, eligible, blockers in (
        (IPPE_CODE, eligibility.g0402_eligible, eligibility.g0402_blockers),
        ("G0438", eligibility.g0438_eligible, eligibility.g0438_blockers),
        ("G0439", eligibility.g0439_eligible, eligibility.g0439_blockers),
    ):
        if eligible or not blockers:
            continue
        blocker_phrases = _describe_blockers(blockers, explanation_map=CODE_BLOCKER_EXPLANATIONS)
        code_summaries.append(f"{CODE_DISPLAY_NAMES[code]} is not supported because {_join_phrases(blocker_phrases)}")

    if code_summaries:
        return f"AWV precheck failed: a Medicare AWV/IPPE code cannot be supported with the available eligibility information: {'; '.join(code_summaries)}."
    return "AWV precheck failed: a Medicare AWV/IPPE code cannot be supported with the available eligibility information."


def _build_failed_awv_result(
    state: Mapping[str, Any],
    eligibility: AWVEligibilityOutput,
    *,
    mode: Literal["deterministic", "llm"],
) -> dict[str, Any]:
    return {
        "procedure": [],
        "customer_name": _get_customer_name_from_state(state),
        "additional_output": {
            "awv_precheck": {
                "status": "failed",
                "mode": mode,
                "hard_blocks": list(eligibility.hard_blocks),
                "justification": _build_failure_justification(eligibility),
                "eligibility": eligibility.model_dump(),
            }
        },
    }


def _extract_soap_awv_fallback_code(note_text: Any) -> Optional[str]:
    text = str(note_text or "").lower()
    if not text.strip():
        return None

    if any(re.search(pattern, text) for pattern in IPPE_FALLBACK_PATTERNS):
        return "G0402"
    if any(re.search(pattern, text) for pattern in INITIAL_AWV_FALLBACK_PATTERNS):
        return "G0438"
    if any(re.search(pattern, text) for pattern in AWV_FALLBACK_PATTERNS):
        return "G0439"
    return None


def _structured_awv_context_incomplete(state: Mapping[str, Any]) -> bool:
    return not _coerce_health_plan_entries(state.get("health_plan")) or not _coerce_awv_history_entries(state.get("awv_history"))


def _soap_fallback_blocked_by_structured_context(
    candidate_code: str,
    state: Mapping[str, Any],
    eligibility: AWVEligibilityOutput,
) -> bool:
    plan_entries = _coerce_health_plan_entries(state.get("health_plan"))
    if plan_entries and eligibility.medicare_part_b_likely is not True:
        return True
    if plan_entries and eligibility.plan_active_on_encounter is not True:
        return True

    if candidate_code == "G0402" and eligibility.prior_ippe_exists:
        return True
    if candidate_code == "G0438" and eligibility.prior_awv_exists:
        return True
    if candidate_code == "G0439" and "SUBSEQUENT_AWV_TOO_SOON" in eligibility.hard_blocks:
        return True
    return False


def _build_fallback_awv_result(
    state: Mapping[str, Any],
    eligibility: AWVEligibilityOutput,
    *,
    mode: Literal["deterministic", "llm"],
    code: str,
) -> dict[str, Any]:
    return {
        "procedure": [
            {
                "description": AWV_FALLBACK_DESCRIPTIONS[code],
                "code": code,
                "system": "HCPCS",
                "justification": _build_fallback_procedure_justification(code, state),
            }
        ],
        "customer_name": _get_customer_name_from_state(state),
        "additional_output": {
            "awv_precheck": {
                "status": "fallback",
                "mode": mode,
                "selected_code": code,
                "hard_blocks": list(eligibility.hard_blocks),
                "justification": _build_fallback_precheck_justification(code, state),
                "eligibility": eligibility.model_dump(),
            }
        },
    }


def _maybe_build_fallback_awv_result(
    state: Mapping[str, Any],
    eligibility: AWVEligibilityOutput,
    *,
    mode: Literal["deterministic", "llm"],
) -> Optional[dict[str, Any]]:
    if not _structured_awv_context_incomplete(state):
        return None

    candidate_code = _extract_soap_awv_fallback_code(state.get("text"))
    if candidate_code is None:
        return None
    if _soap_fallback_blocked_by_structured_context(candidate_code, state, eligibility):
        return None

    return _build_fallback_awv_result(state, eligibility, mode=mode, code=candidate_code)


def _finalize_awv_precheck(
    state: Mapping[str, Any],
    eligibility: AWVEligibilityOutput,
    *,
    mode: Literal["deterministic", "llm"],
) -> dict[str, Any]:
    fallback_result = _maybe_build_fallback_awv_result(state, eligibility, mode=mode)
    if fallback_result is not None:
        return {
            "awv_eligibility": eligibility.model_dump(),
            "awv_precheck_passed": False,
            "awv_result": fallback_result,
        }

    passed = not eligibility.hard_blocks and any((eligibility.g0402_eligible, eligibility.g0438_eligible, eligibility.g0439_eligible))
    result: dict[str, Any] = {
        "awv_eligibility": eligibility.model_dump(),
        "awv_precheck_passed": passed,
    }
    if not passed:
        result["awv_result"] = _build_failed_awv_result(state, eligibility, mode=mode)
    return result


def _normalize_prompt_version_key(prompt_version_config: Any) -> Optional[str]:
    if isinstance(prompt_version_config, Mapping):
        return prompt_version_config.get("version")
    if isinstance(prompt_version_config, str):
        return prompt_version_config
    return None


def _parse_date(raw_value: Any) -> Optional[date]:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    for candidate in (text, text.split("T")[0], text.split(" ")[0]):
        if not candidate:
            continue
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            pass

    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _months_between(start_date: date, end_date: date) -> int:
    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    if end_date.day < start_date.day:
        months -= 1
    return months


def _coerce_health_plan_entries(raw_health_plan: Any) -> List[Mapping[str, Any]]:
    if raw_health_plan is None:
        return []
    if isinstance(raw_health_plan, BaseModel):
        return [raw_health_plan.model_dump()]
    if isinstance(raw_health_plan, Mapping):
        return [dict(raw_health_plan)]
    if isinstance(raw_health_plan, Sequence) and not isinstance(raw_health_plan, (str, bytes)):
        entries = []
        for item in raw_health_plan:
            if isinstance(item, BaseModel):
                entries.append(item.model_dump())
            elif isinstance(item, Mapping):
                entries.append(dict(item))
        return entries
    return []


def _coerce_awv_history_entries(raw_history: Any) -> List[AWVHistory]:
    if raw_history is None:
        return []
    if isinstance(raw_history, BaseModel):
        raw_history = [raw_history]
    elif not isinstance(raw_history, list):
        raw_history = [raw_history]

    normalized: List[AWVHistory] = []
    for item in raw_history:
        try:
            if isinstance(item, AWVHistory):
                normalized.append(item)
            elif isinstance(item, BaseModel):
                normalized.append(AWVHistory(**item.model_dump()))
            elif isinstance(item, Mapping):
                normalized.append(AWVHistory(**item))
        except Exception:
            continue
    return normalized


def _plan_display_name(plan_entry: Mapping[str, Any]) -> str:
    return str(plan_entry.get("health_plan") or "unknown").strip() or "unknown"


def _prefix_plan_evidence(plan_entry: Mapping[str, Any], evidence: Sequence[str]) -> List[str]:
    plan_name = _plan_display_name(plan_entry)
    return [f"Plan '{plan_name}': {item}" for item in evidence]


def _determine_plan_active_on_encounter(
    plan_entry: Mapping[str, Any], encounter_day: Optional[date]
) -> tuple[Optional[bool], Optional[date], List[str]]:
    evidence: List[str] = []
    plan_name = _plan_display_name(plan_entry)
    status_value = str(plan_entry.get("active_status") or plan_entry.get("health_plan_status") or "").strip().lower()
    effective_day = _parse_date(plan_entry.get("health_plan_effective_date"))

    explicit_inactive = bool(status_value) and any(re.search(rf"\b{re.escape(term)}\b", status_value) for term in INACTIVE_STATUS_TERMS)
    explicit_active = bool(status_value) and any(re.search(rf"\b{re.escape(term)}\b", status_value) for term in ACTIVE_STATUS_TERMS)

    if explicit_inactive:
        evidence.append(f"Plan '{plan_name}' marked inactive")
    elif explicit_active:
        evidence.append(f"Plan '{plan_name}' marked active")

    if encounter_day and effective_day:
        if effective_day <= encounter_day:
            evidence.append(f"Plan '{plan_name}' effective on or before encounter date")
        else:
            evidence.append(f"Plan '{plan_name}' effective after encounter date")

    if explicit_inactive:
        return False, effective_day, evidence
    if encounter_day and effective_day and effective_day > encounter_day:
        return False, effective_day, evidence
    if explicit_active:
        return True, effective_day, evidence
    if encounter_day and effective_day:
        return True, effective_day, evidence
    return None, effective_day, evidence


def _determine_medicare_part_b_from_text(plan_entries: Sequence[Mapping[str, Any]]) -> tuple[Optional[bool], List[str], bool]:
    if not plan_entries:
        return None, [], False

    evidence: List[str] = []
    plan_texts = [str(entry.get("health_plan") or "").strip() for entry in plan_entries if entry.get("health_plan")]
    normalized_text = " || ".join(plan_texts).lower()

    if not normalized_text:
        return None, evidence, False

    for pattern in MEDICARE_POSITIVE_PATTERNS:
        if pattern in normalized_text:
            evidence.append(f"Matched Medicare signal: {pattern}")
            return True, evidence, False

    has_alpha_numeric_plan = bool(re.search(r"[a-zA-Z]", normalized_text))
    return None, evidence, has_alpha_numeric_plan


def _build_part_b_coverage_facts(
    raw_health_plan: Any,
    encounter_date: Any,
    plan_interpretations: Optional[Mapping[int, PlanInterpretationOutput]] = None,
) -> dict[str, Any]:
    plan_entries = _coerce_health_plan_entries(raw_health_plan)
    encounter_day = _parse_date(encounter_date)
    evidence: List[str] = []
    hard_blocks: List[str] = []
    medicare_candidate_infos: List[dict[str, Any]] = []
    saw_ambiguous_candidate = False

    if encounter_day is None:
        hard_blocks.append("ENCOUNTER_DATE_INVALID")
    if not plan_entries:
        hard_blocks.append("HEALTH_PLAN_MISSING")

    for index, plan_entry in enumerate(plan_entries):
        part_b_likely, part_b_evidence, llm_needed = _determine_medicare_part_b_from_text([plan_entry])
        interpretation = plan_interpretations.get(index) if plan_interpretations else None
        if part_b_likely is None and interpretation is not None:
            part_b_likely = interpretation.medicare_part_b_likely
            part_b_evidence = list(part_b_evidence) + list(interpretation.evidence)

        evidence.extend(_prefix_plan_evidence(plan_entry, part_b_evidence))

        if part_b_likely is True:
            active_on_encounter, effective_day, active_evidence = _determine_plan_active_on_encounter(plan_entry, encounter_day)
            evidence.extend(active_evidence)
            medicare_candidate_infos.append(
                {
                    "active_on_encounter": active_on_encounter,
                    "effective_day": effective_day,
                }
            )
        elif part_b_likely is None and (llm_needed or interpretation is not None):
            saw_ambiguous_candidate = True

    medicare_part_b_likely: Optional[bool]
    if medicare_candidate_infos:
        medicare_part_b_likely = True
    elif saw_ambiguous_candidate:
        medicare_part_b_likely = None
    else:
        medicare_part_b_likely = False

    plan_active_on_encounter: Optional[bool] = None
    first_part_b_effective_day: Optional[date] = None
    months_since_part_b_effective: Optional[int] = None
    within_first_part_b_year: Optional[bool] = None

    if medicare_candidate_infos:
        active_candidates = [item for item in medicare_candidate_infos if item["active_on_encounter"] is True]
        if active_candidates:
            plan_active_on_encounter = True
            historical_effective_days = [
                item["effective_day"]
                for item in medicare_candidate_infos
                if item["effective_day"] is not None and (encounter_day is None or item["effective_day"] <= encounter_day)
            ]
            if historical_effective_days:
                first_part_b_effective_day = min(historical_effective_days)
                if encounter_day is not None:
                    months_since_part_b_effective = _months_between(first_part_b_effective_day, encounter_day)
                    within_first_part_b_year = months_since_part_b_effective < 12
        elif any(item["active_on_encounter"] is None for item in medicare_candidate_infos):
            plan_active_on_encounter = None
        else:
            plan_active_on_encounter = False

    if plan_entries and medicare_part_b_likely is None:
        hard_blocks.append("MEDICARE_PART_B_AMBIGUOUS")
    elif plan_entries and medicare_part_b_likely is False:
        hard_blocks.append("MEDICARE_PART_B_NOT_CONFIRMED")

    if medicare_part_b_likely is True and plan_active_on_encounter is not True:
        hard_blocks.append("PLAN_NOT_ACTIVE_ON_ENCOUNTER")

    if encounter_day is not None and medicare_part_b_likely is True and plan_active_on_encounter is True and first_part_b_effective_day is None:
        hard_blocks.append("PART_B_EFFECTIVE_DATE_INVALID")

    return {
        "encounter_day": encounter_day,
        "medicare_part_b_likely": medicare_part_b_likely,
        "plan_active_on_encounter": plan_active_on_encounter,
        "first_part_b_effective_date": first_part_b_effective_day.isoformat() if first_part_b_effective_day else None,
        "months_since_part_b_effective": months_since_part_b_effective,
        "within_first_part_b_year": within_first_part_b_year,
        "hard_blocks": _dedupe_preserving_order(hard_blocks),
        "evidence": evidence,
    }


def _build_history_facts(raw_history: Any, encounter_day: Optional[date]) -> tuple[dict[str, Any], List[str]]:
    history_entries = _coerce_awv_history_entries(raw_history)
    evidence: List[str] = []

    prior_history_entries: List[tuple[str, date]] = []
    ignored_non_prior_entries: List[tuple[str, date]] = []

    for entry in history_entries:
        code = (entry.hcpcs_code or "").strip().upper()
        claim_day = _parse_date(entry.claim_date)
        if not code or claim_day is None:
            continue

        if encounter_day is not None and claim_day >= encounter_day:
            ignored_non_prior_entries.append((code, claim_day))
            continue

        prior_history_entries.append((code, claim_day))

    ippe_dates = [claim_day for code, claim_day in prior_history_entries if code == IPPE_CODE]
    prior_ippe_exists = bool(ippe_dates)
    latest_ippe_day = max(ippe_dates) if ippe_dates else None
    months_since_latest_ippe = _months_between(latest_ippe_day, encounter_day) if latest_ippe_day and encounter_day else None
    prior_ippe_within_12_months = bool(latest_ippe_day and months_since_latest_ippe is not None and months_since_latest_ippe < 12)
    awv_dates = [claim_day for code, claim_day in prior_history_entries if code in AWV_CODES]
    prior_awv_exists = bool(awv_dates)
    latest_awv_day = max(awv_dates) if awv_dates else None
    months_since_latest_awv = _months_between(latest_awv_day, encounter_day) if latest_awv_day and encounter_day else None

    if prior_ippe_exists:
        evidence.append("Prior G0402 found in AWV history")
    if latest_ippe_day:
        evidence.append(f"Most recent prior IPPE claim date: {latest_ippe_day.isoformat()}")
    if latest_awv_day:
        evidence.append(f"Most recent prior AWV claim date: {latest_awv_day.isoformat()}")
    for code, claim_day in ignored_non_prior_entries:
        history_label = "AWV" if code in AWV_CODES else "AWV/IPPE"
        evidence.append(f"Ignored non-prior {history_label} history entry {code} on {claim_day.isoformat()}")

    facts = {
        "prior_ippe_exists": prior_ippe_exists,
        "latest_ippe_claim_date": latest_ippe_day.isoformat() if latest_ippe_day else None,
        "months_since_latest_ippe": months_since_latest_ippe,
        "prior_ippe_within_12_months": prior_ippe_within_12_months,
        "prior_awv_exists": prior_awv_exists,
        "latest_awv_claim_date": latest_awv_day.isoformat() if latest_awv_day else None,
        "months_since_latest_awv": months_since_latest_awv,
    }
    return facts, evidence


def _build_code_eligibility(part_b_facts: Mapping[str, Any], history_facts: Mapping[str, Any]) -> dict[str, Any]:
    shared_blockers: List[str] = []
    if part_b_facts["medicare_part_b_likely"] is not True:
        shared_blockers.append("MEDICARE_PART_B_NOT_CONFIRMED")
    if part_b_facts["plan_active_on_encounter"] is not True:
        shared_blockers.append("PLAN_NOT_ACTIVE_ON_ENCOUNTER")
    if (
        part_b_facts["medicare_part_b_likely"] is True
        and part_b_facts["plan_active_on_encounter"] is True
        and part_b_facts["first_part_b_effective_date"] is None
    ):
        shared_blockers.append("PART_B_EFFECTIVE_DATE_INVALID")

    g0402_blockers = list(shared_blockers)
    if part_b_facts["within_first_part_b_year"] is False:
        g0402_blockers.append("NOT_WITHIN_FIRST_PART_B_YEAR")
    if history_facts["prior_ippe_exists"]:
        g0402_blockers.append("PRIOR_IPPE_EXISTS")
    if history_facts["prior_awv_exists"]:
        g0402_blockers.append("PRIOR_AWV_EXISTS")

    g0438_blockers = list(shared_blockers)
    if part_b_facts["within_first_part_b_year"] is True:
        g0438_blockers.append("WITHIN_FIRST_PART_B_YEAR")
    if history_facts["prior_awv_exists"]:
        g0438_blockers.append("PRIOR_AWV_EXISTS")
    if history_facts["prior_ippe_within_12_months"]:
        g0438_blockers.append("IPPE_WITHIN_12_MONTHS")

    g0439_blockers = list(shared_blockers)
    if part_b_facts["within_first_part_b_year"] is True:
        g0439_blockers.append("WITHIN_FIRST_PART_B_YEAR")
    if not history_facts["prior_awv_exists"]:
        g0439_blockers.append("NO_PRIOR_AWV_HISTORY")
    elif history_facts["months_since_latest_awv"] is not None and history_facts["months_since_latest_awv"] < 12:
        g0439_blockers.append("AWV_WITHIN_12_MONTHS")
    if history_facts["prior_ippe_within_12_months"]:
        g0439_blockers.append("IPPE_WITHIN_12_MONTHS")

    g0402_blockers = _dedupe_preserving_order(g0402_blockers)
    g0438_blockers = _dedupe_preserving_order(g0438_blockers)
    g0439_blockers = _dedupe_preserving_order(g0439_blockers)

    return {
        "g0402_eligible": not g0402_blockers,
        "g0402_blockers": g0402_blockers,
        "g0438_eligible": not g0438_blockers,
        "g0438_blockers": g0438_blockers,
        "g0439_eligible": not g0439_blockers,
        "g0439_blockers": g0439_blockers,
    }


def _legacy_hard_blocks(part_b_facts: Mapping[str, Any], history_facts: Mapping[str, Any]) -> List[str]:
    hard_blocks = list(part_b_facts["hard_blocks"])
    if "HEALTH_PLAN_MISSING" in hard_blocks and "PLAN_NOT_ACTIVE_ON_ENCOUNTER" not in hard_blocks:
        hard_blocks.append("PLAN_NOT_ACTIVE_ON_ENCOUNTER")
    if history_facts["prior_ippe_exists"]:
        hard_blocks.append("IPPE_ALREADY_BILLED")
    if history_facts["prior_awv_exists"] and history_facts["months_since_latest_awv"] is not None and history_facts["months_since_latest_awv"] < 12:
        hard_blocks.append("SUBSEQUENT_AWV_TOO_SOON")
    return _dedupe_preserving_order(hard_blocks)


def _build_summary(eligibility: AWVEligibilityOutput) -> str:
    coverage_summary = (
        "Medicare Part B coverage is confirmed."
        if eligibility.medicare_part_b_likely is True
        else ("Medicare Part B coverage is not confirmed." if eligibility.medicare_part_b_likely is False else "Medicare Part B coverage is unclear.")
    )
    activity_summary = (
        "The Medicare coverage appears active for the encounter date."
        if eligibility.plan_active_on_encounter is True
        else (
            "The Medicare coverage does not appear active for the encounter date."
            if eligibility.plan_active_on_encounter is False
            else "The plan activity status for the encounter date is unclear."
        )
    )

    timing_details: List[str] = []
    if eligibility.first_part_b_effective_date:
        timing_details.append(f"Part B effective date: {eligibility.first_part_b_effective_date}.")
    if eligibility.latest_ippe_claim_date:
        timing_details.append(f"Most recent prior G0402/IPPE claim date: {eligibility.latest_ippe_claim_date}.")
    if eligibility.latest_awv_claim_date:
        timing_details.append(f"Most recent prior AWV claim date: {eligibility.latest_awv_claim_date}.")

    code_support_summaries: List[str] = []
    for code, blockers in (
        ("G0402", eligibility.g0402_blockers),
        ("G0438", eligibility.g0438_blockers),
        ("G0439", eligibility.g0439_blockers),
    ):
        if blockers:
            blocker_phrases = _describe_blockers(blockers, explanation_map=CODE_BLOCKER_EXPLANATIONS)
            code_support_summaries.append(f"{CODE_DISPLAY_NAMES[code]} is not supported because {_join_phrases(blocker_phrases)}.")
        else:
            code_support_summaries.append(f"{CODE_DISPLAY_NAMES[code]} remains eligible based on the available coverage and claim history.")

    return " ".join([coverage_summary, activity_summary, *timing_details, *code_support_summaries]).strip()


def _run_plan_interpretation_llm(
    plan_entries: Sequence[Mapping[str, Any]],
    config: RunnableConfig,
    enable_prompt_cache_layout: bool,
) -> PlanInterpretationOutput:
    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser = PydanticOutputParserWithRetries(pydantic_object=PlanInterpretationOutput, correction_model=model, num_retries=3)
    chain = model | parser

    prompt_version = (
        _normalize_prompt_version_key(
            config["configurable"].get(HCPCS_AWV_ELIGIBILITY_PROMPT_VERSION, config["configurable"].get(CONFIG.PROMPT_VERSION))
        )
        or "v5.4"
    )
    system_prompt = SystemMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.AWV_ELIGIBILITY_PRECHECK][prompt_version]["system"],
        template_format="jinja2",
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.AWV_ELIGIBILITY_PRECHECK][prompt_version]["user"],
        template_format="jinja2",
    )

    format_instructions = parser.get_format_instructions()
    if enable_prompt_cache_layout:
        schema_message = HumanMessage(
            content=(
                "## Output Format\n"
                "Return your answer as a JSON object matching this schema:\n"
                f"{format_instructions}\n"
                "DO NOT include any other text in your response. Only return the JSON object."
            )
        )
        messages = [
            system_prompt.format(),
            schema_message,
            user_prompt.format(
                health_plan_json=json.dumps(plan_entries, sort_keys=True, ensure_ascii=True),
                format_instructions="(Output schema is provided in the previous message.)",
            ),
        ]
    else:
        messages = [
            system_prompt.format(),
            user_prompt.format(
                health_plan_json=json.dumps(plan_entries, sort_keys=True, ensure_ascii=True),
                format_instructions=format_instructions,
            ),
        ]

    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        merged_parts = [messages[0].content]
        if enable_prompt_cache_layout and len(messages) > 2:
            merged_parts.append(messages[1].content)
            user_message_index = 2
        else:
            user_message_index = 1
        merged_parts.append(messages[user_message_index].content)
        user_message = messages[user_message_index]
        user_message.content = "\n\n".join(part for part in merged_parts if part)
        messages = [user_message]

    return chain.invoke(messages)


def _build_awv_eligibility(
    raw_health_plan: Any,
    raw_history: Any,
    encounter_date: Any,
    plan_interpretations: Optional[Mapping[int, PlanInterpretationOutput]] = None,
) -> AWVEligibilityOutput:
    part_b_facts = _build_part_b_coverage_facts(
        raw_health_plan=raw_health_plan,
        encounter_date=encounter_date,
        plan_interpretations=plan_interpretations,
    )
    history_facts, history_evidence = _build_history_facts(raw_history, part_b_facts["encounter_day"])
    code_eligibility = _build_code_eligibility(part_b_facts, history_facts)

    eligibility = AWVEligibilityOutput(
        medicare_part_b_likely=part_b_facts["medicare_part_b_likely"],
        plan_active_on_encounter=part_b_facts["plan_active_on_encounter"],
        first_part_b_effective_date=part_b_facts["first_part_b_effective_date"],
        months_since_part_b_effective=part_b_facts["months_since_part_b_effective"],
        within_first_part_b_year=part_b_facts["within_first_part_b_year"],
        prior_ippe_exists=history_facts["prior_ippe_exists"],
        latest_ippe_claim_date=history_facts["latest_ippe_claim_date"],
        months_since_latest_ippe=history_facts["months_since_latest_ippe"],
        prior_ippe_within_12_months=history_facts["prior_ippe_within_12_months"],
        prior_awv_exists=history_facts["prior_awv_exists"],
        latest_awv_claim_date=history_facts["latest_awv_claim_date"],
        months_since_latest_awv=history_facts["months_since_latest_awv"],
        g0402_eligible=code_eligibility["g0402_eligible"],
        g0402_blockers=code_eligibility["g0402_blockers"],
        g0438_eligible=code_eligibility["g0438_eligible"],
        g0438_blockers=code_eligibility["g0438_blockers"],
        g0439_eligible=code_eligibility["g0439_eligible"],
        g0439_blockers=code_eligibility["g0439_blockers"],
        hard_blocks=_legacy_hard_blocks(part_b_facts, history_facts),
        evidence=part_b_facts["evidence"] + history_evidence,
        summary_for_classifier="",
    )
    eligibility.summary_for_classifier = _build_summary(eligibility)
    return eligibility


def awv_eligibility_precheck_deterministic(state: Mapping[str, Any], config: RunnableConfig) -> dict:
    eligibility = _build_awv_eligibility(
        raw_health_plan=state.get("health_plan"),
        raw_history=state.get("awv_history"),
        encounter_date=state.get("encounter_date"),
        plan_interpretations=None,
    )

    return _finalize_awv_precheck(state, eligibility, mode="deterministic")


@track_tokens("model_usage")
def awv_eligibility_precheck(state: Mapping[str, Any], config: RunnableConfig) -> dict:
    enable_prompt_cache_layout = _as_bool(config["configurable"].get("enable_prompt_cache_layout", True), default=True)
    plan_entries = _coerce_health_plan_entries(state.get("health_plan"))
    plan_interpretations: dict[int, PlanInterpretationOutput] = {}
    for index, plan_entry in enumerate(plan_entries):
        medicare_part_b_likely, _, llm_needed = _determine_medicare_part_b_from_text([plan_entry])
        if llm_needed and medicare_part_b_likely is None:
            plan_interpretations[index] = _run_plan_interpretation_llm([plan_entry], config, enable_prompt_cache_layout)

    eligibility = _build_awv_eligibility(
        raw_health_plan=state.get("health_plan"),
        raw_history=state.get("awv_history"),
        encounter_date=state.get("encounter_date"),
        plan_interpretations=plan_interpretations or None,
    )

    return _finalize_awv_precheck(state, eligibility, mode="llm")


def build_awv_eligibility_subgraph(args: Namespace):
    return awv_eligibility_precheck
