import json
import re
from argparse import Namespace
from datetime import date, datetime
from typing import Any, List, Mapping, Optional, Sequence

from langchain_core.messages import HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from cdrc.common.data.schema import AWVHistory, HealthPlan
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.hcpcs_codes.models.baseline_llm.constants import HCPCS_AWV_ELIGIBILITY_PROMPT_VERSION
from cdrc.profees_coding.hcpcs_codes.models.utils import CONFIG, PROMPTS

from .hcpcs_code import _as_bool, _get_model_kwargs_and_possibly_change_seed

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


class PlanInterpretationOutput(BaseModel):
    medicare_part_b_likely: Optional[bool] = Field(None, description="Whether the plan likely represents active Medicare Part B coverage")
    evidence: List[str] = Field(default_factory=list, description="Short evidence snippets from the plan text")


class AWVEligibilityOutput(BaseModel):
    medicare_part_b_likely: Optional[bool]
    plan_active_on_encounter: Optional[bool]
    prior_ippe_exists: bool
    prior_awv_exists: bool
    latest_awv_claim_date: Optional[str]
    months_since_latest_awv: Optional[int]
    g0402_history_allows: bool
    g0438_history_allows: bool
    g0439_history_allows: bool
    hard_blocks: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    summary_for_classifier: str


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


def _determine_plan_active_on_encounter(plan_entries: Sequence[Mapping[str, Any]], encounter_day: Optional[date]) -> tuple[Optional[bool], List[str]]:
    if not plan_entries:
        return None, []

    evidence: List[str] = []
    saw_explicit_active = False
    saw_explicit_inactive = False
    any_active = False
    any_inactive = False

    for plan in plan_entries:
        plan_name = str(plan.get("health_plan") or "").strip()
        status_value = str(plan.get("active_status") or plan.get("health_plan_status") or "").strip().lower()
        effective_day = _parse_date(plan.get("health_plan_effective_date"))

        if status_value:
            if any(re.search(rf"\b{re.escape(term)}\b", status_value) for term in INACTIVE_STATUS_TERMS):
                saw_explicit_inactive = True
                any_inactive = True
                evidence.append(f"Plan '{plan_name or 'unknown'}' marked inactive")
            elif any(re.search(rf"\b{re.escape(term)}\b", status_value) for term in ACTIVE_STATUS_TERMS):
                saw_explicit_active = True
                any_active = True
                evidence.append(f"Plan '{plan_name or 'unknown'}' marked active")

        if encounter_day and effective_day:
            if effective_day <= encounter_day:
                any_active = True
                evidence.append(f"Plan '{plan_name or 'unknown'}' effective on or before encounter date")
            else:
                any_inactive = True
                evidence.append(f"Plan '{plan_name or 'unknown'}' effective after encounter date")

    if saw_explicit_inactive and not saw_explicit_active:
        return False, evidence
    if saw_explicit_active and not saw_explicit_inactive:
        return True, evidence
    if any_active and not any_inactive:
        return True, evidence
    if any_inactive and not any_active:
        return False, evidence
    if saw_explicit_active or saw_explicit_inactive:
        return None, evidence
    return None, evidence


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


def _build_history_facts(raw_history: Any, encounter_day: Optional[date]) -> tuple[dict, List[str]]:
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

    prior_ippe_exists = any(code == IPPE_CODE for code, _ in prior_history_entries)
    awv_dates = [claim_day for code, claim_day in prior_history_entries if code in AWV_CODES]
    prior_awv_exists = bool(awv_dates)
    latest_awv_day = max(awv_dates) if awv_dates else None
    months_since_latest_awv = _months_between(latest_awv_day, encounter_day) if latest_awv_day and encounter_day else None

    if prior_ippe_exists:
        evidence.append("Prior G0402 found in AWV history")
    if latest_awv_day:
        evidence.append(f"Most recent prior AWV claim date: {latest_awv_day.isoformat()}")
    for code, claim_day in ignored_non_prior_entries:
        evidence.append(f"Ignored non-prior AWV history entry {code} on {claim_day.isoformat()}")

    facts = {
        "prior_ippe_exists": prior_ippe_exists,
        "prior_awv_exists": prior_awv_exists,
        "latest_awv_claim_date": latest_awv_day.isoformat() if latest_awv_day else None,
        "months_since_latest_awv": months_since_latest_awv,
        "g0402_history_allows": not prior_ippe_exists,
        "g0438_history_allows": not prior_awv_exists,
        "g0439_history_allows": bool(prior_awv_exists and months_since_latest_awv is not None and months_since_latest_awv >= 12),
    }
    return facts, evidence


def _build_hard_blocks(eligibility: AWVEligibilityOutput) -> List[str]:
    hard_blocks = list(eligibility.hard_blocks)

    if eligibility.medicare_part_b_likely is not True:
        hard_blocks.append("MEDICARE_PART_B_NOT_CONFIRMED")
    if eligibility.plan_active_on_encounter is not True:
        hard_blocks.append("PLAN_NOT_ACTIVE_ON_ENCOUNTER")
    if not eligibility.g0402_history_allows:
        hard_blocks.append("IPPE_ALREADY_BILLED")
    if eligibility.prior_awv_exists and not eligibility.g0439_history_allows:
        hard_blocks.append("SUBSEQUENT_AWV_TOO_SOON")

    return sorted(set(hard_blocks))


def _build_summary(eligibility: AWVEligibilityOutput) -> str:
    return (
        "AWV eligibility precheck: "
        f"medicare_part_b_likely={eligibility.medicare_part_b_likely}; "
        f"plan_active_on_encounter={eligibility.plan_active_on_encounter}; "
        f"prior_ippe_exists={eligibility.prior_ippe_exists}; "
        f"prior_awv_exists={eligibility.prior_awv_exists}; "
        f"latest_awv_claim_date={eligibility.latest_awv_claim_date}; "
        f"months_since_latest_awv={eligibility.months_since_latest_awv}; "
        f"g0402_history_allows={eligibility.g0402_history_allows}; "
        f"g0438_history_allows={eligibility.g0438_history_allows}; "
        f"g0439_history_allows={eligibility.g0439_history_allows}; "
        f"hard_blocks={','.join(eligibility.hard_blocks) if eligibility.hard_blocks else 'none'}."
    )


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
    plan_interpretation: Optional[PlanInterpretationOutput] = None,
) -> AWVEligibilityOutput:
    plan_entries = _coerce_health_plan_entries(raw_health_plan)
    encounter_day = _parse_date(encounter_date)
    plan_active_on_encounter, activity_evidence = _determine_plan_active_on_encounter(plan_entries, encounter_day)
    medicare_part_b_likely, plan_evidence, llm_needed = _determine_medicare_part_b_from_text(plan_entries)

    if medicare_part_b_likely is None and plan_interpretation is not None:
        medicare_part_b_likely = plan_interpretation.medicare_part_b_likely
        plan_evidence.extend(plan_interpretation.evidence)

    history_facts, history_evidence = _build_history_facts(raw_history, encounter_day)

    hard_blocks: List[str] = []
    if encounter_day is None:
        hard_blocks.append("ENCOUNTER_DATE_INVALID")
    if not plan_entries:
        hard_blocks.append("HEALTH_PLAN_MISSING")
    if llm_needed and plan_interpretation is None and medicare_part_b_likely is None:
        hard_blocks.append("MEDICARE_PART_B_AMBIGUOUS")

    eligibility = AWVEligibilityOutput(
        medicare_part_b_likely=medicare_part_b_likely,
        plan_active_on_encounter=plan_active_on_encounter,
        prior_ippe_exists=history_facts["prior_ippe_exists"],
        prior_awv_exists=history_facts["prior_awv_exists"],
        latest_awv_claim_date=history_facts["latest_awv_claim_date"],
        months_since_latest_awv=history_facts["months_since_latest_awv"],
        g0402_history_allows=history_facts["g0402_history_allows"],
        g0438_history_allows=history_facts["g0438_history_allows"],
        g0439_history_allows=history_facts["g0439_history_allows"],
        hard_blocks=hard_blocks,
        evidence=activity_evidence + plan_evidence + history_evidence,
        summary_for_classifier="",
    )
    eligibility.hard_blocks = _build_hard_blocks(eligibility)
    eligibility.summary_for_classifier = _build_summary(eligibility)
    return eligibility


@track_tokens("model_usage")
def awv_eligibility_precheck(state: Mapping[str, Any], config: RunnableConfig) -> dict:
    enable_prompt_cache_layout = _as_bool(config["configurable"].get("enable_prompt_cache_layout", True), default=True)
    plan_entries = _coerce_health_plan_entries(state.get("health_plan"))
    medicare_part_b_likely, _, llm_needed = _determine_medicare_part_b_from_text(plan_entries)

    plan_interpretation = None
    if llm_needed and medicare_part_b_likely is None:
        plan_interpretation = _run_plan_interpretation_llm(plan_entries, config, enable_prompt_cache_layout)

    eligibility = _build_awv_eligibility(
        raw_health_plan=state.get("health_plan"),
        raw_history=state.get("awv_history"),
        encounter_date=state.get("encounter_date"),
        plan_interpretation=plan_interpretation,
    )

    return {
        "awv_eligibility": eligibility.model_dump(),
    }


def build_awv_eligibility_subgraph(args: Namespace):
    return awv_eligibility_precheck
