from __future__ import annotations

import re
from typing import Any, Mapping

from .profiles import RetrievalProfile
from .summaries import parse_summary_sections

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "visit",
    "with",
}

CONCEPT_MARKERS = {
    "longitudinal_relationship": (
        "longitudinal",
        "established patient-physician relationship",
        "continuing focal point",
        "ongoing care",
        "primary care",
        "transition of care",
        "establish care",
    ),
    "follow_up": (
        "follow-up",
        "follow up",
        "return to clinic",
        "rtc",
        "scheduled follow-up",
    ),
    "monitoring": (
        "monitor",
        "monitoring",
        "lab",
        "labs",
        "ordered",
        "orders",
        "check",
        "recheck",
    ),
    "medication_management": (
        "continue",
        "refill",
        "dose",
        "medication",
        "prescribed",
        "prescription",
        "reconcile",
    ),
}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _tokenize(text: str) -> set[str]:
    tokens = {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2}
    return {token for token in tokens if token not in STOPWORDS}


def _shared_concepts(query_item: str, candidate_item: str) -> list[str]:
    query_text = (query_item or "").lower()
    candidate_text = (candidate_item or "").lower()
    concepts: list[str] = []
    for label, markers in CONCEPT_MARKERS.items():
        if any(marker in query_text for marker in markers) and any(marker in candidate_text for marker in markers):
            concepts.append(label)
    return concepts


def _pair_score(query_item: str, candidate_item: str) -> tuple[float, list[str], list[str]]:
    query_tokens = _tokenize(query_item)
    candidate_tokens = _tokenize(candidate_item)
    shared_terms = sorted(query_tokens & candidate_tokens)
    shared_concepts = _shared_concepts(query_item, candidate_item)
    lexical_score = len(shared_terms) / max(len(query_tokens | candidate_tokens), 1)
    concept_score = 0.08 * len(shared_concepts)
    total_score = lexical_score + concept_score
    return total_score, shared_terms[:8], shared_concepts


def _truncate(text: str, limit: int = 180) -> str:
    value = _normalize_text(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def build_signals_from_sections(
    *,
    profile: RetrievalProfile,
    sections: Mapping[str, list[str]],
    orders_display: str | None = None,
    encounter_route: str | None = None,
) -> list[str]:
    signals: list[str] = []
    visit_type = sections.get("visit_type") or []
    for item in visit_type:
        value = _normalize_text(item)
        if value:
            signals.append(f"visit_type: {value}")

    for section in profile.signal_sections:
        for item in sections.get(section) or []:
            value = _normalize_text(item)
            if value:
                signals.append(value)

    if orders_display:
        order_signal = _normalize_text(orders_display)
        if order_signal:
            signals.append(f"orders: {_truncate(order_signal, limit=220)}")

    if encounter_route:
        signals.append(f"route: {encounter_route}")

    deduped: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        key = signal.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(signal)
    return deduped


def build_query_signals(
    *,
    profile: RetrievalProfile,
    query_summary: Mapping[str, Any],
    encounter,
    encounter_route: str,
) -> list[str]:
    sections = dict(query_summary.get("sections") or {})
    signals = build_signals_from_sections(
        profile=profile,
        sections=sections,
        orders_display=getattr(encounter, "orders_display", None),
        encounter_route=encounter_route,
    )
    if signals:
        return signals

    fallback_sections = parse_summary_sections(
        str(query_summary.get("summary_compact") or query_summary.get("summary_full") or getattr(encounter, "text", ""))
    )
    signals = build_signals_from_sections(
        profile=profile,
        sections=fallback_sections,
        orders_display=getattr(encounter, "orders_display", None),
        encounter_route=encounter_route,
    )
    if signals:
        return signals

    raw_text = _truncate(getattr(encounter, "text", ""), limit=240)
    return [raw_text] if raw_text else []


def _candidate_signals(candidate: Mapping[str, Any], profile: RetrievalProfile) -> list[str]:
    signals = list(candidate.get("candidate_signals") or [])
    if signals:
        return [_normalize_text(signal) for signal in signals if _normalize_text(signal)]

    sections = parse_summary_sections(str(candidate.get("encounter_summary") or ""))
    return build_signals_from_sections(
        profile=profile,
        sections=sections,
        orders_display=None,
        encounter_route=None,
    )


def _score_candidate(
    *,
    query_signals: list[str],
    candidate_signals: list[str],
) -> tuple[float, list[dict[str, Any]], list[str]]:
    matches: list[dict[str, Any]] = []
    for query_signal in query_signals:
        for candidate_signal in candidate_signals:
            score, shared_terms, shared_concepts = _pair_score(query_signal, candidate_signal)
            if score <= 0:
                continue
            matches.append(
                {
                    "query_signal": _truncate(query_signal),
                    "candidate_signal": _truncate(candidate_signal),
                    "score": round(score, 4),
                    "shared_terms": shared_terms,
                    "shared_concepts": shared_concepts,
                }
            )

    matches.sort(key=lambda item: item["score"], reverse=True)
    top_matches = matches[:4]
    signal_score = round(sum(match["score"] for match in top_matches), 4)

    matched_signals: list[str] = []
    for match in top_matches:
        if match["shared_terms"]:
            matched_signals.append(", ".join(match["shared_terms"][:4]))
        elif match["shared_concepts"]:
            matched_signals.append(", ".join(match["shared_concepts"]))
        else:
            matched_signals.append(match["query_signal"])

    deduped: list[str] = []
    seen: set[str] = set()
    for signal in matched_signals:
        normalized = signal.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(signal)
    return signal_score, top_matches, deduped


def rerank_examples(
    *,
    candidates: list[dict[str, Any]],
    query_signals: list[str],
    profile: RetrievalProfile,
    max_examples: int,
    positive_limit: int,
    negative_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_signals = _candidate_signals(candidate, profile)
        signal_score, signal_matches, matched_signals = _score_candidate(
            query_signals=query_signals,
            candidate_signals=candidate_signals,
        )
        dense_score = float(candidate.get("dense_score") or 0.0)
        final_score = round(dense_score + signal_score, 4)
        label = profile.normalize_label(candidate.get("label"))
        teaching_justification = str(candidate.get("teaching_justification") or "").strip()
        if not teaching_justification:
            if label == profile.positive_label:
                teaching_justification = f"{profile.target_code}={label}: this retrieved encounter shows the target longitudinal-care pattern."
            elif label == profile.negative_label:
                teaching_justification = (
                    f"{profile.target_code}={label}: this retrieved encounter is a boundary case and should not receive the code."
                )
            else:
                teaching_justification = f"{profile.target_code} example for retrieval calibration."

        scored_candidates.append(
            {
                **candidate,
                "label": label,
                "similarity_score": final_score,
                "feature_score": signal_score,
                "matched_signals": matched_signals,
                "signal_matches": signal_matches,
                "selection_reason": ("Matched signals: " + "; ".join(matched_signals[:3]) if matched_signals else "Selected by dense similarity."),
                "teaching_justification": teaching_justification,
                "decision_contribution": teaching_justification,
                "dense_score": dense_score,
                "final_score": final_score,
            }
        )

    scored_candidates.sort(
        key=lambda item: (
            item.get("similarity_score", 0.0),
            item.get("feature_score", 0.0),
            item.get("dense_score", 0.0),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    seen_ids: set[tuple[str | None, str | None]] = set()
    label_counts: dict[str, int] = {}

    for candidate in scored_candidates:
        identity = (candidate.get("example_id"), candidate.get("encounter_id"))
        if identity in seen_ids:
            continue
        label = candidate.get("label") or "UNKNOWN"
        if label == profile.positive_label and label_counts.get(label, 0) >= positive_limit:
            continue
        if label == profile.negative_label and label_counts.get(label, 0) >= negative_limit:
            continue
        selected.append(candidate)
        seen_ids.add(identity)
        label_counts[label] = label_counts.get(label, 0) + 1
        if len(selected) >= max_examples:
            break

    if len(selected) < max_examples:
        for candidate in scored_candidates:
            identity = (candidate.get("example_id"), candidate.get("encounter_id"))
            if identity in seen_ids:
                continue
            selected.append(candidate)
            seen_ids.add(identity)
            label = candidate.get("label") or "UNKNOWN"
            label_counts[label] = label_counts.get(label, 0) + 1
            if len(selected) >= max_examples:
                break

    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank

    return selected, label_counts


__all__ = [
    "build_query_signals",
    "build_signals_from_sections",
    "rerank_examples",
]
