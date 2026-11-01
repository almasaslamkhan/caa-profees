from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalProfile:
    target_code: str
    description: str
    allowed_routes: tuple[str, ...]
    signal_sections: tuple[str, ...]
    section_weights: dict[str, float]
    positive_label: str = "YES"
    negative_label: str = "NO"
    max_examples: int = 4
    positive_example_limit: int = 2
    negative_example_limit: int = 2
    prompt_guidance: str = "Use retrieved examples as analogies only. Do not treat them as evidence from the current encounter."

    @property
    def name(self) -> str:
        return self.target_code

    @property
    def code(self) -> str:
        return self.target_code

    def normalize_label(self, value: Any) -> str:
        label = str(value or "").strip().upper()
        if label == self.positive_label:
            return self.positive_label
        if label == self.negative_label:
            return self.negative_label
        return label or "UNKNOWN"


G2211_PROFILE = RetrievalProfile(
    target_code="G2211",
    description="Longitudinal care retrieval for outpatient G2211 review.",
    allowed_routes=("ambulatory",),
    signal_sections=(
        "feature_evidence",
        "continuity",
        "chronic_conditions",
        "management_actions",
    ),
    section_weights={
        "feature_evidence": 0.45,
        "continuity": 0.25,
        "chronic_conditions": 0.20,
        "management_actions": 0.10,
    },
    prompt_guidance=(
        "Use these retrieved G2211 examples as analogies and boundary cases only. "
        "The current encounter's SOAP note and orders remain the only direct evidence."
    ),
)


_PROFILES = {
    G2211_PROFILE.target_code.upper(): G2211_PROFILE,
}


def resolve_retrieval_profile(
    *,
    feature_name: str | None = None,
    target_code: str | None = None,
    candidate_codes: Any = None,
) -> RetrievalProfile:
    resolved_name = target_code or feature_name
    if not resolved_name and candidate_codes:
        if isinstance(candidate_codes, str):
            resolved_name = candidate_codes.split(",")[0].strip()
        else:
            for value in candidate_codes:
                candidate = str(value).strip()
                if candidate:
                    resolved_name = candidate
                    break

    resolved_name = (resolved_name or G2211_PROFILE.target_code).strip().upper()
    if resolved_name not in _PROFILES:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unsupported retrieval feature '{resolved_name}'. Supported features: {supported}")
    return _PROFILES[resolved_name]


__all__ = [
    "G2211_PROFILE",
    "RetrievalProfile",
    "resolve_retrieval_profile",
]
