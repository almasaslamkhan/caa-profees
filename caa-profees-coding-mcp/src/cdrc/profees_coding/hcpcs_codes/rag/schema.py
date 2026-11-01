from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class RagEncounterPayloadModel(BaseModel):
    example_id: str | int | None = None
    text: str = ""
    patient_sex: str | None = None
    patient_age: int | None = None
    orders_display: str | None = None
    health_plan: list[dict[str, Any]] | None = None
    awv_history: list[dict[str, Any]] | None = None
    encounter_date: str | None = None
    encounter_metadata: dict[str, Any] | None = None
    all_patient_problems: list[dict[str, Any]] | None = None
    additional_input: dict[str, Any] | None = None


class RetrievalRequestModel(BaseModel):
    domain: str = "hcpcs"
    task: str = "retrieval"
    feature_name: str
    target_code: str | None = None
    encounter: RagEncounterPayloadModel
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_target_code(self) -> "RetrievalRequestModel":
        if not self.target_code:
            self.target_code = self.feature_name
        return self


class SummaryBundleModel(BaseModel):
    summary_compact: str = ""
    summary_full: str = ""
    source: str = "unknown"
    sections: dict[str, list[str]] = Field(default_factory=dict)


class SharedFeatureModel(BaseModel):
    section: str = ""
    section_label: str = ""
    score: float = 0.0
    query_evidence: str = ""
    candidate_evidence: str = ""
    shared_terms: list[str] = Field(default_factory=list)
    shared_concepts: list[str] = Field(default_factory=list)
    reasoning: str = ""


class RetrievedEncounterModel(BaseModel):
    rank: int
    encounter_id: str | None = None
    example_id: str | None = None
    target_code: str = ""
    label: str | None = None
    encounter_source: str | None = None
    similarity_score: float = 0.0
    encounter_summary: str = ""
    teaching_justification: str = ""
    matched_signals: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Compatibility fields preserved for one migration cycle.
    dense_score: float = 0.0
    feature_score: float = 0.0
    final_score: float = 0.0
    summary_compact: str = ""
    selection_reason: str = ""
    decision_contribution: str = ""
    shared_features: list[SharedFeatureModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def populate_compatibility_fields(self) -> "RetrievedEncounterModel":
        if not self.summary_compact:
            self.summary_compact = self.encounter_summary
        if not self.encounter_summary:
            self.encounter_summary = self.summary_compact
        if not self.teaching_justification:
            self.teaching_justification = self.decision_contribution
        if not self.decision_contribution:
            self.decision_contribution = self.teaching_justification
        if not self.final_score:
            self.final_score = self.similarity_score
        if not self.similarity_score:
            self.similarity_score = self.final_score
        if not self.selection_reason:
            if self.matched_signals:
                self.selection_reason = "Matched signals: " + "; ".join(self.matched_signals[:3])
            else:
                self.selection_reason = "Selected by retrieval similarity."
        if not self.shared_features and self.matched_signals:
            self.shared_features = [
                SharedFeatureModel(
                    section="signal",
                    section_label="signal",
                    score=0.0,
                    query_evidence=signal,
                    candidate_evidence=signal,
                    reasoning="matched retrieval signal",
                )
                for signal in self.matched_signals
            ]
        return self


class PromptInjectionExampleModel(BaseModel):
    label: str | None = None
    encounter_summary: str = ""
    teaching_justification: str = ""
    matched_signals: list[str] = Field(default_factory=list)


class PromptInjectionPayloadModel(BaseModel):
    target_code: str
    guidance: str = ""
    examples: list[PromptInjectionExampleModel] = Field(default_factory=list)


class RetrievalDiagnosticsModel(BaseModel):
    pool_size: int = 0
    selected_count: int = 0
    per_label_limit: int | None = None
    positive_example_limit: int | None = None
    negative_example_limit: int | None = None
    dense_score_min: float | None = None
    dense_score_max: float | None = None
    artifact_build_id: str | None = None
    artifact_source: str | None = None
    embedding_model: str | None = None


class RetrievalResponseModel(BaseModel):
    feature_name: str
    feature_code: str
    target_code: str | None = None
    encounter_route: str = "ambulatory"
    retrieval_status: str = "ok"
    retrieval_status_reason: str | None = None
    query_summary: SummaryBundleModel
    query_signals: list[str] = Field(default_factory=list)
    retrieved_encounters: list[RetrievedEncounterModel] = Field(default_factory=list)
    label_distribution: dict[str, int] = Field(default_factory=dict)
    diagnostics: RetrievalDiagnosticsModel = Field(default_factory=RetrievalDiagnosticsModel)
    prompt_payload: PromptInjectionPayloadModel | None = None

    @model_validator(mode="after")
    def sync_target_code(self) -> "RetrievalResponseModel":
        if not self.target_code:
            self.target_code = self.feature_code
        if self.prompt_payload is None:
            self.prompt_payload = PromptInjectionPayloadModel(target_code=self.target_code or self.feature_code)
        return self
