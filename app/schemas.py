from typing import Literal

from pydantic import BaseModel, Field


class CitedEvidence(BaseModel):
    source: str = Field(..., description="Corpus filename, e.g. 'yellowing_leaves.md'.")
    quote: str = Field(..., min_length=1)
    relevance: Literal["primary", "supporting"]


class RecommendedAction(BaseModel):
    action: str = Field(..., min_length=1)
    urgency: Literal["immediate", "this_week", "monitor"]
    rationale: str = Field(..., min_length=1)


class Diagnosis(BaseModel):
    diagnosis: str = Field(..., min_length=1)
    confidence: Literal["low", "medium", "high"]
    evidence_cited: list[CitedEvidence] = Field(..., min_length=1)
    recommended_actions: list[RecommendedAction] = Field(..., min_length=1)
    differential_diagnoses: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=2000)
    plant_type: str | None = Field(default=None, max_length=120)


class ErrorResponse(BaseModel):
    error: Literal[
        "prompt_injection",
        "off_topic",
        "invalid_citation",
        "model_invalid_json",
        "internal_error",
    ]
    message: str
    detail: dict | None = None
