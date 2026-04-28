"""Schema round-trip and validation tests for the Diagnosis model."""

import json

import pytest
from pydantic import ValidationError

from app.schemas import (
    AskRequest,
    CitedEvidence,
    Diagnosis,
    ErrorResponse,
    RecommendedAction,
)

VALID_DIAG = {
    "diagnosis": "Likely overwatering progressing toward root rot.",
    "confidence": "medium",
    "evidence_cited": [
        {
            "source": "root_rot_overwatering.md",
            "quote": "Lower leaves yellow and drop quickly.",
            "relevance": "primary",
        }
    ],
    "recommended_actions": [
        {
            "action": "Stop watering and check root health.",
            "urgency": "immediate",
            "rationale": "Saturated soil is the proximate trigger.",
        }
    ],
    "differential_diagnoses": ["Natural senescence"],
    "caveats": ["Root inspection needed for confirmation."],
}


def test_diagnosis_round_trip():
    diag = Diagnosis.model_validate(VALID_DIAG)
    assert diag.confidence == "medium"
    assert diag.evidence_cited[0].source == "root_rot_overwatering.md"
    # JSON round-trip preserves shape
    re_parsed = Diagnosis.model_validate_json(diag.model_dump_json())
    assert re_parsed == diag


def test_diagnosis_requires_at_least_one_evidence():
    bad = {**VALID_DIAG, "evidence_cited": []}
    with pytest.raises(ValidationError):
        Diagnosis.model_validate(bad)


def test_diagnosis_requires_at_least_one_action():
    bad = {**VALID_DIAG, "recommended_actions": []}
    with pytest.raises(ValidationError):
        Diagnosis.model_validate(bad)


def test_diagnosis_rejects_invalid_confidence():
    bad = {**VALID_DIAG, "confidence": "very_high"}
    with pytest.raises(ValidationError):
        Diagnosis.model_validate(bad)


def test_diagnosis_rejects_invalid_urgency():
    bad = {
        **VALID_DIAG,
        "recommended_actions": [
            {"action": "do thing", "urgency": "soon-ish", "rationale": "x"}
        ],
    }
    with pytest.raises(ValidationError):
        Diagnosis.model_validate(bad)


def test_diagnosis_rejects_invalid_relevance():
    bad = {
        **VALID_DIAG,
        "evidence_cited": [
            {"source": "x.md", "quote": "y", "relevance": "vibes"}
        ],
    }
    with pytest.raises(ValidationError):
        Diagnosis.model_validate(bad)


def test_ask_request_min_max():
    AskRequest.model_validate({"description": "abc"})
    with pytest.raises(ValidationError):
        AskRequest.model_validate({"description": "ab"})
    with pytest.raises(ValidationError):
        AskRequest.model_validate({"description": "x" * 2001})


def test_error_response_locked_to_known_codes():
    ErrorResponse(error="off_topic", message="nope")
    with pytest.raises(ValidationError):
        ErrorResponse(error="banana", message="x")


def test_diagnosis_partial_components_compose():
    ev = CitedEvidence(source="a.md", quote="b", relevance="primary")
    act = RecommendedAction(action="x", urgency="monitor", rationale="y")
    diag = Diagnosis(
        diagnosis="d",
        confidence="low",
        evidence_cited=[ev],
        recommended_actions=[act],
    )
    payload = json.loads(diag.model_dump_json())
    assert payload["differential_diagnoses"] == []
    assert payload["caveats"] == []
