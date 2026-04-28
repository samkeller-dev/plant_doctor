"""Plant Doctor API.

POST /ask    Diagnose a houseplant problem.
GET  /health Liveness check.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .guardrails import (
    check_on_topic,
    detect_injection,
    validate_citations,
)
from .prompts import SYSTEM_PROMPT, build_query_prompt
from .rag import RagEngine, RagConfig, build_engine
from .schemas import AskRequest, Diagnosis, ErrorResponse

log = logging.getLogger("plant_doctor")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = RagConfig.from_env()
    log.info("Building RAG engine: llm=%s embed=%s", cfg.llm_model, cfg.embed_model)
    app.state.engine = build_engine(cfg)
    log.info(
        "RAG engine ready: %d corpus files, centroid dim=%d, threshold=%.2f",
        len(app.state.engine.corpus_files),
        len(app.state.engine.centroid),
        cfg.offtopic_threshold,
    )
    yield


app = FastAPI(title="Plant Doctor", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    engine: RagEngine | None = getattr(app.state, "engine", None)
    return {
        "status": "ok" if engine else "starting",
        "corpus_files": sorted(engine.corpus_files) if engine else [],
    }


def _err(status: int, error: str, message: str, detail: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=error, message=message, detail=detail).model_dump()
    return JSONResponse(status_code=status, content=body)


def _parse_diagnosis(raw: str) -> Diagnosis:
    """Attempts to parse a Diagnosis from the model's raw text. Tolerates a
    single leading/trailing fence even though the prompt forbids them."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    return Diagnosis.model_validate_json(text)


@app.post("/ask")
def ask(req: AskRequest):
    engine: RagEngine = app.state.engine
    cfg = engine.config

    # Layer 1 — prompt injection.
    hit = detect_injection(req.description)
    if hit:
        return _err(
            400,
            "prompt_injection",
            "Input rejected: looks like a prompt-injection attempt.",
            detail={"matched_pattern": hit.pattern, "matched_text": hit.match},
        )

    # Layer 2 — off-topic.
    topic = check_on_topic(
        req.description,
        engine.centroid,
        engine.embed,
        cfg.offtopic_threshold,
    )
    if not topic.on_topic:
        return _err(
            400,
            "off_topic",
            "Input rejected: not a houseplant problem.",
            detail={
                "similarity": round(topic.similarity, 4),
                "threshold": topic.threshold,
            },
        )

    # Run the model.
    user_prompt = build_query_prompt(req.description, req.plant_type)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    raw = engine.query(full_prompt)

    try:
        diag = _parse_diagnosis(raw)
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        return _err(
            502,
            "model_invalid_json",
            "Model output did not conform to the Diagnosis schema.",
            detail={"raw_response": raw[:2000], "parse_error": str(e)[:500]},
        )

    # Layer 3 — citation validation, with one re-prompt on violation.
    violation = validate_citations(diag, engine.corpus_files)
    if violation:
        log.warning("Citation violation, re-prompting once: %s", violation.bad_sources)
        retry_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Your previous response cited sources that do not exist in the "
            f"corpus: {violation.bad_sources}. The ONLY allowed source "
            f"filenames are: {sorted(engine.corpus_files)}. "
            f"Re-answer using only those sources.\n\n{user_prompt}"
        )
        raw2 = engine.query(retry_prompt)
        try:
            diag = _parse_diagnosis(raw2)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            return _err(
                502,
                "model_invalid_json",
                "Re-prompt also failed schema validation.",
                detail={"raw_response": raw2[:2000], "parse_error": str(e)[:500]},
            )
        violation2 = validate_citations(diag, engine.corpus_files)
        if violation2:
            return _err(
                502,
                "invalid_citation",
                "Model cited sources that do not exist after one re-prompt.",
                detail={
                    "bad_sources": violation2.bad_sources,
                    "raw_response": raw2[:2000],
                },
            )

    return diag
