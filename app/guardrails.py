"""Three-layer guardrails for the plant_doctor RAG endpoint.

Layer 1 — input prompt-injection blocklist (regex).
Layer 2 — input off-topic detection via cosine similarity to corpus centroid.
Layer 3 — output citation validation against the real corpus filename set.

The off-topic detector takes an injected embedder callable so tests can run
without a live Ollama instance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .schemas import Diagnosis

# ---------------------------------------------------------------------------
# Layer 1: prompt-injection blocklist
# ---------------------------------------------------------------------------

# Each pattern is a compiled case-insensitive regex. The list is intentionally
# conservative — it targets well-known phrasings rather than aggressive
# fuzz-matching that would cause false positives on legitimate plant questions.
_TRAILING_NOUNS = r"(?:instructions?|prompts?|rules?|directives?|context|guidance|messages?)"
_ROLE_SWITCH_NOUNS = (
    r"(?:compiler|hacker|sql|shell|terminal|model|ai|assistant|jailbreak|"
    r"developer|root|admin|sudo|chatbot|llm|gpt|chatgpt|unrestricted)"
)

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # "ignore <previous|prior|earlier> <instructions|prompts|rules|...>"
    re.compile(rf"\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|earlier)\s+{_TRAILING_NOUNS}\b", re.I),
    # "ignore (everything|all|the) above" — strong injection signal even without trailing noun
    re.compile(r"\bignore\s+(?:everything\s+|all\s+|the\s+)?above\b", re.I),
    # Same two shapes for "disregard"
    re.compile(rf"\bdisregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|earlier)\s+{_TRAILING_NOUNS}\b", re.I),
    re.compile(r"\bdisregard\s+(?:everything\s+|all\s+|the\s+)?above\b", re.I),
    # Reveal-the-system-prompt variants
    re.compile(r"\b(?:reveal|show|print|repeat|leak|output|tell)\s+(?:me\s+)?(?:your|the)\s+system\s+prompt\b", re.I),
    # Role switch
    re.compile(r"\byou\s+are\s+now\b", re.I),
    # "act as <role-switch noun>" — narrowed to known attacker vocabulary so
    # innocent phrases like "act as a better plant parent" do not false-positive.
    re.compile(rf"\bact\s+as\s+(?:a\s+|an\s+|the\s+)?{_ROLE_SWITCH_NOUNS}\b", re.I),
    # Pretend (must be followed by "to be" / "you are" — "pretends to thrive" is fine)
    re.compile(r"\bpretend\s+(?:to\s+be|you\s+are)\b", re.I),
    # Role-marker spoofing (must start a line)
    re.compile(r"^\s*system\s*[:>]", re.I | re.M),
    re.compile(r"^\s*assistant\s*[:>]", re.I | re.M),
    # Chat-template special tokens
    re.compile(r"<\s*\|?\s*/?\s*(?:system|im_start|im_end)\s*\|?\s*>", re.I),
    # Jailbreak / DAN / developer-mode framing
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bDAN\s*(?:mode|prompt)\b"),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
]


@dataclass(frozen=True)
class InjectionHit:
    pattern: str
    match: str


def detect_injection(text: str) -> InjectionHit | None:
    """Returns the first matching injection pattern, or None."""
    for pat in _INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            return InjectionHit(pattern=pat.pattern, match=m.group(0))
    return None


# ---------------------------------------------------------------------------
# Layer 2: off-topic detection (cosine similarity to corpus centroid)
# ---------------------------------------------------------------------------

Embedder = Callable[[str], list[float]]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Embedding dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def centroid(vectors: Iterable[list[float]]) -> list[float]:
    vectors = list(vectors)
    if not vectors:
        raise ValueError("centroid() requires at least one vector")
    dim = len(vectors[0])
    sums = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            raise ValueError("All vectors must have the same dimensionality")
        for i, x in enumerate(v):
            sums[i] += x
    n = len(vectors)
    return [s / n for s in sums]


@dataclass(frozen=True)
class OffTopicResult:
    similarity: float
    threshold: float
    on_topic: bool


def check_on_topic(
    query: str,
    corpus_centroid: list[float],
    embed: Embedder,
    threshold: float,
) -> OffTopicResult:
    q_vec = embed(query)
    sim = _cosine(q_vec, corpus_centroid)
    return OffTopicResult(similarity=sim, threshold=threshold, on_topic=sim >= threshold)


# ---------------------------------------------------------------------------
# Layer 3: citation validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationViolation:
    bad_sources: list[str]


def validate_citations(diagnosis: Diagnosis, corpus_files: set[str]) -> CitationViolation | None:
    """Returns a violation listing any cited source not present in the corpus."""
    bad = [c.source for c in diagnosis.evidence_cited if c.source not in corpus_files]
    return CitationViolation(bad_sources=bad) if bad else None
