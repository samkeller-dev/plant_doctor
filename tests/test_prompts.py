"""Unit tests for prompt construction."""

from app.prompts import SYSTEM_PROMPT, build_query_prompt


def test_system_prompt_mentions_schema_keys():
    """If the schema keys drift, the prompt must drift with them — fail loudly."""
    for key in [
        "diagnosis",
        "confidence",
        "evidence_cited",
        "recommended_actions",
        "differential_diagnoses",
        "caveats",
        "primary",
        "supporting",
        "immediate",
        "this_week",
        "monitor",
        "low",
        "medium",
        "high",
    ]:
        assert key in SYSTEM_PROMPT, f"system prompt missing schema key: {key!r}"


def test_system_prompt_forbids_code_fences():
    assert "code fences" in SYSTEM_PROMPT.lower() or "code fence" in SYSTEM_PROMPT.lower()


def test_system_prompt_restricts_to_houseplants():
    assert "houseplant" in SYSTEM_PROMPT.lower()


def test_build_query_prompt_includes_user_text_in_tags():
    out = build_query_prompt("my monstera is sad", None)
    assert "<user_description>" in out
    assert "</user_description>" in out
    assert "my monstera is sad" in out


def test_build_query_prompt_includes_plant_type_when_given():
    out = build_query_prompt("yellow leaves", "Monstera deliciosa")
    assert "Monstera deliciosa" in out
    assert "may be wrong" in out  # the hedge that prevents the LLM from over-trusting


def test_build_query_prompt_omits_plant_block_when_none():
    out = build_query_prompt("yellow leaves", None)
    assert "Monstera" not in out
    # The "Plant (user-supplied" preamble should not appear when no type given
    assert "Plant (user-supplied" not in out


def test_build_query_prompt_signals_user_data_is_data_not_instructions():
    """The wrapper must explicitly frame the user content as data so the LLM
    is less likely to follow embedded instructions."""
    out = build_query_prompt("foo", None)
    assert "USER DATA" in out or "user data" in out.lower()
    assert "never instructions" in out.lower() or "not instructions" in out.lower()


def test_build_query_prompt_does_not_leak_system_role_words():
    """The wrapper should not introduce 'system:' or similar tokens that would
    confuse the LLM about role boundaries."""
    out = build_query_prompt("hello", None)
    # Allow 'system' as a noun (e.g., 'plant doctor system') but not as a role marker
    assert "\nsystem:" not in out.lower()
    assert "\nassistant:" not in out.lower()
