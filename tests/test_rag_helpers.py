"""Tests for the lightweight pieces of app.rag — config loading and corpus
filename discovery — that don't require a live Ollama or Chroma instance."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag import RagConfig, corpus_filenames


REPO_CORPUS = Path(__file__).resolve().parent.parent / "corpus"


# ---------------------------------------------------------------------------
# RagConfig
# ---------------------------------------------------------------------------


def test_ragconfig_defaults_when_env_clean(monkeypatch):
    for key in [
        "OLLAMA_BASE_URL",
        "OLLAMA_LLM_MODEL",
        "OLLAMA_EMBED_MODEL",
        "CHROMA_PERSIST_DIR",
        "CHROMA_COLLECTION",
        "CORPUS_DIR",
        "OFFTOPIC_SIM_THRESHOLD",
    ]:
        monkeypatch.delenv(key, raising=False)
    cfg = RagConfig.from_env()
    assert cfg.ollama_base_url == "http://localhost:11434"
    assert cfg.llm_model == "mistral:7b-instruct"
    assert cfg.embed_model == "nomic-embed-text"
    assert cfg.chroma_collection == "plant_doctor"
    assert cfg.offtopic_threshold == pytest.approx(0.35)


def test_ragconfig_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_LLM_MODEL", "llama3:8b")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "all-minilm")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/chroma")
    monkeypatch.setenv("CHROMA_COLLECTION", "test_coll")
    monkeypatch.setenv("CORPUS_DIR", "/tmp/corpus")
    monkeypatch.setenv("OFFTOPIC_SIM_THRESHOLD", "0.7")
    cfg = RagConfig.from_env()
    assert cfg.ollama_base_url == "http://ollama:11434"
    assert cfg.llm_model == "llama3:8b"
    assert cfg.embed_model == "all-minilm"
    assert str(cfg.chroma_dir) in ("/tmp/chroma", "\\tmp\\chroma")
    assert cfg.chroma_collection == "test_coll"
    assert cfg.offtopic_threshold == pytest.approx(0.7)


def test_ragconfig_threshold_invalid_env_raises(monkeypatch):
    monkeypatch.setenv("OFFTOPIC_SIM_THRESHOLD", "not-a-number")
    with pytest.raises(ValueError):
        RagConfig.from_env()


def test_ragconfig_is_frozen():
    cfg = RagConfig.from_env()
    with pytest.raises((AttributeError, TypeError)):  # frozen dataclass
        cfg.llm_model = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# corpus_filenames
# ---------------------------------------------------------------------------


def test_corpus_filenames_returns_md_files():
    files = corpus_filenames(REPO_CORPUS)
    assert len(files) >= 6
    for f in files:
        assert f.endswith(".md")


def test_corpus_filenames_excludes_readme():
    files = corpus_filenames(REPO_CORPUS)
    assert "README.md" not in files
    assert "readme.md" not in files


def test_corpus_filenames_includes_known_topics():
    files = corpus_filenames(REPO_CORPUS)
    expected = {
        "yellowing_leaves.md",
        "root_rot_overwatering.md",
        "underwatering.md",
        "spider_mites.md",
        "fungus_gnats.md",
        "scale_insects.md",
        "light_requirements.md",
        "repotting.md",
    }
    assert expected.issubset(files), f"missing: {expected - files}"


def test_corpus_filenames_empty_dir(tmp_path):
    assert corpus_filenames(tmp_path) == set()


def test_corpus_filenames_ignores_non_markdown(tmp_path):
    (tmp_path / "real.md").write_text("# real")
    (tmp_path / "notes.txt").write_text("not markdown")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    assert corpus_filenames(tmp_path) == {"real.md"}
