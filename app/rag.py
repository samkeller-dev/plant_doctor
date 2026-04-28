"""RAG wrapper around LlamaIndex + Chroma + Ollama (Mistral 7B Instruct).

Heavy imports are deferred into functions so guardrails/schema unit tests do
not require the LlamaIndex / Chroma / Ollama stack to import this package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RagConfig:
    ollama_base_url: str
    llm_model: str
    embed_model: str
    chroma_dir: Path
    chroma_collection: str
    corpus_dir: Path
    offtopic_threshold: float

    @classmethod
    def from_env(cls) -> "RagConfig":
        return cls(
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            llm_model=os.environ.get("OLLAMA_LLM_MODEL", "mistral:7b-instruct"),
            embed_model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            chroma_dir=Path(os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")),
            chroma_collection=os.environ.get("CHROMA_COLLECTION", "plant_doctor"),
            corpus_dir=Path(os.environ.get("CORPUS_DIR", "./corpus")),
            offtopic_threshold=float(os.environ.get("OFFTOPIC_SIM_THRESHOLD", "0.35")),
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class RagEngine:
    """Holds the live query engine, embedder, corpus filename set, and centroid."""

    query: Callable[[str], str]
    embed: Callable[[str], list[float]]
    corpus_files: set[str]
    centroid: list[float]
    config: RagConfig


def corpus_filenames(corpus_dir: Path) -> set[str]:
    return {p.name for p in corpus_dir.glob("*.md") if p.name.lower() != "readme.md"}


def build_engine(config: RagConfig | None = None) -> RagEngine:
    """Constructs the RAG engine. Requires Chroma to have been populated by
    scripts/ingest.py first."""
    from llama_index.core import Settings, StorageContext, VectorStoreIndex
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    cfg = config or RagConfig.from_env()

    embed_model = OllamaEmbedding(
        model_name=cfg.embed_model,
        base_url=cfg.ollama_base_url,
    )
    llm = Ollama(
        model=cfg.llm_model,
        base_url=cfg.ollama_base_url,
        # Generous timeout: cold-loading mistral:7b-instruct on CPU can
        # take a couple of minutes the first time before the model is
        # resident in memory. Subsequent calls finish in seconds.
        request_timeout=600.0,
        json_mode=True,
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    collection = client.get_collection(cfg.chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    query_engine = index.as_query_engine(
        similarity_top_k=4,
        response_mode="compact",
    )

    files = corpus_filenames(cfg.corpus_dir)
    centroid_vec = _compute_centroid_from_chroma(collection)

    def query_fn(prompt: str) -> str:
        resp = query_engine.query(prompt)
        return str(resp)

    def embed_fn(text: str) -> list[float]:
        return embed_model.get_query_embedding(text)

    return RagEngine(
        query=query_fn,
        embed=embed_fn,
        corpus_files=files,
        centroid=centroid_vec,
        config=cfg,
    )


def _compute_centroid_from_chroma(collection) -> list[float]:
    """Pulls all embeddings already stored in Chroma and averages them.

    Reusing the stored vectors avoids re-embedding the entire corpus at
    startup and guarantees the centroid reflects exactly what was indexed.
    """
    from .guardrails import centroid

    data = collection.get(include=["embeddings"])
    # Chroma returns a numpy ndarray (shape: n_docs × dim) when embeddings are
    # present, or None when the field wasn't requested. Avoid `or []` here
    # because `bool(ndarray)` raises ValueError for multi-element arrays.
    embeddings = data.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        raise RuntimeError(
            "Chroma collection is empty. Run `python -m scripts.ingest` first."
        )
    return centroid([list(v) for v in embeddings])
