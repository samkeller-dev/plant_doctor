"""Build the persistent Chroma index from the corpus markdown files.

Usage:
    python scripts/ingest.py
    docker compose run --rm app python scripts/ingest.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.rag import RagConfig

log = logging.getLogger("ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    cfg = RagConfig.from_env()
    log.info("Corpus dir:      %s", cfg.corpus_dir)
    log.info("Chroma dir:      %s", cfg.chroma_dir)
    log.info("Collection:      %s", cfg.chroma_collection)
    log.info("Embed model:     %s @ %s", cfg.embed_model, cfg.ollama_base_url)

    if not cfg.corpus_dir.exists():
        log.error("Corpus directory not found: %s", cfg.corpus_dir)
        return 1

    md_files = [p for p in cfg.corpus_dir.glob("*.md") if p.name.lower() != "readme.md"]
    if not md_files:
        log.error("No markdown files in corpus dir.")
        return 1
    log.info("Found %d corpus files: %s", len(md_files), [p.name for p in md_files])

    Settings.embed_model = OllamaEmbedding(
        model_name=cfg.embed_model,
        base_url=cfg.ollama_base_url,
    )

    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    # Recreate the collection so re-ingests are deterministic and old chunks
    # do not linger after corpus edits.
    try:
        client.delete_collection(cfg.chroma_collection)
        log.info("Deleted existing collection (rebuilding clean)")
    except Exception:
        pass
    collection = client.create_collection(cfg.chroma_collection)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = SimpleDirectoryReader(
        input_files=[str(p) for p in md_files],
        filename_as_id=True,
    ).load_data()
    # Preserve the original filename in node metadata so the LLM has a
    # citation-grade source identifier in every retrieved chunk.
    for d in docs:
        src = Path(d.metadata.get("file_path", d.metadata.get("file_name", ""))).name
        d.metadata["source"] = src

    log.info("Embedding %d documents…", len(docs))
    VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    log.info("Index built. Collection size: %d", collection.count())
    return 0


if __name__ == "__main__":
    sys.exit(main())
