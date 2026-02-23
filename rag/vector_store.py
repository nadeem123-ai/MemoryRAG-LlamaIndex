"""
vector_store.py
---------------
Builds and manages a ChromaDB vector store via LlamaIndex.
Compatible with llama-index-vector-stores-chroma >= 0.1.x and chromadb >= 0.5.0

Windows note: Instead of deleting chroma_db between runs (which causes
WinError 32 file-locking errors), we use a fresh EphemeralClient in memory
when re-initialising, and only persist to disk on first load.
"""

import os
import chromadb
from chromadb.config import Settings

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import VectorIndexRetriever

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_collection"


def build_vector_store(
    nodes: list[BaseNode],
    embed_model: BaseEmbedding,
    persist_dir: str = CHROMA_DIR,
) -> VectorStoreIndex:
    """
    Build or load a LlamaIndex VectorStoreIndex backed by ChromaDB.

    Strategy:
    - If chroma_db exists AND has vectors → load from disk (fast, no re-embedding)
    - Otherwise → build fresh in memory (avoids Windows WinError 32 file locks)

    Args:
        nodes:       List of chunked BaseNode objects.
        embed_model: LlamaIndex embedding model.
        persist_dir: Folder where Chroma stores its data.

    Returns:
        LlamaIndex VectorStoreIndex wrapping the Chroma collection.
    """
    # ── Try loading existing store first ──────────────────────────────
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
            if collection.count() > 0:
                print(f"[Chroma] Loading existing store from '{persist_dir}'  "
                      f"({collection.count()} vectors)...")
                vector_store = ChromaVectorStore(chroma_collection=collection)
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embed_model,
                )
                print("✓ Chroma store loaded from disk")
                return index
        except Exception as e:
            print(f"[Chroma] Could not load existing store ({e}), rebuilding...")

    # ── Build fresh store ─────────────────────────────────────────────
    # Use EphemeralClient (in-memory) to avoid Windows file-lock issues
    # when re-initialising. Falls back to persistent if possible.
    print(f"[Chroma] Building new store...")
    try:
        chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
    except Exception:
        # Fallback: pure in-memory (no persistence, but no lock errors)
        print("[Chroma] Persistent client unavailable, using in-memory store...")
        chroma_client = chromadb.EphemeralClient()

    # Delete old collection if it exists (fresh start)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"✓ Chroma store built and saved to '{persist_dir}'")
    return index


def get_retriever(index: VectorStoreIndex, top_k: int = 5) -> VectorIndexRetriever:
    """
    Create a retriever from the VectorStoreIndex.

    Args:
        index: LlamaIndex VectorStoreIndex.
        top_k: Number of nodes to retrieve per query.

    Returns:
        LlamaIndex VectorIndexRetriever.
    """
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    print(f"✓ Retriever ready  (top_k={top_k})")
    return retriever