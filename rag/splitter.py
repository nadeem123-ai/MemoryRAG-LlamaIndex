"""
splitter.py
-----------
Splits LlamaIndex Documents into smaller overlapping nodes (chunks).
Compatible with llama-index-core >= 0.10.x
"""

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode


def split_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[BaseNode]:
    """
    Split a list of Documents into smaller nodes (chunks).

    LlamaIndex uses SentenceSplitter — it tries to split at sentence
    boundaries first, then falls back to character level.
    This mirrors LangChain's RecursiveCharacterTextSplitter behaviour.

    Args:
        docs:       List of LlamaIndex Document objects (from loader).
        chunk_size: Max tokens per chunk.
        overlap:    Tokens shared between consecutive chunks.

    Returns:
        List of BaseNode objects (chunks) with original metadata preserved.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    nodes = splitter.get_nodes_from_documents(docs)
    print(f"✓ Split into {len(nodes)} chunks  "
          f"(chunk_size={chunk_size}, overlap={overlap})")
    return nodes