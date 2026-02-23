"""
embedder.py
-----------
Creates embeddings using LlamaIndex's HuggingFaceEmbedding wrapper.
Compatible with llama-index-embeddings-huggingface >= 0.2.x
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # fast, 384-dim — same size as MiniLM


def get_embeddings(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbedding:
    """
    Load and return a LlamaIndex HuggingFaceEmbedding object.

    Note: LlamaIndex uses BAAI/bge-small-en-v1.5 as default which is
    slightly better than all-MiniLM-L6-v2 on retrieval benchmarks.
    You can pass model_name="sentence-transformers/all-MiniLM-L6-v2"
    to keep the exact same model as the LangChain version.

    Args:
        model_name: HuggingFace model name.

    Returns:
        HuggingFaceEmbedding instance ready to use.
    """
    print(f"Loading embedding model '{model_name}'...")
    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        device="cpu",
        embed_batch_size=32,
    )
    print("✓ Embedding model loaded")
    return embed_model