"""
pipeline.py
-----------
Orchestrates the full LlamaIndex RAG pipeline with conversation memory.
"""

from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage

from rag.loader import load_pdfs
from rag.splitter import split_documents
from rag.embedder import get_embeddings
from rag.vector_store import build_vector_store, get_retriever
from rag.llm import get_llm


class RAGPipeline:
    """
    LlamaIndex-powered RAG pipeline with conversation memory.

    Supports single PDF, multiple PDFs, or an entire folder:
        pipeline = RAGPipeline(pdf_path="docs/resume.pdf")
        pipeline = RAGPipeline(pdf_path=["docs/a.pdf", "docs/b.pdf"])
        pipeline = RAGPipeline(pdf_path="docs/")
    """

    def __init__(
        self,
        pdf_path: str | list[str],
        chunk_size: int = 1000,
        overlap: int = 150,
        top_k: int = 5,
        provider: str = "ollama",
        model: str = "mistral",
        temperature: float = 0.0,
        persist_dir: str = "chroma_db",
    ):
        print("=" * 70)
        print("Initialising LlamaIndex RAG Pipeline")
        print("=" * 70)

        print("\n[1/5] Loading PDF(s)...")
        docs = load_pdfs(pdf_path)

        print("\n[2/5] Splitting into chunks...")
        nodes = split_documents(docs, chunk_size=chunk_size, overlap=overlap)

        print("\n[3/5] Loading embeddings...")
        embed_model = get_embeddings()
        Settings.embed_model = embed_model

        print("\n[4/5] Building vector store...")
        index = build_vector_store(nodes, embed_model, persist_dir)
        self.retriever = get_retriever(index, top_k=top_k)

        print("\n[5/5] Setting up LLM and memory...")
        llm = get_llm(provider=provider, model=model, temperature=temperature)
        Settings.llm = llm

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=self.retriever,
            memory=self.memory,
            llm=llm,
            verbose=False,
        )

        print("\n✓ Pipeline ready!\n")

    def ask(self, question: str) -> dict:
        """
        Ask a question. Returns a dict with answer + sources.

        Returns:
            {
                "answer": str,
                "sources": [{"file": str, "page": str, "preview": str}, ...]
            }
        """
        response = self.chat_engine.chat(question)

        raw_sources = response.source_nodes if hasattr(response, "source_nodes") else []
        sources = []
        for node in raw_sources:
            sources.append({
                "file": node.metadata.get("file_name", "unknown"),
                "page": node.metadata.get("page_label", node.metadata.get("page", "?")),
                "preview": node.get_content().replace("\n", " ")[:120],
            })

        return {
            "answer": str(response),
            "sources": sources,
        }

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory.reset()

    def get_history(self) -> list[ChatMessage]:
        """Return the full conversation history."""
        return self.memory.get_all()