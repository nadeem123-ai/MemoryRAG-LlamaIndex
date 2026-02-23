"""
loader.py
---------
Loads one or more PDF documents using LlamaIndex's SimpleDirectoryReader.
Compatible with llama-index >= 0.10.x

Supports:
    - Single PDF  : load_pdfs("docs/resume.pdf")
    - Multiple PDFs : load_pdfs(["docs/resume.pdf", "docs/report.pdf"])
    - Entire folder : load_pdfs("docs/")   ← loads every PDF in the folder
"""

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
import os


def load_pdfs(pdf_input: str | list[str]) -> list[Document]:
    """
    Load one or more PDF files and return a list of LlamaIndex Document objects.

    Each Document contains:
        - text     : the text content
        - metadata : { file_name, page_label, file_path, ... }
                     ← file_name lets you trace which PDF each chunk came from

    Args:
        pdf_input: One of:
            - str path to a single PDF      → "docs/resume.pdf"
            - str path to a folder          → "docs/"  (all PDFs loaded)
            - list of PDF paths             → ["docs/a.pdf", "docs/b.pdf"]

    Returns:
        List of Document objects (pages across all PDFs).
    """
    # ── Resolve input into a list of file paths or a directory ────────
    if isinstance(pdf_input, list):
        # Multiple explicit files
        for path in pdf_input:
            if not os.path.exists(path):
                raise FileNotFoundError(f"PDF not found: '{path}'")
        print(f"Loading {len(pdf_input)} PDFs...")
        reader = SimpleDirectoryReader(input_files=pdf_input)

    elif os.path.isdir(pdf_input):
        # Entire directory — picks up all .pdf files automatically
        print(f"Loading all PDFs from folder: '{pdf_input}'...")
        reader = SimpleDirectoryReader(
            input_dir=pdf_input,
            required_exts=[".pdf"],
            recursive=True,          # also scan sub-folders
        )

    else:
        # Single file path (str)
        if not os.path.exists(pdf_input):
            raise FileNotFoundError(f"PDF not found: '{pdf_input}'")
        print(f"Loading PDF: '{pdf_input}'...")
        reader = SimpleDirectoryReader(input_files=[pdf_input])

    docs = reader.load_data()

    # Print summary grouped by source file
    sources: dict[str, int] = {}
    for doc in docs:
        fname = doc.metadata.get("file_name", "unknown")
        sources[fname] = sources.get(fname, 0) + 1

    print(f"✓ Loaded {len(docs)} pages from {len(sources)} file(s):")
    for fname, pages in sources.items():
        print(f"    • {fname}  ({pages} pages)")

    return docs