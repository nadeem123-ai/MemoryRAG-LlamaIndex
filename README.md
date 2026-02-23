# 🧠 MemoryRAG — LlamaIndex Edition

A production-grade **Multi-PDF RAG pipeline** built with **LlamaIndex** that remembers your conversation history. Available as both a **Streamlit web app** and a **terminal CLI**.

> **Part 3 of the DeepRAG series.**
> [Part 1 — Built from scratch](https://github.com/nadeem123-ai/DeepRAG) | [Part 2 — LangChain](your-link) | Part 3 — LlamaIndex ← you are here

---

## ✨ What Makes This Special

Most RAG systems treat every question independently. **MemoryRAG remembers.**

```
You:  "What are his technical skills?"
AI:   "He knows Python, LlamaIndex, ChromaDB..."

You:  "Tell me more about the first one"   ← no context given!
AI:   "Python is used for..."              ← knows "first one" = Python ✅
```

And it works across **multiple PDFs at once** — ask about your resume, then your report, then compare them. It knows which document each answer came from.

---

## 🚀 Features

- 🧠 **Conversation Memory** — remembers all previous Q&A in a session
- 📄 **Multi-PDF Support** — load a single file, multiple files, or an entire folder
- 🌐 **Streamlit Web UI** — beautiful dark-themed chat interface with source pills
- 💻 **Terminal CLI** — classic interactive mode still available
- ✂️ **Smart Chunking** — SentenceSplitter preserves natural sentence boundaries
- 🔢 **HuggingFace Embeddings** — `BAAI/bge-small-en-v1.5` (384-dim vectors)
- 🗄️ **Chroma DB** — persistent vector store, no re-embedding on restart
- 🤖 **Dual LLM Support** — Ollama (local/free) or OpenAI (cloud)
- 🪟 **Windows Compatible** — handles Chroma file-locking gracefully

---

## 🏗️ Project Structure

```
MemoryRAG-LlamaIndex/
├── docs/                    ← put your PDF files here
├── rag/
│   ├── __init__.py          ← package entry point
│   ├── loader.py            ← multi-PDF loading (file / list / folder)
│   ├── splitter.py          ← SentenceSplitter (chunk + overlap)
│   ├── embedder.py          ← HuggingFaceEmbedding (BAAI/bge-small-en-v1.5)
│   ├── vector_store.py      ← ChromaDB persistent store (Windows-safe)
│   ├── llm.py               ← Ollama + OpenAI unified interface
│   └── pipeline.py          ← CondensePlusContextChatEngine + ChatMemoryBuffer
├── app.py                   ← Streamlit web UI
├── main.py                  ← terminal CLI
├── requirements.txt
└── .env                     ← OpenAI API key (optional)
```

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/MemoryRAG-LlamaIndex.git
cd MemoryRAG-LlamaIndex
```

### 2. Create virtual environment
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ Takes 5–10 minutes — downloads PyTorch + sentence-transformers.

### 4. Install Ollama + pull a model
Download from **https://ollama.com**, then:
```bash
# Recommended for 4–8 GB RAM
ollama pull qwen2.5:1.5b

# Alternatives
ollama pull qwen2.5:0.5b   # lightest (1 GB RAM)
ollama pull phi3:mini       # best quality on low RAM
ollama pull mistral         # needs 8 GB RAM
```

### 5. (Optional) OpenAI setup
```bash
# Create .env file
echo OPENAI_API_KEY=sk-... > .env
```

---

## 🖥️ Usage

### Web UI (Streamlit) — recommended
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

1. Select PDF source in the sidebar (folder or upload)
2. Choose your model
3. Click **Load & Initialize**
4. Start chatting!

### Terminal CLI
```bash
# Load entire docs/ folder
python main.py --pdf docs/

# Load specific files
python main.py --pdf docs/resume.pdf docs/report.pdf

# Use OpenAI
python main.py --pdf docs/ --provider openai --model gpt-4o-mini

# Skip demo, jump to chat
python main.py --pdf docs/ --model qwen2.5:1.5b --no-demo
```

### Terminal commands
```
clear    → reset conversation memory
history  → show all previous Q&A
exit     → quit
```

---

## 🔍 How It Works

```
Your Question
      ↓
CondensePlusContextChatEngine
      ↓
  [ChatMemoryBuffer] condenses question with history
      ↓
  [ChromaDB] finds top-k relevant nodes across all PDFs
      ↓
  [LLM] generates answer using nodes + history
      ↓
  [ChatMemoryBuffer] saves Q&A for next turn
      ↓
Answer + Source Pills [resume.pdf · p1]  [islamiyat.pdf · p3]
```

---

## 🔄 LangChain → LlamaIndex Mapping

| LangChain | LlamaIndex | Role |
|-----------|-----------|------|
| `PyPDFLoader` | `SimpleDirectoryReader` | PDF loading |
| `RecursiveCharacterTextSplitter` | `SentenceSplitter` | Chunking |
| `HuggingFaceEmbeddings` | `HuggingFaceEmbedding` | Embeddings |
| `Chroma` (LC wrapper) | `ChromaVectorStore` + `VectorStoreIndex` | Vector store |
| `OllamaLLM` / `ChatOpenAI` | `Ollama` / `OpenAI` | LLM |
| `ConversationBufferMemory` | `ChatMemoryBuffer` | Memory |
| `ConversationalRetrievalChain` | `CondensePlusContextChatEngine` | Chain / Engine |
| `source_documents` | `source_nodes` | Source tracking |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LlamaIndex 0.10.x |
| Web UI | Streamlit |
| Embeddings | HuggingFace — BAAI/bge-small-en-v1.5 |
| Vector DB | Chroma DB (persistent, Windows-safe) |
| Local LLM | Ollama — qwen2.5:1.5b / mistral / phi3 |
| Cloud LLM | OpenAI — GPT-4o / GPT-4o-mini |
| Memory | ChatMemoryBuffer (token-limited) |
| Engine | CondensePlusContextChatEngine |
| PDF | SimpleDirectoryReader |
| Language | Python 3.10+ |

---

## 🪟 Windows Notes

ChromaDB holds file locks on Windows which causes `WinError 32` when re-initialising. This project handles it automatically:

- `app.py` releases the pipeline and calls `gc.collect()` before touching `chroma_db/`
- `vector_store.py` falls back to an in-memory `EphemeralClient` if the persistent store is locked
- No manual deletion of `chroma_db/` needed

---

## 💡 Key Learnings

**LlamaIndex's `CondensePlusContextChatEngine` > LangChain's `ConversationalRetrievalChain`**
It separately condenses the question AND retrieves context before answering — better follow-up accuracy.

**`Settings` global is cleaner than passing objects everywhere**
One line `Settings.llm = llm` and every component downstream uses it automatically.

**Build from scratch first**
I built DeepRAG (Part 1) manually before using LangChain (Part 2) and LlamaIndex (Part 3). Every abstraction made sense because I had already implemented it myself.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

**Muhammad Nadeem**
AI / ML Engineer · LlamaIndex · RAG · Generative AI · LLM Systems

⭐ If you found this useful, please give it a star!