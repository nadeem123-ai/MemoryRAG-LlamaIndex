"""
app.py
------
Streamlit web UI for the LlamaIndex Multi-PDF RAG pipeline.

Run:
    streamlit run app.py
"""

import os
import time
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MemoryRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem 2rem; max-width: 100%; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e3a;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Header ── */
.rag-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.2rem 1.6rem;
    background: linear-gradient(135deg, #0f0f1a 0%, #13132a 100%);
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    margin-bottom: 1.5rem;
}
.rag-header-icon {
    font-size: 2.2rem;
    line-height: 1;
}
.rag-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.7rem;
    margin: 0;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.rag-header p {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #555580;
    margin: 2px 0 0 0;
    letter-spacing: 0.04em;
}

/* ── Chat messages ── */
.chat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1.2rem;
    animation: fadeSlideUp 0.3s ease;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-row.user { flex-direction: row-reverse; }

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar.user-av  { background: linear-gradient(135deg, #6d28d9, #3b82f6); }
.avatar.bot-av   { background: linear-gradient(135deg, #1e1e3a, #2a2a4a); border: 1px solid #3a3a6a; }

.bubble {
    max-width: 72%;
    padding: 0.9rem 1.1rem;
    border-radius: 14px;
    font-size: 0.92rem;
    line-height: 1.65;
}
.bubble.user-bubble {
    background: linear-gradient(135deg, #2d1b69, #1e3a5f);
    border: 1px solid #4c3a99;
    border-top-right-radius: 4px;
    color: #ddd6fe;
}
.bubble.bot-bubble {
    background: #13132a;
    border: 1px solid #1e1e3a;
    border-top-left-radius: 4px;
    color: #e8e8f0;
}

/* ── Sources pills ── */
.sources-wrap {
    margin-top: 0.65rem;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.source-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 9px;
    border-radius: 20px;
    background: #1a1a30;
    border: 1px solid #2a2a50;
    color: #7c7caa;
    cursor: default;
    transition: border-color 0.2s;
}
.source-pill:hover { border-color: #a78bfa; color: #c4b5fd; }

/* ── Status bar ── */
.status-bar {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #555580;
    padding: 0.4rem 0.8rem;
    background: #0d0d1a;
    border: 1px solid #1a1a30;
    border-radius: 8px;
    margin-bottom: 1rem;
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
.dot-green  { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.dot-yellow { background: #eab308; }
.dot-red    { background: #ef4444; }

/* ── Input area ── */
.stTextInput > div > div > input {
    background: #0f0f1a !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6d28d9 !important;
    box-shadow: 0 0 0 2px rgba(109,40,217,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #3b82f6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Secondary buttons */
.stButton > button[kind="secondary"] {
    background: #1a1a30 !important;
    border: 1px solid #2a2a50 !important;
    color: #a0a0c8 !important;
}

/* ── Selectbox / inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div {
    background: #0f0f1a !important;
}

/* ── Sidebar labels ── */
.sidebar-section {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #555580;
    text-transform: uppercase;
    margin: 1.2rem 0 0.5rem 0;
}

/* ── Welcome screen ── */
.welcome-card {
    text-align: center;
    padding: 3rem 2rem;
    border: 1px dashed #1e1e3a;
    border-radius: 16px;
    margin: 2rem auto;
    max-width: 500px;
}
.welcome-card h2 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    color: #a78bfa;
    margin-bottom: 0.5rem;
}
.welcome-card p {
    font-size: 0.88rem;
    color: #555580;
    line-height: 1.7;
}
.step-badge {
    display: inline-block;
    background: #1a1a30;
    border: 1px solid #2a2a50;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.3rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #7c7caa;
}

/* ── Spinner override ── */
.stSpinner > div { border-top-color: #a78bfa !important; }

/* ── Scrollable chat area ── */
.chat-container {
    height: calc(100vh - 280px);
    overflow-y: auto;
    padding-right: 0.5rem;
    scrollbar-width: thin;
    scrollbar-color: #2a2a50 transparent;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_info" not in st.session_state:
    st.session_state.pipeline_info = {}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <span style="font-size:2.5rem">🧠</span>
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.2rem;
             background:linear-gradient(90deg,#a78bfa,#60a5fa);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            MemoryRAG
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:#555580; margin-top:4px;">
            LLAMAINDEX · MULTI-PDF · MEMORY
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── PDF Source ──
    st.markdown('<div class="sidebar-section">📂 PDF Source</div>', unsafe_allow_html=True)
    pdf_source = st.radio(
        "Load from",
        ["Folder (docs/)", "Upload PDFs"],
        label_visibility="collapsed",
    )

    uploaded_files = []
    pdf_folder = "docs/"

    if pdf_source == "Upload PDFs":
        uploaded_files = st.file_uploader(
            "Drop PDF files here",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    else:
        pdf_folder = st.text_input("Folder path", value="docs/", label_visibility="collapsed")

    # ── Model Settings ──
    st.markdown('<div class="sidebar-section">🤖 Model Settings</div>', unsafe_allow_html=True)

    provider = st.selectbox("Provider", ["ollama", "openai"], label_visibility="collapsed")

    if provider == "ollama":
        model = st.selectbox(
            "Model",
            ["qwen2.5:1.5b", "qwen2.5:0.5b", "mistral", "llama3", "phi3:mini", "tinyllama"],
            label_visibility="collapsed",
        )
    else:
        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o"],
            label_visibility="collapsed",
        )
        openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

    # ── Advanced ──
    with st.expander("⚙️ Advanced"):
        top_k = st.slider("Top-K chunks", 1, 10, 5)
        chunk_size = st.slider("Chunk size", 256, 2048, 1000, step=128)
        overlap = st.slider("Overlap", 0, 300, 150, step=50)

    st.divider()

    # ── Load / Initialize button ──
    load_btn = st.button("🚀 Load & Initialize", use_container_width=True)

    if load_btn:
        pdf_paths = []

        # Handle uploads
        if pdf_source == "Upload PDFs" and uploaded_files:
            os.makedirs("docs", exist_ok=True)
            for f in uploaded_files:
                save_path = os.path.join("docs", f.name)
                with open(save_path, "wb") as out:
                    out.write(f.read())
                pdf_paths.append(save_path)
        elif pdf_source == "Folder (docs/)":
            pdf_paths = pdf_folder
        else:
            st.error("Please select or upload PDF files.")
            st.stop()

        with st.spinner("Initialising pipeline..."):
            try:
                from rag import RAGPipeline
                import shutil, gc

                # Windows fix: release Chroma file handles before deleting
                if st.session_state.pipeline is not None:
                    try:
                        st.session_state.pipeline = None
                        gc.collect()
                    except Exception:
                        pass

                # Now safe to delete chroma_db on Windows
                if os.path.exists("chroma_db"):
                    try:
                        shutil.rmtree("chroma_db")
                    except PermissionError:
                        pass  # If still locked, reuse existing store

                st.session_state.pipeline = RAGPipeline(
                    pdf_path=pdf_paths,
                    top_k=top_k,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    provider=provider,
                    model=model,
                )
                st.session_state.messages = []
                st.session_state.pipeline_info = {
                    "provider": provider,
                    "model": model,
                    "top_k": top_k,
                    "source": pdf_source,
                }
                st.success("✓ Pipeline ready!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # ── Memory controls ──
    st.markdown('<div class="sidebar-section">💬 Conversation</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear", use_container_width=True):
            if st.session_state.pipeline:
                st.session_state.pipeline.clear_memory()
                st.session_state.messages = []
                st.rerun()
    with col2:
        msg_count = len(st.session_state.messages)
        st.markdown(
            f'<div style="text-align:center; font-family:Space Mono,monospace; '
            f'font-size:0.7rem; color:#555580; padding-top:0.5rem;">'
            f'{msg_count} msgs</div>',
            unsafe_allow_html=True,
        )


# ── Main area ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="rag-header">
    <div class="rag-header-icon">🧠</div>
    <div>
        <h1>MemoryRAG</h1>
        <p>MULTI-PDF · CONVERSATIONAL · LLAMAINDEX</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Status bar
if st.session_state.pipeline:
    info = st.session_state.pipeline_info
    st.markdown(f"""
    <div class="status-bar">
        <span><span class="status-dot dot-green"></span>PIPELINE READY</span>
        <span>MODEL: {info.get('model','–')}</span>
        <span>PROVIDER: {info.get('provider','–').upper()}</span>
        <span>TOP-K: {info.get('top_k','–')}</span>
        <span>SOURCE: {info.get('source','–')}</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-bar">
        <span><span class="status-dot dot-yellow"></span>NOT INITIALISED — configure & click Load in the sidebar</span>
    </div>
    """, unsafe_allow_html=True)


# ── Chat area ─────────────────────────────────────────────────────────────────
if not st.session_state.pipeline:
    st.markdown("""
    <div class="welcome-card">
        <h2>Get Started</h2>
        <p>Configure your pipeline in the sidebar, then click <strong>Load & Initialize</strong> to begin chatting with your PDFs.</p>
        <br>
        <div class="step-badge">① Select PDF source</div>
        <div class="step-badge">② Choose model</div>
        <div class="step-badge">③ Click Load</div>
        <div class="step-badge">④ Ask anything</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Render chat history
    chat_html = '<div class="chat-container" id="chat-bottom">'
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f"""
            <div class="chat-row user">
                <div class="avatar user-av">👤</div>
                <div class="bubble user-bubble">{msg['content']}</div>
            </div>"""
        else:
            sources_html = ""
            if msg.get("sources"):
                pills = "".join(
                    f'<span class="source-pill" title="{s["preview"]}">📄 {s["file"]} · p{s["page"]}</span>'
                    for s in msg["sources"]
                )
                sources_html = f'<div class="sources-wrap">{pills}</div>'
            chat_html += f"""
            <div class="chat-row">
                <div class="avatar bot-av">🧠</div>
                <div class="bubble bot-bubble">
                    {msg['content']}
                    {sources_html}
                </div>
            </div>"""
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # ── Input ──
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Message",
                placeholder="Ask anything about your documents...",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send →")

    if submitted and user_input.strip():
        question = user_input.strip()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})

        # Get answer
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.pipeline.ask(question)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {e}",
                    "sources": [],
                })

        st.rerun()