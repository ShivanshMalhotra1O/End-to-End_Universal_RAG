import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Medical RAG Agent",
    page_icon="🏥",
    layout="centered"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: #0f1117;
    color: #e8e8e8;
}

/* Title area */
.rag-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #f0f0f0;
    margin-bottom: 0.2rem;
}

.rag-subtitle {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 2rem;
    letter-spacing: 0.03em;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #1a1d27 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-bottom: 0.75rem !important;
}

/* Input box */
[data-testid="stChatInput"] {
    background-color: #1a1d27 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 12px !important;
    color: #e8e8e8 !important;
}

/* Cache badge */
.cache-badge {
    display: inline-block;
    background-color: #1e3a2f;
    color: #4ade80;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 999px;
    margin-bottom: 6px;
    letter-spacing: 0.05em;
    font-family: 'DM Sans', sans-serif;
}

/* Status pills */
.status-pill {
    display: inline-block;
    background-color: #1e2a3a;
    color: #60a5fa;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 1rem;
}

/* Divider */
hr {
    border-color: #2a2d3a;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper: Find answer in history ───────────────────────────────────────────
def find_in_history(question: str, history: list) -> str | None:
    for i, message in enumerate(history):
        if message.type == "human" and message.content.lower() == question.lower():
            if i + 1 < len(history):
                return history[i + 1].content
    return None


# ─── Initialize index + graph ONCE ────────────────────────────────────────────
# Streamlit reruns the whole script on every interaction.
# session_state persists data between reruns.
# Without this check, the index would rebuild on every message!

if "initialized" not in st.session_state:
    with st.spinner("Loading and indexing document... please wait"):
        from src.injestion import build_index
        from src.graph import build_graph

        config_path = os.getenv("CONFIG_PATH")
        index = build_index(config_path)
        graph = build_graph(index, config_path)

        st.session_state.index = index
        st.session_state.graph = graph
        st.session_state.chat_history = []
        st.session_state.config_path = config_path
        st.session_state.initialized = True


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="rag-title">🏥 Medical RAG Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="rag-subtitle">Powered by Gale Encyclopedia of Medicine</div>', unsafe_allow_html=True)
st.markdown('<div class="status-pill">● Ready</div>', unsafe_allow_html=True)


# ─── Display chat history ──────────────────────────────────────────────────────
# This loop runs on every rerender to show all previous messages
for message in st.session_state.chat_history:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message.content)


# ─── Chat input ───────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a medical question...")

if user_input:

    # Show user message immediately
    with st.chat_message("user"):
        st.write(user_input)

    # Check cache first
    cached_answer = find_in_history(user_input, st.session_state.chat_history)

    if cached_answer:
        # Serve from history - no API call needed
        with st.chat_message("assistant"):
            st.markdown('<div class="cache-badge">⚡ FROM HISTORY</div>', unsafe_allow_html=True)
            st.write(cached_answer)

    else:
        # Run full pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                result = st.session_state.graph.invoke({
                    "user_question": user_input,
                    "retrieved_chunks": [],
                    "answer": "",
                    "messages": st.session_state.chat_history
                })

            st.write(result["answer"])

        # Update history
        st.session_state.chat_history = result["messages"]