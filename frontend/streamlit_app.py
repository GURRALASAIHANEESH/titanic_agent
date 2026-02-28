# ==============================================================
# FILE: titanic_agent/frontend/streamlit_app.py
# PURPOSE: Streamlit chat interface — calls the FastAPI backend.
# ==============================================================

import base64
import os
import time
from io import BytesIO
from typing import Optional

import requests
import streamlit as st
from PIL import Image

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------

BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:8000")
CHAT_ENDPOINT: str = f"{BACKEND_URL}/chat"
HEALTH_ENDPOINT: str = f"{BACKEND_URL}/health"
REQUEST_TIMEOUT: int = 120  # seconds

EXAMPLE_QUESTIONS: list[str] = [
    "What percentage of passengers were male?",
    "Show me a histogram of passenger ages",
    "What was the average ticket fare?",
    "How many passengers embarked from each port?",
    "Show survival rate by gender",
    "Show survival rate by passenger class",
    "Show a bar chart of passenger class distribution",
    "What is the correlation between all features?",
    "Show age distribution by passenger class",
    "How many passengers traveled alone?",
    "What was the median age of survivors versus non-survivors?",
    "What was the survival rate overall?",
]

# --------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------

st.set_page_config(
    page_title="Titanic Chat Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* General background */
    .stApp { background-color: #0d1117; color: #c9d1d9; }

    /* User message bubble */
    .bubble-user {
        background-color: #1c2e4a;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
        color: #e2e8f0;
    }

    /* Bot message bubble */
    .bubble-bot {
        background-color: #161b22;
        border-left: 3px solid #22c55e;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
        color: #d1d5db;
    }

    /* Timing label */
    .timing-label {
        font-size: 11px;
        color: #6b7280;
        margin-top: 4px;
        padding-left: 2px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }

    /* Remove default padding on chat input */
    .stChatInput { padding-top: 0; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------
# Session state initialisation
# --------------------------------------------------------------

def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


# --------------------------------------------------------------
# Backend communication
# --------------------------------------------------------------

def check_backend_health() -> dict:
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=6)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Connection refused. Ensure the FastAPI server is running."}
    except requests.exceptions.Timeout:
        return {"error": "Health check timed out."}
    except Exception as exc:
        return {"error": str(exc)}


def call_backend(question: str) -> dict:
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={"question": question},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot reach the backend. Is the FastAPI server running?"}
    except requests.exceptions.Timeout:
        return {"error": "The request timed out. The agent may be processing a complex query."}
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return {"error": f"Backend returned an error: {detail}"}
    except Exception as exc:
        return {"error": f"Unexpected client error: {exc}"}


def decode_chart(b64_string: str) -> Optional[Image.Image]:
    try:
        image_bytes = base64.b64decode(b64_string)
        return Image.open(BytesIO(image_bytes))
    except Exception:
        return None


# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Titanic Chat Agent")
        st.caption(
            "Ask natural language questions about Titanic passenger data. "
            "The agent computes answers dynamically from the dataset."
        )
        st.divider()

        # Backend health check
        st.markdown("#### Backend Connection")
        if st.button("Check Connection", use_container_width=True):
            with st.spinner("Connecting..."):
                health = check_backend_health()
            if "error" in health:
                st.error(health["error"])
            else:
                st.success(
                    f"Connected\n\n"
                    f"- Dataset rows: **{health.get('dataset_rows', 'N/A')}**\n"
                    f"- Columns: **{health.get('dataset_columns', 'N/A')}**\n"
                    f"- Model: **{health.get('model', 'N/A')}**"
                )

        st.divider()

        # Example questions
        st.markdown("#### Example Questions")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True, key=f"btn_{hash(question)}"):
                st.session_state.pending_question = question

        st.divider()

        # Clear history
        if st.button("Clear Chat History", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("Backend: FastAPI + LangChain\nFrontend: Streamlit")


# --------------------------------------------------------------
# Chat history rendering
# --------------------------------------------------------------

def render_chat_history() -> None:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-user"><strong>You</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bubble-bot"><strong>Agent</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(msg["content"])

            if msg.get("chart_b64"):
                image = decode_chart(msg["chart_b64"])
                if image:
                    st.image(image, use_column_width=True)
                else:
                    st.warning("Chart data was returned but could not be rendered.")

            if msg.get("processing_time_ms") is not None:
                st.markdown(
                    f'<div class="timing-label">Processed in {msg["processing_time_ms"]} ms</div>',
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------
# Question processing
# --------------------------------------------------------------

def process_question(question: str) -> None:
    """Append user message, call backend, append agent response."""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Analysing dataset..."):
        t0 = time.perf_counter()
        result = call_backend(question)
        client_elapsed = round((time.perf_counter() - t0) * 1000, 1)

    if "error" in result:
        bot_message = {
            "role": "assistant",
            "content": f"**Error:** {result['error']}",
            "chart_b64": None,
            "processing_time_ms": client_elapsed,
        }
    else:
        bot_message = {
            "role": "assistant",
            "content": result.get("answer", "No answer was returned."),
            "chart_b64": result.get("chart_base64") if result.get("has_chart") else None,
            "processing_time_ms": result.get("processing_time_ms", client_elapsed),
        }

    st.session_state.messages.append(bot_message)
    st.rerun()


# --------------------------------------------------------------
# Main application
# --------------------------------------------------------------

def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("Titanic Dataset Chat Agent")
    st.caption(
        "Ask questions in plain English. "
        "The agent reads from the actual Titanic dataset — no hardcoded answers."
    )
    st.divider()

    render_chat_history()

    # Consume a question injected via sidebar button
    pending = st.session_state.pending_question
    if pending:
        st.session_state.pending_question = None
        process_question(pending)
        return  # rerun will handle re-rendering

    # Standard chat input
    user_input: Optional[str] = st.chat_input(
        "Ask a question about the Titanic dataset..."
    )
    if user_input:
        process_question(user_input.strip())


if __name__ == "__main__":
    main()
