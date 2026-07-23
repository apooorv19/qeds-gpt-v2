import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import logging
import time
from typing import Any

import requests
import streamlit as st

from api_client import get_chat_result, submit_chat_job
from memory import ChatManager
from ui import render_api_sources, render_chat_history, render_header, render_sidebar


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Configure and run the QEDS-GPT Streamlit app."""

    st.set_page_config(page_title="QEDS-GPT", page_icon=":books:", layout="wide")
    render_header()

    chat_manager = ChatManager()
    _initialize_async_state()

    semester_filter = render_sidebar(chat_manager)
    render_chat_history(chat_manager)
    _render_pending_job(chat_manager)

    user_query = st.chat_input(
        "Ask your question...",
        disabled=st.session_state.pending_job is not None,
    )
    if not user_query:
        return

    context_messages = chat_manager.get_context_messages()
    try:
        response = submit_chat_job(user_query, semester_filter, context_messages)
        st.session_state.pending_job = {
            "job_id": response["job_id"],
            "user_query": user_query,
            "submitted_at": time.time(),
        }
        logger.info("Submitted chat job: %s", response["job_id"])
        st.rerun()
    except requests.RequestException as exc:
        logger.error("Failed to submit chat job: %s", exc, exc_info=True)
        st.error("The API is unavailable. Make sure FastAPI and Redis are running.")


def _initialize_async_state() -> None:
    """Initialize Streamlit state used for asynchronous job polling."""

    if "pending_job" not in st.session_state:
        st.session_state.pending_job = None


def _render_pending_job(chat_manager: ChatManager) -> None:
    """Poll and render the currently pending chat job, if any."""

    pending_job: dict[str, Any] | None = st.session_state.pending_job
    if pending_job is None:
        return

    user_query = pending_job["user_query"]
    job_id = pending_job["job_id"]

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        try:
            result = get_chat_result(job_id)
        except requests.RequestException as exc:
            logger.error("Failed to poll chat job %s: %s", job_id, exc, exc_info=True)
            status_placeholder.error("Could not poll the API. Retrying in 1 second...")
            time.sleep(1)
            st.rerun()

        status_name = result["status"]
        if status_name in {"queued", "running"}:
            elapsed = time.time() - float(pending_job["submitted_at"])
            status_placeholder.info(f"Job {status_name}. Waiting for result... ({elapsed:.1f}s)")
            time.sleep(1)
            st.rerun()

        if status_name == "completed":
            answer = result.get("answer") or ""
            status_placeholder.markdown(answer)
            render_api_sources(result)
            chat_manager.add_turn(
                user_query,
                answer,
                sources=result.get("sources", []),
                retrieved_chunks=result.get("retrieved_chunks", []),
                used_retrieval=bool(result.get("used_retrieval")),
            )
            st.session_state.pending_job = None
            time.sleep(0.2)
            st.rerun()

        error_message = result.get("error") or "The queued job failed."
        status_placeholder.error(error_message)
        chat_manager.add_turn(user_query, error_message)
        st.session_state.pending_job = None


if __name__ == "__main__":
    main()
