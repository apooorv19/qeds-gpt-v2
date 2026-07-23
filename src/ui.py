"""Streamlit UI rendering helpers."""

import time
from typing import Optional

import streamlit as st

from text_processor import format_source


FALLBACK_PHRASE = "I couldn't find this exact topic in the semester notes"


def render_header() -> None:
    """Render the app title and caption."""

    st.title("📘 QEDS-GPT")
    st.caption("Hybrid RAG System for Economics & Data Science Notes")


def render_sidebar(chat_manager) -> Optional[int]:
    """Render sidebar controls and return the selected semester filter."""

    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("🧹 Clear Chat History", use_container_width=True):
            chat_manager.clear()
            st.rerun()

        st.divider()
        st.subheader("🔍 Filters")
        semester_filter = st.selectbox(
            "Filter by Semester",
            options=[None, 1, 2, 3, 4, 5, 6],
            format_func=lambda value: "All Semesters" if value is None else f"Semester {value}",
        )

        st.divider()
        st.subheader("📊 Session Stats")
        st.metric("Messages in Memory", len(st.session_state.get("chat_messages", [])))
        if st.session_state.get("conversation_summary"):
            with st.expander("📝 Summary"):
                st.write(st.session_state.conversation_summary)

    return semester_filter


def render_chat_history(chat_manager) -> None:
    """Render visible chat history."""

    for message in chat_manager.get_context_messages():
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    render_api_sources(message)


def render_immediate_response(response: str) -> None:
    """Render an assistant response that does not need retrieval."""

    with st.chat_message("assistant"):
        st.markdown(response)


def render_api_sources(result: dict) -> None:
    """Render API-provided sources and retrieved chunks."""

    if not result.get("used_retrieval"):
        return

    sources = result.get("sources", [])
    if sources:
        st.markdown("**📚 Sources Used:**\n" + "\n".join([f"- {source['label']}" for source in sources]))

    chunks = result.get("retrieved_chunks", [])
    if chunks:
        with st.expander("View Retrieved Chunks", expanded=False):
            for chunk in chunks:
                st.markdown(f"### Chunk {chunk['index']}")
                st.markdown(f"**Source:** {chunk['source']}\n\n{chunk['content_preview']}")
                st.divider()


def render_sources(answer: str, docs: list) -> None:
    """Render sources and retrieved chunks when the answer used retrieved notes."""

    if FALLBACK_PHRASE in answer or not docs:
        return

    sources = list(dict.fromkeys([format_source(doc.metadata) for doc in docs]))
    st.markdown("**📚 Sources Used:**\n" + "\n".join([f"- {source}" for source in sources]))

    with st.expander("View Retrieved Chunks", expanded=False):
        for index, doc in enumerate(docs, 1):
            st.markdown(f"### Chunk {index}")
            st.markdown(f"**Source:** {format_source(doc.metadata)}\n\n{doc.page_content[:1000]}...")
            st.divider()


def render_academic_response(rag_service, chat_manager, user_query: str, semester_filter: Optional[int]) -> None:
    """Run and render the RAG response flow."""

    with st.chat_message("assistant"):
        start_time = time.time()
        with st.status("Processing...", expanded=False) as status:
            status.update(label="Loading retrieval stack and fusing notes...")
            try:
                docs = rag_service.retrieve(user_query, semester_filter)
            except Exception as exc:
                status.update(label="Retrieval failed", state="error", expanded=True)
                st.error("The retrieval system could not start. The app will answer from general knowledge.")
                st.exception(exc)
                docs = []

            if not docs:
                status.update(label="No matching notes; using general knowledge...")

            status.update(label="Preparing Context...")
            context_messages = chat_manager.get_context_messages()

            status.update(label="Generating Answer...")
            answer = rag_service.build_academic_answer(user_query, docs, context_messages)
            status.update(label="Done", state="complete")

        st.markdown(answer)
        st.divider()
        render_sources(answer, docs)
        st.caption(f"⏱️ {time.time() - start_time:.2f}s")

    chat_manager.add_turn(user_query, answer)
