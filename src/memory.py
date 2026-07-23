"""Conversation state and memory management."""

from collections import deque
from typing import Any

import streamlit as st

from config import Config
from prompts import get_summary_prompt


class ChatManager:
    """Manages Streamlit session state and conversational memory."""

    def __init__(self, llm_service=None) -> None:
        self.llm = llm_service
        self.initialize_state()

    def initialize_state(self) -> None:
        """Initialize Streamlit session state keys used by chat memory."""

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = deque(maxlen=Config.MAX_TURNS_IN_MEMORY * 2)
            st.session_state.conversation_summary = ""
            st.session_state.turn_count = 0

    def add_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        sources: list[dict[str, Any]] | None = None,
        retrieved_chunks: list[dict[str, Any]] | None = None,
        used_retrieval: bool = False,
    ) -> None:
        """Add a complete user/assistant turn to memory."""

        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": assistant_msg,
                "sources": sources or [],
                "retrieved_chunks": retrieved_chunks or [],
                "used_retrieval": used_retrieval,
            }
        )
        st.session_state.turn_count += 1

        if st.session_state.turn_count >= Config.SUMMARY_THRESHOLD // 2:
            self.create_summary()
            st.session_state.turn_count = 0

    def create_summary(self) -> None:
        """Summarize older conversation turns."""

        if self.llm is None:
            return

        if len(st.session_state.chat_messages) < 4:
            return

        messages = list(st.session_state.chat_messages)[:4]
        convo_text = "\n".join(
            [f"{message['role'].upper()}: {message['content'][:200]}" for message in messages]
        )
        prompt = get_summary_prompt(st.session_state.conversation_summary, convo_text)
        new_summary = self.llm.call_llm(
            [{"role": "user", "content": prompt}],
            temp=0.0,
            max_tokens=150,
        )

        if "error" not in new_summary.lower():
            st.session_state.conversation_summary = new_summary

    def get_context_messages(self) -> list[dict[str, Any]]:
        """Return summary and recent messages for display and prompt context."""

        messages = []
        if st.session_state.conversation_summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Context: {st.session_state.conversation_summary}",
                }
            )
        messages.extend(list(st.session_state.chat_messages))
        return messages

    def clear(self) -> None:
        """Clear chat memory."""

        st.session_state.chat_messages.clear()
        st.session_state.conversation_summary = ""
        st.session_state.turn_count = 0
