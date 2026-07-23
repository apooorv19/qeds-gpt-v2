"""Groq LLM client wrapper."""

import logging
import os
from typing import Any

from groq import Groq

from config import Config

logger = logging.getLogger(__name__)


class LLMService:
    """Handles Groq API communication."""

    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY") or self._read_streamlit_secret()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        self.client = Groq(api_key=api_key)

    def _read_streamlit_secret(self) -> str | None:
        """Read the Groq API key from Streamlit secrets when available."""

        try:
            import streamlit as st

            return st.secrets.get("GROQ_API_KEY")
        except Exception:
            return None

    def call_llm(
        self,
        messages: list[dict[str, Any]],
        temp: float = Config.TEMPERATURE,
        max_tokens: int = Config.MAX_TOKENS,
    ) -> str:
        """Call the configured Groq chat model."""

        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
                top_p=Config.TOP_P,
                frequency_penalty=Config.FREQUENCY_PENALTY,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("LLM API Error: %s", exc, exc_info=True)
            return "I encountered an error processing your request. Please try again."
