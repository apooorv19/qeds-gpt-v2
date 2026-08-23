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

        logger.info(
            "Calling Groq | model=%s | temperature=%s | max_tokens=%s",
            Config.LLM_MODEL,
            temp,
            max_tokens,
        )

        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
                top_p=Config.TOP_P,
            )

            logger.info("Groq request completed successfully")

            content = response.choices[0].message.content

            if not content:
                logger.error("Groq returned an empty response: %s", response)
                return "The language model returned an empty response."

            return content.strip()

        except Exception as exc:
            logger.exception("Groq LLM request failed")

            raise RuntimeError(
                f"Groq LLM request failed: "
                f"{type(exc).__name__}: {str(exc)}"
            ) from exc
