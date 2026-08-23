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
        api_key = os.environ.get(
            "GROQ_API_KEY"
        ) or self._read_streamlit_secret()

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
                reasoning_effort="low",
            )

            message = response.choices[0].message

            content = message.content

            if content is None or not content.strip():
                logger.error(
                    "Groq returned empty content. "
                    "finish_reason=%s | reasoning=%s",
                    response.choices[0].finish_reason,
                    getattr(message, "reasoning", None),
                )

                raise RuntimeError(
                    "Groq returned an empty response."
                )

            return content.strip()

        except Exception as exc:
            logger.exception(
                "Groq request failed | model=%s",
                Config.LLM_MODEL,
            )

            raise RuntimeError(
                f"LLM request failed using model "
                f"'{Config.LLM_MODEL}': {exc}"
            ) from exc
