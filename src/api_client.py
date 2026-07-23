"""HTTP client helpers used by the Streamlit frontend."""

import os
from typing import Any

import requests

DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1"


def get_api_base_url() -> str:
    """Return the configured FastAPI base URL."""

    return (_read_config_value("QEDS_API_URL") or DEFAULT_API_BASE_URL).rstrip("/")


def _read_config_value(name: str) -> str | None:
    """Read frontend configuration from environment or Streamlit secrets."""

    value = os.environ.get(name)
    if value:
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        return str(secret_value) if secret_value else None
    except Exception:
        return None


def submit_chat_job(message: str, semester_filter: int | None, history: list[dict[str, Any]]) -> dict[str, Any]:
    """Submit a chat job to the FastAPI backend."""

    response = requests.post(
        f"{get_api_base_url()}/chat",
        json={
            "message": message,
            "semester_filter": semester_filter,
            "history": history,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def get_chat_result(job_id: str) -> dict[str, Any]:
    """Poll a queued chat job result from the FastAPI backend."""

    response = requests.get(f"{get_api_base_url()}/result/{job_id}", timeout=15)
    response.raise_for_status()
    return response.json()
