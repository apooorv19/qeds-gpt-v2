"""Optional Pydantic Logfire observability configuration."""

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIGURED_SERVICES: set[str] = set()


def configure_logfire(service_name: str) -> bool:
    """Configure Logfire once for a process service.

    Logfire is optional. If it is disabled or unavailable, the application
    continues with standard Python logging.
    """

    if service_name in _CONFIGURED_SERVICES:
        return True

    if not _is_logfire_enabled():
        logger.info("Logfire disabled for service: %s", service_name)
        return False

    try:
        import logfire

        token = _read_secret("LOGFIRE_TOKEN") or _read_secret("LOGFIRE_API_KEY")
        token_source = _secret_source("LOGFIRE_TOKEN") or _secret_source("LOGFIRE_API_KEY")
        logfire.configure(
            token=token,
            send_to_logfire=os.environ.get("LOGFIRE_SEND_TO_LOGFIRE", "if-token-present"),
            service_name=service_name,
            service_version=os.environ.get("QEDS_APP_VERSION", "1.0.0"),
            environment=os.environ.get("APP_ENV", "local"),
        )

        if _env_bool("LOGFIRE_INSTRUMENT_REDIS", default=False):
            logfire.instrument_redis(capture_statement=False)

        _CONFIGURED_SERVICES.add(service_name)
        logger.info("Logfire configured for service: %s using %s", service_name, token_source or "default configuration")
        return True
    except Exception as exc:
        logger.warning("Logfire could not be configured: %s", exc, exc_info=True)
        return False


def instrument_fastapi_app(app: Any) -> None:
    """Instrument a FastAPI app without recording request bodies."""

    if not configure_logfire("qeds-gpt-api"):
        return

    try:
        import logfire

        logfire.instrument_fastapi(
            app,
            capture_headers=False,
            request_attributes_mapper=_safe_fastapi_request_attributes,
        )
    except Exception as exc:
        logger.warning("FastAPI Logfire instrumentation failed: %s", exc, exc_info=True)


def logfire_span(name: str, **attributes: Any):
    """Return a Logfire span when available, otherwise a no-op context manager."""

    if not _is_logfire_enabled():
        return _NoOpSpan()

    try:
        import logfire

        return logfire.span(name, **attributes)
    except Exception:
        return _NoOpSpan()


def _safe_fastapi_request_attributes(request: Any, attributes: dict[str, Any]) -> dict[str, Any]:
    """Keep FastAPI validation errors, but avoid recording valid request values."""

    safe_attributes: dict[str, Any] = {
        "http.route": getattr(request, "url", None).path if getattr(request, "url", None) else None,
        "http.method": getattr(request, "method", None),
    }
    if attributes.get("errors"):
        safe_attributes["errors"] = attributes["errors"]
    return {key: value for key, value in safe_attributes.items() if value is not None}


def _is_logfire_enabled() -> bool:
    """Return whether Logfire should be attempted."""

    if _env_bool("LOGFIRE_ENABLED", default=False):
        return True
    return bool(_read_secret("LOGFIRE_TOKEN") or _read_secret("LOGFIRE_API_KEY"))


def _read_secret(name: str) -> str | None:
    """Read a secret from environment, Streamlit secrets, or local secrets.toml."""

    value = os.environ.get(name)
    if value:
        return value

    streamlit_value = _read_streamlit_secret(name)
    if streamlit_value:
        return streamlit_value

    return _read_local_secrets_toml(name) or _read_local_dotenv(name)


def _secret_source(name: str) -> str | None:
    """Return where a secret would be read from, without exposing the value."""

    if os.environ.get(name):
        return f"environment variable {name}"
    if _read_streamlit_secret(name):
        return f"Streamlit secret {name}"
    if _read_local_secrets_toml(name):
        return f".streamlit/secrets.toml key {name}"
    if _read_local_dotenv(name):
        return f".env key {name}"
    return None


def _read_streamlit_secret(name: str) -> str | None:
    """Read a value from Streamlit secrets when Streamlit is available."""

    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def _read_local_secrets_toml(name: str) -> str | None:
    """Read .streamlit/secrets.toml for local worker/API development."""

    secrets_path = Path.cwd() / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None

    try:
        data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
        value = data.get(name)
        return str(value) if value else None
    except Exception as exc:
        logger.debug("Could not read local secrets.toml for %s: %s", name, exc)
        return None


def _read_local_dotenv(name: str) -> str | None:
    """Read simple KEY=value entries from a local .env file."""

    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return None

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            if key.strip() != name:
                continue

            return value.strip().strip('"').strip("'") or None
    except Exception as exc:
        logger.debug("Could not read local .env for %s: %s", name, exc)
    return None


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable."""

    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


class _NoOpSpan:
    """No-op context manager used when Logfire is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False
