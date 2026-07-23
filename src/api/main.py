"""FastAPI application entrypoint for QEDS-GPT."""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, health, version
from observability import instrument_fastapi_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="QEDS-GPT API",
        description="FastAPI backend for the QEDS-GPT Hybrid RAG application.",
        version="1.0.0",
    )

    allowed_origins = _get_allowed_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(version.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    instrument_fastapi_app(app)

    return app


def _get_allowed_origins() -> list[str]:
    """Read allowed CORS origins from the environment."""

    raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = create_app()
