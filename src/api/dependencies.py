"""FastAPI dependency providers."""

import logging
from functools import lru_cache

from fastapi import HTTPException, status

from services import RAGService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _build_rag_service() -> RAGService:
    """Create a singleton RAG service for the API process."""

    return RAGService()


def get_rag_service() -> RAGService:
    """Provide the RAG service through FastAPI dependency injection."""

    try:
        return _build_rag_service()
    except Exception as exc:
        logger.error("Failed to initialize RAG service: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG service could not be initialized. Check server configuration.",
        ) from exc
