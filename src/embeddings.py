"""Embedding providers for retrieval."""

import logging
import math
import os
from typing import Any

import requests
from langchain_core.embeddings import Embeddings

from config import Config

logger = logging.getLogger(__name__)


class RemoteHuggingFaceEmbeddings(Embeddings):
    """Hugging Face Inference API embeddings without local Torch."""

    def __init__(self, model_name: str = Config.EMBED_MODEL) -> None:
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.api_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.timeout = int(os.environ.get("HF_EMBED_TIMEOUT_SECONDS", "60"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents through the remote inference API."""

        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query through the remote inference API."""

        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        response = requests.post(
            self.api_url,
            headers=headers,
            json={
                "inputs": text,
                "options": {"wait_for_model": True},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return _normalize(_coerce_embedding(response.json()))


def get_embeddings() -> Embeddings:
    """Return the configured embedding provider.

    The default is remote embeddings to avoid installing Torch inside Docker.
    Set EMBEDDING_BACKEND=local only in an environment where
    langchain-huggingface and sentence-transformers are intentionally installed.
    """

    backend = os.environ.get("EMBEDDING_BACKEND", "remote").lower()
    if backend == "local":
        logger.info("Using local HuggingFace embeddings. This requires Torch.")
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    logger.info("Using remote HuggingFace embeddings for model: %s", Config.EMBED_MODEL)
    return RemoteHuggingFaceEmbeddings(Config.EMBED_MODEL)


def _coerce_embedding(payload: Any) -> list[float]:
    """Convert common Hugging Face feature-extraction shapes to one vector."""

    if not isinstance(payload, list) or not payload:
        raise ValueError("Embedding response was empty or invalid.")

    if all(isinstance(value, (int, float)) for value in payload):
        return [float(value) for value in payload]

    if all(isinstance(value, list) for value in payload):
        first = payload[0]
        if first and all(isinstance(value, (int, float)) for value in first):
            if len(payload) == 1:
                return [float(value) for value in first]
            return _mean_pool(payload)

        if first and all(isinstance(value, list) for value in first):
            return _mean_pool(first)

    raise ValueError("Unsupported embedding response shape.")


def _mean_pool(token_vectors: list[list[float]]) -> list[float]:
    """Mean-pool token vectors when the API returns token-level embeddings."""

    if not token_vectors:
        raise ValueError("Cannot pool an empty embedding response.")

    dimensions = len(token_vectors[0])
    pooled = [0.0] * dimensions
    for vector in token_vectors:
        for index, value in enumerate(vector):
            pooled[index] += float(value)

    return [value / len(token_vectors) for value in pooled]


def _normalize(vector: list[float]) -> list[float]:
    """L2-normalize an embedding vector to match the existing vector index."""

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
