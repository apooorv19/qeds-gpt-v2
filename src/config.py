"""Application configuration and shared constants."""

import os


class Config:
    """Centralized configuration for the application."""

    DB_PATH = os.environ.get(
        "CHROMA_DB_PATH",
        "/app/chroma_db" if os.path.exists("/app/chroma_db") else "chroma_db",
    )
    COLLECTION_NAME = "semester_notes"
    EMBED_MODEL = "BAAI/bge-m3"
    LLM_MODEL = "llama-3.1-8b-instant"
    TOP_K = 3
    SIMILARITY_THRESHOLD = 0.65
    MAX_CONTEXT_CHARS = 5000
    MAX_TURNS_IN_MEMORY = 3
    SUMMARY_THRESHOLD = 6
    TEMPERATURE = 0.1
    MAX_TOKENS = 1024
    TOP_P = 0.9
    FREQUENCY_PENALTY = 0.15


class QueryCategory:
    """Supported query classification categories."""

    ACADEMIC = "ACADEMIC"
    GREETING = "GREETING"
    META = "META"
    GENERAL = "GENERAL"
    INJECTION = "INJECTION"
