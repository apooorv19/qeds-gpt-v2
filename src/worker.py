"""Background worker for asynchronous QEDS-GPT RAG jobs."""

import logging
import os
from functools import lru_cache
from typing import Any

from rq import SimpleWorker, Worker
from rq.job import get_current_job
from rq.timeouts import TimerDeathPenalty

from config import QueryCategory
from job_queue import get_queue, get_redis_connection
from observability import configure_logfire, logfire_span
from services import RAGService
from text_processor import format_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FALLBACK_PHRASE = "I couldn't find this exact topic in the semester notes"


@lru_cache(maxsize=1)
def get_worker_rag_service() -> RAGService:
    """Create one RAG service per worker process."""

    return RAGService()


def process_chat_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a queued chat job using the existing RAG pipeline."""

    configure_logfire("qeds-gpt-worker")
    job = get_current_job()
    job_id = job.id if job else "unknown"
    logger.info("Job started: %s", job_id)

    if job:
        job.meta["status"] = "running"
        job.save_meta()

    try:
        message = str(payload["message"])
        semester_filter = payload.get("semester_filter")
        context_messages = payload.get("history", [])

        rag_service = get_worker_rag_service()
        with logfire_span("qeds.worker.classify_query", job_id=job_id, history_count=len(context_messages)):
            category = rag_service.classify_query(message, context_messages)
        immediate_response = rag_service.get_immediate_response(category, message, context_messages)

        if immediate_response is not None:
            logger.info("LLM completed for immediate job: %s", job_id)
            return {
                "answer": immediate_response,
                "category": category,
                "used_retrieval": False,
                "sources": [],
                "retrieved_chunks": [],
            }

        docs = []
        if category == QueryCategory.ACADEMIC:
            with logfire_span("qeds.worker.retrieve", job_id=job_id, semester_filter=semester_filter):
                docs = rag_service.retrieve(message, semester_filter)
            logger.info("Retrieval completed for job %s with %s document(s)", job_id, len(docs))

        with logfire_span("qeds.worker.generate_answer", job_id=job_id, category=category, doc_count=len(docs)):
            answer = rag_service.build_academic_answer(message, docs, context_messages)
        logger.info("LLM completed for job: %s", job_id)

        used_retrieval = bool(docs) and FALLBACK_PHRASE not in answer
        result = {
            "answer": answer,
            "category": category,
            "used_retrieval": used_retrieval,
            "sources": _build_sources(docs, used_retrieval),
            "retrieved_chunks": _build_retrieved_chunks(docs, used_retrieval),
        }

        if job:
            job.meta["status"] = "completed"
            job.save_meta()

        logger.info("Job finished: %s", job_id)
        return result
    except Exception as exc:
        logger.error("Job failed: %s", job_id, exc_info=True)
        if job:
            job.meta["status"] = "error"
            job.meta["error"] = str(exc)
            job.save_meta()
        raise


def _build_sources(docs: list, used_retrieval: bool) -> list[dict[str, str]]:
    """Build unique source labels from retrieved documents."""

    if not used_retrieval:
        return []

    source_labels = list(dict.fromkeys(format_source(doc.metadata) for doc in docs))
    return [{"label": label} for label in source_labels]


def _build_retrieved_chunks(docs: list, used_retrieval: bool) -> list[dict[str, Any]]:
    """Build retrieved chunk previews for API clients."""

    if not used_retrieval:
        return []

    return [
        {
            "index": index,
            "source": format_source(doc.metadata),
            "content_preview": f"{doc.page_content[:1000]}...",
            "metadata": doc.metadata,
        }
        for index, doc in enumerate(docs, 1)
    ]


def main() -> None:
    """Start an RQ worker for the configured queue."""

    configure_logfire("qeds-gpt-worker")
    queue = get_queue()
    connection = get_redis_connection()
    logger.info("Worker listening on queue: %s", queue.name)
    worker_class = SimpleWorker if os.name == "nt" else Worker
    logger.info("Using worker class: %s", worker_class.__name__)
    worker = worker_class([queue], connection=connection)
    if os.name == "nt":
        worker.death_penalty_class = TimerDeathPenalty
    worker.work()


if __name__ == "__main__":
    main()
