"""Chat job routes for the QEDS-GPT API."""

import logging

from fastapi import APIRouter, HTTPException, status
from rq.exceptions import NoSuchJobError

from api.models import ChatJobResponse, ChatRequest, ChatResultResponse
from job_queue import get_job, get_job_timeout_seconds, get_queue, get_result_ttl_seconds
from observability import logfire_span

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatJobResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_chat_job(request: ChatRequest) -> ChatJobResponse:
    """Queue a chat request and return immediately with a job id."""

    try:
        queue = get_queue()
        with logfire_span("qeds.api.enqueue_chat_job", history_count=len(request.history)):
            job = queue.enqueue(
                "worker.process_chat_job",
                request.dict(),
                job_timeout=get_job_timeout_seconds(),
                result_ttl=get_result_ttl_seconds(),
                failure_ttl=get_result_ttl_seconds(),
            )
        logger.info("Job queued: %s", job.id)
        return ChatJobResponse(job_id=job.id, status="queued")
    except Exception as exc:
        logger.error("Failed to queue chat job: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat queue is unavailable.",
        ) from exc


@router.get("/result/{job_id}", response_model=ChatResultResponse)
def get_chat_result(job_id: str) -> ChatResultResponse:
    """Return the current status and result for a queued chat job."""

    try:
        job = get_job(job_id)
    except NoSuchJobError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        ) from exc

    with logfire_span("qeds.api.get_chat_result", job_id=job_id):
        status_name = _map_rq_status(job.get_status(refresh=True))

    if status_name == "completed":
        result = job.result or {}
        return ChatResultResponse(
            job_id=job.id,
            status="completed",
            answer=result.get("answer"),
            category=result.get("category"),
            used_retrieval=bool(result.get("used_retrieval", False)),
            sources=result.get("sources", []),
            retrieved_chunks=result.get("retrieved_chunks", []),
        )

    if status_name == "error":
        error = job.meta.get("error") or str(job.exc_info or "Job failed.")
        return ChatResultResponse(job_id=job.id, status="error", error=error)

    return ChatResultResponse(job_id=job.id, status=status_name)


def _map_rq_status(rq_status: str) -> str:
    """Map RQ's internal statuses to the public API statuses."""

    if rq_status == "finished":
        return "completed"
    if rq_status == "started":
        return "running"
    if rq_status in {"failed", "stopped", "canceled"}:
        return "error"
    return "queued"
