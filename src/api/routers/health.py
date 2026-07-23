"""Health-check routes."""

from fastapi import APIRouter, HTTPException, status
from rq import Worker

from api.models import HealthResponse, QueueHealthResponse, WorkerHealthResponse
from job_queue import get_queue, get_redis_connection

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return basic process health."""

    return HealthResponse(status="ok")


@router.get("/queue", response_model=QueueHealthResponse)
def queue_health_check() -> QueueHealthResponse:
    """Return Redis queue health."""

    try:
        connection = get_redis_connection()
        connection.ping()
        queue = get_queue()
        return QueueHealthResponse(
            status="ok",
            queue_name=queue.name,
            queued_jobs=len(queue),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Queue is unavailable: {exc}",
        ) from exc


@router.get("/worker", response_model=WorkerHealthResponse)
def worker_health_check() -> WorkerHealthResponse:
    """Return RQ worker health."""

    try:
        connection = get_redis_connection()
        workers = Worker.all(connection=connection)
        return WorkerHealthResponse(
            status="ok" if workers else "degraded",
            worker_count=len(workers),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Worker health could not be checked: {exc}",
        ) from exc
