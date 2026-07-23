"""Redis Queue configuration for asynchronous RAG jobs."""

import os

from redis import Redis
from rq import Queue
from rq.job import Job

DEFAULT_QUEUE_NAME = "qeds-rag-jobs"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def get_redis_connection() -> Redis:
    """Create a Redis connection from environment configuration."""

    redis_url = os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    return Redis.from_url(redis_url)


def get_queue() -> Queue:
    """Return the configured RQ queue."""

    queue_name = os.environ.get("RQ_QUEUE_NAME", DEFAULT_QUEUE_NAME)
    return Queue(name=queue_name, connection=get_redis_connection())


def get_job(job_id: str) -> Job:
    """Fetch an RQ job by id."""

    return Job.fetch(job_id, connection=get_redis_connection())


def get_job_timeout_seconds() -> int:
    """Return the maximum runtime for a single RAG job."""

    return int(os.environ.get("RQ_JOB_TIMEOUT_SECONDS", "600"))


def get_result_ttl_seconds() -> int:
    """Return how long completed job results remain available in Redis."""

    return int(os.environ.get("RQ_RESULT_TTL_SECONDS", "3600"))
