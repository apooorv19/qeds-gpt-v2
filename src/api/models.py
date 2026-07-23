"""Pydantic models for the QEDS-GPT API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


ChatRole = Literal["system", "user", "assistant"]
JobStatus = Literal["queued", "running", "completed", "error"]


class ChatMessage(BaseModel):
    """A single chat message supplied by an API client."""

    role: ChatRole = Field(..., description="Message role.")
    content: str = Field(..., min_length=1, description="Message content.")


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(..., min_length=1, description="The user's current message.")
    semester_filter: int | None = Field(
        default=None,
        ge=1,
        le=6,
        description="Optional semester filter from 1 to 6.",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous chat turns. APIs are stateless, so clients send history explicitly.",
    )


class Source(BaseModel):
    """A human-readable retrieved source."""

    label: str


class RetrievedChunk(BaseModel):
    """A retrieved chunk returned for inspection or debugging."""

    index: int
    source: str
    content_preview: str
    metadata: dict[str, Any]


class ChatJobResponse(BaseModel):
    """Response body returned immediately after a chat job is queued."""

    job_id: str
    status: JobStatus


class ChatResultResponse(BaseModel):
    """Response body for polling a chat job result."""

    job_id: str
    status: JobStatus
    answer: str | None = None
    category: str | None = None
    used_retrieval: bool = False
    sources: list[Source] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    error: str | None = None


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str


class QueueHealthResponse(BaseModel):
    """Queue health endpoint response."""

    status: str
    queue_name: str
    queued_jobs: int


class WorkerHealthResponse(BaseModel):
    """Worker health endpoint response."""

    status: str
    worker_count: int


class VersionResponse(BaseModel):
    """Version endpoint response."""

    app_name: str
    version: str
    api_version: str
