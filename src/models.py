from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: HealthStatus
    message: str
    timestamp: str
    service_name: str = "video-composer-job-service"


class JobStatus(str, Enum):
    """Job status enumeration."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """Job model."""

    id: str
    status: JobStatus
    progress: int = 0
    input_params: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class StartJobRequest(BaseModel):
    """Request to start a job."""

    job_id: str
    input_params: Optional[dict[str, Any]] = None


class ExtractFramesRequest(BaseModel):
    """Request to extract frames from a video."""

    input_file: str
    output_dir: str


class CancelJobRequest(BaseModel):
    """Request to cancel a job."""

    pass
