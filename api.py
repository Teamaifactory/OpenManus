"""
FastAPI web server for OpenManus.

Exposes HTTP endpoints so users can submit video-creation (or any agent)
jobs through a web interface instead of the CLI, and poll for results.
"""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

import boto3
from botocore.client import Config as BotoConfig
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.logger import logger


# ---------------------------------------------------------------------------
# Job store (in-memory; replace with Redis / DB for multi-instance deploys)
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    prompt: str
    title: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    result: Optional[str] = None
    error: Optional[str] = None
    storage_url: Optional[str] = None


_jobs: Dict[str, Job] = {}


# ---------------------------------------------------------------------------
# Wasabi / S3-compatible storage helper
# ---------------------------------------------------------------------------

def _get_s3_client():
    """Return a boto3 S3 client pointed at Wasabi (or any S3-compatible store)."""
    endpoint = os.environ.get("WASABI_ENDPOINT_URL", "https://s3.wasabisys.com")
    region = os.environ.get("WASABI_REGION", "us-east-1")
    access_key = os.environ.get("WASABI_ACCESS_KEY_ID")
    secret_key = os.environ.get("WASABI_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        logger.warning(
            "Wasabi credentials not set (WASABI_ACCESS_KEY_ID / WASABI_SECRET_ACCESS_KEY). "
            "Storage upload will be skipped."
        )
        return None

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(signature_version="s3v4"),
    )


def _upload_to_wasabi(local_path: str, job_id: str) -> Optional[str]:
    """Upload a local file to Wasabi and return its public URL (or None on failure)."""
    bucket = os.environ.get("WASABI_BUCKET")
    if not bucket:
        logger.warning("WASABI_BUCKET not set — skipping upload.")
        return None

    client = _get_s3_client()
    if client is None:
        return None

    key = f"videos/{job_id}/{os.path.basename(local_path)}"
    try:
        client.upload_file(local_path, bucket, key)
        endpoint = os.environ.get("WASABI_ENDPOINT_URL", "https://s3.wasabisys.com")
        url = f"{endpoint}/{bucket}/{key}"
        logger.info(f"Uploaded {local_path} → {url}")
        return url
    except Exception as exc:
        logger.error(f"Wasabi upload failed for job {job_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Manus agent singleton
# ---------------------------------------------------------------------------

_agent_lock = asyncio.Lock()
_agent = None  # Initialised in lifespan


async def _get_agent():
    """Return the shared Manus agent, creating it if necessary."""
    global _agent
    if _agent is None:
        async with _agent_lock:
            if _agent is None:
                from app.agent.manus import Manus
                logger.info("Initialising Manus agent…")
                _agent = await Manus.create()
                logger.info("Manus agent ready.")
    return _agent


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Don't initialize agent on startup - do it lazily on first request
    yield

    # Shutdown: clean up agent resources.
    global _agent
    if _agent is not None:
        try:
            await _agent.cleanup()
            logger.info("Manus agent cleaned up.")
        except Exception as exc:
            logger.error(f"Error during Manus agent cleanup: {exc}")
        _agent = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenManus API",
    description="HTTP interface for the OpenManus agent — submit video-creation jobs and poll for results.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class VideoCreateRequest(BaseModel):
    prompt: str = Field(..., description="Natural-language description of the video to create")
    title: Optional[str] = Field(None, description="Optional title / filename hint for the output")
    extra: Optional[Dict[str, Any]] = Field(
        None,
        description="Any additional key-value parameters forwarded to the agent prompt",
    )


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    prompt: str
    title: Optional[str]
    created_at: str
    updated_at: str
    result: Optional[str]
    error: Optional[str]
    storage_url: Optional[str]


# ---------------------------------------------------------------------------
# Background task: run the agent
# ---------------------------------------------------------------------------

async def _run_agent_job(job_id: str, full_prompt: str) -> None:
    """Execute the Manus agent for a job and update the job store."""
    job = _jobs[job_id]
    job.status = JobStatus.RUNNING
    job.updated_at = datetime.now(timezone.utc).isoformat()

    try:
        agent = await _get_agent()

        # Each job needs a fresh agent state so previous memory doesn't bleed in.
        # We reset the agent's memory and step counter before running.
        agent.memory.clear()
        agent.current_step = 0
        from app.schema import AgentState
        agent.state = AgentState.IDLE

        logger.info(f"[job={job_id}] Starting agent run.")
        result = await agent.run(full_prompt)
        logger.info(f"[job={job_id}] Agent run completed.")

        job.result = result
        job.status = JobStatus.COMPLETED

        # Attempt to find and upload any generated video file.
        from app.config import WORKSPACE_ROOT
        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
        for candidate in sorted(WORKSPACE_ROOT.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if candidate.suffix.lower() in video_extensions:
                job.storage_url = _upload_to_wasabi(str(candidate), job_id)
                break

    except Exception as exc:
        logger.error(f"[job={job_id}] Agent run failed: {exc}", exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(exc)
    finally:
        job.updated_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health():
    """Health check — returns 200 when the server is up."""
    return {"status": "ok", "agent_ready": _agent is not None}


@app.post("/create-video", response_model=JobResponse, status_code=202, tags=["jobs"])
async def create_video(request: VideoCreateRequest, background_tasks: BackgroundTasks):
    """
    Submit a video-creation job.

    The agent runs asynchronously; the response contains a `job_id` you can
    use to poll `/status/{job_id}` for progress and the final result.
    """
    job_id = str(uuid.uuid4())

    # Build the full prompt sent to the agent.
    parts = [request.prompt]
    if request.title:
        parts.append(f"Title: {request.title}")
    if request.extra:
        for k, v in request.extra.items():
            parts.append(f"{k}: {v}")
    full_prompt = "\n".join(parts)

    job = Job(job_id=job_id, prompt=request.prompt, title=request.title)
    _jobs[job_id] = job

    background_tasks.add_task(_run_agent_job, job_id, full_prompt)

    logger.info(f"Accepted video-creation job {job_id}.")
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Job {job_id} accepted. Poll /status/{job_id} for updates.",
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_status(job_id: str):
    """
    Retrieve the current status and result of a job.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        prompt=job.prompt,
        title=job.title,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=job.result,
        error=job.error,
        storage_url=job.storage_url,
    )


@app.get("/jobs", tags=["jobs"])
async def list_jobs():
    """List all jobs (most recent first)."""
    sorted_jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "title": j.title,
            "created_at": j.created_at,
            "updated_at": j.updated_at,
        }
        for j in sorted_jobs
    ]
