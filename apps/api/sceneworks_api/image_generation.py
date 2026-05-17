from __future__ import annotations

import secrets
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from .jobs import queue_summary


router = APIRouter(prefix="/image", tags=["image"])

ImageMode = Literal["text_to_image", "edit_image", "character_image", "style_variations"]


class ImageJobRequest(BaseModel):
    projectId: str = Field(min_length=1)
    projectName: str | None = None
    mode: ImageMode = "text_to_image"
    prompt: str = Field(min_length=1, max_length=4000)
    negativePrompt: str = ""
    model: str = "z_image_turbo"
    count: int = Field(default=4, ge=1, le=8)
    seed: int | None = None
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    stylePreset: str = "cinematic"
    loras: list[dict[str, Any]] = Field(default_factory=list)
    sourceAssetId: str | None = None
    requestedGpu: str = "auto"
    advanced: dict[str, Any] = Field(default_factory=dict)


def random_image_seeds(count: int) -> list[int]:
    return [secrets.randbits(32) for _ in range(count)]


def image_job_payload(payload: ImageJobRequest) -> dict[str, Any]:
    job_payload = payload.model_dump(exclude={"requestedGpu"})
    if job_payload["seed"] is None:
        job_payload["seeds"] = random_image_seeds(payload.count)
    return job_payload


@router.post("/jobs", status_code=201)
def create_image_job(payload: ImageJobRequest, request: Request) -> dict:
    job_type = "image_edit" if payload.mode == "edit_image" else "image_generate"
    job = request.app.state.jobs_store.create_job(
        job_type=job_type,
        project_id=payload.projectId,
        project_name=payload.projectName,
        payload=image_job_payload(payload),
        requested_gpu=payload.requestedGpu,
    )
    request.app.state.event_hub.publish("job.updated", job)
    request.app.state.event_hub.publish("queue.updated", queue_summary(request))
    return job
