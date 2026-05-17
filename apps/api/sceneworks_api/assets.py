from __future__ import annotations

import json
import mimetypes
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .projects import find_project_path, utc_now


router = APIRouter(prefix="/projects/{project_id}", tags=["assets"])

ASSET_SIDECAR_PATTERN = "*.sceneworks.json"
MEDIA_FOLDERS = ("assets/images", "assets/videos", "assets/uploads", "assets/frames", "assets/renders")
ALLOWED_IMPORT_PREFIXES = ("image/", "video/")


class AssetStatusUpdate(BaseModel):
    favorite: bool | None = None
    rating: int | None = Field(default=None, ge=0, le=5)
    rejected: bool | None = None
    trashed: bool | None = None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def safe_filename(value: str, fallback: str = "upload") -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(name).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem.strip()).strip("-").lower()
    return slug[:64] or fallback


def media_type_for_mime(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"
    raise HTTPException(status_code=400, detail="Only image and video uploads are supported")


def index_asset_db(project_path: Path, asset: dict[str, Any]) -> None:
    db_path = project_path / "project.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            create table if not exists assets (
              id text primary key,
              type text not null,
              display_name text not null,
              file_path text not null,
              generation_set_id text,
              created_at text not null,
              favorite integer not null default 0,
              rating integer not null default 0,
              rejected integer not null default 0,
              trashed integer not null default 0
            )
            """
        )
        connection.execute(
            """
            insert or replace into assets (
              id, type, display_name, file_path, generation_set_id, created_at,
              favorite, rating, rejected, trashed
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset["id"],
                asset["type"],
                asset["displayName"],
                asset["file"]["path"],
                asset.get("generationSetId"),
                asset["createdAt"],
                int(asset["status"]["favorite"]),
                int(asset["status"]["rating"]),
                int(asset["status"]["rejected"]),
                int(asset["status"]["trashed"]),
            ),
        )


def normalize_asset(project_id: str, project_path: Path, sidecar_path: Path) -> dict[str, Any]:
    asset = read_json(sidecar_path)
    rel_media = asset.get("file", {}).get("path", "")
    if rel_media:
        normalized_path = rel_media.replace("\\", "/")
        asset["url"] = f"/api/v1/projects/{project_id}/files/{normalized_path}"
    asset["sidecarPath"] = str(sidecar_path.relative_to(project_path)).replace("\\", "/")
    return asset


def find_asset_sidecar(project_path: Path, asset_id: str) -> Path:
    for folder in MEDIA_FOLDERS:
        for sidecar_path in (project_path / folder).glob(ASSET_SIDECAR_PATTERN):
            try:
                payload = read_json(sidecar_path)
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("id") == asset_id:
                return sidecar_path
    raise HTTPException(status_code=404, detail="Asset not found")


@router.get("/assets")
def list_assets(
    project_id: str,
    request: Request,
    includeRejected: bool = Query(default=False),
    includeTrashed: bool = Query(default=False),
) -> list[dict[str, Any]]:
    project_path = find_project_path(request.app.state.settings, project_id)
    assets = []
    for folder in MEDIA_FOLDERS:
        for sidecar_path in (project_path / folder).glob(ASSET_SIDECAR_PATTERN):
            try:
                asset = normalize_asset(project_id, project_path, sidecar_path)
            except (OSError, json.JSONDecodeError):
                continue
            status = asset.get("status", {})
            if status.get("rejected") and not includeRejected:
                continue
            if status.get("trashed") and not includeTrashed:
                continue
            assets.append(asset)

    return sorted(assets, key=lambda item: item.get("createdAt", ""), reverse=True)


@router.post("/assets", status_code=201)
def import_asset(project_id: str, request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    upload_dir = project_path / "assets" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    guessed_mime, _ = mimetypes.guess_type(file.filename or "")
    content_type = file.content_type or ""
    mime_type = guessed_mime if content_type in {"", "application/octet-stream"} else content_type
    mime_type = mime_type or "application/octet-stream"
    if not mime_type.startswith(ALLOWED_IMPORT_PREFIXES):
        raise HTTPException(status_code=400, detail="Only image and video uploads are supported")

    asset_id = f"asset_{uuid4().hex}"
    created_at = utc_now()
    extension = Path(file.filename or "").suffix.lower() or mimetypes.guess_extension(mime_type) or ".bin"
    filename = f"{safe_filename(file.filename or '', asset_id)}-{asset_id[-8:]}{extension}"
    media_path = upload_dir / filename
    media_rel = str(media_path.relative_to(project_path)).replace("\\", "/")

    try:
        with media_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    finally:
        file.file.close()

    if media_path.stat().st_size == 0:
        media_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    asset = {
        "schemaVersion": 1,
        "id": asset_id,
        "projectId": project_id,
        "generationSetId": None,
        "type": media_type_for_mime(mime_type),
        "displayName": Path(file.filename or filename).name,
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": mime_type,
            "width": None,
            "height": None,
            "duration": None,
            "fps": None,
        },
        "status": {
            "favorite": False,
            "rating": 0,
            "rejected": False,
            "trashed": False,
        },
        "recipe": {
            "mode": "upload",
            "model": "manual-import",
            "adapter": "api-upload",
            "prompt": file.filename or filename,
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {},
            "rawAdapterSettings": {"contentType": mime_type},
        },
        "lineage": {
            "parents": [],
            "sourceAssetId": None,
            "sourceTimestamp": None,
            "jobId": None,
        },
    }
    sidecar_path = media_path.with_suffix(".sceneworks.json")
    write_json(sidecar_path, asset)
    index_asset_db(project_path, asset)
    return normalize_asset(project_id, project_path, sidecar_path)


@router.patch("/assets/{asset_id}/status")
def update_asset_status(
    project_id: str,
    asset_id: str,
    payload: AssetStatusUpdate,
    request: Request,
) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    sidecar_path = find_asset_sidecar(project_path, asset_id)
    asset = read_json(sidecar_path)
    status = asset.setdefault("status", {})
    changes = payload.model_dump(exclude_none=True)
    status.update(changes)
    write_json(sidecar_path, asset)
    return normalize_asset(project_id, project_path, sidecar_path)


@router.delete("/assets/{asset_id}")
def delete_asset(project_id: str, asset_id: str, request: Request) -> dict[str, str]:
    project_path = find_project_path(request.app.state.settings, project_id)
    sidecar_path = find_asset_sidecar(project_path, asset_id)
    asset = read_json(sidecar_path)
    media_path = project_path / asset.get("file", {}).get("path", "")

    if media_path.exists() and media_path.is_file():
        media_path.unlink()
    sidecar_path.unlink()
    return {"id": asset_id, "status": "deleted"}


@router.get("/files/{relative_path:path}")
def get_project_file(project_id: str, relative_path: str, request: Request) -> FileResponse:
    project_path = find_project_path(request.app.state.settings, project_id)
    target = (project_path / relative_path).resolve()
    try:
        target.relative_to(project_path.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project file path") from None
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)
