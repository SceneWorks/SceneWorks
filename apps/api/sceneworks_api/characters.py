from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from sceneworks_shared import find_asset_sidecar_path, read_json, utc_now, write_json

from .jobs import queue_summary
from .projects import find_project_path


router = APIRouter(prefix="/projects/{project_id}/characters", tags=["characters"])

CHARACTER_SIDECAR_PATTERN = "*.sceneworks.character.json"
CharacterType = Literal["person", "creature", "object"]


class CharacterCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    type: CharacterType = "person"
    description: str = Field(default="", max_length=2000)


class CharacterUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    type: CharacterType | None = None
    description: str | None = Field(default=None, max_length=2000)
    archived: bool | None = None


class CharacterReferenceRequest(BaseModel):
    assetId: str = Field(min_length=1)
    approved: bool = False
    role: str = Field(default="reference", max_length=80)
    notes: str = Field(default="", max_length=1000)


class CharacterReferenceUpdate(BaseModel):
    approved: bool | None = None
    role: str | None = Field(default=None, max_length=80)
    notes: str | None = Field(default=None, max_length=1000)


class CharacterLookRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    approvedReferenceIds: list[str] = Field(default_factory=list)
    recipeSettings: dict[str, Any] = Field(default_factory=dict)


class CharacterLookUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    approvedReferenceIds: list[str] | None = None
    recipeSettings: dict[str, Any] | None = None


class CharacterLoraRequest(BaseModel):
    loraId: str | None = None
    name: str = Field(min_length=1, max_length=160)
    sourcePath: str | None = None
    triggerWords: list[str] = Field(default_factory=list)
    defaultWeight: float = Field(default=0.8, ge=-2, le=2)
    compatibility: dict[str, Any] = Field(default_factory=dict)
    scope: Literal["project", "global", "external"] = "project"


class CharacterLoraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    triggerWords: list[str] | None = None
    defaultWeight: float | None = Field(default=None, ge=-2, le=2)
    compatibility: dict[str, Any] | None = None
    scope: Literal["project", "global", "external"] | None = None


class CharacterTestRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    model: str = "z_image_turbo"
    count: int = Field(default=4, ge=1, le=8)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    requestedGpu: str = "auto"
    lookId: str | None = None


def character_dir(project_path: Path) -> Path:
    return project_path / "characters"


def character_file(project_path: Path, character_id: str) -> Path:
    return character_dir(project_path) / f"{character_id}.sceneworks.character.json"


def find_character_file(project_path: Path, character_id: str) -> Path:
    path = character_file(project_path, character_id)
    if path.exists():
        return path
    for candidate in character_dir(project_path).glob(CHARACTER_SIDECAR_PATTERN):
        try:
            if read_json(candidate).get("id") == character_id:
                return candidate
        except (OSError, json.JSONDecodeError):
            continue
    raise HTTPException(status_code=404, detail="Character not found")


def read_character(project_path: Path, character_id: str) -> dict[str, Any]:
    return read_json(find_character_file(project_path, character_id))


def write_character(project_path: Path, character: dict[str, Any]) -> dict[str, Any]:
    character["updatedAt"] = utc_now()
    write_json(character_file(project_path, character["id"]), character)
    return character


def asset_summary(project_id: str, project_path: Path, asset_id: str) -> dict[str, Any] | None:
    sidecar = find_asset_sidecar_path(project_path, asset_id)
    if sidecar is None:
        return None
    try:
        asset = read_json(sidecar)
    except (OSError, json.JSONDecodeError):
        return None
    file_path = asset.get("file", {}).get("path")
    if file_path:
        normalized_file_path = file_path.replace("\\", "/")
        asset["url"] = f"/api/v1/projects/{project_id}/files/{normalized_file_path}"
    return {
        "id": asset.get("id"),
        "type": asset.get("type"),
        "displayName": asset.get("displayName"),
        "url": asset.get("url"),
        "status": asset.get("status", {}),
        "file": asset.get("file", {}),
    }


def hydrate_character(project_id: str, project_path: Path, character: dict[str, Any]) -> dict[str, Any]:
    hydrated = dict(character)
    references = []
    for reference in hydrated.get("references", []):
        references.append({**reference, "asset": asset_summary(project_id, project_path, reference.get("assetId", ""))})
    hydrated["references"] = references
    hydrated["approvedReferences"] = [item for item in references if item.get("approved")]
    return hydrated


def update_asset_character_link(
    project_path: Path,
    character_id: str,
    reference: dict[str, Any],
    *,
    remove: bool = False,
) -> None:
    sidecar = find_asset_sidecar_path(project_path, reference["assetId"])
    if sidecar is None:
        raise HTTPException(status_code=404, detail="Reference asset not found")
    asset = read_json(sidecar)
    metadata = asset.setdefault("metadata", {})
    links = metadata.setdefault("characterReferences", [])
    links = [item for item in links if item.get("characterId") != character_id]
    if not remove:
        links.append(
            {
                "characterId": character_id,
                "approved": bool(reference.get("approved")),
                "role": reference.get("role", "reference"),
                "linkedAt": reference.get("addedAt") or utc_now(),
                "approvedAt": reference.get("approvedAt"),
            }
        )
    metadata["characterReferences"] = links
    write_json(sidecar, asset)


def copy_lora_into_project(project_path: Path, character_id: str, source_path_text: str | None) -> tuple[str | None, bool]:
    if not source_path_text:
        return None, False
    source_path = Path(source_path_text)
    if not source_path.exists() or not source_path.is_file():
        return source_path_text, False
    target_dir = project_path / "loras" / "characters" / character_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source_path.name
    if source_path.resolve() != target.resolve():
        shutil.copy2(source_path, target)
    return str(target.relative_to(project_path)).replace("\\", "/"), True


@router.get("")
def list_characters(
    project_id: str,
    request: Request,
    includeArchived: bool = False,
) -> list[dict[str, Any]]:
    project_path = find_project_path(request.app.state.settings, project_id)
    items = []
    for path in character_dir(project_path).glob(CHARACTER_SIDECAR_PATTERN):
        try:
            character = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if character.get("status", {}).get("archived") and not includeArchived:
            continue
        items.append(hydrate_character(project_id, project_path, character))
    return sorted(items, key=lambda item: item.get("updatedAt", ""), reverse=True)


@router.post("", status_code=201)
def create_character(project_id: str, payload: CharacterCreate, request: Request) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    now = utc_now()
    character_id = f"character_{uuid4().hex}"
    character = {
        "schemaVersion": 1,
        "id": character_id,
        "projectId": project_id,
        "name": payload.name.strip(),
        "type": payload.type,
        "description": payload.description.strip(),
        "createdAt": now,
        "updatedAt": now,
        "status": {"archived": False},
        "references": [],
        "looks": [],
        "loras": [],
        "trainedLoras": [],
    }
    write_json(character_file(project_path, character_id), character)
    return hydrate_character(project_id, project_path, character)


@router.get("/{character_id}")
def get_character(project_id: str, character_id: str, request: Request) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    return hydrate_character(project_id, project_path, read_character(project_path, character_id))


@router.patch("/{character_id}")
def update_character(
    project_id: str,
    character_id: str,
    payload: CharacterUpdate,
    request: Request,
) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    changes = payload.model_dump(exclude_none=True)
    if "name" in changes:
        character["name"] = changes["name"].strip()
    if "type" in changes:
        character["type"] = changes["type"]
    if "description" in changes:
        character["description"] = changes["description"].strip()
    if "archived" in changes:
        character.setdefault("status", {})["archived"] = changes["archived"]
    return hydrate_character(project_id, project_path, write_character(project_path, character))


@router.delete("/{character_id}")
def archive_character(project_id: str, character_id: str, request: Request) -> dict[str, str]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    character.setdefault("status", {})["archived"] = True
    write_character(project_path, character)
    return {"id": character_id, "status": "archived"}


@router.post("/{character_id}/references", status_code=201)
def add_reference(
    project_id: str,
    character_id: str,
    payload: CharacterReferenceRequest,
    request: Request,
) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    now = utc_now()
    references = [item for item in character.get("references", []) if item.get("assetId") != payload.assetId]
    reference = {
        "assetId": payload.assetId,
        "approved": payload.approved,
        "role": payload.role or "reference",
        "notes": payload.notes,
        "addedAt": now,
        "approvedAt": now if payload.approved else None,
    }
    update_asset_character_link(project_path, character_id, reference)
    character["references"] = [reference, *references]
    return hydrate_character(project_id, project_path, write_character(project_path, character))


@router.patch("/{character_id}/references/{asset_id}")
def update_reference(
    project_id: str,
    character_id: str,
    asset_id: str,
    payload: CharacterReferenceUpdate,
    request: Request,
) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    references = character.get("references", [])
    reference = next((item for item in references if item.get("assetId") == asset_id), None)
    if reference is None:
        raise HTTPException(status_code=404, detail="Reference not found")
    changes = payload.model_dump(exclude_none=True)
    if "approved" in changes:
        reference["approved"] = changes["approved"]
        reference["approvedAt"] = utc_now() if changes["approved"] else None
    if "role" in changes:
        reference["role"] = changes["role"] or "reference"
    if "notes" in changes:
        reference["notes"] = changes["notes"]
    update_asset_character_link(project_path, character_id, reference)
    return hydrate_character(project_id, project_path, write_character(project_path, character))


@router.delete("/{character_id}/references/{asset_id}")
def remove_reference(project_id: str, character_id: str, asset_id: str, request: Request) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    references = character.get("references", [])
    reference = next((item for item in references if item.get("assetId") == asset_id), None)
    if reference is None:
        raise HTTPException(status_code=404, detail="Reference not found")
    update_asset_character_link(project_path, character_id, reference, remove=True)
    character["references"] = [item for item in references if item.get("assetId") != asset_id]
    return hydrate_character(project_id, project_path, write_character(project_path, character))


@router.post("/{character_id}/looks", status_code=201)
def create_look(project_id: str, character_id: str, payload: CharacterLookRequest, request: Request) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    now = utc_now()
    look = {
        "id": f"look_{uuid4().hex}",
        "name": payload.name.strip(),
        "description": payload.description.strip(),
        "approvedReferenceIds": payload.approvedReferenceIds,
        "recipeSettings": payload.recipeSettings,
        "createdAt": now,
        "updatedAt": now,
    }
    character["looks"] = [look, *character.get("looks", [])]
    return hydrate_character(project_id, project_path, write_character(project_path, character))


@router.patch("/{character_id}/looks/{look_id}")
def update_look(
    project_id: str,
    character_id: str,
    look_id: str,
    payload: CharacterLookUpdate,
    request: Request,
) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    look = next((item for item in character.get("looks", []) if item.get("id") == look_id), None)
    if look is None:
        raise HTTPException(status_code=404, detail="Look not found")
    changes = payload.model_dump(exclude_none=True)
    look.update(changes)
    if "name" in changes:
        look["name"] = look["name"].strip()
    if "description" in changes:
        look["description"] = look["description"].strip()
    look["updatedAt"] = utc_now()
    return hydrate_character(project_id, project_path, write_character(project_path, character))


@router.delete("/{character_id}/looks/{look_id}")
def delete_look(project_id: str, character_id: str, look_id: str, request: Request) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    character["looks"] = [item for item in character.get("looks", []) if item.get("id") != look_id]
    return hydrate_character(project_id, project_path, write_character(project_path, character))


@router.post("/{character_id}/loras", status_code=201)
def attach_lora(project_id: str, character_id: str, payload: CharacterLoraRequest, request: Request) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    project_lora_path, copied = copy_lora_into_project(project_path, character_id, payload.sourcePath)
    now = utc_now()
    link = {
        "id": f"character_lora_{uuid4().hex}",
        "loraId": payload.loraId,
        "name": payload.name.strip(),
        "sourcePath": payload.sourcePath,
        "projectPath": project_lora_path,
        "copiedIntoProject": copied,
        "category": "character",
        "scope": payload.scope,
        "triggerWords": payload.triggerWords,
        "defaultWeight": payload.defaultWeight,
        "compatibility": payload.compatibility,
        "createdAt": now,
        "updatedAt": now,
    }
    character["loras"] = [link, *character.get("loras", [])]
    return hydrate_character(project_id, project_path, write_character(project_path, character))


@router.patch("/{character_id}/loras/{link_id}")
def update_lora(
    project_id: str,
    character_id: str,
    link_id: str,
    payload: CharacterLoraUpdate,
    request: Request,
) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    link = next((item for item in character.get("loras", []) if item.get("id") == link_id), None)
    if link is None:
        raise HTTPException(status_code=404, detail="Character LoRA not found")
    changes = payload.model_dump(exclude_none=True)
    link.update(changes)
    if "name" in changes:
        link["name"] = link["name"].strip()
    link["updatedAt"] = utc_now()
    return hydrate_character(project_id, project_path, write_character(project_path, character))


@router.delete("/{character_id}/loras/{link_id}")
def detach_lora(project_id: str, character_id: str, link_id: str, request: Request) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    character["loras"] = [item for item in character.get("loras", []) if item.get("id") != link_id]
    return hydrate_character(project_id, project_path, write_character(project_path, character))


@router.post("/{character_id}/test-jobs", status_code=201)
def create_character_test_job(
    project_id: str,
    character_id: str,
    payload: CharacterTestRequest,
    request: Request,
) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    character = read_character(project_path, character_id)
    look = next((item for item in character.get("looks", []) if item.get("id") == payload.lookId), None)
    job = request.app.state.jobs_store.create_job(
        job_type="image_generate",
        project_id=project_id,
        project_name=None,
        payload={
            "mode": "character_image",
            "prompt": payload.prompt,
            "negativePrompt": "",
            "model": payload.model,
            "count": payload.count,
            "seed": None,
            "width": payload.width,
            "height": payload.height,
            "stylePreset": "character-test",
            "sourceAssetId": None,
            "loras": character.get("loras", []),
            "characterId": character_id,
            "characterLookId": payload.lookId,
            "advanced": {
                "characterName": character.get("name"),
                "characterType": character.get("type"),
                "approvedReferenceIds": [item["assetId"] for item in character.get("references", []) if item.get("approved")],
                "look": look,
            },
        },
        requested_gpu=payload.requestedGpu,
    )
    request.app.state.event_hub.publish("job.updated", job)
    request.app.state.event_hub.publish("queue.updated", queue_summary(request))
    return job
