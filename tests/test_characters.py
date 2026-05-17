from __future__ import annotations

import json
from types import SimpleNamespace

from sceneworks_api.characters import (
    CharacterCreate,
    CharacterLoraRequest,
    CharacterLookRequest,
    CharacterReferenceRequest,
    CharacterReferenceUpdate,
    add_reference,
    archive_character,
    attach_lora,
    create_character,
    create_look,
    list_characters,
    update_reference,
)
from sceneworks_shared import index_asset, read_json, write_json


def request_for_project(tmp_path, project_path, jobs_store=None):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    registry_path = data_dir / "recent-projects.json"
    registry_path.write_text(
        json.dumps([{"id": "project-1", "name": "Project", "path": str(project_path)}]),
        encoding="utf-8",
    )
    settings = SimpleNamespace(registry_path=registry_path)
    state = SimpleNamespace(settings=settings)
    if jobs_store is not None:
        state.jobs_store = jobs_store
        state.event_hub = SimpleNamespace(publish=lambda *_args, **_kwargs: None)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def write_asset(project_path, asset_id="asset-1"):
    image_dir = project_path / "assets" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    asset = {
        "id": asset_id,
        "projectId": "project-1",
        "type": "image",
        "displayName": "Reference",
        "createdAt": "2026-05-17T00:00:00Z",
        "generationSetId": None,
        "file": {"path": f"assets/images/{asset_id}.png", "mimeType": "image/png"},
        "status": {"favorite": False, "rating": 0, "rejected": False, "trashed": False},
    }
    sidecar = image_dir / f"{asset_id}.sceneworks.json"
    write_json(sidecar, asset)
    index_asset(project_path, asset)
    return sidecar


def test_character_crud_reference_approval_and_archive(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    project_path.mkdir()
    asset_sidecar = write_asset(project_path)
    request = request_for_project(tmp_path, project_path)

    character = create_character("project-1", CharacterCreate(name="Mira", type="person"), request)
    assert character["name"] == "Mira"
    assert character["type"] == "person"
    assert (project_path / "characters" / f"{character['id']}.sceneworks.character.json").exists()

    with_reference = add_reference(
        "project-1",
        character["id"],
        CharacterReferenceRequest(assetId="asset-1", approved=False),
        request,
    )
    assert with_reference["references"][0]["asset"]["displayName"] == "Reference"

    updated = update_reference(
        "project-1",
        character["id"],
        "asset-1",
        CharacterReferenceUpdate(approved=True, role="hero"),
        request,
    )
    assert updated["approvedReferences"][0]["assetId"] == "asset-1"

    asset = read_json(asset_sidecar)
    assert asset["metadata"]["characterReferences"][0]["characterId"] == character["id"]
    assert asset["metadata"]["characterReferences"][0]["approved"] is True

    archive_character("project-1", character["id"], request)
    assert list_characters("project-1", request) == []
    assert len(list_characters("project-1", request, includeArchived=True)) == 1


def test_character_looks_and_lora_copy(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    project_path.mkdir()
    lora_source = tmp_path / "mira.safetensors"
    lora_source.write_bytes(b"lora")
    request = request_for_project(tmp_path, project_path)
    character = create_character("project-1", CharacterCreate(name="Mira"), request)

    with_look = create_look(
        "project-1",
        character["id"],
        CharacterLookRequest(name="Rain coat", approvedReferenceIds=["asset-1"], recipeSettings={"style": "noir"}),
        request,
    )
    assert with_look["looks"][0]["recipeSettings"] == {"style": "noir"}

    with_lora = attach_lora(
        "project-1",
        character["id"],
        CharacterLoraRequest(
            name="Mira LoRA",
            sourcePath=str(lora_source),
            compatibility={"families": ["sdxl"]},
            triggerWords=["mira"],
        ),
        request,
    )
    link = with_lora["loras"][0]
    assert link["copiedIntoProject"] is True
    assert link["projectPath"].startswith(f"loras/characters/{character['id']}/")
    assert (project_path / link["projectPath"]).read_bytes() == b"lora"
