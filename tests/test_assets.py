from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sceneworks_api.assets import AssetStatusUpdate, get_project_file, index_asset_db, update_asset_status, write_json


def request_for_project(tmp_path, project_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    registry_path = data_dir / "recent-projects.json"
    registry_path.write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    settings = SimpleNamespace(registry_path=registry_path)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))


def test_project_file_rejects_path_traversal(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    project_path.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        get_project_file("project-1", "../outside.txt", request_for_project(tmp_path, project_path))

    assert exc_info.value.status_code == 400


def test_status_patch_updates_project_db(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    image_dir = project_path / "assets" / "images"
    image_dir.mkdir(parents=True)
    asset = {
        "id": "asset-1",
        "type": "image",
        "displayName": "Image",
        "createdAt": "2026-05-17T00:00:00Z",
        "generationSetId": None,
        "file": {"path": "assets/images/image.png"},
        "status": {"favorite": False, "rating": 0, "rejected": False, "trashed": False},
    }
    write_json(image_dir / "image.sceneworks.json", asset)
    index_asset_db(project_path, asset)

    update_asset_status(
        "project-1",
        "asset-1",
        AssetStatusUpdate(favorite=True, rating=4, rejected=True),
        request_for_project(tmp_path, project_path),
    )

    with sqlite3.connect(project_path / "project.db") as connection:
        row = connection.execute("select favorite, rating, rejected, trashed from assets where id = ?", ("asset-1",)).fetchone()

    assert row == (1, 4, 1, 0)
