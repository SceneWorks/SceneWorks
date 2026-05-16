from pathlib import Path
import sqlite3
import sys
import tempfile

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.sceneworks_api.main import create_app
from apps.api.sceneworks_api.persistence import PROJECT_FOLDERS
from apps.api.sceneworks_api.settings import Settings


def assert_status(response, status_code):
    if response.status_code != status_code:
        raise AssertionError(f"Expected {status_code}, got {response.status_code}: {response.text}")


with tempfile.TemporaryDirectory() as temp_dir:
    root = Path(temp_dir)
    settings = Settings()
    settings.data_dir = root / "data"
    settings.config_dir = root / "config"
    settings.access_token = ""

    client = TestClient(create_app(settings))

    created_response = client.post("/api/v1/projects", json={"name": "Epic Spine"})
    assert_status(created_response, 201)
    project = created_response.json()
    project_path = Path(project["path"])
    assert (project_path / "project.json").exists()
    assert (project_path / "project.db").exists()
    for folder in PROJECT_FOLDERS:
        assert (project_path / folder).is_dir(), folder

    opened_response = client.post("/api/v1/projects/open", json={"path": str(project_path)})
    assert_status(opened_response, 200)
    assert opened_response.json()["id"] == project["id"]

    import_response = client.post(
        f"/api/v1/projects/{project['id']}/assets/import",
        files=[
            ("files", ("noir still.png", b"\x89PNG\r\n\x1a\nsceneworks", "image/png")),
            ("files", ("alley shot.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")),
        ],
    )
    assert_status(import_response, 201)
    imported_assets = import_response.json()["assets"]
    assert len(imported_assets) == 2
    assert {asset["type"] for asset in imported_assets} == {"image", "video"}
    for asset in imported_assets:
        media_path = project_path / asset["file"]["path"]
        sidecar_path = media_path.with_suffix(".sceneworks.json")
        assert media_path.exists()
        assert sidecar_path.exists()

    connection = sqlite3.connect(project_path / "project.db")
    try:
        count = connection.execute("select count(*) from assets").fetchone()[0]
    finally:
        connection.close()
    assert count == 2

    first_asset = imported_assets[0]
    patch_response = client.patch(
        f"/api/v1/projects/{project['id']}/assets/{first_asset['id']}",
        json={"rating": 5, "favorite": True, "rejected": True, "notes": "keeper after review"},
    )
    assert_status(patch_response, 200)
    updated_asset = patch_response.json()
    assert updated_asset["status"]["rating"] == 5
    assert updated_asset["status"]["favorite"] is True
    assert updated_asset["status"]["rejected"] is True
    assert updated_asset["notes"] == "keeper after review"

    hidden_response = client.get(f"/api/v1/projects/{project['id']}/assets")
    assert_status(hidden_response, 200)
    assert hidden_response.json()["total"] == 1

    shown_response = client.get(f"/api/v1/projects/{project['id']}/assets?includeRejected=true")
    assert_status(shown_response, 200)
    assert shown_response.json()["total"] == 2

    delete_response = client.delete(f"/api/v1/projects/{project['id']}/assets/{first_asset['id']}")
    assert_status(delete_response, 200)
    trashed_asset = delete_response.json()
    assert trashed_asset["status"]["trashed"] is True
    assert trashed_asset["file"]["path"].startswith("trash/")

    reindex_response = client.post(f"/api/v1/projects/{project['id']}/reindex")
    assert_status(reindex_response, 200)
    reindex = reindex_response.json()
    assert reindex["discovered"] == 2
    assert reindex["indexed"] == 2
    assert reindex["errors"] == []

print("SceneWorks API persistence check passed.")
