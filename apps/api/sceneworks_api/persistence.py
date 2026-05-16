from datetime import UTC, datetime
from contextlib import contextmanager
import json
import mimetypes
import re
import shutil
import sqlite3
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from .models import (
    SCHEMA_VERSION,
    AssetFileMetadata,
    AssetLineage,
    AssetSidecar,
    AssetStatus,
    AssetSummary,
    ProjectDocument,
    ProjectSummary,
    Recipe,
)
from .settings import Settings


PROJECT_FOLDERS = [
    "assets/images",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "characters",
    "loras",
    "recipes",
    "timelines",
    "trash",
    "cache",
]

ASSET_DIR_BY_TYPE = {
    "image": "assets/images",
    "video": "assets/videos",
    "upload": "assets/uploads",
    "frame": "assets/frames",
    "render": "assets/renders",
    "character_reference": "characters",
}

IMAGE_EXTENSIONS = {".apng", ".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "asset"


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{path.name} contains invalid JSON: {exc.msg}") from exc


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def relative_to_project(project_path: Path, path: Path) -> str:
    return path.relative_to(project_path).as_posix()


def ensure_data_dirs(settings: Settings) -> None:
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("models", "loras", "cache"):
        (settings.data_dir / folder).mkdir(parents=True, exist_ok=True)


def ensure_project_structure(project_path: Path) -> None:
    project_path.mkdir(parents=True, exist_ok=True)
    for folder in PROJECT_FOLDERS:
        (project_path / folder).mkdir(parents=True, exist_ok=True)


def load_registry(settings: Settings) -> list[dict]:
    if not settings.registry_path.exists():
        return []
    payload = read_json(settings.registry_path)
    if isinstance(payload, list):
        return payload
    raise HTTPException(status_code=422, detail="Recent projects registry must be a list")


def save_registry(settings: Settings, projects: list[dict]) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings.registry_path, projects[:20])


def touch_registry(settings: Settings, project: ProjectSummary) -> None:
    registry = [item for item in load_registry(settings) if item.get("id") != project.id]
    registry.insert(
        0,
        {
            "id": project.id,
            "name": project.name,
            "path": project.path,
            "lastOpenedAt": utc_now(),
        },
    )
    save_registry(settings, registry)


def db_path(project_path: Path) -> Path:
    return project_path / "project.db"


@contextmanager
def connect_project_db(project_path: Path):
    connection = sqlite3.connect(db_path(project_path))
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def migrate_project_db(project_path: Path) -> list[str]:
    applied: list[str] = []
    with connect_project_db(project_path) as connection:
        connection.execute(
            """
            create table if not exists project_metadata (
              key text primary key,
              value text not null
            )
            """
        )
        connection.execute(
            """
            create table if not exists assets (
              id text primary key,
              project_id text not null,
              generation_set_id text,
              type text not null,
              display_name text not null,
              relative_path text not null unique,
              sidecar_path text not null unique,
              mime_type text not null,
              size_bytes integer not null,
              created_at text not null,
              updated_at text not null,
              favorite integer not null default 0,
              rating integer not null default 0,
              rejected integer not null default 0,
              trashed integer not null default 0,
              notes text not null default '',
              prompt text,
              model text,
              lineage_json text not null default '{}'
            )
            """
        )
        connection.execute(
            """
            create table if not exists generation_sets (
              id text primary key,
              project_id text not null,
              created_at text not null,
              mode text not null,
              recipe_id text
            )
            """
        )
        connection.execute(
            """
            create table if not exists jobs (
              id text primary key,
              project_id text,
              type text not null,
              status text not null,
              progress real not null default 0,
              payload_json text not null default '{}',
              error text,
              created_at text not null,
              updated_at text not null
            )
            """
        )
        connection.execute(
            """
            create table if not exists characters (
              id text primary key,
              project_id text not null,
              name text not null,
              type text not null,
              document_json text not null,
              created_at text not null,
              updated_at text not null
            )
            """
        )
        connection.execute(
            """
            create table if not exists timelines (
              id text primary key,
              project_id text not null,
              name text not null,
              document_json text not null,
              created_at text not null,
              updated_at text not null
            )
            """
        )
        connection.execute("create index if not exists idx_assets_type on assets(type)")
        connection.execute("create index if not exists idx_assets_status on assets(rejected, trashed, favorite)")
        connection.execute("create index if not exists idx_assets_created_at on assets(created_at)")
        connection.execute(
            "insert or replace into project_metadata (key, value) values (?, ?)",
            ("schemaVersion", str(SCHEMA_VERSION)),
        )
        connection.execute("pragma user_version = 1")
        applied.append("sqlite_schema_v1")
    return applied


def create_project_document(settings: Settings, project_path: Path, project_id: str, name: str) -> ProjectDocument:
    now = utc_now()
    document = ProjectDocument(
        appVersion=settings.app_version,
        id=project_id,
        name=name,
        createdAt=now,
        updatedAt=now,
        folders={folder.replace("/", "_").replace("-", "_"): folder for folder in PROJECT_FOLDERS},
    )
    write_json(project_path / "project.json", document.model_dump(mode="json"))
    return document


def load_project_document(project_path: Path) -> ProjectDocument:
    project_file = project_path / "project.json"
    if not project_file.exists():
        raise HTTPException(status_code=404, detail="Project file not found")
    try:
        return ProjectDocument.model_validate(read_json(project_file))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid project.json: {exc}") from exc


def project_asset_count(project_path: Path) -> int:
    if not db_path(project_path).exists():
        return 0
    with connect_project_db(project_path) as connection:
        row = connection.execute("select count(*) as count from assets where trashed = 0").fetchone()
    return int(row["count"])


def read_project_summary(project_path: Path) -> ProjectSummary:
    document = load_project_document(project_path)
    return ProjectSummary(
        id=document.id,
        name=document.name,
        path=str(project_path),
        createdAt=document.createdAt,
        updatedAt=document.updatedAt,
        assetCount=project_asset_count(project_path),
    )


def create_project(settings: Settings, name: str) -> ProjectSummary:
    ensure_data_dirs(settings)
    project_id = f"project_{uuid4().hex}"
    project_path = settings.projects_dir / f"{slugify(name)}.sceneworks"
    if project_path.exists():
        project_path = settings.projects_dir / f"{slugify(name)}-{project_id[-8:]}.sceneworks"
    ensure_project_structure(project_path)
    create_project_document(settings, project_path, project_id, name)
    migrate_project_db(project_path)
    summary = read_project_summary(project_path)
    touch_registry(settings, summary)
    return summary


def open_project(settings: Settings, project_path: Path) -> ProjectSummary:
    resolved = project_path.expanduser().resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Project folder not found")
    ensure_project_structure(resolved)
    summary = read_project_summary(resolved)
    migrate_project_db(resolved)
    touch_registry(settings, summary)
    return read_project_summary(resolved)


def find_project_path(settings: Settings, project_id: str) -> Path:
    for item in load_registry(settings):
        if item.get("id") == project_id:
            path = Path(item["path"])
            if path.exists():
                return path
    raise HTTPException(status_code=404, detail="Project not found")


def classify_upload(filename: str, content_type: str | None) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    mime_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    if mime_type.startswith("image/") or suffix in IMAGE_EXTENSIONS:
        return "image", mime_type
    if mime_type.startswith("video/") or suffix in VIDEO_EXTENSIONS:
        return "video", mime_type
    return "upload", mime_type


def asset_summary_from_sidecar(project_id: str, sidecar: AssetSidecar) -> AssetSummary:
    return AssetSummary(
        id=sidecar.id,
        projectId=project_id,
        generationSetId=sidecar.generationSetId,
        type=sidecar.type,
        displayName=sidecar.displayName,
        createdAt=sidecar.createdAt,
        updatedAt=sidecar.updatedAt,
        file=sidecar.file,
        status=sidecar.status,
        notes=sidecar.notes,
        recipe=sidecar.recipe,
        lineage=sidecar.lineage,
        previewUrl=f"/api/v1/projects/{project_id}/assets/{sidecar.id}/content",
    )


def load_sidecar(path: Path) -> AssetSidecar:
    try:
        return AssetSidecar.model_validate(read_json(path))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid asset sidecar {path.name}: {exc}") from exc


def sidecar_path_for_media(media_path: Path) -> Path:
    return media_path.with_suffix(".sceneworks.json")


def upsert_asset_index(project_path: Path, sidecar: AssetSidecar) -> None:
    with connect_project_db(project_path) as connection:
        connection.execute(
            """
            insert into assets (
              id, project_id, generation_set_id, type, display_name, relative_path, sidecar_path,
              mime_type, size_bytes, created_at, updated_at, favorite, rating, rejected, trashed,
              notes, prompt, model, lineage_json
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(id) do update set
              generation_set_id = excluded.generation_set_id,
              type = excluded.type,
              display_name = excluded.display_name,
              relative_path = excluded.relative_path,
              sidecar_path = excluded.sidecar_path,
              mime_type = excluded.mime_type,
              size_bytes = excluded.size_bytes,
              updated_at = excluded.updated_at,
              favorite = excluded.favorite,
              rating = excluded.rating,
              rejected = excluded.rejected,
              trashed = excluded.trashed,
              notes = excluded.notes,
              prompt = excluded.prompt,
              model = excluded.model,
              lineage_json = excluded.lineage_json
            """,
            (
                sidecar.id,
                sidecar.projectId,
                sidecar.generationSetId,
                sidecar.type,
                sidecar.displayName,
                sidecar.file.path,
                f"{sidecar.file.path.rsplit('.', 1)[0]}.sceneworks.json",
                sidecar.file.mimeType,
                sidecar.file.sizeBytes,
                sidecar.createdAt,
                sidecar.updatedAt,
                int(sidecar.status.favorite),
                sidecar.status.rating,
                int(sidecar.status.rejected),
                int(sidecar.status.trashed),
                sidecar.notes,
                sidecar.recipe.prompt if sidecar.recipe else None,
                sidecar.recipe.model if sidecar.recipe else None,
                json.dumps(sidecar.lineage.model_dump(mode="json")),
            ),
        )


def import_asset(settings: Settings, project_id: str, upload: UploadFile) -> AssetSummary:
    project_path = find_project_path(settings, project_id)
    project = load_project_document(project_path)
    migrate_project_db(project_path)

    asset_type, mime_type = classify_upload(upload.filename or "upload.bin", upload.content_type)
    destination_dir = project_path / ASSET_DIR_BY_TYPE[asset_type]
    destination_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(upload.filename or "upload").stem
    extension = Path(upload.filename or "").suffix.lower() or mimetypes.guess_extension(mime_type) or ".bin"
    asset_id = f"asset_{uuid4().hex}"
    date_prefix = datetime.now(UTC).strftime("%Y-%m-%d")
    filename = f"{date_prefix}_{slugify(original_name)}_{asset_id[-8:]}{extension}"
    media_path = destination_dir / filename
    with media_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)

    relative_path = relative_to_project(project_path, media_path)
    now = utc_now()
    sidecar = AssetSidecar(
        appVersion=settings.app_version,
        id=asset_id,
        projectId=project.id,
        type=asset_type,
        displayName=Path(upload.filename or filename).stem,
        createdAt=now,
        updatedAt=now,
        file=AssetFileMetadata(
            path=relative_path,
            mimeType=mime_type,
            sizeBytes=media_path.stat().st_size,
        ),
        status=AssetStatus(),
        notes="",
        recipe=Recipe(appVersion=settings.app_version, id=None),
        lineage=AssetLineage(),
    )
    write_json(sidecar_path_for_media(media_path), sidecar.model_dump(mode="json"))
    upsert_asset_index(project_path, sidecar)
    update_project_timestamp(project_path)
    return asset_summary_from_sidecar(project.id, sidecar)


def update_project_timestamp(project_path: Path) -> None:
    document = load_project_document(project_path)
    document.updatedAt = utc_now()
    write_json(project_path / "project.json", document.model_dump(mode="json"))


def get_asset_sidecar_path(project_path: Path, asset_id: str) -> Path:
    with connect_project_db(project_path) as connection:
        row = connection.execute("select sidecar_path from assets where id = ?", (asset_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Asset not found")
    return project_path / row["sidecar_path"]


def get_asset(project_path: Path, asset_id: str) -> AssetSummary:
    sidecar = load_sidecar(get_asset_sidecar_path(project_path, asset_id))
    return asset_summary_from_sidecar(sidecar.projectId, sidecar)


def list_assets(
    project_path: Path,
    asset_type: str | None,
    include_rejected: bool,
    include_trashed: bool,
    favorites_only: bool,
    search: str | None,
    sort: str,
) -> tuple[list[AssetSummary], int]:
    query = "select sidecar_path from assets where 1 = 1"
    params: list[object] = []
    if asset_type:
        query += " and type = ?"
        params.append(asset_type)
    if not include_rejected:
        query += " and rejected = 0"
    if not include_trashed:
        query += " and trashed = 0"
    if favorites_only:
        query += " and favorite = 1"
    if search:
        query += " and (display_name like ? or notes like ? or prompt like ? or model like ?)"
        needle = f"%{search}%"
        params.extend([needle, needle, needle, needle])

    order_by = {
        "oldest": "created_at asc",
        "rating": "rating desc, created_at desc",
        "type": "type asc, created_at desc",
        "name": "display_name collate nocase asc",
    }.get(sort, "created_at desc")
    query += f" order by {order_by}"

    with connect_project_db(project_path) as connection:
        rows = connection.execute(query, params).fetchall()

    assets: list[AssetSummary] = []
    for row in rows:
        sidecar = load_sidecar(project_path / row["sidecar_path"])
        assets.append(asset_summary_from_sidecar(sidecar.projectId, sidecar))
    return assets, len(assets)


def update_asset(project_path: Path, asset_id: str, **updates: object) -> AssetSummary:
    sidecar_path = get_asset_sidecar_path(project_path, asset_id)
    sidecar = load_sidecar(sidecar_path)
    if updates.get("displayName") is not None:
        sidecar.displayName = str(updates["displayName"])
    if updates.get("favorite") is not None:
        sidecar.status.favorite = bool(updates["favorite"])
    if updates.get("rating") is not None:
        sidecar.status.rating = int(updates["rating"])
    if updates.get("rejected") is not None:
        sidecar.status.rejected = bool(updates["rejected"])
    if updates.get("notes") is not None:
        sidecar.notes = str(updates["notes"])
    sidecar.updatedAt = utc_now()
    write_json(sidecar_path, sidecar.model_dump(mode="json"))
    upsert_asset_index(project_path, sidecar)
    update_project_timestamp(project_path)
    return asset_summary_from_sidecar(sidecar.projectId, sidecar)


def trash_asset(project_path: Path, asset_id: str, hard_delete: bool = False) -> AssetSummary | None:
    sidecar_path = get_asset_sidecar_path(project_path, asset_id)
    sidecar = load_sidecar(sidecar_path)
    media_path = project_path / sidecar.file.path
    if hard_delete and not sidecar.lineage.parents and not sidecar.generationSetId:
        media_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        with connect_project_db(project_path) as connection:
            connection.execute("delete from assets where id = ?", (asset_id,))
        update_project_timestamp(project_path)
        return None

    trash_dir = project_path / "trash" / sidecar.type
    trash_dir.mkdir(parents=True, exist_ok=True)
    trashed_media = trash_dir / media_path.name
    trashed_sidecar = sidecar_path_for_media(trashed_media)
    if media_path.exists():
        shutil.move(str(media_path), str(trashed_media))
    if sidecar_path.exists():
        shutil.move(str(sidecar_path), str(trashed_sidecar))
    sidecar.file.path = relative_to_project(project_path, trashed_media)
    sidecar.status.trashed = True
    sidecar.updatedAt = utc_now()
    write_json(trashed_sidecar, sidecar.model_dump(mode="json"))
    upsert_asset_index(project_path, sidecar)
    update_project_timestamp(project_path)
    return asset_summary_from_sidecar(sidecar.projectId, sidecar)


def reindex_project(project_path: Path) -> tuple[int, int, list[str], list[str]]:
    migrations = migrate_project_db(project_path)
    discovered = 0
    indexed = 0
    errors: list[str] = []
    for sidecar_path in project_path.glob("**/*.sceneworks.json"):
        if sidecar_path.name == "project.sceneworks.json":
            continue
        discovered += 1
        try:
            sidecar = load_sidecar(sidecar_path)
            media_path = project_path / sidecar.file.path
            if media_path.exists():
                sidecar.file.sizeBytes = media_path.stat().st_size
            upsert_asset_index(project_path, sidecar)
            indexed += 1
        except HTTPException as exc:
            errors.append(f"{relative_to_project(project_path, sidecar_path)}: {exc.detail}")
    return discovered, indexed, errors, migrations
