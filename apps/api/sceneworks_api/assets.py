from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from .models import (
    AssetImportResponse,
    AssetListResponse,
    AssetSort,
    AssetSummary,
    AssetType,
    AssetUpdateRequest,
    ReindexResponse,
)
from .persistence import (
    find_project_path,
    get_asset,
    get_asset_sidecar_path,
    import_asset,
    list_assets,
    reindex_project,
    trash_asset,
    update_asset,
)
from .projects import get_settings_from_request


router = APIRouter(prefix="/projects/{project_id}", tags=["assets"])


@router.get("/assets", response_model=AssetListResponse)
def list_project_assets(
    project_id: str,
    request: Request,
    asset_type: AssetType | None = Query(default=None, alias="type"),
    include_rejected: bool = Query(default=False, alias="includeRejected"),
    include_trashed: bool = Query(default=False, alias="includeTrashed"),
    favorites_only: bool = Query(default=False, alias="favoritesOnly"),
    search: str | None = None,
    sort: AssetSort = "newest",
) -> AssetListResponse:
    project_path = find_project_path(get_settings_from_request(request), project_id)
    assets, total = list_assets(project_path, asset_type, include_rejected, include_trashed, favorites_only, search, sort)
    return AssetListResponse(assets=assets, total=total)


@router.post("/assets/import", response_model=AssetImportResponse, status_code=201)
async def import_project_assets(
    project_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
) -> AssetImportResponse:
    settings = get_settings_from_request(request)
    imported: list[AssetSummary] = []
    for upload in files:
        imported.append(import_asset(settings, project_id, upload))
    return AssetImportResponse(assets=imported)


@router.get("/assets/{asset_id}", response_model=AssetSummary)
def get_project_asset(project_id: str, asset_id: str, request: Request) -> AssetSummary:
    project_path = find_project_path(get_settings_from_request(request), project_id)
    return get_asset(project_path, asset_id)


@router.patch("/assets/{asset_id}", response_model=AssetSummary)
def update_project_asset(
    project_id: str,
    asset_id: str,
    payload: AssetUpdateRequest,
    request: Request,
) -> AssetSummary:
    project_path = find_project_path(get_settings_from_request(request), project_id)
    return update_asset(project_path, asset_id, **payload.model_dump(exclude_unset=True))


@router.delete("/assets/{asset_id}", response_model=AssetSummary | None)
def delete_project_asset(
    project_id: str,
    asset_id: str,
    request: Request,
    hard_delete: bool = Query(default=False, alias="hardDelete"),
) -> AssetSummary | None:
    project_path = find_project_path(get_settings_from_request(request), project_id)
    return trash_asset(project_path, asset_id, hard_delete)


@router.get("/assets/{asset_id}/content")
def get_project_asset_content(project_id: str, asset_id: str, request: Request) -> FileResponse:
    project_path = find_project_path(get_settings_from_request(request), project_id)
    sidecar = get_asset(project_path, asset_id)
    media_path = (project_path / sidecar.file.path).resolve()
    if not media_path.exists() or project_path not in media_path.parents:
        raise HTTPException(status_code=404, detail="Asset file not found")
    return FileResponse(media_path, media_type=sidecar.file.mimeType, filename=media_path.name)


@router.post("/reindex", response_model=ReindexResponse)
def reindex_project_assets(project_id: str, request: Request) -> ReindexResponse:
    project_path = find_project_path(get_settings_from_request(request), project_id)
    discovered, indexed, errors, migrations = reindex_project(project_path)
    return ReindexResponse(
        projectId=project_id,
        discovered=discovered,
        indexed=indexed,
        errors=errors,
        migrationsApplied=migrations,
    )
