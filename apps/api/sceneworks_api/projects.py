from pathlib import Path

from fastapi import APIRouter, Request

from .models import ProjectCreateRequest, ProjectOpenRequest, ProjectSummary
from .persistence import (
    create_project as create_project_on_disk,
    ensure_data_dirs,
    find_project_path,
    load_registry,
    open_project as open_project_on_disk,
    read_project_summary,
)
from .settings import Settings


router = APIRouter(prefix="/projects", tags=["projects"])


def get_settings_from_request(request: Request) -> Settings:
    return request.app.state.settings


@router.get("", response_model=list[ProjectSummary])
def list_projects(request: Request) -> list[ProjectSummary]:
    settings = get_settings_from_request(request)
    ensure_data_dirs(settings)
    projects: list[ProjectSummary] = []
    for item in load_registry(settings):
        path = Path(item["path"])
        if path.exists():
            projects.append(read_project_summary(path))
    return projects


@router.post("", response_model=ProjectSummary, status_code=201)
def create_project(payload: ProjectCreateRequest, request: Request) -> ProjectSummary:
    return create_project_on_disk(get_settings_from_request(request), payload.name)


@router.post("/open", response_model=ProjectSummary)
def open_project(payload: ProjectOpenRequest, request: Request) -> ProjectSummary:
    return open_project_on_disk(get_settings_from_request(request), Path(payload.path))


@router.get("/{project_id}", response_model=ProjectSummary)
def get_project(project_id: str, request: Request) -> ProjectSummary:
    return read_project_summary(find_project_path(get_settings_from_request(request), project_id))
