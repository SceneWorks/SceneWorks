from __future__ import annotations

from datetime import UTC, datetime
import json
import re
from pathlib import Path
from typing import Any


class ProjectNotFound(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str, *, fallback: str = "item", max_length: int | None = None) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower() or fallback
    return slug[:max_length] if max_length else slug


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def safe_float(value: Any, default: float, minimum: float = 0, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    parsed = max(minimum, parsed)
    return min(maximum, parsed) if maximum is not None else parsed


def load_registry(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_project_path(registry_path: Path, project_id: str) -> Path:
    for project in load_registry(registry_path):
        if project.get("id") == project_id:
            project_path = Path(project["path"])
            if project_path.exists():
                return project_path
            break
    raise ProjectNotFound(f"Project not found: {project_id}")
