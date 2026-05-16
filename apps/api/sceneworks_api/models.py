from typing import Any, Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = 1

AssetType = Literal["image", "video", "upload", "frame", "render", "character_reference"]
AssetSort = Literal["newest", "oldest", "rating", "type", "name"]
JobStatus = Literal[
    "queued",
    "preparing",
    "downloading",
    "loading_model",
    "running",
    "saving",
    "completed",
    "failed",
    "canceled",
    "interrupted",
]


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ProjectOpenRequest(BaseModel):
    path: str = Field(min_length=1)


class ProjectSummary(BaseModel):
    id: str
    name: str
    path: str
    createdAt: str
    updatedAt: str | None = None
    assetCount: int = 0


class ProjectDocument(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    appVersion: str
    id: str
    name: str
    createdAt: str
    updatedAt: str
    folders: dict[str, str]


class AssetFileMetadata(BaseModel):
    path: str
    mimeType: str
    sizeBytes: int
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    fps: float | None = None


class AssetStatus(BaseModel):
    favorite: bool = False
    rating: int = Field(default=0, ge=0, le=5)
    rejected: bool = False
    trashed: bool = False


class Recipe(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    appVersion: str
    id: str | None = None
    mode: str | None = None
    model: str | None = None
    prompt: str | None = None
    negativePrompt: str | None = None
    seed: int | None = None
    loras: list[dict[str, Any]] = Field(default_factory=list)
    normalizedSettings: dict[str, Any] = Field(default_factory=dict)
    rawAdapterSettings: dict[str, Any] = Field(default_factory=dict)


class AssetLineage(BaseModel):
    parents: list[str] = Field(default_factory=list)
    sourceAssetId: str | None = None
    sourceTimestamp: float | None = None
    sourceTimelineId: str | None = None
    jobId: str | None = None


class AssetSidecar(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    appVersion: str
    id: str
    projectId: str
    generationSetId: str | None = None
    type: AssetType
    displayName: str
    createdAt: str
    updatedAt: str
    file: AssetFileMetadata
    status: AssetStatus = Field(default_factory=AssetStatus)
    notes: str = ""
    recipe: Recipe | None = None
    lineage: AssetLineage = Field(default_factory=AssetLineage)


class AssetSummary(BaseModel):
    id: str
    projectId: str
    generationSetId: str | None = None
    type: AssetType
    displayName: str
    createdAt: str
    updatedAt: str
    file: AssetFileMetadata
    status: AssetStatus
    notes: str = ""
    recipe: Recipe | None = None
    lineage: AssetLineage
    previewUrl: str


class AssetUpdateRequest(BaseModel):
    displayName: str | None = Field(default=None, min_length=1, max_length=180)
    favorite: bool | None = None
    rating: int | None = Field(default=None, ge=0, le=5)
    rejected: bool | None = None
    notes: str | None = Field(default=None, max_length=4000)


class AssetListResponse(BaseModel):
    assets: list[AssetSummary]
    total: int


class AssetImportResponse(BaseModel):
    assets: list[AssetSummary]


class GenerationSet(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    appVersion: str
    id: str
    projectId: str
    createdAt: str
    mode: str
    recipeId: str | None = None
    assetIds: list[str] = Field(default_factory=list)


class Job(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    appVersion: str
    id: str
    projectId: str | None = None
    type: str
    status: JobStatus = "queued"
    progress: float = Field(default=0, ge=0, le=1)
    createdAt: str
    updatedAt: str
    error: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ModelManifest(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    id: str
    name: str
    family: str
    type: Literal["image", "video", "utility"]
    adapter: str
    capabilities: list[str] = Field(default_factory=list)
    downloads: list[dict[str, Any]] = Field(default_factory=list)
    paths: dict[str, str] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)
    loraCompatibility: dict[str, Any] = Field(default_factory=dict)
    ui: dict[str, Any] = Field(default_factory=dict)


class LoRAManifest(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    id: str
    name: str
    scope: Literal["global", "project"]
    category: Literal["style", "enhance", "character", "motion", "clothing_object", "experimental"]
    compatibleFamilies: list[str] = Field(default_factory=list)
    path: str
    triggerWords: list[str] = Field(default_factory=list)
    defaultWeight: float = 1.0
    builtIn: bool = False
    ui: dict[str, Any] = Field(default_factory=dict)


class Character(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    appVersion: str
    id: str
    projectId: str
    name: str
    type: Literal["person", "creature", "object"]
    createdAt: str
    updatedAt: str
    referenceAssetIds: list[str] = Field(default_factory=list)
    approvedReferenceAssetIds: list[str] = Field(default_factory=list)
    lookIds: list[str] = Field(default_factory=list)
    loraIds: list[str] = Field(default_factory=list)
    notes: str = ""


class TimelineItem(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    id: str
    timelineId: str
    assetId: str
    track: str = "main"
    sourceIn: float = 0
    sourceOut: float | None = None
    timelineStart: float
    timelineEnd: float
    speed: float = 1
    activeVersionAssetId: str | None = None
    versionAssetIds: list[str] = Field(default_factory=list)


class Timeline(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    appVersion: str
    id: str
    projectId: str
    name: str
    createdAt: str
    updatedAt: str
    aspectRatio: Literal["16:9", "9:16", "1:1", "source"] = "16:9"
    fps: int = 24
    items: list[TimelineItem] = Field(default_factory=list)


class ReindexResponse(BaseModel):
    projectId: str
    discovered: int
    indexed: int
    errors: list[str] = Field(default_factory=list)
    migrationsApplied: list[str] = Field(default_factory=list)
