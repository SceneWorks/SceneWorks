# SceneWorks API

FastAPI service for versioned backend routes and all project filesystem writes.

Current routes:

- `GET /api/v1/health`
- `GET /api/v1/access`
- `POST /api/v1/auth/verify`
- `GET /api/v1/projects`
- `POST /api/v1/projects`
- `POST /api/v1/projects/open`
- `GET /api/v1/projects/{project_id}`
- `GET /api/v1/projects/{project_id}/assets`
- `POST /api/v1/projects/{project_id}/assets/import`
- `GET /api/v1/projects/{project_id}/assets/{asset_id}`
- `PATCH /api/v1/projects/{project_id}/assets/{asset_id}`
- `DELETE /api/v1/projects/{project_id}/assets/{asset_id}`
- `GET /api/v1/projects/{project_id}/assets/{asset_id}/content`
- `POST /api/v1/projects/{project_id}/reindex`
- `GET /api/v1/jobs/events`

The API owns project filesystem writes. Imported media is copied into the project, sidecar JSON is validated with typed models, and SQLite indexes assets for Library list/search/filter operations.
