# SceneWorks Rust API

This binary is the Rust backend migration target. Docker Compose can run it as
the `api` service by setting the migration switch in the repository `.env`:

```text
SCENEWORKS_API_RUNTIME=rust
SCENEWORKS_API_DOCKERFILE=docker/rust-api.Dockerfile
```

The Python FastAPI service remains the default and rollback runtime:

```text
SCENEWORKS_API_RUNTIME=python
SCENEWORKS_API_DOCKERFILE=docker/api.Dockerfile
```

Both API runtimes use the same compose contracts:

- `SCENEWORKS_API_HOST` and `SCENEWORKS_API_PORT` control the container bind address.
- `SCENEWORKS_DATA_DIR=/sceneworks/data` maps to the repository `./data` directory.
- `SCENEWORKS_CONFIG_DIR=/sceneworks/config` maps read-only to `./config`.
- `SCENEWORKS_JOBS_DB_PATH=/sceneworks/runtime/jobs.db` stores queue state in the `api-runtime` volume.
- `SCENEWORKS_ACCESS_TOKEN`, `SCENEWORKS_CORS_ORIGINS`, and `SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE` are honored by the Rust API.

Compose checks `GET /api/v1/health` inside the container before starting
dependent services.

Use the root Rust scripts to format, lint, test, and build this workspace.
