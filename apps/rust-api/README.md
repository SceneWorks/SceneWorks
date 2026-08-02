# SceneWorks Rust API

This binary is the SceneWorks backend API. Docker Compose runs it as the `api`
service with `docker/rust.Dockerfile (target rust-api)`.

The API uses these compose contracts:

- `SCENEWORKS_API_HOST` and `SCENEWORKS_API_PORT` control the bind address. The
  binary defaults `SCENEWORKS_API_HOST` to `127.0.0.1` (loopback); compose sets it to
  `0.0.0.0` so the published host port can reach the container, while
  `SCENEWORKS_API_PUBLISH_HOST` defaults the host-side publish to `127.0.0.1`. A
  non-loopback bind or publish with no `SCENEWORKS_ACCESS_TOKEN` serves every endpoint
  unauthenticated and logs a startup warning.
- `SCENEWORKS_DATA_DIR=/sceneworks/data` maps to `${SCENEWORKS_DATA_BIND:-./data}`.
- `SCENEWORKS_CONFIG_DIR=/sceneworks/config` maps writable to `${SCENEWORKS_CONFIG_BIND:-./config}` for user manifests.
- `SCENEWORKS_CREDENTIALS_DIR=/sceneworks/credentials` maps writable to
  `${SCENEWORKS_CREDENTIALS_BIND:-~/.sceneworks/credentials}` and holds the `0600`
  `credentials.json` written by Settings → Service credentials. Kept off the config
  bind on purpose (sc-16540): that one defaults to the repo checkout, and the store is
  plaintext tokens. Outside Docker the variable is normally unset and the store
  resolves to the OS app-config dir; set it only to pin the store to a specific volume.
  An existing `<config>/credentials.json` is migrated here automatically on first start.
- `SCENEWORKS_JOBS_DB_PATH=/sceneworks/data/cache/jobs.db` stores queue state on the existing data bind mount.
- `SCENEWORKS_ACCESS_TOKEN`, `SCENEWORKS_CORS_ORIGINS`, and `SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE` are honored by the Rust API.

Compose checks `GET /api/v1/health` inside the container before starting
dependent services. The health payload reports `runtime: "rust"`.

Rebuild the API image before starting it after dependency or source changes:

```powershell
docker compose build api
docker compose up -d api
```

Use the root Rust scripts to format, lint, test, and build this workspace.
