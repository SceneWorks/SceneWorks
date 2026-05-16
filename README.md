# SceneWorks

SceneWorks is a local Docker-based AI image and video generation studio. This repository currently contains the runtime skeleton plus the project, asset, and persistence spine: a Vite/React Library shell, FastAPI backend, placeholder Python worker, shared config/data folders, SQLite project indexes, and portable JSON sidecars.

## Quick Start

```powershell
npm run dev
```

This starts the local stack with Docker Compose:

- Web: http://localhost:5173
- API: http://localhost:8000/api/v1/health

Run the lightweight scaffold checks:

```powershell
npm run check
npm run check:api
```

Build the web app:

```powershell
npm --workspace apps/web run build
```

## Projects And Assets

Projects are inspectable `.sceneworks` folders under `data/projects` by default. Each project contains `project.json`, `project.db`, and portable asset folders:

```text
assets/images
assets/videos
assets/uploads
assets/frames
assets/renders
characters
loras
recipes
timelines
trash
cache
```

Imported images and videos are copied into the project. Each asset gets a sidecar JSON file next to the media file, while `project.db` indexes assets for fast listing, search, filtering, curation, trash, and future repair/reindex workflows.

## Local Access Control

Local-only development is open by default. To require a simple pairing token for LAN or shared-machine use, copy `.env.example` to `.env` and set:

```text
SCENEWORKS_ACCESS_TOKEN=choose-a-private-token
```

When a token is configured, API requests other than health/access discovery must include either:

```text
Authorization: Bearer choose-a-private-token
```

or:

```text
X-SceneWorks-Token: choose-a-private-token
```

This is for privacy and control over local media, model downloads, and long-running GPU work. It is not a content moderation system.

## Structure

```text
apps/
  web/       React + Vite app shell
  api/       FastAPI service and backend filesystem owner
  worker/    Placeholder worker package
packages/
  schemas/   Shared JSON schema contracts
  shared/    Cross-app shared notes/helpers placeholder
config/
  manifests/ Built-in and user model/LoRA manifests
data/
  projects/  Local SceneWorks projects
  models/    App-managed model storage
  loras/     App-managed LoRA storage
  cache/     Runtime cache
docker/      Service Dockerfiles
```
