# SceneWorks Schemas

Shared JSON schema contracts live here so the API, worker, and web app can agree on project, asset, recipe, job, model, LoRA, character, and timeline shapes.

The FastAPI models in `apps/api/sceneworks_api/models.py` are the runtime contract for API responses and sidecar validation. These JSON schemas mirror those shapes for portable project data and future worker/web validation.
