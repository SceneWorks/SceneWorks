#!/usr/bin/env python
"""sc-3031 A/B: generate ONE image through the REAL Python Mlx*Adapter (the old path
this epic replaces) for a matched job payload, and copy the output PNG to $SC3031_OUT.

Run with the MAIN worker venv python, PYTHONPATH=apps/worker:packages/shared,
SCENEWORKS_DATA_DIR + HF_HOME + SCENEWORKS_MLX_FLUX_PYTHON set."""
import glob
import json
import os
import shutil
import tempfile
from pathlib import Path

from scene_worker.image_adapters import (
    MlxFlux2Adapter,
    MlxFluxAdapter,
    MlxQwenAdapter,
    MlxZImageAdapter,
    image_request_from_job,
)
from scene_worker.settings import WorkerSettings

# NOTE: SDXL has no old MLX adapter to A/B against — the vendored `_vendor/mlx_sd`
# path was retired in sc-3060; the Python SDXL path is now torch-only. SDXL parity
# is covered by Phase A (settings) + Phase B (E2E) + the mlx-gen-sdxl engine goldens.
ADAPTERS = {
    "z_image_turbo": MlxZImageAdapter,
    "flux_schnell": MlxFluxAdapter,
    "flux_dev": MlxFluxAdapter,
    "qwen_image": MlxQwenAdapter,
    "flux2_klein_9b": MlxFlux2Adapter,
    "flux2_klein_9b_kv": MlxFlux2Adapter,
    "flux2_klein_9b_true_v2": MlxFlux2Adapter,
}

payload = json.loads(os.environ["SC3031_PAYLOAD"])
out = os.environ["SC3031_OUT"]
model = payload["model"]

settings = WorkerSettings()
job = {"id": "sc3031ab", "payload": payload}
request = image_request_from_job(job)
proj = Path(tempfile.mkdtemp(prefix="sc3031_pyproj_"))

adapter = ADAPTERS[model]()


def _progress(*_a, **_k):
    return None


result = adapter.generate(
    settings=settings,
    job=job,
    request=request,
    project_path=proj,
    progress=_progress,
    cancel_requested=lambda: False,
)

pngs = sorted(glob.glob(str(proj / "assets" / "images" / "**" / "*.png"), recursive=True))
assert pngs, f"no PNG produced under {proj}; result={result}"
shutil.copy(pngs[0], out)
print(f"sc3031 python dump: model={model} -> {out} (from {pngs[0]})")
