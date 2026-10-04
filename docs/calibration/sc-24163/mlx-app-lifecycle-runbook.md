# MLX app lifecycle evidence (E11)

Dispatch `.github/workflows/qwen-image-2-1-app-lifecycle.yml` after the reviewed
inference repair has merged and SceneWorks has its final permanent pin. Set
`source_sha` to the exact 40-character SceneWorks feature commit and
`inference_sha` to its exact 40-character inference pin. The controller rejects
a dirty checkout or mismatched manifest/lock revisions. Dispatch runs only;
ordinary pushes and PRs never launch this campaign.

Reserve the idle **primary** `nax-macos` runner with the unique `rw-starvector`
label (128 GB M5 Max). Run it after the separate inference real-weight probe,
serially with other Metal work on this physical Mac. The workflow concurrency
group prevents duplicate app campaigns; it cannot serialize another repository's
manual engine jobs. Allow several hours: this is 400 completed training steps,
plus the interrupted T2I steps, three model-load/cache passes and four previews.
The controller has a ten-hour bound, and the workflow a twelve-hour bound,
including build/setup. These bounds are not measured runtime estimates.

The engine lane must already have staged the immutable dense snapshot in
`~/sceneworks-rw-weights/hub/models--Qwen--Qwen-Image-2.1/snapshots/790c92633540aa0cb11d9abf19eb46d861714758`.
The app uses this same physical task-owned hub offline, verifies the resolved
API plan uses that snapshot, and never downloads, copies or deletes model weights.
The task-owned weights survive the run for the epic's remaining evidence; remove
that directory only during the separately authorized epic-end cleanup.

The source-built Release API and its desktop-style standalone MLX worker share
isolated data/config/jobs/credentials under the attempt's `RUNNER_TEMP` directory.
Port 17921 must be free. Only the two owned process groups are terminated at the
end. The controller uses public synthetic 768-pixel images and a valid 200-step
target, rank 4, batch/accumulation 1, checkpoints every 50 steps, and one eight-step
preview every 100 steps. It does not lower product target floors.

Acceptance requires both research-license 403 negative twins, exact typed API
routes and immutable base paths, an acknowledged T2I cancellation after a real
fingerprinted step-50 optimizer bundle, a retry preserving its plan and continuing
from that checkpoint, **200 completed T2I steps and 200 completed edit LoKr steps**,
live-adapter PNGs served by the app, and automatic project-library registration
of each declared final adapter. Library provenance and safetensors derivative
metadata must agree. Checkpoint imports cannot satisfy finalization.

The artifact `qwen-image-2-1-mlx-app-<run>-<attempt>` includes `receipt.json`,
owned build/API/worker/Metal-preflight logs, public inputs and generated previews.
The receipt remains `failed` on any missing gate, wrong runtime or cleanup error.
It records adapter/checkpoint hashes and metadata; large weights, adapters,
databases and credentials are excluded from the uploaded artifact. Retain the
artifact and fold its compact receipt into `docs/calibration/sc-24163/mlx/`
alongside the single accepted engine campaign after independent review.

This app campaign proves E11 lifecycle/registration. The existing CUDA app
evidence, accepted Phase 1 MLX bf16/q8/q4 anchors and memory matrix remain intact.
The one inference engine campaign owns learned effect, stacked-overlay and
training-footprint acceptance; no catalog, memory or base-anchor recapture is
performed by this workflow.
