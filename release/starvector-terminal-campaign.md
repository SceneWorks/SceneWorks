# StarVector terminal campaign

This is the single post-permanent-pin, dispatch-only SC-22261 campaign profile.
It never downloads snapshots, package wheels, or LPIPS weights: the selected
runner must already provide an immutable inference checkout, both model
inventories, an exact metric environment, and the current-tree typed
`vector_generate` route entrypoint. The controller rejects any missing or
changed identity, a non-matching Cargo inference pin, job-time downloads, or a
second campaign marker for the same permanent pin.

The workflow executes exactly four serial tuples: MLX 1B, MLX 8B, Candle/CUDA
1B, then Candle/CUDA 8B. Each must emit all 120 ordered image-quality cases and
20 deterministic parity cases. The last tuple also emits the one 200-case
hostile sanitizer suite and the one 60-case raster-to-vector prompt suite.
Every tuple artifact uploads even on a failed route; the final CPU-only seal
step accepts a receipt only after all raw evidence, source/output/transcript
digests, metrics lock, model inventories, and same campaign identifier agree.
It builds the exact inference receipt shape and invokes the pinned inference
`validate-receipt` command, which independently recomputes the quality,
parity, sanitizer, prompt, lifecycle, limit, memory, and 8B uplift gates.

The checked-in metric lock fixes white-composited 512x512 sRGB8 rendering,
scikit-image SSIM parameters, official LPIPS Alex evaluation parameters, and
the separately hashed LPIPS linear and AlexNet trunk weights. SceneWorks owns
the product route, sanitizer, rasterizer, asset publication and artifact
materialization; inference owns the terminal receipt schema and validator.

## Pre-provisioned input contract

The metrics root contains `starvector-terminal-metrics-environment-v1.json`.
It names exactly the seven package versions in the checked-in metrics lock,
the regular non-symlink LPIPS linear and AlexNet files with their fixed hashes,
and a local OpenCLIP object with `provider_id`, `model`, immutable `revision`,
and `checkpoint.path`/`checkpoint.sha256`. The controller validates every path
below the metrics root before execution; the source-owned Python runner then
reads the installed package versions and hashes the files again. Its receipt
transcript is generated from that same execution, not supplied in this file.

The weights root contains `starvector-terminal-weights-v1.json`. In addition
to the exact `models` entries and `terminal_service_closure`, it contains a
`prompt_raster` object with the native `provider_id`, catalog `model`, immutable
`revision`, and a regular `inventory_path`/`inventory_sha256` pair. The route
accepts this identity only when all 60 completed product workflows report the
same raster model and revision.

The inference preflight index contains the exact inference `main` head,
workflow run id/attempt, two inventory artifacts, and four native-hook logs.
Every artifact entry supplies a relative `path` and `sha256`; the controller
rejects missing files, symlinks, duplicate tuples, and byte drift, then copies
the checked bytes into the terminal receipt closure.
