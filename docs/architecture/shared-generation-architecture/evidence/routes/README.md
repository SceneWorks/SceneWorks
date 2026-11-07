# PIPE-18 route reconciliation (SW-6931)

The executable artifact is `sceneworks-core::jobs_store::generation_route_coverage` and
its `check-generation-route-coverage` command. It joins the existing published product
matrix to the current production routing predicates, canonical requests, native platform
snapshots, actual bespoke selectors, and the source-identified CPU/iOS profiles. Its output
is a temporary review/CI artifact, **not another checked-in capability matrix** or a source
of product eligibility.

The old product input is `config/backend-capabilities/matrix.json` at SceneWorks
`5a28d413dd9f579cc9607d44f252a045d025bd8e`. A test freezes its semantic model digest
(`105abeb6caea45068044f0d4b6e6fcf8a6bcf58f003e81de7c016514fa454c82`), independently of
source-capture digest refreshes. Regenerating a candidate export cannot silently remove an
old positive or negative witness. A new model or axis requires an explicit baseline
disposition. Preserve the old inventory when the legacy matrix is eventually retired.

Run from the SceneWorks root:

```sh
cargo run -p sceneworks-core --no-default-features --bin check-generation-route-coverage -- /tmp/generation-route-coverage.json
cargo test -p sceneworks-core --no-default-features --lib route_coverage
node --test scripts/check-generation-route-profiles.test.mjs
node scripts/check-generation-route-profiles.mjs /path/to/inference-checkout
```

The last command uses `git show` at the receipt's immutable inference revision; it checks
source bytes and extracts the real profile assertions independently. It neither checks out
another revision nor compiles an engine. The receipt in
`crates/sceneworks-core/tests/fixtures/generation-route-profiles.json` names the exact source
files/hashes, including the direct FLUX.1/FLUX.2 conditioning contracts. The iOS media-enabled
profile has seven generators and Iris restoration; LLM-only excludes both media surfaces.
CPU composes the same Candle media/audio catalogs as CUDA with its own LLM list and admission.
Neither iOS registration nor a CPU/CUDA shared catalog proves device fit.

The report retains all 38 `JobType` lifecycles and parses the actual native dispatch arms
with `syn`, rather than accepting comments as dispatch evidence. Its product rows preserve
both backend answers for every operation, conditioning, adapter, precision, guidance and
preview cell; imports and training retain their separate typed rows. Registry entries retain
provider descriptors, public consumers or direct engine disposition, and the callable bundle
entry point. Model aliases retain their origin even when the target is absent from the media
registry: all four Candle Qwen edit aliases remain visible, with their public eligibility
checked separately.

Conditioning retains the canonical production request and distinguishes FLUX.1 XLabs IP,
PuLID/InstantID identity and structural control from VAE initialization. Direct FLUX.2-dev
MLX img2img has its own source witness; base FLUX.1 img2img remains explicitly unsupported.
SenseNova VQA/interleave keep their concrete worker entry points and text/ordered-document
outputs. Output variants retain Qwen 2.1 straight-alpha RGBA, prompt/count batch output, joint
H3 AV and optional LTX AV. Vector output retains canonical sanitized SVG, derived preview and
terminal metadata.

The mutation test removes StarVector from each candidate text-LLM list while preserving the
old product witness. It must name exactly:

```text
mlx/starvector_1b/operation/image_to_svg
candle/starvector_1b/operation/image_to_svg
mlx/starvector_8b/operation/image_to_svg
candle/starvector_8b/operation/image_to_svg
```

Success means zero missing old supported cells, complete job lifecycle/dispatch coverage,
reviewed dispositions for newly introduced cells, and valid source/profile provenance. This
is metadata-only route evidence. It makes no learned-output, latency, memory-fit or AP13
migration-acceptance claim. Future migration owners still attach the appropriate native
receipts to these routes. Ordinary CI executes no CPU/GPU model forward for this check;
there is no workflow dispatch, device hold or terminal measurement campaign.
