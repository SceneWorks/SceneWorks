# PIPE-18 route reconciliation (SW-6931, SW-F331 E1)

The executable artifact is `sceneworks-core::jobs_store::generation_route_coverage` and its
`check-generation-route-coverage` command. It joins the public product routes to the platform
engine registries by provider, mode, conditioning and output on each backend. Its output is a
temporary review artifact, **not a checked-in capability matrix**, a source of product
eligibility, or a gate with a pinned digest.

The two sides of the join are written independently:

- **Product side (authored).** Public operations (`capabilities`, `ui.img2img`) and authored
  conditioning carriers (`ui.controlModes`, `ui.multiReference`, `audio.conditioning`) of
  `config/manifests/builtin.models.jsonc`; the vector providers each product names; the
  production routing predicates (`routing/mlx.rs`, `routing/candle.rs`, which apply the authored
  `config/manifests/registry/route-policy.json` and every bespoke worker lane) that decide whether
  a backend routes the canonical request; and the native job arms of
  `crates/sceneworks-worker/src/utility_dispatch.rs`. No field the core build copies out of a
  capability dump is read on this side.
- **Engine side.** The platform registries in
  `config/engine-capabilities/runtime/capabilities.{mlx,candle}.json`, produced by the worker's
  `engine_capability_facts.rs` dumper: model and video-mode mappings, registered generators with
  their descriptor conditioning, text-LLM, audio and utility registries, and worker job arms.

A routed product cell binds when the platform's worker advertises the job arm (the production
claim gate) and either a registered engine serves it (mapped generator with the mode's
conditioning, video mapping meeting the mode's conditioning requirements, named vector provider,
audio registration) or a bespoke worker lane claims it. A routed cell with neither is **missing**.
An authored public cell that no backend routes is also missing. The approved backend exception
register (`config/manifests/registry/approved-exceptions.json`) records deliberate MLX-only cells:
the Candle side it explains is reported under `exceptions`, never dropped, and it can never
explain an MLX loss.

On the current tree the report has 451 route cells (393 operation, 58 conditioning; 413
registry-bound, 38 on bespoke lanes), zero missing cells and nine exception-covered Candle cells
(the LTX-2.3 Eros withdrawal and the Krea Realtime sequencing decisions). It also carries all
39 `JobType` lifecycles with the platforms that advertise each arm, every registry surface with
its owner and callable entry point, model aliases, and the source-identified iOS/CPU profiles.

Run from the SceneWorks root:

```sh
cargo run -p sceneworks-core --no-default-features --bin check-generation-route-coverage -- /tmp/generation-route-coverage.json
cargo test -p sceneworks-core --no-default-features --lib route_coverage
node --test scripts/check-generation-route-profiles.test.mjs
node scripts/check-generation-route-profiles.mjs /path/to/inference-checkout
```

The binary exits non-zero when any cell is missing. The profile command uses `git show` at the
receipt's immutable inference revision; it checks source bytes and extracts the real profile
assertions independently. It neither checks out another revision nor compiles an engine. The
receipt in `crates/sceneworks-core/tests/fixtures/generation-route-profiles.json` names the exact
source files/hashes. The iOS media-enabled profile has seven generators and Iris restoration;
LLM-only excludes both media surfaces. CPU composes the same Candle media/audio catalogs as CUDA
with its own LLM list and admission. Neither iOS registration nor a CPU/CUDA shared catalog
proves device fit.

The job join parses the actual native dispatch arms with `syn`, rather than accepting comments as
dispatch evidence. Conditioning keeps the canonical production request and distinguishes FLUX.1
XLabs IP, PuLID/InstantID identity and structural control from VAE initialization; SenseNova
VQA/interleave keep their concrete worker entry points and text/ordered-document outputs.

The negative control removes StarVector from both platforms' text-LLM registries in memory,
leaving the product side untouched. It must name exactly:

```text
macos/mlx/starvector_1b/operation/image_to_svg
cuda/candle/starvector_1b/operation/image_to_svg
macos/mlx/starvector_8b/operation/image_to_svg
cuda/candle/starvector_8b/operation/image_to_svg
```

A second control removes Krea Realtime's MLX registrations: its three MLX cells become missing
while its approved Candle exception cells stay reported, unchanged.

This is metadata-only route evidence. It makes no learned-output, latency, memory-fit or AP13
migration-acceptance claim. Ordinary CI executes no CPU/GPU model forward for this check; there
is no workflow dispatch, device hold or terminal measurement campaign.
