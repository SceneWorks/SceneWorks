# FLUX3 Image Klein: one MLX application smoke (sc-26234)

This is S1's install → select → ordinary typed API job → native worker → **saved
project-library image** evidence recipe. It is not a provider PNG probe, hosted
fixture CI evidence, a full epic acceptance matrix, or a memory campaign.

## Admission and exact identities

**Do not dispatch yet.** Inference PR #1263 must first merge green; the coordinator
must explicitly authorize the SceneWorks feature tracking pin, and the coherent
source containing that actual manifest/lock pin must be published. `inference_sha`
validates that pin; it never overrides dependencies or repairs an old-pin build.
No pin is changed by this harness.

Reserve one idle physical Apple Silicon Mac serially with other Metal work:
`secondary` = `rw-mage` / `nax-macos-2`; `primary` = `rw-starvector` / `nax-macos`.
The workflow only serializes its own runs on the selected target, not other repos
or profiles. It requires Darwin arm64 and at least 32 GB unified memory. Windows
CPU tests cannot build MLX or certify real weights. No local GPU run is needed.

That Mac runner account needs its **own stored Hugging Face login**, with access
to `black-forest-labs/flux-3-image-klein` under the FLUX Kommunity terms. The
controller reads `HF_TOKEN_PATH`, or the token file under the runner's existing
`HF_HOME` (default `~/.cache/huggingface/token`), once in memory. It probes an actual
pinned config download before build. A 401/403 or missing login is an actionable
runner-account authorization failure, not evidence the user's Windows account
has not accepted the gate. Never print, upload or transfer token values; no
repository secret is inherited, and no unauthorized model rehost is used.

Freeze `SOURCE_BRANCH` at exactly `SOURCE_SHA` (40 hex); use its matching pinned
`INFERENCE_SHA` (40 hex). Once coordinator-authorized, dispatch the already
registered workflow **on that frozen branch**, not on main:

```bash
gh workflow run memory-catalog-campaign.yml --ref "$SOURCE_BRANCH" \
  -f profile=flux3-image-klein-app-lifecycle \
  -f ref="$SOURCE_SHA" -f inference_sha="$INFERENCE_SHA" \
  -f flux3_app_runner_target=secondary
```

Relative reusable-workflow calls resolve at that dispatch commit. The controller
rejects mismatches among dispatch `GITHUB_SHA`, supplied source SHA, clean checkout
HEAD, every inference manifest pin and Cargo.lock resolution. Keep the branch
frozen through dispatch acceptance. Do not dispatch/re-run simply to discover a
known missing pin or credential. This profile skips **both catalog jobs**, and
never walks, measures, regenerates matrices, edits pins, commits, pushes or opens
PRs. Omitted/default profile retains existing catalog behavior.

## What runs and what counts

The bounded reusable workflow builds the exact source Release rust-api, verifies
Metal, and starts only its isolated API and standalone native MLX worker. State,
config, credentials, jobs DB, build target and evidence are isolated under the
attempt's RUNNER_TEMP directory; port 17921 must be free. Only owned process
groups are terminated. The controller has a 150-minute bound, workflow 180;
these are safety bounds, not measured runtime estimates.

Using the ordinary model-library relocation and model-download API, it installs
(or re-verifies) the catalog's upstream package at revision
`8454017f9ebaada30c998fccd9f67a80c7580f68` in the persistent task-owned
`~/sceneworks-flux3-app-weights/hub`. It checks gated/license identity, the exact
catalog snapshot path, DiT, AE, bundled Qwen3-VL encoder/config/tokenizer/shards,
and LICENSE/NOTICE, retaining component byte counts and SHA-256 hashes. The
ordinary downloader receives the runner login only in its environment. No
provider-local download or fake install/completion/import substitutes for this.
Weights remain on the runner; the workflow does not delete shared caches.

It creates a fresh project and calls `POST /api/v1/image/jobs` with
`model=flux3_klein`, `mode=text_to_image`, one 1024² image, seed 26234,
`advanced={steps:16,sampler:euler,guidanceScale:0}` and no adapters/references/pup.
It waits for the owned MLX worker's completed ordinary `image_generate` job.
Success additionally requires API-transformed `assets`/`assetIds` (no residual
`assetWrites`), library-index lookup, persisted sidecar equality, native
`mlx_flux3` recipe, actual job/generation-set provenance and restored settings.
The project's ordinary file route must serve bytes identical to the nonempty
saved 1024² PNG. There is no manual image upload/registration: the worker writes
media and reports facts; the API's production asset persistence registers them.

The artifact `flux3-image-klein-mlx-app-<run>-<attempt>` retains `receipt.json`,
`saved-asset.png`, its sidecar, and owned build/API/worker/Metal logs for 30 days.
It excludes weights, databases, credentials, build trees and private raw logs.
Subprocess logs are written outside the uploaded tree, then promoted only after
owned cleanup with raw/URL-encoded known credentials redacted; hard cancellation
cannot upload the private logs. A passed preflight
receipt says **phase=preflight** and is not completed-job evidence. An accepted
full receipt must say `status=passed`, `phase=lifecycle`, contain the completed
job, indexed library asset, saved-file hash, exact source/inference/model and
runner/run identities, and successful owned-process cleanup. Failures retain a
failed receipt. Retain the artifact URL and compact identity/asset summary in S1's
handoff; do not call S1's real-weight AC proven until that actual run succeeds.

Backend-neutral request/asset validators are exported for a future separately
admitted Candle launcher (`backend=candle`, `adapter=candle_flux3`). This Mac
workflow makes **no Candle evidence claim**. The epic terminal owner still owns
the single full acceptance/resource campaign and final main-reachable pin.

CPU-only harness checks: `node --test scripts/flux3-image-klein-app-lifecycle.test.mjs`.
Those fixture/negative-mutation tests verify refusal and receipt contracts only;
they are not real application generation evidence.
