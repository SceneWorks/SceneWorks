# Packaging rule: never bake model weights into a published artifact

**Status:** enforced in CI (sc-10526, epic 10512 "Anima model support").

## The rule

SceneWorks ships its **code** under AGPL-3.0-or-later. The model **weights** it runs
are a **separate, license-gated concern** and are governed by their own upstream
licenses — not by SceneWorks' code license.

Several supported model families are distributed under **Non-Commercial (NC)**
licenses. A "Derivative" under these licenses includes **any converted or quantized
checkpoint and any LoRA / fine-tune** of the weights. Redistributing a Derivative
obliges the distributor to ship the license text, an attribution notice, and a
statement of modification.

SceneWorks avoids all of those obligations with one architectural posture:

> **Convert at install. Pull at runtime. Never redistribute weights.**

Weights are downloaded from Hugging Face into the **user's own machine or instance**
and converted/quantized there. SceneWorks is never the party handing someone a copy
of the weights — so the redistribution terms never attach.

**That guarantee is only as strong as our packaging.** If a published artifact ever
embeds model weights, SceneWorks becomes a distributor of a Derivative and the
obligations attach immediately. Published artifacts include:

- **Desktop installers / DMGs** (Tauri bundle, `apps/desktop/`).
- **Docker / RunPod image layers** (`docker/rust.Dockerfile`, epic 10362).
- **Re-hosted Hugging Face repos** (the epic 5594 pattern).

So the rule is: **no model-weight file may appear inside any published artifact
payload.**

### Outputs are explicitly fine

Do not over-correct. The NC licenses permit **commercial use of Outputs** — the
generated images (e.g. the CircleStone Labs Non-Commercial License v1.2 §2(e)). Only
the **Model and its Derivatives** (the weights) are restricted. This rule is about
weights, never about generations.

### Re-hosting an NC model (if we ever choose to)

Re-hosting converted NC tiers on the SceneWorks HF org (the epic 5594 pattern) is
permitted **non-commercially**, but it is a redistribution, so each such repo **must
carry the upstream license text + an attribution notice + a "we modified this model"
statement.** Do not do it accidentally. That path is out of scope for this guard,
which covers installers and image layers.

## The guard

`scripts/check-no-nc-weights.mjs` fails the build if a model-weight file is found in
an artifact-payload directory.

Run it directly:

```sh
npm run check:nc-weights              # scan the default payload trees
node scripts/check-no-nc-weights.mjs --dir <path>   # scan a built artifact tree
node scripts/check-no-nc-weights.mjs --self-test     # prove the gate fires
```

### What it scans

- **Default:** `config/`, `crates/`, `apps/` — exactly the trees the Docker image
  COPYs in and where the Tauri desktop bundle stages its resources. Build caches
  (`node_modules/`, `target/`, `dist/`, `.git/`) are skipped.
- **`--dir <path>` (repeatable):** a real built artifact — an extracted Docker image
  rootfs, a built `.app` / bundle tree, or a RunPod build context. The release
  workflow points this at `target/release/bundle` after building the desktop bundle.

### What it deliberately does NOT scan, and why

- **`data/`** — the developer's local model store. `.dockerignore` already excludes
  `data/models|loras|cache` from the image, so it is never artifact payload.
- **`~/.cache/huggingface`** — the runtime HF cache. A developer running
  convert-at-install has NC weights here (e.g.
  `models--circlestone-labs--Anima/`). That is the normal, correct state and must
  **not** trip the guard — so the cache is never scanned.

### How NC families are declared (data-driven)

The set of NC families is **derived from the manifests**
(`config/manifests/builtin.models.jsonc` + `builtin.loras.jsonc`): an entry is NC if
it declares `nonCommercial: true` **or** its license text carries an NC signal
("non-commercial", "NSCL", "NVIDIA Open Model License", the FLUX / Ideogram / Krea NC
licenses). Each NC entry's HF repo(s) become on-disk match tokens (`org/name`,
`models--org--name`, and the bare `name`). **The next NC model inherits the guard
the moment it lands in the manifest** — there is no second list to maintain.

The hard failure does not depend on that classification: **any** weight file in the
payload that is not on the allowlist fails the build. NC classification only enriches
the error so it can name the specific license/family. A brand-new NC model therefore
still fails the build even before it is tagged NC.

> **Bootstrap:** Anima (epic 10512) is not in the manifest yet, so its tokens
> (`anima`, `circlestone-labs`) are seeded directly in the guard. Remove that seed
> once Anima's manifest entry lands with its repo + `nonCommercial: true`.

### The allowlist

A tiny set of **permissively-licensed** weights are legitimately compiled into a
SceneWorks binary and are allowed in the payload. Today that is only the LAION
improved-aesthetic-predictor (`aesthetic-v2-sac-logos-ava1-l14.safetensors`, MIT), a
~4 MB CLIP-head regressor linked into `crates/sceneworks-image-quality`.

To add a new legitimately-bundled permissive weight, add its basename **with a
justification** to the `ALLOWLIST` array in `scripts/check-no-nc-weights.mjs`.
**Never add an NC weight to the allowlist** — the whole point is that NC weights are
pulled at runtime, not shipped. An allowlisted basename found inside an NC repo/cache
path is still flagged.

## Where it runs in CI

- **`.github/workflows/check.yml`** (every PR): runs `--self-test` then the default
  scan over the Docker build context / Tauri resource trees.
- **`.github/workflows/release.yml`** (desktop release, macOS + Windows): scans the
  freshly-built `target/release/bundle` tree before signing/publishing — the
  real-artifact form of the check.

## Known limitation

The `.dmg` / `.msi` / `-setup.exe` installers are opaque compressed archives; the
`--dir` release scan walks the loose `.app` / bundle tree (which carries the same
`Contents/Resources` payload the DMG wraps) but does not decompress the final
installer. The exhaustive resource-source scan in `check.yml` covers the resource
trees that feed those installers, so a weight cannot reach an installer without first
tripping the source scan.

The **RunPod image (epic 10362) does not exist yet** — there is no RunPod-specific
Dockerfile in the repo. The existing `docker/rust.Dockerfile` (the base the RunPod
lane will build on) is verified to pull-at-runtime: its COPYs are limited to
`config/`, `crates/`, `apps/`, and the built binaries, and `.dockerignore` excludes
`data/models|loras|cache`. When the RunPod image lands, run the guard over its build
context / extracted rootfs (`--dir`) as part of that lane.
