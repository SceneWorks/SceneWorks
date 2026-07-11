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
node scripts/check-no-nc-weights.mjs --dir <path> --scan-archives --skip-uninspectable
                                                     # + decompress-and-scan archives
node scripts/check-no-nc-weights.mjs --self-test     # prove the gate fires
```

### What it scans

- **Default:** `config/`, `crates/`, `apps/` — exactly the trees the Docker image
  COPYs in and where the Tauri desktop bundle stages its resources. Build caches
  (`node_modules/`, `target/`, `dist/`, `.git/`) are skipped.
- **`--dir <path>` (repeatable):** a real built artifact — an extracted Docker image
  rootfs, a built `.app` / bundle tree, or a RunPod build context. The release
  workflow points this at `target/release/bundle` after building the desktop bundle.
- **Tauri bundle-resource config (always):** on every run the guard also reads the
  committed `apps/desktop/tauri*.conf.json` files and inspects the declared
  `bundle.resources` / `bundle.externalBin` specs. It **fails** if any spec is an
  archive/container (`.zip`, `.tar.gz`, `.dmg`, …), stages a weight file directly,
  matches an NC family token, or is rooted at a weights directory (`weights/`,
  `checkpoints/`, `loras/`). This is the cheap defense for the one vector the
  file-tree scans cannot see into: a weight sealed inside an archive staged as a
  resource. The current resources (`onnxruntime/**/*`, `ffmpeg/**/*`, `mlx/**/*`) are
  clean.
- **`--scan-archives` (opt-in, sc-10551):** decompress-and-scan the CONTENTS of every
  archive under the scan roots, running the exact same weight/NC-token checks over the
  entry paths inside — **recursively** (an archive-in-archive is caught too). This is
  defense-in-depth for a weight that reached a built artifact sealed inside an archive
  some other way than a declared Tauri resource. Pure Node (no new dependency); it
  reads `.zip` / `.nsis.zip` (central-directory listing, stored + deflate), `.tar`,
  `.tar.gz` / `.tgz`, and bare `.gz`. The real release payloads it opens are the macOS
  `bundle/macos/*.app.tar.gz` updater tarball and the Windows `*.nsis.zip` updater. It
  is **off by default** so it never slows an ordinary build; the release lanes turn it
  on.

### Archive-bomb guards (`--scan-archives`)

A malicious or accidental archive bomb (a tiny file that declares or inflates to an
enormous payload, a "many tiny files" bomb, or a deeply-nested quine) must never
exhaust memory or disk. The scan **fails closed** — it refuses the archive and fails
the build rather than OOM — if any of these caps is exceeded:

| Limit                    | Default  | CLI override           |
| ------------------------ | -------- | ---------------------- |
| On-disk archive size     | 2 GiB    | `--max-archive-bytes`  |
| Total uncompressed bytes | 3 GiB    | `--max-uncompressed`   |
| Per-entry uncompressed   | 1 GiB    | `--max-entry-bytes`    |
| Entry count              | 200 000  | `--max-entries`        |
| Nesting depth            | 8        | `--max-depth`          |

For a `.zip` the size caps are checked against the **declared** central-directory
sizes **before any entry is inflated**, so a zip bomb is refused up front. For gzip the
total cap is enforced as the gunzip output limit (Node throws before inflating past
it). All limits are per top-level archive tree (they accumulate across nesting levels).

### Opaque installers (`--skip-uninspectable`)

The final installers — `.dmg` (macOS), `.msi` and the NSIS `-setup.exe` (Windows) —
have **no pure-JS reader**. By default `--scan-archives` **fails closed** on a
container it cannot read (it never silently passes an un-verified archive). The release
lanes pass **`--skip-uninspectable`** to downgrade those to a *warning*, because the
opaque installer's CONTENTS duplicate the `.app` / resource tree the loose-tree walk in
the same step already scanned. A bomb refusal is **never** downgraded by this flag —
only the "no reader for this format" case is.

### The `.bin` question (not a blind spot)

A bare `.bin` extension is **not** in the weight-extension set — fonts, wasm, and
misc blobs also use `.bin`, so treating every `.bin` as a weight would false-trip.
But an NC `.bin` is still caught **by default, with no flags**, three ways:

1. The **strong NC repo-path match** (`org/name`, `models--org--name`) runs on every
   file regardless of extension, so any blob inside a redistributed NC repo/cache
   directory fails (this is the `pytorch_model.bin` under `models--…--Anima/` case).
2. A `.bin` whose path matches a **bare NC family token** (e.g. `anima-base.bin`) is
   promoted to a weight.
3. The **canonical Hugging Face `.bin` weight names** — `pytorch_model.bin`,
   `diffusion_pytorch_model.bin`, `open_clip_pytorch_model.bin` and their sharded
   forms (the second dominant HF weight format after safetensors) — count as weights
   by name even with no NC token.

`--include-bin` additionally promotes **every** `.bin` to a weight (belt-and-suspenders).

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

- **`.github/workflows/check.yml`** (every PR): runs `--self-test` (which now also
  exercises the archive-scan positive/negative/bomb cases) then the default scan over
  the Docker build context / Tauri resource trees.
- **`.github/workflows/release.yml`** (desktop release, macOS + Windows): scans the
  freshly-built `target/release/bundle` tree before signing/publishing with
  `--scan-archives --skip-uninspectable` — the real-artifact form of the check, which
  additionally decompresses the `*.app.tar.gz` / `*.nsis.zip` updater payloads.

## Known limitation

**Opaque installers still are not decompressed.** The `.dmg` / `.msi` / `-setup.exe`
installers have no pure-JS reader, so `--scan-archives` cannot open them (it warns
under `--skip-uninspectable`, or fails closed without it). This is not a real gap: the
`--dir` release scan walks the loose `.app` / bundle tree — which carries the same
`Contents/Resources` payload the DMG/MSI wraps — and the exhaustive resource-source
scan in `check.yml` covers the trees that feed those installers, so a weight cannot
reach an installer without first tripping one of those scans. The archives that are NOT
otherwise covered — the `*.app.tar.gz` / `*.nsis.zip` updater payloads — ARE now
decompressed and scanned by `--scan-archives` (sc-10551).

`include_bytes!`-compiled weights (a weight embedded in a Rust binary at compile time)
are not detectable by any file or archive scan post-compile; the source-tree scan
catches the weight *file* that such a macro would embed, before it is compiled in.

The **RunPod image (epic 10362) does not exist yet** — there is no RunPod-specific
Dockerfile in the repo. The existing `docker/rust.Dockerfile` (the base the RunPod
lane will build on) is verified to pull-at-runtime: its COPYs are limited to
`config/`, `crates/`, `apps/`, and the built binaries, and `.dockerignore` excludes
`data/models|loras|cache`. When the RunPod image lands, run the guard over its build
context / extracted rootfs with `--dir <rootfs> --scan-archives` as part of that lane
(the tar-based image layers are exactly the format `--scan-archives` reads).
