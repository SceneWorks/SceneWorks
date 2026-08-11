# MiniMax-H3 — CUDA runner provisioning and per-tier VRAM reachability (sc-18677)

Epic [sc-17137](https://app.shortcut.com/trefry/epic/17137) · story
[sc-18677](https://app.shortcut.com/trefry/story/18677) · measured **2026-08-11**.

This story unblocks [sc-17153](https://app.shortcut.com/trefry/story/17153) and
[sc-17156](https://app.shortcut.com/trefry/story/17156), which need **measured** per-tier VRAM on
real H3 weights with each tier in its own process. It does three things: it lands the weights on
the CUDA runner, it generalizes `windows-candle.yml`'s Krea-hardcoded provisioning so it can point
at any model, and it records what this hardware can and cannot physically hold.

It measures no generate peak. §5 is a *necessary-condition* screen — it says which tiers are worth
sending a measurement job at, and rules one configuration out arithmetically so sc-17153 does not
spend a self-hosted lane discovering it.

---

## 1. The runner (AC2)

The `cuda` pool is **four** runners across **two registration levels**, and they are all listener
processes on **one physical box**, `MICHAEL-TRX50`, running as `MICHAEL-TRX50\Michael`. So "the
runner's GPU" is this box's GPU, and concurrent jobs share it.

| Runner | Level | Install | Labels |
| --- | --- | --- | --- |
| `cuda-windows` (2313) | **org** | `D:\actions-runner` | `self-hosted, Windows, X64, cuda, `**`real-weights`** |
| `cuda-windows-2` (2619) | **org** | `D:\actions-runner-2` | `self-hosted, Windows, X64, cuda, `**`real-weights`** |
| `cuda-windows-3` (23) | repo | `D:\actions-runner-3` | `self-hosted, Windows, X64, cuda` |
| `cuda-windows-4` (24) | repo | `D:\actions-runner-4` | `self-hosted, Windows, X64, cuda` |

`gh api repos/SceneWorks/SceneWorks/actions/runners` reports only the repo-level pair, which is how
this pool gets undercounted — the org-level pair is where a `real-weights` job belongs and is the
half this lane never asked for. That is fixed here (§4.1).

| Fact | Value |
| --- | --- |
| GPU | 2 × **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition** |
| VRAM per card | **97,887 MiB** = 102.642 GB = 95.593 GiB |
| Free per card (idle) | GPU0 97,420 MiB (102.152 GB) · GPU1 96,686 MiB (101.383 GB) |
| Compute capability | **12.0** (sm_120, Blackwell) |
| Driver | 596.36 |
| `display_active` | Disabled on both — no desktop VRAM tax |
| Lane's `CUDA_COMPUTE_CAP` | `"120"` — matches the hardware, so nvcc emits native SASS |

**The budget is ONE card, not the pair — 102.642 GB.** A generation is bounded by a single device:
`runtime_cuda::media::default_device()` is a hardcoded `new_cuda(0)`, and under the `auto`
supervisor each per-GPU child is spawned with `CUDA_VISIBLE_DEVICES=<its gpu id>` so device 0 *is*
that child's card (`crates/sceneworks-worker/src/lib.rs:1030-1034`). There is no tensor-parallel
path, so the two cards do not sum into a 205 GB budget for one job.

### 1.1 Disk headroom

| Volume | Used | Free |
| --- | ---: | ---: |
| C: | 3,061.4 GB | **663.6 GB** |
| D: | 2,390.8 GB | **1,335.2 GB** |
| E: | 7,192 GB | **3,983 GB** |

The story flags that ~200 GB of provisioning competes with CI disk on this box. **It cost nothing.**
The full 498 GB upstream repo was already resident in `E:\huggingface\hub` before this story, and the
revision this epic pins differs from the cached one *only in `README.md`* (§2.2) — so provisioning
re-links existing blobs rather than downloading. Nothing was stranded and no CI check was starved.

### 1.2 The quantized-matmul patch (scope item 4)

Confirmed present and in lockstep. The root `Cargo.toml` carries

```toml
[patch."https://github.com/huggingface/candle"]
candle-kernels = { git = "https://github.com/SceneWorks/inference", rev = "b965641e388f4db646e4c60ab3f75219737e2cc8" }
```

and that rev equals the worker's `sceneworks-gen-core` / `runtime-cuda` pins
(`crates/sceneworks-worker/Cargo.toml:40,99`). This matters more here than anywhere else in the
repo: without the vendored multi-arch kernels, `libmoe.a` is an Ampere-only cubin with no PTX and
**every quantized matmul silently returns zeros on sm_120** — which would turn a q4/q8 VRAM
measurement into a confident wrong number rather than an error. Two guards keep it honest, and both
run on this lane: `candle_kernels_patch_guard.rs` (file-only, proves the patch is live and in
lockstep from `Cargo.lock`) and `cuda_quant_smoke.rs` (deliberately not `#[ignore]`d, exercises the
patched kernels on the real GPU every run).

---

## 2. What is resident, and where (AC1)

### 2.1 The cache is on E:, not in the profile

The box sets `HF_HOME=E:\huggingface` as a user environment variable, which the runner listeners
inherit. The workflow's provisioning step, however, passes an explicit `cache_dir` and therefore
**ignores `HF_HOME`**. The two caches hold different things:

| Cache | Holds |
| --- | --- |
| `C:\Users\Michael\.cache\huggingface\hub` | the Krea reference artifact (`models--SceneWorks--krea-2-turbo-mlx`), plus small LLMs |
| `E:\huggingface\hub` | **`models--MiniMaxAI--MiniMax-H3`**, and the rest of the app's weights |

This is why the generalization added a `provision_cache_dir` input rather than switching the default
to `HF_HOME`: honoring `HF_HOME` would silently relocate Krea's cache and break the "Krea behaves
identically" contract (§4).

### 2.2 The pinned revision

**`939557dc319dd91227e30195a763f272ba7f8765`** — current upstream `main`, and the exact snapshot
sc-17139's sizing table was derived from.

The cache already held an *older* snapshot, `6818f6c32d12b210915e44ad56a4228c2608f160`. Comparing
LFS OIDs across the two revisions over all 280 files:

- **210,331,731,510 bytes of component weights are byte-identical**;
- the only difference is `README.md` (38,479 B → 38,391 B).

Hugging Face keys `blobs/` by etag, so provisioning the pinned revision re-links the existing blobs
and downloads ~38 KB. Pinning the revision sc-17139 measured therefore costs nothing, and keeps this
epic's sizing table and its resident weights on the same commit.

### 2.3 The component set

Fetched by component, one `allow_patterns` entry each. The repo totals 498.475 GB only because
`FL2VA/` and `Ref2VA/` re-package the same components; only the root layout is referenced by
`modular_model_index.json` and only it is the conversion source (sc-17139 §2).

| Pattern | Files | GB | GiB |
| --- | ---: | ---: | ---: |
| `transformer/*` | 16 | 66.281 | 61.729 |
| `text_encoder/*` | 23 | 66.727 | 62.144 |
| `vae/*` | 5 | 10.416 | 9.700 |
| `audio_vae/*` | 2 | 0.605 | 0.564 |
| `tokenizer/*` | 4 | 0.011 | 0.011 |
| `processor/*` | 7 | 0.011 | 0.011 |
| `scheduler/*`, `audio_scheduler/*` | 2 | 0.000 | 0.000 |
| `model_index.json`, `modular_model_index.json`, `LICENSE` | 3 | 0.000 | 0.000 |
| `FL2VA/model_index.json`, `Ref2VA/model_index.json` | 2 | 0.000 | 0.000 |
| **selected** | **64** | **144.051** | **134.158** |
| *whole repo, for contrast* | *280* | *498.475* | *464.241* |
| **avoided** | **216** | **354.424** | **330.083** |

Notes on the selection:

- **`transformer_ref` is deliberately excluded.** The story defers it ("later, 66.3 GB"), and it is
  sc-17149's entry. Adding it takes the set to the full 210.332 GB distinct total.
- The two partition `model_index.json` files are included at a cost of ~2 KB. sc-17139 §2 identifies
  them as the only place `sigma_shift_scales` and the task map live, so sc-17150 needs them; leaving
  them out would mean re-dispatching a provisioning run for two kilobytes.
- Patterns are matched with `fnmatch`, whose `*` **crosses `/`**. A lazy `*.json` would therefore
  have dragged in `FL2VA/**/*.json` and `Ref2VA/**/*.json` too. The root files are named
  individually for that reason, and the selection was verified to have zero partition leakage.

> **Unit warning.** The story text mixes units: its "`vae` 9.7 GB, `audio_vae` 0.6 GB,
> `text_encoder` 62 GB, `transformer` 62 GB" are **GiB**, while its "`transformer_ref` 66.3 GB" is
> decimal GB. Both describe the same components. Every figure in this document is decimal GB unless
> a column says GiB.

---

## 3. The exact snapshot path (AC1)

```
E:\huggingface\hub\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765
```

The resolve step proves it with **three** assertions, not one, and throws on each:

1. the resolved directory exists;
2. its *canonical* path ends with `\models--<owner>--<name>\snapshots\<revision>[\<subdir>]`, so a
   stale cache entry, a different revision, or a lookalike repo cannot satisfy it;
3. every declared component's literal path prefix is present under the snapshot root.

(3) is what makes this a real proof for H3. Krea resolves a *tier subdirectory* (`…\<rev>\q4`), so
its directory cannot exist unless the tier was fetched. H3 resolves the snapshot **root**, which
exists the moment any single file lands — a `README.md` alone would satisfy (1) and (2) while every
weight was missing. For Krea, (3) is exactly equivalent to the pre-existing check: the sole pattern
`q4/**` yields `<snapshot>\q4`, which *is* the resolved root. So it adds no new way for a Krea
dispatch to fail.

---

## 4. The generalization, and the Krea invariance proof (AC3)

`windows-candle.yml`'s provisioning was hardcoded to Krea in four places: the
`provision_krea_snapshot` input, a `snapshot_download` literal naming `SceneWorks/krea-2-turbo-mlx`
with `allow_patterns=["q4/**"]`, a `cache_dir` built from `USERPROFILE`, and a resolve step that
re-spelled `models--SceneWorks--krea-2-turbo-mlx\snapshots\<rev>\q4` by hand.

The inputs were **renamed, not duplicated**, so exactly one provisioning path exists:

| Was | Now | Default |
| --- | --- | --- |
| `provision_krea_snapshot` | `provision_snapshot` | `false` |
| `krea_repository` | `provision_repository` | `SceneWorks/krea-2-turbo-mlx` |
| `krea_revision` | `provision_revision` | — |
| — | `provision_patterns` | `q4/**` |
| — | `provision_subdir` | `q4` |
| — | `provision_cache_dir` | *(empty → `%USERPROFILE%\.cache\huggingface\hub`)* |

Renaming rather than adding a parallel family is what keeps the workflow at **8 inputs** against
GitHub's hard cap of 10, leaving headroom for the epic's later stories.

Three things deliberately did **not** change:

- **`SCENEWORKS_KREA_ROOT` / `_REPOSITORY` / `_REVISION` keep their names.** They are the
  memory-adapter binaries' runtime contract, read via `required_env` in
  `crates/sceneworks-memory-adapter/src/bin/{candle,mlx}.rs`. Renaming them would break the
  five-rung capture at run time, not at lint time. `SCENEWORKS_KREA_ROOT` is now exported only when
  the resolved repository *is* Krea, so an H3 dispatch cannot hand the adapter a MiniMax root under
  a Krea-shaped name.
- **Every five-rung guard survives** — exact 40-hex inference revision, exact 40-hex artifact
  revision, the fixed Krea repository, and the `INFERENCE_PIN` match — just keyed on the new names.
- **The default cache dir still ignores `HF_HOME`** (§2.1).

One coupling was removed on purpose: `provision_snapshot` no longer requires
`run_five_rung_reference`. The old throw existed because provisioning had exactly one consumer, so
provisioning alone was necessarily a mistake. This epic makes it a first-class outcome — H3 weights
must land on the box and there is no H3 five-rung fixture to run.

### 4.1 One deliberate behaviour change: weights dispatches now request `real-weights`

This is the single place the change does **not** preserve prior behaviour, and it is intentional.

`runs-on` was the flat list `[self-hosted, Windows, X64, cuda]`, so a five-rung capture could be
scheduled onto any of the four runners in §1 — including the repo-level pair that does not carry
`real-weights`. It is now conditional, exactly mirroring `macos-mlx.yml:450`:

```yaml
runs-on: ${{ (github.event_name == 'workflow_dispatch' && (inputs.provision_snapshot || inputs.run_five_rung_reference))
             && fromJSON('["self-hosted","Windows","X64","cuda","real-weights"]')
             || fromJSON('["self-hosted","Windows","X64","cuda"]') }}
```

The Mac lane has done this since it grew a weights dispatch; this lane's lack of it is the
asymmetry epic 17137 recorded as *"no `weights` label and no HF provisioning input — unlike the Mac
lane"*. Ordinary PR and push runs are untouched and keep the full four-runner pool, so the ~24m
lane loses no throughput.

**Observable effect today: none.** All four runners are on the same box and share the same caches,
so a Krea dispatch resolves the same snapshot either way. It matters the day a `cuda` runner joins
on a second machine: without the label the job lands somewhere with no snapshot and dies at the
resolve step; with it the job queues for a box that has the weights.

### 4.2 How "identical" was proven

**Executed, not asserted.** The PowerShell step bodies were extracted from the YAML and run locally
against the real caches, with a simulated `GITHUB_ENV` carried between steps exactly as Actions does.
A Krea dispatch leaving every `provision_*` input at its default resolves to

```
C:\Users\Michael\.cache\huggingface\hub\models--SceneWorks--krea-2-turbo-mlx\snapshots\d009674080cc1bccf2b629d834c34bf5eccdb723\q4
```

— character-for-character the path the hardcoded step produced. Eight cases pass: the historical
five-rung + provision dispatch, five-rung without provisioning, H3 provision-only, and five negative
cases (non-40-hex revision, `provision_subdir` traversal, empty allow-list, absent snapshot,
declared-but-missing component) each failing at the intended step.

**Pinned so it cannot rot.** Four tests in `scripts/platform-review-contracts.test.mjs` (inside
`npm run check`) assert the input defaults, the path-construction expressions, the surviving
five-rung guards, and the never-a-whole-repo-fetch guards. They are structural rather than
prose-matching: the defaults and the suffix expression together *determine* the resolved path. All
ten mutations tried against them go red — including flipping `provision_subdir` to `q8`, switching
the cache default to `HF_HOME`, exporting `SCENEWORKS_KREA_ROOT` unconditionally, dropping the
snapshots path segment, deleting either loud-failure guard, and pushing the input count over 10.

---

## 5. Per-tier reachability (AC2)

Component bytes are sc-17139's; the tier factors are its validated MLX-affine model
(`group_size: 64` → q8 = 8.5/16 = 53.125% of bf16, q4 = 4.5/16 = 28.125%), which held to +0.03% /
+0.48% against six shipped artifacts. The VAEs stay dense in every tier, so
**11.044 GB is a fixed floor**.

Against **one card = 102.642 GB**, using the *trimmed* text encoder (what sc-17150 will host):

| Tier | DiT | TE | VAEs | **Resident total** | % of card | Headroom | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q4 | 18.641 | 14.486 | 11.044 | **44.171** | 43.0% | +58.471 | **fits** |
| q8 | 35.211 | 27.363 | 11.044 | **73.618** | 71.7% | +29.024 | **fits** |
| bf16 | 66.280 | 51.506 | 11.044 | **128.830** | 125.5% | **−26.188** | **exceeds** |

Under **sequential** residency — the phase order the schema describes, text encoder dropped before
the DiT — the peak is `max(TE, DiT + VAEs)`:

| Tier | TE stage | DiT+VAEs stage | **Sequential peak** | % of card | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| q4 | 14.486 | 29.685 | **29.685** | 28.9% | +72.957 |
| q8 | 27.363 | 46.255 | **46.255** | 45.1% | +56.387 |
| bf16 | 51.506 | 77.324 | **77.324** | 75.3% | +25.318 |

With the **untrimmed** TE currently on the box, resident bf16 is 144.039 GB (140.3%) and resident q8
is 81.697 GB (79.6%); the sequential peaks are unchanged, because the DiT stage dominates at every
tier.

### Verdicts

- **q4 — reachable.** 43.0% of the card resident. Ample activation headroom.
- **q8 — reachable.** 71.7% resident, 29.02 GB spare. Comfortable resident, trivially fine sequential.
- **bf16 — reachable ONLY sequentially.** Co-resident is not a measurement question: the weights
  alone overrun the card by 26.19 GB before a single activation is allocated, so sc-17153 should not
  spend a lane on it. Sequential fits at 77.32 GB with 25.32 GB for activations.

### 5.1 The AdaLN evict makes the bf16 sequential verdict comfortable, not marginal

The sequential table above is deliberately conservative: it assumes both VAEs are co-resident with a
*whole* DiT. The epic's headline memory lever changes that. `adaln_proj` is **26.021 GB of the
66.280 GB DiT (39.3%)** and is a function of timestep only, so sc-17145 precomputes all 18
modulation vectors for the entire schedule and then evicts those weights; sc-18665 makes that evict
a typed memory-resident exclusion so the ladder's arithmetic can actually see it.

With the phases ordered TE → (evict) → DiT load + AdaLN precompute → (evict AdaLN) → denoise → VAE
decode, the bf16 stage peaks are:

| Stage | Resident | GB |
| --- | --- | ---: |
| text encode | TE alone | 51.506 |
| DiT load + AdaLN precompute | full DiT | **66.280** |
| denoise | DiT − AdaLN | 40.260 |
| VAE decode | DiT − AdaLN + VAEs | 51.303 |

Peak **66.280 GB — 64.6% of the card, 36.36 GB of headroom.** That is the number that matters for
whether bf16 is worth measuring, and it is comfortable rather than marginal. The 77.324 GB figure in
§5 remains the correct *conservative* bound for a provider that has not implemented the evict, and
the 25.32 GB headroom there is still positive — so bf16-sequential is reachable either way. Neither
number changes the resident verdict: co-resident bf16 is out by 26.19 GB.

**These are weights floors, not peaks.** `vramGbByTier` is the max across the whole generate and
"the denoise peak dominates"; for a joint audio+video model at video sequence lengths, attention
scales as B·H·S²·2·bytes and can be large. So a "fits" verdict is a *necessary* condition that
justifies sending a measurement job — the real ceiling is sc-17153/sc-17156's to measure. Only the
bf16-resident row is a *sufficient* rejection.

---

## 6. Manifest consequences for sc-17150 / sc-17158 (AC5)

The story asks that an unreachable tier be recorded as "must not be advertised on the candle lane".
**Both premises behind that instruction are wrong, and the correct consequence is different.**

**Premise 1 — "the largest `candle.vramGbByTier` in the manifest today is `bf16: 40.7`".** It is
**128.0** (`flux2_dev`). Eleven entries already exceed 40.7:

| id | q4 | q8 | bf16 | `minMemoryGb` | seq-capable |
| --- | ---: | ---: | ---: | ---: | --- |
| `flux2_dev` | 44.0 | 70.7 | **128.0** | 56 | yes |
| `qwen_image_edit_2511_lightning` | 57.0 | 65.8 | 87.4 | 59 | yes |
| `qwen_image` | 53.7 | 65.3 | 82.5 | 56 | yes |
| `qwen_image_edit_2511` | 56.7 | 69.0 | 81.7 | 59 | yes |
| `flux2_klein_9b` | 46.2 | 50.6 | 75.5 | 25 | yes |

**Premise 2 — "a tier that cannot run on the candle lane must not be advertised there".** That is
not the repo's convention. `flux2_dev` advertises `bf16: 128.0`, which **exceeds this very box's
card**, and ships. The manifest does not encode "runnable here"; it encodes *measured cost*, and the
runtime decides:

- `minMemoryGb` gates the **default (lightest) tier only** — the schema says so explicitly, and
  heavier hosted tiers are expected to exceed it (`model-manifest.schema.json:1040`).
- `vramGbByTier` is the **resident** peak "that the fit-gate compares against free VRAM to pick
  sequential offload vs reject" (`:1044`).
- `sequentialPeakGb` is the **second-stage** check: when the resident peak forces sequential offload
  but the measured sequential peak still exceeds free VRAM, reject before load instead of running
  into a reactive OOM (`:1054`, sc-10856).

So the correct consequences are:

1. **Do not drop bf16 from the H3 manifest entry on VRAM grounds.** Hosting it (sc-17150) and
   advertising it (sc-17158) stays correct and matches five shipped entries that are heavier than
   H3's bf16 resident total.
2. **bf16 on candle requires sequential residency**, and that is now a hard requirement rather than
   an optimization: 128.830 GB against a 102.642 GB card. sc-17158 must declare
   `supportsSequentialOffload: true` **and** populate `sequentialPeakGb` for bf16. Without the
   second-stage number the gate keeps its best-effort behaviour — run sequentially, reactive-OOM
   backstop — which on a tier that cannot possibly fit resident is a guaranteed bad experience.
3. **If sc-17154/sc-17155 finds the candle H3 provider does not honor `LoadSpec` sequential
   residency, then bf16 genuinely is unreachable on candle** and *that* is when the platform arrays
   must exclude it. This is the real decision point the story was reaching for, and it is a provider
   capability question, not a VRAM one.
4. `minMemoryGb` must be derived from the measured **q4** peak, not from any bf16 figure.
5. The `11.044 GB` dense VAE floor is tier-invariant, so it lifts every tier's number equally; do
   not model it as part of the tier ladder.

---

## 7. Evidence

| Claim | How it was established |
| --- | --- |
| GPU model, VRAM, free, compute cap, driver | `nvidia-smi --query-gpu` on the runner box |
| Both `cuda` runners are one box | `gh api …/actions/runners` + `Win32_Process` owner of each `Runner.Listener.exe` |
| Single-device budget | `lib.rs:1030-1034`, `runtime_cuda::media::default_device()` = `new_cuda(0)` |
| Disk headroom | `Get-PSDrive -PSProvider FileSystem` |
| candle-kernels patch in lockstep | root `Cargo.toml` `[patch]` rev vs `sceneworks-worker/Cargo.toml:40,99`; `candle_kernels_patch_guard.rs` |
| 498 GB already cached; per-component bytes | symlink-resolving walk of the cache snapshot |
| Two revisions differ only in `README.md` | `HfApi.model_info(files_metadata=True)` LFS-OID diff over all 280 files |
| Pattern set selects 64 files / 144.051 GB, no leakage | `filter_repo_objects` against the live file list |
| Krea path is character-identical | step bodies executed locally with a simulated `GITHUB_ENV` |
| Contract tests cannot rot | 10/10 mutations go red |
| Dispatch provisions and resolves H3 | §8 |

---

## 8. Dispatch run (AC4)

<!-- sc-18677:dispatch-evidence -->
_Recorded on the story and filled in here once the dispatch completes._
