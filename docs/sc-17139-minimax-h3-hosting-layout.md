# MiniMax-H3 — hosted artifact layout and tier decision (sc-17139)

Epic [sc-17137](https://app.shortcut.com/trefry/epic/17137) · story
[sc-17139](https://app.shortcut.com/trefry/story/17139) · status: **decided**, 2026-08-11.

This is the layout decision that gates sc-17150 (build and upload the tiers) and sc-17158
(author the manifest entry). It decides *what we host, under what names, in what shape*. It
does not build anything and does not add a manifest entry.

> ## ⚠️ Amendment — what actually shipped differs from §1–§4. Read this before authoring anything.
>
> Recorded by **sc-17158** on 2026-08-13. This document, and its executable half
> `tests/test_minimax_h3_download_shape.py`, remain as the record of the decision **as taken**;
> they are not the record of the artifacts that exist. Three things changed during sc-17150 /
> sc-17143 and the manifest in `config/manifests/builtin.models.jsonc` follows the changed facts:
>
> 1. **One repository, not three.** `SceneWorks/minimax-h3-mlx`
>    @ `f22bc294f46894584645aec59a513ee411450c96` holds *both* DiT partitions at every tier as
>    `<tier>/transformer/` and `<tier>/transformer_ref/` (97 files, 240,725,488,299 B, public).
>    **`SceneWorks/minimax-h3-ref-mlx` and `SceneWorks/minimax-h3-components-mlx` were never
>    created.** The two manifest entries share the one repo and differ only in their per-tier
>    `files` predicate, which keeps a per-tier delete scoped to its own subtree.
> 2. **The text encoder is not re-hosted and not tiered.** sc-17143 proved all 14 shards
>    byte-for-byte identical to `Qwen/Qwen3-VL-32B-Instruct`, so sc-17150 (activity 18712) decided
>    to source it upstream rather than mirror it — which also makes §6's layer-50 trim moot as a
>    *hosting* step. The manifest fetches `text_encoder/` from `MiniMaxAI/MiniMax-H3`
>    @ `939557dc319dd91227e30195a763f272ba7f8765` and omits **shard 13 only** (4,875,990,584 B,
>    7.3 %) — the engine reads shards 1–12 and 14, so §6's projected 15.209 GB saving is 4.88 GB in
>    practice. A packed (tiered) text encoder is filed as **sc-19120**, which explicitly reopens the
>    source-upstream decision because a packed artifact is *derived* and cannot come from Qwen.
> 3. **Both VAEs come from the same upstream snapshot**, at their exact directory sizes
>    (`vae/` 10,415,634,127 B, `audio_vae/` 605,431,611 B — the safetensors totals in §2/§9 plus
>    each directory's `config.json` and index).
>
> What did NOT change: the DiT is the only tiered component; the text encoder and both VAEs are
> shared `coRequisite` rows, byte-identical across both entries so the floor downloads once;
> `transformer_ref` is a second entry (`minimax_h3_ref`, §7); and the About→Licenses gate registers
> **two** ids (§8).

Downstream consumers of this document:

| Story | Consumes |
| --- | --- |
| sc-17143 | the text-encoder trim contract (§6) — the engine must load a 50-layer artifact |
| sc-17146 | the AdaLN quantization caveat (§5.4) |
| sc-17150 | repo names, tier subdirs, conversion inputs (§2, §3), sizing targets (§5) |
| sc-17158 | the `downloads[]` shape (§4), two manifest ids, the license gate (§8) |
| sc-17149 | `transformer_ref` lands as its own catalog entry (§7) |

---

## 1. Summary of decisions

1. **Three repositories on the SceneWorks HF org**, not one and not one-per-tier:
   `SceneWorks/minimax-h3-mlx` (the t2va/fl2va DiT, tiered), `SceneWorks/minimax-h3-ref-mlx`
   (the ref2va DiT, tiered), `SceneWorks/minimax-h3-components-mlx` (the shared text encoder,
   tiered; both VAEs, dense and tier-agnostic).
2. **Only the DiT and the text encoder are tiered.** Both VAEs stay dense bf16 in every tier.
   This is the LTX/Mage-Flow layout, and Wan's self-contained-tier layout is explicitly rejected.
3. **`transformer_ref` becomes a second manifest entry** (`minimax_h3_ref`), not a variant or a
   co-requisite of one entry. The two entries share one components mirror.
4. **The text encoder is trimmed** on upload: decoder layers 50–63 and `lm_head` are dropped.
   Verified saving **15.209 GB** bf16 (22.8% of the text encoder), **27.566 GB** across the three
   hosted tiers.
5. Chosen layout hosts **345.02 GB** for the full capability surface against **493.92 GB** for the
   Wan layout — a **148.90 GB** (43%) saving, of which only 22.09 GB comes from the tier split
   itself and the rest from the second partition sharing the same floor.

Every bf16 number in this document is exact and derived from the upstream safetensors index
manifests (§9). Every q8/q4 number is an **ESTIMATE** from a validated bits-per-weight model and
must be replaced with measured hosted sizes in sc-17150 before `estimatedSizeBytes` / `footprint`
are written in sc-17158.

---

## 2. What is actually distinct upstream

`MiniMaxAI/MiniMax-H3` totals ~498 GB because the repo ships three overlapping layouts: the root
(modular) layout plus `FL2VA/` and `Ref2VA/` partition re-packagings. Only the root layout is
referenced by `modular_model_index.json`, and it is the conversion source.

| Component | Bytes (exact, from the index manifest) | GB | Tensors |
| --- | ---: | ---: | ---: |
| `text_encoder` (Qwen3-VL-32B) | 66,714,780,128 | 66.715 | 1058 |
| `transformer` (t2va + fl2va DiT) | 66,280,430,080 | 66.280 | 638 |
| `transformer_ref` (ref2va DiT) | 66,280,430,080 | 66.280 | 638 |
| `vae` (video) | 10,415,475,936 | 10.415 | 703 |
| `audio_vae` | 605,306,340 | 0.605 | 1087 |
| `tokenizer` + `processor` + `scheduler` + `audio_scheduler` | 22,990,623 | 0.023 | — |
| **distinct total** | **210,319,413,187** | **210.319** | |

Two findings that change the conversion plan:

- **`transformer` and `transformer_ref` are structurally identical.** They resolve to the *same*
  `diffusion_pytorch_model.safetensors.index.json` blob in the HF cache — identical tensor names,
  identical shapes, identical 66,280,430,080-byte total. One conversion pipeline serves both, and
  every DiT sizing number in this document applies unchanged to both.
- **The `FL2VA/` re-packaging is a different tensor naming, not a copy.** `FL2VA/transformer` has
  **535** tensor names totalling 66,280,430,**144** bytes against root's **638** names totalling
  66,280,430,**080**. Same parameters, fused differently for the older `MiniMaxH3DiTModel` /
  `MiniMaxH3Pipeline` classes. The text encoder *is* byte-identical across all three layouts (same
  index blob), so the "partitions duplicate the components" claim holds for the TE and the VAEs but
  **not** for the DiT.

  **Conversion consequence:** take weights from the **root** layout (638-name,
  `MiniMaxH3Transformer3DModel`) and take the partition metadata from the **partition**
  `model_index.json` files, which are the only place `sigma_shift_scales` and the task map live.
  Both partitions declare `{"video": 12.0, "audio": 3.0}`, matching `scheduler/config.json`
  (`shift: 12.0`) and `audio_scheduler/config.json` (`shift: 3.0`), so the two scheduler configs
  are partition-invariant and belong in the shared mirror.

---

## 3. Repository naming and tier subdirectory layout

### 3.1 Repositories

```
SceneWorks/minimax-h3-mlx                 # t2va + fl2va DiT, tiered
  bf16/transformer/{config.json, *.safetensors, *.safetensors.index.json}
  q8/transformer/...
  q4/transformer/...
  LICENSE, README.md

SceneWorks/minimax-h3-ref-mlx             # ref2va DiT, tiered — same shape, different weights
  bf16/transformer/...
  q8/transformer/...
  q4/transformer/...
  LICENSE, README.md

SceneWorks/minimax-h3-components-mlx      # shared across BOTH entries
  bf16/text_encoder/...                   # includes tokenizer.json, vocab.json, merges.txt,
  q8/text_encoder/...                     #   chat_template.json, preprocessor_config.json —
  q4/text_encoder/...                     #   exactly as upstream text_encoder/ already ships them
  video_vae/                              # dense bf16, tier-agnostic
  audio_vae/                              # dense bf16, tier-agnostic
  scheduler/  audio_scheduler/            # config-only, tier- and partition-invariant
  LICENSE, README.md
```

Naming follows the shipped precedent rather than inventing a scheme:

- `<family>-mlx` for the tiered primary artifact — `SceneWorks/ltx-2.3-mlx`,
  `SceneWorks/wan2.2-t2v-a14b-mlx`.
- `<family>-Components-mlx` for a shared multi-component mirror — `SceneWorks/Mage-Flow-Components-mlx`
  is the only existing instance and is the direct template for this layout
  (`config/manifests/builtin.models.jsonc:85-150`).
- `<tier>/` top-level subdirs with the component directory beneath, so a tier's `files` glob is a
  single `"<tier>/*"` (DiT repos) or `"<tier>/text_encoder/*"` (components mirror). The worker's
  glob crosses `/` (`scripts/check-download-patterns.mjs:55-58`), so `q4/*` legitimately matches
  `q4/transformer/config.json`.

**Tokenizer and processor live inside `<tier>/text_encoder/`, not as separate components.** This
matches upstream, where `text_encoder/` already carries `tokenizer.json`, `vocab.json`,
`merges.txt`, `chat_template.json`, `preprocessor_config.json` and `video_preprocessor_config.json`
alongside the weights. Duplicating those ~23 MB across three tiers costs 46 MB extra and buys a
self-contained component that one `subdir` fully stages. Splitting them into their own
`componentId` would add two more co-requisite rows per entry for no benefit.

### 3.2 The `-mlx` suffix and the candle backend

**Decision: one repo triple, all three platforms on every row. Do not publish `-candle` forks.**

The `-mlx` suffix is a naming artifact, not a backend contract, and the reason a family sometimes
needs a second repo is **not** that candle cannot read MLX-quantized weights. It can, and does:
the candle loader packed-detects a `.scales` sibling per `Linear` independently of the requested
quant and degrades to dense when there is none
(`crates/sceneworks-worker/src/image_jobs/flux2_edit_candle.rs:109`,
`image_jobs/base.rs:918-922`). Candle's LTX repo constant is literally the MLX repo —
`const CANDLE_LTX_REPO: &str = "SceneWorks/ltx-2.3-mlx";`
(`crates/sceneworks-worker/src/video_jobs/candle.rs:59`). The clearest statement of the rule is
Mochi's (`crates/sceneworks-worker/src/video_jobs/mochi.rs:14-19`, `:38-41`):

> **ONE repo, TWO backends.** … the `.scales`-detect seam lets candle ingest the mlx-affine tiers
> 1:1 … `SceneWorks/mochi-1-candle` was never published.

**A second repo is a *directory-layout* problem.** Wan needs one because its MLX tier is flat
(`model.safetensors`, `t5_encoder.safetensors`, `vae.safetensors`, `tokenizer.json`, `config.json`
— `crates/sceneworks-worker/src/wan_ti2v_5b_tier_build.rs:47-55`) while `candle-gen-wan` resolves a
diffusers tree of `transformer/`, `text_encoder/`, `vae/`, `tokenizer/` directories
(`video_jobs/candle.rs:254-261`). Bernini splits for the same reason and is named
`SceneWorks/bernini` with **no** suffix — proof the suffix carries no meaning. The historical
direction is the other way: `candle_lens_repo`, `candle_ideogram_repo`, `candle_boogu_repo` and
`candle_krea_repo` were all retired once candle learned to packed-detect the `-mlx` tier.

**So the requirement this imposes on sc-17150 is a layout requirement, not a publishing one:** emit
each tier subdir as the *component-directory* tree (`<tier>/transformer/`,
`<tier>/text_encoder/`), which is the shape §3.1 already specifies and the shape a candle provider
resolves. Do not emit Wan's flat form. At 345 GB of hosted artifacts a `-candle` fork would double
the upload and every future re-upload for identical weights.

Residual risk is small and the fallback is additive: if sc-17154/sc-17155 finds the candle provider
cannot resolve this tree, add `-candle` repos and split the `platforms` arrays — nothing in the
`downloads[]` shape below changes. Note that omitting `platforms` entirely also means "all
platforms" (`apps/rust-api/src/models.rs:4885-4907`), and that platform filtering is a no-op unless
at least one entry in the model is platform-tagged; declaring all three explicitly is the clearer
authoring choice and matches LTX.

### 3.3 Build and upload tooling for sc-17150

The precedent is a `#[ignore]`d on-device Rust test per family under
`crates/sceneworks-worker/src/`: `wan_ti2v_5b_tier_build.rs`, `wan_t2v_14b_tier_build.rs`,
`wan_i2v_14b_tier_build.rs`, `bernini_tier_build.rs`. Each builds `bf16/` + `q8/` + `q4/`, prints a
`[[TIER]] {json}` line carrying the exact byte totals that backfill `estimatedSizeBytes`, and prints
the manual `hf upload … --include 'bf16/*' 'q8/*' 'q4/*'` command
(`wan_ti2v_5b_tier_build.rs:217`). sc-17150 should follow that shape; the quantization helpers it
calls live behind `runtime_macos::providers::…` in the inference repo.

Because these are `#[ignore]`d, sc-17150 must assert the test actually **ran** — a 0.00 s result is
a silent skip, not a pass.

---

## 4. What is tiered, what is shared, and the `downloads[]` shape

### 4.1 The split

| Component | Tiered? | Why |
| --- | --- | --- |
| `transformer` | **yes** — q4/q8/bf16 | 66.28 GB; the whole point of tiering |
| `transformer_ref` | **yes** — q4/q8/bf16 | identical structure; own manifest entry (§7) |
| `text_encoder` | **yes** — q4/q8/bf16 | 66.72 GB. Too large to leave dense; a dense TE would put a 66.7 GB floor under the q4 tier and make q4 pointless |
| `video_vae` | **no** — dense bf16 | Wan/LTX/Mage precedent: the VAE stays dense in every tier. Mage ships byte-identical `vae` sizes across q4/q8/bf16 (`builtin.models.jsonc:119-150`) |
| `audio_vae` | **no** — dense bf16 | 0.605 GB, and 91.7% of it is 1-D/3-D convolution weight that MLX affine quantization cannot pack at all |
| tokenizer / processor | **no** | rides inside `<tier>/text_encoder/` |
| schedulers | **no** | config-only, 8 KB |

The tiered text encoder is what makes this the *Mage-Flow* layout rather than the *LTX* layout.
LTX's shared Gemma co-requisite is tier-agnostic and dense; Mage's is per-tier. H3 needs per-tier
because its TE is 66.7 GB, so the co-requisite rows carry `variant` + `subdir`, which is exactly
what sc-14980 added the `subdir` key for.

### 4.2 Sketched `downloads[]` for `minimax_h3`

Every key below is verified present in `packages/schemas/model-manifest.schema.json`; none require a
schema change (§4.4). Sizes are placeholders — `<…>` marks values sc-17150 backfills.

```jsonc
"downloads": [
  // ---- tiered DiT: q4 default, one repo, one glob per tier ----
  {
    "provider": "huggingface",
    "repo": "SceneWorks/minimax-h3-mlx",
    "revision": "<40-hex upload commit>",
    "variant": "q4",
    "default": true,
    "files": ["q4/*"],
    "platforms": ["macos", "windows", "linux"],
    "estimatedSizeBytes": <q4 DiT>,
    "footprint": { "diskSizeBytes": <q4 DiT>, "residentMemoryBytes": null, "peakMemoryBytes": null }
  },
  { /* variant: "q8",  files: ["q8/*"]  */ },
  { /* variant: "bf16", files: ["bf16/*"] */ },

  // ---- shared text encoder: one row per tier, scoped by `variant` ----
  {
    "provider": "huggingface",
    "repo": "SceneWorks/minimax-h3-components-mlx",
    "revision": "<40-hex components commit>",
    "coRequisite": true,
    "componentId": "text_encoder",
    "variant": "q4",
    "subdir": "q4/text_encoder",
    "files": ["q4/text_encoder/*"],
    "platforms": ["macos", "windows", "linux"],
    "estimatedSizeBytes": <q4 TE>
  },
  { /* variant: "q8",   subdir: "q8/text_encoder",   files: ["q8/text_encoder/*"]   */ },
  { /* variant: "bf16", subdir: "bf16/text_encoder", files: ["bf16/text_encoder/*"] */ },

  // ---- shared VAEs: tier-agnostic (no `variant`), always applicable ----
  {
    "provider": "huggingface",
    "repo": "SceneWorks/minimax-h3-components-mlx",
    "revision": "<40-hex components commit>",
    "coRequisite": true,
    "componentId": "video_vae",
    "subdir": "video_vae",
    "files": ["video_vae/*"],
    "platforms": ["macos", "windows", "linux"],
    "estimatedSizeBytes": 10415475936
  },
  { /* componentId: "audio_vae", subdir: "audio_vae", files: ["audio_vae/*"], 605306340 */ }
]
```

`minimax_h3_ref` is the same array with `repo` swapped to `SceneWorks/minimax-h3-ref-mlx` on the
three tier rows. **The five co-requisite rows are byte-for-byte identical between the two entries**,
including `repo` and `revision` — that is what makes the shared floor download once.

### 4.3 Why this satisfies the co-requisite pre-download-floor contract

The story asks for confirmation that this path satisfies co-requisite floor semantics. It does, and
the mechanism is worth stating precisely because a co-requisite is a **weights floor, not an
alternative to tiers**:

- **Install gates on them.** `install_state_for` (`apps/rust-api/src/models.rs:3389-3484`) computes
  `installed: primary_installed && hard_co_requisites_installed`. Hard is the default. The model
  cannot report installed with the TE or a VAE missing.
- **Tier-scoped co-requisites gate per tier.** For rows carrying `variant`
  (`models.rs:3403-3444`) the gate is satisfied when at least one tier's full set is cached — so a
  q4 user is not held hostage to the bf16 TE.
- **Install queues one job per row.** `models.rs:536-556`; the worker is one repo per job.
- **The job-time seam does not fetch.** `resolve_co_requisites_for_tier`
  (`crates/sceneworks-worker/src/model_jobs.rs:1892-2044`) resolves from cache only and returns
  `WorkerError::InvalidPayload` when a declared component is absent, so a render fails *before* the
  engine load rather than triggering a mid-render hub fetch. That is the floor property.
- **Tier disambiguation is enforced.** With several rows sharing a `componentId`, a resolved tier is
  required or the worker refuses to guess (`model_jobs.rs:1946-1972`) — "picking the wrong one would
  silently serve another tier's weights". Our three `text_encoder` rows have distinct `variant`s, so
  this resolves cleanly; the two VAE rows have distinct `componentId`s and no `variant`, so they are
  single-candidate and tier-agnostic.
- **`subdir` is confined at runtime.** Applied through `safe_join` (`model_jobs.rs:2055-2072`), so
  the mirror layout cannot escape the snapshot. Note the schema pattern
  `^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$` **does not** stop traversal — `..` is spelled entirely from
  characters it allows, so `../q8/text_encoder` validates. It rejects only absolute paths and
  exotic characters. `safe_join` is the sole containment guard, and the value reaches it from the
  payload `modelManifestEntry`, unvalidated over the LAN jobs API. Do not read the pattern as a
  containment guarantee; `test_subdir_pattern_does_not_stop_traversal_only_safe_join_does` pins
  this.
- **They survive per-tier delete structurally.** `delete_model_variant` resolves only
  non-co-requisite rows (`models.rs:1188`, `:5004`), and `remove_tier_artifacts`
  (`models.rs:1314-1426`) adds every non-selected path's canonicalized blob to `retained_reals`,
  which blocks removal. Deleting the bf16 tier can never strand the q4 user's TE.

**Binding requirement for sc-17143/sc-17154:** the componentIds `text_encoder`, `video_vae` and
`audio_vae` must appear in the engine's `ModelDescriptor::required_components`. The manifest audit
explicitly cannot see that (`tests/test_builtin_manifest_audit.py:1243-1247`) — it is a runtime
resolve check only. A mismatch is a job-time failure with no CI signal.

### 4.4 Schema conformance — no new keys

| Key used | Schema location | Constraint satisfied |
| --- | --- | --- |
| `provider`, `repo` | `model-manifest.schema.json:1323`, `:1327` (both listed in `required` at `:1295`) | `downloads.items` is `additionalProperties: false` (`:1296`), so an undeclared key is a hard error |
| `revision` | `:1331` | 40-hex lowercase; **required** on co-requisites by `tests/test_builtin_manifest_audit.py:1059-1096`, and by `:1099-1110` on any `SceneWorks/*` row — so all eight rows must pin |
| `variant` | `:1351` | enum `bf16 \| q8 \| q4 \| int8-convrot \| training` (`:95-99`) |
| `default` | `:1347` | exactly one applicable non-co-requisite per OS (`test_builtin_manifest_audit.py:1000-1036`) |
| `coRequisite` | `:1355` | — |
| `componentId` | `:1336` | `^[a-z][a-z0-9_]*$`; `allOf` requires `coRequisite: true` (`:1314-1319`) |
| `subdir` | `:1341` | path pattern; `allOf` requires `coRequisite: true` (`:1306-1311`) |
| `estimatedSizeBytes`, `footprint`, `files`, `platforms` | `:1369`, `:1373`, `:1377`, `:1382` | — |

`downloads` has no `maxItems`, and multiple distinct repos among co-requisite rows already ship
(`mmaudio_small_16k` spans two upstream repos; Mage spans two SceneWorks repos). Eight rows per entry
is unremarkable.

**No schema change is required by this layout.** A regression test asserting exactly that ships with
this document: `tests/test_minimax_h3_download_shape.py`.

Two authoring-time gates sc-17158 must satisfy that the schema does not express:

- `node scripts/check-download-patterns.mjs --model minimax_h3` — every declared glob must match a
  real file. It is a **manual preflight, deliberately not in CI** (it calls the HF API;
  `scripts/check-download-patterns.mjs:24-31`). A glob matching zero files hard-fails the user's
  download, not the build.
- `apps/desktop/licenses/manifest.json` — see §8.

---

## 5. Sizing table

### 5.1 The bits-per-weight model, and its validation

SceneWorks packs every tier at **`group_size: 64`** — established, not assumed:
`crates/sceneworks-worker/src/wan_ti2v_5b_tier_build.rs:57` (`const GROUP_SIZE: i32 = 64;`),
`wan_t2v_14b_tier_build.rs:34-35` and `wan_i2v_14b_tier_build.rs:34-36`
(`[("bf16", None), ("q8", Some((8, 64))), ("q4", Some((4, 64)))]`), and
`bernini_tier_build.rs:71-73` ("the canonical mflux/reference default"). The sidecar written next
to each tier is `{"bits": bits, "group_size": 64}`.

MLX affine quantization stores an fp16 scale and an fp16 bias per group. At `group_size: 64` that
is 32 bits of metadata per 64 weights = 0.5 bits/weight, so:

```
bytes(tier) = bytes_quantizable_bf16 × (bits + 0.5) / 16  +  bytes_dense
q8 → 8.5/16 = 53.125% of bf16      q4 → 4.5/16 = 28.125% of bf16
```

A tensor is quantizable when it is 2-D and its last dimension is a multiple of 64. For the H3 DiT
that is **99.98%** of the bytes; the dense remainder is the AdaLN biases (9.68 MB) and
`proj_in.weight` `[5376, 96]` (2.06 MB, in-features 96 is not a multiple of 64).

Validated against six shipped SceneWorks DiT artifacts (Mage-Flow, whose tier subdirs are DiT-only
and therefore a clean signal):

| | bf16 shipped | q8 shipped | q8 predicted | err | q4 shipped | q4 predicted | err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all six `mage_flow*` | 8,231,571,754 | 4,374,163,324 | 4,373,022,494 | **+0.03%** | 2,326,294,167 | 2,315,129,555 | **+0.48%** |

The +0.48% q4 residual is the documented sc-15071 8-bit floor on `norm_out.linear`. The model is
sound to well under 1%.

**The 0.5 bits/weight overhead is derived from the MLX format and validated above; it is not stated
anywhere in this repository.** Do not cross-reference the "~4.5 effective bits/weight" figure in
`crates/sceneworks-worker/src/image_jobs/flux2_comfyui_candle.rs:213-214` — that is **NVFP4**
(E2M1 elements with FP8-E4M3 block scales in a W4A4 regime), a deliberately distinct tier that
`image_jobs/tier_resolver.rs:77-79` calls out as "**not** an int4-affine equivalent". The
coincidence of the number is a trap.

### 5.2 Per-component (ESTIMATES for q8/q4)

| Component | bf16 (exact) | q8 (est.) | q4 (est.) |
| --- | ---: | ---: | ---: |
| DiT — `transformer` **or** `transformer_ref` | 66.280 GB | 35.218 GB | 18.651 GB |
| Text encoder, **trimmed** (§6) | 51.506 GB | 27.491 GB | 14.683 GB |
| *(text encoder, untrimmed, for reference)* | *66.715 GB* | *35.570 GB* | *18.960 GB* |
| video VAE — dense in every tier | 10.415 GB | 10.415 GB | 10.415 GB |
| audio VAE — dense in every tier | 0.605 GB | 0.605 GB | 0.605 GB |
| schedulers + configs | 0.023 GB | 0.023 GB | 0.023 GB |
| **shared floor (VAEs + configs)** | **11.044 GB** | **11.044 GB** | **11.044 GB** |

### 5.3 Per-tier install and hosted totals

What one user downloads:

| Tier | base entry (t2va + fl2va) | + ref2va entry | ref2va delta |
| --- | ---: | ---: | ---: |
| q4 | **44.378 GB** | 63.029 GB | +18.651 GB |
| q8 | **73.752 GB** | 108.970 GB | +35.218 GB |
| bf16 | **128.830 GB** | 195.111 GB | +66.280 GB |

What we host:

| Layout | full surface (both DiTs × 3 tiers) | base entry only |
| --- | ---: | ---: |
| **Chosen** — shared components mirror | **345.022 GB** | 224.873 GB |
| Rejected — Wan self-contained tiers | 493.921 GB | 246.960 GB |
| delta | **+148.898 GB (+43%)** | +22.088 GB |

**Honest correction to the story text.** The story justifies the decision with "three
self-contained tiers would be ~250 GB+". That figure is right (246.96 GB) but it is not the
argument: the *chosen* layout is 224.87 GB for the same scope, so the tier-split alone saves only
22.09 GB — two extra copies of the 11.04 GB dense VAE floor. The decision earns its keep on two
other axes:

1. **Tiering the text encoder at all** is worth far more than the sharing: 66.72 GB → 14.68 GB at
   q4. Without it the q4 tier's floor is 77.8 GB and q4 is pointless. The story frames the TE as a
   *shared dense* component; it must be shared **and tiered**.
2. **The second partition.** Ref2VA doubles the self-contained layout to 493.92 GB but adds only
   the DiT to the shared one. That is where the 148.90 GB comes from, and it is only visible once
   `transformer_ref` is in scope — which is the §7 decision.

### 5.4 Risk-adjusted ceilings — read this before pinning q4 targets

Two q4 assumptions could break, and both have a shipped precedent for breaking:

- **The text encoder.** sc-15071 found that Qwen3-VL packed uniformly at 4 bits "did not render the
  prompt at all, it produced a repeating tiled texture", and the fix was an 8-bit floor on all
  decoder layers (`builtin.models.jsonc:48-56`). Mage's shipped q4 TE is **91.5%** of its q8 TE, not
  28% of bf16. H3's TE is the *same architecture family at 32B*. Assume this floor may be required.
- **The DiT's AdaLN projection.** `adaln_proj.linear` is 39.3% of the DiT (26.021 GB bf16) and
  drives LayerNorm scale/shift — structurally the same role as `norm_out.linear`, the one projection
  Mage had to floor at 8 bits.

| | optimistic (uniform q4) | risk-adjusted ceiling |
| --- | ---: | ---: |
| q4 DiT | 18.651 GB | **25.168 GB** (AdaLN + `norm_out` at 8 bits) |
| q4 TE (trimmed) | 14.683 GB | **26.872 GB** (all decoder layers at 8 bits) |
| q4 base install | 44.378 GB | **63.084 GB** |

**sc-17150 must build and eyeball q4 before the number is pinned**, and if a floor is needed the
manifest carries `precisionFloors` for exactly this (Mage declares
`{ "component": "textEncoder", "selectedTier": "q4", "residentTier": "q8" }`,
`builtin.models.jsonc:19-23`). sc-17158 should expect to need it.

**Cross-story caveat for sc-17146.** The epic's AdaLN precompute-and-evict lever interacts with
quantization: the modulation vectors are computed *from* `adaln_proj` before it is evicted, so a q4
AdaLN perturbs every layer's modulation for the whole schedule, with no per-step averaging to hide
it. Whatever tier the DiT ships at, sc-17146 should evaluate keeping AdaLN at 8 bits — the disk
cost is +6.5 GB at q4 and the denoise-resident cost is **zero**, because the weights are evicted
after the precompute pass. That is an unusually cheap safety margin and the numbers above make it
purchasable.

For reference, DiT composition (exact, bf16):

| Bucket | Bytes | GB | Share |
| --- | ---: | ---: | ---: |
| `transformer_blocks` excluding AdaLN | 38,536,268,800 | 38.536 | 58.1% |
| `adaln_proj` (timestep-only, evictable) | 26,020,915,200 | 26.021 | 39.3% |
| `token_refiner` (2 blocks) | 1,541,461,504 | 1.541 | 2.3% |
| stem/head (embedders, `norm_out`, `proj_*`, `audio_proj_*`) | 181,784,576 | 0.182 | 0.3% |

---

## 6. Trimming the text encoder — evaluated, adopted, with conditions

The epic proposes dropping the text encoder's unused upper layers. **Adopted**, and it should also
drop `lm_head`.

### 6.1 The saving is real and larger than proposed

The TE is Qwen3-VL-32B: 64 decoder layers at exactly 975,196,672 bytes each (uniform),
`hidden_size` 5120, `vocab_size` 151936, `tie_word_embeddings: false`, plus a 27-layer / 1152-wide
vision tower (1.191 GB) and a 1.556 GB token embedding.

| Dropped | Bytes | GB | Share of TE |
| --- | ---: | ---: | ---: |
| decoder layers 50–63 (14 × 975,196,672) | 13,652,753,408 | 13.653 | 20.5% |
| `lm_head` (151936 × 5120, untied) | 1,555,824,640 | 1.556 | 2.3% |
| **total** | **15,208,578,048** | **15.209** | **22.8%** |

Resulting trimmed TE: **51,506,202,080 B (51.506 GB)**. Across the three hosted tiers the saving is
**27.566 GB** (bf16 15.209 + q8 8.080 + q4 4.278).

`lm_head` is a bonus the epic did not name: because `tie_word_embeddings` is `false` it is a real
1.556 GB tensor, and a text *encoder* never runs it. The token embedding (`embed_tokens`, also
1.556 GB) **is** needed and stays.

### 6.2 Does anything else need those layers? — No, with one caveat

- The DiT context is the layer-50 hidden state. Both readings of "layer 50" (1-indexed layer 50, or
  HF-convention `hidden_states[50]` where index 0 is the embedding output) resolve to *the output of
  decoder index 49*, so layers 50–63 are unreachable under either. Layers ≥50 cannot influence
  `hidden_states[50]` — a decoder is causal in depth.
- `deepstack_visual_indexes` is `[8, 16, 24]`, all far below 50, and indexes the **vision tower**,
  which is kept whole.
- Nothing else in the H3 pipeline runs the TE. `modular_model_index.json` uses it only as
  `text_encoder`.

**Caveat, and it is the one real risk:** the withheld `H3-Context-IR` prompt-refiner is a *separate*
hosted model, so nothing in the shipped pipeline needs generation from this TE today. But if a
future story reaches for this Qwen3-VL-32B as a local prompt refiner (SceneWorks already has a
prompt-refine path), a trimmed artifact cannot do it — `lm_head` and 14 layers are gone. The
mitigation is that this is a *hosting* decision: re-uploading the untrimmed TE is a build step, not
a code change. Recorded, not blocking.

### 6.3 Does dropping them break loading? — Yes, unless the hosted config is rewritten

This is a binding requirement on sc-17150, not a free saving:

1. **`config.json` must be rewritten** in the hosted artifact: `text_config.num_hidden_layers`
   64 → 50. A loader that constructs 64 layers against a 50-layer weight map fails, and one that
   constructs 50 against a 64-layer config silently mismatches.
2. **`architectures` / the class name.** Upstream declares `Qwen3VLForConditionalGeneration`, which
   expects `lm_head`. The hosted artifact is an encoder-only derivative; sc-17143 must load it as
   such rather than through a strict full-model path.
3. **The index manifest must be regenerated** so no `weight_map` entry points at a dropped tensor.
4. **Prove the layer index before trimming.** Cost of being wrong by one layer is 0.975 GB of
   re-upload plus a wrong-context regression that parity fixtures might not catch if they are
   captured *against the trimmed artifact*. **sc-17143 must assert `hidden_states[50]` equality
   between the untrimmed upstream weights and the trimmed artifact** — the tensors are bit-identical
   under a correct trim, so this is an exact-equality check, not a tolerance. Do this before
   sc-17150 uploads.

---

## 7. `transformer_ref` — a second manifest entry

**Decision: `minimax_h3_ref` is its own catalog entry, sharing one components mirror with
`minimax_h3` via identical co-requisite rows.**

### 7.1 The readiness constraint, established from code

The story's premise — "the readiness check is one repo per entry" — is **true on the primary
(non-co-requisite) side and false on the co-requisite side**. Establishing it precisely:

- `install_state_for` derives `managed_path`, `cache_path`, receipt file sets, stale-file
  detection and breaking-update detection from **one** repo: the `default: true` non-co-requisite
  row's (`apps/rust-api/src/models.rs:3262-3504`, via `model_download`, `:4910-4925`).
  `receipt_file_sets` filters `if entry.get("repo") != Some(repo) { return None }` (`:2229-2231`).
- `model_artifact_paths`, which whole-model delete consumes, uses **only** that repo
  (`models.rs:5759-5791`). A second primary repo is orphaned on delete.
- Per-tier state is the exception: `model_variant_states` reads `repo` per row (`models.rs:3651-3668`).
- It is a **de-facto** rule, not an assertion. There is no catalog-wide check, and a two-repo
  primary side already ships — `wan_2_2_t2v_14b` puts candle q4/q8 in `SceneWorks/wan2.2-t2v-a14b-candle`
  and bf16 in `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, which forced a bespoke worker resolver
  (`crates/sceneworks-worker/src/video_jobs/candle.rs:181-190`).
- **Co-requisite rows are explicitly unconstrained**: multiple distinct repos already ship
  (`mmaudio_small_16k` spans two), the schema states "A co-requisite may be shared by several
  models, so deleting one model does NOT remove it" (`model-manifest.schema.json:1357`), and there
  is no row cap.

So: keeping each entry's *primary* side on one repo keeps install state, stale detection,
breaking-update detection and whole-model delete correct for free. Sharing the *floor* across
entries is a supported, shipped pattern.

### 7.2 What breaks under each option

| Option | Shape | What breaks |
| --- | --- | --- |
| **A — chosen.** Two entries, `minimax_h3` + `minimax_h3_ref` | each entry: 3 tier rows on its own repo + 5 identical co-req rows | Two catalog rows for one upstream model. `capabilities` split across them, so routing, `VIDEO_UI_MODES` and the `validate_video_job` allow-list must send ref2va jobs to the second id. Two `apps/desktop/licenses/manifest.json` ids. Two memory-matrix rows. UI must not present them as unrelated models. |
| B — one entry, `transformer_ref` as a **hard** per-tier co-requisite | 3 tier rows + 8 co-req rows | Every t2va user is forced to download the ref DiT to reach `installed`: **+18.65 GB on a 44.38 GB q4 install (+42%)**, +66.28 GB at bf16. Unacceptable. |
| B′ — same, but `required: "soft"` | as above | `soft_co_requisite_update` sets `update_available: true` **permanently** for anyone who never wants Ref2VA (`models.rs:3458-3464`, `:3475-3484`) — a phantom update badge that can never be cleared. Also forces `transformer_ref` out of `required_components` onto the bespoke `resolve_optional_component` seam, used today by exactly one component (`audio_jobs.rs:713-718`). |
| C — one entry, ref DiT as extra tier rows | — | Impossible. `variant` is the enum `bf16\|q8\|q4\|int8-convrot\|training` (`schema:95-99`); there is no axis for "which checkpoint". |

Option A also matches shipped precedent directly: `wan_2_2_t2v_14b` and `wan_2_2_i2v_14b` are two
entries for one upstream family, differing in checkpoint and `capabilities`. And upstream's own
`modular_model_index.json` listing both DiTs in one pipeline is not a counter-argument — it
describes a *pipeline*, and SceneWorks' catalog unit is a *downloadable artifact with one install
state*.

### 7.3 Consequence to hand to sc-17158 / sc-17149

- `minimax_h3` declares the t2va/fl2va capabilities; `minimax_h3_ref` declares the reference-driven
  one. Do not declare ref2va capability on `minimax_h3` — R5 (declaration is not reachability) cuts
  both ways, and a capability whose weights are in another entry is a 400 waiting to happen.
- The five co-requisite rows must be **identical in `repo` and `revision`** across both entries, or
  the shared floor downloads twice.
- `ui.description` on both entries must say they are two halves of one model.

### 7.4 Known gap, not absorbed

Deleting the last of the two entries leaves the components mirror on disk with no owner —
`model_artifact_paths` never enumerates co-requisite repos, by design (`schema:1357`). What is
stranded is the 11.04 GB of dense VAEs plus whichever text-encoder tiers are cached — **25.7 GB
after a q4-only install, 62.6 GB after bf16** — far more noticeable than the existing instances of
this behaviour (chatterbox's `perth`). This is **pre-existing repo-wide behaviour, not introduced here**; recorded as a follow-up
rather than absorbed into this story.

---

## 8. Gates

| Gate | Applies to this story? | Applies to sc-17150 / sc-17158 |
| --- | --- | --- |
| About→Licenses coverage | **no** | **yes, sc-17158** |
| `packages/schemas/model-manifest.schema.json` change | **no** — §4.4 | no |
| `node scripts/check-download-patterns.mjs` | no | **yes, after upload** |
| `npm run check` | run, passes | yes |
| `python -m pytest tests/ -m "not e2e and not parity"` | run, passes | yes |

**Correction to the epic's prep comment.** Finding #9 on the epic names
`apps/web/src/simple/licenseTerms.js` as the About→Licenses gate. That file is **not a registry** —
it is 61 lines of three pure functions that *derive* a commercial/non-commercial badge from a
free-text licence string. The actual gate is `scripts/check-license-coverage.mjs:785-790`:

```
model "<id>" ships weights but has NO About→Licenses entry. Add its upstream license
under apps/desktop/licenses/<component>/, list the id in that component's `models`,
and wire the document key in bundledLicenses.js.
```

It runs in the `parity-scaffold` job (`.github/workflows/check.yml:205-207`), and its trigger is
**a catalog model with a non-empty `downloads[]`**, satisfied by listing the model `id` under a
component's `models[]` in `apps/desktop/licenses/manifest.json`. Consequences:

- This docs-only story adds no manifest entry, so the gate is inert here. Correct placement is
  **sc-17158**.
- sc-17158 must register **two** ids, `minimax_h3` and `minimax_h3_ref`.
- The licence text to vendor is the MiniMax H3 Community License Agreement; upstream also ships
  `docs/QA-about-License.md` in the model repo. The licence *classification* call is sc-17227's.

There is no markdown lint, link checker, spell check, docs index or citation gate in this
repository; the only mechanical constraints on a file under `docs/` are LF endings
(`.gitattributes`) and no exotic C0 bytes (`scripts/check-source-control-bytes.mjs`).

---

## 9. Provenance of the numbers

Every bf16 figure is derived from `MiniMaxAI/MiniMax-H3` at snapshot
`939557dc319dd91227e30195a763f272ba7f8765` in the local HF cache, by:

1. reading `metadata.total_size` and the full `weight_map` from each component's safetensors index
   manifest;
2. parsing safetensors headers from the shards present on disk for exact `(dtype, shape)` per
   tensor;
3. extrapolating the remaining tensors by structural pattern (path segments that are pure digits
   normalized to a wildcard), then **validating the reconstruction against `total_size`**.

Reconstruction is **byte-exact**: text encoder 66,714,780,128 = index total; DiT 66,280,430,080 =
index total. The video VAE (10,415,475,936) and audio VAE (605,306,340) are fully on disk and
summed directly, not extrapolated. The DiT's two `token_refiner` blocks were the only tensors in no
downloaded shard; their shapes are recovered exactly from the main-block patterns and confirmed by
the residual (2 × 770,725,376 = 1,541,450,752, closing the total to the byte).

q8 and q4 figures are **ESTIMATES** from §5.1, validated to +0.03% / +0.48% against six shipped
artifacts. They are planning targets. sc-17150 replaces them with measured hosted directory sizes
before sc-17158 writes `estimatedSizeBytes` and `footprint`.
