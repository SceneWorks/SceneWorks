# Video-Model Feasibility Matrix (sc-1174)

> **Story:** [sc-1174 — Validate video model feasibility matrix](https://app.shortcut.com/trefry/story/1174)
> **Epic:** [1093 — SceneWorks: Research Tracks](https://app.shortcut.com/trefry/epic/1093)
> **Last updated:** 2026-07-27 (Krea Realtime 14B empirical entry — sc-8446 / epic 8431)
> **Status:** Validated — chosen model spiked empirically; comparison set web-verified (June 2026). Two claims in the story were tested and corrected.

**Provenance:** ⚙️ = empirically run on this machine (Apple M5 Max) · 🌐 = web-verified June 2026 · 📄 = SceneWorks code/manifest.

## Recommendation (v1 ordering)

1. **`ltx_2_3` — primary.** ~5.7× faster than Wan2.2 A14B on 24 GB-class HW (distilled 8-step); the
   **only** option with native **synchronized audio** (a real differentiator for a media app);
   richest native conditioning (first/last/keyframe-bridge + region-regen); best Apple/MLX story;
   longer practical clips (~15–20 s). ⚙️ Ran natively here. **License caveat below.**
2. **`wan_2_2` TI2V-5B — safe default fallback.** Dense 5B, ~10 GB, 24 fps native, **Apache-2.0 /
   ungated**, fits 24 GB trivially (even ~8 GB). Removes all license risk; the workhorse for
   low-VRAM and license-sensitive users.
3. **`wan_2_2` A14B — quality tier / multi-GPU path.** 27B MoE (14B active); higher fidelity but
   needs fp8/GGUF + expert-swap on 24 GB (bf16 wants 80 GB) and is ~5–6× slower. First-class
   multi-GPU (FSDP + Ulysses) if SceneWorks ever scales out.

> **Update (sc-8446, July 2026):** a **fifth** video family now runs natively on Mac — joining
> `ltx_2_3`, the Wan2.2 entries, `scail2_14b` and `bernini` — **`krea_realtime_14b`** (Krea Realtime
> 14B, Wan-2.1-T2V-14B distilled via Self-Forcing into an autoregressive chunk generator, Apache-2.0).
> It is the only SceneWorks video engine that is *not* full-clip block diffusion, and it generates a
> full 81-frame 832×480 clip inside 28 GiB of MLX-active memory. 🔴 **Its shipped VAE-decode tiling
> currently corrupts the output (sc-15325)** — the generation is sound, the decode in front of it is
> not. Measured numbers below.

**Do not** make MLX video a v1 *gating requirement* on Mac — treat it as best-effort; NVIDIA is the
supported path. (Apple still runs LTX-2.3 natively — see empirical result — it's just memory-bound.)

## ⚠️ Two story claims, tested and corrected 🌐

The story description asserts two specifics. Validation says:

- **"LTX-2.3 best at ~15 s or less" → PARTIALLY TRUE — it's a quality sweet-spot, not the ceiling.**
  Hard model max is **~20 s**; ~15 s is a real quality knee (subject-drift beyond 12–15 s on portrait
  clips; "Standard" tier recommended "for temporal consistency across 15 s"). **Encode 15 s as a
  recommended sweet-spot, not a hard limit.**
- **"Wan2.2 may loop cleanly around ~7 s" → UNVERIFIED / LIKELY INCORRECT.** Native clips are **~5 s**
  (81 frames @16 fps A14B; 121 @24 fps TI2V-5B). Seamless looping is an **open feature request**, only
  achievable via community first=last-frame / last-frame chaining. **No source pins anything to 7 s.
  Do not encode a "7 s native loop" capability.**

## Empirical result — LTX-2.3 ⚙️

`ltx_real_weights_with_audio` on Apple M5 Max (LTX-2.3 q4 + Gemma-3-12B TE, 256², 9 frames, 24 fps,
synchronized audio, native MLX, in-process):

| Metric | Value |
|---|---|
| Gen time | 14.49 s (in-process) / 15.65 s wall |
| **Peak unified memory** | **53.4 GB** |
| Host RSS | 23.0 GB |
| Manifest `mlx.minMemoryGb` | 31 (`builtin.models.jsonc:2199`) |

> ⚠️ Measured peak (**53.4 GB**) is **~1.7× the 31 GB manifest estimate** — the Gemma-3-12B text
> encoder loads dequantized (~24 GB) alongside the q4 DiT + VAEs. Even a *minimal* 256²/9-frame clip
> uses 53 of 64 GB, so **64 GB Macs have little headroom for real-size LTX clips**. Feeds sc-1177.

## Empirical result — Krea Realtime 14B ⚙️ (sc-8446, 2026-07-27)

> 🔴 **The shipped decode path currently corrupts the output.** Read the "Decode defect" subsection
> before quoting anything here as a quality result. The *model* works; the VAE decode tiling in front
> of it does not.

**Harness configuration**, not a shipped one: `mlx-gen-krea-realtime`'s `generate_smoke` on Apple
M5 Max / 128 GB, **Q4 tier** of `SceneWorks/krea-realtime-14b-mlx` @ `e68e9a3d`, 832×480, **81 frames
@ 24 fps, 5 steps/chunk** (the reference Self-Forcing `denoising_step_list`), native MLX, in-process.

⚠️ Neither number matches the shipped manifest: `defaults.steps` is **6**, and `limits.durations`
`[2,3,4,5] @ 24 fps` snaps to 45/69/93/117 frames, so **81 frames is not requestable at all** and the
default 4 s clip is **93 frames**. A user at defaults should expect roughly **6–6.5 minutes**, not the
4.3 minutes below — see the projection under the table.

| Metric | Value (harness: 81 f, 5 steps) |
|---|---|
| Whole-clip wall | **256 s** |
| Load + UMT5 encode | 4.8–5.9 s |
| AR denoise | 211 s — **≈5.3 s/denoise step**, **31 s/chunk** mean (20 s first chunk → 36 s once the KV window fills) |
| ↳ why 31 × 7 = 217 ≠ 211 | the 211 s window runs first-step-mark → last-step-mark, so it *excludes* the first step's own compute; the per-chunk figures impute it. The ~6 s gap is that one step. |
| VAE decode (84→81 frames) | 38–39 s |
| **MLX peak, active** | **27.9 GiB** |
| MLX peak, active + allocator cache (sampled ⇒ lower bound) | ≥90.4 GiB |
| KV-cache residency (measured) | **7.14 GiB**, flat from chunk 2 on |
| Manifest `mlx.minMemoryGb` | 64 |

> **Per-step vs per-chunk.** A chunk boundary also carries the S5 clean-context KV-recompute forward,
> which emits no progress event. Averaging across boundaries inflates the per-step figure to 6.2 s; the
> honest per-*step* cost is **≈5.3 s** and the recompute is accounted inside the 31–36 s/chunk. Use
> **chunk** time for latency planning.

**Projection to shipped defaults** (4 s @ 24 fps = 93 frames = 24 latent = 8 chunks; 6 steps + 1
recompute = 7 forwards/chunk at ~6 s): ≈336 s of AR + ~44 s decode + ~6 s load ≈ **385 s (~6.4 min)**.

### Memory

Cumulative **active** peak by phase: 9.6 GiB through the first denoise step → 17.4 GiB after the AR
loop → **27.9 GiB** after the decode. The AR plateau closes against the components (Q4 DiT 7.8 + KV
7.14 + VAE 0.5 ≈ 15.4 GiB); the decode adds ~10.5 GiB.

- **`mlx.minMemoryGb: 64` is now a closed derivation, but not because of the decode.** That term is
  +10.5 GiB, not the ~2× once feared. A single per-model value must admit the heaviest *installable*
  tier, and **bf16** swaps a 7.8 GiB DiT for a 26.6 GiB one → **~47 GiB** active peak. Per tier:
  **Q4 ~40 / Q8 ~48 / bf16 64** (now carried in the per-download `footprint` blocks).
- **The ceiling does not grow with clip length** — 33- and 81-frame clips both peak at 27.6–27.9 GiB,
  because the decode is tiled. ⚠️ That is also what breaks the image; see below.
- ⚠️ `mlx::get_peak_memory` is the **active** high-water mark only. Active is nonetheless the right
  basis for a floor, because the allocator's cache is reclaimable under pressure — the ≥90.4 GiB
  active+cache figure is the reason not to quote `get_peak_memory` as "the memory this uses", not
  itself a requirement. (It is polled at 50 ms, so it is a lower bound.)

**KV cache.** 40 layers × 9360 tokens (a 6-latent-frame window × 1560 tok/frame) × 40 heads × 128 dims
× {k,v} × bf16 = **7.14 GiB**, confirming the S1 estimate exactly. It holds *activations*, so it is
bf16 on **every** weight tier — Q4 does not shrink it. 720p (3600 tok/frame → ~16.5 GiB) is therefore
the memory-relevant jump, not the tier.

### 🔴 Decode defect — the tiled VAE decode corrupts the clip (sc-15325)

**The AR model is fine; the decode in front of it is not.** Decoding the *same* latents single-pass
versus through the shipped tiling, at 832×480/36 frames (the largest geometry with a valid single-pass
reference — see the write-bound note):

| decode | latent tile / overlap | mean \|Δ\| vs single-pass | highlight clipping mean / worst | MLX active peak |
|---|---|---|---|---|
| single-pass (reference) | — | 0 | **0.08% / 0.25%** | 85.1 GiB |
| **tile 8 / overlap 2 — the ORIGINAL shipped value** | 2 / **0** | **18.5 /255** | **9.7% / 26.6%** | 19.8 GiB |
| **tile 8 / overlap 4 — what sc-8446 ships** | 2 / **1** | 17.1 | 5.2% / 14.7% | 19.8 GiB |
| tile 16 / overlap 4 | 4 / 1 | 7.5 | 1.8% / 12.9% | 38.5 GiB |
| tile 16 / overlap 8 | 4 / 2 | 6.4 | 1.4% / 7.0% | 38.5 GiB |
| tile 32 / overlap 8 | 8 / 2 | 2.5 | 0.08% / 0.25% | 75.8 GiB |
| tile 32 / overlap 16 | 8 / 4 | 2.0 | 0.14% / 1.1% | 75.8 GiB |

⚠️ Overlap is expressed in **output** frames but applied in **latent** space (÷4), and `split_spatial`
then clamps it to `tile − 1`. At the shipped latent tile of **2 the overlap cannot exceed 1**, so the
sc-8446 fix already takes it to the maximum available there and the tile-2 rows say nothing about how
much blending is worth.

At the shipped setting roughly **one frame in eight blows a quarter of its pixels to near-white**, with
violet/green chroma separation and rainbow fringing along moving edges. The same latents decode to a
clean photographic image single-pass. The artifact period is **8 output frames = the decode tile
stride**, which is what identifies the cause: it does **not** align with the 12-output-frame AR chunk,
so this is not the `sink_size = 0` bounded-window drift of sc-15127.

**Mechanism — read the pairs at a FIXED tile, where overlap has room to move:**

| comparison | mean \|Δ\| | worst-frame clipping | MLX active peak |
|---|---|---|---|
| overlap ×2 at latent tile 4 (1→2) | 7.5 → 6.4 (**−15%**) | 12.9% → 7.0% (**−46%**) | 38.5 → 38.5 GiB |
| overlap ×2 at latent tile 8 (2→4) | 2.5 → 2.0 (**−22%**) | already at the single-pass floor | 75.8 → 75.8 GiB |
| tile ×2 at matched overlap (4/1 → 8/2) | 7.5 → 2.5 (**−67%**) | 12.9% → 0.25% | 38.5 → **75.8 GiB** |
| tile ×2 at matched overlap (4/2 → 8/4) | 6.4 → 2.0 (**−70%**) | 7.0% → 1.1% | 38.5 → **75.8 GiB** |

**Tile size dominates** — roughly −67…−70% per doubling — because two latent frames is less temporal
context than the z16 decoder's convolutions need. But **overlap is a real secondary term**, not a
negligible one: −15…−22% on mean error and −46% on worst-frame clipping where it has room. An earlier
cut of these notes called blending a minor contributor on the strength of the tile-2 rows; that was a
degenerate comparison (the clamp had already capped it) and the claim is withdrawn.

Practical consequence for sc-15325: **raise the tile *and* scale the overlap with it.** Overlap is free
in *peak* (identical GiB across each pair — it changes the stride and so the number of passes, i.e.
wall time, not the resident window), so there is no reason to buy a bigger tile and leave the overlap
at one latent frame.

**Status.** sc-8446 shipped the free half — the overlap now survives the ÷4 to latent space (it was
literally 0), which halves the clipping at identical memory. The tile size is *not* raised here because
tile size **is** the memory bound (19.8 → 38.5 → 75.8 GiB), and raising it re-opens `mlx.minMemoryGb`
for this engine **and** for `mlx-gen-scail2`, which computes the identical budget. Tracked as
**sc-15325**.

**Blast radius.** The zero-overlap collapse needs `tile_frames ∈ {8, 12}`, i.e. `budget_frames < 16`,
i.e. **≥ ~233k px/frame**: 640×384, 512×512, 768×512, 832×480 and 1280×720 all collapse — but
512×384 does **not** (budget 17 → tile 16 → latent overlap 1). Both engines with hand-rolled pxframe
budgets are exposed: `krea_realtime_14b` and **`scail2_14b`**, which computes the identical
`overlap = tile_frames/4` and is shipping it unmeasured.

Once the root cause is tile size, the right metric is **latent** frames, not output frames. On that
metric: Wan z16/z48 route through `budgeted_plan` and bottom out at latent tile 8 / overlap 2 —
**clear**, by the table above. **LTX is not cleared**: its `temporal_scale` is 8, so its smallest
candidate bottoms out at latent tile **3** / overlap 1 — below the latent-4 tile that still measured
6.4/255 here. Unmeasured; on sc-15325's step 1 alongside scail2.

⚠️ **Write-bound note.** A single-pass z16 decode is only valid below `96 · frames · h · w ≤ i32::MAX`
— at 832×480 that is **56 output frames**. An 84-frame single-pass decode is 1.5× over it and MLX
writes silently wrong results, so it cannot be used as a reference at the full clip length. (This
caught a bad measurement during S13: against that corrupt reference every tiled candidate scored
58–71/255 and saturation collapsed 0.33 → 0.07.)

### Coherence — what the clip actually looks like

**Not** "coherent with minor stylization". Through the shipped decode the clip is **visibly broken**:
violet snow and rainbow fringing by frame 13 (half a second in), and a heavy colour cast late in the
clip. Measured on the 81-frame run, mean frame brightness walks **137 → 122** head-to-tail; on the
33-frame comparison the shipped decode holds saturation at **0.293 → 0.261** where a single-pass decode
of the same latents relaxes to **0.293 → 0.199**, i.e. the tiling *adds* saturation as the clip runs.
Highlight clipping is the clearest single number: **9.71% of pixels mean, 26.63% worst** against
**0.08% / 0.25%** single-pass.

The *underlying generation* is good — subject identity, gait and background hold across all 81 frames,
and single-pass decoding of the same latents yields a clean photographic clip. There is a *separate*
and much milder progressive stylization consistent with the sc-15127 bounded-window drift, but it is
**not** what a viewer notices and it was wrongly blamed for this in the first cut of these notes.

## Capability & spec matrix 🌐

| Dimension | **LTX-2.3 (22B)** | **Wan2.2 A14B (27B MoE/14B active)** | **Wan2.2 TI2V-5B (dense)** | **Krea Realtime 14B (dense AR)** ⚙️ |
|---|---|---|---|---|
| Text-to-video | Yes | Yes | Yes (unified) | Yes |
| Image-to-video | Yes | Yes | Yes | Yes (a still warms the AR KV cache) |
| First-frame cond. | Yes (native) | Yes (I2V) | Yes | Yes (that IS the i2v path) |
| Last-frame / first+last bridge | **Yes, native** (keyframe interp) | **Not in core Wan2.2** — via node / Fun-InP (FLF was Wan2.1) | via node, same caveat |
| Video extend / region-regen | **Yes** (retake/region) | No native continue-clip | No native | v2v restyle only (strength-controlled AR init); no extend/region |
| LoRA (infer / train) | Yes / Yes (`ltx-trainer`) | Yes / Yes (high+low-noise pair) | Yes / Yes | **Infer: the low-rank half of any Wan-2.1-14B-T2V LoRA** ⚙️ (a step-distill file's 647 `.diff`/`.diff_b` keys are silently dropped — sc-15326) / no trainer |
| **Native FPS** | 24/25/48/50; **24–25 rec.** | **16 fps** | **24 fps** | **24 fps** |
| Practical duration | **~20 s max; ~12–15 s sweet-spot** | **~5 s** (81f@16) | **~5 s** (121f@24) | AR loop is unbounded in principle; visible stylization drift accumulates past ~3 s (sc-15127) |
| **Audio** | **Yes — native synced A/V (24 kHz)** | No native audio | No | No |
| Frame constraint | `(F-1) % 8 == 0`, dims ÷32 | `(F-1) % 4 == 0` | same | `(F-1) % 4 == 0`, dims ÷16; latent frames in 3-frame AR chunks |
| **License** | **Custom "LTX-2 Community License"** — free commercial **only under $10M ARR**; anti-compete clause; **NOT Apache** | **Apache-2.0** | **Apache-2.0** | **Apache-2.0** |
| Gating | effectively ungated (verify acceptance click-through) | ungated | ungated | ungated |
| CFG | Yes | Yes | Yes | **No** — Self-Forcing distilled guidance out; one batch-1 forward/step, no negative prompt |
| Mac (measured) | 53.4 GB **unified-memory** peak, 256²/9f | no measured 14B Mac time | — | **27.9 GiB MLX-**active** peak (≥90.4 GiB active+cache), 256 s for 832×480/81f @ Q4, harness config** ⚙️ |
| ↳ basis caveat | host-level unified memory | — | — | MLX allocator only — **not** like-for-like with the LTX cell; the comparable figure is the ≥90.4 GiB active+cache |

## 24 GB NVIDIA VRAM & runtime 🌐

| | **LTX-2.3 22B** | **Wan2.2 A14B** | **Wan2.2 TI2V-5B** |
|---|---|---|---|
| bf16 on 24 GB | DiT fits; **Gemma 3-12B encoder is the squeeze (~24–27 GB)** | **No** (min 80 GB; 28.6 GB/expert ×2) | **Yes** (min 24 GB → ~8 GB offloaded) |
| GGUF Q4_K_M | distilled **17.8 GB** single file | **9.65 GB/expert** (~18 GB, loaded 1 at a time) | **3.43 GB** |
| 24 GB unlock | CPU-offload or **fp4 encoder ≈ 8.8 GB**; FFN chunking | MoE expert-swap + `--t5_cpu` + offload → ~6–8 GB @480p | fits natively |
| Head-to-head (RTX 5090, Q4, 832×480, 81f I2V) | **22.1 s warm / 48.5 s cold** | 125 s warm / 143.9 s cold | — |
| → speed | **~5.7× faster** (distilled 8-step) | baseline | 5 s/720p in <9 min |
| Multi-GPU | **none native** (community sharding only) | **first-class** (FSDP + Ulysses) | same framework |

LTX's 5.7× lead reflects the **distilled 8-step** checkpoint at 480p/Q4; it narrows with the full
non-distilled pipeline (~9 min multimodal) and at 4K.

## Apple / MPS / MLX feasibility 🌐⚙️
- **LTX-2 / 2.3 MLX is the strongest video MLX story** (multiple real ports; q4≈12 GB/16 GB-Mac,
  q8≈21 GB/32 GB-Mac, bf16≈42 GB/64 GB-Mac). ⚙️ SceneWorks' own `ltx_2_3` MLX engine ran here at
  53.4 GB peak (q4 + Gemma-12B), 15.6 s for a minimal clip.
- **Wan2.2 MLX is experimental** — only via `mlx-video`; on M2 Max/32 GB Wan2.2-14B "uses almost all
  32 GB" while LTX-2.3-22B ≈ 19.4 GB; only Wan2.1-1.3B runs comfortably. **No measured 14B/22B Mac
  generation times exist** (assume minutes).
- ComfyUI on MPS breaks on fp8 (silent CPU fallback); image-only MLX stacks (mflux/DiffusionKit) have
  no Wan/LTX video.

## Rust backend target 📄

The repo **already implements** this contract; below are validated deltas, not greenfield design.

- **Manifest:** `ltx_2_3` (adapter `ltx_video`) and `wan_2_2` / `wan_2_2_t2v_14b` / `wan_2_2_i2v_14b`
  already exist as `ModelKind::Video` (`builtin.models.jsonc:2394,2501,2608`). **Multi-file resources
  already supported** via typed per-platform `downloads` *and* the untyped `resources` named-slot map
  (`checkpoint`/`spatialUpscaler`/`distilledLora`/`gemma`); LTX-2.3 already uses it (`:2137`).
  - **Gaps:** (a) no per-file `sha256`/url/size — only HF `repo`+`file`+`estimatedSizeBytes`; consider
    per-file hashes for supply-chain integrity (more pointed given the LTX license obligation). (b)
    VRAM is untyped (`mlx.minMemoryGb` only, and the LTX run shows it undercounts ~1.7×) — add a
    **precision-keyed VRAM block** (`bf16`/`fp8`/`gguf_q4` + `text_encoder_vram`) given the LTX
    24-vs-32 GB encoder cliff and Wan A14B's 80 GB-bf16 / 24 GB-fp8 split.
- **Capability flags** (`ModelCapability`, `contracts.rs:542-552`): has `text_to_video`,
  `image_to_video`, `video_extend`, `video_bridge`. **Skew to fix:** manifest video entries use mode
  strings not in the enum — `first_last_frame`, `extend_clip`, `replace_person` — and `video_extend`
  (enum) ≠ `extend_clip` (manifest), a latent naming collision. Reconcile (add `first_last_frame` to
  the enum or document that `capabilities` are `ContractMode` strings).
- **Job payload:** `VideoGenerate/VideoExtend/VideoBridge/VideoUpscale` job types + a typed
  `VideoRequest` already carry `mode, prompt, duration, fps, width/height, seed, loras[],
  source_asset_id, last_frame_asset_id, source_clip_asset_id, bridge_right_clip_asset_id,
  model_manifest_entry` — **the full LTX/Wan conditioning surface**. Frame-snapping helpers exist
  (`ltx_frame_count` → `8k+1`, `wan_frame_count` → `4k+1`), matching the verified constraints.
  **Add:** surface the LTX **audio** track on the job result/asset (the muxed MP4 carries it, but
  audio-track metadata isn't separately modeled).
- **Scheduling constraints:** routing is capability-based, not VRAM-based. Given the LTX 24-vs-32 GB
  cliff and Wan A14B offload-dependence, add a **precision-aware admission check** (does
  model+precision fit the assigned GPU/unified free memory?) using the typed VRAM block — don't
  discover OOM at runtime. `JobSnapshot` already records post-hoc peak GPU mem.
- **Asset outputs:** `AssetFile` already video-capable (`path, mime_type, width, height, duration,
  fps`) — no new fields needed except optional audio-track metadata.

## Caveats / could-not-verify 🌐 (research limits, not story gaps)

**Krea Realtime (sc-8446):** the headline clip was initially characterized as "coherent with
progressive stylization" and the degradation attributed to sc-15127 bounded-window drift. Both were
wrong — the dominant artifact is the tiled VAE decode (**sc-15325**), identified by its 8-output-frame
period matching the decode tile stride rather than the 12-frame AR chunk. The bf16/Q8 memory ladder is
**derived** from measured Q4 figures by weight-size substitution, not separately measured. The
`~47 GiB` bf16 peak and the shipped-default timing projection are likewise derived, not run.

"Wan ~7 s loop" unsupported by any source; "LTX ≤15 s" true as sweet-spot only; Wan A14B fp8 on
exactly 24 GB @720p unverified (480p is the safe claim); no measured 14B/22B Mac generation times;
**"LTX-2.3 is Apache-2.0" is a widespread but incorrect claim** (authoritative LICENSE is the custom
Community License — high-confidence refutation); LTX HF acceptance click-through not ruled out; Wan
Fun-InP (FLF path) license not separately fetched. Full source list retained in research notes.

## Sources
Empirical: `ltx_real_weights_with_audio` on M5 Max; `mlx-gen-krea-realtime`'s
`tests/generate_smoke.rs` + `tests/style_lora_real_weights.rs` on M5 Max (sc-8446, inference repo). Web (June 2026): github.com/Lightricks/LTX-2
(+LICENSE), HF Lightricks/LTX-2.3, ltx.io/model/license, github.com/Wan-Video/Wan2.2, HF
Wan-AI/Wan2.2-{I2V-A14B,TI2V-5B}, QuantStack/Wan2.2-*-GGUF, comfy.org Wan2.2 docs, RTX-5090
LTX-vs-Wan benchmark (zenn.dev), github.com/Blaizzy/mlx-video, github.com/dgrauet/ltx-2-mlx. Code:
`crates/sceneworks-core/src/contracts.rs`, `config/manifests/builtin.models.jsonc`. Prior:
`documents/VIDEO_MODEL_RESEARCH.md`, `documents/EPIC_NATIVE_LTX23_VIDEO_ADAPTER.md`.
