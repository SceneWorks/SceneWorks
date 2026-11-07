# C — Component inventory for the shared component library

Researcher C of 8. Read-only survey of `inference` at **origin/main = a43724b82**. Every path below
is on origin/main. Conventions:

- `mlx/` = `crates/media/mlx-gen/`, `cdl/` = `crates/media/candle-gen/`, `llm-m/` = `crates/llm/mlx-llm/src/`,
  `llm-c/` = `crates/llm/candle-llm/src/`, `aud/` = `crates/audio/`, `gc/` = `crates/contracts/`.
  `mlx/core` = `mlx/src/` (the `mlx-gen` core crate), `cdl/core` = `cdl/candle-gen/src/`.
- **LOC** = `git show origin/main:<file> | wc -l` (whole file, comments included). "src LOC" for a
  crate excludes `tests/`, `*tests.rs`, `tools/`, `examples/`, `scripts/`.
- **Confidence** tags: **[V]** verified by reading the code or the module's own doc comment on
  origin/main; **[I]** inferred from names, structure, dependency edges, or upstream lineage
  knowledge without reading the math; **[E]** an estimate (always LOC-to-delete figures).
- Counts of "definitions" come from a regex sweep over `struct`/`fn` declarations (non-test,
  non-vendor); they over-count helpers and under-count inline code. Treat them as order-of-magnitude.

---

## 0. Size context

Total `.rs` under `crates/`: 4,034 files, **2.28 M lines; 1.68 M src lines** (non-test).

| Group | src LOC | notes |
|---|---:|---|
| `mlx-llm` / `candle-llm` | 95,125 / 62,989 | LLM engines (decode, KV, campaigns, provider) |
| `mlx/core` / `cdl/core` | 52,992 / 49,735 | shared media cores (nn, qkv, adapters, quant, train, tiling) |
| `gc/gen-core` (+family/h3/ltx/iris/flux3/…) | 68,951 + ~70k | tensor-free contracts, sampling, policy |
| 40 `mlx-gen-*` family crates | ~610k | biggest: sdxl 63k, z-image 46k, iris 39k, ltx 39k, h3 33k, wan 31k |
| 38 `candle-gen-*` family crates | ~520k | biggest: ltx 42k, h3 37k, krea 36k, wan 28k, sdxl 25k |
| `cdl/vendor` (candle-core/kernels/flash-attn/metal) | 43,736 | vendored framework, not components |
| audio (16 crates) | ~110k | candle-only |

Cross-cutting per-family files (sum over family crates, src only) — these are where most component
duplication hides:

| file stem | MLX LOC (crates) | Candle LOC (crates) |
|---|---:|---:|
| `memory_strategy` | 57,723 (28) | 56,607 (27) |
| `training` | 41,148 (13) | 35,100 (13) |
| `text_encoder*` (stem only) | 19,301 (20) | 11,607 (17) |
| `vae*` (stem only) | 16,839 (16) | 11,042 (16) |
| `convert` | 12,966 (25) | 3,477 (5) |
| `adapters` | 9,125 (18) | **17,393 (16)** |
| `loader` | 7,774 (14) | 8,037 (6) |
| `block_stream` | 6,493 (15) | 173 (1) |
| `preview` | 1,294 (7) | 9,977 (16) |
| `quant` | 1,628 (12) | **6,393 (17)** |

The asymmetries (candle `adapters`/`quant`/`preview` 2–8× MLX; MLX `block_stream`/`convert` many×
candle) are the fingerprints of which backend already centralized what (§6, §3.4).

---

## 1. Text / condition encoders

### 1.1 Decoder-LM text encoders (Qwen/Mistral/Gemma/GLM/gpt-oss)

Every row is one *implementation* (its own attention/MLP/decoder-layer code). "Consumers" are crates
that reuse it rather than re-implement.

| Arch (checkpoint) | MLX implementation (LOC) | Candle implementation (LOC) | Consumers | Covered by llm `CausalLm`? |
|---|---|---|---|---|
| **Qwen3 dense** shared block | `mlx/core/text_decoder.rs` (295) `Qwen3TextDecoderLayer` :197, `Qwen3TextAttention` :55 — "Krea, Boogu, Ideogram, Mage, MiniMax-H3 and FLUX.2 each carried a private copy" (:5) [V] | — (no candle equivalent) | flux2, boogu, ideogram, krea, mage, minimax-h3 | yes (Qwen3, Qwen3-VL text) |
| Qwen3 (FLUX.2-klein 4B/9B) + **Mistral** (FLUX.2-dev, `qk_norm=false`) | `mlx-gen-flux2/src/text_encoder/*` (926) on core block | `cdl/candle-gen-flux2/src/text_encoder.rs` (958) own `Attention` :32 `Mlp` :178 `DecoderLayer` :207 [V] | ideogram (cdl) "Adapted from Flux2PromptEncoder" | yes (Qwen3, Mistral) |
| Qwen3-VL-8B text (Boogu) | `mlx-gen-boogu/src/text_encoder/*` (492) on core block | `cdl/candle-gen-boogu/src/text_encoder.rs` (581) own `Attention` :55 `Mlp` :141 `DecoderLayer` :169 [V] | mage (both), krea (mlx loader), minimax-h3 (vision only) | yes (qwen3_vl) |
| Qwen3-VL-8B text, 13-layer interleave (Ideogram 4) | `mlx-gen-ideogram/src/text_encoder/*` (196) on core block | `cdl/candle-gen-ideogram/src/text_encoder.rs` (366) own layer [V] | — | yes |
| Qwen3-VL-4B, 12-layer stack (Krea 2) | `mlx-gen-krea/src/text_encoder/*` (1,185) on core block | `cdl/candle-gen-krea/src/text_encoder.rs` (890) own `Attention` :236 `Mlp` :315 `DecoderLayer` :337 [V] | — | yes |
| Qwen3-VL-4B (Mage) | `mlx-gen-mage/src/text_encoder/*` (2,442; own MRoPE `rope.rs` 299) on core block | `cdl/candle-gen-mage/src/text_encoder.rs` (295) **reuses** `candle_gen_boogu::BooguTextEncoder` [V] | — | yes |
| Qwen3-VL-32B, tap @50/64 (MiniMax-H3) | `mlx-gen-minimax-h3/src/text_encoder/*` (3,120 incl. tokenizer 727, degeneracy 322) on core block | `cdl/candle-gen-minimax-h3/src/text_encoder/*` (3,037) own `Qwen3Attention` :114 `Qwen3Mlp` :214 `Qwen3DecoderLayer` :241 [V] | — | yes, but needs early-exit (§9) |
| Qwen3 dense, penultimate un-normed (Z-Image) | `mlx-gen-z-image/src/text_encoder/*` (1,432; `stream.rs` 584 compiled/streamed variant) `EncoderLayer` layer.rs:13 [V] | `cdl/candle-gen-z-image/src/packed_te.rs` (374) — "faithful copy of the stock candle-transformers z_image::text_encoder" vendored for packed load [V] | anima (mlx), qwen-image-2-1 (mlx) reuse `EncoderLayer` [V] | yes |
| Qwen3-0.6B base (Anima) | `mlx-gen-anima/src/text_encoder.rs` (205) reuses z-image `EncoderLayer` [V] | `cdl/candle-gen-anima/src/text_encoder.rs` (183) own `DecoderLayer` :35 + `nn.rs` (120) [V] | — | yes |
| Qwen3-VL text (Qwen-Image 2.1) | `mlx-gen-qwen-image-2-1/src/text_encoder.rs` (490) z-image `EncoderLayer` + `mlx_llm::Qwen3VLVisionModel` + deepstack helpers [V] | `cdl/candle-gen-qwen-image-2-1/src/text_encoder.rs` (788) own `Attention` :149 `Mlp` :279 `DecoderLayer` :303 + `candle_llm::Qwen35VisionModel` [V] | — | yes (text), vision already llm |
| Qwen3-VL-4B (FLUX 3) | `mlx-gen-flux3/src/text_encoder.rs` (308) → `mlx_llm::CausalLm` [V] | `cdl/candle-gen-flux3/src/text_encoder.rs` (457) → `candle_llm::CausalLm` [V] | — | **already on llm** |
| Qwen3-VL (Iris) | `mlx-gen-iris/src/text_encoder.rs` (1,327) → `mlx_llm::CausalLm` [V] | `cdl/candle-gen-iris/src/text_encoder.rs` (398) → `candle_llm::CausalLm` [V] | — | **already on llm** |
| Qwen3.5 hybrid GDN (MOSS-Image) | `mlx-gen-moss-image/src/text_encoder.rs` (177) → `mlx_llm` Qwen35 [V] | `cdl/candle-gen-moss-image/src/text_encoder.rs` (151) → `candle_llm::models::qwen35::Qwen35Model` [V] | — | **already on llm** |
| **Qwen2.5-VL** text (bias q/k/v, no qk-norm), Qwen-Image | `mlx-gen-qwen-image/src/text_encoder/{attention,encoder,layer,mlp,mod}.rs` (342) + `vision_language.rs` (143) [V] | `cdl/candle-gen-qwen-image/src/text_encoder.rs` (218) + `vision_language.rs` (209) [V] | krea/anima consume qwen-image VAE, not TE | yes (Qwen2 arch) + needs MRoPE variant |
| Qwen2.5-VL-7B text, `hidden_states[-2]`, 3D MRoPE (Bernini planner) | `mlx-gen-bernini/src/qwen2_5_vl.rs` (493) `Qwen25VlText` :235 [V] | `cdl/candle-gen-bernini/src/qwen2_5_vl.rs` (797) `Qwen25VlText` :179 [V] | — | yes (Qwen2) — **duplicate of the Qwen-Image tower** [V: both docs say "architecturally stock HF Qwen2.5-VL"] |
| NEO-Unify dual-path (MoT) Qwen3 (SenseNova-U1) | `mlx-gen-sensenova/src/qwen3.rs` (1,051) `Qwen3Backbone` :337 [V] | `cdl/candle-gen-sensenova/src/qwen3.rs` (819) `Qwen3Backbone` :313 [V] | — | no (two weight sets per layer; also the denoiser) |
| Qwen3 blocks (audio) | — | `aud/candle-audio/src/qwen3.rs` (457) "Shared Qwen3-style transformer blocks" [V] | moss-tts-realtime, … | yes |
| **Gemma-2-2B-IT** (PiD, SANA) | `mlx-gen-pid/src/gemma2.rs` (890) `Gemma2` :412 [V] | `cdl/candle-gen-pid/src/gemma2.rs` (437) `Gemma2` :301 [V] | sana (both; `text_encoder.rs` 139 / 205 thin wrappers) [V] | **yes — `Architecture::Gemma2`** (`gc/core-llm/src/model_config.rs:499`) [V] |
| **Gemma-3-12B** (LTX-2.3) | `mlx-gen-ltx/src/gemma.rs` (936) `GemmaModel` :165 + `text_encoder.rs` (349) [V] | `cdl/candle-gen-ltx/src/gemma.rs` (532) `GemmaEncoder` :57 + `text_encoder.rs` (283) [V] | — | **no Gemma 3 arch** in core-llm enum (Gemma2/Gemma4/Gemma4Unified only) [V] |
| Gemma 4 (LTX-2.5 TE) | `mlx-gen-ltx/src/gemma4_te.rs` (1,068) → `mlx_llm::CausalLm` [V] | `cdl/candle-gen-ltx/src/gemma4_te.rs` (881) → `candle_llm::CausalLm` [V] | — | **already on llm** |
| **ChatGLM3-6B** (Kolors) | `mlx-gen-kolors/src/chatglm3.rs` (876) `ChatGlmModel` :318 + own `ChatGlmLinear`/`ChatGlmQuant` :88–95 [V] | `cdl/candle-gen-kolors/src/chatglm3.rs` (647) `ChatGlmModel` :57 [V] | — | llm has GLM-4 (partial rotary) [V]; ChatGLM3 half-dim *adjacent-pair* RoPE (`cdl/.../chatglm3.rs` doc) — compatibility with GLM-4 path **not verified** [I] |
| **gpt-oss-20b** MoE + MXFP4 (Lens) | `mlx-gen-lens/src/text_encoder/*` (1,980: `gpt_oss.rs` 1,433, `mxfp4.rs` 92) [V] | `cdl/candle-gen-lens/src/text_encoder.rs` (2,199) own `SparseMoe` :524, `Attention` :236 [V] | — | **no** gpt-oss arch (llm has MoE + YaRN primitives) [V] |

**Implementation counts for the Qwen-family decoder block** (Qwen2/2.5/3/3-VL/Mistral):

- MLX: **6** distinct (core `text_decoder`, z-image `EncoderLayer`, qwen-image `QwenEncoderLayer`,
  bernini, sensenova, mlx-llm `CausalLm`) serving 15 families. MLX already collapsed 6 copies into
  `text_decoder.rs` (sc-27326).
- Candle: **13** distinct (flux2, boogu, krea, ideogram, minimax-h3, qwen-image-2-1, z-image
  packed_te, anima, qwen-image, bernini, sensenova, `candle-audio::qwen3`, candle-llm). Candle has
  **no** shared Qwen3 text block in `cdl/core`.

Equivalence classes (same math, different code):

| Class | Members | Differences carried as data | Conf. |
|---|---|---|---|
| TE-A "HF dense GQA decoder" | all Qwen2/2.5/3/3-VL/Mistral rows above except sensenova | bias on q/k/v (Qwen2/2.5), per-head qk RMSNorm (Qwen3), θ, eps, head_dim decoupled from hidden (Krea 4B), RoPE type 1-D vs MRoPE (sections), compute dtype f32/bf16, tap policy (layers, final-norm on/off, prefix-drop, interleave vs stack vs concat) | **[V]** — mlx `text_decoder.rs:5-16` states the differing data explicitly; candle docs for flux2/ideogram/krea/boogu/h3 enumerate the same knobs |
| TE-B "Gemma family" | PiD/SANA Gemma-2, LTX Gemma-3, LTX-2.5 Gemma-4 | (1+w) RMSNorm, √d embed scale, soft-cap (G2), per-layer RoPE base + sliding window + qk-norm (G3), per-layer head dims (G4) | [V] (docs) — Gemma-3 is Gemma-2 shape + G3 knobs; llm already handles G2 and G4, so G3 is "add a config arm" [I] |
| TE-C "T5 encoder" | FLUX.1/Chroma/Mochi/SD3/Supra2 T5-XXL v1.1; Wan UMT5-XXL; (audio T5Gemma separate) | per-layer vs shared relative-position bias, vocab, masked vs unmasked encode | [V] (`cdl/candle-gen-wan/src/text_encoder.rs` doc lists the 3 UMT5 deviations; mochi/chroma docs state masked reuse of FLUX T5) |
| TE-D "CLIP text" | SDXL CLIP-L + OpenCLIP-bigG, FLUX.1 CLIP-L, SD3 CLIP-L/G, SAM3 CLIP-H text | width/depth, projection head, causal + padding mask, penultimate vs last | [V] (`mlx-gen-sam3/src/text.rs` doc: "A standard CLIP text transformer") |
| TE-E singletons | ChatGLM3 (GLM-1 style), gpt-oss MoE, SenseNova MoT | — | [V] |

**Collapse estimate (TE-A)**: MLX 6 → 1 (llm `CausalLm`) + per-family *tap policy*; Candle 13 → 1.
Deletable ≈ **8,000–9,500 LOC** [E]: candle ≈ 5,000–5,500 (flux2 ~650, boogu ~450, krea ~600,
ideogram ~250, h3 encoder ~900, qwen-image-2-1 ~450, z-image packed_te ~330, anima ~250, qwen-image
~330, bernini ~600, audio qwen3 ~350); MLX ≈ 3,000–3,500 (core text_decoder 295, z-image TE minus
streaming ~800, qwen-image TE ~300, bernini ~400, per-family encoder loops in flux2/boogu/ideogram/
krea/mage/h3 ~1,500). TE-B: ~2,300 [E] (Gemma-2 both backends 1,327 + Gemma-3 1,468 minus ~500 glue).
TE-C/TE-D see §1.3.

### 1.2 Vision / image encoders

| Arch | MLX (LOC) | Candle (LOC) | Consumers | Notes / conf. |
|---|---|---|---|---|
| CLIP ViT-L/14 (proj 768) | `mlx-gen-sdxl/src/vision_encoder.rs` (360) `ClipVisionEncoder` :209 (configurable) [V]; `mlx-gen-flux/src/image_encoder.rs` (127) wraps it [V] | `cdl/candle-gen-sdxl/src/vision_encoder.rs` (523) `ClipVisionEncoder` :215 [V]; `cdl/candle-gen-flux/src/ip_image_encoder.rs` (181) wraps it [V] | sdxl IP, flux XLabs IP, `*-clip` Dataset-Doctor embedder (mlx 332 / cdl 744), instantid | [V] |
| CLIP ViT-L/14-336 | same struct, `VisionConfig::vit_l_14_336` | same | kolors IP-Adapter-Plus (`mlx-gen-kolors/src/ip_adapter.rs` 103, `cdl/.../ip_provider.rs` 806) | [V] |
| OpenCLIP ViT-H/14 (proj 1024) | same sdxl struct; `mlx-gen-svd/src/image_encoder.rs` (69) reuses it [V] | **`cdl/candle-gen-svd/src/image_encoder.rs` (184) is its own port** ("Mirrors mlx-gen-svd … which reuses mlx-gen-sdxl's ViT-H body") [V] | sdxl IP, svd | duplicate on candle [V] |
| OpenCLIP XLM-R ViT-H/14 visual, 31 blocks, penultimate | `mlx-gen-scail2/src/clip.rs` (218) `ScailClip` :137 [V] | `cdl/candle-gen-scail2/src/clip.rs` (199) `ScailClip` :132 [V] | scail2 | same transformer math as ViT-H/14 with different key names + `use_31_block` [I] |
| EVA-CLIP (PuLID) | `mlx-gen-pulid/src/eva_clip/*` (669) [V] | `cdl/candle-gen-pulid/src/eva_clip/*` (792) [V] | pulid | 2-D vision RoPE + SwiGLU ViT [I] |
| SigLIP-so400m | `llm-m/models/siglip.rs` (339) [V] | `llm-c/models/siglip.rs` (399) [V] | joycaption (via llm) | already shared [V] |
| StarVector CLIP vision | `llm-m/models/starvector_vision.rs` (224) | `llm-c/starvector*.rs` | starvector | [V] |
| DINOv2 (Depth-Anything) | `mlx-gen-depth/src/backbone.rs` (223) `Dinov2Backbone` :123 [V] | `cdl/candle-gen-depth/src/backbone.rs` (199) :103 [V] | depth, perceptual, ltx/wan/krea/mage/z-image (depth control) | [V] |
| Pixtral ViT + Mistral3 projector (FLUX.2-dev edit) | `mlx-gen-flux2/src/vision/*` (480) [V] | `cdl/candle-gen-flux2/src/vision.rs` (907) [V] | flux2 dev edit | 2-D RoPE ViT [V] |
| **Qwen2.5-VL vision** (window attn @[7,15,23,31], merger) | `mlx-gen-qwen-image/src/text_encoder/vision/*` (710) [V] **and** `mlx-gen-bernini/src/vision.rs` (558) [V] | `cdl/candle-gen-qwen-image/src/vision.rs` (535) [V] **and** `cdl/candle-gen-bernini/src/vision.rs` (584) [V] | qwen-image-edit, bernini | **2 copies per backend of the same HF `Qwen2_5_VisionTransformerPretrainedModel`** [V by docs] |
| **Qwen3-VL vision** (+deepstack) | `mlx-gen-boogu/src/vision/mod.rs` (521) `VisionTower` [V] **and** `llm-m/models/qwen35_vision.rs` (1,202) `Qwen3VLVisionModel` ("`Qwen3VLVisionModel == Qwen35VisionModel`", `llm-m/models/mod.rs` doc) [V] | `cdl/candle-gen-boogu/src/vision/mod.rs` (1,163) [V] **and** `llm-c/models/qwen35_vision.rs` (973) [V] | boogu/krea/mage/minimax-h3 use boogu tower; qwen-image-2-1 uses llm tower [V] | 2 copies per backend [V] |
| SAM3 ViT (rope) / SAM2 Hiera | `mlx-gen-sam3/src/vision.rs` (547), `mlx-gen-sam2/src/hiera.rs` (500) | `cdl/candle-gen-sam3/src/vision.rs` (589) | sam | singletons [V] |
| Face stack: SCRFD, ArcFace iResNet, BiSeNet | `mlx-gen-face/src/*` (2,602) | `cdl/candle-gen-face/src/*` (3,026) | pulid, instantid, perceptual | singletons (CNNs) [V] |
| Body: ViTPose, Sapiens, HybrIK | `mlx-gen-body` (1,353) | `cdl/candle-gen-body` (1,396) | perceptual | [V] |

Equivalence classes:

| Class | Members | Collapse | Conf. |
|---|---|---|---|
| VE-A "CLIP-family pre-norm ViT" | CLIP-L/14, L/14-336, OpenCLIP-H/14 (sdxl struct), SVD candle copy, SCAIL OpenCLIP-XLMR-H, StarVector CLIP | 1 per backend (keys via remap; `n_layers_used`, pooled vs tokens, proj head) | [V] for sdxl/svd; [I] for scail2/starvector |
| VE-B "Qwen-VL vision" | Qwen2.5-VL ×2 per backend; Qwen3-VL ×2 per backend | 1 per backend in llm (window-attn flag, deepstack taps, patch 14 vs 16, merge 2) | [V] same HF class per pair; [I] for one tower serving both 2.5 and 3 |
| VE-C "RoPE-2D ViT" | Pixtral, EVA-CLIP, SAM3 ViT | possible generic ViT with 2-D rope; low priority | [I] |
| VE-D "DINOv2/SigLIP" | singletons per backend | keep | [V] |

Deletable [E]: VE-A ~400 (candle svd 184 + scail ~250 if absorbed); VE-B ~2,300 (MLX qwen-image vs
bernini ~550, boogu vs llm ~520; candle ~530 + ~1,000 — the larger candle boogu tower 1,163 vs llm
973); total vision ≈ **2,700–3,000 LOC**.

### 1.3 T5 / CLIP-text detail

| Component | MLX | Candle |
|---|---|---|
| T5-XXL v1.1 encoder | `mlx-gen-flux/src/text_encoder.rs` `T5TextEncoder` :315 + `T5BlockStream` :376 (~620 of 935) — reused by chroma (`text.rs` 113), mochi (263), sd3 (`text.rs` 310 aggregator), supra2 [V] | **3 paths**: stock `candle_transformers::t5::T5EncoderModel` (chroma `text.rs`, sd3 `conditioning.rs`, supra2 `model.rs`, flux `flux1_load.rs`) + vendored `PackedT5Encoder` in `cdl/candle-gen-flux/src/packed_te.rs:556` (mochi reuses it) [V] |
| UMT5-XXL | `mlx-gen-wan/src/text_encoder.rs` (705) `Umt5Encoder` :81 [V] | `cdl/candle-gen-wan/src/text_encoder.rs` (284) `Umt5Encoder` :137 + `text_encode.rs` (378) [V] |
| CLIP-L / bigG text | `mlx-gen-sdxl/src/text_encoder.rs` (1,435) `ClipTextEncoder` :901 **and** `mlx-gen-flux/src/text_encoder.rs` `ClipTextEncoder` :110 (~200) [V] | stock `candle_transformers::clip` (sd3, `candle-gen-clip`, flux1_load) **and** vendored `cdl/candle-gen-sdxl/src/clip.rs` (928) "faithful copy of candle-transformers … ClipTextTransformer" **and** `PackedClipText` in flux `packed_te.rs:215` [V] |
| CLIP-H text (SAM3) | `mlx-gen-sam3/src/text.rs` (261) | `cdl/candle-gen-sam3/src/text.rs` (270) |
| CLIP BPE tokenizer | `mlx-gen-sdxl/src/tokenizer.rs` (824) | `cdl/candle-gen-sd3/src/clip_tokenizer.rs` (393) + `gc/gen-core-tokenizer` (436) |
| T5Gemma (audio) | — | `aud/candle-audio-stable-audio-3/src/t5gemma.rs` (1,081) |

Collapse: T5 + UMT5 → 1 per backend (flag `per_layer_rel_bias`); CLIP text 3 → 1 per backend.
Deletable ≈ **2,000–2,500 LOC** [E], plus removal of the stock `candle-transformers` T5/CLIP
dependency paths (which differ numerically from the packed vendored paths; §11).

### 1.4 Do the llm crates already contain these architectures?

`core-llm` `Architecture` (`gc/core-llm/src/model_config.rs:29`, detection :482–515): Llama, Qwen3,
Phi3, Qwen2Moe, Gemma2, Glm4, DeepseekV2, Qwen35, Qwen3Vl, Gemma4Unified, Gemma4 (+ Qwen2 and
Mistral routed through Llama, `llm-m/config.rs` tests :507–573). The generic decoder
(`llm-m/models/llama.rs:1-12`) covers "Llama / Mistral / Qwen2 / Qwen3 / Phi-3 / Qwen2-MoE / Gemma-2
/ GLM-4 / DeepSeek-V2 / Gemma 4" with q/k-norm, q/k/v bias, Gemma sandwich norms, MoE, MLA,
per-layer-type attention. [V]

| Media encoder | llm today | gap |
|---|---|---|
| Qwen3 / Qwen3-VL text (11 families) | yes | tap policy + early exit (§9) |
| Qwen2.5-VL text (qwen-image, bernini) | Qwen2 yes; 3-D MRoPE via `mrope_positions_mm` exists on `CausalLm` (`llm-m/models/llama.rs:896`) [V] | Qwen2.5 MRoPE section layout vs Qwen3-VL interleaved — **not verified** [I] |
| Mistral (FLUX.2-dev) | yes | packed-tier load prefix `language_model.model.*` |
| Gemma-2 | yes | none known |
| Gemma-3 | **no** | add arch (G2 + qk-norm + per-layer rope base + sliding) |
| Gemma-4 | yes (already used by LTX-2.5) | — |
| Qwen3.5 | yes (already used by MOSS) | — |
| ChatGLM3 | GLM-4 only | rotary convention / MQA grouping check [I] |
| gpt-oss | **no** | add arch (sinks, YaRN, MoE, MXFP4 expert dequant) |
| T5 / UMT5 / CLIP text | **no** (decoder-only engine) | encoder-only bidirectional arch needed or keep in media |
| SigLIP | yes | — |
| Qwen3-VL vision | yes | — |
| Qwen2.5-VL vision | **no** | add (window attention) |

---

## 2. VAEs / latent codecs

### 2.1 Inventory

| Codec | MLX implementation (LOC) | Candle implementation (LOC) | Consumers |
|---|---|---|---|
| **SD/SDXL AutoencoderKL** (4-ch, sf 0.13025) | `mlx-gen-sdxl/src/vae.rs` (2,284; structs :1184, :1484) + `vae/pricing.rs` (846) [V] | `cdl/candle-gen-sdxl/src/vae_decoder.rs` (416) `:256` native decoder + `unet/vae_encode.rs` (155) + **stock** `candle_transformers::stable_diffusion::vae` (kolors `common.rs`, `pipeline.rs`, supra2) [V] | sdxl, kolors, instantid, supra2, ios-catalog |
| **FLUX.1 / Z-Image AutoencoderKL 16-ch** | `mlx-gen-z-image/src/vae/*` (1,647; 11 files) [V] — reused by flux, chroma, boogu, sd3 [V] | **4 paths**: vendored BFL `cdl/candle-gen-flux/src/vae/native.rs` (515), vendored diffusers `vae/diffusers.rs` (726) (both "faithful copies … vendored solely so the single unguarded SDPA can be bounded", `vae/mod.rs`) + stock `candle_transformers::z_image::vae` (z-image `common.rs`, boogu `pipeline.rs`, sd3 `vae.rs` 195) + **own port** `cdl/candle-gen-chroma/src/vae.rs` (278, "adapted from the flux2 decoder") [V] | flux, chroma, z-image, boogu, sd3, pulid, mochi(no) |
| **SD3.5 16-ch** | `mlx-gen-sd3/src/vae.rs` (542) = z-image VAE with factors + converter ("config.json byte-for-byte identical to Z-Image's … differing only in scaling/shift", :4–12) [V] | `cdl/candle-gen-sd3/src/vae.rs` (195) = stock z_image VAE re-export [V] | sd3 |
| **FLUX.2 AutoencoderKL-Flux2** (32-ch + 2×2 patchify + BN stats) | `mlx-gen-flux2/src/vae.rs` (1,395) — "Structurally identical to the SDXL VAE … block_out (128,256,512,512), latent 32, GN eps 1e-6" (:6–10) [V] | `cdl/candle-gen-flux2/src/vae.rs` (880) [V] | flux2, ideogram, lens, moss-image (all reuse) [V] |
| **FLUX 3 autoencoder** (8× KL + 2×2 patchify + BN) | `mlx-gen-flux3/src/autoencoder.rs` (1,004) :546 [V] | `cdl/candle-gen-flux3/src/autoencoder.rs` (1,029) + `autoencoder/bounded.rs` (756) [V] | flux3 |
| **Wan 2.1 z16** causal-Conv3d (stride 4×8×8) | `mlx-gen-wan/src/vae.rs` (1,378) + `vae_common.rs` (429) [V] | `cdl/candle-gen-wan/src/vae16.rs` (1,349) :320 [V] | wan A14B, bernini (`ProviderVae = candle_gen_wan::…`), scail2, krea-realtime, qwen-image (mlx alt decoder) [V] |
| **Qwen-Image AutoencoderKLQwenImage** (z16) | `mlx-gen-qwen-image/src/vae/{mod,blocks}.rs` (1,817) `QwenVae` :174 [V] | `cdl/candle-gen-qwen-image/src/vae.rs` (781) `QwenVae` :238, `QwenVaeEncoder` :570 [V] | anima, krea reuse it (both backends) [V] |
| **Wan 2.2 z48** (stride 4×16×16, residual) | `mlx-gen-wan/src/vae22.rs` (1,842) [V] | `cdl/candle-gen-wan/src/vae.rs` (1,667) :730 [V] | wan 5B |
| **Qwen-Image 2.1 RGBA** (z64, 16×, "Wan 2.2 residual VAE family") | `mlx-gen-qwen-image-2-1/src/vae.rs` (752) [V] | `cdl/candle-gen-qwen-image-2-1/src/vae.rs` (808) [V] | qwen-image-2-1 |
| LTX-2.3 video VAE (128-ch, 32× spatial) | `mlx-gen-ltx/src/vae.rs` (1,222) :551 | `cdl/candle-gen-ltx/src/vae.rs` (1,101) :387 | ltx |
| LTX-2.5 DiffVAE (diffusion decoder) | `mlx-gen-ltx/src/diff_vae.rs` (2,355) + `budget.rs` (679) | `cdl/candle-gen-ltx/src/diff_vae.rs` (2,529) + `budget.rs` (654) | ltx |
| LTX audio VAE + vocoder (HiFi-GAN/BigVGAN) | `audio_vae.rs` (430), `vocoder.rs` (867) | `audio_vae.rs` (295), `vocoder.rs` (741) | ltx |
| Mochi AsymmVAE decoder | `mlx-gen-mochi/src/vae.rs` (682) | `cdl/candle-gen-mochi/src/vae.rs` (627) | mochi |
| SVD temporal-decoder VAE | `mlx-gen-svd/src/vae.rs` (519) | `cdl/candle-gen-svd/src/vae.rs` (898) | svd |
| SeedVR2 3-D causal VAE | `mlx-gen-seedvr2/src/vae.rs` (618) | `cdl/candle-gen-seedvr2/src/vae.rs` (522) | seedvr2 |
| MiniMax-H3 video VAE (ViT decoder + causal-CNN encoder) | `vae.rs` (802) + `vae_encoder.rs` (755) | `vae.rs` (853) + `vae_encoder.rs` (1,083) | h3 |
| MiniMax-H3 audio VAE (DAC enc + BigVGAN dec) | `audio_vae.rs` (726) + `audio_vae_encoder.rs` (1,358) | `audio_vae.rs` (793) + `audio_vae_encoder.rs` (1,270) | h3 |
| DC-AE f32c32 (SANA) | `mlx-gen-sana/src/dc_ae.rs` (1,509) | `cdl/candle-gen-sana/src/dc_ae.rs` (1,817) | sana |
| Mage-VAE (one-step diffusion codec, 128-ch/16×) | `mlx-gen-mage/src/vae/*` (2,048) | `cdl/candle-gen-mage/src/vae.rs` (774) | mage |
| PiD pixel-diffusion decoder (swap-in for any VAE) | `mlx-gen-pid/src/decoder.rs` (233) via `mlx/core/decoder.rs` (217) `LatentDecoder` seam [V] | `cdl/candle-gen-pid/src/decoder.rs` (165) via `cdl/core/decoder.rs` (216) [V] | all image engines |
| TAESD / TAEHV (training perceptual) | `mlx/core/train/tae.rs` (1,095), `taehv.rs` (1,167) | `cdl/core/train/tae.rs` (1,090), `taehv.rs` (1,089) | trainers |
| Iris / SenseNova / PiD | pixel-space, **no VAE** (`mlx-gen-iris/src/dit.rs` doc "No VAE") [V] | — | — |
| Audio codecs (DAC-like, Snake) | — | `aud/candle-audio/src/neural_codec.rs` (740), `candle-audio-moss-tts-realtime/src/codec.rs` (2,374), yue codec (673) + vocoder (529), mmaudio vae (573), acestep vae (506), yue2 vae (2,037), moss-sfx vae (137) | audio |

Shared VAE plumbing already present: `mlx/core/vae_tiling.rs` (2,338), `cdl/core/vae_tiling.rs` (921),
`cdl/core/bounded_decode.rs` (270, `bounded_kl_decode`), `cdl/core/comfyui_vae.rs` (589, Wan key
remaps), `mlx-gen-wan/src/pipeline/vae{16,22}_tiling.rs` (544). Latent constants
`QWEN_WAN_Z16_MEAN/STD` are already in gen-core and imported by both qwen-image VAEs
(`mlx-gen-qwen-image/src/vae/mod.rs:16`, `cdl/candle-gen-qwen-image/src/vae.rs:17`) [V].

### 2.2 Equivalence classes

| Class | Members (per backend) | Parameters that differ | Collapse | Conf. |
|---|---|---|---|---|
| **VAE-A AutoencoderKL-2D** | MLX: sdxl, z-image(=flux1=sd3=chroma=boogu), flux2, flux3, svd-encoder → **4–5 impls**. Candle: sdxl native + stock SD, flux native + flux diffusers + stock z_image + chroma own, flux2, flux3 (+bounded), svd-encoder → **7–8 impls + 2 stock** | latent ch (4/16/32), block_out, GN eps (1e-5/1e-6), sf/shift, quant_conv presence, BFL vs diffusers key layout, optional 2×2 patchify + BN-stats normalization (flux2/flux3) | **1 per backend** + key-remap table + `PackedLatent` post-op | [V] flux2 doc says "structurally identical to SDXL"; sd3 doc "byte-for-byte identical config to Z-Image"; chroma "adapted from flux2 decoder" |
| **VAE-B Wan causal-3D lineage** | Wan2.1 z16, Qwen-Image z16, Wan2.2 z48, QwenImage2.1 RGBA z64 → **4 impls per backend** | z_dim, in/out ch (3/4), dim_mult, residual/avg-shortcut + patchify (2.2), temporal down/up flags, layout (NCTHW vs NTHWC in MLX) | **1 per backend** | Qwen-Image≡Wan2.1: [V] shared latent constants + MLX qwen-image already loads a Wan single-frame decoder as alternate (`mlx-gen-qwen-image/src/model.rs:20,292`). 2.1↔2.2 and 2.1-RGBA: [V by doc lineage] "the Wan 2.2 residual VAE family", but the wan vae22 doc lists 3 structural axes of difference → configurable, medium |
| VAE-C causal-3D others | LTX conv VAE, Mochi, SeedVR2, H3 encoder, SVD temporal decoder | distinct graphs | share **primitives only** (CausalConv3d, per-frame GroupNorm, depth-to-space, temporal cache, tiling) | [V] distinct; primitives [I] |
| VAE-D diffusion/one-step codecs | LTX DiffVAE, Mage-VAE, PiD | distinct | keep; share `LatentDecoder` seam (exists) | [V] |
| VAE-E audio vocoders | LTX vocoder (BigVGAN, 25 mentions), H3 audio decoder (BigVGAN, 14), `candle-audio/neural_codec.rs` (Snake/DAC) | — | shared Snake/AMP-ResBlock/ConvTranspose1d kit | [I] (keyword evidence only) |
| VAE-F DC-AE | sana | — | keep | [V] |

CausalConv3d-type structs: 14 across 12 crates (`mlx-gen-wan` ×2, ltx ×3, h3 ×2, mochi ×2,
seedvr2 ×2, qwen-image, svd), plus three candle `conv3d.rs` files (wan 425, seedvr2 432, svd 112).

Deletable [E]: VAE-A **~5,500–6,500** (MLX ~2,600; candle ~3,400 incl. vendored native/diffusers 1,241
and chroma 278); VAE-B **~5,000–6,000** (MLX ~3,300: qwen-image 1,817 + 70% of qwen-image-2-1 +
partial vae/vae22 merge; candle ~2,200); VAE-C primitives **~1,500**.

---

## 3. Transformer blocks

### 3.1 DiT taxonomy (one row per denoiser architecture)

| Shape | Families (MLX DiT LOC / Candle DiT LOC) | Conf. |
|---|---|---|
| **MMDiT double + single stream (FLUX.1 lineage)** | flux (`transformer.rs` 2,623 / stock + vendored `packed_dit.rs` 2,513 + `ip_dit.rs` 1,759), chroma ("FLUX MMDiT skeleton (19 dual + 38 single)", 1,260 / 761), flux2 (8 double + 24 single, 2,739 / 1,503), boogu (mixed, 1,707 / 1,221), iris (hybrid dual/single + PiT head, 3,547 / 1,212), flux3 (per-mode + single, 829 / 1,017), pulid (on flux backbone) | [V] docs |
| **MMDiT all-joint (dual stream)** | qwen-image (2,186 / 1,778), lens ("near-twin of mlx-gen-qwen-image's MMDiT", 1,347 / 1,177), sd3 (1,762 / 1,984), seedvr2 (1,398 / 1,143), mage NR-MMDiT (3,278 / 1,241), mochi asymmetric (1,462 / 1,138), pid PixDiT dual-stream + PiT (1,058 / 1,169) | [V] docs |
| **Single-stream over concatenated tokens** | z-image S3-DiT (3,729 / stock + `dit.rs` 1,192 + `packed_dit.rs` 1,986), ideogram (662 / 1,298), krea (3,380 / 3,047 + `train_dit.rs` 1,406), qwen-image-2-1 (1,583 / 2,684), minimax-h3 (7,690 / 8,666), moss-image (1,530 / 1,852) | [V] |
| **Cross-attention DiT + adaLN** | wan (2,989 / 1,812 + `dit_train.rs` 659) and its riders scail2, krea-realtime (`causal.rs` 2,576), bernini; ltx AV (4,020 / 3,133 + `dit_train.rs` 1,575); anima Cosmos (3,465 / 1,548); sana linear-attn (2,396 / 1,723); supra2 (263 / 292) | [V] |
| **MoT LLM-as-denoiser** | sensenova (qwen3.rs + t2i.rs) | [V] |
| **UNet** | sdxl (8,461 incl. 2,642 pricing / vendored 5,170), kolors (reuses sdxl UNet: 23 / 430), svd spatio-temporal (922 / 920), instantid (sdxl) | [V] |

Equivalence notes: chroma ≈ FLUX.1 DiT with pruned adaLN (separate impl on both backends, [V]);
lens ≈ qwen-image MMDiT (separate impl on both backends, [V] doc "near-twin"); scail2/krea-realtime/
bernini ride the Wan DiT ([V] deps); kolors/instantid ride the SDXL UNet ([V]).

### 3.2 Sub-block duplication counts (regex sweep, non-test, non-vendor, all of `crates/`)

| Sub-block | Definitions | Crates | Notes |
|---|---:|---:|---|
| `struct …Attention…` | 194 | 64 | prior audit's "147 in 58 crates" was media-only; this includes llm/audio |
| `struct …(Mlp/FeedForward/SwiGlu/GeGlu/Ffn)…` | 135 | 60 | `mlx/core/nn.rs:1392 SwiGluMlp` is the only shared MLX MLP |
| rope fns `fn …(rope|rotary)…` | 440 (304 media-only) | 74 | media count matches prior audit (305) |
| `struct …(Rope|Rotary|PosEmbed|EmbedNd)…` | 84 | 54 | |
| adaLN `struct …(Modulation|AdaLayerNorm|AdaLn)…` | 53 | 26 | |
| `fn …modulat…` | 109 | 44 | `mlx/core/nn.rs:926 modulate` shared but not universal |
| sinusoidal timestep fns | ~45 media (≈30 MLX+candle families, both backends) | ~35 | shared: `mlx/core/nn.rs:896 timestep_sincos`, `cdl/core/rope.rs:23 sinusoidal_timestep_embedding` — 30 files use them; ~45 still define their own (list in §3.3) |
| `fn …layer_norm…` / `struct …LayerNorm…` | 61 / 22 | 39 / 14 | mostly wrappers over `fast::layer_norm` / `candle_nn` |
| `fn …rms_norm…` / `struct …RmsNorm…` | 62 / 8 | 26 / 8 | 12 in candle-quant-kernels (fused) |
| GroupNorm defs | 40 | 21 | 10 in `mlx/core` (tiling/global GN) |
| `struct …(Resnet|ResBlock)…` | 62 | 37 | VAE + UNet + audio |
| `struct …Conv3d…` | 14 | 12 | §2.2 |
| Linear wrapper types (`Linear`, `QLinear`, `QEmbedding`, `Proj`, `TieredLinear`, `LinearBias/NoBias`, `ConvRotLinear`, `*Linear` enums) | ~50 | ~33 | §6 |

### 3.3 Timestep-embedding re-implementations (evidence for the count above)

Own sinusoid fn (file:line): `cdl/candle-gen-{anima/nn.rs:74, bernini/clip_diff.rs:118, boogu/transformer/mod.rs:326,
chroma/transformer.rs:80, flux2/transformer.rs:58, flux3/dit.rs:659, iris/dit.rs:690, lens/transformer.rs:185,
ltx/transformer.rs:81, ltx/dit_train.rs:64, ltx/diff_vae.rs:2004, minimax-h3/dit/heads.rs:182, mochi/nn.rs:73,
pid/backbone/layers.rs:63, qwen-image-2-1/transformer.rs:328, qwen-image/transformer.rs:138, sana/transformer.rs:263,
sd3/transformer.rs:82, sensenova/fm.rs:93, svd/embeddings.rs:13, wan/transformer.rs:619, z-image/packed_dit.rs:127}`;
`mlx/mlx-gen-{bernini/clip_diff.rs:136, chroma/transformer.rs:58, flux/transformer.rs:1442, flux2/transformer.rs:255,
flux3/dit.rs:533, iris/dit.rs:2376, lens/dit/transformer.rs:55, ltx/diff_vae.rs:1831, mage/vae/timestep.rs:56,
minimax-h3/dit/heads.rs:169, pid/backbone/layers.rs:107, qwen-image/transformer/timesteps.rs:10, sensenova/fm.rs:109,
svd/embeddings.rs:18, z-image/timestep_embedder.rs:72}`; audio: acestep, chatterbox, moss-sfx, moss-tts. [V by grep]
Knobs: cos-first vs sin-first ("flip"), `max_period`, scale (×1000), downscale_freq_shift, f32 host vs
device. One function with 4 knobs covers them [I].

### 3.4 Existing shared seams (what is already "the library")

| Seam | Where | Consumers | Conf. |
|---|---|---|---|
| Parameterized QK-norm + RoPE + layout + fused-QKV prologue, 6+ knobs (`QkNormPlacement`, `RopeStyle`, `RopeSpec` rot_dims/q/k) | `mlx/core/qkv.rs` (1,671; doc :1–30 "each family open-coded that prologue … ~20 times") | 14 MLX families (anima, boogu, chroma, flux2, ideogram, lens, ltx, mage, mochi, pid, sana, sd3, sensenova + core) [V] | [V] |
| Qwen3 text decoder block | `mlx/core/text_decoder.rs` (295) | 6 MLX families | [V] |
| `TextRope`, `apply_text_rope`, `SwiGluMlp`, `repeat_kv`, `build_mask`, `modulate`, `timestep_sincos`, `gated`, `conv3d`, `group_norm`, `window_partition` | `mlx/core/nn.rs` (3,215; :833–1529) | broad | [V] |
| Bounded/chunked SDPA | `mlx/core/attention.rs` (1,224), `cdl/core/attention.rs` (857) + `fused_attention.rs` (849) | broad | [V] |
| Interleaved RoPE apply, `AxialRope3`, sinusoid | `cdl/core/rope.rs` (346; "byte-identical bodies the Lens, Qwen-Image, FLUX.2 and Wan transformers each carried", :1–3) | 11 candle crates [V] | [V] |
| `Rotary`, `mrope_positions`, `mrope_cos_sin`, `causal_mask` | `cdl/core/grounding.rs` (378) | candle TEs (ideogram, krea, qwen-image-2-1, h3) | [V] |
| Block streaming | per-family `block_stream.rs` in **15 MLX crates (6,493 LOC)**; candle only `ltx` (173) + `cdl/core/block_window.rs` (442) | — | [V] — a per-family re-implementation of one cross-cutting hook |

**Candle has no `qkv.rs` equivalent**; its families open-code the prologue that MLX centralized.

### 3.5 Candle "vendor-to-get-a-seam" copies (the strongest single dedupe signal)

Root cause stated in the files themselves: stock `candle-transformers` modules build frozen
`candle_nn::Linear` with no seam (packed load, LoRA residual, IP injection), and candle's fused
`softmax_last_dim`/`rms_norm` CustomOps have **no backward**, so trainers copy the DiT with
composable ops. [V]

| Copy | LOC | Copy of | Reason (doc) |
|---|---:|---|---|
| `cdl/candle-gen-flux/src/ip_dit.rs` | 1,759 | stock `flux::model` | no per-double-block injection point (IP-Adapter) |
| `cdl/candle-gen-flux/src/packed_dit.rs` | 2,513 | diffusers FLUX DiT | packed-load seam |
| `cdl/candle-gen-flux/src/packed_te.rs` | 775 | stock CLIP + T5 | packed-load seam |
| `cdl/candle-gen-flux/src/vae/{native,diffusers}.rs` | 1,241 | stock flux AE + z_image VAE | bound one SDPA |
| `cdl/candle-gen-z-image/src/dit.rs` | 1,192 | stock z_image transformer | trainable residual seam |
| `cdl/candle-gen-z-image/src/packed_dit.rs` | 1,986 | stock z_image transformer | packed-load seam |
| `cdl/candle-gen-z-image/src/packed_te.rs` | 374 | stock z_image text encoder | packed-load seam |
| `cdl/candle-gen-krea/src/train_dit.rs` | 1,406 | own inference `Krea2Transformer` | fused ops lack backward |
| `cdl/candle-gen-lens/src/dit_train.rs` | 809 | own `LensTransformer` | LoRA trainable projections |
| `cdl/candle-gen-ltx/src/dit_train.rs` | 1,575 | own `AvDiT` (video leaves) | trainable projections |
| `cdl/candle-gen-wan/src/dit_train.rs` | 659 | own `WanTransformer` | trainable projections |
| `cdl/candle-gen-sdxl/src/unet/*` + `clip.rs` | ~5,000 + 928 | stock UNet / CLIP | packed load / controlnet / LoRA |
| **Total** | **~20,000** | | |

MLX has none of these: `AdaptableLinear` (`mlx/core/adapters.rs:604`) is the seam, and the MLX
trainers train the inference DiT directly ([V]: no `train_dit`/`dit_train` file exists under
`mlx-gen`). Deletable if candle gets one hookable Linear + composable-op mode + block hooks:
**~8,000–11,000 LOC** [E] (one of each pair kept; SDXL partially).

---

## 4. Schedulers / sigma builders outside gen-core

gen-core already owns policy: `gc/gen-core/src/sampling*.rs` (4,953 LOC: `compute_mu` :233,
`time_shift_exponential` :252, `build_flow_sigmas` :259, `sdxl_edm_sigmas` :1012,
`flow_match_static_shift_sigmas` :1176, `schedulers.rs` karras/exponential/normal/simple/ddim/beta
:128–246, `solvers.rs`, `unified.rs`). Backends carry thin tensor application:
`mlx/core/sampler.rs` (2,095) and `cdl/core/sampler.rs` (1,811, "candle twin of mlx-gen's
MlxLatentOps") [V]. `gc/gen-core-family` hosts krea/qwen-image-2-1 schedules [V].

Still per-family (file:line) [V by grep]:

| Family | MLX | Candle |
|---|---|---|
| anima | `pipeline.rs:38 anima_sigmas`, `sampling.rs` (125) | `pipeline.rs:23 anima_sigmas` |
| boogu | `pipeline.rs:1198/1211/1267/1297` (dmd/turbo/base/time_shift_v1) | `pipeline.rs:1359/1386/1396/1410` (same four) |
| chroma | `beta.rs` (155) `base_sigmas` :101 | `beta.rs` (158) :104, `pipeline.rs:528 sigmas` |
| flux | `pipeline.rs:35 build_sigmas_with`, :113 `build_linear_sigmas` | `preview.rs:555`, stock `flux::sampling::get_schedule` |
| flux2 | (gen-core) | `pipeline.rs:114 compute_mu` — **deliberate f64 fork** ("differs_from_the_f32_shared_one", :241) |
| qwen-image | `pipeline.rs:427 qwen_sigmas`, `sampler.rs:52 lightning_sigmas` | `pipeline.rs:122/163` |
| sd3 | `pipeline.rs:46 resolve_sd3_sigmas` | `pipeline.rs:123/148` |
| sdxl / kolors | `sampler.rs:321 edm_sigmas` (516), `sampling.rs` (1,085); kolors `sampler.rs:106 edm_sigmas` (283) | `sdxl/sampler.rs` (494), kolors `sampler.rs` (217) |
| wan | `scheduler.rs` (731) 5 `sigmas` fns :49–468 | `scheduler.rs` (331) :56/77/271 |
| mochi / svd / mage / lens / sana / krea-realtime / sensenova / pid / supra2 / iris | `scheduler.rs` 214 / 140 / — / `schedule.rs` 50 / `sampling.rs` 555 + `scm.rs` 173 / 536 / `fm.rs` 195 / `sampler.rs` 245 / 93 / `solver.rs` 135 | 218 / 145 / `scheduler.rs` 120 / 163 / `pipeline.rs:145` / — / 220 / 209 / 93 / 68 |

Per-family scheduler files total ≈ **5,600 MLX + 2,700 candle** (excluding ios-catalog 1,176).
Equivalence: most are (linspace | beta | EDM/Karras | saved-list) × (static shift | exponential μ-shift
| none) × terminal-zero policy — all already expressible in `gen_core::sampling::schedulers` [I].
Collapse to gen-core named schedules + per-family *data* (shift, μ table, saved sigmas):
**~3,000–4,000 LOC** [E]. Risk: flux2 candle f64-vs-f32 μ fork shows that numerics, not math, drove
some forks.

---

## 5. Image pre/post-processing, video frame packing

| Function | MLX | Candle | Conf. |
|---|---|---|---|
| decoded `[-1,1]` → RGB8 / RGBA / frames | **shared** `mlx/core/image.rs` (389): `decoded_to_image` :22, `decoded_to_rgba_image` :47, `u8_frames_to_images` :132, `video_bcfhw_to_u8_frames` :175 [V] | only `cdl/core/lib.rs:65 round_rgb8`; **~20 family copies**: `to_image` in anima :324, flux :1582, flux2 lib :1294, ideogram :1111, iris :218, kolors common :119, krea :2956, lens :925, qwen-image control_common :124, sdxl :1459; `frames_to_images` in ltx :779, h3 :332, mochi :247, svd :343, wan :153; `decoded_to_image` z-image common :366, qwen-image-2-1 :640, seedvr2 :82; sensenova `tensor_to_image` :1854; sana `decode_to_image` :340 [V] | [V] |
| RGBA over white | gen-core `media.rs:137 to_rgb_over_white` + mlx image.rs | `cdl/candle-gen-qwen-image-2-1/src/pipeline.rs:546` own | [V] |
| image → model pixels | `mlx/core/img2img.rs` (172) + boogu `pipeline.rs:1159` | boogu `:1268`, krea `:2745` | [V] |
| Qwen-VL smart_resize + bicubic + normalize + patchify | qwen-image `image_processor.rs` (148), bernini `vit_preprocess.rs` (267), boogu `vision/preprocess.rs` (293) [V] | qwen-image (229), bernini (272), boogu (278) [V] | — |
| …the same, already shared for LLM engines | `gc/core-llm/src/vision_preprocess.rs` (545; "MLX 7157 preprocessor and its Candle 7262 port were line-for-line copies", :1–3) + `llm-m/image.rs` (289) / `llm-c/image.rs` (327) [V] | | [V] |
| SVD / depth / scail2 / bernini-VAE preprocess | svd 274, depth 163, scail2 91 + `resize.rs` 150, bernini `vae_preprocess.rs` 115 | svd 265, depth 182, scail2 164 + 177 | [V] |
| SeedVR2 color fix (LAB/wavelet) | `color.rs` (242) | `color.rs` (394) | singleton |
| HDR/EXR | `gc/gen-core-image` (699) tensor-free, shared [V] | | |

Collapse: Qwen-VL preprocessors 6 → reuse `core-llm::vision_preprocess` (it is host-side and
tensor-free) — **~1,300 LOC** [E, high confidence]; candle `to_image` family copies → one
`cdl/core/image.rs` mirroring MLX — **~500–700 LOC** [E]. Video frame packing (`frames_to_images`
5 candle copies) included.

---

## 6. Quantized linear + adapter-capable linear

### 6.1 Linear/projection types

| Layer | MLX | Candle |
|---|---|---|
| **Core adapter-capable linear** | `mlx/core/adapters.rs` `LinearBase` :547, `AdaptableLinear` :604 (bases: dense, `QuantizedLinear`, `ConvRotLinear`, `W4A8Linear`; residual LoRA / LoKr / LoHa; `merge_dense_delta` :777) [V] | `cdl/core/quant/mod.rs` `DenseLinear` :392, `QuantWeight` :473, `QLinear` :520 (Dense / W4A8 / Quantized); `cdl/core/quant/adapt.rs` `AdaptLinear` :713 (forward-time LoRA/LoKr residual "the candle twin of mlx-gen's AdaptableLinear"); `eight_bit_linear.rs` `Fp8Linear` :30 `Int8Linear` :91; `w4a8_int8.rs` :24; `convrot_int8.rs` :7; `train/lora.rs LoraLinear` :236 [V] |
| **Per-family wrappers** | flux3 `nn.rs Linear` :94, iris `nn.rs Linear` :437 (+`AdaptedLinears`, `BackboneQuant`), kolors `ChatGlmLinear`, ltx `GemmaLinear` + `transformer/linear.rs LinearKind/Linear` (360), lens dit `Linear` + gpt-oss `LinearBias`/`StackedQuant`, mage ×2, minimax-h3 ×4 (+pruned adaLN linears), mochi `MochiLinear`/`MochiQuant`, sdxl `HeaderLinear`, seedvr2 ×2, ltx diff_vae `Linear` [V] | boogu `QLinear`/`QEmbedding`, chroma `QLinear`, flux/flux2/ideogram/z-image `QEmbedding`, krea `QLinear`/`QEmbedding`/`ComposableLinear`, ltx `QLinear`/`QLinearBase`/`ConvRotLinear`/`QEmbedding`, minimax-h3 `TieredLinear`/`Linear`/`LinearBias`/`LinearNoBias`/`Proj`, flux3 `Linear`, iris `Linear`, depth/sam3/moss-image/seedvr2×2 `Linear`, sana nvfp4 [V] |
| **Per-family `quant.rs`** | 12 crates, 1,628 LOC (mostly ≤60 LOC config) | **17 crates, 6,393 LOC**; e.g. `candle-gen-boogu/src/quant.rs` (300) and `candle-gen-chroma/src/quant.rs` (278) are "thin dense-or-packed enum + detect wrapper" over `candle_gen::quant` (doc) [V] |
| **llm** | `llm-m/primitives/projection.rs` `QuantSpec` :16 (affine group 64, `QuantGrid` Computed/Stored :29), `Projection` :62 (Dense / Quantized / Prism) — **no adapters** | `llm-c/primitives/projection.rs` `QuantSpec` :27, `ProjectionFormat` :145 (Ggml Q4_K/Q8_0 / Nvfp4), `Projection` :233 (Dense / Quantized / Prism / Nvfp4) — **no adapters** [V] |

### 6.2 Quant formats supported (by stack)

| Format | mlx media | candle media | mlx-llm | candle-llm |
|---|---|---|---|---|
| MLX affine group-wise Q4/Q8 (gs 64; boogu 32) | native (`mlx/core/quant.rs` 1,029) | **repacked**: Q4 → GGML `Q4_1` lossless; Q8 → `Q8_0` re-quant (0.56% rel RMS) (`cdl/core/quant/mod.rs` doc; `repack.rs` 792) | native (`QuantSpec`) | — (quantizes to GGML on load) |
| GGML Q4_K / Q8_0 | — | Wan GGUF loader (`cdl/candle-gen-wan/src/gguf.rs` 1,767, Q4_K_M) | GGUF read + dequant + IQ grids (`llm-m/gguf/*` ~4,400) | quantize-on-load (`ProjectionFormat::Ggml`) |
| NVFP4 (E2M1 + UE4M3) | — | `kernels/candle-quant-kernels/nvfp4_linear.rs` :523, `cdl/core/quant/nvfp4_kitchen.rs` (310), krea `nvfp4_dit.rs` (1,738), sana `nvfp4_dit.rs` (524), flux2 `nvfp4_roles.rs` (552) | — | `Projection::Nvfp4` + `nvfp4_path.rs` |
| MXFP4 (gpt-oss experts) | lens `text_encoder/mxfp4.rs` (92) dequant | lens (inline) | — | — |
| FP8 / INT8 | `int8_gemm.rs` (540), `convrot_int8.rs` (303), `w4a8_int8.rs` (526) | `eight_bit_linear.rs` (265), `convrot.rs`+`convrot_int8.rs` (411), `w4a8_int8.rs` (377), Veda int8 kernels | — | — |
| ComfyUI scaled-fp8 → dense | — | `cdl/candle-gen-wan/src/comfyui.rs` (540), z-image `comfyui.rs` (834) | — | — |
| Prism ternary / Bonsai | — | — | `primitives/prism.rs` (884) | `primitives/prism.rs` (2,391) |
| "sidecar" pre-quant artifacts | — | `cdl/core/quant/sidecar.rs` (2,554) | — | — |

### 6.3 Adapters

MLX: runtime residual on `AdaptableLinear`; loaders centralized (`mlx/core/adapters/loader/*`:
bfl, kohya, peft, lycoris, lokr, loha, diff, configured, `source_facts.rs` 3,397) — per-family
`adapters.rs` mostly 18–68 LOC (key→module map only), exceptions sdxl 3,431, minimax-h3 1,450, iris
1,200+895, ltx 899, moss 600 [V]. Candle: **both** merge-at-load (`cdl/core/train/merge.rs` 1,085,
`merge_dit.rs` 204) and runtime residual (`quant/adapt.rs`, `quant/adapters.rs` 1,229 "Shared
forward-time LoRA/LoKr installer"); per-family `adapters.rs` are 505–2,705 LOC each (krea 2,705, sdxl
2,251, sd3 1,458, minimax-h3 1,410, wan 1,214, qwen-image 1,209, z-image 1,014, …) re-homing the
"classify/merge core" onto each key namespace (krea/sd3/qwen-image docs) [V]. TE adapter keys are
**skipped** on both backends (`mlx/core/adapters/loader/bfl.rs:172–188`, sdxl adapters :1251) [V].

Collapse [E]: per-family candle wrappers (`quant.rs` + Linear types) → core `AdaptLinear`/`QLinear`:
**~4,000–5,000 LOC**; candle per-family `adapters.rs` → declarative key-map tables + shared
installer (MLX already proves the shape): **~7,000–9,000 LOC** (medium-low confidence: merge vs
residual semantics and golden byte-parity are deliberate, §11).

---

## 7. ControlNet / IP-Adapter / PuLID / InstantID / reference conditioning

| Module | MLX (LOC) | Candle (LOC) | Mechanism |
|---|---|---|---|
| SDXL ControlNet | `mlx-gen-sdxl/src/unet/controlnet.rs` (635) | `cdl/candle-gen-sdxl/src/unet/controlnet.rs` (640) + `control_provider.rs` (772) | UNet-encoder copy → zero-conv residuals |
| Kolors ControlNet | (via sdxl UNet in `model.rs`) | `cdl/candle-gen-kolors/src/control.rs` (948) | same as SDXL |
| FLUX.1 Fun-ControlNet-Union | `control_transformer.rs` (299) + `model_control.rs` (748) | `control.rs` (1,037) + `control_provider.rs` (633) | partial DiT copy (6 double blocks) → per-block residual |
| FLUX.2 control | `model_control.rs` (960) | `control_provider.rs` (1,076) | |
| Qwen-Image Fun control | `control_transformer.rs` (262) + `model_control.rs` (655) | `control_fun.rs` (793) + `control_common.rs` (207) | per-block residual branch |
| Z-Image Fun control (VACE-style dual injection) | `control_transformer.rs` (1,391) + `_block.rs` (195) + `model_control.rs` (706) + `model_base_control.rs` (516) | `control.rs` (2,136) | parallel control stack threading a hint state |
| Krea 2 control | `control.rs` (697) + `model_control.rs` (1,366) | `control.rs` (1,801) + `control_provider.rs` (1,124) + train (1,264) | |
| Wan VACE / VACE-Fun | `vace.rs` (1,526) + `model_vace.rs` (1,813) | `vace.rs` (1,106) + `model_vace.rs` (1,048) + `model_vace_fun.rs` (1,180) + `vace_fun_tier.rs` (561) | VACE context blocks → hint injection |
| SDXL IP-Adapter (+Plus Resampler) | `ip_adapter.rs` (742) + pricing (477) | `ip_adapter.rs` (724) + `ip_provider.rs` (466) | decoupled cross-attn |
| Kolors IP-Adapter-Plus | `ip_adapter.rs` (103, config over sdxl) | `ip_provider.rs` (806) | same |
| FLUX XLabs IP-Adapter | `ip_adapter.rs` (211) | `ip_adapter.rs` (441) + `ip_dit.rs` (1,759 vendored DiT) + `ip_provider.rs` (542) | decoupled cross-attn in double blocks |
| PuLID-FLUX | `pulid_flux.rs` (966) + `idformer.rs` (276) + `ca.rs` (215) + EVA (669) | 769 + 277 + 271 + 792 | ID tokens → per-block CA injection |
| InstantID | `model.rs` (1,769) + kps/openpose/restore (440) | `model.rs` (2,012) + 440 | IP-Adapter(face) + ControlNet(kps) on SDXL |
| Reference / edit (in-context tokens) | flux2 (in model), qwen-image `model_edit.rs` (840), qwen-image-2-1 `reference.rs` (443), flux3 `references.rs` (50), moss `edit.rs` (823), boogu edit, mage, h3 ref2va/fl2va | flux2 `edit_provider.rs` (1,483), qwen-image `edit.rs` (2,211), qwen-image-2-1 `reference.rs` (440), moss `edit.rs` (451) + `transformer/edit.rs` (408), mage `edit_provider.rs` (570), sdxl/z-image edit providers | token concatenation + position offsets |
| Bernini CLIP-diff guidance | `clip_diff.rs` (449) | 408 | |

Equivalence classes:

| Class | Members | Generic form | Conf. |
|---|---|---|---|
| CN-A "residual control branch" | SDXL/Kolors ControlNet, FLUX.1 Fun-CN-Union, Qwen Fun, FLUX.2 control, Krea control | `ControlBranch<B: Block>` = N copies of the family's own block + zero-init proj + `post_block(i) += scale·r_i` hook | [I] (docs describe "per-block residual injected into the frozen base") |
| CN-B "hint-state parallel stack" | Z-Image Fun 2.1, Wan VACE / VACE-Fun | parallel stack threading `c` + `hint_i` injection at mapped layers | [I] (z-image candle doc: "VACE-style dual-injection") |
| IP-A "decoupled cross-attention" | SDXL/Kolors IP, FLUX XLabs IP, InstantID face IP, PuLID CA | `extra_kv(i)` attention hook + image-proj/resampler | [I] |
| REF-A "in-context reference tokens" | flux2, qwen-image edit, qwen-image-2-1, flux3, moss, boogu, mage, h3 | token-sequence assembly + rope offsets | [I] |

Totals: control/IP/ref modules ≈ **15,000 MLX + 21,000 candle** LOC (above rows). Collapse via
three DiT hooks (post-block residual, extra-KV attention, sequence-assembly) [E]:
**~8,000–10,000 LOC**, low-medium confidence (much of the mass is provider/validation glue).

---

## 8. Other notable repeated components (brief)

- **DINOv2 / depth head**, **face** (SCRFD/ArcFace/BiSeNet), **body**, **perceptual** (`*-perceptual` lib.rs 1,302 / 1,246), **SAM2/SAM3** are singletons per backend: no intra-backend dup; only the 2× backend dup.
- **Tokenizers**: gen-core-tokenizer (436) exists; still per-family wrappers (`boogu/tokenizer.rs` 322/396, `krea/tokenizer.rs` 265/225, `minimax-h3` 727/908, `sdxl/tokenizer.rs` 824, sd3 `clip_tokenizer.rs` 393, `kolors` 151/155, `anima` 141/92, `ltx` 175/193, `mochi` 74/36, `qwen-image/vl_tokenizer.rs` 324/174) ≈ 5,500 LOC — template/drop policy, mostly legitimately family-specific [I].
- **memory_strategy.rs** ~114k LOC across 55 crates is the single largest per-family mass (out of scope for C; flagged for the admission/hook researchers).

---

## 9. The llm crates as the encoder library

### 9.1 What they implement

| | mlx-llm | candle-llm |
|---|---|---|
| Decoders | `models/llama.rs` (2,702) generic `CausalLm`; `qwen35.rs` (2,680) GDN hybrid; `gemma4_mm.rs` (1,030); `gpt_bigcode.rs`, `starcoder2.rs`; `qwen35_vision.rs` (1,202); `siglip.rs` (339); `starvector_vision.rs` (224); `deepstack.rs` (250) | `models/llama.rs` (4,071); `qwen35.rs` (7,078); `gemma4_mm.rs` (1,163); `starcoder2.rs`, `starvector*.rs`; `qwen35_vision.rs` (973); `siglip.rs` (399); `llava.rs` (1,128); `deepstack.rs` (368) |
| Primitives | attention (1,983), rope (937: standard/llama3/yarn/proportional/partial — `config.rs:37-423`), kv_cache (1,481), paged KV (882), packed/compressed KV (~11k), moe (884), gated_delta (2,250), sampler (1,342), quant (463), projection (241), prism (884) | attention (1,954), rope (661), kv/step/paged caches (~3k), moe (1,588), gated_delta (3,044), nvfp4 path, prism (2,391), device_positions (439) |
| Decode engines | continuous/batched/speculative/prefix (~9k) | engine/graph/step/batched/prefix (~12k) |
| Load/admission | `load_memory.rs` (2,281), `residency.rs` (246) | `provider/admission.rs` (933), `device.rs` (520) |
| Encoder API | `hidden_states` :1081, `hidden_states_with_mask` :1102, `hidden_states_with_mask_at` :1123 (explicit positions — covers FLUX 3 pad trick), `hidden_states_from_embeds{,_with_mask}` :1177/:1189, `splice_vision_features` :853, `mrope_positions_mm` :896, `decode_logits_from_embeds_mrope_deepstack` :923 [V] | same set :1289–1509 [V] |
| `VlmDecode` trait (vision splice + MRoPE + deepstack) | `models/mod.rs` [V] | equivalent |

### 9.2 Media consumers today

`mlx-gen` core depends on mlx-llm (caption, `text_sample.rs`, `coherence.rs`). Family consumers [V]:
flux3, iris (`CausalLm` as TE), ltx (`gemma4_te.rs` `CausalLm`; `enhance.rs` prompt enhancer),
moss-image (`Qwen35Model`), qwen-image-2-1 (vision tower + deepstack helpers only), flux2
(KV cache + speculative decode for caption upsampling only), lens (`reasoner.rs` decode only),
sensenova (sampler/decode only), joycaption (whole VLM). Same set on candle (+ candle flux2
`llava::splice_image_features`). **9 media families**, but only **5** use it as the text-encoder
forward.

### 9.3 Blockers to "llm = the encoder library"

| # | Blocker | Evidence | Severity |
|---|---|---|---|
| B1 | No **tap set / early exit**: `hidden_states*` returns all `num_layers+1` states and runs every layer; H3 needs layer 50 of 64, Lens 23 of 24, flux2 9/18/27, ideogram 13 taps, krea 12-stack | `llm-m/models/llama.rs:1081-1200` returns `Vec` of all layers [V] | medium (API addition; memory on MLX is lazy so unused handles are free, but compute isn't) |
| B2 | Encoder forward requires `&mut dyn KvCache` | signatures :1081/:1102 [V] | low (pass a throwaway; but `ContiguousKvCache` allocates per layer) |
| B3 | **Quant spine mismatch**: llm Projection has Dense/Quantized/Prism(/Nvfp4); media needs `AdaptableLinear` bases incl. ConvRot-int8, W4A8, and candle-media's MLX-packed→Q4_1 lossless repack; candle-llm quantizes to GGML **Q4_K**/Q8_0 at load — different numerics from candle-media's Q4_1 | `llm-c/primitives/projection.rs:145-250`, `cdl/core/quant/mod.rs` doc [V] | **high** for tier parity (q4/q8 goldens) |
| B4 | **No adapter residuals** on llm `Projection` | 0 lora/adapter mentions in either `projection.rs`; llama.rs "lora" hits are DeepSeek MLA ranks [V] | low today (TE adapter keys are skipped) but blocks future TE LoRA |
| B5 | **Residency/streaming/memory admission** are a parallel system (`load_memory.rs` 2,281; `provider/admission.rs`) vs media `block_residency`, `request_scope`, per-family `memory_strategy`, `block_stream` | [V] files exist | high (two admission models for one process) |
| B6 | Missing archs: Gemma 3, gpt-oss, ChatGLM3 (verify vs GLM-4), Qwen2.5-VL vision, T5/UMT5/CLIP (encoder-only bidirectional) | §1.4 [V] | medium |
| B7 | Checkpoint layout: SceneWorks packed component dirs (`text_encoder/` with `quantization` block, boogu gs 32, mistral `language_model.model.*`, z-image packed TE) | `from_weights(prefix)` exists; packed gs-32 load path **not verified** [I] | medium |
| B8 | Precision conventions: several media encoders run **f32** (qwen-image, pid gemma2, svd) or bf16 with f32 RMS reduce (UMT5); llm "Cached decode runs in bf16" with `from_weights_dtype` | docs [V] | medium (parity) |
| B9 | MLX command-buffer chunking: guidance "evaluate returned states in order" (`llama.rs:1160-1172`) — media encoders have their own streaming (`z-image/text_encoder/stream.rs` 584, `flux T5BlockStream`) | [V] | low |
| B10 | Batching: llm decode engines batch decode requests; media encoders batch CFG pos/neg prompts — `hidden_states_with_mask` is `[batch, seq]` capable [V] | low |

Verdict: the llm `CausalLm` is already a correct and battle-tested **decoder-LM encoder core** for
TE-A and TE-B (5 families prove it), and should become *the* implementation. It needs B1, B3, B5
resolved — B3/B5 argue for moving `Projection` to the media `AdaptableLinear`/`QLinear` spine (or a
shared trait) rather than the reverse.

---

## 10. Top 15 dedupe opportunities

Score = est. deletable LOC (midpoint, both backends) × families touched. All LOC are **[E]**.

| Rank | Opportunity | Families touched | Deletable LOC | Score | Conf. |
|---|---|---:|---:|---:|---|
| 1 | **Candle per-family adapters → declarative key-maps + shared installer** (the MLX shape, §6.3) | 16 | 8,000 | 128k | medium-low |
| 2 | **Qwen-family decoder TEs → llm `CausalLm` + tap policy** (13 candle + 5 MLX impls → 1/backend, §1.1) | 14 (flux2, boogu, ideogram, krea, mage, h3, z-image, anima, qwen-image, qwen-image-2-1, bernini, iris, flux3, + audio) | 8,700 | 122k | medium-high |
| 3 | **Timestep sinusoid + adaLN modulate + gated residual + MLP glue kit** (§3.2–3.3) | 35 | 3,500 | 122k | medium (simple math, but byte-parity flips and op orders vary) |
| 4 | **Per-family Linear/QLinear wrappers + candle `quant.rs`** → core types (§6.1) | 20 | 4,500 | 90k | medium |
| 5 | **RoPE engines** → MLX `qkv.rs` everywhere + a candle port of `qkv.rs`; axial/MRoPE/2-D tables as data (§3.4) | 30+ | 3,000 | 90k | medium |
| 6 | **Control branches / VACE / IP / PuLID as three DiT hooks** (§7) | 10 (sdxl, kolors, flux, flux2, qwen-image, z-image, krea, wan, pulid, instantid) | 9,000 | 90k | low-medium |
| 7 | **AutoencoderKL-2D family → 1 configurable VAE/backend** (§2.2 VAE-A) | 14+ (sdxl, kolors, instantid, supra2, flux, chroma, z-image, boogu, sd3, pulid, flux2, ideogram, lens, moss, flux3, svd-enc) | 6,000 | 84k | medium-high |
| 8 | **Candle vendored DiT/TE/VAE copies → hookable Linear + composable-op mode + block hooks** (§3.5) | 8 (flux, z-image, krea, lens, ltx, wan, sdxl, pulid/chroma) | 9,500 | 76k | medium-high |
| 9 | **Per-family sigma/scheduler builders → gen-core named schedules + data** (§4) | 20 | 3,500 | 70k | medium |
| 10 | **Wan-lineage causal-3D VAE → 1/backend** (§2.2 VAE-B) | 8 (wan, qwen-image, qwen-image-2-1, anima, krea, bernini, scail2, krea-realtime) | 5,500 | 44k | medium |
| 11 | **Candle `to_image`/frames/RGBA + Qwen-VL preprocess** → `cdl/core/image.rs` + `core-llm::vision_preprocess` (§5) | 20 | 2,000 | 40k | high |
| 12 | **ViT/CLIP-vision & Qwen-VL vision towers** (VE-A, VE-B, §1.2) | 12 | 2,800 | 34k | medium-high |
| 13 | **T5/UMT5 + CLIP-text → 1 each/backend** (drop stock candle-transformers paths, §1.3) | 12 (flux, chroma, mochi, sd3, supra2, wan, scail2, krea-rt, bernini, sdxl, kolors-IP, sam3) | 2,300 | 28k | high |
| 14 | **Causal-3D VAE primitives** (CausalConv3d, per-frame GN, depth-to-space, feature cache) + BigVGAN/Snake vocoder kit (§2.2 VAE-C/E) | 8 (ltx, mochi, seedvr2, h3, svd, wan, + audio) | 2,000 | 16k | low-medium |
| 15 | **Gemma-2/3 TEs → llm** (+ add a Gemma 3 arch, §1.1 TE-B) | 3 (pid, sana, ltx) | 2,300 | 7k | medium |

Sum of midpoints ≈ **~72,000 LOC** deletable [E], about 4–5% of src. It is concentrated in the code
every new model copies. The high-confidence core is ranks 2, 7, 8, 10, 11, 12 and 13 (~37k).
Ranking by score rewards wide shallow glue (ranks 3–5); ranking by confidence × LOC puts ranks 2, 8,
7 and 10 first. Design implication 8 uses the second ordering.

Not counted: the inherent **×2 backend** duplication (every row above exists once in MLX and once
in candle); see Design implications.

---

## 11. Risks

1. **Numerical parity / byte-identical goldens.** Many forks exist *because of numerics*, not math:
   candle-flux2 `compute_mu` f64 vs shared f32 (`cdl/candle-gen-flux2/src/pipeline.rs:241`); merge-
   vs-residual LoRA "(W+δ)·x ≠ W·x + δ·x to ~1 ULP" (`cdl/core/quant/adapt.rs` doc); bf16-parity cast
   in MLX `AdaptableLinear::quantize`; f32-vs-bf16 encoders; cos-first vs sin-first timestep;
   Q8→`Q8_0` double-quant (0.56% RMS). A shared component must carry these as explicit knobs with
   per-family golden tests, or parity gates will fail.
2. **Checkpoint-format variance.** Same math, different keys/layouts: BFL vs diffusers FLUX AE and
   DiT; original-Qwen vs diffusers Qwen VAE naming (`anima/vae.rs` docs); open-clip vs HF CLIP;
   MLX `[out,kT,kH,kW,in]` vs torch conv layouts; MLX-packed triples (gs 64 vs 32) vs GGUF vs NVFP4
   sidecars; `language_model.model.*` prefixes. A component library needs a **key-map + layout
   adapter layer** (gen-core already holds some: `gen-core-family::sd3_encoder_artifacts`,
   `cdl/core/comfyui_vae.rs`).
3. **MLX vs candle API differences.** MLX is channels-last for conv/GN, lazy (command-buffer
   size/watchdog — sc-24245 memory: "NEVER eager-read inside a block stream"), `&self` forwards over
   flattened `Array` leaves (`AdaptableHost` dotted-path install); candle is channels-first, eager,
   `Module` trait, fused CustomOps without backward (forces training copies), `VarBuilder` vs raw
   mmap loaders (boogu doc). The same component therefore stays two implementations unless a
   backend-neutral tensor trait is introduced — the realistic target is "one spec (config + keys +
   tests) in gen-core, one impl per backend".
4. **Quant-spine divergence between llm and media** (B3): adopting llm TEs on candle changes Q4
   numerics (Q4_K vs Q4_1).
5. **Memory/admission coupling**: per-family `memory_strategy.rs` (114k LOC) prices *specific*
   component instances (e.g. `mlx-gen-sdxl/src/vae/pricing.rs`, `ip_adapter/pricing.rs`,
   iOS pricing certificates referencing `vae_graph.rs`). Swapping a component invalidates those
   prices/calibration (memory: "pin bump stales calib"). Dedupe must move pricing with the component.
6. **Hidden behavioral deltas** inside "identical" components: Chroma masked vs FLUX unmasked T5;
   candle Chroma encodes at natural length vs MLX padded-masked (`cdl/candle-gen-chroma/src/text.rs`
   doc) — the backends already disagree; consolidation forces choosing one.
7. **iOS build identity / feature gates** (`ios_build_identity.rs` in z-image, sana, anima, pid,
   qwen-image, wan) tie crate graphs to iOS bundles; moving components across crates touches those.

---

## Design implications

1. **Two-level component model: spec in gen-core, impl per backend.** Each shared component
   (AutoencoderKL-2D, WanCausalVae, HfDecoderEncoder, T5Encoder, ClipText, ClipVision/ViT, QwenVlVision,
   AxialRope, Timestep/AdaLN kit) gets a tensor-free spec in gen-core (config struct, key map per
   checkpoint layout, latent constants, tap policy, golden fixtures) and exactly **one** MLX and one
   candle implementation. Families become *config + glue*. This is the pattern already emerging
   (`QWEN_WAN_Z16_MEAN`, `gen_core_iris::assemble_window`, `gen_core_flux3::chat_text`).
2. **Make llm `CausalLm` the decoder-LM encoder** for TE-A/TE-B, with: (a) `TapPolicy {layers,
   final_norm, early_exit, combine: concat|stack|interleave, drop_prefix}`; (b) cache-free encoder
   entry; (c) its `Projection` replaced by (or unified with) the media `AdaptableLinear`/`QLinear`
   spine so quant tiers and adapters are identical across LLM and media; (d) add Gemma 3, gpt-oss,
   Qwen2.5-VL vision. Delete the 17 media copies.
3. **One hookable Linear per backend is the root fix for candle vendoring.** Port the MLX
   `AdaptableLinear` contract fully to candle (bases: dense, packed-affine, GGML, NVFP4, int8/fp8,
   W4A8, ConvRot; residuals: LoRA/LoKr/LoHa; trainable mode that swaps fused CustomOps for
   composable ops). With it, `ip_dit`, `packed_dit`, `dit_train`, `train_dit`, `packed_te`
   (~20k LOC) collapse into the inference modules.
4. **Hooks at every block boundary, defined once.** The control/IP/PuLID/reference/preview/
   block-streaming/residency features are all "per-block" cross-cutting concerns implemented per
   family today (15 MLX `block_stream.rs`, 10 control families, 4 IP variants). A shared
   `BlockStack` driver with hooks — `pre_block(i)`, `extra_kv(i)` (IP/PuLID CA), `post_block(i)`
   (ControlNet/VACE residual, preview), `stream/evict(i)` (residency) — lets each feature be built
   once. MLX `qkv.rs` proves the "knobs as data" approach works for the attention prologue; extend
   to the block level and port to candle.
5. **VAE library: 3 parametric codecs cover ~80% of VAE mass** (AutoencoderKL-2D with optional
   patchify+BN; Wan causal-3D lineage; plus a causal-3D primitive kit). Keep LTX/DC-AE/Mage/H3 as
   singletons on the shared kit and the existing `LatentDecoder` seam (PiD swap) + tiling.
6. **Schedules are data.** Move every per-family sigma builder into gen-core named schedules with
   per-family parameters; forbid per-family `sigmas` fns outside gen-core (a CI grep is enough).
7. **Host-side preprocessing lives in gen-core/core-llm only.** Qwen-VL smart-resize, CLIP
   normalize, to_image/RGBA/frames packing are tensor-free or trivially backend-thin; the candle
   core should mirror `mlx/core/image.rs`.
8. **Sequence the work by confidence × score**: (1) candle Linear/adapter spine + vendored-copy
   removal; (2) VAE-A and VAE-B; (3) TE-A onto llm with TapPolicy; (4) RoPE/qkv port to candle +
   timestep/adaLN kit; (5) control/IP hooks; (6) schedules. Each step must carry per-family golden
   parity and re-home the matching `memory_strategy` pricing.

---

## Uncertain / not verified

- **LOC-to-delete figures are estimates** [E] from file sizes and a retained-glue factor (20–40%);
  no code was diffed line-by-line. The ~72k total could be off by ±30%.
- **ChatGLM3 ↔ llm GLM-4** compatibility (rotary convention, MQA grouping, bias layout) not
  verified; may need its own arch arm.
- **Qwen2.5-VL 3-D MRoPE** layout vs llm's Qwen3-VL interleaved MRoPE not compared in code.
- **One Qwen-VL vision tower for both 2.5 (windowed) and 3 (deepstack, learned pos-embed
  interpolation)** is plausible but unverified; might remain two towers in the llm crate.
- **Wan 2.1 vs 2.2 vs QwenImage-2.1 VAE** unification: lineage verified by docs; the vae22 doc lists
  structural differences (channels-last, residual shortcuts, 16× spatial) — the parametric merge is
  assumed feasible, not prototyped.
- **SCAIL-2 OpenCLIP-XLMR ViT-H** equivalence to the SDXL ViT-H body inferred (key names and 31-block
  truncation differ); not read line-by-line.
- **BigVGAN/Snake** sharing across LTX vocoder, H3 audio decoder and `candle-audio` codecs is keyword
  evidence only.
- **Control/IP equivalence classes** (CN-A/CN-B/IP-A/REF-A) are from module docs, not code reads;
  the deletable estimate there is the least reliable.
- **Candle per-family adapters** reduction assumes the merge-at-load path can be expressed as key-map
  data; candle's deliberate "merge, don't residual" for byte-parity on dense bases may keep more code.
- The regex definition counts (194 Attention, 135 MLP, 440/304 rope fns, ~45 sinusoid fns) include
  helpers/test-adjacent code in non-test files and miss inline implementations.
- Whether mlx-llm loads **SceneWorks MLX-packed gs-32** snapshots (Boogu tier) directly was not
  checked.
- Audio crates were inventoried only where they overlap media components (Qwen3 block, T5Gemma,
  codecs); their internal duplication was not audited.
- The working tree was not used; everything is origin/main a43724b82. No builds/tests were run.
