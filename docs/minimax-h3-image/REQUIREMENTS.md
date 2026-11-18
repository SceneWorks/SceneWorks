# MiniMax-H3-Image: Requirements (draft 0.2)

| | |
|---|---|
| **Status** | Approved (draft 0.2, Michael answered D1–D6 on 2026-11-10). Implemented as CodeGraph feature SW-F363, complete 2026-11-18; all 18 items complete. LoRA training's app-level proof (AT10) moved to SW-F369. |
| **Date** | 2026-11-10 |
| **Models** | [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) and its prompt expander [MiniMaxAI/MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE), both released 2026-11-10 |
| **Base** | Existing MiniMax-H3 support (CodeGraph SW-F245, SW-F329 and follow-ons) |
| **Evidence** | HF configs and safetensors headers read 2026-11-10. Code mapped against SceneWorks `c0892e14d1` and its inference pin `4a36a7f8c` (on inference `main`). |

## 1. What the models are

### 1.1 MiniMax-H3-Image

H3-Image is a 33.1B single-stream DiT for text-to-image and multi-reference image editing. MiniMax built it from the H3 video/audio model and removed the audio branch.

| Component | H3-Image | Relation to our H3 code |
|---|---|---|
| DiT | 50 layers, hidden 5376, 56 heads × 128, 2 refiner layers, FFN 14336, text_dim 5120, 24 latent channels, patch 1×2×2, AdaLN width 96768 (= 6 × 3 × 5376) | Same shapes as H3 minus the audio tensors. AdaLN width matches `MODALITY_NUM = 3`. New weights. |
| Per-step output heads | `pdd_fused_heads.safetensors`: `video_out.weight` F32 [6, 96, 5376], bias [6, 96]. One head per sampling step replaces `final_layer.video_out`. | New |
| Sampler | 6 fixed sigmas `[1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.2320, 0.0]`, velocity Euler, no shift, no CFG, no negative prompt | H3's velocity Euler and `SigmaSchedule::from_sigmas` are reusable |
| Text encoder | Qwen3-VL, 64 layers, hidden 5120, layer-50 tap | **Byte-identical** to H3's `text_encoder/` (same LFS hashes) |
| VAE encoder | 3D-conv CNN, 24 channels, 16× spatial | Same architecture as H3. Latent mean/std are identical to H3's. |
| VAE decoder | 2D ViT decoder: 24 layers, 16 heads × 64, 4 register tokens, rope θ 100, rope ratio 0.5, tile 256 / overlap 64 | Different from H3's 36-layer 3D ViT decoder. H3's `ViT3dDecoder` needs a 2D mode. |
| Edit references | Up to 16 images, named `<Picture N>` in the prompt. References go through the VL encoder **and** the VAE, and all tokens form one packed sequence. | Mirrors H3's ref2va path, but the caps differ (H3 allows 9 images) |
| Resolution | 4–20 MP working range, best near 16 MP (4096², 5472×3072, 6272×2688). Default area 4 MP (2048²). | Far above H3's 1.03 MP canvas cap and SceneWorks' 4096 px max edge |
| License | MiniMax-H3-Image Community License. The text matches the H3 license except for the model name and date. §IV adds commercial terms. | Same gating pattern as H3 |

Reference implementation: diffusers `main` (`MiniMaxH3ImagePipeline` and the modular pipeline), plus SGLang, vLLM-Omni and ComfyUI recipes.

### 1.2 MiniMax-H3-Image-PE (prompt expander)

- **What it does.** PE turns a short prompt into the long, detailed English description H3-Image expects. It has a t2i mode and an edit mode, and the edit mode sees the reference images. It also proposes an output size (the HF repo ships `revised_size` examples).
- **Architecture.** A fine-tune of Qwen3.8-27B (`qwen3_5`). It is a hybrid model: 64 layers, hidden 5120, three linear-attention layers to every full-attention layer, and a vision tower.
- **Weights.** NVFP4 (ModelOpt, group 16), 17.9 GB. NVFP4 runs natively only on Blackwell NVIDIA GPUs. The vision tower, embeddings and the linear-attention conv/gate projections are unquantized.
- **Prompts.** The system and task prompt files ship in the repo under `prompt/`, with sha256 hashes in `prompt/manifest.json`.
- **Relation to our code.** Our LLM lane already runs `qwen3_5` models through `Architecture::Qwen35` on `mlx-llama` and `candle-llama`, with the vision tower. The Qwen-Image-2.1 PE rewriters (9.4B, `crates/sceneworks-worker/src/qwen_prompt_rewrite.rs`) use that lane and are the template here.
- **Not the video expander.** H3 video's expander is H3-Context-IR, which MiniMax offers only through its API; no weights exist. PE is image-only (its prompts cover t2i and edit only).

## 2. Goals

- G1. A SceneWorks user can generate an image from text with MiniMax-H3-Image in Image Studio, on Mac (MLX) and CUDA (Candle).
- G2. A SceneWorks user can edit or compose an image from 1–16 reference images with `<Picture N>` prompts.
- G3. The model runs at its full native resolution range (up to 20 MP) on both backends. Memory admission picks the memory strategy, or refuses truthfully when nothing fits.
- G4. The full H3 memory ladder works from the first release. The model never ships resident-only.
- G5. A user can expand a short prompt with the local PE model on both backends.
- G6. A user can apply community H3-Image LoRAs, and can train their own H3-Image LoRAs in Training Studio, on both backends.
- G7. Installing H3-Image after H3 does not re-download the 66.7 GB text encoder.
- G8. The code is shaped so that moving it to the shared generation architecture (SGA) later is a mechanical port.

## 3. Requirements

### 3.1 Image generation (inference)

- **E1. Image-mode H3 DiT.** The H3 DiT runs a single-frame image request without audio. The audio projections become optional at load, and T=1 geometry is accepted for this family only. H3 video behaviour does not change.
- **E2. Per-step output heads.** Step *i* of 6 applies head *i* from `pdd_fused_heads.safetensors` instead of `final_layer.video_out`. The step index is passed down to where the head is applied.
- **E3. Fixed distilled sampler.** Steps are fixed at 6 with the published sigmas, velocity Euler, guidance 1, and no negative prompt. A request with any other step count or guidance is refused with a clear error (upstream returns HTTP 400). It is never silently changed.
- **E4. 2D ViT VAE decoder.** H3's ViT decoder gains a 2D mode: a 2-axis rope at ratio 0.5, a spatial-only output patch, no temporal chunking or token drop, and spatial tiling at 256 / 64 overlap. Its output matches the diffusers `AutoencoderKLMiniMaxH3Image` decoder.
- **E5. Text-to-image.** The family exposes `t2i` with the Qwen3-VL prompt template from H3-Image's `processor/chat_template.json`. Prompts are truncated at 4096 tokens.
- **E6. Multi-reference edit.** The family exposes `edit` with 1–16 references, matching diffusers. The references go through both the VL encoder (as vision tokens with `<Picture N>` naming) and the VAE (as latents in the packed sequence). Output size comes from the request, not the references.
- **E7. Full resolution on both backends.** Both MLX and Candle accept any size in the published range: 1024 px² to 16,777,216 px² of `min_generation_area`, output up to 20 MP, on a 32 px stride. The H3 limits of a 1.03 MP canvas and a 2016 px edge do not apply to this family.
- **E8. Tiers.** The DiT has bf16, q8 and q4 tiers on both backends, with H3's policy of keeping the output heads and AdaLN dense. The text encoder reuses H3's existing bf16/q8/q4 artifacts.
- **E9. Parity on both backends.** E1–E8 hold on MLX and Candle. Each backend matches the diffusers reference within the tolerance the family's parity test states.
- **E10. Registration.** The family registers as its own model id with an image modality and an image descriptor. It does not copy H3's `Modality::Video` or `mac_only` flags.

### 3.2 Memory (inference)

- **E11. Full memory ladder from the first release.** H3-Image ships every memory strategy H3 has at the pinned inference rev, on both backends, selected automatically. The H3 baseline it must match:

  | Rung | H3 MLX | H3 Candle | H3-Image requirement |
  |---|---|---|---|
  | 0 Resident | Implemented | Implemented | Implemented |
  | 1 Staged residency (text encoder, DiT and VAE never co-resident) | Implemented, forced | Implemented, forced sequential | Implemented, forced. PE is staged out before the text encoder loads. |
  | 2 Bounded decode (spatial tiling) | Implemented, 256 / 64 tiles | Implemented | Implemented on the **new 2D ViT decoder** (E4) |
  | 3 Bounded attention | Not applicable (measured, sc-18661) | Implemented (SW-F329) | Candle: implemented and priced at image geometry (up to ~20k image tokens plus 16 references). MLX: keep H3's not-applicable verdict only if a measurement at image geometry confirms it; otherwise implement it. |
  | 4 Bounded transformer residency (deferred load, block window) | Implemented for the DiT **and** the text encoder | Implemented for the DiT and the text encoder (SW-F329) | Implemented for the H3-Image DiT, including the per-step output heads (E2), and for the shared text encoder |

  The H3 AdaLN cache eviction carries over. Admission computes host and GPU bytes separately for each phase, using H3's SW-F329 accounting, so a request that can stream is admitted instead of being rejected at the resident peak.
- **E12. Memory ladder with LoRAs.** Rung 4 stays available when LoRAs are applied, on both backends. Today H3 reports rung 4 as `Missing` when an adapter is configured (sc-18650), because a streamed block is rebuilt from the stored weights and would drop the LoRA. H3-Image applies the LoRA to each block as it is streamed in. The fix lives in the shared H3 code, so H3 video gains it too.
- **E13. Memory evidence.** Per the standing rule: two anchors per backend and tier (fully resident and fully optimized), with the rest derived. It is measured once, at the end of the feature.

### 3.3 Prompt expander (inference + SceneWorks)

- **E14. PE runtime on both backends.** PE runs on the existing `Qwen35` LLM lane on MLX and Candle, including the vision tower for edit mode. If the 27B hybrid layout needs changes to `Qwen35Model` (for example the layer pattern or sizes), those changes are part of this feature.
- **E15. PE tiers.**
  - CUDA on Blackwell: the published NVFP4 weights, run natively.
  - MLX and non-Blackwell CUDA: q8 and q4 tiers converted from NVFP4, and bf16 if the conversion preserves it. These are published per D3.
  - Each tier states its quality versus the NVFP4 original on a fixed prompt set.
- **E16. PE prompts and output.**
  - **Prompt files:** PE loads its system and task prompts from the model's own `prompt/` files and checks them against `prompt/manifest.json`. Following the Qwen rewriter pattern, the licensed prompt text is never copied into our repos.
  - **Output:** the expanded prompt replaces the user's prompt only after the user accepts it, and the proposed size is offered, not forced.
  - **Edit mode:** PE receives the reference images and the `<Picture N>` mapping.
- **E17. PE in the product.** Image Studio offers "Expand prompt" for H3-Image, in both the advanced and simple views. PE has its own manifest entry, license gate (same license family), download, and memory admission. It is staged out of memory before generation starts (E11 rung 1). The H3-Image prompt guide stays as written guidance next to it.

### 3.4 LoRA (inference + SceneWorks)

- **E18. LoRA inference.** Both backends apply up to the SceneWorks LoRA limit (currently 10) of H3-Image LoRAs, with per-LoRA scale, at every tier and every memory rung (E12).
  - **Accepted formats:** the formats the community is likely to publish: diffusers/PEFT names, kohya `lora_unet_` names (the format our H3 trainer-receipt test already reads), and ComfyUI exports.
  - **Unknown keys:** an unknown key layout is refused with a message naming the unmatched keys. It is never partially applied.
- **E19. LoRA training.** Training Studio trains H3-Image LoRAs on MLX and Candle. H3 has no native trainer today, so this trainer is new, built on `gen-core-train` with the existing image trainers as templates (`mlx-gen-qwen-image-2-1/src/training.rs`, `candle-gen-krea/src/training.rs`). It covers t2i training. It also covers edit-pair training (references plus a target), because edit is half of the model's capability.
- **E20. Trainer memory.** Training a 33B DiT has to fit by design: a quantized frozen base, gradient checkpointing, and block streaming for the frozen weights. The training-memory estimate is computed the same way as for inference (E11, E13). Training resolution defaults to 2048² (the model's default 4 MP area) and is configurable.
- **E21. Trainer output.** A trained LoRA saves in a format E18 accepts. It loads back on both backends and reproduces the trainer's own sample within the stated tolerance. Save and resume produce the same next update as uninterrupted training.

### 3.5 SceneWorks

- **E22. Manifest and download.**
  - **New entry:** a new built-in model entry with bf16/q8/q4 tiers.
  - **Shared text encoder:** its text-encoder rows point at the **existing H3 rows** (the `MiniMaxAI/MiniMax-H3` and `SceneWorks/minimax-h3-mlx` text-encoder files), so an existing H3 install satisfies them.
  - **Files:** it downloads `transformer_diffusers/` (D2) and `pdd_fused_heads.safetensors`, the VAE, and the processor/tokenizer. It never downloads the second DiT copy in `transformer/`.
- **E23. License gate.** Both models require license acknowledgment, each with its own license entry and notice, under the H3 use-restriction safeguards (S1–S8). This includes §IV's requirement to display "MiniMax-H3-Image" in a commercial product's UI.
- **E24. Image Studio controls.** Image Studio (advanced and simple) offers t2i and edit for this model:
  - **Shown:** prompt, size presets from the published aspect table, seed, up to 16 references with `<Picture N>` insertion, LoRAs, and Expand prompt.
  - **Hidden:** steps, guidance and negative prompt.
- **E25. Resolution limits.** SceneWorks' image request validation and UI allow this model's full range (up to a 6272 px edge, for 21:9) on both backends, without raising the limits for any other model.
- **E26. Family isolation.**
  - **Family token:** H3-Image has its own family token for checkpoints, LoRAs and training.
  - **No misfiling:** the checkpoint inspector, the H3 fine-tune import (`minimax-h3`) and the LoRA-family detection never misfile H3-Image weights or LoRAs as H3, and never misfile H3 weights or LoRAs as H3-Image.
- **E27. Routing and admission.** The worker routes both backends, and the memory gates admit or refuse by tier and size, with LoRAs counted. Both models appear in the capability, preview-support and memory-anchor config.
- **E28. Publication.** The converted DiT tiers (`SceneWorks/minimax-h3-image-mlx`) and the PE tiers are published to SceneWorks HF repos (D3). Each publish still gets Michael's go-ahead at the moment it happens.

### 3.6 Fit with the shared generation architecture

- **E29. SGA-ready shape.** The SGA plan's new-model rule (PAT-3) defines a new model as: config, graph, weight plan, route declaration, manifest entry, and measured anchors as data. H3-Image keeps those six things separable in today's code:
  - **Host math:** geometry, the sigma list, reference caps, the resolution table and the size rules live in a tensor-free `gen-core` module, not in backend crates.
  - **Sampler as data:** the 6-step schedule is a named sigma list, not code.
  - **Graph:** the DiT, decoder and heads are built from H3's existing layers and add no new family-local copies of shared layers.
  - **Memory as data:** memory anchors and estimates are data files, with no per-family memory constants in SceneWorks code.
  - **SceneWorks tables:** no new SceneWorks code tables are added where a manifest field can carry the fact.
  - **Runner switch:** if SGA ENG-1a (SW-F331, the runner core) merges to inference `main` before the denoise-loop item starts, H3-Image targets the runner (SGA plan §7.1 rule 7).

## 4. Feature acceptance tests

Real-weight runs (AT1–AT4, AT8–AT11) run in the explicit real-weights workflow, never in ordinary PR CI.

- **AT1.** On the Mac (MLX): t2i renders at 2048², 4096² and 6272×2688 complete from a fresh process at the tier and strategy admission picks. The 2048² image is coherent and matches the prompt.
- **AT2.** On a CUDA runner (Candle): the same three renders complete.
- **AT3.** On each backend: a 3-reference edit using `<Picture 1>`–`<Picture 3>` produces an output that visibly draws on each reference.
- **AT4.** Parity: for a fixed seed and prompt, each backend's first-step DiT output and VAE decode match diffusers within the stated tolerance.
- **AT5.** With H3 already installed, installing H3-Image downloads no text-encoder bytes.
- **AT6.** These requests are refused with a clear message: steps ≠ 6, guidance ≠ 1, a negative prompt, 17 references, or a LoRA whose keys don't match.
- **AT7.** Installing or running either model without license acknowledgment is refused.
- **AT8.** On each backend: a 16 MP render that does not fit resident is admitted and completes, **with two LoRAs applied**, through bounded transformer residency and bounded decode. The run record shows the selected rungs.
- **AT9.** On each backend: PE expands a short t2i prompt and a 2-reference edit prompt. The expanded t2i prompt produces an image that is more detailed than the one from the short prompt.
- **AT10.** On each backend: a short LoRA training run on a small dataset completes and produces a LoRA. The LoRA visibly changes output when applied, and it loads on the other backend.
- **AT11.** On each backend: a community H3-Image LoRA from HF, in each accepted format available at the end of the feature, applies without error.

## 5. Decisions

### 5.1 Answered by Michael (2026-11-10)

| # | Decision | Answer |
|---|---|---|
| D1 | Resolution | Full native range, on Mac **and** CUDA (E7, E25) |
| D2 | DiT copy | `transformer_diffusers/` (E22) |
| D3 | Converted tiers | Publish them (E28) |
| D4 | Prompt expander | Support PE now (E14–E17) |
| D5 | LoRA | Support LoRA inference **and** LoRA training (E18–E21) |
| D6 | Architecture | Build on today's code, shaped for an easy move to SGA (E29) |

### 5.2 Decided while drafting (open to override)

- **D7. LoRA streaming fix in shared H3 code (E12). High confidence.** H3-Image reuses H3's streaming code, so fixing it there is less code than forking it, and it also lifts H3 video's resident-only fallback when LoRAs are applied.
- **D8. Default training resolution is 2048² (E20). Medium confidence.** It matches the model's default area. Training at 16 MP would take much more memory and time per step.
- **D9. Edit-pair training is in scope (E19). Medium confidence.** Edit is one of the model's two tasks, and edit LoRAs are a common community format for edit models.

## 6. Traps found while mapping the code

1. H3's canvas cap (`CANVAS_MAX_PIXELS = 768×1344`, `MAX_CANVAS_EDGE = 2016`) and SceneWorks' 4096 px max edge both block the native range (E7, E25).
2. H3's reference caps are 9 images / 12 total with a 2048 px short edge. H3-Image needs 16 (E6).
3. `DitProjections::TENSOR_COUNT = 17` and a 638-tensor test assume the audio tensors exist (E1).
4. H3's ViT decoder config validation rejects a rope dim that is not a multiple of 6. H3-Image's is 32 (E4).
5. H3's memory constants were calibrated on 1344×768 video and do not transfer (E11).
6. The H3 text-encoder rows omit shard 13. H3-Image's rows must reuse the same row set, not list all 14 shards (E22).
7. The H3 descriptor is `Modality::Video` and `mac_only: true`. The new descriptor must not copy those flags (E10).
8. Memory rungs are declared per family. A new family id starts with none of H3's rungs wired, even though the code is shared. That is how a model ends up resident-only (E11).
9. H3's rung 4 turns off when any LoRA is configured (sc-18650). Without E12, every LoRA render at 16 MP would fall back to resident.
10. The checkpoint inspector labels anything shaped like H3 as `minimax-h3`. Without E26, H3-Image weights and LoRAs would be misfiled.
11. PE's NVFP4 weights don't run on Mac or pre-Blackwell CUDA. Converted tiers are required, not optional (E15).
