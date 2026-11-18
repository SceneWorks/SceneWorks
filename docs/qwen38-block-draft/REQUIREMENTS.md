# Qwen3.8-27B block-draft speculative decoding on MLX: requirements (DRAFT)

Status: requirements approved 2026-11-18. Michael accepted every recommendation in section 6
and added E15: the 4-bit tier ships too, for smaller cards.

Tracked in CodeGraph:

- **SW-F374:** the SceneWorks describer and H3 prompt expander. It holds E1 as consumed, E12,
  E13, E15, the tier work, and AT1, AT2, AT6, AT7.
- **SW-F360:** widened to DFlash2 and DSpine. It holds E2–E5, E10, and draft quantization from
  E11 and E15.
- **SW-F328:** tree checking, E6 and E8.
- **SW-F357:** choosing the check size from measured cost, E7.

The CodeGraph features use their own E numbers.

## 1. Goal

Make Qwen3.8-27B decode at least twice as fast on Apple silicon at the **int8** tier. The engine
gets a new speculative proposer: a small draft model that reads the target's hidden states and
proposes a block of tokens in one pass. The target then checks a tree of the draft's candidates
in one forward pass.

The two Qwen3.8-27B products this serves:

| Product | Model | Request shape today |
|---|---|---|
| Batch image describer (`vision_describe_qwen3_8_27b`) | Huihui Qwen3.8-27B abliterated, re-hosted as `SceneWorks/qwen3.8-27b-abliterated-mlx` (q4 default, q8) | Image + text, JSON-constrained, thinking off, temperature 0.4, `speculative: auto` (native MTP) |
| MiniMax-H3-Image prompt expander (`minimax-h3-image-pe`) | MiniMax-H3-Image-PE, a Qwen3.8-27B fine-tune | Text-LLM lane |

## 2. Evidence (measured 2026-11-18, M5 Max 128 GB, greedy, same 5 prompts, reasoning off)

Decode tokens per second, mean over the 5 prompts:

| Weights | mlx-lm, no speculation | DSpine (linear check, 5 drafts) | Splash 1.7.0 + DFlash2 (tree check) |
|---|---:|---:|---:|
| Base `mlx-community/Qwen3.8-27B-4bit` | 29.3 | 63.6 | 96.0 |
| Abliterated, fully int8 (our q8 with `lm_head`, `embed_tokens`, `in_proj_a/b` also int8; 30.0 GB) | 12.9 | 29.5 | 55.3 |

Our engine (mlx-llm, inference `bab8ac028`) on the shipped abliterated tiers, its own 5-prompt set:

| Tier | No speculation | Native MTP head |
|---|---:|---:|
| q4 | 25.4 | 29.4 (1.16×; demoted on 3 of 5 prompts) |
| q8 | 11.5 | 18.5 (1.61×; demoted on 2 of 5 prompts) |

What the numbers say:

- Splash commits about 1.55× as many tokens per target pass as DSpine (5.4 against 3.5 on the
  4-bit runs), and its cost per pass is about the same (about 42 ms). Its lead comes from
  checking a tree of candidates, not from faster kernels at one request.
- The base-model drafts lose little on the abliterated fine-tune: DSpine's accepted tokens per
  check were 3.57 against 3.34 on math and 1.43 against 1.34 on the caption prompt.
- Short outputs gain least. The caption prompt gained 1.45–1.65× at 4-bit.
- Background Rust builds were running during the int8 runs. Plain mlx-lm int8 measured 12.9
  tok/s against DSpine's published 17.9, so the int8 figures are likely low for every engine.

Not yet measured: the describer's real request shape (image prompt, temperature 0.4, JSON
constraint), the H3 prompt expander, and Splash's single-line checking mode, which would separate
the draft's contribution from tree checking's.

## 3. Scope

In scope:

- mlx-llm and candle-llm in the inference repo, and the core-llm speculative contract they
  share (D2);
- the describer's int8 and 4-bit tiers on both backends (D3, E15);
- the SceneWorks manifest and worker changes that ship them.

Out of scope: decoding several requests at once (D4), and training new drafts (D5).

## 4. Requirements

- **E1. New proposer.** core-llm's speculative vocabulary gains a block-draft proposer beside
  `mtp`, `prompt_lookup` and `draft_model`. A request can name it, and `auto` prefers it over MTP
  when a compatible draft is installed. The speculative monitor's demotion to plain decoding
  still applies.
- **E2. Both draft formats.** The proposer loads DFlash2 (`DFlash2DraftModel`, e.g.
  `incoai/Qwen3.8-27B-DFlash2`) and DSpine (`DSpineDraftModel`, `tencent/Qwen3.8-27B-DSpine-b16`)
  checkpoints, selected by the checkpoint's `architectures` (D1).
- **E3. Compatibility check before load.** The draft's config is checked against the target:
  hidden size, vocabulary, mask token id, target layer ids within the target's depth, attention
  geometry. On a mismatch the draft is refused with a typed reason, the target still loads, and
  the request runs without the proposer. This follows the companion-MTP pattern in
  `core-llm/src/mtp_head.rs`.
- **E4. Hidden states from selected layers.** The target returns hidden rows from the draft's
  `target_layer_ids` (DFlash2 27B: 5, 19, 33, 47, 61; DSpine 27B: 1, 16, 31, 46, 61), not only
  the final normalized row. They come for every prompt position, image positions included, and
  for every committed position after a check. Today `TargetOutput.hidden` carries only the final
  normalized rows.
- **E5. Draft context window.** The proposer keeps the draft's own cache over the draft's
  trained window (DFlash2 2,048 tokens, DSpine 8,192) and feeds a fresh draft only those rows.
  The draft's memory does not grow with context length past its window.
- **E6. Tree checking in one target pass.** The target checks a tree of candidate tokens
  (anchor plus up to 16 rows) in one forward pass:
  - attention uses a tree mask and per-row positions;
  - each GatedDeltaNet layer advances its recurrent state along each row's path from the
    pre-step state;
  - the accepted path's state becomes the cache state with no replay forward, whichever branch
    was accepted.
  A single line of drafts is the special case of a tree with one branch.
- **E7. Tree size chosen per step.** Before each step the engine chooses the number of rows to
  check, from measured step costs by row count and the expected accepted tokens per row. It
  never checks more rows than the remaining token budget allows.
- **E8. Output preserved.**
  - *Greedy:* the output matches the same engine's non-speculative greedy output. Where they
    differ, the report names the first differing position.
  - *Sampled:* the committed tokens follow the target's distribution exactly, using
    multi-candidate rejection sampling over the tree with the engine's seeded sampler.
  This holds under the request's knobs (temperature, top-p, top-k, min-p).
- **E9. Constrained decoding.** Candidates pass through the request's rewindable constraint mask
  (JSON for the describer) before they are checked. A token the constraint forbids is never
  committed, and the mask is rewound to the accepted path after each step.
- **E10. Image prompts.** A request with images runs the proposer. The draft's context
  includes the target's hidden rows at image positions, under the same multimodal position
  rules the MTP proposer uses (`Qwen35MtpMultimodalPrompt`).
- **E11. Memory admission.** Admission prices the draft weights, the draft cache, the tree
  check's activation and recurrent-state buffers, and the hidden-row buffers. The draft is
  quantized to 8-bit at load by default.
- **E12. Int8 describer tier runs fully int8.** mlx-llm runs a snapshot whose `embed_tokens` and
  `lm_head` are quantized. Today it fails with `rms_norm width mismatch … [1, 266, 1280] … [5120]`.
  The shipped tier is decided by D3.
- **E13. Distribution.** Each supported draft is re-hosted under SceneWorks on Hugging Face at a
  pinned revision. It is downloaded with the 27B tier that uses it and recorded in the manifest
  and license inventory: DFlash2 Apache-2.0, DSpine Apache-2.0. The base-model drafts serve the
  fine-tunes (D5).
- **E14. Reporting.** Each decode report names the proposer and draft, and records:
  - target passes and rows checked per pass;
  - accepted tokens per pass;
  - the tree sizes chosen;
  - demotions;
  - draft time and target time separately.
- **E15. A 4-bit tier for smaller cards.** The describer ships a 4-bit tier beside the int8
  tier:
  - The transformer body is 4-bit; `lm_head` and `embed_tokens` are int8, as in the int8 tier.
  - The block-draft proposer runs on it, with the draft quantized to 4-bit.
  - Every card admitted to today's q4 tier is still admitted: Candle's floor admits a 24 GB card
    with up to 2,036 MiB already in use.
  - When the draft and its buffers don't fit beside the target, the request runs without the
    proposer. The tier is never refused for the draft's sake.

## 5. Feature acceptance tests

- **AT1. Speed, real request shape.** The describer runs on a fixed set of 20 images with its
  production request (temperature 0.4, JSON, thinking off). Mean decode is at least 2.0× the
  same engine with speculation off. This is measured at the int8 and 4-bit tiers, on MLX (the
  dev M5 Max) and on Candle CUDA.
- **AT2. Speed, text.** The 5-prompt set from section 2, greedy: mean decode is at least 2.5×
  the same engine with speculation off. This is measured at the int8 and 4-bit tiers, on both
  backends.
- **AT3. State equivalence.** On a tiny random `qwen3_5` model, every tree shape up to 16 rows
  and every accepted branch leaves a cache equal to direct decoding of the accepted path. The
  cache covers the attention KV, the DeltaNet recurrent and convolution state, and the draft
  cache.
- **AT4. Distribution.** On a tiny model, sampled tree decoding matches the target's
  distribution, under the constraint mask and without it. This is checked exhaustively,
  following the existing `speculative` unit tests.
- **AT5. Refusal.** A draft whose geometry does not match the target is refused with the typed
  reason, and the request still completes.
- **AT6. Describer output.** On the AT1 image set, the share of replies that parse as valid
  describer JSON is no lower than with speculation off.
- **AT7. Smaller cards.** At the 4-bit tier, every card today's q4 tier admits is still
  admitted. On a card where the draft doesn't fit, the request completes without the proposer,
  and the decode report says why.

## 6. Decisions (all recommendations accepted 2026-11-18)

- **D1. Which drafts.**
  - *Both DFlash2 and DSpine (recommended).* They share E3–E11. DSpine's linear proposals form a
    one-branch tree, and its logits give the per-row candidates.
  - *DFlash2 only:* less loader work. Its candidate selector is what Splash's tree uses.
- **D2. Candle (Windows/Linux CUDA).**
  - *Include it in this feature (recommended):* the describer ships on Candle too, and core-llm
    holds both backends to the same accept-and-refuse rules. SGLang and vLLM already implement
    DFLASH and DSpine on CUDA, which helps.
  - *MLX only:* that's what you asked to measure. It would leave the CUDA describer on MTP.
- **D3. The shipped int8 tier.**
  - *Re-quantize the remaining matrices to int8 (recommended):* `lm_head`, `embed_tokens` and
    `in_proj_a/b`. That's 2.4 GB smaller and needs E12, and the outputs change slightly.
  - *Keep today's q8 unchanged:* about 4% more weight reads per token, and it can't run
    under Splash.
- **D4. Concurrent requests.** Describe jobs decode one image at a time today (the per-item
  loop at `crates/sceneworks-worker/src/batch_describe_jobs.rs:1962`).
  - *Single request per step (recommended):* matches how jobs run today.
  - *Also batched lanes:* would pay off only if describe jobs start decoding several images at
    once. Splash measured 3.9× at 4 concurrent requests.
- **D5. Drafts for our fine-tunes.**
  - *Use the base-model drafts (recommended):* DSpine lost little on the abliterated weights.
  - *Train drafts for the abliterated model and the H3 prompt expander:* the DSpine recipe
    trains through SpecForge.
- **D6. Speed targets.** AT1 and AT2 use the proposed figures. On the 4-bit runs Splash reached
  3.3× over mlx-lm with no speculation, and DSpine reached 2.2×.
- **Added by Michael:** ship a 4-bit tier for users with smaller cards (E15, AT7). Its
  `lm_head` and `embed_tokens` stay int8. Quantizing those two matrices to 4-bit saves only
  about 1.2 GB, and they are where low-bit error shows up most in the next-token ranking. My
  call, medium confidence.

## 7. Overlap with existing CodeGraph features (found 2026-11-18)

| Existing feature | State | Overlap with this document |
|---|---|---|
| SW-F328 Native tree and two-level verification | Approved 2026-11-04; 7 items (1 ready, 6 backlog) | E6 and E8 (tree checking, Qwen hybrid state, exact acceptance), plus part of E9 and E11. Also covers two-level verification, which this document does not ask for. |
| SW-F360 Native DFlash2 drafting on MLX and Candle/CUDA | No items | E1–E5 and E11 for DFlash2 only. It does not cover DSpine, image prompts, or the describer. |
| SW-F357 Adaptive speculative proposer and proposal depth | No items | E7 (choosing the check size from measured cost) |

All three describe ChatWorks as the consumer. Only this document covers: DSpine (E2), image
prompts (E10), the int8 and 4-bit describer tiers and the quantized-embedding loader fix (E12,
E15), draft distribution for SceneWorks (E13), and the describer acceptance tests (AT1, AT6,
AT7).

## 8. References

- Splash (Apache-2.0, C++/Metal): tree checking and GatedDeltaNet tree kernels, `docs/models.md`
  "Drafts" and `docs/internals.md`. https://github.com/incoai/splash
- DSpine (Apache-2.0): MLX reference including exact GatedDeltaNet rollback (`dspine_mlx/hybrid.py`).
  https://github.com/tencent/dspine
- DFlash (MIT): https://github.com/z-lab/dflash. The DFlash2 model card names SGLang and vLLM
  implementations.
- mlx-llm speculative engine and rollback: `crates/llm/mlx-llm/src/decode/engine.rs`
  (`Proposer`, `SpeculativeTarget`, `CheckpointRingRollback`).
