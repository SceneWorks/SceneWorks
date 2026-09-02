# SC-22414 — Mac2 LTX-2.5 MLX cold-load investigation (2026-08-30 → 09-02)

Artifacts from the Mac2 (Mac17,6, 128 GB, macOS 26.6.1) investigation of the LTX-2.5
measured-vs-warm parity breach (`max=1.000000 mean=0.160441 rms=0.215802`). Read
`RUNBOOK.md` first — it is the chronological log with every experiment, score, and the exact
reproducer recipe. Nothing here re-enables Mac2 for evidence; the campaign grid finished on Mac1.

## Bottom line

- Reproducer (300 GB blob read sweep + rotating dirty-page writer + case
  `ltx-2-5-mlx-q4-dev-conv-512x768-f145`): unpatched **0/6 pass**.
- Localized to the once-loaded 2 GB text-encoder embedding table reading as **exact zeros on the
  GPU during render 1's first pass** while the CPU view of the same buffer is always correct
  (`logs/act-audit.log`, `logs/act-fix5.log`, `logs/act-fix8.log`). Seeded noise, positions,
  tokenizer and every TE layer weight are bit-identical between renders.
- Falsified: OS version, sampler/RNG, CPU→GPU fence race, MLX read errors/short reads, tokenizer,
  uncached reads alone, read churn alone, a GPU-touch at load, and heap-staged reads (2/3 — the
  passes were timing).
- Mechanism: CPU/GPU view divergence of a freshly allocated Metal shared buffer under a full page
  cache with dirty write-back — the GPU reads the fresh buffer as zeros while the CPU already holds
  the bytes, and the views converge with time. Not an OS-version defect: the same hardware on
  26.5.2 passed only because the reproducer's pressure was never applied there.

## Resolution (2026-09-02, inference PR — see the story's external links)

Runbook option (a), the GPU-side verify loop, shipped in inference as a load-boundary guard:

- `mlx_llm::primitives::coherence` (mirrored as `mlx_gen::coherence`): a wrapping-`u32` checksum
  of a buffer's raw bytes read through the CPU stream and again through the GPU stream. Modular
  addition is reduction-order independent, so the two are bit-identical for the same bytes.
  On divergence the GPU read is retried with a 50 ms → 2 s back-off (12 reads, ~15 s); exhausting
  it is a typed `IncoherentLoad` error naming the tensor and both checksums — never a silent render.
- Wired at every load seam the LTX-2.5 MLX path (and every other MLX model) crosses:
  `CausalLm::build` (embeddings / final norm / LM head, and the whole stack under `Resident`),
  `SequentialStack::run_layer` (each streamed layer, every pass), and mlx-gen's
  `Weights::materialize` / `materialize_accessed` (every provider loader and block-window stream).
- Cost measured on Mac17,6: CPU checksum 18 GB/s cold / 68 GB/s warm, GPU 250–330 GB/s — under
  1.5 s per 26 GB encoder pass.
- Incidence is observable: `coherence::retries()` counts every GPU re-read this process needed.

Validation on the reproducer (recipe in `RUNBOOK.md` "To resume") is the remaining step and needs
Mac2: expected outcome is a pass with non-zero `retries()`; a typed `IncoherentLoad` refusal would
mean the views take longer than ~15 s to converge and the budget needs widening.

The mlx-rs fork's error-propagation half (`mlx/stream_error.h`, commit `d6c5a5ff` on
`sc-22414/load-read-staging`, local to Mac2) is still valid but unrelated to this mechanism —
reads never failed in the common mode — and stays unpushed pending Michael's decision.

## Contents

- `RUNBOOK.md` — chronological runbook + resume recipe.
- `PR-DRAFT-mlx-rs.md` — draft PR text for the mlx-rs fork; the "fix" claim is SUPERSEDED (see
  runbook 05:16/07:15); the error-propagation half (`mlx/stream_error.h`) remains valid.
- `patches/` — the MLX patch (committed d6c5a5ff shape and the later working shape), the
  `build.rs` registration diff, the regression test, the inference diagnostic probes
  (`stage_hash`, load/layer/activation audits), and the SceneWorks diagnostic `[patch]` table.
- `repro/` — captured adapter stdin payloads (replay standalone), `case-env.sh` generator
  (`make-env.mjs`), tee wrapper, and `analyze-layer-audit.py`.
- `logs/` — stage-hash logs, weight/activation audits, load audits, and every replay's stderr.
