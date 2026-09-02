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
- Open: CPU/GPU view divergence of a freshly allocated Metal shared buffer under a full page
  cache with dirty write-back. Options in `RUNBOOK.md` (GPU-side verify loop; Apple/MLX report;
  no-load-under-write-back mitigation).

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
