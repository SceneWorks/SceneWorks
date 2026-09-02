# SC-22414 Mac2 cold-load investigation — paused 2026-08-31 ~11:10, resumable

## Confirmed so far (see memory: sc-18791-mac2-determinism-blocker)
- Warm-parity failures (max=1.000000 mean=0.160441 rms=0.215802) are caused by corrupt
  **cold-load of the Gemma TE**: stage hashes prove noise/positions/latents identical between
  renders, only `v_ctx`/`a_ctx` differ, and the **measured (first) render is the corrupt one** —
  the warm repeat matches known-good hashes. Matches Mac1's MLX `Load::eval_gpu` gap.
- Reliable repro recipe: (1) sequential read of all 300 GB of blobs
  (`cat /Volumes/Data/huggingface/hub/models--SceneWorks--ltx-2.5-mlx/blobs/* > /dev/null`, ~51 min),
  (2) rotating new-extent dirty writer, (3) run the case. Failed 2/2 with the recipe; passed 7/7 without.
- Passing-run reference hashes: stage1 out v=af2ea7de10ca78b1 a=aa7e7ab1c939a675;
  final latents video=ccbbede022124677 audio=6ff9cb7c911dea9c; corrupt render-1 stage1 out
  v=30c6f2f2cf876c77. Full logs: repro/stage-b.log (pass), repro/stage-c.log (fail), repro/load-audit.log
  (18 MATCH, no MISMATCH — but the TE and the <64MB LoRA A/B tensors were NOT covered by that audit).

## What was interrupted
The final discriminating run: TE-audited replay (te load_backbone policy line + per-tensor
first/second double-read hashes in the Resident branch of gemma4_te.rs). The sweep was ~25/51 min
in; nothing of it is lost except time. Binary at SceneWorks/target/release/memory-mlx-adapter
(built 10:39, includes all probes: stage_hash boundaries + STEPS/ENTRY gates, Weights::from_file
load audit, gemma4_te policy log + TE double-read audit).

## To resume (run from /Volumes/Data/calibration/sc-18791/diagnostic, referred to as $D)

1. Start the dirty writer:
   nohup zsh -c 'i=0; while true; do dd if=/dev/zero of=$D/repro/dirty/n$i bs=16m count=1600 2>/dev/null; rm -f $D/repro/dirty/n$((i-6)); i=$((i+1)); done' & (mkdir $D/repro/dirty first)
2. Cache-reset sweep (~51 min): cat /Volumes/Data/huggingface/hub/models--SceneWorks--ltx-2.5-mlx/blobs/* > /dev/null
3. Replay with all probes (from $D/SceneWorks):
   source $D/repro/case-env.sh
   SCENEWORKS_LTX25_LOAD_AUDIT=$D/repro/load-audit2.log \
   SCENEWORKS_LTX25_STAGE_LOG=$D/repro/stage-d.log \
   SCENEWORKS_LTX25_STAGE_STEPS=1 SCENEWORKS_LTX25_STAGE_ENTRY=1 \
   SCENEWORKS_MEMORY_CAPTURE_DIR=$D/repro/raw \
   SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX=docs/calibration/sc-18791/e4d37922-b971699f \
   ./target/release/memory-mlx-adapter < $D/repro/stdin-1788173412.json
4. Read: grep "te load_backbone\|te tensor" $D/repro/stage-d.log
   - policy=Sequential → corruption happens in per-layer mid-forward loads (fix = eval barrier in
     mlx-llm SequentialStack::run_layer path); the Resident-branch tensor audit won't fire.
   - policy=Resident + any "te tensor" line where first!=second → corrupt-at-load proven per tensor.
   - If ctx differs again (compare two "generate_av_latents entry" lines) but TE audit is clean,
     next suspect is the connector/heads or the tokenizer-side arrays.
5. Afterwards kill the writer and rm -rf $D/repro/dirty.

## Ground rules still in force
- Pinned campaign checkouts at /Volumes/Data/calibration/sc-18791/{SceneWorks,inference} stay
  untouched; all experiments live in this diagnostic clone (patched via [patch] table appended to
  $D/SceneWorks/Cargo.toml pointing at $D/inference-patched; DIAGNOSTIC commits are never merged).
- Nothing here re-enables Mac2 for evidence; the campaign grid finishes on Mac1 (sc-22414 is the
  mechanism-confirmation + upstream-fix story: explicit materialization/eval barrier after load on
  the LTX-2.5 MLX path).
- Evidence branch evidence/sc-18791-mac2 holds only the seed commit; the Mac1 coordinator knows.

## RESULT (resumed 2026-08-31 16:15, finished 17:10)
- Recipe reproduced the failure a 3rd time (canonical metrics). Logs: repro/stage-d.log, repro/load-audit2.log (18 MATCH).
- `te load_backbone policy=Sequential` on BOTH renders → TE loads per layer inside the forward via
  mlx-llm `SequentialStack::run_layer` (residency.rs ~191/213/228); `materialize_in_batches` is skipped on
  this branch, so the cold per-layer `Load` evaluates mid-graph. Render 1 ctx corrupt (9ecb47fcf2ebb926),
  render 2 correct (91d66f82657ad6e4) — identical to the previous failing run. Investigation complete.
- Machine left quiet: writer killed, dirty files removed, no adapter/harness running.

## FIX ATTEMPTS (2026-09-01 evening)
- Attempt 1 (inference-side barrier: `view.materialize_accessed()` before `layer.forward` in
  mlx-llm `SequentialStack::run_layer`): FAILED — identical corrupt ctx 9ecb47fcf2ebb926. Reverted.
  Conclusion: not a CPU→GPU fence/race; the READ itself returns wrong bytes (deterministic).
- MLX v0.32.0 read path (fetched source under target/.../_deps/mlx-src) has two real bugs:
  (a) `mlx/backend/common/load.cpp` Load::eval_cpu: read task runs in a packaged_task; the CPU-stream
      task does `fut.wait()` — exceptions from a failed pread are CAPTURED and never re-raised →
      silent success with an unfilled buffer.
  (b) `mlx/io/load.cpp` ParallelFileReader::read(offset): on a SHORT pread the loop advances
      buffer/size but NOT offset → re-reads the same range → silent duplicated bytes.
  Patch written: mlx-rs-fork/mlx-sys/patches/load-read-error-propagation.patch (fut.get(), EINTR
  retry, errno to stderr, offset += m) and registered in mlx-rs-fork/mlx-sys/build.rs patch_files.
  Diagnostic SceneWorks Cargo.toml now [patch]es https://github.com/michaeltrefry/mlx-rs to the local fork.
- TE load audit at 1 MB floor (load-audit3.log) is CONFOUNDED: casting ~330 cold tensors on the GPU
  tripped the watchdog (kIOGPUCommandBufferCallbackErrorSubmissionsIgnored); the one MISMATCH
  (layers.8.mlp.up_proj, first_diff=0, no zeros) may be a GPU artifact. Audit hashers now use the
  CPU stream (as_dtype_device(F32, cpu)). Re-run with 1 MB floor only if needed.
- NEXT: recipe with the MLX-patched adapter (no audit env; STAGE_LOG + ENTRY=1). Outcomes:
  pass → fixed; loud "[read] pread(...)" error → mechanism (a) confirmed with errno, decide retry
  policy; silent identical failure → IO layer exonerated, back to Metal.
- 20:19 result: MLX read-path patch (fut.get + EINTR + offset advance) also FAILED identically
  (ctx 9ecb47fcf2ebb926) with NO "[read] pread(" lines → pread never failed; bytes off disk are
  correct. IO layer exonerated. Remaining suspects: CPU-written MTLBuffer visibility to the GPU
  (event-path Fence has no input_coherent kernel; the FAST_SYNCH path does), or the TE forward
  consuming a buffer image before the copyout lands. Next: F_NOCACHE knob on the reader
  (MLX_PMETAL_NOCACHE=1) to reproduce cold reads without the 43-min sweep, then
  MLX_METAL_FAST_SYNCH=1 as the visibility discriminator.
- 20:44: F_NOCACHE replay (MLX_PMETAL_NOCACHE=1, writer live, NO sweep) PASSED — uncached/slow reads
  alone do not trigger it; the sweep (cached path under eviction) stays mandatory. Next probe:
  per-layer CPU-stream fnv vs GPU-stream sum of each TE layer's weights inside SequentialStack
  (SCENEWORKS_LTX25_LAYER_AUDIT=<log>) to split "wrong bytes in memory" from "GPU sees a different
  image of the buffer".
- 21:50: per-layer weight audit run REPRODUCED the failure (ctx 9ecb47…) while ALL 664 TE layer
  tensors were identical across all 4 stack passes (2 per render: positive/negative prompt) at
  BOTH the CPU view (fnv) and the GPU view (sum): weights are NOT corrupted. Divergence is in
  activations or in the resident/once-loaded inputs (embed_tokens, norms, heads, connector).
  Next: hash layer-0 input and every layer's output carry per pass (act-audit.log); first
  differing layer between render-1 pass and render-2 pass (same prompt) names the culprit.
- 22:41: activation-probe run crashed (kIOGPUCommandBufferCallbackErrorTimeout → abort) — probe
  artifact: as_dtype_device(cpu) on the lazy GPU embedding output made a cross-stream dependency.
  Probe switched to GPU cast + host readback (same path stage_hash uses). Rerun launched 22:45.
- 23:55 DECISIVE (act-audit.log): render-1 pass-1 `layer 0 IN` (token-embedding output) is EXACTLY
  ZERO (sum 0, hash of zeros) and every layer OUT stays zero; pass 2 (negative prompt, same process,
  minutes later) and both render-2 passes are non-zero and correct. Weights of every layer were
  correct at CPU and GPU views. => The once-loaded, cold 2 GB embed_tokens buffer reads as zeros on
  its FIRST GPU use and as real data afterwards: CPU-written buffer not yet GPU-visible/resident at
  first use under dirty-page pressure. Tokenizer exonerated (empty prompt embeds non-zero).
  FIX DIRECTION: GPU-touch (reduce + wait) the resident tensors at load, before any forward, in
  mlx-llm from_file_sequential (campaign-legal, inference-side); MLX-level residency fix optional.
- 00:46 ROOT CAUSE NAMED: with fut.get()+errno patch active, the reader logged
  `pread(fd=11, size=33554432, offset=6506862082) returned -1: Bad address` (x2, text_encoder
  embedding region) and the run failed LOUDLY (CPU task threw → GPU wait timed out). Pristine MLX
  swallows that EFAULT → zero-filled buffer → all-zero embedding → deterministic garbage ctx.
  Mechanism: kernel copyout into a freshly allocated Metal shared buffer fails with EFAULT under
  a full page cache + dirty write-back. Fix: in ParallelFileReader readfn, on EFAULT fault the
  destination range in from user space and retry (log "[read] EFAULT recovery"), keep fut.get()
  so exhausted retries fail loudly. GPU-touch in from_file_sequential REVERTED (not the fix).
  Verification run launched 00:50 (expect pass + "EFAULT recovery" lines on stderr).
- 01:52: EFAULT-recovery run FAILED identically with ZERO pread errors and ZERO recovery lines →
  in the common mode the read succeeds yet the embedding is zero: the DESTINATION memory loses the
  write (or never retains it). Probes for next run: Load::eval_cpu reads via heap staging + memcpy +
  immediate memcmp (logs "[load] destination did not retain write"), and a CPU-stream hash of
  embed_tokens at every embed() call (act-fix5.log "embed_tokens ..."). Launched 01:58.
- 03:00 FIRST PASS UNDER THE REPRODUCER (exit 0, both renders ctx 91d66f…, embed_tokens CPU hash
  identical/non-zero at every call, pass-1 layer-0 IN correct, ZERO "did not retain" lines):
  reading via heap staging + user-space memcpy into the Metal buffer FIXES it. Mechanism: kernel
  copyout (pread) straight into the Metal shared-buffer mapping silently drops the write (or
  EFAULTs) under a full page cache + dirty write-back; user-space stores are retained. Patch
  reshaped to production form (per-chunk staging in ParallelFileReader::read, fut.get(), EINTR,
  offset advance, errno; NOCACHE knob dropped) — confirmation run #2 launched 03:05.
- 04:07 CONFIRMATION #2 PASSED with the production-shaped patch (exit 0, ctx 91d66f… both renders,
  pass-1 layer-0 IN correct). Score: fix 2/2 pass vs unfixed 6/6 fail under the reproducer.
  Stressors stopped, dirty files removed. Remaining: regression test run, commit on a fork branch
  (sc-22414/load-read-staging), push + PR to michaeltrefry/mlx-rs (ASK FIRST), then inference
  mlx-rs rev bump and SceneWorks inference pin bump (coordinate with Mac1 / RELEASING.md).
- 04:14: error propagation finalized (stream-task failure recorded in a process-global slot,
  mlx/stream_error.h; host Event::wait() re-raises → Rust Err, no abort, no hang). Regression test
  PASSES ("[read] pread(...) returned 0 (EOF)" then Err). Committed locally on fork branch
  sc-22414/load-read-staging (NOT pushed). Final recipe run on the finished patch launched 04:16.
- 05:16: FINAL-SHAPE RUN #3 FAILED (canonical metrics, no reader errors) with the SAME per-chunk
  staging read path that passed run #2 → staging+memcpy is NOT a reliable fix (2/3). A completed
  user-space write can still end up as zeros ⇒ the destination's backing pages are replaced
  AFTER the write (residency/wiring migration, slow under dirty write-back). Fits EFAULT, the
  F_NOCACHE pass (slow read → write lands after migration) and the read-churn pass.
  Fork commit d6c5a5ff (branch sc-22414/load-read-staging) must NOT be pushed as-is.
  Next (launched 05:30): whole-tensor heap staging + delayed re-verify loop (two clean memcmp
  checks 150 ms apart, rewrite on mismatch, log "[load] destination lost write ..."), with the
  embed CPU-hash probe on. A logged lost-write event confirms the theory; the loop is the fix if
  the CPU view is truthful. If the run still fails with zero lost-write events, the CPU and GPU
  views have diverged (mapping split) and the verify must be done from the GPU side.
- 07:15: verify-rewrite run PASSED but with ZERO "lost write" events over the whole load, and the
  render took 67 min (150 ms verify delays per tensor). So the CPU view of every buffer was
  always correct; the theory "data lost from memory" is NOT supported. Combined with the 05:16
  failure of the lean variant: the GPU intermittently sees ZEROS where the CPU sees data —
  CPU/GPU view divergence (stale GPU mapping of freshly faulted pages), timing-dependent; every
  timing-heavy variant passed (3/3), the lean one failed (1/1), unpatched 6/6 fail.
  Score summary: unpatched 0/6, staging-lean 2/3, staging+delays 2/2.
  Machine quiet (writer killed, dirty removed). Fork branch sc-22414/load-read-staging still local.
  OPTIONS: (a) GPU-side verify loop (GPU reduction vs CPU sum after load; re-touch/rewrite until
  the GPU agrees) — targets the observed divergence directly; (b) report to Apple/MLX with the
  reproducer; (c) operational mitigation: no model load during/after heavy write-back.
