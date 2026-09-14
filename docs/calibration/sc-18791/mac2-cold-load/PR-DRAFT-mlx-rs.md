# fix(mlx-sys): stage safetensors reads through the heap and surface read failures (sc-22414)

**Branch:** `sc-22414/load-read-staging` (michaeltrefry/mlx-rs, off `bd8f0e3`)
**Commit:** `d6c5a5ff`
**Follow-ups:** bump `mlx-rs`/`mlx-sys` rev in `SceneWorks/inference` `Cargo.toml` (line 61–62); then a SceneWorks inference-pin bump per RELEASING.md.

## Problem

MLX 0.32.0's `ParallelFileReader::read(char*, size_t, size_t offset)` `pread()`s each 32 MB
chunk straight into the destination — a Metal shared buffer. On a full page cache with dirty
write-back pressure, the kernel copyout into that mapping silently leaves the pages zero (once
observed as `EFAULT` instead). Separately, `Load::eval_cpu` only `wait()`ed on the reader's
`packaged_task` future, so even a loud failure was captured and dropped; evaluation reported
success over an unfilled buffer.

Observed effect: the cold 2 GB Gemma text-encoder embedding table read back as all zeros on its
first use, the whole text encoder ran on empty input, and LTX-2.5 produced deterministic garbage
conditioning — the measured-vs-warm parity breach (`max=1.000000 mean=0.160441 rms=0.215802`)
that blocked the SC-18791 MLX campaign on Mac2. Byte-identical across days because zeros are
deterministic.

## Evidence (Mac2, M5 Max 128 GB, macOS 26.6.1)

Reproducer: sequential read of the 300 GB model blob set (page-cache reset) + a rotating
new-extent `dd` writer + the campaign case `ltx-2-5-mlx-q4-dev-conv-512x768-f145` via the
memory adapter. Unpatched: **6/6 fail** with the canonical metrics. Stage hashes proved seeded
noise, positions and every TE layer's weights (CPU and GPU views) identical between the two
renders; only `layer 0 IN` (embedding output) differed — exactly zero on the first pass.

Ruled out by experiment (each a full reproducer run): macOS version; sampler/RNG; CPU→GPU fence
race (pre-forward materialization barrier: no effect); GPU-touch of resident tensors at load
(no effect — bytes never landed); uncached reads alone (`F_NOCACHE`: passes); read churn alone
(passes); EFAULT user-space fault-in retry (no effect); tokenizer (empty prompt embeds non-zero).

With per-chunk heap staging + user-space `memcpy`: **2/2 pass** (renders bit-identical, first-pass
embedding correct, no reader errors). Final patch shape re-verified on a third run (see runbook).

## Change

`mlx-sys/patches/load-read-error-propagation.patch` (registered in `build.rs` `patch_files`):

- `mlx/io/load.cpp` — `readfn` reads each chunk into a heap buffer and `memcpy`s it into the
  destination (overhead: one ≤32 MB chunk per worker); retries `EINTR`; advances the file offset
  on short preads (upstream re-read the same range into the advanced buffer); names the errno.
- `mlx/backend/common/load.cpp` — the stream task no longer drops the reader's exception; it
  records it (a throw there would `std::terminate`).
- `mlx/stream_error.h` + `mlx/backend/metal/event.cpp` — process-global pending stream-task
  error, re-raised by the host-side `Event::wait()` so the caller gets an `Err`. (Metal backend
  only; non-Metal builds keep upstream behaviour for the propagation half.)

Regression test: `mlx-tests/tests/load_read_error_propagation.rs` — a payload truncated after
the lazy load must evaluate to an error (unpatched: `SIGABRT` with `fut.get()`, silent zeros
with upstream `wait()`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
