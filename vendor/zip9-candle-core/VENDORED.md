# Vendored `candle-core` — CUDA-graph parameter cache (sc-24441)

This directory is `candle-core` `0.10.2` from

    https://github.com/huggingface/candle @ 1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d

(the revision the workspace pins `candle-core` / `candle-nn` / `candle-transformers` /
`candle-flash-attn` to) with **two upstream commits cherry-picked, unmodified**:

| Upstream commit | Title | Files |
| --- | --- | --- |
| `cabbc30191b352814d8bb94e821e82faa883c7b5` | Add htod cache for cuda graphs (#3564) | `src/cuda_backend/device.rs` |
| `f53ed3bfbd2e3291c70e4db81eaf0faeccbb8e83` | Parameter cache for CUDA kernel launch (#3598) | `src/cuda_backend/{device,mod}.rs` |

It is wired into the build by the workspace `[patch]` (root `Cargo.toml`), next to the vendored
`candle-kernels`, and is **excluded** from the workspace members so no workspace-wide `fmt` /
`clippy` / `test` rewrites or lints upstream code.

## Why

The decode CUDA-graph runner (`candle-llm` `decode/graph.rs`, epic sc-24432 story sc-24441)
records a decode step with stream capture. At `1e6aa85e`, every candle CUDA op whose kernel takes
a layout — `index_select`, `where_cond`, comparisons, reductions, gathers / scatters, and any
unary / binary / copy on a non-contiguous operand — uploads its `[dims, strides]` from a
temporary host `Vec` (`SlicePtrOrNull::params_from_layout` / `CudaDevice::clone_htod`). A capture
records that as a memcpy node reading a host address freed as soon as the op returns, so a replay
reads garbage (sc-24134 finding 6: 851 such uploads in one Qwen3.8-27B step; the runner's census
refuses them as `host_upload_in_capture`).

`f53ed3bf` routes those parameter vectors through a per-`(device, params)` cache of device buffers
**while a `CudaDevice::enable_cuda_graph_htod_cache` guard is alive**: the eager warm-up fills it,
the capture reuses it, and any host<->device copy the cache does not cover is an error during
capture instead of a silently broken graph. `cabbc301` introduces that guard (and a content-keyed
cache for small `clone_htod` / `memcpy_htod` uploads made under it); `f53ed3bf` does not compile
without it, and it is not an ancestor of `1e6aa85e` (the pin sits on a different upstream branch
line), so it is taken as the prerequisite.

**Not taken:** `3ba428d0` (Improve CUDA stream consistency, #3565), which sits between the two
upstream and moves `Device::new_cuda` onto the per-thread stream — an eager behaviour change this
vendor must not carry. A full candle bump was rejected for the same reason: it brings ~110 commits
including "Optimize CUDA kernels #3600", which moves media numerics and stales their parity
evidence.

## Changes vs upstream `1e6aa85e`

1. `src/cuda_backend/device.rs`, `src/cuda_backend/mod.rs` — exactly the two cherry-picks above
   (`git cherry-pick cabbc301 f53ed3bf` on an upstream `1e6aa85e` checkout reproduces these files
   byte for byte).
2. `Cargo.toml` — upstream inherits every field and dependency from its workspace
   (`*.workspace = true`), which does not exist here. Each inherited entry is spelled out with the
   value upstream's root manifest at `1e6aa85e` declares; the three sibling crates upstream
   reaches by path (`candle-kernels`, `candle-metal-kernels`, `candle-ug`) are the same crates at
   the same git revision the pin resolved before, so `Cargo.lock` changes by one line (the
   `source` of `candle-core`). The root `[patch]` then redirects `candle-kernels` and
   `candle-metal-kernels` to their own vendored copies (each with its own `VENDORED.md`). Features, targets, dev-dependencies, benches and examples are
   upstream's.
3. `src/quantized/cuda.rs` — one local fix, not upstream (sc-24446; upstream `main` still has
   the defect as of 2026-10): the four `QCudaStorage` quantizers (`quantize`, `quantize_imatrix`,
   `quantize_imatrix_onto`, `quantize_onto`) allocate their `MATRIX_ROW_PADDING` tail with
   `alloc_zeros` instead of `unsafe alloc`, as `load_quantized` and `QCudaStorage::zeros` already
   do. See *Quantized padding is zeroed* below.
4. `src/backprop.rs` — one local patch, not upstream (sc-25686; upstream `main` still has the
   behaviour as of 2026-10): `Tensor::backward` computes and stores a gradient only for an operand
   that is itself in `sorted_nodes` (i.e. leads to a `Var`). Every `if needs_grad(..)` guard in the
   backward `match`, the `tracked` set and the `needs_grad` closure above it (delimited by
   `// sc-25686` comments) are the whole change. See *Untracked operands get no gradient* below.
5. `VENDORED.md` — this file.
6. `src/backprop.rs` `Op::MaxPool2D` arm + `tests/grad_tests.rs` `max_pool2d_grad` — one local fix,
   not upstream (sc-25688; upstream `main` still has the defect as of 2026-10): the max-pool
   backward routes each window's gradient to its maximum (split evenly over ties) and gives the
   rows / columns a floor-division pool drops zero gradient. See *Max-pool backward routes the
   window gradient* below.

Everything else (`src/**`, `tests/**`, `benches/**`, `examples/**`, `README.md`, `LICENSE`) is
byte-for-byte upstream. Diff against an upstream checkout to confirm these are the sole deltas.

## Quantized padding is zeroed

A CUDA `QTensor` is stored with `MATRIX_ROW_PADDING` (512) elements' worth of blocks after its
data. The fast MMQ kernels (`fast_mmq`, any quantized matmul over more than 8 rows) load a tile of
`MMQ_ITER_K` (256) elements per row whatever the row length, so for a row length that is not a
multiple of 256 (YuE stage-2's 5504-wide `down_proj`, every synthetic fixture under 256 wide) the
last row's tile reads blocks out of that padding. Their quants meet zero activations, but each
block's scale is still multiplied in: an inf/NaN f16 scale in the padding turns `0 · scale` into
NaN in the output. Upstream's `quantize*` left the padding uninitialized, so the result depended on
what the stream-ordered pool last stored there — a nondeterministic NaN (sc-24446: the CUDA lane's
`stage2::tests::production_load_upsamples_through_candle_llm` failed with NaN logits on an
unchanged code path). llama.cpp zeroes the same padding for the same reason. With finite padding
the term is an exact `0`, so zeroing it changes no finite result. candle-llm's
`primitives::quant::cuda_tests::quantize_on_load_zeroes_the_padding_mmq_reads` pins it on the CUDA
lane.

## Untracked operands get no gradient

Upstream's backward walks only the nodes that lead to a `Var`, but inside a multi-input op's arm it
computes and accumulates a gradient for **every** floating operand. A frozen (non-`Var`) weight
used as a matmul rhs — every base projection of an adapter run — therefore costs a full
weight-gradient GEMM (`lhsᵀ · grad`, `[in, out]`) plus a `zeros_like` accumulator, and since that
operand is never a node of the walk its entry is never consumed: it stays in the `GradStore` until
the store drops (sc-25686: the Iris LoRA step peaked at 45 GB on CUDA against MLX's ~15 GB; the
Qwen-Image 2.1 trainer's `DEAD_WEIGHT_GRAD_COPIES` priced the same retention at 3 copies; re-measured
on CUDA after this patch it is 0, and 24 of its 36 `BACKWARD_RETAINED_HIDDEN` chain tensors went with
it). The patch guards each
operand of the multi-input arms (`Binary` add/sub/mul/div/min/max, `WhereCond`, the convolutions,
`SliceScatter0`, `Scatter`/`ScatterAdd`/`IndexAdd`, `Matmul`, `Cat`, and the `CustomOp2`/`CustomOp3`
results) with membership in the walked set; single-input arms are reached only through a walked node
whose one operand is walked, so they need none. A tracked tensor's gradient is unchanged — the same
ops accumulated in the same order — so this changes no trained result; it only stops computing and
storing gradients nobody reads. A `GradStore` from `backward()` now holds `Var` gradients only (the
root's is consumed as before). `candle-gen-iris`'s
`train_parity::backward_stores_gradients_only_for_trainable_tensors` pins it (full, LoRA, LoKr), and
upstream's `grad_tests` / `custom_op_tests` / `conv_tests` / `matmul_tests` / `tensor_tests` pass on
this copy.

## Max-pool backward routes the window gradient

Upstream's `MaxPool2D` backward builds the "is the window maximum" mask, then multiplies the
upsampled gradient by `avg_pool(mask)` — the **fraction** of the window that is maximal. A unique
maximum therefore received `grad / (kh · kw)` (a quarter of the true gradient for the 2×2 pools VGG
uses), a tie received `grad · ties / (kh · kw)` instead of `grad / ties`, and on an odd input
(`h` or `w` not a multiple of the kernel) the nearest-upsampled `[oh, ow]` gradient was stretched over
the whole `[h, w]` input, misaligning every window. The patch divides by the per-window maximum count
(`avg_pool(mask) · kh · kw`), computes the mask and the routed gradient over the covered
`[oh·kh, ow·kw]` region only, and zero-pads the dropped trailing rows / columns — the
`torch.nn.functional.max_pool2d` gradient (which, on ties, picks one index; an even split is the
same total). Only the pool's own gradient changes; its forward is upstream's.

**Who it changes.** Every backward through a candle `max_pool2d`:

* the latent-LPIPS auxiliary loss (`candle-gen/src/train/latent_lpips.rs` `Op::MaxPool`, built by
  `candle-gen-perceptual`), whose gradient now reaches the trainer at the true scale — the trainers
  that declare a latent-LPIPS family are `candle-gen-sdxl` and `candle-gen-kolors` (SDXL),
  `candle-gen-sd3` (SD3) and `candle-gen-z-image` (Flux); `mage`, `anima`, `krea`, `ltx`,
  `qwen-image-2-1`, `lens` and `wan` declare none (`latent_lpips: None`, the loss is refused) and
  run no max-pool backward;
* the Iris restoration trainer's LPIPS-VGG (`candle-gen-iris/src/train/lpips.rs`), whose gradient is
  pinned against torch in `restoration_train_parity`.

In E-LatentLPIPS the first three VGG pools are identities and only the fourth is a real 2×2
max-pool, so before the fix the gradient reaching the latent through the taps behind it was a
quarter of its true value (and misaligned when that layer's spatial size was odd); the shallower
taps were unaffected. A run that weights `latent_lpips_loss` now trains that part of the term at the
strength the weight states. The Iris LPIPS-VGG has five real pools (the last unused), so the fix
matters there for every tap past the first. `tests/grad_tests.rs::max_pool2d_grad` pins
a unique maximum, a tie and an odd trailing row / column on CPU, CUDA and Metal; upstream's
`grad_tests` / `pool_tests` / `conv_tests` / `tensor_tests` pass on this copy. This crate is outside
the workspace, so CI never runs those tests; the CI-run pin is `candle-gen`'s
`train::latent_lpips::tests::max_pool_backward_is_the_finite_difference_gradient` (an f64 central
difference on a tie-free odd input, torch's gradient on the tie case, and the refused stride < kernel
backward).

## Eager behaviour is unchanged outside the guard

Every behavioural line the cherry-picks add is behind `cuda_graph_htod_cache_enabled()` — a
thread-local guard depth that only `enable_cuda_graph_htod_cache` raises:

* `SlicePtrOrNull::params_from_vec` with no guard returns `SlicePtrOrNull::Ptr(dev.clone_htod(..))`,
  the exact call every replaced site made before; `params_from_layout` still returns `Null` for a
  contiguous layout.
* `clone_htod` / `memcpy_htod` with no guard take the unchanged `self.stream.*` call.
* `clone_dtoh` / `memcpy_dtoh` / `to_cpu_storage` call `check_capture_copy`, which returns `Ok(())`
  immediately with no guard.
* The only unconditional differences are type-level: `T: 'static` on `clone_htod` /
  `memcpy_htod` (every candle dtype and every workspace caller satisfies it — the CUDA compile
  lane builds them all) and the `SlicePtrOrNull::Cached` variant, which is constructed only under
  the guard.

In this workspace the guard is held only by the graph runner, and only around its warm-up and
capture steps (`candle-llm` `decode/graph.rs`). The runtime evidence is the CUDA lane: the
existing candle-llm / candle-gen / candle-audio CUDA suites run unchanged against this copy, and
`candle-llm`'s `cuda_graphs` suite checks the layout-bearing ops bit for bit with and without the
guard.

## The parameter cache lives for the process

`f53ed3bf`'s cache (`CUDA_PARAM_CACHE`, `src/cuda_backend/mod.rs`) is a process-global
`HashMap<(DeviceId, Vec<usize>), Arc<CudaSlice<usize>>>` with **no eviction and no public way to
clear it** — neither cherry-pick exposes one, and this vendor adds none (it stays byte-for-byte
upstream). Consequences a consumer must know:

* Every entry keeps its small device buffer — and, through the `CudaSlice`, the CUDA stream it
  was allocated on — alive until the process exits, including after the model whose captures
  created it is unloaded.
* `DeviceId` is per `CudaDevice` instance, and `candle-llm` opens a new device at every load, so a
  reload of the same model adds its own entries rather than reusing the earlier load's.
* Entries are added only under the graph runner's guard (warm-up and capture), keyed by the value
  of a layout's `[dims, strides]`: a decoder step has a few hundred distinct layouts at most.

`candle-llm` therefore prices it **per load**, not per request: a CUDA load the graph runner wraps
admits `graph_param_cache_load_bytes()` (`PARAM_CACHE_ADMISSION_BYTES` = 1 MiB for each of the
16 capturable step token counts in both logits scopes = 32 MiB) on top of its device bound
(`LoadMemoryEstimate::graph_param_cache_bytes`), and what earlier loads left behind is already in
use when a later load or request reads the device's free memory. A process that loads and unloads
graph-captured models many times accumulates up to that allowance per load; restarting the
worker process is what reclaims it. Upstream eviction (or a clear API) would let the provider free
a load's entries at unload — revisit when the candle pin is next bumped.

## Consumers

**A `[patch]` only takes effect in the top-level workspace.** Any consumer that builds these
crates by git with `--features cuda` (SceneWorks, ChatWorks) resolves upstream `candle-core` unless
its own root manifest patches it to this directory — and upstream `1e6aa85e` has no
`enable_cuda_graph_htod_cache`, so such a build does not compile. Add, next to the existing
`candle-kernels` entry:

```toml
[patch."https://github.com/huggingface/candle"]
candle-core = { git = "https://github.com/SceneWorks/inference", rev = "<the pinned inference rev>" }
```

## MAINTENANCE — re-vendor on every candle pin bump

`scripts/bump_pins.py` refuses a candle bump while this copy exists. To bump:

1. Re-copy `candle-core/` from the new revision over this directory (keep this file).
2. Re-apply `cabbc301` and `f53ed3bf` unless the new revision contains them (upstream main after
   2026-06-24 does), and the zeroed quantized padding (`src/quantized/cuda.rs`) unless the new
   revision zeroes it itself, and the untracked-operand backward guards (`src/backprop.rs`) unless
   the new revision skips untracked operands itself — only then can this vendor and its `[patch]` be
   dropped.
3. Re-derive `Cargo.toml` from the new upstream workspace manifest as described above.
4. Move every pin (`Cargo.toml`, `scripts/check-workspace.py` `PINNED_WORKSPACE_DEPENDENCIES` and
   `VENDORED_PACKAGES`), re-vendor `candle-kernels` and `candle-metal-kernels` (their own
   `VENDORED.md`), regenerate the lock, and run the CUDA lane (`gh workflow run ci.yml -f lanes=windows-cuda`).
