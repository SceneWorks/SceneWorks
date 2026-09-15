# sc-23584: MLX active working-set admission

SceneWorks 0.8.9 could refuse an otherwise executable memory strategy because its
fallback charged allocator cache as mandatory physical memory. The estimate also
missed optional staging compositions and treated complete transformer weights as
interchangeable streaming blocks. These corrections apply to shared image/video
admission; they do not create new calibration measurements or measured authority.

## Accounting and execution

The budget compares live allocations with an active pipeline peak. Reclaimable
MLX cache is excluded. The retained same-cell active recapture spread is
0.1260183508475594, derived by `scripts/derive-ladder-margins.mjs`. Image estimates
receive it in the selector. Video anchor estimates include it in the core law
and receive no second selector widening. Counted Candle allocations retain their
existing separate policy.

Providers declare actual phase ownership. TwoStage retains the nonstreamable
heavy trunk through decode. ThreeStage sheds the denoiser before decode; both
conservatively keep decoder weights during denoise. Optional staging is a
request parameter and is recorded in the executed composition and evidence key.
Legacy keys with no override remain byte-compatible. Measured evidence takes
precedence over estimates of the same execution, not every composition at a rung.

Exact stream inventories retain fixed tensors and count the largest materialized
window among sequential independent stacks. They must reconcile with the loaded
transformer's total. TE-only streaming earns no DiT credit. Unknown inventories
and bounded auxiliary networks remain charged in full. Generic MLX activation
fallbacks do not borrow dense-attention or whole-tail tiling savings.

## Existing observations and derived workspace

Z-Image's retained 1024-square staged q4 decode sweep is in inference Actions
[31918926833, job 95095446324](https://github.com/SceneWorks/inference/actions/runs/31918926833/job/95095446324).
It reports active GiB of 4.457, 4.548, 4.708, 5.376, 6.129, 6.840 and 8.595
for tile edges 256, 320, 384, 448, 512, 640 and 768. The calculation fits global
feature-map and tile-area terms from 256 and 512, includes the printed rounding
interval, subtracts the 97,583,622 decoder-weight bytes, and checks the remaining
edges as held-out points. Below the observed image size the global term is kept.
It is restricted to the actual shared layer-wise convolution/GroupNorm decoder:
f32, output channels [512,512,256,128], input channels [512,512,512,256], and
spatial divisors [8,4,2,1]. Whole-tail decoders are outside this calculation.

Z-Image denoise uses its compatible resident anchor's active residue with linear
image-token growth because its fused SDPA does not allocate a dense score matrix.
SDXL uses the retained UNet-only 1024-square CFG-batch-two, 77-token seam in
`mlx-gen-sdxl/src/memory_strategy.rs`: 6.5868 GiB minus the reference UNet's
5,134,927,368 fp16 bytes. The upper printed endpoint is used. Dual-CLIP workspace
is calculated from both encoder constructors, charging every layer's intermediates
at f32 plus full score tensors. Prompt demand is bounded from lowercase UTF-8 bytes
before byte-BPE merges and is propagated from the actual request. Multiple padded
windows increase conditioning and denoise workspace. A fitted cell may not be
reused with changed tile parameters or uncovered SDXL conditioning demand.

## Coverage and verification boundaries

Registry/common-builder review covers TwoStage SDXL, Kolors, FLUX1/PuLID, FLUX2
Dev/DevEdit/DevControl/Klein, Chroma, SD3.5, Qwen base/edit/control, Ideogram and
Krea base/native/control; ThreeStage Z-Image, Anima, SANA/Sprint, Lens's explicit
ladder path, Mage and InstantID. Other image providers and video routes still
benefit from shared cache accounting and conservative fallbacks. Undeclared
stream/workspace facts remain unknown; schedule coverage does not certify a new
measured fit for every model.

The ignored `reported_q4_requests_credit_the_implemented_memory_ladder` regression
reads the exact shipped q4 snapshots and runs production preflight at 8 GiB total
with 2 GiB reserved. At 1024-square it calculates 5.090 GiB for Z-Image count two
(sequential) and 5.996 GiB for SDXL count one, including uncertainty. These are
estimates on the actual artifacts, not proof from a physical 8 GiB Mac.

SDXL's new 256/64 quality policy is independently sealed from five production-latent
comparisons against dense decode using the existing maximum RGB-u8 error limit of
48. No admission measurement matrix is generated. The manifest contains the exact
artifact, geometry, implementation stamp, fixture hashes and observed errors.

The opt-in `SC23584_RENDER=all` path of that same regression loads the selected
production providers and begins the selected memory scopes. Four denoise steps
per image, including the cold load in active-memory accounting, passed on this
Apple/Metal host: Z-Image count two peaked at 4,292,400,896 bytes for both serial
items; SDXL count one peaked at 4,953,965,672 bytes. Each output had the requested
1024-square geometry and nonuniform pixels. Both are below the six-GiB request
budget. This verifies selected-strategy execution on a larger host; physical
8-GiB OS pressure and long-duration use remain outside this local check.
