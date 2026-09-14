# Image generation failures (sc-23187)

The source fix pairs with inference PR #972. The reported jobs were inspected from
local job records and worker logs; user prompts are not copied into this document.

## Request preparation and model recovery

Checkpoint-plan generation now filters negative prompts using the selected provider's
capability, matching ordinary image generation. Krea 2 ignores unused negatives.

Boogu variants share one repository. A cached Base or Edit variant is not evidence
that Turbo exists. On-demand repair now includes default q8 variants and checks
component/tokenizer presence before loading. The observed Turbo snapshot initially
had no `turbo/` directory; the native model downloader completed that variant at
revision `a459e614d408bfdf57089c32cc3da706f5a017de`. The tokenizer and each of the three
single-file weight components were present after download.

## Flux2 budget accounting

The original Flux2 Dev request asked for bf16, 1024x1024, count 2. Image count is a
sequential output loop; request geometry uses batch 1. The lower-resolution advice
path now follows that same batch convention.

The old reported budget was an incremental pool: 128 GiB physical memory minus
2 GiB reserve minus approximately 44.72 GiB active MLX allocations. Cache sequencing
drops the previous logical generator before loading Flux2, and the refusal occurred
after Flux2 load completion. The installed Dev text encoder contains 44.72465 GiB
of tensor data, supporting attribution to the current encoder. Historical logs do
not prove the identity of every live allocation. New diagnostics record active bytes,
the pre-load external baseline, provider credit, and reserve separately.

All ladder candidates now compare complete-pipeline peaks with a consistently
normalized budget. Only current-provider bytes above the recorded external baseline,
bounded by the provider's resident envelope, are credited. Unrelated active allocations
remain charged. Resident estimate allowances preserve their previous incremental
basis, keeping existing resident admission decisions unchanged.

A second defect padded fixed OS/app reserve as though it were model activations.
The actual staged transformer plus VAE is 60.333699625 GiB. Generic headroom adds
14 GiB activations and 2 GiB fixed reserve. The former calculation applied the
3.104173817 allocator allowance to all 16 GiB, producing 126.000480698 GiB against
126 GiB available. The correction retains the full raw peak and reserve but charges
only 14 GiB of activations: 119.792133064 GiB. Law-derived activation terms and the
allowance coefficient are unchanged.

The catalog declaration scan also required request contexts on an unrelated legacy
Resident inventory row before reaching Flux2's exact staging declaration. It now
filters by strategy before validating that strategy's context. The real shipped
bf16/q4/q8 declarations select eager sequential staging; missing, malformed, duplicate,
or provider-refused staging declarations remain rejected.

Regression coverage includes sequential count, unrelated allocations, retained
resident admission, staged selection, fixed reserve, and geometry scaling. The
ignored installed-checkpoint admission test uses the real Dev component metadata
and shipped catalog through production load-shape/policy selection, without executing
GPU tensors. It passed at the permanent inference pin: both count 1 and count 2
select StagedResidency with a 76.33 GiB raw peak. See the inference diagnostic note for Unicode and
True-V2 content-identity verification. Source/test evidence is distinct from a
rebuilt desktop binary or a fresh full-resolution render.
