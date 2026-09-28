# Installed receipt recovery: SDXL on the 8 GiB VM

On 2026-09-16, the installed SceneWorks 0.8.11 MLX worker refused manifest SDXL q4
at 1024x1024, count 1: 16.90 GiB requested versus 6.00 GiB safely available
(8.00 GiB total, 2.00 GiB reserved). Its log reported that the decode-quality
table had no independently resolved artifact identity.

The receipt existed and pinned `SceneWorks/sdxl-base-mlx` revision
`36699bb8a6353e61c920e3bf19f0e6f8e4151c55`, variant `q4`. The recorded metadata
stamp was reproduced exactly from the VM's current file metadata by substituting
only the old filesystem device number: `16777234` instead of `16777233`.
macOS device numbering is mount-dependent. Including it in a persistent receipt
caused unchanged artifacts to lose identity and disabled SDXL's verified tiled
decoder. The prior real-artifact test constructed an identity from the cache path
and did not exercise this failure.

The fix removes the device number from new persistent stamps, retains legacy
stamp verification, and migrates a still-matching legacy baseline under the
receipt lock. A stale HF baseline is replaced only after all recorded files match
the pinned upstream revision: local LFS bytes must match the upstream SHA-256,
and small non-LFS files are compared with their exact pinned remote bytes.
Converted artifacts use their existing independent content digest. A changed
receipt or file during verification is refused. Recovery occurs lazily as each
installed tier is considered; unused lower tiers cannot block a fitting request.

## Actual VM validation

The reviewed worker test binary ran in Michael's existing UTM VM, configured with
8 GiB memory. It copied the VM's actual install receipts into temporary bookkeeping;
the original installed receipts and 0.8.11 application were not changed. It resolved
weights through production receipt/path code, repaired identity with the production
verifier, bound the shipped decode-quality policy, prepared the production load
shape and residency policy, and used the selected strategy for generation.

- Prompt: `A cinematic frame of a woman in a red raincoat holding a red umbrella standing in the middle of a neon street on a rainy night`
- Negative prompt: none.
- SDXL q4, 1024x1024, count 1, seed 1234, four validation steps.
- Selected decode: 256-pixel tiles, 64-pixel overlap, transformer residency window 1.
- Conservative admitted estimate: 5.996 GiB, including 0.671 GiB allowance.
- Cold-inclusive peak active allocation: **4,954,305,800 bytes (4.61 GiB)**.
- Output: one nonblank image; dimensions and nonuniform pixels asserted by the test.
- Test result: passed in 52.01 seconds. Z-Image and SANA Sprint admission also passed;
  only SDXL executed a render in this run. Refusals at 4 GiB total remained enforced.

This is an 8 GiB VM runtime check, not a physical 8 GiB Mac or a full desktop queue
acceptance run. Four steps exercise the pipeline and memory strategy; they are not
an image-quality comparison or a new measurement matrix. No memory estimate,
reserve, uncertainty allowance or decode-quality constraint changed in this fix.

The standalone executable required its own matching Metal library. Initial attempts
with a missing or different packaged library failed before rendering. The successful
run used the exact build's library through the VM's existing shared folder, avoiding
its near-full disk. The executable and library hashes were verified after transfer:

```
executable: 23639765c74dd26bbde71ed3b9bebc750efa55fa3295632fc4ef4dcd66369ed4
metallib:   ed247b9c448cdcb7f5a196bdce0ed3e7c5f564a1317f26c22777287bbe4af774
```

Command (paths specific to this VM):

```sh
PMETAL_METALLIB_PATH='/Volumes/My Shared Files/Shared/sc23648-validation/validation.metallib' \
HF_HOME=/Users/michael/.cache/huggingface \
SC23648_INSTALL_DATA='/Users/michael/Library/Application Support/SceneWorks/data' \
SC23584_RENDER=sdxl \
'/Volumes/My Shared Files/Shared/sc23648-validation/receipt-validation' \
reported_q4_requests_credit_the_implemented_memory_ladder --ignored --nocapture
```

See [vm-runtime.log](vm-runtime.log) for the complete successful run. Generated
memory-matrix changes in this hotfix only refresh source hashes; there was no
catalog measurement campaign.
