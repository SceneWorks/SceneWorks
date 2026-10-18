# MOSS-Image product integration

The catalog exposes `moss_image_base` and `moss_image_sft` as native text-to-image models. Ordinary execution requests 50 steps, CFG 3, schedule shift 5, training shift 1 and context 4096. The native training grid can deduplicate timesteps; the completed receipt records both requested settings and the steps that ran.

SFT's **8-step acceleration** starts OFF. Its two explicit policies are `unmerged_bf16` (BF16 backbone with the published FP32 residual/time/modulation operations) and `full_fp32` (the published already distilled FP32 transformer, without a second LoRA). Both select a trained prefix of 1–8 steps, defaulting to 8. Installing components or requesting eight ordinary steps never enables acceleration.

## Requests and replay

Typed `POST /api/v1/image/jobs`, MCP `generate_image`, presets, retry and duplicate use the same final payload validator, after preset expansion or replay merging. The existing raw `POST /api/v1/jobs` security boundary refuses `image_generate` and `image_edit` for every model. Its directly allowed `image_detail` operation refuses MOSS because that provider implements text-to-image only.

```json
{
  "projectId": "your-project-id",
  "model": "moss_image_sft",
  "prompt": "日本語 <tag> a ceramic cup {\"items\":[]}",
  "advanced": {
    "mossImage": { "acceleration": "unmerged_bf16", "max_sequence_length": 4096 }
  }
}
```

Generic steps, guidance, negative prompt and schedule shift retain their ordinary wire fields. `advanced.mossImage` accepts the native strict options object: acceleration, context, `uncond_skip_layers`, `training_shift` and `slg` (layers, scale, start and stop). For unconditional skip layers, null/omission means the author's `[22]`; `[]` disables skipping. An empty SLG layer list remains empty. CFG 0 retains its negative prompt. Accelerated requests refuse explicit incompatible controls instead of discarding them. The studio disables those controls and retains ordinary choices locally for switching OFF.

Prompt text is preserved, including multilingual prose, tags and authored JSON. Native prompt batches preserve order; outputs are prompt-major and image-minor. Frozen advanced tensor inputs remain available through the native contract, without a product tensor-upload endpoint.

Requested options persist through studio state, saved presets, recipes and image history. PNG workflow exports preserve canonical options and a validated actual execution receipt, replacing any local adapter path with its frozen filename. Imported receipts are display metadata. Generation scrubs supplied or replayed reports and reads a new report only after successful, uncanceled provider execution while holding the serialized generator transaction.

## Assets and admission

The primary Base/SFT transformer and required SFT text encoder/tokenizer/VAE/scheduler snapshot are separate pinned download components. DMD components are optional downloads selected by precision. Their provenance distinguishes bundled official Apache-2.0 licenses from the DMD README's Apache-2.0 metadata, for which no separate LICENSE file is published.

Before loading tensors, the worker authenticates the frozen repository, revision, relative path, physical size and SHA-256 of each selected file. Cache identities include mode, precision, components and content pins. Admission prices the selected execution closure and request workspace; cached unselected optional packages do not increase it. Warm admission credits only the current generator's actual owned residency, bounded by current committed device memory. Normal worker progress, heartbeats, cancellation and artifact writes surround the native provider.

## Small terminal product acceptance

The ignored test below calls `run_image_generate_job`, including normal route selection, asset preparation, admission, serialized cache, default native loader, progress consumption and PNG/recipe writing. It does not inject a generator. Run it alone only after the coordinator admits the device and provisions the exact frozen snapshots; source compilation does not prove native execution.

```sh
# Apple Silicon; the coordinator supplies the variables listed below.
cargo test --locked --release -p sceneworks-worker --lib \
  image_jobs::moss_consumer_tests::moss_real_native_product_worker_campaign \
  -- --exact --ignored --nocapture --test-threads=1
# Windows CUDA uses the same command plus --features backend-candle before --lib.
```

Required environment: `MOSS_BASE_SNAPSHOT`, `MOSS_SHARED_SNAPSHOT` and `MOSS_DMD_SNAPSHOT` name canonical snapshots in one existing Hugging Face hub; `MOSS_TRANSPORT_ARTIFACT_ROOT` names a fresh absolute directory outside the checkout. `MOSS_NATIVE_TERMINAL_ADMITTED` is `sc-26325-mlx` or `sc-26325-physical-gpu0`. The CUDA controller additionally supplies `SCENEWORKS_GPU_ID=0`, `MOSS_PHYSICAL_GPU_INDEX=0` and the independently verified `MOSS_EXPECTED_CUDA_UUID`. The source test performs no acquisition or hardware reservation itself.

`product-worker/case-summary.json` uses protocol `sc-26325-product-worker-v1` and records the backend and five case rows. Four successful rows (`base_off`, `sft_unmerged_bf16`, `sft_full_fp32`, `sft_off`) each produce a 64×64 PNG, a typed actual receipt and a JSON array of normal HTTP progress posts. Success rows request one step and context 64. `sft_off_warm_cancel` reuses ordinary SFT after its lazy renderer has materialized, requests a 1,000-step ordinary schedule and cancellation after actual step progress (the schedule remains active through the normal two-second cancellation poll) and must publish no image or receipt. The controller validates each actual receipt and outcome; counts alone are insufficient.

The separate native backend acceptance retains advanced tensor and cancellation coverage. Neither this smoke nor CPU consumer tests establish comparative quality or a benchmark verdict.

## Web bundle evidence

The feature keeps every initial, raw and route limit enabled. Against the clean `875666c7c` web baseline, initial JavaScript changes from 474,820 / 146,257 / 124,981 bytes (raw/gzip/Brotli) to 465,936 / 143,132 / 122,586. Sharing disclosure details load through the existing interaction boundary; preference persistence remains on the initial path.

Total JavaScript changes from 2,888,178 / 859,234 / 733,239 to 2,894,510 / 861,383 / 735,294 bytes. The compressed total limits add exactly the positive measured feature deltas, 2,149 gzip and 2,055 Brotli bytes: 862,149 and 737,055. This preserves the previous baseline headroom of 766 and 1,761 bytes. The raw total limit and all coverage checks remain unchanged.
