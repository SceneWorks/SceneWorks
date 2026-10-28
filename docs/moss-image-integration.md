# MOSS-Image product integration

The catalog exposes `moss_image_base` and `moss_image_sft` as native text-to-image models. Ordinary execution requests 50 steps, CFG 3, schedule shift 5, training shift 1 and context 4096. The native training grid can deduplicate timesteps; the completed receipt records both requested settings and the steps that ran.

SFT's **8-step acceleration** starts OFF. Its two explicit policies are `unmerged_bf16` (BF16 backbone with the published FP32 residual/time/modulation operations) and `full_fp32` (the published already distilled FP32 transformer, without a second LoRA). Both select a trained prefix of 1–8 steps, defaulting to 8. Installing components or requesting eight ordinary steps never enables acceleration.

## Adapter training

Training Studio exposes **MOSS-Image Base LoRA / LoKr** and **MOSS-Image SFT LoRA / LoKr**. Each trains transformer adapters while keeping the parent weights, text encoder and VAE frozen.

| Parent target | Network type | Apple Silicon (MLX) | CUDA (candle) |
| --- | --- | --- | --- |
| MOSS-Image Base | LoRA | Supported | Supported |
| MOSS-Image Base | LoKr | Supported | Supported |
| MOSS-Image SFT | LoRA | Supported | Supported |
| MOSS-Image SFT | LoKr | Supported | Supported |

The table describes implemented training combinations. Requirements and acceptance evidence for learning, recovery, generation and opposite-backend import are tracked in [epic sc-26584](https://app.shortcut.com/trefry/epic/26584). The table makes no performance or output-quality comparison.

1. Install the selected parent and its required components, then prepare images and captions in **Data Sets**. Base uses its own transformer with the shared SFT text encoder/VAE components; choosing Base does not remove those dependencies.
2. In Training Studio, select the dataset and the exact **Target**, give the adapter a **LoRA name**, and set **Steps**, **Checkpoint cadence** and resolution. Under advanced controls, select **Network type**: LoRA or **LoKr (LyCORIS Kronecker)**. LoKr additionally exposes **LoKr factor** (`-1` selects automatic decomposition).
3. Set **Rank**, **Alpha**, **Batch size**, **Gradient accumulation**, **Optimizer**, **Learning rate** and schedule as needed. MOSS accepts positive fractional alpha and a literal nonnegative **Seed**, including zero; **Randomize seed** saves a concrete value. **MOSS target modules** accepts one exact, case-sensitive native path per line; leave it blank for all supported projections. **Cache VAE latents** and **Cache text embeddings** trade retained cache memory for repeated encoding when disabled. Native VAE preparation uses the deterministic posterior mean.
4. Choose **Start training** to submit a real job. Watch progress and checkpoint saves in the queue. A completed run registers its adapter in the normal library for selection in Image Studio. Select the matching Base or SFT model and adjust adapter strength through the usual LoRA controls.

For API clients, `GET /api/v1/training/targets` supplies current target defaults and limits. The target IDs are `moss_image_base_lora` and `moss_image_sft_lora`; `config.advanced.networkType` chooses `lora` or `lokr`. Submit a dataset, output name and configuration through `POST /api/v1/projects/:project_id/training/jobs`. Use `dryRun: true` for plan validation, then `dryRun: false` for real execution. Training Studio's **Start training** always submits a real run; it has no dry-run selector.

### Checkpoints, cancellation and memory

**Cancel** requests cancellation of the active job. When a complete checkpoint is available, the stopped queue card offers **Resume from step N**; **Restart from step 0** starts over. Resume restores adapter and optimizer state, scheduler position, seed/draw cursor and any pending gradient accumulation. Keep the original dataset, parent, settings and total step horizon: mismatched or incomplete state is refused. A standalone exported adapter is for generation/import, not a substitute for a complete resume bundle. Cancellation does not claim a completed final adapter.

Training uses the configured GPU memory budget and authenticates required model files before loading weights. An over-budget plan is refused rather than silently changing its precision or training controls. Review the refusal and reduce the requested workload or cache residency before submitting a new plan; a dry run validates the plan but does not establish that actual device execution fits.

### Saved adapters and generation

The exported `.safetensors` adapter can be imported through **Import LoRA** and used on MLX or CUDA with the same exact parent checkpoint, method, rank and alpha metadata. Parent identity is preserved through library registration and import: a Base adapter is not advertised as compatible with SFT, or vice versa. Cross-backend portability here concerns the adapter export; it does not promise cross-backend transfer of an in-progress optimizer checkpoint.

Training targets the ordinary Base or SFT model. **8-step acceleration** is a separate SFT generation choice, initially OFF. A custom trained SFT adapter and the selected DMD accelerator have separate roles and remain separately selected; the custom adapter does not replace the accelerator's required components. Base with DMD is refused. Turning acceleration OFF returns to ordinary generation with the selected custom adapter. Disabling that adapter's strength removes its contribution without enabling or changing acceleration.

## Requests and replay

Typed `POST /api/v1/image/jobs`, MCP `generate_image`, presets, retry and duplicate use the same final payload validator, after preset expansion or replay merging. The existing raw `POST /api/v1/jobs` security boundary refuses `image_generate` and `image_edit` for every model. MOSS supports typed text-to-image and explicit `edit_image` routes; `image_detail` remains unsupported.

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

Prompt text is preserved, including multilingual prose, tags and authored JSON. Native prompt batches preserve order; outputs are prompt-major and image-minor. Batch recipes retain images per prompt, while generation-set counts report the total output count. Optional native tensor inputs are also exposed for explicit MOSS edit requests through project NPY uploads and the **Import tensor NPY (optional)** control.

Requested options persist through studio state, saved presets, recipes and image history. PNG workflow exports preserve canonical options and a validated actual execution receipt, replacing any local adapter path with its frozen filename. Imported receipts are display metadata. Generation scrubs supplied or replayed reports and reads a new report only after successful, uncanceled provider execution while holding the serialized generator transaction.

## Assets and admission

The primary Base/SFT transformer and required SFT text encoder/tokenizer/VAE/scheduler snapshot are separate pinned download components. DMD components are optional downloads selected by precision. Their provenance distinguishes bundled official Apache-2.0 licenses from the DMD README's Apache-2.0 metadata, for which no separate LICENSE file is published.

Before loading tensors, the worker authenticates the frozen repository, revision, relative path, physical size and SHA-256 of each selected file. Cache identities include mode, precision, components and content pins. Admission prices the selected execution closure and request workspace; cached unselected optional packages do not increase it. Warm admission credits only the current generator's actual owned residency, bounded by current committed device memory. Normal worker progress, heartbeats, cancellation and artifact writes surround the native provider.

## Historical capability facts capture (sc-26325)

These commands, device settings and measurements describe the earlier sc-26325 delivery. They are retained with their original provenance. Current training acceptance uses [Training acceptance dispatch](#training-acceptance-dispatch) below; current web budgets are defined in [apps/web/bundle-budgets.json](../apps/web/bundle-budgets.json).

The earlier delivery used the following Windows descriptor-capture profile without running GPU or runtime tests:

```sh
gh workflow run windows-candle.yml --ref story/sc-26325-epic-26321-moss-image-evaluation -f capability_facts_only=true
```

This optional input defaults to false. The explicit profile hides CUDA devices, compiles both test suites with `--no-run`, retains the sidecar/memory-adapter checks and Clippy, and runs the native weights-free descriptor dumper with all three strict content comparisons. It uploads the actual `backend-capability-facts-candle` artifact on successful capture or comparison mismatch; a mismatch still fails the job. Mixed provisioning/runtime profiles are refused, and terminal `always()` steps and separate GPU jobs cannot run under this profile. Normal CI and default manual runtime tests remain enabled.

This is CPU descriptor capture, not required CI GPU-test coverage or native model acceptance. The coordinator must still run the normal required checks and separately admitted real product campaign.

## Historical generation-only acceptance (sc-26325)

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

## Historical web bundle evidence (sc-26325)

All initial-load, raw-size and largest-route limits remain enabled and unchanged. Against the clean `875666c7c` baseline, initial JavaScript changes from 474,820 / 146,257 / 124,981 bytes (raw/gzip/Brotli) to 467,260 / 143,621 / 122,981. Sharing disclosure details load through the existing interaction boundary; preference persistence remains on the initial path.

The final reviewed web profile totals 2,897,898 / 862,405 / 736,096 bytes (raw/gzip/Brotli). Its compressed total caps match the measured totals: 862,405 gzip and 736,096 Brotli bytes, with zero remaining headroom. Relative to the previous feature caps of 862,196 and 737,239, gzip increases by 209 bytes and Brotli tightens by 1,143 bytes. The earlier headroom figures of 766 / 1,761 describe a superseded feature snapshot. The raw total cap, initial-load and largest-route caps, coverage checks and negative budget tests remain unchanged.


## Training acceptance dispatch

The existing `memory-catalog-campaign.yml` dispatcher keeps training acceptance isolated from
catalog and the other model profiles. Dispatch the exact reviewed source with `--ref`,
`moss_options_json={"enabled":true,"runner_label":"real-weights","phase":"capture"}`.
The other eligible runner labels are `rw-llm`, `rw-mage`, and `rw-krea`; choose one
with all required cached snapshots. The default `{}` disables MOSS. The four MOSS
options share this one strict object so the dispatcher preserves all delivered FLUX3
inputs within GitHub's 25-input limit; unknown keys, malformed JSON and mixed profiles
refuse before compiler or device setup.
The source is the actual dispatched GitHub SHA. There is no separate `workflowSource` input.

For finalization, use the same `moss_options_json` with `"phase":"finalize"` and put the
existing inspection document in its `inspection` object. Its positive
string `captureRunId` selects the original capture artifacts and is checked again by the
unchanged evidence verifier against the inspection and artifact provenance. There is no
separate `moss_evidence_run_id` input. These two derived values keep the combined dispatcher
within GitHub's 25-input limit without removing either acceptance phase or any other profile.

Training acceptance measures positive native current-process RSS through the protected engine's
existing syscall, with a sampled maximum and the unchanged resolved memory ceiling. All API,
training and generation work is joined in the launched Rust process. The parent binds its PID
and exact binary SHA and requires child exit and stdio closure before continuing. This scope
makes no full-process-tree, metadata-helper absence or kernel lifetime-peak claim. Cancellation,
recovery, infeasible-cap refusal and the original one-hour cell deadline remain mandatory.
