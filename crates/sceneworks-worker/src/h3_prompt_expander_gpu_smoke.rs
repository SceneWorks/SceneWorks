//! Real-weight validation of the MiniMax-H3-Image prompt expander on its fully quantized tiers
//! (SW-7202, feature SW-F374): one production `prompt_refine` job with the
//! `minimax_h3_image_expand` task, on the q4 or q8 tier a user picks, on CUDA (Candle layout,
//! `candle-q4-full/` / `candle-q8-full/`) and on MLX (`q4-full/` / `q8-full/`).
//!
//! `SCENEWORKS_H3_EXPANDER_MODEL` picks the expander model (SW-7208): `minimax` (the default,
//! MiniMax-H3-Image-PE from `SceneWorks/minimax-h3-image-pe`) or `abliterated` (the abliterated
//! Qwen3.8-27B the describer uses, from its own `q4-full/` / `q8-full/` folder of
//! `SceneWorks/qwen3.8-27b-abliterated-{mlx,candle}`). Both read the same MiniMax prompt files and
//! must meet the same output contract; the abliterated job carries the describer's entry with the
//! expander's `prompt` co-requisite added, exactly as the API builds it, and its result must name
//! the abliterated model as the one that ran.
//!
//! Runs ONLY from the dispatch-only `qwen38-describer-validation.yml` workflow with `job=expander`
//! (CUDA under the exact-UUID guard `scripts/ci/windows_cuda_execution.py`; MLX through
//! `qwen38-describer-mlx-validation.yml` under the external phys_footprint guard). It is never part
//! of ordinary PR or push CI.
//!
//! The job is the one a user's expand request runs: [`crate::prompt_refine_jobs::run_prompt_refine_job`]
//! with the shipped catalog entry as `modelManifestEntry` (every platform's rows, as the API forwards
//! it) and the tier as `peTier`, against a local stub of the job API. The worker resolves the tier
//! through the pinned-revision snapshot path (`huggingface_pinned_snapshot_dir`), reads the
//! upstream MiniMax `prompt/` files from `MiniMaxAI/MiniMax-H3-Image-PE` at its pinned revision and
//! verifies them against `prompt/manifest.json`, generates, and parses the reply strictly. The smoke
//! then requires:
//!
//! * the job LOADED the catalog's pinned folder for that tier on this host's layout: the result's
//!   `executionIdentity.weightsDir` (the `LoadSpec` source the job handed the loader) is that
//!   folder at the pinned revision;
//! * a completed job whose expanded prompt is non-empty, differs from the input, carries no
//!   reasoning block, and was not cut off (`finishReason` `stop`, under the budget). A reply that
//!   stops at its budget (`length`) is a real product failure — the user gets no expansion — so it
//!   fails here rather than being retried with a larger budget;
//! * a proposed aspect ratio that is one of the recipe's own ratios (`request_params.json`), the
//!   text-to-image workflow, and the requested tier named on the suggestion block.
//!
//! The expander repository's other tier folders present in the snapshot are recorded
//! (`other_tiers_present`, `tier_isolated`). Run against a fresh run-scoped hub
//! (`fresh_hub: true` on the workflow), which holds only the selected tier and the prompt files,
//! `tier_isolated=true` shows the job needed nothing else.
//!
//! SW-7204: the job names the installed block draft (DFlash2 / DSpine, provisioned with the tier)
//! and its result's `executionIdentity.decode` records the proposer, draft, precision, target
//! passes, rows checked and accepted tokens per pass. The expansion sends `speculative: auto`,
//! checked linearly. `SCENEWORKS_QWEN38_BLOCK_DRAFT=on` requires the draft to load and be admitted,
//! checked linearly; where `auto` prefers it (MLX q4/q8, SW-7200) it must run (block draft, the
//! tier's draft precision, more than one accepted token per pass), and elsewhere `auto` must keep
//! its MTP head; `no_fit` caps the engine's load budget so the draft
//! cannot fit beside the target and requires the job to complete without it, its record saying
//! why. The default `record` records it and asserts nothing about it.
//!
//! Fail-loud: the tier, the prompt files and the cache must all be present. Nothing is downloaded
//! (`qwen38-describer-provision.yml` with `tiers: expander-q4|expander-q8`, or the validation
//! workflow's `fresh_hub`, fills the CUDA cache; the MLX run script fetches the selected tier).
//!
//! ```text
//! $env:HF_HUB_CACHE = "E:\huggingface\hub"
//! $env:SCENEWORKS_H3_EXPANDER_TIER = "q4"
//! $env:SCENEWORKS_H3_EXPANDER_MODEL = "abliterated"   # or "minimax" (the default)
//! cargo test -p sceneworks-worker --features backend-candle --release --lib h3_prompt_expander_gpu_smoke -- --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
#[cfg(not(target_os = "macos"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_os = "macos"))]
use std::sync::Arc;

use serde_json::{json, Value};

use super::*;
use crate::minimax_h3_image_prompt_expand as expand;

const TIER_ENV: &str = "SCENEWORKS_H3_EXPANDER_TIER";
const MODEL_ENV: &str = "SCENEWORKS_H3_EXPANDER_MODEL";
const JOB_ID: &str = "sw-7202-h3-expander-smoke";
const WORKER_ID: &str = "h3-expander-smoke-worker";
const STAMP: &str = "2026-11-19T00:00:00Z";
/// The expander model under test and where its tiers live.
#[derive(Clone, Copy, Debug)]
struct Expander {
    /// The workflow's `expander_model` input: `minimax` or `abliterated`.
    input: &'static str,
    /// The catalog id the job runs and its result must name.
    catalog_id: &'static str,
    /// The repository this host's tier rows load from.
    tier_repo: &'static str,
}

/// `$SCENEWORKS_H3_EXPANDER_MODEL`: `minimax` (default) or `abliterated`.
fn requested_expander() -> Expander {
    let model = std::env::var(MODEL_ENV).unwrap_or_default();
    match model.trim() {
        "" | "minimax" => Expander {
            input: "minimax",
            catalog_id: expand::PE_MODEL_ID,
            tier_repo: "SceneWorks/minimax-h3-image-pe",
        },
        "abliterated" => Expander {
            input: "abliterated",
            catalog_id: expand::DESCRIBER_MODEL_ID,
            tier_repo: if cfg!(target_os = "macos") {
                sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_MLX_REPO
            } else {
                sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_CANDLE_REPO
            },
        },
        other => panic!("${MODEL_ENV} must be minimax or abliterated, got {other:?}"),
    }
}
/// A short text-to-image prompt: the expander's own use case.
const PROMPT: &str = "a lighthouse on a rocky coast at dusk, waves breaking below";

/// `$SCENEWORKS_H3_EXPANDER_TIER`: one of the two published converted tiers.
fn requested_tier() -> &'static str {
    let tier = std::env::var(TIER_ENV).unwrap_or_else(|_| panic!("set ${TIER_ENV} to q4 or q8"));
    match tier.trim() {
        "q4" => "q4",
        "q8" => "q8",
        other => panic!("${TIER_ENV} must be q4 or q8, got {other:?}"),
    }
}

/// One pinned snapshot folder, relative to the hub cache.
#[derive(Debug)]
struct Pinned {
    repo: String,
    revision: String,
    relative_dir: PathBuf,
}

fn relative_snapshot(repo: &str, revision: &str) -> PathBuf {
    PathBuf::from(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(revision)
}

/// The entry the API forwards for `expander`: the expander's own, or the describer's with the
/// expander's `describer_prompt_expansion` prompt files added as a co-requisite (every platform).
fn job_entry(expander: Expander) -> Value {
    let pe = Value::Object(crate::test_env::shipped_catalog_entry(expand::PE_MODEL_ID));
    if expander.catalog_id == expand::PE_MODEL_ID {
        return pe;
    }
    let mut entry = Value::Object(crate::test_env::shipped_catalog_entry(expander.catalog_id));
    let prompts = sceneworks_core::model_usage_policy::conditional_component_downloads(
        &pe,
        sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPAND_PROMPTS_PURPOSE,
    )
    .expect("the expander declares its prompt files for the describer");
    entry["downloads"]
        .as_array_mut()
        .expect("the describer declares download rows")
        .extend(prompts);
    entry
}

/// [`job_entry`] with only this host's rows (MLX layout on macOS, Candle elsewhere).
fn host_entry(expander: Expander) -> Value {
    let mut entry = job_entry(expander);
    sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os(
        &mut entry,
        std::env::consts::OS,
    );
    entry
}

fn host_rows(entry: &Value) -> Vec<&Value> {
    entry["downloads"]
        .as_array()
        .expect("the expander declares download rows")
        .iter()
        .collect()
}

/// The one folder a converted row's files name (`candle-q4-full/*` ⇒ `candle-q4-full`; the
/// describer's `q4-full/config.json`, `q4-full/model.safetensors`, ... ⇒ `q4-full`).
fn row_folder(row: &Value) -> String {
    let folders: std::collections::BTreeSet<&str> = row["files"]
        .as_array()
        .expect("the row names its files")
        .iter()
        .map(|file| {
            let file = file.as_str().expect("file selectors are strings");
            file.split_once('/')
                .unwrap_or_else(|| {
                    panic!("a converted tier row names files in a folder, got {file}")
                })
                .0
        })
        .collect();
    assert_eq!(
        folders.len(),
        1,
        "a converted tier row names one folder: {folders:?}"
    );
    folders.into_iter().next().unwrap().to_owned()
}

/// The catalog's pinned folder of `tier` on this host's layout.
fn pinned_tier(entry: &Value, expander: Expander, tier: &str) -> Pinned {
    let tier_repo = expander.tier_repo;
    let rows: Vec<&Value> = host_rows(entry)
        .into_iter()
        .filter(|row| row["repo"] == tier_repo && row["variant"] == tier)
        .collect();
    assert_eq!(
        rows.len(),
        1,
        "the catalog has one {tier_repo} {tier} row for {}",
        std::env::consts::OS
    );
    let row = rows[0];
    assert!(
        row.get("pendingArtifact").is_none(),
        "the {tier} row is published"
    );
    let revision = row["revision"]
        .as_str()
        .expect("pinned revision")
        .to_owned();
    Pinned {
        repo: tier_repo.to_owned(),
        relative_dir: relative_snapshot(tier_repo, &revision).join(row_folder(row)),
        revision,
    }
}

/// Every other published tier folder of the converted repository on this host's layout.
fn other_tier_folders(entry: &Value, expander: Expander, tier: &str) -> Vec<String> {
    host_rows(entry)
        .into_iter()
        .filter(|row| {
            row["repo"] == expander.tier_repo
                && row["variant"] != tier
                && row.get("pendingArtifact").is_none()
        })
        .map(row_folder)
        .collect()
}

/// The `prompt` co-requisite's pinned upstream snapshot.
fn pinned_prompt(entry: &Value) -> Pinned {
    let row = host_rows(entry)
        .into_iter()
        .find(|row| row["coRequisite"] == true && row["componentId"] == "prompt")
        .expect("the catalog declares the prompt co-requisite");
    let repo = row["repo"].as_str().expect("prompt repo").to_owned();
    let revision = row["revision"]
        .as_str()
        .expect("prompt revision")
        .to_owned();
    assert_eq!(repo, gen_core::core_llm::minimax_h3_image_pe::REPO);
    assert_eq!(revision, gen_core::core_llm::minimax_h3_image_pe::REVISION);
    Pinned {
        relative_dir: relative_snapshot(&repo, &revision),
        repo,
        revision,
    }
}

/// What the smoke measured outside the job, for the RESULT line and the evidence.
struct Run {
    expander: Expander,
    tier: &'static str,
    weights_dir: PathBuf,
    pinned: Pinned,
    prompt: Pinned,
    /// Other tier folders present beside the selected one in the pinned snapshot.
    other_tiers_present: Vec<String>,
    result: Value,
    wall_seconds: f64,
    /// The block-draft check this run made (SW-7204).
    draft_check: crate::qwen38_block_draft::SmokeDraftCheck,
}

/// Run one expand job with `expander` on `tier` through the production entry point and check its
/// result against the expander's output contract.
async fn expand_on_tier(expander: Expander, tier: &'static str) -> Run {
    let hf_hub_cache = PathBuf::from(
        std::env::var("HF_HUB_CACHE").expect("set $HF_HUB_CACHE to the real-weights hub"),
    );
    assert!(
        hf_hub_cache.is_dir(),
        "HF_HUB_CACHE is not a directory: {}",
        hf_hub_cache.display()
    );
    let entry = job_entry(expander);
    let host = host_entry(expander);
    let pinned = pinned_tier(&host, expander, tier);
    let prompt = pinned_prompt(&host);

    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let (api_url, posts) = crate::test_api_stub::smoke_api(
        crate::test_api_stub::smoke_job("", "prompt_refine", None, None, WORKER_ID, STAMP),
        crate::test_api_stub::smoke_worker("", JOB_ID, STAMP),
        None,
    )
    .await;
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.api_url = api_url;
    settings.worker_id = WORKER_ID.to_owned();

    // The resolution the job makes below, read first so a wrong folder fails before any load.
    let entry_object = entry.as_object().expect("catalog entry object");
    let dirs = expand::resolve_dirs(&settings, entry_object, Some(tier), None)
        .unwrap_or_else(|error| panic!("the {tier} tier resolves from HF_HUB_CACHE: {error}"));
    assert_eq!(
        dirs.tier,
        Some(tier),
        "an explicit tier is never substituted"
    );
    assert_eq!(
        dirs.model_id, expander.catalog_id,
        "the chosen model resolves"
    );
    assert!(
        dirs.weights.ends_with(&pinned.relative_dir),
        "the {tier} tier resolved to {} instead of the pinned folder {}",
        dirs.weights.display(),
        pinned.relative_dir.display()
    );
    assert!(
        dirs.prompt_root.ends_with(&prompt.relative_dir),
        "the prompt files resolved to {} instead of the pinned upstream snapshot {}",
        dirs.prompt_root.display(),
        prompt.relative_dir.display()
    );
    let snapshot = dirs
        .weights
        .parent()
        .expect("a tier folder sits in its snapshot");
    let other_tiers_present: Vec<String> = other_tier_folders(&host, expander, tier)
        .into_iter()
        .filter(|folder| snapshot.join(folder).exists())
        .collect();
    // The recipe's own ratios: the verified prompt files' `request_params.json`.
    let allowed_ratios = expand::load_prompt_files(&dirs.prompt_root)
        .expect("the pinned prompt files verify against prompt/manifest.json")
        .params()
        .aspect_ratios
        .clone();

    let mut snapshot_job =
        crate::test_api_stub::smoke_job(JOB_ID, "prompt_refine", None, None, WORKER_ID, STAMP);
    snapshot_job["payload"] = json!({
        "task": sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPAND_TASK,
        "prompt": PROMPT,
        "modelId": sceneworks_core::contracts::MINIMAX_H3_IMAGE_MODEL_ID,
        (sceneworks_core::contracts::MINIMAX_H3_IMAGE_PE_TIER_KEY): tier,
        (sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPANDER_MODEL_KEY): expander.catalog_id,
        "modelManifestEntry": entry,
    });
    let job: JobSnapshot = serde_json::from_value(snapshot_job).expect("smoke job deserializes");
    // SW-7204: the block draft the job will name, and (`no_fit`) the load-budget cap held for it.
    let draft_check = crate::qwen38_block_draft::SmokeDraftCheck::from_env();
    let draft = draft_check.installed_draft(&settings.data_dir, expander.catalog_id, Some(tier));
    let _no_fit_cap = match (draft_check, &draft) {
        (crate::qwen38_block_draft::SmokeDraftCheck::NoFit, Some(draft)) => {
            Some(crate::qwen38_block_draft::force_draft_no_fit(&dirs.weights, draft).await)
        }
        _ => None,
    };
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    let outcome = crate::prompt_refine_jobs::run_prompt_refine_job(&api, &settings, &job).await;
    let wall_seconds = started.elapsed().as_secs_f64();
    let posts = posts.lock().expect("progress posts lock").clone();
    if let Err(error) = &outcome {
        panic!("the {tier} expand job failed after {wall_seconds:.1} s: {error}");
    }
    let result = posts
        .iter()
        .rev()
        .find(|(id, body)| id == JOB_ID && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .unwrap_or_else(|| panic!("the job posted no completed result: {posts:#?}"));
    eprintln!(
        "job result:\n{}",
        serde_json::to_string_pretty(&result).unwrap_or_default()
    );

    // The output contract, on what the job published.
    let expanded = result["refinedPrompt"]
        .as_str()
        .expect("the result carries the expanded prompt");
    assert!(!expanded.trim().is_empty(), "the expanded prompt is empty");
    assert_ne!(expanded.trim(), PROMPT, "the prompt came back unexpanded");
    assert!(
        expanded.len() > PROMPT.len(),
        "the expansion is shorter than the prompt: {expanded:?}"
    );
    assert!(
        !expanded.contains("<think>") && !expanded.contains("</think>"),
        "the reasoning block leaked into the expansion: {expanded:?}"
    );
    assert_eq!(result["originalPrompt"], PROMPT, "the original is kept");
    let generation = &result["generation"];
    assert_eq!(
        generation["finishReason"], "stop",
        "the reply was not completed: {generation}"
    );
    let generated = generation["usage"]["generatedTokens"].as_u64();
    let budget = generation["maxNewTokens"].as_u64();
    assert!(
        matches!((generated, budget), (Some(g), Some(b)) if g > 0 && g < b),
        "the reply ran to its budget: {generation}"
    );
    let suggestion = &result["rewriteSuggestion"];
    let ratio = suggestion["whRatio"].as_str().unwrap_or_default();
    assert!(
        gen_core::core_llm::minimax_h3_image_pe::is_ratio(ratio)
            && allowed_ratios.iter().any(|allowed| allowed == ratio),
        "the proposed ratio {ratio:?} is not one of the recipe's {allowed_ratios:?}"
    );
    assert_eq!(suggestion["workflow"], "t2i");
    assert_eq!(
        suggestion["rewriterModelId"], expander.catalog_id,
        "the suggestion names the model that ran"
    );
    assert_eq!(
        result["executionIdentity"]["expanderModelId"], expander.catalog_id,
        "the job record names the model that ran"
    );
    assert_eq!(
        suggestion["tier"], tier,
        "the suggestion block names the requested tier"
    );
    // What the job actually loaded: the `LoadSpec` source it handed the loader.
    let loaded = PathBuf::from(
        result["executionIdentity"]["weightsDir"]
            .as_str()
            .expect("the expansion result records the weights folder it loaded"),
    );
    assert!(
        loaded.ends_with(&pinned.relative_dir),
        "the job loaded {} instead of the pinned {tier} folder {}",
        loaded.display(),
        pinned.relative_dir.display()
    );
    assert_eq!(
        loaded, dirs.weights,
        "the job loaded the folder it resolved"
    );
    let decode = &result["executionIdentity"]["decode"];
    eprintln!("DECODE block_draft_check={} {decode}", draft_check.label());
    let auto_runs_draft = draft
        .as_ref()
        .is_some_and(|draft| crate::qwen38_block_draft::auto_runs_draft(draft, tier));
    draft_check
        .check(decode, draft.as_ref(), auto_runs_draft)
        .unwrap_or_else(|error| panic!("block-draft check {}: {error}", draft_check.label()));

    Run {
        expander,
        tier,
        weights_dir: loaded,
        pinned,
        prompt,
        other_tiers_present,
        result,
        wall_seconds,
        draft_check,
    }
}

/// The evidence file the workflow validates and uploads.
fn write_result(run: &Run, backend: &str, memory: Value) -> String {
    let test_fqn = std::thread::current()
        .name()
        .expect("the libtest case has an actual FQN")
        .to_owned();
    if let Some(output_dir) = std::env::var_os("MODEL_PROOF_OUTPUT_DIR") {
        let output_dir = PathBuf::from(output_dir);
        assert!(output_dir.is_absolute() && output_dir.is_dir());
        let evidence = json!({
            "schema_version": 1,
            "job": "expander",
            "expander_model": run.expander.input,
            "catalog_id": run.expander.catalog_id,
            "rewriter_model_id": run.result["rewriteSuggestion"]["rewriterModelId"],
            "test_fqn": test_fqn,
            "backend": backend,
            "repo": run.pinned.repo,
            "revision": run.pinned.revision,
            "tier": run.tier,
            "weights_dir": run.weights_dir,
            "other_tiers_present": run.other_tiers_present,
            "tier_isolated": run.other_tiers_present.is_empty(),
            "prompt_files": { "repo": run.prompt.repo, "revision": run.prompt.revision },
            "prompt": PROMPT,
            "expanded_prompt": run.result["refinedPrompt"],
            "wh_ratio": run.result["rewriteSuggestion"]["whRatio"],
            "workflow": run.result["rewriteSuggestion"]["workflow"],
            "suggestion_tier": run.result["rewriteSuggestion"]["tier"],
            "generation": run.result["generation"],
            "block_draft": {
                "check": run.draft_check.label(),
                "decode": run.result["executionIdentity"]["decode"],
            },
            "wall_seconds": run.wall_seconds,
            "wall_time_scope": "one prompt_refine job on a cold process: model load plus one expansion",
            "memory": memory,
        });
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_dir.join("expander-result.json"))
            .expect("create the owned expander result, never replace an existing file");
        serde_json::to_writer_pretty(&mut file, &evidence).expect("persist the expander result");
    }
    test_fqn
}

fn result_line(run: &Run, backend: &str, memory: &str) -> String {
    format!(
        "RESULT model={} job=expander backend={backend} tier={} {memory} \
         wall_s={:.1} generated_tokens={} wh_ratio={} tier_isolated={} other_tiers_present={:?} \
         loaded_weights={}",
        run.expander.catalog_id,
        run.tier,
        run.wall_seconds,
        run.result["generation"]["usage"]["generatedTokens"],
        run.result["rewriteSuggestion"]["whRatio"],
        run.other_tiers_present.is_empty(),
        run.other_tiers_present,
        run.weights_dir.display(),
    )
}

#[cfg(not(target_os = "macos"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight prompt-expander GPU validation; needs a pinned Candle tier of the selected expander model and the upstream MiniMax prompt files under HF_HUB_CACHE and a CUDA device"]
async fn h3_prompt_expander_expands_a_prompt_on_cuda() {
    let tier = requested_tier();
    let expander = requested_expander();
    // The same card reading as the describer smoke: the job's card by its nvidia-smi
    // (PCI-bus-ordered) index, which under `CUDA_VISIBLE_DEVICES` is the visible device's index.
    let gpu_index = std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .and_then(|devices| devices.split(',').next().map(|d| d.trim().to_owned()))
        .filter(|device| !device.is_empty())
        .unwrap_or_else(|| "0".to_owned());
    let baseline_mib = crate::iris_gpu_smoke::nvidia_memory_used_mib(&gpu_index)
        .expect("nvidia-smi memory.used reads");
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let stop = Arc::clone(&stop);
        let index = gpu_index.clone();
        std::thread::spawn(move || {
            let mut peak = 0_u64;
            while !stop.load(Ordering::Relaxed) {
                if let Some(used) = crate::iris_gpu_smoke::nvidia_memory_used_mib(&index) {
                    peak = peak.max(used);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            peak
        })
    };
    // A panic inside the job still stops the sampler: the guard then sees this process exit.
    let run = tokio::spawn(expand_on_tier(expander, tier)).await;
    stop.store(true, Ordering::Relaxed);
    let peak_mib = sampler.join().expect("sampler thread joins");
    let run = run.unwrap_or_else(|error| std::panic::resume_unwind(error.into_panic()));
    let delta_mib = peak_mib.saturating_sub(baseline_mib);
    eprintln!(
        "{}",
        result_line(
            &run,
            "candle",
            &format!(
                "gpu={gpu_index} baseline_mib={baseline_mib} peak_mib={peak_mib} \
                 peak_delta_mib={delta_mib}"
            )
        )
    );
    let test_fqn = write_result(
        &run,
        "candle",
        json!({
            "gpu_index": gpu_index,
            "baseline_mib": baseline_mib,
            "peak_mib": peak_mib,
            "peak_delta_mib": delta_mib,
            "method": "nvidia-smi memory.used on the job's card, polled every 100 ms; card-wide high-water"
        }),
    );
    // The CUDA guard refuses a card above 5% utilisation as soon as this process exits, and
    // `nvidia-smi` averages over the last second: stay idle until the average decays (as the
    // describer smoke does).
    std::thread::sleep(std::time::Duration::from_secs(5));
    eprintln!("MODEL_INFERENCE_BODY_COMPLETED {test_fqn}");
}

/// The MLX twin: the same job and checks on macOS. Memory is measured outside the process by
/// `tree_guard.py` (`root_lifetime_max`), which `guarded_run.sh` records.
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight prompt-expander MLX validation; needs a pinned MLX tier of the selected expander model and the upstream MiniMax prompt files under HF_HUB_CACHE"]
async fn h3_prompt_expander_expands_a_prompt_on_mlx() {
    let tier = requested_tier();
    let run = expand_on_tier(requested_expander(), tier).await;
    eprintln!(
        "{}",
        result_line(&run, "mlx", "memory=tree_guard_root_lifetime_max")
    );
    let test_fqn = write_result(
        &run,
        "mlx",
        json!("measured outside the process by tree_guard.py (root_lifetime_max)"),
    );
    eprintln!("MODEL_INFERENCE_BODY_COMPLETED {test_fqn}");
}
