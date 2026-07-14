//! On-device build helper for the Bernini quant matrix (sc-9945, epic 8506).
//!
//! Bernini is a COMPOSITE — a Qwen2.5-VL-7B semantic planner + a dual-expert Wan2.2-A14B renderer —
//! so its tiers pack THREE weight files (the planner LLM backbone + both renderer experts) and leave
//! everything else (the vision tower, connector, clip-diff flow head, T5, VAE) dense, exactly matching
//! the sc-5146 load-time quant policy. Produces the three hosted tier subdirs — `bf16/` + `q8/` +
//! `q4/` — for `SceneWorks/bernini-mlx`.
//!
//! Unlike the Wan tier builders (which convert from a native checkpoint), ALL three Bernini tiers are
//! derived **worker-side from the already-hosted lean bf16 snapshot** — the ~82 GiB self-contained
//! MLX snapshot the turnkey `bernini` / `bernini_image` ids ship today (the diffusers-format
//! `t5_text_encoder`/`t5_tokenizer`/`vae`/`scheduler` dirs the engine never loads are already
//! excluded). This avoids re-downloading the ~168 GB raw `ByteDance/Bernini-Diffusers` package + a
//! base-Wan snapshot, and guarantees the tiers are byte-parity with what ships today:
//!
//! - `bf16/`: a verbatim copy of the lean bf16 snapshot.
//! - `q8/` + `q4/`: derived from the bf16 tier with NO extra convert — the three packable files are
//!   quantized worker-side (group 64), the dense remainder is copied, and the two config sidecars get
//!   the `{bits, group_size}` block the load path reads as authoritative:
//!     * `qwen2_5_vl.safetensors` → `runtime_macos::providers::bernini::convert::quantize_qwen_planner_backbone`
//!       (packs the LLM backbone attention + SwiGLU linears; vision tower + embeddings + norms stay
//!       dense) + patch `qwen2_5_vl_config.json` `quantization`.
//!     * `high_noise_model.safetensors` + `low_noise_model.safetensors` →
//!       `runtime_macos::providers::wan::convert::quantize_wan_transformer` (the same public helper the Wan tiers use) +
//!       patch the renderer `config.json` `quantization`.
//!
//! A resolved tier then loads packed with the config sidecars authoritative and `quant = None` — no
//! install-time convert peak, no dense-staging of the ~56 GB experts.
//!
//! This is an `#[ignore]`d test, not part of CI — it needs the lean bf16 `SceneWorks/bernini-mlx`
//! snapshot on disk and takes minutes per tier on an Apple-Silicon Mac. Run one-off to produce the
//! artifacts, then `hf upload` the `bf16/`/`q8/`/`q4/` subdirs to `SceneWorks/bernini-mlx`.
//!
//! ```sh
//! # The lean bf16 snapshot, e.g. `hf download SceneWorks/bernini-mlx --local-dir <bf16>` (or point
//! # at the cached HF snapshot dir). Auto-resolves the cached turnkey when the var is unset.
//! export SCENEWORKS_BERNINI_BF16_DIR=<bf16-snapshot>
//! export SCENEWORKS_BERNINI_TIER_OUT=<out-root>            # bf16/ q8/ q4/ written here
//! cargo test -p sceneworks-worker --release bernini_build_tiers -- --ignored --nocapture
//! ```
//!
//! Each tier prints a `[[TIER]] {json}` line (tier, dir, diskSizeBytes) so the manifest
//! `estimatedSizeBytes`/`footprint.diskSizeBytes` can be backfilled with the exact hosted sizes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_rs::Array;

/// The runtime files that make a Bernini tier subdir COMPLETE for the load path (mirrors
/// `video_jobs::BERNINI_TIER_FILES`): the planner components + both renderer experts + the shared
/// dense T5/VAE + tokenizer + the config sidecars + the planner's `mllm/tokenizer.json`.
const TIER_FILES: &[&str] = &[
    "qwen2_5_vl.safetensors",
    "qwen2_5_vl_config.json",
    "connector.safetensors",
    "vit_decoder.safetensors",
    "mask_tokens.safetensors",
    "bernini_planner.json",
    "high_noise_model.safetensors",
    "low_noise_model.safetensors",
    "t5_encoder.safetensors",
    "vae.safetensors",
    "tokenizer.json",
    "config.json",
    "bernini_renderer.json",
    "mllm/tokenizer.json",
];

/// The two quantized tiers to derive from the bf16 tier: `(subdir, bits)`.
const QUANT_TIERS: &[(&str, i32)] = &[("q8", 8), ("q4", 4)];

/// The quant group size — the canonical mflux/reference default the load path
/// (`WanModelConfig::from_config_json` / `QwenVlTextConfig::from_config_json`) reconstructs. Matches
/// the Wan A14B renderer experts, so the two halves pack at the same group.
const GROUP_SIZE: i32 = 64;

/// The three weight files a tier PACKS (planner backbone + both renderer experts). Everything else in
/// the snapshot is copied dense.
const PLANNER_WEIGHTS: &str = "qwen2_5_vl.safetensors";
const RENDERER_EXPERTS: &[&str] = &[
    "high_noise_model.safetensors",
    "low_noise_model.safetensors",
];

/// The config sidecars a quant tier patches with the `{bits, group_size}` block the load path reads as
/// authoritative: the planner config gates `Qwen25VlText::from_weights` packed-load; the renderer
/// config gates `WanTransformer::from_weights`.
const PLANNER_CONFIG: &str = "qwen2_5_vl_config.json";
const RENDERER_CONFIG: &str = "config.json";

/// Recursively sum the byte size of every file under `dir` (the on-disk size of a built tier).
fn dir_size_bytes(dir: &Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            total += dir_size_bytes(&path);
        } else if let Ok(meta) = path.metadata() {
            total += meta.len();
        }
    }
    total
}

/// Recursively copy a directory tree (following symlinks to real files, like the HF cache blobs).
fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap_or_else(|e| panic!("mkdir {}: {e:?}", dst.display()));
    for entry in std::fs::read_dir(src)
        .unwrap_or_else(|e| panic!("read_dir {}: {e:?}", src.display()))
        .flatten()
    {
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if std::fs::symlink_metadata(&from)
            .map(|m| m.is_dir())
            .unwrap_or(false)
        {
            copy_tree(&from, &to);
        } else {
            std::fs::copy(&from, &to)
                .unwrap_or_else(|e| panic!("copy {} → {}: {e:?}", from.display(), to.display()));
        }
    }
}

/// Assert every load-path file exists in a built `tier` dir.
fn assert_tier_complete(out_dir: &Path, tier: &str) {
    for file in TIER_FILES {
        assert!(
            out_dir.join(file).is_file(),
            "built {tier} tier is missing {file}"
        );
    }
}

/// Print the machine-readable `[[TIER]]` line for backfilling the manifest sizes.
fn report_tier(tier: &str, out_dir: &Path) {
    let size = dir_size_bytes(out_dir);
    eprintln!(
        "[[TIER]] {{\"tier\":\"{tier}\",\"dir\":\"{}\",\"diskSizeBytes\":{size}}}",
        out_dir.display()
    );
}

/// Load a bf16 safetensors map, apply `quantize` to it, materialize, and save to `out`.
fn quantize_file(
    bf16_file: &Path,
    out_file: &Path,
    quantize: impl FnOnce(
        HashMap<String, Array>,
    ) -> runtime_macos::media::Result<HashMap<String, Array>>,
) {
    let dense: HashMap<String, Array> = Array::load_safetensors(bf16_file)
        .unwrap_or_else(|e| panic!("load {}: {e:?}", bf16_file.display()));
    let packed =
        quantize(dense).unwrap_or_else(|e| panic!("quantize {}: {e:?}", bf16_file.display()));
    mlx_rs::transforms::eval(packed.values().collect::<Vec<_>>())
        .unwrap_or_else(|e| panic!("eval packed {}: {e:?}", out_file.display()));
    Array::save_safetensors(
        packed.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        out_file,
    )
    .unwrap_or_else(|e| panic!("save {}: {e:?}", out_file.display()));
    drop(packed);
    mlx_rs::memory::clear_cache();
}

/// Add the `{"bits", "group_size"}` quantization block to a JSON config in place.
fn patch_config_quant(config_file: &Path, bits: i32) {
    let text = std::fs::read_to_string(config_file)
        .unwrap_or_else(|e| panic!("read {}: {e:?}", config_file.display()));
    let mut config: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("parse {}: {e:?}", config_file.display()));
    config["quantization"] = serde_json::json!({ "bits": bits, "group_size": GROUP_SIZE });
    std::fs::write(config_file, serde_json::to_string_pretty(&config).unwrap())
        .unwrap_or_else(|e| panic!("write {}: {e:?}", config_file.display()));
}

/// Derive a quantized tier (`q8`/`q4`) from the already-built dense `bf16/` tier: copy the whole tier,
/// then overwrite the three packable weight files with their quantized packs and patch the two config
/// sidecars. Byte-identical to what an inline convert with `quantize = Some((bits, GROUP_SIZE))` would
/// emit (the planner packs the same backbone linears `Qwen25VlText::quantize` would; the experts run
/// the same `quantize_wan_transformer` the renderer assembler runs).
fn build_quant_tier(bf16_dir: &Path, out_dir: &Path, tier: &str, bits: i32) {
    // 1. Copy the full bf16 tier (dense remainder: connector, vit_decoder, mask_tokens, T5, VAE,
    //    tokenizer, sidecars, mllm/). The three packable files are overwritten below.
    copy_tree(bf16_dir, out_dir);

    // 2. Planner backbone — pack the LLM attention + SwiGLU linears (vision/embeddings/norms stay
    //    dense) and patch the planner config so `Qwen25VlText::from_weights` loads the packs.
    quantize_file(
        &bf16_dir.join(PLANNER_WEIGHTS),
        &out_dir.join(PLANNER_WEIGHTS),
        |m| {
            runtime_macos::providers::bernini::convert::quantize_qwen_planner_backbone(
                m, bits, GROUP_SIZE,
            )
        },
    );
    patch_config_quant(&out_dir.join(PLANNER_CONFIG), bits);

    // 3. Renderer experts — pack both dual-expert DiTs and patch the renderer config so
    //    `WanTransformer::from_weights` loads them packed.
    for expert in RENDERER_EXPERTS {
        quantize_file(&bf16_dir.join(expert), &out_dir.join(expert), |m| {
            runtime_macos::providers::wan::convert::quantize_wan_transformer(m, bits, GROUP_SIZE)
        });
    }
    patch_config_quant(&out_dir.join(RENDERER_CONFIG), bits);

    assert_tier_complete(out_dir, tier);
    report_tier(tier, out_dir);
}

/// Resolve the source lean bf16 snapshot: the explicit env override, else the cached
/// `SceneWorks/bernini-mlx` turnkey under the HF hub cache. Panics with actionable guidance if none is
/// found (the tiers are derived from it).
fn resolve_bf16_source() -> PathBuf {
    if let Ok(explicit) = std::env::var("SCENEWORKS_BERNINI_BF16_DIR") {
        let path = PathBuf::from(explicit.trim());
        assert!(
            path.join("qwen2_5_vl.safetensors").is_file(),
            "SCENEWORKS_BERNINI_BF16_DIR={} is not a Bernini bf16 snapshot (no qwen2_5_vl.safetensors)",
            path.display()
        );
        return path;
    }
    if let Some(home) = std::env::var_os("HOME") {
        let snapshots = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join("models--SceneWorks--bernini-mlx")
            .join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                if entry.path().join("qwen2_5_vl.safetensors").is_file() {
                    return entry.path();
                }
            }
        }
    }
    panic!(
        "no Bernini bf16 snapshot found — set SCENEWORKS_BERNINI_BF16_DIR or cache the turnkey \
         (hf download SceneWorks/bernini-mlx --local-dir <bf16>)"
    );
}

/// Build all three Bernini tier subdirs (bf16 = copy of the lean snapshot; q8/q4 derived from it).
/// `#[ignore]`d — run on-device with the env vars above; not exercised in CI (needs the ~82 GiB
/// snapshot + minutes/tier).
#[test]
#[ignore = "on-device tier build: needs the lean SceneWorks/bernini-mlx bf16 snapshot + minutes/tier"]
fn bernini_build_tiers() {
    let bf16_source = resolve_bf16_source();
    let out_root = PathBuf::from(
        std::env::var("SCENEWORKS_BERNINI_TIER_OUT")
            .expect("set SCENEWORKS_BERNINI_TIER_OUT to the output root for bf16/ q8/ q4/"),
    );
    std::fs::create_dir_all(&out_root).unwrap();

    // Cap the MLX buffer cache to 0 so freed buffers return to the OS immediately between the heavy
    // per-file loads (sc-5567 batch pattern) — a build-time realloc-cost trade the generation hot path
    // deliberately avoids.
    mlx_rs::memory::set_cache_limit(0);

    // 1. bf16 tier = verbatim copy of the lean snapshot.
    let bf16_dir = out_root.join("bf16");
    eprintln!(
        "building bf16 tier → {} (copy of lean snapshot)",
        bf16_dir.display()
    );
    copy_tree(&bf16_source, &bf16_dir);
    assert_tier_complete(&bf16_dir, "bf16");
    report_tier("bf16", &bf16_dir);

    // 2. q8/q4 derived from the bf16 tier (worker-side quantize of the three packable files).
    for (tier, bits) in QUANT_TIERS {
        let out_dir = out_root.join(tier);
        eprintln!("building {tier} tier → {} (bits={bits})", out_dir.display());
        build_quant_tier(&bf16_dir, &out_dir, tier, *bits);
        mlx_rs::memory::clear_cache();
    }

    eprintln!(
        "done — upload the tier subdirs:\n  hf upload SceneWorks/bernini-mlx {} --include 'bf16/*' 'q8/*' 'q4/*'",
        out_root.display()
    );
}
