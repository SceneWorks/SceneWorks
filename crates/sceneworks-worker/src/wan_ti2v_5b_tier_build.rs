//! On-device build helper for the Wan2.2 TI2V-5B quant matrix (sc-9941, epic 8506).
//!
//! The single-expert sibling of `wan_t2v_14b_tier_build` / `wan_i2v_14b_tier_build`. Produces the
//! three hosted tier subdirs — `bf16/` + `q8/` + `q4/` — for `SceneWorks/wan2.2-ti2v-5b-mlx`:
//!
//! - `bf16/`: the dense converter output. Drives the SAME byte-parity-validated converter the turnkey
//!   uses (`runtime_macos::providers::wan::convert::convert_ti2v_5b`) → one transformer (`model.safetensors`) + the UMT5
//!   T5 encoder + the z16 VAE + `config.json`; this helper then copies the `tokenizer.json` the
//!   converter does not emit.
//! - `q8/` + `q4/`: derived **worker-side** from the bf16 tier with NO mlx-gen change — load the bf16
//!   `model.safetensors`, quantize the DiT Linears via the public
//!   `runtime_macos::providers::wan::convert::quantize_wan_transformer` (group 64, the canonical Wan group), save, reuse
//!   the shared dense T5/VAE/tokenizer, and patch `config.json` with the `{bits, group_size}` block.
//!   This is byte-identical to what an inline `convert_ti2v_5b(quantize)` would emit (the converter's
//!   transformer path is exactly `sanitize → cast bf16 → quantize_wan_transformer`, and the config
//!   patch is what `WanModelConfig::to_json()` writes), so a resolved tier loads packed with
//!   `config.json` authoritative and no install-time convert peak.
//!
//! This is an `#[ignore]`d test, not part of CI — it needs the native `Wan-AI/Wan2.2-TI2V-5B`
//! checkpoint on disk (transformer `diffusion_pytorch_model-*.safetensors` shards +
//! `models_t5_umt5-xxl-enc-bf16.pth` + `Wan2.2_VAE.pth`) and takes minutes per tier on an
//! Apple-Silicon Mac. Run one-off to produce the artifacts, then `hf upload` the `bf16/`/`q8/`/`q4/`
//! subdirs to `SceneWorks/wan2.2-ti2v-5b-mlx`.
//!
//! ```sh
//! # Native checkpoint, e.g. `hf download Wan-AI/Wan2.2-TI2V-5B --local-dir <native>`:
//! export SCENEWORKS_WAN_TI2V_5B_NATIVE_DIR=<native>
//! export SCENEWORKS_WAN_TI2V_5B_TIER_OUT=<out-root>          # bf16/ q8/ q4/ written here
//! # tokenizer.json to copy into each tier (the converter does not emit it). Defaults to
//! # <native>/tokenizer.json, then <native>/google/umt5-xxl/tokenizer.json, then the cached
//! # SceneWorks/wan2.2-ti2v-5b-mlx turnkey's tokenizer.json.
//! export SCENEWORKS_WAN_TI2V_5B_TOKENIZER=<path/to/tokenizer.json>   # optional
//! cargo test -p sceneworks-worker --release wan_ti2v_5b_build_tiers -- --ignored --nocapture
//! ```
//!
//! Each tier prints a `[[TIER]] {json}` line (tier, dir, diskSizeBytes) so the manifest
//! `estimatedSizeBytes`/`footprint.diskSizeBytes` can be backfilled with the exact hosted sizes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_rs::Array;

/// The five files that make a TI2V-5B tier subdir COMPLETE for the load path
/// (`video_jobs::wan_tier_is_complete` with `WAN_TI2V_5B_TIER_FILES`): one transformer + the shared
/// dense T5/VAE + tokenizer + `config.json`.
const TIER_FILES: &[&str] = &[
    "model.safetensors",
    "t5_encoder.safetensors",
    "vae.safetensors",
    "tokenizer.json",
    "config.json",
];

/// The two quantized tiers to derive from the bf16 output: `(subdir, bits)`. Group 64 (the canonical
/// Wan/mflux default the load path `WanModelConfig::from_config_json` assumes and reconstructs).
const QUANT_TIERS: &[(&str, i32)] = &[("q8", 8), ("q4", 4)];

/// The Wan transformer quant group size (mflux/reference default; matches the A14B tiers).
const GROUP_SIZE: i32 = 64;

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

/// Resolve the `tokenizer.json` to copy into each tier: the explicit env override, then
/// `<native>/tokenizer.json`, then the native `google/umt5-xxl/tokenizer.json` (where the upstream
/// checkpoint ships the UMT5 tokenizer), then the cached `SceneWorks/wan2.2-ti2v-5b-mlx` turnkey's
/// `tokenizer.json` under the HF hub cache. Panics with actionable guidance if none is found — the
/// tier is incomplete without it.
fn resolve_tokenizer(native_dir: &Path) -> PathBuf {
    if let Ok(explicit) = std::env::var("SCENEWORKS_WAN_TI2V_5B_TOKENIZER") {
        let path = PathBuf::from(explicit.trim());
        assert!(
            path.is_file(),
            "SCENEWORKS_WAN_TI2V_5B_TOKENIZER={} is not a file",
            path.display()
        );
        return path;
    }
    let beside = native_dir.join("tokenizer.json");
    if beside.is_file() {
        return beside;
    }
    let native_umt5 = native_dir.join("google/umt5-xxl/tokenizer.json");
    if native_umt5.is_file() {
        return native_umt5;
    }
    if let Some(home) = std::env::var_os("HOME") {
        let snapshots = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join("models--SceneWorks--wan2.2-ti2v-5b-mlx")
            .join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                let candidate = entry.path().join("tokenizer.json");
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
    }
    panic!(
        "no tokenizer.json found — set SCENEWORKS_WAN_TI2V_5B_TOKENIZER, place one at {}, or cache \
         the SceneWorks/wan2.2-ti2v-5b-mlx turnkey (hf download … --include tokenizer.json)",
        beside.display()
    );
}

/// Assert the five load-path files exist in a built `tier` dir.
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

/// Derive a quantized tier (`q8`/`q4`) from the already-built dense `bf16/` tier: quantize the DiT
/// transformer worker-side, reuse the shared dense T5/VAE/tokenizer, and patch `config.json` with the
/// `{bits, group_size}` block the load path reads as authoritative. Byte-identical to an inline
/// `convert_ti2v_5b(Some((bits, GROUP_SIZE)))`.
fn build_quant_tier(bf16_dir: &Path, out_dir: &Path, tier: &str, bits: i32) {
    std::fs::create_dir_all(out_dir).unwrap();

    // 1. Transformer — load the dense bf16 `model.safetensors`, quantize the same Linear set the
    //    converter's `convert_expert` would (attn q/k/v/o + FFN fc1/fc2), and save the packed triple.
    let dense: HashMap<String, Array> = Array::load_safetensors(bf16_dir.join("model.safetensors"))
        .unwrap_or_else(|e| panic!("load bf16 model.safetensors for {tier}: {e:?}"));
    let packed =
        runtime_macos::providers::wan::convert::quantize_wan_transformer(dense, bits, GROUP_SIZE)
            .unwrap_or_else(|e| panic!("quantize_wan_transformer {tier}: {e:?}"));
    // Materialize before writing (lazy MLX graph, mirrors mlx-gen-wan's save_map).
    mlx_rs::transforms::eval(packed.values().collect::<Vec<_>>())
        .unwrap_or_else(|e| panic!("eval packed {tier}: {e:?}"));
    Array::save_safetensors(
        packed.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        out_dir.join("model.safetensors"),
    )
    .unwrap_or_else(|e| panic!("save packed model.safetensors for {tier}: {e:?}"));
    drop(packed);
    mlx_rs::memory::clear_cache();

    // 2. Shared dense components — the T5/VAE/tokenizer are identical across tiers (only the DiT is
    //    quantized), so copy them from the bf16 tier rather than re-converting the heavy .pth sources.
    for file in [
        "t5_encoder.safetensors",
        "vae.safetensors",
        "tokenizer.json",
    ] {
        std::fs::copy(bf16_dir.join(file), out_dir.join(file))
            .unwrap_or_else(|e| panic!("copy {file} into {tier}: {e:?}"));
    }

    // 3. config.json — take the dense config and add the quantization manifest the load path reads
    //    (`{"bits": bits, "group_size": 64}`, exactly what `WanModelConfig::to_json()` emits).
    let text = std::fs::read_to_string(bf16_dir.join("config.json"))
        .unwrap_or_else(|e| panic!("read bf16 config.json for {tier}: {e:?}"));
    let mut config: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse bf16 config.json: {e:?}"));
    config["quantization"] = serde_json::json!({ "bits": bits, "group_size": GROUP_SIZE });
    std::fs::write(
        out_dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap_or_else(|e| panic!("write {tier} config.json: {e:?}"));

    assert_tier_complete(out_dir, tier);
    report_tier(tier, out_dir);
}

/// Build all three Wan2.2 TI2V-5B tier subdirs (bf16 via the converter, q8/q4 derived from it).
/// `#[ignore]`d — run on-device with the env vars above; not exercised in CI (needs the native
/// weights).
#[test]
#[ignore = "on-device tier build: needs the native Wan2.2-TI2V-5B checkpoint + minutes/tier"]
fn wan_ti2v_5b_build_tiers() {
    let native_dir = PathBuf::from(
        std::env::var("SCENEWORKS_WAN_TI2V_5B_NATIVE_DIR")
            .expect("set SCENEWORKS_WAN_TI2V_5B_NATIVE_DIR to the native checkpoint dir"),
    );
    assert!(
        native_dir.join("models_t5_umt5-xxl-enc-bf16.pth").is_file(),
        "{} is not a native Wan2.2 TI2V-5B checkpoint (expected the transformer \
         diffusion_pytorch_model-*.safetensors shards, models_t5_umt5-xxl-enc-bf16.pth, \
         Wan2.2_VAE.pth)",
        native_dir.display()
    );
    let out_root = PathBuf::from(
        std::env::var("SCENEWORKS_WAN_TI2V_5B_TIER_OUT")
            .expect("set SCENEWORKS_WAN_TI2V_5B_TIER_OUT to the output root for bf16/ q8/ q4/"),
    );
    let tokenizer = resolve_tokenizer(&native_dir);
    std::fs::create_dir_all(&out_root).unwrap();

    // The converter loads the T5 (.pth) and transformer as full buffers; at MLX's default cache limit
    // the freed buffers are RETAINED as cache. Cap the buffer cache to 0 so freed buffers return to
    // the OS immediately between stages (sc-5567 batch pattern) — a build-time realloc-cost trade the
    // generation hot path deliberately avoids.
    mlx_rs::memory::set_cache_limit(0);

    // 1. Dense bf16 tier via the byte-parity-validated converter, then copy the tokenizer it omits.
    let bf16_dir = out_root.join("bf16");
    eprintln!("building bf16 tier → {}", bf16_dir.display());
    runtime_macos::providers::wan::convert::convert_ti2v_5b(&native_dir, &bf16_dir)
        .unwrap_or_else(|e| panic!("convert_ti2v_5b bf16 failed: {e:?}"));
    mlx_rs::memory::clear_cache();
    std::fs::copy(&tokenizer, bf16_dir.join("tokenizer.json"))
        .unwrap_or_else(|e| panic!("copy tokenizer.json into bf16 failed: {e:?}"));
    assert_tier_complete(&bf16_dir, "bf16");
    report_tier("bf16", &bf16_dir);

    // 2. q8/q4 derived from the bf16 tier (worker-side quantize; no re-convert of the heavy sources).
    for (tier, bits) in QUANT_TIERS {
        let out_dir = out_root.join(tier);
        eprintln!("building {tier} tier → {} (bits={bits})", out_dir.display());
        build_quant_tier(&bf16_dir, &out_dir, tier, *bits);
        mlx_rs::memory::clear_cache();
    }

    eprintln!(
        "done — upload the tier subdirs:\n  hf upload SceneWorks/wan2.2-ti2v-5b-mlx {} --include 'bf16/*' 'q8/*' 'q4/*'",
        out_root.display()
    );
}
