//! On-device build helper for the Wan2.2 T2V-A14B quant matrix (sc-9942, epic 8506).
//!
//! Produces the three hosted tier subdirs — `bf16/` + `q8/` + `q4/` — from the native
//! `Wan-AI/Wan2.2-T2V-A14B` checkpoint by driving the SAME byte-parity-validated converter the
//! turnkey uses (`mlx_gen_wan::convert::convert_t2v_14b`), once per tier with the matching quant.
//! Each tier is a COMPLETE self-contained dual-expert snapshot (both MoE experts, the UMT5 T5, the
//! z16 VAE and `config.json` with the quant baked in); this helper additionally copies the
//! `tokenizer.json` the converter does not emit into every tier so the load path
//! (`video_jobs::wan_a14b_tier_is_complete`) treats each as present.
//!
//! This is an `#[ignore]`d test, not part of CI — it needs the ~126 GB native fp32 checkpoint on
//! disk and takes minutes per tier on an Apple-Silicon Mac. Run one-off to produce the artifacts,
//! then `hf upload` the `bf16/`/`q8/`/`q4/` subdirs to `SceneWorks/wan2.2-t2v-a14b-mlx`.
//!
//! ```sh
//! # Native checkpoint (high_noise_model/ + low_noise_model/ + models_t5_umt5-xxl-enc-bf16.pth +
//! # Wan2.1_VAE.pth), e.g. `hf download Wan-AI/Wan2.2-T2V-A14B --local-dir <native>`:
//! export SCENEWORKS_WAN_T2V_14B_NATIVE_DIR=<native>
//! export SCENEWORKS_WAN_T2V_14B_TIER_OUT=<out-root>          # bf16/ q8/ q4/ written here
//! # tokenizer.json to copy into each tier (the converter does not emit it). Defaults to
//! # <native>/tokenizer.json, then the cached SceneWorks/wan2.2-t2v-a14b-mlx turnkey's tokenizer.json.
//! export SCENEWORKS_WAN_T2V_14B_TOKENIZER=<path/to/tokenizer.json>   # optional
//! cargo test -p sceneworks-worker --release wan_t2v_14b_build_tiers -- --ignored --nocapture
//! ```
//!
//! Each tier prints a `[[TIER]] {json}` line (tier, dir, diskSizeBytes) so the manifest
//! `estimatedSizeBytes`/`footprint.diskSizeBytes` can be backfilled with the exact hosted sizes.

use std::path::{Path, PathBuf};

/// The three tiers to build: `(subdir, Option<(bits, group_size)>)`. bf16 is dense (`None`); q8/q4
/// pack the Linear-DiT experts at group 64 — the canonical Wan group (mflux/reference default; the
/// same group `WanModelConfig::from_config_json` assumes and the load path reconstructs).
const TIERS: &[(&str, Option<(i32, i32)>)] =
    &[("bf16", None), ("q8", Some((8, 64))), ("q4", Some((4, 64)))];

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
/// `<native>/tokenizer.json`, then the cached `SceneWorks/wan2.2-t2v-a14b-mlx` turnkey's
/// `tokenizer.json` under the HF hub cache. Panics with actionable guidance if none is found — the
/// tier is incomplete without it.
fn resolve_tokenizer(native_dir: &Path) -> PathBuf {
    if let Ok(explicit) = std::env::var("SCENEWORKS_WAN_T2V_14B_TOKENIZER") {
        let path = PathBuf::from(explicit.trim());
        assert!(
            path.is_file(),
            "SCENEWORKS_WAN_T2V_14B_TOKENIZER={} is not a file",
            path.display()
        );
        return path;
    }
    let beside = native_dir.join("tokenizer.json");
    if beside.is_file() {
        return beside;
    }
    // Cached turnkey: ~/.cache/huggingface/hub/models--SceneWorks--wan2.2-t2v-a14b-mlx/snapshots/*/tokenizer.json
    if let Some(home) = std::env::var_os("HOME") {
        let snapshots = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join("models--SceneWorks--wan2.2-t2v-a14b-mlx")
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
        "no tokenizer.json found — set SCENEWORKS_WAN_T2V_14B_TOKENIZER, place one at {}, or cache \
         the SceneWorks/wan2.2-t2v-a14b-mlx turnkey (hf download … --include tokenizer.json)",
        beside.display()
    );
}

/// Build all three Wan2.2 T2V-A14B tier subdirs from the native checkpoint. `#[ignore]`d — run
/// on-device with the env vars above; not exercised in CI (needs the native weights).
#[test]
#[ignore = "on-device tier build: needs the ~126GB native Wan2.2-T2V-A14B checkpoint + minutes/tier"]
fn wan_t2v_14b_build_tiers() {
    let native_dir = PathBuf::from(
        std::env::var("SCENEWORKS_WAN_T2V_14B_NATIVE_DIR")
            .expect("set SCENEWORKS_WAN_T2V_14B_NATIVE_DIR to the native checkpoint dir"),
    );
    assert!(
        native_dir.join("high_noise_model").is_dir() && native_dir.join("low_noise_model").is_dir(),
        "{} is not a native Wan2.2 A14B checkpoint (expected high_noise_model/ + low_noise_model/ \
         subdirs, models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth)",
        native_dir.display()
    );
    let out_root = PathBuf::from(
        std::env::var("SCENEWORKS_WAN_T2V_14B_TIER_OUT")
            .expect("set SCENEWORKS_WAN_T2V_14B_TIER_OUT to the output root for bf16/ q8/ q4/"),
    );
    let tokenizer = resolve_tokenizer(&native_dir);
    std::fs::create_dir_all(&out_root).unwrap();

    // The converter loads each expert as a full fp32 buffer (~57 GB) then casts to bf16; at MLX's
    // default cache limit the freed fp32 buffers are RETAINED as cache, so after both experts the
    // residue (~114 GB) plus the T5 f32 load OOM-SIGKILLs the process on a 128 GB box (sc-5567 batch
    // pattern). Cap the buffer cache to 0 so freed buffers return to the OS immediately between
    // stages — a build-time realloc-cost trade the generation hot path deliberately avoids.
    mlx_rs::memory::set_cache_limit(0);

    for (tier, quant) in TIERS {
        let out_dir = out_root.join(tier);
        eprintln!(
            "building {tier} tier → {} (quant={:?})",
            out_dir.display(),
            quant
        );
        mlx_gen_wan::convert::convert_t2v_14b(&native_dir, &out_dir, *quant)
            .unwrap_or_else(|e| panic!("convert_t2v_14b {tier} failed: {e:?}"));
        // Release any retained buffers before the next (heavier) tier so residue never accumulates.
        mlx_rs::memory::clear_cache();
        // The converter does not emit tokenizer.json; copy it so the tier is complete for the load
        // path (video_jobs::wan_a14b_tier_is_complete).
        std::fs::copy(&tokenizer, out_dir.join("tokenizer.json"))
            .unwrap_or_else(|e| panic!("copy tokenizer.json into {tier} failed: {e:?}"));
        // Sanity: the six files the load path requires must all exist.
        for file in [
            "high_noise_model.safetensors",
            "low_noise_model.safetensors",
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
            "config.json",
        ] {
            assert!(
                out_dir.join(file).is_file(),
                "built {tier} tier is missing {file}"
            );
        }
        let size = dir_size_bytes(&out_dir);
        // Machine-readable line for backfilling the manifest sizes.
        eprintln!(
            "[[TIER]] {{\"tier\":\"{tier}\",\"dir\":\"{}\",\"diskSizeBytes\":{size}}}",
            out_dir.display()
        );
    }
    eprintln!(
        "done — upload the tier subdirs:\n  hf upload SceneWorks/wan2.2-t2v-a14b-mlx {} --include 'bf16/*' 'q8/*' 'q4/*'",
        out_root.display()
    );
}
