//! Local real-weight MLX smoke for the Mage-Flow **Q8** worker lane (sc-14980 / sc-14979).
//! `#[ignore]`d — run by hand on an Apple-Silicon Mac.
//!
//! This is the ORDERING-SAFETY proof for the per-tier re-host. Mage is the first family whose tier
//! subdir is NOT engine-complete: `<snapshot>/q8/` holds the DiT and scheduler only, while the text
//! encoder and VAE — bit-identical across all six variants — live once in
//! `SceneWorks/Mage-Flow-Components-mlx` and are staged as per-tier `coRequisite` components. So the
//! failure this guards against is specific and was real before the pin bump: the worker resolves
//! `<snapshot>/q8` via `standard_tier_subdir`, and an engine that expects a FLAT diffusers root then
//! looks for `<snapshot>/q8/text_encoder/` and dies at load. A manifest that ships per-tier downloads
//! against an engine pinned before the split is exactly that break.
//!
//! It drives the real seam `generate_stream` uses, minus the API/job plumbing:
//!   1. `standard_tier_subdir` → `<snapshot>/q8` (asserted to be DiT-only, so the check is honest),
//!   2. `resolve_co_requisites_for_tier(.., Some("q8"))` → the shared TE + VAE component dirs,
//!   3. `LoadSpec::new(Dir(<snapshot>/q8)).with_quant(Q8)` + those components,
//!   4. `inference_runtime::load("mage_flow_base", &spec)` — the real engine load,
//!   5. a render, checked non-degenerate.
//!
//! **Why q8 and not the default q4 tier.** Historical, and no longer a statement about q4's health.
//! When this smoke was written, Mage's q4 tier did not render correctly — at 512x512/30 steps it
//! produced a repeating tiled texture where bf16 and q8 both produced the prompt — so q8 was the
//! only tier on which "a correct render" could prove the split layout had wired the right weights.
//! **That defect is fixed (sc-15071):** two projections needed an 8-bit floor (the DiT's
//! `norm_out.linear` and the Qwen3-VL LM's decoder layers), the q4 artifacts on all seven mirrors
//! were regenerated, and q4 now renders the prompt.
//!
//! The smoke stays on q8 because the thing it gates is the **split layout** — DiT-only tier subdir
//! plus co-requisite TE/VAE dirs — which is identical across tiers, and q8 is what the cache is
//! already primed with. Tier *correctness* is not this test's job and is no longer left to a
//! non-degenerate-render check: it belongs to
//! `mlx-gen-mage/tests/quant_real_weights.rs::every_tier_renders_the_reference_scene_and_a_uniform_q4_head_fails_the_gate`,
//! which compares each tier against the bf16 render at a fixed seed and is proven by mutation to
//! fail on exactly the tiled output this file's original note describes.
//!
//! Setup — with the q8 tier of BOTH repos in the HF cache, no env is needed:
//! ```text
//! # optional: MAGE_Q8_STEPS=4 MAGE_Q8_W=512 MAGE_Q8_H=512 MAGE_Q8_PROMPT="..." MAGE_Q8_OUT_DIR=/tmp/mage_q8_smoke
//! cargo test -p sceneworks-worker --release mage_flow_q8_mlx_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

use super::smoke_support::{env_or, image_mean, image_std, is_all_zero, save_png};

const VARIANT_REPO: &str = "SceneWorks/Mage-Flow-Base";
const COMPONENTS_REPO: &str = "SceneWorks/Mage-Flow-Components-mlx";
const MODEL_ID: &str = "mage_flow_base";

/// The hub cache root this smoke reads from: whatever the environment already points at, else the
/// standard `~/.cache/huggingface/hub` that `huggingface_hub` downloads into.
fn hub_cache_root() -> Option<PathBuf> {
    for key in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"] {
        if let Ok(value) = std::env::var(key) {
            if !value.trim().is_empty() {
                return Some(PathBuf::from(value));
            }
        }
    }
    if let Ok(value) = std::env::var("HF_HOME") {
        if !value.trim().is_empty() {
            return Some(PathBuf::from(value).join("hub"));
        }
    }
    Some(PathBuf::from(std::env::var("HOME").ok()?).join(".cache/huggingface/hub"))
}

/// The cached snapshot of `repo` whose `q8/` subdir exists. `None` when that tier is not pulled.
fn cached_q8_snapshot(hub: &Path, repo: &str) -> Option<PathBuf> {
    let snapshots = hub
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|entry| entry.ok())
        .find_map(|entry| {
            let path = entry.path();
            path.join("q8").is_dir().then_some(path)
        })
}

#[test]
#[ignore = "real-weight MLX smoke; needs the SceneWorks/Mage-Flow-Base + Mage-Flow-Components-mlx q8 tiers cached + an Apple-Silicon Mac"]
fn mage_flow_q8_mlx_gpu_smoke() {
    let hub = hub_cache_root().expect("a resolvable HF hub cache root");
    // Pin the worker's cache resolution to the SAME hub this smoke discovered in, so the
    // co-requisite resolver reads the tiers we just verified rather than a different data_dir's
    // cache. Without this the resolve fails with "not installed" even though the tiers are present.
    let _env = crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", &hub.display().to_string())]);
    let Some(variant) = cached_q8_snapshot(&hub, VARIANT_REPO) else {
        panic!("{VARIANT_REPO} q8 tier is not in the HF cache — pull it before running this smoke");
    };
    let Some(components) = cached_q8_snapshot(&hub, COMPONENTS_REPO) else {
        panic!(
            "{COMPONENTS_REPO} q8 tier is not in the HF cache — the shared text encoder + VAE are a \
             co-requisite of every Mage variant, so this smoke cannot prove the split layout without it"
        );
    };
    let tier = variant.join("q8");

    // The premise of the whole test: the tier is DiT-only. If a future re-host put the text encoder
    // back inside the tier, the load below would pass for the WRONG reason (the flat-root fallback),
    // and this smoke would silently stop testing the co-requisite path.
    assert!(
        tier.join("transformer/diffusion_pytorch_model.safetensors")
            .is_file(),
        "the q8 tier must carry the packed DiT"
    );
    assert!(
        !tier.join("text_encoder").exists() && !tier.join("vae").exists(),
        "the q8 tier must NOT contain text_encoder/ or vae/ — they are shared co-requisites, and \
         their presence here would make this smoke pass without exercising the component seam"
    );

    // 2. The component seam, tier-matched — the exact call `attach_required_components` makes.
    let descriptor = crate::inference_runtime::media_descriptor(MODEL_ID)
        .unwrap_or_else(|| panic!("{MODEL_ID} is a registered media provider"));
    assert_eq!(
        descriptor.required_components,
        &["text_encoder", "vae"],
        "the pinned engine must advertise the shared components; an engine pinned BEFORE the \
         sc-14979 split advertises &[] and would look for {}/text_encoder instead",
        tier.display()
    );
    let manifest = builtin_manifest_entry(MODEL_ID);
    // The REAL environment's HF cache resolution — the same `Settings` a running worker uses, so
    // the co-requisite resolves out of the actual hub cache rather than a synthetic fixture.
    let settings = crate::Settings::from_env();
    let staged = crate::model_jobs::resolve_co_requisites_for_tier(
        &descriptor,
        &manifest,
        &settings,
        Some("q8"),
    )
    .expect("the q8 shared components resolve from the components mirror");

    // They must resolve into the COMPONENTS repo's q8 subtree — not the variant snapshot.
    for (id, source) in &staged {
        let gen_core::WeightsSource::Dir(dir) = source else {
            panic!("{id} must stage as a directory");
        };
        assert!(
            dir.starts_with(&components) && dir.ends_with(format!("q8/{id}")),
            "{id} must resolve to {}/q8/{id}, got {}",
            components.display(),
            dir.display()
        );
        assert!(dir.is_dir(), "{id} component dir must exist on disk");
    }
    println!("staged components: {:?}", staged.keys().collect::<Vec<_>>());

    // 3-4. The real load, exactly as generate_stream builds it.
    let spec = staged.into_iter().fold(
        LoadSpec::new(WeightsSource::Dir(tier.clone())).with_quant(Quant::Q8),
        |spec, (id, source)| spec.with_component(id, source),
    );
    let model = crate::inference_runtime::load(MODEL_ID, &spec).unwrap_or_else(|error| {
        panic!(
            "the q8 tier at {} must load with the shared components staged — this is the exact \
             failure a stale engine pin produces: {error}",
            tier.display()
        )
    });

    // 5. Render and check it is a real image.
    let steps: u32 = env_or("MAGE_Q8_STEPS", "8").parse().expect("steps");
    let width: u32 = env_or("MAGE_Q8_W", "512").parse().expect("width");
    let height: u32 = env_or("MAGE_Q8_H", "512").parse().expect("height");
    let prompt = env_or(
        "MAGE_Q8_PROMPT",
        "a red cube on a white table, studio lighting",
    );
    let request = GenerationRequest {
        prompt: prompt.clone(),
        width,
        height,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(5.0),
        ..Default::default()
    };
    let mut last = String::new();
    let output = model
        .generate(&request, &mut |progress| {
            let line = format!("{progress:?}");
            if line != last {
                println!("[progress] {line}");
                last = line;
            }
        })
        .expect("q8 generation");
    let GenerationOutput::Images(images) = output else {
        panic!("mage_flow_base must produce images");
    };
    let image = images.first().expect("one image");

    assert!(!is_all_zero(image), "the q8 render must not be all-zero");
    let (mean, std) = (image_mean(image), image_std(image));
    println!("q8 render: mean={mean:.2} std={std:.2} ({width}x{height}, {steps} steps)");
    // A dynamic-range floor alone is NOT a quality gate: Mage's q4 tier renders a repeating tiled
    // texture that scores std ~64 and would sail past any such check (see the q4 note above). The
    // meaningful assertions here are the structural ones ABOVE — the tier is DiT-only, the engine
    // advertises the components, and they resolve out of the shared mirror — plus a sanity floor so
    // a flat/black frame still fails loudly.
    assert!(
        std >= 20.0 && (10.0..245.0).contains(&mean),
        "the q8 render looks degenerate (mean {mean:.2}, std {std:.2})"
    );

    let out_dir = PathBuf::from(env_or("MAGE_Q8_OUT_DIR", "/tmp/mage_q8_smoke"));
    std::fs::create_dir_all(&out_dir).expect("out dir");
    save_png(image, &out_dir.join("mage_flow_base_q8.png"));
    println!("wrote {}", out_dir.join("mage_flow_base_q8.png").display());
}

/// The shipped catalog entry for `model_id`, read straight from the embedded builtin manifest — the
/// same bytes the worker resolves co-requisites against.
fn builtin_manifest_entry(model_id: &str) -> serde_json::Value {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc is embedded");
    let manifest: serde_json::Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
            .expect("builtin.models.jsonc parses");
    manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|entry| entry.get("id").and_then(serde_json::Value::as_str) == Some(model_id))
        .cloned()
        .unwrap_or_else(|| panic!("builtin entry {model_id} present"))
}
