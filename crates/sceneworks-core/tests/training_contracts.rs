use std::fs;
use std::path::{Path, PathBuf};

use sceneworks_core::training::target_supports_resolution_buckets;
use sceneworks_core::training::{
    build_training_plan, builtin_training_targets, validate_training_config_for_target,
    BuildTrainingPlan, LoraTrainingRequest, TrainingConfig, TrainingDataset, TrainingModality,
    TrainingOutputKind, TrainingPlan, TrainingPlanError, TrainingPresetRegistry,
    TrainingProvenance, TrainingTargetLimitError, TrainingTargetRegistry, GRADIENT_NOISE_ETA_KEY,
    GRADIENT_NOISE_ETA_MAX, GRADIENT_NOISE_ETA_SUGGESTED, GRADIENT_NOISE_GAMMA_DEFAULT,
    GRADIENT_NOISE_GAMMA_KEY, GRADIENT_NOISE_GAMMA_MAX, RESOLUTION_BUCKETS_MAX,
    RESOLUTION_BUCKET_REPEATS_MAX, RESOLUTION_BUCKET_STRIDE,
    SUBJECT_MASK_BACKGROUND_WEIGHT_DEFAULT, SUBJECT_MASK_SUBJECT_WEIGHT_DEFAULT,
    SUBJECT_MASK_WEIGHT_MAX, TRAINING_CONTRACT_SCHEMA_VERSION, TRAINING_PLAN_VERSION,
    WEIGHT_NOISE_SIGMA_KEY, WEIGHT_NOISE_SIGMA_MAX, WEIGHT_NOISE_SIGMA_SUGGESTED,
};
use sceneworks_core::training::{target_supports_gradient_noise, target_supports_weight_noise};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};

fn fixture_path(relative_path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("rust_migration_contracts")
        .join("training")
        .join(relative_path)
}

fn load_fixture(relative_path: &str) -> Value {
    let path = fixture_path(relative_path);
    let payload = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&payload)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn assert_round_trip<T>(relative_path: &str)
where
    T: DeserializeOwned + Serialize,
{
    let original = load_fixture(relative_path);
    let typed: T = serde_json::from_value(original.clone())
        .unwrap_or_else(|error| panic!("failed to deserialize {relative_path}: {error}"));
    let encoded = serde_json::to_value(typed)
        .unwrap_or_else(|error| panic!("failed to serialize {relative_path}: {error}"));

    assert_eq!(
        encoded, original,
        "{relative_path} drifted after typed round-trip"
    );
}

#[test]
fn training_dataset_round_trips() {
    assert_round_trip::<TrainingDataset>("dataset.json");
}

#[test]
fn training_config_round_trips() {
    assert_round_trip::<TrainingConfig>("training-config.json");
}

#[test]
fn lora_training_request_round_trips() {
    assert_round_trip::<LoraTrainingRequest>("lora-training-request.json");
}

#[test]
fn training_plan_round_trips() {
    assert_round_trip::<TrainingPlan>("training-plan.json");
}

#[test]
fn training_provenance_round_trips() {
    assert_round_trip::<TrainingProvenance>("training-provenance.json");
}

#[test]
fn training_target_registry_round_trips() {
    assert_round_trip::<TrainingTargetRegistry>("target-registry.json");
}

#[test]
fn training_preset_registry_round_trips() {
    assert_round_trip::<TrainingPresetRegistry>("preset-registry.json");
}

#[test]
fn builtin_registry_matches_committed_snapshot() {
    let expected = load_fixture("target-registry.json");
    let encoded =
        serde_json::to_value(builtin_training_targets()).expect("builtin registry serializes");

    assert_eq!(
        encoded, expected,
        "builtin target registry drifted from committed snapshot"
    );
}

#[test]
fn builtin_targets_gate_network_types() {
    let registry = builtin_training_targets();

    let advertises = |target: &sceneworks_core::training::TrainingTarget, value: &str| {
        target
            .limits
            .get("networkTypes")
            .and_then(Value::as_array)
            .is_some_and(|types| types.iter().any(|entry| entry.as_str() == Some(value)))
    };

    // Every target advertises `lora`; validated native backends (epic 2193) advertise `lokr`:
    // the Z-Image/SDXL image backends (v1), the Kolors image backend
    // (SDXL-architecture, epic 1929 / sc-2217), the Lens sidecar backend (sc-2218), and
    // the Wan2.2 5B video backend (sc-2211). Krea 2 (epic 7565) is the first *native-MLX*
    // LoKr target: its `mlx-gen-krea` trainer reports `supports_lokr` and builds LoKr targets
    // natively, and the Turbo inference loader applies LoKr through the shared adapter seam
    // (sc-7911). SD3.5 Large + Medium (epic 7841 T3 sc-7884) likewise expose native MLX and
    // Candle LoKr targets; both providers report `supports_lokr` and share the
    // `apply_sd3_adapters` inference seam. LTX and the Wan MoE targets stay lora-only.
    let lokr_targets: Vec<&str> = registry
        .targets
        .iter()
        .filter(|target| {
            // Control-branch targets (krea_control, epic 10159) train a conditioning branch, not a
            // LoRA/LoKr network, so they legitimately advertise no `networkTypes`. The lora/lokr
            // invariant applies only to LoRA-output targets.
            if !matches!(target.output_kind, TrainingOutputKind::Lora) {
                return false;
            }
            assert!(
                advertises(target, "lora"),
                "target {} must advertise lora in limits.networkTypes",
                target.id
            );
            advertises(target, "lokr")
        })
        .map(|target| target.id.as_str())
        .collect();

    assert_eq!(
        lokr_targets,
        [
            "z_image_turbo_lora",
            "sdxl_lora",
            // Illustrious v1.0/v2.0 are vanilla-SDXL anime finetunes sharing the `sdxl_lora`
            // kernel; both advertise native-MLX LoKr through the same SDXL trainer (epic 10609).
            "illustrious_xl_v1_lora",
            "illustrious_xl_v2_lora",
            "kolors_lora",
            "lens_turbo_lora",
            "krea_2_raw_lora",
            "sd3_5_large_lora",
            "sd3_5_medium_lora",
            "anima_base_lora",
            // T2V-14B retains its existing Candle LoKr path. The single-DiT TI2V-5B and I2V-14B
            // generated-matrix obligations remain LoRA-only.
            "wan_t2v_14b_lora",
            // Mage-Flow (sc-14055, offered in sc-14056): the `mlx-gen-mage` trainer reports
            // `supports_lokr` and has a real LoKr branch, and Mage inference installs LoKr through
            // the same strict adapter seam as LoRA (`apply_mage_adapters`). It was absent from this
            // list only because the target never advertised `lokr` — shipped but unreachable.
            // Base only: `mage_flow_edit_base_lora` is no longer offered (sc-15277 — no edit
            // trainer exists, so it could only ever claim-then-fail; restoring it is sc-15320).
            "mage_flow_base_lora"
        ]
    );
}

#[test]
#[ignore = "regen helper: run with REGEN_FIXTURE=1 to rewrite target-registry.json"]
fn regen_target_registry_fixture() {
    if std::env::var("REGEN_FIXTURE").is_err() {
        return;
    }
    let pretty = serde_json::to_string_pretty(&builtin_training_targets())
        .expect("builtin registry serializes");
    fs::write(fixture_path("target-registry.json"), pretty + "\n").expect("write fixture");
}

#[test]
#[ignore = "regen helper: run with REGEN_FIXTURE=1 to rewrite preset-registry.json"]
fn regen_preset_registry_fixture() {
    if std::env::var("REGEN_FIXTURE").is_err() {
        return;
    }
    let pretty =
        serde_json::to_string_pretty(&sceneworks_core::training::builtin_training_presets())
            .expect("builtin preset registry serializes");
    fs::write(fixture_path("preset-registry.json"), pretty + "\n").expect("write fixture");
}

#[test]
fn builtin_preset_registry_matches_committed_snapshot() {
    let expected = load_fixture("preset-registry.json");
    let encoded = serde_json::to_value(sceneworks_core::training::builtin_training_presets())
        .expect("builtin preset registry serializes");

    assert_eq!(
        encoded, expected,
        "builtin preset registry drifted from committed snapshot"
    );
}

#[test]
fn builtin_preset_registry_exposes_optimizer_sensitive_defaults() {
    let registry = sceneworks_core::training::builtin_training_presets();
    let prodigy = registry
        .presets
        .iter()
        .find(|preset| preset.id == "z_image_turbo_lora.character.prodigyopt.balanced")
        .expect("prodigy preset present");

    assert_eq!(prodigy.target_id, "z_image_turbo_lora");
    assert_eq!(prodigy.optimizer, "prodigyopt");
    assert_eq!(prodigy.config.optimizer, "prodigyopt");
    assert_eq!(prodigy.config.learning_rate.as_f64(), Some(1.0));
    assert_eq!(prodigy.config.steps, 1600);
    assert_eq!(prodigy.config.advanced["sampleEvery"], 200);
    assert_eq!(prodigy.config.advanced["sampleSteps"], 8);
    assert_eq!(prodigy.config.advanced["sampleGuidanceScale"], 1.0);
    assert_eq!(prodigy.ui["experimental"], true);

    let balanced = registry
        .presets
        .iter()
        .find(|preset| preset.id == "z_image_turbo_lora.character.adamw8bit.balanced")
        .expect("balanced character preset present");
    assert_eq!(balanced.config.steps, 3000);
    assert_eq!(
        balanced.config.advanced["trainingAdapterRepo"],
        "ostris/zimage_turbo_training_adapter"
    );
    assert_eq!(balanced.config.advanced["timestepType"], "sigmoid");
    assert_eq!(balanced.config.advanced["timestepBias"], "high_noise");
    assert_eq!(balanced.config.advanced["gradientCheckpointing"], true);
}

#[test]
fn builtin_registry_exposes_z_image_turbo_target() {
    let registry = builtin_training_targets();
    assert_eq!(registry.schema_version, TRAINING_CONTRACT_SCHEMA_VERSION);

    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("z_image_turbo_lora target present");

    assert_eq!(target.modality, TrainingModality::Image);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "z-image");
    assert_eq!(target.base_model, "z_image_turbo");
    // sc-13860: the DEFAULT training base trains off the SceneWorks/z-image-turbo-mlx turnkey the
    // catalog + engine install, NOT the flat upstream `Tongyi-MAI/Z-Image-Turbo` the epic-8506 re-host
    // stopped downloading (false "not installed"); the trainer reads its dense `bf16/` tier.
    assert_eq!(
        target.base_model_repo.as_deref(),
        Some("SceneWorks/z-image-turbo-mlx")
    );
    assert_eq!(target.kernel, "z_image_lora");
    assert_eq!(target.defaults.rank, 16);
    assert_eq!(target.defaults.resolution, 1024);
    assert_eq!(target.defaults.trigger_word, None);
    assert_eq!(
        target.defaults.advanced.get("sampleGuidanceScale"),
        Some(&serde_json::json!(1.0))
    );
}

/// Mage-Flow offers exactly ONE training target — the Base checkpoint (epic 14034: "Base = training
/// target", F6 is scoped "vs Base"), which is also the only id `mlx-gen-mage` registers a trainer
/// under. sc-15277 removed the speculative `mage_flow_edit_base_lora` contract sc-14054 had
/// registered ahead of any trainer: once sc-14056 routed `mage_flow_lora` to the mlx worker, that
/// target stopped queueing and started being CLAIMED and then failed ~2s later. The negative half
/// below is the guard — re-offering an edit target without an edit trainer must fail here (and in
/// the worker's `every_mlx_routed_training_kernel_resolves_to_a_trainer_…` invariant), not in a
/// user's job. Restoring it with a real trainer is sc-15320.
#[test]
fn builtin_registry_exposes_the_mage_flow_base_target_and_no_edit_target() {
    let registry = builtin_training_targets();
    assert!(
        !registry
            .targets
            .iter()
            .any(|target| target.base_model.starts_with("mage_flow_edit")),
        "no Mage EDIT training target may be advertised until `mlx-gen-mage` registers an edit \
         trainer (sc-15320) — `engine_trainer_id` resolves nothing for it, so the mlx worker would \
         claim the job and immediately fail it"
    );
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "mage_flow_base_lora")
        .expect("mage_flow_base_lora target present");
    assert_eq!(target.modality, TrainingModality::Image);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "mage-flow");
    assert_eq!(target.base_model, "mage_flow_base");
    assert_eq!(
        target.base_model_repo.as_deref(),
        Some("SceneWorks/Mage-Flow-Base")
    );
    assert_eq!(target.kernel, "mage_flow_lora");
    assert_eq!(target.defaults.resolution, 1024);
    // The Training Studio's network-type picker is built from exactly this list, so this
    // assertion is what keeps each capability OFFERABLE. All three of Mage's trainer paths must
    // appear: `lora` and `lokr` (both advertised by the `mlx-gen-mage` trainer and both loadable
    // back through `apply_mage_adapters`), plus `full` — the base fine-tune, which Mage-Flow is
    // the only family to have. `lokr` and `full` were each shipped-but-unreachable purely
    // because they were absent here (sc-14055 / sc-14056).
    assert_eq!(
        target.limits.get("networkTypes"),
        Some(&serde_json::json!(["lora", "lokr", "full"]))
    );
}

/// sc-14056: `full` is a Mage-Flow-only network type. Every other built-in target must keep
/// advertising only adapter networks — a stray `full` on a family whose trainer has no full path
/// would offer the user a run the engine rejects (gen-core's `validate_full_finetune_request` floor).
#[test]
fn only_mage_flow_targets_advertise_the_full_finetune_network_type() {
    let registry = builtin_training_targets();
    let full_targets: Vec<&str> = registry
        .targets
        .iter()
        .filter(|target| {
            target
                .limits
                .get("networkTypes")
                .and_then(Value::as_array)
                .is_some_and(|types| types.iter().any(|entry| entry.as_str() == Some("full")))
        })
        .map(|target| target.id.as_str())
        .collect();
    assert_eq!(
        full_targets,
        vec!["mage_flow_base_lora"],
        "only the Mage-Flow Base target may advertise the full base fine-tune network type"
    );
}

#[test]
fn builtin_registry_exposes_sdxl_target() {
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "sdxl_lora")
        .expect("sdxl_lora target present");

    assert_eq!(target.modality, TrainingModality::Image);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "sdxl");
    assert_eq!(target.base_model, "sdxl");
    // Trains off the SceneWorks turnkey the catalog + engine install, NOT the flat upstream
    // `stabilityai/stable-diffusion-xl-base-1.0` nothing downloads (issue #1694); the trainer
    // reads its dense `bf16/` tier (rust-api resolve_base_model_path descends into it).
    assert_eq!(
        target.base_model_repo.as_deref(),
        Some("SceneWorks/sdxl-base-mlx")
    );
    assert_eq!(target.kernel, "sdxl_lora");
    assert_eq!(target.defaults.rank, 16);
    assert_eq!(target.defaults.resolution, 1024);
    // Real CFG previews (higher positive guidance), unlike the distilled Z-Image target.
    assert_eq!(
        target.defaults.advanced.get("sampleGuidanceScale"),
        Some(&serde_json::json!(7.0))
    );
    // SDXL UNet attention modules drive the LoRA injection.
    assert_eq!(
        target.defaults.advanced.get("loraTargetModules"),
        Some(&serde_json::json!(["to_q", "to_k", "to_v", "to_out.0"]))
    );
}

/// sc-13860 — registry↔catalog parity guard.
///
/// Every training target's `base_model_repo` must be a repo the target's own model CATALOG entry
/// actually downloads. The pre-flight install gate (`training_base_model_installed`, rust-api) keys on
/// `base_model_repo` in the HF cache; if a target names an UPSTREAM repo the app no longer downloads
/// (the epic-8506 MLX re-host moved the catalog to `SceneWorks/*-mlx` turnkeys), the gate never finds
/// it and 400s "not installed" on every real run — even when the model IS installed for generation.
/// That was issue #1694 (SDXL) and this whole class (z-image / sd3.5 / kolors). Reading the SHIPPED
/// `builtin.models.jsonc` and asserting the invariant here means the class can't silently drift again.
///
/// The join is per-target (target `base_model` == catalog `id`), so it catches "named the wrong repo",
/// not merely "named a dead one". NOTE lens (`SceneWorks/Lens` flat-diffusers training variant) and wan
/// (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`, the Windows/Linux torch bf16 variant) INTENTIONALLY do not name
/// their `*-mlx` inference turnkey — the guard still passes because the catalog installs those exact
/// repos as separate download variants (verified sc-13860; do NOT "fix" them to the turnkey).
#[test]
fn every_training_base_repo_is_installed_by_its_catalog_entry() {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;
    use std::collections::BTreeSet;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc embedded in BUILTIN_MANIFESTS");
    let catalog: Value = serde_json::from_str(&strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses as JSON");
    let models = catalog["models"]
        .as_array()
        .expect("catalog has a models array");
    // Mutation-check on the parser itself: a silently-empty catalog would make every assertion below
    // vacuously pass. The real catalog has dozens of models.
    assert!(
        models.len() > 20,
        "catalog parsed to only {} models — the manifest parse likely broke, not a real result",
        models.len()
    );

    // model id -> the set of every repo any of its download variants pulls.
    let mut downloads_by_id: std::collections::BTreeMap<&str, BTreeSet<&str>> =
        std::collections::BTreeMap::new();
    for model in models {
        let Some(id) = model["id"].as_str() else {
            continue;
        };
        let repos = downloads_by_id.entry(id).or_default();
        if let Some(downloads) = model["downloads"].as_array() {
            for download in downloads {
                if let Some(repo) = download["repo"].as_str() {
                    repos.insert(repo);
                }
            }
        }
    }

    for target in builtin_training_targets().targets {
        let Some(repo) = target.base_model_repo.as_deref() else {
            // A target with no explicit repo resolves via the app-managed models dir, not the HF-cache
            // gate keyed on `base_model_repo`, so there is nothing to reconcile against the catalog.
            continue;
        };
        let catalog_repos = downloads_by_id.get(target.base_model.as_str()).unwrap_or_else(|| {
            panic!(
                "training target `{}` trains base `{}`, but no model in builtin.models.jsonc has that \
                 id — the training base has no catalog entry to install it from (sc-13860)",
                target.id, target.base_model
            )
        });
        assert!(
            catalog_repos.contains(repo),
            "training target `{}` names base_model_repo `{}`, which the `{}` catalog entry does NOT \
             download (it installs: {:?}). The pre-flight install gate keys on this repo, so it would \
             report the installed base as missing and 400 every real run (sc-13860). Point it at the \
             repo the catalog installs.",
            target.id,
            repo,
            target.base_model,
            catalog_repos
        );
    }
}

/// sc-14054 — Mage's three logical quality variants all install the same flat dense snapshot.
/// They must remain available on every desktop platform and point at the target's live mirror;
/// otherwise Training Studio can offer a target whose platform-specific catalog view cannot install
/// the repo used by the API pre-flight resolver.
#[test]
fn mage_flow_training_repos_are_available_on_macos_windows_and_linux() {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc embedded in BUILTIN_MANIFESTS");
    let catalog: Value = serde_json::from_str(&strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses as JSON");
    let models = catalog["models"].as_array().expect("catalog models array");

    // Base only: `mage_flow_edit_base` is still a first-class GENERATION model in the catalog, but
    // it is no longer a training target (sc-15277 / sc-15320), so its tiers are not a training
    // pre-flight concern and this training contract must not claim otherwise.
    {
        let (base_model, repo) = ("mage_flow_base", "SceneWorks/Mage-Flow-Base");
        let model = models
            .iter()
            .find(|model| model["id"] == base_model)
            .unwrap_or_else(|| panic!("{base_model} catalog entry present"));
        let downloads = model["downloads"]
            .as_array()
            .unwrap_or_else(|| panic!("{base_model} downloads array"));
        for tier in ["q4", "q8", "bf16"] {
            let download = downloads
                .iter()
                .find(|download| download["variant"] == tier)
                .unwrap_or_else(|| panic!("{base_model} has {tier} training tier"));
            assert_eq!(download["repo"], repo);
            assert!(
                download.get("platforms").is_none(),
                "{base_model}/{tier} must be installable on macOS, Windows, and Linux"
            );
        }
    }
}

/// sc-13878 — platform-aware training-base guard (the sc-13860 guard above is platform-BLIND).
///
/// The three Wan video targets name a Windows/Linux torch/diffusers `base_model_repo`
/// (`Wan-AI/Wan2.2-*-Diffusers`) the manifest gates to `platforms:[windows,linux]`. macOS installs a
/// DIFFERENT repo — the pre-converted MLX turnkey (`SceneWorks/wan2.2-*-mlx`, `platforms:[macos]`) — and
/// runs the native MLX trainer against it. The sc-13860 guard only checks `base_model_repo ∈ downloads`
/// (which PASSES here — the diffusers repo IS a win/linux download), so it structurally cannot catch that
/// on macOS the gate/resolver must key off a DIFFERENT repo; that mismatch was this bug. This guard
/// asserts, per Wan target, that the manifest offers a macOS-installable MLX turnkey for the base AND that
/// the diffusers repo is win/linux-only — the platform split the rust-api `macos_wan_*` gate/resolver
/// depend on. A dropped macOS turnkey download, or the diffusers repo (wrongly) tagged for macOS, fails
/// here regardless of which OS runs CI.
#[test]
fn macos_wan_targets_have_a_macos_installable_mlx_turnkey_base() {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc embedded in BUILTIN_MANIFESTS");
    let catalog: Value = serde_json::from_str(&strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses as JSON");
    let models = catalog["models"]
        .as_array()
        .expect("catalog has a models array");
    assert!(
        models.len() > 20,
        "catalog parsed to only {} models — the manifest parse likely broke",
        models.len()
    );

    // A download applies to an OS when it has no `platforms` key (all OSes) or lists that OS.
    fn download_targets_os(download: &Value, os: &str) -> bool {
        match download.get("platforms").and_then(Value::as_array) {
            None => true,
            Some(platforms) => platforms.iter().any(|p| p.as_str() == Some(os)),
        }
    }
    let model_entry = |id: &str| -> Value {
        models
            .iter()
            .find(|m| m["id"].as_str() == Some(id))
            .unwrap_or_else(|| panic!("builtin.models.jsonc has a `{id}` entry"))
            .clone()
    };

    // The macOS MLX turnkey each Wan base installs — the per-platform counterpart the rust-api
    // `macos_wan_mlx_repo` map returns. Kept in lockstep with that map (a typo here or there fails).
    let expected_macos_turnkey = |base_model: &str| -> &'static str {
        match base_model {
            "wan_2_2" => "SceneWorks/wan2.2-ti2v-5b-mlx",
            "wan_2_2_t2v_14b" => "SceneWorks/wan2.2-t2v-a14b-mlx",
            "wan_2_2_i2v_14b" => "SceneWorks/wan2.2-i2v-a14b-mlx",
            other => panic!("unexpected Wan base_model `{other}` — extend the macOS turnkey map"),
        }
    };

    let wan_targets: Vec<_> = builtin_training_targets()
        .targets
        .into_iter()
        .filter(|t| t.family == "wan-video")
        .collect();
    // Anti-vacuous: the dense 5B + both A14B MoE targets. A silently-empty filter must not pass.
    assert_eq!(
        wan_targets.len(),
        3,
        "expected the 3 wan-video training targets (5B + T2V/I2V A14B), got {}",
        wan_targets.len()
    );

    for target in wan_targets {
        let entry = model_entry(&target.base_model);
        let downloads = entry["downloads"]
            .as_array()
            .unwrap_or_else(|| panic!("`{}` has a downloads array", target.base_model));

        // 1) macOS installs the MLX turnkey the native trainer loads.
        let turnkey = expected_macos_turnkey(&target.base_model);
        assert!(
            downloads.iter().any(|d| d["repo"].as_str() == Some(turnkey)
                && download_targets_os(d, "macos")),
            "training target `{}` (base `{}`) has no macOS-installable `{}` download — macOS Wan LoRA \
             training has no base to run against (sc-13878)",
            target.id,
            target.base_model,
            turnkey
        );

        // 2) The diffusers `base_model_repo` is the Windows/Linux torch base — NOT installed on macOS.
        //    This is exactly why keying the gate on `base_model_repo` (sc-13860) breaks on macOS.
        let diffusers = target
            .base_model_repo
            .as_deref()
            .expect("wan target names a diffusers base_model_repo");
        let diffusers_downloads: Vec<_> = downloads
            .iter()
            .filter(|d| d["repo"].as_str() == Some(diffusers))
            .collect();
        assert!(
            !diffusers_downloads.is_empty(),
            "`{}` diffusers base `{diffusers}` must still be a catalog download (the off-Mac torch base)",
            target.base_model
        );
        assert!(
            diffusers_downloads
                .iter()
                .all(|d| !download_targets_os(d, "macos")),
            "`{}` diffusers base `{diffusers}` is tagged for macOS — it must be Windows/Linux only, or the \
             sc-13860 gate would (wrongly) look for it on macOS (sc-13878)",
            target.base_model
        );
    }
}

#[test]
fn builtin_registry_exposes_kolors_target() {
    // Kolors (epic 1929) is an SDXL-architecture U-Net target served by the
    // `kolors_lora` kernel; it reuses the SDXL attention modules + LoKr support
    // (epic 2193 / sc-2217) and resolves the SceneWorks/kolors-mlx turnkey's dense bf16 tier.
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "kolors_lora")
        .expect("kolors_lora target present");

    assert_eq!(target.modality, TrainingModality::Image);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "kolors");
    assert_eq!(target.base_model, "kolors");
    assert_eq!(target.kernel, "kolors_lora");
    // sc-13860: trains off the SceneWorks/kolors-mlx turnkey the catalog + engine install, NOT the flat
    // upstream `Kwai-Kolors/Kolors-diffusers` the epic-8506 re-host stopped downloading (false
    // "not installed"); the trainer reads its dense `bf16/` tier.
    assert_eq!(
        target.base_model_repo.as_deref(),
        Some("SceneWorks/kolors-mlx")
    );
    // SDXL-shared attention modules + LoKr advertised (sc-2217).
    assert_eq!(
        target.defaults.advanced.get("loraTargetModules"),
        Some(&serde_json::json!(["to_q", "to_k", "to_v", "to_out.0"]))
    );
    assert_eq!(
        target.limits.get("networkTypes"),
        Some(&serde_json::json!(["lora", "lokr"]))
    );
}

#[test]
fn builtin_registry_exposes_krea_target() {
    // Krea 2 (epic 7565) trains on the undistilled `krea/Krea-2-Raw` 12B single-stream
    // DiT via the `krea_lora` kernel and applies at Krea 2 Turbo inference (family `krea_2`,
    // no base-model gating). Rust-native on BOTH backends — mlx (Apple Silicon) + candle
    // (Windows/Linux NVIDIA, sc-8614).
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "krea_2_raw_lora")
        .expect("krea_2_raw_lora target present");

    assert_eq!(target.modality, TrainingModality::Image);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "krea_2");
    assert_eq!(target.base_model, "krea_2_raw");
    assert_eq!(target.kernel, "krea_lora");
    // Path 1 (epic 9992): training shares the generation turnkey re-host; the trainer reads its dense
    // `bf16/` tier (rust-api resolve_base_model_path descends into it).
    assert_eq!(
        target.base_model_repo.as_deref(),
        Some("SceneWorks/krea-2-raw-mlx")
    );
    // Single-stream DiT attention modules (separate q/k/v + the joint-attention output
    // projection `to_out.0`), matching the engine trainer's default target set.
    assert_eq!(
        target.defaults.advanced.get("loraTargetModules"),
        Some(&serde_json::json!(["to_q", "to_k", "to_v", "to_out.0"]))
    );
    // First native-MLX LoKr target (the `mlx-gen-krea` trainer + the Turbo inference seam
    // both handle LoKr).
    assert_eq!(
        target.limits.get("networkTypes"),
        Some(&serde_json::json!(["lora", "lokr"]))
    );
    // Rust-native on BOTH backends (mlx + candle, sc-8614), so it is
    // NOT Apple-Silicon-only: the `appleSiliconOnly`/`requiresBackend` mlx-only markers are gone
    // (matching the candle-capable z-image/lens targets; routing is enforced by the worker tables).
    assert_eq!(target.limits.get("appleSiliconOnly"), None);
    assert_eq!(target.limits.get("requiresBackend"), None);
}

#[test]
fn builtin_registry_exposes_sd3_targets() {
    // SD3.5 (epic 7841 T3 sc-7884) exposes TWO native MLX/Candle LoRA/LoKr training bases on the shared
    // `sd3_lora` kernel: Large (`mlx-gen-sd3` trainer id `sd3_5_large`) and the MMDiT-X Medium
    // (`sd3_5_medium`). Both stamp `family: sd3` and apply back at their base via the
    // `apply_sd3_adapters` seam (family-match, no base-model gating) on both native backends.
    let registry = builtin_training_targets();

    let joint_modules = serde_json::json!([
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out"
    ]);

    // sc-13860: both train off the SceneWorks/sd3.5-*-mlx turnkeys the catalog + engine install, NOT
    // the flat upstream `stabilityai/stable-diffusion-3.5-*` the epic-8506 re-host stopped downloading
    // (false "not installed"); the native-MLX trainer reads each turnkey's dense `bf16/` tier.
    for (id, base_model, repo) in [
        (
            "sd3_5_large_lora",
            "sd3_5_large",
            "SceneWorks/sd3.5-large-mlx",
        ),
        (
            "sd3_5_medium_lora",
            "sd3_5_medium",
            "SceneWorks/sd3.5-medium-mlx",
        ),
    ] {
        let target = registry
            .targets
            .iter()
            .find(|target| target.id == id)
            .unwrap_or_else(|| panic!("{id} target present"));

        assert_eq!(target.modality, TrainingModality::Image);
        assert_eq!(target.output_kind, TrainingOutputKind::Lora);
        assert_eq!(target.family, "sd3", "{id} stamps the sd3 LoRA family");
        assert_eq!(target.base_model, base_model);
        assert_eq!(target.kernel, "sd3_lora");
        assert_eq!(target.base_model_repo.as_deref(), Some(repo));
        // SD3.5 MMDiT joint-block attention: both joint streams (image + text), matching the
        // `mlx-gen-sd3` trainer's DEFAULT_TARGET_MODULES.
        assert_eq!(
            target.defaults.advanced.get("loraTargetModules"),
            Some(&joint_modules),
            "{id} targets both joint attention streams"
        );
        // SD3 native logit-normal flow-match recipe (the trainer default).
        assert_eq!(
            target.defaults.advanced.get("timestepType"),
            Some(&serde_json::json!("logit_normal")),
            "{id} uses the SD3 logit-normal recipe"
        );
        // Native-MLX LoRA + LoKr (supports_lokr + apply_sd3_adapters).
        assert_eq!(
            target.limits.get("networkTypes"),
            Some(&serde_json::json!(["lora", "lokr"])),
            "{id} advertises lora + lokr"
        );
        assert!(target.limits.get("appleSiliconOnly").is_none(), "{id}");
        assert!(target.limits.get("requiresBackend").is_none(), "{id}");
    }
}

#[test]
fn builtin_presets_expose_sdxl_character_default() {
    let registry = sceneworks_core::training::builtin_training_presets();
    let default_character = registry
        .presets
        .iter()
        .find(|preset| preset.id == "sdxl_lora.character.adamw8bit.balanced")
        .expect("sdxl character balanced preset present");

    assert_eq!(default_character.target_id, "sdxl_lora");
    assert_eq!(default_character.optimizer, "adamw8bit");
    assert_eq!(default_character.config.rank, 16);
    assert_eq!(
        default_character.ui.get("default"),
        Some(&serde_json::json!(true))
    );

    let style = registry
        .presets
        .iter()
        .find(|preset| preset.id == "sdxl_lora.style.adamw8bit.balanced")
        .expect("sdxl style preset present");
    assert_eq!(style.config.rank, 32);
    assert_eq!(style.config.alpha, 16);
}

#[test]
fn builtin_registry_exposes_ltx_video_target() {
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "ltx_video_lora")
        .expect("ltx_video_lora target present");

    assert_eq!(target.modality, TrainingModality::Video);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "ltx-video");
    assert_eq!(target.base_model, "ltx_2_3");
    assert_eq!(target.kernel, "ltx_mlx_lora");
    assert_eq!(target.defaults.rank, 32);
    assert_eq!(target.defaults.resolution, 768);
    assert_eq!(target.defaults.optimizer, "adamw");
    assert!(
        target.limits.get("appleSiliconOnly").is_none()
            && target.limits.get("requiresBackend").is_none(),
        "LTX training is available on native MLX and candle/CUDA"
    );
    assert!(target.ui.get("appleSiliconOnly").is_none());
    assert!(target.ui.get("backend").is_none());
}

#[test]
fn ltx_video_target_resolves_image_dataset_into_plan() {
    // The video target consumes the image dataset fixture unchanged: build a plan
    // and confirm it carries the video target's kernel/family/modality.
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "ltx_video_lora")
        .expect("ltx_video_lora target present");

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_ltx",
        target,
        dataset: &dataset,
        config: target.defaults.clone(),
        preset: None,
        lora_id: "lora_ltx_new",
        base_model_path: "/models/ltx".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_ltx_new"),
        file_name: "ltx_character.safetensors".to_owned(),
        created_at: "2026-05-22T00:00:00Z".to_owned(),
    })
    .expect("ltx plan resolves");

    assert_eq!(plan.plan_version, TRAINING_PLAN_VERSION);
    assert_eq!(plan.target.kernel, "ltx_mlx_lora");
    assert_eq!(plan.target.family, "ltx-video");
    assert_eq!(plan.target.modality, TrainingModality::Video);
    assert!(!plan.dataset.items.is_empty());
}

#[test]
fn training_plan_preserves_model_specific_item_options() {
    let mut dataset = dataset_fixture();
    dataset.items[0].extra.insert(
        "ltxPreparedBundlePath".to_owned(),
        serde_json::json!("prepared/item-0001.safetensors"),
    );
    dataset.items[0].extra.insert(
        "ltxPreparedBundleSize".to_owned(),
        serde_json::json!(123_u64),
    );
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "ltx_video_lora")
        .expect("ltx_video_lora target present");

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_ltx_options",
        target,
        dataset: &dataset,
        config: target.defaults.clone(),
        preset: None,
        lora_id: "lora_ltx_options",
        base_model_path: "/models/ltx".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_ltx_options"),
        file_name: "ltx_options.safetensors".to_owned(),
        created_at: "2026-08-28T00:00:00Z".to_owned(),
    })
    .expect("ltx plan resolves");

    assert_eq!(
        plan.dataset.items[0].extra.get("ltxPreparedBundlePath"),
        Some(&serde_json::json!("prepared/item-0001.safetensors"))
    );
    assert_eq!(
        plan.dataset.items[0].extra.get("ltxPreparedBundleSize"),
        Some(&serde_json::json!(123_u64))
    );
}

#[test]
fn ltx_2_5_plan_preserves_prepared_multimodal_bundle_input() {
    let mut dataset = dataset_fixture();
    dataset.items[0].extra.insert(
        "ltxPreparedBundlePath".to_owned(),
        serde_json::json!("prepared/item-0001.safetensors"),
    );
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "ltx_2_5_video_lora")
        .expect("LTX-2.5 target present");

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_ltx25",
        target,
        dataset: &dataset,
        config: target.defaults.clone(),
        preset: None,
        lora_id: "lora_ltx25_new",
        base_model_path: "/models/ltx25".to_owned(),
        dataset_root: std::path::Path::new("/datasets/ltx25"),
        output_dir: std::path::Path::new("/outputs/ltx25"),
        file_name: "trained.safetensors".to_owned(),
        created_at: "2026-08-27T00:00:00Z".to_owned(),
    })
    .expect("build LTX-2.5 plan");

    assert_eq!(
        plan.dataset.items[0].extra.get("ltxPreparedBundlePath"),
        Some(&serde_json::json!("prepared/item-0001.safetensors"))
    );
}

#[test]
fn builtin_registry_exposes_wan_target() {
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "wan_lora")
        .expect("wan_lora target present");

    assert_eq!(target.modality, TrainingModality::Video);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "wan-video");
    assert_eq!(target.base_model, "wan_2_2");
    assert_eq!(target.kernel, "wan_lora");
    assert_eq!(target.defaults.rank, 32);
    assert_eq!(target.defaults.resolution, 512);
    // Cross-platform plan default: plain AdamW.
    assert_eq!(target.defaults.optimizer, "adamw");
    // Wan transformer attention projections drive the LoRA injection.
    assert_eq!(
        target.defaults.advanced.get("loraTargetModules"),
        Some(&serde_json::json!(["to_q", "to_k", "to_v", "to_out.0"]))
    );
    // Still-image training: each item encodes to a single Wan-VAE latent frame.
    assert_eq!(
        target.defaults.advanced.get("numFrames"),
        Some(&serde_json::json!(1))
    );
    // The shared contract carries no `appleSiliconOnly` marker, while current execution is the
    // native MLX trainer against the installed MLX turnkey on macOS (sc-13878); unsupported
    // off-Mac work remains queued. The macOS base-weight resolution lives in the rust-api
    // `macos_wan_*` gate/resolver, guarded by `macos_wan_targets_have_a_macos_installable_mlx_turnkey_base`.
    assert_eq!(target.limits.get("appleSiliconOnly"), None);
    assert_eq!(target.limits["networkTypes"], serde_json::json!(["lora"]));
}

#[test]
fn wan_target_resolves_image_dataset_into_plan() {
    // Like LTX, the Wan video target consumes the image dataset fixture unchanged.
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "wan_lora")
        .expect("wan_lora target present");

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_wan",
        target,
        dataset: &dataset,
        config: target.defaults.clone(),
        preset: None,
        lora_id: "lora_wan_new",
        base_model_path: "/models/wan".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_wan_new"),
        file_name: "wan_character.safetensors".to_owned(),
        created_at: "2026-05-26T00:00:00Z".to_owned(),
    })
    .expect("wan plan resolves");

    assert_eq!(plan.plan_version, TRAINING_PLAN_VERSION);
    assert_eq!(plan.target.kernel, "wan_lora");
    assert_eq!(plan.target.family, "wan-video");
    assert_eq!(plan.target.modality, TrainingModality::Video);
    assert!(!plan.dataset.items.is_empty());
}

#[test]
fn builtin_registry_exposes_wan_moe_targets() {
    let registry = builtin_training_targets();
    for (id, base_model) in [
        ("wan_t2v_14b_lora", "wan_2_2_t2v_14b"),
        ("wan_i2v_14b_lora", "wan_2_2_i2v_14b"),
    ] {
        let target = registry
            .targets
            .iter()
            .find(|target| target.id == id)
            .unwrap_or_else(|| panic!("{id} target present"));
        assert_eq!(target.modality, TrainingModality::Video);
        assert_eq!(target.output_kind, TrainingOutputKind::Lora);
        assert_eq!(target.family, "wan-video");
        assert_eq!(target.base_model, base_model);
        // Both A14B variants share the dual-expert MoE kernel.
        assert_eq!(target.kernel, "wan_moe_lora");
        assert_eq!(target.defaults.rank, 32);
        assert_eq!(target.defaults.resolution, 512);
        let expected = if id == "wan_t2v_14b_lora" {
            serde_json::json!(["lora", "lokr"])
        } else {
            serde_json::json!(["lora"])
        };
        assert_eq!(target.limits["networkTypes"], expected, "{id}");
    }
}

#[test]
fn unknown_training_fields_and_values_are_preserved() {
    let mut dataset = load_fixture("dataset.json");
    // Unknown enum value falls back to the string-enum `Unknown` variant.
    dataset["status"] = Value::String("curating".to_owned());
    // Unknown top-level and nested keys survive via flattened `extra` maps.
    dataset["futureField"] = Value::String("kept".to_owned());
    dataset["items"][0]["caption"]["futureCaptionField"] = Value::Bool(true);

    let typed: TrainingDataset =
        serde_json::from_value(dataset.clone()).expect("unknown dataset fields parse");
    let encoded = serde_json::to_value(typed).expect("unknown dataset fields serialize");

    assert_eq!(encoded, dataset);
}

#[test]
fn training_plan_fixture_pins_current_plan_version() {
    let plan = load_fixture("training-plan.json");
    assert_eq!(
        plan["planVersion"].as_u64(),
        Some(u64::from(TRAINING_PLAN_VERSION))
    );
}

/// sc-15036 — the plan's declared `outputKind` must describe THIS JOB, not what its target usually
/// produces. The Mage target's own `output_kind` is `Lora` and its `limits.networkTypes` offers
/// `"full"`, so the same target yields an adapter or an ~8 GB base checkpoint depending on one
/// config value. Copying `target.output_kind` — what the builder did before — makes every full
/// fine-tune declare itself a LoRA, which is precisely what routed it to the LoRA registrar and left
/// the checkpoint with no home.
///
/// Discriminating: two plans built from the SAME target and the SAME dataset, differing ONLY in
/// `advanced.networkType`, must declare DIFFERENT kinds. A builder that copies the target cannot
/// tell them apart, and neither can one that keys off the target id.
#[test]
fn build_training_plan_derives_the_output_kind_from_the_run_not_the_target() {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.base_model == "mage_flow_base")
        .expect("the Mage-Flow Base training target is registered");
    assert_eq!(
        target.output_kind,
        TrainingOutputKind::Lora,
        "fixture sanity: the target itself declares an adapter — that is the whole point"
    );
    assert!(
        target
            .limits
            .get("networkTypes")
            .and_then(|value| value.as_array())
            .is_some_and(|types| types.iter().any(|value| value.as_str() == Some("full"))),
        "fixture sanity: this target offers a full fine-tune"
    );

    let kind_for = |network_type: &str| {
        let mut config = target.defaults.clone();
        config.advanced.insert(
            "networkType".to_owned(),
            serde_json::Value::String(network_type.to_owned()),
        );
        build_training_plan(BuildTrainingPlan {
            job_id: "job_kind",
            target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: "out_kind",
            base_model_path: "/data/cache/huggingface/SceneWorks/Mage-Flow-Base".to_owned(),
            dataset_root: Path::new("/data/training/ds_abc123"),
            output_dir: Path::new("/data/models/finetunes/out_kind"),
            file_name: "out.safetensors".to_owned(),
            created_at: "2026-07-27T00:00:00Z".to_owned(),
        })
        .expect("plan resolves")
        .target
        .output_kind
    };

    assert_eq!(kind_for("lora"), TrainingOutputKind::Lora);
    assert_eq!(
        kind_for("lokr"),
        TrainingOutputKind::Lora,
        "a LoKr is an adapter too — only \"full\" changes the artifact class"
    );
    assert_eq!(
        kind_for("full"),
        TrainingOutputKind::BaseCheckpoint,
        "a full base fine-tune produces a base checkpoint; the plan is what the worker and the \
         completion registrar read to decide where the artifact goes"
    );
}

#[test]
fn build_training_plan_preserves_platform_effective_mage_full_requirements() {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.base_model == "mage_flow_base")
        .expect("Mage-Flow Base target");
    let mut config = target.defaults.clone();
    config
        .advanced
        .insert("networkType".to_owned(), json!("full"));

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_mage_full",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "finetune_mage",
        base_model_path: "/data/models/mage_flow_base".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/models/finetunes/finetune_mage"),
        file_name: "mage.safetensors".to_owned(),
        created_at: "2026-08-13T00:00:00Z".to_owned(),
    })
    .expect("Mage full plan resolves");

    assert_eq!(plan.target.output_kind, TrainingOutputKind::BaseCheckpoint);
    assert_eq!(plan.config.advanced["mixedPrecision"], json!("bf16"));
    assert_eq!(plan.config.advanced["gradientCheckpointing"], json!(true));
    assert_eq!(
        plan.provenance.config_snapshot["advanced"]["mixedPrecision"],
        json!("bf16"),
        "the MLX plan retains the platform catalog's submitted full-tune dtype"
    );
    assert_eq!(
        plan.provenance.config_snapshot["advanced"]["gradientCheckpointing"],
        json!(true),
        "the MLX plan retains its pre-existing checkpointing default"
    );

    let mut candle_target = target.clone();
    candle_target.defaults.advanced.insert(
        "fullFinetuneConfig".to_owned(),
        json!({
            "mixedPrecision": "f32",
            "gradientCheckpointing": false
        }),
    );

    for (field, value) in [
        ("mixedPrecision", json!("bf16")),
        ("gradientCheckpointing", json!(true)),
    ] {
        let mut invalid = candle_target.defaults.clone();
        invalid
            .advanced
            .insert("networkType".to_owned(), json!("full"));
        invalid
            .advanced
            .insert("mixedPrecision".to_owned(), json!("f32"));
        invalid
            .advanced
            .insert("gradientCheckpointing".to_owned(), json!(false));
        invalid.advanced.insert(field.to_owned(), value);
        let error = build_training_plan(BuildTrainingPlan {
            job_id: "job_mage_full_invalid",
            target: &candle_target,
            dataset: &dataset,
            config: invalid,
            preset: None,
            lora_id: "finetune_mage",
            base_model_path: "/data/models/mage_flow_base".to_owned(),
            dataset_root: Path::new("/data/training/ds_abc123"),
            output_dir: Path::new("/data/models/finetunes/finetune_mage"),
            file_name: "mage.safetensors".to_owned(),
            created_at: "2026-08-13T00:00:00Z".to_owned(),
        })
        .expect_err("unsupported Mage full execution config must fail before queueing");
        assert!(
            matches!(error, TrainingPlanError::InvalidConfig(_)),
            "{field}: {error:?}"
        );
    }
}

#[test]
fn build_training_plan_rejects_network_types_outside_the_target_surface() {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    for (target_id, network_type) in [
        ("wan_lora", "lokr"),
        ("wan_i2v_14b_lora", "lokr"),
        ("mage_flow_base_lora", "mystery"),
        ("sd3_5_large_lora", "full"),
    ] {
        let target = registry
            .targets
            .iter()
            .find(|target| target.id == target_id)
            .unwrap_or_else(|| panic!("target {target_id}"));
        let mut config = target.defaults.clone();
        config
            .advanced
            .insert("networkType".to_owned(), json!(network_type));
        let error = build_training_plan(BuildTrainingPlan {
            job_id: "job_network_reject",
            target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: "out",
            base_model_path: "/data/models/base".to_owned(),
            dataset_root: Path::new("/data/training/ds_abc123"),
            output_dir: Path::new("/data/loras/out"),
            file_name: "out.safetensors".to_owned(),
            created_at: "2026-08-13T00:00:00Z".to_owned(),
        })
        .expect_err("unadvertised network type must fail before queueing");
        let TrainingPlanError::InvalidConfig(detail) = error else {
            panic!("expected InvalidConfig for {target_id}/{network_type}");
        };
        assert!(detail.contains("does not support networkType"), "{detail}");
    }
}

#[test]
fn core_targets_preserve_the_preexisting_mlx_checkpoint_and_preview_defaults() {
    let registry = builtin_training_targets();
    for target_id in [
        "kolors_lora",
        "mage_flow_base_lora",
        "sd3_5_large_lora",
        "sd3_5_medium_lora",
    ] {
        let target = registry
            .targets
            .iter()
            .find(|target| target.id == target_id)
            .unwrap_or_else(|| panic!("target {target_id}"));
        assert_eq!(
            target.defaults.advanced["gradientCheckpointing"],
            json!(true),
            "{target_id} keeps its MLX checkpointing default; the API projects Candle defaults"
        );
        assert!(
            target.defaults.advanced["sampleEvery"]
                .as_u64()
                .is_some_and(|value| value > 0),
            "{target_id} keeps its MLX preview cadence; the API projects Candle defaults"
        );
    }
}

fn dataset_fixture() -> TrainingDataset {
    serde_json::from_value(load_fixture("dataset.json")).expect("dataset fixture parses")
}

#[test]
fn build_training_plan_resolves_paths_ids_and_provenance() {
    let mut dataset = dataset_fixture();
    dataset.items[0].caption.text = "a portrait of woman, soft light".to_owned();
    let registry = builtin_training_targets();
    let target = registry.targets.first().expect("a builtin target exists");
    let dataset_root = Path::new("/data/training/ds_abc123");
    let output_dir = Path::new("/data/loras/lora_new");
    let mut config = target.defaults.clone();
    config.trigger_word = Some("auroraStyle".to_owned());

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_test",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_new",
        base_model_path: "/data/cache/huggingface/Tongyi-MAI/Z-Image-Turbo".to_owned(),
        dataset_root,
        output_dir,
        file_name: "aurora_style.safetensors".to_owned(),
        created_at: "2026-05-21T00:00:00Z".to_owned(),
    })
    .expect("plan resolves");

    assert_eq!(plan.plan_version, TRAINING_PLAN_VERSION);
    assert_eq!(plan.job_id, "job_test");
    // The plan is self-referential so the kernel never needs the job record.
    assert_eq!(plan.provenance.source_job_id, "job_test");
    assert_eq!(plan.target.target_id, target.id);
    assert_eq!(plan.target.kernel, target.kernel);
    assert_eq!(
        plan.target.base_model_path,
        "/data/cache/huggingface/Tongyi-MAI/Z-Image-Turbo"
    );
    assert_eq!(plan.dataset.dataset_id, "ds_abc123");
    assert_eq!(plan.dataset.dataset_version, 3);
    assert_eq!(plan.dataset.items.len(), 2);
    assert_eq!(
        plan.dataset.items[0].caption,
        "auroraStyle, a portrait of woman, soft light"
    );
    // Item paths resolve under the dataset root with the host separator.
    let mut expected_image = dataset_root.to_path_buf();
    for component in Path::new("images/001.png").components() {
        expected_image.push(component);
    }
    assert_eq!(
        plan.dataset.items[0].image_path,
        expected_image.display().to_string()
    );
    assert_eq!(plan.output.lora_id, "lora_new");
    assert_eq!(plan.output.file_name, "aurora_style.safetensors");
    assert_eq!(plan.output.trigger_words, vec!["auroraStyle".to_owned()]);
    assert_eq!(plan.provenance.output_lora_id, "lora_new");
    assert_eq!(plan.provenance.dataset_version, 3);
}

#[test]
fn build_training_plan_omits_trigger_words_when_unset() {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry.targets.first().expect("a builtin target exists");

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_test",
        target,
        dataset: &dataset,
        config: target.defaults.clone(),
        preset: None,
        lora_id: "lora_new",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_new"),
        file_name: "aurora.safetensors".to_owned(),
        created_at: "2026-05-21T00:00:00Z".to_owned(),
    })
    .expect("plan resolves");

    assert!(plan.output.trigger_words.is_empty());
}

#[test]
fn build_training_plan_rejects_empty_dataset() {
    let mut dataset = dataset_fixture();
    dataset.items.clear();
    let registry = builtin_training_targets();
    let target = registry.targets.first().expect("a builtin target exists");

    let error = build_training_plan(BuildTrainingPlan {
        job_id: "job_test",
        target,
        dataset: &dataset,
        config: target.defaults.clone(),
        preset: None,
        lora_id: "lora_new",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_new"),
        file_name: "aurora.safetensors".to_owned(),
        created_at: "2026-05-21T00:00:00Z".to_owned(),
    })
    .expect_err("empty dataset is rejected");

    assert_eq!(error, TrainingPlanError::EmptyDataset);
}

/// Builds a plan from the Z-Image target defaults with the learning-rate
/// scheduler knobs overridden, so the validation tests exercise the same submit
/// path the API uses (`build_training_plan` → `validate_training_config`).
fn build_plan_with_lr_overrides(
    scheduler: Option<Value>,
    warmup: Option<Value>,
) -> Result<TrainingPlan, TrainingPlanError> {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("z_image_turbo_lora target present");
    let mut config = target.defaults.clone();
    match scheduler {
        Some(value) => {
            config.advanced.insert("lrScheduler".to_owned(), value);
        }
        None => {
            config.advanced.remove("lrScheduler");
        }
    }
    if let Some(value) = warmup {
        config.advanced.insert("lrWarmupSteps".to_owned(), value);
    }

    build_training_plan(BuildTrainingPlan {
        job_id: "job_lr",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_lr",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_lr"),
        file_name: "lr.safetensors".to_owned(),
        created_at: "2026-05-23T00:00:00Z".to_owned(),
    })
}

#[test]
fn build_training_plan_accepts_supported_lr_schedulers() {
    // Canonical names plus case/whitespace variants the validator normalizes.
    for name in ["constant", "linear", "cosine", "Cosine", " LINEAR "] {
        let plan = build_plan_with_lr_overrides(Some(Value::String(name.to_owned())), None)
            .unwrap_or_else(|error| panic!("scheduler {name:?} should be accepted: {error}"));
        // The submitted value is preserved verbatim in the resolved config; only
        // validation normalizes for the membership check.
        assert_eq!(
            plan.config.advanced["lrScheduler"],
            Value::String(name.to_owned())
        );
    }
}

#[test]
fn build_training_plan_rejects_unknown_lr_scheduler() {
    let error = build_plan_with_lr_overrides(Some(Value::String("warmup_cosine".to_owned())), None)
        .expect_err("unknown scheduler is rejected");
    match error {
        TrainingPlanError::InvalidConfig(detail) => {
            assert!(detail.contains("Unsupported lrScheduler"), "got: {detail}");
            assert!(
                detail.contains("warmup_cosine"),
                "names the bad value: {detail}"
            );
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn build_training_plan_rejects_non_string_lr_scheduler() {
    let error = build_plan_with_lr_overrides(Some(Value::Bool(true)), None)
        .expect_err("non-string scheduler is rejected");
    assert!(matches!(error, TrainingPlanError::InvalidConfig(_)));
}

#[test]
fn build_training_plan_validates_lr_warmup_steps() {
    // A non-negative warmup shorter than the 3000-step run is accepted.
    let plan =
        build_plan_with_lr_overrides(Some(Value::String("cosine".to_owned())), Some(json!(100)))
            .expect("warmup within range is accepted");
    assert_eq!(plan.config.advanced["lrWarmupSteps"], json!(100));

    // Warmup at or beyond the total step count is rejected.
    let error =
        build_plan_with_lr_overrides(Some(Value::String("cosine".to_owned())), Some(json!(5000)))
            .expect_err("warmup beyond the run is rejected");
    assert!(matches!(error, TrainingPlanError::InvalidConfig(_)));

    // Negative (and otherwise non-integer) warmup is rejected.
    let error = build_plan_with_lr_overrides(None, Some(json!(-5)))
        .expect_err("negative warmup is rejected");
    assert!(matches!(error, TrainingPlanError::InvalidConfig(_)));
}

#[test]
fn build_training_plan_rejects_invalid_config() {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry.targets.first().expect("a builtin target exists");
    let mut config = target.defaults.clone();
    config.rank = 0;

    let error = build_training_plan(BuildTrainingPlan {
        job_id: "job_test",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_new",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_new"),
        file_name: "aurora.safetensors".to_owned(),
        created_at: "2026-05-21T00:00:00Z".to_owned(),
    })
    .expect_err("zero rank is rejected");

    assert!(matches!(
        error,
        TrainingPlanError::TargetLimit(TrainingTargetLimitError::BelowMinimum { ref field, .. })
            if field == "rank"
    ));
}

fn build_plan_for_target(
    target: &sceneworks_core::training::TrainingTarget,
    config: TrainingConfig,
) -> Result<TrainingPlan, TrainingPlanError> {
    let dataset = dataset_fixture();
    build_training_plan(BuildTrainingPlan {
        job_id: "job_target_limit",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_target_limit",
        base_model_path: "/data/models/target".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_target_limit"),
        file_name: "target_limit.safetensors".to_owned(),
        created_at: "2026-08-26T00:00:00Z".to_owned(),
    })
}

fn set_training_numeric_field(config: &mut TrainingConfig, field: &str, value: u32) {
    match field {
        "rank" => config.rank = value,
        "alpha" => config.alpha = value,
        "steps" => config.steps = value,
        "batchSize" => config.batch_size = value,
        "resolutions" => config.resolution = value,
        other => panic!("missing test mutation for advertised numeric limit {other}"),
    }
}

/// The target registry is the public trainer-capability contract. Each table
/// row mutates exactly one submitted field; the shared plan builder is the same
/// boundary used by the API before it allocates a training job.
#[test]
fn trainer_capability_numeric_boundaries_are_typed_and_preserved() {
    let registry = builtin_training_targets();

    for target in &registry.targets {
        for (field, advertised) in target.limits.iter().filter(|(_, value)| {
            value
                .as_array()
                .is_some_and(|values| values.iter().any(Value::is_number))
        }) {
            let values = advertised
                .as_array()
                .expect("numeric target limit is an array")
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .expect("numeric target limit is an unsigned integer")
                })
                .collect::<Vec<_>>();
            assert!(
                matches!(
                    field.as_str(),
                    "rank" | "alpha" | "steps" | "batchSize" | "resolutions"
                ),
                "{}/{} is advertised but has no shared request-field validator",
                target.id,
                field
            );

            let mut boundary_values = values.clone();
            boundary_values.sort_unstable();
            boundary_values.dedup();
            for boundary in boundary_values {
                let boundary = u32::try_from(boundary).expect("builtin bound fits u32");
                let mut config = target.defaults.clone();
                set_training_numeric_field(&mut config, field, boundary);
                let plan = build_plan_for_target(target, config.clone()).unwrap_or_else(|error| {
                    panic!(
                        "{}/{}={boundary} should reach the trainer: {error}",
                        target.id, field
                    )
                });
                assert_eq!(
                    plan.config, config,
                    "{}/{} boundary must reach the intended trainer unchanged",
                    target.id, field
                );
            }

            let minimum = *values.iter().min().expect("numeric limit has a bound");
            let maximum = *values.iter().max().expect("numeric limit has a bound");
            for (direction, attempted) in [
                ("below", minimum.saturating_sub(1)),
                ("above", maximum.saturating_add(1)),
            ] {
                let mut config = target.defaults.clone();
                set_training_numeric_field(
                    &mut config,
                    field,
                    u32::try_from(attempted).expect("test value fits u32"),
                );
                let error = build_plan_for_target(target, config)
                    .expect_err("a value outside an advertised target capability must be rejected");
                match (field.as_str(), direction, error) {
                    (
                        "resolutions",
                        _,
                        TrainingPlanError::TargetLimit(
                            TrainingTargetLimitError::UnsupportedNumericValue {
                                field: error_field,
                                value,
                                ..
                            },
                        ),
                    ) => {
                        assert_eq!(error_field, "resolution");
                        assert_eq!(value, attempted);
                    }
                    (
                        _,
                        "below",
                        TrainingPlanError::TargetLimit(TrainingTargetLimitError::BelowMinimum {
                            field: error_field,
                            value,
                            minimum: error_minimum,
                        }),
                    ) => {
                        assert_eq!(error_field, field.as_str());
                        assert_eq!(value, attempted);
                        assert_eq!(error_minimum, minimum);
                    }
                    (
                        _,
                        "above",
                        TrainingPlanError::TargetLimit(TrainingTargetLimitError::AboveMaximum {
                            field: error_field,
                            value,
                            maximum: error_maximum,
                        }),
                    ) => {
                        assert_eq!(error_field, field.as_str());
                        assert_eq!(value, attempted);
                        assert_eq!(error_maximum, maximum);
                    }
                    (_, _, other) => panic!(
                        "{}/{} {direction} bound returned the wrong typed error: {other:?}",
                        target.id, field
                    ),
                }
            }
        }
    }
}

#[test]
fn trainer_capability_optimizer_and_unknown_numeric_limits_fail_closed() {
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .first()
        .expect("a builtin trainer target exists");

    let mut unsupported_optimizer = target.defaults.clone();
    unsupported_optimizer.optimizer = "not-advertised".to_owned();
    let error = build_plan_for_target(target, unsupported_optimizer)
        .expect_err("an optimizer absent from the advertised target list must fail");
    assert!(matches!(
        error,
        TrainingPlanError::TargetLimit(TrainingTargetLimitError::UnsupportedValue { ref field, .. })
            if field == "optimizer"
    ));

    let mut provider_target = target.clone();
    provider_target
        .limits
        .insert("futureNumericBudget".to_owned(), json!([1, 2]));
    let error = build_plan_for_target(&provider_target, provider_target.defaults.clone())
        .expect_err("a provider cannot advertise a numeric limit the shared validator ignores");
    assert!(matches!(
        error,
        TrainingPlanError::TargetLimit(TrainingTargetLimitError::UnsupportedNumericLimit {
            ref field
        }) if field == "futureNumericBudget"
    ));
}

/// Build a Z-Image plan whose `advanced` carries `weightNoiseSigma` (and optionally a network type).
fn build_plan_with_weight_noise(
    sigma: Value,
    network_type: Option<&str>,
) -> Result<TrainingPlan, TrainingPlanError> {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("z_image_turbo_lora target present");
    let mut config = target.defaults.clone();
    config.advanced.insert("weightNoiseSigma".to_owned(), sigma);
    if let Some(network_type) = network_type {
        config
            .advanced
            .insert("networkType".to_owned(), json!(network_type));
    }
    build_training_plan(BuildTrainingPlan {
        job_id: "job_wn",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_wn",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_wn"),
        file_name: "wn.safetensors".to_owned(),
        created_at: "2026-10-04T00:00:00Z".to_owned(),
    })
}

/// sc-24826 (epic 2123 E6): an in-range weight-noise sigma survives into the plan's config
/// snapshot verbatim; negative / above-limit / non-numeric values and the full-fine-tune
/// combination are field-level errors naming `weightNoiseSigma`.
#[test]
fn build_training_plan_validates_weight_noise_sigma_as_a_field_error() {
    for sigma in [
        json!(0),
        json!(WEIGHT_NOISE_SIGMA_SUGGESTED),
        json!(WEIGHT_NOISE_SIGMA_MAX),
    ] {
        let plan = build_plan_with_weight_noise(sigma.clone(), None)
            .unwrap_or_else(|error| panic!("{sigma} must be accepted: {error}"));
        assert_eq!(plan.config.advanced["weightNoiseSigma"], sigma);
    }
    for (sigma, network_type) in [
        (json!(-0.0001), None),
        (json!(WEIGHT_NOISE_SIGMA_MAX + 0.0001), None),
        (json!("0.0125"), None),
        (json!(WEIGHT_NOISE_SIGMA_SUGGESTED), Some("full")),
    ] {
        match build_plan_with_weight_noise(sigma.clone(), network_type) {
            Err(TrainingPlanError::InvalidField { field, .. }) => {
                assert_eq!(field, "weightNoiseSigma", "{sigma}/{network_type:?}")
            }
            other => panic!(
                "{sigma}/{network_type:?}: expected a weightNoiseSigma field error, got {other:?}"
            ),
        }
    }
}

/// sc-24826 (epic 2123 E6): the web form's weight-noise bound is the API's bound. The web
/// constant lives in `apps/web/src/training/trainingConfig.js`; read it so the two cannot drift.
#[test]
fn web_weight_noise_bound_matches_the_api_bound() {
    let source = include_str!("../../../apps/web/src/training/trainingConfig.js");
    let read = |name: &str| -> f64 {
        let prefix = format!("export const {name} = ");
        let line = source
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("{name} is not exported by trainingConfig.js"));
        line[prefix.len()..]
            .trim_end_matches(';')
            .trim()
            .parse()
            .unwrap_or_else(|error| panic!("{name} is not a numeric literal: {line} ({error})"))
    };
    assert_eq!(read("weightNoiseSigmaMax"), WEIGHT_NOISE_SIGMA_MAX);
    assert_eq!(
        read("weightNoiseSigmaSuggested"),
        WEIGHT_NOISE_SIGMA_SUGGESTED
    );
}

/// Build a Z-Image plan whose `advanced` carries the given extra keys.
fn build_plan_with_depth_advanced(
    extra: &[(&str, Value)],
) -> Result<TrainingPlan, TrainingPlanError> {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("z_image_turbo_lora target present");
    let mut config = target.defaults.clone();
    for (key, value) in extra {
        config.advanced.insert((*key).to_owned(), value.clone());
    }
    build_training_plan(BuildTrainingPlan {
        job_id: "job_da",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_da",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_da"),
        file_name: "da.safetensors".to_owned(),
        created_at: "2026-10-04T00:00:00Z".to_owned(),
    })
}

/// Build a Z-Image plan whose `advanced` carries the given subject-masked-loss keys.
fn build_plan_with_subject_mask_advanced(
    extra: &[(&str, Value)],
) -> Result<TrainingPlan, TrainingPlanError> {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("z_image_turbo_lora target present");
    let mut config = target.defaults.clone();
    for (key, value) in extra {
        config.advanced.insert((*key).to_owned(), value.clone());
    }
    build_training_plan(BuildTrainingPlan {
        job_id: "job_sm",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_sm",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_sm"),
        file_name: "sm.safetensors".to_owned(),
        created_at: "2026-10-04T00:00:00Z".to_owned(),
    })
}

/// sc-24828 (epic 2123 E6): subject-masked-loss keys — in-range values survive into the plan
/// verbatim (and resolve with defaults when absent); a non-boolean toggle, a background weight
/// outside [0, max], a subject weight outside (0, max] or a non-number are field-level errors
/// naming the offending key — even while the toggle is off.
#[test]
fn build_training_plan_validates_subject_mask_loss_as_field_errors() {
    use sceneworks_core::training::subject_mask_loss_weights;
    for extra in [
        vec![("subjectMaskLoss", json!(true))],
        vec![
            ("subjectMaskLoss", json!(true)),
            ("subjectMaskBackgroundWeight", json!(0)),
            ("subjectMaskSubjectWeight", json!(SUBJECT_MASK_WEIGHT_MAX)),
        ],
        vec![("subjectMaskLoss", json!(false))],
    ] {
        let plan = build_plan_with_subject_mask_advanced(&extra)
            .unwrap_or_else(|error| panic!("{extra:?} must be accepted: {error}"));
        for (key, value) in &extra {
            assert_eq!(&plan.config.advanced[*key], value);
        }
    }
    let resolve = |extra: &[(&str, Value)]| {
        let plan = build_plan_with_subject_mask_advanced(extra).unwrap();
        subject_mask_loss_weights(&plan.config.advanced).unwrap()
    };
    assert_eq!(resolve(&[]), None, "absent toggle = off");
    assert_eq!(
        resolve(&[("subjectMaskLoss", json!(true))]),
        Some((
            SUBJECT_MASK_BACKGROUND_WEIGHT_DEFAULT,
            SUBJECT_MASK_SUBJECT_WEIGHT_DEFAULT
        ))
    );
    assert_eq!(
        resolve(&[
            ("subjectMaskLoss", json!(true)),
            ("subjectMaskBackgroundWeight", json!(0.25)),
            ("subjectMaskSubjectWeight", json!(0.5)),
        ]),
        Some((0.25, 0.5))
    );

    for (extra, field) in [
        (vec![("subjectMaskLoss", json!("yes"))], "subjectMaskLoss"),
        (
            vec![("subjectMaskBackgroundWeight", json!(-0.01))],
            "subjectMaskBackgroundWeight",
        ),
        (
            vec![(
                "subjectMaskBackgroundWeight",
                json!(SUBJECT_MASK_WEIGHT_MAX + 0.01),
            )],
            "subjectMaskBackgroundWeight",
        ),
        (
            vec![
                ("subjectMaskLoss", json!(true)),
                ("subjectMaskSubjectWeight", json!(0)),
            ],
            "subjectMaskSubjectWeight",
        ),
        (
            vec![(
                "subjectMaskSubjectWeight",
                json!(SUBJECT_MASK_WEIGHT_MAX + 0.5),
            )],
            "subjectMaskSubjectWeight",
        ),
        (
            vec![
                ("subjectMaskLoss", json!(true)),
                ("subjectMaskSubjectWeight", json!("1")),
            ],
            "subjectMaskSubjectWeight",
        ),
    ] {
        match build_plan_with_subject_mask_advanced(&extra) {
            Err(TrainingPlanError::InvalidField { field: got, .. }) => {
                assert_eq!(got, field, "{extra:?}")
            }
            other => panic!("{extra:?}: expected a {field} field error, got {other:?}"),
        }
    }
}

/// sc-24828 (epic 2123 E6): the web form's subject-mask weight bound and defaults are the API's.
#[test]
fn web_subject_mask_bounds_match_the_api_bounds() {
    let source = include_str!("../../../apps/web/src/training/trainingConfig.js");
    let read = |name: &str| -> f64 {
        let prefix = format!("export const {name} = ");
        let line = source
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("{name} is not exported by trainingConfig.js"));
        line[prefix.len()..]
            .trim_end_matches(';')
            .trim()
            .parse()
            .unwrap_or_else(|error| panic!("{name} is not a numeric literal: {line} ({error})"))
    };
    assert_eq!(read("subjectMaskWeightMax"), SUBJECT_MASK_WEIGHT_MAX);
    assert_eq!(
        read("subjectMaskBackgroundWeightDefault"),
        SUBJECT_MASK_BACKGROUND_WEIGHT_DEFAULT
    );
    assert_eq!(
        read("subjectMaskSubjectWeightDefault"),
        SUBJECT_MASK_SUBJECT_WEIGHT_DEFAULT
    );
}

/// sc-2125 (epic 2123 E6): in-range depth-anchoring values survive into the plan verbatim; every
/// out-of-range / wrong-type value is a field-level error naming the offending key.
#[test]
fn build_training_plan_validates_depth_anchoring_as_field_errors() {
    use sceneworks_core::training::depth_anchoring::*;
    let good = [
        (
            DEPTH_ANCHORING_WEIGHT_KEY,
            json!(DEPTH_ANCHORING_WEIGHT_SUGGESTED),
        ),
        (DEPTH_ANCHORING_MODEL_KEY, json!("large")),
        (DEPTH_ANCHORING_MIN_T_KEY, json!(0.2)),
        (DEPTH_ANCHORING_MAX_T_KEY, json!(0.9)),
        (DEPTH_ANCHORING_EVERY_KEY, json!(DEPTH_ANCHORING_EVERY_MAX)),
    ];
    let plan = build_plan_with_depth_advanced(&good).expect("in-range depth anchoring is accepted");
    for (key, value) in &good {
        assert_eq!(&plan.config.advanced[*key], value, "{key}");
    }
    let settings = depth_anchoring_settings(&plan.config.advanced)
        .unwrap()
        .expect("weight > 0 is on");
    assert_eq!(settings.model, "large");
    assert_eq!(settings.every, DEPTH_ANCHORING_EVERY_MAX as u32);
    // Weight 0 (or absent) is off.
    let off = build_plan_with_depth_advanced(&[(DEPTH_ANCHORING_WEIGHT_KEY, json!(0))]).unwrap();
    assert_eq!(
        depth_anchoring_settings(&off.config.advanced).unwrap(),
        None
    );

    for (key, value, extra) in [
        (DEPTH_ANCHORING_WEIGHT_KEY, json!(-0.01), None),
        (
            DEPTH_ANCHORING_WEIGHT_KEY,
            json!(DEPTH_ANCHORING_WEIGHT_MAX + 0.001),
            None,
        ),
        (DEPTH_ANCHORING_WEIGHT_KEY, json!("0.1"), None),
        (DEPTH_ANCHORING_MODEL_KEY, json!("giant"), None),
        (DEPTH_ANCHORING_MIN_T_KEY, json!(-0.1), None),
        (DEPTH_ANCHORING_MAX_T_KEY, json!(1.5), None),
        (
            DEPTH_ANCHORING_MAX_T_KEY,
            json!(0.3),
            Some((DEPTH_ANCHORING_MIN_T_KEY, json!(0.6))),
        ),
        (DEPTH_ANCHORING_EVERY_KEY, json!(0), None),
        (
            DEPTH_ANCHORING_EVERY_KEY,
            json!(DEPTH_ANCHORING_EVERY_MAX + 1),
            None,
        ),
        (DEPTH_ANCHORING_EVERY_KEY, json!(2.5), None),
    ] {
        let mut advanced = vec![
            (DEPTH_ANCHORING_WEIGHT_KEY, json!(0.1)),
            (key, value.clone()),
        ];
        if let Some(pair) = extra {
            advanced.push(pair);
        }
        match build_plan_with_depth_advanced(&advanced) {
            Err(TrainingPlanError::InvalidField { field, .. }) => {
                assert_eq!(field, key, "{key}={value}")
            }
            other => panic!("{key}={value}: expected a field error naming {key}, got {other:?}"),
        }
    }
}

/// Read `export const <name> = <literal>;` from the web training config module.
fn web_training_const(name: &str) -> String {
    let source = include_str!("../../../apps/web/src/training/trainingConfig.js");
    let prefix = format!("export const {name} = ");
    let line = source
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("{name} is not exported by trainingConfig.js"));
    line[prefix.len()..].trim_end_matches(';').trim().to_owned()
}

/// sc-2125 (epic 2123 E6): the web form's depth-anchoring bounds are the API's bounds.
#[test]
fn web_depth_anchoring_bounds_match_the_api_bounds() {
    use sceneworks_core::training::depth_anchoring::*;
    let num = |name: &str| -> f64 {
        web_training_const(name)
            .parse()
            .unwrap_or_else(|error| panic!("{name} is not numeric ({error})"))
    };
    assert_eq!(num("depthAnchoringWeightMax"), DEPTH_ANCHORING_WEIGHT_MAX);
    assert_eq!(
        num("depthAnchoringWeightSuggested"),
        DEPTH_ANCHORING_WEIGHT_SUGGESTED
    );
    assert_eq!(
        num("depthAnchoringEveryMax"),
        DEPTH_ANCHORING_EVERY_MAX as f64
    );
    assert_eq!(
        num("depthAnchoringEveryDefault"),
        DEPTH_ANCHORING_EVERY_DEFAULT as f64
    );
    let models: Vec<String> =
        serde_json::from_str(&web_training_const("depthAnchoringModelOptions"))
            .expect("depthAnchoringModelOptions is a JSON-compatible string array");
    assert_eq!(models, DEPTH_ANCHORING_MODELS.to_vec());
    let workflows: Vec<String> =
        serde_json::from_str(&web_training_const("depthAnchoringNoVideoLtxWorkflows"))
            .expect("depthAnchoringNoVideoLtxWorkflows is a JSON-compatible string array");
    assert_eq!(workflows, DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS.to_vec());
}

/// sc-24830 review: submit refuses the depth-anchoring combinations the engine refuses even on an
/// advertising target — a Mage full base fine-tune and each LTX-2.5 workflow with no generated
/// video — as a `depthAnchoringWeight` field error; the adapter run and a video LTX-2.5 workflow
/// are admitted. Mutation: drop the `depth_anchoring_combination_refusal` check from
/// `validate_support` ⇒ the refused cases are admitted ⇒ red.
#[test]
fn depth_anchoring_refuses_full_finetune_and_no_video_ltx_workflows_at_submit() {
    use sceneworks_core::training::depth_anchoring::DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS;
    let registry = builtin_training_targets();
    let target = |id: &str| {
        registry
            .targets
            .iter()
            .find(|t| t.id == id)
            .unwrap_or_else(|| panic!("{id}"))
            .clone()
    };
    let with = |target: &sceneworks_core::training::TrainingTarget, extra: &[(&str, Value)]| {
        let mut config = target.defaults.clone();
        config
            .advanced
            .insert("depthAnchoringWeight".to_owned(), json!(0.1));
        for (k, v) in extra {
            config.advanced.insert((*k).to_owned(), v.clone());
        }
        validate_training_config_for_target(target, &config)
    };
    let refused = |r: Result<(), TrainingPlanError>, what: &str| match r {
        Err(TrainingPlanError::InvalidField { field, .. }) => {
            assert_eq!(field, "depthAnchoringWeight", "{what}")
        }
        other => panic!("{what}: expected a depthAnchoringWeight field error, got {other:?}"),
    };
    let mage = registry
        .targets
        .iter()
        .find(|t| t.base_model == "mage_flow_base")
        .expect("mage target")
        .clone();
    with(&mage, &[("networkType", json!("lora"))]).expect("Mage LoRA admits depth");
    refused(with(&mage, &[("networkType", json!("full"))]), "mage full");
    let ltx = target("ltx_2_5_video_lora");
    with(&ltx, &[("ltxWorkflow", json!("t2v_lora"))]).expect("LTX-2.5 t2v admits depth");
    for workflow in DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS {
        refused(with(&ltx, &[("ltxWorkflow", json!(workflow))]), workflow);
    }
    // The same workflow name on LTX-2.3 is not an LTX-2.5 bundle workflow: no extra refusal.
    let ltx23 = target("ltx_video_lora");
    assert!(
        sceneworks_core::training::depth_anchoring::depth_anchoring_combination_refusal(
            &ltx23,
            &ltx23.defaults
        )
        .is_none()
    );
}

/// sc-2125 / sc-24830: every auxiliary model depth anchoring loads — each family's tiny x0 decoder
/// and every Depth-Anything-V2 size — is a `componentOnly` utility entry in the shipped catalog
/// whose repo / revision / file are exactly the worker's resolution constants, so "install it from
/// the Models screen" installs the bytes the trainer loads. Mutation: drop or re-pin one decoder
/// entry in `builtin.models.jsonc` ⇒ red.
#[test]
fn depth_anchoring_aux_models_are_cataloged_at_the_loaded_revision() {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;
    use sceneworks_core::training::depth_anchoring::{DEPTH_ANYTHING_V2_MODELS, X0_DECODER_MODELS};

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc embedded");
    let catalog: Value = serde_json::from_str(&strip_jsonc_comments(raw)).unwrap();
    let models = catalog["models"].as_array().unwrap();
    for aux in X0_DECODER_MODELS
        .iter()
        .chain(DEPTH_ANYTHING_V2_MODELS.iter())
    {
        let entry = models
            .iter()
            .find(|m| m["id"] == aux.id)
            .unwrap_or_else(|| panic!("{} has no catalog entry", aux.id));
        assert_eq!(entry["type"], "utility", "{}", aux.id);
        assert_eq!(entry["componentOnly"], true, "{}", aux.id);
        let download = &entry["downloads"][0];
        assert_eq!(download["repo"], aux.repo, "{}", aux.id);
        assert_eq!(download["revision"], aux.revision, "{}", aux.id);
        assert!(
            download["files"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f == aux.file),
            "{} does not download {}",
            aux.id,
            aux.file
        );
    }
}

/// sc-24830: each native trainer's latent family maps to its x0 decoder — the matching tiny
/// decoder, or the trainer's own VAE for Mage-Flow (no tiny decoder exists for its 128-channel
/// latent) — and the two trainers that cannot decode x0 have none. Mutation: map Wan 2.2 TI2V-5B
/// (48-channel latent) to TAEW2.1 ⇒ red.
#[test]
fn every_depth_trainer_maps_to_its_latent_familys_x0_decoder() {
    use sceneworks_core::training::depth_anchoring::{
        x0_decoder_for_trainer, X0DecoderSource, TAEF1_MODEL, TAEF2_MODEL, TAELTX2_3_MODEL,
        TAESD3_MODEL, TAESDXL_MODEL, TAEW2_1_MODEL, TAEW2_2_MODEL,
    };
    use X0DecoderSource::{BaseModelVae, Catalog};
    for (trainer, expected) in [
        ("z_image_turbo", Catalog(&TAEF1_MODEL)),
        ("sdxl", Catalog(&TAESDXL_MODEL)),
        ("kolors", Catalog(&TAESDXL_MODEL)),
        ("sd3_5_large", Catalog(&TAESD3_MODEL)),
        ("sd3_5_medium", Catalog(&TAESD3_MODEL)),
        ("lens", Catalog(&TAEF2_MODEL)),
        ("krea_2_raw", Catalog(&TAEW2_1_MODEL)),
        ("anima_base", Catalog(&TAEW2_1_MODEL)),
        ("wan2_2_t2v_14b", Catalog(&TAEW2_1_MODEL)),
        ("wan2_2_i2v_14b", Catalog(&TAEW2_1_MODEL)),
        ("wan2_2_ti2v_5b", Catalog(&TAEW2_2_MODEL)),
        ("ltx_2_3", Catalog(&TAELTX2_3_MODEL)),
        ("ltx_2_5", Catalog(&TAELTX2_3_MODEL)),
        ("ltx_2_5_distilled", Catalog(&TAELTX2_3_MODEL)),
        ("mage_flow_base", BaseModelVae),
    ] {
        assert_eq!(x0_decoder_for_trainer(trainer), Some(expected), "{trainer}");
    }
    for trainer in ["krea_2_control", "unknown"] {
        assert_eq!(x0_decoder_for_trainer(trainer), None, "{trainer}");
    }
}

/// Build a Z-Image plan (target resolutions 512/768/1024) whose `advanced` carries
/// `resolutionBuckets`.
fn build_plan_with_buckets(buckets: Value) -> Result<TrainingPlan, TrainingPlanError> {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("z_image_turbo_lora target present");
    let mut config = target.defaults.clone();
    config
        .advanced
        .insert("resolutionBuckets".to_owned(), buckets);
    build_training_plan(BuildTrainingPlan {
        job_id: "job_rb",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_rb",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_rb"),
        file_name: "rb.safetensors".to_owned(),
        created_at: "2026-10-04T00:00:00Z".to_owned(),
    })
}

/// sc-2127 (epic 2123 E6): a well-formed bucket list survives into the plan verbatim; an empty
/// list, a non-positive / non-integer / over-limit repeat, an off-stride / zero / unsupported
/// resolution, a duplicate, too many rows and a non-list are all field-level errors naming
/// `resolutionBuckets`.
#[test]
fn build_training_plan_validates_resolution_buckets_as_a_field_error() {
    let good = json!([
        { "resolution": 512, "repeats": 16 },
        { "resolution": 768, "repeats": 4 },
        { "resolution": 1024, "repeats": RESOLUTION_BUCKET_REPEATS_MAX },
    ]);
    let plan = build_plan_with_buckets(good.clone()).expect("16:4:N buckets accepted");
    assert_eq!(plan.config.advanced["resolutionBuckets"], good);
    assert_eq!(
        sceneworks_core::training::training_max_resolution(&plan.config),
        1024
    );

    let too_many: Vec<Value> = (1..=RESOLUTION_BUCKETS_MAX as u64 + 1)
        .map(|i| json!({ "resolution": 32 * i, "repeats": 1 }))
        .collect();
    for bad in [
        json!([]),
        json!([{ "resolution": 512, "repeats": 0 }]),
        json!([{ "resolution": 512, "repeats": -2 }]),
        json!([{ "resolution": 512, "repeats": 1.5 }]),
        json!([{ "resolution": 512, "repeats": RESOLUTION_BUCKET_REPEATS_MAX + 1 }]),
        json!([{ "resolution": 512 }]),
        json!([{ "resolution": 0, "repeats": 1 }]),
        json!([{ "resolution": RESOLUTION_BUCKET_STRIDE * 16 + 16, "repeats": 1 }]),
        // On-stride but not a resolution this target trains at.
        json!([{ "resolution": 1536, "repeats": 1 }]),
        json!([{ "resolution": 512, "repeats": 1 }, { "resolution": 512, "repeats": 2 }]),
        json!(too_many),
        json!({ "resolution": 512, "repeats": 1 }),
    ] {
        match build_plan_with_buckets(bad.clone()) {
            Err(TrainingPlanError::InvalidField { field, .. }) => {
                assert_eq!(field, "resolutionBuckets", "{bad}")
            }
            other => panic!("{bad}: expected a resolutionBuckets field error, got {other:?}"),
        }
    }
}

/// sc-2127 (epic 2123 E7): memory admission sizes for the largest bucket, not `resolution`.
#[test]
fn training_max_resolution_is_the_largest_bucket() {
    let mut config = builtin_training_targets()
        .targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("target")
        .defaults
        .clone();
    config.resolution = 768;
    assert_eq!(
        sceneworks_core::training::training_max_resolution(&config),
        768
    );
    config.advanced.insert(
        "resolutionBuckets".to_owned(),
        json!([{ "resolution": 512, "repeats": 4 }, { "resolution": 1024, "repeats": 1 }]),
    );
    assert_eq!(
        sceneworks_core::training::training_max_resolution(&config),
        1024
    );
}

/// sc-2127 (epic 2123 E6): the web form's bucket limits are the API's. The web constants live in
/// `apps/web/src/training/trainingConfig.js`; read them so the two cannot drift.
#[test]
fn web_resolution_bucket_limits_match_the_api_limits() {
    let source = include_str!("../../../apps/web/src/training/trainingConfig.js");
    let read = |name: &str| -> u64 {
        let prefix = format!("export const {name} = ");
        let line = source
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("{name} is not exported by trainingConfig.js"));
        line[prefix.len()..]
            .trim_end_matches(';')
            .trim()
            .parse()
            .unwrap_or_else(|error| panic!("{name} is not an integer literal: {line} ({error})"))
    };
    assert_eq!(read("resolutionBucketsMax"), RESOLUTION_BUCKETS_MAX as u64);
    assert_eq!(
        read("resolutionBucketRepeatsMax"),
        RESOLUTION_BUCKET_REPEATS_MAX
    );
    assert_eq!(read("resolutionBucketStride"), RESOLUTION_BUCKET_STRIDE);
}

/// Build a Z-Image plan whose `advanced` carries `key = value` (and optionally a network type).
fn build_plan_with_advanced(
    key: &str,
    value: Value,
    network_type: Option<&str>,
) -> Result<TrainingPlan, TrainingPlanError> {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("z_image_turbo_lora target present");
    let mut config = target.defaults.clone();
    config.advanced.insert(key.to_owned(), value);
    if let Some(network_type) = network_type {
        config
            .advanced
            .insert("networkType".to_owned(), json!(network_type));
    }
    build_training_plan(BuildTrainingPlan {
        job_id: "job_gn",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_gn",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_gn"),
        file_name: "gn.safetensors".to_owned(),
        created_at: "2026-10-04T00:00:00Z".to_owned(),
    })
}

/// sc-24827 (epic 2123 E6): in-range gradient-noise eta/gamma survive into the plan verbatim;
/// negative / above-limit / non-numeric values, and eta with a full fine-tune, are field-level
/// errors naming the offending field.
#[test]
fn build_training_plan_validates_gradient_noise_as_field_errors() {
    for (key, value) in [
        (GRADIENT_NOISE_ETA_KEY, json!(0)),
        (GRADIENT_NOISE_ETA_KEY, json!(GRADIENT_NOISE_ETA_SUGGESTED)),
        (GRADIENT_NOISE_ETA_KEY, json!(GRADIENT_NOISE_ETA_MAX)),
        (GRADIENT_NOISE_GAMMA_KEY, json!(0)),
        (
            GRADIENT_NOISE_GAMMA_KEY,
            json!(GRADIENT_NOISE_GAMMA_DEFAULT),
        ),
        (GRADIENT_NOISE_GAMMA_KEY, json!(GRADIENT_NOISE_GAMMA_MAX)),
    ] {
        let plan = build_plan_with_advanced(key, value.clone(), None)
            .unwrap_or_else(|error| panic!("{key}={value} must be accepted: {error}"));
        assert_eq!(plan.config.advanced[key], value);
    }
    for (key, value, network_type) in [
        (GRADIENT_NOISE_ETA_KEY, json!(-0.0001), None),
        (
            GRADIENT_NOISE_ETA_KEY,
            json!(GRADIENT_NOISE_ETA_MAX + 0.0001),
            None,
        ),
        (GRADIENT_NOISE_ETA_KEY, json!("0.01"), None),
        (
            GRADIENT_NOISE_ETA_KEY,
            json!(GRADIENT_NOISE_ETA_SUGGESTED),
            Some("full"),
        ),
        (GRADIENT_NOISE_GAMMA_KEY, json!(-0.1), None),
        (
            GRADIENT_NOISE_GAMMA_KEY,
            json!(GRADIENT_NOISE_GAMMA_MAX + 0.0001),
            None,
        ),
        (GRADIENT_NOISE_GAMMA_KEY, json!("0.55"), None),
    ] {
        match build_plan_with_advanced(key, value.clone(), network_type) {
            Err(TrainingPlanError::InvalidField { field, .. }) => {
                assert_eq!(field, key, "{key}={value}/{network_type:?}")
            }
            other => panic!(
                "{key}={value}/{network_type:?}: expected a {key} field error, got {other:?}"
            ),
        }
    }
}

/// sc-24827 (epic 2123 E3): the control-branch target advertises neither adapter-noise flag (its
/// native trainer trains a full-weight branch and declares neither), so both knobs are refused at
/// submit time with a field-level error; a plain control request still validates.
#[test]
fn adapter_noise_is_refused_for_a_control_branch_target() {
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.output_kind == TrainingOutputKind::ControlBranch)
        .expect("a control-branch target is registered");
    validate_training_config_for_target(target, &target.defaults)
        .expect("the control target's defaults validate");
    for (key, value) in [
        (WEIGHT_NOISE_SIGMA_KEY, json!(WEIGHT_NOISE_SIGMA_SUGGESTED)),
        (GRADIENT_NOISE_ETA_KEY, json!(GRADIENT_NOISE_ETA_SUGGESTED)),
    ] {
        let mut config = target.defaults.clone();
        config.advanced.insert(key.to_owned(), value);
        match validate_training_config_for_target(target, &config) {
            Err(TrainingPlanError::InvalidField { field, message }) => {
                assert_eq!(field, key);
                assert!(message.contains("does not support"), "{message}");
            }
            other => panic!("{key}: expected a field error, got {other:?}"),
        }
        // Explicitly off is fine.
        let mut off = target.defaults.clone();
        off.advanced.insert(key.to_owned(), json!(0));
        validate_training_config_for_target(target, &off).expect("off validates");
    }
}

/// sc-24827 (epic 2123 E6): the web form's gradient-noise bounds and defaults are the API's. The
/// web constants live in `apps/web/src/training/trainingConfig.js`; read them so they cannot drift.
#[test]
fn web_gradient_noise_bounds_match_the_api_bounds() {
    let source = include_str!("../../../apps/web/src/training/trainingConfig.js");
    let read = |name: &str| -> f64 {
        let prefix = format!("export const {name} = ");
        let line = source
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("{name} is not exported by trainingConfig.js"));
        line[prefix.len()..]
            .trim_end_matches(';')
            .trim()
            .parse()
            .unwrap_or_else(|error| panic!("{name} is not a numeric literal: {line} ({error})"))
    };
    assert_eq!(read("gradientNoiseEtaMax"), GRADIENT_NOISE_ETA_MAX);
    assert_eq!(
        read("gradientNoiseEtaSuggested"),
        GRADIENT_NOISE_ETA_SUGGESTED
    );
    assert_eq!(read("gradientNoiseGammaMax"), GRADIENT_NOISE_GAMMA_MAX);
    assert_eq!(
        read("gradientNoiseGammaDefault"),
        GRADIENT_NOISE_GAMMA_DEFAULT
    );
}

/// sc-24826 review, updated for sc-24827: every LoRA/LoKr target advertises both adapter-noise
/// flags (all their trainers declare both, on both backends) and admits both knobs; a target that
/// does not advertise a flag gets a submit-time field error for that knob — never a refusal after
/// the job is queued.
#[test]
fn adapter_noise_is_gated_on_the_target_flags() {
    let registry = builtin_training_targets();
    let lora_targets: Vec<_> = registry
        .targets
        .iter()
        .filter(|target| target.output_kind != TrainingOutputKind::ControlBranch)
        .collect();
    assert!(!lora_targets.is_empty());
    for target in &lora_targets {
        assert!(target_supports_weight_noise(target), "{}", target.id);
        assert!(target_supports_gradient_noise(target), "{}", target.id);
        for (key, value) in [
            (WEIGHT_NOISE_SIGMA_KEY, json!(WEIGHT_NOISE_SIGMA_SUGGESTED)),
            (GRADIENT_NOISE_ETA_KEY, json!(GRADIENT_NOISE_ETA_SUGGESTED)),
        ] {
            let mut config = target.defaults.clone();
            config.advanced.insert(key.to_owned(), value);
            validate_training_config_for_target(target, &config)
                .unwrap_or_else(|e| panic!("{} {key}: {e}", target.id));
        }
    }

    // Withdraw one flag at a time: that knob becomes a field error, the other stays admitted.
    let z_image = lora_targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("Z-Image target");
    for (flag, key, value, other_key, other_value) in [
        (
            "supportsWeightNoise",
            WEIGHT_NOISE_SIGMA_KEY,
            json!(WEIGHT_NOISE_SIGMA_SUGGESTED),
            GRADIENT_NOISE_ETA_KEY,
            json!(GRADIENT_NOISE_ETA_SUGGESTED),
        ),
        (
            "supportsGradientNoise",
            GRADIENT_NOISE_ETA_KEY,
            json!(GRADIENT_NOISE_ETA_SUGGESTED),
            WEIGHT_NOISE_SIGMA_KEY,
            json!(WEIGHT_NOISE_SIGMA_SUGGESTED),
        ),
    ] {
        let mut target = (*z_image).clone();
        target.limits.remove(flag);
        let with = |key: &str, value: &Value| {
            let mut config = target.defaults.clone();
            config.advanced.insert(key.to_owned(), value.clone());
            validate_training_config_for_target(&target, &config)
        };
        with(key, &json!(0)).unwrap_or_else(|e| panic!("{flag}: off must pass: {e}"));
        match with(key, &value) {
            Err(TrainingPlanError::InvalidField { field, .. }) => assert_eq!(field, key, "{flag}"),
            other => panic!("{flag}: expected a {key} field error, got {other:?}"),
        }
        with(other_key, &other_value)
            .unwrap_or_else(|e| panic!("{flag}: the other technique must stay admitted: {e}"));
    }
}

/// sc-2127 review: every builtin target advertises `supportsResolutionBuckets` except LTX-2.5
/// (prepared latent packs, no spatial edge) — one catalog serves both platforms since sc-24827 — and a bucket list on a target that
/// does not advertise it is a submit-time `resolutionBuckets` field error, never a queued job.
#[test]
fn resolution_buckets_are_refused_at_submit_on_targets_that_do_not_advertise_them() {
    let registry = builtin_training_targets();
    let withheld: Vec<&str> = registry
        .targets
        .iter()
        .filter(|target| !target_supports_resolution_buckets(target))
        .map(|target| target.id.as_str())
        .collect();
    assert_eq!(withheld, ["ltx_2_5_video_lora"]);

    let buckets = |target: &sceneworks_core::training::TrainingTarget| {
        let allowed = target.limits["resolutions"]
            .as_array()
            .expect("resolutions")[0]
            .clone();
        let mut config = target.defaults.clone();
        config.advanced.insert(
            "resolutionBuckets".to_owned(),
            json!([{ "resolution": allowed, "repeats": 2 }]),
        );
        validate_training_config_for_target(target, &config)
    };
    let by_id = |id: &str| {
        registry
            .targets
            .iter()
            .find(|target| target.id == id)
            .unwrap_or_else(|| panic!("{id} target present"))
            .clone()
    };
    buckets(&by_id("z_image_turbo_lora")).expect("Z-Image admits buckets");
    match buckets(&by_id("ltx_2_5_video_lora")) {
        Err(TrainingPlanError::InvalidField { field, message }) => {
            assert_eq!(field, "resolutionBuckets");
            assert!(
                message.contains("does not support multi-resolution buckets"),
                "{message}"
            );
        }
        other => panic!("expected a resolutionBuckets field error, got {other:?}"),
    }
}

/// sc-2125 / sc-24830: only targets whose trainer declares depth anchoring advertise it — every
/// LoRA target except the Krea ControlNet branch, on both platforms — and an enabled
/// depth weight on any other target is a submit-time `depthAnchoringWeight` field error.
/// Mutation: drop `depth_anchoring::validate_support` from `validate_training_config_for_target` ⇒
/// the Krea ControlNet case is accepted ⇒ red.
#[test]
fn depth_anchoring_is_refused_at_submit_on_targets_that_do_not_advertise_it() {
    use sceneworks_core::training::depth_anchoring::target_supports_depth_anchoring;
    let registry = builtin_training_targets();
    let by_id = |id: &str| {
        registry
            .targets
            .iter()
            .find(|target| target.id == id)
            .unwrap_or_else(|| panic!("{id} target present"))
            .clone()
    };
    let with_weight = |target: &sceneworks_core::training::TrainingTarget, weight: Value| {
        let mut config = target.defaults.clone();
        config
            .advanced
            .insert("depthAnchoringWeight".to_owned(), weight);
        validate_training_config_for_target(target, &config)
    };

    let unsupported: Vec<&str> = registry
        .targets
        .iter()
        .filter(|target| !target_supports_depth_anchoring(target))
        .map(|target| target.id.as_str())
        .collect();
    assert_eq!(unsupported, ["krea_2_control"]);

    for target in &registry.targets {
        with_weight(target, json!(0)).unwrap_or_else(|e| panic!("{}: {e}", target.id));
        match (
            with_weight(target, json!(0.1)),
            target_supports_depth_anchoring(target),
        ) {
            (Ok(()), true) => {}
            (Err(TrainingPlanError::InvalidField { field, .. }), false) => {
                assert_eq!(field, "depthAnchoringWeight", "{}", target.id)
            }
            (other, supported) => panic!(
                "{} (advertised {supported}): unexpected depth validation {other:?}",
                target.id
            ),
        }
    }
    let z_image = by_id("z_image_turbo_lora");
    with_weight(&z_image, json!(0.1)).expect("Z-Image admits depth anchoring");
}

/// sc-24828 review: every LoRA target advertises subject-masked loss except LTX-2.5 (prepared
/// latent bundles) and the Krea ControlNet branch; mask loss on a non-advertising target is a
/// submit-time `subjectMaskLoss` field error (never queued), and off is always admitted. The flag
/// is static per target (served identically on both platforms).
#[test]
fn subject_mask_loss_is_refused_at_submit_on_targets_that_do_not_advertise_it() {
    use sceneworks_core::training::target_supports_subject_mask_loss;
    let registry = builtin_training_targets();
    let with_mask = |target: &sceneworks_core::training::TrainingTarget, on: bool| {
        let mut config = target.defaults.clone();
        config
            .advanced
            .insert("subjectMaskLoss".to_owned(), json!(on));
        validate_training_config_for_target(target, &config)
    };
    let unsupported: Vec<&str> = registry
        .targets
        .iter()
        .filter(|target| !target_supports_subject_mask_loss(target))
        .map(|target| target.id.as_str())
        .collect();
    assert_eq!(unsupported, ["krea_2_control", "ltx_2_5_video_lora"]);
    for target in &registry.targets {
        with_mask(target, false).unwrap_or_else(|e| panic!("{} off: {e}", target.id));
        match (
            with_mask(target, true),
            target_supports_subject_mask_loss(target),
        ) {
            (Ok(()), true) => {}
            (Err(TrainingPlanError::InvalidField { field, .. }), false) => {
                assert_eq!(field, "subjectMaskLoss", "{}", target.id)
            }
            (other, supported) => panic!(
                "{} (advertised {supported}): unexpected mask-loss validation {other:?}",
                target.id
            ),
        }
    }
}

/// sc-24833 (epic 2123 E6): in-range VAE-anchor / E-LatentLPIPS values survive into the plan
/// verbatim and parse with upstream's defaults (window `[0, 0.5]`, additive `every = 1`); every
/// out-of-range / wrong-type value is a field-level error naming the offending key. Mutation: widen
/// the weight bound in `latent_loss_settings` to `0.0..=f64::MAX` ⇒ the over-max weight is accepted
/// ⇒ red.
#[test]
fn build_training_plan_validates_latent_perceptual_losses_as_field_errors() {
    use sceneworks_core::training::latent_perceptual::*;
    for spec in LATENT_LOSSES {
        let plan = build_plan_with_depth_advanced(&[(spec.weight_key, json!(0.5))])
            .unwrap_or_else(|e| panic!("{}: {e}", spec.weight_key));
        assert_eq!(
            latent_loss_settings(&spec, &plan.config.advanced).unwrap(),
            Some(LatentLossSettings {
                weight: 0.5,
                min_t: 0.0,
                max_t: LATENT_LOSS_MAX_T_DEFAULT,
                every: LATENT_LOSS_EVERY_DEFAULT,
            }),
            "{} defaults",
            spec.weight_key
        );
        let good = [
            (spec.weight_key, json!(LATENT_LOSS_WEIGHT_MAX)),
            (spec.min_t_key, json!(0.1)),
            (spec.max_t_key, json!(0.9)),
            (spec.every_key, json!(LATENT_LOSS_EVERY_MAX)),
        ];
        let plan = build_plan_with_depth_advanced(&good).expect("in-range values are accepted");
        for (key, value) in &good {
            assert_eq!(&plan.config.advanced[*key], value, "{key}");
        }
        let off = build_plan_with_depth_advanced(&[(spec.weight_key, json!(0))]).unwrap();
        assert_eq!(
            latent_loss_settings(&spec, &off.config.advanced).unwrap(),
            None
        );

        for (key, value, extra) in [
            (spec.weight_key, json!(-0.01), None),
            (spec.weight_key, json!(LATENT_LOSS_WEIGHT_MAX + 0.001), None),
            (spec.weight_key, json!("0.5"), None),
            (spec.min_t_key, json!(-0.1), None),
            (spec.max_t_key, json!(1.5), None),
            (
                spec.max_t_key,
                json!(0.3),
                Some((spec.min_t_key, json!(0.6))),
            ),
            (spec.every_key, json!(0), None),
            (spec.every_key, json!(LATENT_LOSS_EVERY_MAX + 1), None),
            (spec.every_key, json!(1.5), None),
        ] {
            let mut advanced = vec![(spec.weight_key, json!(0.5)), (key, value.clone())];
            if let Some(pair) = extra {
                advanced.push(pair);
            }
            match build_plan_with_depth_advanced(&advanced) {
                Err(TrainingPlanError::InvalidField { field, .. }) => {
                    assert_eq!(field, key, "{key}={value}")
                }
                other => {
                    panic!("{key}={value}: expected a field error naming {key}, got {other:?}")
                }
            }
        }
    }
}

/// sc-24833 (epic 2123 E6): the web form's latent-perceptual bounds are the API's bounds.
#[test]
fn web_latent_perceptual_bounds_match_the_api_bounds() {
    use sceneworks_core::training::latent_perceptual::*;
    let num = |name: &str| -> f64 {
        web_training_const(name)
            .parse()
            .unwrap_or_else(|error| panic!("{name} is not numeric ({error})"))
    };
    for prefix in ["vaeAnchor", "latentLpips"] {
        assert_eq!(
            num(&format!("{prefix}WeightMax")),
            LATENT_LOSS_WEIGHT_MAX,
            "{prefix}"
        );
        assert_eq!(
            num(&format!("{prefix}WeightSuggested")),
            LATENT_LOSS_WEIGHT_SUGGESTED,
            "{prefix}"
        );
        assert_eq!(
            num(&format!("{prefix}MaxTDefault")),
            LATENT_LOSS_MAX_T_DEFAULT,
            "{prefix}"
        );
        assert_eq!(
            num(&format!("{prefix}EveryDefault")),
            LATENT_LOSS_EVERY_DEFAULT as f64,
            "{prefix}"
        );
        assert_eq!(
            num(&format!("{prefix}EveryMax")),
            LATENT_LOSS_EVERY_MAX as f64,
            "{prefix}"
        );
    }
}

/// sc-24833: every auxiliary model the latent-perceptual losses load is a `componentOnly` utility
/// entry in the shipped catalog whose repo / revision / file are exactly the worker's resolution
/// constants; the FLUX.2 VAE component is the VAE inside the FLUX.2 [dev] entry's own repo at its
/// pinned revision, so an installed FLUX.2 [dev] tier satisfies it too.
#[test]
fn latent_perceptual_aux_models_are_cataloged_at_the_loaded_revision() {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;
    use sceneworks_core::training::latent_perceptual::{
        ELATENTLPIPS_MODELS, FLUX2_VAE_MODEL, FLUX2_VAE_TIER_DIRS,
    };

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc embedded");
    let catalog: Value = serde_json::from_str(&strip_jsonc_comments(raw)).unwrap();
    let models = catalog["models"].as_array().unwrap();
    for aux in std::iter::once(&FLUX2_VAE_MODEL).chain(ELATENTLPIPS_MODELS.iter()) {
        let entry = models
            .iter()
            .find(|m| m["id"] == aux.id)
            .unwrap_or_else(|| panic!("{} has no catalog entry", aux.id));
        assert_eq!(entry["type"], "utility", "{}", aux.id);
        assert_eq!(entry["componentOnly"], true, "{}", aux.id);
        let download = &entry["downloads"][0];
        assert_eq!(download["repo"], aux.repo, "{}", aux.id);
        assert_eq!(download["revision"], aux.revision, "{}", aux.id);
        assert!(
            download["files"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f == aux.file),
            "{} does not download {}",
            aux.id,
            aux.file
        );
    }
    assert!(FLUX2_VAE_MODEL.file.starts_with(FLUX2_VAE_TIER_DIRS[0]));
    // Every FLUX.2 [dev] tier download is the same repo + revision, and each tier directory is one
    // the worker accepts the VAE from.
    let dev = models
        .iter()
        .find(|m| m["id"] == "flux2_dev")
        .expect("flux2_dev entry");
    for download in dev["downloads"].as_array().unwrap() {
        assert_eq!(download["repo"], FLUX2_VAE_MODEL.repo);
        assert_eq!(download["revision"], FLUX2_VAE_MODEL.revision);
        let tier = download["variant"].as_str().unwrap();
        assert!(
            FLUX2_VAE_TIER_DIRS.contains(&format!("{tier}/vae").as_str()),
            "{tier}"
        );
    }
}

/// sc-24833 (S1 mechanism): exactly the targets in `VAE_ANCHOR_TARGETS` / `LATENT_LPIPS_TARGETS`
/// advertise each loss (static per target, identical on both platforms); the VAE-anchor set is the
/// depth-anchoring set (same x0-decoder seam); an enabled weight on any other target is a
/// submit-time field error on that weight key. Mutation: drop `latent_perceptual::validate_support`
/// from `validate_training_config_for_target` ⇒ the Krea control case is accepted ⇒ red.
#[test]
fn latent_perceptual_losses_are_refused_at_submit_on_targets_that_do_not_advertise_them() {
    use sceneworks_core::training::depth_anchoring::target_supports_depth_anchoring;
    use sceneworks_core::training::latent_perceptual::{
        target_supports, LATENT_LOSSES, LATENT_LPIPS_TARGETS, VAE_ANCHOR_TARGETS,
    };
    let registry = builtin_training_targets();
    let by_id = |id: &str| {
        registry
            .targets
            .iter()
            .find(|target| target.id == id)
            .unwrap_or_else(|| panic!("{id} target present"))
            .clone()
    };
    let sorted = |mut v: Vec<String>| {
        v.sort();
        v
    };
    let depth: Vec<String> = sorted(
        registry
            .targets
            .iter()
            .filter(|target| target_supports_depth_anchoring(target))
            .map(|target| target.id.clone())
            .collect(),
    );
    for (spec, table) in [
        (LATENT_LOSSES[0], &VAE_ANCHOR_TARGETS[..]),
        (LATENT_LOSSES[1], &LATENT_LPIPS_TARGETS[..]),
    ] {
        let advertising = sorted(
            registry
                .targets
                .iter()
                .filter(|target| target_supports(&spec, target))
                .map(|target| target.id.clone())
                .collect(),
        );
        assert_eq!(
            advertising,
            sorted(table.iter().map(|s| s.to_string()).collect()),
            "{}",
            spec.support_limit
        );
        if spec.weight_key == LATENT_LOSSES[0].weight_key {
            assert_eq!(
                advertising, depth,
                "the VAE anchor rides the depth-anchoring decoders"
            );
        }
        let with_weight = |target: &sceneworks_core::training::TrainingTarget, weight: Value| {
            let mut config = target.defaults.clone();
            config.advanced.insert(spec.weight_key.to_owned(), weight);
            validate_training_config_for_target(target, &config)
        };
        with_weight(&by_id("z_image_turbo_lora"), json!(0.5)).expect("Z-Image admits it");
        let unsupported = if spec.weight_key == LATENT_LOSSES[0].weight_key {
            "krea_2_control"
        } else {
            "krea_2_raw_lora"
        };
        let target = by_id(unsupported);
        with_weight(&target, json!(0)).unwrap_or_else(|e| panic!("{}: {e}", target.id));
        match with_weight(&target, json!(0.5)) {
            Err(TrainingPlanError::InvalidField { field, .. }) => {
                assert_eq!(field, spec.weight_key, "{}", target.id)
            }
            other => panic!(
                "{}: expected a {} field error, got {other:?}",
                target.id, spec.weight_key
            ),
        }
    }
}

/// sc-24833: the submit-time combination refusals mirror the engine's — neither latent loss on a
/// full fine-tune (Mage-Flow advertises both `full` and the VAE anchor), and the decoded-x0 VAE
/// anchor not on an LTX-2.5 workflow that generates no video; a video-generating workflow is
/// admitted. Mutation: drop the `latent_loss_combination_refusal` check from `validate_support` ⇒
/// the full-fine-tune case is accepted ⇒ red.
#[test]
fn latent_perceptual_losses_refuse_the_combinations_the_engine_refuses() {
    use sceneworks_core::training::depth_anchoring::DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS;
    use sceneworks_core::training::latent_perceptual::{
        latent_loss_combination_refusal, VAE_ANCHOR,
    };
    let registry = builtin_training_targets();
    let by_id = |id: &str| {
        registry
            .targets
            .iter()
            .find(|target| target.id == id)
            .unwrap_or_else(|| panic!("{id} target present"))
            .clone()
    };
    let field_error = |target: &sceneworks_core::training::TrainingTarget,
                       extra: &[(&str, Value)]| {
        let mut config = target.defaults.clone();
        config
            .advanced
            .insert(VAE_ANCHOR.weight_key.to_owned(), json!(0.5));
        for (k, v) in extra {
            config.advanced.insert((*k).to_owned(), v.clone());
        }
        validate_training_config_for_target(target, &config)
    };
    let mage = by_id("mage_flow_base_lora");
    field_error(&mage, &[]).expect("Mage-Flow LoRA admits the VAE anchor");
    match field_error(&mage, &[("networkType", json!("full"))]) {
        Err(TrainingPlanError::InvalidField { field, message }) => {
            assert_eq!(field, VAE_ANCHOR.weight_key);
            assert!(message.contains("full fine-tune"), "{message}");
        }
        other => panic!("expected a full-fine-tune refusal, got {other:?}"),
    }
    let ltx = by_id("ltx_2_5_video_lora");
    for workflow in DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS {
        let mut config = ltx.defaults.clone();
        config
            .advanced
            .insert("ltxWorkflow".to_owned(), json!(workflow));
        assert!(
            latent_loss_combination_refusal(&VAE_ANCHOR, &ltx, &config).is_some(),
            "{workflow}"
        );
    }
    let mut config = ltx.defaults.clone();
    config
        .advanced
        .insert("ltxWorkflow".to_owned(), json!("t2v_lora"));
    assert_eq!(
        latent_loss_combination_refusal(&VAE_ANCHOR, &ltx, &config),
        None
    );
}
