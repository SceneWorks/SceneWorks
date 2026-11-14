//! SW-6936: the FLUX.1 conditional routes (XLabs IP-Adapter, ControlNet, PuLID) compile their render
//! plans before load. The shared fixture here pins the base weights directory each route binds into
//! its plan: an external `modelPath` wins over the hub snapshot, and the snapshot's selected packed or
//! dense tier is the one loaded.

use super::super::ImageRequest;
use crate::test_env::EnvVars;
use crate::PathBuf;
use crate::Settings;
use crate::WorkerResult;
use serde_json::json;
use serde_json::Value;

/// One installed hub snapshot of `repo` holding packed `q4`/`q8` and dense `bf16` tiers, plus an
/// app-managed external base directory, under an isolated data dir and hub cache.
struct ConditionalBaseFixture {
    _root: tempfile::TempDir,
    _hub_env: EnvVars,
    settings: Settings,
    snapshot: PathBuf,
    external: PathBuf,
}

impl ConditionalBaseFixture {
    fn new(repo: &str) -> Self {
        let root = tempfile::tempdir().expect("fixture root");
        let hub = root.path().join("hub");
        let hub_env = EnvVars::set(&[
            ("HF_HUB_CACHE", hub.to_str().expect("utf-8 hub dir")),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ]);
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let repo_dir = hub.join(format!("models--{}", repo.replace('/', "--")));
        let snapshot = repo_dir.join("snapshots").join(revision);
        for tier in ["q4", "q8", "bf16"] {
            write_weights(&snapshot.join(tier));
        }
        std::fs::create_dir_all(repo_dir.join("refs")).expect("refs dir");
        std::fs::write(repo_dir.join("refs").join("main"), revision).expect("refs/main");

        let external = root.path().join("models").join("external-flux1-base");
        write_weights(&external);

        let mut settings = crate::test_env::offline_settings();
        settings.data_dir = root.path().to_path_buf();
        Self {
            _root: root,
            _hub_env: hub_env,
            settings,
            snapshot,
            external,
        }
    }
}

fn write_weights(dir: &std::path::Path) {
    let file = dir
        .join("transformer")
        .join("diffusion_pytorch_model.safetensors");
    std::fs::create_dir_all(file.parent().expect("component dir")).expect("component dir");
    std::fs::write(file, b"weights").expect("weights");
}

fn canonical(path: Option<PathBuf>) -> Option<PathBuf> {
    path.map(|path| std::fs::canonicalize(&path).unwrap_or(path))
}

/// Assert one conditional route's base resolution: an external `modelPath` (advanced or manifest)
/// wins over an installed hub snapshot and is used as-is, and without one the route descends into
/// the selected packed (`q4`/`q8`) or dense (`bf16`) tier of the snapshot.
pub(in crate::image_jobs) fn assert_conditional_base_resolution(
    route: &str,
    payload: Value,
    repo: &str,
    resolve: impl Fn(&ImageRequest, &Settings) -> WorkerResult<Option<PathBuf>>,
) {
    let fixture = ConditionalBaseFixture::new(repo);
    let request = |advanced: Value, model_path_in_manifest: Option<&str>| {
        let mut payload = payload.clone();
        let object = payload.as_object_mut().expect("payload object");
        let mut manifest = json!({ "repo": repo, "mlx": { "standardTierLayout": true } });
        if let Some(path) = model_path_in_manifest {
            manifest["modelPath"] = json!(path);
        }
        object.insert("modelManifestEntry".to_owned(), manifest);
        let mut merged = object
            .get("advanced")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        merged.extend(advanced.as_object().cloned().unwrap_or_default());
        object.insert("advanced".to_owned(), Value::Object(merged));
        ImageRequest::from_payload(object)
    };
    let external = fixture.external.display().to_string();

    for (label, req) in [
        (
            "advanced.modelPath",
            request(json!({ "modelPath": external, "mlxQuantize": 4 }), None),
        ),
        (
            "manifest modelPath",
            request(json!({ "mlxQuantize": 8 }), Some(&external)),
        ),
    ] {
        let resolved = resolve(&req, &fixture.settings)
            .unwrap_or_else(|error| panic!("{route}: {label} must resolve: {error}"));
        assert_eq!(
            canonical(resolved),
            canonical(Some(fixture.external.clone())),
            "{route}: an external {label} must win over the installed hub snapshot, untiered"
        );
    }

    for (bits, tier) in [(4, "q4"), (8, "q8"), (0, "bf16")] {
        let resolved = resolve(
            &request(json!({ "mlxQuantize": bits }), None),
            &fixture.settings,
        )
        .unwrap_or_else(|error| panic!("{route}: mlxQuantize {bits} must resolve: {error}"));
        assert_eq!(
            canonical(resolved),
            canonical(Some(fixture.snapshot.join(tier))),
            "{route}: mlxQuantize {bits} must select the {tier} tier of the hub snapshot"
        );
    }
}

/// The MLX FLUX.1 ControlNet and XLabs IP-Adapter routes resolve their base through the shared
/// generic resolver.
#[cfg(target_os = "macos")]
#[test]
fn mlx_control_and_ip_base_prefers_external_model_path_then_the_selected_tier() {
    for (model, repo) in [
        ("flux_dev", "SceneWorks/flux1-dev-mlx"),
        ("flux_schnell", "SceneWorks/flux1-schnell-mlx"),
    ] {
        assert_conditional_base_resolution(
            &format!("FLUX.1 conditional base (MLX, {model})"),
            json!({
                "projectId": "p", "model": model, "prompt": "a portrait", "count": 1,
                "mode": "character_image", "referenceAssetId": "asset_1"
            }),
            repo,
            super::super::artifact_resolution::resolve_weights_dir,
        );
    }
}

/// SW-6936: every MLX FLUX.1 conditional route compiles through the registry before load. A request
/// the provider cannot serve — a reference or control without its overlay in the load contract,
/// several references, or IP-Adapter mixed with control — is refused there as a payload error the
/// job reports, and the matching well-formed request compiles.
#[cfg(target_os = "macos")]
#[test]
fn mlx_conditional_preload_refusals_are_payload_errors() {
    use gen_core::{Conditioning, ControlKind, GenerationRequest, LoadSpec, WeightsSource};

    let root = tempfile::tempdir().expect("fixture root");
    let dir = |name: &str| {
        let path = root.path().join(name);
        std::fs::create_dir_all(&path).expect("fixture dir");
        path
    };
    let base = dir("base");
    let ip_adapter = dir("ip_adapter");
    let control_file = root.path().join("control.safetensors");
    std::fs::write(&control_file, b"control").expect("control fixture");
    let identity = gen_core::IdentityWeights {
        encoder: Some(WeightsSource::File(root.path().join("pulid.safetensors"))),
        eva: Some(WeightsSource::File(root.path().join("eva.safetensors"))),
        face_dir: Some(WeightsSource::Dir(dir("face"))),
    };

    let image = gen_core::Image {
        width: 512,
        height: 512,
        pixels: vec![96; 512 * 512 * 3],
    };
    let reference = Conditioning::Reference {
        image: image.clone(),
        strength: Some(0.7),
    };
    let control = Conditioning::Control {
        image,
        kind: ControlKind::Pose,
        scale: Some(0.7),
    };
    let request = |conditioning: Vec<Conditioning>| GenerationRequest {
        prompt: "a portrait".to_owned(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(7),
        steps: Some(4),
        guidance: Some(3.5),
        conditioning,
        ..Default::default()
    };
    let base_spec = LoadSpec::new(WeightsSource::Dir(base.clone()));
    let ip_spec = base_spec
        .clone()
        .with_ip_adapter(WeightsSource::Dir(ip_adapter));
    let control_spec = base_spec
        .clone()
        .with_control(WeightsSource::File(control_file.clone()));
    let mut pulid_spec = base_spec.clone();
    pulid_spec.identity = Some(identity);
    let registry = crate::inference_runtime::media();

    for (case, engine, spec, conditioning) in [
        (
            "XLabs IP-Adapter",
            "flux1_dev",
            &ip_spec,
            vec![reference.clone()],
        ),
        (
            "ControlNet",
            "flux1_dev_control",
            &control_spec,
            vec![control.clone()],
        ),
        ("PuLID", "pulid_flux", &pulid_spec, vec![reference.clone()]),
    ] {
        let plan = super::compile_render_plan(registry, engine, spec, &request(conditioning))
            .unwrap_or_else(|error| panic!("{case}: a well-formed request must compile: {error}"));
        assert!(
            plan.is_some(),
            "{case}: the route must have a pre-load plan"
        );
    }

    let mixed_spec = ip_spec
        .clone()
        .with_control(WeightsSource::File(control_file));
    for (case, engine, spec, conditioning) in [
        (
            "IP reference without its overlay",
            "flux1_dev",
            &base_spec,
            vec![reference.clone()],
        ),
        (
            "IP with multiple references",
            "flux1_dev",
            &ip_spec,
            vec![reference.clone(), reference.clone()],
        ),
        (
            "mixed IP-Adapter + control",
            "flux1_dev",
            &mixed_spec,
            vec![reference.clone(), control.clone()],
        ),
        (
            "control without its overlay",
            "flux1_dev_control",
            &base_spec,
            vec![control.clone()],
        ),
        (
            "control mixed with a reference",
            "flux1_dev_control",
            &control_spec,
            vec![control.clone(), reference.clone()],
        ),
        (
            "PuLID without its identity overlay",
            "pulid_flux",
            &base_spec,
            vec![reference.clone()],
        ),
        (
            "PuLID with multiple references",
            "pulid_flux",
            &pulid_spec,
            vec![reference.clone(), reference.clone()],
        ),
    ] {
        match super::compile_render_plan(registry, engine, spec, &request(conditioning)) {
            Err(crate::WorkerError::InvalidPayload(message)) => assert!(
                message.contains("cannot compile render plan before load"),
                "{case}: {message}"
            ),
            other => panic!("{case}: expected a pre-load payload refusal, got {other:?}"),
        }
    }
}

/// SW-6936: an MLX XLabs IP-Adapter Hires.fix request compiles both passes before load — the first
/// pass and a refinement pass that consumes the first pass's output stretched to the target size —
/// and the refinement refuses a first-pass image of any other size as a payload error. Base FLUX.1
/// has no reference input without the IP-Adapter, so its refinement pass is refused before load.
#[cfg(target_os = "macos")]
#[test]
fn mlx_ip_hires_fix_compiles_both_passes_before_load() {
    use gen_core::{Conditioning, GenerationRequest, LoadSpec, WeightsSource};

    let root = tempfile::tempdir().expect("fixture root");
    let base = root.path().join("base");
    let ip_adapter = root.path().join("ip_adapter");
    std::fs::create_dir_all(&base).expect("base dir");
    std::fs::create_dir_all(&ip_adapter).expect("ip dir");
    let base_spec = LoadSpec::new(WeightsSource::Dir(base));
    let ip_spec = base_spec
        .clone()
        .with_ip_adapter(WeightsSource::Dir(ip_adapter));
    let request = |side: u32, steps: u32, conditioning: Vec<Conditioning>| GenerationRequest {
        prompt: "a portrait".to_owned(),
        width: side,
        height: side,
        count: 1,
        seed: Some(7),
        steps: Some(steps),
        guidance: Some(3.5),
        conditioning,
        ..Default::default()
    };
    let reference = Conditioning::Reference {
        image: gen_core::Image {
            width: 512,
            height: 512,
            pixels: vec![96; 512 * 512 * 3],
        },
        strength: Some(0.7),
    };
    let hires = super::HiresFixPlan {
        width: 1024,
        height: 1024,
        steps: 6,
        guidance: Some(3.5),
        true_cfg: None,
        provider_reference_strength: 0.45,
    };
    let first = request(512, 4, vec![reference]);
    let refine = request(1024, 6, Vec::new());
    let registry = crate::inference_runtime::media();

    let (first_plan, refine_plan) =
        super::compile_hires_render_plans(registry, "flux1_dev", &ip_spec, &first, &refine, &hires)
            .expect("an IP-Adapter Hires.fix request compiles both passes before load");
    assert_eq!(first_plan.pass_index(), 0);
    assert_eq!(first_plan.output_size(), (512, 512));
    assert_eq!(refine_plan.pass_index(), 1);
    assert_eq!(refine_plan.output_size(), (1024, 1024));
    assert_eq!(
        refine_plan.produced_reference(),
        Some(gen_core::ProducedReference {
            from_pass: 0,
            width: 1024,
            height: 1024,
            strength: Some(0.45),
        })
    );

    let stretched = |side: u32| gen_core::Image {
        width: side,
        height: side,
        pixels: vec![0; (side * side * 3) as usize],
    };
    super::check_produced_reference(&refine_plan, &stretched(1024))
        .expect("the declared first-pass size is accepted");
    match super::check_produced_reference(&refine_plan, &stretched(768)) {
        Err(crate::WorkerError::InvalidPayload(message)) => {
            assert!(message.contains("expects a 1024x1024"), "{message}")
        }
        other => panic!("a mismatched first-pass size must be a payload error, got {other:?}"),
    }
    assert!(
        matches!(
            super::check_produced_reference(&first_plan, &stretched(1024)),
            Err(crate::WorkerError::InvalidPayload(_))
        ),
        "a plan that consumes no produced reference cannot take one"
    );

    match super::compile_hires_render_plans(
        registry,
        "flux1_dev",
        &base_spec,
        &request(512, 4, Vec::new()),
        &refine,
        &hires,
    ) {
        Err(crate::WorkerError::InvalidPayload(message)) => assert!(
            message.contains("cannot compile render plan before load"),
            "{message}"
        ),
        other => panic!("base FLUX.1 Hires.fix must be refused before load, got {other:?}"),
    }
}
