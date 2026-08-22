//! One fresh-process execution entrypoint for the serialized epic-20738 CUDA terminal profile.
//!
//! The Node controller owns immutable provisioning, repository identity, VRAM sampling, and receipt
//! upload. This module owns the reviewable native work: exact-tier load specs, deterministic inputs,
//! production registry loads, generation, and non-degenerate output checks. It has one ignored test;
//! the controller invokes it exactly once for each of the 19 checked-in cells with `--test-threads=1`.

use std::path::{Path, PathBuf};

use gen_core::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, Quant,
    ReplacementMode, WeightsSource,
};
use serde::Deserialize;
use serde_json::{json, Value};

use super::smoke_support::{image_std, save_png, DEGENERATE_STD_FLOOR_DEFAULT};

const ENABLE_ENV: &str = "SCENEWORKS_ENABLE_EPIC_20738_TERMINAL_CUDA";
const CELL_ENV: &str = "SCENEWORKS_EPIC_20738_CELL_FILE";
const OUTPUT_ENV: &str = "SCENEWORKS_EPIC_20738_OUTPUT_DIR";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Artifact {
    id: String,
    role: String,
    repository: String,
    revision: String,
    subdirectory: String,
    root: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Request {
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    #[serde(default)]
    frames: Option<u32>,
    #[serde(default)]
    fps: Option<u32>,
    #[serde(default)]
    guidance: Option<f32>,
    #[serde(default)]
    true_cfg: Option<f32>,
    #[serde(default)]
    sampler: Option<String>,
    #[serde(default)]
    reference_pairs: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Cell {
    id: String,
    kind: String,
    model_id: String,
    engine_id: String,
    requested_tier: String,
    #[serde(default)]
    capability: Option<String>,
    request: Request,
    artifacts: Vec<Artifact>,
}

/// The terminal profile selects immutable packed roots. Only SDXL's production loader still uses
/// an advisory Q4 selector, and Candle SCAIL carries the exact selected tier as its production load
/// hint. Chroma, FLUX, and LTX consume the selected packed root without asking the provider to
/// quantize it again. Keep this policy pure and shared by both the load-spec builder and the runtime
/// receipt so the evidence cannot describe a different load than the one the worker requested.
fn family_load_spec_quant_bits(kind: &str, engine_id: &str, requested_tier: &str) -> Option<u64> {
    match (kind, engine_id) {
        ("image", "chroma1_base" | "chroma1_flash" | "chroma1_hd")
        | ("image", "flux1_dev" | "flux1_schnell")
        | ("ltx", "ltx_2_3_distilled") => None,
        ("sdxlOpenPose", "sdxl") => Some(4),
        ("scail2", "scail2_14b") => match requested_tier {
            "q4" => Some(4),
            "q8" => Some(8),
            tier => panic!("unreviewed terminal SCAIL tier {tier}"),
        },
        _ => panic!("unreviewed terminal load-quant family {kind}/{engine_id}"),
    }
}

/// The reviewed terminal requests deliberately leave the production request-memory policy at its
/// default. In particular, FLUX on an ample CUDA device is resident/non-streamed and therefore does
/// not create bounded-transformer sidecars. This helper makes that absence reviewable and rejects a
/// new family before it can silently inherit a different harness policy.
fn family_generation_memory(kind: &str, engine_id: &str) -> Option<gen_core::GenerationMemory> {
    match (kind, engine_id) {
        ("image", "chroma1_base" | "chroma1_flash" | "chroma1_hd")
        | ("image", "flux1_dev" | "flux1_schnell")
        | ("ltx", "ltx_2_3_distilled")
        | ("sdxlOpenPose", "sdxl")
        | ("scail2", "scail2_14b") => None,
        _ => panic!("unreviewed terminal request-memory family {kind}/{engine_id}"),
    }
}

fn primary_load_spec(cell: &Cell, primary: &Artifact) -> LoadSpec {
    let spec = LoadSpec::new(WeightsSource::Dir(primary.root.clone()))
        .with_resolved_route(cell.model_id.clone());
    match family_load_spec_quant_bits(&cell.kind, &cell.engine_id, &cell.requested_tier) {
        None => spec,
        Some(4) => spec.with_quant(Quant::Q4),
        Some(8) => spec.with_quant(Quant::Q8),
        Some(bits) => panic!("unreviewed terminal load quant q{bits}"),
    }
}

fn required_env(name: &str) -> String {
    std::env::var(name)
        .unwrap_or_else(|_| panic!("{name} is required by the terminal CUDA controller"))
        .trim()
        .to_owned()
}

fn artifact<'a>(cell: &'a Cell, role: &str) -> &'a Artifact {
    let matches = cell
        .artifacts
        .iter()
        .filter(|artifact| artifact.role == role)
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "{} needs exactly one {role} artifact",
        cell.id
    );
    matches[0]
}

fn exact_file(root: &Path, name: &str, label: &str) -> PathBuf {
    let path = root.join(name);
    assert!(path.is_file(), "{label} is missing at {}", path.display());
    path
}

fn collect_quant_bits(value: &Value, in_quant: bool, bits: &mut Vec<u64>) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                let quant = in_quant || key.to_ascii_lowercase().contains("quant");
                if quant && key == "bits" {
                    if let Some(value) = value.as_u64() {
                        bits.push(value);
                    }
                }
                collect_quant_bits(value, quant, bits);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_quant_bits(value, in_quant, bits);
            }
        }
        _ => {}
    }
}

fn visit_json(root: &Path, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).expect("read artifact directory") {
        let path = entry.expect("read artifact entry").path();
        if path.is_dir() {
            visit_json(&path, files);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            files.push(path);
        }
    }
}

/// Bind execution to the requested packed tier rather than trusting the directory label alone.
fn verify_exact_packed_tier(cell: &Cell, primary: &Artifact) -> u64 {
    assert_eq!(primary.subdirectory, cell.requested_tier);
    assert_eq!(
        primary.root.file_name().and_then(|name| name.to_str()),
        Some(cell.requested_tier.as_str()),
        "{} selected root must be the exact requested tier subdirectory",
        cell.id
    );
    let expected = match cell.requested_tier.as_str() {
        "q4" => 4,
        "q8" => 8,
        other => panic!("terminal profile cannot request dense or unknown tier {other}"),
    };
    let mut files = Vec::new();
    visit_json(&primary.root, &mut files);
    let mut bits = Vec::new();
    for file in files {
        let Ok(body) = std::fs::read_to_string(&file) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&body) else {
            continue;
        };
        collect_quant_bits(&value, false, &mut bits);
    }
    assert!(
        bits.contains(&expected),
        "{} has no packed quantization marker for q{expected}; refusing a path-only or dense fallback claim",
        cell.id
    );
    assert!(
        bits.iter().all(|bits| *bits == expected),
        "{} contains mixed packed-tier markers {:?}",
        cell.id,
        bits
    );
    expected
}

fn pattern_image(width: u32, height: u32, salt: u8) -> Image {
    Image {
        width,
        height,
        pixels: (0..width * height * 3)
            .map(|index| ((index.wrapping_mul(17) + u32::from(salt) * 29) % 251) as u8)
            .collect(),
    }
}

fn pose_control_image(width: u32, height: u32) -> Image {
    let mut pixels = vec![0_u8; width as usize * height as usize * 3];
    let mut line = |start: (i32, i32), end: (i32, i32), color: [u8; 3]| {
        let (mut x, mut y) = start;
        let dx = (end.0 - start.0).abs();
        let sx = if start.0 < end.0 { 1 } else { -1 };
        let dy = -(end.1 - start.1).abs();
        let sy = if start.1 < end.1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            for offset_y in -2..=2 {
                for offset_x in -2..=2 {
                    let px = x + offset_x;
                    let py = y + offset_y;
                    if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                        let index = (py as usize * width as usize + px as usize) * 3;
                        pixels[index..index + 3].copy_from_slice(&color);
                    }
                }
            }
            if (x, y) == end {
                break;
            }
            let doubled = 2 * error;
            if doubled >= dy {
                error += dy;
                x += sx;
            }
            if doubled <= dx {
                error += dx;
                y += sy;
            }
        }
    };
    let scale = |x: u32, y: u32| ((x * width / 512) as i32, (y * height / 512) as i32);
    for (start, end, color) in [
        (scale(256, 82), scale(256, 138), [255, 255, 255]),
        (scale(190, 158), scale(322, 158), [255, 128, 0]),
        (scale(256, 138), scale(256, 296), [255, 255, 0]),
        (scale(190, 158), scale(145, 238), [0, 255, 0]),
        (scale(145, 238), scale(116, 320), [0, 255, 255]),
        (scale(322, 158), scale(370, 225), [0, 128, 255]),
        (scale(370, 225), scale(404, 292), [0, 0, 255]),
        (scale(214, 296), scale(298, 296), [255, 0, 255]),
        (scale(214, 296), scale(194, 396), [255, 0, 0]),
        (scale(194, 396), scale(176, 482), [128, 0, 255]),
        (scale(298, 296), scale(330, 390), [255, 64, 128]),
        (scale(330, 390), scale(366, 472), [128, 255, 0]),
    ] {
        line(start, end, color);
    }
    Image {
        width,
        height,
        pixels,
    }
}

fn mask_image(width: u32, height: u32, pair: usize) -> Image {
    const COLORS: [[u8; 3]; 6] = [
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
    ];
    let color = COLORS[pair];
    Image {
        width,
        height,
        pixels: color.repeat((width * height) as usize),
    }
}

fn image_output(output: GenerationOutput, context: &str) -> Image {
    match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("{context} expected Images, got {other:?}"),
    }
}

fn generate_image(cell: &Cell, primary: &Artifact, output: &Path) -> Value {
    let spec = primary_load_spec(cell, primary);
    let generator = crate::inference_runtime::load(&cell.engine_id, &spec)
        .unwrap_or_else(|error| panic!("{} load failed: {error}", cell.id));
    let request = GenerationRequest {
        prompt: "a red fox in a sunlit forest, detailed, coherent".to_owned(),
        negative_prompt: cell
            .engine_id
            .starts_with("chroma1_")
            .then(|| "blurry, distorted, low quality".to_owned()),
        width: cell.request.width,
        height: cell.request.height,
        count: 1,
        seed: Some(cell.request.seed),
        steps: Some(cell.request.steps),
        guidance: cell.request.guidance,
        true_cfg: cell.request.true_cfg,
        sampler: cell.request.sampler.clone(),
        memory: family_generation_memory(&cell.kind, &cell.engine_id),
        ..Default::default()
    };
    let image = image_output(
        generator
            .generate(&request, &mut |_| {})
            .expect("terminal image generation"),
        &cell.id,
    );
    assert_eq!(
        (image.width, image.height),
        (cell.request.width, cell.request.height)
    );
    let std = image_std(&image);
    assert!(
        std > DEGENERATE_STD_FLOOR_DEFAULT,
        "{} output is degenerate (std {std})",
        cell.id
    );
    let path = output.join("output.png");
    save_png(&image, &path);
    json!({ "kind": "image", "width": image.width, "height": image.height, "pixelStd": std })
}

fn generate_sdxl_openpose(cell: &Cell, primary: &Artifact, bits: u64, output: &Path) -> Value {
    assert_eq!(
        bits, 4,
        "the reviewed five-backbone OpenPose profile is q4-only"
    );
    let control = artifact(cell, "control");
    let tokenizer_l = artifact(cell, "tokenizer_clip_l");
    let tokenizer_bigg = artifact(cell, "tokenizer_clip_bigg");
    let vae = artifact(cell, "vae_fp16_fix");
    let spec = primary_load_spec(cell, primary)
        .with_control(WeightsSource::File(exact_file(
            &control.root,
            "diffusion_pytorch_model.safetensors",
            "SDXL OpenPose ControlNet",
        )))
        .with_component(
            "tokenizer_clip_l",
            WeightsSource::File(exact_file(
                &tokenizer_l.root,
                "tokenizer.json",
                "SDXL CLIP-L tokenizer",
            )),
        )
        .with_component(
            "tokenizer_clip_bigg",
            WeightsSource::File(exact_file(
                &tokenizer_bigg.root,
                "tokenizer.json",
                "SDXL CLIP-bigG tokenizer",
            )),
        )
        .with_component(
            "vae_fp16_fix",
            WeightsSource::File(exact_file(
                &vae.root,
                "diffusion_pytorch_model.safetensors",
                "SDXL fp16 VAE",
            )),
        )
        .with_resolved_route(cell.model_id.clone());
    // Keep this mutable construction explicit: review can see there is exactly one control and no
    // adapter, IP-Adapter, identity, PiD, extra-control, or dense alternate.
    assert!(spec.extra_controls.is_empty());
    assert!(spec.adapters.is_empty());
    let generator = crate::inference_runtime::load("sdxl", &spec)
        .unwrap_or_else(|error| panic!("{} SDXL OpenPose load failed: {error}", cell.id));
    // Use an actual deterministic whole-body stick pose, not arbitrary pixels carrying Pose metadata.
    let control_image = pose_control_image(cell.request.width, cell.request.height);
    save_png(&control_image, &output.join("input-openpose.png"));
    let request = GenerationRequest {
        prompt: "a dancer holding the supplied pose, studio photograph".to_owned(),
        width: cell.request.width,
        height: cell.request.height,
        count: 1,
        seed: Some(cell.request.seed),
        steps: Some(cell.request.steps),
        guidance: cell.request.guidance,
        sampler: cell.request.sampler.clone(),
        conditioning: vec![Conditioning::Control {
            image: control_image,
            kind: ControlKind::Pose,
            scale: Some(1.0),
        }],
        memory: family_generation_memory(&cell.kind, &cell.engine_id),
        ..Default::default()
    };
    let image = image_output(
        generator
            .generate(&request, &mut |_| {})
            .expect("terminal SDXL OpenPose generation"),
        &cell.id,
    );
    let std = image_std(&image);
    assert!(
        std > DEGENERATE_STD_FLOOR_DEFAULT,
        "{} pose output is degenerate",
        cell.id
    );
    save_png(&image, &output.join("output.png"));
    json!({ "kind": "sdxlOpenPose", "width": image.width, "height": image.height, "pixelStd": std })
}

fn video_stats(frames: &[Image]) -> (f64, f64) {
    assert!(!frames.is_empty(), "video output is empty");
    let mean_std = frames.iter().map(image_std).sum::<f64>() / frames.len() as f64;
    let first = &frames[0];
    let last = frames.last().expect("last frame");
    assert_eq!(first.pixels.len(), last.pixels.len());
    let motion = first
        .pixels
        .iter()
        .zip(&last.pixels)
        .map(|(left, right)| (f64::from(*left) - f64::from(*right)).abs())
        .sum::<f64>()
        / first.pixels.len() as f64;
    (mean_std, motion)
}

fn save_video_witnesses(frames: &[Image], output: &Path) {
    save_png(&frames[0], &output.join("frame-first.png"));
    save_png(
        frames.last().expect("last frame"),
        &output.join("frame-last.png"),
    );
}

fn generate_scail2(cell: &Cell, primary: &Artifact, output: &Path) -> Value {
    let pair_count = cell.request.reference_pairs.unwrap_or(1);
    assert!((1..=6).contains(&pair_count));
    assert_eq!(
        cell.capability.as_deref() == Some("multiReference"),
        pair_count == 6
    );
    let frames = cell.request.frames.expect("SCAIL2 frames");
    let mut conditioning = Vec::with_capacity(pair_count * 2 + 1);
    for pair in 0..pair_count {
        // The engine contract is order-sensitive: each Reference is immediately followed by its Mask.
        let reference = pattern_image(cell.request.width, cell.request.height, pair as u8);
        let mask = mask_image(cell.request.width, cell.request.height, pair);
        save_png(
            &reference,
            &output.join(format!("input-reference-{}.png", pair + 1)),
        );
        save_png(
            &mask,
            &output.join(format!("input-reference-mask-{}.png", pair + 1)),
        );
        conditioning.push(Conditioning::Reference {
            image: reference,
            strength: None,
        });
        conditioning.push(Conditioning::Mask { image: mask });
    }
    let driving = (0..frames)
        .map(|frame| pattern_image(cell.request.width, cell.request.height, frame as u8))
        .collect::<Vec<_>>();
    let masks = (0..frames)
        .map(|frame| mask_image(cell.request.width, cell.request.height, frame as usize % 6))
        .collect::<Vec<_>>();
    save_png(&driving[0], &output.join("input-driving-first.png"));
    save_png(
        driving.last().expect("last driving frame"),
        &output.join("input-driving-last.png"),
    );
    save_png(&masks[0], &output.join("input-driving-mask-first.png"));
    conditioning.push(Conditioning::ControlClip {
        frames: driving,
        mask: masks,
        masking_strength: 1.0,
        start_frame: 0,
        mode: ReplacementMode::default(),
    });
    assert_eq!(conditioning.len(), pair_count * 2 + 1);
    let spec = primary_load_spec(cell, primary);
    let exact_tier_hint = match cell.requested_tier.as_str() {
        "q4" => matches!(spec.quantize.as_ref(), Some(Quant::Q4)),
        "q8" => matches!(spec.quantize.as_ref(), Some(Quant::Q8)),
        _ => false,
    };
    assert!(
        exact_tier_hint,
        "Candle SCAIL2 must retain its exact production tier hint"
    );
    let generator = crate::inference_runtime::load("scail2_14b", &spec)
        .unwrap_or_else(|error| panic!("{} SCAIL2 load failed: {error}", cell.id));
    let request = GenerationRequest {
        prompt: "the reference characters perform the driving motion".to_owned(),
        width: cell.request.width,
        height: cell.request.height,
        frames: Some(frames),
        fps: cell.request.fps.or(Some(16)),
        steps: Some(cell.request.steps),
        seed: Some(cell.request.seed),
        conditioning,
        video_mode: Some("animation".to_owned()),
        memory: family_generation_memory(&cell.kind, &cell.engine_id),
        ..Default::default()
    };
    let output_value = generator
        .generate(&request, &mut |_| {})
        .expect("terminal SCAIL2 generation");
    let output_frames = match output_value {
        GenerationOutput::Video { frames, .. } => frames,
        other => panic!("{} expected Video, got {other:?}", cell.id),
    };
    let (mean_std, motion) = video_stats(&output_frames);
    assert!(
        mean_std > DEGENERATE_STD_FLOOR_DEFAULT,
        "{} output is degenerate",
        cell.id
    );
    assert!(motion > 0.01, "{} first/last frames do not move", cell.id);
    save_video_witnesses(&output_frames, output);
    json!({ "kind": "scail2", "frames": output_frames.len(), "referencePairs": pair_count, "meanFrameStd": mean_std, "firstLastMeanAbsDelta": motion })
}

fn generate_ltx(cell: &Cell, primary: &Artifact, output: &Path) -> Value {
    let text_encoder = artifact(cell, "text_encoder");
    let spec = primary_load_spec(cell, primary)
        .with_text_encoder(WeightsSource::Dir(text_encoder.root.clone()));
    assert!(
        spec.quantize.is_none(),
        "LTX prepacked q8 must load with quant=None"
    );
    let generator = crate::inference_runtime::load("ltx_2_3_distilled", &spec)
        .unwrap_or_else(|error| panic!("{} LTX load failed: {error}", cell.id));
    let request = GenerationRequest {
        prompt: "a red fox walks through a sunlit forest, cinematic".to_owned(),
        width: cell.request.width,
        height: cell.request.height,
        frames: cell.request.frames,
        fps: cell.request.fps,
        steps: Some(cell.request.steps),
        seed: Some(cell.request.seed),
        memory: family_generation_memory(&cell.kind, &cell.engine_id),
        ..Default::default()
    };
    let generated = generator
        .generate(&request, &mut |_| {})
        .expect("terminal LTX generation");
    let frames = match generated {
        GenerationOutput::Video { frames, .. } => frames,
        other => panic!("{} expected Video, got {other:?}", cell.id),
    };
    let (mean_std, motion) = video_stats(&frames);
    assert!(
        mean_std > DEGENERATE_STD_FLOOR_DEFAULT,
        "{} output is degenerate",
        cell.id
    );
    assert!(motion > 0.01, "{} first/last frames do not move", cell.id);
    save_video_witnesses(&frames, output);
    json!({ "kind": "ltx", "frames": frames.len(), "meanFrameStd": mean_std, "firstLastMeanAbsDelta": motion })
}

#[test]
#[ignore = "terminal epic-20738 real-weight CUDA campaign; workflow controller only"]
fn epic_20738_terminal_cuda_cell() {
    assert_eq!(
        required_env(ENABLE_ENV),
        "1",
        "terminal CUDA execution is opt-in"
    );
    let cell_path = PathBuf::from(required_env(CELL_ENV));
    let output = PathBuf::from(required_env(OUTPUT_ENV));
    assert!(
        output.is_dir(),
        "controller output directory must already exist"
    );
    let cell: Cell =
        serde_json::from_slice(&std::fs::read(&cell_path).expect("read controller cell file"))
            .expect("parse controller cell file");
    assert!(cell
        .id
        .bytes()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'));
    let primary = artifact(&cell, "primary");
    let bits = verify_exact_packed_tier(&cell, primary);

    let input_summary = json!({
        "cell": cell.id,
        "modelId": cell.model_id,
        "engineId": cell.engine_id,
        "requestedTier": cell.requested_tier,
        "request": {
            "width": cell.request.width, "height": cell.request.height, "steps": cell.request.steps,
            "frames": cell.request.frames, "fps": cell.request.fps, "seed": cell.request.seed,
            "referencePairs": cell.request.reference_pairs,
        },
        "artifacts": cell.artifacts.iter().map(|artifact| json!({
            "id": artifact.id, "role": artifact.role, "repository": artifact.repository,
            "revision": artifact.revision, "subdirectory": artifact.subdirectory,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(
        output.join("generated-inputs.json"),
        serde_json::to_vec_pretty(&input_summary).expect("serialize inputs"),
    )
    .expect("write generated inputs");

    let metrics = match cell.kind.as_str() {
        "image" => generate_image(&cell, primary, &output),
        "sdxlOpenPose" => generate_sdxl_openpose(&cell, primary, bits, &output),
        "scail2" => generate_scail2(&cell, primary, &output),
        "ltx" => generate_ltx(&cell, primary, &output),
        other => panic!("unreviewed terminal cell kind {other}"),
    };
    let quant = family_load_spec_quant_bits(&cell.kind, &cell.engine_id, &cell.requested_tier);
    let result = json!({
        "schemaVersion": 1,
        "cell": cell.id,
        "requestedTier": cell.requested_tier,
        "resolvedTier": primary.subdirectory,
        "denseFallback": false,
        "loadSpecQuantBits": quant,
        "metrics": metrics,
    });
    std::fs::write(
        output.join("runtime-result.json"),
        serde_json::to_vec_pretty(&result).expect("serialize runtime result"),
    )
    .expect("write runtime result");
}

#[cfg(test)]
mod cpu_contract_tests {
    use super::*;

    #[test]
    fn ordered_six_pair_fixture_is_reference_mask_pairs_then_control_clip() {
        let mut conditioning = Vec::new();
        for pair in 0..6 {
            conditioning.push(Conditioning::Reference {
                image: pattern_image(32, 32, pair),
                strength: None,
            });
            conditioning.push(Conditioning::Mask {
                image: mask_image(32, 32, pair as usize),
            });
        }
        conditioning.push(Conditioning::ControlClip {
            frames: vec![pattern_image(32, 32, 99)],
            mask: vec![mask_image(32, 32, 0)],
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        });
        assert_eq!(conditioning.len(), 13);
        for pair in 0..6 {
            assert!(matches!(
                conditioning[pair * 2],
                Conditioning::Reference { .. }
            ));
            assert!(matches!(
                conditioning[pair * 2 + 1],
                Conditioning::Mask { .. }
            ));
        }
        assert!(matches!(conditioning[12], Conditioning::ControlClip { .. }));
    }

    #[test]
    fn quant_marker_parser_requires_bits_beneath_quantization() {
        let mut bits = Vec::new();
        collect_quant_bits(
            &json!({
                "layers": 24,
                "quantization": { "bits": 4 },
                "unrelated": { "bits": 8 },
            }),
            false,
            &mut bits,
        );
        assert_eq!(bits, vec![4]);
    }

    #[test]
    fn family_load_quant_policy_covers_all_19_terminal_cells() {
        let cases = [
            ("chroma1-base-q4", "image", "chroma1_base", "q4", None),
            ("chroma1-base-q8", "image", "chroma1_base", "q8", None),
            ("chroma1-flash-q4", "image", "chroma1_flash", "q4", None),
            ("chroma1-flash-q8", "image", "chroma1_flash", "q8", None),
            ("chroma1-hd-q4", "image", "chroma1_hd", "q4", None),
            ("chroma1-hd-q8", "image", "chroma1_hd", "q8", None),
            ("flux1-dev-q4", "image", "flux1_dev", "q4", None),
            ("flux1-dev-q8", "image", "flux1_dev", "q8", None),
            ("flux1-schnell-q4", "image", "flux1_schnell", "q4", None),
            ("flux1-schnell-q8", "image", "flux1_schnell", "q8", None),
            ("scail2-q4", "scail2", "scail2_14b", "q4", Some(4)),
            (
                "scail2-multi-reference-q4",
                "scail2",
                "scail2_14b",
                "q4",
                Some(4),
            ),
            ("scail2-q8", "scail2", "scail2_14b", "q8", Some(8)),
            ("ltx-2-3-q8", "ltx", "ltx_2_3_distilled", "q8", None),
            ("sdxl-openpose", "sdxlOpenPose", "sdxl", "q4", Some(4)),
            ("realvisxl-openpose", "sdxlOpenPose", "sdxl", "q4", Some(4)),
            (
                "realvisxl-lightning-openpose",
                "sdxlOpenPose",
                "sdxl",
                "q4",
                Some(4),
            ),
            (
                "illustrious-v1-openpose",
                "sdxlOpenPose",
                "sdxl",
                "q4",
                Some(4),
            ),
            (
                "illustrious-v2-openpose",
                "sdxlOpenPose",
                "sdxl",
                "q4",
                Some(4),
            ),
        ];
        assert_eq!(cases.len(), 19);
        for (id, kind, engine_id, requested_tier, expected) in cases {
            assert_eq!(
                family_load_spec_quant_bits(kind, engine_id, requested_tier),
                expected,
                "{id} load-quant policy"
            );
        }
    }

    #[test]
    fn family_request_memory_policy_keeps_all_19_terminal_cells_at_default() {
        let cases = [
            ("chroma1-base-q4", "image", "chroma1_base"),
            ("chroma1-base-q8", "image", "chroma1_base"),
            ("chroma1-flash-q4", "image", "chroma1_flash"),
            ("chroma1-flash-q8", "image", "chroma1_flash"),
            ("chroma1-hd-q4", "image", "chroma1_hd"),
            ("chroma1-hd-q8", "image", "chroma1_hd"),
            ("flux1-dev-q4", "image", "flux1_dev"),
            ("flux1-dev-q8", "image", "flux1_dev"),
            ("flux1-schnell-q4", "image", "flux1_schnell"),
            ("flux1-schnell-q8", "image", "flux1_schnell"),
            ("scail2-q4", "scail2", "scail2_14b"),
            ("scail2-multi-reference-q4", "scail2", "scail2_14b"),
            ("scail2-q8", "scail2", "scail2_14b"),
            ("ltx-2-3-q8", "ltx", "ltx_2_3_distilled"),
            ("sdxl-openpose", "sdxlOpenPose", "sdxl"),
            ("realvisxl-openpose", "sdxlOpenPose", "sdxl"),
            ("realvisxl-lightning-openpose", "sdxlOpenPose", "sdxl"),
            ("illustrious-v1-openpose", "sdxlOpenPose", "sdxl"),
            ("illustrious-v2-openpose", "sdxlOpenPose", "sdxl"),
        ];
        assert_eq!(cases.len(), 19);
        for (id, kind, engine_id) in cases {
            assert!(
                family_generation_memory(kind, engine_id).is_none(),
                "{id} must keep GenerationRequest.memory at its reviewed default"
            );
        }
        assert!(
            std::panic::catch_unwind(|| family_generation_memory("image", "unreviewed_engine"))
                .is_err(),
            "an unreviewed family must not silently inherit the default request-memory contract"
        );
    }

    #[test]
    fn chroma_prepacked_root_builds_a_none_quant_load_spec_before_device_load() {
        let cell = Cell {
            id: "chroma1-base-q4".to_owned(),
            kind: "image".to_owned(),
            model_id: "chroma1_base".to_owned(),
            engine_id: "chroma1_base".to_owned(),
            requested_tier: "q4".to_owned(),
            capability: None,
            request: Request {
                width: 32,
                height: 32,
                steps: 1,
                seed: 1,
                frames: None,
                fps: None,
                guidance: None,
                true_cfg: None,
                sampler: None,
                reference_pairs: None,
            },
            artifacts: Vec::new(),
        };
        let primary = Artifact {
            id: "chroma1-base-q4".to_owned(),
            role: "primary".to_owned(),
            repository: "SceneWorks/chroma1-base-candle".to_owned(),
            revision: "0".repeat(40),
            subdirectory: "q4".to_owned(),
            root: PathBuf::from("fixture/chroma1_base/q4"),
        };
        let spec = primary_load_spec(&cell, &primary);
        assert!(
            spec.quantize.is_none(),
            "the exact prepacked Chroma root must reach the provider without a load-time requant"
        );
    }
}
