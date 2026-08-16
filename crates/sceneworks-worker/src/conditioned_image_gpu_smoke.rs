//! sc-18476 ignored real-weight CUDA evidence for the newly exposed registered Kolors source-edit
//! and Krea singular-reference worker paths. Compile in ordinary Candle test builds; run only on the
//! coordinator-owned physical GPU with env-provided weights/output paths.

use std::path::PathBuf;

use gen_core::{
    AdapterKind, AdapterSpec, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec,
    WeightsSource,
};

use super::smoke_support::{
    env_or, image_std, mean_abs_frame_delta, save_png, DEGENERATE_STD_FLOOR_DEFAULT,
};

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set ${key}")))
}

fn reference(width: u32, height: u32, inverted: bool) -> Image {
    Image {
        width,
        height,
        pixels: (0..(width * height * 3))
            .map(|index| {
                let value = (index % 251) as u8;
                if inverted {
                    255 - value
                } else {
                    value
                }
            })
            .collect(),
    }
}

fn one_image(output: GenerationOutput, context: &str) -> Image {
    match output {
        GenerationOutput::Images(mut images) => images.pop().expect(context),
        other => panic!("expected images from {context}, got {other:?}"),
    }
}

#[test]
#[ignore = "real-weight GPU smoke; needs KOLORS_DIR and CUDA"]
fn kolors_source_edit_candle_gpu_smoke() {
    let dir = env_path("KOLORS_DIR");
    let out_dir = env_path("CONDITIONED_IMAGE_OUT_DIR");
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    let generator =
        crate::inference_runtime::load("kolors", &LoadSpec::new(WeightsSource::Dir(dir)))
            .expect("load registered Kolors generator");
    let (width, height) = (512, 512);
    let request = |inverted| GenerationRequest {
        prompt: env_or(
            "KOLORS_EDIT_PROMPT",
            "turn the scene into a watercolor painting",
        ),
        width,
        height,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        guidance: Some(5.0),
        conditioning: vec![Conditioning::Reference {
            image: reference(width, height, inverted),
            strength: Some(0.6),
        }],
        ..Default::default()
    };
    let image_a = one_image(
        generator
            .generate(&request(false), &mut |_| {})
            .expect("Kolors registered source edit A"),
        "Kolors output A",
    );
    let image_b = one_image(
        generator
            .generate(&request(true), &mut |_| {})
            .expect("Kolors registered source edit B"),
        "Kolors output B",
    );
    assert!(image_std(&image_a) > DEGENERATE_STD_FLOOR_DEFAULT);
    assert!(image_std(&image_b) > DEGENERATE_STD_FLOOR_DEFAULT);
    assert!(
        mean_abs_frame_delta(&image_a, &image_b) > 0.01,
        "same-shape Kolors renders must respond to materially different reference pixels"
    );
    save_png(&image_a, &out_dir.join("kolors_source_edit_a.png"));
    save_png(&image_b, &out_dir.join("kolors_source_edit_b.png"));
}

#[test]
#[ignore = "real-weight GPU smoke; needs KREA_DIR, KREA_EDIT_LORA, and CUDA"]
fn krea_singular_reference_candle_gpu_smoke() {
    let dir = env_path("KREA_DIR");
    let edit_lora = env_path("KREA_EDIT_LORA");
    let out_dir = env_path("CONDITIONED_IMAGE_OUT_DIR");
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    let device = runtime_cuda::media::default_device().expect("CUDA device");
    let adapters = vec![AdapterSpec::new(edit_lora, 1.0, AdapterKind::Lora)];
    let components =
        runtime_cuda::providers::krea::pipeline::load_components(&dir, &device, &adapters, None)
            .expect("load Krea components");
    let edit = runtime_cuda::providers::krea::pipeline::load_edit_components(&dir, &device)
        .expect("load Krea edit components");
    let request = GenerationRequest {
        prompt: env_or(
            "KREA_EDIT_PROMPT",
            "preserve the person and move them to a snowy forest",
        ),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        guidance: Some(3.5),
        ..Default::default()
    };
    let render = |inverted| {
        runtime_cuda::providers::krea::pipeline::render_edit(
            &components,
            &edit,
            &request,
            &[reference(512, 512, inverted)],
            false,
            &device,
            &mut |_| {},
        )
        .expect("Krea singular-reference edit")
        .pop()
        .expect("Krea output")
    };
    let image_a = render(false);
    let image_b = render(true);
    assert!(image_std(&image_a) > DEGENERATE_STD_FLOOR_DEFAULT);
    assert!(image_std(&image_b) > DEGENERATE_STD_FLOOR_DEFAULT);
    assert!(
        mean_abs_frame_delta(&image_a, &image_b) > 0.01,
        "same-shape Krea renders must respond to materially different singular-reference pixels"
    );
    save_png(
        &image_a,
        &out_dir.join("krea_singular_reference_edit_a.png"),
    );
    save_png(
        &image_b,
        &out_dir.join("krea_singular_reference_edit_b.png"),
    );
}
