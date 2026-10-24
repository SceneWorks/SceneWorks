# mlx-gen

Rust-native inference for generative **image and video** models on Apple [MLX](https://github.com/ml-explore/mlx), built on [`mlx-rs`](https://crates.io/crates/mlx-rs).

> **Status:** active — two dozen model provider crates with merged, parity-validated engines spanning image, video, upscaling, identity, and understanding models. Built as a Rust library workspace consumed in-process; not yet published to crates.io. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

A from-scratch Rust reimplementation of the MLX image/video model stack (a divergence from the Python `mflux` / `mlx-video` lineage), collapsing on-device inference into a single statically-linked component with no Python sidecar. Each model family is its own provider crate registered through the core `mlx-gen` `Generator` contract.

**Supported models**

- **Image:** FLUX.1 (schnell/dev, incl. Hyper few-step), FLUX.2 (klein-9b and dev — txt2img, edit, ControlNet, KV-cache edit; Qwen3 text encoder + 32-ch VAE), Chroma (`chroma1_hd`/`base`/`flash`), Qwen-Image (+ Edit, + ControlNet), Stable Diffusion XL (+ inpaint/outpaint, IP-Adapter, tile-ControlNet, LCM/Lightning/Hyper), Kolors (bilingual, ChatGLM3 text encoder), Z-Image (incl. ControlNet), SenseNova-U1 (unified understanding + generation: T2I, image-edit, VQA, interleaved document), Boogu-Image (Lumina-Image-2.0 / OmniGen2 lineage; base/turbo/edit, Qwen3-VL encoder), Ideogram 4.0 (+ Turbo), Lens / Lens-Turbo (Microsoft; gpt-oss-20b MoE encoder + dual-stream MMDiT)
- **Video:** Wan2.2 (text/image/TI2V, incl. VACE and VACE-Fun), Bernini renderer (ByteDance; Wan2.2-A14B dual-expert MoE + source-id rotary + APG guidance), SCAIL-2 (controlled character animation / motion transfer; Wan2.1-14B I2V backbone), LTX-2.3 (text-to-video + audio), Stable Video Diffusion (image-to-video)
- **Upscaling / restoration:** SeedVR2 (one-step diffusion super-resolution, image and video; 3B/7B)
- **Identity:** PuLID-FLUX and InstantID, over a native MLX face stack (SCRFD + ArcFace + BiSeNet)
- **Understanding & utility:** JoyCaption (captioning), SAM2 / SAM3 (segmentation; SAM3 adds open-vocabulary concept segmentation + video tracking)
- **Adapters:** LoRA, LoKr (reconstruct + forward-time residual + stacking, quant-safe), ControlNet, IP-Adapter
- **Training:** native MLX LoRA / LoKr fine-tuning for SDXL, Z-Image, Kolors, Wan2.2, LTX-2.3, and Lens (adamw / adam / rose / prodigy optimizers, dataset + checkpoint plumbing)
- **Quantization:** group-wise affine Q4 / Q8 (byte-identical to the reference packing)
- **Weight loading:** most models load directly from their Hugging Face / diffusers snapshot — no conversion step. Families that ship in a non-loadable source format include a native Rust converter (FLUX.2 single-file → diffusers; Wan2.2 torch `.pth` reader, T2V/I2V/TI2V + VAE; LTX-2.3 single-file → split MLX). A few models that ship as fp8 or scattered torch checkpoints are provisioned by an offline Python converter under `tools/` (Ideogram 4 fp8, InstantID, and the face-stack sub-models)

Requires a Mac with full Xcode + the Metal Toolchain (MLX's Metal kernels compile from source).

## Usage

mlx-gen is a Rust library workspace consumed in-process. Each model family lives in its own
provider crate and exports an explicit registry builder. Depend on `mlx-gen` plus the provider
families you want, then construct and query the exact catalog you intend to ship:

```toml
# Cargo.toml
[dependencies]
mlx-gen = { git = "https://github.com/SceneWorks/inference", tag = "runtime-2026.07.0" }
mlx-gen-z-image = { git = "https://github.com/SceneWorks/inference", tag = "runtime-2026.07.0" }
```

```rust
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};

fn main() -> mlx_gen::gen_core::Result<()> {
    // Load a model by id from a Hugging Face snapshot directory.
    let spec = LoadSpec::new(WeightsSource::Dir("/path/to/Z-Image-Turbo".into()));
    let registry = mlx_gen_z_image::provider_registry()?;
    let model = registry.load("z_image_turbo", &spec)?;

    let req = GenerationRequest {
        prompt: "a red fox in a snowy forest".into(),
        width: 1024,
        height: 1024,
        seed: Some(42),
        ..Default::default()
    };

    let out = model.generate(&req, &mut |p| {
        if let Progress::Step { current, total } = p {
            println!("step {current}/{total}");
        }
    })?;

    if let GenerationOutput::Images(images) = out {
        let img = &images[0];
        // `img.pixels` is interleaved RGB (`img.width` × `img.height`); encode with any
        // image crate (e.g. `image::save_buffer`) to write a PNG.
        println!("generated {}×{}", img.width, img.height);
    }
    Ok(())
}
```

Inspect a family registry with `registry.generators()`, `registry.trainers()`, and
`registry.captioners()`, or use `mlx-gen-catalog` for the complete shipped MLX platform surface.
The SAM2 / SAM3 segmenters are plain utility APIs used directly, not through the registry.
(Prompt-refine —
Llama-3.2-3B-Instruct rewriting — is served through the `core-llm` LLM contract and lives in the
worker / candle-gen; mlx-gen's own `mlx-gen-prompt-refine` crate was retired by the mlx-llm
engine in sc-7158, and the `load_textllm` registry entry point was removed in sc-7189.)

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). You are free to
use, modify, and distribute mlx-gen, **including commercially**, under those terms.

## Acknowledgements

mlx-gen is an independent Rust reimplementation and includes no copied source,
but it stands on the work of others:

- [Apple MLX](https://github.com/ml-explore/mlx) (MIT) and [mlx-rs](https://crates.io/crates/mlx-rs) (Apache-2.0 OR MIT) — the on-device tensor stack
- [mflux](https://github.com/filipstrand/mflux) (MIT) — the MLX diffusion lineage mlx-gen diverged from and validates parity against
- [Apple mlx-examples](https://github.com/ml-explore/mlx-examples) (MIT)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) (Apache-2.0) — the upstream model architectures

See [NOTICE](NOTICE) for full attribution.
