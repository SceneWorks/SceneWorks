//! # mlx-gen
//!
//! Rust-native inference for generative **image and video** models on Apple
//! [MLX](https://github.com/ml-explore/mlx), built on top of `mlx-rs`.
//!
//! **Status: active** — multiple merged, parity-validated provider crates spanning image,
//! video, identity, and understanding models, consumed in-process as a Rust library.
//!
//! Families: FLUX.1 / FLUX.2, Chroma, Qwen-Image (+ Edit), SDXL, Kolors, Z-Image,
//! SenseNova-U1 (image); Wan2.2, LTX-2.3, SVD (video); PuLID-FLUX, InstantID (identity);
//! JoyCaption, SAM2 (understanding). Adapters: LoRA, LoKr (with stacking), ControlNet,
//! IP-Adapter. Plus native MLX LoRA/LoKr training and group-wise Q4/Q8 quantization.
//!
//! Architecture: a *disciplined hybrid* of the frozen Python mflux fork — see
//! [`ARCHITECTURE.md`](https://github.com/michaeltrefry/mlx-gen/blob/main/ARCHITECTURE.md).

// The backend-neutral contract layer (epic 3720). gen-core owns the contracts, registry, request/
// output types, and pure policy math; mlx-gen re-exports them at the historical `mlx_gen::…` paths
// below so every downstream `use mlx_gen::…` keeps compiling. Re-exported as a module too, so
// `mlx_gen::gen_core::{Error, Result}` (the neutral contract error) is reachable by name.
pub use ::gen_core;
pub use gen_core::{
    impl_generator, register_captioner, register_generators, register_image_embedder,
    register_text_embedder, register_trainer,
};

// Local MLX modules (tensor ops, weights, quant, samplers' tensor application, error w/ mlx variants).
pub mod adapters;
// Narrowing helpers for the provider-declared architecture axes (epic SC-22657): shared so a
// fabricated or zeroed axis cannot appear in one family and not another.
pub mod architecture_facts;
pub mod array;
pub mod asset_facts;
// Query-row bounded attention (SC-15615): the MLX half of ladder rung 3, shared so no family forks it.
pub mod attention;
// Bounded transformer residency (SC-15750): ladder rung 4, likewise shared — see the module docs for
// the lazy-graph trap that makes a hand-rolled version silently save nothing.
pub mod block_residency;
// Production capability switches for the shared optimization surface (sc-18316/sc-18318): P1
// retained compilation and P3 exact epilogues default ON in production, with the benchmark keeping
// A/B authority and a truthful opt-out for any path that cannot run one.
pub mod capability;
pub mod coherence;
pub mod error;
pub mod img2img;
pub mod logical_weights;
pub mod memory;
pub mod mllm;
pub mod nn;
pub mod preview;
pub mod pth;
pub mod quant;
// The parameterized QK-norm + RoPE + layout primitive and the adapter/quant-aware fused QKV
// projection (SC-18319, epic 18304 P4). Shared so the ~10 expressible families stop open-coding the
// same attention prologue; the ~11 structural exemptions are enumerated in `qkv::EXEMPTIONS`.
pub mod qkv;
pub mod request_scope;
pub mod residency;
pub mod sampler;
pub mod sc20686;
pub mod scheduler;
pub mod text_sample;
pub mod weights;

// Split modules: contract types in gen-core, MLX impls + lifts local (caption→joycaption,
// train→kernels, tokenizer→to_arrays, image→decoded_to_image).
pub mod caption;
pub mod decoder;
pub mod diagnostics;
pub mod image;
pub mod tokenizer;
pub mod train;

// Moved-verbatim contract modules — re-exported from gen-core at their old paths.
pub mod control {
    pub use gen_core::control::*;
}
pub mod generator {
    pub use gen_core::generator::*;
}
pub mod media {
    pub use gen_core::media::*;
}
pub mod registry {
    pub use gen_core::registry::*;
}
pub mod runtime {
    pub use gen_core::runtime::*;
}
pub mod tiling {
    pub use gen_core::tiling::*;
}
pub mod transform {
    pub use gen_core::transform::*;
}
// Array-level tiled-decode blend loop (sc-11747): the MLX half of the gen-core tiling seam, shared by
// every VAE that tiles a decode (Wan z16/z48, Qwen-Image). gen-core carries the pure geometry
// ([`tiling::TilePlan`]); this carries the tensor loop.
pub mod memory_probe;
pub mod vae_tiling;

pub use attention::{
    sdpa_budgeted_bhsd, AttentionBudget, AttentionPlan, CONSTRAINED_ATTN_SCORES_BUDGET,
};
pub use caption::{
    CaptionCapabilities, CaptionFinishReason, CaptionOptions, CaptionOutput, CaptionRequest,
    CaptionSampling, Captioner, CaptionerDescriptor,
};
pub use control::{
    require_base_dir, require_base_snapshot, require_component_file, require_control,
    AcceptedControlKinds, ControlBranch,
};
pub use decoder::{ensure_decoder_compatible, ensure_decoder_layout, LatentDecoder};
pub use error::{Error, Result};
pub use gen_core::sampling::{
    flow_capture_plan, schedule_sigmas, vp_capture_plan, vp_sigma_from_edm, CapturePlan,
    DiscreteModelSampling, EdmModelSampling, ModelSampling, PredictionType, Scheduler, Solver,
    TimestepConvention, VpCapturePlan,
};
pub use gen_core::weightsmeta::{safetensors_dir_bytes, safetensors_path_bytes};
pub use generator::{
    default_seed, ActivationMemoryAnchor, Capabilities, Conditioning, ConditioningKind,
    ControlClipRef, ControlKind, GenerationOutput, GenerationPhase, GenerationRequest, Generator,
    KeyframeRef, Modality, ModelDescriptor, OutputChannels, PhaseAdapter, ReplacementMode,
    SizeFloor, StagedResidencyAvailability, StepSupport, VideoClipRef,
};
pub use media::{AudioTrack, Image, RgbaImage};
pub use registry::{
    CaptionerRegistration, ModelRegistration, PerComponentBytes, ProviderRegistry,
    ProviderRegistryBuilder, TrainerRegistration, TransformRegistration,
};
pub use residency::{Residency, StagedHeavy};
pub use runtime::{
    AdapterApplyReport, AdapterKind, AdapterSpec, CancelFlag, IdentityWeights, LoadPhase,
    LoadShape, LoadSpec, MoeExpert, OffloadPolicy, PidWeights, PinnedWeightsFile, Precision,
    PreparedFilePins, PreviewFrame, PreviewSink, Progress, Quant, WeightsSource,
    BASE_SNAPSHOT_COMPONENT, COMFYUI_TEXT_ENCODER_COMPONENT, COMFYUI_VAE_COMPONENT, VAE_COMPONENT,
};
pub use sampler::{
    curated_sampler_names, curated_scheduler_names, resolve_flow_schedule, resolve_schedule,
    run_av_curated_sampler, run_cfgpp_sampler, run_cfgpp_sampler_with_latent_hook,
    run_curated_sampler, run_curated_sampler_with_latent_hook, run_flow_sampler,
    run_flow_sampler_with_latent_hook, AlphaSchedule, AvLatents, DiffusionSampler,
    FlowMatchSampler, LcmSampler, LightningSampler, MlxAvLatentOps, MlxLatentOps, TcdSampler,
};
pub use scheduler::FlowMatchEuler;
pub use tiling::{TilingConfig, VaeTiling, VideoDecodeMemoryProfile};
pub use train::{
    LrSchedule, NetworkType, TrainOptimizer, Trainer, TrainerDescriptor, TrainingConfig,
    TrainingItem, TrainingOutput, TrainingProgress, TrainingRequest,
};
pub use transform::{
    InputBudget, InputSizing, TargetSize, Transform, TransformCapabilities, TransformDescriptor,
    TransformRequest,
};
