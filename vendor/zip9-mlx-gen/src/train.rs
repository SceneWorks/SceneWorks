//! The `Trainer` contract + the pure LR-schedule policy live in gen-core; mlx-gen hosts the MLX
//! training kernels (checkpoint / dataset / LoRA factor reconstruction / optimizer) and re-exports
//! the contract types — plus the schedule module and the MLX-bound [`TrainOptimizer`] — at the
//! historical `mlx_gen::train::…` paths (epic 3720, D4 / Appendix A).

pub mod checkpoint;
pub mod dataset;
pub mod formula;
pub mod latent_lpips;
pub mod lazy;
pub mod lora;
pub mod loss;
pub mod optim;
pub mod perceptual;
pub mod tae;
pub mod taehv;
pub mod vae_anchor;

/// The pure LR-schedule policy (`LrSchedule`, `lr_multiplier`, `schedule_updates`) moved to
/// gen-core; re-exported here so `mlx_gen::train::schedule::…` keeps resolving for the family
/// trainers.
pub mod schedule {
    pub use gen_core::train::schedule::*;
}

// `TrainOptimizer` wraps an mlx-rs `AdamW` — it stays in mlx-gen. The contract types (Trainer,
// TrainingConfig, LrSchedule, …) come from gen-core. The local `schedule` module above shadows the
// `schedule` name the glob would otherwise pull in (explicit-wins; not an error).
pub use gen_core::train::*;
pub use optim::TrainOptimizer;
