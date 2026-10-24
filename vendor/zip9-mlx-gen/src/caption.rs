//! The `Captioner` contract lives in gen-core (tensor-free request/output types). mlx-gen hosts the
//! MLX [`joycaption`] implementation and re-exports the contract types at the historical
//! `mlx_gen::caption::…` paths (epic 3720, D4 / Appendix A).

pub mod joycaption;

pub use gen_core::caption::*;
