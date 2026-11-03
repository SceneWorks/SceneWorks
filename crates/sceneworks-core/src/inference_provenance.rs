//! Provenance for newly produced runtime and estimate evidence.
//!
//! Captured measurements retain the producer revision stored in their record. This value is
//! descriptive only: it is never a memory-evidence currency or admission gate.
pub const INFERENCE_REVISION: &str = env!("SCENEWORKS_INFERENCE_REVISION");
