//! sc-17776 research probe — a BEHAVIOUR WITNESS for `flux2_dev`'s declared memory ladder.
//!
//! Not part of the crate's shipped surface: this file is applied to a scratch worktree by
//! `git apply` and never committed, exactly like the mutation patches in
//! `docs/calibration/sc-17776/mutations.md`.
//!
//! It renders what the ladder DECLARES rather than what the compiler emitted: the registered
//! provider contract for a grid of load specs. `MemoryProviderContract` derives `Debug` and
//! `crates/contracts/gen-core/src/memory_strategy.rs` is byte-identical across the window under
//! test, so the rendering is comparable between the two revisions by construction.
//!
//! No weights, no GPU work, no device allocation — it only builds `LoadSpec`s and asks the
//! provider what it would do.

use std::path::PathBuf;

use candle_gen::gen_core::{LoadShape, LoadSpec, Quant, WeightsSource};
use candle_gen_flux2::memory_strategy;

fn main() {
    let mut rendering = String::new();
    for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
        for load_shape in [LoadShape::DeferredMaterialization, LoadShape::EagerMaterialization] {
            let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/sc17776/flux2-dev")));
            if let Some(quant) = quant {
                spec = spec.with_quant(quant);
            }
            spec.load_shape = load_shape;
            let contract = match memory_strategy::provider_contract(&spec) {
                Ok(contract) => format!("{contract:?}"),
                Err(error) => format!("Err({error})"),
            };
            let tier = match memory_strategy::resolved_numeric_tier(&spec) {
                Ok(tier) => format!("{tier:?}"),
                Err(error) => format!("Err({error})"),
            };
            rendering.push_str(&format!(
                "quant={quant:?} load_shape={load_shape:?}\n  tier={tier}\n  contract={contract}\n"
            ));
        }
    }
    print!("{rendering}");
    eprintln!("SC17776_WITNESS_BYTES {}", rendering.len());
}
