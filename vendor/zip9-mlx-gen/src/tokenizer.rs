//! Text tokenization. The tokenizer itself (HF `tokenizers` wrapper, chat templating, padding)
//! lives in gen-core and is re-exported here unchanged; mlx-gen adds only [`to_arrays`], the MLX
//! lift that turns the neutral host-vec [`TokenizerOutput`] into `(input_ids, attention_mask)`
//! arrays at the text-encoder seam (epic 3720, D4).

use mlx_rs::Array;

pub use gen_core::tokenizer::*;

/// Lift a gen-core [`TokenizerOutput`] (host vecs) into `(1, L)` int32 `(input_ids, attention_mask)`
/// MLX arrays. An empty prompt yields `(1, 0)` arrays (the gen-core empty convention) — preserving
/// the previous `empty_row()` behavior for the empty-negative-prompt / true-CFG paths.
pub fn to_arrays(t: &TokenizerOutput) -> (Array, Array) {
    let len = t.ids.len() as i32;
    (
        Array::from_slice(&t.ids, &[1, len]),
        Array::from_slice(&t.mask, &[1, len]),
    )
}
