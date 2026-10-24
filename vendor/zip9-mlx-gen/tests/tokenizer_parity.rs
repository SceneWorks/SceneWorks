//! sc-2341: text-tokenizer parity vs the Python fork's Z-Image `LanguageTokenizer`.
//!
//! Gated `#[ignore]` + env var: it needs the real Qwen2 `tokenizer.json` (~11 MB), which is
//! too large to commit and would bloat the published crate. Byte-level BPE parity is the
//! `tokenizers` crate's responsibility (the same Rust core `transformers` wraps), so the
//! regression risk lives in *our* orchestration (chat template, padding, mask) — covered by
//! the unit tests in `src/tokenizer.rs` that DO run on CI. Run this locally:
//!
//!   MLX_GEN_ZIMAGE_TOKENIZER=/path/to/Z-Image-Turbo/tokenizer/tokenizer.json \
//!     cargo test --test integration -- tokenizer_parity:: --ignored
//!
//! Fixture `tests/fixtures/tokenizer_zimage.safetensors` ← `tools/dump_z_image_tokenizer.py`.

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_rs::ops::array_eq;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tokenizer_zimage.safetensors"
);

// Must match tools/dump_z_image_tokenizer.py exactly.
const PROMPTS: [&str; 3] = [
    "a red fox",
    "A serene mountain lake at sunset, photorealistic",
    "café — naïve façade, 日本語 prompt, emoji 🦊",
];

fn z_image_config() -> TokenizerConfig {
    TokenizerConfig {
        max_length: 512,
        pad_token_id: 151643, // Qwen2 <|endoftext|>
        chat_template: ChatTemplate::QwenInstruct,
        pad_to_max_length: true,
    }
}

#[test]
#[ignore = "needs the ~11MB Z-Image tokenizer.json; set MLX_GEN_ZIMAGE_TOKENIZER"]
fn z_image_tokenizer_matches_fork() {
    let path = std::env::var("MLX_GEN_ZIMAGE_TOKENIZER")
        .expect("set MLX_GEN_ZIMAGE_TOKENIZER to the Z-Image tokenizer.json path");
    let tok = TextTokenizer::from_file(&path, z_image_config()).unwrap();
    let w = Weights::from_file(FIXTURE).unwrap();

    for (i, prompt) in PROMPTS.iter().enumerate() {
        let out = tok.tokenize(prompt).unwrap();
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&out);
        let want_ids = w.require(&format!("p{i}.input_ids")).unwrap();
        let want_mask = w.require(&format!("p{i}.attention_mask")).unwrap();

        assert_eq!(input_ids.shape(), want_ids.shape(), "p{i}: input_ids shape");
        assert!(
            array_eq(&input_ids, want_ids, false)
                .unwrap()
                .item::<bool>(),
            "p{i}: input_ids diverged from the fork"
        );
        assert!(
            array_eq(&attention_mask, want_mask, false)
                .unwrap()
                .item::<bool>(),
            "p{i}: attention_mask diverged from the fork"
        );
    }
}
