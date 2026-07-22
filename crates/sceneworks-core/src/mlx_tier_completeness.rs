//! Per-family tier-completeness predicates for the MLX turnkeys that ship NO `model_index.json`
//! (Anima / Boogu / SANA).
//!
//! These families lay their weights out in a bespoke tree rather than a diffusers `model_index.json`
//! pipeline, so the generic `<tier>/*` presence checks — the worker's `tier_components_present`
//! (reads the tier's own `model_index.json`) and rust-api's coarse glob / `tier_subdir_has_weights`
//! — are a no-op for them: a TORN tier (backbone present, text-encoder / VAE / tokenizer missing)
//! passes the coarse check yet fails to load. Both the worker's tier resolvers (which pick a loadable
//! tier) AND rust-api's catalog completeness (which reports `installed` vs `incomplete`) need the SAME
//! concrete per-family predicate, so it lives here in the shared crate — a single source of truth that
//! the two consumers cannot drift apart (sc-13513). Hidden AppleDouble `._*` sidecars never count
//! (SceneWorks#1333).

use std::path::Path;

/// Whether `dir` holds at least one non-hidden file whose name ends with `suffix` (e.g. `.safetensors`
/// / `.index.json`). The building block for the per-family completeness checks: a hidden AppleDouble
/// `._*` sidecar never counts (SceneWorks#1333).
pub fn dir_has_visible_file_ending(dir: &Path, suffix: &str) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            !crate::lora_family::is_hidden_file(&entry.path())
                && entry.file_name().to_string_lossy().ends_with(suffix)
        })
    })
}

/// Whether an Anima tier `dir` is COMPLETE and loadable (issue #850 class, generalized). Anima ships no
/// `model_index.json`, so the shared presence guards are a no-op for it and a torn tier (DiT present,
/// text-encoder/VAE absent) reached the loader, which then died with a raw "No such file or directory"
/// (mlx-gen-anima `load_text_phase` / `load_vae`). A tier is complete when all three concrete inputs are
/// present: the DiT (any visible `.safetensors` in `diffusion_models/`), the dense text encoder, and the
/// VAE. Tokenizers are vendored into the binary, so there is no tokenizer file to check here.
pub fn anima_tier_complete(dir: &Path) -> bool {
    dir_has_visible_file_ending(&dir.join("diffusion_models"), ".safetensors")
        && dir
            .join("text_encoders/qwen_3_06b_base.safetensors")
            .is_file()
        && dir.join("vae/qwen_image_vae.safetensors").is_file()
}

/// Whether a Boogu tier `dir` is COMPLETE and loadable. Boogu ships no `model_index.json`, so the shared
/// guard is a no-op; the loader crashes FIRST on a missing `mllm/tokenizer.json`, then on an absent
/// `mllm`/`transformer`/`vae` weight. A tier is complete when the transformer (packed single-file OR
/// sharded index) plus its `config.json`, the Qwen3-VL `mllm/` (weights + `tokenizer.json`), and the VAE
/// weights are all present.
pub fn boogu_tier_complete(dir: &Path) -> bool {
    let transformer = dir.join("transformer");
    let transformer_weights = transformer
        .join("diffusion_pytorch_model.safetensors")
        .is_file()
        || transformer
            .join("diffusion_pytorch_model.safetensors.index.json")
            .is_file();
    transformer_weights
        && transformer.join("config.json").is_file()
        && dir.join("mllm/tokenizer.json").is_file()
        && dir_has_visible_file_ending(&dir.join("mllm"), ".safetensors")
        && dir_has_visible_file_ending(&dir.join("vae"), ".safetensors")
}

/// Whether a SANA tier `dir` is COMPLETE and loadable. The `SceneWorks/Sana_*_mlx` turnkeys ship no
/// `model_index.json`, so the standard-tier guard is a no-op for SANA specifically (flux/qwen/etc. DO
/// ship one and stay protected by the `model_index.json` check); the SANA loader dies with a raw OS
/// error on an absent Gemma text encoder or its tokenizer (mlx-gen-sana `SanaTextEncoder::from_snapshot`).
/// A tier is complete when the Linear-DiT transformer, the DC-AE VAE, and the Gemma-2 text encoder + its
/// tokenizer (both bundled inside `text_encoder/`) are all present.
pub fn sana_tier_complete(dir: &Path) -> bool {
    dir_has_visible_file_ending(&dir.join("transformer"), ".safetensors")
        && dir_has_visible_file_ending(&dir.join("vae"), ".safetensors")
        && dir.join("text_encoder/gemma-2-2b-it.safetensors").is_file()
        && dir.join("text_encoder/tokenizer.json").is_file()
}

/// Whether a Wan2.2 MLX turnkey tier `dir` is COMPLETE and loadable by the native MLX Wan trainer
/// (sc-13878). The `SceneWorks/wan2.2-{ti2v-5b,t2v-a14b,i2v-a14b}-mlx` turnkeys ship a FLAT MLX layout
/// (a `config.json` + top-level `*.safetensors`, NOT a diffusers `model_index.json` / `transformer/`
/// component tree), so both the shared `model_index.json` guard and rust-api's diffusers-shaped
/// training-base checks (`bf16_component_tree_present`, which probes `transformer/`|`unet/` dirs) are a
/// no-op for them: a torn tier clears the coarse `<tier>/*` glob yet the trainer
/// (mlx-gen-wan `build_trainer_concrete`) dies on the first absent flat file. A tier is complete when the
/// UMT5 T5 encoder, the Wan VAE, the tokenizer, the `config.json`, and the DiT are all present — the DiT
/// being either the single-expert `model.safetensors` (dense TI2V-5B) OR both MoE experts
/// `low_noise_model.safetensors` + `high_noise_model.safetensors` (A14B T2V/I2V). Exact-path `.is_file()`
/// checks are AppleDouble-safe by construction — a hidden `._name` sidecar has a different name
/// (SceneWorks#1333) — and pin the exact filenames the trainer's `LoadSpec` dir must contain.
pub fn wan_tier_complete(dir: &Path) -> bool {
    let shared = dir.join("config.json").is_file()
        && dir.join("t5_encoder.safetensors").is_file()
        && dir.join("vae.safetensors").is_file()
        && dir.join("tokenizer.json").is_file();
    let dense_expert = dir.join("model.safetensors").is_file();
    let dual_expert = dir.join("low_noise_model.safetensors").is_file()
        && dir.join("high_noise_model.safetensors").is_file();
    shared && (dense_expert || dual_expert)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn touch(path: &Path) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"x").unwrap();
    }

    /// Write a COMPLETE Anima tier tree, then return its dir. Callers delete one component to probe that
    /// each is load-bearing (a mutation check — a torn tier must go RED, not pass on the backbone alone).
    fn seed_anima(dir: &Path) {
        touch(&dir.join("diffusion_models/anima-base-v1.0.safetensors"));
        touch(&dir.join("text_encoders/qwen_3_06b_base.safetensors"));
        touch(&dir.join("vae/qwen_image_vae.safetensors"));
    }

    fn seed_boogu(dir: &Path) {
        touch(&dir.join("transformer/diffusion_pytorch_model.safetensors"));
        touch(&dir.join("transformer/config.json"));
        touch(&dir.join("mllm/model.safetensors"));
        touch(&dir.join("mllm/tokenizer.json"));
        touch(&dir.join("vae/diffusion_pytorch_model.safetensors"));
    }

    fn seed_sana(dir: &Path) {
        touch(&dir.join("transformer/diffusion_pytorch_model.safetensors"));
        touch(&dir.join("vae/diffusion_pytorch_model.safetensors"));
        touch(&dir.join("text_encoder/gemma-2-2b-it.safetensors"));
        touch(&dir.join("text_encoder/tokenizer.json"));
    }

    /// Write a COMPLETE dense (single-expert, TI2V-5B) Wan MLX tier tree.
    fn seed_wan_dense(dir: &Path) {
        touch(&dir.join("config.json"));
        touch(&dir.join("t5_encoder.safetensors"));
        touch(&dir.join("vae.safetensors"));
        touch(&dir.join("tokenizer.json"));
        touch(&dir.join("model.safetensors"));
    }

    /// Write a COMPLETE dual-expert (A14B MoE) Wan MLX tier tree.
    fn seed_wan_moe(dir: &Path) {
        touch(&dir.join("config.json"));
        touch(&dir.join("t5_encoder.safetensors"));
        touch(&dir.join("vae.safetensors"));
        touch(&dir.join("tokenizer.json"));
        touch(&dir.join("low_noise_model.safetensors"));
        touch(&dir.join("high_noise_model.safetensors"));
    }

    #[test]
    fn anima_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q8");
        seed_anima(&dir);
        assert!(anima_tier_complete(&dir), "fully-seeded tier is complete");

        // Mutation check: removing ANY single component must flip the verdict to incomplete.
        for component in [
            "diffusion_models/anima-base-v1.0.safetensors",
            "text_encoders/qwen_3_06b_base.safetensors",
            "vae/qwen_image_vae.safetensors",
        ] {
            let torn = tmp.path().join("torn");
            seed_anima(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !anima_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn anima_wrong_text_encoder_filename_is_incomplete() {
        // The predicate pins the EXACT dense-TE filename; a differently-named `.safetensors` in
        // `text_encoders/` does not satisfy it (guards a typo'd rename from silently passing).
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q8");
        seed_anima(&dir);
        fs::remove_file(dir.join("text_encoders/qwen_3_06b_base.safetensors")).unwrap();
        touch(&dir.join("text_encoders/some_other.safetensors"));
        assert!(!anima_tier_complete(&dir));
    }

    #[test]
    fn anima_ignores_appledouble_dit_sidecar() {
        // A hidden `._` DiT sidecar is not a real weight (SceneWorks#1333).
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q8");
        seed_anima(&dir);
        fs::remove_file(dir.join("diffusion_models/anima-base-v1.0.safetensors")).unwrap();
        touch(&dir.join("diffusion_models/._anima-base-v1.0.safetensors"));
        assert!(!anima_tier_complete(&dir));
    }

    #[test]
    fn boogu_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("base");
        seed_boogu(&dir);
        assert!(boogu_tier_complete(&dir));

        for component in [
            "transformer/diffusion_pytorch_model.safetensors",
            "transformer/config.json",
            "mllm/model.safetensors",
            "mllm/tokenizer.json",
            "vae/diffusion_pytorch_model.safetensors",
        ] {
            let torn = tmp.path().join("torn");
            seed_boogu(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !boogu_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn boogu_accepts_sharded_transformer_index() {
        // A sharded transformer (index.json, no single-file weight) is loadable too.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("base");
        seed_boogu(&dir);
        fs::remove_file(dir.join("transformer/diffusion_pytorch_model.safetensors")).unwrap();
        touch(&dir.join("transformer/diffusion_pytorch_model.safetensors.index.json"));
        assert!(boogu_tier_complete(&dir));
    }

    #[test]
    fn sana_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q4");
        seed_sana(&dir);
        assert!(sana_tier_complete(&dir));

        for component in [
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/gemma-2-2b-it.safetensors",
            "text_encoder/tokenizer.json",
        ] {
            let torn = tmp.path().join("torn");
            seed_sana(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !sana_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn wan_dense_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bf16");
        seed_wan_dense(&dir);
        assert!(
            wan_tier_complete(&dir),
            "fully-seeded dense tier is complete"
        );

        // Mutation check: removing ANY single component must flip the verdict to incomplete.
        for component in [
            "config.json",
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
            "model.safetensors",
        ] {
            let torn = tmp.path().join("torn");
            seed_wan_dense(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !wan_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn wan_moe_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bf16");
        seed_wan_moe(&dir);
        assert!(wan_tier_complete(&dir), "fully-seeded MoE tier is complete");

        // Mutation check: BOTH experts are load-bearing (a single-expert dir is not a valid A14B tier)
        // alongside every shared component.
        for component in [
            "config.json",
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
            "low_noise_model.safetensors",
            "high_noise_model.safetensors",
        ] {
            let torn = tmp.path().join("torn");
            seed_wan_moe(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !wan_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn wan_ignores_appledouble_expert_sidecar() {
        // A hidden `._` DiT sidecar is not a real weight (SceneWorks#1333); exact-path checks reject it.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bf16");
        seed_wan_dense(&dir);
        fs::remove_file(dir.join("model.safetensors")).unwrap();
        touch(&dir.join("._model.safetensors"));
        assert!(!wan_tier_complete(&dir));
    }

    #[test]
    fn missing_tier_dir_is_incomplete_for_every_family() {
        // A tier subdir that does not exist at all (never converted / never downloaded) is incomplete,
        // not a panic — the catalog treats it as a clean "missing".
        let tmp = tempfile::tempdir().unwrap();
        let absent = tmp.path().join("nope");
        assert!(!anima_tier_complete(&absent));
        assert!(!boogu_tier_complete(&absent));
        assert!(!sana_tier_complete(&absent));
        assert!(!wan_tier_complete(&absent));
    }
}
