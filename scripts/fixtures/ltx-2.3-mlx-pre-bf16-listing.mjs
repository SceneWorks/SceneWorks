// Real upstream file listing of SceneWorks/ltx-2.3-mlx at revision
// 254989c3ca7ee691187647f350b112c0c448789d, recorded from
// https://huggingface.co/api/models/SceneWorks/ltx-2.3-mlx/revision/<rev>?blobs=false.
//
// This is the LITERAL sc-18853 defect, frozen as data. That revision (2026-06-14) is what the
// ltx_2_3 bf16 download row was pinned to, and it predates the upstream bf16/ upload
// (01df27d308466533aa09d251e3aebdcc627d07eb, 2026-07-05) by three weeks -- note the top level
// below holds gemma/, q4/ and q8/ and NO bf16/. The pre-sc-18809 gate resolved bf16/* against the
// DEFAULT BRANCH, where bf16/ did exist, so it passed the row; `hf download --include 'bf16/*'`
// then exited 0 having fetched nothing and the engine failed later with
// `missing transformer.safetensors`.
//
// scripts/check-download-patterns.test.mjs replays the REAL manifest bf16/* pattern against this
// listing to prove the gate now reds it, and against the current recorded listing to prove it does
// not red a good pin. A commit SHA is immutable, so this fixture cannot rot.
export const LTX_2_3_MLX_REPO = "SceneWorks/ltx-2.3-mlx";
export const LTX_2_3_MLX_PRE_BF16_REVISION = "254989c3ca7ee691187647f350b112c0c448789d";
export const LTX_2_3_MLX_PRE_BF16_FILES = Object.freeze([
    ".gitattributes",
    "LICENSE",
    "README.md",
    "gemma/README.md",
    "gemma/added_tokens.json",
    "gemma/chat_template.json",
    "gemma/config.json",
    "gemma/generation_config.json",
    "gemma/model-00001-of-00005.safetensors",
    "gemma/model-00002-of-00005.safetensors",
    "gemma/model-00003-of-00005.safetensors",
    "gemma/model-00004-of-00005.safetensors",
    "gemma/model-00005-of-00005.safetensors",
    "gemma/model.safetensors.index.json",
    "gemma/preprocessor_config.json",
    "gemma/processor_config.json",
    "gemma/special_tokens_map.json",
    "gemma/tokenizer.json",
    "gemma/tokenizer.model",
    "gemma/tokenizer_config.json",
    "q4/audio_vae.safetensors",
    "q4/config.json",
    "q4/connector.safetensors",
    "q4/embedded_config.json",
    "q4/quantize_config.json",
    "q4/split_model.json",
    "q4/transformer.safetensors",
    "q4/upsampler.safetensors",
    "q4/vae_decoder.safetensors",
    "q4/vae_encoder.safetensors",
    "q4/vocoder.safetensors",
    "q8/audio_vae.safetensors",
    "q8/config.json",
    "q8/connector.safetensors",
    "q8/embedded_config.json",
    "q8/quantize_config.json",
    "q8/split_model.json",
    "q8/transformer.safetensors",
    "q8/upsampler.safetensors",
    "q8/vae_decoder.safetensors",
    "q8/vae_encoder.safetensors",
    "q8/vocoder.safetensors"
  ]);
