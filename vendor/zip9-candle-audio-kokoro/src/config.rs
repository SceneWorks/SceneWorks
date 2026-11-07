//! Kokoro-82M `config.json` parsing — the model hyperparameters and the phoneme→id vocab
//! (sc-12836). Mirrors the fields the reference `KModel.__init__` reads
//! (`hexgrad/kokoro`'s `model.py`); everything else in the file is ignored.

use std::collections::HashMap;
use std::path::Path;

use candle_audio::{AudioError, Result};
use serde_json::Value;

/// iSTFT-Net vocoder head hyperparameters (the `istftnet` block).
#[derive(Clone, Debug)]
pub struct IstftNetConfig {
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub gen_istft_hop_size: usize,
    pub gen_istft_n_fft: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
}

/// PLBERT (ALBERT) hyperparameters (the `plbert` block). ALBERT shares ONE transformer layer's
/// parameters across all `num_hidden_layers` iterations; `embedding_size` is the ALBERT default
/// 128 (the checkpoint's factorized embedding width), not present in the file.
#[derive(Clone, Debug)]
pub struct PlbertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
}

/// The parsed Kokoro model configuration.
#[derive(Clone, Debug)]
pub struct KokoroConfig {
    pub istftnet: IstftNetConfig,
    pub dim_in: usize,
    pub hidden_dim: usize,
    pub max_dur: usize,
    pub n_layer: usize,
    pub n_mels: usize,
    pub n_token: usize,
    pub style_dim: usize,
    pub text_encoder_kernel_size: usize,
    pub plbert: PlbertConfig,
    /// Phoneme character → input id (the model's fixed symbol table). Characters not present are
    /// silently dropped at tokenization, exactly like the reference `KModel.forward`.
    pub vocab: HashMap<char, u32>,
}

fn get_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| {
            AudioError::Msg(format!("kokoro config.json: missing integer field {key:?}"))
        })
}

fn get_usize_vec(v: &Value, key: &str) -> Result<Vec<usize>> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|n| n as usize)
                .collect::<Vec<_>>()
        })
        .ok_or_else(|| {
            AudioError::Msg(format!("kokoro config.json: missing integer array {key:?}"))
        })
}

impl KokoroConfig {
    /// Parse a `hexgrad/Kokoro-82M` `config.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| AudioError::Msg(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&text)
    }

    /// Parse the configuration from raw JSON text (unit-testable without a snapshot).
    pub fn from_json_str(text: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(text)
            .map_err(|e| AudioError::Msg(format!("kokoro config.json parse: {e}")))?;

        let istft = v
            .get("istftnet")
            .ok_or_else(|| AudioError::Msg("kokoro config.json: missing istftnet block".into()))?;
        let istftnet = IstftNetConfig {
            upsample_kernel_sizes: get_usize_vec(istft, "upsample_kernel_sizes")?,
            upsample_rates: get_usize_vec(istft, "upsample_rates")?,
            gen_istft_hop_size: get_usize(istft, "gen_istft_hop_size")?,
            gen_istft_n_fft: get_usize(istft, "gen_istft_n_fft")?,
            resblock_dilation_sizes: istft
                .get("resblock_dilation_sizes")
                .and_then(Value::as_array)
                .map(|rows| {
                    rows.iter()
                        .map(|row| {
                            row.as_array()
                                .map(|a| {
                                    a.iter()
                                        .filter_map(Value::as_u64)
                                        .map(|n| n as usize)
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default()
                        })
                        .collect::<Vec<_>>()
                })
                .ok_or_else(|| {
                    AudioError::Msg("kokoro config.json: missing resblock_dilation_sizes".into())
                })?,
            resblock_kernel_sizes: get_usize_vec(istft, "resblock_kernel_sizes")?,
            upsample_initial_channel: get_usize(istft, "upsample_initial_channel")?,
        };

        let plbert_v = v
            .get("plbert")
            .ok_or_else(|| AudioError::Msg("kokoro config.json: missing plbert block".into()))?;
        let plbert = PlbertConfig {
            hidden_size: get_usize(plbert_v, "hidden_size")?,
            num_attention_heads: get_usize(plbert_v, "num_attention_heads")?,
            intermediate_size: get_usize(plbert_v, "intermediate_size")?,
            max_position_embeddings: get_usize(plbert_v, "max_position_embeddings")?,
            num_hidden_layers: get_usize(plbert_v, "num_hidden_layers")?,
        };

        let vocab = v
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| AudioError::Msg("kokoro config.json: missing vocab map".into()))?
            .iter()
            .filter_map(|(k, id)| {
                let mut chars = k.chars();
                match (chars.next(), chars.next(), id.as_u64()) {
                    (Some(c), None, Some(id)) => Some((c, id as u32)),
                    _ => None,
                }
            })
            .collect::<HashMap<char, u32>>();
        if vocab.is_empty() {
            return Err(AudioError::Msg("kokoro config.json: empty vocab".into()));
        }

        Ok(Self {
            istftnet,
            dim_in: get_usize(&v, "dim_in")?,
            hidden_dim: get_usize(&v, "hidden_dim")?,
            max_dur: get_usize(&v, "max_dur")?,
            n_layer: get_usize(&v, "n_layer")?,
            n_mels: get_usize(&v, "n_mels")?,
            n_token: get_usize(&v, "n_token")?,
            style_dim: get_usize(&v, "style_dim")?,
            text_encoder_kernel_size: get_usize(&v, "text_encoder_kernel_size")?,
            plbert,
            vocab,
        })
    }

    /// Map a phoneme string to input ids, silently dropping characters outside the vocab —
    /// the reference `KModel.forward` filter, which is what makes odd G2P output (ties,
    /// unknown-word markers) degrade gracefully instead of erroring.
    pub fn phonemes_to_ids(&self, phonemes: &str) -> Vec<u32> {
        phonemes
            .chars()
            .filter_map(|c| self.vocab.get(&c).copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI: &str = r#"{
        "istftnet": {
            "upsample_kernel_sizes": [20, 12],
            "upsample_rates": [10, 6],
            "gen_istft_hop_size": 5,
            "gen_istft_n_fft": 20,
            "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
            "resblock_kernel_sizes": [3,7,11],
            "upsample_initial_channel": 512
        },
        "dim_in": 64, "hidden_dim": 512, "max_dur": 50, "n_layer": 3, "n_mels": 80,
        "n_token": 178, "style_dim": 128, "text_encoder_kernel_size": 5,
        "plbert": {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 2048,
                   "max_position_embeddings": 512, "num_hidden_layers": 12},
        "vocab": {"a": 43, "b": 44, "ˈ": 156, " ": 16}
    }"#;

    #[test]
    fn parses_the_reference_shape() {
        let c = KokoroConfig::from_json_str(MINI).unwrap();
        assert_eq!(c.istftnet.upsample_rates, [10, 6]);
        assert_eq!(c.istftnet.gen_istft_n_fft, 20);
        assert_eq!(c.plbert.num_hidden_layers, 12);
        assert_eq!(c.hidden_dim, 512);
        assert_eq!(c.vocab[&'a'], 43);
    }

    #[test]
    fn tokenization_drops_out_of_vocab_chars() {
        let c = KokoroConfig::from_json_str(MINI).unwrap();
        // The tie (U+200D) and the unknown-word marker must vanish, not error.
        assert_eq!(
            c.phonemes_to_ids("a\u{200d}b \u{2753}ˈa"),
            [43, 44, 16, 156, 43]
        );
    }

    #[test]
    fn rejects_malformed_configs() {
        assert!(KokoroConfig::from_json_str("{}").is_err());
        assert!(KokoroConfig::from_json_str("not json").is_err());
    }
}
