//! PLBERT — the ALBERT encoder that contextualizes phoneme ids for the prosody predictor
//! (sc-12836). A faithful component port of HF `AlbertModel` restricted to what Kokoro uses:
//! batch-1, full attention (no padding mask), `last_hidden_state` output, `gelu_new`
//! activation, one shared layer group iterated `num_hidden_layers` times (the ALBERT parameter
//! sharing that makes 82 M total possible).
//!
//! Checkpoint names mirror HF: `embeddings.{word,position,token_type}_embeddings`,
//! `embeddings.LayerNorm`, `encoder.embedding_hidden_mapping_in`,
//! `encoder.albert_layer_groups.0.albert_layers.0.{attention.{query,key,value,dense,LayerNorm},
//! ffn, ffn_output, full_layer_layer_norm}`. The pooler is unused (Kokoro reads the hidden
//! states).

use candle_audio::candle_core::Tensor;
use candle_audio::Result;
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};

use crate::config::PlbertConfig;

/// ALBERT's factorized embedding width (the HF `AlbertConfig.embedding_size` default the
/// checkpoint was trained with; not present in Kokoro's config.json).
pub const EMBEDDING_SIZE: usize = 128;
/// HF `AlbertConfig.layer_norm_eps` default.
const LN_EPS: f64 = 1e-12;

/// `gelu_new` — the tanh GELU approximation HF ALBERT defaults to.
fn gelu_new(x: &Tensor) -> Result<Tensor> {
    // 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
    let inner = ((x * 0.044715f64)?.mul(&x.sqr()?)? + x)?;
    let inner = (inner * (2f64 / std::f64::consts::PI).sqrt())?;
    Ok(((inner.tanh()? + 1.0)? * 0.5)?.mul(x)?)
}

struct AlbertLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    attn_norm: LayerNorm,
    ffn: Linear,
    ffn_output: Linear,
    full_norm: LayerNorm,
    n_heads: usize,
    head_dim: usize,
}

impl AlbertLayer {
    fn new(cfg: &PlbertConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let attn = vb.pp("attention");
        Ok(Self {
            query: linear(h, h, attn.pp("query"))?,
            key: linear(h, h, attn.pp("key"))?,
            value: linear(h, h, attn.pp("value"))?,
            dense: linear(h, h, attn.pp("dense"))?,
            attn_norm: layer_norm(h, LN_EPS, attn.pp("LayerNorm"))?,
            ffn: linear(h, cfg.intermediate_size, vb.pp("ffn"))?,
            ffn_output: linear(cfg.intermediate_size, h, vb.pp("ffn_output"))?,
            full_norm: layer_norm(h, LN_EPS, vb.pp("full_layer_layer_norm"))?,
            n_heads: cfg.num_attention_heads,
            head_dim: h / cfg.num_attention_heads,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (b, t, _h) = hidden.dims3()?;
        let shape = (b, t, self.n_heads, self.head_dim);
        let q = self
            .query
            .forward(hidden)?
            .reshape(shape)?
            .transpose(1, 2)?;
        let k = self.key.forward(hidden)?.reshape(shape)?.transpose(1, 2)?;
        let v = self
            .value
            .forward(hidden)?
            .reshape(shape)?
            .transpose(1, 2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.contiguous()?.matmul(&k.contiguous()?.t()?)? * scale)?;
        let probs = candle_nn::ops::softmax(&scores, 3)?;
        let ctx = probs
            .matmul(&v.contiguous()?)? // [b, heads, t, head_dim]
            .transpose(1, 2)?
            .reshape((b, t, self.n_heads * self.head_dim))?;
        let attn_out = self
            .attn_norm
            .forward(&(self.dense.forward(&ctx)? + hidden)?)?;
        let ffn = gelu_new(&self.ffn.forward(&attn_out)?)?;
        let ffn = self.ffn_output.forward(&ffn)?;
        Ok(self.full_norm.forward(&(ffn + attn_out)?)?)
    }
}

/// The PLBERT encoder: embeddings → hidden mapping → the shared layer iterated N times.
pub struct Plbert {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    embed_norm: LayerNorm,
    hidden_mapping: Linear,
    shared_layer: AlbertLayer,
    num_layers: usize,
    max_positions: usize,
}

impl Plbert {
    pub fn new(n_token: usize, cfg: &PlbertConfig, vb: VarBuilder) -> Result<Self> {
        let emb = vb.pp("embeddings");
        let enc = vb.pp("encoder");
        Ok(Self {
            word_embeddings: embedding(n_token, EMBEDDING_SIZE, emb.pp("word_embeddings"))?,
            position_embeddings: embedding(
                cfg.max_position_embeddings,
                EMBEDDING_SIZE,
                emb.pp("position_embeddings"),
            )?,
            token_type_embeddings: embedding(2, EMBEDDING_SIZE, emb.pp("token_type_embeddings"))?,
            embed_norm: layer_norm(EMBEDDING_SIZE, LN_EPS, emb.pp("LayerNorm"))?,
            hidden_mapping: linear(
                EMBEDDING_SIZE,
                cfg.hidden_size,
                enc.pp("embedding_hidden_mapping_in"),
            )?,
            shared_layer: AlbertLayer::new(cfg, enc.pp("albert_layer_groups.0.albert_layers.0"))?,
            num_layers: cfg.num_hidden_layers,
            max_positions: cfg.max_position_embeddings,
        })
    }

    /// The encoder's context length (input ids, including the two `0` sentinels).
    pub fn context_length(&self) -> usize {
        self.max_positions
    }

    /// `input_ids: [1, T] (u32) → last_hidden_state [1, T, hidden]`. No attention mask: Kokoro
    /// always runs a single unpadded sequence, so the mask is all-ones and drops out.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let device = input_ids.device();
        let positions = Tensor::arange(0u32, t as u32, device)?.unsqueeze(0)?;
        let token_types = Tensor::zeros((1, t), candle_audio::candle_core::DType::U32, device)?;
        let emb = (self.word_embeddings.forward(input_ids)?
            + self.position_embeddings.forward(&positions)?)?;
        let emb = (emb + self.token_type_embeddings.forward(&token_types)?)?;
        let emb = self.embed_norm.forward(&emb)?;
        let mut hidden = self.hidden_mapping.forward(&emb)?;
        for _ in 0..self.num_layers {
            hidden = self.shared_layer.forward(&hidden)?;
        }
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::Device;

    #[test]
    fn gelu_new_matches_reference_values() {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], (4,), &dev).unwrap();
        let y: Vec<f32> = gelu_new(&x).unwrap().to_vec1().unwrap();
        // torch.nn.functional.gelu(x, approximate="tanh")
        let expect = [-0.15880801, 0.0, 0.841192, 1.9545977];
        for (a, e) in y.iter().zip(expect) {
            assert!((a - e).abs() < 1e-5, "{y:?}");
        }
    }
}
