//! StyleTTS2 `TextEncoder` (modules.py) — phoneme embedding → three weight-normed conv blocks
//! (conv + channel layer-norm + LeakyReLU) → a bidirectional LSTM. Produces the aligned-text
//! features (`t_en`) the decoder consumes (sc-12836). Batch-1, no padding mask (a single
//! unpadded sequence, so every reference `masked_fill` is a no-op).

use candle_audio::candle_core::Tensor;
use candle_audio::Result;
use candle_nn::{conv1d, embedding, ops, Conv1d, Conv1dConfig, Embedding, Module, VarBuilder};

use crate::nn::{BiLstm, ChannelLayerNorm, LRELU_SLOPE_BLOCKS};

pub struct TextEncoder {
    embedding: Embedding,
    convs: Vec<(Conv1d, ChannelLayerNorm)>,
    lstm: BiLstm,
}

impl TextEncoder {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        depth: usize,
        n_symbols: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embedding = embedding(n_symbols, channels, vb.pp("embedding"))?;
        let padding = (kernel_size - 1) / 2;
        let mut convs = Vec::with_capacity(depth);
        for i in 0..depth {
            let block = vb.pp(format!("cnn.{i}"));
            convs.push((
                conv1d(
                    channels,
                    channels,
                    kernel_size,
                    Conv1dConfig {
                        padding,
                        ..Default::default()
                    },
                    block.pp("0"),
                )?,
                ChannelLayerNorm::new(channels, block.pp("1"))?,
            ));
        }
        let lstm = BiLstm::new(channels, channels / 2, vb.pp("lstm"))?;
        Ok(Self {
            embedding,
            convs,
            lstm,
        })
    }

    /// `input_ids: [1, T] (u32) → t_en [1, channels, T]`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let x = self.embedding.forward(input_ids)?; // [1, T, C]
        let mut x = x.transpose(1, 2)?.contiguous()?; // [1, C, T]
        for (conv, norm) in &self.convs {
            x = conv.forward(&x)?;
            x = norm.forward(&x)?;
            x = ops::leaky_relu(&x, LRELU_SLOPE_BLOCKS)?;
        }
        let x = x.transpose(1, 2)?.contiguous()?; // [1, T, C]
        let x = self.lstm.forward(&x)?; // [1, T, C]
        Ok(x.transpose(1, 2)?.contiguous()?) // [1, C, T]
    }
}
