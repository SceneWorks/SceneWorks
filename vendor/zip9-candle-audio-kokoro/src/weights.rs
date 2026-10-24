//! Checkpoint + voice-pack loading for Kokoro-82M (sc-12836).
//!
//! `kokoro-v1_0.pth` is a torch zip whose top level maps five module names — `bert`,
//! `bert_encoder`, `predictor`, `text_encoder`, `decoder` — to state dicts whose keys carry a
//! DataParallel `module.` prefix and OLD-style weight-norm pairs (`weight_g` / `weight_v`).
//! The pinned candle pickle reader (`candle_core::pickle::PthTensors`) reads one such section
//! per `key`; this module loads each section, strips the `module.` prefix, resolves every
//! weight-norm pair into a plain `weight` (`w = g · v / ‖v‖`, norm over all dims except 0 — the
//! torch `weight_norm(dim=0)` default used throughout StyleTTS2), and hands back a
//! [`VarBuilder`] per section.
//!
//! Voice style vectors (`voices/<voice>.pt`) are torch zips holding ONE bare `[510, 1, 256]`
//! f32 tensor (not a state dict), a shape the candle reader does not expose — so the raw
//! little-endian storage entry (`<voice>/data/0`) is read directly. The files are pinned by
//! commit SHA, so the layout cannot drift under us.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use candle_audio::candle_core::pickle::PthTensors;
use candle_audio::candle_core::{DType, Device, Tensor, D};
use candle_audio::{AudioError, Result};
use candle_nn::VarBuilder;

/// The five top-level sections of `kokoro-v1_0.pth`, in reference order.
pub const SECTIONS: [&str; 5] = [
    "bert",
    "bert_encoder",
    "predictor",
    "text_encoder",
    "decoder",
];

/// Load one checkpoint section into a name→tensor map with the `module.` prefix stripped and
/// weight-norm pairs resolved to plain `weight` tensors.
pub fn load_section(pth: &Path, section: &str) -> Result<HashMap<String, Tensor>> {
    let tensors = PthTensors::new(pth, Some(section))
        .map_err(|e| AudioError::Msg(format!("open {} [{section}]: {e}", pth.display())))?;
    let mut raw: HashMap<String, Tensor> = HashMap::new();
    for name in tensors.tensor_infos().keys() {
        let t = tensors
            .get(name)
            .map_err(|e| AudioError::Msg(format!("read {section}.{name}: {e}")))?
            .ok_or_else(|| AudioError::Msg(format!("tensor {section}.{name} vanished")))?
            .to_dtype(DType::F32)?;
        let stripped = name.strip_prefix("module.").unwrap_or(name).to_string();
        raw.insert(stripped, t);
    }
    resolve_weight_norm(raw)
}

/// Fold every `X.weight_g` / `X.weight_v` pair into `X.weight = g · v / ‖v‖` (norm per
/// out-channel — all dims except 0, the torch `weight_norm(dim=0)` default). Non-paired
/// tensors pass through unchanged.
fn resolve_weight_norm(raw: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(raw.len());
    let g_suffix = ".weight_g";
    for (name, tensor) in &raw {
        if let Some(base) = name.strip_suffix(g_suffix) {
            let v = raw
                .get(&format!("{base}.weight_v"))
                .ok_or_else(|| AudioError::Msg(format!("{base}: weight_g without weight_v")))?;
            // ‖v‖ over all dims except 0, kept broadcastable against v.
            let mut sq = v.sqr()?;
            for d in 1..v.rank() {
                sq = sq.sum_keepdim(d)?;
            }
            let norm = sq.sqrt()?;
            let w = v.broadcast_mul(&tensor.broadcast_div(&norm)?)?;
            out.insert(format!("{base}.weight"), w);
        } else if !name.ends_with(".weight_v") {
            out.insert(name.clone(), tensor.clone());
        }
    }
    Ok(out)
}

/// [`load_section`] wrapped as a [`VarBuilder`] on `device` (f32 — the checkpoint's dtype).
pub fn section_var_builder(
    pth: &Path,
    section: &str,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let tensors = load_section(pth, section)?
        .into_iter()
        .map(|(k, v)| Ok((k, v.to_device(device)?)))
        .collect::<Result<HashMap<_, _>>>()?;
    Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
}

/// A loaded voice pack: `rows` style vectors of width 256, one per input-token count — the
/// reference pipeline selects row `min(len(tokens) - 1, rows - 1)`.
#[derive(Clone, Debug)]
pub struct VoicePack {
    /// `[rows, 256]` — row-major style vectors (`ref_s`; first 128 = decoder style, last 128 =
    /// prosody style).
    pub data: Vec<f32>,
    pub rows: usize,
    pub dim: usize,
}

impl VoicePack {
    /// Style-vector width (`ref_s` length): decoder style ‖ prosody style.
    pub const STYLE_WIDTH: usize = 256;

    /// Read a `voices/<voice>.pt` torch zip. Validates the little-endian byteorder marker and
    /// that the single storage entry divides into `[rows, 1, 256]` f32 rows.
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| AudioError::Msg(format!("open voice {}: {e}", path.display())))?;
        let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file)).map_err(|e| {
            AudioError::Msg(format!("voice {}: not a torch zip: {e}", path.display()))
        })?;

        let names: Vec<String> = zip
            .file_names()
            .map(|name| name.map(|name| name.into_owned()))
            .collect::<zip::result::ZipResult<_>>()
            .map_err(|e| AudioError::Msg(format!("voice entry name: {e}")))?;
        if let Some(byteorder) = names.iter().find(|n| n.ends_with("/byteorder")) {
            let mut s = String::new();
            zip.by_name(byteorder)
                .map_err(|e| AudioError::Msg(format!("voice byteorder entry: {e}")))?
                .read_to_string(&mut s)
                .map_err(|e| AudioError::Msg(format!("voice byteorder read: {e}")))?;
            if s.trim() != "little" {
                return Err(AudioError::Msg(format!(
                    "voice {}: unsupported byteorder {s:?}",
                    path.display()
                )));
            }
        }
        let data_entry = names
            .iter()
            .find(|n| n.ends_with("/data/0"))
            .ok_or_else(|| {
                AudioError::Msg(format!(
                    "voice {}: no tensor storage entry (expected <name>/data/0)",
                    path.display()
                ))
            })?;
        let mut bytes = Vec::new();
        zip.by_name(data_entry)
            .map_err(|e| AudioError::Msg(format!("voice storage entry: {e}")))?
            .read_to_end(&mut bytes)
            .map_err(|e| AudioError::Msg(format!("voice storage read: {e}")))?;

        let row_bytes = Self::STYLE_WIDTH * 4;
        if bytes.is_empty() || !bytes.len().is_multiple_of(row_bytes) {
            return Err(AudioError::Msg(format!(
                "voice {}: storage of {} bytes does not divide into [rows, 1, {}] f32",
                path.display(),
                bytes.len(),
                Self::STYLE_WIDTH
            )));
        }
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if !data.iter().all(|x| x.is_finite()) {
            return Err(AudioError::Msg(format!(
                "voice {}: non-finite style values",
                path.display()
            )));
        }
        Ok(Self {
            rows: data.len() / Self::STYLE_WIDTH,
            dim: Self::STYLE_WIDTH,
            data,
        })
    }

    /// The `ref_s` row for a token count: `min(n_tokens - 1, rows - 1)` (reference pipeline
    /// indexing, clamped so long inputs still resolve a style).
    pub fn ref_s(&self, n_tokens: usize) -> &[f32] {
        let row = n_tokens.saturating_sub(1).min(self.rows - 1);
        &self.data[row * self.dim..(row + 1) * self.dim]
    }
}

/// Split a `ref_s` slice into `(decoder_style, prosody_style)` tensors of width 128 each on
/// `device` — `ref_s[..128]` feeds the decoder/vocoder AdaIN stack, `ref_s[128..]` the prosody
/// predictor (reference `KModel.forward_with_tokens`).
pub fn split_ref_s(ref_s: &[f32], device: &Device) -> Result<(Tensor, Tensor)> {
    let half = ref_s.len() / 2;
    let decoder = Tensor::from_slice(&ref_s[..half], (1, half), device)?;
    let prosody = Tensor::from_slice(&ref_s[half..], (1, half), device)?;
    Ok((decoder, prosody))
}

/// Convenience: `t.dim(D::Minus1)` errors mapped into [`AudioError`] (used by module code).
pub fn last_dim(t: &Tensor) -> Result<usize> {
    Ok(t.dim(D::Minus1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_norm_resolution_matches_the_definition() {
        let dev = Device::Cpu;
        // v: [2, 1, 3]; g: [2, 1, 1] → w[o] = g[o] * v[o] / ||v[o]||.
        let v = Tensor::from_slice(&[3.0f32, 4.0, 0.0, 0.0, 5.0, 12.0], (2, 1, 3), &dev).unwrap();
        let g = Tensor::from_slice(&[10.0f32, 26.0], (2, 1, 1), &dev).unwrap();
        let mut raw = HashMap::new();
        raw.insert("conv.weight_v".to_string(), v);
        raw.insert("conv.weight_g".to_string(), g);
        raw.insert(
            "conv.bias".to_string(),
            Tensor::zeros(2, DType::F32, &dev).unwrap(),
        );
        let out = resolve_weight_norm(raw).unwrap();
        let w: Vec<f32> = out["conv.weight"].flatten_all().unwrap().to_vec1().unwrap();
        // ||v0|| = 5, ||v1|| = 13 → w0 = 2*[3,4,0] = [6,8,0]; w1 = 2*[0,5,12] = [0,10,24].
        assert_eq!(w, [6.0, 8.0, 0.0, 0.0, 10.0, 24.0]);
        assert!(out.contains_key("conv.bias"));
        assert!(!out.contains_key("conv.weight_g"));
    }

    #[test]
    fn weight_norm_without_its_pair_is_an_error() {
        let dev = Device::Cpu;
        let mut raw = HashMap::new();
        raw.insert(
            "conv.weight_g".to_string(),
            Tensor::zeros((2, 1, 1), DType::F32, &dev).unwrap(),
        );
        assert!(resolve_weight_norm(raw).is_err());
    }
}
