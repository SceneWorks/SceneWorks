//! Audio-lane snapshot-preparation accommodation for Kokoro (sc-12836).
//!
//! The audio lane carries the candle snapshot preparer so audio model weights are preparable
//! through `catalog.audio_preparers()` on every platform (sc-12835). That preparer's HF path
//! demands `config.json` **and** `tokenizer.json` — a Kokoro snapshot has `config.json` but no
//! tokenizer (its "tokenizer" is the phoneme vocab inside `config.json`) and pickle weights
//! rather than safetensors, so `candle-llm`'s `can_prepare` would accept it (config.json
//! present) and `prepare` would then fail on the missing tokenizer — a lying probe.
//!
//! The accommodation (composed by `candle-audio-catalog` into the lane's single `candle`
//! registration, WITHOUT weakening the LLM preparer): recognize a Kokoro audio snapshot by its
//! `istftnet` config block + checkpoint file, and prepare it as a validated **passthrough** —
//! the snapshot is already in its loadable form (dense pickle + voices; there is no
//! quantized/converted variant to materialize), so preparation verifies and returns it.
//! A requested quantization is a typed `Unsupported`, never a silent dense fallback.

use std::path::Path;

use candle_audio::candle_core::pickle::read_pth_tensor_info;
use core_llm::{Error as CoreError, ModelFormat, PrepareReport, PrepareSpec, Result as CoreResult};

use crate::pipeline::CHECKPOINT_FILE;
use crate::weights::SECTIONS;

/// Weightless probe: is `dir` a Kokoro audio snapshot (a `config.json` carrying the
/// `istftnet` vocoder block + the pickle checkpoint)? Reads only `config.json` metadata,
/// never a weight shard.
pub fn is_kokoro_snapshot(dir: &Path) -> bool {
    if !dir.is_dir() || !dir.join(CHECKPOINT_FILE).is_file() {
        return false;
    }
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .map(|v| v.get("istftnet").is_some() && v.get("vocab").is_some())
        .unwrap_or(false)
}

/// [`is_kokoro_snapshot`] over a [`PrepareSpec`] — the probe the composed audio-lane
/// registration consults before delegating to the LLM preparer.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_kokoro_snapshot(&spec.source)
}

/// Prepare (verify + passthrough) a Kokoro snapshot. Counts the checkpoint's tensors from
/// pickle metadata (no storage reads) so the report is honest about what the snapshot holds.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    if !is_kokoro_snapshot(&spec.source) {
        return Err(CoreError::Unsupported(format!(
            "prepare: {} is not a Kokoro audio snapshot",
            spec.source.display()
        )));
    }
    if let Some(q) = spec.quantize {
        return Err(CoreError::Unsupported(format!(
            "prepare: kokoro audio snapshots have no {q:?} form — the 82M checkpoint ships \
             dense-only"
        )));
    }
    let pth = spec.source.join(CHECKPOINT_FILE);
    let mut num_tensors = 0usize;
    for section in SECTIONS {
        let infos = read_pth_tensor_info(&pth, false, Some(section)).map_err(|e| {
            CoreError::Msg(format!("prepare: {} section {section}: {e}", pth.display()))
        })?;
        if infos.is_empty() {
            return Err(CoreError::Msg(format!(
                "prepare: {} is missing checkpoint section {section:?}",
                pth.display()
            )));
        }
        num_tensors += infos.len();
    }
    Ok(PrepareReport {
        // core-llm's own detect_format calls a config.json-bearing snapshot dir Safetensors
        // (the HF-snapshot shape); the pickle checkpoint rides inside that shape.
        input_format: ModelFormat::Safetensors,
        quantized: None,
        out_dir: spec.source.clone(),
        num_tensors,
        passthrough: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(dir: &Path) -> PrepareSpec {
        PrepareSpec {
            source: dir.to_path_buf(),
            out_dir: dir.join("out"),
            quantize: None,
        }
    }

    #[test]
    fn probe_rejects_non_kokoro_layouts() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        // Empty dir → no.
        assert!(!can_prepare(&spec(&dir)));
        // LLM-shaped config.json (no istftnet) + a pth → still no.
        std::fs::write(dir.join("config.json"), r#"{"hidden_size": 8}"#).unwrap();
        std::fs::write(dir.join(CHECKPOINT_FILE), b"not a checkpoint").unwrap();
        assert!(!can_prepare(&spec(&dir)));
        // Kokoro-shaped config.json → yes (probe reads no weights).
        std::fs::write(
            dir.join("config.json"),
            r#"{"istftnet": {}, "vocab": {"a": 1}}"#,
        )
        .unwrap();
        assert!(can_prepare(&spec(&dir)));
    }

    #[test]
    fn prepare_refuses_quantization_typed() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        std::fs::write(
            dir.join("config.json"),
            r#"{"istftnet": {}, "vocab": {"a": 1}}"#,
        )
        .unwrap();
        std::fs::write(dir.join(CHECKPOINT_FILE), b"stub").unwrap();
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
    }
}
