//! Dedicated Actions-only cold/warm quantization proof. This is compiled as a small standalone
//! harness so validation does not need to link every application provider or CV runtime.
use gen_core::QuantizationCacheContext;
use sceneworks_core::quantization_cache::SavedQuantizations;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "candle")]
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(feature = "candle")]
fn run(_root: &Path, cold: bool) -> Result<Vec<Value>> {
    use candle_gen::candle_core::{
        quantized::{GgmlDType, QTensor},
        safetensors::MmapedSafetensors,
        DType, Device,
    };
    let source = PathBuf::from(std::env::var("SCENEWORKS_BF16_CHECKPOINT")?);
    let key = "model.diffusion_model.blocks.0.attn.wq.weight";
    // SAFETY: read-only checkpoint, retained for this proof; no source write is performed.
    let weights = unsafe { MmapedSafetensors::new(&source)? };
    let weight = weights.load(key, &Device::Cpu)?.to_dtype(DType::F32)?;
    let device = Device::new_cuda(0)?;
    let mut cases = Vec::new();
    for (bits, dtype) in [(4, GgmlDType::Q4_0), (8, GgmlDType::Q8_0)] {
        let actual = candle_gen::quant::quantize_cached(&weight, dtype, &device)?;
        let bytes = actual.data()?;
        if cold {
            let baseline = QTensor::quantize_onto(&weight, dtype, &device)?;
            assert_eq!(
                bytes.as_ref(),
                baseline.data()?.as_ref(),
                "cold packing changed"
            );
        }
        cases.push(
            json!({"bits": bits, "shape": weight.dims(), "packedSha256": digest(bytes.as_ref())}),
        );
    }
    Ok(cases)
}

#[cfg(feature = "mlx")]
fn run(root: &Path, cold: bool) -> Result<Vec<Value>> {
    use mlx_rs::{Array, Dtype};
    let source = root.join("source/model.safetensors");
    if cold {
        let values: Vec<f32> = (0..16384)
            .map(|n| ((n * 37 % 1021) as f32 - 510.0) / 127.0)
            .collect();
        let weight = Array::from_slice(&values, &[128, 128]).as_dtype(Dtype::Bfloat16)?;
        Array::save_safetensors([("weight", &weight)], None, &source)?;
    }
    let mut tensors: std::collections::HashMap<String, Array> =
        Array::load_safetensors(&source)?.into_iter().collect();
    let weight = tensors.remove("weight").ok_or("missing source weight")?;
    let hash_parts = |parts: &(Array, Array, Array)| -> Result<String> {
        let mut hash = Sha256::new();
        for value in parts.0.try_as_slice::<u32>()? {
            hash.update(value.to_le_bytes());
        }
        for array in [&parts.1, &parts.2] {
            hash.update(format!("{:?}", array.dtype()).as_bytes());
            let values = array.as_dtype(Dtype::Float32)?;
            for value in values.try_as_slice::<f32>()? {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        Ok(format!("{:x}", hash.finalize()))
    };
    let mut cases = Vec::new();
    for bits in [4, 8] {
        let actual = mlx_gen::quant::quantize_cached(&weight, 64, bits)?;
        let actual_hash = hash_parts(&actual)?;
        if cold {
            let baseline = mlx_rs::ops::quantize(&weight, 64, bits)?;
            assert_eq!(actual_hash, hash_parts(&baseline)?, "cold packing changed");
        }
        cases.push(json!({"bits": bits, "shape": weight.shape(), "packedSha256": actual_hash}));
    }
    Ok(cases)
}

fn main() -> Result<()> {
    if std::env::var("GITHUB_ACTIONS").as_deref() != Ok("true") {
        return Err("run this numerical proof through its dedicated GitHub Actions profile".into());
    }
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or("expected cold or warm")?;
    if !matches!(mode.as_str(), "cold" | "warm") {
        return Err("expected cold or warm".into());
    }
    let root = PathBuf::from(args.next().ok_or("expected proof output root")?);
    std::fs::create_dir_all(root.join("data"))?;
    std::fs::create_dir_all(root.join("source"))?;
    let source = if cfg!(feature = "candle") {
        PathBuf::from(std::env::var("SCENEWORKS_BF16_CHECKPOINT")?)
    } else {
        root.join("source")
    };
    let cache = Arc::new(SavedQuantizations::new(
        &root.join("data"),
        source,
        "quantization-proof".into(),
        "quantization-proof".into(),
    ));
    let context = QuantizationCacheContext::new(cache.clone());
    let _guard = context.install();
    let started = std::time::Instant::now();
    let cases = run(&root, mode == "cold")?;
    let (hits, writes) = cache.counts();
    if mode == "cold" {
        assert_eq!((hits, writes), (0, 2));
    } else {
        assert_eq!(
            (hits, writes),
            (2, 0),
            "warm process must load both saved formats without quantizing"
        );
        let previous: Value = serde_json::from_slice(&std::fs::read(root.join("cold.json"))?)?;
        assert_eq!(
            previous["cases"],
            json!(cases),
            "saved weights changed across processes"
        );
    }
    let report = json!({"mode": mode, "cases": cases, "cacheHits": hits, "quantizerRuns": writes, "elapsedMs": started.elapsed().as_millis(), "processId": std::process::id()});
    std::fs::write(
        root.join(format!("{mode}.json")),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
