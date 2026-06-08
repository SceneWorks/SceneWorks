//! sc-3487 Stage A — confirm `ort` builds, links onnxruntime, binds CoreML, and
//! runs the RTMW + YOLOX onnx with the expected input/output shapes. Prints the
//! resolved ort version's I/O metadata so the full pipeline (detect.rs) can be
//! written against the real API.

use anyhow::Result;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::Session;

fn cache() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home).join(".cache/rtmlib/hub/checkpoints")
}

fn describe(path: &std::path::Path, coreml: bool) -> Result<()> {
    let mut builder = Session::builder()?;
    if coreml {
        builder = builder.with_execution_providers([CoreMLExecutionProvider::default().build()])?;
    }
    let session = builder.commit_from_file(path)?;
    println!("== {} (coreml={})", path.file_name().unwrap().to_string_lossy(), coreml);
    for inp in &session.inputs {
        println!("   IN  {:<16} {:?}", inp.name, inp.input_type);
    }
    for out in &session.outputs {
        println!("   OUT {:<16} {:?}", out.name, out.output_type);
    }
    Ok(())
}

fn main() -> Result<()> {
    let c = cache();
    let det = c.join("yolox_m_8xb8-300e_humanart-c2c7a14a.onnx");
    let pose = c.join("rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx");
    println!("ort probe — det={} pose={}", det.exists(), pose.exists());
    describe(&det, false)?;
    describe(&pose, false)?;
    println!("-- now with CoreML --");
    describe(&det, true)?;
    describe(&pose, true)?;
    println!("OK");
    Ok(())
}
