//! Fork a completed generic trainer's recovery state into a new output namespace.
//! Tensor bytes, optimizer moments, update counters and dataset identities stay unchanged;
//! only the explicitly extended schedule horizon changes in the copied metadata.
use std::{
    fs,
    io::{Seek, SeekFrom, Write},
    path::Path,
};

use crate::{WorkerError, WorkerResult};
use serde_json::{json, Value};

fn refused(message: impl Into<String>) -> WorkerError {
    WorkerError::InvalidPayload(message.into())
}

/// Both directories must be server-owned and confined before calling. `destination` must
/// not exist. Publication is atomic; failures leave the original and destination untouched.
pub fn fork_completed_training_checkpoint(
    source: &Path,
    destination: &Path,
    file_name: &str,
    completed_steps: u32,
    total_steps: u32,
) -> WorkerResult<()> {
    if completed_steps == 0 || total_steps <= completed_steps {
        return Err(refused("Additional training steps must be positive."));
    }
    let file_name = crate::paths::safe_weight_filename(file_name, "Training fileName")?;
    let stem = crate::training_resume_stem(&file_name);
    let source = source.canonicalize()?;
    let snapshot_name = format!("{stem}-step{completed_steps:06}.resume.safetensors");
    let snapshot = source.join(&snapshot_name).canonicalize().map_err(|_| refused(
        "This run has no recovery checkpoint at its completed step. Its exported LoRA is preserved, but cannot be continued with optimizer state."
    ))?;
    if !snapshot.starts_with(&source) || !snapshot.is_file() {
        return Err(refused("Training checkpoint escapes the source run."));
    }
    if !sceneworks_core::safetensors::file_is_structurally_valid(&snapshot) {
        return Err(refused("Training checkpoint is truncated or malformed."));
    }
    let mut input = fs::File::open(&snapshot)?;
    let source_metadata = input.metadata()?;
    let descriptor = sceneworks_core::safetensors::read_header(
        &mut input,
        source_metadata.len(),
        sceneworks_core::safetensors::MAX_ADAPTER_HEADER_BYTES,
        sceneworks_core::safetensors::HeaderJsonPolicy::RejectDuplicateKeys,
    )
    .map_err(|error| refused(format!("Invalid training checkpoint: {error}")))?;
    let mut header = descriptor.value;
    let meta = header
        .get_mut("__metadata__")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| refused("Training checkpoint is missing recovery metadata."))?;
    if meta
        .get("step")
        .and_then(Value::as_str)
        .and_then(|n| n.parse::<u32>().ok())
        != Some(completed_steps)
        || meta
            .get("update_idx")
            .and_then(Value::as_str)
            .and_then(|n| n.parse::<u32>().ok())
            .is_none()
    {
        return Err(refused(
            "Training checkpoint does not match the completed step.",
        ));
    }
    let candle = meta.get("format").and_then(Value::as_str) == Some("candle-gen-resume-v1");
    let mlx =
        !meta.contains_key("format") && meta.get("optimizer").and_then(Value::as_str).is_some();
    if !candle && !mlx {
        return Err(refused(
            "This trainer's checkpoint format does not support continuation yet.",
        ));
    }
    if let Some(identity) = meta.get("training_config").and_then(Value::as_str) {
        let prefix = format!("steps={completed_steps};");
        let remainder = identity
            .strip_prefix(&prefix)
            .ok_or_else(|| refused("Checkpoint schedule does not match the completed run."))?;
        meta.insert(
            "training_config".into(),
            json!(format!("steps={total_steps};{remainder}")),
        );
    } else if candle {
        return Err(refused(
            "Training checkpoint is missing its configuration identity.",
        ));
    }
    let parent = destination
        .parent()
        .ok_or_else(|| refused("Invalid continuation output directory."))?;
    fs::create_dir_all(parent)?;
    if destination.exists() {
        return Err(refused("Continuation output directory already exists."));
    }
    let staging = tempfile::Builder::new()
        .prefix(".continue-")
        .tempdir_in(parent)?;
    if mlx {
        let name = snapshot_name.replace(".resume.safetensors", ".optim.safetensors");
        let optimizer = source
            .join(&name)
            .canonicalize()
            .map_err(|_| refused("Training checkpoint is missing its optimizer state."))?;
        if !optimizer.starts_with(&source) || !optimizer.is_file() {
            return Err(refused("Training optimizer state escapes the source run."));
        }
        if !sceneworks_core::safetensors::file_is_structurally_valid(&optimizer) {
            return Err(refused(
                "Training optimizer checkpoint is truncated or malformed.",
            ));
        }
        fs::copy(optimizer, staging.path().join(name))?;
    }
    input.seek(SeekFrom::Start(descriptor.data_start))?;
    let mut encoded = serde_json::to_vec(&header).map_err(|error| refused(error.to_string()))?;
    encoded.resize(encoded.len().next_multiple_of(8), b' ');
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(staging.path().join(snapshot_name))?;
    output.write_all(&(encoded.len() as u64).to_le_bytes())?;
    output.write_all(&encoded)?;
    std::io::copy(&mut input, &mut output)?;
    let after = input.metadata()?;
    if after.len() != source_metadata.len()
        || after.modified().ok() != source_metadata.modified().ok()
    {
        return Err(refused(
            "Source checkpoint changed during the copy; wait for its writer to stop.",
        ));
    }
    output.sync_all()?;
    drop(output);
    fs::rename(staging.path(), destination)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_snapshot(path: &Path, metadata: Value) -> Vec<u8> {
        let header = json!({"__metadata__":metadata,
            "factor.test":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
            "optim.test":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}});
        let mut encoded = serde_json::to_vec(&header).unwrap();
        encoded.resize(encoded.len().next_multiple_of(8), b' ');
        let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
        bytes.extend(encoded);
        bytes.extend([1.25f32.to_le_bytes(), 0.75f32.to_le_bytes()].concat());
        fs::write(path, &bytes).unwrap();
        bytes
    }

    #[test]
    fn continuation_forks_only_the_horizon_and_preserves_original_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("original");
        fs::create_dir(&source).unwrap();
        let snapshot = source.join("style-step003000.resume.safetensors");
        let original = write_snapshot(
            &snapshot,
            json!({"format":"candle-gen-resume-v1",
            "step":"3000","update_idx":"3000", "optimizer":"adamw",
            "training_config":"steps=3000;accum=1;scheduler=Cosine;execution_profile=fixture",
            "request_fingerprint":"same-dataset"}),
        );
        for total in [4000, 5000] {
            let destination = temp.path().join(total.to_string());
            fork_completed_training_checkpoint(
                &source,
                &destination,
                "style.safetensors",
                3000,
                total,
            )
            .unwrap();
            let fork = fs::read(destination.join(snapshot.file_name().unwrap())).unwrap();
            assert_eq!(&fork[fork.len() - 8..], &original[original.len() - 8..]);
            let header = sceneworks_core::lora_family::read_safetensors_header(
                &destination.join(snapshot.file_name().unwrap()),
            )
            .unwrap();
            assert_eq!(
                header["__metadata__"]["training_config"],
                format!("steps={total};accum=1;scheduler=Cosine;execution_profile=fixture")
            );
            assert_eq!(header["__metadata__"]["update_idx"], "3000");
            assert_eq!(
                header["__metadata__"]["request_fingerprint"],
                "same-dataset"
            );
            assert_eq!(fs::read(&snapshot).unwrap(), original);
            assert!(fork_completed_training_checkpoint(
                &source,
                &destination,
                "style.safetensors",
                3000,
                total
            )
            .is_err());
        }
    }

    #[test]
    fn mlx_continuation_copies_its_optimizer_companion_without_changing_state() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("original");
        fs::create_dir(&source).unwrap();
        let snapshot_name = "style-step003000.resume.safetensors";
        let optimizer_name = "style-step003000.optim.safetensors";
        let original = write_snapshot(
            &source.join(snapshot_name),
            json!({"step":"3000", "update_idx":"3000", "optimizer":"adamw"}),
        );
        let optimizer = write_snapshot(&source.join(optimizer_name), json!({}));
        let destination = temp.path().join("continued");
        fork_completed_training_checkpoint(&source, &destination, "style.safetensors", 3000, 4000)
            .unwrap();
        assert_eq!(fs::read(destination.join(snapshot_name)).unwrap(), original);
        assert_eq!(
            fs::read(destination.join(optimizer_name)).unwrap(),
            optimizer
        );
        assert_eq!(fs::read(source.join(snapshot_name)).unwrap(), original);
        assert_eq!(fs::read(source.join(optimizer_name)).unwrap(), optimizer);
    }

    #[test]
    fn incomplete_or_mismatched_state_never_publishes_an_output() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("original");
        fs::create_dir(&source).unwrap();
        let destination = temp.path().join("continued");
        let snapshot = source.join("style-step003000.resume.safetensors");
        for metadata in [
            json!({}),
            json!({"step":"2999", "update_idx":"2999"}),
            json!({"step":"3000", "update_idx":"3000", "optimizer":"adamw"}),
            json!({"step":"3000", "update_idx":"3000", "format":"unknown"}),
            json!({"step":"3000", "update_idx":"3000", "format":"candle-gen-resume-v1", "training_config":"steps=1000;accum=1;"}),
        ] {
            write_snapshot(&snapshot, metadata);
            assert!(fork_completed_training_checkpoint(
                &source,
                &destination,
                "style.safetensors",
                3000,
                4000
            )
            .is_err());
            assert!(!destination.exists());
        }
        assert!(fork_completed_training_checkpoint(
            &source,
            &destination,
            "../style.safetensors",
            3000,
            4000
        )
        .is_err());
        assert!(fork_completed_training_checkpoint(
            &source,
            &destination,
            "style.safetensors",
            3000,
            3000
        )
        .is_err());
    }
}
