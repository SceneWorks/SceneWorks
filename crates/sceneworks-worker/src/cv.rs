//! CV application boundary. Engines belong to the pinned runtime; file loading
//! remains here to preserve validated ONNX external data and load-progress events.
#[cfg(not(target_os = "macos"))]
pub(crate) use runtime_cuda::cv::*;
#[cfg(target_os = "macos")]
pub(crate) use runtime_macos::cv::*;

pub(crate) struct ModelIoLoader;

impl session::SessionLoader for ModelIoLoader {
    fn prepare_accelerator(&mut self) {
        #[cfg(not(target_os = "macos"))]
        crate::ort_cuda::preload_cuda_dylibs();
    }

    fn load(
        &mut self,
        builder: ort::session::builder::SessionBuilder,
        path: &std::path::Path,
    ) -> Result<ort::session::Session> {
        #[cfg(test)]
        let builder = if let Ok(directory) = std::env::var("CV_PROOF_OUTPUT") {
            static NEXT_PROFILE: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let sequence = NEXT_PROFILE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            builder
                .with_profiling(std::path::Path::new(&directory).join(format!("ort-{sequence}")))
                .map_err(Error::from)?
        } else {
            builder
        };
        crate::model_io::ort_session(builder, path).map_err(|error| Error::Load(Box::new(error)))
    }
}

impl From<Error> for crate::WorkerError {
    fn from(error: Error) -> Self {
        match error {
            Error::Engine(message) => Self::Engine(message),
            Error::InvalidPayload(message) => Self::InvalidPayload(message),
            Error::Canceled(message) => Self::Canceled(message),
            Error::Load(source) => match source.downcast::<Self>() {
                Ok(error) => *error,
                Err(error) => Self::Engine(error.to_string()),
            },
        }
    }
}

#[cfg(test)]
pub(crate) fn proof_complete(name: &str, device: &str, result: serde_json::Value) {
    // The dispatch controller requires this receipt from the completed body,
    // in addition to exact-test discovery and a successful libtest summary.
    if let Ok(expected) = std::env::var("CV_EXPECTED_DEVICE") {
        assert_eq!(device, expected, "selected CV provider silently fell back");
    }
    if let Ok(directory) = std::env::var("CV_PROOF_OUTPUT") {
        let receipt = serde_json::json!({"case": name, "device": device, "completed": true, "result": result});
        std::fs::write(
            std::path::Path::new(&directory).join("case.json"),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .expect("CV proof receipt");
    }
}

#[cfg(test)]
pub(crate) fn proof_ort_complete(
    name: &str,
    device: &str,
    mut result: serde_json::Value,
    profiles: Vec<std::path::PathBuf>,
) {
    let provider = match device {
        "cuda" => "CUDAExecutionProvider",
        "coreml" => "CoreMLExecutionProvider",
        _ => panic!("hardware provider proof fell back to {device}"),
    };
    let mut summaries = Vec::new();
    for file in profiles {
        let events: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(&file).expect("ORT profile"))
                .expect("ORT profile JSON");
        let mut counts = std::collections::BTreeMap::<String, usize>::new();
        for event in events {
            if let Some(provider) = event["args"]["provider"].as_str() {
                *counts.entry(provider.to_owned()).or_default() += 1;
            }
        }
        assert!(
            counts.get(provider).is_some_and(|count| *count > 0),
            "required {provider} executed no profiled nodes: {counts:?}"
        );
        summaries.push(serde_json::json!({"file":file.file_name().unwrap().to_string_lossy(), "executions":counts}));
    }
    assert!(
        !summaries.is_empty(),
        "ORT proof needs executed provider profiles"
    );
    result["providerExecutions"] = serde_json::json!(summaries);
    proof_complete(name, device, result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_and_engine_errors_keep_worker_classification() {
        for original in [
            crate::WorkerError::InvalidPayload("bad external data".into()),
            crate::WorkerError::ExternalLibraryUnavailable("unmounted".into()),
            crate::WorkerError::Io(std::io::Error::from(std::io::ErrorKind::NotFound)),
        ] {
            let variant = std::mem::discriminant(&original);
            let message = original.to_string();
            let roundtrip = crate::WorkerError::from(Error::Load(Box::new(original)));
            assert_eq!(std::mem::discriminant(&roundtrip), variant);
            assert_eq!(roundtrip.to_string(), message);
        }
        assert!(matches!(
            crate::WorkerError::from(Error::Canceled("cancel".into())),
            crate::WorkerError::Canceled(_)
        ));
        assert!(matches!(
            crate::WorkerError::from(Error::InvalidPayload("shape".into())),
            crate::WorkerError::InvalidPayload(_)
        ));
        assert!(matches!(
            crate::WorkerError::from(Error::Engine("ORT".into())),
            crate::WorkerError::Engine(_)
        ));
    }
}
