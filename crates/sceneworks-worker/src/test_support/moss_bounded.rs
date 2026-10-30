//! Manual proof observation on the production blocking training thread.
use super::*;
pub use candle_gen::testkit::PeakSampler;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub type BoundedMossReceipt = Arc<Mutex<serde_json::Value>>;

/// Called by the worker's existing blocking loader. This loads the production
/// registry trainer, with no TrainingProbe or alternate trainer implementation.
pub fn real_bounded_moss_trainer(
    engine: &str,
    spec: &gen_core::LoadSpec,
    metrics: PathBuf,
    receipt: BoundedMossReceipt,
) -> gen_core::Result<Box<dyn gen_core::Trainer>> {
    if engine != "moss_image_base" {
        return Err(gen_core::Error::Unsupported(
            "bounded product case requires MOSS Base".into(),
        ));
    }
    let inner = crate::inference_runtime::load_trainer(engine, spec)?;
    *receipt
        .lock()
        .map_err(|e| gen_core::Error::Msg(e.to_string()))? = serde_json::json!({
        "provider":inner.descriptor().id, "backend":inner.descriptor().backend,
        "actualLoadSpec":format!("{spec:#?}"), "steps":[], "nativeReturned":false
    });
    Ok(Box::new(Observed {
        inner,
        metrics,
        receipt,
    }))
}

struct Observed {
    inner: Box<dyn gen_core::Trainer>,
    metrics: PathBuf,
    receipt: BoundedMossReceipt,
}
impl gen_core::Trainer for Observed {
    fn descriptor(&self) -> &gen_core::TrainerDescriptor {
        self.inner.descriptor()
    }
    fn validate(&self, request: &gen_core::TrainingRequest) -> gen_core::Result<()> {
        self.inner.validate(request)
    }
    fn train(
        &mut self,
        request: &gen_core::TrainingRequest,
        progress: &mut dyn FnMut(gen_core::TrainingProgress),
    ) -> gen_core::Result<gen_core::TrainingOutput> {
        // COHORT is thread-local. The async caller cannot own this scope.
        let receipt = Arc::clone(&self.receipt);
        let mut observation_error = None;
        let mut native_result = None;
        // A partial/cancelled run cannot satisfy the proof's four-update assertions.
        // Retain the original native error instead of turning cancellation into a
        // blocking-task panic; a genuinely panicking native trainer still unwinds.
        let proof = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            candle_gen::train::profile::with_bounded_cohort_for(
                &self.metrics,
                &[
                    "moss:geometry-storage-hit",
                    "moss:frozen-caption-storage-hit",
                    "moss:frozen-projection-pack",
                ],
                || {
                    native_result = Some(self.inner.train(request, &mut |event| {
                        if let gen_core::TrainingProgress::Training { step, total, loss } = &event {
                            match receipt.lock() {
                                Ok(mut r) => {
                                    r["steps"].as_array_mut().expect("owned steps").push(
                                        serde_json::json!({"step":step,"total":total,"loss":loss}),
                                    );
                                    if *total != 4 || !loss.is_finite() {
                                        observation_error = Some(
                                            "changed native horizon or nonfinite loss".to_owned(),
                                        );
                                    }
                                }
                                Err(e) => observation_error = Some(e.to_string()),
                            }
                        }
                        progress(event); // Forward every original event exactly once.
                    }));
                },
            )
        }));
        let result = match native_result {
            Some(result) => result,
            None => match proof {
                Err(panic) => std::panic::resume_unwind(panic),
                Ok(()) => return Err(gen_core::Error::Msg("native trainer did not return".into())),
            },
        };
        let mut r = receipt
            .lock()
            .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
        r["nativeReturned"] = serde_json::json!(true);
        match &result {
            Ok(output) => {
                r["output"] = serde_json::json!({"adapterPath":output.adapter_path,"steps":output.steps,"finalLoss":output.final_loss});
            }
            Err(error) => {
                r["error"] = serde_json::json!(error.to_string());
            }
        }
        r["boundedObserverCompleted"] = serde_json::json!(proof.is_ok());
        if result.is_err() {
            return result;
        }
        if proof.is_err() {
            return Err(gen_core::Error::Msg(
                "native training returned but bounded observer did not complete".into(),
            ));
        }
        if let Some(error) = observation_error {
            return Err(gen_core::Error::Msg(error));
        }
        result
    }
}
