//! video_admission scail2 owner.
use super::*;

/// Worker seal for both SCAIL-2 carriers (`animate_character` and `replace_person`), which share
/// one physical assembly built by `video_jobs::scail2`: one or more ordered `Reference` stills and their painted
/// `Mask`s, and a `ControlClip` of driving frames plus per-frame color masks. The engine TASK is a
/// separate axis because animation and replacement are different working sets over the same shapes.
pub(crate) fn scail2_carrier_receipt(
    lane: VideoLane,
    mode: &str,
    width: u32,
    height: u32,
    conditioning: &[Conditioning],
) -> Result<String, String> {
    gen_core::generator::Scail2AnimationConditioningRef::catalog_receipt(
        backend_axis(lane),
        mode,
        width,
        height,
        conditioning,
    )
    .map_err(|error| error.to_string())
}

#[cfg(all(test, native_lane))]
pub(crate) use plural::Scail2FitFixture;
#[cfg(native_lane)]
pub(crate) use plural::{Scail2FitProbe, Scail2PluralFit};

#[cfg(native_lane)]
mod plural {
    use super::*;
    use gen_core::generator::scail2_fit::{Scail2FitPrice, Scail2FitProfile, Scail2FitWitness};
    use gen_core::{LoadSpec, MemoryBudgetDomain, MemoryCacheState};
    use serde::Deserialize;
    use std::{sync::OnceLock, time::Duration};
    use tokio::runtime::Handle;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct QualifiedWitness {
        platform: String,
        inference_revision: String,
        device: String,
        runtime: String,
        evidence_sha256: String,
        witness: Scail2FitWitness,
    }

    /// The digest names the reviewed physical capture that supplied both domains.
    pub(crate) struct WitnessSource {
        digest: String,
        witness: Scail2FitWitness,
    }

    struct Quotation {
        evidence: String,
        price: Scail2FitPrice,
    }

    fn packaged() -> crate::WorkerResult<&'static [QualifiedWitness]> {
        static ROWS: OnceLock<Result<Vec<QualifiedWitness>, String>> = OnceLock::new();
        ROWS.get_or_init(|| {
            serde_json::from_str(include_str!("../../../../config/scail2-plural-memory.json"))
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map(Vec::as_slice)
        .map_err(|e| {
            crate::WorkerError::InvalidPayload(format!(
                "invalid SCAIL-2 plural memory evidence: {e}"
            ))
        })
    }

    async fn hardware(gpu_id: &str) -> Option<(String, String)> {
        use tokio::process::Command;
        #[cfg(target_os = "macos")]
        let (command, separator) = {
            let _ = gpu_id;
            let mut command = Command::new("/usr/sbin/sysctl");
            command.args(["-n", "machdep.cpu.brand_string", "kern.osversion"]);
            (command, '\n')
        };
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        let (command, separator) = {
            let mut command = Command::new("nvidia-smi");
            command.arg(format!("--id={gpu_id}")).args([
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ]);
            (command, ',')
        };
        let output = crate::gpu::run_bounded_command(command, Duration::from_secs(3))
            .await
            .ok()?;
        let output = std::str::from_utf8(&output.stdout).ok()?.trim();
        let (device, runtime) = output.split_once(separator)?;
        let (device, runtime) = (device.trim(), runtime.trim());
        // The generator cache binds CUDA ordinal zero in the worker's visible-device
        // context. A supervisor-supplied nvidia-smi ordinal may refer to another class.
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        {
            use runtime_cuda::media::candle_core::cuda::cudarc::driver::result;
            result::init().ok()?;
            let native = result::device::get_name(result::device::get(0).ok()?).ok()?;
            if native != device {
                return None;
            }
        }
        (!device.is_empty() && !runtime.is_empty() && !runtime.contains(['\n', ',']))
            .then(|| (device.to_owned(), runtime.to_owned()))
    }

    #[cfg(test)]
    pub(crate) struct Scail2FitFixture {
        pub(crate) rows: Vec<Scail2FitWitness>,
        pub(crate) budget: gen_core::MemoryBudget,
        pub(crate) physical_host: Option<u64>,
    }

    /// Lives only for this request. A cold miss and a cache hit read their own fresh
    /// snapshots on the serialized generator owner.
    pub(crate) struct Scail2FitProbe {
        pub(crate) runtime: Handle,
        pub(crate) gpu: String,
        #[cfg(test)]
        pub(crate) fixture: Option<Scail2FitFixture>,
    }

    impl Scail2FitProbe {
        pub(crate) fn witnesses(&self) -> crate::WorkerResult<Vec<WitnessSource>> {
            #[cfg(test)]
            if let Some(fixture) = &self.fixture {
                return Ok(fixture
                    .rows
                    .iter()
                    .cloned()
                    .map(|witness| WitnessSource {
                        digest: "test-capture".into(),
                        witness,
                    })
                    .collect());
            }
            let rows = packaged()?;
            let current = |row: &QualifiedWitness| {
                row.platform == std::env::consts::OS
                    && row.inference_revision
                        == sceneworks_core::inference_provenance::INFERENCE_REVISION
            };
            if !rows.iter().any(current) {
                return Ok(Vec::new());
            }
            let (device, version) =
                self.runtime.block_on(hardware(&self.gpu)).ok_or_else(|| {
                    crate::WorkerError::InvalidPayload(
                        "SCAIL-2 device and runtime identity unavailable".into(),
                    )
                })?;
            Ok(rows
                .iter()
                .filter(|r| current(r))
                .filter(|r| r.device == device && r.runtime == version)
                .filter(|r| {
                    r.evidence_sha256.len() == 64
                        && r.evidence_sha256
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
                .map(|r| WitnessSource {
                    digest: r.evidence_sha256.clone(),
                    witness: r.witness.clone(),
                })
                .collect())
        }

        fn budget(
            &self,
            spec: &LoadSpec,
            cache: MemoryCacheState,
            resident: u64,
        ) -> crate::WorkerResult<(gen_core::MemoryBudget, Option<u64>)> {
            #[cfg(test)]
            if let Some(fixture) = &self.fixture {
                return Ok((fixture.budget, fixture.physical_host));
            }
            let state =
                live_video_runtime_state("scail2_14b", cache, spec.offload_policy, resident, None)
                    .map_err(|error| match error {
                        crate::WorkerError::Engine(detail) => crate::WorkerError::InvalidPayload(
                            format!("SCAIL-2 accelerator budget is unavailable: {detail}"),
                        ),
                        other => other,
                    })?
                    .ok_or_else(|| {
                        crate::WorkerError::InvalidPayload(
                            "SCAIL-2 accelerator budget is unavailable".into(),
                        )
                    })?;
            #[cfg(target_os = "macos")]
            {
                let mut budget = state.budget;
                let working_set =
                    crate::fit_gate::working_set_ceiling_bytes().ok_or_else(|| {
                        crate::WorkerError::InvalidPayload(
                            "SCAIL-2 Metal working set is unavailable".into(),
                        )
                    })?;
                budget.total_bytes = budget.total_bytes.min(working_set);
                Ok((
                    budget,
                    self.runtime
                        .block_on(crate::gpu::available_unified_memory_bytes()),
                ))
            }
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            {
                Ok((
                    state.budget,
                    self.runtime
                        .block_on(crate::yue2_admission::host_available_bytes()),
                ))
            }
        }
    }

    pub(crate) struct Scail2PluralFit {
        profile: Scail2FitProfile,
        prices: Vec<Quotation>,
        selected: OnceLock<usize>,
        geometry: MemoryGeometry,
        source: u64,
        accelerator_buffers: u64,
        host_buffers: u64,
        receipt: String,
        job: String,
        reference_digests: String,
        probe: Scail2FitProbe,
    }

    impl Scail2PluralFit {
        pub(crate) fn prepare(
            reservation: &PlannedVideoReservation,
            spec: &LoadSpec,
            request: &gen_core::GenerationRequest,
            job: &str,
            probe: Scail2FitProbe,
            rows: &[WitnessSource],
        ) -> crate::WorkerResult<Self> {
            let geometry = reservation.plan.geometry();
            let missing = |reason: &str| {
                crate::WorkerError::InvalidPayload(format!(
                "SCAIL-2 has no verified accelerator and host memory fit for {} ordered character references at {}x{} across {} frames on {}: {reason}",
                geometry.reference_count, geometry.width, geometry.height, geometry.frames, LANE.as_key()
            ))
            };
            if rows.is_empty() {
                return Err(missing("no matching device and runtime measurement"));
            }
            let profile = Scail2FitProfile::from_plan(spec, &reservation.plan, request)
                .map_err(|e| missing(&e.to_string()))?
                .ok_or_else(|| missing("missing native plural profile"))?;
            // A row prices all physical domains together. Two measurements can be
            // incomparable, so retain every distinct qualified quotation until the live probe.
            let mut prices = Vec::new();
            for row in rows {
                if let Ok(price) = profile.price(std::slice::from_ref(&row.witness)) {
                    prices.push(Quotation {
                        evidence: row.digest.clone(),
                        price,
                    });
                }
            }
            if prices.is_empty() {
                return Err(missing("no completed measurement covers the native plan"));
            }
            let domain = reservation.plan.layout().domain;
            let host = if domain == MemoryBudgetDomain::UnifiedMemory {
                domain
            } else {
                MemoryBudgetDomain::HostRam
            };
            Ok(Self {
                profile,
                prices,
                selected: OnceLock::new(),
                geometry,
                source: reservation.plan.source_bytes(host),
                accelerator_buffers: reservation
                    .plan
                    .minimum_live_bytes(domain)
                    .map_err(|e| missing(&e.to_string()))?,
                host_buffers: reservation
                    .plan
                    .minimum_live_bytes(host)
                    .map_err(|e| missing(&e.to_string()))?,
                receipt: reservation
                    .overlay
                    .clone()
                    .ok_or_else(|| missing("missing ordered carrier identity"))?,
                job: job.to_owned(),
                reference_digests: reservation
                    .plan
                    .reference_rgb_digests()
                    .iter()
                    .map(|digest| sceneworks_core::hashing::hex(digest))
                    .collect::<Vec<_>>()
                    .join(","),
                probe,
            })
        }

        pub(crate) fn with_execution<R>(
            &self,
            generate: impl FnOnce() -> crate::WorkerResult<R>,
        ) -> crate::WorkerResult<R> {
            let price = self.selected.get().ok_or_else(|| {
                crate::WorkerError::InvalidPayload(
                    "SCAIL-2 execution has no admitted memory price".into(),
                )
            })?;
            self.prices[*price].price.with_execution(generate)
        }

        pub(crate) fn check_live(
            &self,
            spec: &LoadSpec,
            cache: MemoryCacheState,
            stage: &str,
            resident: u64,
        ) -> crate::WorkerResult<()> {
            self.profile.validate_load(spec).map_err(|e| {
                crate::WorkerError::InvalidPayload(format!("SCAIL-2 fit source changed: {e}"))
            })?;
            // Read on the cache owner. Both the cold factory and warmed generator have the
            // selected backend context bound; no unrelated process or generator earns credit.
            let (budget, physical) = self.probe.budget(spec, cache, resident)?;
            let credit = resident.min(budget.committed_bytes);
            let free = budget.effective_bytes();
            let mut refusal = None;
            for (index, quote) in self.prices.iter().enumerate() {
                if self
                    .selected
                    .get()
                    .is_some_and(|selected| *selected != index)
                {
                    continue;
                }
                let price = &quote.price;
                let (accelerator, host) = required(
                    price,
                    LANE,
                    self.source,
                    self.accelerator_buffers,
                    self.host_buffers,
                    credit,
                )?;
                let result = check_available(accelerator, host, Some(free), physical);
                tracing::info!(event="scail2_plural_fit", jobId=%self.job, stage, backend=LANE.as_key(),
                    tier=%self.profile.axes().effective_tier, artifactTier=%self.profile.axes().artifact_tier,
                    referenceRgbSha256=%self.reference_digests, evidenceSha256=%quote.evidence,
                    grossHostBytes=price.host_bytes, grossDeviceBytes=price.device_bytes, grossUnifiedBytes=price.unified_bytes,
                    acceleratorReserveBytes=budget.reserved_headroom_bytes,
                    artifactSha256=%self.profile.axes().artifact_sha256,
                    width=self.geometry.width, height=self.geometry.height, frames=self.geometry.frames,
                    referenceCount=self.geometry.reference_count, carrierReceipt=%self.receipt,
                    requiredAcceleratorBytes=accelerator, availableAcceleratorBytes=free,
                    requiredPhysicalHostBytes=host, availablePhysicalHostBytes=physical,
                    providerCreditBytes=credit, admitted=result.is_ok(), "Checked SCAIL-2 plural memory before generation");
                match result {
                    Ok(()) => {
                        self.selected.get_or_init(|| index);
                        return Ok(());
                    }
                    Err(error) => refusal = Some(error),
                }
            }
            Err(refusal.unwrap_or_else(|| {
                crate::WorkerError::InvalidPayload("SCAIL-2 has no verified memory price".into())
            }))
        }
    }

    fn required(
        price: &Scail2FitPrice,
        lane: VideoLane,
        source: u64,
        accelerator_buffers: u64,
        host_buffers: u64,
        credit: u64,
    ) -> crate::WorkerResult<(u64, u64)> {
        let (accelerator, physical) = match lane {
            VideoLane::Mlx => (price.unified_bytes, price.unified_bytes),
            VideoLane::Candle => (price.device_bytes, price.host_bytes),
        };
        let new_accelerator = accelerator.saturating_sub(credit).max(accelerator_buffers);
        let physical_credit = if lane == VideoLane::Mlx { credit } else { 0 };
        let new_host = physical
            .saturating_sub(physical_credit)
            .max(host_buffers)
            .checked_sub(source)
            .ok_or_else(|| {
                crate::WorkerError::InvalidPayload(
                    "SCAIL-2 host price is below its owned source bytes".into(),
                )
            })?;
        Ok((new_accelerator, new_host))
    }

    fn check_available(
        accelerator: u64,
        host: u64,
        free: Option<u64>,
        physical: Option<u64>,
    ) -> crate::WorkerResult<()> {
        for (pool, required, available) in [
            ("accelerator", accelerator, free),
            ("physical host", host, physical),
        ] {
            let available = available.ok_or_else(|| crate::WorkerError::InvalidPayload(format!("SCAIL-2 plural fit requires {required} {pool} bytes; available capacity could not be verified")))?;
            if available < required {
                return Err(crate::WorkerError::InvalidPayload(format!("SCAIL-2 plural fit requires {required} {pool} bytes, only {available} are available")));
            }
        }
        Ok(())
    }
}
