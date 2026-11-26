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

#[cfg(native_lane)]
#[path = "scail2/evidence.rs"]
mod evidence;

#[cfg(native_lane)]
pub(crate) use plural::{check_scail2_plural_package, Scail2FitProbe, Scail2PluralFit};
#[cfg(all(test, native_lane))]
pub(crate) use plural::{Scail2FitFixture, SCAIL2_TEST_MACHINE};

#[cfg(all(test, native_lane))]
pub(crate) fn scail2_verified_test_quotes(
    bytes: &[u8],
    index_digest: &str,
    os: &str,
    count: usize,
) -> Result<Vec<gen_core::generator::scail2_fit::Scail2FitWitness>, String> {
    Ok(evidence::qualify_index(bytes, index_digest)?
        .into_iter()
        .filter(|q| q.covers(os, count))
        .map(|q| q.witness)
        .collect())
}

#[cfg(native_lane)]
mod plural {
    use super::evidence::{
        meets_macos_minimum, packaged, HostClass, MACOS_MINIMUM_PRODUCT_VERSION,
    };
    use super::*;
    use gen_core::generator::scail2_fit::{Scail2FitPrice, Scail2FitProfile, Scail2FitWitness};
    use gen_core::{LoadSpec, MemoryBudgetDomain, MemoryCacheState};
    use std::{sync::OnceLock, time::Duration};
    use tokio::runtime::Handle;

    /// The digest names the reviewed physical capture that supplied both domains.
    pub(crate) struct WitnessSource {
        artifact_id: u64,
        capture_revision: String,
        digest: String,
        witness: Scail2FitWitness,
    }

    struct Quotation {
        evidence_references: u32,
        artifact_id: u64,
        capture_revision: String,
        evidence: String,
        price: Scail2FitPrice,
    }

    #[cfg(test)]
    tokio::task_local! { pub(crate) static SCAIL2_TEST_MACHINE: (String, String, Option<u64>); }

    /// Parses `nvidia-smi --query-gpu=name,driver_version,memory.total,uuid` (csv, noheader,
    /// nounits) into the hardware class and the queried card's UUID.
    #[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
    fn cuda_class(output: &str) -> Option<(HostClass, String)> {
        let fields: Vec<_> = output.trim().split(',').collect();
        let [device, runtime, total, uuid] = fields.as_slice() else {
            return None;
        };
        let uuid = uuid.trim();
        let id = uuid::Uuid::try_parse(uuid.strip_prefix("GPU-")?).ok()?;
        if cuda_uuid_text(*id.as_bytes()) != uuid {
            return None;
        }
        let class = HostClass::new(device, runtime, Some(total.trim().parse().ok()?))?;
        Some((class, uuid.to_owned()))
    }

    /// `nvidia-smi`'s spelling of a CUDA device UUID.
    #[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
    fn cuda_uuid_text(bytes: [u8; 16]) -> String {
        format!("GPU-{}", uuid::Uuid::from_bytes(bytes).hyphenated())
    }

    async fn hardware(gpu_id: &str) -> Option<HostClass> {
        use tokio::process::Command;
        #[cfg(test)]
        if let Ok((device, runtime, total)) = SCAIL2_TEST_MACHINE.try_with(Clone::clone) {
            return HostClass::new(&device, &runtime, total);
        }
        #[cfg(target_os = "macos")]
        let command = {
            let _ = gpu_id;
            let mut command = Command::new("/usr/sbin/sysctl");
            command.args(["-n", "machdep.cpu.brand_string", "kern.osproductversion"]);
            command
        };
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        let command = {
            let mut command = Command::new("nvidia-smi");
            command.arg(format!("--id={gpu_id}")).args([
                "--query-gpu=name,driver_version,memory.total,uuid",
                "--format=csv,noheader,nounits",
            ]);
            command
        };
        let output = crate::gpu::run_bounded_command(command, Duration::from_secs(3))
            .await
            .ok()?;
        let output = std::str::from_utf8(&output.stdout).ok()?.trim();
        #[cfg(target_os = "macos")]
        let class = {
            let (device, runtime) = output.split_once('\n')?;
            HostClass::new(device, runtime, None)?
        };
        // The generator cache binds CUDA ordinal zero in the worker's visible-device
        // context. A supervisor-supplied nvidia-smi ordinal may refer to another card, so
        // its model, memory and driver count only when it is that same physical device.
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        let class = {
            use runtime_cuda::media::candle_core::cuda::cudarc::driver::result;
            let (class, queried) = cuda_class(output)?;
            result::init().ok()?;
            let device = result::device::get(0).ok()?;
            let native = result::device::get_name(device).ok()?;
            let uuid = result::device::get_uuid(device)
                .ok()?
                .bytes
                .map(|byte| byte as u8);
            if native != class.device || cuda_uuid_text(uuid) != queried {
                return None;
            }
            class
        };
        Some(class)
    }

    #[cfg(test)]
    #[test]
    fn scail_plural_cuda_class_reads_model_driver_and_total_memory() {
        let uuid = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe";
        let (class, queried) = cuda_class(&format!(
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 596.36, 97887, {uuid}\n"
        ))
        .unwrap();
        assert_eq!(queried, uuid);
        let native: [u8; 16] = [
            0xe4, 0xb7, 0x99, 0x31, 0x7b, 0xe6, 0xf2, 0x16, 0x46, 0x0a, 0xf5, 0x40, 0x5c, 0xfa,
            0xff, 0xfe,
        ];
        assert_eq!(cuda_uuid_text(native), uuid);
        assert_ne!(cuda_uuid_text([0; 16]), uuid);
        assert_eq!(
            class,
            HostClass::new(
                "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                "596.36",
                Some(97_887)
            )
            .unwrap()
        );
        for invalid in [
            String::new(),
            "board, 596.36".into(),
            "board, 596.36, 97887".into(),
            format!("board, 596.36, [N/A], {uuid}"),
            format!("board, 596.36, 0, {uuid}"),
            format!("board, 596.36, 97887, {uuid}, extra"),
            format!(", 596.36, 97887, {uuid}"),
            "board, 596.36, 97887, e4b79931-7be6-f216-460a-f5405cfafffe".into(),
            "board, 596.36, 97887, GPU-E4B79931-7BE6-F216-460A-F5405CFAFFFE".into(),
            "board, 596.36, 97887, GPU-e4b799317be6f216460af5405cfafffe".into(),
            "board, 596.36, 97887, [N/A]".into(),
        ] {
            assert_eq!(cuda_class(&invalid), None, "{invalid}");
        }
    }

    async fn hardware_witnesses(
        gpu: &str,
        count: usize,
    ) -> crate::WorkerResult<Vec<WitnessSource>> {
        let rows = packaged()?;
        let eligible =
            |r: &super::evidence::QualifiedWitness| r.covers(std::env::consts::OS, count);
        if !rows.iter().any(eligible) {
            return Ok(Vec::new());
        }
        let host = hardware(gpu).await.ok_or_else(|| {
            crate::WorkerError::InvalidPayload(
                "SCAIL-2 device and runtime identity unavailable".into(),
            )
        })?;
        if std::env::consts::OS == "macos" && !meets_macos_minimum(&host.runtime) {
            return Err(crate::WorkerError::InvalidPayload(format!(
                "SCAIL-2 with {count} ordered character references requires macOS {MACOS_MINIMUM_PRODUCT_VERSION} or later; this Mac reports macOS {}",
                host.runtime
            )));
        }
        Ok(rows
            .iter()
            .filter(|r| eligible(r) && r.matches_host(&host))
            .map(|r| WitnessSource {
                artifact_id: r.artifact_id,
                capture_revision: r.inference_revision.clone(),
                digest: r.evidence_sha256.clone(),
                witness: r.witness.clone(),
            })
            .collect())
    }

    pub(crate) async fn check_scail2_plural_package(
        gpu: &str,
        count: usize,
    ) -> crate::WorkerResult<()> {
        if count > 1 && hardware_witnesses(gpu, count).await?.is_empty() {
            return Err(crate::WorkerError::InvalidPayload(format!("SCAIL-2 has no verified accelerator and host memory fit for {count} ordered character references on this device and runtime")));
        }
        Ok(())
    }

    #[cfg(any(windows, test))]
    fn windows_available_host_bytes(text: &str) -> Option<u64> {
        let mut counters = text.lines().map(|s| s.trim().parse::<u64>().ok());
        let physical = counters.next()??;
        let commit_limit = counters.next()??;
        let committed = counters.next()??;
        if counters.next().is_some() || commit_limit == 0 {
            return None;
        }
        Some(physical.min(commit_limit.checked_sub(committed)?))
    }

    #[cfg(windows)]
    async fn available_host() -> Option<u64> {
        let mut command = tokio::process::Command::new("powershell.exe");
        command.args(["-NoProfile", "-NonInteractive", "-Command",
            "$m = Get-CimInstance Win32_PerfRawData_PerfOS_Memory -ErrorAction Stop; $m.AvailableBytes; $m.CommitLimit; $m.CommittedBytes"]);
        let output = crate::gpu::run_bounded_command(command, Duration::from_secs(10))
            .await
            .ok()?;
        windows_available_host_bytes(std::str::from_utf8(&output.stdout).ok()?)
    }

    #[cfg(test)]
    #[test]
    fn scail_plural_windows_host_capacity_requires_both_ram_and_commit() {
        assert_eq!(windows_available_host_bytes("40\n80\n20\n"), Some(40));
        assert_eq!(
            windows_available_host_bytes("100\r\n80\r\n20\r\n"),
            Some(60)
        );
        assert_eq!(windows_available_host_bytes("100\n80\n80"), Some(0));
        for invalid in ["100\n80", "100\n80\n90", "100\n0\n0", "broken\n80\n5"] {
            assert_eq!(windows_available_host_bytes(invalid), None);
        }
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
        pub(crate) fn witnesses(&self, count: usize) -> crate::WorkerResult<Vec<WitnessSource>> {
            #[cfg(test)]
            if let Some(fixture) = &self.fixture {
                return Ok(fixture
                    .rows
                    .iter()
                    .cloned()
                    .map(|witness| WitnessSource {
                        artifact_id: 0,
                        capture_revision: "1".repeat(40),
                        digest: "test-capture".into(),
                        witness,
                    })
                    .collect());
            }
            self.runtime.block_on(hardware_witnesses(&self.gpu, count))
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
                // A Windows Job/WDDM capture includes private commit and resident RAM.
                // A Linux MemAvailable reading lacks the effective cgroup ancestor limits.
                #[cfg(windows)]
                let physical = self.runtime.block_on(available_host());
                #[cfg(not(windows))]
                let physical = None;
                Ok((state.budget, physical))
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
                        evidence_references: row.witness.axes.references,
                        artifact_id: row.artifact_id,
                        capture_revision: row.capture_revision.clone(),
                        evidence: row.digest.clone(),
                        price,
                    });
                }
            }
            // A higher reference count can measure cheaper than a lower one (a sampled
            // allocation burst, run-to-run variation), so price only at the covering
            // measurement whose reference count is closest to and at least this request's.
            if let Some(nearest) = prices.iter().map(|q| q.evidence_references).min() {
                prices.retain(|q| q.evidence_references == nearest);
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
                    referenceRgbSha256=%self.reference_digests, evidenceSha256=%quote.evidence, evidenceArtifactId=quote.artifact_id, evidenceReferences=quote.evidence_references,
                    evidenceInferenceRevision=%quote.capture_revision, inferenceRevision=sceneworks_core::inference_provenance::INFERENCE_REVISION,
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
