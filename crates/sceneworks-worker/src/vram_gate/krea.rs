//! vram_gate krea owner.
use super::*;

/// Translate the shared strategy vocabulary to Krea's historical manifest keys at the evidence-read
/// boundary. This is deliberately a function, not a second rung enum: selection order and execution
/// composition remain owned by [`gen_core::MemoryStrategy`] and the provider contract.
pub(super) fn krea_turbo_manifest_key(strategy: gen_core::MemoryStrategy) -> &'static str {
    match strategy {
        gen_core::MemoryStrategy::Resident => "resident",
        gen_core::MemoryStrategy::StagedResidency => "threeStage",
        gen_core::MemoryStrategy::BoundedDecode => "tiledVae",
        gen_core::MemoryStrategy::BoundedAttention => "chunkedAttention",
        gen_core::MemoryStrategy::BoundedTransformerResidency => "streamedBlocks",
    }
}

impl BindingPhase {
    /// The phase index (0 text, 1 denoise, 2 decode) the sc-18097 comparison seam is written in.
    pub(crate) fn index(self) -> u8 {
        match self {
            Self::Text => 0,
            Self::Denoise => 1,
            Self::Decode => 2,
        }
    }
}

impl KreaTurboPhasePeaks {
    pub(crate) fn peak_gb(self) -> f64 {
        self.text_gb.max(self.denoise_gb).max(self.decode_gb)
    }

    /// The phase carrying the peak. Ties resolve to the LATER phase deterministically, mirroring
    /// the MLX gate's `binding_phase` (sc-18096).
    ///
    /// This is why sc-18810 put the temporal term on each phase curve rather than on the
    /// aggregate: `peak_gb` is a `max` over three curves, and a max of linear pieces is not
    /// linear. Fitting one temporal coefficient to the aggregate missed it by >= 10.26 GiB on
    /// held-out geometries — about 94x the measured noise floor — while the ADOPTED per-phase
    /// form stays inside 0.44 GiB: 0.0185 decode, 0.1741 denoise, 0.4438 text. Keeping the curves
    /// per phase is what makes both `peak_gb` and this question answerable at an unmeasured
    /// geometry.
    ///
    /// That per-phase band is 0.019-0.44 GiB, not the tighter 0.30 an earlier revision of this
    /// comment quoted: 0.30 is `area_only`'s TEXT residual, and no cross-form phase equals it.
    /// The distinction matters because the win is not uniform. On text the rejected additive form
    /// scores 0.3445 against cross's 0.4438, so **cross is 1.29x worse there**. Cross is still the
    /// correct adoption: it beats additive 355x on decode and 15.4x on denoise — the phases that
    /// actually carry the temporal response and decide admission at scale — it strictly contains
    /// the constrained candidates sharing its regressors (`area_only`, `output_voxels`), and the
    /// 0.0993 GiB it concedes on text is under that phase's own 0.1095 GiB replicate floor, the
    /// largest of the three by orders of magnitude. Every figure here is graded from the committed
    /// artifact rather than from this comment, by the test
    /// `the_adopted_form_is_the_only_accurate_one_that_extends_the_shipped_curve`.
    pub(crate) fn binding_phase(self) -> BindingPhase {
        let mut phase = BindingPhase::Text;
        let mut peak = self.text_gb;
        if self.denoise_gb >= peak {
            phase = BindingPhase::Denoise;
            peak = self.denoise_gb;
        }
        if self.decode_gb >= peak {
            phase = BindingPhase::Decode;
        }
        phase
    }
}

impl KreaTurboFit {
    /// The `image_memory_strategy_selected` telemetry payload for a selection this lane made
    /// (sc-22667, epic 22657 E7) — the same shape the shared candle ladder emits
    /// (`candle_memory_strategy::CandleMemoryEvaluation::selection_telemetry`): the selected rung
    /// and its parameters, the basis that priced it, the three phase peaks where a phase
    /// decomposition priced it (every optimized rung: the measured curves, the fitted curves and
    /// the anchor derivation are all per phase), and the selector's admission figures. `None`
    /// for a refusal or an unverified result, which select nothing.
    pub(crate) fn selection_telemetry(&self, tier: &str, width: u32, height: u32) -> Option<Value> {
        let (selection, basis, phases, admitted, predicted_peak_gb) = match self {
            Self::Resident {
                peak_gb,
                selection,
                admitted,
                ..
            } => (
                selection,
                crate::memory_strategy::CandidateBasis::Measured,
                None,
                admitted,
                *peak_gb,
            ),
            Self::Fits {
                phases,
                selection,
                basis,
                admitted,
                ..
            } => (selection, *basis, Some(*phases), admitted, phases.peak_gb()),
            Self::Reject { .. } | Self::Unverified { .. } => return None,
        };
        let bytes = |gb: f64| (gb * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64;
        let parameters = selection.parameters;
        Some(serde_json::json!({
            "backend": "candle",
            "route": "krea_2_turbo",
            "actualTier": tier,
            "mode": "text_to_image",
            "geometry": {
                "width": width,
                "height": height,
                "batch": 1,
                "frames": KREA_LANE_FRAMES,
            },
            "referenceCount": 0,
            "overlay": Value::Null,
            "strategy": crate::candle_memory_strategy::strategy_label(selection.strategy),
            "parameters": {
                "decodeTileEdge": parameters.decode_tile_edge,
                "decodeOverlap": parameters.decode_overlap,
                "attentionChunkSize": parameters.attention_chunk_size,
                "transformerWindowSize": parameters.transformer_window_size,
            },
            "basis": basis.as_key(),
            "authority": if basis.is_estimate() { "estimated" } else { "calibrated" },
            "predictedPeakBytes": bytes(predicted_peak_gb),
            "phasePeakBytes": phases.map(|phases| serde_json::json!({
                "conditioning": bytes(phases.text_gb),
                "denoise": bytes(phases.denoise_gb),
                "decode": bytes(phases.decode_gb),
            })),
            "admittedPeakGb": admitted.needed_gb,
            "availableGb": admitted.available_gb,
            "reserveGb": admitted.reserve_gb,
            "evidenceRevision": KREA_TURBO_SCENEWORKS_REVISION,
        }))
    }
}

pub(crate) const KREA_TURBO_SCENEWORKS_REVISION: &str = "sc-15449-contract-v1";

impl KreaRuntimeEvidenceContext {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inspect(
        resolved_route: &str,
        backend: &str,
        gpu_id: &str,
        compute_capability: Option<f32>,
        artifact_provider: &str,
        artifact_repository: &str,
        resolved_revision: &str,
        tier_root: &str,
        resolved_artifact_root: &std::path::Path,
        pinned_snapshot_root: &std::path::Path,
    ) -> Option<Self> {
        let compute_capability = compute_capability?;
        Self::inspect_artifact_structure(tier_root, resolved_artifact_root, pinned_snapshot_root)?;
        Some(Self {
            resolved_route: resolved_route.to_owned(),
            backend: backend.to_owned(),
            gpu_id: gpu_id.to_owned(),
            compute_capability,
            artifact_provider: artifact_provider.to_owned(),
            artifact_repository: artifact_repository.to_owned(),
            resolved_revision: resolved_revision.to_owned(),
            tier_root: tier_root.to_owned(),
            provider_contract: None,
        })
    }

    /// Inspect the stock Turbo artifact and acquire its full registered provider contract once.
    ///
    /// Krea Control also uses [`Self::inspect`] for structural base-snapshot validation, but its
    /// execution contract is different. Keep that path metadata-only; only the Turbo fit ladder
    /// needs this sealed, reusable provider contract.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inspect_turbo_fit(
        resolved_route: &str,
        backend: &str,
        gpu_id: &str,
        compute_capability: Option<f32>,
        artifact_provider: &str,
        artifact_repository: &str,
        resolved_revision: &str,
        tier_root: &str,
        resolved_artifact_root: &std::path::Path,
        pinned_snapshot_root: &std::path::Path,
    ) -> Option<Self> {
        let mut context = Self::inspect(
            resolved_route,
            backend,
            gpu_id,
            compute_capability,
            artifact_provider,
            artifact_repository,
            resolved_revision,
            tier_root,
            resolved_artifact_root,
            pinned_snapshot_root,
        )?;
        // The registered lookup performs complete production load-source validation, including
        // encoder ArtifactSeal acquisition. The path-free result is safe to reuse for every fit
        // against this inspected request-local artifact; the eventual provider load independently
        // validates its retained seals before opening model bytes.
        let provider_contract = crate::inference_runtime::media()
            .memory_strategy_contract(
                "krea_2_turbo",
                &gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(
                    resolved_artifact_root.to_owned(),
                )),
            )
            .ok()
            .flatten()?;
        context.provider_contract = Some(provider_contract);
        Some(context)
    }

    /// Cheap snapshot-layout validation kept separate from the provider's full sealed-source
    /// validation so its missing-shard behavior can be exercised with a tiny structural fixture.
    pub(super) fn inspect_artifact_structure(
        tier_root: &str,
        resolved_artifact_root: &std::path::Path,
        pinned_snapshot_root: &std::path::Path,
    ) -> Option<()> {
        let expected = pinned_snapshot_root.join(tier_root);
        if resolved_artifact_root.canonicalize().ok()? != expected.canonicalize().ok()? {
            return None;
        }
        let components = sceneworks_core::mlx_tier_completeness::tier_declared_components(
            resolved_artifact_root,
        )?;
        if components.is_empty()
            || !components.iter().all(|component| {
                std::fs::read_dir(resolved_artifact_root.join(component)).is_ok_and(|entries| {
                    entries
                        .flatten()
                        .any(|entry| !sceneworks_core::lora_family::is_hidden_file(&entry.path()))
                })
            })
        {
            return None;
        }
        let parse_json = |path: &std::path::Path| {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        };
        let weights_loadable = |component: &str| {
            let dir = resolved_artifact_root.join(component);
            if parse_json(&dir.join("config.json")).is_none() {
                return false;
            }
            let index = std::fs::read_dir(&dir).ok().and_then(|entries| {
                entries.flatten().map(|entry| entry.path()).find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with(".safetensors.index.json"))
                })
            });
            if let Some(index) = index {
                return parse_json(&index)
                    .and_then(|value| value.get("weight_map")?.as_object().cloned())
                    .is_some_and(|weight_map| {
                        !weight_map.is_empty()
                            && weight_map.values().all(|shard| {
                                shard
                                    .as_str()
                                    .is_some_and(|shard| dir.join(shard).is_file())
                            })
                    });
            }
            sceneworks_core::mlx_tier_completeness::dir_has_visible_file_ending(
                &dir,
                ".safetensors",
            )
        };
        if !["transformer", "text_encoder", "vae"]
            .into_iter()
            .all(weights_loadable)
            || parse_json(&resolved_artifact_root.join("tokenizer/tokenizer.json")).is_none()
            || parse_json(&resolved_artifact_root.join("scheduler/scheduler_config.json")).is_none()
        {
            return None;
        }
        Some(())
    }

    #[cfg(test)]
    pub(crate) fn verified_for_test(tier_root: &str) -> Self {
        Self {
            resolved_route: "krea_2_turbo".into(),
            backend: "candle".into(),
            gpu_id: "test-gpu".into(),
            compute_capability: 12.0,
            artifact_provider: "huggingface".into(),
            artifact_repository: "SceneWorks/krea-2-turbo-mlx".into(),
            resolved_revision: "d009674080cc1bccf2b629d834c34bf5eccdb723".into(),
            tier_root: tier_root.into(),
            provider_contract: Some(krea_test_provider_contract(tier_root).clone()),
        }
    }
}

/// Sparse, weights-free Krea source used by the Candle-only memory-contract tests.
///
/// The provider's memory lookup now shares its complete production load-spec validation, including
/// the encoder and tokenizer contracts. An empty path therefore no longer represents a valid source.
/// Keep the tests on the real lookup seam with a structurally truthful fixture; no model tensor is
/// materialized and no memory evidence is measured or rewritten here.
///
/// ## Windows disk discipline (StorageFull incident, 2026-08-20)
///
/// "Sparse" above was only true on APFS/ext4. `gen_core_testkit::write_encoder_contract_fixture`
/// ends with a multi-gigabyte `set_len` — the encoder's dense logical size, which must stay because
/// the weight-bytes gates read it — and on NTFS that allocates in FULL. This fixture was measured
/// costing 21.84 GB of `C:` per test process, and because it lives in a `static` (never dropped)
/// every process left that behind permanently. Two mitigations, both load-bearing:
///
/// 1. Each tier's `.safetensors` tail is released right after the tier is written
///    ([`crate::test_fixture_disk::sparsify_written_safetensors`]). Logical size and every written
///    header byte are unchanged, so the gates read exactly the same numbers.
/// 2. Initialization sweeps prior runs' leftovers, which a `static`'s guard can never do for
///    itself. The directory name carries this process's PID so the sweep can tell its own from
///    another concurrent run's.
#[cfg(test)]
pub(super) fn krea_test_artifact_root(tier: &str) -> std::path::PathBuf {
    struct Fixture {
        _temp: tempfile::TempDir,
        root: std::path::PathBuf,
    }

    /// Distinctive enough that the sweep can match nothing else under the shared temp root.
    const FIXTURE_FAMILY: &str = "sceneworks-krea-memfix-";

    static FIXTURE: std::sync::OnceLock<Fixture> = std::sync::OnceLock::new();
    let fixture = FIXTURE.get_or_init(|| {
        crate::test_fixture_disk::sweep_stale_fixtures(FIXTURE_FAMILY);
        let temp = tempfile::Builder::new()
            .prefix(&crate::test_fixture_disk::process_keyed_prefix(
                FIXTURE_FAMILY,
            ))
            .tempdir()
            .expect("Krea memory-contract fixture root");
        let root = temp.path().to_path_buf();
        let contract = crate::inference_runtime::media_encoder_contract("krea_2_turbo")
            .expect("Krea encoder contract");
        for (tier, bits) in [("q4", Some(4)), ("q8", Some(8)), ("bf16", None)] {
            let tier_root = root.join(tier);
            // AT THE TIER'S QUANTIZATION, not dense (sc-22667). The `bits` this loop already uses
            // for the transformer's `config.json` describe the whole tier's snapshot: a q4 Krea
            // store holds a q4 text encoder too. Writing a dense encoder for every tier was
            // invisible while the pinned contract published `MemoryAssetFacts::default()`, but the
            // sc-22657 contract prices `conditioning_bytes` from this fixture's safetensors
            // HEADERS — dense made q4's encoder 15.69 GB, four times the packaged anchor's whole
            // 3.83 GB measured conditioning peak, so the derivation law correctly refused a
            // conditioning residency the measurement contradicts and every Krea derivation test
            // below went unpriceable. At the tier's own width it is 3.76 GB, i.e. the anchor's
            // measured peak less ~64 MB of activation — the composition that measurement saw.
            gen_core_testkit::write_encoder_contract_fixture_with_quant(
                &tier_root.join("text_encoder"),
                contract,
                bits,
            )
            .expect("write sparse Krea encoder fixture");
            // Per tier, not once at the end: the moment between the testkit's `set_len` and the
            // release is the only fully-allocated one, and this bounds it to ONE tier's file.
            crate::test_fixture_disk::sparsify_written_safetensors(&tier_root.join("text_encoder"));
            if let Some(bits) = bits {
                let transformer = tier_root.join("transformer");
                std::fs::create_dir_all(&transformer).expect("Krea transformer fixture dir");
                std::fs::write(
                    transformer.join("config.json"),
                    serde_json::to_vec(&serde_json::json!({
                        "quantization": { "bits": bits, "group_size": 64 }
                    }))
                    .expect("Krea transformer fixture config"),
                )
                .expect("write Krea transformer fixture config");
            }
        }
        Fixture { _temp: temp, root }
    });
    assert!(
        matches!(tier, "q4" | "q8" | "bf16"),
        "unsupported Krea fixture tier {tier}"
    );
    fixture.root.join(tier)
}

#[cfg(test)]
pub(super) fn krea_test_load_spec(tier: &str) -> gen_core::LoadSpec {
    gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(krea_test_artifact_root(tier)))
}

/// One complete provider validation per immutable test tier.
///
/// ArtifactSeal acquisition deliberately hashes the full logical encoder payload. The shared
/// Krea fixture is process-lifetime immutable, so reacquiring that same seal on every fit only
/// repeats multi-gigabyte I/O without testing another state. Production contexts acquire their
/// contract in [`KreaRuntimeEvidenceContext::inspect_turbo_fit`]; this cache is the test equivalent.
#[cfg(test)]
pub(super) fn krea_test_provider_contract(tier: &str) -> &'static gen_core::MemoryProviderContract {
    static CONTRACTS: std::sync::OnceLock<[gen_core::MemoryProviderContract; 3]> =
        std::sync::OnceLock::new();
    let contracts = CONTRACTS.get_or_init(|| {
        ["q4", "q8", "bf16"].map(|fixture_tier| {
            crate::inference_runtime::media()
                .memory_strategy_contract("krea_2_turbo", &krea_test_load_spec(fixture_tier))
                .expect("Krea contract lookup succeeds")
                .expect("Krea contract exists")
        })
    });
    match tier {
        "q4" => &contracts[0],
        "q8" => &contracts[1],
        "bf16" => &contracts[2],
        _ => panic!("unsupported Krea fixture tier {tier}"),
    }
}

/// Read a measured phase curve `fixedGb + perMpxGb * megapixels + perMpxFrameGb * megapixels *
/// frames`. The manifest stores fixed weight / allocator residency separately from the
/// geometry-dependent activation slopes. Invalid or incomplete evidence fails closed to `None`;
/// callers retain the established sequential gate instead of inventing a fit.
///
/// ## The temporal term (sc-18812, form chosen by sc-18810)
///
/// `perMpxFrameGb` is OPTIONAL and absent on every committed image curve. Absent is read as
/// `0.0`, and the sum is deliberately written so that the absent case reduces to the pre-sc-18812
/// expression **bit for bit** — the area term keeps its original association
/// (`per_mpx * pixels as f64 / 1_000_000.0`, not `per_mpx * (pixels as f64 / 1_000_000.0)`, which
/// is a DIFFERENT f64 in general), and `x + 0.0` is bitwise `x` for every finite `x` EXCEPT
/// `x == -0.0`, where IEEE-754 round-to-nearest gives `+0.0` — a different bit pattern for the
/// same numeric value. Reaching that here takes a curve declaring BOTH `fixedGb` and `perMpxGb`
/// as `-0.0` (either one alone leaves the pre-term sum at `+0.0`); such a curve validates, since
/// JSON Schema `minimum: 0` accepts `-0.0`, and clears the `< 0.0` guards below. No committed
/// curve declares one. It is documented rather than guarded: a signed-zero branch would exist
/// only to preserve the sign bit of a zero-valued phase prediction, which no consumer can
/// observe. The identity itself is pinned bitwise over the real committed curves, not asserted
/// against a default.
///
/// A PRESENT `perMpxFrameGb` that `json_f64` cannot read still fails closed, and so does a
/// negative or non-finite one. Only true absence is zero; a malformed value must not silently
/// degrade a video curve into an image curve. "Unreadable" is `json_f64`'s notion, not JSON's:
/// it accepts a NUMERIC STRING, so `"0.3"` evaluates exactly as `0.3` would — deliberately, and
/// identically to `fixedGb`/`perMpxGb`, which have always been read the same way. Rejecting a
/// string-typed coefficient is the SCHEMA's job (`model-manifest.schema.json`
/// `#/$defs/phaseVramCurve` types all three as `number`), and
/// `test_schema_admits_the_temporal_coefficient_additively` is where that rejection is pinned.
/// The string case is carried here as an explicitly-ACCEPTED control so the fail-closed set below
/// is not read as broader than it is.
///
/// SC-16514 recovered the q8/bf16 768² captures from SC-15205 activity 15272 and SC-15206 activity
/// 15314 into `turboFit.evidenceRecords`. Every tier now carries 768² and 1024² phase cells, and every
/// `perMpxGb` is fitted from that tier's own phase delta. The recovered cells are explicitly
/// `phase_fit_only`: the cited activities do not establish geometry-specific 768² output parity, so
/// they characterize the bounded curve without authorizing exact runtime admission. Q8's 7.98
/// denoise slope equals q4's because both measured rises are 3.658 GiB, not because the coefficient
/// was borrowed. Zero geometry-sensitive slopes remain only where the two samples are flat or
/// decrease; the manifest names each such pair. `maxMeasuredPixels` remains 1024² because larger
/// attention shapes have not been validated, so the curve is fitted within that bound rather than
/// extrapolated beyond it.
pub(super) fn krea_phase_curve(phase: &JsonObject, geometry: CurveGeometry) -> Option<f64> {
    let fixed = phase.get("fixedGb").and_then(json_f64)?;
    let per_mpx = phase.get("perMpxGb").and_then(json_f64)?;
    let per_mpx_frame = match phase.get("perMpxFrameGb") {
        None => 0.0,
        Some(value) => json_f64(value)?,
    };
    if !fixed.is_finite()
        || !per_mpx.is_finite()
        || !per_mpx_frame.is_finite()
        || fixed < 0.0
        || per_mpx < 0.0
        || per_mpx_frame < 0.0
    {
        return None;
    }
    // A zero-frame request is not a still image, it is a nonsense geometry. Fail closed rather
    // than silently pricing it as the intercept.
    if geometry.frames == 0 {
        return None;
    }
    let area_term = per_mpx * geometry.pixels as f64 / 1_000_000.0;
    let temporal_term =
        per_mpx_frame * geometry.pixels as f64 / 1_000_000.0 * f64::from(geometry.frames);
    Some(fixed + area_term + temporal_term)
}

/// The typed materialization shape the shipped Krea Turbo curves were measured under (sc-17097).
///
/// `None` for a missing or unrecognized value, which fails the fit closed rather than defaulting: a
/// silently assumed shape is what let calibration ABI 2's load-shape axis do no work on this route.
pub(crate) fn krea_turbo_load_shape(turbo_fit: &Value) -> Option<gen_core::LoadShape> {
    match turbo_fit.get("loadShape")?.as_str()? {
        "eager_materialization" => Some(gen_core::LoadShape::EagerMaterialization),
        "deferred_materialization" => Some(gen_core::LoadShape::DeferredMaterialization),
        _ => None,
    }
}

/// The largest output VOXEL count (pixels x frames) a fit block's curves were measured across
/// (sc-18812). Absent is read as `pixels`, i.e. one output frame — which is what every image-lane
/// fit is, so omitting the key refuses exactly the multi-frame requests that lane never measured.
///
/// ## Why voxels and not a frame count
///
/// 1. Voxels are the regressor `perMpxFrameGb` multiplies, so this bounds the extrapolation of
///    the term it governs rather than a loosely correlated proxy.
/// 2. **The tiling discontinuity is itself a constant-voxel surface.** At the pinned revision
///    `VaeTiling::writable_frame_cap(out_h, out_w)` is `MAX_WRITABLE_ELEMS / (full_res_channels *
///    out_h * out_w)` with `MAX_WRITABLE_ELEMS = i32::MAX`, so a single pass is legal exactly
///    while `out_voxels <= i32::MAX / full_res_channels` — 268,435,455 for LTX's 8 full-res
///    channels. That one surface is the 297-output-frame cap quoted at 0.90 MP and 682 at
///    0.39 MP. A scalar frame bound would admit a small-area request and refuse an
///    identically-priced large-area one, which is the wrong shape of guard.
///
/// The bound is not politeness about unvalidated territory — past it the affine form is KNOWN
/// wrong. Single-pass decode climbs to ~94.3 GB at the cap and tiled decode drops it to ~63.8 GB
/// on this 128 GiB host; no affine curve represents that step, so the fit is refused across it
/// rather than extrapolated through it. Note the cap is only ONE-SIDED machine-independent: no
/// host exceeds it single-pass, but a smaller host tiles EARLIER via the memory bound, and tiled
/// cost RISES with host memory because the selector keeps the largest tile that fits.
pub(super) fn max_measured_voxels(fit: &Value, pixels: u64) -> Option<u64> {
    match fit.get("maxMeasuredVoxels") {
        None => Some(pixels),
        Some(value) => value.as_u64().filter(|max| *max >= 1),
    }
}

pub(super) fn krea_rung_phase_peaks(
    manifest_entry: &JsonObject,
    tier: &str,
    strategy: gen_core::MemoryStrategy,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<KreaTurboPhasePeaks> {
    let turbo_fit = manifest_entry.get("candle")?.get("turboFit")?;
    let pixels = u64::from(width).checked_mul(u64::from(height))?;
    let max_measured_pixels = turbo_fit.get("maxMeasuredPixels")?.as_u64()?;
    if pixels > max_measured_pixels {
        return None;
    }
    // `frames == 0` needs no check here: `pixels * 0` clears any bound, and `krea_phase_curve`
    // refuses the geometry outright a few lines later. A second check would be a guard no test
    // could kill.
    let voxels = pixels.checked_mul(u64::from(frames))?;
    if voxels > max_measured_voxels(turbo_fit, pixels)? {
        return None;
    }
    let rung = turbo_fit
        .get("phaseCurvesByTier")?
        .get(tier)?
        .get(krea_turbo_manifest_key(strategy))?
        .as_object()?;
    let geometry = CurveGeometry { pixels, frames };
    let phase = |name: &str| {
        rung.get(name)
            .and_then(Value::as_object)
            .and_then(|curve| krea_phase_curve(curve, geometry))
    };
    Some(KreaTurboPhasePeaks {
        text_gb: phase("text")?,
        denoise_gb: phase("denoise")?,
        decode_gb: phase("decode")?,
    })
}

/// The per-phase predicted peaks a measured evidence record declares for one rung — the measured
/// basis a fitted-curve estimate extrapolates from (sc-18097). `None` when the record does not
/// carry a complete phase triple for the rung, which fails the estimate closed.
pub(super) fn krea_record_phase_peaks(
    record: &Value,
    manifest_rung: &str,
) -> Option<KreaTurboPhasePeaks> {
    let phases = record.get("predictedPhasesGb")?.get(manifest_rung)?;
    Some(KreaTurboPhasePeaks {
        text_gb: phases.get("text").and_then(json_f64)?,
        denoise_gb: phases.get("denoise").and_then(json_f64)?,
        decode_gb: phases.get("decode").and_then(json_f64)?,
    })
}

/// The phase index (0 text, 1 denoise, 2 decode) carrying the peak of a phase triple (sc-18097).
/// Ties resolve to the LATER phase deterministically, mirroring the MLX gate's `binding_phase`
/// (sc-18096); the comparison below only ever contrasts two triples produced by the same per-phase
/// curves, so tie handling cannot manufacture a flip on its own.
///
/// sc-18812 moved the rule onto [`KreaTurboPhasePeaks::binding_phase`] so that the same question
/// can be asked of a CURVE-predicted triple at an arbitrary geometry, not only of a recorded one.
/// This wrapper keeps the index vocabulary the sc-18097 comparison seam below is written in.
pub(super) fn krea_binding_phase(peaks: KreaTurboPhasePeaks) -> u8 {
    peaks.binding_phase().index()
}

pub(super) fn krea_rung_parameters(
    turbo_fit: &Value,
    strategy: gen_core::MemoryStrategy,
) -> Option<gen_core::MemoryStrategyParameters> {
    let parameters = turbo_fit
        .get("strategyParameters")?
        .get(krea_turbo_manifest_key(strategy))?
        .as_object()?;
    let value = |name| {
        parameters
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    };
    Some(gen_core::MemoryStrategyParameters {
        stage_residency: None,
        decode_tile_edge: value("decodeTileEdge"),
        decode_overlap: value("decodeOverlap"),
        attention_chunk_size: value("attentionChunkSize"),
        transformer_window_size: value("transformerWindowSize"),
        // Rung 4's window scope (SC-15794). `Dit` is the only scope that is correct here:
        // `candle-gen-krea` leaves `transformer_window_components` empty, which the contract reads
        // as the DiT-only pre-SC-15794 behaviour, and every measured `strategyParameters` row in
        // the manifest was collected against DiT block streaming. It is also the published default
        // SC-15794 upheld -- a text-encoder scope cut z-image conditioning 46.5% but moved the
        // request peak 0.0%, so widening the scope buys an admission gate nothing.
        //
        // The scope is declared only on the rung that owns it. `validate_selected_parameters`
        // rejects any `Some(..)` below `BoundedTransformerResidency` as "irrelevant: the selection
        // does not engage its owning strategy rung", and the three cheaper rungs here pair with
        // cheaper strategies
        // (see `rung_pairs`); a blanket `Some` would fail `validate_selection` on three of the
        // four candidates and silently downgrade their evidence verdicts. `None` on those rungs
        // carries the identical DiT meaning without tripping that check.
        transformer_window_component: (strategy
            == gen_core::MemoryStrategy::BoundedTransformerResidency)
            .then_some(gen_core::TransformerComponent::Dit),
    })
}

/// Krea Turbo is a text-to-IMAGE route: every request and every measured record on it produces one
/// output frame. Named rather than repeated as a literal `1` so the curve evaluation and the
/// `MemoryGeometry` it is graded against cannot drift apart — sc-18810 found four of six MLX
/// capture arms hardcoding `frames: 1` into their admission context while reading only width and
/// height, which is exactly how a geometry gets attested that was never asked for.
pub(super) const KREA_LANE_FRAMES: u32 = 1;

/// The packaged measured memory anchor for one Krea Turbo candle tier (sc-22509, epic 22505), with
/// every identity conjunct that can be answered without the loaded provider contract.
///
/// Fail-closed and deliberately narrow: the anchor was measured on ONE catalog model, route,
/// provider and mode, overlay-free and reference-free. The remaining conjunct (materialization
/// shape) is applied by the caller once the contract is available.
///
/// Currency is NOT a conjunct (sc-22738). The anchor's loader-closure digest (sc-22511) is what
/// the probe tooling re-captures on; the calibration fingerprint and ABI are PROVENANCE bound to
/// the source record by `validate_anchor`. None of them can demote the anchor here: it prices the
/// request whether or not the loader has moved since it was measured.
/// The anchor store this lane reads. Production has exactly one — the packaged, validated store.
///
/// The `cfg(test)` override lets a test grade the derivation ITSELF on a store whose contents it
/// states — the same shape `candle_memory_strategy::synthesize_estimate_floors` already takes as
/// a parameter, and the same test-seam precedent as
/// `video_admission::admit_video_generation_with_curves`.
pub(super) fn krea_anchor_store(
) -> Option<&'static sceneworks_core::memory_anchor::MemoryAnchorStore> {
    #[cfg(test)]
    if let Some(store) = tests::injected_anchor_store() {
        return Some(store);
    }
    sceneworks_core::memory_anchor::packaged_memory_anchors()
}

pub(super) fn krea_store_anchor(
    tier: &str,
    allow_streamed_blocks: bool,
) -> Option<&'static sceneworks_core::memory_anchor::MemoryAnchor> {
    // A job carrying load-time adapters is exactly the case this lane keeps on its resident/staged
    // paths, and the anchor render carried no overlay at all.
    if !allow_streamed_blocks {
        return None;
    }
    krea_anchor_store()
        .and_then(|store| {
            store.image_anchor_for(
                "krea_2_turbo",
                sceneworks_core::memory_anchor::AnchorBackend::Candle,
                tier,
            )
        })
        .filter(|anchor| {
            anchor.route == "krea_2_turbo"
                && anchor.provider == "krea_2_turbo"
                && anchor.mode == "text_to_image"
                && anchor.overlay.is_none()
                && anchor.reference_count == 0
        })
}

/// Select the least-cost measured Krea Turbo fit rung for this tier, geometry, and live budget.
/// `allow_streamed_blocks` is false when the job carries load-time adapters: the provider preserves
/// those jobs on its existing resident/staged paths rather than silently omitting their residuals.
///
/// Returns `None` when live budget or complete measured manifest evidence is absent. This is distinct
/// from `Reject`: unknown evidence must not masquerade as proof that a configuration cannot fit.
///
/// Since sc-18097 (epic 18093 R1b) an in-envelope request geometry with no exact measured record
/// no longer freezes to `Unverified`: each optimized rung carries a fitted-curve ESTIMATE
/// candidate anchored to a verified measured record, graded by the shared selector behind the
/// candle estimate margin — see the synthesis block below for the anchoring and binding-phase
/// rules.
///
/// Since sc-22667 (epic 22657 feature-end round, E3/E4/E7) the lane is on the same law and the
/// same reserve as the shared candle ladder:
///
/// * an OUT-OF-ENVELOPE rung — one the fitted `turboFit` curves cannot price — is derived through
///   `MemoryAnchor::derive_phase_peaks` with the rung's OWN regime
///   (`candle_memory_strategy::request_regime` over the manifest's `strategyParameters`: the
///   decode tile, the attention score budget, the transformer window) and the contract's
///   architecture facts (`video_admission::architecture_facts_from_contract`), so rung 3 and
///   rung 4 price their bounded working sets rather than every rung repeating the shallow staged
///   derivation. In-envelope rungs keep the fitted curves: they are measured evidence.
/// * `reserve_gb` is the operational reserve — `ladder_reserve_gb` of the caller's RAW probe,
///   handed in explicitly so a reclaimable-credited `budget` can never derive it — charged
///   EXACTLY ONCE, on the budget side, against every candidate
///   (`memory_strategy::ReserveCharge::EveryCandidate`): every candidate this lane grades is a
///   device delta above the pre-load residency (the measured records, the curves fitted over
///   them, and the anchor derivation alike), so none carries the reserve inside its peak. The
///   quoted `needed_gb` is the admitted peak plus that reserve.
/// * the selection reports its basis and admission figures, so the caller can emit the
///   `image_memory_strategy_selected` event with the three phase peaks
///   ([`KreaTurboFit::selection_telemetry`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn krea_turbo_fit_with_runtime(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    allow_streamed_blocks: bool,
    runtime: Option<&KreaRuntimeEvidenceContext>,
) -> Option<KreaTurboFit> {
    krea_turbo_fit_priced(
        manifest_entry,
        tier,
        width,
        height,
        budget,
        reserve_gb,
        allow_streamed_blocks,
        runtime,
        None,
    )
}

/// [`krea_turbo_fit_with_runtime`] with the architecture facts the derivation scales by as an
/// explicit input: `None` reads them off the loaded contract (production — the contract's own
/// `architecture_facts` block through `architecture_facts_from_contract`, sc-22667), `Some` lets
/// a fixture grade the rung ratios with facts it states directly.
#[allow(clippy::too_many_arguments)]
pub(super) fn krea_turbo_fit_priced(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    allow_streamed_blocks: bool,
    runtime: Option<&KreaRuntimeEvidenceContext>,
    facts_override: Option<sceneworks_core::memory_anchor::ArchitectureFacts>,
) -> Option<KreaTurboFit> {
    use crate::memory_strategy::{self, Budget, Candidate, RequestScope, Selection};
    use gen_core::{
        MemoryConformanceState, MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey,
        MemoryEvidenceVerdict, MemoryGeometry, MemoryParityContract, MemoryParityResult,
        MemorySelection, MemoryStrategy, MemoryStrategyParameters,
    };

    let budget = budget?;
    let turbo_fit = manifest_entry.get("candle")?.get("turboFit")?;
    // sc-22735: the identity the LOADED contract publishes is keyed on the artifact-proven tier, so
    // the fingerprint this fit is graded against has to be read per tier. The scalar remains the
    // measured q4 string and is the fallback for a manifest that declares no per-tier map.
    let calibration_fingerprint = turbo_fit
        .get("calibrationFingerprintByTier")
        .and_then(|by_tier| by_tier.get(tier))
        .or_else(|| turbo_fit.get("calibrationFingerprint"))?
        .as_str()?;
    let calibration_abi = turbo_fit.get("calibrationAbi")?.as_u64()? as u32;
    // sc-17097: calibration ABI 2 added the typed load shape, but this route never read it - the
    // worker took the shape from the provider alone, so the axis could not detect drift here. The
    // manifest now states the shape its curves were MEASURED under and the handshake below compares
    // the two; eager and deferred measurements are not interchangeable.
    let declared_load_shape = krea_turbo_load_shape(turbo_fit)?;
    let scene_works_revision = turbo_fit.get("sceneWorksRevision")?.as_str()?;
    let inference_revision = turbo_fit.get("inferenceRevision")?.as_str()?;
    let max_pixels = turbo_fit.get("maxMeasuredPixels")?.as_u64()?;
    let geometry = MemoryGeometry {
        width,
        height,
        batch: 1,
        frames: KREA_LANE_FRAMES,
        reference_count: 0,
    };
    let pixels = u64::from(width).checked_mul(u64::from(height))?;
    // sc-22509 (epic 22505): a geometry the campaign never measured is no longer a refusal on its
    // own. The measured memory ANCHOR for this `(model, tier, candle lane)` prices it analytically,
    // so the hull only refuses when there is no anchor to derive from. The identity guards this
    // lookup cannot answer yet (contract load shape, live calibration identity) are applied once
    // `provider_contract` is resolved, and the hull refusal is re-asserted there if they fail.
    let store_anchor = krea_store_anchor(tier, allow_streamed_blocks);
    let out_of_envelope = pixels > max_pixels;
    if out_of_envelope && store_anchor.is_none() {
        return Some(KreaTurboFit::Unverified {
            reason: MemoryEvidenceVerdict::OutOfEnvelope,
        });
    }
    // Runtime artifact identity is a required evidence dimension. Preserve the explicit unverified
    // result when it is unavailable instead of attempting provider source validation against an
    // empty path and collapsing to `None`; callers keep the same resident-or-reject fallback, while
    // diagnostics retain the truthful reason.
    if runtime.is_none() {
        return Some(KreaTurboFit::Unverified {
            reason: MemoryEvidenceVerdict::Unverified,
        });
    }
    let provider_contract = runtime.and_then(|runtime| runtime.provider_contract.as_ref())?;
    let numeric_tier = gen_core::MemoryNumericTier {
        precision: gen_core::Precision::Bf16,
        quant: match tier {
            "q4" => Some(gen_core::Quant::Q4),
            "q8" => Some(gen_core::Quant::Q8),
            "bf16" => None,
            _ => return None,
        },
        component_precision_floors: &[],
    };
    // The remaining anchor identity conjunct, now that the loaded contract is available: the anchor
    // was measured under one materialization shape, and eager and deferred are not interchangeable.
    // Currency was already graded in `krea_store_anchor` on the loader closure and is deliberately
    // not re-asked here against the calibration campaign (sc-22511: provenance, not currency).
    let anchor = store_anchor.filter(|anchor| {
        let anchor_load_shape = match anchor.load_shape {
            sceneworks_core::memory_anchor::AnchorLoadShape::EagerMaterialization => {
                gen_core::LoadShape::EagerMaterialization
            }
            sceneworks_core::memory_anchor::AnchorLoadShape::DeferredMaterialization => {
                gen_core::LoadShape::DeferredMaterialization
            }
        };
        anchor_load_shape == provider_contract.load_shape
    });
    if out_of_envelope && anchor.is_none() {
        return Some(KreaTurboFit::Unverified {
            reason: MemoryEvidenceVerdict::OutOfEnvelope,
        });
    }
    if !reserve_gb.is_finite() || reserve_gb < 0.0 {
        return None;
    }
    // The law's inputs for the out-of-envelope rungs (sc-22667): the component bytes of the
    // loaded tier and the model's architecture facts, both off the contract.
    let components = crate::video_admission::anchor_component_bytes(provider_contract.asset_facts);
    let facts = facts_override.unwrap_or_else(|| {
        crate::video_admission::architecture_facts_from_contract(provider_contract)
    });
    let request = RequestScope {
        resolved_route: "krea_2_turbo",
        backend: "candle",
        tier: numeric_tier,
        mode: "text_to_image",
        overlay: (!allow_streamed_blocks).then_some("adapter"),
        geometry,
    };
    let resident_peak_gb = manifest_entry
        .get("candle")?
        .get("vramGbByTier")?
        .get(tier)
        .and_then(json_f64)?;
    if !resident_peak_gb.is_finite() || resident_peak_gb < 0.0 {
        return None;
    }
    let resident_parameters = turbo_fit
        .get("strategyParameters")?
        .get("resident")?
        .as_object()?;
    if !resident_parameters.is_empty() {
        return None;
    }
    let bytes = |gb: f64| (gb * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64;
    let tier_records = turbo_fit
        .get("evidenceRecords")?
        .as_array()?
        .iter()
        .filter(|record| {
            record.get("evidenceScope").and_then(Value::as_str) == Some("exact_request")
                && record.get("tier").and_then(Value::as_str) == Some(tier)
        })
        .filter_map(|record| {
            let record_width = u32::try_from(record.get("width")?.as_u64()?).ok()?;
            let record_height = u32::try_from(record.get("height")?.as_u64()?).ok()?;
            Some((record, record_width, record_height))
        })
        .collect::<Vec<_>>();
    let evidence_record = tier_records
        .iter()
        .find(|(_, record_width, record_height)| *record_width == width && *record_height == height)
        .map(|(record, _, _)| *record);
    // `record`/`at_geometry` are parameters rather than captures (sc-18097): the request cell's
    // candidates are graded against the request's own record and geometry, while the estimate
    // synthesis below re-grades a DIFFERENT record at its own measured geometry to decide whether
    // it is a verified extrapolation anchor.
    let make_evidence = |selection: MemorySelection,
                         manifest_rung: Option<&str>,
                         fallback_predicted_peak_gb: f64,
                         fallback_phases: Option<KreaTurboPhasePeaks>,
                         evidence_record: Option<&Value>,
                         at_geometry: MemoryGeometry| {
        let record_peak = |field: &str| {
            let rung = manifest_rung?;
            evidence_record?.get(field)?.get(rung).and_then(json_f64)
        };
        let record_phase = |phase: &str| {
            let rung = manifest_rung?;
            evidence_record?
                .get("predictedPhasesGb")?
                .get(rung)?
                .get(phase)
                .and_then(json_f64)
        };
        let engaged_composition = manifest_rung.map_or_else(
            || provider_contract.engaged_composition(selection.strategy),
            |rung| match evidence_record {
                Some(record) => record
                    .get("measuredCompositions")
                    .and_then(|compositions| compositions.get(rung))
                    .and_then(Value::as_array)
                    .and_then(|composition| {
                        composition
                            .iter()
                            .map(|rung| match rung.as_str()? {
                                "resident" => Some(MemoryStrategy::Resident),
                                "staged_residency" => Some(MemoryStrategy::StagedResidency),
                                "bounded_decode" => Some(MemoryStrategy::BoundedDecode),
                                "bounded_attention" => Some(MemoryStrategy::BoundedAttention),
                                "bounded_transformer_residency" => {
                                    Some(MemoryStrategy::BoundedTransformerResidency)
                                }
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .unwrap_or_default(),
                None => provider_contract.engaged_composition(selection.strategy),
            },
        );
        let predicted_peak_gb =
            record_peak("predictedPeaksGb").unwrap_or(fallback_predicted_peak_gb);
        let observed_peak_gb = record_peak("observedPeaksGb");
        let harness_version = evidence_record
            .and_then(|record| record.get("harnessVersion"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let parity = evidence_record
            .and_then(|record| record.get("parity"))
            .and_then(Value::as_object);
        let verdict = |condition, failure| {
            if condition {
                MemoryEvidenceVerdict::Satisfied
            } else {
                failure
            }
        };
        let static_valid = provider_contract.conformance_errors().is_empty()
            && provider_contract.validate_selection(&selection).is_ok();
        let declared_calibration = provider_contract
            .calibration
            .as_ref()
            .is_some_and(|identity| {
                identity.abi == calibration_abi
                    && identity.fingerprint == calibration_fingerprint
                    && identity.load_shape == declared_load_shape
            });
        let valid_commit = |field: &str| {
            evidence_record
                .and_then(|record| record.get(field))
                .and_then(Value::as_str)
                .is_some_and(|revision| {
                    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
                })
        };
        // sc-17774: `compatibleInferenceRevision` is gone. It was a per-record hand-declared "this
        // capture is also valid at revision X" claim — the same one-shot hatch `flux2_dev` had, in a
        // second spelling. Compatibility is now decided by the lane's closure digest below.
        let historical_compatible = valid_commit("sceneWorksCommit")
            && valid_commit("inferenceCommit")
            && evidence_record
                .and_then(|record| record.get("compatibleSceneWorksRevision"))
                .and_then(Value::as_str)
                == Some(KREA_TURBO_SCENEWORKS_REVISION);
        let loadability = evidence_record
            .and_then(|record| record.get("loadability"))
            .and_then(Value::as_object);
        let expected_compute_capability = loadability
            .and_then(|loadability| loadability.get("computeCapability"))
            .and_then(json_f64);
        // `inference_revision` stays parsed above as capture provenance for the receipt, and since
        // sc-22738 so does `turboFit.inferenceClosureDigest`: the lane's compile closure is a
        // re-capture signal for the probe tooling, never a conjunct that can un-verify this
        // dimension at runtime.
        let current_environment = scene_works_revision == KREA_TURBO_SCENEWORKS_REVISION
            && turbo_fit.get("measured").and_then(Value::as_bool) == Some(true)
            && runtime.is_some_and(|runtime| {
                !runtime.gpu_id.trim().is_empty()
                    && expected_compute_capability.is_some_and(|expected| {
                        (f64::from(runtime.compute_capability) - expected).abs() < f64::EPSILON
                    })
            });
        let loadability_matches_record =
            loadability
                .zip(runtime)
                .is_some_and(|(loadability, runtime)| {
                    loadability.get("provider").and_then(Value::as_str)
                        == Some(runtime.artifact_provider.as_str())
                        && loadability.get("repository").and_then(Value::as_str)
                            == Some(runtime.artifact_repository.as_str())
                        && loadability.get("resolvedRevision").and_then(Value::as_str)
                            == Some(runtime.resolved_revision.as_str())
                        && loadability.get("tierRoot").and_then(Value::as_str)
                            == Some(runtime.tier_root.as_str())
                        && loadability.get("route").and_then(Value::as_str)
                            == Some(runtime.resolved_route.as_str())
                        && loadability.get("backend").and_then(Value::as_str)
                            == Some(runtime.backend.as_str())
                        && runtime.resolved_route == provider_contract.provider_id
                        && runtime.backend == provider_contract.backend.backend_id()
                        && runtime.tier_root == tier
                        && manifest_entry
                            .get("downloads")
                            .and_then(Value::as_array)
                            .is_some_and(|downloads| {
                                downloads.iter().any(|download| {
                                    download.get("provider").and_then(Value::as_str)
                                        == loadability.get("provider").and_then(Value::as_str)
                                        && download.get("repo").and_then(Value::as_str)
                                            == loadability.get("repository").and_then(Value::as_str)
                                        && download.get("revision").and_then(Value::as_str)
                                            == loadability
                                                .get("resolvedRevision")
                                                .and_then(Value::as_str)
                                        && download.get("variant").and_then(Value::as_str)
                                            == Some(tier)
                                })
                            })
                });
        let phases_match = fallback_phases.is_some_and(|phases| {
            [
                ("text", phases.text_gb),
                ("denoise", phases.denoise_gb),
                ("decode", phases.decode_gb),
            ]
            .into_iter()
            .all(|(name, predicted)| {
                record_phase(name).is_some_and(|recorded| (recorded - predicted).abs() <= 0.01)
            })
        });
        let exact_parameters = manifest_rung.is_some()
            && record_peak("predictedPeaksGb").is_some_and(|record_peak_gb| {
                (record_peak_gb - fallback_predicted_peak_gb).abs() <= 0.01
            })
            && phases_match
            && provider_contract.validate_selection(&selection).is_ok();
        let evidence_dimensions = MemoryEvidenceDimensions {
            static_implementation: verdict(static_valid, MemoryEvidenceVerdict::Invalid),
            declared_calibration: verdict(
                declared_calibration,
                MemoryEvidenceVerdict::FingerprintMismatch,
            ),
            historical_verification: verdict(
                historical_compatible,
                if evidence_record.is_some() {
                    MemoryEvidenceVerdict::Stale
                } else {
                    MemoryEvidenceVerdict::Missing
                },
            ),
            current_environment_verification: verdict(
                current_environment,
                if evidence_record.is_some() {
                    MemoryEvidenceVerdict::Stale
                } else {
                    MemoryEvidenceVerdict::Missing
                },
            ),
            canonical_route_loadability: verdict(
                loadability_matches_record,
                if evidence_record.is_some() {
                    MemoryEvidenceVerdict::Unverified
                } else {
                    MemoryEvidenceVerdict::Missing
                },
            ),
            exact_strategy_parameters: verdict(
                exact_parameters,
                MemoryEvidenceVerdict::OutOfEnvelope,
            ),
        };
        let verified = observed_peak_gb.is_some()
            && !harness_version.is_empty()
            && parity.is_some_and(|parity| {
                parity.get("contract").and_then(Value::as_str) == Some("golden")
                    && parity.get("result").and_then(Value::as_str) == Some("passed")
            })
            && evidence_dimensions.all_satisfied();
        let parity_contract = parity.map_or(MemoryParityContract::Exact, |parity| {
            MemoryParityContract::Golden {
                fixture: parity
                    .get("fixture")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                metric: parity
                    .get("metric")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                maximum_error: parity
                    .get("maximumError")
                    .and_then(json_f64)
                    .unwrap_or_default(),
            }
        });
        MemoryEvidence {
            key: MemoryEvidenceKey {
                model_family: "krea_2_turbo".to_owned(),
                resolved_route: "krea_2_turbo".to_owned(),
                backend: gen_core::MemoryBackend::Candle,
                tier: numeric_tier,
                load_shape: provider_contract.load_shape,
                mode: gen_core::MemoryMode::TextToImage,
                reference_shape: gen_core::MemoryReferenceShape::None,
                // The existing measurements cover ordinary T2I only.
                overlay: None,
                geometry: at_geometry,
                frames_per_second: None,
                strategy: selection.strategy,
                engaged_composition,
                parameters: selection.parameters,
            },
            conformance: if verified {
                MemoryConformanceState::Verified
            } else {
                MemoryConformanceState::ImplementedUnverified
            },
            dimensions: evidence_dimensions,
            calibration_abi,
            calibration_fingerprint: calibration_fingerprint.to_owned(),
            sceneworks_revision: scene_works_revision.to_owned(),
            inference_revision: inference_revision.to_owned(),
            harness_version: harness_version.to_owned(),
            predicted_peak_bytes: bytes(predicted_peak_gb),
            observed_peak_bytes: observed_peak_gb.map(bytes),
            parity: parity_contract,
            parity_result: if verified {
                MemoryParityResult::Passed
            } else {
                MemoryParityResult::NotRun
            },
        }
    };
    let resident_selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: MemoryStrategyParameters::default(),
        tier: numeric_tier,
    };

    let optimized_strategies = [
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ];
    // sc-22509 review (blocker): the resident rung is priced by `candle.vramGbByTier[tier]`, a
    // SCALAR measured at `vramMeasuredPixels` that does not scale with geometry at all. Inside the
    // measured hull that is the row the lane has always quoted; OUTSIDE it the row is simply not a
    // price for the request, and the anchor cannot supply one either — the derivation refuses
    // `staged_residency: false`, because the shallow staged anchor does not price a whole-model
    // resident working set. So an out-of-envelope request emits NO resident candidate: a rung with
    // no priced basis is honestly Missing, not `Measured` at a 1024x1024 number. Without this the
    // hull-refusal bypass above would admit e.g. a 2048x2048 q4 request as `Resident` at 25.7 GiB
    // on a 32 GB card, which previously returned `OutOfEnvelope`.
    let mut evidence = Vec::new();
    let mut selections = Vec::new();
    if !out_of_envelope {
        evidence.push(make_evidence(
            resident_selection,
            None,
            resident_peak_gb,
            None,
            evidence_record,
            geometry,
        ));
        selections.push(resident_selection);
    }
    let mut measured = Vec::new();
    // sc-22509: rungs the manifest curves cannot price at this geometry, derived from the anchor.
    // They are candidates in their own right — never `Measured` — so they are collected apart from
    // the curve-priced evidence above.
    let mut anchor_derived: Vec<(MemorySelection, MemoryEvidence)> = Vec::new();
    for strategy in optimized_strategies {
        let parameters = krea_rung_parameters(turbo_fit, strategy)?;
        let selection = MemorySelection {
            strategy,
            parameters,
            tier: numeric_tier,
        };
        let curve_phases = krea_rung_phase_peaks(
            manifest_entry,
            tier,
            strategy,
            width,
            height,
            geometry.frames,
        );
        let Some(phases) = curve_phases else {
            // No measured curve reaches this cell. Derive it from the anchor for the exact rung
            // composition the provider would execute. A rung the anchor cannot price — no anchor at
            // all, or a composition running whole-model resident, which the shallow staged anchor
            // does not cover — is SKIPPED, never propagated out of the function (sc-22509 review):
            // `?` here would collapse every other rung's candidate and destroy the selector's own
            // Missing/Reject verdict, turning a truthful `Unverified { OutOfEnvelope }` into a bare
            // `None`.
            let Some(anchor) = anchor else {
                continue;
            };
            if anchor.underived_reason.is_some()
                || anchor.backend != sceneworks_core::memory_anchor::AnchorBackend::Candle
            {
                continue;
            }
            let engaged = provider_contract.engaged_composition(strategy);
            // sc-22667: the rung's OWN regime — its engaged composition and the parameters the
            // manifest selects for it — through the one law, with the contract's facts. Rungs 3
            // and 4 therefore price their chunk and window where the shallow shim priced every
            // rung at the staged working set. A rung whose parameters are incomplete cannot be
            // executed as selected and is skipped, like a rung the law refuses.
            let Some(regime) = crate::candle_memory_strategy::request_regime(&engaged, &parameters)
            else {
                continue;
            };
            let Some(derived) = anchor.derive_phase_peaks(
                &sceneworks_core::memory_anchor::ImageDeriveRequest {
                    width,
                    height,
                    batch: 1,
                    conditioning_tokens: None,
                    regime,
                },
                components,
                facts,
            ) else {
                continue;
            };
            let phases = KreaTurboPhasePeaks {
                text_gb: derived.conditioning as f64 / BYTES_PER_GIB,
                denoise_gb: derived.denoise as f64 / BYTES_PER_GIB,
                decode_gb: derived.decode as f64 / BYTES_PER_GIB,
            };
            let predicted_peak_bytes = bytes(phases.peak_gb());
            tracing::info!(
                route = "krea_2_turbo",
                backend = "candle",
                ?strategy,
                anchor = anchor.id.as_str(),
                anchor_geometry = format!("{}x{}", anchor.geometry.width, anchor.geometry.height),
                conditioning_peak_bytes = derived.conditioning,
                denoise_peak_bytes = derived.denoise,
                decode_peak_bytes = derived.decode,
                raw_peak_bytes = predicted_peak_bytes,
                "synthesized anchor-derived estimate candidate for a geometry outside the \
                 measured curve envelope"
            );
            measured.push((
                strategy,
                phases,
                phases.peak_gb() + reserve_gb,
                selection,
                true,
            ));
            anchor_derived.push((
                selection,
                sceneworks_core::memory_contracts::unverified_estimate(
                    MemoryEvidenceKey {
                        model_family: "krea_2_turbo".to_owned(),
                        resolved_route: "krea_2_turbo".to_owned(),
                        backend: gen_core::MemoryBackend::Candle,
                        tier: numeric_tier,
                        load_shape: provider_contract.load_shape,
                        mode: gen_core::MemoryMode::TextToImage,
                        reference_shape: gen_core::MemoryReferenceShape::None,
                        overlay: None,
                        geometry,
                        frames_per_second: None,
                        strategy,
                        engaged_composition: engaged,
                        parameters,
                    },
                    calibration_abi,
                    calibration_fingerprint.to_owned(),
                    scene_works_revision.to_owned(),
                    predicted_peak_bytes,
                ),
            ));
            continue;
        };
        let phase_peak_gb = phases.peak_gb();
        let needed_gb = phase_peak_gb + reserve_gb;
        measured.push((strategy, phases, needed_gb, selection, false));
        evidence.push(make_evidence(
            selection,
            Some(krea_turbo_manifest_key(strategy)),
            phase_peak_gb,
            Some(phases),
            evidence_record,
            geometry,
        ));
        selections.push(selection);
    }
    // ── sc-18097 (epic 18093 R1b): fitted-curve estimate candidates for unmeasured cells. ──
    //
    // The candle mirror of `mlx_fit_gate::synthesize_estimate_ladder`'s fitted arm. The manifest's
    // per-phase curves ARE the fitted model over this tier's measured cells, so an in-envelope
    // request geometry nobody measured gets an estimate candidate per optimized rung at the
    // curve-predicted peak, graded by the shared selector behind the candle ESTIMATE margin
    // (`crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD`). Where the exact request cell has a
    // verified record, the selector's measured-supersedes-estimate rule keeps admission
    // byte-for-byte unchanged.
    //
    // A rung's fitted estimate is emitted only when EVERY `exact_request` record of this tier —
    // graded at ITS OWN geometry through the same `make_evidence` conjuncts as the request path —
    // passes the full measured eligibility predicate (`optimized_eligibility`). Whole-tier rather
    // than best-anchor, because the curve is fitted across every cell (see the fail-closed note at
    // the loop head). That check carries every restriction sc-18096 established for extrapolation
    // bases —
    // closure-CURRENT capture (a stale record may serve its own cell behind the stale margin but
    // may not seed an extrapolation; the estimate margin was derived over same-closure re-capture
    // variance and cannot also absorb closure drift), the loaded contract's calibration identity
    // (a drifted provider must not receive fitted candidates built from another identity's
    // records), artifact + hardware loadability, measured-composition agreement, parity, and
    // curve↔record phase agreement at the anchor cell (so a mutated curve that no longer describes
    // the measurement cannot smuggle its numbers back in as an estimate).
    //
    // `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE` (sc-18094) is honored at this synthesis
    // seam: if the curve moves the request peak onto a different phase than the anchor record's
    // binding phase, the fitted candidate is NOT emitted. No weights+headroom floor path exists on
    // this lane (the constraint's scope exemption is therefore never exercised here): every rung
    // always has a curve, and a floor from the resident row could never admit deeper than the
    // resident baseline it equals, so a suppressed rung honestly falls out of estimate admission.
    let mut estimates: Vec<(MemorySelection, MemoryEvidence)> = Vec::new();
    for (strategy, phases, _, selection, from_anchor) in &measured {
        if *from_anchor {
            // Priced by the anchor derivation, not by the tier's curves: the fitted-estimate
            // synthesis below has nothing to say about it and must not relabel it `fitted`.
            continue;
        }
        let manifest_rung = krea_turbo_manifest_key(*strategy);
        let record_is_eligible = |record: &Value, record_width: u32, record_height: u32| {
            let anchor_geometry = MemoryGeometry {
                width: record_width,
                height: record_height,
                batch: 1,
                frames: KREA_LANE_FRAMES,
                reference_count: 0,
            };
            let Some(anchor_phases) = krea_rung_phase_peaks(
                manifest_entry,
                tier,
                *strategy,
                record_width,
                record_height,
                anchor_geometry.frames,
            ) else {
                return false;
            };
            // The FULL measured-eligibility predicate, not just `Verified` conformance: it
            // additionally requires the record's measured composition to agree with the loaded
            // contract's and the calibration identity to match (the mirror of
            // `mlx_fit_gate::collect_estimate_bases`' engaged-composition filter).
            make_evidence(
                *selection,
                Some(manifest_rung),
                anchor_phases.peak_gb(),
                Some(anchor_phases),
                Some(record),
                anchor_geometry,
            )
            .optimized_eligibility(provider_contract)
            .is_ok()
        };
        // FAIL CLOSED ON THE WHOLE TIER (sc-18097 review, major finding). The tier's phase curves
        // are fitted across EVERY measured cell, and the measured path's own
        // `exact_strategy_parameters` conjunct requires the record peak and the curve to agree
        // within 0.01 GiB — so the curve evaluated at an excluded cell's geometry reproduces that
        // cell's number. Anchoring on a surviving SIBLING record would therefore let a
        // structurally excluded cell (composition mismatch, drifted identity, stale capture) be
        // re-admitted at its own numbers behind nothing but the 4% estimate margin, laundering
        // the per-cell structural exclusion the epic requires to keep excluding. So a rung emits
        // fitted candidates only when EVERY `exact_request` record of the tier is eligible for it:
        // one bad row disqualifies the tier's curve for that rung, not merely that row.
        if let Some((_, bad_width, bad_height)) = tier_records
            .iter()
            .find(|(record, width, height)| !record_is_eligible(record, *width, *height))
        {
            tracing::info!(
                route = "krea_2_turbo",
                backend = "candle",
                ?strategy,
                ineligible_record_geometry = format!("{bad_width}x{bad_height}"),
                "fitted-curve estimates suppressed for this rung: a measured record of this tier \
                 is structurally excluded, and the tier's curve is fitted across it"
            );
            continue;
        }
        let anchor = tier_records
            .iter()
            .copied()
            // Prefer the largest measured cell: the curve is fitted within `maxMeasuredPixels`,
            // and the top sample is the anchor closest to that envelope.
            .max_by_key(|(_, record_width, record_height)| {
                u64::from(*record_width) * u64::from(*record_height)
            });
        let Some((anchor_record, anchor_width, anchor_height)) = anchor else {
            continue;
        };
        let measured_binding =
            krea_record_phase_peaks(anchor_record, manifest_rung).map(krea_binding_phase);
        let request_binding = krea_binding_phase(*phases);
        if crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE
            && measured_binding != Some(request_binding)
        {
            // The pinned sc-18094 constraint: the corpus shows a per-phase re-capture spread no
            // margin in the policy absorbs, so a curve that moves the request peak onto a
            // different phase than the one the anchor measured is refused rather than margined.
            tracing::info!(
                route = "krea_2_turbo",
                backend = "candle",
                ?strategy,
                anchor_geometry = format!("{anchor_width}x{anchor_height}"),
                ?measured_binding,
                request_binding,
                "fitted-curve estimate rejected: the curve moves the request peak onto a \
                 different phase than the measured anchor's \
                 (ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE)"
            );
            continue;
        }
        let predicted_peak_bytes = bytes(phases.peak_gb());
        tracing::info!(
            route = "krea_2_turbo",
            backend = "candle",
            ?strategy,
            anchor_geometry = format!("{anchor_width}x{anchor_height}"),
            raw_peak_bytes = predicted_peak_bytes,
            "synthesized fitted-curve estimate candidate from the anchored phase curves"
        );
        estimates.push((
            *selection,
            sceneworks_core::memory_contracts::unverified_estimate(
                MemoryEvidenceKey {
                    model_family: "krea_2_turbo".to_owned(),
                    resolved_route: "krea_2_turbo".to_owned(),
                    backend: gen_core::MemoryBackend::Candle,
                    tier: numeric_tier,
                    load_shape: provider_contract.load_shape,
                    mode: gen_core::MemoryMode::TextToImage,
                    reference_shape: gen_core::MemoryReferenceShape::None,
                    overlay: None,
                    geometry,
                    frames_per_second: None,
                    strategy: selection.strategy,
                    engaged_composition: provider_contract.engaged_composition(selection.strategy),
                    parameters: selection.parameters,
                },
                calibration_abi,
                calibration_fingerprint.to_owned(),
                scene_works_revision.to_owned(),
                predicted_peak_bytes,
            ),
        ));
    }
    let mut candidates = selections
        .iter()
        .zip(&evidence)
        .map(|(selection, evidence)| Candidate {
            selection: *selection,
            evidence,
            basis: memory_strategy::CandidateBasis::Measured,
            unmodeled_activation_bytes: None,
        })
        .collect::<Vec<_>>();
    candidates.extend(estimates.iter().map(|(selection, evidence)| Candidate {
        selection: *selection,
        evidence,
        basis: memory_strategy::CandidateBasis::EstimateFittedCurve,
        // A fitted per-phase curve carries no weights/activation split (sc-22508); its remaining
        // uncertainty is the same-cell recapture spread the policy charges on the whole peak.
        unmodeled_activation_bytes: None,
    }));
    // sc-22509: anchor-derived rungs.
    candidates.extend(
        anchor_derived
            .iter()
            .map(|(selection, evidence)| Candidate {
                selection: *selection,
                evidence,
                basis: memory_strategy::CandidateBasis::EstimateAnchorDerived {
                    lane: memory_strategy::AnchorDerivationLane::Image,
                },
                // No weights/activation split (sc-22508), for the same reason the fitted-curve
                // candidate above has none: the anchor derivation decomposes its peak by PHASE
                // (conditioning/denoise/decode), not into counted weights plus an activation
                // remainder. Only the manifest-row FLOOR declares that split, so the allowance is
                // charged on the whole peak here.
                unmodeled_activation_bytes: None,
            }),
    );
    // sc-22667: the operational reserve is charged ONCE, on the budget side, against every
    // candidate (`ReserveCharge::EveryCandidate`): none of this lane's candidates carries a pad
    // inside its peak — see the function doc. The legacy fixed `HEADROOM_GB` is gone from both
    // sides of the comparison.
    let selection = memory_strategy::select_strategy_charging(
        request,
        provider_contract,
        Some(Budget {
            available_gb: budget.free_gb,
            reclaimable_gb: 0.0,
            total_gb: budget.total_gb,
            reserved_headroom_gb: reserve_gb,
        }),
        &candidates,
        memory_strategy::ReserveCharge::EveryCandidate,
    );
    let admitted =
        |needed_gb: f64, available_gb: f64| crate::candle_memory_strategy::AdmittedBudget {
            needed_gb,
            available_gb,
            reserve_gb,
        };
    match selection {
        Selection::Selected {
            selection:
                selected @ MemorySelection {
                    strategy: MemoryStrategy::Resident,
                    ..
                },
            needed_gb,
            available_gb,
        } => Some(KreaTurboFit::Resident {
            peak_gb: resident_peak_gb,
            needed_gb: needed_gb + reserve_gb,
            selection: selected,
            admitted: admitted(needed_gb, available_gb),
        }),
        Selection::Selected {
            selection: selected,
            needed_gb,
            available_gb,
        } => {
            let (_, phases, _, _, from_anchor) = measured
                .into_iter()
                .find(|(_, _, _, selection, _)| selection.strategy == selected.strategy)?;
            // What priced the rung: the anchor derivation, else the request cell's own measured
            // record, else the tier's fitted curve (the only estimate an in-envelope cell without
            // a record can be carried by — see `Fits::estimate_scoped`).
            let basis = if from_anchor {
                memory_strategy::CandidateBasis::EstimateAnchorDerived {
                    lane: memory_strategy::AnchorDerivationLane::Image,
                }
            } else if evidence_record.is_some() {
                memory_strategy::CandidateBasis::Measured
            } else {
                memory_strategy::CandidateBasis::EstimateFittedCurve
            };
            let memory = gen_core::GenerationMemory {
                tile_vae_decode: provider_contract
                    .engages(selected.strategy, MemoryStrategy::BoundedDecode),
                chunk_attention: provider_contract
                    .engages(selected.strategy, MemoryStrategy::BoundedAttention),
                stream_transformer_blocks: provider_contract.engages(
                    selected.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                ),
                ..Default::default()
            };
            Some(KreaTurboFit::Fits {
                phases,
                needed_gb: needed_gb + reserve_gb,
                selection: selected,
                memory,
                basis,
                admitted: admitted(needed_gb, available_gb),
                // See the field doc: with no `exact_request` record at this geometry the rung's
                // measured candidates are structurally excluded, so only a synthesized estimate
                // can have carried it.
                estimate_scoped: evidence_record.is_none(),
            })
        }
        Selection::Reject { needed_gb, .. } => {
            let (_, phases, _, _, _) = measured.last().copied()?;
            Some(KreaTurboFit::Reject {
                phases,
                needed_gb: needed_gb + reserve_gb,
            })
        }
        // sc-18097 narrowed this arm's meaning: an in-envelope unmeasured geometry now carries a
        // fitted-curve estimate per rung, so the selector lands here only when a rung has NO
        // eligible candidate at all — a stale-closure or otherwise unverifiable manifest (whose
        // records may not seed extrapolation), a mutated record/curve pair, a binding-phase flip,
        // or an overlay the measurements do not cover. Those remain the explicit fallback to the
        // established generic gate, exactly as before.
        Selection::Unverified { reason } => Some(KreaTurboFit::Unverified { reason }),
    }
}

/// The reserve the threshold fixtures below were calibrated against: the legacy 2 GB, which is
/// exactly what [`ladder_reserve_gb`] produces for a card whose idle baseline is at or above
/// `HEADROOM_GB - LADDER_RESERVE_MARGIN_GB` (the ceiling arm). The reserve is an INPUT of the lane
/// since sc-22667 — production derives it from the raw probe — so the fixtures state it rather
/// than re-deriving a different number from their `free == total` budgets, which would move
/// every calibrated admission threshold for no reason connected to what they grade. The
/// derivation itself is graded by the `ladder_reserve_*` tests and by
/// `an_out_of_envelope_krea_request_charges_the_ladder_reserve_once`.
#[cfg(test)]
pub(super) const KREA_FIXTURE_RESERVE_GB: f64 = HEADROOM_GB;

#[cfg(test)]
pub(super) fn krea_turbo_fit(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    allow_streamed_blocks: bool,
) -> Option<KreaTurboFit> {
    let runtime = KreaRuntimeEvidenceContext::verified_for_test(tier);
    krea_turbo_fit_with_runtime(
        manifest_entry,
        tier,
        width,
        height,
        budget,
        KREA_FIXTURE_RESERVE_GB,
        allow_streamed_blocks,
        Some(&runtime),
    )
}

/// Highest lower-pixel manifest bucket that the deepest available Krea Turbo rung can actually fit.
/// Used only to make rejection copy truthful: if this returns `None`, lowering resolution is not
/// presented as an escape hatch.
///
/// The advice is restricted to MEASURED admissions (sc-18097): an estimate-scoped `Fits` — the
/// fitted-curve admission of a geometry nobody measured — is not offered here, mirroring
/// `mlx_fit_gate::verified_lower_alternative` ("no formula, interpolation, tier heuristic, or
/// aspect-ratio rewrite is admitted"). Naming a fallback that itself rests on an estimate invites
/// a second refusal at the geometry the user was just told to switch to. `Resident` keeps its
/// pre-sc-18097 acceptance: that verdict is the manifest's geometry-independent `vramGbByTier`
/// row, which this helper has always been allowed to quote.
#[allow(clippy::too_many_arguments)]
pub(crate) fn krea_turbo_smaller_fit_with_runtime(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    allow_streamed_blocks: bool,
    runtime: Option<&KreaRuntimeEvidenceContext>,
) -> Option<(u32, u32)> {
    let current_pixels = u64::from(width) * u64::from(height);
    let resolutions = manifest_entry
        .get("limits")?
        .get("resolutions")?
        .as_array()?;
    let mut candidates = resolutions
        .iter()
        .filter_map(Value::as_str)
        .filter_map(|resolution| resolution.split_once('x'))
        .filter_map(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .filter(|(w, h)| u64::from(*w) * u64::from(*h) < current_pixels)
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(w, h)| std::cmp::Reverse(u64::from(*w) * u64::from(*h)));
    candidates.into_iter().find(|(w, h)| {
        matches!(
            krea_turbo_fit_with_runtime(
                manifest_entry,
                tier,
                *w,
                *h,
                budget,
                reserve_gb,
                allow_streamed_blocks,
                runtime,
            ),
            Some(
                KreaTurboFit::Resident { .. }
                    | KreaTurboFit::Fits {
                        estimate_scoped: false,
                        ..
                    }
            )
        )
    })
}

#[cfg(test)]
pub(super) fn krea_turbo_smaller_fit(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    allow_streamed_blocks: bool,
) -> Option<(u32, u32)> {
    let runtime = KreaRuntimeEvidenceContext::verified_for_test(tier);
    krea_turbo_smaller_fit_with_runtime(
        manifest_entry,
        tier,
        width,
        height,
        budget,
        KREA_FIXTURE_RESERVE_GB,
        allow_streamed_blocks,
        Some(&runtime),
    )
}
