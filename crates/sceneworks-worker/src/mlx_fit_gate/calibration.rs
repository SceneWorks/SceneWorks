//! mlx_fit_gate calibration owner.
use super::*;

/// Suffix shared by the worker's route/tier Mage calibration identities.
/// The complete identities must remain identical to mlx-gen-mage's loaded provider contract. sc-22733 (inference #953,
/// epic sc-22723) retired the single string `mage-flow-mlx-shared-ladder-2026-08-03-v1` for a
/// table keyed on (route, PROVEN artifact tier) — `mage-flow-<route>-<tier>-mlx-shared-ladder-v1`,
/// eighteen cells — because the three tiers of one route are three different resident sets and one
/// anchor cannot price all three. The engine binds a cell only when the tier is proven off the
/// `quantization.bits` marker of the component directories the loader itself opened (DiT and text
/// encoder agreeing) and equals `LoadSpec::quantize`, so the plan's `tier.quant` IS the tier axis.
/// The worker mirrors the table here rather than calling the engine because the Resident-path
/// handshake below is compiled on every platform; the macOS test
/// `mage_estimator_fingerprint_matches_the_linked_provider_contract` pins every cell to the linked
/// engine's `production_calibration_fingerprint`, so a move on either side is red at test time.
/// The prior generation-peak token predated the shared ladder and caused every exact Mage request
/// to fail the provider/gate handshake before selection.
pub(super) const MAGE_CALIBRATION_FINGERPRINT_SUFFIX: &str = "-mlx-shared-ladder-v1";

/// The route label inside a Mage-Flow calibration string, mirroring
/// `mlx_gen_mage::model::MageVariant::route_label` over the six registered ids. `None` for any id
/// the engine does not serve: a plan naming one could only reach an unpaired estimator.
pub(super) fn mage_calibration_route_label(engine_id: &str) -> Option<&'static str> {
    Some(match engine_id {
        "mage_flow" => "mage-flow",
        "mage_flow_base" => "mage-flow-base",
        "mage_flow_turbo" => "mage-flow-turbo",
        "mage_flow_edit" => "mage-flow-edit",
        "mage_flow_edit_base" => "mage-flow-edit-base",
        "mage_flow_edit_turbo" => "mage-flow-edit-turbo",
        _ => return None,
    })
}

/// The tier label inside a Mage-Flow calibration string, mirroring
/// `mlx_gen_mage::model::calibration_tier_label`. `None` for a tier the family does not ship, so an
/// unshipped tier is unnameable rather than collapsed onto a neighbour.
pub(super) fn mage_calibration_tier_label(quant: Option<gen_core::Quant>) -> Option<&'static str> {
    match quant {
        None => Some("bf16"),
        Some(gen_core::Quant::Q4) => Some("q4"),
        Some(gen_core::Quant::Q8) => Some("q8"),
        Some(_) => None,
    }
}

/// The production calibration identity the loaded `mlx-gen-mage` route publishes for one
/// (route, tier) cell, or `None` when no cell exists for the pair.
pub(super) fn mage_calibration_fingerprint(
    engine_id: &str,
    quant: Option<gen_core::Quant>,
) -> Option<String> {
    let route = mage_calibration_route_label(engine_id)?;
    let tier = mage_calibration_tier_label(quant)?;
    Some(format!(
        "mage-flow-{route}-{tier}{MAGE_CALIBRATION_FINGERPRINT_SUFFIX}"
    ))
}

impl DecodeGeometryPolicyWire {
    pub(super) fn into_core(self, model_id: &str) -> WorkerResult<MemoryDecodeGeometryPolicy> {
        if self.resolved_route != model_id {
            return Err(WorkerError::InvalidPayload(format!(
                "decode-quality policy route {:?} cannot authorize model {:?}",
                self.resolved_route, model_id
            )));
        }
        let backend = match self.backend.as_str() {
            "mlx" => MemoryBackend::Mlx,
            other => {
                return Err(WorkerError::InvalidPayload(format!(
                    "MLX decode-quality policy cannot use backend {other:?}"
                )))
            }
        };
        let tier = match self.tier.as_str() {
            "bf16" => MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            "q4" => MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
            "q8" => MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q8),
                component_precision_floors: &[],
            },
            "nvfp4" => MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Nvfp4),
                component_precision_floors: &[],
            },
            "fp32" => MemoryNumericTier {
                precision: Precision::Fp32,
                quant: None,
                component_precision_floors: &[],
            },
            other => {
                return Err(WorkerError::InvalidPayload(format!(
                    "unsupported decode-quality tier {other:?}"
                )))
            }
        };
        let load_shape = match self.load_shape.as_str() {
            "eager_materialization" => gen_core::LoadShape::EagerMaterialization,
            "deferred_materialization" => gen_core::LoadShape::DeferredMaterialization,
            other => {
                return Err(WorkerError::InvalidPayload(format!(
                    "unsupported decode-quality load shape {other:?}"
                )))
            }
        };
        let mode = match self.mode.as_str() {
            "text_to_image" => MemoryMode::TextToImage,
            "image_to_image" => MemoryMode::ImageToImage,
            "edit" => MemoryMode::Edit,
            other => MemoryMode::Other(other.to_owned()),
        };
        let disposition = match self.disposition {
            DecodeQualityDispositionWire::Admitted => MemoryDecodeQualityDisposition::Admitted,
            DecodeQualityDispositionWire::Refused { reason } => {
                MemoryDecodeQualityDisposition::Refused { reason }
            }
        };
        Ok(MemoryDecodeGeometryPolicy {
            quality_abi: self.quality_abi,
            family: self.family,
            resolved_route: self.resolved_route,
            backend,
            tier,
            load_shape,
            artifact: MemoryDecodeArtifactIdentity {
                repository: self.artifact.repository,
                revision: self.artifact.revision,
                variant: self.artifact.variant,
                fingerprint: self.artifact.fingerprint,
            },
            implementation_fingerprint: self.implementation_fingerprint,
            mode,
            overlay: self.overlay,
            geometry: MemoryGeometry {
                width: self.geometry.width,
                height: self.geometry.height,
                batch: self.geometry.batch,
                frames: self.geometry.frames,
                reference_count: self.geometry.reference_count,
            },
            use_pid: self.use_pid,
            tile_edge: self.tile_edge,
            overlap: self.overlap,
            metric: self.metric,
            maximum_error: self.maximum_error,
            fixtures: self
                .fixtures
                .into_iter()
                .map(|fixture| MemoryDecodeQualityFixture {
                    seed: fixture.seed,
                    production_latent_provenance: fixture.production_latent_provenance,
                    production_latent_sha256: fixture.production_latent_sha256,
                    dense_output_sha256: fixture.dense_output_sha256,
                    tiled_output_sha256: fixture.tiled_output_sha256,
                    observed_error: fixture.observed_error,
                })
                .collect(),
            production_evidence_sha256: self.production_evidence_sha256,
            disposition,
        })
    }
}

pub(crate) fn manifest_declares_decode_quality_policies(
    manifest: &JsonObject<String, Value>,
) -> bool {
    manifest
        .get("mlx")
        .and_then(Value::as_object)
        .and_then(|mlx| mlx.get("memoryStrategyContract"))
        .and_then(|contract| contract.get("implementations"))
        .and_then(Value::as_array)
        .is_some_and(|implementations| {
            implementations.iter().any(|implementation| {
                implementation
                    .get("parameterRanges")
                    .and_then(|ranges| ranges.get("decodeGeometryPolicies"))
                    .and_then(Value::as_array)
                    .is_some_and(|rows| !rows.is_empty())
            })
        })
}

/// Read the semantic P9 table from the model's derived static contract. This does not read or
/// mutate the performance/calibration bundle: quality rows have their own ABI, hashes, and runtime
/// path. Missing policy preserves the historical provider contract and its estimated fallback.
pub(crate) fn decode_quality_policies_from_manifest(
    manifest: &JsonObject<String, Value>,
    model_id: &str,
) -> WorkerResult<Vec<MemoryDecodeGeometryPolicy>> {
    let Some(contract) = manifest
        .get("mlx")
        .and_then(Value::as_object)
        .and_then(|mlx| mlx.get("memoryStrategyContract"))
    else {
        return Ok(Vec::new());
    };
    let implementations = contract
        .get("implementations")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "mlx.memoryStrategyContract.implementations must be an array".to_owned(),
            )
        })?;
    let mut policies = Vec::new();
    for implementation in implementations {
        let Some(ranges) = implementation.get("parameterRanges") else {
            continue;
        };
        let Some(rows) = ranges.get("decodeGeometryPolicies") else {
            continue;
        };
        if implementation.get("rung").and_then(Value::as_str) != Some("bounded_decode") {
            return Err(WorkerError::InvalidPayload(
                "decodeGeometryPolicies may be declared only by bounded_decode".to_owned(),
            ));
        }
        let rows = rows.as_array().ok_or_else(|| {
            WorkerError::InvalidPayload("decodeGeometryPolicies must be an array".to_owned())
        })?;
        for row in rows {
            let wire: DecodeGeometryPolicyWire =
                serde_json::from_value(row.clone()).map_err(|error| {
                    WorkerError::InvalidPayload(format!("invalid decode-quality policy: {error}"))
                })?;
            policies.push(wire.into_core(model_id)?);
        }
    }
    Ok(policies)
}

/// Bind packaged semantic rows to provenance resolved independently beside the exact weights tree.
/// An unavailable or mismatched identity refuses only the optimized decode table and is emitted as
/// a typed audit decision; it never turns a valid model install into a dense-decode load failure.
pub(crate) fn bind_decode_quality_policies_from_manifest(
    manifest: &JsonObject<String, Value>,
    model_id: &str,
    resolved_artifact: Option<&ResolvedArtifactProvenance>,
) -> WorkerResult<DecodeQualityPolicyBinding> {
    let policies = decode_quality_policies_from_manifest(manifest, model_id)?;
    if policies.is_empty() {
        return Ok(DecodeQualityPolicyBinding {
            policies,
            authoritative: false,
            runtime_identity: None,
            decision: DecodeQualityBindingDecision::Legacy,
        });
    }
    let Some(resolved_artifact) = resolved_artifact else {
        return Ok(DecodeQualityPolicyBinding {
            policies: Vec::new(),
            authoritative: true,
            runtime_identity: None,
            decision: DecodeQualityBindingDecision::Refused {
                reason: "decode-quality table has no independently resolved artifact identity"
                    .to_owned(),
            },
        });
    };
    let artifact = MemoryDecodeArtifactIdentity {
        repository: resolved_artifact.identity.repository.clone(),
        revision: resolved_artifact.identity.revision.clone(),
        variant: resolved_artifact.identity.variant.clone(),
        fingerprint: resolved_artifact.identity.fingerprint.clone(),
    };
    if let Some(foreign) = policies.iter().find(|policy| policy.artifact != artifact) {
        return Ok(DecodeQualityPolicyBinding {
            policies: Vec::new(),
            authoritative: true,
            runtime_identity: None,
            decision: DecodeQualityBindingDecision::Refused {
                reason: format!(
                    "decode-quality artifact {:?}@{}:{} ({}) does not match resolved artifact {:?}@{}:{} ({})",
                    foreign.artifact.repository,
                    foreign.artifact.revision,
                    foreign.artifact.variant,
                    foreign.artifact.fingerprint,
                    artifact.repository,
                    artifact.revision,
                    artifact.variant,
                    artifact.fingerprint,
                ),
            },
        });
    }
    let runtime_identity = MemoryDecodeQualityRuntimeIdentity {
        artifact: artifact.clone(),
    };
    let decision = DecodeQualityBindingDecision::Bound {
        artifact,
        row_count: policies.len(),
    };
    Ok(DecodeQualityPolicyBinding {
        policies,
        authoritative: true,
        runtime_identity: Some(runtime_identity),
        decision,
    })
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn attach_decode_quality_binding(
    mut spec: LoadSpec,
    binding: DecodeQualityPolicyBinding,
    route: &str,
) -> LoadSpec {
    tracing::info!(
        event = "mlx_decode_quality_binding",
        route,
        decision = ?binding.decision,
        "resolved semantic decode-quality load binding"
    );
    spec = spec
        .with_decode_geometry_policy_authority(binding.authoritative)
        .with_decode_geometry_policies(binding.policies);
    if let Some(identity) = binding.runtime_identity {
        spec = spec.with_decode_quality_runtime_identity(identity);
    }
    spec
}

impl MlxRequestPlan {
    /// Bind the already-resolved artifact tier without reinterpreting `LoadSpec::quantize`.
    /// Prepacked Krea q4/q8 artifacts intentionally load with `quantize=None`; this explicit axis is
    /// therefore required for request/evidence identity even when no calibration provenance exists.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn with_resolved_artifact_tier(
        mut self,
        resolved_tier: Option<&str>,
    ) -> WorkerResult<Self> {
        let Some(resolved_tier) = resolved_tier else {
            return Err(WorkerError::InvalidPayload(format!(
                "{} has no resolved MLX artifact tier",
                self.model_id
            )));
        };
        self.tier = numeric_tier_for_resolved(
            resolved_tier,
            self.tier,
            declared_component_floors(self.engine_id),
        )
        .map_err(|reason| WorkerError::InvalidPayload(format!("{}: {reason}", self.model_id)))?;
        Ok(self)
    }

    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn for_spec_and_manifest(
        engine_id: &'static str,
        model_id: &str,
        spec: &LoadSpec,
        manifest: Option<&JsonObject<String, Value>>,
        resolved_artifact: Option<ResolvedArtifactProvenance>,
    ) -> Self {
        let media = crate::inference_runtime::media();
        let provider_footprint = media.footprint(engine_id, spec).ok().flatten();
        let activation_anchor_bytes = media.activation_memory_bytes_1024(engine_id).ok().flatten();
        Self::for_spec_and_manifest_with_provider_facts(
            engine_id,
            model_id,
            spec,
            manifest,
            resolved_artifact,
            provider_footprint,
            activation_anchor_bytes,
        )
    }

    /// Prepared-spec planner used by imported single-file routes. Provider queries and token
    /// validation are fallible contract operations here: failures propagate to the job instead of
    /// being silently converted into an unmeasured/raw-restat estimate.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn try_for_spec_and_manifest(
        engine_id: &'static str,
        model_id: &str,
        spec: &LoadSpec,
        manifest: Option<&JsonObject<String, Value>>,
        resolved_artifact: Option<ResolvedArtifactProvenance>,
    ) -> WorkerResult<Self> {
        let media = crate::inference_runtime::media();
        let provider_footprint = media
            .footprint(engine_id, spec)
            .map_err(|error| crate::classify_engine_error("MLX footprint query failed", error))?;
        let activation_anchor_bytes =
            media
                .activation_memory_bytes_1024(engine_id)
                .map_err(|error| {
                    crate::classify_engine_error("MLX activation-memory query failed", error)
                })?;
        let (asset_bytes, _, headroom) = spec_component_bytes_with_provider_footprint_checked(
            engine_id,
            spec,
            provider_footprint,
        )
        .map_err(|error| {
            crate::classify_engine_error("MLX prepared source accounting failed", error)
        })?;
        let file_sizes = prepared_file_sizes(spec).map_err(|error| {
            crate::classify_engine_error("MLX prepared source validation failed", error)
        })?;
        let folded_control_bytes = spec
            .control
            .as_ref()
            .map_or(0, |source| prepared_source_bytes(source, &file_sizes));
        let folded_adapter_source_bytes = spec.adapters.iter().fold(0_u64, |total, adapter| {
            total.saturating_add(prepared_source_bytes(
                &WeightsSource::File(adapter.path.clone()),
                &file_sizes,
            ))
        });
        let folded_adapter_bytes =
            adapter_resident_source_bytes(engine_id, spec, folded_adapter_source_bytes);
        Ok(Self::for_spec_and_manifest_with_accounting(
            engine_id,
            model_id,
            spec,
            manifest,
            resolved_artifact,
            asset_bytes,
            folded_control_bytes,
            folded_adapter_bytes,
            headroom,
            activation_anchor_bytes,
        ))
    }

    /// The platform-neutral core of [`Self::for_spec_and_manifest`]. Production supplies both facts
    /// from the active provider registry; tests may inject the same provider-owned facts when the
    /// host deliberately links no MLX catalog (the default Linux workspace build). Keeping the
    /// filesystem accounting and plan construction here prevents an audit from replacing the live
    /// path with hand-written `asset + headroom` arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn for_spec_and_manifest_with_provider_facts(
        engine_id: &'static str,
        model_id: &str,
        spec: &LoadSpec,
        manifest: Option<&JsonObject<String, Value>>,
        resolved_artifact: Option<ResolvedArtifactProvenance>,
        provider_footprint: Option<PerComponentBytes>,
        activation_anchor_bytes: Option<u64>,
    ) -> Self {
        let (asset_bytes, _, headroom) =
            spec_component_bytes_with_provider_footprint(engine_id, spec, provider_footprint);
        let folded_control_bytes = spec.control.as_ref().map_or(0, weights_source_bytes);
        let folded_adapter_bytes = adapter_source_bytes_for_gate(engine_id, spec);
        Self::for_spec_and_manifest_with_accounting(
            engine_id,
            model_id,
            spec,
            manifest,
            resolved_artifact,
            asset_bytes,
            folded_control_bytes,
            folded_adapter_bytes,
            headroom,
            activation_anchor_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn for_spec_and_manifest_with_accounting(
        engine_id: &'static str,
        model_id: &str,
        spec: &LoadSpec,
        manifest: Option<&JsonObject<String, Value>>,
        resolved_artifact: Option<ResolvedArtifactProvenance>,
        asset_bytes: u64,
        folded_control_bytes: u64,
        folded_adapter_bytes: u64,
        headroom: HeadroomAllowance,
        activation_anchor_bytes: Option<u64>,
    ) -> Self {
        let declared_floors = declared_component_floors(engine_id);
        let spec_tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: active_component_floors(declared_floors, spec.quantize),
        };
        let (tier, calibration) = match manifest {
            Some(manifest) => match MlxCalibrationBinding::from_manifest(manifest) {
                Ok(Some(bindings)) => match resolved_artifact {
                    Some(resolved) => match resolved.fixed_artifact_tier.as_deref() {
                        Some(fixed_tier) => {
                            match numeric_tier_for_resolved(fixed_tier, spec_tier, declared_floors)
                            {
                                Ok(tier) => (
                                    tier,
                                    MlxCalibrationConfig::Valid(MlxCalibrationSet {
                                        bindings,
                                        resolved,
                                    }),
                                ),
                                Err(reason) => (spec_tier, MlxCalibrationConfig::Invalid(reason)),
                            }
                        }
                        None => (
                            spec_tier,
                            MlxCalibrationConfig::Valid(MlxCalibrationSet { bindings, resolved }),
                        ),
                    },
                    // The manifest opts in, but the resolver could not PROVE which artifact is on
                    // disk. That is a property of the install, not of the opt-in: a receipt written
                    // before download-time tree stamps existed (`backfill_current_receipt` never
                    // writes `artifactTreeStamp`) resolves a perfectly loadable snapshot and still
                    // yields no provenance. Refusing the request outright would strand every such
                    // install with no repair path, so this is a NON-COVERING state and routes to the
                    // established legacy selector — the conservative gate — exactly like every other
                    // one. `Invalid` stays reserved for a malformed opt-in, which is an authoring bug.
                    None => (spec_tier, MlxCalibrationConfig::Unproven),
                },
                Ok(None) => (spec_tier, MlxCalibrationConfig::Absent),
                Err(reason) => (spec_tier, MlxCalibrationConfig::Invalid(reason)),
            },
            None => (spec_tier, MlxCalibrationConfig::Absent),
        };
        // Request budgeting already removes the legacy 2 GiB unified reserve from the available
        // envelope. Keep the remainder of the 4 GiB OS/app reserve fixed, and source only the bare
        // 1024² activation term from the provider's exact-tier measurement. An absent measurement
        // retains the load allowance's conservative activation component.
        let fixed_reserve_bytes = gib_to_bytes(
            (OS_APP_RESERVE_GB - crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB).max(0.0),
        );
        let activation_anchor_bytes = activation_anchor_bytes
            .unwrap_or_else(|| gib_to_bytes((headroom.total_gb - headroom.os_reserve_gb).max(0.0)));
        Self {
            engine_id,
            model_id: model_id.to_owned(),
            tier,
            asset_bytes,
            folded_control_bytes,
            folded_adapter_bytes,
            // This field remains the combined request allowance so the generic formula can hold the
            // fixed slice out before scaling: fixed reserve + provider activation anchor.
            activation_headroom_bytes: activation_anchor_bytes.saturating_add(fixed_reserve_bytes),
            fixed_reserve_bytes,
            calibration,
            load_shape_declaration_result: spec.load_shape_declaration_result,
        }
    }

    /// Request peak for the generic/legacy path: resident weights + a FIXED OS/app reserve + the
    /// activation transient scaled by output area (sc-16195).
    ///
    /// The predecessor scaled the WHOLE `activation_headroom_bytes` by megapixels. That constant is
    /// `HEADROOM_GB` (a 1024²-only calibration) minus the legacy reserve, and its own doc comment
    /// decomposes it as *transient + a ~4 GiB macOS/app reserve*. Scaling the reserve was simply
    /// wrong: the OS's working set does not grow because this request renders 2048² instead of 1024².
    ///
    /// The AREA term, by contrast, is scaled exactly as before — because the sc-16195 sweep
    /// (`crate::resolution_sweep`, 7 tiers across 5 families × 5 cells, `docs/sc-16195/`) found the
    /// measured transient to be PROPORTIONAL to megapixels above 1024². Each tier's transient
    /// normalised by its OWN 1024² value, against the proportional target:
    ///
    /// | tier                     | 1024² GiB | ×1.50 area | ×2.25 area | ×4.00 area |
    /// |--------------------------|----------:|-----------:|-----------:|-----------:|
    /// | illustrious q8 (SDXL)    |     14.04 |      1.497 |      2.245 |      3.922 |
    /// | sdxl bf16 (SDXL, dense)  |     14.04 |      1.497 |      2.245 |      3.922 |
    /// | z_image_turbo q4 (DiT)   |     14.04 |      1.497 |      2.244 |      3.921 |
    /// | lens q4 (DiT)            |     14.04 |      1.497 |      2.245 |      3.922 |
    /// | qwen_image q8 (tiled VAE)|      7.66 |      1.498 |      2.246 |      3.994 |
    /// | krea_2_turbo q8 (tiled)  |      7.67 |      1.498 |      2.245 |      3.992 |
    /// | krea_2_turbo bf16 (tiled)|      7.67 |      1.498 |      2.245 |      3.992 |
    /// | **proportional target**  |         — |      1.500 |      2.250 |      4.000 |
    ///
    /// Maximum deviation from proportional across every cell above the anchor: **1.97%**.
    ///
    /// This matters more as a REFUTATION than as a confirmation. The story that filed this work
    /// reasoned from the only prior evidence — two sc-5567 points suggesting the transient was
    /// markedly SUBLINEAR (16× area → 3.8× memory, i.e. an exponent near 0.48) — and expected the fix
    /// to be a fitted sub-linear curve. Fitting that exponent would have predicted illustrious q8 at
    /// 2048² as 27.3 GiB against a measured 55.06 — a ~28 GiB UNDER-prediction on a gate whose
    /// permissive-side failure mode is an OS Jetsam SIGKILL. The area term is left linear because it
    /// measured linear, not because it was not examined.
    ///
    /// What the sweep did NOT settle is the per-family LEVEL: the 1024² column above splits 14.04 vs
    /// 7.66 on whether the family's VAE decode is tiled (sc-11747), and every family is currently
    /// charged 14. That is the larger error, and it is tracked as sc-16209 — reducing a family's
    /// anchor lowers a SIGKILL-guarding margin, which is a different kind of change from this one.
    ///
    /// The `.max(1.0)` floor on the scale is likewise kept: below 1024² the measured transient stops
    /// falling off proportionally (illustrious 0.305× and qwen 0.512× of their anchors at 0.25×
    /// area, both ABOVE the 0.25× a proportional term would predict), so the floor is the
    /// conservative reading of the data rather than a leftover.
    ///
    /// `batch` continues to multiply the area term only — a genuine batched pass renders more
    /// pixels, but it does not run more copies of macOS. See [`request_batch`]: on this lane it is
    /// always 1, because a multi-image job is a sequential loop (sc-16194).
    ///
    /// Provider anchors are bare activation measurements. The request planner adds the remaining
    /// fixed OS/app reserve separately, so even an anchor larger than the generic fallback (Lens
    /// dense) keeps its whole measured value in the area term.
    pub(super) fn generic_total_peak_bytes(&self, geometry: MemoryGeometry) -> u64 {
        self.asset_bytes
            .saturating_add(self.generic_headroom_bytes(geometry))
    }

    /// The non-weights slice of [`Self::generic_total_peak_bytes`]: the fixed OS/app reserve plus
    /// the area-scaled activation transient. Factored out (sc-18096) so the estimate-floor
    /// candidates charge the exact same headroom convention as the resident baseline — same fixed
    /// slice, same anchor, same area law — with only their weights term differing per rung.
    pub(super) fn generic_headroom_bytes(&self, geometry: MemoryGeometry) -> u64 {
        let megapixel_scale =
            (f64::from(geometry.width) * f64::from(geometry.height) / (1024.0 * 1024.0)).max(1.0);
        let request_scale = megapixel_scale * f64::from(geometry.batch.max(1));
        let fixed_reserve_bytes = self.fixed_reserve_bytes.min(self.activation_headroom_bytes);
        let area_bytes = self.activation_headroom_bytes - fixed_reserve_bytes;
        let baseline = fixed_reserve_bytes.saturating_add(
            (area_bytes as f64 * request_scale)
                .round()
                .clamp(0.0, u64::MAX as f64) as u64,
        );
        let extra =
            flux3_cfg_additional_workspaces(self.engine_id, geometry.width, geometry.height);
        baseline.saturating_add(extra.0.max(extra.1))
    }

    /// Convert the legacy whole-spec estimate into the base-only scalar expected by the additive
    /// component contract. `spec_component_bytes` folds raw control and resident adapter sources
    /// into the legacy scalar. A typed declaration replaces the corresponding raw source bytes with
    /// the provider's load-exact residency. Non-adopting providers retain the legacy scalar.
    ///
    /// A [`gen_core::MemoryComponentResidency::PrecomputedThenEvicted`] declaration deliberately
    /// does NOT move this figure (sc-19721). This is the resident BASELINE's peak: it is the one
    /// candidate that must always cover the widest instant of the whole pipeline, and the precompute
    /// instant — where the evictable sub-stack is fully materialized — is inside it. The eviction is
    /// a steady-state fact and reaches the floor candidates through
    /// [`estimate_floor_weights_bytes`], never this leg.
    pub(super) fn contract_base_peak_bytes(
        &self,
        legacy_total_peak_bytes: u64,
        contract: &MemoryProviderContract,
    ) -> u64 {
        let declares_control_branch = contract
            .resident_components()
            .iter()
            .any(|component| component.kind == gen_core::MemoryComponentKind::ControlBranch);
        let declares_adapter_stack = contract
            .resident_components()
            .iter()
            .any(|component| component.kind == gen_core::MemoryComponentKind::AdapterStack);
        // Mage's request estimator is provider-specific and already returns the adapter-free base
        // peak, unlike the generic legacy estimator whose asset term contains external adapter
        // files. Do not normalize bytes that were never present or the provider's later additive
        // declaration would cancel to zero.
        let legacy_includes_adapter_sources = !self.engine_id.starts_with("mage_flow");
        legacy_total_peak_bytes
            .saturating_sub(if declares_control_branch {
                self.folded_control_bytes
            } else {
                0
            })
            .saturating_sub(
                if declares_adapter_stack && legacy_includes_adapter_sources {
                    self.folded_adapter_bytes
                } else {
                    0
                },
            )
    }
}

#[cfg(native_lane)]
pub(super) fn declared_component_floors(
    engine_id: &str,
) -> &'static [gen_core::ComponentPrecisionFloor] {
    crate::inference_runtime::media_descriptor(engine_id)
        .map(|descriptor| descriptor.capabilities.component_precision_floors)
        .unwrap_or(&[])
}

#[cfg(not(native_lane))]
pub(super) fn declared_component_floors(_: &str) -> &'static [gen_core::ComponentPrecisionFloor] {
    // The platform-neutral worker build has no active media registry. Its fit-gate decision remains
    // usable for pure contract tests, but there is no provider declaration to attach to the tier.
    &[]
}

impl MlxCalibrationBinding {
    /// Read the optional closed collection of exact-cell MLX bindings. A model may carry many tiers
    /// and many request cells per tier; request-time routing selects exactly one by resolved source
    /// plus mode/overlay/geometry. Duplicate selectors are rejected here as ambiguous opt-in.
    pub(super) fn from_manifest(
        manifest: &JsonObject<String, Value>,
    ) -> Result<Option<Vec<Self>>, String> {
        let Some(calibrations) = manifest.get("mlx").and_then(|mlx| mlx.get("calibrations")) else {
            return Ok(None);
        };
        let calibrations = calibrations
            .as_array()
            .filter(|items| !items.is_empty())
            .ok_or_else(|| "mlx.calibrations must be a non-empty array".to_owned())?;
        let mut bindings = Vec::with_capacity(calibrations.len());
        for (index, calibration) in calibrations.iter().enumerate() {
            bindings.push(Self::parse(calibration, index)?);
        }
        for left in 0..bindings.len() {
            for right in (left + 1)..bindings.len() {
                if bindings[left].same_selector(&bindings[right]) {
                    return Err(format!(
                        "mlx.calibrations[{left}] and mlx.calibrations[{right}] are ambiguous duplicates"
                    ));
                }
            }
        }
        Ok(Some(bindings))
    }

    pub(super) fn parse(calibration: &Value, index: usize) -> Result<Self, String> {
        const CALIBRATION_FIELDS: [&str; 19] = [
            "abi",
            "loadShape",
            "fingerprint",
            "sceneWorksRevision",
            "matrixSourceRevision",
            "inferenceRevision",
            "inferenceClosureDigest",
            "provider",
            "tier",
            "mode",
            "overlay",
            "geometry",
            "artifactRepository",
            "artifactResolvedRevision",
            "artifactVariant",
            "resolvedPathFingerprint",
            "rung",
            "engagedRungs",
            "parameters",
        ];
        let calibration = calibration
            .as_object()
            .ok_or_else(|| format!("mlx.calibrations[{index}] must be an object"))?;
        if let Some(field) = calibration
            .keys()
            .find(|field| !CALIBRATION_FIELDS.contains(&field.as_str()))
        {
            return Err(format!(
                "mlx.calibrations[{index}] contains unknown field {field:?}"
            ));
        }
        let text = |name| {
            calibration
                .get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    format!("mlx.calibrations[{index}].{name} must be a non-empty string")
                })
        };
        let rung = match text("rung")?.as_str() {
            "resident" => StrategyRung::Resident,
            "staged_residency" => StrategyRung::StagedResidency,
            "bounded_decode" => StrategyRung::BoundedDecode,
            "bounded_attention" => StrategyRung::BoundedAttention,
            "bounded_transformer_residency" => StrategyRung::BoundedTransformerResidency,
            other => {
                return Err(format!(
                    "unsupported mlx.calibrations[{index}].rung {other:?}"
                ))
            }
        };
        let parameters = calibration
            .get("parameters")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| format!("mlx.calibrations[{index}].parameters must be an object"))?;
        let declared_engaged = match calibration.get("engagedRungs") {
            None => None,
            Some(Value::Array(values)) => Some(
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .and_then(crate::memory_strategy::rung_from_key)
                            .ok_or_else(|| {
                                format!(
                                    "mlx.calibrations[{index}].engagedRungs contains an unsupported rung {value}"
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Some(_) => {
                return Err(format!(
                    "mlx.calibrations[{index}].engagedRungs must be an array"
                ))
            }
        };
        let engaged_rungs =
            crate::memory_strategy::engaged_composition(rung, declared_engaged.as_deref())
                .map_err(|reason| format!("mlx.calibrations[{index}]: {reason}"))?;
        let geometry = calibration
            .get("geometry")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("mlx.calibrations[{index}].geometry must be an object"))?;
        const GEOMETRY_FIELDS: [&str; 4] = ["width", "height", "batch", "frames"];
        if let Some(field) = geometry
            .keys()
            .find(|field| !GEOMETRY_FIELDS.contains(&field.as_str()))
        {
            return Err(format!(
                "mlx.calibrations[{index}].geometry contains unknown field {field:?}"
            ));
        }
        let geometry_value = |name| {
            geometry
                .get(name)
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    format!("mlx.calibrations[{index}].geometry.{name} must be a positive u32")
                })
        };
        let abi = calibration
            .get("abi")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("mlx.calibrations[{index}].abi must be a u32"))?;
        // ABI 2 keys receipts by the typed materialization shape, so the opt-in must declare the
        // shape its receipts were measured under. ABI-1 opt-ins predate the axis; they are stale on
        // the ABI check alone, so the placeholder value is never compared.
        let load_shape = match calibration.get("loadShape").and_then(Value::as_str) {
            Some("eager_materialization") => LoadShapeKey::EagerMaterialization,
            Some("deferred_materialization") => LoadShapeKey::DeferredMaterialization,
            Some(other) => {
                return Err(format!(
                    "mlx.calibrations[{index}].loadShape {other:?} is not a known materialization shape"
                ))
            }
            None if abi >= 2 => {
                return Err(format!(
                    "mlx.calibrations[{index}].loadShape is required at calibration ABI {abi}"
                ))
            }
            None => LoadShapeKey::EagerMaterialization,
        };
        // The selected rung's DECLARED prerequisite graph is deliberately not re-checked here. At
        // the pinned contract revision rung 4's shared prerequisite is
        // `LoadShape::DeferredMaterialization`, and that axis is already owned by
        // `EvidenceBundle::evidence_for`, which degrades a load-shape mismatch to
        // `EvidenceMismatchReason::LoadShape` and the legacy selector rather than rejecting the opt-in.
        // The rung-1 edge some providers add for rung 4 is realization-specific
        // (`MemoryProviderContract::additional_prerequisites`) and is enforced by
        // `validate_selection` against the provider contract, which no manifest reader holds.
        let selection_parameters = parse_evidence_parameters(rung, &engaged_rungs, &parameters)
            .map_err(|reason| format!("mlx.calibrations[{index}]: {reason}"))?;
        Ok(Self {
            query: CalibrationBinding {
                abi,
                load_shape,
                fingerprint: text("fingerprint")?,
                scene_works_revision: text("sceneWorksRevision")?,
                matrix_source_revision: text("matrixSourceRevision")?,
                inference_revision: text("inferenceRevision")?,
                inference_closure_digest: text("inferenceClosureDigest")?,
                artifact_repository: text("artifactRepository")?,
                artifact_resolved_revision: text("artifactResolvedRevision")?,
                artifact_variant: text("artifactVariant")?,
                resolved_path_fingerprint: text("resolvedPathFingerprint")?,
            },
            provider: text("provider")?,
            tier: text("tier")?,
            mode: text("mode")?,
            overlay: text("overlay")?,
            geometry: CalibrationGeometry {
                width: geometry_value("width")?,
                height: geometry_value("height")?,
                batch: geometry_value("batch")?,
                frames: geometry_value("frames")?,
            },
            rung,
            parameters,
            selection_parameters,
        })
    }

    pub(super) fn same_selector(&self, other: &Self) -> bool {
        self.query == other.query
            && self.provider == other.provider
            && self.tier == other.tier
            && self.mode == other.mode
            && self.overlay == other.overlay
            && self.geometry == other.geometry
            && self.rung == other.rung
            && self.parameters == other.parameters
    }
}

pub(super) fn numeric_tier_for_resolved(
    tier: &str,
    fallback: MemoryNumericTier,
    declared_floors: &'static [gen_core::ComponentPrecisionFloor],
) -> Result<MemoryNumericTier, String> {
    let resolved = match tier {
        "q4" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q4),
            component_precision_floors: active_component_floors(
                declared_floors,
                Some(gen_core::Quant::Q4),
            ),
        },
        "q8" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Q8),
            component_precision_floors: &[],
        },
        "nvfp4" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(gen_core::Quant::Nvfp4),
            component_precision_floors: &[],
        },
        "bf16" => MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        },
        "fp32" => MemoryNumericTier {
            precision: gen_core::Precision::Fp32,
            quant: None,
            component_precision_floors: &[],
        },
        other => return Err(format!("unsupported resolver-supplied MLX tier {other:?}")),
    };
    if fallback.quant.is_some() && fallback != resolved {
        return Err(format!(
            "resolver tier {tier:?} conflicts with the LoadSpec numeric tier"
        ));
    }
    Ok(resolved)
}

pub(super) fn active_component_floors(
    declared: &'static [gen_core::ComponentPrecisionFloor],
    selected: Option<gen_core::Quant>,
) -> &'static [gen_core::ComponentPrecisionFloor] {
    sceneworks_core::memory_contracts::active_component_floors(declared, selected)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn plan_tier_key(tier: MemoryNumericTier) -> &'static str {
    match tier.quant {
        Some(gen_core::Quant::Q4) => "q4",
        Some(gen_core::Quant::Q8) => "q8",
        Some(gen_core::Quant::Nvfp4) => "nvfp4",
        Some(gen_core::Quant::Fp8Rowwise) => "fp8r",
        None if tier.precision == gen_core::Precision::Fp32 => "fp32",
        None => "bf16",
    }
}
