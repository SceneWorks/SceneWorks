use super::*;
use runtime_cuda::gen_core::{Error, Modality, PrecisionFloorComponent};
use std::cell::Cell;

/// A text-encoder-above-tier floor, the shape `candle-gen-mage` declares for q4.
const Q4_TEXT_ENCODER_FLOOR: &[ComponentPrecisionFloor] = &[ComponentPrecisionFloor {
    component: PrecisionFloorComponent::TextEncoder,
    selected_tier: Quant::Q4,
    resident_tier: Quant::Q8,
}];

fn descriptor(floors: &'static [ComponentPrecisionFloor]) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        id: "sc22738-stub",
        family: "sc22738",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            component_precision_floors: floors,
            ..Capabilities::default()
        },
        required_components: &[],
        control_kinds: None,
    }
}

fn context(strategy: MemoryStrategy) -> MemoryRunContext {
    MemoryRunContext {
        selection: MemorySelection {
            strategy,
            parameters: MemoryStrategyParameters::default(),
            tier: numeric_tier("bf16").unwrap(),
        },
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: runtime_cuda::gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: "sc22738-stub-v1".to_owned(),
        load_shape: LoadShape::EagerMaterialization,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        reference_resources: None,
        overlay: None,
        budget: MemoryBudget {
            total_bytes: 1 << 34,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "sc22738-stub".to_owned(),
    }
}

/// A request scope that accepts every lifecycle call.
struct AcceptingScope;

impl MemoryRequestScope for AcceptingScope {
    fn configure_request(
        &mut self,
        _: &mut GenerationRequest,
    ) -> runtime_cuda::gen_core::Result<()> {
        Ok(())
    }
    fn enter_phase(&mut self, _: MemoryPhase) -> runtime_cuda::gen_core::Result<()> {
        Ok(())
    }
    fn leave_phase(&mut self, _: MemoryPhase) -> runtime_cuda::gen_core::Result<()> {
        Ok(())
    }
    fn configure_decode(
        &mut self,
        _: u32,
        _: u32,
        _: MemoryGeometry,
    ) -> runtime_cuda::gen_core::Result<()> {
        Ok(())
    }
    fn configure_attention(&mut self, _: u32) -> runtime_cuda::gen_core::Result<()> {
        Ok(())
    }
    fn materialize_transformer_window(
        &mut self,
        _: u32,
        _: u32,
    ) -> runtime_cuda::gen_core::Result<()> {
        Ok(())
    }
    fn finish(&mut self, _: MemoryRunOutcome) -> runtime_cuda::gen_core::Result<()> {
        Ok(())
    }
}

/// The engine-side admission registry in miniature: `candle-gen-chroma`, `-flux2` and `-mage`
/// each take an approval INSIDE `memory_strategy_safety_check` and consume it in
/// `begin_memory_strategy_request`, refusing a `begin` with none on file by the exact sentence
/// the campaign logged.
struct HandshakeStub {
    descriptor: ModelDescriptor,
    approved: Cell<bool>,
    /// `false` models `candle-gen-minimax-h3`: no override of either seam, so `begin` yields
    /// `Ok(None)` for its resident-only contract.
    opens_scope: bool,
    /// `Some` models a provider that loaded WITHOUT a contract and carries the reason
    /// (`candle-gen-bernini`'s `memory_refusal`).
    refusal: Option<&'static str>,
    safety_checks: Cell<u32>,
}

impl HandshakeStub {
    fn new(opens_scope: bool) -> Self {
        Self {
            descriptor: descriptor(&[]),
            approved: Cell::new(false),
            opens_scope,
            refusal: None,
            safety_checks: Cell::new(0),
        }
    }
}

impl Generator for HandshakeStub {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, _: &GenerationRequest) -> runtime_cuda::gen_core::Result<()> {
        Ok(())
    }

    fn generate(
        &self,
        _: &GenerationRequest,
        _: &mut dyn FnMut(Progress),
    ) -> runtime_cuda::gen_core::Result<GenerationOutput> {
        Err(Error::Unsupported("the stub renders nothing".to_owned()))
    }

    fn memory_strategy_safety_check(&self, _: &MemoryRunContext) -> MemorySafetyDecision {
        self.safety_checks.set(self.safety_checks.get() + 1);
        if let Some(reason) = self.refusal {
            return MemorySafetyDecision::Reject {
                reason: reason.to_owned(),
            };
        }
        self.approved.set(true);
        MemorySafetyDecision::Accept
    }

    fn begin_memory_strategy_request(
        &self,
        _: &MemoryRunContext,
    ) -> runtime_cuda::gen_core::Result<Option<Box<dyn MemoryRequestScope + '_>>> {
        if !self.approved.replace(false) {
            return Err(Error::Unsupported(
                "sc22738-stub: memory request skipped the safety handshake".to_owned(),
            ));
        }
        Ok(self
            .opens_scope
            .then(|| Box::new(AcceptingScope) as Box<dyn MemoryRequestScope>))
    }
}

/// Class 1 (chroma1_*, flux2_*): the arm performs the provider's safety handshake before
/// `begin`, so a registry-guarded provider opens its scope instead of refusing.
#[test]
fn the_capture_admits_through_the_safety_check_before_begin() {
    let stub = HandshakeStub::new(true);
    let mut scope = open_memory_request_scope(
        &stub,
        &context(MemoryStrategy::StagedResidency),
        "sc22738 stub",
    )
    .expect("an approved begin opens the scope");
    assert_eq!(
        stub.safety_checks.get(),
        1,
        "the handshake ran exactly once"
    );
    assert!(
        scope.scope.is_some(),
        "the provider's scope is the one the capture drives"
    );
    scope.finish(MemoryRunOutcome::Complete).unwrap();
    // Direct proof that the stub's guard is real: a bare `begin` on the same provider — the
    // sequence the arm ran before this fix — is refused by the campaign's exact sentence.
    let error = stub
        .begin_memory_strategy_request(&context(MemoryStrategy::StagedResidency))
        .err()
        .expect("no approval on file");
    assert!(
        error.to_string().contains("skipped the safety handshake"),
        "{error}"
    );
}

/// A provider that refuses the context is refused by ITS reason, before any `begin`.
#[test]
fn a_rejected_safety_check_names_the_providers_reason() {
    let mut stub = HandshakeStub::new(true);
    stub.refusal = Some("bernini: no sealed memory receipt for the loaded artifact (x)");
    let error =
        open_memory_request_scope(&stub, &context(MemoryStrategy::Resident), "sc22738 stub")
            .err()
            .expect("a rejected context never reaches begin");
    assert!(error.contains("no sealed memory receipt"), "{error}");
}

/// Class 3 (minimax_h3, minimax_h3_ref): a resident selection on a provider that opens no
/// scope renders without one — the worker's `generate_with_scope` — and the lifecycle calls
/// the capture makes around the render are no-ops rather than failures.
#[test]
fn a_resident_selection_without_a_provider_scope_is_rendered_scopeless() {
    let stub = HandshakeStub::new(false);
    let mut scope = open_memory_request_scope(
        &stub,
        &context(MemoryStrategy::Resident),
        "minimax_h3 capture",
    )
    .expect("a resident selection needs no scope");
    assert!(scope.scope.is_none());
    let mut generation = GenerationRequest::default();
    scope.configure_request(&mut generation).unwrap();
    scope.enter_phase(MemoryPhase::Conditioning).unwrap();
    scope.leave_phase(MemoryPhase::Conditioning).unwrap();
    scope.finish(MemoryRunOutcome::Complete).unwrap();
    // A parameter with no scope to receive it is refused, not dropped.
    let error = scope
        .configure_decode(256, 32, context(MemoryStrategy::Resident).geometry)
        .expect_err("no scope can take a decode tile");
    assert!(
        error.to_string().contains("opened no request scope"),
        "{error}"
    );
}

/// …while an OPTIMIZED selection that opened no scope is the state the worker refuses by name.
#[test]
fn an_optimized_selection_without_a_provider_scope_is_refused() {
    let stub = HandshakeStub::new(false);
    let error = open_memory_request_scope(
        &stub,
        &context(MemoryStrategy::StagedResidency),
        "sc22738 stub",
    )
    .err()
    .expect("an optimized rung needs a scope");
    assert!(error.contains("without opening a request scope"), "{error}");
}

/// Class 4 (bernini q4/q8, bernini_image ×3): a provider that loaded with no contract answers
/// the arm's probe with its own refusal, so the record names the cause instead of the symptom.
#[test]
fn a_missing_contract_carries_the_providers_refusal() {
    let mut stub = HandshakeStub::new(true);
    stub.refusal = Some("bernini: no sealed memory receipt for the loaded artifact (dense)");
    let request = json!({ "planned": { "target": { "tier": "q4" } } });
    let reason = missing_contract_reason(&stub, &request);
    assert!(reason.contains("no sealed memory receipt"), "{reason}");
    let accepting = HandshakeStub::new(true);
    assert_eq!(
        missing_contract_reason(&accepting, &request),
        "no memory-strategy contract"
    );
}

/// Class 2 (mage_flow* q4): the selected tier carries the floors the loaded provider declares
/// for the planned quant — the worker's `active_component_floors` — and none otherwise.
#[test]
fn the_selected_tier_carries_the_loaded_providers_component_floors() {
    let mage_like = descriptor(Q4_TEXT_ENCODER_FLOOR);
    assert_eq!(
        loaded_numeric_tier("q4", &mage_like).unwrap(),
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: Q4_TEXT_ENCODER_FLOOR,
        }
    );
    // The declared table names q4; q8 and the dense base carry no active floor.
    assert!(loaded_numeric_tier("q8", &mage_like)
        .unwrap()
        .component_precision_floors
        .is_empty());
    assert!(loaded_numeric_tier("bf16", &mage_like)
        .unwrap()
        .component_precision_floors
        .is_empty());
    // A provider that declares none selects exactly what `numeric_tier` selects.
    assert_eq!(
        loaded_numeric_tier("q4", &descriptor(&[])).unwrap(),
        numeric_tier("q4").unwrap()
    );
    // …and the selection the arms hand the provider is built from that tier.
    let request = json!({ "planned": {
        "target": { "tier": "q4" },
        "strategy": { "rung": "resident", "parameters": {} },
    }});
    assert_eq!(
        planned_selection_for(&request, &mage_like)
            .unwrap()
            .tier
            .component_precision_floors,
        Q4_TEXT_ENCODER_FLOOR
    );
}

/// Class 4: both Bernini arms hand `candle-gen-bernini` the worker's spec — eager, the packed
/// tier's quant bound, the catalog entry as the resolved route.
#[test]
fn the_bernini_arms_bind_the_workers_load_spec() {
    for (tier, quant) in [
        ("bf16", None),
        ("q4", Some(Quant::Q4)),
        ("q8", Some(Quant::Q8)),
    ] {
        let root = PathBuf::from("/nonexistent/bernini").join(tier);
        // Video arm.
        let video = bernini_candle_load_spec(
            LoadSpec::new(WeightsSource::Dir(root.clone()))
                .with_offload_policy(OffloadPolicy::Sequential),
            tier,
            BERNINI_CANDLE_VIDEO_MODEL_ID,
        )
        .unwrap();
        assert_eq!(video.quantize, quant, "video {tier}");
        assert_eq!(
            video.load_shape,
            LoadShape::EagerMaterialization,
            "video {tier}"
        );
        assert_eq!(
            video.resolved_route.as_deref(),
            Some(BERNINI_CANDLE_VIDEO_MODEL_ID),
            "video {tier}"
        );
        // Still arm, through the shared five-rung spec builder.
        let request = json!({ "planned": {
            "backend": "candle",
            "target": {
                "provider": BERNINI_CANDLE_ID,
                "modelId": BERNINI_CANDLE_IMAGE_MODEL_ID,
                "tier": tier,
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 848, "height": 848, "batch": 1, "frames": 1 },
            },
            "loadShape": "eager_materialization",
            "strategy": { "rung": "resident", "parameters": {} },
            "calibrationFingerprint": format!("bernini-image-{tier}-candle-dual-expert-ladder-v1"),
            "fixture": format!("fresh-five-rung-bernini-image-{tier}-848-seed16402-step2"),
        }});
        let still = five_rung_load_spec(&request, BERNINI_CANDLE_ID, tier, root).unwrap();
        assert_eq!(still.quantize, quant, "still {tier}");
        assert_eq!(
            still.load_shape,
            LoadShape::EagerMaterialization,
            "still {tier}"
        );
        assert_eq!(
            still.resolved_route.as_deref(),
            Some(BERNINI_CANDLE_IMAGE_MODEL_ID),
            "still {tier}"
        );
    }
}
