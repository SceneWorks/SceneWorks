use super::super::memory::optimized_shared_memory_context;
use super::super::memory::shared_image_reference_count;
use super::super::LoadSpec;
use super::super::WeightsSource;
use super::apply_candle_image_load_shape;
use std::sync::{Arc, Mutex};

fn fixture_spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
        "qwen-image-fixture",
    )))
}

fn legacy_shape(engine_id: &str, spec: LoadSpec) -> LoadSpec {
    apply_candle_image_load_shape(engine_id, spec)
}

/// `qwen_image_edit` is deliberately absent from this list. SC-20790 moved its Candle rule off
/// `legacy_shaping` onto the declaration-owned shaper, and `apply_candle_image_load_shape`
/// consults ONLY legacy rules — so the legacy shaper returning the spec untouched is now the
/// correct answer for that route, not a regression. The route is not left uncovered: the arm
/// below pins the two facts that make the removal accountable, so a route cannot fall off this
/// list by accident. The declaration-owned shape itself is exercised at the qwen edit call
/// site, which is the only place that assembles the exact tier/mode/source coordinate the
/// declared shaper requires.
#[test]
fn adopting_native_image_routes_use_deferred_materialization() {
    for engine_id in [
        "qwen_image",
        "flux1_schnell",
        "flux1_dev",
        "flux2_dev",
        "flux2_klein_9b",
        "mage_flow_base",
        "mage_flow",
        "mage_flow_turbo",
        "mage_flow_edit_base",
        "mage_flow_edit",
        "mage_flow_edit_turbo",
    ] {
        assert_eq!(
            legacy_shape(engine_id, fixture_spec()).load_shape,
            gen_core::LoadShape::DeferredMaterialization
        );
    }
    // Qwen edit's own accounting. Note that being declaration-owned is NOT what excludes it —
    // several routes above (the Mage six, `flux2_dev`) carry a declared rule alongside their
    // legacy one and still shape here. What excludes qwen edit is that its legacy Candle rule
    // was REPLACED rather than supplemented, so the legacy shaper finds nothing and correctly
    // returns the spec untouched; its production call site reaches the declared shaper instead.
    assert!(
        crate::memory_route_registry::candle_declaration_owns_load_shape("qwen_image_edit"),
        "if qwen edit ever returns to legacy shaping it belongs back in the list above"
    );
    assert_eq!(
        legacy_shape("qwen_image_edit", fixture_spec()).load_shape,
        fixture_spec().load_shape,
        "no legacy Candle rule remains for qwen edit, so the legacy shaper is a no-op for it"
    );
    assert!(include_str!("../qwen_edit_candle.rs")
        .contains("apply_candle_qwen_load_shape(\n            QWEN_EDIT_PROVIDER_ID,"));
}

#[test]
fn unsupported_overlays_and_non_streamable_routes_remain_eager() {
    let adapter = gen_core::AdapterSpec::new(
        std::path::PathBuf::from("adapter.safetensors"),
        1.0,
        gen_core::AdapterKind::Lora,
    );
    let cases = [
        ("qwen_image", fixture_spec().with_adapters(vec![adapter])),
        (
            "qwen_image_edit",
            fixture_spec().with_pid(
                WeightsSource::File(std::path::PathBuf::from("pid.safetensors")),
                WeightsSource::Dir(std::path::PathBuf::from("gemma")),
            ),
        ),
        (
            "mage_flow_edit",
            fixture_spec().with_pid(
                WeightsSource::File(std::path::PathBuf::from("pid.safetensors")),
                WeightsSource::Dir(std::path::PathBuf::from("gemma")),
            ),
        ),
        ("z_image", fixture_spec()),
    ];
    for (engine_id, spec) in cases {
        assert_eq!(
            legacy_shape(engine_id, spec).load_shape,
            gen_core::LoadShape::EagerMaterialization
        );
    }
}

#[test]
fn shared_reference_count_preserves_mage_multi_reference_geometry() {
    assert_eq!(shared_image_reference_count(0, false, false, false), 0);
    assert_eq!(shared_image_reference_count(0, true, false, false), 1);
    assert_eq!(shared_image_reference_count(1, true, false, false), 1);
    assert_eq!(shared_image_reference_count(8, true, false, false), 8);
    assert_eq!(shared_image_reference_count(0, true, true, false), 2);
    assert_eq!(shared_image_reference_count(8, false, true, false), 9);
    assert_eq!(shared_image_reference_count(8, true, true, true), 1);
}

#[test]
fn flux_single_control_or_ip_overlay_keeps_deferred_materialization() {
    for spec in [
        fixture_spec().with_control(WeightsSource::File(std::path::PathBuf::from(
            "control.safetensors",
        ))),
        fixture_spec().with_ip_adapter(WeightsSource::Dir(std::path::PathBuf::from("ip"))),
    ] {
        assert_eq!(
            legacy_shape("flux1_dev", spec).load_shape,
            gen_core::LoadShape::DeferredMaterialization
        );
    }
    let combined = fixture_spec()
        .with_control(WeightsSource::File(std::path::PathBuf::from(
            "control.safetensors",
        )))
        .with_ip_adapter(WeightsSource::Dir(std::path::PathBuf::from("ip")));
    assert_eq!(
        legacy_shape("flux1_dev", combined).load_shape,
        gen_core::LoadShape::EagerMaterialization
    );
    assert_eq!(
        legacy_shape(
            "flux2_dev",
            fixture_spec().with_control(WeightsSource::File(std::path::PathBuf::from(
                "control.safetensors",
            )))
        )
        .load_shape,
        gen_core::LoadShape::DeferredMaterialization
    );
}

struct ResidentOverwriteScope;

impl gen_core::MemoryRequestScope for ResidentOverwriteScope {
    fn configure_request(
        &mut self,
        request: &mut gen_core::GenerationRequest,
    ) -> gen_core::Result<()> {
        request.memory = Some(gen_core::GenerationMemory::default());
        Ok(())
    }

    fn enter_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
        Ok(())
    }
    fn leave_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
        Ok(())
    }
    fn configure_decode(
        &mut self,
        _tile_edge: u32,
        _overlap: u32,
        _geometry: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        Ok(())
    }
    fn configure_attention(&mut self, _chunk_size: u32) -> gen_core::Result<()> {
        Ok(())
    }
    fn materialize_transformer_window(
        &mut self,
        _first_block: u32,
        _block_count: u32,
    ) -> gen_core::Result<()> {
        Ok(())
    }
    fn finish(&mut self, _outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
        Ok(())
    }
}

struct LegacyFallbackGenerator {
    observed_stage_residency: Arc<Mutex<Option<bool>>>,
    descriptor: gen_core::ModelDescriptor,
}

impl LegacyFallbackGenerator {
    fn new(observed_stage_residency: Arc<Mutex<Option<bool>>>) -> Self {
        Self {
            observed_stage_residency,
            descriptor: gen_core::ModelDescriptor {
                denoiser_output_latent_space: None,
                id: "legacy_fallback_fixture",
                family: "test",
                backend: "candle",
                modality: gen_core::Modality::Image,
                capabilities: Default::default(),
                encoder_contract: None,
                required_components: &[],
                control_kinds: None,
            },
        }
    }
}

impl gen_core::Generator for LegacyFallbackGenerator {
    fn descriptor(&self) -> &gen_core::ModelDescriptor {
        &self.descriptor
    }

    fn begin_memory_strategy_request(
        &self,
        _context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        Ok(Some(Box::new(ResidentOverwriteScope)))
    }

    fn validate(&self, _request: &gen_core::GenerationRequest) -> gen_core::Result<()> {
        Ok(())
    }

    fn generate(
        &self,
        request: &gen_core::GenerationRequest,
        _on_progress: &mut dyn FnMut(gen_core::Progress),
    ) -> gen_core::Result<gen_core::GenerationOutput> {
        *self.observed_stage_residency.lock().unwrap() =
            Some(request.memory.unwrap_or_default().stage_residency);
        Ok(gen_core::GenerationOutput::Images(Vec::new()))
    }
}

fn resident_context() -> gen_core::MemoryRunContext {
    gen_core::MemoryRunContext {
        selection: gen_core::MemorySelection {
            strategy: gen_core::MemoryStrategy::Resident,
            parameters: Default::default(),
            tier: gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        },
        optimization_authority: gen_core::MemoryOptimizationAuthority::Resident,
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: String::new(),
        load_shape: gen_core::LoadShape::DeferredMaterialization,
        mode: gen_core::MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: gen_core::MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        reference_resources: None,
        overlay: None,
        budget: gen_core::MemoryBudget {
            total_bytes: 24,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 20,
        cache_state: gen_core::MemoryCacheState::Cold,
        evidence_revision: "resident-sentinel".to_owned(),
    }
}

#[test]
fn resident_shared_context_cannot_overwrite_legacy_sequential_request() {
    let observed = Arc::new(Mutex::new(None));
    let generator = LegacyFallbackGenerator::new(Arc::clone(&observed));
    let mut request = gen_core::GenerationRequest {
        memory: Some(gen_core::GenerationMemory {
            stage_residency: true,
            ..Default::default()
        }),
        ..Default::default()
    };
    let context = optimized_shared_memory_context("legacy_provider", resident_context());
    assert!(
        context.is_none(),
        "Resident is a fallback sentinel, not an execution scope"
    );

    crate::memory_strategy::generate_with_scope(
        &generator,
        &mut request,
        context.as_ref(),
        &mut |_| {},
    )
    .expect("legacy fallback generation");

    assert_eq!(*observed.lock().unwrap(), Some(true));
    assert!(request.memory.unwrap().stage_residency);
}

#[test]
fn receipt_priced_resident_context_remains_request_scoped() {
    let context = resident_context();
    for provider in [
        "ideogram_4",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
    ] {
        assert_eq!(
            optimized_shared_memory_context(provider, context.clone()),
            Some(context.clone()),
            "{provider}"
        );
    }
}
