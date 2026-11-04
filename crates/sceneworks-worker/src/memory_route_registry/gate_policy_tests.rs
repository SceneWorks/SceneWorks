use super::*;
#[test]
fn family_policy_is_closed_and_preserves_backend_route_identities() {
    for (id, family) in [
        ("ideogram_4_turbo", MemoryGateFamily::Ideogram),
        ("flux1_dev_control", MemoryGateFamily::Flux1),
        ("flux3_klein_base", MemoryGateFamily::Flux3),
        ("candle_kolors_control", MemoryGateFamily::KolorsBespoke),
        ("krea_2_turbo_edit", MemoryGateFamily::Krea),
        ("krea_2_turbo_control", MemoryGateFamily::KreaControl),
    ] {
        assert_eq!(gate_family(id), Some(family));
    }
    for unknown in ["", "future_ideogram", "ideogram_5", "unknown"] {
        assert_eq!(gate_family(unknown), None);
    }
}
#[test]
fn every_registered_image_route_retains_gate_policy_without_a_parallel_family_list() {
    for backend in [MemoryRouteBackend::Mlx, MemoryRouteBackend::Candle] {
        assert!(has_gate_policy(backend, "krea_2_turbo_control"));
    }
    for row in crate::engines::MODEL_TABLE {
        assert!(
            has_gate_policy(MemoryRouteBackend::Mlx, row.engine_id),
            "{}",
            row.engine_id
        );
        assert!(
            has_gate_policy(MemoryRouteBackend::Candle, row.engine_id),
            "{}",
            row.engine_id
        );
    }
    #[cfg(target_os = "macos")]
    assert!(
        has_gate_policy(MemoryRouteBackend::Mlx, "qwen_image_control"),
        "registered control contracts share the known base family without a duplicate id list"
    );
    assert!(!has_gate_policy(
        MemoryRouteBackend::Mlx,
        "unregistered_future_provider"
    ));
    assert!(!has_gate_policy(
        MemoryRouteBackend::Candle,
        "unregistered_future_provider"
    ));
}
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn unknown_candle_provider_cannot_inherit_declaration_revision() {
    assert_eq!(candle_gate_evidence_revision("unknown"), None);
    assert_eq!(
        candle_gate_evidence_revision("ideogram_4"),
        Some(IDEOGRAM_REQUEST_EVIDENCE_REVISION)
    );
    assert_eq!(
        candle_gate_evidence_revision("sdxl"),
        Some(SDXL_REQUEST_EVIDENCE_REVISION)
    );
}
