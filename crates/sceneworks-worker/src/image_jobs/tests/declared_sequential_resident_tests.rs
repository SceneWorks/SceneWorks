use super::{admission_needs_no_staging, declared_sequential_resident_alternative};
use gen_core::{LoadShapeDeclarationResult, LoadSpec, OffloadPolicy, WeightsSource};

const GIB: u64 = 1024 * 1024 * 1024;

/// Sparse stand-ins for a Klein-sized bf16 artifact: the fit gate reads file lengths only.
fn klein_sized_dir() -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("temp dir");
    for (component, bytes) in [("transformer", 18 * GIB), ("text_encoder", 14 * GIB)] {
        let dir = root.path().join(component);
        std::fs::create_dir_all(&dir).expect("component dir");
        std::fs::File::create(dir.join("model.safetensors"))
            .and_then(|file| file.set_len(bytes))
            .expect("sparse weights");
    }
    root
}

fn declared_sequential(before: &LoadSpec) -> LoadSpec {
    before
        .clone()
        .with_applied_load_shape_declaration()
        .with_offload_policy(OffloadPolicy::Sequential)
}

#[test]
fn declared_sequential_is_released_only_when_the_eager_load_fits_resident() {
    let root = klein_sized_dir();
    let before = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()))
        .with_resolved_route("flux2_klein_9b_true_v2");
    let after = declared_sequential(&before);
    let cap = crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV;

    let resident = crate::test_env::temp_env_var(cap, "128", || {
        declared_sequential_resident_alternative("flux2_klein_9b", &before, &after)
    })
    .expect("a 32 GiB pipeline fits a 128 GiB host resident");
    assert_eq!(resident.offload_policy, OffloadPolicy::Resident);
    assert_eq!(resident.load_shape, before.load_shape);
    assert_eq!(
        resident.load_shape_declaration_result,
        LoadShapeDeclarationResult::Eligible
    );

    // A host where only the staged peak fits keeps the declared Sequential spec.
    assert!(crate::test_env::temp_env_var(cap, "32", || {
        declared_sequential_resident_alternative("flux2_klein_9b", &before, &after)
    })
    .is_none());
    // Sequential chosen before the declaration step, or never chosen, is not ours to release.
    let already = before
        .clone()
        .with_offload_policy(OffloadPolicy::Sequential);
    assert!(declared_sequential_resident_alternative(
        "flux2_klein_9b",
        &already,
        &declared_sequential(&already)
    )
    .is_none());
    assert!(declared_sequential_resident_alternative("flux2_klein_9b", &before, &before).is_none());
}

/// The text from `declaration` to the end of its body; these lanes load real MLX weights, so
/// their wiring is what can be asserted without a GPU.
fn function_body(source: &'static str, declaration: &str) -> &'static str {
    let body = &source[source.find(declaration).expect(declaration)..];
    let mut depth = 0_usize;
    for (offset, byte) in body.bytes().enumerate().skip(body.find('{').expect("body")) {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &body[..=offset];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced body for {declaration}");
}

#[test]
fn every_lane_that_binds_a_declared_policy_offers_the_resident_release() {
    // FLUX.1 control and PuLID compile their render plans on the owner before the load (SW-6936).
    let lanes = [
        (
            include_str!("../flux2.rs"),
            "async fn generate_flux2_edit_stream(",
            "start_cached_gen_stream_with_request_state(",
        ),
        (
            include_str!("../flux2.rs"),
            "async fn generate_flux2_dev_control_stream(",
            "start_cached_gen_stream_with_request_state(",
        ),
        (
            include_str!("../flux1_control.rs"),
            "async fn generate_flux1_dev_control_stream(",
            "start_cached_gen_stream_with_request_state_using_prepared(",
        ),
        (
            include_str!("../pulid.rs"),
            "async fn generate_pulid_flux_stream(",
            "start_cached_gen_stream_with_request_state_using_prepared(",
        ),
    ];
    for (source, declaration, handoff) in lanes {
        let body = function_body(source, declaration);
        let before = body
            .find("let pre_policy_spec = spec.clone();")
            .expect(declaration);
        let shaped = body
            .find("evaluate_declared_mlx_load_shape_for_request(")
            .expect(declaration);
        let release = body
            .find("release_declared_sequential_for_request(")
            .expect(declaration);
        let load = body.find(handoff).expect(declaration);
        assert!(
            before < shaped && shaped < release && release < load,
            "{declaration}"
        );
    }
    let base = function_body(include_str!("../mlx.rs"), "async fn generate_stream_using(");
    let prepared = base
        .find("prepare_mlx_load_policy(")
        .expect("base policy step");
    let alternative = base
        .find("declared_sequential_resident_alternative(engine_id, &pre_policy_spec, &spec)")
        .expect("base release alternative");
    assert!(base[..prepared].contains("let pre_policy_spec = spec.clone();"));
    assert!(prepared < alternative);
    assert!(
        base.contains("prepare(dir, true)"),
        "base tier chooser never re-prepares Resident"
    );
}

#[test]
fn rejected_admission_never_counts_as_needing_no_staging() {
    let rejected = crate::mlx_fit_gate::MlxRequestAdmission::Rejected(
        crate::WorkerError::InvalidPayload("too big".to_owned()),
    );
    assert!(!admission_needs_no_staging(&rejected));
}
