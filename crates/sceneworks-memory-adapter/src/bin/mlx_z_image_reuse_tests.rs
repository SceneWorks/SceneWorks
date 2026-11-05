use super::*;

#[test]
fn shape_independent_fingerprint_still_keeps_eager_and_deferred_loads_distinct() {
    let fingerprint = "z-image-mlx-independent-materialization-v4";
    assert_ne!(
        z_image_reuse_identity(fingerprint, LoadShape::EagerMaterialization),
        z_image_reuse_identity(fingerprint, LoadShape::DeferredMaterialization),
    );
}

/// A canonical five-rung batch, differing from a real Z-Image one ONLY in `target.provider`.
/// Every other check in `validate_z_image_batch` — length, canonical rung order, one exact
/// target tuple — passes on this input, which is precisely why the provider check has to exist.
fn foreign_five_rung_batch(provider: &str) -> Value {
    let target = json!({ "provider": provider, "tier": "q4", "mode": "text_to_image" });
    let planned: Vec<Value> = [
        "resident",
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
    ]
    .iter()
    .map(|rung| json!({ "target": target, "strategy": { "rung": rung } }))
    .collect();
    json!({ "action": "assess_batch", "planned": planned })
}

/// sc-18104: the batch action had the same silent-misroute hole `run` had. `validate_z_image_batch`
/// never read `target.provider`, and `assess_z_image_batch` hardcodes `Z_IMAGE_PROVIDER` when it
/// reads the contract — so a foreign five-rung batch was misrouted into the Z-Image contract and
/// failed on a Z-Image-shaped complaint AFTER `runtime_macos::catalog()` did real environment
/// work. Refusal must therefore be by name and must happen inside validation, before that call.
#[test]
fn the_batch_action_refuses_a_foreign_provider_by_name_during_validation() {
    for provider in ["flux2_dev", "qwen_image", "krea_2_turbo_control"] {
        let error = validate_z_image_batch(&foreign_five_rung_batch(provider))
            .expect_err("a foreign provider must not reach the Z-Image contract");
        assert_eq!(
            error,
            format!("MLX five-rung batch assessment does not implement provider {provider:?}")
        );
        assert!(
            !error.contains("fingerprint") && !error.contains("contract"),
            "refusal came from the Z-Image contract, so validation let it through: {error}"
        );
    }
}

/// The companion direction: the real Z-Image provider must still pass validation unchanged, so
/// the new check cannot be satisfied by refusing everything.
#[test]
fn the_batch_action_still_accepts_the_z_image_provider() {
    let batch = foreign_five_rung_batch("z_image_turbo");
    let planned = validate_z_image_batch(&batch).expect("the Z-Image batch must still validate");
    assert_eq!(planned.len(), 5);
}

/// The refusal must not be conditional on the batch happening to be five long. `mlx-gen-flux2`
/// implements only the resident rung, so `assess-reuse` on a flux2 fixture submits a ONE-element
/// batch — and if the length check ran first, that lane would be told
/// `Z-Image rung batch must contain exactly 5 cases`, a Z-Image-named complaint about a provider
/// that is not Z-Image. The provider check is hoisted above the length check for exactly this.
#[test]
fn a_short_foreign_batch_is_still_refused_by_name_not_by_length() {
    // 1 is the flux2 case specifically: one implemented rung, so one planned case.
    for length in [1, 2, 3, 4] {
        let mut batch = foreign_five_rung_batch("flux2_dev");
        batch["planned"].as_array_mut().unwrap().truncate(length);
        let error = validate_z_image_batch(&batch)
            .expect_err("a foreign batch of any length must be refused");
        assert_eq!(
            error,
            "MLX five-rung batch assessment does not implement provider \"flux2_dev\""
        );
        assert!(
            !error.contains("exactly"),
            "length {length} read as a Z-Image arity problem, not a foreign provider: {error}"
        );
    }
}

/// ...but a genuinely Z-Image batch of the wrong length must still fail on arity, so hoisting the
/// provider check did not shadow the length check it now precedes.
#[test]
fn a_short_z_image_batch_still_fails_on_arity() {
    let mut batch = foreign_five_rung_batch(Z_IMAGE_PROVIDER);
    batch["planned"].as_array_mut().unwrap().truncate(3);
    let error = validate_z_image_batch(&batch).expect_err("a 3-case Z-Image batch must still fail");
    assert_eq!(
        error,
        "Z-Image rung batch must contain exactly 5 cases, got 3"
    );
}
