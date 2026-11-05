use super::super::artifact_resolution::resolved_mlx_artifact_tier_for_model;
use super::super::standard_tier_subdir;
use super::super::ImageRequest;
use super::super::Quant;
use super::dense_te_requested_tier_bits;
use super::fixed_mlx_artifact_quant;
use super::mlx_load_quant_for_resolved_artifact;
use super::reconcile_resolved_tier_quant;
use super::resolve_quant;
use super::tier_quant_from_resolved_dir;
use crate::JsonObject;
use crate::Path;
use serde_json::json;

/// The resolved basename → precision map used to record the tier that ran (not the one requested).
#[test]
fn tier_quant_from_resolved_dir_maps_basename_to_precision() {
    let root = std::path::Path::new("/models/sd3_5_large-mlx");
    // Standard `q4`/`q8`/`bf16` tier dirs → their precision.
    assert_eq!(
        tier_quant_from_resolved_dir(&JsonObject::new(), &root.join("q4")),
        Some((Some(Quant::Q4), Some(4)))
    );
    assert_eq!(
        tier_quant_from_resolved_dir(&JsonObject::new(), &root.join("q8")),
        Some((Some(Quant::Q8), Some(8)))
    );
    assert_eq!(
        tier_quant_from_resolved_dir(&JsonObject::new(), &root.join("bf16")),
        Some((None, None))
    );
    // Boogu `<variant>-<tier>` and bare `<variant>` (= the packed Q8 default).
    assert_eq!(
        tier_quant_from_resolved_dir(&JsonObject::new(), &root.join("base-q4")),
        Some((Some(Quant::Q4), Some(4)))
    );
    assert_eq!(
        tier_quant_from_resolved_dir(&JsonObject::new(), &root.join("turbo-bf16")),
        Some((None, None))
    );
    assert_eq!(
        tier_quant_from_resolved_dir(&JsonObject::new(), &root.join("edit")),
        Some((Some(Quant::Q8), Some(8)))
    );
    // A fell-all-the-way-back-to-root (or modelPath) dir is not a recognizable tier → None, so the
    // caller keeps the request-derived quant.
    assert_eq!(tier_quant_from_resolved_dir(&JsonObject::new(), root), None);
}

/// The end-to-end reconcile is macOS-only (the MLX generate path). When the resolved tier matches
/// the request it's a pass-through; when it differs it records the tier that ran, and the
/// dense-TE guard keeps the load quant `None` while still correcting the recorded bits.
#[test]
fn reconcile_records_the_resolved_tier_on_fallback() {
    // Requested q8 present as q8 → pass through, records q8.
    assert_eq!(
        reconcile_resolved_tier_quant(
            (Some(Quant::Q8), Some(8)),
            std::path::Path::new("/m/q8"),
            &JsonObject::new(),
            true,
            "sd3_5_large",
            "job1",
            "mlx",
        ),
        (Some(Quant::Q8), Some(8)),
    );
    // Requested bf16 but only q4 downloaded → resolved dir is `q4`; record Q4 (not dense) and
    // rewrite the load quant to Q4 (safe no-op on already-packed weights).
    assert_eq!(
        reconcile_resolved_tier_quant(
            (None, None),
            std::path::Path::new("/m/q4"),
            &JsonObject::new(),
            true,
            "sd3_5_large",
            "job1",
            "mlx",
        ),
        (Some(Quant::Q4), Some(4)),
    );
    // Dense-TE turnkey: same q4 fallback, but the load quant STAYS `None` (never re-quantize the
    // dense bf16 TE) while the recorded bits are still corrected to Q4.
    assert_eq!(
        reconcile_resolved_tier_quant(
            (None, None),
            std::path::Path::new("/m/q4"),
            &JsonObject::new(),
            false,
            "flux2_klein_9b",
            "job1",
            "mlx",
        ),
        (None, Some(4)),
    );
    // Unrecognized resolved dir (fell back to repo root / modelPath) → keep the request quant.
    assert_eq!(
        reconcile_resolved_tier_quant(
            (Some(Quant::Q8), Some(8)),
            std::path::Path::new("/m/root"),
            &JsonObject::new(),
            true,
            "sd3_5_large",
            "job1",
            "mlx",
        ),
        (Some(Quant::Q8), Some(8)),
    );
}

#[test]
fn krea_and_flux2_klein_turnkeys_keep_packed_tier_out_of_load_quantization() {
    for engine_id in [
        "krea_2_raw",
        "krea_2_turbo",
        "krea_2_edit",
        "krea_2_turbo_edit",
        "flux2_klein_9b",
        "flux2_klein_9b_edit",
        "flux2_klein_9b_kv_edit",
    ] {
        assert_eq!(
            mlx_load_quant_for_resolved_artifact(engine_id, Some(Quant::Q4)),
            None,
            "{engine_id} q4 is an already-packed artifact tier"
        );
        assert_eq!(
            mlx_load_quant_for_resolved_artifact(engine_id, Some(Quant::Q8)),
            None,
            "{engine_id} q8 is an already-packed artifact tier"
        );
    }
    assert_eq!(
        mlx_load_quant_for_resolved_artifact("qwen_image", Some(Quant::Q4)),
        Some(Quant::Q4),
        "unrelated loaders retain their existing load-time quantization contract"
    );
    for engine_id in ["flux1_schnell", "flux1_dev"] {
        assert_eq!(
            mlx_load_quant_for_resolved_artifact(engine_id, Some(Quant::Q4)),
            Some(Quant::Q4),
            "{engine_id} preserves the provider's matching packed-tier quant marker"
        );
    }
    assert_eq!(
        fixed_mlx_artifact_quant("flux2_klein_9b_true_v2"),
        Some((None, None)),
        "True-V2's converted artifact is fixed dense BF16"
    );
    assert_eq!(
        resolved_mlx_artifact_tier_for_model(
            "flux2_klein_9b_true_v2",
            Path::new("/models/mlx/flux2_klein_9b_true_v2"),
            None,
        ),
        Some("bf16"),
        "the converted root is not a tier subdirectory, so its fixed BF16 tier must be explicit"
    );
    assert_eq!(
        fixed_mlx_artifact_quant("flux2_klein_9b"),
        None,
        "the selectable base tier matrix remains request/resolution-derived"
    );
}

/// End-to-end tier resolution + recording: a bf16 request against a turnkey where ONLY `q4/` is
/// downloaded resolves to `q4/`, and the reconciled recipe records Q4 — the precision that ran —
/// not the requested dense bf16. Guards the epic 8506 A/B workflow against telemetry that lies.
#[test]
fn bf16_request_with_only_q4_present_records_q4() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let dir = root.join("q4").join("transformer");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), b"x").unwrap();

    let req = ImageRequest::from_payload(
        json!({ "model": "sd3_5_large", "advanced": { "mlxQuantize": 0 } })
            .as_object()
            .unwrap(),
    );
    // The tier resolver falls through to q4.
    let resolved = standard_tier_subdir(root, &req);
    assert_eq!(resolved, root.join("q4"));
    // The request would have recorded dense bf16 — reconcile corrects it to the resolved Q4 tier.
    let requested = resolve_quant(&req, Some(&resolved));
    assert_eq!(requested, (None, None), "bf16 request derives dense");
    let (quant, bits) = reconcile_resolved_tier_quant(
        requested,
        &resolved,
        &JsonObject::new(),
        true,
        "sd3_5_large",
        "job1",
        "mlx",
    );
    assert_eq!((quant, bits), (Some(Quant::Q4), Some(4)));
}

/// sc-9362 (F-018 follow-up): the dense-TE transformer tier the request asks for is derived from
/// `advanced.mlxQuantize` exactly like [`standard_tier_subdir`] — no explicit pick → `q8` (sc-10726),
/// `<=0 → bf16 (None)`, `>4 → q8`, else `q4` — regardless of the always-`None` load quant
/// `resolve_quant` returns for dense-TE. This is what reconcile compares the resolved tier against so
/// a straight default job (resolving the q8 tier) isn't flagged as a spurious downgrade.
#[test]
fn dense_te_requested_tier_bits_mirrors_standard_tier_mapping() {
    let req = |mlx_quantize: serde_json::Value| {
        ImageRequest::from_payload(
                json!({ "model": "flux2_klein_9b", "advanced": { "mlxQuantize": mlx_quantize }, "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } } })
                    .as_object()
                    .unwrap(),
            )
    };
    let default = ImageRequest::from_payload(
            json!({ "model": "flux2_klein_9b", "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } } }).as_object().unwrap(),
        );
    // No selection → the q8 default (matches standard_tier_subdir's preferred, sc-10726).
    assert_eq!(dense_te_requested_tier_bits(&default), Some(8));
    assert_eq!(dense_te_requested_tier_bits(&req(json!(0))), None); // bf16 opt-out
    assert_eq!(dense_te_requested_tier_bits(&req(json!(4))), Some(4)); // explicit q4 honored
    assert_eq!(dense_te_requested_tier_bits(&req(json!(8))), Some(8));
    assert_eq!(dense_te_requested_tier_bits(&req(json!("8"))), Some(8)); // numeric-string
    assert_eq!(dense_te_requested_tier_bits(&req(json!(-1))), None);
}

/// sc-9362 (F-018 follow-up) + sc-10726: a straight (no-fallback) dense-TE default job — its q8
/// transformer tier is downloaded and resolves as the new q8 default — records the ACTUAL
/// transformer tier (Q8) while keeping the load quant `None` (the dense bf16 TE is never
/// re-quantized). The requested tier ([`dense_te_requested_tier_bits`], now q8 by default) MATCHES
/// the resolved q8 tier, so reconcile pass-throughs it with no spurious `quant_tier_downgraded`.
#[test]
fn dense_te_no_fallback_records_transformer_tier() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // The klein q4 + q8 transformer tiers are present (dense bf16 TE lives alongside in each tier);
    // a default job takes the q8 default.
    for tier in ["q4", "q8"] {
        let dir = root.join(tier).join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), b"x").unwrap();
    }

    let req = ImageRequest::from_payload(
            json!({ "model": "flux2_klein_9b", "advanced": {}, "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } } })
                .as_object()
                .unwrap(),
        );
    // The tier resolver lands on the q8 default tier (no fallback).
    let resolved = standard_tier_subdir(root, &req);
    assert_eq!(resolved, root.join("q8"));
    // resolve_quant keeps dense-TE at `(None, None)` (never re-quantize the dense bf16 TE)…
    assert_eq!(resolve_quant(&req, Some(&resolved)), (None, None));
    // …but reconcile against the REQUESTED transformer tier (q8) records the real transformer
    // precision (Q8) with the load quant still `None`, and — since resolved == requested — with no
    // downgrade (the requested/resolved tiers match, so it's a clean pass-through).
    let requested_for_reconcile = (None, dense_te_requested_tier_bits(&req));
    assert_eq!(requested_for_reconcile, (None, Some(8)));
    let (quant, bits) = reconcile_resolved_tier_quant(
        requested_for_reconcile,
        &resolved,
        &JsonObject::new(),
        false, // dense-TE: keep the load quant None
        "flux2_klein_9b",
        "job1",
        "mlx",
    );
    assert_eq!(
        (quant, bits),
        (None, Some(8)),
        "records the actual q8 transformer tier, load quant stays None"
    );
}

/// sc-9362: a GENUINE dense-TE fallback still surfaces — the request asks for q8 but only q4 is
/// downloaded, so the resolver falls through to q4; reconcile records q4 (the tier that ran) while
/// the load quant stays `None`. This is the case that legitimately warns/emits.
#[test]
fn dense_te_genuine_fallback_records_resolved_tier() {
    let req = ImageRequest::from_payload(
            json!({ "model": "flux2_klein_9b", "advanced": { "mlxQuantize": 8 }, "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } } })
                .as_object()
                .unwrap(),
        );
    // Requested q8 tier bits, but only the q4 tier exists on disk (resolver fell through).
    assert_eq!(dense_te_requested_tier_bits(&req), Some(8));
    let (quant, bits) = reconcile_resolved_tier_quant(
        (None, dense_te_requested_tier_bits(&req)),
        std::path::Path::new("/m/q4"),
        &JsonObject::new(),
        false,
        "flux2_klein_9b",
        "job1",
        "mlx",
    );
    assert_eq!(
        (quant, bits),
        (None, Some(4)),
        "genuine q8→q4 fallback records the resolved q4 tier, load quant stays None"
    );
}
