use super::*;
use serde_json::json;

fn request(advanced: serde_json::Value) -> ImageRequest {
    ImageRequest::from_payload(
        json!({ "model": "sd3_5_large", "advanced": advanced })
            .as_object()
            .unwrap(),
    )
}

/// A4 (sc-10189): the generic img2img arm keys off the `ui.img2img` manifest flag the catalog
/// forwards as `modelManifestEntry`, NOT a hardcoded model string — so Krea + SD3.5 + any future
/// `ui.img2img` model route uniformly, and a model without the flag stays plain txt2img.
#[cfg(native_lane)]
#[test]
fn model_supports_img2img_reads_the_ui_manifest_flag() {
    let entry = |manifest: serde_json::Value| {
        ImageRequest::from_payload(
            json!({ "model": "m", "modelManifestEntry": manifest })
                .as_object()
                .unwrap(),
        )
    };
    // ui.img2img: true → opted in (SD3.5 + Krea shape).
    assert!(model_supports_img2img(&entry(
        json!({ "ui": { "img2img": true } })
    )));
    // Flag explicitly false, or no `ui`, or no flag → plain txt2img.
    assert!(!model_supports_img2img(&entry(
        json!({ "ui": { "img2img": false } })
    )));
    assert!(!model_supports_img2img(&entry(json!({ "family": "sd3" }))));
    assert!(!model_supports_img2img(&entry(json!({ "ui": {} }))));
}

/// sc-18475: both SANA variants use the generic non-edit singular-reference gate on Candle as
/// MLX. The scheduler rejects plural/control carriers before dispatch; this covers the worker
/// fields after dispatch and prevents an edit or missing reference from becoming img2img.
#[cfg(native_lane)]
#[test]
fn sana_variants_use_the_generic_img2img_worker_gate() {
    let request = |model: &str, mode: &str, manifest_flag: bool| {
        ImageRequest::from_payload(
            json!({
                "model": model,
                "mode": mode,
                "referenceAssetId": "reference-1",
                "modelManifestEntry": { "ui": { "img2img": manifest_flag } },
            })
            .as_object()
            .unwrap(),
        )
    };

    for model in ["sana_1600m", "sana_sprint_1600m"] {
        let img2img = request(model, "text_to_image", true);
        assert!(uses_generic_img2img(&img2img, true));
        assert!(!uses_generic_img2img(&img2img, false));
        assert!(!uses_generic_img2img(
            &request(model, "edit_image", true),
            true
        ));
        assert!(!uses_generic_img2img(
            &request(model, "text_to_image", false),
            true
        ));
    }
}

/// A4.5 (sc-10193): on Z-Image t2i the Character Studio identity-init (`referenceStrength`, sc-3619)
/// keeps precedence; the generic `ui.img2img` reference-guided init (Image Studio "Image reference"
/// tile: `advanced.strength`, no `referenceStrength`) fires only when identity-init doesn't engage,
/// the model opts into `ui.img2img`, AND a reference is present. Otherwise plain txt2img.
#[cfg(target_os = "macos")]
#[test]
fn zimage_generic_img2img_yields_to_identity_reference() {
    let req = |advanced: serde_json::Value| {
        ImageRequest::from_payload(
            json!({
                "model": "z_image_turbo",
                "modelManifestEntry": { "ui": { "img2img": true } },
                // Both surfaces carry a reference asset; `identity_strength` reads it too.
                "referenceAssetId": "asset-1",
                "advanced": advanced,
            })
            .as_object()
            .unwrap(),
        )
    };
    // Image Studio "Image reference": strength only, no referenceStrength, a reference present →
    // generic img2img takes over.
    assert!(zimage_uses_generic_img2img(
        &req(json!({ "strength": 0.6 })),
        true
    ));
    // Character Studio identity: referenceStrength set → identity-init keeps precedence, generic
    // img2img yields (even though ui.img2img is on and a reference is present).
    assert!(!zimage_uses_generic_img2img(
        &req(json!({ "referenceStrength": 0.7 })),
        true
    ));
    // No reference asset present → neither surface engages (plain txt2img).
    assert!(!zimage_uses_generic_img2img(
        &req(json!({ "strength": 0.6 })),
        false
    ));
    // Model without the ui.img2img flag → never the generic path.
    let no_flag = ImageRequest::from_payload(
        json!({ "model": "z_image_turbo", "modelManifestEntry": { "ui": {} }, "advanced": {} })
            .as_object()
            .unwrap(),
    );
    assert!(!zimage_uses_generic_img2img(&no_flag, true));
}

/// A4.4 (sc-10192): Ideogram opts into the generic `ui.img2img` surface (the Image Studio "Image
/// reference" tile, `text_to_image` mode) while ALSO owning the bespoke Remix/inpaint edit arm
/// (`edit_image` mode, [`resolve_ideogram_edit`], sc-6303). The two arms in
/// [`resolve_generic_lane_conditioning`] are mutually exclusive by mode — the edit arm is checked
/// first and gates on `mode == "edit_image"`, the generic img2img arm on `mode != "edit_image"` — so a
/// plain-t2i reference routes to the generic init (a single `Conditioning::Reference`, no mask, which
/// the native engine's edit path denoises as plain img2img) while an Edit-tab job keeps the
/// mask-capable path. No engine change was needed: mlx-gen `resolve_edit` already treats a Reference
/// with no Mask as img2img. This tripwire locks the flag + mode-split; the disk-backed resolve
/// (asset decode) is validated on-device.
#[cfg(target_os = "macos")]
#[test]
fn ideogram_img2img_routes_by_mode() {
    let req = |model: &str, mode: &str| {
        ImageRequest::from_payload(
            json!({
                "model": model,
                "mode": mode,
                "modelManifestEntry": { "ui": { "img2img": true } },
                "referenceAssetId": "asset-1",
            })
            .as_object()
            .unwrap(),
        )
    };
    for model in ["ideogram_4", "ideogram_4_turbo"] {
        let is_ideogram_edit = |r: &ImageRequest| {
            matches!(r.model.as_str(), "ideogram_4" | "ideogram_4_turbo") && r.mode == "edit_image"
        };
        // Plain t2i + reference: the generic img2img arm's gate holds (flag on, non-edit mode) and the
        // earlier Ideogram edit arm does not — so the reference takes the generic img2img init.
        let t2i = req(model, "text_to_image");
        assert!(model_supports_img2img(&t2i) && t2i.mode != "edit_image");
        assert!(!is_ideogram_edit(&t2i));
        // Edit tab: the Ideogram edit arm claims it first and the generic arm yields (edit mode).
        let edit = req(model, "edit_image");
        assert!(is_ideogram_edit(&edit));
        assert!(!(model_supports_img2img(&edit) && edit.mode != "edit_image"));
    }
}

/// Write a minimal present `<tier>/transformer/<file>` so [`standard_tier_subdir`]'s
/// filename-agnostic probe sees the tier as downloaded.
fn seed_tier(root: &Path, tier: &str, file: &str) {
    let dir = root.join(tier).join("transformer");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(file), b"x").unwrap();
}

#[test]
fn defaults_to_q8_and_honors_quantize_selection() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Packed q4/q8 single-file + dense sharded bf16 (only the index.json shape).
    seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
    seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
    seed_tier(
        root,
        "bf16",
        "diffusion_pytorch_model.safetensors.index.json",
    );

    // No selection → q8 default (epic 10721 / sc-10726), clamped to installed.
    assert_eq!(
        standard_tier_subdir(root, &request(json!({}))),
        root.join("q8")
    );
    // An explicit Q4 pick is still honored (never overridden by the q8 default).
    assert_eq!(
        standard_tier_subdir(root, &request(json!({ "mlxQuantize": 4 }))),
        root.join("q4")
    );
    // mlxQuantize 8 → q8; 0/"none" → bf16; numeric-string accepted.
    assert_eq!(
        standard_tier_subdir(root, &request(json!({ "mlxQuantize": 8 }))),
        root.join("q8")
    );
    assert_eq!(
        standard_tier_subdir(root, &request(json!({ "mlxQuantize": 0 }))),
        root.join("bf16")
    );
    assert_eq!(
        standard_tier_subdir(root, &request(json!({ "mlxQuantize": "8" }))),
        root.join("q8")
    );
}

/// **sc-10732 — acceptance #1: the app-wide default-tier revert guard.**
///
/// Epic 10721 moved the gen-time default tier off the old blind `q4` to `q8` (sc-10726): the shared
/// [`preferred_tier`] returns `"q8"` with no explicit `mlxQuantize` pick and no per-model floor, and
/// BOTH tier resolvers ([`standard_tier_subdir`] and [`anima_tier_subdir`]) inherit it. If a future
/// change reverts that default back to `q4` — in `preferred_tier`'s `None => …` arm, or a resolver
/// special-case — this test FAILS LOUDLY. That is the whole point of the sc-10732 lock: the finer
/// resolver/floor tests each imply it, but this one names it at the revert site so the intent is
/// unmissable. Deliberately redundant.
#[test]
fn default_tier_is_q8_not_q4_regression() {
    // The shared default-tier primitive: no explicit `mlxQuantize`, no per-model floor → q8, never q4.
    assert_eq!(
            preferred_tier(None, None, false),
            "q8",
            "app-wide gen default MUST be q8 (epic 10721 / sc-10726) — a revert to q4 is the regression \
             this guards"
        );
    assert_ne!(
            preferred_tier(None, None, false),
            "q4",
            "the pre-epic-10721 blind-q4 default has been reverted — do NOT reinstate it (sc-10726/sc-10732)"
        );

    // Disk-backed through the standard resolver: a default job (no mlxQuantize) with ALL tiers
    // installed resolves the q8 subdir, not the washed q4 — the revert is caught end-to-end too.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    for tier in ["bf16", "q8", "q4"] {
        seed_tier(root, tier, "diffusion_pytorch_model.safetensors");
    }
    let default_job = request(json!({}));
    assert_eq!(
        standard_tier_subdir(root, &default_job),
        root.join("q8"),
        "standard_tier_subdir default MUST land on q8 (not q4) when all tiers are installed"
    );
    assert_ne!(standard_tier_subdir(root, &default_job), root.join("q4"));

    // The Anima resolver shares the same default (sc-10714 → sc-10731): its no-pick default is q8 too,
    // so a revert to q4 also washes Anima — the exact quality bug epic 10721 fixed.
    #[cfg(native_lane)]
    {
        let anima_root = tempfile::tempdir().unwrap();
        for tier in ["bf16", "q8", "q4"] {
            let dir = anima_root.path().join(tier).join("diffusion_models");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("anima-base-v1.0.safetensors"), b"x").unwrap();
        }
        let anima_default =
            ImageRequest::from_payload(json!({ "model": "anima_base" }).as_object().unwrap());
        assert_eq!(
            anima_tier_subdir(anima_root.path(), &anima_default),
            anima_root.path().join("q8"),
            "anima_tier_subdir default MUST be q8 (sc-10714), never the washed q4"
        );
        assert_ne!(
            anima_tier_subdir(anima_root.path(), &anima_default),
            anima_root.path().join("q4")
        );
    }
}

#[test]
fn falls_back_when_preferred_tier_absent() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Only q4 downloaded: a q8/bf16 request still resolves to the present q4 rather than a
    // half-empty subdir, so a partial turnkey surfaces as a load error, not a silent half-load.
    seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
    assert_eq!(
        standard_tier_subdir(root, &request(json!({ "mlxQuantize": 8 }))),
        root.join("q4")
    );
    // sc-10726: the q8 default is CLAMPED to installed — with only q4 on disk a default job
    // (no mlxQuantize) resolves q4, never a tier the user didn't download (no OOM risk).
    assert_eq!(
        standard_tier_subdir(root, &request(json!({}))),
        root.join("q4")
    );
    // Nothing present → the repo root (engine surfaces the missing-weights error).
    let empty = tempfile::tempdir().unwrap();
    assert_eq!(
        standard_tier_subdir(empty.path(), &request(json!({}))),
        empty.path().to_path_buf()
    );
}

/// sc-8746: the SDXL-family turnkeys pack their backbone under `unet/`, not `transformer/`, so
/// [`standard_tier_subdir`]'s probe must recognize a `unet/` component as a present tier.
#[test]
fn resolves_sdxl_unet_backbone_tiers() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let seed_unet = |tier: &str| {
        let dir = root.join(tier).join("unet");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), b"x").unwrap();
    };
    seed_unet("q4");
    seed_unet("q8");
    seed_unet("bf16");
    // Default q8, q8 selection, bf16 opt-out all resolve to the unet-backed tier subdir.
    assert_eq!(
        standard_tier_subdir(root, &request(json!({}))),
        root.join("q8")
    );
    assert_eq!(
        standard_tier_subdir(root, &request(json!({ "mlxQuantize": 8 }))),
        root.join("q8")
    );
    assert_eq!(
        standard_tier_subdir(root, &request(json!({ "mlxQuantize": 0 }))),
        root.join("bf16")
    );
}

/// sc-8771: SenseNova-U1 is a unified MoT turnkey — no `transformer/`/`unet/` component, the whole
/// backbone is a flat `model.safetensors` (q4/q8) or sharded `*.index.json` (bf16) directly in the
/// tier dir. [`standard_tier_subdir`]'s probe must recognize weights at the tier root itself.
#[test]
fn resolves_flat_unified_backbone_tiers() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let seed_flat = |tier: &str, file: &str| {
        let dir = root.join(tier);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), b"x").unwrap();
    };
    // Packed q4/q8 single-file, dense sharded bf16 (index.json shape) — the SenseNova layout.
    seed_flat("q4", "model.safetensors");
    seed_flat("q8", "model.safetensors");
    seed_flat("bf16", "model.safetensors.index.json");
    assert_eq!(
        standard_tier_subdir(root, &request(json!({}))),
        root.join("q8")
    );
    assert_eq!(
        standard_tier_subdir(root, &request(json!({ "mlxQuantize": 8 }))),
        root.join("q8")
    );
    assert_eq!(
        standard_tier_subdir(root, &request(json!({ "mlxQuantize": 0 }))),
        root.join("bf16")
    );
}

/// sc-14432: a SenseNova tier's backbone probe passes on the flat `model.safetensors` alone, so a
/// tier missing the rest of what its loader reads short-circuited the chain and then died at load
/// ("complete but unloadable"). Under the real `sensenova_u1_8b` id the resolver now folds in
/// `sensenova_tier_complete` and falls through to a complete sibling instead. Note the tokenizer is
/// model-wide: a tier that ships none still loads by borrowing a sibling tier's copy, so it is
/// `config.json` + the backbone that make a tier tier-specifically complete.
#[test]
fn sensenova_tier_selection_skips_a_torn_tier_for_a_complete_sibling() {
    let sensenova_request = |bits: serde_json::Value| {
        ImageRequest::from_payload(
            json!({ "model": "sensenova_u1_8b", "advanced": { "mlxQuantize": bits } })
                .as_object()
                .unwrap(),
        )
    };
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let seed_tier = |tier: &str, complete: bool| {
        let dir = root.join(tier);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
        if complete {
            std::fs::write(dir.join("config.json"), b"x").unwrap();
            std::fs::write(dir.join("tokenizer.json"), b"x").unwrap();
        }
    };

    // Only a torn q4 is installed: nothing complete anywhere, so the resolver still lands on it
    // (pre-sc-12279 behavior — never strand a model that might load) and warns.
    seed_tier("q4", false);
    assert_eq!(
        standard_tier_subdir(root, &sensenova_request(json!(4))),
        root.join("q4")
    );

    // Add a complete bf16: the explicit q4 pick now falls through to it rather than loading the torn
    // tier. This is the assertion that goes RED without the completeness fold.
    seed_tier("bf16", true);
    assert_eq!(
        standard_tier_subdir(root, &sensenova_request(json!(4))),
        root.join("bf16")
    );

    // Complete q4 with its own `config.json` — it borrows bf16's tokenizer (the engine's sibling
    // resolution), so it is loadable and the explicit pick is honored again.
    std::fs::write(root.join("q4").join("config.json"), b"x").unwrap();
    assert_eq!(
        standard_tier_subdir(root, &sensenova_request(json!(4))),
        root.join("q4")
    );
}

/// sc-10517 / sc-10714 / sc-10731: Anima is convert-at-install with a q4/q8/bf16 MATRIX under the
/// injected `modelPath` root (`bf16/ q8/ q4/`, each a `diffusion_models/<variant>.safetensors` tree —
/// NOT a `transformer/` component). [`anima_tier_subdir`] picks the tier by `mlxQuantize`
/// (**default Q8**; `<= 0` → bf16; `1..=4` → q4; `> 4` → q8) and falls back clean-tiers-first through
/// q8 → bf16 → q4 → root, so a partial install surfaces as a load error, never a silent half-load onto
/// the washed q4. The Q8 default (sc-10714) is the fix for base/aesthetic rendering smudgy at q4 —
/// q4 × CFG amplifies quant error; Q8 is near-lossless. [`is_anima_model`] gates only the three ids.
///
/// sc-10731 reconciled the old anima-specific `None => "q8"` hardcode into the shared, floor-driven
/// [`preferred_tier`]: with no manifest floor the app-wide q8 default still stands (the assertions
/// above), and the added assertions below prove the floor now DRIVES the default — a manifest
/// `mlx.minQualityTier` clamps the default UP (capped by installed), while an explicit pick is honored.
#[test]
fn anima_tier_subdir_selects_and_falls_back() {
    let anima_request = |bits: serde_json::Value| {
        ImageRequest::from_payload(
            json!({ "model": "anima_base", "advanced": { "mlxQuantize": bits } })
                .as_object()
                .unwrap(),
        )
    };
    assert!(is_anima_model("anima_base"));
    assert!(is_anima_model("anima_aesthetic") && is_anima_model("anima_turbo"));
    assert!(!is_anima_model("sd3_5_large"));

    let seed_tier = |root: &std::path::Path, tier: &str| {
        let dir = root.join(tier).join("diffusion_models");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("anima-base-v1.0.safetensors"), b"x").unwrap();
    };
    // All three tiers present → exact selection.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    for tier in ["bf16", "q8", "q4"] {
        seed_tier(root, tier);
    }
    assert_eq!(
        anima_tier_subdir(root, &anima_request(json!(null))),
        root.join("q8"),
        "no opt-in → the Q8 default (sc-10714), not the washed q4"
    );
    assert_eq!(
        anima_tier_subdir(root, &anima_request(json!(4))),
        root.join("q4"),
        "an explicit q4 pick is still honored"
    );
    assert_eq!(
        anima_tier_subdir(root, &anima_request(json!(8))),
        root.join("q8")
    );
    assert_eq!(
        anima_tier_subdir(root, &anima_request(json!(0))),
        root.join("bf16"),
        "explicit bf16 (mlxQuantize <= 0) is honored"
    );
    // Only q4 downloaded, but the default (q8) is requested → falls through clean-tiers-first to the
    // present q4, never a bare root. (An all-tiers install lands on q8 above; this is the partial case.)
    let tmp2 = tempfile::tempdir().unwrap();
    seed_tier(tmp2.path(), "q4");
    assert_eq!(
        anima_tier_subdir(tmp2.path(), &anima_request(json!(null))),
        tmp2.path().join("q4")
    );
    // q8 default absent (only bf16 + q4 present) → fallback prefers the clean bf16 over the washed q4.
    let tmp2b = tempfile::tempdir().unwrap();
    seed_tier(tmp2b.path(), "bf16");
    seed_tier(tmp2b.path(), "q4");
    assert_eq!(
        anima_tier_subdir(tmp2b.path(), &anima_request(json!(null))),
        tmp2b.path().join("bf16"),
        "q8 default absent → fallback prefers clean bf16 over washed q4"
    );
    // Nothing present → the root itself (the loader then surfaces a clear error).
    let tmp3 = tempfile::tempdir().unwrap();
    assert_eq!(
        anima_tier_subdir(tmp3.path(), &anima_request(json!(null))),
        tmp3.path().to_path_buf()
    );

    // sc-10731 — the per-model quality FLOOR now DRIVES the anima default (was a hardcode).
    // A floored request carries `advanced.mlxQuantize` AND the forwarded manifest floor.
    let floored_request = |bits: serde_json::Value, floor: &str| {
        ImageRequest::from_payload(
            json!({
                "model": "anima_base",
                "advanced": { "mlxQuantize": bits },
                "modelManifestEntry": { "mlx": { "minQualityTier": floor } }
            })
            .as_object()
            .unwrap(),
        )
    };
    // Anima base's PRODUCTION shape: floor q8, all tiers present, no explicit pick → the default is
    // q8 — now floor-DERIVED (the hardcode is gone), not a resolver special-case.
    assert_eq!(
        anima_tier_subdir(root, &floored_request(json!(null), "q8")),
        root.join("q8"),
        "floor q8 drives the default to q8 (reconciled from the sc-10714 hardcode)"
    );
    // Floor CAPPED by installed: floor q8 but only q4 on disk → q4 (the floor never selects an
    // uninstalled tier — a heavy model never resolves a tier the user didn't download).
    assert_eq!(
        anima_tier_subdir(tmp2.path(), &floored_request(json!(null), "q8")),
        tmp2.path().join("q4"),
        "floor q8 but only q4 installed → q4 (floor capped by installed)"
    );
    // Floor RAISES above the q8 default: a synthetic bf16 floor + no explicit pick → bf16 (proves the
    // clamp genuinely lifts the default, not just coincides with the app-wide q8).
    assert_eq!(
        anima_tier_subdir(root, &floored_request(json!(null), "bf16")),
        root.join("bf16"),
        "floored default → the floor tier (bf16 floor raises above the q8 default)"
    );
    // An EXPLICIT below-floor pick is HONORED even against a q8 floor — the worker never overrides a
    // deliberate quant choice (the web surfaces the advisory instead).
    assert_eq!(
        anima_tier_subdir(root, &floored_request(json!(4), "q8")),
        root.join("q4"),
        "explicit q4 honored despite the q8 floor (below-floor pick is the user's choice)"
    );
}

/// sc-10731: the shared, floor-aware default-tier logic behind both [`standard_tier_subdir`] and
/// [`anima_tier_subdir`]. An explicit `mlxQuantize` pick maps directly and is honored regardless of
/// the floor; with NO pick, the app-wide q8 default is clamped UP to the floor (never DOWN).
#[test]
fn preferred_tier_clamps_default_up_to_the_floor_only() {
    // No explicit pick, no floor → the app-wide q8 default (unchanged, non-floored models).
    assert_eq!(preferred_tier(None, None, false), "q8");
    // Floor at/below q8 leaves the q8 default untouched (the floor only ever RAISES).
    assert_eq!(preferred_tier(None, Some("q8"), false), "q8");
    assert_eq!(preferred_tier(None, Some("q4"), false), "q8");
    // A floor ABOVE q8 raises the default to the floor tier.
    assert_eq!(preferred_tier(None, Some("bf16"), false), "bf16");
    // Explicit picks map directly and are HONORED even below the floor (no clamp on an explicit pick).
    assert_eq!(preferred_tier(Some(4), Some("q8"), false), "q4");
    assert_eq!(preferred_tier(Some(0), Some("q8"), false), "bf16");
    assert_eq!(preferred_tier(Some(8), None, false), "q8");
    assert_eq!(preferred_tier(Some(2), None, false), "q4");
    // Rank order + normalization helpers.
    assert!(tier_quality_rank("bf16") > tier_quality_rank("q8"));
    assert!(tier_quality_rank("q8") > tier_quality_rank("q4"));
    assert_eq!(tier_quality_rank("mystery"), 0);
}

/// sc-11042 / epic 11037 SC#5 — **the regression guard for "no existing tier changes"**.
///
/// The whole `mlxQuantize` bits map must resolve EXACTLY as it did before NVFP4 existed, for every
/// (bits, floor) pair, on every host. `nvfp4: false` is what every non-NVFP4 request passes — i.e.
/// every request in existence today — so this pins the claim that adding the tier is purely
/// additive. If someone ever "helpfully" routes q4 → NVFP4 on Blackwell (the Option B this story
/// rejected), these assertions fail.
#[test]
fn preferred_tier_bits_map_is_unchanged_by_the_nvfp4_tier() {
    // The exact pre-sc-11042 mapping, re-asserted with the new parameter defaulted off.
    for floor in [None, Some("bf16"), Some("q8"), Some("q4"), Some("mystery")] {
        // Explicit bits picks: `<= 0` → bf16, `> 4` → q8, `1..=4` → q4 — floor-independent.
        assert_eq!(preferred_tier(Some(0), floor, false), "bf16");
        assert_eq!(preferred_tier(Some(-1), floor, false), "bf16");
        assert_eq!(preferred_tier(Some(4), floor, false), "q4");
        assert_eq!(preferred_tier(Some(1), floor, false), "q4");
        assert_eq!(preferred_tier(Some(8), floor, false), "q8");
        assert_eq!(preferred_tier(Some(16), floor, false), "q8");
    }
    // No pick → the q8 default, clamped UP to a higher floor only.
    assert_eq!(preferred_tier(None, None, false), "q8");
    assert_eq!(preferred_tier(None, Some("q4"), false), "q8");
    assert_eq!(preferred_tier(None, Some("q8"), false), "q8");
    assert_eq!(preferred_tier(None, Some("bf16"), false), "bf16");
    // And NOTHING in the bits map can ever produce the NVFP4 tier: only the explicit flag does.
    for bits in [None, Some(-1), Some(0), Some(1), Some(4), Some(8), Some(16)] {
        for floor in [None, Some("bf16"), Some("q8"), Some("q4")] {
            assert_ne!(
                preferred_tier(bits, floor, false),
                NVFP4_TIER,
                "NVFP4 must never be reachable from the bits map (bits={bits:?}, floor={floor:?})"
            );
        }
    }
}

/// sc-11042: an explicit, host-eligible NVFP4 pick resolves the distinct `nvfp4` tier and takes no
/// part in the bits map or the floor clamp — it is a tier identity, not a rung on the fidelity ladder.
#[test]
fn preferred_tier_resolves_the_distinct_nvfp4_tier_on_an_explicit_pick() {
    // Wins regardless of what bits/floor say — NVFP4 has no honest `mlxQuantize` integer, so a
    // stray/stale bits value must not steer a request that explicitly named the NVFP4 tier.
    for bits in [None, Some(0), Some(4), Some(8)] {
        for floor in [None, Some("bf16"), Some("q8")] {
            assert_eq!(preferred_tier(bits, floor, true), NVFP4_TIER);
        }
    }
    // It is NOT floor-clamped: a bf16 floor does not raise/replace an explicit NVFP4 pick.
    assert_eq!(preferred_tier(None, Some("bf16"), true), NVFP4_TIER);
    // The label is distinct from q4 — the aliasing SC#5 forbids (see `video_quant_label`).
    assert_ne!(NVFP4_TIER, "q4");
}

/// sc-11042: [`nvfp4_requested`] reads ONLY the explicit `advanced.quantTier: "nvfp4"` label, and no
/// `mlxQuantize` value — not even `4` — can stand in for it. This is the request-side half of SC#5.
#[test]
fn nvfp4_requested_reads_only_the_explicit_quant_tier_label() {
    // The explicit label, tolerant of surrounding whitespace / casing like the sibling parsers.
    assert!(nvfp4_requested(&request(json!({ "quantTier": "nvfp4" }))));
    assert!(nvfp4_requested(&request(
        json!({ "quantTier": "  nvfp4 " })
    )));
    assert!(nvfp4_requested(&request(json!({ "quantTier": "NVFP4" }))));
    // Nothing else asks for NVFP4: no label, another tier's label, a non-string, or ANY bits value.
    assert!(!nvfp4_requested(&request(json!({}))));
    assert!(!nvfp4_requested(&request(json!({ "quantTier": "q4" }))));
    assert!(!nvfp4_requested(&request(
        json!({ "quantTier": "int8-convrot" })
    )));
    assert!(!nvfp4_requested(&request(json!({ "quantTier": 4 }))));
    assert!(!nvfp4_requested(&request(json!({ "quantTier": true }))));
    for bits in [0, 4, 8] {
        assert!(
                !nvfp4_requested(&request(json!({ "mlxQuantize": bits }))),
                "mlxQuantize {bits} must never request NVFP4 — it is a bits-valued knob naming q4/q8/bf16"
            );
    }
}

/// sc-11042 — **the SC#5 opt-in guard at the resolver**: with NO explicit `quantTier` label, a
/// Blackwell-eligible host resolves EXACTLY the tier it resolved before this story. Being on sm_120
/// selects nothing by itself; NVFP4 is only ever reached by a deliberate user pick.
#[test]
fn nvfp4_never_selected_without_an_explicit_pick() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
    seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
    seed_tier(
        root,
        "bf16",
        "diffusion_pytorch_model.safetensors.index.json",
    );
    seed_tier(root, NVFP4_TIER, "diffusion_pytorch_model.safetensors");

    // The injected `true` says "the HOST is Blackwell-eligible" — the hardware gate is ON, but no
    // request below carries the `quantTier` label. Every answer is the pre-sc-11042 one, even though
    // an `nvfp4/` tier is sitting right there on disk. That is SC#5: hardware selects nothing.
    for (advanced, expected) in [
        (json!({}), "q8"),                   // default
        (json!({ "mlxQuantize": 4 }), "q4"), // the tier NVFP4 must never silently replace
        (json!({ "mlxQuantize": 8 }), "q8"),
        (json!({ "mlxQuantize": 0 }), "bf16"),
    ] {
        assert_eq!(
                standard_tier_subdir_gated(root, &request(advanced.clone()), true),
                root.join(expected),
                "an unlabeled request must resolve {expected} on a Blackwell host (advanced={advanced})"
            );
    }
}

/// sc-11042: the NVFP4 tier resolves on an sm_120 host and falls back CLEANLY off Blackwell.
///
/// The two host classes are exercised through the injected gate rather than a live compute-cap
/// probe, so the rig isn't needed; [`nvfp4_host_eligible`] is what maps hardware → this bool, and
/// its floor is pinned separately by `gpu::tests`.
#[test]
fn nvfp4_tier_resolves_on_blackwell_and_falls_back_off_it() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
    seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
    seed_tier(root, NVFP4_TIER, "diffusion_pytorch_model.safetensors");
    let picked = request(json!({ "quantTier": "nvfp4" }));

    // sm_120 + the tier installed → the distinct `nvfp4/` dir.
    assert_eq!(
        standard_tier_subdir_gated(root, &picked, true),
        root.join(NVFP4_TIER)
    );
    // NOT Blackwell (pre-sm_120 NVIDIA, macOS/MLX, or the neither build) → the label is ignored and
    // the request lands on an installed tier via the normal chain. A clean fallback, not an error.
    assert_eq!(
        standard_tier_subdir_gated(root, &picked, false),
        root.join("q8")
    );

    // sm_120 but the tier ISN'T converted yet (sc-11043 owns the converter; the shipping case today)
    // → rejoins the same clean chain rather than failing the load.
    let bare = tempfile::tempdir().unwrap();
    seed_tier(bare.path(), "q4", "diffusion_pytorch_model.safetensors");
    seed_tier(bare.path(), "q8", "diffusion_pytorch_model.safetensors");
    assert_eq!(
        standard_tier_subdir_gated(bare.path(), &picked, true),
        bare.path().join("q8")
    );
    // …and with only q4 on disk it clamps to q4 — never a half-load, never an FP4 load with no
    // FP4 weights.
    let only_q4 = tempfile::tempdir().unwrap();
    seed_tier(only_q4.path(), "q4", "diffusion_pytorch_model.safetensors");
    assert_eq!(
        standard_tier_subdir_gated(only_q4.path(), &picked, true),
        only_q4.path().join("q4")
    );
}

/// sc-11042: [`resolve_quant`] returns the distinct `Quant::Nvfp4` for an explicit, host-eligible
/// pick — with **no** bit count (NVFP4 is ~4.5 EFFECTIVE bits/weight; `Some(4)` would alias the
/// recipe onto q4) — and is otherwise completely unchanged.
#[cfg(native_lane)]
#[test]
fn resolve_quant_returns_the_distinct_nvfp4_tier_only_on_an_explicit_blackwell_pick() {
    let picked = request(json!({ "quantTier": "nvfp4" }));
    let nvfp4_dir = PathBuf::from("/models/klein").join(NVFP4_TIER);
    let q8_dir = PathBuf::from("/models/klein").join("q8");
    // Explicit pick + Blackwell + the nvfp4 tier RESOLVED → the distinct tier, and NOT `Some(4)` bits.
    assert_eq!(
        resolve_quant_gated(&picked, true, Some(&nvfp4_dir)),
        (Some(Quant::Nvfp4), None)
    );
    // Off Blackwell → the label is ignored; the unchanged q8 default. A clean fallback.
    assert_eq!(
        resolve_quant_gated(&picked, false, Some(&nvfp4_dir)),
        (Some(Quant::Q8), Some(8))
    );
    // On Blackwell but the resolver landed on q8 (the `nvfp4/` dir isn't installed — the shipping
    // case today): the LOAD quant must be q8 too. Loading FP4 against q8 weights is not a mislabel,
    // it's a wrong load.
    assert_eq!(
        resolve_quant_gated(&picked, true, Some(&q8_dir)),
        (Some(Quant::Q8), Some(8))
    );
    // Tier dir unknown ⇒ never claim NVFP4 (see `tier_dir_is_nvfp4`).
    assert_eq!(
        resolve_quant_gated(&picked, true, None),
        (Some(Quant::Q8), Some(8))
    );

    // SC#5: every existing mapping is untouched on BOTH host classes and on EVERY tier dir — being
    // on sm_120, even with the `nvfp4/` tier resolved, never converts an unlabeled q4/q8/bf16
    // request into an NVFP4 one.
    for nvfp4_host in [false, true] {
        for dir in [None, Some(&nvfp4_dir), Some(&q8_dir)] {
            assert_eq!(
                resolve_quant_gated(&request(json!({})), nvfp4_host, dir.map(PathBuf::as_path)),
                (Some(Quant::Q8), Some(8))
            );
            assert_eq!(
                resolve_quant_gated(
                    &request(json!({ "mlxQuantize": 4 })),
                    nvfp4_host,
                    dir.map(PathBuf::as_path)
                ),
                (Some(Quant::Q4), Some(4)),
                "a q4 pick must stay int4-affine q4 on every host (epic 11037 SC#5)"
            );
            assert_eq!(
                resolve_quant_gated(
                    &request(json!({ "mlxQuantize": 8 })),
                    nvfp4_host,
                    dir.map(PathBuf::as_path)
                ),
                (Some(Quant::Q8), Some(8))
            );
            assert_eq!(
                resolve_quant_gated(
                    &request(json!({ "mlxQuantize": 0 })),
                    nvfp4_host,
                    dir.map(PathBuf::as_path)
                ),
                (None, None)
            );
        }
    }
}

/// sc-11042 — **the dense-TE carve-out outranks the NVFP4 tier** (sc-8711 / sc-9362).
///
/// `flux2_klein_9b`/`_kv` declare `mlx.denseTextEncoderTier` AND ride the candle txt2img lane, whose
/// `resolve_quant` call is gated only by `model.supports_quant()`. With the NVFP4 arm ordered ahead
/// of the carve-out, a crafted `quantTier: "nvfp4"` returned `Some(Nvfp4)` and skipped it — which
/// would re-quantize the bf16 text encoder those stories deliberately kept dense. The carve-out is
/// the wider invariant (the TE must never be quantized by ANY tier), so it wins outright.
#[cfg(native_lane)]
#[test]
fn the_nvfp4_arm_never_short_circuits_the_dense_te_carve_out() {
    let nvfp4_dir = PathBuf::from("/models/klein").join(NVFP4_TIER);
    // The declared dense-TE turnkey (sc-15799: the manifest flag is the only form) and a novel
    // manifest-flag model, with EVERY gate for the NVFP4 arm satisfied: explicit pick, Blackwell
    // host, and the `nvfp4/` tier resolved.
    let by_id = ImageRequest::from_payload(
        json!({
            "model": "flux2_klein_9b",
            "advanced": { "quantTier": "nvfp4" },
            "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } }
        })
        .as_object()
        .unwrap(),
    );
    let by_manifest = ImageRequest::from_payload(
        json!({
            "model": "some_matrix_model",
            "advanced": { "quantTier": "nvfp4" },
            "modelManifestEntry": { "mlx": { "denseTextEncoderTier": true } }
        })
        .as_object()
        .unwrap(),
    );

    for dense_te in [&by_id, &by_manifest] {
        assert!(is_dense_te_tier(dense_te), "test fixture must be dense-TE");
        assert_eq!(
            resolve_quant_gated(dense_te, true, Some(&nvfp4_dir)),
            (None, None),
            "a dense-TE turnkey must load with quant None — the NVFP4 arm must not preempt the \
                 sc-8711 carve-out that keeps its bf16 text encoder dense"
        );
    }

    // The carve-out is not a general NVFP4 kill-switch: a NON-dense-TE model on the same terms
    // still selects the tier, so the fix can't be masking the arm entirely.
    assert_eq!(
        resolve_quant_gated(
            &request(json!({ "quantTier": "nvfp4" })),
            true,
            Some(&nvfp4_dir)
        ),
        (Some(Quant::Nvfp4), None)
    );
}

/// sc-11042 / epic 11037 SC#5 — **the image lane's aliasing guard**, the sibling of
/// `video_quant_label_never_aliases_nvfp4_to_q4`.
///
/// [`effective_quant_label`] maps [`resolve_quant`]'s BIT COUNT onto a label, and NVFP4's bits are
/// deliberately `None`, so a bits-only match drops it into the `_ => bf16` arm — stamping a 4-bit
/// NVFP4 render as full-precision `"bf16"`. The inverse of the video lane's `q4` mislabel, the same
/// violation. Matching the variant is what pins it.
#[cfg(native_lane)]
#[test]
fn effective_quant_label_never_aliases_nvfp4_to_bf16_or_q4() {
    let picked = request(json!({ "quantTier": "nvfp4" }));
    let nvfp4_dir = PathBuf::from("/models/klein").join(NVFP4_TIER);
    let (label, bits) = effective_quant_label_gated(&picked, true, Some(&nvfp4_dir));
    assert_eq!(label, Some("nvfp4".to_owned()));
    // Neither mislabel is possible: not the "bf16" a bits-only match produced…
    assert_ne!(label, Some("bf16".to_owned()));
    // …nor the "q4" the video lane's bits-derived form produced.
    assert_ne!(label, Some("q4".to_owned()));
    // No honest integer width — same reason the tier reports None everywhere else.
    assert_eq!(bits, None);

    // Off Blackwell the label reports what actually ran (the q8 fallback), not the tier asked for.
    assert_eq!(
        effective_quant_label_gated(&picked, false, Some(&nvfp4_dir)),
        (Some("q8".to_owned()), Some(8))
    );
    // SC#5: the existing labels are unchanged on both host classes.
    for nvfp4_host in [false, true] {
        assert_eq!(
            effective_quant_label_gated(
                &request(json!({ "mlxQuantize": 4 })),
                nvfp4_host,
                Some(&nvfp4_dir)
            ),
            (Some("q4".to_owned()), Some(4))
        );
        assert_eq!(
            effective_quant_label_gated(
                &request(json!({ "mlxQuantize": 8 })),
                nvfp4_host,
                Some(&nvfp4_dir)
            ),
            (Some("q8".to_owned()), Some(8))
        );
        assert_eq!(
            effective_quant_label_gated(
                &request(json!({ "mlxQuantize": 0 })),
                nvfp4_host,
                Some(&nvfp4_dir)
            ),
            (Some("bf16".to_owned()), None)
        );
        // The int8-convrot tier still wins its early return.
        assert_eq!(
            effective_quant_label_gated(
                &request(json!({ "convRot": true })),
                nvfp4_host,
                Some(&nvfp4_dir)
            ),
            (Some("int8-convrot".to_owned()), None)
        );
    }
}

/// sc-16025: provider-local floors are part of the effective numeric identity. The label stored
/// on the asset/effective-settings record must not collapse Mage's mixed q4/q8 load to plain q4.
#[cfg(native_lane)]
#[test]
fn mage_q4_label_includes_every_descriptor_declared_component_floor() {
    let request = ImageRequest::from_payload(
        json!({ "model": "mage_flow", "advanced": { "mlxQuantize": 4 } })
            .as_object()
            .unwrap(),
    );
    let q4_dir = PathBuf::from("/models/mage_flow/q4");
    let (label, bits) = effective_quant_label_gated(&request, false, Some(&q4_dir));
    assert_eq!(
        label.as_deref(),
        Some("q4+[textEncoder:q8,transformerHead:q8]")
    );
    assert_ne!(label.as_deref(), Some("q4"));
    assert_eq!(bits, None, "a mixed-width profile has no single bit count");
}

/// sc-11042 / epic 11037 SC#5 — **the recorded label must describe the tier that RAN.**
///
/// The regression this pins: `effective_quant_label` was host-aware but DISK-BLIND. It returned
/// `Nvfp4` from `nvfp4_requested && nvfp4_host` alone, while [`standard_tier_subdir`] independently
/// (and correctly) fell back to `q8/` when the `nvfp4/` dir was absent — **the shipping case on
/// every model today**, since sc-11043 has not converted a tier yet. So on a Blackwell candle host
/// the resolver loaded the **q8** weights and the asset record stamped them **`"nvfp4"`**: a q8
/// render sold as NVFP4. Exactly the SC#5 creative-choice aliasing this tier exists to eliminate,
/// displaced out of selection and into telemetry.
///
/// The suite already contained both halves of the contradiction — `standard_tier_subdir_gated(bare,
/// picked, true) == q8` and `effective_quant_label_gated(picked, true) == "nvfp4"` — and never
/// connected them. This test connects them: it drives the SAME request through the resolver and the
/// label and asserts they agree, so the two can never drift apart again.
#[cfg(native_lane)]
#[test]
fn effective_quant_label_reports_the_resolved_tier_not_the_requested_one() {
    let picked = request(json!({ "quantTier": "nvfp4" }));

    // A Blackwell host whose turnkey has NO `nvfp4/` dir — the shipping case today.
    let bare = tempfile::tempdir().unwrap();
    seed_tier(bare.path(), "q4", "diffusion_pytorch_model.safetensors");
    seed_tier(bare.path(), "q8", "diffusion_pytorch_model.safetensors");

    // The resolver falls back to q8 (this half already passed before the fix)…
    let resolved = standard_tier_subdir_gated(bare.path(), &picked, true);
    assert_eq!(resolved, bare.path().join("q8"));
    // …so the label MUST say q8. Before the fix this returned `Some("nvfp4")` with bits None —
    // a q8 render recorded as an NVFP4 one.
    let (label, bits) = effective_quant_label_gated(&picked, true, Some(&resolved));
    assert_ne!(
        label,
        Some(NVFP4_TIER.to_owned()),
        "an NVFP4 pick that RESOLVED q8 must never be recorded as nvfp4 (epic 11037 SC#5)"
    );
    assert_eq!((label, bits), (Some("q8".to_owned()), Some(8)));

    // Same host, same request, but the tier IS installed → the label is nvfp4. This is what proves
    // the fix pins the label to the DISK and isn't just disabling the tier.
    let converted = tempfile::tempdir().unwrap();
    seed_tier(
        converted.path(),
        "q8",
        "diffusion_pytorch_model.safetensors",
    );
    seed_tier(
        converted.path(),
        NVFP4_TIER,
        "diffusion_pytorch_model.safetensors",
    );
    let resolved = standard_tier_subdir_gated(converted.path(), &picked, true);
    assert_eq!(resolved, converted.path().join(NVFP4_TIER));
    assert_eq!(
        effective_quant_label_gated(&picked, true, Some(&resolved)),
        (Some(NVFP4_TIER.to_owned()), None)
    );

    // The resolver and the label agree on EVERY tier/host/install combination — the invariant, not
    // just the two cases above. Each `expected` is read off the resolver's own output.
    for install in [vec!["q4", "q8"], vec!["q4", "q8", NVFP4_TIER]] {
        let root = tempfile::tempdir().unwrap();
        for tier in &install {
            seed_tier(root.path(), tier, "diffusion_pytorch_model.safetensors");
        }
        for nvfp4_host in [false, true] {
            for advanced in [
                json!({ "quantTier": "nvfp4" }),
                json!({}),
                json!({ "mlxQuantize": 4 }),
            ] {
                let req = request(advanced.clone());
                let resolved = standard_tier_subdir_gated(root.path(), &req, nvfp4_host);
                let (label, _) = effective_quant_label_gated(&req, nvfp4_host, Some(&resolved));
                let resolved_tier = resolved.file_name().unwrap().to_str().unwrap();
                assert_eq!(
                    label.as_deref() == Some(NVFP4_TIER),
                    resolved_tier == NVFP4_TIER,
                    "label {label:?} disagrees with the resolved tier {resolved_tier} \
                         (host={nvfp4_host}, advanced={advanced}, installed={install:?})"
                );
            }
        }
    }
}

/// sc-11042: the Krea 2 Turbo resolver (epic 11037's named SC#1/SC#2 validation vehicle, sc-12110)
/// wires the NVFP4 tier on the same terms — explicit pick + Blackwell, clean fallback otherwise —
/// and its q4/q8/bf16 selection is unchanged.
#[test]
fn krea_model_subdir_wires_nvfp4_without_changing_its_existing_tiers() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    for tier in ["q4", "q8", NVFP4_TIER] {
        seed_tier(root, tier, "diffusion_pytorch_model.safetensors");
    }
    let picked = request(json!({ "quantTier": "nvfp4" }));
    // Explicit pick on Blackwell → the distinct tier.
    assert_eq!(
        krea_model_subdir_gated(root, &picked, true),
        root.join(NVFP4_TIER)
    );
    // Off Blackwell → clean fallback to the shipped q8 default.
    assert_eq!(
        krea_model_subdir_gated(root, &picked, false),
        root.join("q8")
    );
    // SC#5: krea's existing tiers resolve exactly as before on a Blackwell host with an `nvfp4/`
    // dir present.
    for (advanced, expected) in [
        (json!({}), "q8"),
        (json!({ "mlxQuantize": 4 }), "q4"),
        (json!({ "mlxQuantize": 8 }), "q8"),
    ] {
        assert_eq!(
            krea_model_subdir_gated(root, &request(advanced), true),
            root.join(expected)
        );
    }
}

/// sc-10731: [`min_quality_floor`] reads the forwarded manifest `mlx.minQualityTier`, honoring only a
/// valid bf16/q8/q4 value and treating an absent/bogus one as no floor.
#[test]
fn min_quality_floor_reads_valid_manifest_values_only() {
    let with_floor = |mlx: serde_json::Value| {
        ImageRequest::from_payload(
            json!({ "model": "anima_base", "modelManifestEntry": { "mlx": mlx } })
                .as_object()
                .unwrap(),
        )
    };
    assert_eq!(
        min_quality_floor(&with_floor(json!({ "minQualityTier": "q8" }))),
        Some("q8")
    );
    assert_eq!(
        min_quality_floor(&with_floor(json!({ "quantize": 4 }))),
        None
    );
    assert_eq!(
        min_quality_floor(&with_floor(json!({ "minQualityTier": "q2" }))),
        None
    );
    // No manifest entry at all → no floor.
    assert_eq!(min_quality_floor(&request(json!({}))), None);
}

/// sc-10731: [`standard_tier_subdir`] applies the same floor clamp — a floored standard-tier model's
/// DEFAULT lands at the floor (capped by installed), while a non-floored one is unchanged and an
/// explicit below-floor pick is honored.
#[test]
fn standard_tier_subdir_clamps_default_to_the_floor() {
    let floored = |bits: Option<i64>, floor: &str| {
        let advanced = match bits {
            Some(b) => json!({ "mlxQuantize": b }),
            None => json!({}),
        };
        ImageRequest::from_payload(
            json!({
                "model": "some_matrix_model",
                "advanced": advanced,
                "modelManifestEntry": { "mlx": { "minQualityTier": floor } }
            })
            .as_object()
            .unwrap(),
        )
    };
    // All three tiers present.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    for tier in ["bf16", "q8", "q4"] {
        seed_tier(root, tier, "diffusion_pytorch_model.safetensors");
    }
    // Floor bf16, no explicit pick → bf16 (floored default → floor tier).
    assert_eq!(
        standard_tier_subdir(root, &floored(None, "bf16")),
        root.join("bf16")
    );
    // Floor q8, no explicit pick → q8; a non-floored default is q8 too, but here it is floor-driven.
    assert_eq!(
        standard_tier_subdir(root, &floored(None, "q8")),
        root.join("q8")
    );
    // Explicit q4 honored despite a bf16 floor (below-floor pick is the user's choice).
    assert_eq!(
        standard_tier_subdir(root, &floored(Some(4), "bf16")),
        root.join("q4")
    );
    // Floor capped by installed: floor bf16 but only q4 on disk → q4.
    let only_q4 = tempfile::tempdir().unwrap();
    seed_tier(only_q4.path(), "q4", "diffusion_pytorch_model.safetensors");
    assert_eq!(
        standard_tier_subdir(only_q4.path(), &floored(None, "bf16")),
        only_q4.path().join("q4")
    );
    // Non-floored model unaffected — the plain q8 default still stands (acceptance #3).
    assert_eq!(
        standard_tier_subdir(root, &request(json!({}))),
        root.join("q8")
    );
}

/// Regression guard for **sc-10578** (epic 10512), pinning the worker HALF of the fix that mlx-gen
/// #681 (this pin bump → `a5c1fcd`) delivered.
///
/// The bug: on mlx-gen ≤ `6a10ae1`, `mlx-gen-anima`'s `load` rejected
/// `spec.quantize.is_some() && !spec.adapters.is_empty()`. The worker defaults EVERY MLX model's
/// tier to Q8 ([`resolve_quant`]'s `None` arm), so `spec.quantize` is `Some(..)` on the default
/// path — which meant **every** Anima LoRA/LoKr generation failed at model load, at the DEFAULT
/// tier, with no tier selection by the user. The only escape was an explicit bf16 pick
/// (`mlxQuantize <= 0`). That combination — default tier + adapter — is exactly what no prior test
/// exercised, which is how the model shipped with `supports_lora`, an official style LoRA, and a
/// trainer whose output could not be loaded.
///
/// This asserts the two worker-owned facts that made the bug fire, so a future change that
/// reintroduces either is caught here. The end-to-end proof that the engine now ACCEPTS this spec
/// lives in mlx-gen's real-weights `tests/packed_adapters.rs` (a Mac + weights are needed to run it,
/// so it cannot live in this crate).
#[test]
fn anima_default_tier_with_adapter_builds_loadable_spec() {
    use gen_core::{AdapterKind, AdapterSpec, Quant};

    // 1. A default Anima request IS quantized — the premise that made the guard fire everywhere.
    let base_default =
        ImageRequest::from_payload(json!({ "model": "anima_base" }).as_object().unwrap());
    assert_eq!(
        resolve_quant(&base_default, None),
        (Some(Quant::Q8), Some(8)),
        "anima_base with no mlxQuantize must default to Q8 — the reason adding an adapter used to \
             fail at load on the DEFAULT tier"
    );
    // aesthetic/turbo ship manifest `mlx.quantize: 4`; still quantized, so still hit the guard.
    let aesthetic = ImageRequest::from_payload(
        json!({ "model": "anima_aesthetic", "modelManifestEntry": { "mlx": { "quantize": 4 } } })
            .as_object()
            .unwrap(),
    );
    assert_eq!(resolve_quant(&aesthetic, None), (Some(Quant::Q4), Some(4)));
    // Only an explicit bf16 opt-out escaped the bug.
    let bf16 = ImageRequest::from_payload(
        json!({ "model": "anima_base", "advanced": { "mlxQuantize": 0 } })
            .as_object()
            .unwrap(),
    );
    assert_eq!(resolve_quant(&bf16, None), (None, None));

    // 2. The LoadSpec the worker hands the engine carries a quant AND the adapter together — the
    //    exact `quantize.is_some() && !adapters.is_empty()` shape mlx-gen-anima rejected on
    //    `6a10ae1` and accepts on `a5c1fcd`.
    let (quant, _) = resolve_quant(&base_default, None);
    let adapters = vec![AdapterSpec::new(
        PathBuf::from("/tmp/anima-style-lora.safetensors"),
        1.0,
        AdapterKind::Lora,
    )];
    let spec = load_spec(PathBuf::from("/tmp/anima-q8-tier"), quant, adapters, None);
    assert!(
        spec.quantize.is_some(),
        "the default Anima tier is quantized, so the spec carries a quant"
    );
    assert!(
        !spec.adapters.is_empty(),
        "the adapter is present alongside the quant — the combination that used to fail"
    );
    // A dense-tier spec (the bf16 escape hatch) never carried a quant — documents the one path
    // that worked before the fix.
    let dense = load_spec(
        PathBuf::from("/tmp/anima-bf16-tier"),
        None,
        vec![AdapterSpec::new(
            PathBuf::from("/tmp/anima-style-lora.safetensors"),
            1.0,
            AdapterKind::Lora,
        )],
        None,
    );
    assert!(dense.quantize.is_none());
}

/// sc-10676: off-Mac candle dense-load resolution. [`anima_dense_split_files_dir`] descends into the
/// `split_files/` subdir of the HF snapshot (the raw dense DiT tree, NOT a converted q4/q8/bf16 tier),
/// falling back to the snapshot root when `split_files/` is absent (the candle loader accepts the
/// parent; a partial download stays a loud load error). This is the off-Mac counterpart to the mac
/// convert-at-install [`anima_tier_subdir`] — there are no tier subdirs off-Mac.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn anima_dense_split_files_dir_descends_into_split_files_else_root() {
    // Snapshot holds `split_files/diffusion_models/...` → resolve there (the loader reads it directly).
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("split_files").join("diffusion_models")).unwrap();
    assert_eq!(
        anima_dense_split_files_dir(root.to_path_buf()),
        root.join("split_files"),
        "descends into split_files/ when present"
    );
    // No `split_files/` yet (partial/absent download) → the snapshot root; the loader surfaces a
    // clear "not an Anima split_files dir" error rather than this silently pointing at a wrong dir.
    let tmp2 = tempfile::tempdir().unwrap();
    assert_eq!(
        anima_dense_split_files_dir(tmp2.path().to_path_buf()),
        tmp2.path().to_path_buf(),
        "falls back to the snapshot root when split_files/ is absent"
    );
}

/// sc-14249 — the candle SenseNova lane resolves the tier the REQUEST asks for.
///
/// This replaces the two sc-13817 tests that pinned it to `bf16/`. That force existed only
/// because `candle-gen-sensenova` mmapped at a hardcoded F32 and rejected `spec.quantize`, so
/// packed weights were unreadable; the engine now packed-detects every backbone projection, so
/// the family takes the ordinary `standard_tier_subdir` descent and the cheap tiers are reachable
/// at last. Asserting the descent HERE is what would catch a re-introduced dense force — the
/// symptom of which is silent: every job still renders, just on the 70.5 GB tier.
///
/// All three tiers are seeded COMPLETE so a result cannot be an accident of a tier being absent
/// (the completeness fallback would reach a sibling on its own — the shape that masked sc-13817
/// on a half-provisioned host).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn sensenova_candle_tier_follows_the_request_not_a_forced_dense_tier() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    for tier in ["q4", "q8", "bf16"] {
        std::fs::create_dir_all(root.join(tier)).unwrap();
        std::fs::write(root.join(tier).join("model.safetensors"), "weights").unwrap();
    }
    let req = |model: &str, advanced: serde_json::Value| {
        ImageRequest::from_payload(
            json!({
                "model": model,
                "advanced": advanced,
                "modelManifestEntry": { "mlx": { "standardTierLayout": true } }
            })
            .as_object()
            .unwrap(),
        )
    };
    for model in [
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
        "sensenova_u1_8b_infographic_v2",
        "sensenova_u1_8b_infographic_v3",
        "sensenova_u1_8b_infographic_v2_fast",
        "sensenova_u1_8b_infographic_v3_fast",
    ] {
        // An explicit pick is honored — the packed tiers are the point of the story.
        for (bits, tier) in [(4, "q4"), (8, "q8"), (0, "bf16")] {
            assert_eq!(
                standard_tier_subdir(root, &req(model, json!({ "mlxQuantize": bits }))),
                root.join(tier),
                "{model} with mlxQuantize {bits} must resolve {tier}"
            );
        }
        // ...and with no pick, the app-wide q8 default — NOT a forced bf16.
        assert_eq!(
            standard_tier_subdir(root, &req(model, json!({}))),
            root.join("q8"),
            "{model} default must be the shared q8, not a dense force"
        );
    }
}

/// sc-8746 on-device verify (MLX): drive the ACTUAL worker seam against a downloaded SceneWorks
/// realvisxl-mlx turnkey — `standard_tier_subdir` resolves the `q4/` subdir from the tier root,
/// then `crate::inference_runtime::load("sdxl", …)` with `Quant::Q4` loads the packed tier and renders. Asserts a
/// non-degenerate image (per-pixel std above the all-black/NaN floor). `#[ignore]`d — run by hand
/// on a Mac with the tier downloaded:
/// ```text
/// hf download SceneWorks/realvisxl-mlx --include "q4/*" --local-dir /tmp/realvisxl-q4
/// SDXL_TIER_ROOT=/tmp/realvisxl-q4 cargo test -p sceneworks-worker --lib \
///   sdxl_realvisxl_q4_tier_mlx_smoke -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight MLX smoke; needs a downloaded SceneWorks/realvisxl-mlx q4 tier (SDXL_TIER_ROOT)"]
fn sdxl_realvisxl_q4_tier_mlx_smoke() {
    use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

    let root = PathBuf::from(
        std::env::var("SDXL_TIER_ROOT")
            .expect("set SDXL_TIER_ROOT to the downloaded realvisxl-mlx tier root")
            .trim(),
    );
    // The worker resolution: a default `realvisxl` request (no mlxQuantize) prefers the q8 default
    // (sc-10726) but CLAMPS to installed — only the q4 tier was downloaded here (`--include "q4/*"`),
    // so it lands on q4/.
    let req = ImageRequest::from_payload(
        json!({ "model": "realvisxl", "advanced": {} })
            .as_object()
            .unwrap(),
    );
    let tier = standard_tier_subdir(&root, &req);
    assert_eq!(
        tier,
        root.join("q4"),
        "worker must resolve the q4 tier subdir"
    );
    assert!(
        tier.join("model_index.json").is_file() && tier.join("unet").is_dir(),
        "q4 tier subdir missing turnkey layout (model_index.json + unet/): {}",
        tier.display()
    );

    // Load the packed q4 tier through the MLX `sdxl` engine (Quant::Q4 = harmless no-op on the
    // already-packed weights) and render a 768x768 image.
    let spec = LoadSpec::new(WeightsSource::Dir(tier.clone())).with_quant(Quant::Q4);
    let generator =
        crate::inference_runtime::load("sdxl", &spec).expect("load MLX sdxl provider on q4 tier");
    let gen_req = GenerationRequest {
        prompt: "a photorealistic portrait of a red fox in a snowy forest, golden hour".to_owned(),
        width: 768,
        height: 768,
        count: 1,
        seed: Some(42),
        steps: Some(20),
        guidance: Some(7.0),
        ..Default::default()
    };
    let output = generator
        .generate(&gen_req, &mut |_p| {})
        .expect("sdxl q4 tier generate");
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("no image returned"),
        other => panic!("expected Images output, got {other:?}"),
    };
    // Cheap degenerate-floor check: an all-black / NaN-clamped decode collapses toward std 0.
    let n = image.pixels.len() as f64;
    assert!(n > 0.0, "empty image buffer");
    let mean = image.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let std = (image
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    println!(
        "[sc-8746 smoke] realvisxl q4 tier render {}x{} std {std:.2}",
        image.width, image.height
    );
    assert!(
        std > 5.0,
        "render looks degenerate (std {std:.2}) — possible NaN / all-black decode"
    );
}

/// A model built with a manifest entry so `uses_standard_tier_layout` / `is_dense_te_tier`
/// can be exercised without touching the hardcoded registries.
#[cfg(native_lane)]
fn manifest_request(model: &str, mlx: serde_json::Value) -> ImageRequest {
    ImageRequest::from_payload(
        json!({ "model": model, "modelManifestEntry": { "mlx": mlx } })
            .as_object()
            .unwrap(),
    )
}

/// sc-8508: the standard-tier routing is manifest-driven — a model NOT in
/// [`STANDARD_TIER_MODELS`] opts in via `mlx.standardTierLayout: true`, while a registry model
/// stays true without the flag and an ordinary model stays false. Back-compat guard.
#[cfg(native_lane)]
#[test]
fn standard_tier_layout_is_manifest_driven_with_registry_backcompat() {
    // Registry member, no manifest flag → still true.
    assert!(uses_standard_tier_layout(&manifest_request(
        "flux2_dev",
        json!({})
    )));
    // Novel id (not in the registry) opts in from the manifest alone.
    assert!(uses_standard_tier_layout(&manifest_request(
        "some_new_matrix_model",
        json!({ "standardTierLayout": true })
    )));
    // Novel id without the flag → not a standard-tier model.
    assert!(!uses_standard_tier_layout(&manifest_request(
        "some_dense_model",
        json!({})
    )));
}

/// sc-15799: the dense-TE guard is MANIFEST-ONLY. An above-tier residency the shared decision cannot
/// see is the defect that story removes, so the id alone must no longer grant the carve-out —
/// `flux2_klein_9b` gets it from the catalog (pinned against the shipped manifest by
/// `tests/gpu_and_manifest.rs`) or not at all.
#[cfg(native_lane)]
#[test]
fn dense_te_tier_is_declared_only_in_the_manifest() {
    assert!(
        !is_dense_te_tier(&manifest_request("flux2_klein_9b", json!({}))),
        "a bare id must NOT grant the carve-out any more — the deleted hardcoded registry is \
             exactly the invisible exception sc-15799 removes"
    );
    assert!(is_dense_te_tier(&manifest_request(
        "flux2_klein_9b",
        json!({ "denseTextEncoderTier": true })
    )));
    assert!(is_dense_te_tier(&manifest_request(
        "some_dense_te_model",
        json!({ "denseTextEncoderTier": true })
    )));
    assert!(!is_dense_te_tier(&manifest_request("flux2_dev", json!({}))));
}

/// sc-10614: the candle SDXL lanes (edit / IP-Adapter) read DENSE weights, so a tiered-turnkey
/// snapshot must resolve to its `bf16/` tier — its root holds no component tree, and the loader
/// would find no `unet/`. Flat upstream diffusers snapshots (what `sdxl` and `realvisxl` fall
/// back to today) must pass through untouched.
#[test]
fn dense_tier_subdir_descends_into_turnkeys_and_passes_flat_snapshots_through() {
    // Tiered turnkey: tier subdirs, no backbone at the root.
    let turnkey = tempfile::tempdir().unwrap();
    for tier in ["q4", "q8", "bf16"] {
        std::fs::create_dir_all(turnkey.path().join(tier).join("unet")).unwrap();
    }
    assert_eq!(
        dense_tier_subdir(turnkey.path().to_path_buf()),
        turnkey.path().join("bf16"),
        "an SDXL turnkey resolves to its dense bf16 tier, never a quantized one"
    );

    // Flat SDXL diffusers snapshot: `unet/` at the root.
    let flat_unet = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(flat_unet.path().join("unet")).unwrap();
    assert_eq!(
        dense_tier_subdir(flat_unet.path().to_path_buf()),
        flat_unet.path().to_path_buf(),
        "stabilityai/stable-diffusion-xl-base-1.0 and SG161222/RealVisXL_V5.0 pass through"
    );

    // Flat DiT snapshot: `transformer/` at the root — even alongside a stray `bf16/` dir.
    let flat_dit = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(flat_dit.path().join("transformer")).unwrap();
    std::fs::create_dir_all(flat_dit.path().join("bf16")).unwrap();
    assert_eq!(
        dense_tier_subdir(flat_dit.path().to_path_buf()),
        flat_dit.path().to_path_buf(),
        "a rooted component tree wins over a tier dir sitting beside it"
    );

    // Neither shape: return the root and let the loader raise a real load error.
    let bare = tempfile::tempdir().unwrap();
    assert_eq!(
        dense_tier_subdir(bare.path().to_path_buf()),
        bare.path().to_path_buf()
    );
}

/// sc-9092 (epic 9083 gap #3): the candle Lens lane no longer resolves a SEPARATE bf16 diffusers
/// rehost (`SceneWorks/Lens{,-Turbo}`, the retired `candle_lens_repo`) — it packed-loads the SAME
/// `SceneWorks/lens{,-turbo}-mlx` MLX turnkey the macOS path uses, routed through the shared
/// `standard_tier_subdir` (`lens`/`lens_turbo` opt in via `mlx.standardTierLayout`, exactly like the
/// MLX lane). This proves the shared tier resolver picks the requested q4/q8/bf16 subdir of a Lens
/// turnkey snapshot off-Mac — the candle-lane sibling of the SD3.5 `standard_tier_subdir` tests
/// above — so the retired resolver is fully replaced by the standard machinery.
#[test]
fn krea_model_subdir_selects_tier_and_falls_back_to_downloaded() {
    let krea_request = |bits: serde_json::Value| {
        ImageRequest::from_payload(
            json!({ "model": "krea_2_raw", "advanced": { "mlxQuantize": bits } })
                .as_object()
                .unwrap(),
        )
    };
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Krea turnkey: packed q4/q8 single-file transformer + dense sharded bf16 (index.json only).
    seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
    seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
    seed_tier(
        root,
        "bf16",
        "diffusion_pytorch_model.safetensors.index.json",
    );

    // q8-default (no selection) / q4 (bits<=4) / bf16 (bits<=0) each resolve to their tier subdir.
    assert_eq!(
        krea_model_subdir(root, &krea_request(json!(null))),
        root.join("q8")
    );
    assert_eq!(
        krea_model_subdir(root, &krea_request(json!(4))),
        root.join("q4")
    );
    assert_eq!(
        krea_model_subdir(root, &krea_request(json!(0))),
        root.join("bf16")
    );

    // Regression guard (epic 9992): with ONLY the `bf16/` training-base tier downloaded (the Path-1
    // unify scenario), a q8-default generation must fall back to the present bf16 tier — NOT the repo
    // root (which has no `tokenizer/`, the reported "tokenizer: No such file or directory" load error).
    let bf16_only = tempfile::tempdir().unwrap();
    seed_tier(
        bf16_only.path(),
        "bf16",
        "diffusion_pytorch_model.safetensors.index.json",
    );
    assert_eq!(
        krea_model_subdir(bf16_only.path(), &krea_request(json!(null))),
        bf16_only.path().join("bf16"),
        "q8-default must fall back to the only downloaded tier (bf16), not the repo root"
    );
}

/// Seed a diffusers-shaped tier: a backbone, a `model_index.json` declaring `declared` as
/// `[library, class]` components, and an on-disk dir for each of `on_disk`. The index also carries
/// the three NON-component value shapes a real one does (see [`is_component_entry`]) so every test
/// through this helper proves they aren't mistaken for required dirs.
fn seed_diffusers_tier(root: &Path, tier: &str, declared: &[&str], on_disk: &[&str]) {
    seed_tier(root, tier, "diffusion_pytorch_model.safetensors");
    let dir = root.join(tier);
    let mut index = serde_json::Map::new();
    index.insert("_class_name".to_owned(), json!("Krea2Pipeline"));
    // Config array + scalar (krea's real `text_encoder_select_layers` / `patch_size`), and an
    // ABSENT optional component (realvisxl's real `feature_extractor`). None is a required dir.
    index.insert("text_encoder_select_layers".to_owned(), json!([2, 5, 8]));
    index.insert("patch_size".to_owned(), json!(2));
    index.insert("feature_extractor".to_owned(), json!([null, null]));
    for component in declared {
        index.insert(
            (*component).to_owned(),
            json!(["transformers", "SomeClass"]),
        );
    }
    std::fs::write(
        dir.join("model_index.json"),
        serde_json::to_vec(&Value::Object(index)).unwrap(),
    )
    .unwrap();
    for component in on_disk {
        let component_dir = dir.join(component);
        std::fs::create_dir_all(&component_dir).unwrap();
        std::fs::write(component_dir.join("config.json"), b"{}").unwrap();
    }
}

/// The Krea tier component set, as the real `q8/model_index.json` declares it.
const KREA_COMPONENTS: &[&str] = &[
    "transformer",
    "tokenizer",
    "text_encoder",
    "vae",
    "scheduler",
];

/// sc-12279 (issue #850): a TORN tier — backbone landed, `tokenizer/` did not — must not
/// short-circuit the fallback chain. Before this, `present()` accepted a tier on its transformer
/// alone, so the q8 default resolved to the torn `q8/` and the loader died on
/// `tokenizer: No such file or directory (os error 2)` even though a complete `bf16/` was installed.
#[test]
fn krea_tier_probe_skips_a_torn_tier_for_a_complete_sibling() {
    let krea_request = |bits: serde_json::Value| {
        ImageRequest::from_payload(
            json!({ "model": "krea_2_raw", "advanced": { "mlxQuantize": bits } })
                .as_object()
                .unwrap(),
        )
    };
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // q8 (the default tier) is torn: everything but `tokenizer/`. bf16 is complete.
    seed_diffusers_tier(
        root,
        "q8",
        KREA_COMPONENTS,
        &["text_encoder", "vae", "scheduler"],
    );
    seed_diffusers_tier(
        root,
        "bf16",
        KREA_COMPONENTS,
        &["tokenizer", "text_encoder", "vae", "scheduler"],
    );
    assert_eq!(
        krea_model_subdir(root, &krea_request(json!(null))),
        root.join("bf16"),
        "the q8 default is torn (no tokenizer/), so the chain must land on the complete bf16 tier"
    );
    // An EXPLICIT pick of the torn tier is also redirected — the user asked for a tier that cannot
    // load, and a working render beats an os-error-2 on a request we can serve.
    assert_eq!(
        krea_model_subdir(root, &krea_request(json!(8))),
        root.join("bf16"),
        "an explicit q8 pick still skips the torn tier for the complete sibling"
    );
}

/// sc-12279: with NO complete tier, the torn one is still returned — pass 2 of the chain preserves
/// the pre-sc-12279 result exactly. The user gets the same loader error as before (they have no
/// loadable tier), never a silent no-op, and the resolver logs which tier is short.
#[test]
fn krea_tier_probe_still_returns_a_torn_tier_when_it_is_all_there_is() {
    let request = ImageRequest::from_payload(
        json!({ "model": "krea_2_raw", "advanced": {} })
            .as_object()
            .unwrap(),
    );
    let tmp = tempfile::tempdir().unwrap();
    seed_diffusers_tier(
        tmp.path(),
        "q8",
        KREA_COMPONENTS,
        &["text_encoder", "vae", "scheduler"],
    );
    assert_eq!(
        krea_model_subdir(tmp.path(), &request),
        tmp.path().join("q8"),
        "no complete tier: the torn tier still resolves (unchanged behavior), not the repo root"
    );
}

/// sc-12279: [`tier_components_present`] reads the tier's OWN `model_index.json`, and must not
/// mistake diffusers' non-component value shapes for required dirs. Each of these is live in a
/// shipping turnkey, so getting any wrong would fail a perfectly good tier.
#[test]
fn tier_components_present_reads_model_index_and_ignores_non_components() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    // Complete: every declared component on disk. The `[null, null]` optional, the config array and
    // the config scalar that `seed_diffusers_tier` always writes must NOT be required as dirs —
    // this asserting true IS the proof (no `feature_extractor/` dir exists).
    seed_diffusers_tier(
        root,
        "q8",
        KREA_COMPONENTS,
        &["tokenizer", "text_encoder", "vae", "scheduler"],
    );
    assert!(tier_components_present(&root.join("q8")));

    // Torn: `tokenizer/` declared but absent.
    seed_diffusers_tier(
        root,
        "q4",
        KREA_COMPONENTS,
        &["text_encoder", "vae", "scheduler"],
    );
    assert!(!tier_components_present(&root.join("q4")));

    // A component dir holding ONLY an AppleDouble sidecar has no tokenizer (SceneWorks#1333).
    let sidecar = root.join("q4").join("tokenizer");
    std::fs::create_dir_all(&sidecar).unwrap();
    std::fs::write(sidecar.join("._tokenizer.json"), b"x").unwrap();
    assert!(
        !tier_components_present(&root.join("q4")),
        "a hidden sidecar must not satisfy a declared component"
    );
    std::fs::write(sidecar.join("tokenizer.json"), b"{}").unwrap();
    assert!(tier_components_present(&root.join("q4")));

    // No `model_index.json` at all (flat unified turnkeys — SenseNova-U1 MoT roots its weights +
    // `tokenizer.json` directly in the tier dir): nothing to verify, so the backbone probe rules.
    seed_tier(root, "bf16", "model.safetensors");
    assert!(
        tier_components_present(&root.join("bf16")),
        "a tier with no model_index.json has no component tree to check"
    );
}

// ---- sc-12279 generalized to the no-`model_index.json` families (SANA / Boogu / Anima) ----
// These turnkeys ship no `model_index.json`, so the shared `tier_components_present` guard is a
// no-op for them and a torn tier used to short-circuit the chain and crash the loader on the first
// missing component. Each resolver now routes through a concrete per-family completeness predicate.

/// Seed a SANA MLX tier (`SceneWorks/Sana_*_mlx` layout): transformer + (when `complete`) the DC-AE
/// VAE and the Gemma-2 text encoder with its bundled tokenizer. No `model_index.json` (as shipped).
fn seed_sana_tier(root: &Path, tier: &str, complete: bool) {
    let dir = root.join(tier);
    std::fs::create_dir_all(dir.join("transformer")).unwrap();
    std::fs::write(
        dir.join("transformer/diffusion_pytorch_model.safetensors"),
        b"x",
    )
    .unwrap();
    if complete {
        std::fs::create_dir_all(dir.join("vae")).unwrap();
        std::fs::write(dir.join("vae/diffusion_pytorch_model.safetensors"), b"x").unwrap();
        std::fs::create_dir_all(dir.join("text_encoder")).unwrap();
        std::fs::write(dir.join("text_encoder/gemma-2-2b-it.safetensors"), b"x").unwrap();
        std::fs::write(dir.join("text_encoder/tokenizer.json"), b"{}").unwrap();
    }
}

/// A torn SANA tier (transformer only, TE/VAE absent) must fall through to a complete sibling rather
/// than reaching the loader, which would die on the missing Gemma text encoder. SANA ships no
/// `model_index.json`, so this is caught by the concrete `sana_tier_complete` check, not the no-op
/// `tier_components_present` guard.
#[test]
fn sana_torn_tier_falls_through_to_a_complete_sibling() {
    let request = |bits: serde_json::Value| {
        ImageRequest::from_payload(
            json!({ "model": "sana_1600m", "advanced": { "mlxQuantize": bits } })
                .as_object()
                .unwrap(),
        )
    };
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    seed_sana_tier(root, "q8", false); // torn: transformer only
    seed_sana_tier(root, "bf16", true); // complete
    assert_eq!(
            standard_tier_subdir(root, &request(json!(null))),
            root.join("bf16"),
            "the torn q8 default must skip to the complete bf16 tier, not crash on the missing Gemma TE"
        );
    assert_eq!(
        standard_tier_subdir(root, &request(json!(8))),
        root.join("bf16"),
        "an explicit q8 pick is redirected too — a working render beats an os-error-2 we can avoid"
    );
}

/// Regression contract: with NO complete SANA tier, the torn one still resolves (pass 2), unchanged
/// from before — the user gets the same loader error they would have, never a silent no-op.
#[test]
fn sana_torn_tier_is_returned_when_it_is_all_there_is() {
    let request = ImageRequest::from_payload(
        json!({ "model": "sana_1600m", "advanced": {} })
            .as_object()
            .unwrap(),
    );
    let tmp = tempfile::tempdir().unwrap();
    seed_sana_tier(tmp.path(), "q8", false);
    assert_eq!(
        standard_tier_subdir(tmp.path(), &request),
        tmp.path().join("q8"),
        "no complete tier: the torn tier still resolves, not the repo root"
    );
}

/// Non-SANA standard turnkeys (flux/qwen/…) are unaffected: they DO ship `model_index.json`, so
/// their completeness stays `tier_components_present` — the added SANA branch never touches them.
#[test]
fn non_sana_standard_turnkey_ignores_the_sana_completeness_branch() {
    let request = ImageRequest::from_payload(
        json!({ "model": "flux_dev", "advanced": {} })
            .as_object()
            .unwrap(),
    );
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // A flux tier with only a transformer and no text_encoder dir would FAIL sana_tier_complete, but
    // flux is not SANA — it resolves on its backbone/model_index exactly as before.
    seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
    assert_eq!(standard_tier_subdir(root, &request), root.join("q8"));
}

/// Seed a Boogu tier folder (`<variant>` / `<variant>-q4` / `<variant>-bf16`): packed transformer +
/// its config, and (when `complete`) the Qwen3-VL `mllm/` with tokenizer and the VAE. No index.
fn seed_boogu_tier(root: &Path, folder: &str, complete: bool) {
    let dir = root.join(folder);
    std::fs::create_dir_all(dir.join("transformer")).unwrap();
    std::fs::write(
        dir.join("transformer/diffusion_pytorch_model.safetensors"),
        b"x",
    )
    .unwrap();
    std::fs::write(dir.join("transformer/config.json"), b"{}").unwrap();
    if complete {
        std::fs::create_dir_all(dir.join("mllm")).unwrap();
        std::fs::write(dir.join("mllm/model.safetensors"), b"x").unwrap();
        std::fs::write(dir.join("mllm/tokenizer.json"), b"{}").unwrap();
        std::fs::create_dir_all(dir.join("vae")).unwrap();
        std::fs::write(dir.join("vae/diffusion_pytorch_model.safetensors"), b"x").unwrap();
    }
}

/// A torn Boogu tier (transformer present, `mllm/tokenizer.json` + VAE absent) must fall through to a
/// complete sibling rather than crash the loader on the first-read `mllm/tokenizer.json`.
#[test]
fn boogu_torn_tier_falls_through_to_a_complete_sibling() {
    let request = ImageRequest::from_payload(
        json!({ "model": "boogu_image", "advanced": {} })
            .as_object()
            .unwrap(),
    );
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // `base` is the Q8 default folder (torn); `base-bf16` is complete.
    seed_boogu_tier(root, "base", false);
    seed_boogu_tier(root, "base-bf16", true);
    assert_eq!(
            boogu_model_subdir(root, &request),
            root.join("base-bf16"),
            "the torn Q8 `base/` must skip to the complete `base-bf16`, not crash on `mllm/tokenizer.json`"
        );
}

/// `resolved_tier_is_complete` (the pre-flight completeness dispatcher behind the friendly
/// `mlx_weights_gap` message) reports a torn SANA tier as incomplete and a whole one as complete.
#[cfg(target_os = "macos")]
#[test]
fn resolved_tier_is_complete_flags_a_torn_sana_tier() {
    let request = ImageRequest::from_payload(
        json!({ "model": "sana_1600m", "advanced": {} })
            .as_object()
            .unwrap(),
    );
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    seed_sana_tier(root, "q8", false);
    seed_sana_tier(root, "bf16", true);
    assert!(!resolved_tier_is_complete(&request, &root.join("q8")));
    assert!(resolved_tier_is_complete(&request, &root.join("bf16")));
}

/// The same pre-flight dispatcher for SenseNova-U1 (sc-14432): a torn tier reads incomplete, and the
/// distilled `_fast` id additionally requires the pre-merged `distill_merged.json` marker — the base
/// id must NOT, or every base install would report a spurious gap.
#[cfg(target_os = "macos")]
#[test]
fn resolved_tier_is_complete_flags_a_torn_sensenova_tier_and_a_fast_tier_missing_its_marker() {
    let sensenova = |model: &str| {
        ImageRequest::from_payload(
            json!({ "model": model, "advanced": {} })
                .as_object()
                .unwrap(),
        )
    };
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let seed = |tier: &str, complete: bool| {
        let dir = root.join(tier);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
        if complete {
            std::fs::write(dir.join("config.json"), b"x").unwrap();
            std::fs::write(dir.join("tokenizer.json"), b"x").unwrap();
        }
    };
    seed("q8", false);
    seed("bf16", true);

    let base = sensenova("sensenova_u1_8b");
    assert!(!resolved_tier_is_complete(&base, &root.join("q8")));
    assert!(resolved_tier_is_complete(&base, &root.join("bf16")));

    // The bf16 tier satisfies the BASE contract but has no distill marker, so it is complete for the
    // base id and incomplete for the fast one — the discriminating pair for the fast arm.
    let fast = sensenova("sensenova_u1_8b_fast");
    assert!(!resolved_tier_is_complete(&fast, &root.join("bf16")));
    std::fs::write(root.join("bf16").join("distill_merged.json"), b"{}").unwrap();
    assert!(resolved_tier_is_complete(&fast, &root.join("bf16")));
}

/// Seed an Anima tier (`bf16/ q8/ q4/`, `split_files` shape): the DiT + (when `complete`) the dense
/// text encoder and the VAE. Tokenizers are vendored into the binary, so none is on disk. No index.
#[cfg(native_lane)]
fn seed_anima_tier(root: &Path, tier: &str, complete: bool) {
    let dir = root.join(tier);
    std::fs::create_dir_all(dir.join("diffusion_models")).unwrap();
    std::fs::write(
        dir.join("diffusion_models/anima-base-v1.0.safetensors"),
        b"x",
    )
    .unwrap();
    if complete {
        std::fs::create_dir_all(dir.join("text_encoders")).unwrap();
        std::fs::write(dir.join("text_encoders/qwen_3_06b_base.safetensors"), b"x").unwrap();
        std::fs::create_dir_all(dir.join("vae")).unwrap();
        std::fs::write(dir.join("vae/qwen_image_vae.safetensors"), b"x").unwrap();
    }
}

/// A torn Anima tier (DiT present, text-encoder/VAE absent) must fall through to a complete sibling
/// rather than reaching the loader, which would die on the missing `text_encoders/…` (mlx-gen-anima).
#[cfg(native_lane)]
#[test]
fn anima_torn_tier_falls_through_to_a_complete_sibling() {
    let request = ImageRequest::from_payload(
        json!({ "model": "anima_base", "advanced": {} })
            .as_object()
            .unwrap(),
    );
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    seed_anima_tier(root, "q8", false); // torn: DiT only
    seed_anima_tier(root, "bf16", true); // complete
    assert_eq!(
            anima_tier_subdir(root, &request),
            root.join("bf16"),
            "the torn q8 default must skip to the complete bf16 tier, not crash on the missing text encoder"
        );
}

/// sc-10845 (epic 10721): [`krea_model_subdir`] — the last bespoke tier resolver — routes its DEFAULT
/// through the shared, floor-aware [`preferred_tier`]`(bits, `[`min_quality_floor`]`(request))`, exactly
/// like [`standard_tier_subdir`] / [`anima_tier_subdir`] / [`ideogram_model_subdir`] /
/// [`boogu_model_subdir`], so a floored model's picker-less default clamps UP to `mlx.minQualityTier`
/// (capped by installed) instead of blindly landing on the q8 default. No shipping Krea model declares
/// a floor today, so every current model is byte-identical to the prior q8 default; this pins the
/// generality. Mirrors [`standard_tier_subdir_clamps_default_to_the_floor`].
#[test]
fn krea_default_tier_clamps_to_the_floor() {
    // A krea request with the given `bits` (null = no explicit pick) and an optional per-model quality
    // floor forwarded via `modelManifestEntry.mlx.minQualityTier`.
    let floored = |bits: serde_json::Value, floor: Option<&str>| {
        let payload = match floor {
            Some(f) => json!({
                "model": "krea_2_raw",
                "advanced": { "mlxQuantize": bits },
                "modelManifestEntry": { "mlx": { "minQualityTier": f } },
            }),
            None => json!({
                "model": "krea_2_raw",
                "advanced": { "mlxQuantize": bits },
            }),
        };
        ImageRequest::from_payload(payload.as_object().unwrap())
    };
    // Krea turnkey carries packed q4/q8 + dense sharded bf16.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
    seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
    seed_tier(
        root,
        "bf16",
        "diffusion_pytorch_model.safetensors.index.json",
    );

    // Floor bf16, no explicit pick → the dense bf16 (raised above the q8 default).
    assert_eq!(
        krea_model_subdir(root, &floored(json!(null), Some("bf16"))),
        root.join("bf16"),
        "floored Krea default → the bf16 floor tier (raised above the q8 default)"
    );
    // Floor q8 (at the default) → q8, unchanged (the floor only ever RAISES).
    assert_eq!(
        krea_model_subdir(root, &floored(json!(null), Some("q8"))),
        root.join("q8")
    );
    // An explicit below-floor q4 pick is HONORED even against a bf16 floor (the worker never overrides
    // a deliberate quant choice — the web surfaces the advisory instead).
    assert_eq!(
        krea_model_subdir(root, &floored(json!(4), Some("bf16"))),
        root.join("q4")
    );
    // Non-floored default is unchanged — the q8 default still stands.
    assert_eq!(
        krea_model_subdir(root, &floored(json!(null), None)),
        root.join("q8")
    );

    // Floor capped by installed: bf16 floor but only q8 on disk → q8.
    let only_q8 = tempfile::tempdir().unwrap();
    seed_tier(only_q8.path(), "q8", "diffusion_pytorch_model.safetensors");
    assert_eq!(
            krea_model_subdir(only_q8.path(), &floored(json!(null), Some("bf16"))),
            only_q8.path().join("q8"),
            "bf16 floor with only q8 installed → q8 (a floor tier not on disk falls to the best installed)"
        );
}

#[test]
fn candle_lens_resolves_packed_turnkey_tier_subdir() {
    let lens_request = |bits: serde_json::Value| {
        ImageRequest::from_payload(
            json!({ "model": "lens", "advanced": { "mlxQuantize": bits } })
                .as_object()
                .unwrap(),
        )
    };
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // A Lens turnkey ships packed per-tier subdirs (transformer + gpt-oss-20b MoE TE + FLUX.2 VAE).
    seed_tier(root, "q4", "diffusion_pytorch_model.safetensors");
    seed_tier(root, "q8", "diffusion_pytorch_model.safetensors");
    seed_tier(
        root,
        "bf16",
        "diffusion_pytorch_model.safetensors.index.json",
    );

    // A/B tier toggle: default (q8, sc-10726) / mlxQuantize:8 (q8) / mlxQuantize:0 (bf16) each
    // resolve to their tier subdir — the default now prefers the clean q8 tier (clamped to
    // installed), matching `resolve_quant`'s Q8 default.
    assert_eq!(
        standard_tier_subdir(root, &lens_request(json!(null))),
        root.join("q8")
    );
    assert_eq!(
        standard_tier_subdir(root, &lens_request(json!(8))),
        root.join("q8")
    );
    assert_eq!(
        standard_tier_subdir(root, &lens_request(json!(0))),
        root.join("bf16")
    );
}

/// sc-9092 (epic 9083 gap #3, review fix): Ideogram + Boogu were left `macOnly:true` with only
/// off-Mac diffusers download entries, which the PR deleted — so off-Mac they resolved to the MLX
/// turnkey (`SceneWorks/{ideogram-4,boogu-image}-mlx`) whose download entries were macOS-only and
/// thus never fetched (a "candle weights snapshot not found" load error). The fix flips them
/// `macOnly:false` and extends the turnkey download `platforms` to windows/linux, so both lanes
/// packed-load the SAME turnkey the macOS path uses (candle-gen sc-9412 / sc-9410). This asserts the
/// per-family subdir resolvers pick the shipped tier of a turnkey snapshot regardless of platform —
/// the ideogram/boogu sibling of `candle_lens_resolves_packed_turnkey_tier_subdir` above.
#[test]
fn candle_ideogram_boogu_resolve_packed_turnkey_tier_subdir() {
    let model_request = |model: &str, bits: serde_json::Value| {
        ImageRequest::from_payload(
            json!({ "model": model, "advanced": { "mlxQuantize": bits } })
                .as_object()
                .unwrap(),
        )
    };

    // Ideogram turnkey (`SceneWorks/ideogram-4-mlx`): q4 + q8 tiers. `ideogram_model_subdir` probes
    // `<tier>/transformer/model.safetensors`.
    {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_tier(root, "q4", "model.safetensors");
        seed_tier(root, "q8", "model.safetensors");
        for model in ["ideogram_4", "ideogram_4_turbo"] {
            // Default (no mlxQuantize) → q8 when installed (epic 10721 / sc-10726).
            assert_eq!(
                ideogram_model_subdir(root, &model_request(model, json!(null))),
                root.join("q8")
            );
            // mlxQuantize:8 → q8 when present.
            assert_eq!(
                ideogram_model_subdir(root, &model_request(model, json!(8))),
                root.join("q8")
            );
            // An explicit Q4 pick is still honored.
            assert_eq!(
                ideogram_model_subdir(root, &model_request(model, json!(4))),
                root.join("q4")
            );
        }
    }

    // Boogu turnkey (`SceneWorks/boogu-image-mlx`): Q8 `base/`/`turbo/`/`edit/` is the shipped
    // (off-Mac) default. `boogu_model_subdir` probes `<variant>/transformer/diffusion_pytorch_model.safetensors`.
    {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for variant in ["base", "turbo", "edit"] {
            seed_tier(root, variant, "diffusion_pytorch_model.safetensors");
        }
        for (model, variant) in [
            ("boogu_image", "base"),
            ("boogu_image_turbo", "turbo"),
            ("boogu_image_edit", "edit"),
        ] {
            // Default (no mlxQuantize) → the shipped Q8 `<variant>/` subfolder.
            assert_eq!(
                boogu_model_subdir(root, &model_request(model, json!(null))),
                root.join(variant)
            );
        }
    }
}

/// sc-10777 (epic 10721): [`ideogram_model_subdir`] / [`boogu_model_subdir`] route their DEFAULT tier
/// through the shared, floor-aware [`preferred_tier`]`(bits, `[`min_quality_floor`]`(request))` — the
/// same clamp [`standard_tier_subdir`] / [`anima_tier_subdir`] use — so a floored model's picker-less
/// default clamps UP to `mlx.minQualityTier` (capped by installed) instead of blindly landing on the
/// per-family q8/Q8 default. No shipping Ideogram/Boogu model declares a floor today, so every current
/// model is byte-identical to the prior default; this pins the generality so a future floored model
/// can't silently regress below its floor on the worker default path. Mirrors
/// [`standard_tier_subdir_clamps_default_to_the_floor`].
#[test]
fn ideogram_boogu_default_tier_clamps_to_the_floor() {
    // A request for `model` with the given `bits` (null = no explicit pick) and an optional per-model
    // quality floor forwarded via `modelManifestEntry.mlx.minQualityTier`.
    let floored = |model: &str, bits: serde_json::Value, floor: Option<&str>| {
        let payload = match floor {
            Some(f) => json!({
                "model": model,
                "advanced": { "mlxQuantize": bits },
                "modelManifestEntry": { "mlx": { "minQualityTier": f } },
            }),
            None => json!({
                "model": model,
                "advanced": { "mlxQuantize": bits },
            }),
        };
        ImageRequest::from_payload(payload.as_object().unwrap())
    };

    // --- Boogu: a bf16 floor RAISES the picker-less default above the packed Q8 (the meaningful case,
    // since Boogu's turnkey carries a bf16 tier above its Q8 default). ---
    {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_tier(root, "base", "diffusion_pytorch_model.safetensors");
        seed_tier(root, "base-q4", "diffusion_pytorch_model.safetensors");
        // bf16 ships as the dense diffusers tree (SHARDED → only the `.index.json`), like the loader.
        seed_tier(
            root,
            "base-bf16",
            "diffusion_pytorch_model.safetensors.index.json",
        );
        // Floor bf16, no explicit pick → the dense `base-bf16` (raised above the Q8 default).
        assert_eq!(
            boogu_model_subdir(root, &floored("boogu_image", json!(null), Some("bf16"))),
            root.join("base-bf16"),
            "floored Boogu default → the bf16 floor tier (raised above the packed Q8 default)"
        );
        // Floor q8 (at the default) → the packed Q8 `base/`, unchanged (the floor only ever RAISES).
        assert_eq!(
            boogu_model_subdir(root, &floored("boogu_image", json!(null), Some("q8"))),
            root.join("base")
        );
        // An explicit below-floor q4 pick is HONORED even against a bf16 floor (the worker never
        // overrides a deliberate quant choice — the web surfaces the advisory instead).
        assert_eq!(
            boogu_model_subdir(root, &floored("boogu_image", json!(4), Some("bf16"))),
            root.join("base-q4")
        );
        // Non-floored default is unchanged — the packed Q8 default still stands (acceptance #3).
        assert_eq!(
            boogu_model_subdir(root, &floored("boogu_image", json!(null), None)),
            root.join("base")
        );
    }
    // Boogu floor capped by installed: bf16 floor but only the packed Q8 `base/` on disk → Q8.
    {
        let only_q8 = tempfile::tempdir().unwrap();
        seed_tier(
            only_q8.path(),
            "base",
            "diffusion_pytorch_model.safetensors",
        );
        assert_eq!(
                boogu_model_subdir(
                    only_q8.path(),
                    &floored("boogu_image", json!(null), Some("bf16"))
                ),
                only_q8.path().join("base"),
                "bf16 floor with only Q8 installed → Q8 (a floor tier not on disk falls to the best installed)"
            );
    }

    // --- Ideogram: the turnkey carries only `q4/`/`q8/` (bf16 is a separate repo) and the default
    // already prefers q8, so the floor routing is inert for every in-turnkey floor — but it must still
    // resolve correctly and stay capped by installed. ---
    {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_tier(root, "q4", "model.safetensors");
        seed_tier(root, "q8", "model.safetensors");
        // Floor q8, no explicit pick → q8 (the default already prefers q8; confirms the floor routing
        // resolves through the shared helper).
        assert_eq!(
            ideogram_model_subdir(root, &floored("ideogram_4", json!(null), Some("q8"))),
            root.join("q8")
        );
        // A bf16 floor has NO in-turnkey tier → falls through the clean q8 → q4 chain to the best
        // installed (q8), never erroring on the absent bf16.
        assert_eq!(
            ideogram_model_subdir(root, &floored("ideogram_4", json!(null), Some("bf16"))),
            root.join("q8"),
            "bf16 floor with no in-turnkey bf16 tier → best installed (q8)"
        );
        // An explicit below-floor q4 pick is HONORED even against a q8 floor.
        assert_eq!(
            ideogram_model_subdir(root, &floored("ideogram_4", json!(4), Some("q8"))),
            root.join("q4")
        );
        // Non-floored default is unchanged — q8 when installed (acceptance #3).
        assert_eq!(
            ideogram_model_subdir(root, &floored("ideogram_4", json!(null), None)),
            root.join("q8")
        );
    }
    // Ideogram floor capped by installed: q8 floor but only the shipped q4 on disk → q4.
    {
        let only_q4 = tempfile::tempdir().unwrap();
        seed_tier(only_q4.path(), "q4", "model.safetensors");
        assert_eq!(
                ideogram_model_subdir(only_q4.path(), &floored("ideogram_4", json!(null), Some("q8"))),
                only_q4.path().join("q4"),
                "q8 floor with only q4 installed → q4 (a floor tier not on disk falls to the best installed)"
            );
    }
}
