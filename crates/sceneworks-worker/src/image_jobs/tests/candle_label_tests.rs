use super::*;

#[test]
fn moss_edit_route_is_named_counted_labeled_and_separate_from_ordinary() {
    let temp = tempfile::tempdir().unwrap();
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.backend_candle_enabled = true;
    for model in ["moss_image_base", "moss_image_sft"] {
        let request = ImageRequest::from_payload(
            json!({"model":model,"mode":"edit_image",
                "sourceAssetId":"ref","prompts":["blue cup","red cup"],"prompt":"blue cup","count":2
            })
            .as_object()
            .unwrap(),
        );
        let route = resolve_candle_image_route_with_prepared_availability(
            &request, &settings, false, false, false, false, false, false, false,
        )
        .unwrap();
        assert_eq!(route, CandleImageRoute::MossImageEdit);
        assert_eq!(route.image_count(&request, &settings), 4);
        assert_eq!(route.adapter_label(&request), candle_adapter_label(model));
        assert!(!route.applies_request_loras(&request));
        let ordinary = ImageRequest::from_payload(
            json!({"model":model,"mode":"text_to_image"})
                .as_object()
                .unwrap(),
        );
        assert_eq!(
            resolve_candle_image_route_with_prepared_availability(
                &ordinary, &settings, false, false, false, false, false, false, false
            ),
            Some(CandleImageRoute::CandleTxt2Img)
        );
    }
    settings.backend_candle_enabled = false;
    let request = ImageRequest::from_payload(
        json!({"model":"moss_image_sft","mode":"edit_image","sourceAssetId":"ref"})
            .as_object()
            .unwrap(),
    );
    assert_eq!(
        resolve_candle_image_route_with_prepared_availability(
            &request, &settings, false, false, false, false, false, false, false
        ),
        None
    );
}

/// sc-24163: a LoRA/LoKr on a Qwen-Image 2.1 reference edit must pass the fail-closed route
/// guard — the edit route renders through the same adapter-bearing generic stream as 2.1 T2I.
#[test]
fn qwen_image_2_1_edit_route_admits_the_request_adapter_stack() {
    let request = ImageRequest::from_payload(
        json!({
            "model": "qwen_image_2_1",
            "mode": "edit_image",
            "referenceAssetIds": ["asset_a", "asset_b"],
            "loras": [{ "id": "edit_style", "weight": 1.0 }]
        })
        .as_object()
        .unwrap(),
    );
    assert!(
        CandleImageRoute::CandleTxt2Img.applies_request_loras(&request),
        "precondition: 2.1 T2I consumes the adapter stack"
    );
    assert!(
        CandleImageRoute::QwenImage21Edit.applies_request_loras(&request),
        "the 2.1 edit route must consume the same stack instead of refusing it"
    );
}

#[test]
fn krea_convrot_load_spec_preserves_adapters() {
    let request = ImageRequest::from_payload(
        json!({
            "model": "krea_2_turbo",
            "advanced": { "convRot": true }
        })
        .as_object()
        .unwrap(),
    );
    assert!(
        CandleImageRoute::CandleTxt2Img.applies_request_loras(&request),
        "the fail-closed route guard must admit ConvRot adapters before the load seam"
    );
    assert!(
        CandleImageRoute::KreaImported.applies_request_loras(&request),
        "the imported Krea lane must preserve the user adapters added by sc-18480"
    );
    assert!(
        CandleImageRoute::KolorsEdit.applies_request_loras(&request),
        "the registered Kolors edit stream consumes the same adapter-bearing load spec"
    );
    assert!(
        !CandleImageRoute::SenseNovaEdit.applies_request_loras(&request),
        "SenseNova descriptors expose no user-adapter slot"
    );
    assert!(
            !CandleImageRoute::FluxIpAdapter.applies_request_loras(&request),
            "the bespoke FLUX IP-Adapter provider has no user-adapter input, so the route guard must refuse LoRAs"
        );

    let adapter = AdapterSpec::new(
        PathBuf::from("/nonexistent/krea-style-lora.safetensors"),
        0.6,
        AdapterKind::Lora,
    );
    let spec = load_spec(
        PathBuf::from("/nonexistent/krea-bf16-surface"),
        None,
        vec![adapter.clone()],
        None,
    );
    let convrot = PathBuf::from("/nonexistent/krea-int8-convrot.safetensors");
    // The ConvRot DiT rides its own named component slot, NOT `text_encoder`: the pinned
    // gen-core reserves `text_encoder` for the manifest-selected encoder receipt.
    let spec = spec.with_component(
        gen_core::KREA_CONVROT_DIT_COMPONENT,
        WeightsSource::File(convrot.clone()),
    );

    assert!(matches!(
        spec.components.get(gen_core::KREA_CONVROT_DIT_COMPONENT),
        Some(WeightsSource::File(path)) if path == &convrot
    ));
    assert!(
        spec.text_encoder.is_none(),
        "the ConvRot DiT must not squat on the selected-text-encoder slot"
    );
    assert_eq!(spec.adapters.len(), 1);
    assert_eq!(spec.adapters[0].path, adapter.path);
}

#[test]
fn registered_zimage_identity_routes_scored_source_metadata() {
    let request = ImageRequest::from_payload(
        json!({
            "model": "z_image_turbo",
            "mode": "character_image",
            "referenceAssetId": "character-42",
            "advanced": { "referenceStrength": 0.7 }
        })
        .as_object()
        .unwrap(),
    );
    assert!(zimage_identity_candle_strength(&request).is_some());
    assert!(character_image_likeness_requested(&request));

    let fact = json!({ "sourceAssetId": "character-42" })
        .as_object()
        .unwrap()
        .clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel(2);
    drive_gen_items_scored_reported(
        tx,
        [()],
        move |_index, (), _preview, _reports, _progress| {
            Ok(Some((17, 1, 1, vec![0_u8; 3], Some(fact.clone()))))
        },
    )
    .unwrap();
    match rx.try_recv().expect("scored image event") {
        GenEvent::Image {
            face_likeness: Some(face_likeness),
            ..
        } => assert_eq!(face_likeness["sourceAssetId"], "character-42"),
        _ => panic!("expected scored image event"),
    }
}

#[test]
fn mage_edit_preserves_primary_then_ordered_extra_references() {
    let request = ImageRequest::from_payload(
        json!({
            "mode": "edit_image",
            "model": "mage_flow_edit_turbo",
            "sourceAssetId": "primary",
            "referenceAssetIds": ["second", "third"]
        })
        .as_object()
        .unwrap(),
    );
    assert_eq!(
        mage_edit_reference_ids(&request),
        ["primary", "second", "third"]
    );
    assert_eq!(
        CandleImageRoute::MageEdit.adapter_label(&request),
        "candle_mage",
        "Mage edit assets and generation sets must retain the Candle family label"
    );

    let missing_primary = ImageRequest::from_payload(
        json!({
            "mode": "edit_image",
            "model": "mage_flow_edit",
            "referenceAssetIds": ["orphan"]
        })
        .as_object()
        .unwrap(),
    );
    assert!(
        mage_edit_reference_ids(&missing_primary).is_empty(),
        "Mage edit must not reinterpret an optional reference as its required primary source"
    );
}

#[test]
fn candle_image_adapter_labels_are_per_family() {
    assert_eq!(candle_adapter_label("z_image_turbo"), "candle_z_image");
    assert_eq!(candle_adapter_label("flux_schnell"), "candle_flux");
    assert_eq!(candle_adapter_label("flux_dev"), "candle_flux");
    assert_eq!(candle_adapter_label("flux2_klein_9b"), "candle_flux2");
    assert_eq!(candle_adapter_label("flux2_dev"), "candle_flux2");
    // sc-7459: the klein weight variants share the FLUX.2 family label.
    assert_eq!(candle_adapter_label("flux2_klein_9b_kv"), "candle_flux2");
    assert_eq!(
        candle_adapter_label("flux2_klein_9b_true_v2"),
        "candle_flux2"
    );
    assert_eq!(candle_adapter_label("qwen_image"), "candle_qwen");
    assert_eq!(candle_adapter_label("chroma1_hd"), "candle_chroma");
    assert_eq!(candle_adapter_label("chroma1_base"), "candle_chroma");
    assert_eq!(candle_adapter_label("chroma1_flash"), "candle_chroma");
    assert_eq!(candle_adapter_label("lens"), "candle_lens");
    assert_eq!(candle_adapter_label("lens_turbo"), "candle_lens");
    assert_eq!(candle_adapter_label("kolors"), "candle_kolors");
    assert_eq!(candle_adapter_label("sensenova_u1_8b"), "candle_sensenova");
    assert_eq!(
        candle_adapter_label("sensenova_u1_8b_fast"),
        "candle_sensenova"
    );
    assert_eq!(candle_adapter_label("ideogram_4"), "candle_ideogram");
    assert_eq!(candle_adapter_label("ideogram_4_turbo"), "candle_ideogram");
    // Boogu (sc-7524): all three variants share the `candle_boogu` asset stamp.
    assert_eq!(candle_adapter_label("boogu_image"), "candle_boogu");
    assert_eq!(candle_adapter_label("boogu_image_turbo"), "candle_boogu");
    assert_eq!(candle_adapter_label("boogu_image_edit"), "candle_boogu");
    for model in [
        "mage_flow_base",
        "mage_flow",
        "mage_flow_turbo",
        "mage_flow_edit_base",
        "mage_flow_edit",
        "mage_flow_edit_turbo",
    ] {
        assert_eq!(candle_adapter_label(model), "candle_mage");
    }
    // Krea 2 Turbo (sc-7581): the candle asset stamp.
    assert_eq!(candle_adapter_label("krea_2_turbo"), "candle_krea");
    // Supra2-IMG (sc-24214): its own stamp, never the `candle_sdxl` fallback.
    assert_eq!(candle_adapter_label("supra2_img"), "candle_supra2");
    assert_eq!(candle_adapter_label("sdxl"), "candle_sdxl");
    assert_eq!(candle_adapter_label("realvisxl"), "candle_sdxl");
    assert_eq!(candle_adapter_label("iris_3b"), "candle_iris");
    assert_eq!(candle_adapter_label("bernini_image"), "candle_bernini");
    // Only the SDXL-engine ids may fall through to `candle_sdxl`; any other candle-routed model
    // landing there is missing from the map and would stamp the wrong provenance.
    const SDXL_ENGINE_IDS: &[&str] = &[
        "sdxl",
        "realvisxl",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
        "realvisxl_lightning",
    ];
    for model in sceneworks_core::jobs_store::candle_routed_image_models() {
        if !SDXL_ENGINE_IDS.contains(model) {
            assert_ne!(candle_adapter_label(model), CANDLE_ADAPTER, "{model}");
        }
    }
    // Every wired engine carries a `candle_`-prefixed label, distinct from the `mlx_` labels.
    for model in [
        "z_image_turbo",
        "flux_schnell",
        "flux_dev",
        "flux2_klein_9b",
        "flux2_dev",
        "qwen_image",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "lens",
        "lens_turbo",
        "kolors",
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
        "ideogram_4",
        "ideogram_4_turbo",
        "boogu_image",
        "boogu_image_turbo",
        "boogu_image_edit",
        // SD3.5 (sc-7880): Large / Large Turbo / Medium carry the `candle_sd3` stamp.
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "sdxl",
        "realvisxl",
    ] {
        assert!(candle_adapter_label(model).starts_with("candle_"));
    }
    // SD3.5 (sc-7880): the candle asset stamp.
    assert_eq!(candle_adapter_label("sd3_5_large"), "candle_sd3");
    assert_eq!(candle_adapter_label("sd3_5_large_turbo"), "candle_sd3");
    assert_eq!(candle_adapter_label("sd3_5_medium"), "candle_sd3");
    // Anima 2B (sc-10676): the candle asset stamp, the off-Mac sibling of `mlx_anima`.
    assert_eq!(candle_adapter_label("anima_base"), "candle_anima");
    assert_eq!(candle_adapter_label("anima_aesthetic"), "candle_anima");
    assert_eq!(candle_adapter_label("anima_turbo"), "candle_anima");
}

/// sc-12090: the reject tail suggests only INSTALLED, smaller tiers — never the rejected tier, and
/// never the picker (hidden when ≤1 tier is installed, the case that produced the misleading
/// "Pick a lower tier (Q4/Q8)" on the #1516 q4-only install).
#[test]
fn vram_reject_tail_names_only_installed_smaller_tiers() {
    // Two smaller tiers installed → both offered, uppercased, highest-fidelity first.
    let tail = vram_reject_tail(&["q8", "q4"], ResolutionLever::Priced);
    assert!(
        tail.contains("Q8 / Q4"),
        "lists installed smaller tiers: {tail}"
    );
    assert!(
        !tail.contains("picker"),
        "never points at the picker: {tail}"
    );
    // One smaller tier installed.
    assert!(vram_reject_tail(&["q4"], ResolutionLever::Priced).contains("(Q4)"));
    // None smaller installed (the single-tier / q4-only case) → says so, no tier list, no picker.
    let none = vram_reject_tail(&[], ResolutionLever::Priced);
    assert!(none.contains("No smaller tier is installed"), "{none}");
    assert!(!none.contains("Select a smaller"), "{none}");
}

/// Issue #3197: the tail is printed under numbers `predicted_peak_gb` /
/// `predicted_sequential_peak_gb` derive from a manifest row alone, so it must not send the
/// operator to the resolution knob — and must say that outright, since the estimate reads as
/// though it were activation-shaped.
#[test]
fn vram_reject_tail_never_offers_the_resolution_knob() {
    for tail in [
        vram_reject_tail(&[], ResolutionLever::Blind),
        vram_reject_tail(&["q4"], ResolutionLever::Blind),
        vram_reject_tail(&["q8", "q4"], ResolutionLever::Blind),
    ] {
        assert!(
            !tail.contains("lower the output resolution"),
            "resolution advice is false against a per-tier row: {tail}"
        );
        assert!(
            tail.contains("Lowering the output resolution will not help"),
            "the retry this costs is worth one sentence: {tail}"
        );
        assert!(
            tail.contains("run on a GPU with more VRAM"),
            "the card lever survives: {tail}"
        );
    }
    // The lane that DOES price geometry keeps the knob, verbatim as it has always read — the
    // rule is "name a knob only when the gate priced it", not "never mention resolution".
    for tail in [
        vram_reject_tail(&[], ResolutionLever::Priced),
        vram_reject_tail(&["q8", "q4"], ResolutionLever::Priced),
    ] {
        assert!(
            tail.contains("lower the output resolution"),
            "a geometry-aware reject still offers the real lever: {tail}"
        );
        assert!(
            !tail.contains("will not help"),
            "and never explains away a knob it just offered: {tail}"
        );
    }
}

/// sc-10733 / issue #3197: the capability clamp rejects on `candle_tier_fit`, which reads
/// `candle.vramGbByTier` and never the request geometry. The message may therefore name the
/// card, must not name the resolution, and must not offer a smaller tier — `smallest_tier` is
/// already the least-demanding tier the clamp scored.
#[test]
fn capability_reject_offers_only_truthful_escapes() {
    // SDXL Q4 on the reporter's 8 GB card: the q4 row (12) + HEADROOM_GB (2) against ~6 GB free.
    let message =
        candle_capability_reject_message("sdxl", "q4", 14.0, 6.0, "0", ResolutionLever::Blind);
    assert!(
        message.starts_with(
            "sdxl needs ~14 GB of VRAM even at the smallest installed tier it can run (q4)"
        ),
        "{message}"
    );
    assert!(message.contains("GPU 0 has ~6 GB available."), "{message}");
    // #3197 read that ~14 GB against Model Manager's 2.5 GiB q4 row: say which number is which.
    assert!(message.contains("not the tier's on-disk size"), "{message}");
    assert!(
        message.contains("Run on a GPU with more VRAM."),
        "the one lever that can change the verdict: {message}"
    );
    assert!(
        !message.contains("Lower the output resolution"),
        "the clamp prices a geometry-blind row, so this retry can never pass: {message}"
    );
    assert!(
        message.contains("does not scale with the output size"),
        "name the reason, or the operator retries smaller anyway: {message}"
    );
    assert!(
        !message.contains("Select a smaller"),
        "q4 IS the smallest tier the clamp scored: {message}"
    );

    // The SDXL admission path prices geometry through the shared selector, so the knob is real.
    let priced =
        candle_capability_reject_message("sdxl", "q4", 14.0, 6.0, "0", ResolutionLever::Priced);
    assert!(
        priced.contains("Lower the output resolution or run on a GPU with more VRAM."),
        "{priced}"
    );
    assert!(!priced.contains("will not help"), "{priced}");
    assert!(priced.contains("not the tier's on-disk size"), "{priced}");
}
