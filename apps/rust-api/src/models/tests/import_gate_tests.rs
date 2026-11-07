//! The queue-time base-checkpoint compatibility gate (sc-14019, epic 14015): `import_source_supported`
//! must accept a dense-bf16 or int8-per-row Krea 2 DiT and refuse unsupported triples with an
//! actionable reason, so the LAN-exposed import endpoint can never queue an un-runnable file.
use super::*;

/// Write a safetensors file whose header declares `(name, dtype)` tensors. The declared tensor
/// data must be present or the header is rejected as truncated (sc-6072), so the payload is padded
/// to the declared offsets — mirrors `external_base_models::tests::write_safetensors`.
fn write_safetensors(path: &FsPath, entries: &[(&str, &str)]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent dir");
    }
    let mut header = serde_json::Map::new();
    for (index, (key, dtype)) in entries.iter().enumerate() {
        let start = index * 4;
        header.insert(
            (*key).to_owned(),
            json!({ "dtype": dtype, "shape": [1], "data_offsets": [start, start + 4] }),
        );
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("header json");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend(std::iter::repeat_n(0_u8, entries.len() * 4));
    std::fs::write(path, bytes).expect("write safetensors");
}

/// A ComfyUI-native dense-bf16 Krea 2 DiT: the unique `txtfusion.` tower + BFL-style `blocks.*`
/// keys, all BF16 → detector `(krea_2, Transformer, Bf16)` — one supported import triple.
fn krea2_bf16_dit_keys() -> Vec<(&'static str, &'static str)> {
    vec![
        ("model.diffusion_model.blocks.0.attn.wq.weight", "BF16"),
        ("model.diffusion_model.blocks.0.mod.lin", "BF16"),
        (
            "model.diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight",
            "BF16",
        ),
        ("model.diffusion_model.txtfusion.projector.weight", "BF16"),
    ]
}

fn fused_sdxl_f16_keys() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "model.diffusion_model.input_blocks.7.0.out_layers.3.weight",
            "F16",
        ),
        (
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight",
            "F16",
        ),
        (
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "F16",
        ),
        (
            "conditioner.embedders.1.model.transformer.resblocks.9.attn.in_proj_weight",
            "F16",
        ),
        ("first_stage_model.encoder.conv_in.weight", "F16"),
        ("first_stage_model.decoder.conv_out.weight", "F16"),
    ]
}

fn legacy_user_entry(family: &str, path: &FsPath) -> JsonObject {
    json!({
        "id": "legacy_user_model",
        "family": family,
        "paths": { "model": path.to_string_lossy() },
    })
    .as_object()
    .expect("legacy entry")
    .clone()
}

#[test]
fn krea2_bf16_upload_is_accepted() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("kreamania_variant5.safetensors");
    write_safetensors(&file, &krea2_bf16_dit_keys());
    assert!(
        import_source_supported(&file).is_ok(),
        "a dense-bf16 Krea 2 DiT upload must pass the import gate"
    );
}

#[test]
fn legacy_source_shape_backfill_uses_the_exact_loader_structure() {
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path().join("data");
    let models = data_dir.join("models");

    let krea_dir = models.join("imports/krea");
    let krea_file = krea_dir.join("transformer.safetensors");
    write_safetensors(&krea_file, &krea2_bf16_dit_keys());
    let mut krea = legacy_user_entry("krea_2", &krea_dir);
    stamp_legacy_import_source_shape(&mut krea, &data_dir, true);
    assert_eq!(krea["importSourceShape"], json!("transformer_file"));

    let sdxl_file = models.join("imports/sdxl.safetensors");
    write_safetensors(&sdxl_file, &fused_sdxl_f16_keys());
    let mut sdxl = legacy_user_entry("sdxl", &sdxl_file);
    stamp_legacy_import_source_shape(&mut sdxl, &data_dir, true);
    assert_eq!(sdxl["importSourceShape"], json!("fused_checkpoint"));

    let mage_dir = models.join("imports/mage");
    std::fs::create_dir_all(&mage_dir).unwrap();
    std::fs::write(mage_dir.join("config.json"), b"{}").unwrap();
    std::fs::write(
        mage_dir.join("diffusion_pytorch_model.safetensors"),
        b"weights",
    )
    .unwrap();
    let mut mage = legacy_user_entry("mage-flow", &mage_dir);
    stamp_legacy_import_source_shape(&mut mage, &data_dir, true);
    assert_eq!(mage["importSourceShape"], json!("transformer_directory"));
}

#[test]
fn legacy_source_shape_backfill_refuses_ambiguous_or_mismatched_trees() {
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path().join("data");
    let models = data_dir.join("models");

    let ambiguous = models.join("imports/ambiguous");
    write_safetensors(&ambiguous.join("one.safetensors"), &krea2_bf16_dit_keys());
    write_safetensors(&ambiguous.join("two.safetensors"), &krea2_bf16_dit_keys());
    let mut multiple = legacy_user_entry("krea_2", &ambiguous);
    stamp_legacy_import_source_shape(&mut multiple, &data_dir, true);
    assert!(multiple.get("importSourceShape").is_none());

    let nested = models.join("imports/nested");
    write_safetensors(
        &nested.join("transformer/model.safetensors"),
        &krea2_bf16_dit_keys(),
    );
    let mut recursive = legacy_user_entry("krea_2", &nested);
    stamp_legacy_import_source_shape(&mut recursive, &data_dir, true);
    assert!(recursive.get("importSourceShape").is_none());

    let sdxl_file = models.join("imports/wrong-family.safetensors");
    write_safetensors(&sdxl_file, &fused_sdxl_f16_keys());
    let mut wrong_family = legacy_user_entry("krea_2", &sdxl_file);
    stamp_legacy_import_source_shape(&mut wrong_family, &data_dir, true);
    assert!(wrong_family.get("importSourceShape").is_none());
}

#[test]
fn krea2_int8_per_row_upload_is_accepted() {
    // Header detection deliberately identifies the convention from bulk I8 weights plus
    // `.comfy_quant`; the inference loader validates the actual descriptor JSON and scale shapes
    // before dequantization. This reduced fixture pins the queue-time classification/gate seam.
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("kreamania_variant4.safetensors");
    let mut names = Vec::new();
    for index in 0..6 {
        names.push((
            format!("model.diffusion_model.blocks.{index}.attn.wq.weight"),
            "I8",
        ));
        names.push((
            format!("model.diffusion_model.blocks.{index}.attn.wq.weight_scale"),
            "F32",
        ));
        names.push((
            format!("model.diffusion_model.blocks.{index}.attn.wq.comfy_quant"),
            "U8",
        ));
    }
    names.push(("model.diffusion_model.blocks.0.mod.lin".to_owned(), "BF16"));
    names.push((
        "model.diffusion_model.txtfusion.projector.weight".to_owned(),
        "F32",
    ));
    let entries: Vec<(&str, &str)> = names
        .iter()
        .map(|(name, dtype)| (name.as_str(), *dtype))
        .collect();
    write_safetensors(&file, &entries);

    assert!(
        import_source_supported(&file).is_ok(),
        "a Krea 2 int8-per-row DiT upload must pass the import gate"
    );
}

#[test]
fn krea2_scalar_fp8_upload_is_accepted() {
    // A ComfyUI `ModelSave` of an fp8-scaled Krea 2 DiT: `F8_E4M3` weights with an `F32`
    // per-tensor `.weight_scale` and a `.comfy_quant` descriptor per projection. Before
    // sc-27092 this classified as `ComfyQuantPacked` — the fp4/mxfp4 bucket — and was refused
    // with a reason naming a format the file does not use.
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("krea2_fp8_scaled.safetensors");
    let mut names = Vec::new();
    for index in 0..6 {
        names.push((
            format!("model.diffusion_model.blocks.{index}.attn.wq.weight"),
            "F8_E4M3",
        ));
        names.push((
            format!("model.diffusion_model.blocks.{index}.attn.wq.weight_scale"),
            "F32",
        ));
        names.push((
            format!("model.diffusion_model.blocks.{index}.attn.wq.comfy_quant"),
            "U8",
        ));
    }
    names.push(("model.diffusion_model.blocks.0.mod.lin".to_owned(), "BF16"));
    names.push((
        "model.diffusion_model.txtfusion.projector.weight".to_owned(),
        "F32",
    ));
    let entries: Vec<(&str, &str)> = names
        .iter()
        .map(|(name, dtype)| (name.as_str(), *dtype))
        .collect();
    write_safetensors(&file, &entries);

    assert!(
        import_source_supported(&file).is_ok(),
        "a Krea 2 scalar-fp8 DiT upload must pass the import gate"
    );
}

#[test]
fn krea2_packed_quant_upload_is_refused_with_reason() {
    // Same family/component but `.comfy_quant` fp4/mxfp4-packed — nibble-packed `U8` weights
    // with no `I8` weight anywhere → the refuse bucket → 400 with a reason. (The fixture is
    // genuinely packed: sc-27092 replaced the old "bulk `I8` or it is packed" proxy with a
    // per-projection reading, so a file that really is packed must say so in its dtypes.)
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("kreamania_variant4.safetensors");
    write_safetensors(
        &file,
        &[
            ("model.diffusion_model.blocks.0.attn.wq.weight", "U8"),
            ("model.diffusion_model.blocks.0.attn.wq.comfy_quant", "U8"),
            ("model.diffusion_model.blocks.0.mod.lin", "BF16"),
            (
                "model.diffusion_model.txtfusion.refiner_blocks.0.attn.wq.comfy_quant",
                "U8",
            ),
        ],
    );
    let error = import_source_supported(&file).expect_err("packed quant must be refused");
    assert_eq!(error.status, StatusCode::BAD_REQUEST);
    assert!(error.detail.contains("bf16"), "detail: {}", error.detail);
}

#[test]
fn unsupported_family_upload_is_refused() {
    // A qwen-image plain-fp8 DiT is recognized but has no import loader → refused.
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("qwen_image.safetensors");
    write_safetensors(
        &file,
        &[
            ("model.diffusion_model.img_in.weight", "F8_E4M3"),
            (
                "model.diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
                "F8_E4M3",
            ),
            (
                "model.diffusion_model.transformer_blocks.0.img_mlp.net.0.proj.weight",
                "F8_E4M3",
            ),
        ],
    );
    let error = import_source_supported(&file).expect_err("qwen import must be refused");
    assert_eq!(error.status, StatusCode::BAD_REQUEST);
    assert!(
        error.detail.contains("qwen-image"),
        "detail should name the unsupported family: {}",
        error.detail
    );
}

#[test]
fn unrecognized_file_upload_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("mystery.safetensors");
    write_safetensors(
        &file,
        &[("some.mystery.tensor", "BF16"), ("another.mystery", "BF16")],
    );
    let error = import_source_supported(&file).expect_err("unrecognized file must be refused");
    assert_eq!(error.status, StatusCode::BAD_REQUEST);
}
