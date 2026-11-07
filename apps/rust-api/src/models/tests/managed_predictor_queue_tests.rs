use super::*;
#[test]
fn both_h3_rows_queue_authoritative_exact_soft_predictor_identity() {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .unwrap()
        .1;
    let catalog: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw)).unwrap();
    // The H3 Veda predictor's own identity, selected by repo: the catalog carries more than one
    // managed single-file pin (sc-24213's Supra2-IMG checkpoint is another), so "the first"
    // is not a stable way to name it.
    let pin = sceneworks_core::managed_file::builtin_identities()
        .unwrap()
        .iter()
        .find(|identity| identity.repo == "Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview")
        .unwrap();
    for id in ["minimax_h3", "minimax_h3_ref"] {
        let model = catalog["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .unwrap();
        let row = model["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["componentId"] == "veda_predictor")
            .unwrap();
        assert_eq!(row["required"], "soft");
        let payload = build_model_download_job_payload(
            model,
            id,
            row,
            None,
            false,
            true,
            FsPath::new("/data"),
        )
        .unwrap();
        assert_eq!(payload["expectedSha256"], pin.sha256);
        assert_eq!(payload["expectedSizeBytes"], pin.bytes);
        assert_eq!(payload["revision"], pin.revision);
        assert_eq!(payload["files"], json!([pin.file]));
        assert_eq!(
            sceneworks_core::managed_file::authoritative_job_identity(&payload).unwrap(),
            Some(pin)
        );
        let mut conflict = payload;
        conflict.insert("expectedSha256".into(), json!("0".repeat(64)));
        assert!(sceneworks_core::managed_file::authoritative_job_identity(&conflict).is_err());
    }
}
