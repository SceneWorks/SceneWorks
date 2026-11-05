use super::*;

fn legacy() -> ModelImportRequest {
    ModelImportRequest {
        ownership_mode: OwnershipModeV1::default(),
        source: None,
        model_id: None,
        name: None,
        model_type: None,
        repo: None,
        source_url: None,
        source_path: None,
        // Linked (in-place) ownership is a different mode entirely (sc-20635); a legacy flat
        // request never names one.
        linked_root_id: None,
        linked_relative_path: None,
        files: Vec::new(),
        family: None,
        base_model: None,
        minimax_h3_partition: None,
        expected_sha256: None,
        license_acknowledged: false,
        uploaded_source_path: false,
    }
}

fn parse(body: &str) -> ModelImportRequest {
    serde_json::from_str(body).expect("body deserializes")
}

/// AC3: a body written before `ownershipMode` and `source` existed still deserializes, and it
/// normalises to MANAGED ownership with its source recorded — not to an unowned import.
#[test]
fn legacy_flat_bodies_still_deserialize_and_normalise_to_managed_ownership() {
    for (body, expected_source, expected_reference) in [
        (
            r#"{"repo":"org/model","type":"image"}"#,
            "huggingface",
            Some("org/model"),
        ),
        (
            r#"{"sourceUrl":"https://host.example/model.safetensors"}"#,
            "url",
            None,
        ),
        (
            r#"{"sourcePath":"/data/models/imports/mine/model.safetensors"}"#,
            "local-copy",
            Some("model.safetensors"),
        ),
    ] {
        let mut payload = parse(body);
        assert_eq!(
            payload.ownership_mode,
            OwnershipModeV1::Managed,
            "an absent ownershipMode is managed: {body}"
        );
        assert!(payload.source.is_none());
        let (provenance, revision) =
            normalize_model_import_source(&mut payload).expect("legacy body normalises");
        assert_eq!(provenance.source, expected_source, "{body}");
        assert_eq!(
            provenance.reference.as_deref(),
            expected_reference,
            "{body}"
        );
        assert_eq!(revision, None);
    }

    // The flat fields survive normalisation untouched, so the rest of the import path — and the
    // worker — see exactly what they saw before this field pair existed.
    let mut payload = parse(r#"{"repo":"org/model","files":["model.safetensors"]}"#);
    normalize_model_import_source(&mut payload).unwrap();
    assert_eq!(payload.repo.as_deref(), Some("org/model"));
    assert_eq!(payload.files, vec!["model.safetensors".to_owned()]);
}

#[test]
fn an_uploaded_legacy_body_is_recorded_as_an_upload_not_a_local_copy() {
    let mut payload = legacy();
    payload.source_path = Some("/data/cache/model-uploads/upload-1/mine.safetensors".to_owned());
    payload.files = vec!["mine.safetensors".to_owned()];
    payload.uploaded_source_path = true;
    let (provenance, _) = normalize_model_import_source(&mut payload).unwrap();
    assert_eq!(provenance.source, "upload");
    assert_eq!(provenance.reference.as_deref(), Some("mine.safetensors"));
}

/// Each discriminated variant collapses onto the flat field the existing transfer code reads,
/// so there is one pipeline rather than a second one beside it.
#[test]
fn every_discriminated_source_normalises_onto_the_existing_transfer_fields() {
    let mut payload = parse(
        r#"{"source":{"kind":"huggingFace","repo":"org/model","revision":"abc123","files":["model.safetensors"]}}"#,
    );
    let (provenance, revision) = normalize_model_import_source(&mut payload).unwrap();
    assert_eq!(payload.repo.as_deref(), Some("org/model"));
    assert_eq!(payload.files, vec!["model.safetensors".to_owned()]);
    assert_eq!(revision.as_deref(), Some("abc123"));
    assert_eq!(provenance.source, "huggingface");
    assert_eq!(provenance.reference.as_deref(), Some("org/model@abc123"));
    assert_eq!(provenance.version_id.as_deref(), Some("abc123"));
    assert_eq!(provenance.file_id.as_deref(), Some("model.safetensors"));

    let mut payload = parse(
        r#"{"source":{"kind":"localPath","path":"/data/models/imports/x/model.safetensors"}}"#,
    );
    let (provenance, _) = normalize_model_import_source(&mut payload).unwrap();
    assert_eq!(
        payload.source_path.as_deref(),
        Some("/data/models/imports/x/model.safetensors")
    );
    assert!(!payload.uploaded_source_path);
    assert_eq!(provenance.source, "local-copy");

    let mut payload = parse(
        r#"{"source":{"kind":"url","url":"https://host.example/m.safetensors","expectedSha256":"ab"}}"#,
    );
    let (provenance, _) = normalize_model_import_source(&mut payload).unwrap();
    assert_eq!(
        payload.source_url.as_deref(),
        Some("https://host.example/m.safetensors")
    );
    assert_eq!(payload.expected_sha256.as_deref(), Some("ab"));
    assert_eq!(provenance.source, "url");
    assert_eq!(provenance.credential_host.as_deref(), Some("host.example"));

    let mut payload = legacy();
    payload.source = Some(ModelImportSourceV1::Upload {
        staged_path: Some("/data/cache/model-uploads/u/m.safetensors".to_owned()),
    });
    let (provenance, _) = normalize_model_import_source(&mut payload).unwrap();
    assert_eq!(
        payload.source_path.as_deref(),
        Some("/data/cache/model-uploads/u/m.safetensors")
    );
    assert!(payload.uploaded_source_path);
    assert_eq!(provenance.source, "upload");
}

/// AC1: Civitai's version/file identity, URL, and the host whose credential authorized the
/// download are recorded; no secret is.
#[test]
fn a_civitai_source_records_its_identity_and_never_a_secret() {
    let mut payload = parse(
        r#"{"source":{"kind":"civitai","url":"https://user:t0ken@civitai.com/api/download/models/9931?type=Model&token=s3cret","modelVersionId":"9931","fileId":"40277","expectedSha256":"abc"}}"#,
    );
    let (provenance, _) = normalize_model_import_source(&mut payload).unwrap();

    // The transfer reuses the ordinary source-URL download, credential attachment included.
    assert_eq!(
        payload.source_url.as_deref(),
        Some("https://user:t0ken@civitai.com/api/download/models/9931?type=Model&token=s3cret")
    );
    assert_eq!(payload.expected_sha256.as_deref(), Some("abc"));

    assert_eq!(provenance.source, "civitai");
    assert_eq!(provenance.version_id.as_deref(), Some("9931"));
    assert_eq!(provenance.file_id.as_deref(), Some("40277"));
    assert_eq!(provenance.reference.as_deref(), Some("modelVersion/9931"));
    assert_eq!(provenance.credential_host.as_deref(), Some("civitai.com"));
    assert_eq!(
        provenance.url.as_deref(),
        Some("https://civitai.com/api/download/models/9931?type=Model"),
        "userinfo and the token parameter must be stripped before provenance is persisted"
    );
    let serialized = provenance.to_json().to_string();
    assert!(
        !serialized.contains("t0ken") && !serialized.contains("s3cret"),
        "provenance must carry no credential: {serialized}"
    );
}

/// A request that sets both spellings is refused rather than silently resolved by precedence:
/// honouring one and dropping the other would fetch something the caller did not ask for.
#[test]
fn a_request_that_sets_both_spellings_is_refused() {
    for body in [
        r#"{"repo":"org/a","source":{"kind":"huggingFace","repo":"org/b"}}"#,
        r#"{"sourceUrl":"https://a.example/m","source":{"kind":"url","url":"https://b.example/m"}}"#,
        r#"{"sourcePath":"/a/m","source":{"kind":"localPath","path":"/b/m"}}"#,
    ] {
        let mut payload = parse(body);
        let error = normalize_model_import_source(&mut payload)
            .expect_err("a conflicting body must refuse");
        assert!(
            format!("{error:?}").contains("not both"),
            "{body} refused as {error:?}"
        );
    }
}

/// `expectedSha256` has the same two spellings as `repo`/`sourceUrl`/`sourcePath`, and it was
/// the only one resolved by PRECEDENCE (`expected_sha256.or(payload.expected_sha256)`). A caller
/// who sent two different digests got the transfer verified against whichever one won and a
/// success report — the exact silent resolution this function's contract refuses. Both set and
/// naming the same digest is fine: sha256 is hex, so case and surrounding whitespace are not a
/// difference.
#[test]
fn conflicting_expected_sha256_digests_are_refused_and_matching_ones_are_accepted() {
    let alpha = "a".repeat(64);
    let beta = "b".repeat(64);
    for (label, body) in [
        (
            "url",
            format!(
                r#"{{"expectedSha256":"{beta}","source":{{"kind":"url","url":"https://h.example/m","expectedSha256":"{alpha}"}}}}"#
            ),
        ),
        (
            "civitai",
            format!(
                r#"{{"expectedSha256":"{beta}","source":{{"kind":"civitai","url":"https://civitai.com/api/download/models/1","expectedSha256":"{alpha}"}}}}"#
            ),
        ),
    ] {
        let mut payload = parse(&body);
        let error = normalize_model_import_source(&mut payload)
            .expect_err("two different digests must refuse");
        let rendered = format!("{error:?}");
        assert!(
            rendered.contains("expectedSha256") && rendered.contains("not both"),
            "{label} refusal must name the field: {rendered}"
        );
        assert!(
            rendered.contains(&alpha) && rendered.contains(&beta),
            "{label} refusal must show both digests: {rendered}"
        );
    }

    // The SAME digest in both places, differing only in case and surrounding whitespace, is one
    // digest spelled twice — accepted, and the value is preserved for the transfer to verify.
    for (label, body) in [
        (
            "url",
            format!(
                r#"{{"expectedSha256":"  {upper}  ","source":{{"kind":"url","url":"https://h.example/m","expectedSha256":"{alpha}"}}}}"#,
                upper = alpha.to_ascii_uppercase()
            ),
        ),
        (
            "civitai",
            format!(
                r#"{{"expectedSha256":"  {upper}  ","source":{{"kind":"civitai","url":"https://civitai.com/api/download/models/1","expectedSha256":"{alpha}"}}}}"#,
                upper = alpha.to_ascii_uppercase()
            ),
        ),
    ] {
        let mut payload = parse(&body);
        normalize_model_import_source(&mut payload).unwrap_or_else(|error| {
            panic!("{label} must accept one digest spelled twice: {error:?}")
        });
        assert_eq!(
            payload.expected_sha256.as_deref(),
            Some(alpha.as_str()),
            "{label} must keep the digest for the transfer to verify"
        );
    }

    // Only one side set is not a conflict, in either direction.
    let mut payload = parse(&format!(
        r#"{{"expectedSha256":"{alpha}","source":{{"kind":"url","url":"https://h.example/m"}}}}"#
    ));
    normalize_model_import_source(&mut payload).expect("a flat-only digest is not a conflict");
    assert_eq!(payload.expected_sha256.as_deref(), Some(alpha.as_str()));
}

/// E7: an ownership mode this route does not serve fails closed at deserialization rather than
/// defaulting to managed.
#[test]
fn an_unserved_ownership_mode_is_refused_instead_of_defaulted() {
    for body in [
        r#"{"repo":"org/model","ownershipMode":"linked"}"#,
        r#"{"repo":"org/model","ownershipMode":"external"}"#,
        r#"{"repo":"org/model","ownershipMode":""}"#,
    ] {
        assert!(
            serde_json::from_str::<ModelImportRequest>(body).is_err(),
            "{body} must be refused"
        );
    }
    // And an unknown discriminated source kind, or an unknown field inside a known one.
    for body in [
        r#"{"source":{"kind":"ftp","url":"ftp://host/m"}}"#,
        r#"{"source":{"kind":"url","url":"https://h/m","token":"s3cret"}}"#,
    ] {
        assert!(
            serde_json::from_str::<ModelImportRequest>(body).is_err(),
            "{body} must be refused"
        );
    }
}
