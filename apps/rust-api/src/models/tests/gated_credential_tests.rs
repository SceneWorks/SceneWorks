use super::*;
use serde_json::json;

fn map(value: Value) -> serde_json::Map<String, Value> {
    value.as_object().expect("object").clone()
}

#[test]
fn derives_huggingface_host_from_provider() {
    let model = map(json!({
        "downloads": [{ "provider": "huggingface", "repo": "black-forest-labs/FLUX.1-dev", "files": [] }]
    }));
    assert_eq!(
        derive_credential_host(&model).as_deref(),
        Some("huggingface.co")
    );
}

#[test]
fn prefers_explicit_download_credential_host() {
    let model = map(json!({
        "downloads": [{ "provider": "civitai", "credentialHost": "https://Civitai.com/", "sourceUrl": "https://civitai.com/api/x" }]
    }));
    assert_eq!(
        derive_credential_host(&model).as_deref(),
        Some("civitai.com")
    );
}

#[test]
fn falls_back_to_source_url_host() {
    let model = map(json!({
        "downloads": [{ "provider": "url", "sourceUrl": "https://models.example.com/path/file.safetensors" }]
    }));
    assert_eq!(
        derive_credential_host(&model).as_deref(),
        Some("models.example.com")
    );
}

#[test]
fn no_downloads_yields_none() {
    assert_eq!(derive_credential_host(&map(json!({}))), None);
}

// sc-7872: the SD3.5 gated entries download direct from the gated stabilityai/*
// repos (no re-host), so the credential host derives to huggingface.co exactly
// like FLUX.2-dev — driving the same stored-HF-token download path.
#[test]
fn derives_huggingface_host_for_stabilityai_sd3_5_repos() {
    for repo in [
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-large-turbo",
        "stabilityai/stable-diffusion-3.5-medium",
    ] {
        let model = map(json!({
            "downloads": [{ "provider": "huggingface", "repo": repo, "files": ["transformer/*"] }]
        }));
        assert_eq!(
            derive_credential_host(&model).as_deref(),
            Some("huggingface.co"),
            "repo {repo} should derive huggingface.co",
        );
    }
}

// sc-7872: a gated SD3.5 entry round-trips through apply_gating_fields with its
// explicit huggingface.co credential host preserved (the field the web client
// reads to gate the download + surface the credential prompt). licenseUrl is
// untouched, so the model card links the stabilityai license page.
#[test]
fn sd3_5_gated_entry_preserves_credential_host_and_license() {
    let mut model = map(json!({
        "id": "sd3_5_large",
        "gated": true,
        "credentialHost": "huggingface.co",
        "licenseUrl": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
        "downloads": [{ "provider": "huggingface", "repo": "stabilityai/stable-diffusion-3.5-large", "files": ["transformer/*"] }]
    }));
    apply_gating_fields(&mut model);
    assert_eq!(model.get("gated").and_then(Value::as_bool), Some(true));
    assert_eq!(
        model.get("credentialHost").and_then(Value::as_str),
        Some("huggingface.co"),
    );
    assert_eq!(
        model.get("licenseUrl").and_then(Value::as_str),
        Some("https://huggingface.co/stabilityai/stable-diffusion-3.5-large"),
    );
}

// sc-17227: an acknowledgment-only entry — MiniMax-H3, whose HF repo is PUBLIC. The catalog
// must carry `requiresLicenseAcknowledgment` + `licenseNotice` through to the web client
// untouched, and must NOT manufacture a `credentialHost` for it: the Models screen keys the
// "Add token in Settings" / "Request access on Hugging Face" affordances off that host, and
// there is no token to add and no access to request. Note the asymmetry with the gated case
// above — `gated` is normalized to an explicit `false`, but the host is left absent, which is
// exactly what `derive_credential_host` would have supplied had the two been coupled.
#[test]
fn license_acknowledgment_entry_keeps_its_fields_and_gains_no_credential_host() {
    let mut model = map(json!({
        "id": "minimax_h3",
        "requiresLicenseAcknowledgment": true,
        "licenseUrl": "https://huggingface.co/MiniMaxAI/MiniMax-H3",
        "licenseNotice": "Applicable Territory excludes the United States of America.",
        "downloads": [{ "provider": "huggingface", "repo": "SceneWorks/minimax-h3-mlx", "files": ["q4/transformer/*"] }]
    }));
    apply_gating_fields(&mut model);
    assert_eq!(
        model
            .get("requiresLicenseAcknowledgment")
            .and_then(Value::as_bool),
        Some(true),
        "the acknowledgment flag must survive to the web client",
    );
    assert_eq!(
        model.get("licenseNotice").and_then(Value::as_str),
        Some("Applicable Territory excludes the United States of America."),
    );
    assert_eq!(
        model.get("licenseUrl").and_then(Value::as_str),
        Some("https://huggingface.co/MiniMaxAI/MiniMax-H3"),
    );
    assert_eq!(model.get("gated").and_then(Value::as_bool), Some(false));
    assert!(
        !model.contains_key("credentialHost"),
        "a public-repo acknowledgment model must not be given a credential host: {model:?}",
    );
}
