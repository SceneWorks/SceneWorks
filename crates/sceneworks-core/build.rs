#[path = "src/registry/assemble.rs"]
mod assemble;
#[path = "src/registry/evidence.rs"]
#[allow(dead_code)]
mod evidence;
#[path = "src/registry/exceptions.rs"]
#[allow(dead_code)]
mod exceptions;
#[path = "src/hf_repo_id.rs"]
mod hf_repo_id;
#[path = "src/jsonc.rs"]
mod jsonc;
#[path = "src/registry/spec.rs"]
mod spec;

use std::path::{Path, PathBuf};

fn build_registry() {
    let read = |name: &str| {
        let path = Path::new("../../config").join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    };
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/hf_repo_id.rs");
    println!("cargo:rerun-if-changed=src/registry/assemble.rs");
    println!("cargo:rerun-if-changed=src/registry/spec.rs");
    let product_manifest = read("manifests/builtin.models.jsonc");
    let product_json = jsonc::strip_jsonc_comments(&product_manifest);
    jsonc::reject_duplicate_keys(&product_json).expect("duplicate builtin manifest key");
    let product_rows: serde_json::Value =
        serde_json::from_str(&product_json).expect("invalid builtin model manifest");
    let evidence_json = read("manifests/registry/evidence.json");
    let _approved_decisions = exceptions::approved_backend_exceptions();
    println!("cargo:rerun-if-changed=../../config/manifests/registry/approved-exceptions.json");
    jsonc::reject_duplicate_keys(&evidence_json).expect("duplicate memory evidence key");
    evidence::validate_evidence_document(&evidence_json, &product_rows)
        .expect("invalid builtin memory evidence");
    let schema: serde_json::Value =
        serde_json::from_str(&read("../packages/schemas/model-manifest.schema.json"))
            .expect("invalid model manifest schema");
    let validator = jsonschema::validator_for(&schema).expect("invalid model manifest schema");
    if let Some(error) = validator.iter_errors(&product_rows).next() {
        panic!("invalid v2 product manifest: {error}");
    }
    let mut effective = product_rows.clone();
    effective["schemaVersion"] = serde_json::json!(1);
    for row in effective["models"]
        .as_array_mut()
        .expect("validated model rows")
    {
        evidence::project_builtin_memory_evidence(row)
            .expect("invalid builtin evidence projection");
    }
    if let Some(error) = validator.iter_errors(&effective).next() {
        panic!("invalid effective builtin memory evidence: {error}");
    }
    let specs = assemble::assemble(
        &read("manifests/registry/products.json"),
        &evidence_json,
        &[
            (
                "mlx",
                &read("engine-capabilities/runtime/capabilities.mlx.json"),
            ),
            (
                "candle",
                &read("engine-capabilities/runtime/capabilities.candle.json"),
            ),
        ],
    )
    .expect("invalid static model registry");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("model-registry.json");
    // A host build dependency can enable serde_json's preserve_order feature. The shipped
    // snapshot has the same bytes under either Cargo feature set.
    let mut snapshot = serde_json::to_value(&specs).expect("serialize model registry");
    snapshot.sort_all_objects();
    std::fs::write(
        out,
        serde_json::to_vec_pretty(&snapshot).expect("serialize model registry"),
    )
    .expect("write model registry");
}

// Derive new evidence provenance from the dependency actually selected by this crate.
fn main() {
    build_registry();
    println!("cargo:rerun-if-changed=Cargo.toml");
    let manifest = std::fs::read_to_string("Cargo.toml").expect("core manifest");
    let declaration = manifest
        .lines()
        .find(|line| line.starts_with("sceneworks-gen-core = "))
        .expect("pinned inference contract dependency");
    let revision = declaration
        .split("rev = \"")
        .nth(1)
        .and_then(|part| part.split('"').next())
        .expect("exact inference revision");
    assert!(
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "inference dependency must use a full revision"
    );
    println!("cargo:rustc-env=SCENEWORKS_INFERENCE_REVISION={revision}");
}
