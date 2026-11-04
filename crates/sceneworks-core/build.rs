// Derive new evidence provenance from the dependency actually selected by this crate.
fn main() {
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
