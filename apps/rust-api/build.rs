use std::path::Path;

// The `embed-web` feature bakes apps/web/dist into the binary via rust-embed at
// compile time. Cargo otherwise has no idea that bundle is a build input, so a
// web-only change (e.g. rebuilding apps/web without touching any .rs file) leaves
// the crate "fresh" and ships a STALE embedded UI. Emit rerun-if-changed for every
// file under the bundle so a rebuilt bundle always forces a recompile + re-embed.
fn main() {
    let dist = Path::new("../web/dist");
    println!("cargo:rerun-if-changed=../web/dist");
    if dist.is_dir() {
        emit_rerun_for_tree(dist);
    }
}

fn emit_rerun_for_tree(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        println!("cargo:rerun-if-changed={}", path.display());
        if path.is_dir() {
            emit_rerun_for_tree(&path);
        }
    }
}
