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
    pin_dynamic_crt();
}

// Resolve the whole link against ONE C runtime on Windows (LNK4098).
//
// `backend-candle` pulls candle-kernels, whose build.rs compiles the MoE/GGUF
// kernels with nvcc into `libmoe.a`. nvcc's MSVC host default is the STATIC CRT,
// so those objects carry `/DEFAULTLIB:LIBCMT` + `/DEFAULTLIB:libcpmt` while rustc
// and every other native dep here (sqlite3, onig, ring) ask for the DYNAMIC
// `msvcrt`. The linker warns and silently resolves the conflict for us — today it
// pulls `std::_Xlength_error` out of the STATIC libcpmt, i.e. the binary really
// does end up with a sliver of a second CRT in it.
//
// Excluding the static pair alone breaks the link (that one C++ symbol goes
// unresolved), so `msvcprt` — the dynamic-CRT sibling of libcpmt — has to be named
// explicitly. Net effect: one CRT, chosen deliberately rather than by link order.
// Adds no new runtime DLL dependency (msvcprt satisfies it statically; the import
// table is unchanged).
//
// The real fix belongs upstream in candle-kernels' build.rs (`-Xcompiler /MD` on
// the MSVC branch); this keeps our link honest until that lands.
fn pin_dynamic_crt() {
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let candle = std::env::var_os("CARGO_FEATURE_BACKEND_CANDLE").is_some();
    if !(msvc && candle) {
        return;
    }
    println!("cargo:rustc-link-arg=/NODEFAULTLIB:libcmt");
    println!("cargo:rustc-link-arg=/NODEFAULTLIB:libcpmt");
    println!("cargo:rustc-link-arg=/DEFAULTLIB:msvcprt");
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
