fn main() {
    // The target triple this crate is compiled for, so tests that shell out to
    // `cargo metadata --filter-platform` resolve exactly this build's graph instead of
    // hardcoding a triple per OS (sc-24163).
    let target = std::env::var("TARGET").expect("cargo sets TARGET for build scripts");
    println!("cargo:rustc-env=SW_HOST_TARGET={target}");
    println!("cargo:rustc-check-cfg=cfg(native_lane)");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS")
        .expect("cargo sets CARGO_CFG_TARGET_OS for build scripts");
    let candle = std::env::var_os("CARGO_FEATURE_BACKEND_CANDLE").is_some();
    if native_lane(&target_os, candle) {
        println!("cargo:rustc-cfg=native_lane");
    }
    println!("cargo:rerun-if-changed=build.rs");
}

// Build scripts and the architecture truth-table test share this exact decision.
pub(crate) fn native_lane(target_os: &str, backend_candle: bool) -> bool {
    match target_os {
        "macos" => true,
        _ => backend_candle,
    }
}
