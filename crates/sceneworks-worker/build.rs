fn main() {
    // The target triple this crate is compiled for, so tests that shell out to
    // `cargo metadata --filter-platform` resolve exactly this build's graph instead of
    // hardcoding a triple per OS (sc-24163).
    let target = std::env::var("TARGET").expect("cargo sets TARGET for build scripts");
    println!("cargo:rustc-env=SW_HOST_TARGET={target}");
    println!("cargo:rerun-if-changed=build.rs");
}
