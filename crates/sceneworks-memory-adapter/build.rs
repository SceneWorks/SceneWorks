fn main() {
    // Kolors overflows Windows' default 1 MiB main-thread stack during capture.
    // Set the executable's reserve: RUST_MIN_STACK only affects spawned threads.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
        && std::env::var_os("CARGO_FEATURE_CANDLE").is_some()
    {
        println!("cargo:rustc-link-arg-bin=memory-candle-adapter=/STACK:33554432");
    }
}
