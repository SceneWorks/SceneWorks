//! Export the compiled, metadata-only static snapshot for consumers without a Rust toolchain.
fn main() {
    println!(
        "{}",
        include_str!(concat!(env!("OUT_DIR"), "/model-registry.json"))
    );
}
