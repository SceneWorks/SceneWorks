//! Stage-1 dumper for the engine-capability facts files (sc-16965, epic 16948).
//!
//! Walks this build's **linked** provider registry weights-free and writes one checked-in facts
//! file per backend under `config/engine-capabilities/`. See
//! [`sceneworks_worker::engine_capability_facts`] for the full rationale — in short, the process
//! that serves `/api/v1/models` is not always the process that links an engine, so the flag must be
//! dumped at build time rather than derived at serve time.
//!
//! Re-run at every **inference pin bump** (`scripts/bump-inference.mjs` fails closed on a stale
//! file, beside the licence re-scan), on each lane that can build a registry:
//!
//! ```text
//! # off-Mac CUDA/candle → config/engine-capabilities/capabilities.candle.json
//! cargo run -p sceneworks-worker --bin dump-engine-capabilities \
//!     --no-default-features --features backend-candle
//!
//! # macOS/MLX → config/engine-capabilities/capabilities.mlx.json
//! cargo run -p sceneworks-worker --bin dump-engine-capabilities
//! ```
//!
//! Exits non-zero and writes **nothing** when the registry is empty, so a lane that links no
//! engines can never leave a valid-looking, entirely wrong facts file behind.

use std::path::PathBuf;

fn main() {
    // Optional explicit output directory (tests, or dumping into a scratch dir to diff); defaults
    // to the checked-in `config/engine-capabilities/` resolved from the crate manifest dir, so the
    // dumper works from any cwd.
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(sceneworks_worker::engine_capability_facts::default_facts_dir);

    match sceneworks_worker::engine_capability_facts::dump_to(&dir) {
        Ok(written) => {
            for path in written {
                println!("wrote {}", path.display());
            }
        }
        Err(error) => {
            eprintln!("dump-engine-capabilities: {error}");
            std::process::exit(1);
        }
    }
}
