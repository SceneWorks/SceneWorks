//! Stage-1 dumper for the engine-capability facts files (sc-16965, epic 16948).
//!
//! Walks this build's **linked** provider registries weights-free and writes checked-in facts files
//! under `config/engine-capabilities/` — one per media backend, plus the audio registry's own dump
//! in `audio/` (sc-17593). See [`sceneworks_worker::engine_capability_facts`] for the full rationale
//! — in short, the process that serves `/api/v1/models` is not always the process that links an
//! engine, so the flag must be dumped at build time rather than derived at serve time.
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
//! Either invocation also writes `audio/capabilities.candle.json`: the audio lane is candle-native
//! on both, links the same catalog on both, and so produces byte-identical facts from either box.
//!
//! Exits non-zero and writes **nothing** when a registry is empty (or, for audio, absent), so a lane
//! that links no engines can never leave a valid-looking, entirely wrong facts file behind. Both
//! dumps are attempted before the exit code is decided, so one lane's failure does not hide the
//! other's.

use std::path::PathBuf;

fn main() {
    // Optional explicit output ROOT (tests, or dumping into a scratch dir to diff); defaults to the
    // checked-in `config/engine-capabilities/` resolved from the crate manifest dir, so the dumper
    // works from any cwd. The audio dump lands in `<root>/audio`.
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(sceneworks_worker::engine_capability_facts::default_facts_dir);

    let media = sceneworks_worker::engine_capability_facts::dump_to(&dir);
    let audio = sceneworks_worker::engine_capability_facts::dump_audio_to(&dir);

    let mut failed = false;
    for outcome in [media, audio] {
        match outcome {
            Ok(written) => {
                for path in written {
                    println!("wrote {}", path.display());
                }
            }
            Err(error) => {
                eprintln!("dump-engine-capabilities: {error}");
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
