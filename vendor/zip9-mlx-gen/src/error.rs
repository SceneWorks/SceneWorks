//! Crate error type. Replaces the early `Box<dyn Error>` placeholder with a typed enum
//! (sc-2373, disciplined-hybrid architecture). `From<&str>`/`From<String>` are provided so
//! existing `"...".into()` / `format!(...).into()` error sites keep compiling, while
//! `#[from]` lets `?` lift `mlx_rs` and IO errors transparently.

use thiserror::Error;

/// Anything that can go wrong in mlx-gen.
#[derive(Debug, Error)]
pub enum Error {
    /// An MLX op (matmul, quantize, SDPA, …) failed on device.
    #[error("MLX op failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    /// A required tensor key was absent from a loaded checkpoint/adapter.
    #[error("missing tensor: {0}")]
    MissingTensor(String),

    /// Filesystem error while traversing a model directory.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// safetensors load/save error from mlx-rs.
    #[error("safetensors I/O failed: {0}")]
    SafeTensors(#[from] mlx_rs::error::IoError),

    /// Generation was cancelled via `CancelFlag` (cooperative, checked at step boundaries). Typed
    /// (not a stringified `Msg`) so it bridges to [`gen_core::Error::Canceled`] 1:1 and the
    /// conformance suite can distinguish cancellation from failure (epic 3720, sc-4481).
    #[error("cancelled")]
    Canceled,

    /// The request asked for something this engine/backend cannot do (a capability gap the descriptor
    /// doesn't advertise). Typed — like [`Error::Canceled`] — so it bridges to
    /// [`gen_core::Error::Unsupported`] 1:1 in both directions; candle gating and the worker match on
    /// the typed variant to tell "unsupported" apart from a generic failure (epic 3720, F-008).
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A pre-render refusal that preserves the requested geometry and carries only a verified
    /// alternative geometry, if one exists.
    #[error(
        "geometry refused: {reason}; requested {requested_width}x{requested_height}; verified alternative: {alternative:?}"
    )]
    GeometryRefused {
        reason: String,
        requested_width: u32,
        requested_height: u32,
        alternative: Option<(u32, u32)>,
    },

    /// A freshly loaded weight buffer whose GPU view never converged on the bytes the CPU holds
    /// (sc-22414). Typed so a consumer can tell "the host's Metal mapping is lagging" from a
    /// generic failure; see [`crate::coherence`]. Bridges to [`gen_core::Error::Msg`] with its full
    /// text — gen-core has no load-coherence variant and the worker only reports it.
    #[error(
        "GPU view of `{name}` ({bytes} bytes) never matched the CPU view after {attempts} reads \
         (cpu checksum {cpu:#x}, gpu checksum {gpu:#x}): the Metal mapping of a freshly loaded \
         buffer is stale — retry the load once the host's write-back pressure subsides (sc-22414)"
    )]
    IncoherentLoad {
        /// The tensor key.
        name: String,
        /// Its byte size.
        bytes: usize,
        /// The CPU-stream checksum (the reference).
        cpu: u64,
        /// The last GPU-stream checksum observed.
        gpu: u64,
        /// GPU reads attempted.
        attempts: u32,
    },

    /// A contextual message (config/validation/adapter-shape errors).
    #[error("{0}")]
    Msg(String),
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Msg(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Msg(s.to_string())
    }
}

/// Bridge the rich mlx-gen error into the backend-neutral [`gen_core::Error`] (epic 3720, D3 /
/// Option B). Legal under the orphan rule because the source type (`mlx_gen::Error`) is local. This
/// is what lets a family crate's `Generator::generate` — whose signature is `gen_core::Result` —
/// keep using `?` on the `mlx_gen::Result` helpers that do the actual tensor work: the device
/// exceptions box into [`gen_core::Error::Backend`], while the typed variants map across 1:1.
impl From<Error> for gen_core::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Mlx(ex) => gen_core::Error::backend(ex),
            Error::SafeTensors(io) => gen_core::Error::backend(io),
            Error::MissingTensor(s) => gen_core::Error::MissingTensor(s),
            Error::Io(io) => gen_core::Error::Io(io),
            Error::Canceled => gen_core::Error::Canceled,
            Error::Unsupported(s) => gen_core::Error::Unsupported(s),
            Error::GeometryRefused {
                reason,
                requested_width,
                requested_height,
                alternative,
            } => gen_core::Error::GeometryRefused {
                reason,
                requested_width,
                requested_height,
                alternative,
            },
            incoherent @ Error::IncoherentLoad { .. } => {
                gen_core::Error::Msg(incoherent.to_string())
            }
            Error::Msg(s) => gen_core::Error::Msg(s),
        }
    }
}

/// The reverse bridge: gen-core contract calls (tokenizer, registry, imageops, capability
/// validation) return `gen_core::Error`; mlx-gen and family code invoke them with `?` inside
/// `mlx_gen::Result` fns, so down-convert here. `Backend` has no rich mlx-gen analog and collapses
/// to `Msg` (keeping the display text); `Canceled` and `Unsupported` map across 1:1 so a round-trip
/// preserves the typed cancellation (sc-4481) and the typed capability gap (F-008) — a provider that
/// delegates to `Capabilities::validate_request` and then returns the error keeps it typed all the
/// way to the worker / candle gating.
impl From<gen_core::Error> for Error {
    fn from(e: gen_core::Error) -> Self {
        match e {
            gen_core::Error::Backend(b) => Error::Msg(b.to_string()),
            gen_core::Error::MissingTensor(s) => Error::MissingTensor(s),
            gen_core::Error::Io(io) => Error::Io(io),
            gen_core::Error::Unsupported(s) => Error::Unsupported(s),
            gen_core::Error::GeometryRefused {
                reason,
                requested_width,
                requested_height,
                alternative,
            } => Error::GeometryRefused {
                reason,
                requested_width,
                requested_height,
                alternative,
            },
            gen_core::Error::Canceled => Error::Canceled,
            gen_core::Error::Msg(s) => Error::Msg(s),
        }
    }
}

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canceled_round_trips_1to1_both_directions() {
        // The reference behavior the Unsupported bridge mirrors (sc-4481): Canceled stays typed.
        let up: gen_core::Error = Error::Canceled.into();
        assert!(matches!(up, gen_core::Error::Canceled));
        let down: Error = gen_core::Error::Canceled.into();
        assert!(matches!(down, Error::Canceled));
    }

    #[test]
    fn unsupported_round_trips_1to1_both_directions() {
        // F-008: a capability gap must stay a typed `Unsupported` across the seam so candle gating /
        // the worker can distinguish it from a generic failure — not be stringified into `Msg`.
        let up: gen_core::Error = Error::Unsupported("hyper sampler".into()).into();
        match &up {
            gen_core::Error::Unsupported(s) => assert_eq!(s, "hyper sampler"),
            other => panic!("mlx-gen→gen-core Unsupported degraded to {other:?}"),
        }
        // Full round-trip gen-core → mlx-gen → gen-core stays typed.
        let down: Error = up.into();
        match &down {
            Error::Unsupported(s) => assert_eq!(s, "hyper sampler"),
            other => panic!("gen-core→mlx-gen Unsupported degraded to {other:?}"),
        }
        let back: gen_core::Error = down.into();
        assert!(matches!(back, gen_core::Error::Unsupported(_)));
    }

    /// sc-22414: the load-coherence refusal crosses to gen-core as a message that still names the
    /// tensor, both checksums and the story, so the worker's report is actionable.
    #[test]
    fn incoherent_load_bridges_with_its_full_text() {
        let up: gen_core::Error = Error::IncoherentLoad {
            name: "model.embed_tokens.weight".into(),
            bytes: 2_013_265_920,
            cpu: 0xdead_beef,
            gpu: 0,
            attempts: 13,
        }
        .into();
        match up {
            gen_core::Error::Msg(text) => {
                for needle in [
                    "model.embed_tokens.weight",
                    "0xdeadbeef",
                    "13 reads",
                    "sc-22414",
                ] {
                    assert!(text.contains(needle), "{needle:?} missing from {text:?}");
                }
            }
            other => panic!("expected Msg, got {other:?}"),
        }
    }

    #[test]
    fn geometry_refusal_round_trips_with_telemetry_fields_intact() {
        let up: gen_core::Error = Error::GeometryRefused {
            reason: "measured infeasible".into(),
            requested_width: 1536,
            requested_height: 1024,
            alternative: Some((1024, 1024)),
        }
        .into();
        let down: Error = up.into();
        assert!(matches!(
            down,
            Error::GeometryRefused {
                requested_width: 1536,
                requested_height: 1024,
                alternative: Some((1024, 1024)),
                ..
            }
        ));
    }
}
