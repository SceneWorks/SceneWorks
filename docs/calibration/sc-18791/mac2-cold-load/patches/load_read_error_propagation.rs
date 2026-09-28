//! sc-22414: a safetensors payload read that fails must surface as an `Err` from `eval`, never as a
//! silently "successful" array over an unfilled buffer.
//!
//! Regression guard for `patches/load-read-error-propagation.patch`. Upstream MLX 0.32.0 runs the
//! `Load` primitive's file read inside a `std::packaged_task` on its IO pool and has the CPU-stream
//! task `wait()` on the future — which never re-raises the captured exception. A `pread` that
//! returns 0/-1 (truncated file, EOF, EIO under pressure) therefore left the destination buffer
//! as freshly allocated memory and evaluation reported success. Observed in production as a
//! deterministic, wrong text-conditioning tensor on a cold page cache under write-back pressure
//! (LTX-2.5 measured-vs-warm parity breach). The patch re-raises via `get()`, retries `EINTR`,
//! names the errno, and advances the offset on short reads.
//!
//! The deterministic way to make `pread` fail is to truncate the file AFTER the lazy load has
//! parsed (and validated) the header: the deferred payload read then hits EOF.

use std::collections::HashMap;
use std::fs::OpenOptions;

use mlx_rs::{transforms::eval, Array};

#[test]
fn truncated_safetensors_payload_is_an_error_not_silent_garbage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.safetensors");

    // 4 MiB of non-zero f32 so an unfilled buffer could never pass as the real payload.
    let values: Vec<f32> = (0..(1 << 20)).map(|i| (i % 251) as f32 + 1.0).collect();
    let array = Array::from_slice(&values, &[1 << 20]);
    let arrays: HashMap<String, Array> = HashMap::from([("w".to_string(), array)]);
    Array::save_safetensors(&arrays, None, &path).unwrap();

    // Open lazily first: the header parses and validates against the full file, and the reader
    // keeps its descriptor. Only then cut the payload in half, so the deferred payload `pread`
    // is what hits EOF — the exact path the patch turns from silent success into an error.
    let loaded = Array::load_safetensors(&path).unwrap();
    let w = loaded.get("w").unwrap();
    let full_len = std::fs::metadata(&path).unwrap().len();
    let file = OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(full_len - (2 << 20)).unwrap();
    drop(file);
    let result = eval([w]);
    assert!(
        result.is_err(),
        "a truncated payload read must fail evaluation; upstream MLX 0.32.0 reports success over \
         an unfilled buffer (sc-22414)"
    );
}
