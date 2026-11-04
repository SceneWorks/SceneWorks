//! Shared image codec owner lives in core, independent of worker/backend dependencies.

pub(crate) use sceneworks_core::image_decode::decode_image_any;
#[cfg(any(test, native_lane))]
pub(crate) use sceneworks_core::image_decode::decode_image_bytes_any;
