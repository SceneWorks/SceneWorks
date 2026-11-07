//! Depth-map assets (Iris-3B monocular depth, epic 25678 / sc-25682): the typed `image_depth` job
//! contract shared by the API and the worker, and the on-disk raw format — a float32 `H × W`
//! row-major little-endian (`<f4`) NumPy `.npy` file.
//!
//! The raw map is the deliverable. It is written once by the depth job and never rewritten: every
//! presentation form (the inferno preview, the near-bright control image a depth-control consumer
//! derives) is computed from a decoded copy. The format is plain NPY v1.0 so any consumer — NumPy,
//! SceneWorks' own control path, a future tool — can read it without knowing which model made it.

use serde::{Deserialize, Serialize};

/// The catalog id of the Iris-3B depth task export (`gen_core_iris::DEPTH_MODEL_ID`).
pub const DEPTH_MODEL_ID: &str = "iris_3b_depth";
/// The asset `type` of a depth map.
pub const DEPTH_ASSET_TYPE: &str = "depth";
/// The `file.mimeType` recorded for the raw map.
pub const NPY_MIME_TYPE: &str = "application/x-npy";
/// Upstream's default `max_side` (`gen_core_iris::depth::DEFAULT_DEPTH_MAX_SIDE`).
pub const DEFAULT_DEPTH_MAX_SIDE: u32 = 1024;
/// The smallest user-selectable cap: below this the 16-pixel patch grid leaves the model almost
/// nothing to look at.
pub const MIN_DEPTH_MAX_SIDE: u32 = 64;
/// The largest user-selectable cap. Native processing is the explicit choice above it.
pub const MAX_DEPTH_MAX_SIDE: u32 = 8192;

/// How large the model input is: upstream's `max_side` (capped, never upscaled) or the source size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "camelCase")]
pub enum DepthResolutionRequest {
    /// Downscale so the long side is at most `maxSide` (never upscale).
    Capped {
        #[serde(rename = "maxSide")]
        max_side: u32,
    },
    /// Run at the source resolution (rounded to the patch grid).
    Native,
}

impl Default for DepthResolutionRequest {
    fn default() -> Self {
        DepthResolutionRequest::Capped {
            max_side: DEFAULT_DEPTH_MAX_SIDE,
        }
    }
}

impl DepthResolutionRequest {
    /// `"capped"` | `"native"`.
    pub fn mode(self) -> &'static str {
        match self {
            DepthResolutionRequest::Capped { .. } => "capped",
            DepthResolutionRequest::Native => "native",
        }
    }

    pub fn validate(self) -> Result<(), String> {
        match self {
            DepthResolutionRequest::Capped { max_side }
                if !(MIN_DEPTH_MAX_SIDE..=MAX_DEPTH_MAX_SIDE).contains(&max_side) =>
            {
                Err(format!(
                    "resolution.maxSide must be between {MIN_DEPTH_MAX_SIDE} and \
                     {MAX_DEPTH_MAX_SIDE} (use mode \"native\" for the source resolution)"
                ))
            }
            _ => Ok(()),
        }
    }
}

/// The typed `image_depth` job payload (`POST /api/v1/image/depth/jobs`, and the stored job
/// payload the worker parses). One job estimates one source image; a batch is one job per image.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageDepthJobPayload {
    pub project_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_name: Option<String>,
    pub source_asset_id: String,
    #[serde(default)]
    pub resolution: DepthResolutionRequest,
    /// Names the result (`"<displayName> (depth)"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// A TRAINED depth model (a Training Studio `iris_3b_depth` run's registered export, sc-25687)
    /// to estimate with instead of the installed release export. `None` (or the release id) = the
    /// release export.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iris_model: Option<String>,
}

impl ImageDepthJobPayload {
    pub fn validate(&self) -> Result<(), String> {
        if self.project_id.trim().is_empty() {
            return Err("projectId is required".to_owned());
        }
        if self.source_asset_id.trim().is_empty() {
            return Err("sourceAssetId is required".to_owned());
        }
        if self
            .display_name
            .as_deref()
            .is_some_and(|name| name.chars().count() > 512)
        {
            return Err("displayName must be at most 512 characters".to_owned());
        }
        if let Some(model) = &self.iris_model {
            if !crate::image_restoration::is_plain_model_id(model) {
                return Err("irisModel must be the id of a trained Iris depth model".to_owned());
            }
        }
        self.resolution.validate()
    }

    /// The trained depth model this request names, `None` for the release export.
    pub fn trained_model(&self) -> Option<&str> {
        self.iris_model
            .as_deref()
            .filter(|model| *model != DEPTH_MODEL_ID)
    }

    /// Parse and validate a stored job payload. Fields the job store or the API's model-source seam
    /// attach (`modelManifestEntries`, `modelArtifactOperation`, …) are not part of the request
    /// contract and are ignored here.
    pub fn from_job_payload(
        payload: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, String> {
        let mut request = serde_json::Map::new();
        for key in [
            "projectId",
            "projectName",
            "sourceAssetId",
            "resolution",
            "displayName",
            "irisModel",
        ] {
            if let Some(value) = payload.get(key).filter(|value| !value.is_null()) {
                request.insert(key.to_owned(), value.clone());
            }
        }
        let parsed: Self = serde_json::from_value(serde_json::Value::Object(request))
            .map_err(|error| format!("invalid image_depth request: {error}"))?;
        parsed.validate()?;
        Ok(parsed)
    }
}

/// A decoded raw depth map.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthArray {
    pub width: u32,
    pub height: u32,
    /// Row-major `height × width`.
    pub values: Vec<f32>,
}

const NPY_MAGIC: &[u8] = b"\x93NUMPY";
/// NumPy aligns the data start to 64 bytes (`numpy.lib.format.ARRAY_ALIGN`).
const NPY_ALIGN: usize = 64;

/// Encode a row-major `height × width` float32 map as NPY v1.0 (`<f4`, C order).
pub fn encode_npy_f32(width: u32, height: u32, values: &[f32]) -> Result<Vec<u8>, String> {
    let expected = width as usize * height as usize;
    if width == 0 || height == 0 || values.len() != expected {
        return Err(format!(
            "a {width}x{height} depth map needs {expected} samples, got {}",
            values.len()
        ));
    }
    let dict =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({height}, {width}), }}");
    // magic(6) + version(2) + header_len(2) + dict + padding + '\n' ≡ 0 (mod 64)
    let unpadded = NPY_MAGIC.len() + 2 + 2 + dict.len() + 1;
    let padding = (NPY_ALIGN - unpadded % NPY_ALIGN) % NPY_ALIGN;
    let header_len = dict.len() + padding + 1;
    let header_len_u16 = u16::try_from(header_len).map_err(|_| "NPY header too long".to_owned())?;
    let mut out = Vec::with_capacity(unpadded + padding + values.len() * 4);
    out.extend_from_slice(NPY_MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&header_len_u16.to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat_n(b' ', padding));
    out.push(b'\n');
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// Decode a 2-D float32 NPY (v1.0 / v2.0 / v3.0 header, `<f4` or `|f4`-equivalent little-endian,
/// C order). Anything else — another dtype, Fortran order, a non-2-D shape, a truncated or padded
/// payload — is refused rather than reinterpreted.
pub fn decode_npy_f32(bytes: &[u8]) -> Result<DepthArray, String> {
    if bytes.len() < 10 || &bytes[..6] != NPY_MAGIC {
        return Err("not an NPY file (bad magic)".to_owned());
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated NPY header".to_owned());
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        other => return Err(format!("unsupported NPY version {other}")),
    };
    let data_start = header_start + header_len;
    let header = bytes
        .get(header_start..data_start)
        .ok_or_else(|| "truncated NPY header".to_owned())?;
    let header = std::str::from_utf8(header).map_err(|_| "NPY header is not text".to_owned())?;
    let descr = dict_value(header, "descr").ok_or_else(|| "NPY header has no descr".to_owned())?;
    if descr != "'<f4'" {
        return Err(format!(
            "depth maps are little-endian float32 ('<f4'); this file is {descr}"
        ));
    }
    let fortran = dict_value(header, "fortran_order")
        .ok_or_else(|| "NPY header has no fortran_order".to_owned())?;
    if fortran != "False" {
        return Err("depth maps are row-major (fortran_order False)".to_owned());
    }
    let shape = dict_value(header, "shape").ok_or_else(|| "NPY header has no shape".to_owned())?;
    let dims: Vec<usize> = shape
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| format!("bad NPY shape {shape}"))
        })
        .collect::<Result<_, _>>()?;
    let [height, width] = dims[..] else {
        return Err(format!("a depth map is 2-D (H, W); this file is {shape}"));
    };
    let (Ok(width32), Ok(height32)) = (u32::try_from(width), u32::try_from(height)) else {
        return Err(format!("NPY shape {shape} is too large"));
    };
    if width == 0 || height == 0 {
        return Err(format!("empty depth map {shape}"));
    }
    let data = &bytes[data_start..];
    let expected = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| format!("NPY shape {shape} is too large"))?;
    if data.len() != expected {
        return Err(format!(
            "NPY payload is {} bytes; a {height}x{width} float32 map is {expected}",
            data.len()
        ));
    }
    let values = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok(DepthArray {
        width: width32,
        height: height32,
        values,
    })
}

/// The raw text of one value in NumPy's header dict (`'key': value,`).
fn dict_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("'{key}':");
    let start = header.find(&needle)? + needle.len();
    let rest = header[start..].trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')')? + 1
    } else {
        rest.find([',', '}'])?
    };
    Some(rest[..end].trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn npy_round_trips_shape_dtype_byte_order_and_bits() {
        // Values that exercise sign, subnormals, infinities, NaN payloads and -0.0: the round trip
        // is compared bit-for-bit, not numerically.
        let values = vec![
            -1.25,
            0.0,
            -0.0,
            1.0e-40,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_1234),
            1.75,
            3.4e38,
            -2.5e-3,
            0.5,
            42.0,
        ];
        let bytes = encode_npy_f32(4, 3, &values).unwrap();
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0, "data is 64-byte aligned");
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
        assert!(header.starts_with("{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }"));
        assert!(header.ends_with('\n'));
        // Little-endian row-major: the first sample's bytes follow the header directly.
        assert_eq!(
            &bytes[10 + header_len..10 + header_len + 4],
            &(-1.25f32).to_le_bytes()
        );
        assert_eq!(bytes.len(), 10 + header_len + 12 * 4);

        let decoded = decode_npy_f32(&bytes).unwrap();
        assert_eq!((decoded.width, decoded.height), (4, 3));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&decoded.values), bits(&values));
    }

    #[test]
    fn npy_decoder_refuses_what_it_cannot_read_faithfully() {
        let good = encode_npy_f32(2, 1, &[1.0, 2.0]).unwrap();
        // Same-length substitutions inside the (ASCII) header only.
        let swap = |from: &str, to: &str| {
            assert_eq!(from.len(), to.len());
            let header_end = 10 + u16::from_le_bytes([good[8], good[9]]) as usize;
            let header = std::str::from_utf8(&good[10..header_end])
                .unwrap()
                .replace(from, to);
            let mut bytes = good.clone();
            bytes[10..header_end].copy_from_slice(header.as_bytes());
            bytes
        };
        assert!(
            decode_npy_f32(&swap("'<f4'", "'>f4'")).is_err(),
            "big-endian"
        );
        assert!(decode_npy_f32(&swap("'<f4'", "'<f8'")).is_err(), "float64");
        assert!(
            decode_npy_f32(&swap("False", "True ")).is_err(),
            "fortran order"
        );
        assert!(
            decode_npy_f32(&good[..good.len() - 1]).is_err(),
            "truncated"
        );
        let mut extra = good.clone();
        extra.extend_from_slice(&[0, 0, 0, 0]);
        assert!(decode_npy_f32(&extra).is_err(), "trailing data");
        assert!(decode_npy_f32(b"not an npy file").is_err());
        assert!(encode_npy_f32(2, 2, &[1.0]).is_err());
    }

    #[test]
    fn the_request_contract_defaults_to_a_1024_cap_and_bounds_the_cap() {
        let payload = |value: serde_json::Value| value.as_object().unwrap().clone();
        let parsed = ImageDepthJobPayload::from_job_payload(&payload(json!({
            "projectId": "p", "sourceAssetId": "a",
            "modelManifestEntries": [{ "id": "iris_3b_depth" }],
        })))
        .unwrap();
        assert_eq!(
            parsed.resolution,
            DepthResolutionRequest::Capped { max_side: 1024 }
        );
        let native = ImageDepthJobPayload::from_job_payload(&payload(json!({
            "projectId": "p", "sourceAssetId": "a", "resolution": { "mode": "native" },
        })))
        .unwrap();
        assert_eq!(native.resolution, DepthResolutionRequest::Native);
        let capped = ImageDepthJobPayload::from_job_payload(&payload(json!({
            "projectId": "p", "sourceAssetId": "a",
            "resolution": { "mode": "capped", "maxSide": 2048 },
        })))
        .unwrap();
        assert_eq!(
            serde_json::to_value(capped.resolution).unwrap(),
            json!({ "mode": "capped", "maxSide": 2048 })
        );
        for bad in [
            json!({ "projectId": "p", "sourceAssetId": "a", "resolution": { "mode": "capped", "maxSide": 0 } }),
            json!({ "projectId": "p", "sourceAssetId": "a", "resolution": { "mode": "capped", "maxSide": 9000 } }),
            json!({ "projectId": "p", "sourceAssetId": "a", "resolution": { "mode": "metric" } }),
            json!({ "projectId": "p", "sourceAssetId": "" }),
            json!({ "projectId": "", "sourceAssetId": "a" }),
        ] {
            assert!(
                ImageDepthJobPayload::from_job_payload(&payload(bad.clone())).is_err(),
                "{bad}"
            );
        }
    }
}
