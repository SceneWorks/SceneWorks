//! Bounded, inert NPY artifacts carried by the existing project asset store.
//! Values are never pickled, converted to a lower precision, or rewritten on import.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub const MIME: &str = "application/x-npy";
pub const MAX_DATA_BYTES: usize = 512 * 1024 * 1024;
pub const MAX_HEADER_BYTES: usize = 4096;
pub const MAX_REQUEST_BYTES: usize = 1280 * 1024 * 1024 + 128 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorMetadata {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_length: usize,
    pub sha256: String,
}

fn header(file: &mut File) -> Result<(TensorMetadata, usize), String> {
    let stat = file.metadata().map_err(|e| e.to_string())?;
    if !stat.is_file() || stat.len() < 10 {
        return Err("tensor must be a nonempty regular NPY file".into());
    }
    let length = stat.len();
    if length > (MAX_DATA_BYTES + MAX_HEADER_BYTES + 12) as u64 {
        return Err("tensor exceeds the 512 MiB artifact budget".into());
    }
    let mut prefix = [0; 10];
    file.read_exact(&mut prefix).map_err(|e| e.to_string())?;
    if &prefix[..6] != b"\x93NUMPY" || prefix[7] != 0 {
        return Err("expected inert NPY version 1, 2 or 3".into());
    }
    let (n, start) = match prefix[6] {
        1 => (u16::from_le_bytes([prefix[8], prefix[9]]) as usize, 10),
        2 | 3 => {
            let mut tail = [0; 2];
            file.read_exact(&mut tail).map_err(|e| e.to_string())?;
            (
                u32::from_le_bytes([prefix[8], prefix[9], tail[0], tail[1]]) as usize,
                12,
            )
        }
        _ => return Err("unsupported NPY version".into()),
    };
    if n == 0 || n > MAX_HEADER_BYTES {
        return Err("tensor NPY header exceeds 4096 bytes".into());
    }
    let mut bytes = vec![0; n];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let text = std::str::from_utf8(&bytes).map_err(|_| "NPY header must be ASCII text")?;
    if !text.is_ascii() || !text.ends_with('\n') {
        return Err("NPY header must be an ASCII dictionary ending in newline".into());
    }
    let dictionary = text
        .trim()
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or("invalid NPY dictionary")?;
    // Parse the small inert dictionary grammar rather than evaluating Python literals.
    let mut fields = std::collections::BTreeMap::new();
    let mut depth = 0usize;
    let mut field_start = 0;
    for (index, character) in dictionary
        .char_indices()
        .chain(std::iter::once((dictionary.len(), ',')))
    {
        match character {
            '(' => {
                if depth != 0 {
                    return Err("nested NPY shapes are unsupported".into());
                }
                depth = 1;
            }
            ')' => {
                if depth != 1 {
                    return Err("invalid NPY shape tuple".into());
                }
                depth = 0;
            }
            ',' if depth == 0 => {
                let field = dictionary[field_start..index].trim();
                field_start = index + 1;
                if field.is_empty() {
                    continue;
                }
                let (key, value) = field
                    .split_once(':')
                    .ok_or("invalid NPY dictionary field")?;
                let key = key.trim();
                let key = key
                    .strip_prefix('\'')
                    .and_then(|s| s.strip_suffix('\''))
                    .or_else(|| key.strip_prefix('"').and_then(|s| s.strip_suffix('"')))
                    .ok_or("NPY keys must be quoted")?;
                if !["descr", "fortran_order", "shape"].contains(&key)
                    || fields.insert(key, value.trim()).is_some()
                {
                    return Err("unsupported or duplicate NPY header field".into());
                }
            }
            _ => (),
        }
    }
    if depth != 0 || fields.len() != 3 {
        return Err("NPY requires exactly descr, fortran_order and shape".into());
    }
    let value = |key: &str| -> Result<&str, String> {
        fields
            .get(key)
            .copied()
            .ok_or_else(|| format!("NPY requires {key}"))
    };
    if value("fortran_order")? != "False" {
        return Err("tensor must use C row-major order".into());
    }
    let descriptor = value("descr")?;
    let descriptor = descriptor
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
        .or_else(|| {
            descriptor
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
        })
        .ok_or("NPY dtype must be quoted")?;
    let dtype = match descriptor {
        "<f4" => "float32",
        "|u1" | "<u1" => "uint8",
        _ => return Err("tensor dtype must be little-endian float32 or uint8".into()),
    };
    let shape_text = value("shape")?;
    if !shape_text.starts_with('(') || !shape_text.ends_with(')') {
        return Err("NPY shape must be a tuple".into());
    }
    let dimensions = shape_text[1..shape_text.len() - 1]
        .trim()
        .trim_end_matches(',');
    let shape = dimensions
        .split(',')
        .map(str::trim)
        .map(|s| {
            if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
                return Err("invalid NPY shape".to_owned());
            }
            s.parse::<usize>()
                .map_err(|_| "invalid NPY shape".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !(2..=4).contains(&shape.len()) || shape.contains(&0) {
        return Err("tensor rank must be 2–4 with positive dimensions".into());
    }
    let elements = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or("tensor shape overflow")?;
    let data = elements
        .checked_mul(if dtype == "float32" { 4 } else { 1 })
        .filter(|n| *n <= MAX_DATA_BYTES)
        .ok_or("tensor payload exceeds budget")?;
    let offset = start + n;
    if length != (offset + data) as u64 {
        return Err("NPY payload length does not exactly match its shape and dtype".into());
    }
    Ok((
        TensorMetadata {
            dtype: dtype.into(),
            shape,
            byte_length: data,
            sha256: String::new(),
        },
        offset,
    ))
}

/// Header/shape/byte bounds precede payload reads; finite/binary checks stream in fixed memory.
pub fn inspect(path: &Path) -> Result<TensorMetadata, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let (mut metadata, offset) = header(&mut file)?;
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut remaining = offset;
    while remaining > 0 {
        let n = remaining.min(buffer.len());
        file.read_exact(&mut buffer[..n])
            .map_err(|e| e.to_string())?;
        hash.update(&buffer[..n]);
        remaining -= n;
    }
    remaining = metadata.byte_length;
    while remaining > 0 {
        let n = remaining.min(buffer.len());
        file.read_exact(&mut buffer[..n])
            .map_err(|e| e.to_string())?;
        hash.update(&buffer[..n]);
        let valid = if metadata.dtype == "float32" {
            buffer[..n]
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().expect("four bytes")).is_finite())
        } else {
            buffer[..n].iter().all(|b| *b <= 1)
        };
        if !valid {
            return Err("tensor values must be finite float32 or binary uint8".into());
        }
        remaining -= n;
    }
    // A concurrent append cannot escape the exact-length check.
    if file.read(&mut buffer[..1]).map_err(|e| e.to_string())? != 0 {
        return Err("tensor changed while reading".into());
    }
    metadata.sha256 = format!("{:x}", hash.finalize());
    Ok(metadata)
}

pub fn read_data(path: &Path, expected: &TensorMetadata) -> Result<Vec<u8>, String> {
    if &inspect(path)? != expected {
        return Err("tensor identity changed; reacquire the exact artifact".into());
    }
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let (meta, offset) = header(&mut file)?;
    if meta.shape != expected.shape
        || meta.dtype != expected.dtype
        || meta.byte_length != expected.byte_length
    {
        return Err("tensor header changed".into());
    }
    let mut bytes = Vec::with_capacity(offset + meta.byte_length);
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    file.take((offset + meta.byte_length + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() != offset + meta.byte_length
        || format!("{:x}", Sha256::digest(&bytes)) != expected.sha256
    {
        return Err("tensor identity changed while loading".into());
    }
    Ok(bytes.split_off(offset))
}

pub fn encode_f32(shape: &[usize], values: &[f32]) -> Result<Vec<u8>, String> {
    let elements = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or("tensor shape overflow")?;
    if !(2..=4).contains(&shape.len())
        || shape.contains(&0)
        || elements != values.len()
        || elements > MAX_DATA_BYTES / 4
        || !values.iter().all(|v| v.is_finite())
    {
        return Err("invalid finite FP32 tensor shape or byte budget".into());
    }
    let dimensions = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dimensions}), }}");
    let padding = (64 - (10 + dict.len() + 1) % 64) % 64;
    let n = dict.len() + padding + 1;
    let mut bytes = Vec::with_capacity(10 + n + values.len() * 4);
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(n as u16).to_le_bytes());
    bytes.extend_from_slice(dict.as_bytes());
    bytes.extend(std::iter::repeat_n(b' ', padding));
    bytes.push(b'\n');
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_fp32_bits_and_identity_survive_inert_storage() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("noise.npy");
        let values = [0., -0., f32::from_bits(1), -1.25];
        let bytes = encode_f32(&[1, 1, 2, 2], &values).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let meta = inspect(&path).unwrap();
        assert_eq!(meta.shape, [1, 1, 2, 2]);
        assert_eq!(meta.dtype, "float32");
        assert_eq!(meta.sha256, format!("{:x}", Sha256::digest(&bytes)));
        let raw = read_data(&path, &meta).unwrap();
        assert_eq!(
            raw,
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        );
        let mut changed = bytes;
        *changed.last_mut().unwrap() ^= 1;
        std::fs::write(&path, changed).unwrap();
        assert!(read_data(&path, &meta).is_err());
    }
    #[test]
    fn every_supported_npy_header_version_uses_its_actual_data_offset() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("version.npy");
        let header = b"{'shape': (1, 2), 'descr': '<f4', 'fortran_order': False, }\n";
        for version in [1u8, 2, 3] {
            let mut bytes = b"\x93NUMPY".to_vec();
            bytes.extend([version, 0]);
            if version == 1 {
                bytes.extend((header.len() as u16).to_le_bytes());
            } else {
                bytes.extend((header.len() as u32).to_le_bytes());
            }
            bytes.extend(header);
            bytes.extend((-0.0f32).to_le_bytes());
            bytes.extend(f32::from_bits(1).to_le_bytes());
            std::fs::write(&path, &bytes).unwrap();
            let metadata = inspect(&path).unwrap();
            let values = read_data(&path, &metadata).unwrap();
            assert_eq!(
                values,
                [(-0.0f32).to_le_bytes(), f32::from_bits(1).to_le_bytes()].concat()
            );
            assert_eq!(metadata.sha256, format!("{:x}", Sha256::digest(&bytes)));
        }
        for header in [
            "{'descr': '<f4', 'fortran_order': False, 'shape': (1,,2), }\n",
            "{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, 'shape': (1,2), }\n",
            "{'descr': '<f4', 'fortran_order': False, 'shape': (1,2), 'pickle': True, }\n",
        ] {
            let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
            bytes.extend((header.len() as u16).to_le_bytes());
            bytes.extend(header.as_bytes());
            bytes.extend([0; 8]);
            std::fs::write(&path, bytes).unwrap();
            assert!(inspect(&path).is_err());
        }
    }
    #[test]
    fn malformed_headers_dtypes_order_nonfinite_lengths_and_budget_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("input.npy");
        let valid = encode_f32(&[1, 2], &[0., 1.]).unwrap();
        let text = |descr: &str, order: &str, shape: &str| {
            let header =
                format!("{{'descr': '{descr}', 'fortran_order': {order}, 'shape': {shape}, }}\n");
            let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
            bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0; 8]);
            bytes
        };
        let mut nonfinite = valid.clone();
        let n = nonfinite.len();
        nonfinite[n - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        for bytes in [
            text("|O", "False", "(1,2)"),
            text(">f4", "False", "(1,2)"),
            text("<f8", "False", "(1,2)"),
            text("<f4", "True", "(1,2)"),
            text("<f4", "False", "(2,)"),
            text("<f4", "False", "(9999999999999999999,9999999999999999999)"),
            nonfinite,
            valid[..valid.len() - 1].to_vec(),
            [valid.clone(), vec![0]].concat(),
        ] {
            std::fs::write(&path, bytes).unwrap();
            assert!(inspect(&path).is_err());
        }
        let file = File::create(&path).unwrap();
        file.set_len((MAX_DATA_BYTES + MAX_HEADER_BYTES + 13) as u64)
            .unwrap();
        assert!(inspect(&path).unwrap_err().contains("budget"));
    }
}
