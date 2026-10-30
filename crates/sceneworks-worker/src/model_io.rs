//! Shared ONNX ingress. Read the graph and the union of selected external tensor ranges once.
//! Compact external buffers retain tensor aliases; only decimal offset strings in the in-memory
//! protobuf are changed (same byte lengths). ORT copies initializers; the builder drops afterwards.
use crate::{WorkerError, WorkerResult};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;

fn read_ranges(path: &Path, ranges: &[Range<u64>]) -> WorkerResult<Vec<u8>> {
    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let total = ranges.iter().try_fold(0u64, |total, range| {
        if range.start > range.end || range.end > size {
            return Err(malformed());
        }
        total
            .checked_add(range.end - range.start)
            .ok_or_else(malformed)
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(total).map_err(|_| malformed())?)
        .map_err(|error| WorkerError::Engine(format!("cannot buffer model bytes: {error}")))?;
    if total == 0 {
        return Ok(bytes);
    }
    let progress = gen_core::LoadActivityContext::capture().file(
        gen_core::LoadActivityPhase::Loading,
        path,
        Some(total),
    );
    let mut chunk = vec![0; 1024 * 1024];
    for range in ranges {
        file.seek(SeekFrom::Start(range.start))?;
        let mut remaining = range.end - range.start;
        while remaining > 0 {
            let count =
                usize::try_from(remaining.min(chunk.len() as u64)).map_err(|_| malformed())?;
            let read = file.read(&mut chunk[..count])?;
            if read == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "model source was truncated during load",
                )
                .into());
            }
            bytes.extend_from_slice(&chunk[..read]);
            progress.add_bytes(read as u64);
            remaining -= read as u64;
        }
    }
    progress.finish();
    Ok(bytes)
}
fn read_weights(path: &Path) -> WorkerResult<Vec<u8>> {
    let length = std::fs::metadata(path)?.len();
    read_ranges(path, std::slice::from_ref(&(0..length)))
}

#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) fn ort_session(
    mut builder: ort::session::builder::SessionBuilder,
    path: &Path,
) -> WorkerResult<ort::session::Session> {
    let mut model = read_weights(path)?;
    for (name, bytes) in external_buffers(path, &mut model)? {
        builder = builder
            .with_external_initializer_file_in_memory(&name, bytes.into())
            .map_err(|error| WorkerError::Engine(format!("onnxruntime: {error}")))?;
    }
    let session = builder
        .commit_from_memory(&model)
        .map_err(|error| WorkerError::Engine(format!("onnxruntime: {error}")));
    drop(builder);
    session
}

#[derive(Debug)]
struct External {
    location: String,
    offset: u64,
    length: u64,
    offset_value: Option<Range<usize>>,
    embed: Option<Range<usize>>,
}

fn external_buffers(path: &Path, model: &mut Vec<u8>) -> WorkerResult<Vec<(String, Vec<u8>)>> {
    // ORT itself selects FlatBuffer models by this standard file identifier. Do not feed an
    // optimized ORT buffer to the ONNX protobuf walker; its embedded bytes are already read.
    if model.get(4..8) == Some(b"ORTM") {
        return Ok(Vec::new());
    }
    let mut references = Vec::new();
    visit(model, 0, Kind::Model, 0, &mut references)?;
    let mut files: BTreeMap<String, Vec<External>> = BTreeMap::new();
    for reference in references {
        files
            .entry(reference.location.clone())
            .or_default()
            .push(reference);
    }
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    let mut buffers = Vec::new();
    let mut embedded = BTreeMap::new();
    for (name, references) in files {
        // Preserve ORT's confinement rule before replacing its file loader with memory buffers.
        let source = external_path(path, directory, &name)?;
        let size = std::fs::metadata(&source)?.len();
        let ranges = union_ranges(&references, size)?;
        let bytes = read_ranges(&source, &ranges)?;
        for reference in &references {
            let offset = compact_offset(&ranges, reference.offset, reference.length)?;
            if let Some(tensor) = &reference.embed {
                let start = usize::try_from(offset).map_err(|_| malformed())?;
                let length = usize::try_from(reference.length).map_err(|_| malformed())?;
                let mut body = Vec::new();
                fields(&model[tensor.clone()], tensor.start, |number, _, _, raw| {
                    if !matches!(number, 9 | 13 | 14) {
                        body.extend_from_slice(raw);
                    }
                    Ok(())
                })?;
                append_field(&mut body, 9, &bytes[start..start + length]);
                embedded.insert(tensor.start, body);
            } else if let Some(value) = &reference.offset_value {
                let replacement = format!("{offset:0width$}", width = value.len());
                if replacement.len() != value.len() {
                    return Err(malformed());
                }
                model[value.clone()].copy_from_slice(replacement.as_bytes());
            } else if offset != 0 {
                return Err(malformed());
            }
        }
        if references.iter().any(|reference| reference.embed.is_none()) {
            buffers.push((name, bytes));
        }
    }
    if !embedded.is_empty() {
        *model = embed_tensors(model, 0, Kind::Model, &mut embedded)?.ok_or_else(malformed)?;
        if !embedded.is_empty() {
            return Err(malformed());
        }
    }
    Ok(buffers)
}
fn external_path(model: &Path, directory: &Path, name: &str) -> WorkerResult<std::path::PathBuf> {
    let relative = Path::new(name);
    if relative.components().any(|part| {
        matches!(
            part,
            std::path::Component::Prefix(_) | std::path::Component::RootDir
        )
    }) {
        return Err(malformed());
    }
    let source = directory.join(relative).canonicalize()?;
    let root = directory.canonicalize()?;
    let real_root = model
        .canonicalize()?
        .parent()
        .ok_or_else(malformed)?
        .to_path_buf();
    if !source.starts_with(root) && !source.starts_with(real_root) {
        return Err(malformed());
    }
    // Retain the logical snapshot path for component attribution after validating its target.
    Ok(directory.join(relative))
}
fn union_ranges(references: &[External], file_size: u64) -> WorkerResult<Vec<Range<u64>>> {
    let mut ranges = Vec::new();
    for reference in references {
        let end = reference
            .offset
            .checked_add(reference.length)
            .ok_or_else(malformed)?;
        if end > file_size {
            return Err(malformed());
        }
        if reference.length > 0 {
            ranges.push(reference.offset..end);
        }
    }
    ranges.sort_by_key(|range| range.start);
    let mut union: Vec<Range<u64>> = Vec::new();
    for range in ranges {
        if let Some(last) = union.last_mut().filter(|last| range.start <= last.end) {
            last.end = last.end.max(range.end);
        } else {
            union.push(range);
        }
    }
    Ok(union)
}
fn compact_offset(ranges: &[Range<u64>], offset: u64, length: u64) -> WorkerResult<u64> {
    if length == 0 {
        return Ok(0);
    }
    let end = offset.checked_add(length).ok_or_else(malformed)?;
    let mut compact = 0u64;
    for range in ranges {
        if offset >= range.start && end <= range.end {
            return Ok(compact + offset - range.start);
        }
        compact = compact
            .checked_add(range.end - range.start)
            .ok_or_else(malformed)?;
    }
    Err(malformed())
}

#[derive(Clone, Copy)]
enum Kind {
    Model,
    Graph,
    Node,
    Attribute,
    Tensor,
    Sparse,
    Training,
    Function,
}
/// Traverse only schema fields containing tensors. Payloads and unknown fields are never parsed.
fn visit(
    bytes: &[u8],
    base: usize,
    kind: Kind,
    depth: usize,
    refs: &mut Vec<External>,
) -> WorkerResult<()> {
    if depth > 128 {
        return Err(malformed());
    }
    if matches!(kind, Kind::Tensor) {
        return tensor(bytes, base, depth != 2, refs);
    }
    fields(bytes, base, |number, value, _, _| {
        let Some((payload, range)) = value else {
            return Ok(());
        };
        let child = child_kind(kind, number);
        if let Some(child) = child {
            visit(payload, range.start, child, depth + 1, refs)?;
        }
        Ok(())
    })
}
fn child_kind(kind: Kind, number: u64) -> Option<Kind> {
    match (kind, number) {
        (Kind::Model, 7) | (Kind::Training, 1 | 2) => Some(Kind::Graph),
        (Kind::Model, 20) => Some(Kind::Training),
        (Kind::Model, 25) => Some(Kind::Function),
        (Kind::Graph, 1) | (Kind::Function, 7) => Some(Kind::Node),
        (Kind::Graph, 5) | (Kind::Attribute, 5 | 10) | (Kind::Sparse, 1 | 2) => Some(Kind::Tensor),
        (Kind::Graph, 15) | (Kind::Attribute, 22 | 23) => Some(Kind::Sparse),
        (Kind::Node, 5) | (Kind::Function, 11) => Some(Kind::Attribute),
        (Kind::Attribute, 6 | 11) => Some(Kind::Graph),
        _ => None,
    }
}
fn embed_tensors(
    bytes: &[u8],
    base: usize,
    kind: Kind,
    embedded: &mut BTreeMap<usize, Vec<u8>>,
) -> WorkerResult<Option<Vec<u8>>> {
    if matches!(kind, Kind::Tensor) {
        return Ok(embedded.remove(&base));
    }
    if embedded.range(base..base + bytes.len()).next().is_none() {
        return Ok(None);
    }
    let mut output = Vec::new();
    let mut changed = false;
    fields(bytes, base, |number, payload, _, raw| {
        if let Some((body, range)) = payload {
            if let Some(child) = child_kind(kind, number) {
                if let Some(replacement) = embed_tensors(body, range.start, child, embedded)? {
                    append_field(&mut output, number, &replacement);
                    changed = true;
                    return Ok(());
                }
            }
        }
        output.extend_from_slice(raw);
        Ok(())
    })?;
    Ok(changed.then_some(output))
}
fn append_varint(output: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 127) as u8;
        value >>= 7;
        output.push(byte | if value == 0 { 0 } else { 128 });
        if value == 0 {
            return;
        }
    }
}
fn append_field(output: &mut Vec<u8>, number: u64, payload: &[u8]) {
    append_varint(output, number * 8 + 2);
    append_varint(output, payload.len() as u64);
    output.extend_from_slice(payload);
}

fn tensor(bytes: &[u8], base: usize, embed: bool, refs: &mut Vec<External>) -> WorkerResult<()> {
    let mut entries: Vec<(String, String, Range<usize>)> = Vec::new();
    let mut external = false;
    let mut dimensions = Vec::new();
    let mut data_type = 0;
    fields(bytes, base, |number, payload, integer, _| {
        match number {
            1 => {
                if let Some(value) = integer {
                    dimensions.push(value);
                } else if let Some((mut packed, _)) = payload {
                    while !packed.is_empty() {
                        dimensions.push(varint(&mut packed)?);
                    }
                }
            }
            2 => data_type = integer.ok_or_else(malformed)?,
            14 => external = integer.ok_or_else(malformed)? == 1,
            13 => {
                let (entry, range) = payload.ok_or_else(malformed)?;
                let (mut key, mut value) = (None, None);
                fields(entry, range.start, |n, data, _, _| {
                    if let Some((data, range)) = data {
                        if n == 1 {
                            key = Some(
                                std::str::from_utf8(data)
                                    .map_err(|_| malformed())?
                                    .to_owned(),
                            );
                        }
                        if n == 2 {
                            value = Some((
                                std::str::from_utf8(data)
                                    .map_err(|_| malformed())?
                                    .to_owned(),
                                range,
                            ));
                        }
                    }
                    Ok(())
                })?;
                let (value, range) = value.ok_or_else(malformed)?;
                entries.push((key.ok_or_else(malformed)?, value, range));
            }
            _ => {}
        }
        Ok(())
    })?;
    if !external {
        return Ok(());
    }
    let find = |key: &str| entries.iter().rev().find(|entry| entry.0 == key);
    let location = find("location")
        .filter(|entry| !entry.1.is_empty())
        .ok_or_else(malformed)?
        .1
        .clone();
    let offset = find("offset").map_or(Ok(0), |entry| decimal(&entry.1))?;
    let length = tensor_bytes(data_type, &dimensions)?;
    if let Some(explicit) = find("length") {
        let declared = unsigned(&explicit.1)?;
        if declared != 0 && declared != length {
            return Err(malformed());
        }
    }
    refs.push(External {
        location: location.clone(),
        offset,
        length,
        offset_value: find("offset").map(|entry| entry.2.clone()),
        embed: embed.then_some(base..base + bytes.len()),
    });
    // ORT's external data can also contain optional prepacked blobs from the same file.
    for (key, value, span) in &entries {
        if embed || !key.starts_with("prepacked") {
            continue;
        }
        let mut position = 0;
        for (index, blob) in value.split('|').enumerate() {
            if index > 0 {
                let parts: Vec<_> = blob.split(';').collect();
                if parts.len() == 3 {
                    refs.push(External {
                        location: location.clone(),
                        offset: decimal(parts[0])?,
                        length: unsigned(parts[1])?,
                        embed: None,
                        offset_value: Some(
                            span.start + position..span.start + position + parts[0].len(),
                        ),
                    });
                }
            }
            position += blob.len() + 1;
        }
    }
    Ok(())
}
fn tensor_bytes(data_type: u64, dimensions: &[u64]) -> WorkerResult<u64> {
    let bits: u64 = match data_type {
        1 | 6 | 12 => 32,
        2 | 3 | 9 | 17..=20 | 24 => 8,
        4 | 5 | 10 | 16 => 16,
        7 | 11 | 13 | 14 => 64,
        15 => 128,
        21..=23 => 4,
        25 | 26 => 2,
        _ => return Err(malformed()),
    };
    let count = dimensions.iter().try_fold(1u64, |count, &dim| {
        if dim > i64::MAX as u64 {
            return Err(malformed());
        }
        count.checked_mul(dim).ok_or_else(malformed)
    })?;
    count
        .checked_mul(bits)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .ok_or_else(malformed)
}
fn decimal(value: &str) -> WorkerResult<u64> {
    let number: i64 = value.parse().map_err(|_| malformed())?;
    u64::try_from(number).map_err(|_| malformed())
}
fn unsigned(value: &str) -> WorkerResult<u64> {
    if value.starts_with("0x") || value.starts_with("0X") {
        u64::from_str_radix(&value[2..], 16).map_err(|_| malformed())
    } else {
        value.parse().map_err(|_| malformed())
    }
}
type Payload<'a> = Option<(&'a [u8], Range<usize>)>;
fn fields(
    bytes: &[u8],
    base: usize,
    mut field: impl FnMut(u64, Payload<'_>, Option<u64>, &[u8]) -> WorkerResult<()>,
) -> WorkerResult<()> {
    let mut input = bytes;
    while !input.is_empty() {
        let before = input;
        let tag = varint(&mut input)?;
        let number = tag >> 3;
        if number == 0 {
            return Err(malformed());
        }
        let (payload, integer) = match tag & 7 {
            0 => (None, Some(varint(&mut input)?)),
            1 => {
                take(&mut input, 8)?;
                (None, None)
            }
            2 => {
                let size = usize::try_from(varint(&mut input)?).map_err(|_| malformed())?;
                let start = base + bytes.len() - input.len();
                (Some((take(&mut input, size)?, start..start + size)), None)
            }
            5 => {
                take(&mut input, 4)?;
                (None, None)
            }
            _ => return Err(malformed()),
        };
        field(
            number,
            payload,
            integer,
            &before[..before.len() - input.len()],
        )?;
    }
    Ok(())
}
fn take<'a>(input: &mut &'a [u8], length: usize) -> WorkerResult<&'a [u8]> {
    if length > input.len() {
        return Err(malformed());
    }
    let (part, rest) = input.split_at(length);
    *input = rest;
    Ok(part)
}
fn varint(input: &mut &[u8]) -> WorkerResult<u64> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let byte = *input.first().ok_or_else(malformed)?;
        *input = &input[1..];
        if shift == 63 && byte > 1 {
            return Err(malformed());
        }
        value |= u64::from(byte & 127) << shift;
        if byte & 128 == 0 {
            return Ok(value);
        }
    }
    Err(malformed())
}
fn malformed() -> WorkerError {
    WorkerError::Engine("invalid ONNX external tensor metadata or range".to_owned())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn shared_reader_reports_only_bytes_actually_read_and_finishes() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.onnx");
        let expected = vec![37; 1024 * 1024 + 17];
        std::fs::write(&path, &expected).unwrap();
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = events.clone();
        let _observer =
            gen_core::LoadActivityObserver::new(move |event| seen.lock().unwrap().push(event));
        assert_eq!(read_weights(&path).unwrap(), expected);
        let events = events.lock().unwrap();
        let last = events
            .iter()
            .filter_map(Option::as_ref)
            .next_back()
            .unwrap();
        assert_eq!(last.bytes_read, expected.len() as u64);
        assert_eq!(last.total_bytes, Some(expected.len() as u64));
        assert!(events.last().unwrap().is_none());
    }

    fn vi(mut value: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let byte = (value & 127) as u8;
            value >>= 7;
            bytes.push(byte | if value > 0 { 128 } else { 0 });
            if value == 0 {
                return bytes;
            }
        }
    }
    fn field(number: u64, data: &[u8]) -> Vec<u8> {
        [vi(number * 8 + 2), vi(data.len() as u64), data.to_vec()].concat()
    }
    fn integer(number: u64, value: u64) -> Vec<u8> {
        [vi(number * 8), vi(value)].concat()
    }
    fn entry(key: &str, value: &str) -> Vec<u8> {
        field(
            13,
            &[field(1, key.as_bytes()), field(2, value.as_bytes())].concat(),
        )
    }
    fn ext(name: &str, offset: u64, count: u64, length: bool) -> Vec<u8> {
        let mut tensor = [
            integer(1, count),
            integer(2, 1),
            field(8, name.as_bytes()),
            integer(14, 1),
            entry("location", "weights.bin"),
            entry("offset", &offset.to_string()),
        ]
        .concat();
        if length {
            tensor.extend(entry("length", &(count * 4).to_string()));
        }
        tensor
    }
    fn refs(model: &[u8]) -> Vec<External> {
        let mut refs = Vec::new();
        visit(model, 0, Kind::Model, 0, &mut refs).unwrap();
        refs
    }
    fn model(tensors: &[Vec<u8>]) -> Vec<u8> {
        field(
            7,
            &tensors
                .iter()
                .flat_map(|tensor| field(5, tensor))
                .collect::<Vec<_>>(),
        )
    }
    #[test]
    fn selected_external_ranges_compact_aliases_overlap_and_preserve_unknown_fields() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.onnx");
        let source = directory.path().join("weights.bin");
        let mut data = std::fs::File::create(&source).unwrap();
        data.set_len(1 << 24).unwrap(); // Unreferenced 16 MiB: bounded on NTFS, whose set_len need not be sparse.
        data.seek(SeekFrom::Start(4096)).unwrap();
        data.write_all(&[9; 32]).unwrap();
        data.seek(SeekFrom::Start(8192)).unwrap();
        data.write_all(&[7; 16]).unwrap();
        let first = ext("x", 4096, 4, true);
        let overlapping = ext("y", 4104, 4, false); // omitted length uses tensor shape, not EOF
        let last = ext("z", 8192, 4, true);
        let nested = field(5, &last);
        let mut graph = [
            field(5, &first),
            field(5, &first),
            field(5, &overlapping),
            nested,
        ]
        .concat();
        graph.extend(field(99, b"unknown field unchanged"));
        let mut model = field(7, &graph);
        let original = model.clone();
        std::fs::write(&path, &model).unwrap();
        let buffers = external_buffers(&path, &mut model).unwrap();
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].1, [vec![9; 24], vec![7; 16]].concat());
        let patched = refs(&model);
        assert_eq!(
            patched.iter().map(|r| r.offset).collect::<Vec<_>>(),
            [0, 0, 8, 24]
        );
        assert_eq!(&model[patched[3].offset_value.clone().unwrap()], b"0024");
        let mut restored = model.clone();
        for reference in refs(&original) {
            let range = reference.offset_value.unwrap();
            restored[range.clone()].copy_from_slice(&original[range]);
        }
        assert_eq!(restored, original, "only offset value bytes may change");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            original,
            "source model stays immutable"
        );
    }
    #[test]
    fn optimized_ort_buffers_bypass_the_onnx_metadata_walker_unchanged() {
        let mut bytes = b"\x08\x00\x00\x00ORTMopaque-flatbuffer".to_vec();
        let original = bytes.clone();
        assert!(external_buffers(Path::new("unused.ort"), &mut bytes)
            .unwrap()
            .is_empty());
        assert_eq!(bytes, original);
    }

    #[test]
    fn invalid_external_ranges_fail_before_reading_or_allocating_source_file_size() {
        let make = |offset, length| External {
            location: "weights.bin".to_owned(),
            offset,
            length,
            offset_value: None,
            embed: None,
        };
        assert!(union_ranges(&[make(u64::MAX, 4)], u64::MAX).is_err());
        assert!(union_ranges(&[make(12, 8)], 16).is_err());
        assert_eq!(
            union_ranges(&[make(4, 4), make(6, 4), make(4, 4)], 64).unwrap(),
            [Range { start: 4, end: 10 }]
        );
        assert_eq!(tensor_bytes(21, &[3]).unwrap(), 2);
        assert_eq!(tensor_bytes(25, &[5]).unwrap(), 2);
        assert!(tensor_bytes(1, &[u64::MAX]).is_err());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.onnx");
        let mut graph = model(&[ext("x", 0, 1, true)]);
        std::fs::write(&path, &graph).unwrap();
        assert!(
            external_buffers(&path, &mut graph).is_err(),
            "missing source"
        );
        std::fs::write(dir.path().join("weights.bin"), [1, 2]).unwrap();
        assert!(
            external_buffers(&path, &mut graph).is_err(),
            "truncated source"
        );
    }
    #[test]
    fn schema_walk_includes_sparse_function_training_and_attribute_tensors() {
        let tensor = ext("x", 4096, 1, true);
        let sparse = field(1, &tensor);
        let graph = [field(15, &sparse), field(1, &field(5, &field(22, &sparse)))].concat();
        let model = [
            field(7, &graph),
            field(20, &field(1, &field(5, &tensor))),
            field(25, &field(11, &field(10, &tensor))),
        ]
        .concat();
        assert_eq!(refs(&model).len(), 4);
        assert!(visit(&[0x3a, 0xff], 0, Kind::Model, 0, &mut Vec::new()).is_err());
    }

    #[test]
    fn nested_external_tensors_embed_selected_bytes_and_keep_unknown_fields() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.onnx");
        let source = directory.path().join("weights.bin");
        let mut file = std::fs::File::create(&source).unwrap();
        file.set_len(65536).unwrap();
        file.seek(SeekFrom::Start(4096)).unwrap();
        file.write_all(&[17; 16]).unwrap();
        let tensor = [ext("nested", 4096, 4, false), field(99, b"preserved")].concat();
        let wrap = |tensor: &[u8]| field(7, &field(1, &field(5, &field(6, &field(5, tensor)))));
        let mut model = wrap(&tensor);
        std::fs::write(&path, &model).unwrap();
        let buffers = external_buffers(&path, &mut model).unwrap();
        let expected = [
            integer(1, 4),
            integer(2, 1),
            field(8, b"nested"),
            field(99, b"preserved"),
            field(9, &[17; 16]),
        ]
        .concat();
        assert_eq!(model, wrap(&expected));
        assert!(refs(&model).is_empty());
        assert!(
            buffers.is_empty(),
            "nested-only file buffers must release before ORT parsing"
        );
    }

    // Session creation can execute ORT graph optimizations. It is deliberately excluded from
    // ordinary CI; the explicit sc-27984 validation profile records this dedicated runtime smoke.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    #[ignore = "dedicated model validation only: ORT session creation may execute graph kernels"]
    fn external_initializer_session_creation_acceptance() {
        fn output(name: &str, count: u64) -> Vec<u8> {
            let shape = field(1, &integer(1, count));
            let tensor_type = [integer(1, 1), field(2, &shape)].concat();
            field(
                12,
                &[field(1, name.as_bytes()), field(2, &field(1, &tensor_type))].concat(),
            )
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.onnx");
        let tensors = [
            ext("x", 4096, 4, true),
            ext("y", 4104, 4, false),
            ext("z", 8192, 4, true),
        ];
        let branch = [
            field(2, b"branch"),
            field(5, &ext("branch_value", 8204, 1, false)),
            output("branch_value", 1),
        ]
        .concat();
        let attribute = |name: &[u8]| {
            field(
                5,
                &[field(1, name), integer(20, 5), field(6, &branch)].concat(),
            )
        };
        let conditional = field(
            1,
            &[
                field(1, b"condition"),
                field(2, b"nested"),
                field(4, b"If"),
                attribute(b"then_branch"),
                attribute(b"else_branch"),
            ]
            .concat(),
        );
        let condition_type = field(1, &[integer(1, 9), field(2, &[])].concat());
        let condition_input = field(
            11,
            &[field(1, b"condition"), field(2, &condition_type)].concat(),
        );
        let graph = [
            field(2, b"range_progress"),
            condition_input,
            conditional,
            output("nested", 1),
            tensors.iter().flat_map(|t| field(5, t)).collect(),
            output("x", 4),
            output("y", 4),
            output("z", 4),
        ]
        .concat();
        let model = [integer(1, 9), field(7, &graph), field(8, &integer(2, 18))].concat();
        std::fs::write(&path, &model).unwrap();
        let mut data = std::fs::File::create(dir.path().join("weights.bin")).unwrap();
        data.set_len(1 << 24).unwrap();
        data.seek(SeekFrom::Start(4096)).unwrap();
        for value in [1f32, 2., 3., 4., 5., 6.] {
            data.write_all(&value.to_le_bytes()).unwrap();
        }
        data.seek(SeekFrom::Start(8192)).unwrap();
        for value in [7f32, 8., 9., 10.] {
            data.write_all(&value.to_le_bytes()).unwrap();
        }
        let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = observed.clone();
        let _observer = gen_core::LoadActivityObserver::new(move |v| sink.lock().unwrap().push(v));
        let mut session = ort_session(ort::session::Session::builder().unwrap(), &path).unwrap();
        assert_eq!(session.outputs().len(), 4);
        let condition = ort::value::Tensor::from_array((Vec::<usize>::new(), vec![true])).unwrap();
        let output = session.run(ort::inputs!["condition" => condition]).unwrap();
        for (name, expected) in [
            ("x", vec![1., 2., 3., 4.]),
            ("y", vec![3., 4., 5., 6.]),
            ("z", vec![7., 8., 9., 10.]),
            ("nested", vec![10.]),
        ] {
            let (_, values) = output[name].try_extract_tensor::<f32>().unwrap();
            assert_eq!(values, expected);
        }
        assert!(observed
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .any(|v| v.file.ends_with("weights.bin")
                && v.bytes_read == 40
                && v.total_bytes == Some(40)));
        assert_eq!(std::fs::read(&path).unwrap(), model);
        // Raw refusal cases cannot invoke ORT and belong in normal CI too.
        invalid_external_ranges_fail_before_reading_or_allocating_source_file_size();
        if let Some(output) = std::env::var_os("SCENEWORKS_LOAD_PROGRESS_ORT_EVIDENCE") {
            std::fs::write(output, r#"{"kind":"sc27980-ort-session-acceptance","passed":true,"tensorValuesVerified":true,"outputs":4,"externalFileBytes":16777216,"selectedExternalBytes":40,"nestedExternalData":true,"omittedLength":true,"aliasesAndOverlaps":true,"decimalOffsetPadding":true,"sourceModelUnchanged":true,"missingTruncatedOverflowRefusals":true}"#).unwrap();
        }
    }
}
