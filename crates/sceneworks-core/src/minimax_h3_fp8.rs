//! Byte-preserving normalization of the shared FP8 codec plan into H3's canonical namespace.
use super::*;
use crate::base_weights::QuantFormat;
use gen_core::checkpoint_codec::{
    classify_checkpoint_projection_sources, LogicalTensorPlan, ScalarScaleSource, TensorCodecSpec,
    BASELINE_CHECKPOINT_CODECS,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub(super) enum PayloadEdit {
    Copy,
    Pad(u64),
    Bytes(Vec<u8>),
    MxScale {
        source_len: usize,
        stored: [usize; 2],
        row_ranges: Vec<std::ops::Range<usize>>,
        output_len: usize,
    },
}

impl PayloadEdit {
    pub(super) fn output_bytes(&self, ranges: &[(u64, u64)]) -> u64 {
        match self {
            Self::Bytes(b) => b.len() as u64,
            Self::MxScale { output_len, .. } => *output_len as u64,
            Self::Copy => ranges.iter().map(|(_, n)| n).sum(),
            Self::Pad(n) => ranges.iter().map(|(_, n)| n).sum::<u64>() + n,
        }
    }
    pub(super) fn write_replacement(
        &self,
        input: &mut File,
        out: &mut impl Write,
        offset: u64,
        check: &dyn Fn() -> Result<(), String>,
    ) -> Result<bool, String> {
        check()?;
        let bytes = match self {
            Self::Copy | Self::Pad(_) => return Ok(false),
            Self::Bytes(bytes) => {
                out.write_all(bytes).map_err(|e| e.to_string())?;
                return Ok(true);
            }
            Self::MxScale {
                source_len,
                stored,
                row_ranges,
                ..
            } => {
                let mut plane = vec![0; *source_len];
                input
                    .seek(SeekFrom::Start(offset))
                    .map_err(|e| e.to_string())?;
                for chunk in plane.chunks_mut(1024 * 1024) {
                    check()?;
                    input.read_exact(chunk).map_err(|e| e.to_string())?;
                }
                let rows: Vec<_> = row_ranges.iter().cloned().flatten().collect();
                gen_core::remap_mxfp8_scales(&plane, *stored, &rows)?
            }
        };
        for chunk in bytes.chunks(1024 * 1024) {
            check()?;
            out.write_all(chunk).map_err(|e| e.to_string())?;
        }
        Ok(true)
    }
}

#[derive(Default)]
pub(super) struct Schedule {
    pub(super) format: Option<QuantFormat>,
    pub(super) consumed: BTreeSet<String>,
    pub(super) tensors: BTreeMap<String, Tensor>,
}

pub(super) struct Geometry<'a> {
    pub(super) native: bool,
    pub(super) config: &'a H3Config,
    pub(super) layout: H3QkvLayout,
    pub(super) expected: &'a BTreeMap<String, Vec<usize>>,
}
struct Inputs<'a, 'b> {
    files: Vec<std::cell::RefCell<std::io::BufReader<File>>>,
    declarations: BTreeMap<String, Value>,
    physical: &'a H3Physical<'b>,
    body: Vec<u64>,
}
impl Inputs<'_, '_> {
    fn descriptor(&self, base: &str, key: &str) -> Result<Option<Value>, String> {
        let mut declaration = self.declarations.get(base).cloned();
        if self.physical.contains_key(key) {
            let source = self.tensor(key)?;
            let len =
                usize::try_from(source.ranges[0].1).map_err(|_| "oversized H3 FP8 descriptor")?;
            if len > crate::checkpoint_inspector::MAX_JSON_DESCRIPTOR_BYTES as usize {
                return Err(format!("H3 {key}: descriptor exceeds size limit"));
            }
            let mut file = self.files[source.source].borrow_mut();
            file.seek(SeekFrom::Start(source.offset))
                .map_err(|e| e.to_string())?;
            let mut bytes = vec![0; len];
            file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            let local: Map<String, Value> =
                serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            declaration
                .get_or_insert_with(|| json!({}))
                .as_object_mut()
                .ok_or("invalid H3 projection declaration")?
                .extend(local);
        }
        Ok(declaration)
    }
    fn tensor(&self, key: &str) -> Result<Tensor, String> {
        let &(source, h) = self
            .physical
            .get(key)
            .ok_or_else(|| format!("missing H3 operand {key}"))?;
        let shape =
            serde_json::from_value(h["shape"].clone()).map_err(|e| format!("H3 {key}: {e}"))?;
        let [start, end]: [u64; 2] = serde_json::from_value(h["data_offsets"].clone())
            .map_err(|e| format!("H3 {key}: {e}"))?;
        Ok(Tensor {
            source,
            key: key.into(),
            dtype: h["dtype"]
                .as_str()
                .ok_or("invalid H3 operand dtype")?
                .into(),
            shape,
            offset: self.body[source] + start,
            ranges: vec![(0, end - start)],
            edit: PayloadEdit::Copy,
        })
    }
}

fn single_shape(
    key: &str,
    native: bool,
    config: &H3Config,
    expected: &BTreeMap<String, Vec<usize>>,
) -> Result<Vec<usize>, String> {
    if native && key.ends_with(".attn.qkv_proj.weight") {
        let heads = config.dim("num_attention_heads")?;
        return Ok(vec![
            3 * heads * config.dim("attention_head_dim")?,
            config.dim("hidden_size")?,
        ]);
    }
    if native && key.ends_with(".mlp.fc1.weight") {
        return Ok(vec![2 * config.dim("ffn_dim")?, config.dim("hidden_size")?]);
    }
    let name = if native {
        native_outputs(key, config, H3QkvLayout::Contiguous, &[], 1)?
            .remove(0)
            .0
    } else {
        key.into()
    };
    expected
        .get(&name)
        .cloned()
        .ok_or_else(|| format!("unconsumed H3 FP8 tensor {key:?} maps to unknown {name:?}"))
}

impl Schedule {
    fn insert(&mut self, key: String, tensor: Tensor) -> Result<(), String> {
        if self.tensors.insert(key.clone(), tensor).is_some() {
            return Err(format!("conflicting H3 canonical operand {key:?}"));
        }
        Ok(())
    }
    fn companion(
        &mut self,
        physical: &str,
        logical: &str,
        inputs: &Inputs<'_, '_>,
        edit: impl FnOnce(&mut Tensor) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut tensor = inputs.tensor(physical)?;
        self.consumed.insert(physical.into());
        edit(&mut tensor)?;
        self.insert(logical.into(), tensor)
    }
    fn projection(
        &mut self,
        plan: &LogicalTensorPlan,
        inputs: &Inputs<'_, '_>,
        geometry: &Geometry<'_>,
    ) -> Result<(), String> {
        let Geometry {
            native,
            config,
            layout,
            expected,
        } = *geometry;
        let key = &plan.physical_key;
        let name = strip_prefix(key);
        let logical_shape = single_shape(name, native, config, expected)?;
        let raw = inputs.tensor(key)?;
        match &plan.codec {
            TensorCodecSpec::Mxfp8 {
                stored_shape,
                logical_shape_declared,
                ..
            } => {
                let logical: [usize; 2] = logical_shape
                    .as_slice()
                    .try_into()
                    .map_err(|_| format!("H3 {key}: MXFP8 requires matrix geometry"))?;
                if gen_core::mxfp8_padded_shape(logical) != *stored_shape
                    || (*logical_shape_declared && plan.shape != logical_shape)
                {
                    return Err(format!("H3 {key}: MXFP8 stored {stored_shape:?}/declared {:?}, expected {logical_shape:?}",plan.shape));
                }
            }
            _ if plan.shape != logical_shape => {
                return Err(format!(
                    "H3 {key}: decoded shape {:?}, expected {logical_shape:?}",
                    plan.shape
                ))
            }
            _ => {}
        }
        let outputs = if native {
            native_outputs(name, config, layout, &logical_shape, 1)?
        } else {
            vec![(
                name.into(),
                logical_shape.clone(),
                vec![(0, logical_shape.iter().product::<usize>() as u64)],
            )]
        };
        self.consumed.insert(key.clone());
        let source_base = key.strip_suffix(".weight").unwrap_or(key);
        let source_desc = format!("{source_base}.comfy_quant");
        if inputs.physical.contains_key(source_desc.as_str()) {
            self.consumed.insert(source_desc.clone());
        }
        let has_source_blob = inputs.physical.contains_key(source_desc.as_str());
        let original = if matches!(plan.codec, TensorCodecSpec::Mxfp8 { .. })
            || inputs.declarations.contains_key(source_base)
        {
            inputs.descriptor(source_base, &source_desc)?
        } else {
            None
        };
        for (logical, shape, ranges) in outputs {
            if expected.get(&logical) != Some(&shape) {
                return Err(format!(
                    "H3 {key}: incompatible logical {logical:?} {shape:?}"
                ));
            }
            let base = logical.strip_suffix(".weight").unwrap_or(&logical);
            let mut codes = raw.clone();
            codes.shape = shape.clone();
            codes.ranges = ranges.clone();
            let descriptor = match &plan.codec {
                TensorCodecSpec::ScalarFp8 {
                    scale,
                    input_scale,
                    full_precision_matrix_mult,
                } => {
                    let format = if raw.dtype == "F8_E5M2" {
                        "float8_e5m2"
                    } else {
                        "float8_e4m3fn"
                    };
                    if let ScalarScaleSource::Companion { physical_key } = scale {
                        let suffix = physical_key.rsplit('.').next().ok_or("invalid scale key")?;
                        self.companion(physical_key, &format!("{base}.{suffix}"), inputs, |_| {
                            Ok(())
                        })?;
                    }
                    if let Some(physical_key) = input_scale {
                        let suffix = physical_key.rsplit('.').next().ok_or("invalid scale key")?;
                        self.companion(physical_key, &format!("{base}.{suffix}"), inputs, |_| {
                            Ok(())
                        })?;
                    }
                    if original.is_none() && !has_source_blob && !*full_precision_matrix_mult {
                        None
                    } else {
                        Some(
                            json!({"format":format,"full_precision_matrix_mult":full_precision_matrix_mult}),
                        )
                    }
                }
                TensorCodecSpec::RowwiseFp8 { scale } => {
                    let width = logical_shape[1] as u64;
                    self.companion(scale, &format!("{base}.weight_scale"), inputs, |tensor| {
                        tensor.shape[0] = shape[0];
                        tensor.ranges = ranges
                            .iter()
                            .map(|(start, n)| (start / width * 4, n / width * 4))
                            .collect();
                        Ok(())
                    })?;
                    Some(json!({"format":"fp8r"}))
                }
                TensorCodecSpec::Mxfp8 {
                    scale,
                    stored_shape,
                    full_precision_matrix_mult,
                    ..
                } => {
                    let input_scale = format!("{source_base}.input_scale");
                    if inputs.physical.contains_key(input_scale.as_str()) {
                        self.companion(
                            &input_scale,
                            &format!("{base}.input_scale"),
                            inputs,
                            |_| Ok(()),
                        )?;
                    }
                    let width = logical_shape[1] as u64;
                    let padded = gen_core::mxfp8_padded_shape([shape[0], shape[1]]);
                    let stride = stored_shape[1] as u64;
                    let identity = ranges.as_slice() == [(0, width * logical_shape[0] as u64)];
                    if !identity {
                        codes.ranges = ranges
                            .iter()
                            .map(|(s, n)| (s / width * stride, n / width * stride))
                            .collect();
                        codes.edit = PayloadEdit::Pad((padded[0] - shape[0]) as u64 * stride);
                    } else {
                        codes.ranges = raw.ranges.clone();
                    }
                    codes.shape = padded.to_vec();
                    self.companion(scale, &format!("{base}.weight_scale"), inputs, |tensor| {
                        if !identity {
                            let source_len = usize::try_from(tensor.ranges[0].1)
                                .map_err(|_| "H3 MXFP8 scale plane too large")?;
                            tensor.shape = gen_core::mxfp8_scale_shape(padded).to_vec();
                            tensor.edit = PayloadEdit::MxScale {
                                source_len,
                                stored: *stored_shape,
                                row_ranges: ranges
                                    .iter()
                                    .map(|(s, n)| (s / width) as usize..((s + n) / width) as usize)
                                    .collect(),
                                output_len: tensor.shape.iter().product(),
                            };
                        }
                        Ok(())
                    })?;
                    Some(
                        json!({"format":"mxfp8","full_precision_matrix_mult":full_precision_matrix_mult,"orig_shape":shape}),
                    )
                }
                _ => return Err(format!("unsupported H3 FP8 codec {}", plan.codec_id)),
            };
            if let Some(mut value) = descriptor {
                let mut desc = codes.clone();
                if inputs.physical.contains_key(source_desc.as_str())
                    && !inputs.declarations.contains_key(source_base)
                    && (!matches!(plan.codec, TensorCodecSpec::Mxfp8 { .. })
                        || original.as_ref().and_then(|v| v.get("orig_shape"))
                            == Some(&json!(shape)))
                {
                    desc = inputs.tensor(&source_desc)?;
                } else {
                    if let Some(Value::Object(mut fields)) = original.clone() {
                        fields.extend(value.as_object_mut().ok_or("invalid descriptor")?.clone());
                        value = Value::Object(fields);
                    }
                    let bytes = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
                    desc.dtype = "U8".into();
                    desc.shape = vec![bytes.len()];
                    desc.edit = PayloadEdit::Bytes(bytes);
                }
                self.insert(format!("{base}.comfy_quant"), desc)?;
            }
            self.insert(logical, codes)?;
        }
        Ok(())
    }
}

pub(super) fn schedule(
    files: &[(&String, &PathBuf)],
    headers: &[Value],
    physical: &H3Physical<'_>,
    geometry: &Geometry<'_>,
    check: &dyn Fn() -> std::io::Result<()>,
) -> Result<Schedule, String> {
    let mut schedule = Schedule::default();
    // W4A8's auxiliary FP8 scale plane has its own existing descriptor and is never an FP8 parameter.
    if !physical
        .iter()
        .any(|(key, (_, entry))| is_fp8_parameter(key, entry["dtype"].as_str()))
    {
        return Ok(schedule);
    }
    let paths: Vec<_> = files.iter().map(|(_, p)| (*p).clone()).collect();
    let mut declarations = BTreeMap::new();
    for header in headers {
        if let Some(table) = header["__metadata__"]["_quantization_metadata"].as_str() {
            let parsed = gen_core::comfy_quant::parse_quantization_metadata(table)
                .map_err(|e| e.to_string())?;
            let prefix = gen_core::checkpoint_codec::resolve_metadata_prefix(physical, &parsed)
                .map_err(|e| e.to_string())?;
            let raw: Value = serde_json::from_str(table).map_err(|e| e.to_string())?;
            for (layer, descriptor) in raw["layers"]
                .as_object()
                .ok_or("invalid FP8 metadata layers")?
            {
                declarations.insert(format!("{prefix}{layer}"), descriptor.clone());
            }
        }
        if header["__metadata__"].as_object().is_some_and(|m| {
            m.contains_key("quant_format") || m.contains_key("quantization_config")
        }) {
            return Err("H3 FP8 source contains competing file quantization declarations".into());
        }
    }
    let registry = CheckpointCodecRegistry::new(BASELINE_CHECKPOINT_CODECS.iter().copied())
        .map_err(|e| e.to_string())?;
    let plan = classify_checkpoint_projection_sources(&paths, &registry)
        .map_err(|e| e.to_string())?
        .ok_or("missing H3 FP8 projection plan")?;
    let format = crate::base_weights::projection_plan_encoding(&plan);
    if !matches!(
        format,
        Some(
            QuantFormat::Fp8E4M3Scalar
                | QuantFormat::Fp8E5M2Scalar
                | QuantFormat::Fp8E4M3Rowwise
                | QuantFormat::Mxfp8
        )
    ) {
        return Err(format!(
            "unsupported or mixed H3 FP8 projection conventions: {format:?}"
        ));
    }
    schedule.format = format;
    let mut body = Vec::new();
    let mut files = Vec::new();
    for path in &paths {
        let mut file = std::io::BufReader::new(File::open(path).map_err(|e| e.to_string())?);
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        body.push(8 + u64::from_le_bytes(bytes));
        files.push(std::cell::RefCell::new(file));
    }
    let inputs = Inputs {
        files,
        physical,
        body,
        declarations,
    };
    for row in &plan.tensors {
        check().map_err(|e| e.to_string())?;
        schedule.projection(row, &inputs, geometry)?;
    }
    if physical.contains_key("scaled_fp8") {
        schedule.companion("scaled_fp8", "scaled_fp8", &inputs, |_| Ok(()))?;
    }
    Ok(schedule)
}
