//! Portable, source-backed video lane eligibility. The native router and shared request
//! validator ask this same static decision; installation, device and claim availability stay
//! with the worker. `VideoRouteFacts` comes from the immutable compiled ModelSpec projection.

use serde_json::{Map, Value};

pub type JsonObject = Map<String, Value>;

/// The five compiled ModelSpec capabilities used by the video scheduler. Missing facts never
/// turn an unknown model into a supported lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VideoRouteFacts {
    pub mlx: bool,
    pub candle: bool,
    pub candle_i2v: bool,
    pub candle_vace: bool,
    pub candle_lora: bool,
}

impl VideoRouteFacts {
    pub const fn new(
        mlx: bool,
        candle: bool,
        candle_i2v: bool,
        candle_vace: bool,
        candle_lora: bool,
    ) -> Self {
        Self {
            mlx,
            candle,
            candle_i2v,
            candle_vace,
            candle_lora,
        }
    }

    /// The server projects these immutable facts from the compiled ModelSpec, not the caller's
    /// request. Imported entries have no builtin route flags and use their declared import plan.
    pub fn from_entry(entry: &JsonObject) -> Self {
        Self::from_value(entry.get("optionRouteFacts"))
    }

    pub fn from_value(value: Option<&Value>) -> Self {
        let facts = value.and_then(Value::as_object);
        let bit = |key| {
            facts
                .and_then(|facts| facts.get(key))
                .and_then(Value::as_bool)
                == Some(true)
        };
        Self::new(
            bit("videoMlxRouted"),
            bit("candleVideoRouted"),
            bit("candleVideoI2v"),
            bit("candleVideoVace"),
            bit("candleVideoLora"),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoJobKind {
    Generate,
    Extend,
    Bridge,
    Replace,
}

impl VideoJobKind {
    pub fn for_mode(mode: &str) -> Self {
        match mode {
            "extend_clip" => Self::Extend,
            "video_bridge" => Self::Bridge,
            "replace_person" => Self::Replace,
            _ => Self::Generate,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Generate => "video_generate",
            Self::Extend => "video_extend",
            Self::Bridge => "video_bridge",
            Self::Replace => "person_replace",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "video_generate" | "videoGenerate" | "video" => Some(Self::Generate),
            "video_extend" | "videoExtend" => Some(Self::Extend),
            "video_bridge" | "videoBridge" => Some(Self::Bridge),
            "person_replace" | "personReplace" => Some(Self::Replace),
            _ => None,
        }
    }

    fn mode(self, payload: &JsonObject) -> Option<&str> {
        match self {
            Self::Extend => Some("extend_clip"),
            Self::Bridge => Some("video_bridge"),
            Self::Replace => Some("replace_person"),
            Self::Generate => match payload.get("mode") {
                None | Some(Value::Null) => Some("image_to_video"),
                Some(Value::String(value)) => Some(value),
                Some(_) => None,
            },
        }
    }
}

fn nonempty(payload: &JsonObject, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}
fn nonempty_array(payload: &JsonObject, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty())
}
fn nonempty_string_array(payload: &JsonObject, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_array)
        .is_some_and(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .any(|value| !value.trim().is_empty())
        })
}
fn present_or_malformed_array(payload: &JsonObject, key: &str) -> bool {
    match payload.get(key) {
        None | Some(Value::Null) => false,
        Some(Value::Array(values)) => !values.is_empty(),
        Some(_) => true,
    }
}
fn well_formed_carriers(payload: &JsonObject) -> bool {
    for key in [
        "model",
        "mode",
        "sourceAssetId",
        "lastFrameAssetId",
        "sourceClipAssetId",
        "bridgeRightClipAssetId",
        "referenceAssetId",
        "maskAssetId",
        "personTrackId",
        "characterId",
    ] {
        if !matches!(
            payload.get(key),
            None | Some(Value::Null) | Some(Value::String(_))
        ) {
            return false;
        }
    }
    for key in [
        "referenceAssetIds",
        "sourceClipAssetIds",
        "referenceAudioAssetIds",
        "loras",
    ] {
        if !matches!(
            payload.get(key),
            None | Some(Value::Null) | Some(Value::Array(_))
        ) {
            return false;
        }
    }
    let Some(advanced) = payload.get("advanced").filter(|value| !value.is_null()) else {
        return true;
    };
    let Some(advanced) = advanced.as_object() else {
        return false;
    };
    match advanced.get("mlxQuantize") {
        None | Some(Value::Null) => {}
        Some(value) if value.as_i64().is_some() => {}
        Some(Value::String(value)) if value.trim().parse::<i64>().is_ok() => {}
        _ => return false,
    }
    if !matches!(
        advanced.get("quantization"),
        None | Some(Value::Null) | Some(Value::String(_))
    ) {
        return false;
    }
    true
}
fn checkpoint_id(payload: &JsonObject) -> Option<&str> {
    payload
        .get("modelManifestEntry")?
        .get("importPlan")?
        .get("checkpointId")?
        .as_str()
        .map(str::trim)
        .filter(|id| !id.is_empty())
}
fn imported_entry(payload: &JsonObject) -> Option<&JsonObject> {
    payload.get("modelManifestEntry")?.as_object()
}

pub fn mlx_mode_eligible(model: &str, mode: &str) -> bool {
    match model {
        "svd" | "wan_2_2_i2v_14b" => mode == "image_to_video",
        "wan_2_2_t2v_14b" | "mochi_1" => mode == "text_to_video",
        "wan_2_2_vace_fun_14b" => mode == "replace_person",
        "bernini" => matches!(
            mode,
            "text_to_video"
                | "video_to_video"
                | "reference_to_video"
                | "reference_video_to_video"
                | "multi_video_to_video"
                | "ads2v"
        ),
        "scail2_14b" => matches!(mode, "animate_character" | "replace_person"),
        "krea_realtime_14b" => {
            matches!(mode, "text_to_video" | "image_to_video" | "video_to_video")
        }
        "minimax_h3" => matches!(
            mode,
            "text_to_video" | "image_to_video" | "first_last_frame"
        ),
        "minimax_h3_ref" => mode == "reference_to_video",
        _ => match mode {
            "text_to_video" | "image_to_video" => true,
            "first_last_frame" | "extend_clip" | "video_bridge" | "replace_person" => {
                matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "ltx_2_5" | "wan_2_2")
            }
            _ => false,
        },
    }
}

pub fn mlx_eligible(kind: VideoJobKind, payload: &JsonObject, facts: VideoRouteFacts) -> bool {
    if !well_formed_carriers(payload) {
        return false;
    }
    let Some(model) = payload
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    else {
        return false;
    };
    if checkpoint_id(payload).is_some() {
        return kind == VideoJobKind::Generate && imported_h3_eligible(model, payload, facts);
    }
    if !facts.mlx {
        return false;
    }
    let Some(mode) = kind.mode(payload) else {
        return false;
    };
    if !mlx_mode_eligible(model, mode) {
        return false;
    }
    if model == "wan_2_2" {
        let source = nonempty(payload, "sourceAssetId");
        let last = nonempty(payload, "lastFrameAssetId");
        if !match mode {
            "text_to_video" => !source && !last,
            "image_to_video" => source && !last,
            "first_last_frame" => source && last,
            _ => true,
        } {
            return false;
        }
    }
    if matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "ltx_2_5")
        && matches!(mode, "extend_clip" | "video_bridge" | "replace_person")
        && !has_ltx_ic_lora(payload)
    {
        return false;
    }
    true
}

fn torch_quant_requested(payload: &JsonObject) -> bool {
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("quantization"))
        .and_then(Value::as_str)
        .is_some_and(|value| {
            let value = value.trim();
            !value.is_empty() && !value.eq_ignore_ascii_case("auto")
        })
}
fn requested_quant_bits(payload: &JsonObject) -> Option<i64> {
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("mlxQuantize"))
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .filter(|bits| *bits > 0)
}
fn ltx_tier(payload: &JsonObject) -> Option<&'static str> {
    let value = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("mlxQuantize"));
    let bits = match value {
        None => return Some("q4"),
        Some(value) => value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())?,
    };
    match bits {
        i64::MIN..=0 => Some("bf16"),
        4 => Some("q4"),
        8 => Some("q8"),
        _ => None,
    }
}
fn ltx_product_tier(model: &str, mode: &str, payload: &JsonObject) -> bool {
    matches!(
        (model, ltx_tier(payload)),
        ("ltx_2_3", Some("q4" | "q8")) | ("ltx_2_5", Some("bf16" | "q4" | "q8"))
    ) && matches!(
        mode,
        "text_to_video"
            | "image_to_video"
            | "first_last_frame"
            | "extend_clip"
            | "video_bridge"
            | "replace_person"
    )
}
fn candle_tier(model: &str, payload: &JsonObject) -> bool {
    if matches!(
        model,
        "wan_2_2" | "wan_2_2_t2v_14b" | "wan_2_2_i2v_14b" | "minimax_h3" | "minimax_h3_ref"
    ) {
        return true;
    }
    payload
        .get("mode")
        .and_then(Value::as_str)
        .is_some_and(|mode| ltx_product_tier(model, mode, payload))
}
fn scail2_tier(payload: &JsonObject) -> Option<&'static str> {
    let value = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("mlxQuantize"));
    match value {
        None | Some(Value::Null) => Some("bf16"),
        Some(value) => match value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
        {
            Some(..=0) => Some("bf16"),
            Some(4) => Some("q4"),
            Some(8) => Some("q8"),
            _ => None,
        },
    }
}
fn scail2_source_tier(payload: &JsonObject) -> bool {
    let adapters = nonempty_array(payload, "loras");
    scail2_tier(payload).is_some_and(|tier| !adapters || tier == "bf16")
}
pub fn scail2_source_tier_eligible(payload: &JsonObject) -> bool {
    scail2_source_tier(payload)
}

pub fn h3_base_shape(payload: &JsonObject) -> bool {
    let source = nonempty(payload, "sourceAssetId");
    let last = nonempty(payload, "lastFrameAssetId");
    let mode = match payload.get("mode").and_then(Value::as_str) {
        Some("text_to_video") => !source && !last,
        Some("image_to_video") => source != last,
        Some("first_last_frame") => source && last,
        _ => false,
    };
    mode && ![
        "sourceClipAssetId",
        "bridgeRightClipAssetId",
        "referenceAssetId",
        "maskAssetId",
    ]
    .iter()
    .any(|key| nonempty(payload, key))
        && ![
            "referenceAssetIds",
            "sourceClipAssetIds",
            "referenceAudioAssetIds",
        ]
        .iter()
        .any(|key| present_or_malformed_array(payload, key))
}

pub fn candle_base_eligible(model: &str, payload: &JsonObject, facts: VideoRouteFacts) -> bool {
    if !well_formed_carriers(payload) || !facts.candle || torch_quant_requested(payload) {
        return false;
    }
    if matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "ltx_2_5") {
        let source = nonempty(payload, "sourceAssetId");
        let last = nonempty(payload, "lastFrameAssetId");
        let clip = nonempty(payload, "sourceClipAssetId");
        let right = nonempty(payload, "bridgeRightClipAssetId");
        match payload.get("mode").and_then(Value::as_str) {
            Some("text_to_video") if !source && !last && !clip && !right => {}
            Some("image_to_video") if source && !last && !clip && !right => {}
            Some("first_last_frame") if source && last && !clip && !right => {}
            Some("extend_clip")
                if clip && !right && !source && !last && has_ltx_ic_lora(payload) => {}
            Some("video_bridge")
                if clip && right && !source && !last && has_ltx_ic_lora(payload) => {}
            _ => return false,
        }
    } else if model == "wan_2_2" {
        let source = nonempty(payload, "sourceAssetId");
        let last = nonempty(payload, "lastFrameAssetId");
        match payload.get("mode").and_then(Value::as_str) {
            Some("text_to_video") if !source && !last => {}
            Some("image_to_video") if source && !last => {}
            Some("first_last_frame") if source && last => {}
            _ => return false,
        }
        if nonempty(payload, "sourceClipAssetId") || nonempty(payload, "bridgeRightClipAssetId") {
            return false;
        }
    } else if model == "minimax_h3" {
        if !h3_base_shape(payload) {
            return false;
        }
    } else if model == "minimax_h3_ref" {
        if payload.get("mode").and_then(Value::as_str) != Some("reference_to_video")
            || [
                "sourceAssetId",
                "lastFrameAssetId",
                "sourceClipAssetId",
                "bridgeRightClipAssetId",
            ]
            .iter()
            .any(|key| nonempty(payload, key))
            || ![
                "referenceAssetIds",
                "sourceClipAssetIds",
                "referenceAudioAssetIds",
            ]
            .iter()
            .any(|key| nonempty_array(payload, key))
        {
            return false;
        }
    } else if facts.candle_i2v {
        if payload.get("mode").and_then(Value::as_str) != Some("image_to_video")
            || !nonempty(payload, "sourceAssetId")
        {
            return false;
        }
    } else if payload.get("mode").and_then(Value::as_str) != Some("text_to_video")
        || nonempty(payload, "sourceAssetId")
    {
        return false;
    }
    if nonempty(payload, "referenceAssetId") || nonempty(payload, "maskAssetId") {
        return false;
    }
    if !facts.candle_lora && nonempty_array(payload, "loras") {
        return false;
    }
    if (requested_quant_bits(payload).is_some() || matches!(model, "ltx_2_3" | "ltx_2_5"))
        && !candle_tier(model, payload)
    {
        return false;
    }
    true
}

fn has_ltx_ic_lora(payload: &JsonObject) -> bool {
    payload
        .get("loras")
        .and_then(Value::as_array)
        .is_some_and(|loras| loras_contain_ltx_ic_lora(loras))
}

/// The same explicit IC-LoRA marker is consumed by portable routing and native execution.
pub fn loras_contain_ltx_ic_lora(loras: &[Value]) -> bool {
    loras.iter().any(lora_looks_like_ltx_ic_lora)
}
pub fn lora_looks_like_ltx_ic_lora(lora: &Value) -> bool {
    let Some(object) = lora.as_object() else {
        return lora
            .as_str()
            .is_some_and(|value| ltx_ic_lora_marker(&value.to_lowercase().replace('_', "-")));
    };
    if object.get("icLora") == Some(&Value::Bool(true))
        || object.get("isIcLora") == Some(&Value::Bool(true))
    {
        return true;
    }
    if object
        .get("conditioningRole")
        .and_then(Value::as_str)
        .is_some_and(|role| role.trim().to_lowercase().replace('-', "_") == "ic_lora")
    {
        return true;
    }
    let source = object.get("source").and_then(Value::as_object);
    let mut values = Vec::new();
    for key in [
        "id",
        "loraId",
        "name",
        "displayName",
        "installedPath",
        "sourcePath",
        "path",
    ] {
        if let Some(value) = object.get(key).and_then(Value::as_str) {
            values.push(value);
        }
    }
    if let Some(source) = source {
        for key in ["repo", "file", "path"] {
            if let Some(value) = source.get(key).and_then(Value::as_str) {
                values.push(value);
            }
        }
    }
    match source
        .and_then(|value| value.get("files"))
        .or_else(|| object.get("files"))
    {
        Some(Value::Array(files)) => values.extend(files.iter().filter_map(Value::as_str)),
        Some(Value::String(file)) => values.push(file),
        _ => {}
    }
    ltx_ic_lora_marker(&values.join(" ").to_lowercase().replace('_', "-"))
}
fn ltx_ic_lora_marker(value: &str) -> bool {
    value.contains("ic-lora") || value.contains("ltx-2-3-ic-")
}
pub fn candle_vace_eligible(
    model: &str,
    payload: &JsonObject,
    kind: VideoJobKind,
    facts: VideoRouteFacts,
) -> bool {
    if model == "wan_2_2_vace_fun_14b" {
        if kind != VideoJobKind::Replace {
            return false;
        }
    } else if !facts.candle_vace {
        return false;
    }
    match kind {
        VideoJobKind::Replace
            if nonempty(payload, "sourceClipAssetId")
                && nonempty(payload, "personTrackId")
                && nonempty(payload, "characterId") => {}
        VideoJobKind::Extend if nonempty(payload, "sourceClipAssetId") => {}
        VideoJobKind::Bridge
            if nonempty(payload, "sourceClipAssetId")
                && nonempty(payload, "bridgeRightClipAssetId") => {}
        _ => return false,
    }
    (model == "wan_2_2_vace_fun_14b" || !nonempty_array(payload, "loras"))
        && requested_quant_bits(payload).is_none()
}
pub fn ltx_replace_eligible(model: &str, payload: &JsonObject) -> bool {
    matches!(model, "ltx_2_3" | "ltx_2_5")
        && nonempty(payload, "sourceClipAssetId")
        && nonempty(payload, "personTrackId")
        && nonempty(payload, "characterId")
        && has_ltx_ic_lora(payload)
        && ltx_product_tier(model, "replace_person", payload)
}
pub fn scail2_animate_eligible(model: &str, payload: &JsonObject) -> bool {
    model == "scail2_14b"
        && payload.get("mode").and_then(Value::as_str) == Some("animate_character")
        && (nonempty(payload, "referenceAssetId")
            || nonempty(payload, "sourceAssetId")
            || nonempty_string_array(payload, "referenceAssetIds"))
        && nonempty(payload, "sourceClipAssetId")
        && scail2_source_tier(payload)
}
pub fn scail2_replace_eligible(model: &str, payload: &JsonObject) -> bool {
    model == "scail2_14b"
        && nonempty(payload, "sourceClipAssetId")
        && nonempty(payload, "personTrackId")
        && nonempty(payload, "characterId")
        && scail2_source_tier(payload)
}
pub fn bernini_eligible(model: &str, payload: &JsonObject) -> bool {
    model == "bernini"
        && matches!(
            payload.get("mode").and_then(Value::as_str),
            Some(
                "text_to_video"
                    | "video_to_video"
                    | "reference_to_video"
                    | "reference_video_to_video"
                    | "multi_video_to_video"
                    | "ads2v"
            )
        )
}

pub fn plan_backed_wan_eligible(payload: &JsonObject) -> bool {
    let Some(entry) = imported_entry(payload) else {
        return false;
    };
    if entry.get("family").and_then(Value::as_str) != Some("wan-video")
        || checkpoint_id(payload).is_none()
        || payload.get("mode").and_then(Value::as_str) != Some("text_to_video")
    {
        return false;
    }
    ![
        "sourceAssetId",
        "lastFrameAssetId",
        "sourceClipAssetId",
        "bridgeRightClipAssetId",
    ]
    .iter()
    .any(|key| nonempty(payload, key))
}

/// Imported H3 task/partition compatibility is sourced from the already-portable declaration
/// parser. The native installer still verifies plan contents and source digests before loading.
pub fn imported_h3_eligible(
    model: &str,
    payload: &JsonObject,
    model_facts: VideoRouteFacts,
) -> bool {
    let Some(entry) = imported_entry(payload) else {
        return false;
    };
    if !crate::h3_import_facts::entry_is_imported_minimax_h3(entry)
        || matches!(model, "minimax_h3" | "minimax_h3_ref")
        || model_facts.mlx
        || model_facts.candle
    {
        return false;
    }
    let references = payload.get("mode").and_then(Value::as_str) == Some("reference_to_video");
    let Ok(partition) = crate::h3_import_facts::MiniMaxH3Partition::from_entry(entry) else {
        return false;
    };
    let route = if references && partition.supports_references() {
        "minimax_h3_ref"
    } else if !references && partition.supports_base() {
        "minimax_h3"
    } else {
        return false;
    };
    // These two shipped partition routes are pinned by the generated ModelSpec route-policy
    // source and a native parity test. They are task semantics, never imported weight identity.
    candle_base_eligible(
        route,
        payload,
        VideoRouteFacts::new(true, true, false, false, true),
    )
}

pub fn candle_eligible(kind: VideoJobKind, payload: &JsonObject, facts: VideoRouteFacts) -> bool {
    if !well_formed_carriers(payload) {
        return false;
    }
    let Some(model) = payload
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    else {
        return false;
    };
    if torch_quant_requested(payload) {
        return false;
    }
    if checkpoint_id(payload).is_some() {
        return kind == VideoJobKind::Generate
            && (plan_backed_wan_eligible(payload) || imported_h3_eligible(model, payload, facts));
    }
    match kind {
        VideoJobKind::Generate => {
            candle_base_eligible(model, payload, facts)
                || scail2_animate_eligible(model, payload)
                || bernini_eligible(model, payload)
        }
        VideoJobKind::Replace => {
            candle_vace_eligible(model, payload, kind, facts)
                || scail2_replace_eligible(model, payload)
                || ltx_replace_eligible(model, payload)
        }
        VideoJobKind::Extend | VideoJobKind::Bridge => {
            candle_base_eligible(model, payload, facts)
                || candle_vace_eligible(model, payload, kind, facts)
        }
    }
}

/// Ask the same static lane decision from a canonical payload and authoritative model entry.
/// No entry or no route facts means no builtin lane; imported plan declarations are checked
/// separately above. Physical worker availability never changes this answer.
pub fn any_eligible(job_type: &str, payload: &JsonObject, entry: &JsonObject) -> bool {
    let Some(kind) = VideoJobKind::parse(job_type) else {
        return false;
    };
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    if entry.get("id").and_then(Value::as_str) != Some(model) {
        return false;
    }
    let facts = VideoRouteFacts::from_entry(entry);
    let mut canonical = payload.clone();
    canonical.insert(
        "modelManifestEntry".to_owned(),
        Value::Object(entry.clone()),
    );
    mlx_eligible(kind, &canonical, facts) || candle_eligible(kind, &canonical, facts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn obj(value: Value) -> JsonObject {
        value.as_object().unwrap().clone()
    }
    #[test]
    fn route_facts_fail_closed_without_authoritative_projection() {
        let payload = obj(json!({"model":"wan_2_2","mode":"text_to_video"}));
        assert!(!any_eligible(
            "video_generate",
            &payload,
            &obj(json!({"id":"wan_2_2"}))
        ));
        let entry = obj(
            json!({"id":"wan_2_2","optionRouteFacts":{"videoMlxRouted":true,"candleVideoRouted":true,"candleVideoI2v":false,"candleVideoVace":true,"candleVideoLora":true}}),
        );
        assert!(any_eligible("video_generate", &payload, &entry));
        assert!(!any_eligible(
            "video_generate",
            &payload,
            &obj(json!({"id":"other","optionRouteFacts":entry["optionRouteFacts"]}))
        ));
        let forged = obj(
            json!({"model":"unknown","mode":"text_to_video","modelManifestEntry":{"family":"wan-video","importPlan":{"checkpointId":"forged"}}}),
        );
        assert!(!any_eligible(
            "video_generate",
            &forged,
            &obj(json!({"id":"unknown"}))
        ));
    }
    #[test]
    fn wan_modes_and_malformed_carriers_do_not_get_a_lane() {
        let facts = VideoRouteFacts::new(true, true, false, true, true);
        for (mode, source, last, expected) in [
            ("text_to_video", None, None, true),
            ("text_to_video", Some("asset"), None, false),
            ("image_to_video", Some("asset"), None, true),
            ("image_to_video", None, None, false),
            ("first_last_frame", Some("asset"), Some("last"), true),
            ("first_last_frame", Some("asset"), None, false),
        ] {
            let mut payload = obj(json!({"model":"wan_2_2","mode":mode}));
            if let Some(source) = source {
                payload.insert("sourceAssetId".into(), json!(source));
            }
            if let Some(last) = last {
                payload.insert("lastFrameAssetId".into(), json!(last));
            }
            assert_eq!(
                mlx_eligible(VideoJobKind::Generate, &payload, facts),
                expected,
                "{mode}"
            );
            assert_eq!(
                candle_eligible(VideoJobKind::Generate, &payload, facts),
                expected,
                "{mode}"
            );
        }
        for malformed in [json!(5), json!(true), json!({"id":"asset"})] {
            let payload =
                obj(json!({"model":"wan_2_2","mode":"text_to_video","sourceAssetId":malformed}));
            assert!(!mlx_eligible(VideoJobKind::Generate, &payload, facts));
            assert!(!candle_eligible(VideoJobKind::Generate, &payload, facts));
        }
    }
    #[test]
    fn specialized_modes_require_their_source_and_tier_shapes() {
        let ltx = VideoRouteFacts::new(true, true, false, false, true);
        let mut request = obj(
            json!({"model":"ltx_2_3","mode":"extend_clip","sourceClipAssetId":"clip","loras":[{"id":"ordinary_adapter"}],"advanced":{"mlxQuantize":4}}),
        );
        // The installed IC-LoRA id is source-defined; a fabricated id cannot satisfy the route.
        assert!(!mlx_eligible(VideoJobKind::Extend, &request, ltx));
        request.insert("loras".into(), json!([]));
        assert!(!candle_eligible(VideoJobKind::Extend, &request, ltx));
        let i2v = VideoRouteFacts::new(true, true, true, false, true);
        assert!(!mlx_mode_eligible("wan_2_2_i2v_14b", "first_last_frame"));
        assert!(!candle_eligible(
            VideoJobKind::Generate,
            &obj(
                json!({"model":"wan_2_2_i2v_14b","mode":"first_last_frame","sourceAssetId":"asset","lastFrameAssetId":"last"})
            ),
            i2v
        ));
    }

    #[test]
    fn specialized_mode_sets_and_candle_tiers_stay_distinct() {
        for (model, allowed, refused) in [
            ("svd", "image_to_video", "text_to_video"),
            ("wan_2_2_t2v_14b", "text_to_video", "image_to_video"),
            ("mochi_1", "text_to_video", "first_last_frame"),
            ("minimax_h3_ref", "reference_to_video", "text_to_video"),
            ("krea_realtime_14b", "video_to_video", "first_last_frame"),
            ("bernini", "ads2v", "image_to_video"),
        ] {
            assert!(mlx_mode_eligible(model, allowed), "{model} {allowed}");
            assert!(!mlx_mode_eligible(model, refused), "{model} {refused}");
        }
        let ltx = VideoRouteFacts::new(true, true, false, false, true);
        for tier in [json!(4), json!(8), json!("4")] {
            assert!(candle_eligible(
                VideoJobKind::Generate,
                &obj(
                    json!({"model":"ltx_2_3","mode":"text_to_video","advanced":{"mlxQuantize":tier}})
                ),
                ltx
            ));
        }
        for tier in [json!(0), json!(6), json!("bad")] {
            assert!(!candle_eligible(
                VideoJobKind::Generate,
                &obj(
                    json!({"model":"ltx_2_3","mode":"text_to_video","advanced":{"mlxQuantize":tier}})
                ),
                ltx
            ));
        }
    }

    #[test]
    fn candle_vace_and_scail2_require_source_and_adapter_policy() {
        let wan = VideoRouteFacts::new(true, true, false, true, true);
        assert!(candle_eligible(
            VideoJobKind::Replace,
            &obj(
                json!({"model":"wan_2_2","mode":"replace_person","sourceClipAssetId":"clip","personTrackId":"track","characterId":"character"})
            ),
            wan
        ));
        assert!(!candle_eligible(
            VideoJobKind::Replace,
            &obj(
                json!({"model":"wan_2_2","mode":"replace_person","sourceClipAssetId":"clip","characterId":"character"})
            ),
            wan
        ));
        let scail = obj(
            json!({"model":"scail2_14b","mode":"animate_character","referenceAssetId":"image","sourceClipAssetId":"clip","advanced":{"mlxQuantize":4}}),
        );
        assert!(candle_eligible(VideoJobKind::Generate, &scail, wan));
        let mut packed_with_adapter = scail;
        packed_with_adapter.insert("loras".into(), json!([{"id":"adapter"}]));
        assert!(!candle_eligible(
            VideoJobKind::Generate,
            &packed_with_adapter,
            wan
        ));
    }

    #[test]
    fn imported_wan_and_h3_use_declared_plan_not_builtin_id() {
        let wan_entry =
            json!({"id":"import_wan","family":"wan-video","importPlan":{"checkpointId":"sha"}});
        let mut wan = obj(
            json!({"model":"import_wan","mode":"text_to_video","modelManifestEntry":wan_entry}),
        );
        assert!(candle_eligible(
            VideoJobKind::Generate,
            &wan,
            VideoRouteFacts::default()
        ));
        wan.insert("sourceAssetId".into(), json!("asset"));
        assert!(!candle_eligible(
            VideoJobKind::Generate,
            &wan,
            VideoRouteFacts::default()
        ));
        let h3_entry = json!({"id":"import_h3","family":"minimax-h3","baseModel":"minimax_h3","minimaxH3Partition":"fl2va","importSourceShape":"transformer_directory","importPlan":{"checkpointId":"sha","assembly":{"schemaVersion":1,"partition":"fl2va"}}});
        let h3 =
            obj(json!({"model":"import_h3","mode":"text_to_video","modelManifestEntry":h3_entry}));
        assert!(mlx_eligible(
            VideoJobKind::Generate,
            &h3,
            VideoRouteFacts::default()
        ));
        assert!(candle_eligible(
            VideoJobKind::Generate,
            &h3,
            VideoRouteFacts::default()
        ));
        let mut forged_builtin = h3;
        forged_builtin.insert("model".into(), json!("minimax_h3"));
        let builtin = VideoRouteFacts::new(true, true, false, false, true);
        assert!(!mlx_eligible(
            VideoJobKind::Generate,
            &forged_builtin,
            builtin
        ));
        assert!(!candle_eligible(
            VideoJobKind::Generate,
            &forged_builtin,
            builtin
        ));
    }
}
