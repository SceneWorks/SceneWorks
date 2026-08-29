//! Vector Studio request, provider, and safe-publication boundary.
//!
//! The route supplies only typed raster/text conditioning. A mode-specific native provider streams
//! the SVG through [`MultimodalVectorProviderAdapter`]; the worker does not create a staging
//! directory until that stream has completed without cancellation. The source is then parsed into
//! a deliberately small inert SVG subset, canonicalized, rendered through resvg (which has no
//! network/resource loader), and published as an SVG+PNG directory rename.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use quick_xml::events::Event;
use quick_xml::{Reader, XmlVersion};
use resvg::usvg;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::*;

const MAX_SVG_BYTES: usize = 256 * 1024;
const MAX_SVG_DEPTH: usize = 32;
const MAX_SVG_ELEMENTS: usize = 2_000;
const MAX_PREVIEW_DIMENSION: u32 = 2_048;
const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";
const VECTOR_SANITIZER_VERSION: &str = "sceneworks-inert-svg-v1";
const VECTOR_RENDERER_VERSION: &str = "resvg-0.45";
const CANCEL_MESSAGE: &str = "Vector generation canceled before publication.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum VectorMode {
    ImageToSvg,
    TextToSvg,
}

impl VectorMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ImageToSvg => "image_to_svg",
            Self::TextToSvg => "text_to_svg",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct VectorSampling {
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) top_k: u32,
    pub(crate) repetition_penalty: f32,
    pub(crate) repetition_context: u32,
    pub(crate) seed: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct VectorDetailBudget {
    pub(crate) max_new_tokens: u32,
    pub(crate) max_svg_bytes: u32,
    pub(crate) max_wall_time_ms: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VectorJobPayload {
    project_id: String,
    mode: VectorMode,
    model: String,
    #[serde(default)]
    source_asset_id: Option<String>,
    #[serde(default)]
    prompt: String,
    sampling: VectorSampling,
    detail_budget: VectorDetailBudget,
    model_manifest_entry: Value,
}

/// Backend-neutral input for the later runtime bridge. The adapter deliberately owns multimodal
/// composition: callers provide the confined raster path and disclosed text guidance separately,
/// and a backend maps them into its native text-provider request without leaking engine types here.
#[derive(Clone, Debug)]
pub(crate) struct VectorProviderRequest {
    pub(crate) mode: VectorMode,
    pub(crate) model: String,
    pub(crate) source_path: Option<PathBuf>,
    pub(crate) prompt: String,
    pub(crate) sampling: VectorSampling,
    pub(crate) detail_budget: VectorDetailBudget,
}

/// Injected multimodal text-provider seam. Implementations must poll `cancel` while decoding and
/// emit only UTF-8 source fragments. Provider discovery/installation is wired by sc-22256; this
/// story owns the lifecycle and atomicity contract that bridge plugs into.
pub(crate) trait MultimodalVectorProviderAdapter: Send + Sync {
    fn provider_id(&self) -> &str;
    fn supports_mode(&self, mode: VectorMode) -> bool;
    fn generate_svg(
        &self,
        request: &VectorProviderRequest,
        cancel: &gen_core::CancelFlag,
        on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
    ) -> WorkerResult<()>;
}

struct UnavailableVectorProvider;

impl MultimodalVectorProviderAdapter for UnavailableVectorProvider {
    fn provider_id(&self) -> &str {
        "unavailable"
    }

    fn supports_mode(&self, _mode: VectorMode) -> bool {
        false
    }

    fn generate_svg(
        &self,
        _request: &VectorProviderRequest,
        _cancel: &gen_core::CancelFlag,
        _on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
    ) -> WorkerResult<()> {
        Err(WorkerError::Engine(
            "no native vector provider bridge is registered".to_owned(),
        ))
    }
}

fn manifest_declares_mode(payload: &VectorJobPayload) -> bool {
    payload
        .model_manifest_entry
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|capabilities| {
            capabilities
                .iter()
                .any(|capability| capability.as_str() == Some(payload.mode.as_str()))
        })
}

fn collect_svg_source(
    provider: &dyn MultimodalVectorProviderAdapter,
    request: &VectorProviderRequest,
    cancel: &gen_core::CancelFlag,
) -> WorkerResult<String> {
    if request.model.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "vector model id is empty".to_owned(),
        ));
    }
    match request.mode {
        VectorMode::ImageToSvg => {
            let source_path = request.source_path.as_deref().ok_or_else(|| {
                WorkerError::InvalidPayload("image_to_svg source path is missing".to_owned())
            })?;
            if !source_path.is_file() {
                return Err(WorkerError::InvalidPayload(format!(
                    "image_to_svg source is no longer available: {}",
                    source_path.display()
                )));
            }
        }
        VectorMode::TextToSvg if request.prompt.trim().is_empty() => {
            return Err(WorkerError::InvalidPayload(
                "text_to_svg prompt is empty".to_owned(),
            ));
        }
        VectorMode::TextToSvg => {}
    }
    if !request.sampling.temperature.is_finite()
        || !request.sampling.top_p.is_finite()
        || !request.sampling.repetition_penalty.is_finite()
    {
        return Err(WorkerError::InvalidPayload(
            "vector sampling values must be finite".to_owned(),
        ));
    }
    if !provider.supports_mode(request.mode) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider {} does not declare {}",
            provider.provider_id(),
            request.mode.as_str()
        )));
    }
    let max_bytes = usize::try_from(request.detail_budget.max_svg_bytes)
        .map_err(|_| WorkerError::InvalidPayload("maxSvgBytes does not fit usize".to_owned()))?;
    let mut source = String::new();
    let mut expected_index = 0u32;
    provider.generate_svg(request, cancel, &mut |fragment, index| {
        if cancel.is_cancelled() {
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
        }
        if index != expected_index {
            return Err(WorkerError::Engine(format!(
                "vector provider emitted non-monotonic source index {index}; expected {expected_index}"
            )));
        }
        expected_index = expected_index.saturating_add(1);
        let next_len = source
            .len()
            .checked_add(fragment.len())
            .ok_or_else(|| WorkerError::InvalidPayload("SVG byte count overflow".to_owned()))?;
        if next_len > max_bytes {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG exceeds the {max_bytes}-byte detail budget"
            )));
        }
        source.push_str(fragment);
        Ok(())
    })?;
    if cancel.is_cancelled() {
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    if source.trim().is_empty() {
        return Err(WorkerError::Engine(
            "vector provider returned no SVG source".to_owned(),
        ));
    }
    Ok(source)
}

pub(crate) async fn run_vector_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    run_vector_job_with_provider(api, settings, job, Arc::new(UnavailableVectorProvider)).await
}

pub(crate) async fn run_vector_job_with_provider(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    provider: Arc<dyn MultimodalVectorProviderAdapter>,
) -> WorkerResult<()> {
    let payload: VectorJobPayload = serde_json::from_value(Value::Object(job.payload.clone()))
        .map_err(|error| WorkerError::InvalidPayload(format!("invalid VectorRequest: {error}")))?;
    if !manifest_declares_mode(&payload) {
        return Err(WorkerError::InvalidPayload(
            "selected model manifest does not declare the requested vector mode".to_owned(),
        ));
    }
    let manifest_adapter = payload
        .model_manifest_entry
        .get("adapter")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if manifest_adapter != provider.provider_id() {
        return Err(WorkerError::InvalidPayload(format!(
            "selected model declares provider {manifest_adapter}, but worker resolved {}",
            provider.provider_id()
        )));
    }
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(&payload.project_id)?;
    let project_path = PathBuf::from(project.path);
    let source_path = match (&payload.mode, payload.source_asset_id.as_deref()) {
        (VectorMode::ImageToSvg, Some(source_asset_id)) => {
            Some(store.resolve_asset_media_path(&payload.project_id, source_asset_id)?)
        }
        (VectorMode::ImageToSvg, None) => {
            return Err(WorkerError::InvalidPayload(
                "image_to_svg requires sourceAssetId".to_owned(),
            ))
        }
        (VectorMode::TextToSvg, None) => None,
        (VectorMode::TextToSvg, Some(_)) => {
            return Err(WorkerError::InvalidPayload(
                "text_to_svg does not accept sourceAssetId".to_owned(),
            ))
        }
    };

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.10,
            "Resolving vector provider and conditioning.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Generating,
            0.25,
            "Streaming SVG from the native vector provider.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let request = VectorProviderRequest {
        mode: payload.mode,
        model: payload.model.clone(),
        source_path,
        prompt: payload.prompt.clone(),
        sampling: payload.sampling.clone(),
        detail_budget: payload.detail_budget.clone(),
    };
    let cancel = gen_core::CancelFlag::new();
    let blocking_cancel = cancel.clone();
    let blocking_provider = provider.clone();
    let blocking_request = request.clone();
    let task = tokio::task::spawn_blocking(move || {
        collect_svg_source(
            blocking_provider.as_ref(),
            &blocking_request,
            &blocking_cancel,
        )
    });
    let source = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        CANCEL_MESSAGE,
        "vector provider stream",
        no_cancel_ack(),
        task,
    )
    .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Rendering,
            0.75,
            "Sanitizing SVG and rendering its preview.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let canonical = sanitize_svg(&source)?;
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;

    let created_at = now_rfc3339();
    let asset_id = fresh_asset_id();
    let generation_set_id = format!("genset_{}", uuid::Uuid::new_v4().simple());
    let base = project_path
        .join("assets")
        .join("images")
        .join(&generation_set_id);
    let staging = base.join(format!(".{asset_id}.tmp"));
    let published = base.join(&asset_id);
    let svg_path = staging.join("vector.svg");
    let preview_path = staging.join("preview.png");

    let publish_result: WorkerResult<()> = async {
        tokio::fs::create_dir_all(&staging).await?;
        tokio::fs::write(&svg_path, canonical.svg.as_bytes()).await?;
        render_preview(
            &canonical.svg,
            canonical.width,
            canonical.height,
            &preview_path,
        )
        .await?;
        check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
        tokio::fs::rename(&staging, &published).await?;
        Ok(())
    }
    .await;
    if publish_result.is_err() {
        let _ = tokio::fs::remove_dir_all(&staging).await;
    }
    publish_result?;

    let media_path = format!("assets/images/{generation_set_id}/{asset_id}/vector.svg");
    let preview_path = format!("assets/images/{generation_set_id}/{asset_id}/preview.png");
    let fact = json!({
        "assetId": asset_id,
        "type": "vector",
        "mediaPath": media_path,
        "mimeType": "image/svg+xml",
        "width": canonical.width,
        "height": canonical.height,
        "createdAt": created_at,
        "mode": payload.mode.as_str(),
        "model": payload.model,
        "adapter": provider.provider_id(),
        "prompt": payload.prompt,
        "negativePrompt": "",
        "sourceAssetId": payload.source_asset_id,
        "sampling": payload.sampling,
        "detailBudget": payload.detail_budget,
        "sanitizerVersion": VECTOR_SANITIZER_VERSION,
        "rendererVersion": VECTOR_RENDERER_VERSION,
        "count": 1,
        "normalizedWidth": canonical.width,
        "normalizedHeight": canonical.height,
        "preview": { "path": preview_path, "mimeType": "image/png", "width": canonical.width, "height": canonical.height },
    });
    let result = json!({
        "generationSetId": generation_set_id,
        "expectedCount": 1,
        "adapter": provider.provider_id(),
        "model": fact["model"],
        "generationSet": {
            "id": generation_set_id,
            "mode": fact["mode"],
            "model": fact["model"],
            "prompt": fact["prompt"],
            "negativePrompt": "",
            "count": 1,
            "createdAt": created_at,
        },
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("vector result is an object");
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Vector SVG stored with a PNG preview.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

struct CanonicalSvg {
    svg: String,
    width: u32,
    height: u32,
}

fn sanitize_svg(input: &str) -> WorkerResult<CanonicalSvg> {
    if input.len() > MAX_SVG_BYTES {
        return Err(WorkerError::InvalidPayload(
            "provider SVG exceeds the 256 KiB sanitizer budget".to_owned(),
        ));
    }
    let mut reader = Reader::from_reader(input.as_bytes());
    reader.config_mut().trim_text(true);
    let mut buffer = Vec::new();
    let mut output = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    let mut stack = Vec::new();
    let mut elements = 0usize;
    let mut dimensions = None;
    let mut root_seen = false;
    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event)) => {
                let (name, mut attrs) = canonical_element(&reader, &event)?;
                if stack.is_empty() && (root_seen || name != "svg") {
                    return Err(WorkerError::InvalidPayload(
                        "provider output root must be <svg>".to_owned(),
                    ));
                }
                if stack.len() >= MAX_SVG_DEPTH {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the element nesting budget".to_owned(),
                    ));
                }
                elements += 1;
                if elements > MAX_SVG_ELEMENTS {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the element budget".to_owned(),
                    ));
                }
                if stack.is_empty() {
                    root_seen = true;
                    if !attrs
                        .iter()
                        .any(|(key, value)| key == "xmlns" && value == SVG_NAMESPACE)
                    {
                        attrs.push(("xmlns".to_owned(), SVG_NAMESPACE.to_owned()));
                        attrs.sort_unstable();
                    }
                    dimensions = Some(svg_dimensions(&attrs)?);
                }
                write_start(&mut output, &name, &attrs, false);
                stack.push(name);
            }
            Ok(Event::Empty(event)) => {
                let (name, attrs) = canonical_element(&reader, &event)?;
                if stack.is_empty() || name == "svg" {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG has an invalid empty root".to_owned(),
                    ));
                }
                elements += 1;
                if elements > MAX_SVG_ELEMENTS {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the element budget".to_owned(),
                    ));
                }
                write_start(&mut output, &name, &attrs, true);
            }
            Ok(Event::End(event)) => {
                let name = std::str::from_utf8(event.name().as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
                if stack.pop().as_deref() != Some(name.as_str()) {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG has mismatched tags".to_owned(),
                    ));
                }
                output.push_str("</");
                output.push_str(&name);
                output.push('>');
            }
            Ok(Event::Text(text)) => {
                let bytes: &[u8] = text.as_ref();
                if !bytes.iter().all(u8::is_ascii_whitespace) {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG text nodes are not supported".to_owned(),
                    ));
                }
            }
            Ok(Event::Comment(_)) | Ok(Event::Decl(_)) => {}
            Ok(Event::Eof) => break,
            Ok(Event::CData(_))
            | Ok(Event::DocType(_))
            | Ok(Event::PI(_))
            | Ok(Event::GeneralRef(_)) => {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG contains a disallowed XML construct".to_owned(),
                ));
            }
            Err(error) => {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG is malformed: {error}"
                )))
            }
        }
        buffer.clear();
    }
    if !stack.is_empty() || dimensions.is_none() {
        return Err(WorkerError::InvalidPayload(
            "provider SVG is incomplete".to_owned(),
        ));
    }
    // A second parse through the renderer proves the canonical subset is renderable before any
    // write. usvg does not load network resources, and our whitelist already removed every URL.
    usvg::Tree::from_str(&output, &usvg::Options::default()).map_err(|error| {
        WorkerError::InvalidPayload(format!("provider SVG cannot be rendered: {error}"))
    })?;
    let (width, height) = dimensions.expect("validated above");
    Ok(CanonicalSvg {
        svg: output,
        width,
        height,
    })
}

fn canonical_element(
    reader: &Reader<&[u8]>,
    event: &quick_xml::events::BytesStart<'_>,
) -> WorkerResult<(String, Vec<(String, String)>)> {
    let name = std::str::from_utf8(event.name().as_ref())
        .map_err(|_| WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned()))?
        .to_owned();
    if !matches!(
        name.as_str(),
        "svg" | "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
    ) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> is not allowed"
        )));
    }
    let mut attrs = Vec::new();
    for attribute in event.attributes().with_checks(true) {
        let attribute = attribute.map_err(|error| {
            WorkerError::InvalidPayload(format!("provider SVG attribute is malformed: {error}"))
        })?;
        let key = std::str::from_utf8(attribute.key.as_ref())
            .map_err(|_| {
                WorkerError::InvalidPayload("provider SVG attribute is not UTF-8".to_owned())
            })?
            .to_owned();
        let value = attribute
            .decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
            .map_err(|error| {
                WorkerError::InvalidPayload(format!("provider SVG attribute is invalid: {error}"))
            })?
            .into_owned();
        // The root namespace is the one URL-looking value admitted by this inert subset; it names
        // SVG syntax and is never dereferenced. Every actual resource-bearing URL is rejected.
        let namespace_ok = key == "xmlns" && value == SVG_NAMESPACE;
        if !allowed_attribute(&name, &key)
            || (key == "xmlns" && !namespace_ok)
            || (!namespace_ok && unsafe_attribute_value(&value))
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is not allowed"
            )));
        }
        attrs.push((key, value));
    }
    attrs.sort_unstable();
    Ok((name, attrs))
}

fn allowed_attribute(element: &str, attribute: &str) -> bool {
    let common = matches!(
        attribute,
        "fill"
            | "fill-opacity"
            | "stroke"
            | "stroke-opacity"
            | "stroke-width"
            | "stroke-linecap"
            | "stroke-linejoin"
            | "opacity"
            | "transform"
    );
    common
        || match element {
            "svg" => matches!(attribute, "xmlns" | "width" | "height" | "viewBox"),
            "path" => attribute == "d",
            "rect" => matches!(attribute, "x" | "y" | "width" | "height" | "rx" | "ry"),
            "circle" => matches!(attribute, "cx" | "cy" | "r"),
            "ellipse" => matches!(attribute, "cx" | "cy" | "rx" | "ry"),
            "line" => matches!(attribute, "x1" | "x2" | "y1" | "y2"),
            "polyline" | "polygon" => attribute == "points",
            "g" => false,
            _ => false,
        }
}

fn unsafe_attribute_value(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.contains("url(")
        || lower.contains("javascript:")
        || lower.contains("data:")
        || lower.contains("http:")
        || lower.contains("https:")
        || lower.contains("//")
}

fn svg_dimensions(attrs: &[(String, String)]) -> WorkerResult<(u32, u32)> {
    let named = |name: &str| {
        attrs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    };
    let viewbox = named("viewBox").map(parse_viewbox).transpose()?;
    let width = named("width")
        .map(parse_dimension)
        .transpose()?
        .or_else(|| viewbox.map(|pair| pair.0))
        .unwrap_or(512);
    let height = named("height")
        .map(parse_dimension)
        .transpose()?
        .or_else(|| viewbox.map(|pair| pair.1))
        .unwrap_or(512);
    if width == 0 || height == 0 || width > MAX_PREVIEW_DIMENSION || height > MAX_PREVIEW_DIMENSION
    {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG dimensions must be 1..={MAX_PREVIEW_DIMENSION}"
        )));
    }
    Ok((width, height))
}

fn parse_dimension(value: &str) -> WorkerResult<u32> {
    let parsed = value
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "provider SVG dimensions must be finite positive numbers".to_owned(),
            )
        })?;
    Ok(parsed.ceil() as u32)
}

fn parse_viewbox(value: &str) -> WorkerResult<(u32, u32)> {
    let numbers = value
        .split(|character: char| character.is_ascii_whitespace() || character == ',')
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<f64>().ok().filter(|value| value.is_finite()))
        .collect::<Option<Vec<_>>>()
        .filter(|values| values.len() == 4 && values[2] > 0.0 && values[3] > 0.0)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "provider SVG viewBox must contain four finite numbers".to_owned(),
            )
        })?;
    Ok((numbers[2].ceil() as u32, numbers[3].ceil() as u32))
}

fn write_start(output: &mut String, name: &str, attrs: &[(String, String)], empty: bool) {
    output.push('<');
    output.push_str(name);
    for (key, value) in attrs {
        output.push(' ');
        output.push_str(key);
        output.push_str("=\"");
        for character in value.chars() {
            match character {
                '&' => output.push_str("&amp;"),
                '<' => output.push_str("&lt;"),
                '"' => output.push_str("&quot;"),
                _ => output.push(character),
            }
        }
        output.push('"');
    }
    output.push_str(if empty { "/>" } else { ">" });
}

async fn render_preview(svg: &str, width: u32, height: u32, path: &Path) -> WorkerResult<()> {
    let svg = svg.to_owned();
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || {
        let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).map_err(|error| {
            WorkerError::InvalidPayload(format!("provider SVG cannot be rendered: {error}"))
        })?;
        let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height).ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG preview dimensions are invalid".to_owned())
        })?;
        // resvg draws only the already-sanitized inert geometry into a fixed-size PNG.
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        pixmap
            .save_png(path)
            .map_err(|error| WorkerError::Io(std::io::Error::other(error)))
    })
    .await
    .map_err(|error| task_join_error("SVG preview render", error))??;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CancelingProvider;

    impl MultimodalVectorProviderAdapter for CancelingProvider {
        fn provider_id(&self) -> &str {
            "starvector"
        }

        fn supports_mode(&self, mode: VectorMode) -> bool {
            mode == VectorMode::TextToSvg
        }

        fn generate_svg(
            &self,
            _request: &VectorProviderRequest,
            cancel: &gen_core::CancelFlag,
            on_source: &mut dyn FnMut(&str, u32) -> WorkerResult<()>,
        ) -> WorkerResult<()> {
            on_source("<svg xmlns=\"http://www.w3.org/2000/svg\">", 0)?;
            cancel.cancel();
            on_source("<rect width=\"1\" height=\"1\"/></svg>", 1)?;
            Ok(())
        }
    }

    #[test]
    fn cancellation_during_streamed_svg_never_reaches_publication() {
        let request = VectorProviderRequest {
            mode: VectorMode::TextToSvg,
            model: "starvector_test".to_owned(),
            source_path: None,
            prompt: "a compact mark".to_owned(),
            sampling: VectorSampling {
                temperature: 0.2,
                top_p: 0.9,
                top_k: 0,
                repetition_penalty: 1.0,
                repetition_context: 0,
                seed: Some(7),
            },
            detail_budget: VectorDetailBudget {
                max_new_tokens: 128,
                max_svg_bytes: 4_096,
                max_wall_time_ms: 1_000,
            },
        };
        let cancel = gen_core::CancelFlag::new();
        let output = collect_svg_source(&CancelingProvider, &request, &cancel);
        assert!(matches!(output, Err(WorkerError::Canceled(_))));

        // This is the exact publication boundary used by `run_vector_job_with_provider`: staging
        // starts only after `collect_svg_source` returns Ok. A canceled stream has no source to
        // sanitize, render, or rename, so neither member of the asset pair can exist.
        let temp = tempfile::tempdir().expect("temp dir");
        let published = temp.path().join("asset");
        if let Ok(source) = output {
            std::fs::create_dir_all(&published).expect("publication dir");
            std::fs::write(published.join("vector.svg"), source).expect("source writes");
        }
        assert!(!published.exists());
    }

    #[test]
    fn canonicalizes_inert_svg_and_rejects_active_or_over_budget_input() {
        let valid = sanitize_svg("<svg height=\"8\" width=\"12\" xmlns=\"http://www.w3.org/2000/svg\"><rect height=\"4\" width=\"3\"/></svg>")
            .expect("valid inert fixture");
        assert_eq!((valid.width, valid.height), (12, 8));
        assert!(
            valid.svg.contains("height=\"8\" width=\"12\""),
            "attributes are canonicalized"
        );
        for malicious in [
            "<svg><script>alert(1)</script></svg>",
            "<svg><rect fill=\"url(https://example.invalid/a)\"/></svg>",
            "<svg><rect onclick=\"alert(1)\"/></svg>",
            "<svg xmlns=\"https://example.invalid/not-svg\"/>",
        ] {
            assert!(sanitize_svg(malicious).is_err(), "{malicious}");
        }
        assert!(sanitize_svg(&format!("<svg>{}</svg>", " ".repeat(MAX_SVG_BYTES))).is_err());
    }

    #[tokio::test]
    async fn renders_a_bounded_png_preview_from_the_canonical_svg() {
        let canonical = sanitize_svg(
            "<svg width=\"12\" height=\"8\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"12\" height=\"8\" fill=\"#ff0000\"/></svg>",
        )
        .expect("valid fixture");
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("preview.png");
        render_preview(&canonical.svg, canonical.width, canonical.height, &path)
            .await
            .expect("preview writes");
        let preview = image::open(&path).expect("preview decodes").to_rgba8();
        assert_eq!(preview.dimensions(), (12, 8));
        assert!(
            preview.get_pixel(0, 0)[0] > 200,
            "preview rendered the red rectangle"
        );
    }
}
