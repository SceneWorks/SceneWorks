//! CPU-only vector walking skeleton (sc-22251).
//!
//! The fixture input is hostile by default: it is parsed into a deliberately small inert SVG
//! subset, canonicalized before persistence, rendered through resvg (which has no network/resource
//! loader), and published as an SVG+PNG directory rename. The API indexes the returned fact only
//! after both files exist, so a malformed SVG never gains a visible asset or sidecar.

use std::path::{Path, PathBuf};

use quick_xml::events::Event;
use quick_xml::{Reader, XmlVersion};
use resvg::usvg;
use serde_json::json;

use super::*;

const MAX_SVG_BYTES: usize = 256 * 1024;
const MAX_SVG_DEPTH: usize = 32;
const MAX_SVG_ELEMENTS: usize = 2_000;
const MAX_PREVIEW_DIMENSION: u32 = 2_048;
const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";

pub(crate) async fn run_image_to_svg_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    if job.payload.get("mode").and_then(serde_json::Value::as_str) != Some("image_to_svg") {
        return Err(WorkerError::InvalidPayload(
            "vector_generate only supports the image_to_svg mode".to_owned(),
        ));
    }
    let fixture = required_payload_string(&job.payload, "fixtureSvg")?;
    let canonical = sanitize_svg(fixture)?;
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let project = ProjectStore::new(settings.data_dir.clone(), "worker").get_project(project_id)?;
    let project_path = PathBuf::from(project.path);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.15,
            "Sanitizing fixture SVG.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(api, &job.id, "Vectorization canceled before publication.").await?;

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
        check_cancel(api, &job.id, "Vectorization canceled before publication.").await?;
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
        "mode": "image_to_svg",
        "model": "fixture_svg",
        "adapter": "fixture_svg",
        "prompt": "",
        "negativePrompt": "",
        "count": 1,
        "normalizedWidth": canonical.width,
        "normalizedHeight": canonical.height,
        "preview": { "path": preview_path, "mimeType": "image/png", "width": canonical.width, "height": canonical.height },
    });
    let result = json!({
        "generationSetId": generation_set_id,
        "expectedCount": 1,
        "adapter": "fixture_svg",
        "model": "fixture_svg",
        "generationSet": {
            "id": generation_set_id,
            "mode": "image_to_svg",
            "model": "fixture_svg",
            "prompt": "",
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
            "Fixture SVG stored with a PNG preview.",
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
            "fixtureSvg exceeds the 256 KiB SVG budget".to_owned(),
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
                        "fixtureSvg root must be <svg>".to_owned(),
                    ));
                }
                if stack.len() >= MAX_SVG_DEPTH {
                    return Err(WorkerError::InvalidPayload(
                        "fixtureSvg exceeds the element nesting budget".to_owned(),
                    ));
                }
                elements += 1;
                if elements > MAX_SVG_ELEMENTS {
                    return Err(WorkerError::InvalidPayload(
                        "fixtureSvg exceeds the element budget".to_owned(),
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
                        "fixtureSvg has an invalid empty root".to_owned(),
                    ));
                }
                elements += 1;
                if elements > MAX_SVG_ELEMENTS {
                    return Err(WorkerError::InvalidPayload(
                        "fixtureSvg exceeds the element budget".to_owned(),
                    ));
                }
                write_start(&mut output, &name, &attrs, true);
            }
            Ok(Event::End(event)) => {
                let name = std::str::from_utf8(event.name().as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("fixtureSvg tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
                if stack.pop().as_deref() != Some(name.as_str()) {
                    return Err(WorkerError::InvalidPayload(
                        "fixtureSvg has mismatched tags".to_owned(),
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
                        "fixtureSvg text nodes are not supported".to_owned(),
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
                    "fixtureSvg contains a disallowed XML construct".to_owned(),
                ));
            }
            Err(error) => {
                return Err(WorkerError::InvalidPayload(format!(
                    "fixtureSvg is malformed: {error}"
                )))
            }
        }
        buffer.clear();
    }
    if !stack.is_empty() || dimensions.is_none() {
        return Err(WorkerError::InvalidPayload(
            "fixtureSvg is incomplete".to_owned(),
        ));
    }
    // A second parse through the renderer proves the canonical subset is renderable before any
    // write. usvg does not load network resources, and our whitelist already removed every URL.
    usvg::Tree::from_str(&output, &usvg::Options::default()).map_err(|error| {
        WorkerError::InvalidPayload(format!("fixtureSvg cannot be rendered: {error}"))
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
        .map_err(|_| WorkerError::InvalidPayload("fixtureSvg tag is not UTF-8".to_owned()))?
        .to_owned();
    if !matches!(
        name.as_str(),
        "svg" | "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
    ) {
        return Err(WorkerError::InvalidPayload(format!(
            "fixtureSvg element <{name}> is not allowed"
        )));
    }
    let mut attrs = Vec::new();
    for attribute in event.attributes().with_checks(true) {
        let attribute = attribute.map_err(|error| {
            WorkerError::InvalidPayload(format!("fixtureSvg attribute is malformed: {error}"))
        })?;
        let key = std::str::from_utf8(attribute.key.as_ref())
            .map_err(|_| {
                WorkerError::InvalidPayload("fixtureSvg attribute is not UTF-8".to_owned())
            })?
            .to_owned();
        let value = attribute
            .decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
            .map_err(|error| {
                WorkerError::InvalidPayload(format!("fixtureSvg attribute is invalid: {error}"))
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
                "fixtureSvg attribute {key} is not allowed"
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
            "fixtureSvg dimensions must be 1..={MAX_PREVIEW_DIMENSION}"
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
                "fixtureSvg dimensions must be finite positive numbers".to_owned(),
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
                "fixtureSvg viewBox must contain four finite numbers".to_owned(),
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
            WorkerError::InvalidPayload(format!("fixtureSvg cannot be rendered: {error}"))
        })?;
        let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height).ok_or_else(|| {
            WorkerError::InvalidPayload("fixtureSvg preview dimensions are invalid".to_owned())
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
