//! Bounded, inert SVG sanitization for vector jobs.
use super::{WorkerError, WorkerResult, MAX_PREVIEW_DIMENSION};
use cssparser::{Parser as CssParser, ParserInput as CssParserInput, Token as CssToken};
use quick_xml::events::Event;
use quick_xml::{Reader, XmlVersion};
use resvg::usvg;
use std::collections::{BTreeMap, BTreeSet};

pub(super) const MAX_SVG_BYTES: usize = 256 * 1024;
pub(super) const MAX_SVG_DEPTH: usize = 32;
pub(super) const MAX_SVG_ELEMENTS: usize = 2_000;
pub(super) const MAX_SVG_ATTRIBUTES: usize = 4_096;
const MAX_SVG_ATTRIBUTES_PER_ELEMENT: usize = 12;
pub(super) const MAX_SVG_ATTRIBUTE_VALUE_BYTES: usize = 64 * 1024;
const MAX_SVG_ATTRIBUTE_VALUE_BYTES_TOTAL: usize = 192 * 1024;
pub(super) const MAX_SVG_PATH_DATA_BYTES: usize = 160 * 1024;
pub(super) const MAX_SVG_PATH_COMMANDS: usize = 16_384;
pub(super) const MAX_SVG_PATH_NUMBERS: usize = 65_536;
pub(super) const MAX_SVG_POINT_NUMBERS: usize = 32_768;
pub(super) const MAX_SVG_TRANSFORM_NUMBERS: usize = 4_096;
pub(super) const MAX_SVG_DASH_NUMBERS: usize = 4_096;
const MAX_STATIC_SVG_CSS_BYTES: usize = 64 * 1024;
const MAX_STATIC_SVG_CSS_TOKENS: usize = 16_384;
const MAX_STATIC_SVG_CSS_BLOCKS: usize = 512;
pub(super) const MAX_SVG_RESOURCES: usize = 128;
pub(super) const MAX_SVG_RESOURCE_REFERENCES: usize = 256;
pub(super) const MAX_SVG_GRADIENT_STOPS: usize = 512;
pub(super) const MAX_SVG_FILTER_PRIMITIVES: usize = 64;
const MAX_SVG_FILTER_BLUR_SIGMA: f64 = 64.0;
// Eight maximum-size RGBA renderer surfaces (128 MiB). The parsed-tree check below charges the
// actual clipped layer for every application, result, and input surface before resvg can allocate.
pub(super) const MAX_SVG_FILTER_OFFSCREEN_PIXELS: u64 =
    MAX_PREVIEW_DIMENSION as u64 * MAX_PREVIEW_DIMENSION as u64 * 8;
const MAX_SVG_COORDINATE_MAGNITUDE: f64 = 1_000_000.0;
const MAX_SVG_VIEWBOX_ORIGIN_MAGNITUDE: f64 = 1_000_000.0;
pub(super) const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";
pub(super) const VECTOR_SANITIZER_VERSION: &str = "sceneworks-static-svg-v2";

#[derive(Debug)]
pub(super) struct CanonicalSvg {
    pub(super) svg: String,
    pub(super) width: u32,
    pub(super) height: u32,
}

#[derive(Default)]
struct SanitizerBudget {
    attributes: usize,
    attribute_value_bytes: usize,
    path_data_bytes: usize,
    path_commands: usize,
    path_numbers: usize,
    point_numbers: usize,
    transform_numbers: usize,
    dash_numbers: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SvgResourceKind {
    ClipPath,
    LinearGradient,
    Filter,
    UsePath,
}

#[derive(Debug)]
struct SvgResourceReference {
    source: Option<String>,
    target: String,
    expected: SvgResourceKind,
}

#[derive(Debug)]
struct SvgResourceCatalog {
    ids: BTreeMap<String, SvgResourceKind>,
    use_paths: BTreeMap<String, Vec<(String, String)>>,
    width: u32,
    height: u32,
}

pub(super) fn sanitize_svg(input: &str) -> WorkerResult<CanonicalSvg> {
    sanitize_svg_bytes(input.as_bytes())
}

#[derive(Debug)]
struct SvgIndexFrame {
    name: String,
    hidden: bool,
    filter_id: Option<String>,
    resource_owner: Option<String>,
}

fn direct_use_path_definition(name: &str, empty: bool, stack: &[SvgIndexFrame]) -> bool {
    name == "path" && empty && stack.len() == 2 && stack[0].name == "svg" && stack[1].name == "defs"
}

fn canonical_use_path_definition(
    attrs: &[(String, String)],
) -> WorkerResult<(String, Vec<(String, String)>)> {
    if attrs.len() != 2
        || attrs
            .iter()
            .any(|(key, _)| !matches!(key.as_str(), "id" | "d"))
    {
        return Err(WorkerError::InvalidPayload(
            "provider SVG local use path definition permits only id and d".to_owned(),
        ));
    }
    let id = attribute_value(attrs, "id").ok_or_else(|| {
        WorkerError::InvalidPayload("provider SVG local use path requires an id".to_owned())
    })?;
    validate_resource_id(id, "local use path id")?;
    let d = attribute_value(attrs, "d").ok_or_else(|| {
        WorkerError::InvalidPayload("provider SVG local use path requires path data".to_owned())
    })?;
    let mut definition_budget = SanitizerBudget::default();
    validate_attribute_resource_budget("path", "d", d, &mut definition_budget)?;
    Ok((id.to_owned(), vec![("d".to_owned(), d.to_owned())]))
}

fn attribute_value<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs
        .iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| value.as_str())
}

fn validate_resource_id(value: &str, label: &str) -> WorkerResult<()> {
    let mut bytes = value.bytes();
    if value.len() > 128
        || !bytes
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} is not a bounded local identifier"
        )));
    }
    Ok(())
}

fn local_url_target(value: &str) -> Option<&str> {
    value
        .strip_prefix("url(#")
        .and_then(|value| value.strip_suffix(')'))
        .filter(|value| !value.is_empty())
}

fn local_href_target(value: &str) -> Option<&str> {
    value.strip_prefix('#').filter(|value| !value.is_empty())
}

fn validate_filter_input(value: &str, results: &BTreeSet<String>, label: &str) -> WorkerResult<()> {
    if matches!(value, "SourceGraphic" | "SourceAlpha") || results.contains(value) {
        Ok(())
    } else {
        Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} must reference an earlier filter result"
        )))
    }
}

fn index_filter_primitive(
    name: &str,
    attrs: &[(String, String)],
    filter_id: Option<&str>,
    filter_results: &mut BTreeMap<String, BTreeSet<String>>,
    filter_primitives: &mut usize,
) -> WorkerResult<()> {
    if !matches!(
        name,
        "feGaussianBlur" | "feOffset" | "feMerge" | "feMergeNode"
    ) {
        return Ok(());
    }
    let filter_id = filter_id.ok_or_else(|| {
        WorkerError::InvalidPayload(format!("provider SVG element <{name}> is outside a filter"))
    })?;
    *filter_primitives = filter_primitives.checked_add(1).ok_or_else(|| {
        WorkerError::InvalidPayload("provider SVG filter primitive count overflow".to_owned())
    })?;
    if *filter_primitives > MAX_SVG_FILTER_PRIMITIVES {
        return Err(WorkerError::InvalidPayload(
            "provider SVG exceeds the filter-primitive budget".to_owned(),
        ));
    }
    let results = filter_results.entry(filter_id.to_owned()).or_default();
    if let Some(input) = attribute_value(attrs, "in") {
        validate_filter_input(input, results, name)?;
    } else if matches!(name, "feGaussianBlur" | "feOffset" | "feMergeNode") {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {name} requires an input"
        )));
    }
    if let Some(result) = attribute_value(attrs, "result") {
        if !matches!(name, "feGaussianBlur" | "feOffset") {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG {name} cannot name a result"
            )));
        }
        validate_resource_id(result, "filter result")?;
        if !results.insert(result.to_owned()) {
            return Err(WorkerError::InvalidPayload(
                "provider SVG filter result is duplicated".to_owned(),
            ));
        }
    } else if matches!(name, "feGaussianBlur" | "feOffset") {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {name} requires a result"
        )));
    }
    Ok(())
}

fn visit_resource(
    id: &str,
    edges: &BTreeMap<String, Vec<String>>,
    visiting: &mut BTreeSet<String>,
    complete: &mut BTreeSet<String>,
) -> WorkerResult<()> {
    if complete.contains(id) {
        return Ok(());
    }
    if !visiting.insert(id.to_owned()) {
        return Err(WorkerError::InvalidPayload(
            "provider SVG resource cycle is not allowed".to_owned(),
        ));
    }
    if let Some(targets) = edges.get(id) {
        for target in targets {
            visit_resource(target, edges, visiting, complete)?;
        }
    }
    visiting.remove(id);
    complete.insert(id.to_owned());
    Ok(())
}

fn validate_resource_cycles(references: &[SvgResourceReference]) -> WorkerResult<()> {
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for reference in references {
        if let Some(source) = &reference.source {
            edges
                .entry(source.clone())
                .or_default()
                .push(reference.target.clone());
        }
    }
    let mut visiting = BTreeSet::new();
    let mut complete = BTreeSet::new();
    for id in edges.keys() {
        visit_resource(id, &edges, &mut visiting, &mut complete)?;
    }
    Ok(())
}

fn index_svg_resources(input: &str) -> WorkerResult<SvgResourceCatalog> {
    let mut reader = Reader::from_reader(input.as_bytes());
    reader.config_mut().trim_text(true);
    let mut buffer = Vec::new();
    let mut stack: Vec<SvgIndexFrame> = Vec::new();
    let mut budget = SanitizerBudget::default();
    let mut elements = 0usize;
    let mut ids = BTreeMap::new();
    let mut use_paths = BTreeMap::new();
    let mut id_counts = BTreeMap::new();
    let mut references = Vec::new();
    let mut filter_results = BTreeMap::new();
    let mut resources = 0usize;
    let mut gradient_stops = 0usize;
    let mut filter_primitives = 0usize;
    let mut dimensions = None;
    let mut xlink_namespace_bound = false;

    loop {
        let (event, empty) = match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event)) => (event, false),
            Ok(Event::Empty(event)) => (event, true),
            Ok(Event::End(_)) => {
                stack.pop();
                buffer.clear();
                continue;
            }
            Ok(Event::Eof) => break,
            Ok(_) => {
                buffer.clear();
                continue;
            }
            Err(error) => {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG is malformed: {error}"
                )))
            }
        };
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
        let name = std::str::from_utf8(event.name().as_ref())
            .map_err(|_| WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned()))?
            .to_owned();
        let is_root = stack.is_empty();
        let attrs = source_attributes(
            &reader,
            &event,
            source_attribute_limit(&name, is_root),
            &mut budget,
        )?;
        if is_root && name == "svg" {
            let mut root_attrs = attrs.clone();
            normalize_root_dimensions(&mut root_attrs)?;
            dimensions = Some(svg_dimensions(&root_attrs)?);
            xlink_namespace_bound =
                attribute_value(&attrs, "xmlns:xlink") == Some("http://www.w3.org/1999/xlink");
        }
        if attrs.iter().any(|(key, _)| key == "xlink:href") && !xlink_namespace_bound {
            return Err(WorkerError::InvalidPayload(
                "provider SVG xlink:href requires the exact root xmlns:xlink binding".to_owned(),
            ));
        }
        if let Some(id) = attribute_value(&attrs, "id") {
            *id_counts.entry(id.to_owned()).or_insert(0usize) += 1;
        }
        let hidden = stack.last().is_some_and(|frame| frame.hidden)
            || (name == "g" && attribute_value(&attrs, "display") == Some("none"));
        let is_use_path_definition = direct_use_path_definition(&name, empty, &stack);
        let resource_kind = if hidden {
            None
        } else {
            match name.as_str() {
                "clipPath" => Some(SvgResourceKind::ClipPath),
                "linearGradient" => Some(SvgResourceKind::LinearGradient),
                "filter" => Some(SvgResourceKind::Filter),
                "path" if is_use_path_definition => Some(SvgResourceKind::UsePath),
                _ => None,
            }
        };
        let resource_id = resource_kind
            .map(|kind| {
                let id = attribute_value(&attrs, "id").ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "provider SVG resource <{name}> requires an id"
                    ))
                })?;
                validate_resource_id(id, "resource id")?;
                resources = resources.checked_add(1).ok_or_else(|| {
                    WorkerError::InvalidPayload("provider SVG resource count overflow".to_owned())
                })?;
                if resources > MAX_SVG_RESOURCES {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the resource budget".to_owned(),
                    ));
                }
                if ids.insert(id.to_owned(), kind).is_some() {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG resource id is duplicated".to_owned(),
                    ));
                }
                Ok(id.to_owned())
            })
            .transpose()?;
        if is_use_path_definition {
            let (id, canonical_attrs) = canonical_use_path_definition(&attrs)?;
            if use_paths.insert(id, canonical_attrs).is_some() {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG local use path id is duplicated".to_owned(),
                ));
            }
        }
        if !hidden && name == "stop" {
            gradient_stops = gradient_stops.checked_add(1).ok_or_else(|| {
                WorkerError::InvalidPayload("provider SVG gradient stop count overflow".to_owned())
            })?;
            if gradient_stops > MAX_SVG_GRADIENT_STOPS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the gradient-stop budget".to_owned(),
                ));
            }
        }
        let filter_id = if name == "filter" {
            resource_id.clone()
        } else {
            stack.last().and_then(|frame| frame.filter_id.clone())
        };
        let resource_owner = resource_id
            .clone()
            .or_else(|| stack.last().and_then(|frame| frame.resource_owner.clone()));
        if !hidden {
            if name == "use" {
                if !empty {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG local use must be empty".to_owned(),
                    ));
                }
                let hrefs = attrs
                    .iter()
                    .filter(|(key, _)| matches!(key.as_str(), "href" | "xlink:href"))
                    .collect::<Vec<_>>();
                if hrefs.len() != 1 {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG local use requires exactly one href".to_owned(),
                    ));
                }
                let target = local_href_target(&hrefs[0].1).ok_or_else(|| {
                    WorkerError::InvalidPayload(
                        "provider SVG local use href must be an exact local fragment".to_owned(),
                    )
                })?;
                validate_resource_id(target, "local use reference")?;
                if references.len() >= MAX_SVG_RESOURCE_REFERENCES {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the resource-reference budget".to_owned(),
                    ));
                }
                references.push(SvgResourceReference {
                    source: resource_owner.clone(),
                    target: target.to_owned(),
                    expected: SvgResourceKind::UsePath,
                });
            }
            index_filter_primitive(
                &name,
                &attrs,
                filter_id.as_deref(),
                &mut filter_results,
                &mut filter_primitives,
            )?;
            for (key, value) in &attrs {
                if key == "style" {
                    for declaration in value.split(';') {
                        let Some((property, property_value)) = declaration.split_once(':') else {
                            continue;
                        };
                        if property.trim() != "fill" {
                            continue;
                        }
                        let Some(target) = local_url_target(property_value.trim()) else {
                            continue;
                        };
                        validate_resource_id(target, "resource reference")?;
                        if references.len() >= MAX_SVG_RESOURCE_REFERENCES {
                            return Err(WorkerError::InvalidPayload(
                                "provider SVG exceeds the resource-reference budget".to_owned(),
                            ));
                        }
                        references.push(SvgResourceReference {
                            source: resource_owner.clone(),
                            target: target.to_owned(),
                            expected: SvgResourceKind::LinearGradient,
                        });
                    }
                }
                let expected = match key.as_str() {
                    "fill" => local_url_target(value).map(|_| SvgResourceKind::LinearGradient),
                    "clip-path" => local_url_target(value).map(|_| SvgResourceKind::ClipPath),
                    "filter" => local_url_target(value).map(|_| SvgResourceKind::Filter),
                    "href" | "xlink:href" if name == "linearGradient" => {
                        local_href_target(value).map(|_| SvgResourceKind::LinearGradient)
                    }
                    _ => None,
                };
                if let Some(expected) = expected {
                    let target = if matches!(key.as_str(), "href" | "xlink:href") {
                        local_href_target(value)
                    } else {
                        local_url_target(value)
                    }
                    .expect("matched above");
                    validate_resource_id(target, "resource reference")?;
                    if references.len() >= MAX_SVG_RESOURCE_REFERENCES {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG exceeds the resource-reference budget".to_owned(),
                        ));
                    }
                    references.push(SvgResourceReference {
                        source: resource_owner.clone(),
                        target: target.to_owned(),
                        expected,
                    });
                }
            }
        }
        if !empty {
            stack.push(SvgIndexFrame {
                name,
                hidden,
                filter_id,
                resource_owner,
            });
        }
        buffer.clear();
    }
    let (width, height) = dimensions
        .ok_or_else(|| WorkerError::InvalidPayload("provider output has no SVG root".to_owned()))?;
    for reference in &references {
        if id_counts.get(&reference.target) != Some(&1)
            || ids.get(&reference.target) != Some(&reference.expected)
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG local resource reference #{} is unresolved or has the wrong type",
                reference.target
            )));
        }
    }
    for id in use_paths.keys() {
        if !references.iter().any(|reference| {
            reference.expected == SvgResourceKind::UsePath
                && reference.target.as_str() == id.as_str()
        }) {
            return Err(WorkerError::InvalidPayload(
                "provider SVG local use path is not referenced".to_owned(),
            ));
        }
    }
    validate_resource_cycles(&references)?;
    Ok(SvgResourceCatalog {
        ids,
        use_paths,
        width,
        height,
    })
}

fn validate_indexed_use_path_definition(
    attrs: &[(String, String)],
    resources: &SvgResourceCatalog,
) -> WorkerResult<()> {
    let (id, canonical_attrs) = canonical_use_path_definition(attrs)?;
    if resources.ids.get(&id) != Some(&SvgResourceKind::UsePath)
        || resources.use_paths.get(&id) != Some(&canonical_attrs)
    {
        return Err(WorkerError::InvalidPayload(
            "provider SVG local use path differs from the validated index".to_owned(),
        ));
    }
    Ok(())
}

struct CanonicalLocalUse {
    wrapper_attrs: Vec<(String, String)>,
    path_attrs: Vec<(String, String)>,
}

fn canonical_local_use(
    source_attrs: Vec<(String, String)>,
    resources: &SvgResourceCatalog,
    budget: &mut SanitizerBudget,
) -> WorkerResult<CanonicalLocalUse> {
    let mut href = None;
    let mut wrapper_attrs = Vec::new();
    for (key, value) in source_attrs {
        if matches!(key.as_str(), "href" | "xlink:href") {
            if href.is_some() {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG local use requires exactly one href".to_owned(),
                ));
            }
            let target = local_href_target(&value).ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "provider SVG local use href must be an exact local fragment".to_owned(),
                )
            })?;
            validate_resource_id(target, "local use reference")?;
            href = Some(target.to_owned());
        } else {
            wrapper_attrs.push((key, value));
        }
    }
    let target = href.ok_or_else(|| {
        WorkerError::InvalidPayload("provider SVG local use requires exactly one href".to_owned())
    })?;
    if resources.ids.get(&target) != Some(&SvgResourceKind::UsePath) {
        return Err(WorkerError::InvalidPayload(
            "provider SVG local use href has an unresolved or mistyped resource".to_owned(),
        ));
    }
    let (_, wrapper_attrs) = canonical_element("g", wrapper_attrs, false, resources, budget)?;
    let definition_attrs = resources.use_paths.get(&target).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "provider SVG local use path is absent from the validated index".to_owned(),
        )
    })?;
    let (_, path_attrs) =
        canonical_element("path", definition_attrs.clone(), false, resources, budget)?;
    Ok(CanonicalLocalUse {
        wrapper_attrs,
        path_attrs,
    })
}

fn record_canonical_output(output: &str, emitted_elements: usize) -> WorkerResult<()> {
    if emitted_elements > MAX_SVG_ELEMENTS {
        return Err(WorkerError::InvalidPayload(
            "provider SVG expansion exceeds the element budget".to_owned(),
        ));
    }
    const XML_DECLARATION: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
    if output.len().saturating_sub(XML_DECLARATION.len()) > MAX_SVG_BYTES {
        return Err(WorkerError::InvalidPayload(
            "provider SVG expansion exceeds the 256 KiB sanitizer budget".to_owned(),
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct StaticSvgSourceNode {
    name: String,
    children: Vec<usize>,
    href_target: Option<String>,
    path_data_bytes: usize,
    path_commands: usize,
    path_numbers: usize,
}

#[derive(Debug)]
struct StaticSvgFrame {
    node: usize,
    name: String,
    filter_id: Option<String>,
    hidden: bool,
    hidden_root_start: Option<usize>,
    style_text: String,
}

#[derive(Debug)]
struct StaticHiddenRange {
    start: usize,
    end: usize,
    node: usize,
}

#[derive(Default)]
struct StaticCssBudget {
    bytes: usize,
    tokens: usize,
    blocks: usize,
}

struct StaticSvgValidationContext<'a> {
    sanitizer_budget: &'a mut SanitizerBudget,
    css_budget: &'a mut StaticCssBudget,
    references: &'a mut Vec<String>,
    typed_references: &'a mut Vec<StaticTypedReference>,
}

#[derive(Clone, Copy, Debug)]
enum StaticReferenceKind {
    Paint,
    ClipPath,
    Filter,
}

#[derive(Debug)]
struct StaticTypedReference {
    source: usize,
    target: String,
    kind: StaticReferenceKind,
}

fn static_svg_element_allowed(name: &str, hidden: bool) -> bool {
    matches!(
        name,
        "svg"
            | "g"
            | "defs"
            | "symbol"
            | "use"
            | "path"
            | "rect"
            | "circle"
            | "ellipse"
            | "line"
            | "polyline"
            | "polygon"
            | "style"
            | "title"
            | "desc"
            | "metadata"
            | "linearGradient"
            | "radialGradient"
            | "stop"
            | "clipPath"
            | "filter"
            | "feGaussianBlur"
            | "feOffset"
            | "feMerge"
            | "feMergeNode"
    ) || (hidden && matches!(name, "text" | "tspan"))
}

fn validate_static_svg_id(value: &str, label: &str) -> WorkerResult<()> {
    if value.is_empty()
        || value.len() > 128
        || value.chars().any(|character| {
            character.is_control() || character.is_whitespace() || character == '#'
        })
    {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} is not a bounded local identifier"
        )));
    }
    Ok(())
}

fn record_static_css_token(budget: &mut StaticCssBudget) -> Result<(), ()> {
    budget.tokens = budget.tokens.checked_add(1).ok_or(())?;
    if budget.tokens > MAX_STATIC_SVG_CSS_TOKENS {
        return Err(());
    }
    Ok(())
}

fn record_static_css_block(budget: &mut StaticCssBudget) -> Result<(), ()> {
    budget.blocks = budget.blocks.checked_add(1).ok_or(())?;
    if budget.blocks > MAX_STATIC_SVG_CSS_BLOCKS {
        return Err(());
    }
    Ok(())
}

fn validate_static_css_url(value: &str, references: &mut Vec<String>) -> Result<(), ()> {
    let target = value
        .strip_prefix('#')
        .filter(|target| !target.is_empty())
        .ok_or(())?;
    validate_static_svg_id(target, "CSS resource reference").map_err(|_| ())?;
    references.push(target.to_owned());
    if references.len() > MAX_SVG_RESOURCE_REFERENCES {
        return Err(());
    }
    Ok(())
}

fn validate_static_css_components<'i, 't>(
    parser: &mut CssParser<'i, 't>,
    budget: &mut StaticCssBudget,
    references: &mut Vec<String>,
) -> Result<(), ()> {
    while !parser.is_exhausted() {
        let token = parser
            .next_including_whitespace_and_comments()
            .map_err(|_| ())?
            .clone();
        record_static_css_token(budget)?;
        if token.is_parse_error() {
            return Err(());
        }
        match token {
            CssToken::AtKeyword(name) if name.eq_ignore_ascii_case("keyframes") => {}
            CssToken::AtKeyword(_) => return Err(()),
            CssToken::Ident(name) if name.starts_with("--") => return Err(()),
            CssToken::UnquotedUrl(value) => validate_static_css_url(&value, references)?,
            CssToken::Function(name)
                if name.eq_ignore_ascii_case("expression")
                    || name.eq_ignore_ascii_case("behavior") =>
            {
                return Err(())
            }
            CssToken::Function(name) if name.eq_ignore_ascii_case("url") => {
                record_static_css_block(budget)?;
                parser
                    .parse_nested_block(|nested| {
                        let value = match nested.next()?.clone() {
                            CssToken::QuotedString(value) | CssToken::UnquotedUrl(value) => value,
                            _ => return Err(nested.new_custom_error::<(), ()>(())),
                        };
                        if !nested.is_exhausted()
                            || validate_static_css_url(&value, references).is_err()
                        {
                            return Err(nested.new_custom_error::<(), ()>(()));
                        }
                        Ok(())
                    })
                    .map_err(|_| ())?;
            }
            CssToken::Function(_)
            | CssToken::ParenthesisBlock
            | CssToken::SquareBracketBlock
            | CssToken::CurlyBracketBlock => {
                record_static_css_block(budget)?;
                parser
                    .parse_nested_block(|nested| {
                        validate_static_css_components(nested, budget, references)
                            .map_err(|_| nested.new_custom_error::<(), ()>(()))
                    })
                    .map_err(|_| ())?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_static_css(
    css: &str,
    budget: &mut StaticCssBudget,
    references: &mut Vec<String>,
) -> WorkerResult<()> {
    budget.bytes = budget.bytes.checked_add(css.len()).ok_or_else(|| {
        WorkerError::InvalidPayload("provider SVG stylesheet byte count overflow".to_owned())
    })?;
    if budget.bytes > MAX_STATIC_SVG_CSS_BYTES {
        return Err(WorkerError::InvalidPayload(
            "provider SVG stylesheets exceed the CSS byte budget".to_owned(),
        ));
    }
    validate_static_css_lexical_boundaries(css)?;
    let mut input = CssParserInput::new(css);
    let mut parser = CssParser::new(&mut input);
    validate_static_css_components(&mut parser, budget, references).map_err(|_| {
        WorkerError::InvalidPayload(
            "provider SVG stylesheet is malformed, active, external, or exceeds its budget"
                .to_owned(),
        )
    })
}

fn static_css_property_allowed(name: &str) -> bool {
    matches!(
        name,
        "fill"
            | "fill-rule"
            | "clip-rule"
            | "fill-opacity"
            | "stroke"
            | "stroke-opacity"
            | "stroke-width"
            | "stroke-linecap"
            | "stroke-linejoin"
            | "stroke-miterlimit"
            | "stroke-dasharray"
            | "stroke-dashoffset"
            | "opacity"
            | "color"
            | "color-interpolation"
            | "color-interpolation-filters"
            | "color-rendering"
            | "display"
            | "visibility"
            | "shape-rendering"
            | "paint-order"
            | "vector-effect"
            | "clip-path"
            | "filter"
            | "stop-color"
            | "stop-opacity"
            | "flood-color"
            | "flood-opacity"
            | "enable-background"
            // usvg intentionally emits no animation. These declarations and their inspected
            // keyframes may influence neither the static initial tree nor the canonical output.
            | "animation"
            | "animation-name"
            | "animation-duration"
            | "animation-delay"
            | "animation-direction"
            | "animation-fill-mode"
            | "animation-iteration-count"
            | "animation-play-state"
            | "animation-timing-function"
    )
}

fn record_static_css_resource_reference(
    property: &str,
    value: &str,
    references: &mut Vec<StaticTypedReference>,
) -> WorkerResult<()> {
    if !value.to_ascii_lowercase().contains("url") {
        return Ok(());
    }
    let target = local_url_target(value.trim()).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "provider SVG CSS property {property} must use an exact local fragment URL"
        ))
    })?;
    let kind = match property {
        "fill" | "stroke" => StaticReferenceKind::Paint,
        "clip-path" => StaticReferenceKind::ClipPath,
        "filter" => StaticReferenceKind::Filter,
        _ => {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG CSS property {property} does not support a resource URL"
            )))
        }
    };
    references.push(StaticTypedReference {
        source: usize::MAX,
        target: target.to_owned(),
        kind,
    });
    Ok(())
}

fn validate_static_css_declarations(
    css: &str,
    stylesheet: bool,
    references: &mut Vec<StaticTypedReference>,
    sanitizer_budget: &mut SanitizerBudget,
) -> WorkerResult<()> {
    let wrapped;
    let source = if stylesheet {
        css
    } else {
        wrapped = format!("*{{{css}}}");
        &wrapped
    };
    let sheet = simplecss::StyleSheet::parse(source);
    let mut declaration_count = 0usize;
    for rule in sheet.rules {
        for declaration in rule.declarations {
            declaration_count += 1;
            if !static_css_property_allowed(declaration.name) || declaration.value.trim().is_empty()
            {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG CSS property {} is unsupported by the static renderer",
                    declaration.name
                )));
            }
            if declaration.name == "vector-effect"
                && !matches!(declaration.value.trim(), "none" | "non-scaling-stroke")
            {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG CSS vector-effect value is unsupported".to_owned(),
                ));
            }
            if declaration.name == "paint-order" {
                validate_paint_order(declaration.value.trim())?;
            }
            validate_attribute_resource_budget(
                "g",
                declaration.name,
                declaration.value.trim(),
                sanitizer_budget,
            )?;
            record_static_css_resource_reference(declaration.name, declaration.value, references)?;
        }
    }
    if !stylesheet && !css.trim().is_empty() && declaration_count == 0 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG inline style has no valid static declarations".to_owned(),
        ));
    }
    Ok(())
}

// CSS Syntax tokenization recovers an EOF-terminated string as a string token. That recovery is
// useful in a browser, but it would let malformed model output silently lose a declaration when
// simplecss builds the usvg tree. Keep cssparser authoritative for tokens and nested blocks, and
// reject only unterminated lexical constructs before asking it to interpret the stylesheet.
fn validate_static_css_lexical_boundaries(css: &str) -> WorkerResult<()> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Normal,
        Comment,
        SingleQuoted,
        DoubleQuoted,
    }

    let bytes = css.as_bytes();
    let mut state = State::Normal;
    let mut index = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        match state {
            State::Normal if byte == b'/' && bytes.get(index + 1) == Some(&b'*') => {
                state = State::Comment;
                index += 2;
                continue;
            }
            State::Normal if byte == b'\'' => state = State::SingleQuoted,
            State::Normal if byte == b'"' => state = State::DoubleQuoted,
            State::Comment if byte == b'*' && bytes.get(index + 1) == Some(&b'/') => {
                state = State::Normal;
                index += 2;
                continue;
            }
            State::SingleQuoted if byte == b'\\' => {
                index = index.saturating_add(2);
                continue;
            }
            State::DoubleQuoted if byte == b'\\' => {
                index = index.saturating_add(2);
                continue;
            }
            State::SingleQuoted if byte == b'\'' => state = State::Normal,
            State::DoubleQuoted if byte == b'"' => state = State::Normal,
            _ => {}
        }
        index += 1;
    }
    if state != State::Normal {
        return Err(WorkerError::InvalidPayload(
            "provider SVG stylesheet has an unterminated string or comment".to_owned(),
        ));
    }
    Ok(())
}

fn static_svg_obvious_external_value(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    let compact = lower
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect::<String>();
    compact.contains("javascript:")
        || compact.contains("data:")
        || compact.contains("file:")
        || compact.contains("http:")
        || compact.contains("https:")
        || compact.contains("ftp:")
        || compact.contains("ws:")
        || compact.contains("wss:")
        || compact.starts_with("//")
}

fn static_svg_attribute_allowed(element: &str, key: &str, hidden: bool) -> bool {
    let common = matches!(key, "id" | "class" | "style");
    let presentation = matches!(
        key,
        "display"
            | "visibility"
            | "color"
            | "color-interpolation"
            | "color-interpolation-filters"
            | "color-rendering"
            | "fill"
            | "fill-opacity"
            | "fill-rule"
            | "clip-rule"
            | "clip-path"
            | "filter"
            | "opacity"
            | "paint-order"
            | "shape-rendering"
            | "stroke"
            | "stroke-opacity"
            | "stroke-width"
            | "stroke-linecap"
            | "stroke-linejoin"
            | "stroke-miterlimit"
            | "stroke-dasharray"
            | "stroke-dashoffset"
            | "transform"
            | "vector-effect"
    );
    common
        || (presentation
            && matches!(
                element,
                "svg"
                    | "g"
                    | "symbol"
                    | "use"
                    | "path"
                    | "rect"
                    | "circle"
                    | "ellipse"
                    | "line"
                    | "polyline"
                    | "polygon"
                    | "text"
                    | "tspan"
            ))
        || match element {
            "svg" => matches!(
                key,
                "xmlns"
                    | "xmlns:xlink"
                    | "version"
                    | "baseProfile"
                    | "width"
                    | "height"
                    | "viewBox"
                    | "preserveAspectRatio"
                    | "overflow"
                    | "x"
                    | "y"
                    | "xml:space"
                    | "enable-background"
            ),
            "defs" | "g" => false,
            "symbol" => matches!(
                key,
                "viewBox" | "preserveAspectRatio" | "overflow" | "x" | "y" | "width" | "height"
            ),
            "use" => matches!(key, "href" | "xlink:href" | "x" | "y" | "width" | "height"),
            "path" => matches!(key, "d" | "pathLength"),
            "rect" => matches!(key, "x" | "y" | "width" | "height" | "rx" | "ry"),
            "circle" => matches!(key, "cx" | "cy" | "r" | "pathLength"),
            "ellipse" => matches!(key, "cx" | "cy" | "rx" | "ry" | "pathLength"),
            "line" => matches!(key, "x1" | "x2" | "y1" | "y2" | "pathLength"),
            "polyline" | "polygon" => matches!(key, "points" | "pathLength"),
            "style" => key == "type",
            "title" | "desc" | "metadata" => key == "id",
            "linearGradient" => matches!(
                key,
                "x1" | "y1"
                    | "x2"
                    | "y2"
                    | "gradientUnits"
                    | "gradientTransform"
                    | "spreadMethod"
                    | "href"
                    | "xlink:href"
            ),
            "radialGradient" => matches!(
                key,
                "cx" | "cy"
                    | "r"
                    | "fx"
                    | "fy"
                    | "fr"
                    | "gradientUnits"
                    | "gradientTransform"
                    | "spreadMethod"
                    | "href"
                    | "xlink:href"
            ),
            "stop" => matches!(key, "offset" | "stop-color" | "stop-opacity"),
            "clipPath" => matches!(key, "clipPathUnits"),
            "filter" => matches!(
                key,
                "x" | "y" | "width" | "height" | "filterUnits" | "primitiveUnits"
            ),
            "feGaussianBlur" => matches!(key, "in" | "result" | "stdDeviation"),
            "feOffset" => matches!(key, "in" | "result" | "dx" | "dy"),
            "feMerge" => false,
            "feMergeNode" => key == "in",
            "text" | "tspan" if hidden => matches!(
                key,
                "x" | "y"
                    | "dx"
                    | "dy"
                    | "font"
                    | "font-family"
                    | "font-size"
                    | "font-style"
                    | "font-weight"
                    | "text-anchor"
                    | "xml:space"
            ),
            _ => false,
        }
}

fn validate_paint_order(value: &str) -> WorkerResult<()> {
    let tokens = value.split_ascii_whitespace().collect::<Vec<_>>();
    if value != "normal"
        && (tokens.is_empty()
            || tokens.len() > 3
            || tokens
                .iter()
                .any(|token| !matches!(*token, "fill" | "stroke" | "markers"))
            || tokens
                .iter()
                .enumerate()
                .any(|(index, token)| tokens[..index].iter().any(|previous| previous == token)))
    {
        return Err(WorkerError::InvalidPayload(
            "provider SVG paint-order value is unsupported".to_owned(),
        ));
    }
    Ok(())
}

fn validate_static_svg_attributes(
    element: &str,
    attrs: &[(String, String)],
    is_root: bool,
    hidden: &mut bool,
    context: &mut StaticSvgValidationContext<'_>,
) -> WorkerResult<(Option<String>, Option<String>)> {
    let mut id = None;
    let mut href_target = None;
    for (key, value) in attrs {
        let lower_key = key.to_ascii_lowercase();
        if lower_key.starts_with("on") || lower_key == "src" || lower_key == "cursor" {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG active attribute {key} is not allowed"
            )));
        }
        if !key.starts_with("xmlns") && static_svg_obvious_external_value(value) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} contains an external or active value"
            )));
        }
        let null_presentation = value == "null"
            && matches!(
                key.as_str(),
                "stroke-dasharray" | "stroke-linecap" | "stroke-linejoin"
            );
        if !static_svg_attribute_allowed(element, key, *hidden) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is unsupported on <{element}> by the static sanitizer"
            )));
        }
        if !null_presentation {
            validate_attribute_resource_budget(element, key, value, context.sanitizer_budget)?;
        }
        match key.as_str() {
            "id" => {
                validate_static_svg_id(value, "element id")?;
                id = Some(value.clone());
            }
            "display" if value == "none" => *hidden = true,
            "display"
                if !matches!(
                    value.as_str(),
                    "inline" | "block" | "none" | "inherit" | "initial" | "unset"
                ) =>
            {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG display value is unsupported".to_owned(),
                ));
            }
            "visibility"
                if !matches!(
                    value.as_str(),
                    "visible" | "hidden" | "collapse" | "inherit" | "initial" | "unset"
                ) =>
            {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG visibility value is unsupported".to_owned(),
                ));
            }
            "vector-effect" if !matches!(value.as_str(), "none" | "non-scaling-stroke") => {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG vector-effect value is unsupported".to_owned(),
                ));
            }
            "paint-order" => {
                validate_paint_order(value)?;
                validate_static_css(value, context.css_budget, context.references)?;
            }
            "baseProfile" if !is_root || value != "full" => {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG baseProfile must be exact root full metadata".to_owned(),
                ));
            }
            "version" if !is_root || !matches!(value.as_str(), "1.0" | "1.1" | "2.0") => {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG version metadata is unsupported".to_owned(),
                ));
            }
            "xml:space" if !matches!(value.as_str(), "default" | "preserve") => {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG xml:space value is unsupported".to_owned(),
                ));
            }
            "enable-background" if element == "svg" => {
                validate_inert_enable_background(value)?;
            }
            "x" | "y" if is_root && !matches!(value.as_str(), "0" | "0px") => {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG inert root attribute {key} must be zero"
                )));
            }
            "style" => {
                validate_static_css(value, context.css_budget, context.references)?;
                validate_static_css_declarations(
                    value,
                    false,
                    context.typed_references,
                    context.sanitizer_budget,
                )?;
                if value
                    .split(';')
                    .filter_map(|declaration| declaration.split_once(':'))
                    .any(|(name, value)| {
                        name.trim().eq_ignore_ascii_case("display")
                            && value.trim().eq_ignore_ascii_case("none")
                    })
                {
                    *hidden = true;
                }
            }
            "href" | "xlink:href" => {
                if !matches!(element, "use" | "linearGradient" | "radialGradient") {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG attribute {key} is not allowed on <{element}>"
                    )));
                }
                let target = local_href_target(value).ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "provider SVG attribute {key} must be an exact local fragment"
                    ))
                })?;
                validate_static_svg_id(target, "local reference")?;
                context.references.push(target.to_owned());
                if context.references.len() > MAX_SVG_RESOURCE_REFERENCES {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the local resource-reference budget".to_owned(),
                    ));
                }
                if href_target.is_some() {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG <{element}> has ambiguous href attributes"
                    )));
                }
                href_target = Some(target.to_owned());
            }
            "xmlns" => {
                if element != "svg" || value != SVG_NAMESPACE {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG default namespace is not exact".to_owned(),
                    ));
                }
            }
            "xmlns:xlink" => {
                if element != "svg" || value != "http://www.w3.org/1999/xlink" {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG xlink namespace is not exact".to_owned(),
                    ));
                }
            }
            _ if key.starts_with("xmlns:") || (key.contains(':') && key != "xml:space") => {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG namespace attribute {key} is not supported by the static sanitizer"
                )));
            }
            "fill" | "stroke" | "clip-path" | "filter" | "mask" | "marker-start" | "marker-mid"
            | "marker-end" => {
                validate_static_css(value, context.css_budget, context.references)?;
                record_static_css_resource_reference(key, value, context.typed_references)?;
            }
            "stroke-dasharray" if value == "null" => {
                // Some vector editors serialize an absent presentation value as the invalid CSS
                // token `null`. usvg deterministically drops it; charging it as a numeric list
                // would reject an otherwise static drawing before that normalization can occur.
            }
            "offset" if element == "stop" => {
                let percent = value.ends_with('%');
                let offset = number_or_percentage(value, "gradient stop offset")?;
                let maximum = if percent { 100.0 } else { 1.0 };
                if !(0.0..=maximum).contains(&offset) {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG gradient stop offset is outside its range".to_owned(),
                    ));
                }
            }
            "stop-opacity" if element == "stop" => {
                let opacity = one_number(value, "stop-opacity", false)?;
                if !(0.0..=1.0).contains(&opacity) {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG stop-opacity must be between zero and one".to_owned(),
                    ));
                }
            }
            "viewBox" if is_root => {
                parse_viewbox_values(value)?;
            }
            "stdDeviation" if element == "feGaussianBlur" => {
                let values = parse_number_list(value, false, "filter stdDeviation", false)?;
                if values.len() > 2
                    || values
                        .iter()
                        .any(|value| *value < 0.0 || *value > MAX_SVG_FILTER_BLUR_SIGMA)
                {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG filter blur exceeds its budget".to_owned(),
                    ));
                }
            }
            _ => {}
        }
    }
    if element == "use" && href_target.is_none() {
        return Err(WorkerError::InvalidPayload(
            "provider SVG local use requires exactly one href".to_owned(),
        ));
    }
    Ok((id, href_target))
}

#[derive(Clone, Copy, Debug)]
struct StaticSvgExpansionCost {
    elements: usize,
    max_depth: usize,
    path_data_bytes: usize,
    path_commands: usize,
    path_numbers: usize,
}

impl StaticSvgExpansionCost {
    fn nested(mut self) -> WorkerResult<Self> {
        self.max_depth = self.max_depth.checked_add(1).ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG expanded depth overflow".to_owned())
        })?;
        if self.max_depth > MAX_SVG_DEPTH {
            return Err(WorkerError::InvalidPayload(
                "provider SVG expansion exceeds the element nesting budget".to_owned(),
            ));
        }
        Ok(self)
    }

    fn checked_add(self, other: Self) -> WorkerResult<Self> {
        let elements = self.elements.checked_add(other.elements).ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG expansion count overflow".to_owned())
        })?;
        let path_data_bytes = self
            .path_data_bytes
            .checked_add(other.path_data_bytes)
            .ok_or_else(|| {
                WorkerError::InvalidPayload("provider SVG expanded path bytes overflow".to_owned())
            })?;
        let path_commands = self
            .path_commands
            .checked_add(other.path_commands)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "provider SVG expanded path command count overflow".to_owned(),
                )
            })?;
        let path_numbers = self
            .path_numbers
            .checked_add(other.path_numbers)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "provider SVG expanded path number count overflow".to_owned(),
                )
            })?;
        if elements > MAX_SVG_ELEMENTS {
            return Err(WorkerError::InvalidPayload(
                "provider SVG local-use expansion exceeds the element budget".to_owned(),
            ));
        }
        if path_data_bytes > MAX_SVG_PATH_DATA_BYTES
            || path_commands > MAX_SVG_PATH_COMMANDS
            || path_numbers > MAX_SVG_PATH_NUMBERS
        {
            return Err(WorkerError::InvalidPayload(
                "provider SVG local-use expansion exceeds the path-work budget".to_owned(),
            ));
        }
        Ok(Self {
            elements,
            max_depth: self.max_depth.max(other.max_depth),
            path_data_bytes,
            path_commands,
            path_numbers,
        })
    }
}

fn static_svg_expansion_cost(
    node: usize,
    nodes: &[StaticSvgSourceNode],
    ids: &BTreeMap<String, usize>,
    visiting: &mut BTreeSet<usize>,
    memo: &mut BTreeMap<usize, StaticSvgExpansionCost>,
) -> WorkerResult<StaticSvgExpansionCost> {
    if let Some(cost) = memo.get(&node) {
        return Ok(*cost);
    }
    if !visiting.insert(node) {
        return Err(WorkerError::InvalidPayload(
            "provider SVG local-use graph is recursive".to_owned(),
        ));
    }
    let mut cost = StaticSvgExpansionCost {
        elements: 1,
        max_depth: 1,
        path_data_bytes: nodes[node].path_data_bytes,
        path_commands: nodes[node].path_commands,
        path_numbers: nodes[node].path_numbers,
    };
    for child in &nodes[node].children {
        cost = cost.checked_add(
            static_svg_expansion_cost(*child, nodes, ids, visiting, memo)?.nested()?,
        )?;
    }
    if let Some(target) = &nodes[node].href_target {
        let target_node = *ids.get(target).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "provider SVG local use reference #{target} is unresolved"
            ))
        })?;
        if nodes[node].name == "use" {
            if !matches!(
                nodes[target_node].name.as_str(),
                "svg"
                    | "g"
                    | "symbol"
                    | "path"
                    | "rect"
                    | "circle"
                    | "ellipse"
                    | "line"
                    | "polyline"
                    | "polygon"
            ) {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG local use reference #{target} has an unsupported target"
                )));
            }
        } else if !matches!(
            nodes[node].name.as_str(),
            "linearGradient" | "radialGradient"
        ) || !matches!(
            nodes[target_node].name.as_str(),
            "linearGradient" | "radialGradient"
        ) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG local href reference #{target} has an unsupported target"
            )));
        }
        cost = cost.checked_add(
            static_svg_expansion_cost(target_node, nodes, ids, visiting, memo)?.nested()?,
        )?;
    }
    visiting.remove(&node);
    memo.insert(node, cost);
    Ok(cost)
}

fn static_svg_node_contains(root: usize, target: usize, nodes: &[StaticSvgSourceNode]) -> bool {
    root == target
        || nodes[root]
            .children
            .iter()
            .any(|child| static_svg_node_contains(*child, target, nodes))
}

fn visit_static_reference_graph(
    node: usize,
    edges: &[Vec<usize>],
    visiting: &mut BTreeSet<usize>,
    complete: &mut BTreeSet<usize>,
) -> WorkerResult<()> {
    if complete.contains(&node) {
        return Ok(());
    }
    if !visiting.insert(node) {
        return Err(WorkerError::InvalidPayload(
            "provider SVG resource graph is recursive".to_owned(),
        ));
    }
    for target in &edges[node] {
        visit_static_reference_graph(*target, edges, visiting, complete)?;
    }
    visiting.remove(&node);
    complete.insert(node);
    Ok(())
}

fn validate_static_svg_source(input: &str) -> WorkerResult<String> {
    let mut reader = Reader::from_reader(input.as_bytes());
    reader.config_mut().trim_text(false);
    let mut buffer = Vec::new();
    let mut nodes = Vec::<StaticSvgSourceNode>::new();
    let mut frames = Vec::<StaticSvgFrame>::new();
    let mut ids = BTreeMap::<String, usize>::new();
    let mut references = Vec::<String>::new();
    let mut typed_references = Vec::<StaticTypedReference>::new();
    let mut sanitizer_budget = SanitizerBudget::default();
    let mut css_budget = StaticCssBudget::default();
    let mut root = None;
    let mut root_closed = false;
    let mut hidden_ranges = Vec::<StaticHiddenRange>::new();
    let mut resources = 0usize;
    let mut gradient_stops = 0usize;
    let mut filter_primitives = 0usize;
    let mut filter_results = BTreeMap::<String, BTreeSet<String>>::new();
    let mut root_xlink_bound = false;
    let mut xlink_reference_seen = false;

    loop {
        let event_start = reader.buffer_position() as usize;
        match reader.read_event_into(&mut buffer) {
            Ok(event @ (Event::Start(_) | Event::Empty(_))) => {
                let (event, empty) = match event {
                    Event::Start(event) => (event, false),
                    Event::Empty(event) => (event, true),
                    _ => unreachable!("matched start or empty"),
                };
                let name = std::str::from_utf8(event.name().as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
                if frames.len() >= MAX_SVG_DEPTH {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the element nesting budget".to_owned(),
                    ));
                }
                if frames.is_empty() && (root.is_some() || root_closed || name != "svg") {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG must contain exactly one SVG root".to_owned(),
                    ));
                }
                let parent_hidden = frames.last().is_some_and(|frame| frame.hidden);
                let is_root = frames.is_empty();
                let attrs = source_attributes(
                    &reader,
                    &event,
                    source_attribute_limit(&name, is_root),
                    &mut sanitizer_budget,
                )?;
                let mut hidden = parent_hidden;
                let path_data_bytes_before = sanitizer_budget.path_data_bytes;
                let path_commands_before = sanitizer_budget.path_commands;
                let path_numbers_before = sanitizer_budget.path_numbers;
                let typed_references_before = typed_references.len();
                let mut validation_context = StaticSvgValidationContext {
                    sanitizer_budget: &mut sanitizer_budget,
                    css_budget: &mut css_budget,
                    references: &mut references,
                    typed_references: &mut typed_references,
                };
                let (id, href_target) = validate_static_svg_attributes(
                    &name,
                    &attrs,
                    is_root,
                    &mut hidden,
                    &mut validation_context,
                )?;
                if is_root {
                    root_xlink_bound = attrs.iter().any(|(key, value)| {
                        key == "xmlns:xlink" && value == "http://www.w3.org/1999/xlink"
                    });
                    let mut normalized_root = attrs.clone();
                    normalize_root_dimensions(&mut normalized_root)?;
                    svg_dimensions(&normalized_root)?;
                }
                xlink_reference_seen |= attrs.iter().any(|(key, _)| key == "xlink:href");
                if matches!(
                    name.as_str(),
                    "clipPath" | "linearGradient" | "radialGradient" | "filter" | "symbol"
                ) {
                    resources = resources.checked_add(1).ok_or_else(|| {
                        WorkerError::InvalidPayload(
                            "provider SVG resource count overflow".to_owned(),
                        )
                    })?;
                    if resources > MAX_SVG_RESOURCES {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG exceeds the resource budget".to_owned(),
                        ));
                    }
                }
                if name == "stop" {
                    gradient_stops = gradient_stops.checked_add(1).ok_or_else(|| {
                        WorkerError::InvalidPayload(
                            "provider SVG gradient stop count overflow".to_owned(),
                        )
                    })?;
                    if gradient_stops > MAX_SVG_GRADIENT_STOPS {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG exceeds the gradient-stop budget".to_owned(),
                        ));
                    }
                }
                let filter_id = if name == "filter" {
                    id.clone()
                } else {
                    frames.last().and_then(|frame| frame.filter_id.clone())
                };
                index_filter_primitive(
                    &name,
                    &attrs,
                    filter_id.as_deref(),
                    &mut filter_results,
                    &mut filter_primitives,
                )?;
                if !static_svg_element_allowed(&name, hidden) {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG element <{name}> is active or unsupported by the static sanitizer"
                    )));
                }
                if is_root && hidden {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG root display:none is unsupported".to_owned(),
                    ));
                }
                if name == "style"
                    && (!matches!(
                        frames.last().map(|frame| frame.name.as_str()),
                        Some("svg" | "defs")
                    ) || attrs
                        .iter()
                        .any(|(key, value)| key != "type" || value != "text/css"))
                {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG style element has unsupported placement or attributes"
                            .to_owned(),
                    ));
                }
                let node = nodes.len();
                if node >= MAX_SVG_ELEMENTS {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG exceeds the element budget".to_owned(),
                    ));
                }
                if let Some(parent) = frames.last() {
                    nodes[parent.node].children.push(node);
                } else {
                    root = Some(node);
                }
                if let Some(id) = id.as_ref() {
                    if ids.insert(id.clone(), node).is_some() {
                        return Err(WorkerError::InvalidPayload(format!(
                            "provider SVG element id {id} is duplicated"
                        )));
                    }
                }
                nodes.push(StaticSvgSourceNode {
                    name: name.clone(),
                    children: Vec::new(),
                    href_target,
                    path_data_bytes: sanitizer_budget.path_data_bytes - path_data_bytes_before,
                    path_commands: sanitizer_budget.path_commands - path_commands_before,
                    path_numbers: sanitizer_budget.path_numbers - path_numbers_before,
                });
                for reference in &mut typed_references[typed_references_before..] {
                    reference.source = node;
                }
                let starts_hidden_subtree = hidden
                    && !parent_hidden
                    && !is_root
                    && !frames
                        .iter()
                        .any(|frame| matches!(frame.name.as_str(), "defs" | "symbol"));
                if empty {
                    if starts_hidden_subtree {
                        hidden_ranges.push(StaticHiddenRange {
                            start: event_start,
                            end: reader.buffer_position() as usize,
                            node,
                        });
                    }
                    if is_root {
                        root_closed = true;
                    }
                } else {
                    frames.push(StaticSvgFrame {
                        node,
                        name,
                        filter_id,
                        hidden,
                        hidden_root_start: starts_hidden_subtree.then_some(event_start),
                        style_text: String::new(),
                    });
                }
            }
            Ok(Event::End(event)) => {
                let name = std::str::from_utf8(event.name().as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
                let frame = frames.pop().ok_or_else(|| {
                    WorkerError::InvalidPayload("provider SVG has mismatched tags".to_owned())
                })?;
                if frame.name != name {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG has mismatched tags".to_owned(),
                    ));
                }
                if frame.name == "style" {
                    let typed_references_before = typed_references.len();
                    validate_static_css(&frame.style_text, &mut css_budget, &mut references)?;
                    validate_static_css_declarations(
                        &frame.style_text,
                        true,
                        &mut typed_references,
                        &mut sanitizer_budget,
                    )?;
                    for reference in &mut typed_references[typed_references_before..] {
                        reference.source = frame.node;
                    }
                }
                if let Some(start) = frame.hidden_root_start {
                    hidden_ranges.push(StaticHiddenRange {
                        start,
                        end: reader.buffer_position() as usize,
                        node: frame.node,
                    });
                }
                if frames.is_empty() {
                    root_closed = true;
                }
            }
            Ok(Event::Text(text)) => {
                let decoded = text.decode().map_err(|error| {
                    WorkerError::InvalidPayload(format!("provider SVG text is malformed: {error}"))
                })?;
                let Some(frame) = frames.last_mut() else {
                    if !decoded.trim().is_empty() {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG has text outside its root".to_owned(),
                        ));
                    }
                    buffer.clear();
                    continue;
                };
                if frame.name == "style" {
                    frame.style_text.push_str(&decoded);
                } else if !frame.hidden
                    && !matches!(frame.name.as_str(), "title" | "desc" | "metadata")
                    && !decoded.trim().is_empty()
                {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG visible text is not supported by the static sanitizer"
                            .to_owned(),
                    ));
                }
            }
            Ok(Event::CData(text)) => {
                let Some(frame) = frames.last_mut() else {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG contains CDATA outside a stylesheet".to_owned(),
                    ));
                };
                if frame.name != "style" {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG contains CDATA outside a stylesheet".to_owned(),
                    ));
                }
                let decoded = text.decode().map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "provider SVG stylesheet CDATA is malformed: {error}"
                    ))
                })?;
                frame.style_text.push_str(&decoded);
            }
            Ok(Event::Comment(_)) | Ok(Event::Decl(_)) => {}
            Ok(Event::Eof) => break,
            Ok(Event::DocType(_)) | Ok(Event::PI(_)) | Ok(Event::GeneralRef(_)) => {
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
    let root = root
        .ok_or_else(|| WorkerError::InvalidPayload("provider output has no SVG root".to_owned()))?;
    if !frames.is_empty() || !root_closed {
        return Err(WorkerError::InvalidPayload(
            "provider SVG is incomplete".to_owned(),
        ));
    }
    if xlink_reference_seen && !root_xlink_bound {
        return Err(WorkerError::InvalidPayload(
            "provider SVG xlink reference requires an exact root xmlns:xlink binding".to_owned(),
        ));
    }
    if nodes
        .iter()
        .any(|node| node.name == "use" && !node.children.is_empty())
    {
        return Err(WorkerError::InvalidPayload(
            "provider SVG local use element must be empty".to_owned(),
        ));
    }
    let mut referenced_nodes = BTreeSet::new();
    for reference in references {
        let Some(node) = ids.get(&reference).copied() else {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG local resource reference #{reference} is unresolved"
            )));
        };
        referenced_nodes.insert(node);
    }
    let mut reference_edges = nodes
        .iter()
        .map(|node| node.children.clone())
        .collect::<Vec<_>>();
    for (source, node) in nodes.iter().enumerate() {
        if let Some(target) = &node.href_target {
            reference_edges[source].push(*ids.get(target).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "provider SVG local resource reference #{target} is unresolved"
                ))
            })?);
        }
    }
    for reference in typed_references {
        let node = ids.get(&reference.target).copied().ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "provider SVG local resource reference #{} is unresolved",
                reference.target
            ))
        })?;
        let matches_kind = match reference.kind {
            StaticReferenceKind::Paint => {
                matches!(
                    nodes[node].name.as_str(),
                    "linearGradient" | "radialGradient"
                )
            }
            StaticReferenceKind::ClipPath => nodes[node].name == "clipPath",
            StaticReferenceKind::Filter => nodes[node].name == "filter",
        };
        if !matches_kind {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG local resource reference #{} has the wrong resource type",
                reference.target
            )));
        }
        reference_edges[reference.source].push(node);
    }
    visit_static_reference_graph(
        root,
        &reference_edges,
        &mut BTreeSet::new(),
        &mut BTreeSet::new(),
    )?;
    static_svg_expansion_cost(
        root,
        &nodes,
        &ids,
        &mut BTreeSet::new(),
        &mut BTreeMap::new(),
    )?;
    for range in &hidden_ranges {
        if referenced_nodes
            .iter()
            .any(|target| static_svg_node_contains(range.node, *target, &nodes))
        {
            return Err(WorkerError::InvalidPayload(
                "provider SVG references a display:none subtree that cannot be canonicalized safely"
                    .to_owned(),
            ));
        }
    }
    if hidden_ranges.is_empty() {
        return Ok(input.to_owned());
    }
    hidden_ranges.sort_by_key(|range| range.start);
    let mut filtered = String::with_capacity(input.len());
    let mut cursor = 0usize;
    for range in hidden_ranges {
        if range.start < cursor || range.end > input.len() || range.start > range.end {
            return Err(WorkerError::InvalidPayload(
                "provider SVG hidden-subtree bounds are malformed".to_owned(),
            ));
        }
        filtered.push_str(&input[cursor..range.start]);
        cursor = range.end;
    }
    filtered.push_str(&input[cursor..]);
    Ok(filtered)
}

fn static_svg_options() -> usvg::Options<'static> {
    usvg::Options {
        image_href_resolver: usvg::ImageHrefResolver {
            resolve_data: Box::new(|_, _, _| None),
            resolve_string: Box::new(|_, _| None),
        },
        ..usvg::Options::default()
    }
}

fn validate_static_tree_size(tree: &usvg::Tree) -> WorkerResult<()> {
    fn visit(group: &usvg::Group, depth: usize, count: &mut usize) -> WorkerResult<()> {
        if depth > MAX_SVG_DEPTH {
            return Err(WorkerError::InvalidPayload(
                "provider SVG parsed expansion exceeds the nesting budget".to_owned(),
            ));
        }
        for node in group.children() {
            *count = count.checked_add(1).ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "provider SVG parsed expansion count overflow".to_owned(),
                )
            })?;
            if *count > MAX_SVG_ELEMENTS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG parsed expansion exceeds the element budget".to_owned(),
                ));
            }
            if matches!(node, usvg::Node::Image(_) | usvg::Node::Text(_)) {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG parsed to unsupported image or text content".to_owned(),
                ));
            }
            if let usvg::Node::Group(child) = node {
                visit(child, depth + 1, count)?;
            }
            let mut subroot_error = None;
            node.subroots(|subroot| {
                if subroot_error.is_none() {
                    subroot_error = visit(subroot, depth + 1, count).err();
                }
            });
            if let Some(error) = subroot_error {
                return Err(error);
            }
        }
        Ok(())
    }
    let mut count = 0;
    visit(tree.root(), 1, &mut count)
}

pub(super) fn sanitize_static_svg_bytes(input: &[u8]) -> WorkerResult<CanonicalSvg> {
    if input.len() > MAX_SVG_BYTES {
        return Err(WorkerError::InvalidPayload(
            "provider SVG exceeds the 256 KiB sanitizer budget".to_owned(),
        ));
    }
    let input = std::str::from_utf8(input)
        .map_err(|_| WorkerError::InvalidPayload("provider SVG is not valid UTF-8".to_owned()))?;
    let input = validate_static_svg_source(input)?;
    let options = static_svg_options();
    let tree = usvg::Tree::from_str(&input, &options).map_err(|error| {
        WorkerError::InvalidPayload(format!("provider SVG is not supported static SVG: {error}"))
    })?;
    validate_static_tree_size(&tree)?;
    let intrinsic = tree.size();
    let width = intrinsic.width().ceil() as u32;
    let height = intrinsic.height().ceil() as u32;
    if width == 0 || height == 0 || width > MAX_PREVIEW_DIMENSION || height > MAX_PREVIEW_DIMENSION
    {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG dimensions must be 1..={MAX_PREVIEW_DIMENSION}"
        )));
    }
    validate_filter_render_budget(
        &tree,
        width,
        height,
        resvg::tiny_skia::Transform::identity(),
    )?;
    // Preserve the parsed f32 geometry through the canonical SVG round trip. The writer default
    // of eight decimal places can shift a single antialiased channel at the 512px oracle size.
    let write_options = usvg::WriteOptions {
        indent: usvg::Indent::None,
        attributes_indent: usvg::Indent::None,
        coordinates_precision: 9,
        transforms_precision: 9,
        ..usvg::WriteOptions::default()
    };
    let body = tree.to_string(&write_options);
    if body.len() > MAX_SVG_BYTES {
        return Err(WorkerError::InvalidPayload(
            "provider SVG expansion exceeds the 256 KiB sanitizer budget".to_owned(),
        ));
    }
    Ok(CanonicalSvg {
        svg: format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>{body}"),
        width,
        height,
    })
}

pub(super) fn sanitize_svg_bytes(input: &[u8]) -> WorkerResult<CanonicalSvg> {
    match sanitize_inert_svg_bytes(input) {
        Ok(canonical) => Ok(canonical),
        // Static SVG accepts safe stylesheets that the narrow profile rejects. Report its
        // actual failure first, retaining any distinct narrow-profile resource-limit diagnosis.
        Err(inert_error) => sanitize_static_svg_bytes(input).map_err(|static_error| {
            match (static_error, inert_error) {
                (
                    WorkerError::InvalidPayload(detail),
                    WorkerError::InvalidPayload(inert_detail),
                ) if detail != inert_detail => WorkerError::InvalidPayload(format!(
                    "{detail} (narrow SVG profile also rejected: {inert_detail})"
                )),
                (error, _) => error,
            }
        }),
    }
}

fn sanitize_inert_svg_bytes(input: &[u8]) -> WorkerResult<CanonicalSvg> {
    if input.len() > MAX_SVG_BYTES {
        return Err(WorkerError::InvalidPayload(
            "provider SVG exceeds the 256 KiB sanitizer budget".to_owned(),
        ));
    }
    let input = std::str::from_utf8(input)
        .map_err(|_| WorkerError::InvalidPayload("provider SVG is not valid UTF-8".to_owned()))?;
    let resources = index_svg_resources(input)?;
    let mut reader = Reader::from_reader(input.as_bytes());
    reader.config_mut().trim_text(true);
    let mut buffer = Vec::new();
    let mut output = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    let mut stack: Vec<RawSvgElement> = Vec::new();
    let mut elements = 0usize;
    let mut emitted_elements = 0usize;
    let mut budget = SanitizerBudget::default();
    let mut dimensions = None;
    let mut root_seen = false;
    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Start(event)) => {
                let event_name = event.name();
                let raw_name = std::str::from_utf8(event_name.as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
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
                let is_root = stack.is_empty();
                let source_attrs = source_attributes(
                    &reader,
                    &event,
                    source_attribute_limit(&raw_name, is_root),
                    &mut budget,
                )?;
                let kind =
                    classify_element(&raw_name, stack.last(), &source_attrs, false, root_seen)?;
                if kind.emitted() {
                    let (name, mut attrs) = canonical_emitted_element(
                        kind,
                        &raw_name,
                        source_attrs,
                        is_root,
                        &resources,
                        &mut budget,
                    )?;
                    if is_root {
                        root_seen = true;
                        if !attrs
                            .iter()
                            .any(|(key, value)| key == "xmlns" && value == SVG_NAMESPACE)
                        {
                            attrs.push(("xmlns".to_owned(), SVG_NAMESPACE.to_owned()));
                            attrs.sort_unstable();
                        }
                        if attrs.len() > MAX_SVG_ATTRIBUTES_PER_ELEMENT {
                            return Err(WorkerError::InvalidPayload(
                                "provider SVG element <svg> exceeds the per-element attribute budget"
                                    .to_owned(),
                            ));
                        }
                        dimensions = Some(svg_dimensions(&attrs)?);
                    }
                    write_start(&mut output, &name, &attrs, false);
                    emitted_elements += 1;
                    record_canonical_output(&output, emitted_elements)?;
                } else if kind == RawSvgElementKind::Hidden {
                    validate_hidden_attributes(&raw_name, &source_attrs, &mut budget)?;
                } else if kind == RawSvgElementKind::UsePathDefinition {
                    validate_indexed_use_path_definition(&source_attrs, &resources)?;
                } else {
                    validate_discarded_attributes(kind, &source_attrs)?;
                }
                stack.push(RawSvgElement {
                    name: raw_name,
                    kind,
                });
            }
            Ok(Event::Empty(event)) => {
                let event_name = event.name();
                let raw_name = std::str::from_utf8(event_name.as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
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
                let source_attrs = source_attributes(
                    &reader,
                    &event,
                    source_attribute_limit(&raw_name, false),
                    &mut budget,
                )?;
                let kind =
                    classify_element(&raw_name, stack.last(), &source_attrs, true, root_seen)?;
                if kind == RawSvgElementKind::Use {
                    if stack.len().saturating_add(2) > MAX_SVG_DEPTH {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG local use expansion exceeds the element nesting budget"
                                .to_owned(),
                        ));
                    }
                    let expanded = canonical_local_use(source_attrs, &resources, &mut budget)?;
                    write_start(&mut output, "g", &expanded.wrapper_attrs, false);
                    write_start(&mut output, "path", &expanded.path_attrs, true);
                    output.push_str("</g>");
                    emitted_elements += 2;
                    record_canonical_output(&output, emitted_elements)?;
                } else if kind.emitted() {
                    let (name, attrs) = canonical_emitted_element(
                        kind,
                        &raw_name,
                        source_attrs,
                        false,
                        &resources,
                        &mut budget,
                    )?;
                    write_start(&mut output, &name, &attrs, true);
                    emitted_elements += 1;
                    record_canonical_output(&output, emitted_elements)?;
                } else if kind == RawSvgElementKind::Hidden {
                    validate_hidden_attributes(&raw_name, &source_attrs, &mut budget)?;
                } else if kind == RawSvgElementKind::UsePathDefinition {
                    validate_indexed_use_path_definition(&source_attrs, &resources)?;
                } else {
                    validate_discarded_attributes(kind, &source_attrs)?;
                }
            }
            Ok(Event::End(event)) => {
                let name = std::str::from_utf8(event.name().as_ref())
                    .map_err(|_| {
                        WorkerError::InvalidPayload("provider SVG tag is not UTF-8".to_owned())
                    })?
                    .to_owned();
                let Some(open) = stack.pop() else {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG has mismatched tags".to_owned(),
                    ));
                };
                if open.name != name {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG has mismatched tags".to_owned(),
                    ));
                }
                if open.kind.emitted() {
                    output.push_str("</");
                    output.push_str(&name);
                    output.push('>');
                    record_canonical_output(&output, emitted_elements)?;
                }
            }
            Ok(Event::Text(text)) => {
                let bytes: &[u8] = text.as_ref();
                let current = stack.last().map(|element| element.kind);
                let dc_format = current == Some(RawSvgElementKind::DcFormat);
                if dc_format {
                    if bytes != b"image/svg+xml" {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG dc:format metadata is not recognized".to_owned(),
                        ));
                    }
                } else if current == Some(RawSvgElementKind::Metadata) {
                    // Exporters such as potrace place inert provenance text directly in the
                    // metadata container. The entire container is omitted from canonical output;
                    // its children and attributes still pass the strict metadata classifiers.
                } else if matches!(
                    current,
                    Some(RawSvgElementKind::Title | RawSvgElementKind::DcTitle)
                ) {
                    // Plain title text is inert metadata and is deliberately omitted.
                } else if !bytes.iter().all(u8::is_ascii_whitespace) {
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
    let tree = usvg::Tree::from_str(&output, &usvg::Options::default()).map_err(|error| {
        WorkerError::InvalidPayload(format!("provider SVG cannot be rendered: {error}"))
    })?;
    let (width, height) = dimensions.expect("validated above");
    validate_filter_render_budget(
        &tree,
        width,
        height,
        resvg::tiny_skia::Transform::identity(),
    )?;
    Ok(CanonicalSvg {
        svg: output,
        width,
        height,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RawSvgElementKind {
    Retained,
    Defs,
    ResourceDefs,
    UsePathDefinition,
    Use,
    ClipPath,
    LinearGradient,
    GradientStop,
    Filter,
    FeGaussianBlur,
    FeOffset,
    FeMerge,
    FeMergeNode,
    Hidden,
    NamedView,
    Grid,
    Metadata,
    RdfLower,
    RdfUpper,
    CcLower,
    CcUpper,
    DcFormat,
    DcType,
    DcTitle,
    Title,
}

impl RawSvgElementKind {
    fn emitted(self) -> bool {
        matches!(
            self,
            Self::Retained
                | Self::ResourceDefs
                | Self::ClipPath
                | Self::LinearGradient
                | Self::GradientStop
                | Self::Filter
                | Self::FeGaussianBlur
                | Self::FeOffset
                | Self::FeMerge
                | Self::FeMergeNode
        )
    }
}

#[derive(Debug)]
struct RawSvgElement {
    name: String,
    kind: RawSvgElementKind,
}

fn source_attribute_limit(name: &str, is_root: bool) -> usize {
    // Some editor-authored paths carry inert source metadata in addition to the visible SVG
    // attributes. The sanitizer strips that metadata before enforcing the unchanged canonical
    // per-element limit below.
    if (is_root && name == "svg") || matches!(name, "path" | "sodipodi:namedview") {
        24
    } else {
        MAX_SVG_ATTRIBUTES_PER_ELEMENT
    }
}

fn source_attributes(
    reader: &Reader<&[u8]>,
    event: &quick_xml::events::BytesStart<'_>,
    per_element_limit: usize,
    budget: &mut SanitizerBudget,
) -> WorkerResult<Vec<(String, String)>> {
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
        if value.len() > MAX_SVG_ATTRIBUTE_VALUE_BYTES {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} exceeds the per-value byte budget"
            )));
        }
        budget.attributes = budget.attributes.checked_add(1).ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG attribute count overflow".to_owned())
        })?;
        budget.attribute_value_bytes = budget
            .attribute_value_bytes
            .checked_add(value.len())
            .ok_or_else(|| {
                WorkerError::InvalidPayload("provider SVG attribute bytes overflow".to_owned())
            })?;
        if budget.attributes > MAX_SVG_ATTRIBUTES {
            return Err(WorkerError::InvalidPayload(
                "provider SVG exceeds the total attribute budget".to_owned(),
            ));
        }
        if budget.attribute_value_bytes > MAX_SVG_ATTRIBUTE_VALUE_BYTES_TOTAL {
            return Err(WorkerError::InvalidPayload(
                "provider SVG exceeds the total attribute-value byte budget".to_owned(),
            ));
        }
        if attrs.iter().any(|(existing, _)| existing == &key) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is duplicated"
            )));
        }
        attrs.push((key, value));
    }
    if attrs.len() > per_element_limit {
        return Err(WorkerError::InvalidPayload(
            "provider SVG element exceeds the per-element attribute budget".to_owned(),
        ));
    }
    Ok(attrs)
}

fn one_number(value: &str, label: &str, allow_px: bool) -> WorkerResult<f64> {
    let values = parse_number_list(value, allow_px, label, true)?;
    if values.len() != 1 {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} must contain exactly one finite number"
        )));
    }
    Ok(values[0])
}

fn number_or_percentage(value: &str, label: &str) -> WorkerResult<f64> {
    if let Some(number) = value.strip_suffix('%') {
        one_number(number, label, false)
    } else {
        one_number(value, label, false)
    }
}

fn canonical_stop_style(value: &str) -> WorkerResult<Vec<(String, String)>> {
    let mut attrs = Vec::new();
    let declarations = value.split(';').collect::<Vec<_>>();
    for (index, declaration) in declarations.iter().enumerate() {
        if declaration.trim().is_empty() {
            if index + 1 == declarations.len() && value.ends_with(';') {
                continue;
            }
            return Err(WorkerError::InvalidPayload(
                "provider SVG gradient stop style is malformed".to_owned(),
            ));
        }
        let (key, value) = declaration.split_once(':').ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG gradient stop style is malformed".to_owned())
        })?;
        let key = key.trim();
        let value = value.trim();
        if !matches!(key, "stop-color" | "stop-opacity")
            || value.is_empty()
            || value.contains(':')
            || unsafe_attribute_value(value)
            || attrs.iter().any(|(existing, _)| existing == key)
        {
            return Err(WorkerError::InvalidPayload(
                "provider SVG gradient stop style is not allowed".to_owned(),
            ));
        }
        if key == "stop-opacity" {
            let opacity = one_number(value, "stop-opacity", false)?;
            if !(0.0..=1.0).contains(&opacity) {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG stop-opacity must be between zero and one".to_owned(),
                ));
            }
        }
        attrs.push((key.to_owned(), value.to_owned()));
    }
    if attrs.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "provider SVG gradient stop style is empty".to_owned(),
        ));
    }
    Ok(attrs)
}

fn canonical_resource_element(
    kind: RawSvgElementKind,
    name: &str,
    source_attrs: Vec<(String, String)>,
    resources: &SvgResourceCatalog,
    budget: &mut SanitizerBudget,
) -> WorkerResult<(String, Vec<(String, String)>)> {
    let mut attrs = Vec::new();
    let expected_id_kind = match kind {
        RawSvgElementKind::ClipPath => Some(SvgResourceKind::ClipPath),
        RawSvgElementKind::LinearGradient => Some(SvgResourceKind::LinearGradient),
        RawSvgElementKind::Filter => Some(SvgResourceKind::Filter),
        _ => None,
    };
    for (key, value) in source_attrs {
        if unsafe_attribute_value(&value)
            && !(kind == RawSvgElementKind::GradientStop && key == "style")
            && !(matches!(key.as_str(), "href" | "xlink:href")
                && local_href_target(&value).is_some())
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG resource attribute {key} is not allowed"
            )));
        }
        match kind {
            RawSvgElementKind::ResourceDefs | RawSvgElementKind::FeMerge => {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG element <{name}> does not allow attributes"
                )));
            }
            RawSvgElementKind::ClipPath => match key.as_str() {
                "id" => attrs.push((key, value)),
                "clipPathUnits"
                    if matches!(value.as_str(), "userSpaceOnUse" | "objectBoundingBox") =>
                {
                    attrs.push((key, value));
                }
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG clipPath attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::LinearGradient => match key.as_str() {
                "id" => attrs.push((key, value)),
                "x1" | "y1" | "x2" | "y2" => {
                    number_or_percentage(&value, &key)?;
                    attrs.push((key, value));
                }
                "gradientUnits"
                    if matches!(value.as_str(), "userSpaceOnUse" | "objectBoundingBox") =>
                {
                    attrs.push((key, value));
                }
                "gradientTransform" => {
                    let numbers = validate_transform_list(&value)?;
                    budget.transform_numbers = budget
                        .transform_numbers
                        .checked_add(numbers)
                        .ok_or_else(|| {
                            WorkerError::InvalidPayload(
                                "provider SVG transform number overflow".to_owned(),
                            )
                        })?;
                    if budget.transform_numbers > MAX_SVG_TRANSFORM_NUMBERS {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG exceeds the transform-number budget".to_owned(),
                        ));
                    }
                    attrs.push((key, value));
                }
                "href" | "xlink:href" => {
                    let target = local_href_target(&value).ok_or_else(|| {
                        WorkerError::InvalidPayload(
                            "provider SVG gradient href must be a local fragment".to_owned(),
                        )
                    })?;
                    if resources.ids.get(target) != Some(&SvgResourceKind::LinearGradient) {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG gradient href has the wrong resource type".to_owned(),
                        ));
                    }
                    attrs.push(("href".to_owned(), value));
                }
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG linearGradient attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::GradientStop => match key.as_str() {
                "offset" => {
                    let percent = value.ends_with('%');
                    let offset = number_or_percentage(&value, "gradient stop offset")?;
                    let maximum = if percent { 100.0 } else { 1.0 };
                    if !(0.0..=maximum).contains(&offset) {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG gradient stop offset is outside its range".to_owned(),
                        ));
                    }
                    attrs.push((key, value));
                }
                "stop-color" => attrs.push((key, value)),
                "stop-opacity" => {
                    let opacity = one_number(&value, "stop-opacity", false)?;
                    if !(0.0..=1.0).contains(&opacity) {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG stop-opacity must be between zero and one".to_owned(),
                        ));
                    }
                    attrs.push((key, value));
                }
                "style" => {
                    for style_attr in canonical_stop_style(&value)? {
                        if attrs.iter().any(|(existing, _)| existing == &style_attr.0) {
                            return Err(WorkerError::InvalidPayload(
                                "provider SVG gradient stop style duplicates an attribute"
                                    .to_owned(),
                            ));
                        }
                        attrs.push(style_attr);
                    }
                }
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG gradient stop attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::Filter => match key.as_str() {
                "id" | "width" | "height" => attrs.push((key, value)),
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG filter attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::FeGaussianBlur => match key.as_str() {
                "in" | "result" => attrs.push((key, value)),
                "stdDeviation" => {
                    let values = parse_number_list(&value, false, "filter blur", true)?;
                    if values.len() > 2
                        || values
                            .iter()
                            .any(|value| *value < 0.0 || *value > MAX_SVG_FILTER_BLUR_SIGMA)
                    {
                        return Err(WorkerError::InvalidPayload(
                            "provider SVG filter blur exceeds its bounded range".to_owned(),
                        ));
                    }
                    attrs.push((key, value));
                }
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG feGaussianBlur attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::FeOffset => match key.as_str() {
                "in" | "result" => attrs.push((key, value)),
                "dx" | "dy" => {
                    one_number(&value, &key, false)?;
                    attrs.push((key, value));
                }
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG feOffset attribute {key} is not allowed"
                    )))
                }
            },
            RawSvgElementKind::FeMergeNode => match key.as_str() {
                "in" => attrs.push((key, value)),
                _ => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG feMergeNode attribute {key} is not allowed"
                    )))
                }
            },
            _ => unreachable!("resource canonicalizer called for {kind:?}"),
        }
    }
    if let Some(expected) = expected_id_kind {
        let id = attribute_value(&attrs, "id").ok_or_else(|| {
            WorkerError::InvalidPayload(format!("provider SVG resource <{name}> requires an id"))
        })?;
        if resources.ids.get(id) != Some(&expected) {
            return Err(WorkerError::InvalidPayload(
                "provider SVG resource id differs from the validated index".to_owned(),
            ));
        }
    }
    if kind == RawSvgElementKind::Filter {
        let width = attribute_value(&attrs, "width").ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG filter requires width".to_owned())
        })?;
        let height = attribute_value(&attrs, "height").ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG filter requires height".to_owned())
        })?;
        bounded_filter_axis(width, resources.width, "width")?;
        bounded_filter_axis(height, resources.height, "height")?;
    }
    let mut keys = BTreeSet::new();
    if attrs.iter().any(|(key, _)| !keys.insert(key.as_str())) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> has duplicate canonical attributes"
        )));
    }
    attrs.sort_unstable();
    Ok((name.to_owned(), attrs))
}

fn bounded_filter_axis(value: &str, viewport: u32, label: &str) -> WorkerResult<u32> {
    let pixels = if let Some(number) = value.strip_suffix('%') {
        let percentage = one_number(number, label, false)?;
        if !(0.0..=200.0).contains(&percentage) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG filter {label} exceeds 200%"
            )));
        }
        f64::from(viewport) * percentage / 100.0
    } else {
        one_number(value, label, false)?
    };
    if pixels <= 0.0 || pixels > f64::from(MAX_PREVIEW_DIMENSION) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG filter {label} exceeds the renderer dimension budget"
        )));
    }
    Ok(pixels.ceil() as u32)
}

fn canonical_emitted_element(
    kind: RawSvgElementKind,
    name: &str,
    attrs: Vec<(String, String)>,
    is_root: bool,
    resources: &SvgResourceCatalog,
    budget: &mut SanitizerBudget,
) -> WorkerResult<(String, Vec<(String, String)>)> {
    if kind == RawSvgElementKind::Retained {
        canonical_element(name, attrs, is_root, resources, budget)
    } else {
        canonical_resource_element(kind, name, attrs, resources, budget)
    }
}

fn validate_hidden_attributes(
    element: &str,
    attrs: &[(String, String)],
    budget: &mut SanitizerBudget,
) -> WorkerResult<()> {
    for (key, value) in attrs {
        let inert = matches!(key.as_str(), "id" | "class" | "data-name");
        let hidden_group = element == "g"
            && matches!(
                key.as_str(),
                "display" | "overflow" | "x" | "y" | "width" | "height"
            );
        if !inert && !hidden_group && !allowed_attribute(element, key) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG hidden attribute {key} is not allowed"
            )));
        }
        let local_paint = key == "fill" && local_url_target(value).is_some();
        if !local_paint && unsafe_attribute_value(value) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG hidden attribute {key} is not allowed"
            )));
        }
        match key.as_str() {
            "display" if value != "none" => {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG hidden display must be none".to_owned(),
                ))
            }
            "overflow" if !matches!(value.as_str(), "visible" | "hidden") => {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG hidden overflow is not recognized".to_owned(),
                ))
            }
            "width" | "height" | "x" | "y" if value.ends_with('%') => {
                let percentage = number_or_percentage(value, key)?;
                if percentage.abs() > 100.0 {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG hidden percentage exceeds its budget".to_owned(),
                    ));
                }
            }
            _ if !inert && !hidden_group && !local_paint => {
                validate_attribute_resource_budget(element, key, value, budget)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn canonical_element(
    name: &str,
    mut source_attrs: Vec<(String, String)>,
    is_root: bool,
    resources: &SvgResourceCatalog,
    budget: &mut SanitizerBudget,
) -> WorkerResult<(String, Vec<(String, String)>)> {
    if !matches!(
        name,
        "svg" | "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
    ) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> is not allowed"
        )));
    }
    if is_root {
        normalize_root_dimensions(&mut source_attrs)?;
    }
    let mut attrs = Vec::new();
    for (key, value) in source_attrs {
        if stripped_retained_attribute(name, &key, &value, is_root)? {
            continue;
        }
        if key == "style" {
            let style_attrs = canonical_presentation_style(name, &value, resources, budget)?;
            if style_attrs
                .iter()
                .any(|(key, _)| attrs.iter().any(|(existing, _)| existing == key))
            {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG style duplicates an attribute".to_owned(),
                ));
            }
            attrs.extend(style_attrs);
            continue;
        }
        let reference_kind = match key.as_str() {
            "fill" if local_url_target(&value).is_some() => Some(SvgResourceKind::LinearGradient),
            "clip-path" => Some(SvgResourceKind::ClipPath),
            "filter" => Some(SvgResourceKind::Filter),
            _ => None,
        };
        if let Some(expected) = reference_kind {
            if matches!(key.as_str(), "clip-path" | "filter")
                && !matches!(
                    name,
                    "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
                )
            {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG attribute {key} is not allowed on <{name}>"
                )));
            }
            let target = local_url_target(&value).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "provider SVG attribute {key} must be an exact local resource reference"
                ))
            })?;
            if resources.ids.get(target) != Some(&expected) {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG attribute {key} has an unresolved or mistyped resource"
                )));
            }
            attrs.push((key, value));
            continue;
        }
        // The default namespace changes the meaning of unprefixed elements, so retain only SVG's
        // namespace on the root. It is syntax, never a dereferenced resource.
        let namespace_ok = key == "xmlns" && is_root && name == "svg" && value == SVG_NAMESPACE;
        if !allowed_attribute(name, &key)
            || (key == "xmlns" && !namespace_ok)
            || (!namespace_ok && unsafe_attribute_value(&value))
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is not allowed"
            )));
        }
        validate_attribute_resource_budget(name, &key, &value, budget)?;
        if attrs.iter().any(|(existing, _)| existing == &key) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is duplicated"
            )));
        }
        attrs.push((key, value));
    }
    if attrs.len() > MAX_SVG_ATTRIBUTES_PER_ELEMENT {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> exceeds the per-element attribute budget"
        )));
    }
    attrs.sort_unstable();
    Ok((name.to_owned(), attrs))
}

fn classify_element(
    name: &str,
    parent: Option<&RawSvgElement>,
    attrs: &[(String, String)],
    empty: bool,
    root_seen: bool,
) -> WorkerResult<RawSvgElementKind> {
    let invalid = || {
        WorkerError::InvalidPayload(format!(
            "provider SVG element <{name}> is not allowed in this context"
        ))
    };
    let Some(parent) = parent else {
        return if !root_seen && !empty && name == "svg" {
            Ok(RawSvgElementKind::Retained)
        } else {
            Err(WorkerError::InvalidPayload(
                "provider output root must be a non-empty <svg> element".to_owned(),
            ))
        };
    };
    if parent.kind == RawSvgElementKind::Hidden {
        return if matches!(
            name,
            "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
        ) {
            Ok(RawSvgElementKind::Hidden)
        } else if name == "title" {
            Ok(RawSvgElementKind::Title)
        } else {
            Err(invalid())
        };
    }
    if parent.kind == RawSvgElementKind::Retained
        && matches!(
            name,
            "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon"
        )
    {
        if name == "g" && attribute_value(attrs, "display") == Some("none") {
            return Ok(RawSvgElementKind::Hidden);
        }
        return Ok(RawSvgElementKind::Retained);
    }
    match (parent.kind, parent.name.as_str(), name, empty) {
        (RawSvgElementKind::Retained, "svg", "defs", true) => Ok(RawSvgElementKind::Defs),
        (RawSvgElementKind::Retained, "svg", "defs", false) => Ok(RawSvgElementKind::ResourceDefs),
        (RawSvgElementKind::ResourceDefs, "defs", "path", true) => {
            Ok(RawSvgElementKind::UsePathDefinition)
        }
        (RawSvgElementKind::Retained, _, "use", true) => Ok(RawSvgElementKind::Use),
        (RawSvgElementKind::ResourceDefs, "defs", "clipPath", false) => {
            Ok(RawSvgElementKind::ClipPath)
        }
        (RawSvgElementKind::ResourceDefs, "defs", "linearGradient", _) => {
            Ok(RawSvgElementKind::LinearGradient)
        }
        (RawSvgElementKind::ResourceDefs, "defs", "filter", false) => Ok(RawSvgElementKind::Filter),
        (
            RawSvgElementKind::ClipPath,
            "clipPath",
            "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon",
            _,
        ) => Ok(RawSvgElementKind::Retained),
        (RawSvgElementKind::LinearGradient, "linearGradient", "stop", true) => {
            Ok(RawSvgElementKind::GradientStop)
        }
        (RawSvgElementKind::Filter, "filter", "feGaussianBlur", true) => {
            Ok(RawSvgElementKind::FeGaussianBlur)
        }
        (RawSvgElementKind::Filter, "filter", "feOffset", true) => Ok(RawSvgElementKind::FeOffset),
        (RawSvgElementKind::Filter, "filter", "feMerge", false) => Ok(RawSvgElementKind::FeMerge),
        (RawSvgElementKind::FeMerge, "feMerge", "feMergeNode", true) => {
            Ok(RawSvgElementKind::FeMergeNode)
        }
        (RawSvgElementKind::Retained, "svg", "sodipodi:namedview", _) => {
            Ok(RawSvgElementKind::NamedView)
        }
        (RawSvgElementKind::Retained, _, "title", _) => Ok(RawSvgElementKind::Title),
        (RawSvgElementKind::NamedView, _, "inkscape:grid", true) => Ok(RawSvgElementKind::Grid),
        (RawSvgElementKind::Retained, "svg", "metadata", false) => Ok(RawSvgElementKind::Metadata),
        (RawSvgElementKind::Metadata, _, "rdf:rdf", false) => Ok(RawSvgElementKind::RdfLower),
        (RawSvgElementKind::Metadata, _, "rdf:RDF", false) => Ok(RawSvgElementKind::RdfUpper),
        (RawSvgElementKind::RdfLower, _, "cc:work", false) => Ok(RawSvgElementKind::CcLower),
        (RawSvgElementKind::RdfUpper, _, "cc:Work", false) => Ok(RawSvgElementKind::CcUpper),
        (RawSvgElementKind::CcLower | RawSvgElementKind::CcUpper, _, "dc:format", false) => {
            Ok(RawSvgElementKind::DcFormat)
        }
        (RawSvgElementKind::CcLower | RawSvgElementKind::CcUpper, _, "dc:type", true) => {
            Ok(RawSvgElementKind::DcType)
        }
        (RawSvgElementKind::CcLower | RawSvgElementKind::CcUpper, _, "dc:title", _) => {
            Ok(RawSvgElementKind::DcTitle)
        }
        _ => Err(invalid()),
    }
}

fn stripped_retained_attribute(
    element: &str,
    key: &str,
    value: &str,
    is_root: bool,
) -> WorkerResult<bool> {
    if key == "xmlns:cc" {
        if is_root
            && element == "svg"
            && matches!(
                value,
                "http://creativecommons.org/ns#" | "http://web.resource.org/cc/"
            )
        {
            return Ok(true);
        }
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG attribute {key} is not allowed"
        )));
    }
    let known_namespace = match key {
        "xmlns:dc" => Some("http://purl.org/dc/elements/1.1/"),
        "xmlns:rdf" => Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
        "xmlns:svg" => Some(SVG_NAMESPACE),
        "xmlns:sodipodi" => Some("http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"),
        "xmlns:inkscape" => Some("http://www.inkscape.org/namespaces/inkscape"),
        "xmlns:xlink" => Some("http://www.w3.org/1999/xlink"),
        "xmlns:ev" => Some("http://www.w3.org/2001/xml-events"),
        "xmlns:ns1" => Some("http://sozi.baierouge.fr"),
        _ => None,
    };
    if let Some(expected) = known_namespace {
        if !is_root || element != "svg" || value != expected {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} is not allowed"
            )));
        }
        return Ok(true);
    }
    if is_root && element == "svg" {
        match key {
            "x" | "y" => {
                let values = parse_number_list(value, true, key, true)?;
                if values.as_slice() != [0.0] {
                    return Err(WorkerError::InvalidPayload(format!(
                        "provider SVG outer {key} must be zero"
                    )));
                }
                return Ok(true);
            }
            "xml:space" if matches!(value, "default" | "preserve") => return Ok(true),
            "baseProfile" if value == "full" => return Ok(true),
            "enable-background" => {
                validate_inert_enable_background(value)?;
                return Ok(true);
            }
            // This editor path is never emitted or dereferenced. Attribute byte budgets still
            // bound it, including legacy Windows paths containing backslashes.
            "sodipodi:docbase" => return Ok(true),
            _ => {}
        }
    }
    let stripped = match element {
        "svg" if is_root => matches!(
            key,
            "id" | "version"
                | "inkscape:version"
                | "sodipodi:docname"
                | "sodipodi:version"
                | "class"
                | "data-z"
                | "data-slots"
                | "data-tags"
        ),
        "g" => matches!(
            key,
            "inkscape:label" | "inkscape:groupmode" | "id" | "class" | "data-name"
        ),
        "path" => matches!(
            key,
            "id" | "class"
                | "data-name"
                | "inkscape:connector-curvature"
                | "inkscape:flatsided"
                | "inkscape:randomized"
                | "inkscape:rounded"
                | "inkscape:transform-center-y"
                | "sodipodi:arg1"
                | "sodipodi:arg2"
                | "sodipodi:cx"
                | "sodipodi:cy"
                | "sodipodi:nodetypes"
                | "sodipodi:r1"
                | "sodipodi:r2"
                | "sodipodi:sides"
                | "sodipodi:type"
        ),
        "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon" => {
            matches!(key, "id" | "class" | "data-name")
        }
        _ => false,
    };
    if stripped && unsafe_attribute_value(value) {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG attribute {key} is not allowed"
        )));
    }
    Ok(stripped)
}

fn validate_discarded_attributes(
    kind: RawSvgElementKind,
    attrs: &[(String, String)],
) -> WorkerResult<()> {
    for (key, value) in attrs {
        let allowed = match kind {
            RawSvgElementKind::Defs => key == "id",
            RawSvgElementKind::NamedView => matches!(
                key.as_str(),
                "id" | "pagecolor"
                    | "bordercolor"
                    | "borderopacity"
                    | "inkscape:pageopacity"
                    | "inkscape:pageshadow"
                    | "inkscape:zoom"
                    | "inkscape:cx"
                    | "inkscape:cy"
                    | "inkscape:document-units"
                    | "inkscape:current-layer"
                    | "showgrid"
                    | "units"
                    | "inkscape:window-width"
                    | "inkscape:window-height"
                    | "inkscape:window-x"
                    | "inkscape:window-y"
                    | "inkscape:window-maximized"
                    | "objecttolerance"
                    | "gridtolerance"
                    | "guidetolerance"
            ),
            RawSvgElementKind::Grid => matches!(
                key.as_str(),
                "type"
                    | "id"
                    | "empspacing"
                    | "visible"
                    | "enabled"
                    | "snapvisiblegridlinesonly"
                    | "units"
                    | "spacingx"
                    | "spacingy"
                    | "originx"
                    | "originy"
            ),
            RawSvgElementKind::Metadata => key == "id",
            RawSvgElementKind::RdfLower | RawSvgElementKind::RdfUpper => false,
            RawSvgElementKind::CcLower | RawSvgElementKind::CcUpper => {
                key == "rdf:about" && value.is_empty()
            }
            RawSvgElementKind::DcFormat | RawSvgElementKind::DcTitle | RawSvgElementKind::Title => {
                false
            }
            RawSvgElementKind::DcType => {
                key == "rdf:resource" && value == "http://purl.org/dc/dcmitype/StillImage"
            }
            RawSvgElementKind::Retained
            | RawSvgElementKind::ResourceDefs
            | RawSvgElementKind::UsePathDefinition
            | RawSvgElementKind::Use
            | RawSvgElementKind::ClipPath
            | RawSvgElementKind::LinearGradient
            | RawSvgElementKind::GradientStop
            | RawSvgElementKind::Filter
            | RawSvgElementKind::FeGaussianBlur
            | RawSvgElementKind::FeOffset
            | RawSvgElementKind::FeMerge
            | RawSvgElementKind::FeMergeNode
            | RawSvgElementKind::Hidden => {
                unreachable!("emitted or hidden attributes are validated separately")
            }
        };
        let known_metadata_uri = kind == RawSvgElementKind::DcType && allowed;
        if !allowed || (!known_metadata_uri && unsafe_attribute_value(value)) {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG discarded attribute {key} is not allowed"
            )));
        }
    }
    Ok(())
}

fn normalize_root_dimensions(attrs: &mut [(String, String)]) -> WorkerResult<()> {
    let viewbox = attrs
        .iter()
        .find(|(key, _)| key == "viewBox")
        .map(|(_, value)| parse_viewbox_values(value))
        .transpose()?;
    for (key, value) in attrs
        .iter_mut()
        .filter(|(key, _)| matches!(key.as_str(), "width" | "height"))
    {
        if value.contains('%') {
            if value != "100%" {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG {key} percentage must be exactly 100%"
                )));
            }
            let values = viewbox.ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "provider SVG percentage dimensions require viewBox".to_owned(),
                )
            })?;
            *value = if key == "width" {
                values[2].to_string()
            } else {
                values[3].to_string()
            };
        } else if let Some(pixels) = physical_length_in_pixels(value)? {
            *value = pixels.to_string();
        }
    }
    Ok(())
}

fn physical_length_in_pixels(value: &str) -> WorkerResult<Option<f64>> {
    let Some((number, end)) = parse_svg_number(value, 0) else {
        return Ok(None);
    };
    let Some(unit) = value.get(end..) else {
        return Ok(None);
    };
    let factor = if unit.eq_ignore_ascii_case("in") {
        96.0
    } else if unit.eq_ignore_ascii_case("cm") {
        96.0 / 2.54
    } else if unit.eq_ignore_ascii_case("mm") {
        96.0 / 25.4
    } else if unit.eq_ignore_ascii_case("q") {
        96.0 / 101.6
    } else if unit.eq_ignore_ascii_case("pt") {
        96.0 / 72.0
    } else if unit.eq_ignore_ascii_case("pc") {
        16.0
    } else {
        return Ok(None);
    };
    let pixels = number * factor;
    if !pixels.is_finite() || pixels <= 0.0 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG dimensions must be finite positive numbers".to_owned(),
        ));
    }
    validate_coordinate(pixels, "dimension")?;
    Ok(Some(pixels))
}

fn validate_inert_enable_background(value: &str) -> WorkerResult<()> {
    let Some(numbers) = value.strip_prefix("new ") else {
        return Err(WorkerError::InvalidPayload(
            "provider SVG enable-background is not recognized".to_owned(),
        ));
    };
    let values = parse_number_list(numbers, false, "enable-background", true)?;
    if values.len() != 4 || values[2] <= 0.0 || values[3] <= 0.0 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG enable-background must be new plus four bounded viewport numbers"
                .to_owned(),
        ));
    }
    Ok(())
}

fn canonical_presentation_style(
    element: &str,
    style: &str,
    resources: &SvgResourceCatalog,
    budget: &mut SanitizerBudget,
) -> WorkerResult<Vec<(String, String)>> {
    let lower = style.to_ascii_lowercase();
    let compact = lower
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect::<String>();
    if style.trim().is_empty()
        || style
            .chars()
            .any(|character| matches!(character, '\\' | '{' | '}' | '@'))
        || compact.contains("/*")
        || compact.contains("*/")
        || compact.contains("expression(")
    {
        return Err(WorkerError::InvalidPayload(
            "provider SVG style is not allowed".to_owned(),
        ));
    }
    let mut attrs = Vec::new();
    let mut seen_properties = Vec::new();
    let declarations = style.split(';').collect::<Vec<_>>();
    for (index, declaration) in declarations.iter().enumerate() {
        if declaration.trim().is_empty() {
            if index + 1 == declarations.len() && style.ends_with(';') {
                continue;
            }
            return Err(WorkerError::InvalidPayload(
                "provider SVG style declaration is malformed".to_owned(),
            ));
        }
        let (key, value) = declaration.split_once(':').ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG style declaration is malformed".to_owned())
        })?;
        let key = key.trim();
        let value = value.trim();
        if key.is_empty()
            || value.contains(':')
            || seen_properties.iter().any(|existing| existing == &key)
        {
            return Err(WorkerError::InvalidPayload(
                "provider SVG style declaration is malformed".to_owned(),
            ));
        }
        seen_properties.push(key);
        if matches!(
            (key, value),
            ("display", "inline")
                | ("stroke-miterlimit", "4")
                | ("stroke-dasharray", "none")
                | ("vector-effect", "none")
        ) {
            continue;
        }
        if key == "enable-background" && element == "svg" {
            validate_inert_enable_background(value)?;
            continue;
        }
        if key == "fill" {
            if let Some(target) = local_url_target(value) {
                if resources.ids.get(target) != Some(&SvgResourceKind::LinearGradient) {
                    return Err(WorkerError::InvalidPayload(
                        "provider SVG fill has an unresolved or mistyped resource".to_owned(),
                    ));
                }
                attrs.push((key.to_owned(), value.to_owned()));
                continue;
            }
        }
        if !allowed_presentation_property(key) || value.is_empty() || unsafe_attribute_value(value)
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG style property {key} is not allowed"
            )));
        }
        validate_attribute_resource_budget(element, key, value, budget)?;
        attrs.push((key.to_owned(), value.to_owned()));
    }
    Ok(attrs)
}

fn allowed_presentation_property(key: &str) -> bool {
    matches!(
        key,
        "fill"
            | "fill-rule"
            | "clip-rule"
            | "fill-opacity"
            | "stroke"
            | "stroke-opacity"
            | "stroke-width"
            | "stroke-linecap"
            | "stroke-linejoin"
            | "stroke-miterlimit"
            | "stroke-dasharray"
            | "stroke-dashoffset"
            | "opacity"
    )
}

fn allowed_attribute(element: &str, attribute: &str) -> bool {
    let common = matches!(
        attribute,
        "fill"
            | "fill-rule"
            | "clip-rule"
            | "fill-opacity"
            | "stroke"
            | "stroke-opacity"
            | "stroke-width"
            | "stroke-linecap"
            | "stroke-linejoin"
            | "stroke-miterlimit"
            | "stroke-dasharray"
            | "stroke-dashoffset"
            | "opacity"
            | "transform"
    );
    common
        || match element {
            "svg" => matches!(
                attribute,
                "xmlns" | "width" | "height" | "viewBox" | "preserveAspectRatio" | "overflow"
            ),
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

fn validate_attribute_resource_budget(
    element: &str,
    key: &str,
    value: &str,
    budget: &mut SanitizerBudget,
) -> WorkerResult<()> {
    match key {
        "fill-rule" | "clip-rule" if !matches!(value, "nonzero" | "evenodd") => {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG attribute {key} must be nonzero or evenodd"
            )));
        }
        "d" => {
            budget.path_data_bytes =
                budget
                    .path_data_bytes
                    .checked_add(value.len())
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload("provider SVG path bytes overflow".to_owned())
                    })?;
            if budget.path_data_bytes > MAX_SVG_PATH_DATA_BYTES {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the path-data byte budget".to_owned(),
                ));
            }
            let (commands, numbers) = scan_path_data(value)?;
            budget.path_commands = budget.path_commands.checked_add(commands).ok_or_else(|| {
                WorkerError::InvalidPayload("provider SVG path command overflow".to_owned())
            })?;
            budget.path_numbers = budget.path_numbers.checked_add(numbers).ok_or_else(|| {
                WorkerError::InvalidPayload("provider SVG path number overflow".to_owned())
            })?;
            if budget.path_commands > MAX_SVG_PATH_COMMANDS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the path-command budget".to_owned(),
                ));
            }
            if budget.path_numbers > MAX_SVG_PATH_NUMBERS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the path-number budget".to_owned(),
                ));
            }
        }
        "points" => {
            let values = parse_number_list(value, false, "points", true)?;
            if values.len() < 2 || values.len() % 2 != 0 {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG points must contain coordinate pairs".to_owned(),
                ));
            }
            budget.point_numbers =
                budget
                    .point_numbers
                    .checked_add(values.len())
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload("provider SVG point count overflow".to_owned())
                    })?;
            if budget.point_numbers > MAX_SVG_POINT_NUMBERS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the point-number budget".to_owned(),
                ));
            }
        }
        "transform" => {
            let numbers = validate_transform_list(value)?;
            budget.transform_numbers =
                budget
                    .transform_numbers
                    .checked_add(numbers)
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload(
                            "provider SVG transform number overflow".to_owned(),
                        )
                    })?;
            if budget.transform_numbers > MAX_SVG_TRANSFORM_NUMBERS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the transform-number budget".to_owned(),
                ));
            }
        }
        "viewBox" => {
            parse_viewbox_values(value)?;
        }
        "preserveAspectRatio" => {
            validate_preserve_aspect_ratio(value)?;
        }
        "overflow" if value != "hidden" => {
            return Err(WorkerError::InvalidPayload(
                "provider SVG root overflow must be hidden".to_owned(),
            ));
        }
        "stroke-miterlimit" => {
            let values = parse_number_list(value, false, key, true)?;
            if values.len() != 1 || values[0] < 1.0 {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG stroke-miterlimit must be one finite number at least one"
                        .to_owned(),
                ));
            }
        }
        "stroke-dasharray" if value != "none" => {
            let values = parse_number_list(value, true, key, true)?;
            if values.iter().any(|value| *value < 0.0) || values.iter().all(|value| *value == 0.0) {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG stroke-dasharray must be none or non-negative finite numbers with a positive sum".to_owned(),
                ));
            }
            budget.dash_numbers =
                budget
                    .dash_numbers
                    .checked_add(values.len())
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload("provider SVG dash number overflow".to_owned())
                    })?;
            if budget.dash_numbers > MAX_SVG_DASH_NUMBERS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG exceeds the dash-number budget".to_owned(),
                ));
            }
        }
        "stroke-dashoffset" => {
            let values = parse_number_list(value, true, key, true)?;
            if values.len() != 1 {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG stroke-dashoffset must be one finite number".to_owned(),
                ));
            }
        }
        "stroke-linecap" if !matches!(value, "butt" | "round" | "square") => {
            return Err(WorkerError::InvalidPayload(
                "provider SVG stroke-linecap is not recognized".to_owned(),
            ));
        }
        "stroke-linejoin" if !matches!(value, "miter" | "round" | "bevel") => {
            return Err(WorkerError::InvalidPayload(
                "provider SVG stroke-linejoin is not recognized".to_owned(),
            ));
        }
        _ if coordinate_attribute(element, key) => {
            let values = parse_number_list(value, true, key, true)?;
            if values.len() != 1 {
                return Err(WorkerError::InvalidPayload(format!(
                    "provider SVG attribute {key} must contain exactly one finite number"
                )));
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_preserve_aspect_ratio(value: &str) -> WorkerResult<()> {
    let mut tokens = value.split_ascii_whitespace();
    let first = tokens.next().ok_or_else(|| {
        WorkerError::InvalidPayload("provider SVG preserveAspectRatio is empty".to_owned())
    })?;
    let align = if first == "defer" {
        tokens.next().ok_or_else(|| {
            WorkerError::InvalidPayload(
                "provider SVG preserveAspectRatio defer requires an alignment".to_owned(),
            )
        })?
    } else {
        first
    };
    if align == "none" {
        if tokens.next().is_none() {
            return Ok(());
        }
    } else if matches!(
        align,
        "xMinYMin"
            | "xMidYMin"
            | "xMaxYMin"
            | "xMinYMid"
            | "xMidYMid"
            | "xMaxYMid"
            | "xMinYMax"
            | "xMidYMax"
            | "xMaxYMax"
    ) {
        match tokens.next() {
            None => return Ok(()),
            Some("meet" | "slice") if tokens.next().is_none() => return Ok(()),
            _ => {}
        }
    }
    Err(WorkerError::InvalidPayload(
        "provider SVG preserveAspectRatio must use the SVG alignment grammar".to_owned(),
    ))
}

fn coordinate_attribute(element: &str, key: &str) -> bool {
    key == "stroke-width"
        || match element {
            "svg" => matches!(key, "width" | "height"),
            "rect" => matches!(key, "x" | "y" | "width" | "height" | "rx" | "ry"),
            "circle" => matches!(key, "cx" | "cy" | "r"),
            "ellipse" => matches!(key, "cx" | "cy" | "rx" | "ry"),
            "line" => matches!(key, "x1" | "x2" | "y1" | "y2"),
            _ => false,
        }
}

fn scan_path_data(value: &str) -> WorkerResult<(usize, usize)> {
    let bytes = value.as_bytes();
    let mut index = 0usize;
    let mut commands = 0usize;
    let mut numbers = 0usize;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() || bytes[index] == b',' {
            index += 1;
            continue;
        }
        if matches!(
            bytes[index],
            b'M' | b'm'
                | b'Z'
                | b'z'
                | b'L'
                | b'l'
                | b'H'
                | b'h'
                | b'V'
                | b'v'
                | b'C'
                | b'c'
                | b'S'
                | b's'
                | b'Q'
                | b'q'
                | b'T'
                | b't'
                | b'A'
                | b'a'
        ) {
            commands += 1;
            index += 1;
            continue;
        }
        let (number, end) = parse_svg_number(value, index).ok_or_else(|| {
            WorkerError::InvalidPayload("provider SVG path data is malformed".to_owned())
        })?;
        validate_coordinate(number, "path")?;
        numbers += 1;
        index = end;
    }
    if commands == 0 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG path data contains no commands".to_owned(),
        ));
    }
    Ok((commands, numbers))
}

fn parse_number_list(
    value: &str,
    allow_px: bool,
    label: &str,
    enforce_coordinate_budget: bool,
) -> WorkerResult<Vec<f64>> {
    let bytes = value.as_bytes();
    let mut index = 0usize;
    let mut values = Vec::new();
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() || bytes[index] == b',' {
            index += 1;
            continue;
        }
        let (number, mut end) = parse_svg_number(value, index).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "provider SVG {label} must contain only finite numbers"
            ))
        })?;
        if allow_px && value.as_bytes().get(end..end + 2) == Some(b"px") {
            end += 2;
        }
        if end < bytes.len()
            && !bytes[end].is_ascii_whitespace()
            && bytes[end] != b','
            && !(bytes[end] == b'+' || bytes[end] == b'-')
        {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG {label} uses an unsupported unit or token"
            )));
        }
        if enforce_coordinate_budget {
            validate_coordinate(number, label)?;
        }
        values.push(number);
        index = end;
    }
    if values.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} contains no numbers"
        )));
    }
    Ok(values)
}

fn parse_svg_number(value: &str, start: usize) -> Option<(f64, usize)> {
    let bytes = value.as_bytes();
    let mut index = start;
    if matches!(bytes.get(index), Some(b'+' | b'-')) {
        index += 1;
    }
    let integer_start = index;
    while bytes.get(index).is_some_and(u8::is_ascii_digit) {
        index += 1;
    }
    let mut digits = index - integer_start;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let fraction_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        digits += index - fraction_start;
    }
    if digits == 0 {
        return None;
    }
    if matches!(bytes.get(index), Some(b'e' | b'E')) {
        index += 1;
        if matches!(bytes.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let exponent_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == exponent_start {
            return None;
        }
    }
    let number = value.get(start..index)?.parse::<f64>().ok()?;
    number.is_finite().then_some((number, index))
}

fn validate_coordinate(number: f64, label: &str) -> WorkerResult<()> {
    if !number.is_finite() || number.abs() > MAX_SVG_COORDINATE_MAGNITUDE {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG {label} exceeds the coordinate-magnitude budget"
        )));
    }
    Ok(())
}

fn validate_transform_list(value: &str) -> WorkerResult<usize> {
    let bytes = value.as_bytes();
    let mut index = 0usize;
    let mut total_numbers = 0usize;
    while index < bytes.len() {
        while bytes
            .get(index)
            .is_some_and(|byte| byte.is_ascii_whitespace() || *byte == b',')
        {
            index += 1;
        }
        if index == bytes.len() {
            break;
        }
        let name_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_alphabetic) {
            index += 1;
        }
        let name = value.get(name_start..index).unwrap_or_default();
        if !matches!(
            name,
            "matrix" | "translate" | "scale" | "rotate" | "skewX" | "skewY"
        ) {
            return Err(WorkerError::InvalidPayload(
                "provider SVG transform function is not allowed".to_owned(),
            ));
        }
        while bytes.get(index).is_some_and(u8::is_ascii_whitespace) {
            index += 1;
        }
        if bytes.get(index) != Some(&b'(') {
            return Err(WorkerError::InvalidPayload(
                "provider SVG transform is malformed".to_owned(),
            ));
        }
        index += 1;
        let args_start = index;
        while bytes.get(index).is_some_and(|byte| *byte != b')') {
            index += 1;
        }
        if bytes.get(index) != Some(&b')') {
            return Err(WorkerError::InvalidPayload(
                "provider SVG transform is incomplete".to_owned(),
            ));
        }
        let args = value.get(args_start..index).unwrap_or_default();
        let count = parse_number_list(args, false, "transform", true)?.len();
        let valid_arity = match name {
            "matrix" => count == 6,
            "translate" | "scale" => matches!(count, 1 | 2),
            "rotate" => matches!(count, 1 | 3),
            "skewX" | "skewY" => count == 1,
            _ => false,
        };
        if !valid_arity {
            return Err(WorkerError::InvalidPayload(format!(
                "provider SVG transform {name} has an invalid argument count"
            )));
        }
        total_numbers += count;
        index += 1;
    }
    if total_numbers == 0 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG transform contains no numbers".to_owned(),
        ));
    }
    Ok(total_numbers)
}

fn unsafe_attribute_value(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    let compact = lower
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect::<String>();
    compact.contains("url(")
        || compact.contains("expression(")
        || lower.contains('\\')
        || compact.contains("/*")
        || compact.contains("*/")
        || lower.contains(';')
        || lower.contains('{')
        || lower.contains('}')
        || lower.contains('@')
        || lower.contains("javascript:")
        || lower.contains("data:")
        || lower.contains("file:")
        || lower.contains("http:")
        || lower.contains("https:")
        || lower.contains("ftp:")
        || lower.contains("ws:")
        || lower.contains("wss:")
        || compact.contains("@import")
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
    let values = parse_number_list(value, true, "dimension", true)?;
    let parsed = values
        .first()
        .copied()
        .filter(|value| *value > 0.0)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "provider SVG dimensions must be finite positive numbers".to_owned(),
            )
        })?;
    if values.len() != 1 {
        return Err(WorkerError::InvalidPayload(
            "provider SVG dimensions must contain exactly one number".to_owned(),
        ));
    }
    Ok(parsed.ceil() as u32)
}

fn parse_viewbox(value: &str) -> WorkerResult<(u32, u32)> {
    let numbers = parse_viewbox_values(value)?;
    Ok((numbers[2].ceil() as u32, numbers[3].ceil() as u32))
}

fn parse_viewbox_values(value: &str) -> WorkerResult<[f64; 4]> {
    // Parse finite SVG numbers first, then apply the viewBox-specific origin/span bounds below so
    // an overlarge origin cannot be hidden behind the generic shape-coordinate refusal.
    let numbers = parse_number_list(value, false, "viewBox", false)?;
    let numbers: [f64; 4] = numbers.try_into().map_err(|_| {
        WorkerError::InvalidPayload(
            "provider SVG viewBox must contain four finite numbers".to_owned(),
        )
    })?;
    if numbers[0].abs() > MAX_SVG_VIEWBOX_ORIGIN_MAGNITUDE
        || numbers[1].abs() > MAX_SVG_VIEWBOX_ORIGIN_MAGNITUDE
    {
        return Err(WorkerError::InvalidPayload(
            "provider SVG viewBox origin exceeds the origin-magnitude budget".to_owned(),
        ));
    }
    if numbers[2] <= 0.0
        || numbers[3] <= 0.0
        || numbers[2] > f64::from(MAX_PREVIEW_DIMENSION)
        || numbers[3] > f64::from(MAX_PREVIEW_DIMENSION)
    {
        return Err(WorkerError::InvalidPayload(format!(
            "provider SVG viewBox dimensions must be 1..={MAX_PREVIEW_DIMENSION}"
        )));
    }
    Ok(numbers)
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

fn filter_input_surface_count(kind: &usvg::filter::Kind) -> WorkerResult<u64> {
    match kind {
        // Pinned resvg's box blur allocates one RGBA back buffer. Its IIR path allocates one f64
        // value per pixel, equivalent to two RGBA surfaces, so charge the larger scratch cost in
        // addition to the primitive result charged by the caller.
        usvg::filter::Kind::GaussianBlur(_) => Ok(2),
        usvg::filter::Kind::Offset(_) => Ok(1),
        usvg::filter::Kind::Merge(merge) => u64::try_from(merge.inputs().len()).map_err(|_| {
            WorkerError::InvalidPayload("provider SVG filter input count overflow".to_owned())
        }),
        _ => Err(WorkerError::InvalidPayload(
            "provider SVG parsed to an unsupported filter primitive".to_owned(),
        )),
    }
}

fn clipped_filter_layer_pixels(
    group: &usvg::Group,
    target_width: u32,
    target_height: u32,
    root_transform: resvg::tiny_skia::Transform,
) -> WorkerResult<u64> {
    let bbox = group
        .abs_layer_bounding_box()
        .transform(root_transform)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "provider SVG filter layer has an invalid transform".to_owned(),
            )
        })?
        .to_int_rect();
    // Keep this identical to resvg::render's max_bbox and render_group intersection. The parsed
    // abs_layer_bounding_box already contains objectBoundingBox filter resolution and every SVG
    // transform, while root_transform contains the requested preview fit.
    let left = bbox.left().max(-(target_width as i32) * 2);
    let top = bbox.top().max(-(target_height as i32) * 2);
    let right = bbox.right().min((target_width as i32) * 3);
    let bottom = bbox.bottom().min((target_height as i32) * 3);
    if right <= left || bottom <= top {
        return Ok(0);
    }
    let width = u64::try_from(right - left).map_err(|_| {
        WorkerError::InvalidPayload("provider SVG filter layer width overflow".to_owned())
    })?;
    let height = u64::try_from(bottom - top).map_err(|_| {
        WorkerError::InvalidPayload("provider SVG filter layer height overflow".to_owned())
    })?;
    width.checked_mul(height).ok_or_else(|| {
        WorkerError::InvalidPayload("provider SVG filter layer pixels overflow".to_owned())
    })
}

pub(super) fn validate_filter_render_budget(
    tree: &usvg::Tree,
    target_width: u32,
    target_height: u32,
    root_transform: resvg::tiny_skia::Transform,
) -> WorkerResult<()> {
    fn visit(
        group: &usvg::Group,
        target_width: u32,
        target_height: u32,
        root_transform: resvg::tiny_skia::Transform,
        total: &mut u64,
    ) -> WorkerResult<()> {
        if !group.filters().is_empty() {
            let layer_pixels =
                clipped_filter_layer_pixels(group, target_width, target_height, root_transform)?;
            // resvg allocates the isolated group layer once. Its filter evaluator retains every
            // primitive result and can clone every input surface while producing the next result.
            let mut surfaces = 1u64;
            for filter in group.filters() {
                for primitive in filter.primitives() {
                    surfaces = surfaces.checked_add(1).ok_or_else(|| {
                        WorkerError::InvalidPayload(
                            "provider SVG filter surface count overflow".to_owned(),
                        )
                    })?;
                    let inputs = filter_input_surface_count(primitive.kind())?;
                    surfaces = surfaces.checked_add(inputs).ok_or_else(|| {
                        WorkerError::InvalidPayload(
                            "provider SVG filter surface count overflow".to_owned(),
                        )
                    })?;
                }
            }
            *total = total
                .checked_add(layer_pixels.checked_mul(surfaces).ok_or_else(|| {
                    WorkerError::InvalidPayload(
                        "provider SVG filter offscreen pixels overflow".to_owned(),
                    )
                })?)
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(
                        "provider SVG filter offscreen pixels overflow".to_owned(),
                    )
                })?;
            if *total > MAX_SVG_FILTER_OFFSCREEN_PIXELS {
                return Err(WorkerError::InvalidPayload(
                    "provider SVG filters exceed the total offscreen-pixel budget".to_owned(),
                ));
            }
        }
        for child in group.children() {
            if let usvg::Node::Group(child) = child {
                visit(child, target_width, target_height, root_transform, total)?;
            }
        }
        Ok(())
    }

    let mut total = 0u64;
    visit(
        tree.root(),
        target_width,
        target_height,
        root_transform,
        &mut total,
    )
}
