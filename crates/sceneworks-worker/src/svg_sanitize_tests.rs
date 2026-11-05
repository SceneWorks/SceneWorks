#[test]
fn canonicalizes_inert_svg_and_accepts_safe_px_dimensions() {
    let valid = sanitize_svg("<svg height=\"8px\" width=\"12px\" xmlns=\"http://www.w3.org/2000/svg\"><rect height=\"4\" width=\"3\"/></svg>")
        .expect("valid inert fixture");
    assert_eq!((valid.width, valid.height), (12, 8));
    assert!(
        valid.svg.contains("height=\"8px\" width=\"12px\""),
        "attributes are canonicalized"
    );
}

#[test]
fn inert_root_namespace_declarations_are_dropped_without_broadening_svg_policy() {
    let declarations = concat!(
        r#"xmlns:dc="http://purl.org/dc/elements/1.1/" "#,
        r#"xmlns:cc="http://creativecommons.org/ns#" "#,
        r#"xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" "#,
        r#"xmlns:svg="http://www.w3.org/2000/svg" "#,
        r#"xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" "#,
        r#"xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" "#,
        r#"xmlns:xlink="http://www.w3.org/1999/xlink" "#,
        r#"xmlns:ns1="http://sozi.baierouge.fr""#,
    );
    let input = format!(r#"<svg {declarations} viewBox="0 0 2 2"><path d="M0 0H2V2H0Z"/></svg>"#);
    let canonical = sanitize_svg(&input).expect("inert namespace declarations");
    assert!(canonical
        .svg
        .contains(&format!("xmlns=\"{SVG_NAMESPACE}\"")));
    assert!(
        !canonical.svg.contains("xmlns:"),
        "declarations are not retained"
    );
    let legacy_cc =
        sanitize_svg(r#"<svg xmlns:cc="http://web.resource.org/cc/"><path d="M0 0H1V1Z"/></svg>"#)
            .expect("legacy Creative Commons declaration is inert");
    assert!(!legacy_cc.svg.contains("xmlns:cc"));

    for input in [
        r#"<svg xmlns="https://example.invalid/svg"><path d="M0 0H1V1Z"/></svg>"#,
        r#"<svg xmlns:evil="https://example.invalid"><evil:path/></svg>"#,
        r#"<svg xmlns:dc="https://example.invalid"><path dc:href="https://example.invalid/a" d="M0 0H1V1Z"/></svg>"#,
        r#"<svg><path xmlns:evil="https://example.invalid" d="M0 0H1V1Z"/></svg>"#,
        r#"<svg xmlns:xml="http://www.w3.org/XML/1998/namespace"><path d="M0 0H1V1Z"/></svg>"#,
        r#"<svg xmlns:bad:prefix="https://example.invalid"><path d="M0 0H1V1Z"/></svg>"#,
        r#"<svg xmlns:xlink="https://example.invalid"><path d="M0 0H1V1Z"/></svg>"#,
        r##"<svg xmlns:xlink="http://www.w3.org/1999/xlink"><path xlink:href="#x" d="M0 0H1V1Z"/></svg>"##,
    ] {
        assert!(sanitize_svg(input).is_err(), "accepted {input}");
    }
}

#[tokio::test]
async fn archived_upstream_case_strips_editor_metadata_and_keeps_visible_geometry() {
    let raw = include_bytes!("../tests/fixtures/starvector/upstream-34835393738-case-02.svg");
    let canonical = terminal_sanitize_svg_bytes(raw).expect("actual archived upstream case");
    assert_eq!((canonical.width, canonical.height), (90, 90));
    for retained in [
        "fill=\"#ffffff\"",
        "fill-opacity=\"1\"",
        "stroke=\"#000000\"",
        "stroke-opacity=\"1\"",
        "stroke-width=\"1.25\"",
        "stroke-linecap=\"butt\"",
        "stroke-linejoin=\"miter\"",
        "transform=\"translate(0,-962.35975)\"",
    ] {
        assert!(
            canonical.canonical_svg.contains(retained),
            "missing {retained}"
        );
    }
    for removed in [
        "xmlns:",
        "<metadata",
        "<defs",
        "sodipodi:",
        "inkscape:",
        "style=",
        "stroke-miterlimit=",
        "stroke-dasharray=",
    ] {
        assert!(
            !canonical.canonical_svg.contains(removed),
            "retained {removed}"
        );
    }
    let temp = tempfile::tempdir().expect("temp dir");
    let (_, preview) = terminal_write_sanitized_pair_with_preview_size(
        &canonical,
        &temp.path().join("actual-upstream-case-02"),
        Some(512),
    )
    .await
    .expect("actual fixture renders through the canonical CLI seam");
    let pixels = image::open(preview).expect("preview PNG").to_rgba8();
    assert_eq!(pixels.dimensions(), (512, 512));
    assert!(
        pixels.pixels().any(|pixel| pixel[3] > 0),
        "actual visible geometry renders"
    );
}

#[test]
fn editor_and_rdf_discarding_is_exact_and_contextual() {
    let valid = concat!(
        "<svg>",
        "<defs id=\"defs4\"/>",
        "<sodipodi:namedview id=\"base\" objecttolerance=\"10\" gridtolerance=\"10\" guidetolerance=\"10\"/>",
        "<metadata id=\"metadata7\"><rdf:rdf><cc:work rdf:about=\"\">",
        "<dc:format>image/svg+xml</dc:format>",
        "<dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>",
        "<dc:title>archived title</dc:title>",
        "</cc:work></rdf:rdf></metadata>",
        "<g><title>visible path description</title><rect id=\"shape\" class=\"geometry\" width=\"1\" height=\"1\"/></g>",
        "</svg>",
    );
    let canonical = sanitize_svg(valid).expect("exact inert editor and RDF metadata");
    assert!(canonical.svg.contains("<rect"));
    for discarded in ["<defs", "namedview", "metadata", "rdf:", "cc:", "dc:"] {
        assert!(!canonical.svg.contains(discarded));
    }

    let potrace = sanitize_svg(concat!(
        "<svg><metadata>",
        "Created by potrace 1.11, written by Peter Selinger 2001-2013",
        "</metadata><g><path d=\"M0 0H1V1Z\"/></g></svg>",
    ))
    .expect("exact inert potrace provenance metadata");
    assert!(potrace.svg.contains("<path"));
    assert!(!potrace.svg.contains("Created by potrace"));

    for malicious in [
        "<svg>Created by potrace 1.11, written by Peter Selinger 2001-2013<path d=\"M0 0H1V1Z\"/></svg>",
        "<svg><g>Created by potrace 1.11, written by Peter Selinger 2001-2013</g></svg>",
        "<svg><metadata><script/></metadata></svg>",
        "<svg><metadata><foreignObject/></metadata></svg>",
        "<svg><metadata><rdf:rdf><cc:work rdf:about=\"\"><dc:creator/></cc:work></rdf:rdf></metadata></svg>",
        "<svg><g><metadata><rdf:rdf></rdf:rdf></metadata></g></svg>",
        "<svg><g><sodipodi:namedview></sodipodi:namedview></g></svg>",
        "<svg><inkscape:grid/></svg>",
        "<svg><metadata onclick=\"alert(1)\"><rdf:rdf></rdf:rdf></metadata></svg>",
        "<svg><sodipodi:namedview pagecolor=\"url(https://example.invalid/a)\"></sodipodi:namedview></svg>",
        "<svg><metadata><rdf:rdf></metadata></rdf:rdf></svg>",
        "<svg><path dc:format=\"image/svg+xml\" d=\"M0 0H1V1Z\"/></svg>",
        "<svg><g inkscape:unknown=\"1\"/></svg>",
        "<svg><rect onclick=\"alert(1)\"/></svg>",
        "<svg><title onclick=\"alert(1)\">metadata</title></svg>",
    ] {
        assert!(sanitize_svg(malicious).is_err(), "accepted {malicious}");
    }
}

#[test]
fn inline_presentation_style_is_strict_and_omits_only_exact_defaults() {
    let canonical = sanitize_svg(concat!(
        "<svg><path ",
        "style=\"fill:#ffffff;fill-opacity:1;stroke:#000000;stroke-opacity:1;",
        "stroke-width:1.25;stroke-linecap:butt;stroke-linejoin:miter;",
        "display:inline;stroke-miterlimit:4;stroke-dasharray:none\" ",
        "d=\"M0 0H1V1Z\"/></svg>",
    ))
    .expect("strict presentation style");
    for retained in [
        "fill=\"#ffffff\"",
        "fill-opacity=\"1\"",
        "stroke=\"#000000\"",
        "stroke-opacity=\"1\"",
        "stroke-width=\"1.25\"",
        "stroke-linecap=\"butt\"",
        "stroke-linejoin=\"miter\"",
    ] {
        assert!(canonical.svg.contains(retained), "missing {retained}");
    }
    for omitted in [
        "style=",
        "display=",
        "stroke-miterlimit=",
        "stroke-dasharray=",
    ] {
        assert!(!canonical.svg.contains(omitted), "retained {omitted}");
    }
    let default_only = sanitize_svg(
        "<svg><g style=\"display:inline;stroke-miterlimit:4;stroke-dasharray:none\"></g></svg>",
    )
    .expect("default-only style is safely omitted");
    assert!(!default_only.svg.contains("style="));
    let visible_stroke = sanitize_svg(
        "<svg><path style=\"stroke:black;stroke-miterlimit:10;stroke-dasharray:0.12,0.06;stroke-dashoffset:0\" d=\"M0 0H1\"/></svg>",
    )
    .expect("bounded visible stroke presentation");
    for retained in [
        "stroke-miterlimit=\"10\"",
        "stroke-dasharray=\"0.12,0.06\"",
        "stroke-dashoffset=\"0\"",
    ] {
        assert!(visible_stroke.svg.contains(retained), "missing {retained}");
    }

    for recoverable in [
        "<svg><rect style=\"fill:red;;stroke:black\"/></svg>",
        "<svg><rect style=\"fill:red;fill:blue\"/></svg>",
        "<svg><rect style=\"display:inline;display:inline\"/></svg>",
        "<svg><rect style=\"stroke-miterlimit:4;stroke-miterlimit:4\"/></svg>",
        "<svg><rect fill=\"red\" style=\"fill:blue\"/></svg>",
        "<svg><rect style=\"fill:blue\" fill=\"red\"/></svg>",
        "<svg><rect style=\"fill:r\\65 d\"/></svg>",
        "<svg><rect style=\"fill:red/*comment*/\"/></svg>",
        "<svg><defs><linearGradient id=\"a\"><stop offset=\"0\" style=\"stop-color:red;;stop-opacity:1\"/></linearGradient></defs></svg>",
    ] {
        let canonical = sanitize_svg(recoverable).expect("valid CSS cascade");
        assert_static_render_equivalent("valid CSS cascade", recoverable, &canonical.svg);
    }

    for malicious in [
        "<svg><rect style=\"fill:url(https://example.invalid/a)\"/></svg>",
        "<svg><rect style=\"@import 'https://example.invalid/a'\"/></svg>",
        "<svg><rect style=\"fill:expression(alert(1))\"/></svg>",
        "<svg><rect style=\"--paint:red\"/></svg>",
        "<svg><rect style=\"fill\"/></svg>",
        "<svg><rect stroke-miterlimit=\"0.5\"/></svg>",
        "<svg><rect stroke-miterlimit=\"10px\"/></svg>",
        "<svg><rect style=\"stroke-miterlimit:10px\"/></svg>",
        "<svg><rect stroke-dasharray=\"0 0\"/></svg>",
        "<svg><rect stroke-dashoffset=\"1 2\"/></svg>",
    ] {
        assert!(sanitize_svg(malicious).is_err(), "accepted {malicious}");
    }
    for normalized in [
        "<svg><rect stroke-linecap=\"null\"/></svg>",
        "<svg><rect stroke-linejoin=\"null\"/></svg>",
    ] {
        let canonical = sanitize_svg(normalized).expect("editor null presentation value");
        assert!(!canonical.svg.contains("null"));
    }
}

#[test]
fn exact_vector_effect_none_is_inert_and_other_values_stay_rejected() {
    let raw = include_str!("../tests/fixtures/starvector/upstream-35238431684-quality-case-06.svg");
    assert_eq!(
        sha256_hex(raw.as_bytes()),
        "182857f6688229e765b7795490ec3d11aaba4ee9a9df891ffeb20d01af48f48b"
    );
    let canonical = sanitize_svg(raw).expect("authenticated quality case 6");
    assert!(!canonical.svg.contains("vector-effect"));
    assert_eq!(
        comparison_pixels(raw, 512),
        comparison_pixels(&canonical.svg, 512),
        "omitting the exact default must preserve the authenticated render"
    );

    let non_scaling = "<svg><path style=\"vector-effect:non-scaling-stroke\" d=\"M0 0H1\"/></svg>";
    let non_scaling_canonical = sanitize_svg(non_scaling).expect("valid static non-scaling stroke");
    assert_static_render_equivalent(
        "inline non-scaling stroke",
        non_scaling,
        &non_scaling_canonical.svg,
    );

    for inert in [
        "<svg><path style=\"vector-effect:none;vector-effect:none\" d=\"M0 0H1\"/></svg>",
        "<svg><path vector-effect=\"none\" d=\"M0 0H1\"/></svg>",
    ] {
        let canonical = sanitize_svg(inert).expect("valid inert vector effect");
        assert_static_render_equivalent("valid inert vector effect", inert, &canonical.svg);
    }

    for invalid in [
        "<svg><path style=\"vector-effect:inherit\" d=\"M0 0H1\"/></svg>",
        "<svg><path style=\"vector-effect:\" d=\"M0 0H1\"/></svg>",
        "<svg><path style=\"vector-effect:url(https://example.invalid/a)\" d=\"M0 0H1\"/></svg>",
    ] {
        assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
    }
}

#[test]
fn exact_local_use_paths_expand_to_bounded_canonical_geometry() {
    let raw = include_str!("../tests/fixtures/starvector/upstream-35231778201-8b-case-15.svg");
    assert_eq!(
        sha256_hex(raw.as_bytes()),
        "4399cf9418f6675511cf504e94270dc490ff2acd950057eb57d313ba1c13e485"
    );
    let canonical = sanitize_svg(raw).expect("authenticated 8B parity case 15");
    for removed in ["<use", "xlink:href", "id=\"a\""] {
        assert!(!canonical.svg.contains(removed), "retained {removed}");
    }
    assert_eq!(canonical.svg.matches("<path ").count(), 2);
    assert_eq!(
        comparison_pixels(raw, 512),
        comparison_pixels(&canonical.svg, 512),
        "local-use expansion must preserve the authenticated render"
    );
    assert_eq!(
        canonical.svg,
        sanitize_svg(raw).expect("repeat sanitation").svg,
        "local-use expansion must be byte deterministic"
    );

    let plain_href =
        sanitize_svg("<svg><defs><path id=\"p\" d=\"M0 0H1V1Z\"/></defs><use href=\"#p\"/></svg>")
            .expect("one exact SVG2 local href");
    assert!(plain_href.svg.contains("<path d=\"M0 0H1V1Z\"/>"));
}

#[test]
fn local_use_expansion_rejects_unsafe_ambiguous_or_unbounded_graphs() {
    for invalid in [
        "<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs><use href=\"https://example.invalid/p\"/></svg>",
        "<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\"><defs><path id=\"p\" d=\"M0 0H1\"/></defs><use href=\"#p\" xlink:href=\"#p\"/></svg>",
        "<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs><use/></svg>",
        "<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs><use href=\"#missing\"/></svg>",
        "<svg><defs><linearGradient id=\"p\"/></defs><use href=\"#p\"/></svg>",
        "<svg><defs><path id=\"p\" d=\"M0 0H1\"/><path id=\"p\" d=\"M0 0H2\"/></defs><use href=\"#p\"/></svg>",
        "<svg><defs><path id=\"p\" d=\"M0 0H1\"/><linearGradient id=\"p\"/></defs><use href=\"#p\"/></svg>",
        "<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs><use href=\"#p\"><path d=\"M0 0H2\"/></use></svg>",
        "<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs><use href=\"#p\" onclick=\"alert(1)\"/></svg>",
        "<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs><use href=\"#p\" style=\"fill:url(https://example.invalid/a)\"/></svg>",
        "<svg xmlns:xlink=\"https://example.invalid\"><defs><path id=\"p\" d=\"M0 0H1\"/></defs><use xlink:href=\"#p\"/></svg>",
        "<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs><use xlink:href=\"#p\"/></svg>",
    ] {
        assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
    }
    sanitize_svg("<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs></svg>")
        .expect("unused static definition is valid and renders empty");

    for valid in [
        "<svg><defs><path id=\"p\" fill=\"red\" d=\"M0 0H1\"/></defs><use href=\"#p\"/></svg>",
        "<svg><defs><g><path id=\"p\" d=\"M0 0H1\"/></g></defs><use href=\"#p\"/></svg>",
        "<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs><use href=\"#p\" x=\"1\"/></svg>",
    ] {
        let canonical = sanitize_svg(valid).expect("safe general local use");
        assert_static_render_equivalent("safe general local use", valid, &canonical.svg);
    }

    let uses = "<use href=\"#p\"/>".repeat(MAX_SVG_RESOURCE_REFERENCES + 1);
    let too_many = format!("<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs>{uses}</svg>");
    assert!(
        invalid_detail(sanitize_svg(&too_many)).contains("resource-reference budget"),
        "local uses must share the resource-reference budget"
    );

    let long_path = "M0 0".repeat(1_000);
    let repeated = "<use href=\"#p\"/>".repeat(40);
    let expanded = format!("<svg><defs><path id=\"p\" d=\"{long_path}\"/></defs>{repeated}</svg>");
    assert!(
        sanitize_svg(&expanded).is_err(),
        "expanded path work was unbounded"
    );

    let nested_use = |groups: usize| {
        format!(
            "<svg><defs><path id=\"p\" d=\"M0 0H1\"/></defs>{}<use href=\"#p\"/>{}</svg>",
            "<g>".repeat(groups),
            "</g>".repeat(groups)
        )
    };
    sanitize_svg(&nested_use(MAX_SVG_DEPTH - 3))
        .expect("expanded path at the exact maximum canonical depth");
    assert!(
        invalid_detail(sanitize_svg(&nested_use(MAX_SVG_DEPTH - 2)))
            .contains("expansion exceeds the element nesting budget"),
        "source-valid use must fail when its expansion would exceed canonical depth"
    );
}

#[test]
fn inert_export_root_attributes_are_validated_then_removed() {
    let canonical = sanitize_svg(concat!(
        "<svg x=\"0px\" y=\"0\" xml:space=\"preserve\" ",
        "enable-background=\"new 0 0 48 48\" ",
        "sodipodi:version=\"0.32\" sodipodi:docbase=\"C:\\\\archive\\\\icons\" ",
        "style=\"enable-background:new 0 0 48 48;\" viewBox=\"0 0 48 48\">",
        "<path d=\"M0 0H48V48Z\"/></svg>",
    ))
    .expect("inert export metadata");
    for removed in [
        " x=",
        " y=",
        "xml:space",
        "enable-background",
        "sodipodi:",
        "style=",
    ] {
        assert!(!canonical.svg.contains(removed), "retained {removed}");
    }
    for invalid in [
        "<svg x=\"1\"><path d=\"M0 0H1\"/></svg>",
        "<svg xml:space=\"other\"><path d=\"M0 0H1\"/></svg>",
        "<svg enable-background=\"new 0 0 -1 1\"><path d=\"M0 0H1\"/></svg>",
        "<svg style=\"enable-background:url(https://example.invalid)\"><path d=\"M0 0H1\"/></svg>",
    ] {
        assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
    }
}

#[test]
fn root_percent_dimensions_use_exact_validated_viewbox_spans() {
    let canonical = sanitize_svg(
        "<svg width=\"100%\" height=\"100%\" viewBox=\"0 0 1.25 2.5\"><rect width=\"1.25\" height=\"2.5\"/></svg>",
    )
    .expect("exact root percentages");
    assert_eq!((canonical.width, canonical.height), (2, 3));
    assert!(canonical.svg.contains("width=\"1.25\""));
    assert!(canonical.svg.contains("height=\"2.5\""));
    let numeric = sanitize_svg("<svg width=\"12.50px\" height=\"8.00\"></svg>")
        .expect("numeric dimensions remain unchanged");
    assert!(numeric.svg.contains("width=\"12.50px\""));
    assert!(numeric.svg.contains("height=\"8.00\""));
    let physical = sanitize_svg(
        "<svg width=\"210mm\" height=\"297mm\" viewBox=\"0 0 210 297\"><rect width=\"210\" height=\"297\"/></svg>",
    )
    .expect("standard absolute physical dimensions");
    assert_eq!((physical.width, physical.height), (794, 1123));
    assert!(physical.svg.contains("width=\"793.7007874015749\""));
    assert!(physical.svg.contains("height=\"1122.5196850393702\""));
    for dimension in [
        "1in", "1IN", "2.54cm", "2.54CM", "25.4mm", "25.4MM", "101.6Q", "101.6q", "72pt", "72PT",
        "6pc", "6PC",
    ] {
        let canonical = sanitize_svg(&format!(
            "<svg width=\"{dimension}\" height=\"1px\"><rect width=\"1\" height=\"1\"/></svg>"
        ))
        .expect("standard absolute length unit");
        assert_eq!(canonical.width, 96, "{dimension}");
    }

    for invalid in [
        "<svg width=\"99%\" viewBox=\"0 0 10 10\"></svg>",
        "<svg width=\"100.0%\" viewBox=\"0 0 10 10\"></svg>",
        "<svg width=\"100%\"></svg>",
        "<svg width=\"100%\" viewBox=\"0 0 0 10\"></svg>",
        "<svg height=\"100%\" viewBox=\"0 0 10 2049\"></svg>",
        "<svg><rect width=\"100%\" height=\"1\"/></svg>",
        "<svg width=\"1em\" height=\"1\"></svg>",
    ] {
        assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
    }
}

#[test]
fn root_preserve_aspect_ratio_uses_only_the_svg_alignment_grammar() {
    for value in ["none", "xMidYMid", "xMinYMax meet", "defer xMaxYMin slice"] {
        let canonical = sanitize_svg(&format!(
            "<svg viewBox=\"0 0 16 8\" preserveAspectRatio=\"{value}\"><rect width=\"16\" height=\"8\"/></svg>"
        ))
        .expect("standard preserveAspectRatio value");
        assert_eq!((canonical.width, canonical.height), (16, 8));
        assert!(canonical
            .svg
            .contains(&format!("preserveAspectRatio=\"{value}\"")));
    }

    for value in [
        "",
        "xMidYMid meet slice",
        "none meet",
        "xCenterYCenter",
        "xMidYMid url(https://example.invalid/a)",
        "xMidYMid javascript:alert(1)",
    ] {
        assert!(
            sanitize_svg(&format!(
                "<svg viewBox=\"0 0 16 8\" preserveAspectRatio=\"{value}\"></svg>"
            ))
            .is_err(),
            "accepted {value:?}"
        );
    }
}

#[test]
fn discarded_content_obeys_all_attribute_and_element_budgets() {
    let attributes = |count: usize| {
        (0..count)
            .map(|index| format!(" x{index}=\"v\""))
            .collect::<String>()
    };
    for input in [
        format!("<svg{}></svg>", attributes(25)),
        format!(
            "<svg><sodipodi:namedview{}></sodipodi:namedview></svg>",
            attributes(25)
        ),
        format!("<svg><g{}/></svg>", attributes(13)),
    ] {
        assert!(
            invalid_detail(sanitize_svg(&input)).contains("per-element attribute budget"),
            "wrong refusal for capped element"
        );
    }

    let namedview = concat!(
        "<sodipodi:namedview id=\"x\" pagecolor=\"x\" bordercolor=\"x\" borderopacity=\"1\" ",
        "inkscape:pageopacity=\"0\" inkscape:pageshadow=\"2\" inkscape:zoom=\"1\" ",
        "inkscape:cx=\"1\" inkscape:cy=\"1\" inkscape:document-units=\"in\" ",
        "inkscape:current-layer=\"x\" showgrid=\"false\" units=\"in\" ",
        "inkscape:window-width=\"1\" inkscape:window-height=\"1\" inkscape:window-x=\"1\" ",
        "inkscape:window-y=\"1\" inkscape:window-maximized=\"1\"></sodipodi:namedview>",
    );
    let global = format!(
        "<svg>{}</svg>",
        namedview.repeat(MAX_SVG_ATTRIBUTES / 18 + 1)
    );
    assert!(invalid_detail(sanitize_svg(&global)).contains("total attribute budget"));

    let oversized = "x".repeat(MAX_SVG_ATTRIBUTE_VALUE_BYTES + 1);
    assert!(invalid_detail(sanitize_svg(&format!(
        "<svg><metadata id=\"{oversized}\"><rdf:rdf></rdf:rdf></metadata></svg>"
    )))
    .contains("per-value byte budget"));

    let value = "x".repeat(50_000);
    let total = format!(
        "<svg>{}</svg>",
        format!("<metadata id=\"{value}\"><rdf:rdf></rdf:rdf></metadata>").repeat(4)
    );
    assert!(invalid_detail(sanitize_svg(&total)).contains("total attribute-value byte budget"));

    let elements = format!("<svg>{}</svg>", "<defs id=\"x\"/>".repeat(MAX_SVG_ELEMENTS));
    assert!(invalid_detail(sanitize_svg(&elements)).contains("element budget"));
}

#[tokio::test]
async fn fill_rules_preserve_holes_and_inherited_geometry() {
    for rule in ["nonzero", "evenodd"] {
        let input = format!(
            r#"<svg width="16" height="16"><g fill-rule="{rule}" clip-rule="{rule}"><path fill="red" d="M0 0H16V16H0Z M4 4H12V12H4Z"/></g></svg>"#
        );
        let canonical = sanitize_svg(&input).expect("inert fill rule");
        assert!(canonical.svg.contains(&format!("fill-rule=\"{rule}\"")));
        assert!(canonical.svg.contains(&format!("clip-rule=\"{rule}\"")));
        let temp = tempfile::tempdir().expect("temp dir");
        let preview = temp.path().join("preview.png");
        render_preview(&canonical.svg, 16, 16, &preview)
            .await
            .expect("render rule");
        let pixels = image::open(preview).expect("PNG").to_rgba8();
        assert_eq!(pixels.get_pixel(1, 1)[3], 255);
        assert_eq!(
            pixels.get_pixel(8, 8)[3],
            if rule == "evenodd" { 0 } else { 255 }
        );
    }
}

#[test]
fn fill_rules_reject_invalid_enums_and_resources() {
    for attribute in ["fill-rule", "clip-rule"] {
        for value in [
            "",
            "inherit",
            "EvenOdd",
            "evenodd nonzero",
            "url(#mask)",
            "url(https://example.invalid/a)",
            "file:///tmp/a",
            "e\\76enodd",
            "evenodd;fill:red",
        ] {
            let input = format!(r#"<svg><path {attribute}="{value}" d="M0 0H1V1Z"/></svg>"#);
            assert!(
                sanitize_svg(&input).is_err(),
                "accepted {attribute}={value}"
            );
        }
    }
    for input in [
        r#"<svg><path fill-rule="evenodd" clip-path="url(#mask)" d="M0 0H1V1Z"/></svg>"#,
        r#"<svg><clipPath clip-rule="evenodd"/></svg>"#,
        r#"<svg><path fill-rule="evenodd" onclick="alert(1)" d="M0 0H1V1Z"/></svg>"#,
    ] {
        assert!(sanitize_svg(input).is_err(), "{input}");
    }
}

#[test]
fn fill_rules_still_consume_attribute_budgets() {
    let input = format!(
        "<svg>{}</svg>",
        r#"<g fill-rule="evenodd" clip-rule="nonzero" fill="red"/>"#
            .repeat(MAX_SVG_ATTRIBUTES / 3 + 1)
    );
    assert!(invalid_detail(sanitize_svg(&input)).contains("total attribute budget"));
    let value = "evenodd".repeat(MAX_SVG_ATTRIBUTE_VALUE_BYTES / 7 + 1);
    let input = format!(r#"<svg fill-rule="{value}"/>"#);
    assert!(invalid_detail(sanitize_svg(&input)).contains("per-value byte budget"));
}

#[test]
fn visible_dash_patterns_obey_a_separate_number_budget() {
    let values = "1 ".repeat(MAX_SVG_DASH_NUMBERS + 1);
    let input = format!("<svg><path stroke-dasharray=\"{values}\" d=\"M0 0H1\"/></svg>");
    assert!(invalid_detail(sanitize_svg(&input)).contains("dash-number budget"));
}

#[tokio::test]
async fn upstream_fill_rule_svg_keeps_source_and_gets_bounded_comparison_preview() {
    let raw = include_bytes!("../tests/fixtures/starvector/upstream-34829516753-case-00.svg");
    let canonical = terminal_sanitize_svg_bytes(raw).expect("actual upstream case accepted");
    assert_eq!((canonical.width, canonical.height), (80, 80));
    assert!(canonical.canonical_svg.contains("fill-rule=\"evenodd\""));
    assert!(canonical.canonical_svg.contains("clip-rule=\"evenodd\""));
    assert!(canonical.canonical_svg.contains("viewBox=\"0 0 80 80\""));
    let temp = tempfile::tempdir().expect("temp dir");
    let (_, intrinsic) = terminal_write_sanitized_pair(&canonical, &temp.path().join("intrinsic"))
        .await
        .expect("intrinsic pair");
    let (source, comparison) = terminal_write_sanitized_pair_with_preview_size(
        &canonical,
        &temp.path().join("comparison"),
        Some(512),
    )
    .await
    .expect("comparison pair");
    assert_eq!(
        image::open(intrinsic).expect("PNG").to_rgba8().dimensions(),
        (80, 80)
    );
    let pixels = image::open(comparison).expect("PNG").to_rgba8();
    assert_eq!(pixels.dimensions(), (512, 512));
    assert!(
        pixels.pixels().any(|pixel| pixel[3] > 0),
        "actual geometry renders"
    );
    assert_eq!(
        std::fs::read_to_string(source).expect("canonical source"),
        canonical.canonical_svg
    );
    for size in [0, MAX_PREVIEW_DIMENSION + 1] {
        let destination = temp.path().join(format!("invalid-{size}"));
        assert!(terminal_write_sanitized_pair_with_preview_size(
            &canonical,
            &destination,
            Some(size)
        )
        .await
        .is_err());
        assert!(!destination.exists());
    }
    // A rectangular viewport is fitted and centered, not stretched to fill the square.
    let rectangle = terminal_sanitize_svg_bytes(
        br#"<svg viewBox="0 0 16 8"><rect width="16" height="8" fill="red"/></svg>"#,
    )
    .expect("rectangle");
    let (_, preview) = terminal_write_sanitized_pair_with_preview_size(
        &rectangle,
        &temp.path().join("rectangle"),
        Some(32),
    )
    .await
    .expect("fitted rectangle");
    let pixels = image::open(preview).expect("PNG").to_rgba8();
    assert_eq!(pixels.get_pixel(16, 2)[3], 0);
    assert_eq!(pixels.get_pixel(16, 16)[3], 255);
    assert_eq!(pixels.get_pixel(16, 29)[3], 0);
}

#[tokio::test]
async fn upstream_case_four_preserves_viewbox_aspect_ratio_rendering() {
    let raw = include_bytes!("../tests/fixtures/starvector/upstream-34841832913-case-04.svg");
    let canonical = terminal_sanitize_svg_bytes(raw).expect("actual upstream case four accepted");
    assert_eq!((canonical.width, canonical.height), (1000, 1000));
    assert!(canonical
        .canonical_svg
        .contains("viewBox=\"0 0 1000 1000\""));
    assert!(canonical
        .canonical_svg
        .contains("preserveAspectRatio=\"xMidYMid\""));
    let temp = tempfile::tempdir().expect("temp dir");
    let (_, preview) = terminal_write_sanitized_pair_with_preview_size(
        &canonical,
        &temp.path().join("actual-upstream-case-04"),
        Some(512),
    )
    .await
    .expect("actual case renders through the canonical CLI seam");
    let pixels = image::open(preview).expect("preview PNG").to_rgba8();
    assert_eq!(pixels.dimensions(), (512, 512));
    assert!(
        pixels.pixels().any(|pixel| pixel[3] > 0),
        "actual visible geometry renders"
    );
}

fn comparison_pixels(svg: &str, size: u32) -> Vec<u8> {
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).expect("comparison SVG");
    let intrinsic = tree.size();
    let scale = (size as f32 / intrinsic.width()).min(size as f32 / intrinsic.height());
    let x = (size as f32 - intrinsic.width() * scale) / 2.0;
    let y = (size as f32 - intrinsic.height() * scale) / 2.0;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size, size).expect("comparison pixmap");
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, x, y),
        &mut pixmap.as_mut(),
    );
    pixmap.data().to_vec()
}

fn assert_static_render_equivalent(label: &str, raw: &str, canonical: &str) {
    let raw_pixels = comparison_pixels(raw, 512);
    let canonical_pixels = comparison_pixels(canonical, 512);
    if raw_pixels != canonical_pixels {
        let samples = raw_pixels
            .iter()
            .zip(&canonical_pixels)
            .enumerate()
            .filter(|(_, (left, right))| left != right)
            .take(8)
            .map(|(index, (left, right))| format!("{index}:{left}->{right}"))
            .collect::<Vec<_>>()
            .join(",");
        let differing_channels = raw_pixels
            .iter()
            .zip(&canonical_pixels)
            .filter(|(left, right)| left != right)
            .count();
        panic!(
            "static canonicalization changed {label} pixels: raw={}, canonical={}, differing_channels={differing_channels}, samples={samples}",
            sha256_hex(&raw_pixels),
            sha256_hex(&canonical_pixels),
        );
    }
}

#[test]
fn static_stylesheets_preserve_all_eight_authenticated_renders() {
    let cases: [(&str, &[u8], &str); 8] = [
        (
            "upstream-1b-15",
            include_bytes!(
                "../tests/fixtures/starvector/terminal-35379924047-upstream-1b-case-15.svg"
            ),
            "7ab80cdacb5a6d2bcbe8420af0388daf5d1fe1437453f4f7ed2fe8908a7b41ae",
        ),
        (
            "upstream-1b-16",
            include_bytes!(
                "../tests/fixtures/starvector/terminal-35379924047-upstream-1b-case-16.svg"
            ),
            "8b33114dc49d34d36e11598f47f3a1314ea0d61ddef98437804800e72e91642e",
        ),
        (
            "upstream-1b-18",
            include_bytes!(
                "../tests/fixtures/starvector/terminal-35379924047-upstream-1b-case-18.svg"
            ),
            "8db17c1632240998bb227f2636900a13f84f31da56c8e49808a5f6442f02911b",
        ),
        (
            "upstream-8b-16",
            include_bytes!(
                "../tests/fixtures/starvector/terminal-35379924047-upstream-8b-case-16.svg"
            ),
            "cd70fdc6885954d9571e06e8f3558ae68f50ea1f69601658a3d05383a64e96fa",
        ),
        (
            "upstream-8b-18",
            include_bytes!(
                "../tests/fixtures/starvector/terminal-35379924047-upstream-8b-case-18.svg"
            ),
            "6af9c20bef7415f795907b299b70a795801d26ac937e56602f6c2fb270b2b54b",
        ),
        (
            "upstream-8b-19",
            include_bytes!(
                "../tests/fixtures/starvector/terminal-35379924047-upstream-8b-case-19.svg"
            ),
            "41e42e97aac2f7ba5b0e9d95bfca4b548641135813096a46c5751894dcdffe2e",
        ),
        (
            "native-quality-25",
            include_bytes!("../tests/fixtures/starvector/terminal-35379924047-quality-25.svg"),
            "7965d511b94488b55e256b002151585f58bfd92b3887d4c7e3ef80334ffa1843",
        ),
        (
            "native-quality-28",
            include_bytes!("../tests/fixtures/starvector/terminal-35379924047-quality-28.svg"),
            "68da04c9d228941dfabbe6aaee56c29b240e41c9d0177000665025a139c13b86",
        ),
    ];
    for (label, bytes, expected_sha256) in cases {
        assert_eq!(sha256_hex(bytes), expected_sha256, "wrong {label} fixture");
        let raw = std::str::from_utf8(bytes).expect("captured provider SVG is UTF-8");
        let canonical = sanitize_svg(raw).unwrap_or_else(|error| panic!("{label} failed: {error}"));
        assert_static_render_equivalent(label, raw, &canonical.svg);
        for removed in ["<style", "class=", "animation", "@keyframes"] {
            assert!(
                !canonical.svg.contains(removed),
                "{label} retained {removed}"
            );
        }
        assert_eq!(
            canonical.svg,
            sanitize_svg(raw).expect("repeat sanitation").svg,
            "{label} is not byte deterministic"
        );
    }
}

#[test]
fn static_fallback_preserves_hidden_ancestor_and_normalizes_null_presentation() {
    let hidden =
        include_bytes!("../tests/fixtures/starvector/terminal-35379924047-upstream-8b-case-10.svg");
    assert_eq!(
        sha256_hex(hidden),
        "f4be71af506b04f3171118c3179542f8c8a3d513991fb715b82921db9ec60abe"
    );
    let hidden_raw = std::str::from_utf8(hidden).expect("hidden fixture UTF-8");
    let hidden_canonical = sanitize_svg(hidden_raw).expect("hidden ancestor fixture");
    assert_static_render_equivalent(
        "display:inline descendants under display:none ancestor",
        hidden_raw,
        &hidden_canonical.svg,
    );
    for removed in ["<text", "TEMPLATE", "EXAMPLE", "display="] {
        assert!(
            !hidden_canonical.svg.contains(removed),
            "retained {removed}"
        );
    }

    let nulls =
        include_bytes!("../tests/fixtures/starvector/terminal-35379924047-upstream-8b-case-11.svg");
    assert_eq!(
        sha256_hex(nulls),
        "6617aa5dca79552264f50c1cbb7888cf1546e76142ca2d1f297b6c158b9e13e3"
    );
    let null_raw = std::str::from_utf8(nulls).expect("null fixture UTF-8");
    let null_canonical = sanitize_svg(null_raw).expect("null presentation fixture");
    assert_static_render_equivalent(
        "invalid null presentation attributes",
        null_raw,
        &null_canonical.svg,
    );
    assert!(!null_canonical.svg.contains("null"));

    let malformed = include_bytes!(
        "../tests/fixtures/starvector/terminal-35379924047-upstream-1b-case-11-malformed.svg"
    );
    assert_eq!(
        sha256_hex(malformed),
        "9ba03b21ab11692a63bfc79aca69719affe32385130860e995379710c237488c"
    );
    assert!(
        terminal_sanitize_svg_bytes(malformed).is_err(),
        "truncated nested SVG must remain rejected"
    );
}

#[test]
fn static_fallback_resolves_cascade_local_references_use_and_nested_svg() {
    let raw = concat!(
        r##"<svg width="32" height="16" viewBox="0 0 32 16" xmlns="http://www.w3.org/2000/svg">"##,
        r##"<style>path{fill:red}.paint{fill:url(#gradient);stroke:#111}g .paint{stroke-width:2}#target{stroke:#00f!important}</style>"##,
        r##"<defs><linearGradient id="gradient"><stop offset="0" stop-color="#0f0"/><stop offset="1" stop-color="#080"/></linearGradient>"##,
        r##"<path id="shape" d="M0 0H12V12H0Z"/></defs>"##,
        r##"<g><use href="#shape" class="paint"/><svg x="16" width="16" height="16" viewBox="0 0 16 16"><path id="target" class="paint" d="M1 1H15V15H1Z"/></svg></g></svg>"##,
    );
    let canonical = sanitize_svg(raw).expect("supported static SVG surface");
    assert_static_render_equivalent("cascade and local references", raw, &canonical.svg);
    for removed in ["<style", "class=", "<use", "<svg x="] {
        assert!(!canonical.svg.contains(removed), "retained {removed}");
    }
    assert!(canonical.svg.matches("<path").count() >= 2);
}

#[test]
fn presentation_only_static_svg_needs_no_unrelated_fallback_trigger() {
    let cases = [
        r#"<svg width="32" height="16"><path vector-effect="non-scaling-stroke" transform="scale(2)" fill="none" stroke="red" d="M1 1H12"/></svg>"#,
        r#"<svg width="32" height="16"><path visibility="hidden" fill="red" d="M0 0H16V16H0Z"/></svg>"#,
        r#"<svg width="32" height="16"><path paint-order="stroke fill" fill="red" stroke="blue" stroke-width="4" d="M4 4H28V12H4Z"/></svg>"#,
    ];
    for raw in cases {
        let canonical = sanitize_svg(raw).expect("safe presentation-only static SVG");
        assert_static_render_equivalent("presentation-only static SVG", raw, &canonical.svg);
    }
    for malicious in [
        r#"<svg><path vector-effect="url(https://example.invalid/a)" d="M0 0H1"/></svg>"#,
        r#"<svg><path visibility="javascript:alert(1)" d="M0 0H1"/></svg>"#,
        r#"<svg><path paint-order="url(#external)" d="M0 0H1"/></svg>"#,
    ] {
        assert!(sanitize_svg(malicious).is_err(), "accepted {malicious}");
    }
}

#[test]
fn static_fallback_accepts_each_declared_resource_budget_boundary_once() {
    let path = format!(
        "M0 0{}",
        "L0 0".repeat(MAX_SVG_PATH_COMMANDS.saturating_sub(1))
    );
    sanitize_svg(&format!(
        r#"<svg><path vector-effect="non-scaling-stroke" d="{path}"/></svg>"#
    ))
    .expect("path commands at the declared boundary");

    let points = "0,0 ".repeat(MAX_SVG_POINT_NUMBERS / 2);
    sanitize_svg(&format!(
        r#"<svg><polyline vector-effect="non-scaling-stroke" points="{points}"/></svg>"#
    ))
    .expect("point numbers at the declared boundary");

    let transform = "translate(0)".repeat(MAX_SVG_TRANSFORM_NUMBERS);
    sanitize_svg(&format!(
        r#"<svg><g vector-effect="non-scaling-stroke" transform="{transform}"/></svg>"#
    ))
    .expect("transform numbers at the declared boundary");

    let dasharray = "1 ".repeat(MAX_SVG_DASH_NUMBERS);
    sanitize_svg(&format!(
        r#"<svg><path vector-effect="non-scaling-stroke" stroke="black" stroke-dasharray="{dasharray}" stroke-dashoffset="1" d="M0 0H1"/></svg>"#
    ))
    .expect("dash numbers at the declared boundary");
}

#[test]
fn stylesheet_rejection_reports_the_authoritative_static_error() {
    let unsupported =
        r#"<svg><style>.x{unknown-static-property:red}</style><path class="x" d="M0 0H1"/></svg>"#;
    let detail = invalid_detail(sanitize_svg(unsupported));
    assert!(
        detail.starts_with("provider SVG CSS property unknown-static-property"),
        "{detail}"
    );
    assert!(
        detail.contains("narrow SVG profile also rejected:"),
        "{detail}"
    );

    let external = r#"<svg><style>.x{fill:url(https://example.invalid/paint)}</style><path class="x" d="M0 0H1"/></svg>"#;
    assert!(
        invalid_detail(sanitize_svg(external)).starts_with(&invalid_detail(
            sanitize_static_svg_bytes(external.as_bytes())
        ),)
    );

    let safe = r#"<svg width="16" height="16"><style>.x{fill:red;stroke:black;stroke-width:1}</style><rect class="x" width="16" height="16"/></svg>"#;
    let canonical = sanitize_svg(safe).expect("safe provider stylesheet");
    assert!(!canonical.svg.contains("<style"));
    assert_static_render_equivalent("safe provider stylesheet", safe, &canonical.svg);
}

#[test]
fn static_fallback_rejects_active_external_and_unsupported_content() {
    let malicious = [
        r#"<svg><script>alert(1)</script></svg>"#,
        r#"<svg><path onload="alert(1)" d="M0 0H1"/></svg>"#,
        r#"<svg><animate attributeName="x" from="0" to="1"/></svg>"#,
        r#"<svg><foreignObject><div>active</div></foreignObject></svg>"#,
        r#"<svg><image href="https://example.invalid/a.png"/></svg>"#,
        r#"<svg><image href="data:image/png;base64,AA=="/></svg>"#,
        r#"<svg><text>visible host-font text</text></svg>"#,
        r#"<svg><filter id="f"><feTurbulence/></filter><path filter="url(#f)" d="M0 0H1"/></svg>"#,
        r#"<svg xmlns="https://example.invalid/not-svg"><path d="M0 0H1"/></svg>"#,
        r#"<svg xmlns:evil="https://example.invalid"><evil:path/></svg>"#,
        r#"<svg><defs><linearGradient id="g" style="fill:url(https://example.invalid/a)"><stop offset="0" stop-color="red"/></linearGradient></defs></svg>"#,
        r#"<svg><defs><clipPath id="c" onclick="alert(1)"><rect/></clipPath></defs></svg>"#,
        r#"<svg><?payload active?><path d="M0 0H1"/></svg>"#,
        r#"<svg><![CDATA[not a stylesheet]]></svg>"#,
        r#"<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg>&xxe;</svg>"#,
    ];
    for input in malicious {
        assert!(sanitize_svg(input).is_err(), "accepted {input}");
    }
}

fn force_static_v2_fallback(svg: &str) -> String {
    let root_end = svg.find('>').expect("SVG root start") + 1;
    format!(
        "{}<style>.compat{{fill:red}}</style>{}",
        &svg[..root_end],
        &svg[root_end..]
    )
}

#[test]
fn static_v2_feature_cannot_bypass_existing_safety_and_resource_bounds() {
    let malicious = [
        r#"<svg><rect style="fill:expression(alert(1))"/></svg>"#.to_owned(),
        r#"<svg><rect style="--paint:red"/></svg>"#.to_owned(),
        r#"<svg><rect style="unknown-static-property:red"/></svg>"#.to_owned(),
        r##"<svg><defs><clipPath id="c"><rect/></clipPath></defs><path style="fill:url(#c)" d="M0 0H1"/></svg>"##.to_owned(),
        r#"<svg><rect fill="u r l( data:image/png;base64,AA== )"/></svg>"#.to_owned(),
        r#"<svg><path fill-rule="EvenOdd" d="M0 0H1"/></svg>"#.to_owned(),
        r#"<svg><rect x="1000001" width="1" height="1"/></svg>"#.to_owned(),
        r#"<svg viewBox="0 0 16 8" preserveAspectRatio="xMidYMid meet slice"></svg>"#
            .to_owned(),
        r#"<svg baseProfile="basic"><path d="M0 0H1"/></svg>"#.to_owned(),
        r#"<svg><path mystery="silently-dropped" d="M0 0H1"/></svg>"#.to_owned(),
        r##"<svg><defs><path id="p" d="M0 0H1"/></defs><use xlink:href="#p"/></svg>"##
            .to_owned(),
    ];
    for input in malicious {
        let forced = force_static_v2_fallback(&input);
        assert!(
            sanitize_svg(&forced).is_err(),
            "accepted forced fallback: {forced}"
        );
    }

    let attributes = (0..13)
        .map(|index| format!(r#" x{index}="v""#))
        .collect::<String>();
    let per_element = format!(r#"<svg><g{attributes}/></svg>"#);
    assert!(sanitize_svg(&force_static_v2_fallback(&per_element)).is_err());

    let resources = (0..=MAX_SVG_RESOURCES)
        .map(|index| format!(r#"<clipPath id="c{index}"><rect/></clipPath>"#))
        .collect::<String>();
    let resource_overflow = format!("<svg><defs>{resources}</defs></svg>");
    assert!(sanitize_svg(&force_static_v2_fallback(&resource_overflow)).is_err());

    let references = (0..=MAX_SVG_RESOURCE_REFERENCES)
        .map(|_| r#"<path fill="url(#g)" d="M0 0H1"/>"#)
        .collect::<String>();
    let reference_overflow = format!(
        r#"<svg><defs><linearGradient id="g"><stop offset="0" stop-color="red"/></linearGradient></defs>{references}</svg>"#
    );
    assert!(sanitize_svg(&force_static_v2_fallback(&reference_overflow)).is_err());

    let mut primitives = String::new();
    let mut input = "SourceAlpha".to_owned();
    for index in 0..=MAX_SVG_FILTER_PRIMITIVES {
        let result = format!("r{index}");
        primitives.push_str(&format!(
            r#"<feOffset in="{input}" result="{result}" dx="1" dy="1"/>"#
        ));
        input = result;
    }
    let filter_overflow =
        format!(r#"<svg><defs><filter id="f">{primitives}</filter></defs></svg>"#);
    assert!(sanitize_svg(&force_static_v2_fallback(&filter_overflow)).is_err());
}

#[test]
fn static_css_parser_rejects_imports_external_urls_and_parser_boundaries() {
    let malicious = [
        r#"<svg><style>@import url(https://example.invalid/a.css);.x{fill:red}</style><path class="x" d="M0 0H1"/></svg>"#,
        r#"<svg><style>@\69mport "https://example.invalid/a.css";.x{fill:red}</style><path class="x" d="M0 0H1"/></svg>"#,
        r#"<svg><style>.x{fill:u\72l(https://example.invalid/a)}</style><path class="x" d="M0 0H1"/></svg>"#,
        r##"<svg><style>.x{fill:url("https://example.invalid/a")}</style><path class="x" d="M0 0H1"/></svg>"##,
        r#"<svg><style>.x{fill:url(data:image/svg+xml;base64,AA==)}</style><path class="x" d="M0 0H1"/></svg>"#,
        r#"<svg><style>.x{fill:url(#missing)}</style><path class="x" d="M0 0H1"/></svg>"#,
        r#"<svg><style>@keyframes x{to{fill:url(https://example.invalid/a)}}.x{fill:red}</style><path class="x" d="M0 0H1"/></svg>"#,
        r#"<svg><style>.x{fill:"unterminated}</style><path class="x" d="M0 0H1"/></svg>"#,
        r#"<svg><style>.x{fill:red}}</style><path class="x" d="M0 0H1"/></svg>"#,
    ];
    for input in malicious {
        assert!(sanitize_svg(input).is_err(), "accepted {input}");
    }
}

#[test]
fn static_fallback_bounds_source_css_and_local_use_expansion() {
    let attributes = (0..65)
        .map(|index| format!(r#" x{index}="v""#))
        .collect::<String>();
    let oversized_element = format!(r#"<svg><path d="M0 0H1"{attributes}/></svg>"#);
    assert!(
        invalid_detail(sanitize_svg(&oversized_element)).contains("per-element attribute budget")
    );

    let css_blocks = format!("<svg><style>{}</style></svg>", ".x{}".repeat(513));
    assert!(
        sanitize_svg(&css_blocks).is_err(),
        "CSS block expansion must be bounded"
    );

    let recursive =
        r##"<svg><defs><g id="loop"><use href="#loop"/></g></defs><use href="#loop"/></svg>"##;
    assert!(
        invalid_detail(sanitize_static_svg_bytes(recursive.as_bytes())).contains("recursive"),
        "local-use cycles must fail before renderer expansion"
    );

    let mut exponential = String::from("<svg><defs><path id=\"n0\" d=\"M0 0H1\"/>");
    for index in 1..16 {
        exponential.push_str(&format!(
            r##"<g id="n{index}"><use href="#n{}"/><use href="#n{}"/></g>"##,
            index - 1,
            index - 1
        ));
    }
    exponential.push_str("</defs><use href=\"#n15\"/></svg>");
    assert!(
        invalid_detail(sanitize_static_svg_bytes(exponential.as_bytes())).contains("expansion"),
        "exponential local-use graphs must fail before renderer expansion"
    );
}

#[test]
fn static_sanitizer_version_distinguishes_the_expanded_policy() {
    assert_eq!(VECTOR_SANITIZER_VERSION, "sceneworks-static-svg-v2");
}

#[test]
fn terminal_campaign_inert_editor_metadata_preserves_captured_renders() {
    let cases: [(&str, &[u8], &str); 3] = [
        (
            "quality-14",
            include_bytes!("../tests/fixtures/starvector/terminal-35379924047-quality-14.svg"),
            "5cf48f9a869e5f641d6f342c930ac4376d58183914de9162a08a25f533f8bc32",
        ),
        (
            "quality-20",
            include_bytes!("../tests/fixtures/starvector/terminal-35379924047-quality-20.svg"),
            "8cb8ac65a95e0025c7a12573a8adfe5ed173cf2f5773b070ddcf98da8cc680e1",
        ),
        (
            "quality-27",
            include_bytes!("../tests/fixtures/starvector/terminal-35379924047-quality-27.svg"),
            "d46970b43771126834a556044cf9879ac1d695448e3f4fd77128a04e575f084b",
        ),
    ];
    for (label, bytes, expected_sha256) in cases {
        assert_eq!(sha256_hex(bytes), expected_sha256, "wrong {label} fixture");
        let raw = std::str::from_utf8(bytes).expect("captured provider SVG is UTF-8");
        let canonical = sanitize_svg(raw).unwrap_or_else(|error| panic!("{label} failed: {error}"));
        assert_eq!(
            comparison_pixels(raw, 512),
            comparison_pixels(&canonical.svg, 512),
            "stripping inert editor metadata changed {label} pixels"
        );
        assert!(
            comparison_pixels(&canonical.svg, 512)
                .as_chunks::<4>()
                .0
                .iter()
                .any(|pixel| pixel[3] != 0),
            "{label} canonical render is empty"
        );
        for removed in [
            "baseProfile",
            "xmlns:ev",
            "xmlns:inkscape",
            "xmlns:sodipodi",
            "inkscape:",
            "sodipodi:",
        ] {
            assert!(
                !canonical.svg.contains(removed),
                "{label} retained {removed}"
            );
        }
        assert_eq!(
            canonical.svg,
            sanitize_svg(raw).expect("repeat sanitation").svg,
            "{label} is not byte deterministic"
        );
    }
}

#[test]
fn terminal_campaign_editor_metadata_remains_exact_scoped_and_bounded() {
    let sodipodi = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd";
    let inkscape = "http://www.inkscape.org/namespaces/inkscape";
    for invalid in [
        r#"<svg baseProfile="basic"/>"#.to_owned(),
        r#"<svg><path baseProfile="full" d="M0 0H1"/></svg>"#.to_owned(),
        r#"<svg xmlns:ev="https://example.invalid/xml-events"/>"#.to_owned(),
        r#"<svg><g xmlns:ev="http://www.w3.org/2001/xml-events"/></svg>"#.to_owned(),
        r#"<svg xmlns:ev="http://www.w3.org/2001/xml-events"><path ev:event="load" d="M0 0H1"/></svg>"#.to_owned(),
        format!(r#"<svg xmlns:sodipodi="{sodipodi}"><path sodipodi:unknown="x" d="M0 0H1"/></svg>"#),
        format!(r#"<svg xmlns:inkscape="{inkscape}"><path inkscape:unknown="x" d="M0 0H1"/></svg>"#),
        r#"<svg><path other:nodetypes="cc" d="M0 0H1"/></svg>"#.to_owned(),
    ] {
        assert!(sanitize_svg(&invalid).is_err(), "accepted {invalid}");
    }

    let editor_attributes = [
        "inkscape:flatsided",
        "inkscape:randomized",
        "inkscape:rounded",
        "inkscape:transform-center-y",
        "sodipodi:arg1",
        "sodipodi:arg2",
        "sodipodi:cx",
        "sodipodi:cy",
        "sodipodi:nodetypes",
        "sodipodi:r1",
        "sodipodi:r2",
        "sodipodi:sides",
        "sodipodi:type",
    ];
    for attribute in editor_attributes {
        let prefix = attribute.split_once(':').expect("prefixed attribute").0;
        let namespace = if prefix == "inkscape" {
            inkscape
        } else {
            sodipodi
        };
        let off_element =
            format!(r#"<svg xmlns:{prefix}="{namespace}"><g {attribute}="x"/></svg>"#);
        assert!(
            sanitize_svg(&off_element).is_err(),
            "accepted {attribute} off path"
        );
        let unsafe_value = format!(
            r#"<svg xmlns:{prefix}="{namespace}"><path {attribute}="url(https://example.invalid/a)" d="M0 0H1"/></svg>"#
        );
        assert!(
            sanitize_svg(&unsafe_value).is_err(),
            "accepted unsafe {attribute} value"
        );
    }

    let source_attributes = (0..24)
        .map(|index| format!(r#" x{index}="v""#))
        .collect::<String>();
    let oversized_source = format!(r#"<svg><path d="M0 0H1"{source_attributes}/></svg>"#);
    assert!(
        invalid_detail(sanitize_svg(&oversized_source)).contains("per-element attribute budget"),
        "path source envelope must remain bounded"
    );

    let oversized_canonical = concat!(
        r#"<svg><path d="M0 0H1" fill="red" fill-rule="nonzero" "#,
        r#"clip-rule="nonzero" fill-opacity="1" stroke="black" stroke-opacity="1" "#,
        r#"stroke-width="1" stroke-linecap="round" stroke-linejoin="round" "#,
        r#"stroke-miterlimit="4" stroke-dasharray="1 1" stroke-dashoffset="0"/></svg>"#,
    );
    let canonical = sanitize_svg(oversized_canonical)
        .expect("static v2 accepts bounded standard path presentation attributes");
    assert_static_render_equivalent(
        "bounded standard path presentation attributes",
        oversized_canonical,
        &canonical.svg,
    );

    let stylesheet =
        r#"<svg><style type="text/css">.st0{fill:red}</style><path class="st0" d="M0 0H1"/></svg>"#;
    let canonical = sanitize_svg(stylesheet).expect("bounded static stylesheet");
    assert_static_render_equivalent("bounded static stylesheet", stylesheet, &canonical.svg);
}

#[test]
fn bounded_internal_resources_preserve_all_five_upstream_renders() {
    let cases = [
        include_str!("../tests/fixtures/starvector/upstream-34847121425-case-01.svg"),
        include_str!("../tests/fixtures/starvector/upstream-34847121425-case-03.svg"),
        include_str!("../tests/fixtures/starvector/upstream-34847121425-case-04.svg"),
        include_str!("../tests/fixtures/starvector/upstream-34847121425-case-09.svg"),
        include_str!("../tests/fixtures/starvector/upstream-34847121425-case-17.svg"),
    ];
    for (index, raw) in cases.into_iter().enumerate() {
        let canonical = sanitize_svg(raw).unwrap_or_else(|error| {
            panic!("upstream internal-resource case {index} failed: {error}")
        });
        let tree = usvg::Tree::from_str(&canonical.svg, &usvg::Options::default())
            .expect("canonical comparison SVG");
        let intrinsic = tree.size();
        let scale = (512.0 / intrinsic.width()).min(512.0 / intrinsic.height());
        let x = (512.0 - intrinsic.width() * scale) / 2.0;
        let y = (512.0 - intrinsic.height() * scale) / 2.0;
        validate_filter_render_budget(
            &tree,
            512,
            512,
            resvg::tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, x, y),
        )
        .unwrap_or_else(|error| {
            panic!("upstream internal-resource case {index} exceeded preview budget: {error}")
        });
        assert_eq!(
            comparison_pixels(raw, 512),
            comparison_pixels(&canonical.svg, 512),
            "upstream internal-resource case {index} changed visible pixels"
        );
        assert_eq!(
            canonical.svg,
            sanitize_svg(raw).expect("repeat sanitation").svg,
            "upstream internal-resource case {index} is not byte deterministic"
        );
    }
}

#[test]
fn internal_resource_graph_is_typed_resolved_acyclic_and_forward_safe() {
    let forward = concat!(
        "<svg width=\"8\" height=\"8\"><rect width=\"8\" height=\"8\" fill=\"url(#g)\"/>",
        "<defs><linearGradient id=\"g\"><stop offset=\"0\" stop-color=\"red\"/>",
        "<stop offset=\"1\" stop-color=\"blue\"/></linearGradient></defs></svg>"
    );
    sanitize_svg(forward).expect("forward local resource reference");
    for invalid in [
        "<svg><path fill=\"url(#missing)\" d=\"M0 0H1\"/></svg>",
        "<svg><defs><clipPath id=\"x\"><rect/></clipPath></defs><path fill=\"url(#x)\" d=\"M0 0H1\"/></svg>",
        "<svg><defs><clipPath id=\"x\"><rect/></clipPath><filter id=\"x\" width=\"1\" height=\"1\"><feGaussianBlur in=\"SourceAlpha\" result=\"b\" stdDeviation=\"1\"/></filter></defs></svg>",
        "<svg><defs><linearGradient id=\"a\" href=\"#b\"/><linearGradient id=\"b\" href=\"#a\"/></defs></svg>",
        "<svg><defs><linearGradient id=\"a\" href=\"#a\"/></defs></svg>",
        "<svg><defs><clipPath id=\"c\"><rect clip-path=\"url(#c)\"/></clipPath></defs></svg>",
        "<svg><defs><linearGradient id=\"a\" href=\"#b\" xlink:href=\"#b\"/><linearGradient id=\"b\"/></defs></svg>",
        "<svg><defs><linearGradient id=\"a\" href=\"https://example.invalid/g\"/></defs></svg>",
        "<svg><defs><linearGradient id=\"a\"/></defs><path fill=\"url( #a )\" d=\"M0 0H1\"/></svg>",
        "<svg><defs><linearGradient id=\"a\"><stop offset=\"2\" stop-color=\"red\"/></linearGradient></defs></svg>",
        "<svg><defs><linearGradient id=\"a\"><stop offset=\"0\" style=\"stop-color:url(#a)\"/></linearGradient></defs></svg>",
    ] {
        assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
    }
    let unbound_xlink = "<svg><defs><linearGradient id=\"a\" xlink:href=\"#b\"/><linearGradient id=\"b\"/></defs></svg>";
    assert!(
        invalid_detail(sanitize_svg(unbound_xlink)).contains("exact root xmlns:xlink binding"),
        "unbound xlink prefix must fail closed"
    );
}

#[test]
fn internal_resource_and_filter_costs_are_independently_bounded() {
    let resources = (0..=MAX_SVG_RESOURCES)
        .map(|index| format!("<clipPath id=\"c{index}\"><rect/></clipPath>"))
        .collect::<String>();
    assert!(invalid_detail(sanitize_svg(&format!(
        "<svg><defs>{resources}</defs></svg>"
    )))
    .contains("resource budget"));

    let references = (0..=MAX_SVG_RESOURCE_REFERENCES)
        .map(|_| "<path fill=\"url(#g)\" d=\"M0 0H1\"/>".to_owned())
        .collect::<String>();
    let input = format!(
        "<svg><defs><linearGradient id=\"g\"><stop offset=\"0\" stop-color=\"red\"/></linearGradient></defs>{references}</svg>"
    );
    assert!(invalid_detail(sanitize_svg(&input)).contains("resource-reference budget"));

    let stops = "<stop offset=\"0\" stop-color=\"red\"/>".repeat(MAX_SVG_GRADIENT_STOPS + 1);
    assert!(invalid_detail(sanitize_svg(&format!(
        "<svg><defs><linearGradient id=\"g\">{stops}</linearGradient></defs></svg>"
    )))
    .contains("gradient-stop budget"));

    let mut primitives = String::new();
    let mut input = "SourceAlpha".to_owned();
    for index in 0..=MAX_SVG_FILTER_PRIMITIVES {
        let result = format!("r{index}");
        primitives.push_str(&format!(
            "<feOffset in=\"{input}\" result=\"{result}\" dx=\"1\" dy=\"1\"/>"
        ));
        input = result;
    }
    let svg = format!(
        "<svg width=\"8\" height=\"8\"><defs><filter id=\"f\" width=\"100%\" height=\"100%\">{primitives}</filter></defs></svg>"
    );
    assert!(invalid_detail(sanitize_svg(&svg)).contains("filter-primitive budget"));

    for invalid in [
        "<svg width=\"8\" height=\"8\"><defs><filter id=\"f\" width=\"100%\" height=\"100%\"><feGaussianBlur in=\"later\" result=\"b\" stdDeviation=\"1\"/></filter></defs></svg>",
        "<svg width=\"8\" height=\"8\"><defs><filter id=\"f\" width=\"100%\" height=\"100%\"><feGaussianBlur in=\"SourceAlpha\" result=\"b\" stdDeviation=\"65\"/></filter></defs></svg>",
        "<svg width=\"8\" height=\"8\"><defs><filter id=\"f\" width=\"100%\" height=\"100%\"><feOffset in=\"SourceAlpha\" result=\"b\"/><feOffset in=\"b\" result=\"b\"/></filter></defs></svg>",
    ] {
        assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
    }

    let scaled = "<svg width=\"900\" height=\"600\"><defs><filter id=\"f\" width=\"200%\" height=\"200%\"><feGaussianBlur in=\"SourceAlpha\" result=\"b\" stdDeviation=\"1\"/></filter></defs><g filter=\"url(#f)\" transform=\"scale(1000)\"><rect width=\"900\" height=\"600\"/></g></svg>";
    assert!(
        invalid_detail(sanitize_svg(scaled)).contains("offscreen-pixel budget"),
        "resolved transformed filter layer must be charged"
    );

    let filter = "<filter id=\"f\" width=\"100%\" height=\"100%\"><feGaussianBlur in=\"SourceAlpha\" result=\"b\" stdDeviation=\"1\"/></filter>";
    let applications = "<g filter=\"url(#f)\"><rect width=\"2048\" height=\"2048\"/></g>".repeat(3);
    let repeated =
        format!("<svg width=\"2048\" height=\"2048\"><defs>{filter}</defs>{applications}</svg>");
    let detail = invalid_detail(sanitize_svg(&repeated));
    assert!(
        detail.contains("offscreen-pixel budget"),
        "filter applications must be charged independently: {detail}"
    );

    let case_17 = sanitize_svg(include_str!(
        "../tests/fixtures/starvector/upstream-34847121425-case-17.svg"
    ))
    .expect("case 17 stays within its intrinsic render budget");
    let tree = usvg::Tree::from_str(&case_17.svg, &usvg::Options::default())
        .expect("case 17 canonical tree");
    let intrinsic = tree.size();
    let scale = (MAX_PREVIEW_DIMENSION as f32 / intrinsic.width())
        .min(MAX_PREVIEW_DIMENSION as f32 / intrinsic.height());
    let x = (MAX_PREVIEW_DIMENSION as f32 - intrinsic.width() * scale) / 2.0;
    let y = (MAX_PREVIEW_DIMENSION as f32 - intrinsic.height() * scale) / 2.0;
    let detail = validate_filter_render_budget(
        &tree,
        MAX_PREVIEW_DIMENSION,
        MAX_PREVIEW_DIMENSION,
        resvg::tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, x, y),
    )
    .expect_err("oversized case 17 preview must be rejected")
    .to_string();
    assert!(
        detail.contains("offscreen-pixel budget"),
        "preview transforms must be included in the allocation bound: {detail}"
    );
}

#[test]
fn hidden_geometry_is_omitted_but_remains_strictly_parsed_and_bounded() {
    let accepted = sanitize_svg(
        "<svg><g display=\"none\"><rect width=\"100%\" height=\"100%\" fill=\"url(#missing)\"/></g><path d=\"M0 0H1\"/></svg>",
    )
    .expect("bounded hidden geometry");
    assert!(!accepted.svg.contains("display"));
    assert!(!accepted.svg.contains("missing"));
    for invalid in [
        "<svg><g display=\"none\"><script/></g></svg>",
        "<svg><g display=\"none\"><use href=\"#x\"/></g></svg>",
        "<svg><g display=\"none\"><rect fill=\"url(https://example.invalid/a)\"/></g></svg>",
        "<svg><g display=\"none\" onclick=\"alert(1)\"><rect/></g></svg>",
    ] {
        assert!(sanitize_svg(invalid).is_err(), "accepted {invalid}");
    }
    let hidden_text = sanitize_svg("<svg><g display=\"none\"><text>hidden</text></g></svg>")
        .expect("hidden text has no font-dependent visible output");
    assert!(!hidden_text.svg.contains("text"));
    let paths = format!(
        "<svg><g display=\"none\"><path d=\"{}\"/></g></svg>",
        "M".repeat(MAX_SVG_PATH_COMMANDS + 1)
    );
    assert!(invalid_detail(sanitize_svg(&paths)).contains("path-command budget"));
}

#[test]
fn rejects_active_resources_css_references_and_text() {
    for malicious in [
        "<svg><script>alert(1)</script></svg>",
        "<svg><rect fill=\"url(https://example.invalid/a)\"/></svg>",
        "<svg><rect fill=\"u r l( data:image/png;base64,AA== )\"/></svg>",
        "<svg><rect fill=\"u\\72l(https://example.invalid/a)\"/></svg>",
        "<svg><rect fill=\"u/*escaped*/rl(https://example.invalid/a)\"/></svg>",
        "<svg><rect fill=\"file:///tmp/payload\"/></svg>",
        "<svg><rect fill=\"http://example.invalid/a\"/></svg>",
        "<svg><use href=\"#shape\"/></svg>",
        "<svg><rect/>not inert</svg>",
        "<svg><rect onclick=\"alert(1)\"/></svg>",
        "<svg xmlns=\"https://example.invalid/not-svg\"/>",
    ] {
        assert!(sanitize_svg(malicious).is_err(), "{malicious}");
    }
}

#[test]
fn enforces_byte_utf8_node_depth_and_attribute_budgets() {
    assert_eq!(
        invalid_detail(sanitize_svg_bytes(&vec![b' '; MAX_SVG_BYTES + 1])),
        "provider SVG exceeds the 256 KiB sanitizer budget"
    );
    assert_eq!(
        invalid_detail(sanitize_svg_bytes(&[0xff])),
        "provider SVG is not valid UTF-8"
    );

    let nodes = format!("<svg>{}</svg>", "<g/>".repeat(MAX_SVG_ELEMENTS));
    assert!(invalid_detail(sanitize_svg(&nodes)).contains("element budget"));

    let depth = format!(
        "<svg>{}{}</svg>",
        "<g>".repeat(MAX_SVG_DEPTH),
        "</g>".repeat(MAX_SVG_DEPTH)
    );
    assert!(invalid_detail(sanitize_svg(&depth)).contains("nesting budget"));

    let nine_attributes = concat!(
        " fill=\"#000\" fill-opacity=\"1\" stroke=\"#000\" stroke-opacity=\"1\"",
        " stroke-width=\"1\" stroke-linecap=\"round\" stroke-linejoin=\"round\"",
        " opacity=\"1\" transform=\"scale(1)\""
    );
    let elements = (MAX_SVG_ATTRIBUTES / 9) + 1;
    let attributes = format!(
        "<svg>{}</svg>",
        format!("<g{nine_attributes}/>").repeat(elements)
    );
    assert!(invalid_detail(sanitize_svg(&attributes)).contains("total attribute budget"));

    let per_element = concat!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1\" height=\"1\" viewBox=\"0 0 1 1\"",
        " fill=\"#000\" fill-opacity=\"1\" stroke=\"#000\" stroke-opacity=\"1\"",
        " stroke-width=\"1\" stroke-linecap=\"round\" stroke-linejoin=\"round\"",
        " opacity=\"1\" transform=\"scale(1)\"></svg>"
    );
    let canonical = sanitize_svg(per_element)
        .expect("static v2 accepts the bounded root presentation envelope");
    assert_static_render_equivalent(
        "bounded root presentation envelope",
        per_element,
        &canonical.svg,
    );

    let value = "x".repeat(MAX_SVG_ATTRIBUTE_VALUE_BYTES + 1);
    assert!(
        invalid_detail(sanitize_svg(&format!("<svg><g fill=\"{value}\"/></svg>")))
            .contains("per-value byte budget")
    );

    let value = "x".repeat(50_000);
    let total_values = format!(
        "<svg><g fill=\"{value}\"/><g fill=\"{value}\"/><g fill=\"{value}\"/><g fill=\"{value}\"/></svg>"
    );
    assert!(
        invalid_detail(sanitize_svg(&total_values)).contains("total attribute-value byte budget")
    );
}

#[test]
fn enforces_path_command_number_and_path_byte_budgets() {
    let commands = "M".repeat(MAX_SVG_PATH_COMMANDS + 1);
    assert!(invalid_detail(sanitize_svg(&format!(
        "<svg><path d=\"{commands}\"/></svg>"
    )))
    .contains("path-command budget"));

    let numbers_per_path = (MAX_SVG_PATH_NUMBERS / 3) + 1;
    let number_path = format!("M{}", " 0".repeat(numbers_per_path));
    let number_svg = format!(
        "<svg><path d=\"{number_path}\"/><path d=\"{number_path}\"/><path d=\"{number_path}\"/></svg>"
    );
    assert!(invalid_detail(sanitize_svg(&number_svg)).contains("path-number budget"));

    let bytes_per_path = (MAX_SVG_PATH_DATA_BYTES / 3) + 1;
    let byte_path = format!("M{}", " 0".repeat(bytes_per_path / 2));
    let byte_svg = format!(
        "<svg><path d=\"{byte_path}\"/><path d=\"{byte_path}\"/><path d=\"{byte_path}\"/></svg>"
    );
    assert!(invalid_detail(sanitize_svg(&byte_svg)).contains("path-data byte budget"));
}

#[test]
fn enforces_points_transform_coordinate_dimension_and_viewbox_budgets() {
    let points_per_element = (MAX_SVG_POINT_NUMBERS / 2) + 2;
    let points = "0 ".repeat(points_per_element);
    let points_svg =
        format!("<svg><polyline points=\"{points}\"/><polyline points=\"{points}\"/></svg>");
    assert!(invalid_detail(sanitize_svg(&points_svg)).contains("point-number budget"));

    let transform = "translate(0)".repeat(MAX_SVG_TRANSFORM_NUMBERS + 1);
    assert!(invalid_detail(sanitize_svg(&format!(
        "<svg><g transform=\"{transform}\"/></svg>"
    )))
    .contains("transform-number budget"));

    assert!(invalid_detail(sanitize_svg(
        "<svg><rect x=\"1000001\" width=\"1\" height=\"1\"/></svg>"
    ))
    .contains("coordinate-magnitude budget"));
    assert!(
        invalid_detail(sanitize_svg("<svg width=\"2049px\" height=\"1px\"></svg>"))
            .contains("dimensions must be")
    );
    assert!(
        invalid_detail(sanitize_svg("<svg viewBox=\"1000001 0 1 1\"></svg>"))
            .contains("viewBox origin")
    );
}

#[tokio::test]
async fn renders_a_bounded_png_preview_from_the_canonical_svg() {
    let canonical = sanitize_svg(
        "<svg width=\"12\" height=\"8\" xmlns=\"http://www.w3.org/2000/svg\"><rect width=\"12\" height=\"8\" fill=\"#ff0000\"/></svg>",
    )
    .expect("valid fixture");
    let temp = tempfile::tempdir().expect("temp dir");
    let path = temp.path().join("preview.png");
    let comparison_path = temp.path().join("comparison-512.png");
    render_preview(&canonical.svg, canonical.width, canonical.height, &path)
        .await
        .expect("preview writes");
    render_preview_with_size(
        &canonical.svg,
        canonical.width,
        canonical.height,
        &comparison_path,
        Some(TERMINAL_COMPARISON_SIZE),
    )
    .await
    .expect("comparison render writes");
    let preview = image::open(&path).expect("preview decodes").to_rgba8();
    let comparison = image::open(&comparison_path)
        .expect("comparison render decodes")
        .to_rgba8();
    assert_eq!(preview.dimensions(), (12, 8));
    assert_eq!(comparison.dimensions(), (512, 512));
    assert!(
        preview.get_pixel(0, 0)[0] > 200,
        "preview rendered the red rectangle"
    );
}
