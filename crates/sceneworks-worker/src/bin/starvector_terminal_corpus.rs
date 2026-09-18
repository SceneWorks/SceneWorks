use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use resvg::usvg;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const LIMIT_CASE_CONTRACT: &str =
    include_str!("../../../../scripts/lib/starvector-terminal-limit-cases.json");

fn shipping_detail_budgets() -> Result<Value, Box<dyn std::error::Error>> {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .ok_or_else(|| fail("builtin.models.jsonc is not compiled into SceneWorks"))?;
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))?;
    let models = manifest
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| fail("builtin.models.jsonc has no models array"))?;
    let mut budgets = serde_json::Map::new();
    for (tier, model_id) in [("1b", "starvector_1b"), ("8b", "starvector_8b")] {
        let matches = models
            .iter()
            .filter(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(fail(format!(
                "shipping manifest must contain one {model_id}"
            )));
        }
        let vector = matches[0]
            .get("vector")
            .and_then(Value::as_object)
            .ok_or_else(|| fail(format!("{model_id} lacks vector limits")))?;
        let max_tokens = vector.get("maxNewTokens").and_then(Value::as_u64);
        let max_bytes = vector.get("maxSvgBytes").and_then(Value::as_u64);
        let max_time = vector.get("maxWallTimeMs").and_then(Value::as_u64);
        if !matches!(max_tokens, Some(1..=16384))
            || !matches!(max_bytes, Some(1..=1_048_576))
            || !matches!(max_time, Some(1..=3_600_000))
        {
            return Err(fail(format!("{model_id} has invalid vector limits")));
        }
        budgets.insert(
            tier.to_owned(),
            json!({"maxNewTokens": max_tokens, "maxSvgBytes": max_bytes, "maxWallTimeMs": max_time}),
        );
    }
    Ok(Value::Object(budgets))
}

fn materialize_limit_cases(
    shipping_budgets: &Value,
) -> Result<serde_json::Map<String, Value>, Box<dyn std::error::Error>> {
    let contract: Value = serde_json::from_str(LIMIT_CASE_CONTRACT)?;
    if contract.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err(fail("limit-case contract schema is unsupported"));
    }
    let sampling = contract
        .get("sampling")
        .ok_or_else(|| fail("limit-case sampling is missing"))?;
    if sampling
        != &json!({"temperature": 0.0, "topP": 1.0, "topK": 1, "repetitionPenalty": 1.0, "seed": 7})
    {
        return Err(fail("limit-case deterministic sampling drifted"));
    }
    let scenarios = contract
        .get("scenarios")
        .and_then(Value::as_array)
        .ok_or_else(|| fail("limit-case scenarios are missing"))?;
    let expected = [
        ("completion", 12_u64),
        ("token", 11),
        ("byte", 11),
        ("wall_time", 11),
        ("queued_cancellation", 11),
        ("in_flight_cancellation", 11),
    ];
    if scenarios.len() != expected.len() {
        return Err(fail("limit-case contract must contain six scenarios"));
    }
    for (index, ((name, source_case_index), template)) in expected.iter().zip(scenarios).enumerate()
    {
        if template.get("scenario").and_then(Value::as_str) != Some(*name)
            || template.get("source_case_index").and_then(Value::as_u64) != Some(*source_case_index)
        {
            return Err(fail(format!(
                "limit-case scenario {index} identity drifted"
            )));
        }
    }
    if scenarios[4]
        .get("cancel_after_create")
        .and_then(Value::as_bool)
        != Some(true)
        || scenarios[4].get("worker_unloaded").and_then(Value::as_bool) != Some(true)
        || scenarios[5]
            .get("cancel_after_progress")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(fail("limit-case cancellation controls are incomplete"));
    }

    let tuples = ["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"];
    let mut materialized = serde_json::Map::new();
    for tuple in tuples {
        let tier = tuple
            .split_once(':')
            .map(|(_, tier)| tier)
            .ok_or_else(|| fail("limit-case tuple lacks a tier"))?;
        let shipping = shipping_budgets
            .get(tier)
            .ok_or_else(|| fail(format!("shipping budget for {tier} is missing")))?;
        if contract.pointer(&format!("/shipping_detail_budgets/{tier}")) != Some(shipping) {
            return Err(fail(format!(
                "limit-case {tier} shipping budget drifted from the compiled manifest"
            )));
        }
        let mut records = Vec::with_capacity(scenarios.len());
        for (case_index, template) in scenarios.iter().enumerate() {
            let mut detail_budget = shipping
                .as_object()
                .cloned()
                .ok_or_else(|| fail(format!("shipping budget for {tier} is invalid")))?;
            let overrides = template
                .get("detail_budget_overrides")
                .and_then(Value::as_object)
                .ok_or_else(|| fail(format!("limit-case scenario {case_index} lacks a budget")))?;
            for (key, value) in overrides {
                if !detail_budget.contains_key(key) || value.as_u64().is_none() {
                    return Err(fail(format!(
                        "limit-case scenario {case_index} has an invalid budget override"
                    )));
                }
                detail_budget.insert(key.clone(), value.clone());
            }
            let scenario = template["scenario"]
                .as_str()
                .expect("validated limit scenario");
            let mut record = json!({
                "case_id": format!("limit-{tuple}-{scenario}"),
                "case_index": case_index,
                "source_case_index": template["source_case_index"],
                "scenario": scenario,
                "sampling": sampling,
                "detailBudget": detail_budget,
            });
            let object = record.as_object_mut().expect("limit record object");
            for key in [
                "cancel_after_create",
                "cancel_after_progress",
                "worker_unloaded",
            ] {
                if let Some(value) = template.get(key) {
                    object.insert(key.to_owned(), value.clone());
                }
            }
            records.push(record);
        }
        materialized.insert(tuple.to_owned(), Value::Array(records));
    }
    Ok(materialized)
}

struct StagingDir(PathBuf);

impl Drop for StagingDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fail(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn row_string(row: &parquet::record::Row, requested: &str) -> Option<String> {
    row.get_column_iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(requested))
        .and_then(|(_, field)| match field {
            Field::Str(value) => Some(value.clone()),
            Field::Bytes(value) => std::str::from_utf8(value.data()).ok().map(str::to_owned),
            Field::Null | Field::Group(_) | Field::ListInternal(_) | Field::MapInternal(_) => None,
            primitive => Some(primitive.to_string()),
        })
}

fn field_kind(field: &Field) -> &'static str {
    match field {
        Field::Null => "null",
        Field::Bool(_) => "bool",
        Field::Byte(_) => "byte",
        Field::Short(_) => "short",
        Field::Int(_) => "int",
        Field::Long(_) => "long",
        Field::UByte(_) => "ubyte",
        Field::UShort(_) => "ushort",
        Field::UInt(_) => "uint",
        Field::ULong(_) => "ulong",
        Field::Float16(_) => "float16",
        Field::Float(_) => "float",
        Field::Double(_) => "double",
        Field::Decimal(_) => "decimal",
        Field::Str(_) => "string",
        Field::Bytes(_) => "bytes",
        Field::Date(_) => "date",
        Field::TimestampMillis(_) => "timestamp-millis",
        Field::TimestampMicros(_) => "timestamp-micros",
        Field::Group(_) => "group",
        Field::ListInternal(_) => "list",
        Field::MapInternal(_) => "map",
    }
}

fn write_png(svg: &str, destination: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default())
        .map_err(|error| fail(format!("source SVG cannot be rendered: {error}")))?;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(512, 512)
        .ok_or_else(|| fail("fixed 512px pixmap allocation failed"))?;
    pixmap.fill(resvg::tiny_skia::Color::WHITE);
    let source = tree.size();
    let scale = (512.0 / source.width()).min(512.0 / source.height());
    let x = (512.0 - source.width() * scale) / 2.0;
    let y = (512.0 - source.height() * scale) / 2.0;
    let transform = resvg::tiny_skia::Transform::from_row(scale, 0.0, 0.0, scale, x, y);
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    pixmap.save_png(destination)?;
    Ok(fs::read(destination)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 8 {
        return Err(fail("usage: <corpus-json> <parquet-root> <assets-root> <inference-revision> <raster-provider> <raster-model> <raster-revision> <raster-inventory-sha256>"));
    }
    let corpus_path = PathBuf::from(&arguments[0]);
    let parquet_root = PathBuf::from(&arguments[1]);
    let final_assets_root = PathBuf::from(&arguments[2]);
    let strings: Vec<String> = arguments[3..]
        .iter()
        .map(|value| value.to_string_lossy().into_owned())
        .collect();
    let [inference_revision, raster_provider, raster_model, raster_revision, raster_inventory]: [String; 5] =
        strings.try_into().map_err(|_| fail("invalid identity arguments"))?;
    if !is_lower_hex(&inference_revision, 40)
        || !is_lower_hex(&raster_inventory, 64)
        || [
            raster_provider.as_str(),
            raster_model.as_str(),
            raster_revision.as_str(),
        ]
        .iter()
        .any(|value| value.is_empty())
    {
        return Err(fail("model and inference identities are incomplete"));
    }
    let corpus: Value = serde_json::from_slice(&fs::read(&corpus_path)?)?;
    let detail_budgets = shipping_detail_budgets()?;
    let sources = corpus
        .pointer("/upstream_image_quality_cases/sources")
        .and_then(Value::as_array)
        .ok_or_else(|| fail("corpus source list is missing"))?;
    if sources.len() != 4 {
        return Err(fail("corpus must contain exactly four sources"));
    }
    if final_assets_root.exists() {
        return Err(fail("assets destination must not already exist"));
    }
    let parent = final_assets_root
        .parent()
        .ok_or_else(|| fail("assets destination must have a parent directory"))?;
    fs::create_dir_all(parent)?;
    let name = final_assets_root
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| fail("assets destination name is invalid"))?;
    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let assets_root = parent.join(format!(".{name}.staging-{}-{unique}", std::process::id()));
    let staging = StagingDir(assets_root.clone());
    for name in ["source-svg", "input-png", "reference-png"] {
        fs::create_dir_all(assets_root.join(name))?;
    }
    let mut rows = Vec::with_capacity(120);
    for (source_index, source) in sources.iter().enumerate() {
        let dataset = source
            .get("dataset")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("dataset missing"))?;
        let revision = source
            .get("revision")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("revision missing"))?;
        let expected = source
            .get("parquet_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("parquet digest missing"))?;
        let parquet_path = parquet_root.join(format!("source-{source_index}.parquet"));
        let parquet_bytes = fs::read(&parquet_path)?;
        if digest(&parquet_bytes) != expected {
            return Err(fail(format!(
                "source {source_index} parquet digest mismatch"
            )));
        }
        let reader = SerializedFileReader::new(File::open(&parquet_path)?)?;
        let mut parquet_rows = reader.get_row_iter(None)?;
        for row_index in 0..30 {
            let row = parquet_rows
                .next()
                .ok_or_else(|| fail(format!("source {source_index} has fewer than 30 rows")))??;
            let columns = row
                .get_column_iter()
                .map(|(name, field)| format!("{name}={}", field_kind(field)))
                .collect::<Vec<_>>()
                .join(",");
            let filename = row_string(&row, "Filename")
                .ok_or_else(|| fail(format!("Filename column missing; observed {columns}")))?;
            let svg = row_string(&row, "Svg")
                .ok_or_else(|| fail(format!("Svg column missing; observed {columns}")))?;
            let case_index = source_index * 30 + row_index;
            let svg_relative = format!("source-svg/{case_index:03}.svg");
            let input_relative = format!("input-png/{case_index:03}.png");
            let reference_relative = format!("reference-png/{case_index:03}.png");
            fs::write(assets_root.join(&svg_relative), svg.as_bytes())?;
            let png = write_png(&svg, &assets_root.join(&input_relative))?;
            fs::write(assets_root.join(&reference_relative), &png)?;
            rows.push(json!({
                "case_index": case_index, "dataset": dataset, "revision": revision,
                "row_index": row_index, "filename": filename,
                "svg_path": svg_relative, "svg_sha256": digest(svg.as_bytes()),
                "input_png_path": input_relative, "png_sha256": digest(&png),
                "reference_png": reference_relative, "reference_png_sha256": digest(&png),
                "sampling": {"temperature": 0.0, "topP": 1.0, "topK": 1, "repetitionPenalty": 1.0, "seed": 7},
                "detail_budgets": detail_budgets
            }));
        }
    }
    let canonical = rows
        .iter()
        .map(|row| {
            format!(
        "{{\"dataset\":{},\"revision\":{},\"row_index\":{},\"filename\":{},\"svg_sha256\":{}}}",
        row["dataset"], row["revision"], row["row_index"], row["filename"], row["svg_sha256"]
    )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let expected_rows = corpus
        .pointer("/upstream_image_quality_cases/row_identity_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| fail("row identity missing"))?;
    let observed_rows = digest(format!("{canonical}\n").as_bytes());
    if observed_rows != expected_rows {
        let filenames = rows
            .iter()
            .take(5)
            .map(|row| row["filename"].as_str().unwrap_or("?"))
            .collect::<Vec<_>>()
            .join(",");
        let first_row =
            SerializedFileReader::new(File::open(parquet_root.join("source-0.parquet"))?)?
                .get_row_iter(None)?
                .next()
                .ok_or_else(|| fail("source 0 is empty"))??;
        let first_fields = first_row
            .get_column_iter()
            .map(|(name, field)| format!("{name}={}", field_kind(field)))
            .collect::<Vec<_>>()
            .join(",");
        let source_hashes = (0..4)
            .map(|source_index| {
                let records = canonical
                    .lines()
                    .skip(source_index * 30)
                    .take(30)
                    .collect::<Vec<_>>()
                    .join("\n");
                digest(format!("{records}\n").as_bytes())
            })
            .collect::<Vec<_>>()
            .join(",");
        return Err(fail(format!(
            "materialized row identity {observed_rows} mismatches corpus {expected_rows}; source identities {source_hashes}; first fields {first_fields}; first filenames {filenames}"
        )));
    }
    for (source_index, source) in sources.iter().enumerate() {
        let records = canonical
            .lines()
            .skip(source_index * 30)
            .take(30)
            .collect::<Vec<_>>()
            .join("\n");
        let observed = digest(format!("{records}\n").as_bytes());
        let expected = source
            .get("row_identity_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("source row identity missing"))?;
        if observed != expected {
            return Err(fail(format!(
                "source {source_index} row identity {observed} mismatches corpus {expected}"
            )));
        }
    }
    let parity = (0..4)
        .flat_map(|source_index| {
            canonical
                .lines()
                .skip(source_index * 30)
                .take(5)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let observed_parity = digest(format!("{parity}\n").as_bytes());
    let expected_parity = corpus
        .pointer("/deterministic_parity_cases/row_identity_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| fail("parity row identity missing"))?;
    if observed_parity != expected_parity {
        return Err(fail(format!(
            "parity row identity {observed_parity} mismatches corpus {expected_parity}"
        )));
    }
    let tuples = ["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"];
    let lifecycle = tuples.iter().map(|tuple| (tuple.to_string(), Value::Array(["load", "unload", "reload", "memory_reported"].iter().enumerate().map(|(index, operation)| json!({"case_id": format!("lifecycle-{tuple}-{index}"), "operation": operation, "case_index": index})).collect()))).collect::<serde_json::Map<_, _>>();
    let limits = materialize_limit_cases(&detail_budgets)?;
    let names = [
        "geometric badge",
        "isometric folder",
        "rounded calendar",
        "minimal rocket",
        "layered landscape",
        "abstract flower",
    ];
    let prompts = (0..60).map(|case_index| {
        let prompt = format!("Create a {} vector illustration, variant {}, with clear silhouette, balanced composition, and no text.", names[case_index / 10], case_index % 10);
        json!({"case_index": case_index, "case_id": format!("prompt-v1-{case_index}"), "prompt_sha256": digest(prompt.as_bytes()), "prompt": prompt,
            "raster_provider_id": raster_provider, "raster_model": raster_model, "vector_model": "starvector_8b",
            "expected_raster_revision": raster_revision, "expected_vector_revision": "518beea8dcb5f7a37c5911e92d1d62a76beee7f9",
            "raster_inventory_sha256": raster_inventory, "seed": case_index, "width": 512, "height": 512,
            "sampling": {"temperature": 0.0, "topP": 1.0, "topK": 1, "repetitionPenalty": 1.0, "seed": case_index},
            "detail_budget": detail_budgets["8b"]})
    }).collect::<Vec<_>>();
    let prompt_identity = digest(
        prompts
            .iter()
            .map(|row| row["prompt_sha256"].as_str().unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            .as_bytes(),
    );
    let expected_prompts = corpus
        .pointer("/sceneworks_owned_suites/prompt_composition/content_identity_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| fail("prompt identity missing"))?;
    if prompt_identity != expected_prompts {
        return Err(fail("generated prompt identity mismatches corpus"));
    }
    let index = json!({"schema_version": 2, "inference_revision": inference_revision, "row_identity_sha256": expected_rows,
        "rows": rows, "lifecycle_cases": lifecycle, "limit_cases": limits, "prompt_composition": prompts});
    fs::write(
        assets_root.join("starvector-terminal-row-index-v1.json"),
        format!("{}\n", serde_json::to_string_pretty(&index)?),
    )?;
    fs::rename(&assets_root, &final_assets_root)?;
    drop(staging);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{materialize_limit_cases, shipping_detail_budgets, write_png};

    #[test]
    fn terminal_corpus_centers_non_square_renders_on_the_fixed_canvas() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("wide.png");
        let bytes = write_png(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="2"><rect width="4" height="2" fill="#ff0000"/></svg>"##,
            &destination,
        )
        .unwrap();
        let pixels = image::load_from_memory(&bytes).unwrap().to_rgba8();

        assert_eq!(pixels.dimensions(), (512, 512));
        assert_eq!(pixels.get_pixel(256, 0).0, [255, 255, 255, 255]);
        assert_eq!(pixels.get_pixel(256, 127).0, [255, 255, 255, 255]);
        assert_eq!(pixels.get_pixel(256, 128).0, [255, 0, 0, 255]);
        assert_eq!(pixels.get_pixel(256, 383).0, [255, 0, 0, 255]);
        assert_eq!(pixels.get_pixel(256, 384).0, [255, 255, 255, 255]);
        assert_eq!(pixels.get_pixel(256, 511).0, [255, 255, 255, 255]);
    }

    #[test]
    fn terminal_detailed_budgets_come_from_the_embedded_shipping_manifest() {
        let budgets = shipping_detail_budgets().unwrap();
        assert_eq!(budgets["1b"]["maxNewTokens"], 7933);
        assert_eq!(budgets["8b"]["maxNewTokens"], 15422);
        for tier in ["1b", "8b"] {
            assert_eq!(budgets[tier]["maxSvgBytes"], 262144);
            assert_eq!(budgets[tier]["maxWallTimeMs"], 300000);
        }
    }

    #[test]
    fn terminal_limit_cases_are_executable_and_tier_specific() {
        let budgets = shipping_detail_budgets().unwrap();
        let limits = materialize_limit_cases(&budgets).unwrap();
        for tuple in ["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"] {
            let records = limits[tuple].as_array().unwrap();
            assert_eq!(records.len(), 6);
            assert_eq!(records[0]["scenario"], "completion");
            assert_eq!(records[0]["source_case_index"], 12);
            assert_eq!(records[1]["scenario"], "token");
            assert_eq!(records[1]["source_case_index"], 11);
            assert_eq!(records[1]["detailBudget"]["maxNewTokens"], 16);
            assert_eq!(records[2]["detailBudget"]["maxSvgBytes"], 1024);
            assert_eq!(records[3]["detailBudget"]["maxWallTimeMs"], 1000);
            assert_eq!(records[4]["cancel_after_create"], true);
            assert_eq!(records[4]["worker_unloaded"], true);
            assert_eq!(records[5]["cancel_after_progress"], true);
            for record in records {
                assert_eq!(record["sampling"]["temperature"], 0.0);
                assert_eq!(record["sampling"]["topP"], 1.0);
                assert_eq!(record["sampling"]["topK"], 1);
                assert_eq!(record["sampling"]["repetitionPenalty"], 1.0);
                assert_eq!(record["sampling"]["seed"], 7);
                assert!(record.get("finish_reason").is_none());
            }
        }
    }
}
