//! Admission for bespoke Candle image routes that load one base model outside
//! `generate_candle_stream` (sc-16093).
//!
//! Built-in tiered models use the catalog's per-tier `candle.vramGbByTier` and
//! `candle.sequentialPeakGb` rows. Imported and ComfyUI checkpoints have no stable
//! catalog tier, so they use an explicitly weaker on-disk weights floor. Both checks
//! run before the handler hands control to `start_*_gen_stream`.

use super::*;

use crate::fit_gate::BYTES_PER_GIB;
use crate::vram_gate::LoadPlan;

fn source_path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path,
    }
}

fn path_weight_bytes(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => metadata.len(),
        Ok(metadata) if metadata.is_dir() => crate::mlx_fit_gate::sum_safetensors_bytes(path),
        _ => 0,
    }
}

/// Sum the exact files/directories a bespoke load consumes without double-counting a
/// file already covered by a recursively scanned directory.
fn distinct_weight_bytes(paths: &[&Path]) -> u64 {
    let mut ordered = paths.to_vec();
    ordered.sort_by_key(|path| path.as_os_str().len());
    let mut kept: Vec<&Path> = Vec::with_capacity(ordered.len());
    for path in ordered {
        if kept.iter().any(|parent| path.starts_with(parent)) {
            continue;
        }
        kept.push(path);
    }
    kept.into_iter().fold(0_u64, |total, path| {
        total.saturating_add(path_weight_bytes(path))
    })
}

/// Count only selected tensor subtrees from every safetensors file below `dir`.
/// Header failures fall back to the whole file so admission never underprices a
/// checkpoint merely because its metadata cannot be inspected.
pub(super) fn safetensors_tensor_bytes_with_prefixes(dir: &Path, prefixes: &[&str]) -> u64 {
    fn tensor_bytes(path: &Path, metadata: &std::fs::Metadata, prefixes: &[&str]) -> u64 {
        let Ok(header) = sceneworks_core::lora_family::read_safetensors_header(path) else {
            return metadata.len();
        };
        header.as_object().map_or(0, |entries| {
            entries.iter().fold(0_u64, |total, (name, tensor)| {
                if !prefixes.iter().any(|prefix| name.starts_with(prefix)) {
                    return total;
                }
                let Some(offsets) = tensor.get("data_offsets").and_then(Value::as_array) else {
                    return total;
                };
                let Some(start) = offsets.first().and_then(Value::as_u64) else {
                    return total;
                };
                let Some(end) = offsets.get(1).and_then(Value::as_u64) else {
                    return total;
                };
                total.saturating_add(end.saturating_sub(start))
            })
        })
    }

    fn visit(
        dir: &Path,
        prefixes: &[&str],
        visited_dirs: &mut std::collections::HashSet<PathBuf>,
    ) -> u64 {
        // Follow HF shard symlinks, but canonicalize directory identities so an
        // operator-provided junction/symlink cycle cannot recurse forever.
        let Ok(canonical_dir) = std::fs::canonicalize(dir) else {
            return 0;
        };
        if !visited_dirs.insert(canonical_dir.clone()) {
            return 0;
        }
        let Ok(entries) = std::fs::read_dir(canonical_dir) else {
            return 0;
        };
        entries.flatten().fold(0_u64, |total, entry| {
            let path = entry.path();
            let Ok(metadata) = std::fs::metadata(&path) else {
                return total;
            };
            if metadata.is_dir() {
                return total.saturating_add(visit(&path, prefixes, visited_dirs));
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.ends_with(".safetensors") || name.starts_with("._") {
                return total;
            }
            total.saturating_add(tensor_bytes(&path, &metadata, prefixes))
        })
    }

    visit(dir, prefixes, &mut std::collections::HashSet::new())
}

fn reject_message(
    model: &str,
    lane: &str,
    tier: Option<&str>,
    needed_gb: f64,
    available_gb: f64,
    gpu_id: &str,
    catalog_evidence: bool,
) -> WorkerError {
    let tier = tier.map_or_else(String::new, |tier| format!(" at the {tier} tier"));
    let evidence = if catalog_evidence {
        "the per-tier catalog peak (including headroom)"
    } else {
        "at least the on-disk weights plus headroom; activations are not measured for this external checkpoint"
    };
    WorkerError::InvalidPayload(format!(
        "{model}{tier} cannot run through the {lane} lane: {evidence} needs ~{} GB of VRAM, but GPU \
         {gpu_id} has ~{} GB available. Select a smaller checkpoint/tier or use a GPU with more VRAM.",
        needed_gb.round() as i64,
        available_gb.round() as i64,
    ))
}

fn has_tier_peak_row(entry: &JsonObject, tier: &str) -> bool {
    let rows = entry
        .get("candle")
        .and_then(|candle| candle.get("vramGbByTier"))
        .and_then(Value::as_object);
    let numeric = |key: &str| rows.and_then(|rows| rows.get(key)).and_then(Value::as_f64);
    numeric(tier).is_some() || (tier == NVFP4_TIER && numeric("q8").is_some())
}

fn builtin_model_entry(id: &str) -> Option<&'static JsonObject> {
    static MANIFEST: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    let manifest = MANIFEST.get_or_init(|| {
        let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, contents)| *contents)
            .expect("builtin.models.jsonc is embedded");
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
            .expect("embedded builtin.models.jsonc parses")
    });
    manifest
        .get("models")?
        .as_array()?
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(id))?
        .as_object()
}

/// Gate one built-in bespoke route on the catalog peak for the tier that its resolved
/// directory actually names. Returns the residency policy the caller must carry into
/// its `LoadSpec`; direct resident-only providers pass `sequential_capable = false` and
/// ignore the returned (necessarily resident) value. An `Alias` records the canonical
/// builtin whose identical base weights supply an alias route's evidence.
pub(super) enum CandleBaseEvidence {
    Catalog,
    Alias(&'static str),
    Ungateable(&'static str),
}

pub(super) async fn admit_candle_base(
    request: &ImageRequest,
    settings: &Settings,
    resolved_dir: &Path,
    lane: &'static str,
    evidence: CandleBaseEvidence,
    adapter_resident_bytes: u64,
    sequential_capable: bool,
) -> WorkerResult<gen_core::OffloadPolicy> {
    let (evidence_entry, unmeasured_reason) = match evidence {
        CandleBaseEvidence::Alias(id) => (builtin_model_entry(id).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{model} cannot run through the {lane} lane because its catalog evidence alias {id} is missing",
                model = request.model,
            ))
        })?, None),
        CandleBaseEvidence::Catalog => (&request.model_manifest_entry, None),
        CandleBaseEvidence::Ungateable(reason) => (&request.model_manifest_entry, Some(reason)),
    };
    let tier = gate_tier_key(
        false,
        resolved_dir,
        &request.advanced,
        &request.model_manifest_entry,
        nvfp4_selected(request, nvfp4_host_eligible(), Some(resolved_dir)),
    );
    if !has_tier_peak_row(evidence_entry, tier) {
        if let Some(reason) = unmeasured_reason {
            tracing::warn!(
                model = %request.model,
                lane,
                tier,
                reason,
                "candle base admission: explicitly un-gateable for this request; admitting without a \
                 per-tier catalog peak (sc-16093)"
            );
            return Ok(gen_core::OffloadPolicy::Resident);
        }
        return Err(WorkerError::InvalidPayload(format!(
            "{model} cannot run through the {lane} lane because its resolved {tier} tier has no \
             candle.vramGbByTier catalog peak. This is a model-catalog error; reinstall or update \
             SceneWorks before retrying.",
            model = request.model,
        )));
    }
    let needed = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        evidence_entry,
        tier,
        adapter_resident_bytes,
    );
    let resident_peak_gb = needed.expect("a resident tier catalog row predicts a peak");
    let sequential_needed = sequential_capable
        .then(|| {
            crate::vram_gate::predicted_sequential_peak_gb_with_adapter_bytes(
                evidence_entry,
                tier,
                adapter_resident_bytes,
            )
        })
        .flatten();
    if sequential_capable && sequential_needed.is_none() {
        tracing::warn!(
            model = %request.model,
            lane,
            tier,
            "candle base admission: resident peak is cataloged but this sequential-capable tier has no \
             candle.sequentialPeakGb row; resident overflow will stage best-effort (sc-16093)"
        );
    }

    let raw_budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    let (plan, budget) = gate_with_evict_reclaim(
        &settings.gpu_id,
        raw_budget,
        |budget| crate::vram_gate::load_plan(needed, sequential_needed, budget, sequential_capable),
        |raw, reclaimed| raw != reclaimed,
    )
    .await?;
    let available_gb = budget.map_or(0.0, |budget| budget.free_gb);

    match plan {
        LoadPlan::Resident => {
            crate::vram_gate::note_loaded_peak(&settings.gpu_id, resident_peak_gb);
            Ok(gen_core::OffloadPolicy::Resident)
        }
        LoadPlan::Sequential => {
            if let Some(peak_gb) = sequential_needed {
                crate::vram_gate::note_loaded_peak(&settings.gpu_id, peak_gb);
            }
            tracing::info!(
                model = %request.model,
                lane,
                tier,
                available_gb,
                "candle base admission: selected sequential component residency (sc-16093)"
            );
            Ok(gen_core::OffloadPolicy::Sequential)
        }
        LoadPlan::Reject => {
            let rejected_peak = crate::vram_gate::sequential_overflow_gb(sequential_needed, budget)
                .unwrap_or(resident_peak_gb);
            Err(reject_message(
                &request.model,
                lane,
                Some(tier),
                rejected_peak,
                available_gb,
                &settings.gpu_id,
                true,
            ))
        }
    }
}

/// Gate an imported/ComfyUI base whose user-owned paths have no stable catalog row.
/// This is intentionally a floor: it can reject only when the weights alone cannot fit.
pub(super) async fn admit_candle_base_floor(
    model: &str,
    lane: &'static str,
    settings: &Settings,
    paths: &[&Path],
) -> WorkerResult<()> {
    let bytes = distinct_weight_bytes(paths);
    let needed = (bytes > 0).then(|| bytes as f64 / BYTES_PER_GIB + crate::vram_gate::HEADROOM_GB);
    let Some(floor_gb) = needed else {
        tracing::warn!(
            model,
            lane,
            "candle base admission: explicitly un-gateable because the external checkpoint paths contain \
             no countable weights; admitting without a floor (sc-16093)"
        );
        return Ok(());
    };
    let raw_budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    let (plan, budget) = gate_with_evict_reclaim(
        &settings.gpu_id,
        raw_budget,
        |budget| crate::vram_gate::load_plan(needed, None, budget, false),
        |raw, reclaimed| raw != reclaimed,
    )
    .await?;
    match plan {
        LoadPlan::Resident => {
            crate::vram_gate::note_loaded_peak(&settings.gpu_id, floor_gb);
            tracing::info!(
                model,
                lane,
                floor_gb,
                "candle base admission: external checkpoint admitted on its on-disk weights floor; \
                 activation peaks remain unmeasured (sc-16093)"
            );
            Ok(())
        }
        LoadPlan::Sequential => unreachable!("a floor-only route is never sequential-capable"),
        LoadPlan::Reject => Err(reject_message(
            model,
            lane,
            None,
            floor_gb,
            budget.map_or(0.0, |budget| budget.free_gb),
            &settings.gpu_id,
            false,
        )),
    }
}

/// Imported SDXL already materializes a `LoadSpec` containing its external checkpoint,
/// adapters, PiD weights, and caller-staged components. Price exactly those sources.
pub(super) async fn admit_candle_load_spec_floor(
    model: &str,
    lane: &'static str,
    settings: &Settings,
    spec: &LoadSpec,
) -> WorkerResult<()> {
    let mut paths = vec![source_path(&spec.weights)];
    paths.extend(spec.adapters.iter().map(|adapter| adapter.path.as_path()));
    paths.extend(spec.components.values().map(source_path));
    if let Some(pid) = &spec.pid {
        paths.push(source_path(&pid.checkpoint));
        paths.push(source_path(&pid.gemma));
    }
    admit_candle_base_floor(model, lane, settings, &paths).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn catalog_evidence_requires_a_per_tier_row_not_the_static_minimum() {
        let static_only = json!({ "candle": { "minMemoryGb": 24 } })
            .as_object()
            .unwrap()
            .clone();
        assert!(!has_tier_peak_row(&static_only, "q4"));

        let evidenced = json!({
            "candle": { "vramGbByTier": { "q4": 18.0, "q8": 24.0 } }
        })
        .as_object()
        .unwrap()
        .clone();
        assert!(has_tier_peak_row(&evidenced, "q4"));
        assert!(
            has_tier_peak_row(&evidenced, NVFP4_TIER),
            "NVFP4 conservatively reuses the catalog q8 row, matching predicted_peak_gb"
        );
        assert!(!has_tier_peak_row(&evidenced, "bf16"));
    }

    #[test]
    fn floor_deduplicates_nested_paths_and_counts_files() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model");
        std::fs::create_dir_all(&model).unwrap();
        let shard = model.join("model.safetensors");
        std::fs::write(&shard, vec![0_u8; 17]).unwrap();
        let external = root.path().join("external.safetensors");
        std::fs::write(&external, vec![0_u8; 23]).unwrap();
        assert_eq!(
            distinct_weight_bytes(&[&model, &shard, &external]),
            40,
            "the shard inside model/ is counted once; the external file is additive"
        );
    }

    #[test]
    fn tensor_prefix_accounting_counts_only_selected_subtrees() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("model.safetensors");
        let header = json!({
            "visual.block.weight": { "dtype": "F32", "shape": [2], "data_offsets": [0, 8] },
            "language_model.weight": { "dtype": "F32", "shape": [4], "data_offsets": [8, 24] },
            "encoder.conv.weight": { "dtype": "F32", "shape": [3], "data_offsets": [24, 36] },
            "decoder.conv.weight": { "dtype": "F32", "shape": [5], "data_offsets": [36, 56] }
        });
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&encoded);
        bytes.extend_from_slice(&[0_u8; 56]);
        std::fs::write(file, bytes).unwrap();

        assert_eq!(
            safetensors_tensor_bytes_with_prefixes(root.path(), &["visual.", "encoder."]),
            20,
            "language-model and decoder tensors are not charged to edit-only residency"
        );
    }
}
