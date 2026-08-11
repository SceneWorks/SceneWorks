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

/// Gate one built-in bespoke route on the catalog peak for the tier that its resolved
/// directory actually names. Returns the residency policy the caller must carry into
/// its `LoadSpec`; direct resident-only providers pass `sequential_capable = false` and
/// ignore the returned (necessarily resident) value.
pub(super) enum CandleBaseEvidence {
    Catalog,
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

/// A live-VRAM floor bound to a registered generator cache miss.
///
/// The runtime handle is captured on the async route before the closure crosses to the dedicated
/// cache thread. [`Self::admit`] can then query `nvidia-smi` only when the cache reports a real miss,
/// while the old different-key entry is still resident and the raw free-VRAM reading is meaningful.
pub(super) struct CachedCandleBaseFloorAdmission {
    model: String,
    lane: &'static str,
    gpu_id: String,
    source_weight_bytes: u64,
    runtime: tokio::runtime::Handle,
}

impl CachedCandleBaseFloorAdmission {
    /// Exact conservative source-weight bytes to bind to the generator if this admission succeeds.
    /// The transient headroom added to the incoming floor is deliberately excluded: evicting a
    /// resident generator cannot promise to reclaim that allowance.
    pub(super) fn reclaimable_weight_bytes(&self) -> u64 {
        self.source_weight_bytes
    }

    /// Admit one cold load. `resident_reclaimable_weight_bytes` comes from the exact different-key
    /// cache entry that remains alive until this gate succeeds. A historical process-global peak is
    /// not valid credit: after large -> small, it can describe memory already returned to the driver.
    pub(super) fn admit(self, resident_reclaimable_weight_bytes: u64) -> WorkerResult<()> {
        let Some(floor_gb) = (self.source_weight_bytes > 0).then(|| {
            self.source_weight_bytes as f64 / BYTES_PER_GIB + crate::vram_gate::HEADROOM_GB
        }) else {
            tracing::warn!(
                model = self.model,
                lane = self.lane,
                "candle base admission: explicitly un-gateable because the external checkpoint paths contain \
                 no countable weights; admitting cold load without a floor (sc-16093/sc-18306)"
            );
            return Ok(());
        };
        let raw_budget = crate::vram_gate::apply_vram_cap(
            self.runtime
                .block_on(crate::gpu::nvidia_vram_budget_gb(&self.gpu_id)),
            crate::vram_gate::cuda_vram_cap_gb(),
        );
        let reclaimable_gb = resident_reclaimable_weight_bytes as f64 / BYTES_PER_GIB;
        let budget = cached_floor_budget(raw_budget, resident_reclaimable_weight_bytes);
        let needed = Some(floor_gb);
        match crate::vram_gate::load_plan(needed, None, budget, false) {
            LoadPlan::Resident => {
                tracing::info!(
                    model = self.model,
                    lane = self.lane,
                    floor_gb,
                    replacing_resident = resident_reclaimable_weight_bytes > 0,
                    reclaimable_gb,
                    "candle cached-base admission: cold external checkpoint admitted on its on-disk \
                     weights floor; activation peaks remain unmeasured (sc-18306)"
                );
                Ok(())
            }
            LoadPlan::Sequential => {
                unreachable!("a floor-only route is never sequential-capable")
            }
            LoadPlan::Reject => Err(reject_message(
                &self.model,
                self.lane,
                None,
                floor_gb,
                budget.map_or(0.0, |budget| budget.free_gb),
                &self.gpu_id,
                false,
            )),
        }
    }
}

/// Prepare the cheap, immutable portion of a cached imported/ComfyUI floor. No GPU query or gate is
/// performed here; those happen only from the cache's cold-load hook.
pub(super) fn prepare_cached_candle_base_floor(
    model: &str,
    lane: &'static str,
    settings: &Settings,
    spec: &LoadSpec,
    companion_dirs: &[&Path],
) -> WorkerResult<CachedCandleBaseFloorAdmission> {
    let bytes = prepared_floor_weight_bytes(lane, spec, companion_dirs)?;
    Ok(CachedCandleBaseFloorAdmission {
        model: model.to_owned(),
        lane,
        gpu_id: settings.gpu_id.clone(),
        source_weight_bytes: bytes,
        runtime: tokio::runtime::Handle::current(),
    })
}

fn cached_floor_budget(
    raw_budget: Option<crate::vram_gate::VramBudget>,
    resident_reclaimable_weight_bytes: u64,
) -> Option<crate::vram_gate::VramBudget> {
    let reclaimable_gb = resident_reclaimable_weight_bytes as f64 / BYTES_PER_GIB;
    raw_budget.map(|budget| crate::vram_gate::with_reclaimable(budget, reclaimable_gb))
}

fn prepared_floor_weight_bytes(
    lane: &str,
    spec: &LoadSpec,
    companion_dirs: &[&Path],
) -> WorkerResult<u64> {
    if !spec.prepared_file_pins().is_prepared() {
        return Err(WorkerError::Engine(format!(
            "{lane} admission requires a finalized prepared LoadSpec"
        )));
    }
    spec.validate_prepared_file_pins().map_err(|error| {
        crate::classify_engine_error(&format!("{lane} source validation failed"), error)
    })?;
    // File bytes come from the exact target fingerprints captured during lexical pinning. This
    // deliberately does not call metadata on File slots; companion directories remain recursive
    // because directory snapshots are outside gen-core's single-file token model.
    let file_bytes = spec
        .prepared_file_pins()
        .iter()
        .fold(0_u64, |total, (_, pin)| {
            total.saturating_add(pin.target_fingerprint().size)
        });
    Ok(file_bytes.saturating_add(distinct_weight_bytes(companion_dirs)))
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
    fn imported_file_floor_excludes_the_replaced_snapshot_transformer() {
        let temp = tempfile::tempdir().expect("tempdir");
        let imported = temp.path().join("imported.safetensors");
        let base = temp.path().join("base");
        let text_encoder = base.join("text_encoder");
        let vae = base.join("vae");
        let replaced_transformer = base.join("transformer");
        std::fs::create_dir_all(&text_encoder).expect("text encoder dir");
        std::fs::create_dir_all(&vae).expect("vae dir");
        std::fs::create_dir_all(&replaced_transformer).expect("transformer dir");
        std::fs::write(&imported, vec![0_u8; 11]).expect("imported weights");
        std::fs::write(text_encoder.join("model.safetensors"), vec![0_u8; 13])
            .expect("text encoder weights");
        std::fs::write(vae.join("model.safetensors"), vec![0_u8; 17]).expect("vae weights");
        std::fs::write(
            replaced_transformer.join("model.safetensors"),
            vec![0_u8; 19],
        )
        .expect("snapshot transformer weights");

        let consumed =
            distinct_weight_bytes(&[imported.as_path(), text_encoder.as_path(), vae.as_path()]);
        let recursively_priced = distinct_weight_bytes(&[imported.as_path(), base.as_path()]);
        assert_eq!(consumed, 11 + 13 + 17);
        assert_eq!(recursively_priced, consumed + 19);
        assert!(
            consumed < recursively_priced,
            "the companion snapshot's replaced transformer must stay outside the imported-file floor"
        );
    }

    #[test]
    fn prepared_floor_uses_token_target_size_and_propagates_stale_source_errors() {
        let temp = tempfile::tempdir().expect("tempdir");
        let imported = temp.path().join("imported.safetensors");
        let companion = temp.path().join("text_encoder");
        std::fs::create_dir_all(&companion).expect("companion creates");
        std::fs::write(&imported, vec![0_u8; 11]).expect("imported weights");
        std::fs::write(companion.join("model.safetensors"), vec![0_u8; 13])
            .expect("companion weights");

        let pin = gen_core::PinnedWeightsFile::pin(&imported).expect("primary pins");
        assert_eq!(pin.target_fingerprint().size, 11);
        let mut spec = LoadSpec::new(WeightsSource::File(imported.clone()));
        spec.prepare_with_file_pins([pin]).expect("spec prepares");
        assert_eq!(
            prepared_floor_weight_bytes("fixture", &spec, &[companion.as_path()])
                .expect("prepared floor accounts"),
            24,
            "File bytes come from the prepared target fingerprint and companions remain recursive"
        );

        std::fs::write(&imported, vec![1_u8; 17]).expect("primary replacement writes");
        let error = prepared_floor_weight_bytes("fixture", &spec, &[companion.as_path()])
            .expect_err("stale prepared sources propagate instead of being restatted/swallowed");
        assert!(
            error.to_string().contains("source validation failed")
                || error.to_string().contains("changed after load"),
            "unexpected stale-source error: {error}"
        );
    }

    #[test]
    fn cached_floor_credits_only_the_exact_current_resident_source_bytes() {
        // Large -> small -> medium on a 24 GiB card. Raw free is 4 GiB while the current small
        // resident occupies a conservative 4 GiB of source weights. A stale 20 GiB historical peak
        // would clamp the budget to 24 GiB and admit the 12 GiB incoming floor even though evicting
        // the small resident can produce only 8 GiB. Cache-bound credit must therefore reject.
        let raw = crate::vram_gate::VramBudget {
            free_gb: 4.0,
            total_gb: 24.0,
        };
        let current_small_bytes = (4.0 * BYTES_PER_GIB) as u64;
        let exact = cached_floor_budget(Some(raw), current_small_bytes).expect("budget");
        assert_eq!(exact.free_gb, 8.0);
        assert_eq!(
            crate::vram_gate::load_plan(Some(12.0), None, Some(exact), false),
            LoadPlan::Reject,
            "the medium load must not receive credit for an older large resident"
        );

        let stale_historical = crate::vram_gate::with_reclaimable(raw, 20.0);
        assert_eq!(stale_historical.free_gb, 24.0);
        assert_eq!(
            crate::vram_gate::load_plan(Some(12.0), None, Some(stale_historical), false),
            LoadPlan::Resident,
            "precondition: the stale global high-water would have over-admitted"
        );
    }

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
