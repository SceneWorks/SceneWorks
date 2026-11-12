//! Catalog memory guidance (SW-7032, TOOL-3): the API-owned answer to "which quant tier fits this
//! host", "what host does this install need" and "which high-resolution buckets fit". The web used
//! to hold these headroom constants and their arithmetic in `tierSuggestion.js` (later
//! `registryTierEvidence.js`) and `resolutionMemory.js`; it now reads the values served here and
//! only compares them with its own host-memory probe.
//!
//! Every value is host-INDEPENDENT. Each "fits" rule of the old client reduces to `host >= t` for a
//! per-(lane, tier) threshold `t`, so the catalog serves `t` and the client compares. Install state
//! and per-host tier eligibility (which candle-only tiers a live worker can serve) stay client-side
//! inputs: the guidance carries one fact row per tier and the client selects the rows of the tiers
//! it has installed. A missing threshold (`null`) means "no basis to predict", which the client
//! treats as fitting — SUGGEST, NEVER WITHHOLD (epic 8506 decision 1).
//!
//! Lanes never cross (sc-15613): the per-download `footprint` block is an MLX unified-memory
//! measurement (`footprint_measure.rs`), and `candle.vramGbByTier` / `candle.minMemoryGb` are
//! discrete-VRAM evidence. Each lane below reads only its own fields.
use serde_json::{json, Map, Value};

use crate::memory_anchor::{
    sdxl_cuda_memory_endpoints::SdxlCudaMemoryEndpoints, BYTES_PER_GIB,
    CANDLE_ACTIVE_RECAPTURE_SPREAD, DEDICATED_VRAM_ALLOCATOR_SLACK_GB,
};
use crate::model_artifacts::artifact_selection::{
    is_co_requisite_download, is_pending_artifact_download, is_supported_model_download,
    manifest_download_size_bytes,
};

/// The catalog field that carries the guidance.
pub const FIELD: &str = "memoryGuidance";

/// Resident-memory estimate over on-disk size, used only when a variant has no measured footprint.
/// sc-8516 measured RESIDENT/disk at 0.81–1.01 (mean 0.94) across sdxl-q8 / z-image-q4 /
/// z-image-turbo-q4 / lens-turbo-q4: packed weights sit resident at roughly their on-disk size.
pub const DISK_TO_RESIDENT_MULTIPLIER: f64 = 1.0;

/// Fixed transient working set (bytes) one 1024² MLX generation needs on top of resident weights.
/// sc-8516 measured PEAK − RESIDENT at 14.04 GiB to within ~4 MiB across a 3.9→16.5 GiB resident
/// spread over three VAEs: resolution-bound and size-independent, so it is an addend, not a factor.
pub const TRANSIENT_HEADROOM_BYTES: u64 = 14 * 1024 * 1024 * 1024;

/// The geometry the transient above (and a sibling-extrapolation anchor) was measured at.
pub const TRANSIENT_HEADROOM_MEASURED_PIXELS: u64 = 1024 * 1024;

/// Margin on the precision-dependent disk delta when a measured sibling tier's peak is extrapolated
/// to an unmeasured tier. Capped at the dense estimate, so it can only correct an over-estimate.
pub const MEASURED_SIBLING_DELTA_TOLERANCE: f64 = 1.05;

/// Fraction of unified memory an MLX tier's estimated peak must fit under to be suggested. The
/// budget is peak-inclusive (weights + the fixed transient), so the remainder is OS/app margin.
pub const MEMORY_HEADROOM_FRACTION: f64 = 0.9;

/// The historical resolution ceiling (pixels). Buckets at or below it shipped before sc-13959 and
/// stay unconditionally offered; the model's memory floor already covers them.
pub const RESOLUTION_BASELINE_PIXELS: u64 = 1536 * 1536;

/// Transient (GB) one single-pass generation needs per megapixel ABOVE the baseline, anchored on the
/// measured ~14 GiB 1024² decode + attention transient (≈13 GB/MP).
pub const HIGHRES_TRANSIENT_GB_PER_MP: f64 = 13.0;

/// Quant tiers in fidelity order, HIGHEST first. The suggestion walks this order.
const TIER_FIDELITY: [&str; 3] = ["bf16", "q8", "q4"];
/// Candle-only selectable tiers with no bits-based `mlxQuantize` value.
const CANDLE_ONLY_TIERS: [&str; 3] = ["int8-convrot", "nvfp4", "fp8r"];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Lane {
    Mlx,
    Candle,
}

impl Lane {
    fn key(self) -> &'static str {
        match self {
            Lane::Mlx => "mlx",
            Lane::Candle => "candle",
        }
    }
}

fn is_quant_tier(tier: &str) -> bool {
    TIER_FIDELITY.contains(&tier)
}

fn is_selectable_tier(tier: &str) -> bool {
    is_quant_tier(tier) || CANDLE_ONLY_TIERS.contains(&tier)
}

/// A JSON number that is finite (the old client's `numberOrNull`); strings are not numbers.
fn number(value: Option<&Value>) -> Option<f64> {
    value
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

fn positive(value: Option<&Value>) -> Option<f64> {
    number(value).filter(|value| *value > 0.0)
}

/// A positive integral JSON number (the old client's `positiveIntegerOrNull`).
fn positive_integer(value: Option<&Value>) -> Option<f64> {
    positive(value).filter(|value| value.fract() == 0.0 && *value <= 9_007_199_254_740_991.0)
}

/// One emitted catalog variant, built from the download rows exactly as the catalog builds its
/// `variants[]`: supported, non-co-requisite rows, keyed by trimmed `variant` (else `default`),
/// first row wins per key.
struct Variant<'a> {
    tier: String,
    footprint: Option<&'a Value>,
    download_size_bytes: Option<f64>,
    pending: bool,
}

struct Entry<'a> {
    model: &'a Value,
    variants: Vec<Variant<'a>>,
    has_variant_matrix: bool,
}

impl<'a> Entry<'a> {
    fn new(model: &'a Value) -> Self {
        let downloads = model
            .get("downloads")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let selectable = || {
            downloads.iter().filter(|download| {
                is_supported_model_download(download) && !is_co_requisite_download(download)
            })
        };
        let has_variant_matrix = selectable().any(|download| {
            download
                .get("variant")
                .and_then(Value::as_str)
                .is_some_and(|variant| !variant.trim().is_empty())
        });
        let mut seen = std::collections::HashSet::new();
        let variants = selectable()
            .filter_map(|download| {
                let tier = download
                    .get("variant")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|variant| !variant.is_empty())
                    .unwrap_or("default")
                    .to_owned();
                seen.insert(tier.clone()).then(|| Variant {
                    tier,
                    footprint: download.get("footprint"),
                    download_size_bytes: manifest_download_size_bytes(model, download)
                        .or_else(|| {
                            download
                                .get("footprint")
                                .and_then(|footprint| footprint.get("diskSizeBytes"))
                                .and_then(crate::payload_util::json_size_to_u64)
                        })
                        .map(|bytes| bytes as f64),
                    pending: is_pending_artifact_download(download),
                })
            })
            .collect();
        Self {
            model,
            variants,
            has_variant_matrix,
        }
    }

    fn block(&self, lane: Lane) -> Option<&'a Value> {
        self.model.get(lane.key())
    }

    fn variant(&self, tier: &str) -> Option<&Variant<'a>> {
        self.variants.iter().find(|variant| variant.tier == tier)
    }

    fn footprint_number(variant: &Variant<'_>, key: &str) -> Option<f64> {
        number(variant.footprint.and_then(|footprint| footprint.get(key)))
    }

    /// `footprint.diskSizeBytes`, else the catalog's `downloadSizeBytes` (both positive).
    fn disk_bytes(variant: &Variant<'_>) -> Option<f64> {
        Self::footprint_number(variant, "diskSizeBytes")
            .filter(|disk| *disk > 0.0)
            .or(variant.download_size_bytes.filter(|bytes| *bytes > 0.0))
    }

    /// A variant's PEAK memory requirement in bytes on the MLX lane, in priority order: the
    /// measured peak; measured resident + the fixed transient; a compatible measured sibling's
    /// extrapolated peak; the dense disk estimate; the download-size estimate.
    fn variant_peak_bytes(&self, index: usize) -> Option<f64> {
        let variant = &self.variants[index];
        if let Some(peak) = Self::footprint_number(variant, "peakMemoryBytes").filter(|v| *v > 0.0)
        {
            return Some(peak);
        }
        if let Some(resident) =
            Self::footprint_number(variant, "residentMemoryBytes").filter(|v| *v > 0.0)
        {
            return Some(resident + TRANSIENT_HEADROOM_BYTES as f64);
        }
        let disk = Self::footprint_number(variant, "diskSizeBytes")
            .filter(|disk| *disk > 0.0)
            .or(variant.download_size_bytes.filter(|bytes| *bytes > 0.0))?;
        Some(
            self.measured_sibling_peak_bytes(index, disk)
                .unwrap_or_else(|| dense_peak_bytes(disk)),
        )
    }

    /// Extrapolate an unmeasured quant tier's peak from a smaller tier measured at the calibrated
    /// geometry, only when product data declares the active share of disk growth
    /// (`mlx.measuredSiblingActiveDiskFraction`). The max usable sibling estimate is taken, capped
    /// at the dense estimate.
    fn measured_sibling_peak_bytes(&self, index: usize, target_disk: f64) -> Option<f64> {
        let target = &self.variants[index].tier;
        let fraction = number(
            self.block(Lane::Mlx)
                .and_then(|mlx| mlx.get("measuredSiblingActiveDiskFraction")),
        )
        .filter(|fraction| *fraction > 0.0 && *fraction <= 1.0)?;
        if !is_quant_tier(target) {
            return None;
        }
        let mut estimate: Option<f64> = None;
        for (other, sibling) in self.variants.iter().enumerate() {
            if other == index || &sibling.tier == target || !is_quant_tier(&sibling.tier) {
                continue;
            }
            let (Some(disk), Some(peak)) = (
                Self::disk_bytes(sibling),
                Self::footprint_number(sibling, "peakMemoryBytes")
                    .filter(|peak| *peak > 0.0)
                    .filter(|_| {
                        Self::footprint_number(sibling, "measuredPixels")
                            == Some(TRANSIENT_HEADROOM_MEASURED_PIXELS as f64)
                    }),
            ) else {
                continue;
            };
            if disk >= target_disk {
                continue;
            }
            let candidate =
                peak + ((target_disk - disk) * fraction * MEASURED_SIBLING_DELTA_TOLERANCE).round();
            if candidate > 0.0
                && candidate.is_finite()
                && estimate.is_none_or(|estimate| candidate > estimate)
            {
                estimate = Some(candidate);
            }
        }
        estimate.map(|estimate| dense_peak_bytes(target_disk).min(estimate))
    }

    /// The lane's own measured per-tier peaks (GB). MLX: the measured `footprint.peakMemoryBytes`
    /// of each quant tier (the max over duplicate keys). Candle: `candle.vramGbByTier` rows of
    /// selectable tiers. A tier present with no number maps to `None`.
    fn measured_peaks_gb(&self, lane: Lane) -> Vec<(String, Option<f64>)> {
        let mut by_tier: Vec<(String, Option<f64>)> = Vec::new();
        match lane {
            Lane::Mlx => {
                for variant in &self.variants {
                    if !is_quant_tier(&variant.tier) {
                        continue;
                    }
                    let gb = Self::footprint_number(variant, "peakMemoryBytes")
                        .filter(|peak| *peak > 0.0)
                        .map(|peak| peak / BYTES_PER_GIB);
                    match by_tier.iter_mut().find(|(tier, _)| *tier == variant.tier) {
                        Some((_, prior)) => {
                            if let Some(gb) = gb {
                                *prior = Some(prior.map_or(gb, |prior| prior.max(gb)));
                            }
                        }
                        None => by_tier.push((variant.tier.clone(), gb)),
                    }
                }
            }
            Lane::Candle => {
                if let Some(rows) = self
                    .block(Lane::Candle)
                    .and_then(|candle| candle.get("vramGbByTier"))
                    .and_then(Value::as_object)
                {
                    for (tier, value) in rows {
                        if is_selectable_tier(tier) {
                            by_tier.push((tier.clone(), positive(Some(value))));
                        }
                    }
                }
            }
        }
        by_tier
    }

    /// A tier's declared peak (GB) on `lane` and whether it is that tier's OWN evidence. A
    /// candle-only tier with no row degrades to the q8 row (never `minMemoryGb`), as a proxy.
    fn declared_peak_gb(
        peaks: &[(String, Option<f64>)],
        lane: Lane,
        tier: &str,
    ) -> (Option<f64>, bool) {
        let get = |key: &str| {
            peaks
                .iter()
                .find(|(candidate, _)| candidate == key)
                .and_then(|(_, gb)| *gb)
        };
        if let Some(own) = get(tier) {
            return (Some(own), true);
        }
        if lane == Lane::Candle && CANDLE_ONLY_TIERS.contains(&tier) {
            let degraded = get("q8");
            return (degraded, degraded.is_none());
        }
        (None, true)
    }

    /// `<lane>.minMemoryGbByTier[tier]` (positive), else the `<lane>.minMemoryGb` scalar.
    fn blanket_floor_gb(&self, lane: Lane, tier: Option<&str>) -> Option<f64> {
        let block = self.block(lane);
        if let Some(tier) = tier.map(str::trim).filter(|tier| !tier.is_empty()) {
            if let Some(row) = positive(
                block
                    .and_then(|block| block.get("minMemoryGbByTier"))
                    .and_then(|rows| rows.get(tier)),
            ) {
                return Some(row);
            }
        }
        positive(block.and_then(|block| block.get("minMemoryGb")))
    }

    /// `tier`'s own `minMemoryGbByTier` row, with no scalar fallback.
    fn tier_floor_row_gb(&self, lane: Lane, tier: &str) -> Option<f64> {
        positive(
            self.block(lane)
                .and_then(|block| block.get("minMemoryGbByTier"))
                .and_then(|rows| rows.get(tier)),
        )
    }

    /// The lightest quant tier that can actually be installed (declared, not pending).
    fn lightest_installable_tier(&self) -> Option<&'static str> {
        if !self.has_variant_matrix {
            return None;
        }
        TIER_FIDELITY.iter().rev().copied().find(|tier| {
            self.variants
                .iter()
                .any(|variant| variant.tier == *tier && !variant.pending)
        })
    }

    fn uncalibrated(&self, lane: Lane) -> bool {
        lane == Lane::Candle
            && self
                .block(Lane::Candle)
                .and_then(|candle| candle.get("measured"))
                .and_then(Value::as_bool)
                == Some(false)
    }

    /// The MLX-lane estimated peak (GB) of an installed tier with no measured number: the variant's
    /// footprint estimate, else a convert-at-install tier's on-disk size from `mlxTierStates`.
    fn mlx_estimated_peak_gb(&self, tier: &str) -> Option<f64> {
        if let Some(index) = self
            .variants
            .iter()
            .position(|variant| variant.tier == tier)
        {
            if let Some(bytes) = self.variant_peak_bytes(index) {
                return Some(bytes / BYTES_PER_GIB);
            }
        }
        let disk = self
            .model
            .get("mlxTierStates")
            .and_then(Value::as_array)
            .and_then(|states| {
                states
                    .iter()
                    .find(|state| state.get("tier").and_then(Value::as_str) == Some(tier))
            })
            .and_then(|state| positive(state.get("diskSizeBytes")))?;
        Some(dense_peak_bytes(disk) / BYTES_PER_GIB)
    }

    /// The selected installed tier's own lane peak and its measurement geometry, or `None` when
    /// the evidence is missing, geometry-less or ambiguous.
    fn tier_peak_evidence(&self, lane: Lane, tier: &str) -> Option<(f64, f64)> {
        match lane {
            Lane::Candle => {
                let candle = self.block(Lane::Candle)?;
                let peak = positive(candle.get("vramGbByTier").and_then(|rows| rows.get(tier)))?;
                let pixels = positive_integer(candle.get("vramMeasuredPixels"))?;
                Some((peak, pixels))
            }
            Lane::Mlx => {
                if !is_quant_tier(tier) {
                    return None;
                }
                let mut selected: Option<(f64, f64)> = None;
                for variant in self.variants.iter().filter(|variant| variant.tier == tier) {
                    let Some(peak) = Self::footprint_number(variant, "peakMemoryBytes")
                        .filter(|peak| *peak > 0.0)
                        .map(|peak| peak / BYTES_PER_GIB)
                    else {
                        continue;
                    };
                    let pixels = positive_integer(
                        variant
                            .footprint
                            .and_then(|footprint| footprint.get("measuredPixels")),
                    );
                    match (pixels, selected) {
                        (None, _) => return None,
                        (Some(pixels), Some((_, chosen))) if pixels != chosen => return None,
                        (Some(pixels), Some((best, _))) if peak <= best => {
                            selected = Some((best, pixels));
                        }
                        (Some(pixels), _) => selected = Some((peak, pixels)),
                    }
                }
                selected
            }
        }
    }

    /// The 1536²-baseline floor (GB) a resolution prediction starts from: the selected tier's own
    /// lane peak normalized to the baseline geometry, else the lane's scalar `minMemoryGb`.
    fn resolution_floor_gb(&self, lane: Lane, tier: Option<&str>) -> Option<f64> {
        if let Some((peak, pixels)) = tier.and_then(|tier| self.tier_peak_evidence(lane, tier)) {
            let geometry_delta_mp = (RESOLUTION_BASELINE_PIXELS as f64 - pixels) / 1_000_000.0;
            let normalized = peak + HIGHRES_TRANSIENT_GB_PER_MP * geometry_delta_mp;
            if normalized.is_finite() && normalized > 0.0 {
                return Some(normalized);
            }
        }
        self.blanket_floor_gb(lane, None)
    }

    /// Minimum host GB per declared resolution bucket above the baseline, from `floor`.
    fn resolution_thresholds(&self, lane: Lane, floor: Option<f64>) -> Map<String, Value> {
        let mut thresholds = Map::new();
        let Some(floor) = floor else {
            return thresholds;
        };
        for bucket in self
            .model
            .get("limits")
            .and_then(|limits| limits.get("resolutions"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            let Some(pixels) = bucket_pixels(bucket) else {
                continue;
            };
            if pixels <= RESOLUTION_BASELINE_PIXELS as f64 {
                continue;
            }
            let extra_mp = (pixels - RESOLUTION_BASELINE_PIXELS as f64) / 1_000_000.0;
            let required = floor + HIGHRES_TRANSIENT_GB_PER_MP * extra_mp;
            // Candle's `minMemoryGb` is already a dedicated-VRAM host requirement; the MLX
            // shared-pool budget leaves `1 - MEMORY_HEADROOM_FRACTION` for the OS.
            let host = match lane {
                Lane::Candle => required,
                Lane::Mlx => required / MEMORY_HEADROOM_FRACTION,
            };
            thresholds.insert(bucket.to_owned(), json!(host));
        }
        thresholds
    }

    /// The tierFits threshold: the minimum host GB at which `variant` fits on `lane`, or `None`
    /// when there is no basis to predict (it then always fits).
    fn tier_fit_min_host_gb(
        &self,
        lane: Lane,
        index: usize,
        endpoint: Option<&EndpointEstimate>,
    ) -> Option<f64> {
        let tier = self.variants[index].tier.as_str();
        if let Some(endpoint) =
            endpoint.filter(|endpoint| lane == Lane::Candle && endpoint.tier == tier)
        {
            return Some(endpoint.optimized_host_gb);
        }
        let floor_row = self.tier_floor_row_gb(lane, tier);
        let lane_threshold = match lane {
            Lane::Candle => self
                .measured_peaks_gb(Lane::Candle)
                .into_iter()
                .find(|(candidate, _)| candidate == tier)
                .and_then(|(_, gb)| gb)
                .map(|peak| peak + DEDICATED_VRAM_ALLOCATOR_SLACK_GB),
            Lane::Mlx => self
                .variant_peak_bytes(index)
                .map(|bytes| bytes / BYTES_PER_GIB / MEMORY_HEADROOM_FRACTION),
        };
        match (floor_row, lane_threshold) {
            (Some(row), Some(threshold)) => Some(row.max(threshold)),
            (row, threshold) => row.or(threshold),
        }
    }

    fn tier_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = Vec::new();
        let mut push = |tier: &str| {
            if !keys.iter().any(|known| known == tier) {
                keys.push(tier.to_owned());
            }
        };
        for variant in &self.variants {
            push(&variant.tier);
        }
        for tier in self
            .model
            .get("runtimeQuantTiers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            push(tier);
        }
        for state in self
            .model
            .get("mlxTierStates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(tier) = state.get("tier").and_then(Value::as_str) {
                push(tier);
            }
        }
        for lane in [Lane::Mlx, Lane::Candle] {
            for key in ["vramGbByTier", "minMemoryGbByTier"] {
                for tier in self
                    .block(lane)
                    .and_then(|block| block.get(key))
                    .and_then(Value::as_object)
                    .into_iter()
                    .flat_map(Map::keys)
                {
                    push(tier);
                }
            }
        }
        keys
    }

    fn lane(&self, lane: Lane, endpoint: Option<&EndpointEstimate>) -> Value {
        let peaks = self.measured_peaks_gb(lane);
        let mut tiers = Map::new();
        for tier in self.tier_keys() {
            let mut row = Map::new();
            if let Some(index) = self
                .variants
                .iter()
                .position(|variant| variant.tier == tier)
            {
                row.insert(
                    "minHostGb".to_owned(),
                    json!(self.tier_fit_min_host_gb(lane, index, endpoint)),
                );
            }
            let (declared, exact) = Self::declared_peak_gb(&peaks, lane, &tier);
            let (installed_peak, estimated) = match declared {
                Some(gb) => (Some(gb), !exact),
                None => match lane {
                    Lane::Mlx => (self.mlx_estimated_peak_gb(&tier), true),
                    Lane::Candle => (None, false),
                },
            };
            row.insert(
                "installedHostGb".to_owned(),
                json!(installed_peak.and_then(|peak| host_gb_for_peak_gb(peak, lane))),
            );
            row.insert(
                "installedEstimated".to_owned(),
                json!(installed_peak.is_some() && estimated),
            );
            row.insert(
                "floorGb".to_owned(),
                json!(self.blanket_floor_gb(lane, Some(&tier))),
            );
            row.insert(
                "measuredHostGb".to_owned(),
                json!(peaks
                    .iter()
                    .find(|(candidate, _)| *candidate == tier)
                    .and_then(|(_, gb)| *gb)
                    .and_then(|peak| host_gb_for_peak_gb(peak, lane))),
            );
            if self.tier_peak_evidence(lane, &tier).is_some() {
                let floor = self.resolution_floor_gb(lane, Some(&tier));
                row.insert(
                    "resolutionMinHostGb".to_owned(),
                    Value::Object(self.resolution_thresholds(lane, floor)),
                );
            }
            tiers.insert(tier, Value::Object(row));
        }
        let suggestion = if self.has_variant_matrix {
            TIER_FIDELITY
                .iter()
                .filter_map(|tier| {
                    let index = self
                        .variants
                        .iter()
                        .position(|variant| variant.tier == *tier)?;
                    Some(json!({
                        "tier": tier,
                        "minHostGb": self.tier_fit_min_host_gb(lane, index, endpoint),
                    }))
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut object = Map::new();
        object.insert("suggestion".to_owned(), Value::Array(suggestion));
        object.insert("tiers".to_owned(), Value::Object(tiers));
        object.insert(
            "blanketGb".to_owned(),
            json!(self.blanket_floor_gb(lane, None)),
        );
        object.insert(
            "entryFloorGb".to_owned(),
            json!(self.blanket_floor_gb(lane, self.lightest_installable_tier())),
        );
        object.insert("uncalibrated".to_owned(), json!(self.uncalibrated(lane)));
        object.insert(
            "resolutionMinHostGb".to_owned(),
            Value::Object(self.resolution_thresholds(lane, self.resolution_floor_gb(lane, None))),
        );
        if lane == Lane::Candle {
            if let Some(endpoint) = endpoint {
                object.insert("endpointEstimate".to_owned(), endpoint.to_json());
            }
        }
        Value::Object(object)
    }
}

fn dense_peak_bytes(disk: f64) -> f64 {
    (disk * DISK_TO_RESIDENT_MULTIPLIER).round() + TRANSIENT_HEADROOM_BYTES as f64
}

/// The host size (GB) at which a raw `peak_gb` satisfies its OWN lane's fit criterion: MLX budgets
/// `peak <= host × MEMORY_HEADROOM_FRACTION`; Candle admits `peak + allocator slack`.
fn host_gb_for_peak_gb(peak_gb: f64, lane: Lane) -> Option<f64> {
    if !peak_gb.is_finite() || peak_gb <= 0.0 {
        return None;
    }
    Some(match lane {
        Lane::Candle => (peak_gb + DEDICATED_VRAM_ALLOCATOR_SLACK_GB).ceil(),
        Lane::Mlx => (peak_gb / MEMORY_HEADROOM_FRACTION).ceil(),
    })
}

/// Pixel count of a `"WxH"` bucket (`None` when malformed).
fn bucket_pixels(bucket: &str) -> Option<f64> {
    let mut parts = bucket.split('x');
    let width = parse_js_number(parts.next()?)?;
    let height = parse_js_number(parts.next()?)?;
    (width > 0.0 && height > 0.0).then_some(width * height)
}

fn parse_js_number(raw: &str) -> Option<f64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

/// The measured SDXL CUDA endpoint guidance for its one measured (model, tier) coordinate.
struct EndpointEstimate {
    tier: String,
    optimized_host_gb: f64,
    label: String,
    description: String,
}

impl EndpointEstimate {
    fn from_pair(pair: &SdxlCudaMemoryEndpoints) -> Option<Self> {
        let required_gb = |peak_bytes: u64| {
            (peak_bytes as f64 * (1.0 + CANDLE_ACTIVE_RECAPTURE_SPREAD)).ceil() / BYTES_PER_GIB
                + DEDICATED_VRAM_ALLOCATOR_SLACK_GB
        };
        let optimized = pair.sequential.overall_allocator_envelope_bytes;
        let resident = pair.resident.overall_allocator_envelope_bytes;
        let optimized_peak_gb = optimized as f64 / BYTES_PER_GIB;
        let resident_peak_gb = resident as f64 / BYTES_PER_GIB;
        let optimized_host_gb = required_gb(optimized);
        let resident_host_gb = required_gb(resident);
        let geometry = &pair.sequential.geometry;
        let defaults = &pair.resolved_defaults;
        let steps = defaults.get("steps").and_then(Value::as_u64)?;
        let guidance = number(defaults.get("guidance"))?;
        let sampler = defaults.get("sampler").and_then(Value::as_str)?;
        let count = pair.settings.get("count").and_then(Value::as_u64)?;
        let tier = pair.sequential.tier.clone();
        let images = if count == 1 {
            "one image".to_owned()
        } else {
            format!("{count} images")
        };
        let model = pair.sequential.model_family.to_ascii_uppercase();
        let tier_label = tier.to_ascii_uppercase();
        Some(Self {
            label: format!("{tier_label} optimized ~{optimized_host_gb:.1} GiB"),
            description: format!(
                "Stock {model} {tier_label}, {} × {}, {images}, default sampling ({steps} steps, \
                 guidance {guidance}, {}), no adapters. Measured CUDA peaks: optimized \
                 {optimized_peak_gb:.2} GiB; resident {resident_peak_gb:.2} GiB. Estimated VRAM \
                 budgets including {}% allowance and {} GiB reserve: optimized \
                 {optimized_host_gb:.2} GiB; resident {resident_host_gb:.2} GiB. Fit depends on \
                 available VRAM; the capture does not validate a physical 8 GiB card.",
                geometry.width,
                geometry.height,
                sampler.to_ascii_uppercase(),
                CANDLE_ACTIVE_RECAPTURE_SPREAD * 100.0,
                DEDICATED_VRAM_ALLOCATOR_SLACK_GB,
            ),
            tier,
            optimized_host_gb,
        })
    }

    fn to_json(&self) -> Value {
        json!({
            "tier": self.tier,
            "optimizedHostGb": self.optimized_host_gb,
            "label": self.label,
            "description": self.description,
        })
    }
}

/// The endpoint estimate for `entry` when the packaged pair measured this model's tier on Candle
/// and that tier is a declared, installable (non-pending) variant.
fn endpoint_for(
    entry: &Entry<'_>,
    pair: Option<&SdxlCudaMemoryEndpoints>,
) -> Option<EndpointEstimate> {
    let pair = pair?;
    let id = entry.model.get("id").and_then(Value::as_str)?;
    if pair.sequential.model_id != id || pair.sequential.mode != "text_to_image" {
        return None;
    }
    if entry
        .variant(&pair.sequential.tier)
        .is_none_or(|variant| variant.pending)
    {
        return None;
    }
    EndpointEstimate::from_pair(pair)
}

/// Compute the guidance for one catalog entry (manifest shape: `downloads`, `mlx`, `candle`,
/// `limits`, plus the catalog's `mlxTierStates` / `runtimeQuantTiers` when present).
pub fn model_memory_guidance(model: &Value) -> Value {
    guidance_with_endpoints(
        model,
        crate::memory_anchor::sdxl_cuda_memory_endpoints::packaged_sdxl_cuda_memory_endpoints(),
    )
}

fn guidance_with_endpoints(model: &Value, pair: Option<&SdxlCudaMemoryEndpoints>) -> Value {
    let entry = Entry::new(model);
    let endpoint = endpoint_for(&entry, pair);
    json!({
        "mlx": entry.lane(Lane::Mlx, None),
        "candle": entry.lane(Lane::Candle, endpoint.as_ref()),
    })
}

/// Attach [`FIELD`] to a catalog entry. Called by the API catalog projection and the native
/// UI-catalog dump, so both serve the same values.
pub fn apply_to_model_entry(object: &mut Map<String, Value>) {
    let guidance = model_memory_guidance(&Value::Object(object.clone()));
    object.insert(FIELD.to_owned(), guidance);
}

#[cfg(test)]
#[path = "memory_guidance_tests.rs"]
mod tests;
