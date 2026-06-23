//! Tier-0 dataset quality evaluation (epic 6529 "Dataset Doctor", sc-6532).
//!
//! This is the **pure** half of Tier-0: the result types plus all decision logic. Given per-item
//! scalars (already extracted from pixels by the worker — see
//! `sceneworks_worker::dataset_quality`), the item dimensions (sc-6531), the content hash
//! (sc-6531), and the dataset context (kind, target bucket, preset minimum), it derives typed
//! per-item quality flags and dataset-level findings.
//!
//! No image decoding happens here, so this module stays codec-free and is fully unit-testable with
//! synthetic scalars — no fixture images required. The pixel extraction that *does* need a decode
//! lives in the worker, behind [`Tier0Scalars`].
//!
//! Catalog, thresholds, and the warn-not-block framing come from the spike,
//! `docs/sc-6530/dataset-doctor-metrics.md`.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::contracts::string_enum;

string_enum! {
    /// A Tier-0 quality check — the cheap, no-model checks from the spike's catalog.
    pub enum QualityCheck {
        Resolution => "resolution",
        CropLoss => "crop_loss",
        Blur => "blur",
        Exposure => "exposure",
        ExactDuplicate => "exact_duplicate",
        NearDuplicate => "near_duplicate",
        Count => "count",
        // Decode: the image could not be decoded, so no pixel scalars exist for it (rare — uploads
        // are normalized to png/jpeg/webp). Surfaced so an undecodable image can't quietly count as
        // "technically fine" in the readiness rollup.
        Decode => "decode",
    }
}

string_enum! {
    /// Severity of a flag. Drives the thumbnail badge (no flag / `Info` → ✓, `Warn` → ⚠,
    /// `Fatal` → ✕) and the readiness gate. Bias to `Warn`; reserve `Fatal` for genuinely
    /// untrainable inputs (the spike's block-vs-warn policy). Declared worst-last so the derived
    /// `Ord` ranks `Info < Warn < Fatal`.
    pub enum Severity {
        Info => "info",
        Warn => "warn",
        Fatal => "fatal",
    }
}

string_enum! {
    /// What a LoRA is being taught. Changes what "good" means, so several thresholds vary by it
    /// (the spike's per-kind column). Maps from the training preset's `recommendedFor` tags.
    pub enum DatasetKind {
        Person => "person",
        Style => "style",
        Object => "object",
    }
}

/// Per-image scalars extracted from pixels by the worker. Blur + exposure are measured on the
/// center-crop→bucket-resize the trainer actually feeds (so they are comparable across source
/// resolutions and capture upscale-to-mush); the perceptual hash is taken on the full image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tier0Scalars {
    /// Variance of the Laplacian on the bucket-resized grayscale crop. Higher = sharper.
    pub blur_variance: f64,
    /// Fraction of pixels crushed to black (luma ≤ cutoff), in `[0, 1]`.
    pub shadow_clip: f64,
    /// Fraction of pixels blown to white (luma ≥ cutoff), in `[0, 1]`.
    pub highlight_clip: f64,
    /// Perceptual-hash bytes from the one pinned `HasherConfig` (fixed length). Hamming distance
    /// over these drives near-duplicate clustering.
    pub phash: Vec<u8>,
}

/// A single quality finding on an item (or, for the dataset-level `Count` check, the dataset).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityFlag {
    pub check: QualityCheck,
    pub severity: Severity,
    /// The measured value behind the flag — Laplacian variance, clip fraction, short-edge px,
    /// crop-loss fraction, item count, or Hamming distance. The "evidence" the readout shows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// The threshold `value` was judged against (for "x of y"-style copy).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    /// Peer item ids for relational checks (exact / near-duplicate clusters). Empty otherwise —
    /// lets the UI say *which* photos are duplicates and lets 6533 reconcile pHash with CLIP pairs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peers: Vec<String>,
    /// The user has dismissed this finding for this image (sc-6534 per-image override). An
    /// acknowledged flag is kept in the report so the UI can show it struck-through, but is
    /// **excluded from every rollup** (the item's badge severity, the technical sub-score, the
    /// severity counts, and the gate). Only non-`Fatal` findings can be acknowledged — a decode
    /// failure is genuinely untrainable, so this stays `false` for it (enforced in `evaluate_tier0`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub acknowledged: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// All Tier-0 flags raised for one item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemQualityFlags {
    pub item_id: String,
    pub flags: Vec<QualityFlag>,
}

/// Result of evaluating Tier-0 over a dataset: per-item flags plus dataset-level findings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tier0Evaluation {
    pub items: Vec<ItemQualityFlags>,
    pub dataset: Vec<QualityFlag>,
}

/// One item's inputs to Tier-0 evaluation: identity, dimensions + content hash (sc-6531), and the
/// pixel scalars (worker-extracted; `None` until they have been computed).
#[derive(Debug, Clone)]
pub struct ItemQualityInput {
    pub item_id: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub content_hash: Option<String>,
    pub scalars: Option<Tier0Scalars>,
    /// Checks the user has dismissed for this image (sc-6534). A flag whose check is listed here is
    /// marked acknowledged (and so dropped from every rollup) — but only when it is not `Fatal`, so
    /// a decode failure cannot be waved through. Resolved by the API from the item's persisted ack,
    /// already filtered to the current content hash.
    pub acknowledged: Vec<QualityCheck>,
}

/// Tier-0 thresholds. Defaults from the spike (`docs/sc-6530`), varying by kind where the kind
/// changes what "good" means. One config surface — calibrate here, not at call sites (sc-6530 §8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tier0Thresholds {
    /// Below `min_resolution_ratio × bucket_edge` on the short side ⇒ upscale-to-mush warning.
    pub min_resolution_ratio: f64,
    /// Center-crop dropping more than this fraction of the long side ⇒ crop-loss warning.
    pub crop_loss_fraction: f64,
    /// Absolute Laplacian-variance floor: below this is "soft" regardless of the dataset.
    pub blur_floor: f64,
    /// An image below `blur_relative_factor × dataset_median` is a soft outlier within a sharp set.
    pub blur_relative_factor: f64,
    /// More than this fraction of pixels clipped at black or white ⇒ exposure warning.
    pub exposure_clip_fraction: f64,
    /// pHash Hamming distance ≤ this ⇒ near-duplicate.
    pub near_dup_hamming: u32,
}

impl Tier0Thresholds {
    /// Spike defaults, tuned per kind. Style tolerates softer images (texture/bokeh); person and
    /// object are stricter on sharpness.
    pub fn for_kind(kind: &DatasetKind) -> Self {
        let blur_floor = match kind {
            DatasetKind::Style => 60.0,
            _ => 100.0,
        };
        Self {
            min_resolution_ratio: 0.75,
            crop_loss_fraction: 0.35,
            blur_floor,
            blur_relative_factor: 0.5,
            exposure_clip_fraction: 0.05,
            near_dup_hamming: 6,
        }
    }
}

/// Below this many usable images a set cannot train at all — the only count that blocks rather
/// than warns (sc-6530 §5).
pub const HARD_MIN_ITEMS: u32 = 4;

/// Evaluate Tier-0 over a dataset. Pure: no IO, no decode. `bucket_edge` is the trainer's target
/// square resolution (the size the blur/exposure scalars were measured at); `min_items` is the
/// chosen preset's recommended minimum.
///
/// Degenerate inputs are handled: an empty set yields only the `Count` finding; a single image has
/// no blur median and forms no duplicate cluster, so neither the relative-blur nor the duplicate
/// checks fire.
pub fn evaluate_tier0(
    items: &[ItemQualityInput],
    bucket_edge: u32,
    min_items: u32,
    thresholds: &Tier0Thresholds,
) -> Tier0Evaluation {
    let mut per_item: Vec<ItemQualityFlags> = items
        .iter()
        .map(|item| ItemQualityFlags {
            item_id: item.item_id.clone(),
            flags: Vec::new(),
        })
        .collect();

    let blur_median = median(
        items
            .iter()
            .filter_map(|item| item.scalars.as_ref().map(|s| s.blur_variance)),
    );

    // Per-image, dataset-independent checks.
    for (idx, item) in items.iter().enumerate() {
        let flags = &mut per_item[idx].flags;
        push_resolution_flags(flags, item, bucket_edge, thresholds);
        push_scalar_flags(flags, item, blur_median, thresholds);
    }

    // Relational checks: exact duplicates (content hash) then near duplicates (pHash).
    push_exact_duplicate_flags(items, &mut per_item);
    push_near_duplicate_flags(items, &mut per_item, thresholds.near_dup_hamming);

    // Per-image overrides (sc-6534): mark dismissed findings acknowledged once all flags exist. A
    // `Fatal` finding is never acknowledgeable — a decode failure is untrainable regardless of the
    // user's wishes — so the gate can't be waved past genuinely broken inputs. Relational checks are
    // marked per side: dismissing a duplicate on image A leaves B's flag standing (B is still a dup
    // of something), which is the intended asymmetry.
    for (idx, item) in items.iter().enumerate() {
        if item.acknowledged.is_empty() {
            continue;
        }
        for flag in &mut per_item[idx].flags {
            if flag.severity != Severity::Fatal && item.acknowledged.contains(&flag.check) {
                flag.acknowledged = true;
            }
        }
    }

    // Dataset-level: too few images for the preset.
    let mut dataset = Vec::new();
    let count = items.len() as u32;
    if count < min_items {
        let severity = if count < HARD_MIN_ITEMS {
            Severity::Fatal
        } else {
            Severity::Warn
        };
        dataset.push(QualityFlag {
            check: QualityCheck::Count,
            severity,
            value: Some(count as f64),
            threshold: Some(min_items as f64),
            peers: Vec::new(),
            acknowledged: false,
        });
    }

    Tier0Evaluation {
        items: per_item,
        dataset,
    }
}

fn push_resolution_flags(
    flags: &mut Vec<QualityFlag>,
    item: &ItemQualityInput,
    bucket_edge: u32,
    thresholds: &Tier0Thresholds,
) {
    let (Some(width), Some(height)) = (item.width, item.height) else {
        return;
    };
    let short = f64::from(width.min(height));
    let long = f64::from(width.max(height));
    let target = f64::from(bucket_edge);

    if target > 0.0 && short < target {
        // Below the bucket is a mild nudge; far below means it will be upscaled to mush.
        let severity = if short < thresholds.min_resolution_ratio * target {
            Severity::Warn
        } else {
            Severity::Info
        };
        flags.push(QualityFlag {
            check: QualityCheck::Resolution,
            severity,
            value: Some(short),
            threshold: Some(target),
            peers: Vec::new(),
            acknowledged: false,
        });
    }

    if long > 0.0 {
        let crop_loss = (long - short) / long;
        if crop_loss > thresholds.crop_loss_fraction {
            flags.push(QualityFlag {
                check: QualityCheck::CropLoss,
                severity: Severity::Warn,
                value: Some(crop_loss),
                threshold: Some(thresholds.crop_loss_fraction),
                peers: Vec::new(),
                acknowledged: false,
            });
        }
    }
}

fn push_scalar_flags(
    flags: &mut Vec<QualityFlag>,
    item: &ItemQualityInput,
    blur_median: Option<f64>,
    thresholds: &Tier0Thresholds,
) {
    let Some(scalars) = &item.scalars else {
        return;
    };

    // Soft if below the absolute floor OR a clear outlier below the dataset median (the spike's
    // "floor AND relative" rule — relative alone would pass a uniformly-soft set).
    let below_floor = scalars.blur_variance < thresholds.blur_floor;
    let below_relative = blur_median
        .is_some_and(|median| scalars.blur_variance < thresholds.blur_relative_factor * median);
    if below_floor || below_relative {
        flags.push(QualityFlag {
            check: QualityCheck::Blur,
            severity: Severity::Warn,
            value: Some(scalars.blur_variance),
            threshold: Some(thresholds.blur_floor),
            peers: Vec::new(),
            acknowledged: false,
        });
    }

    let clip = scalars.shadow_clip.max(scalars.highlight_clip);
    if clip > thresholds.exposure_clip_fraction {
        flags.push(QualityFlag {
            check: QualityCheck::Exposure,
            severity: Severity::Warn,
            value: Some(clip),
            threshold: Some(thresholds.exposure_clip_fraction),
            peers: Vec::new(),
            acknowledged: false,
        });
    }
}

fn push_exact_duplicate_flags(items: &[ItemQualityInput], per_item: &mut [ItemQualityFlags]) {
    let mut by_hash: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, item) in items.iter().enumerate() {
        if let Some(hash) = item.content_hash.as_deref() {
            by_hash.entry(hash).or_default().push(idx);
        }
    }
    for group in by_hash.values().filter(|group| group.len() > 1) {
        for &idx in group {
            let peers = group
                .iter()
                .filter(|&&other| other != idx)
                .map(|&other| items[other].item_id.clone())
                .collect();
            per_item[idx].flags.push(QualityFlag {
                check: QualityCheck::ExactDuplicate,
                severity: Severity::Warn,
                value: Some(0.0),
                threshold: None,
                peers,
                acknowledged: false,
            });
        }
    }
}

fn push_near_duplicate_flags(
    items: &[ItemQualityInput],
    per_item: &mut [ItemQualityFlags],
    near_dup_hamming: u32,
) {
    let hashes: Vec<Option<&[u8]>> = items
        .iter()
        .map(|item| item.scalars.as_ref().map(|s| s.phash.as_slice()))
        .collect();

    // Union-find over pHash pairs within the Hamming threshold.
    let mut uf = UnionFind::new(items.len());
    for (i, hi) in hashes.iter().copied().enumerate() {
        let Some(hi) = hi else { continue };
        for (j, hj) in hashes.iter().copied().enumerate().skip(i + 1) {
            let Some(hj) = hj else { continue };
            if hamming(hi, hj).is_some_and(|distance| distance <= near_dup_hamming) {
                uf.union(i, j);
            }
        }
    }

    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, hash) in hashes.iter().enumerate() {
        if hash.is_some() {
            clusters.entry(uf.find(i)).or_default().push(i);
        }
    }

    for cluster in clusters.values().filter(|cluster| cluster.len() > 1) {
        for &idx in cluster {
            // Exclude byte-identical peers — those are reported as exact duplicates, not near
            // ones, so the same pair is never flagged twice (sc-6530 §2).
            let mut peers = Vec::new();
            let mut nearest = u32::MAX;
            for &other in cluster {
                if other == idx || same_content_hash(items, idx, other) {
                    continue;
                }
                if let (Some(a), Some(b)) = (hashes[idx], hashes[other]) {
                    if let Some(distance) = hamming(a, b) {
                        nearest = nearest.min(distance);
                    }
                }
                peers.push(items[other].item_id.clone());
            }
            if !peers.is_empty() {
                per_item[idx].flags.push(QualityFlag {
                    check: QualityCheck::NearDuplicate,
                    severity: Severity::Warn,
                    value: (nearest != u32::MAX).then_some(f64::from(nearest)),
                    threshold: Some(f64::from(near_dup_hamming)),
                    peers,
                    acknowledged: false,
                });
            }
        }
    }
}

fn same_content_hash(items: &[ItemQualityInput], a: usize, b: usize) -> bool {
    match (
        items[a].content_hash.as_deref(),
        items[b].content_hash.as_deref(),
    ) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// Hamming distance between two equal-length byte hashes, or `None` if the lengths differ (which
/// only happens if two images were hashed with different configs — they never are in practice).
fn hamming(a: &[u8], b: &[u8]) -> Option<u32> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum())
}

/// Median of a sample, or `None` when empty. Even-length samples average the two middle values.
fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<f64> = values.collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    })
}

/// Minimal union-find for near-duplicate clustering.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] != node {
            self.parent[node] = self.parent[self.parent[node]]; // path halving
            node = self.parent[node];
        }
        node
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

// ---------------------------------------------------------------------------
// Readiness report (sc-6533) — the dataset-level rollup the training screens read.
// ---------------------------------------------------------------------------

/// A cached Tier-0 extraction stored on a dataset item. Keyed by **both** the content hash and the
/// bucket edge: blur + exposure are measured on the center-crop→bucket-resize, so the same image at
/// a different training resolution has different scalars. Reuse only when both still match (sc-6533).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CachedTier0Scalars {
    pub content_hash: String,
    pub bucket_edge: u32,
    pub scalars: Tier0Scalars,
}

impl CachedTier0Scalars {
    /// True when this cache entry still applies to an item: same image bytes (content hash) **and**
    /// same bucket edge (blur/exposure are measured at the bucket, so a resolution change
    /// invalidates them). The single source of truth for cache reuse — kept here, pure and tested,
    /// rather than in the API layer.
    pub fn valid_for(&self, content_hash: Option<&str>, bucket_edge: u32) -> bool {
        self.bucket_edge == bucket_edge && content_hash == Some(self.content_hash.as_str())
    }
}

/// A user's per-image quality override (sc-6534): the set of checks they have dismissed for one
/// image. Keyed by content hash — the same precedent as [`CachedTier0Scalars`] — so a dismissal
/// cannot silently apply after the image bytes change (you can't pre-acknowledge a photo you never
/// saw). Stored on the dataset item; resolved to effective checks by the API before evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityAck {
    pub content_hash: String,
    pub checks: Vec<QualityCheck>,
}

impl QualityAck {
    /// True when this ack still applies to an item: same image bytes (content hash). A mismatch
    /// (the image was replaced) silently voids the ack.
    pub fn valid_for(&self, content_hash: Option<&str>) -> bool {
        content_hash == Some(self.content_hash.as_str())
    }

    /// The dismissed checks that still apply, or empty when the ack is stale. The single source of
    /// truth for ack reuse — kept here, pure and tested, not in the API layer.
    pub fn effective_checks(&self, content_hash: Option<&str>) -> Vec<QualityCheck> {
        if self.valid_for(content_hash) {
            self.checks.clone()
        } else {
            Vec::new()
        }
    }
}

string_enum! {
    /// The discrete readiness gate (sc-6530 §4) — deliberately NOT a 0–100 score. Drives whether
    /// Train is enabled and what the readout says.
    pub enum ReadinessGate {
        // Ready: enough usable images, nothing worth surfacing — train freely.
        Ready => "ready",
        // NeedsAttention: trainable, but warnings are present (soft/dup/low-res). Train stays
        // enabled; the readout explains what would make it stronger. The default for most real sets.
        NeedsAttention => "needs_attention",
        // Blocked: genuinely untrainable (a fatal flag, e.g. too few images). Train disabled.
        Blocked => "blocked",
    }
}

/// Interpretable sub-scores (sc-6530 §4) — each a plain share the UI can name, never blended into
/// one number. `technical` comes from Tier-0; the rest are Tier-1 (`None` until the embedding job
/// lands, keeping the report forward-compatible).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadinessSubScores {
    /// Share of items with no technical-quality warning (resolution/crop/blur/exposure/dup/decode).
    pub technical: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diversity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alignment: Option<f64>,
}

/// Per-item readiness: the worst severity (for the thumbnail badge) and the flags behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemReadiness {
    pub item_id: String,
    /// Worst flag severity on the item, or `None` when clean (badge ✓).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<Severity>,
    pub flags: Vec<QualityFlag>,
}

/// Flag counts by severity across items + dataset — the readout's "2 blurry, 3 near-dup" line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SeverityCounts {
    pub info: u32,
    pub warn: u32,
    pub fatal: u32,
}

/// One metric's spread across the dataset, for the Advanced surface's per-metric distribution
/// (sc-6534). The raw per-item values let the UI draw a histogram; `threshold` marks the line a
/// flag is judged against, and `higher_is_better` orients it (sharpness improves upward, clip
/// fractions downward). Carried in the report payload so the web needs no second fetch for scalars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricDistribution {
    pub values: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    pub higher_is_better: bool,
}

/// Per-metric distributions over the items that have pixel scalars (sc-6534). `None` on the report
/// until any scalars exist (an unassessed or empty set).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadinessDistributions {
    pub blur_variance: MetricDistribution,
    pub shadow_clip: MetricDistribution,
    pub highlight_clip: MetricDistribution,
}

/// The complete readiness report the training screens render from one payload (sc-6533). Tier-1
/// fields stay `None`/empty until the embedding job attaches them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetReadinessReport {
    pub gate: ReadinessGate,
    pub sub_scores: ReadinessSubScores,
    pub counts: SeverityCounts,
    pub item_count: u32,
    pub items: Vec<ItemReadiness>,
    /// Dataset-level findings not tied to a single item (e.g. too few images).
    pub dataset_flags: Vec<QualityFlag>,
    /// Per-metric value spreads for the Advanced distribution view (sc-6534). Populated by the
    /// extraction layer (`sceneworks-image-quality`), which holds the scalars; the pure core rollup
    /// leaves it `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distributions: Option<ReadinessDistributions>,
}

/// Resolved inputs for an evaluation, derived from the chosen training target/preset and the
/// dataset's character.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadinessContext {
    pub kind: DatasetKind,
    pub bucket_edge: u32,
    pub min_items: u32,
    pub thresholds: Tier0Thresholds,
}

/// Target resolution assumed when no training target is selected yet (a dataset can be assessed
/// before Teach locks a target).
pub const DEFAULT_BUCKET_EDGE: u32 = 1024;

/// Resolve the evaluation context from the chosen target/preset + the dataset's character. Pure and
/// unit-tested so this easy-to-get-wrong mapping doesn't live only in the (here-unbuildable) API
/// layer. `recommended_for` are the preset/target tags (`"character"`/`"style"`); `character_type`
/// is the dataset's character kind (e.g. `"person"`); `bucket_edge` is floored to a multiple of 32
/// to match the trainer.
pub fn readiness_context(
    target_resolution: Option<u32>,
    recommended_for: &[String],
    character_type: Option<&str>,
    configured_min_items: Option<u32>,
) -> ReadinessContext {
    let kind = resolve_kind(recommended_for, character_type);
    let bucket_edge = target_resolution
        .map(|res| (res / 32).max(1) * 32)
        .unwrap_or(DEFAULT_BUCKET_EDGE);
    let min_items = configured_min_items.unwrap_or_else(|| default_min_items(&kind));
    let thresholds = Tier0Thresholds::for_kind(&kind);
    ReadinessContext {
        kind,
        bucket_edge,
        min_items,
        thresholds,
    }
}

fn resolve_kind(recommended_for: &[String], character_type: Option<&str>) -> DatasetKind {
    let tagged = |tag: &str| {
        recommended_for
            .iter()
            .any(|value| value.eq_ignore_ascii_case(tag))
    };
    if character_type.is_some_and(|value| value.eq_ignore_ascii_case("person"))
        || tagged("character")
        || tagged("person")
    {
        DatasetKind::Person
    } else if tagged("style") {
        DatasetKind::Style
    } else {
        DatasetKind::Object
    }
}

/// Per-kind minimum item count (sc-6530 §3) when the preset doesn't pin one.
fn default_min_items(kind: &DatasetKind) -> u32 {
    match kind {
        DatasetKind::Person => 15,
        DatasetKind::Style => 20,
        DatasetKind::Object => 10,
        DatasetKind::Unknown(_) => 12,
    }
}

/// Roll a Tier-0 evaluation up into the dataset readiness report (sc-6533). Pure: no IO. The gate
/// follows the spike's block-vs-warn policy — `Blocked` on any fatal flag, else `NeedsAttention` on
/// any warning, else `Ready`. An item is "technically fine" only if it carries no warn/fatal flag,
/// so a decode failure (which raises a `Decode` warning) correctly drags the `technical` share down.
pub fn build_readiness_report(evaluation: Tier0Evaluation) -> DatasetReadinessReport {
    let item_count = evaluation.items.len() as u32;
    let mut counts = SeverityCounts::default();
    let mut items = Vec::with_capacity(evaluation.items.len());
    let mut clean_items = 0_u32;

    for entry in &evaluation.items {
        // Acknowledged findings are dropped from every rollup — counts, the item badge, the
        // technical share, and (via counts) the gate — but kept in `flags` so the UI can show the
        // user what they dismissed (sc-6534).
        for flag in entry.flags.iter().filter(|flag| !flag.acknowledged) {
            bump(&mut counts, &flag.severity);
        }
        let severity = worst_severity(&entry.flags);
        if !matches!(severity, Some(Severity::Warn | Severity::Fatal)) {
            clean_items += 1;
        }
        items.push(ItemReadiness {
            item_id: entry.item_id.clone(),
            severity,
            flags: entry.flags.clone(),
        });
    }
    for flag in &evaluation.dataset {
        bump(&mut counts, &flag.severity);
    }

    let technical = if item_count == 0 {
        1.0
    } else {
        f64::from(clean_items) / f64::from(item_count)
    };
    let gate = if counts.fatal > 0 {
        ReadinessGate::Blocked
    } else if counts.warn > 0 {
        ReadinessGate::NeedsAttention
    } else {
        ReadinessGate::Ready
    };

    DatasetReadinessReport {
        gate,
        sub_scores: ReadinessSubScores {
            technical,
            diversity: None,
            identity: None,
            alignment: None,
        },
        counts,
        item_count,
        items,
        dataset_flags: evaluation.dataset,
        distributions: None,
    }
}

/// Worst severity among the item's *active* flags — acknowledged findings (sc-6534) and the
/// forward-compat `Unknown` variant are skipped, so an all-dismissed item badges clean and a future
/// unknown severity can never rank as "worst" (the derived `Ord` would otherwise sort it after
/// `Fatal`).
fn worst_severity(flags: &[QualityFlag]) -> Option<Severity> {
    flags
        .iter()
        .filter(|flag| !flag.acknowledged)
        .map(|flag| &flag.severity)
        .filter(|severity| !matches!(severity, Severity::Unknown(_)))
        .max()
        .cloned()
}

fn bump(counts: &mut SeverityCounts, severity: &Severity) {
    match severity {
        Severity::Info => counts.info += 1,
        Severity::Warn => counts.warn += 1,
        Severity::Fatal => counts.fatal += 1,
        Severity::Unknown(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> Tier0Thresholds {
        Tier0Thresholds::for_kind(&DatasetKind::Person)
    }

    fn item(id: &str) -> ItemQualityInput {
        ItemQualityInput {
            item_id: id.to_owned(),
            width: Some(512),
            height: Some(512),
            content_hash: None,
            scalars: None,
            acknowledged: Vec::new(),
        }
    }

    fn scalars(blur: f64, shadow: f64, highlight: f64, phash: Vec<u8>) -> Tier0Scalars {
        Tier0Scalars {
            blur_variance: blur,
            shadow_clip: shadow,
            highlight_clip: highlight,
            phash,
        }
    }

    fn flags_for<'a>(eval: &'a Tier0Evaluation, id: &str) -> &'a [QualityFlag] {
        &eval
            .items
            .iter()
            .find(|entry| entry.item_id == id)
            .expect("item present")
            .flags
    }

    fn has(eval: &Tier0Evaluation, id: &str, check: QualityCheck) -> bool {
        flags_for(eval, id).iter().any(|f| f.check == check)
    }

    #[test]
    fn small_and_wide_images_flag_resolution_and_crop_loss() {
        let mut tiny = item("tiny");
        tiny.width = Some(128);
        tiny.height = Some(128);
        let mut wide = item("wide");
        wide.width = Some(1000);
        wide.height = Some(300);

        let eval = evaluate_tier0(&[tiny, wide], 512, 1, &thresholds());

        assert!(has(&eval, "tiny", QualityCheck::Resolution));
        assert!(has(&eval, "wide", QualityCheck::CropLoss));
    }

    #[test]
    fn square_bucket_sized_image_has_no_resolution_or_crop_flags() {
        let mut ok = item("ok");
        ok.width = Some(512);
        ok.height = Some(512);
        let eval = evaluate_tier0(std::slice::from_ref(&ok), 512, 1, &thresholds());
        assert!(!has(&eval, "ok", QualityCheck::Resolution));
        assert!(!has(&eval, "ok", QualityCheck::CropLoss));
    }

    #[test]
    fn blur_fires_on_absolute_floor() {
        let mut soft = item("soft");
        soft.scalars = Some(scalars(10.0, 0.0, 0.0, vec![0; 8]));
        let eval = evaluate_tier0(std::slice::from_ref(&soft), 512, 1, &thresholds());
        assert!(has(&eval, "soft", QualityCheck::Blur));
    }

    #[test]
    fn blur_fires_on_relative_outlier_even_above_floor() {
        // A custom low floor so only the relative-to-median rule can trip.
        let mut th = thresholds();
        th.blur_floor = 1.0;
        let sharp = |id: &str, v: f64| {
            let mut it = item(id);
            it.scalars = Some(scalars(v, 0.0, 0.0, vec![0; 8]));
            it
        };
        // Median ≈ 1000; the outlier at 100 is < 0.5 × median but well above the floor.
        let items = [
            sharp("a", 1000.0),
            sharp("b", 1100.0),
            sharp("c", 1050.0),
            sharp("outlier", 100.0),
        ];
        let eval = evaluate_tier0(&items, 512, 1, &th);
        assert!(has(&eval, "outlier", QualityCheck::Blur));
        assert!(!has(&eval, "a", QualityCheck::Blur));
    }

    #[test]
    fn uniformly_soft_set_still_flags_via_floor() {
        // Every image is soft, so the median is soft too — the absolute floor must still catch it.
        let soft = |id: &str| {
            let mut it = item(id);
            it.scalars = Some(scalars(20.0, 0.0, 0.0, vec![0; 8]));
            it
        };
        let eval = evaluate_tier0(&[soft("a"), soft("b"), soft("c")], 512, 1, &thresholds());
        assert!(has(&eval, "a", QualityCheck::Blur));
    }

    #[test]
    fn exposure_fires_on_clipping() {
        let mut dark = item("dark");
        dark.scalars = Some(scalars(5000.0, 0.4, 0.0, vec![1; 8]));
        let eval = evaluate_tier0(std::slice::from_ref(&dark), 512, 1, &thresholds());
        let flag = flags_for(&eval, "dark")
            .iter()
            .find(|f| f.check == QualityCheck::Exposure)
            .expect("exposure flag");
        assert_eq!(flag.value, Some(0.4));
    }

    #[test]
    fn exact_duplicates_reference_each_other() {
        let mut a = item("a");
        a.content_hash = Some("samehash".to_owned());
        let mut b = item("b");
        b.content_hash = Some("samehash".to_owned());
        let mut c = item("c");
        c.content_hash = Some("different".to_owned());

        let eval = evaluate_tier0(&[a, b, c], 512, 1, &thresholds());
        let a_dup = flags_for(&eval, "a")
            .iter()
            .find(|f| f.check == QualityCheck::ExactDuplicate)
            .expect("exact-dup flag on a");
        assert_eq!(a_dup.peers, vec!["b".to_owned()]);
        assert!(!has(&eval, "c", QualityCheck::ExactDuplicate));
    }

    #[test]
    fn near_duplicates_cluster_by_hamming_but_not_exact_pairs() {
        // a/b differ by 1 bit (near); c is far; d is byte-identical to a (exact, not near).
        let near = |id: &str, hash: u8, content: &str| {
            let mut it = item(id);
            it.content_hash = Some(content.to_owned());
            it.scalars = Some(scalars(5000.0, 0.0, 0.0, vec![hash, 0, 0, 0, 0, 0, 0, 0]));
            it
        };
        let items = [
            near("a", 0b0000_0000, "ha"),
            near("b", 0b0000_0001, "hb"),
            near("c", 0b1111_1111, "hc"),
            near("d", 0b0000_0000, "ha"), // same hash AND same content as a
        ];
        let eval = evaluate_tier0(&items, 512, 1, &thresholds());

        let a_near = flags_for(&eval, "a")
            .iter()
            .find(|f| f.check == QualityCheck::NearDuplicate)
            .expect("near-dup flag on a");
        assert!(a_near.peers.contains(&"b".to_owned()));
        // d is an exact duplicate of a, so it must NOT also appear as a near-duplicate peer.
        assert!(!a_near.peers.contains(&"d".to_owned()));
        assert!(a_near.peers.iter().all(|p| p != "c"));
    }

    #[test]
    fn count_warns_then_blocks_below_hard_floor() {
        let warn = evaluate_tier0(
            &[item("a"), item("b"), item("c"), item("d"), item("e")],
            512,
            12,
            &thresholds(),
        );
        let warn_flag = warn
            .dataset
            .iter()
            .find(|f| f.check == QualityCheck::Count)
            .expect("count flag");
        assert_eq!(warn_flag.severity, Severity::Warn);

        let block = evaluate_tier0(&[item("a"), item("b")], 512, 12, &thresholds());
        let block_flag = block
            .dataset
            .iter()
            .find(|f| f.check == QualityCheck::Count)
            .expect("count flag");
        assert_eq!(block_flag.severity, Severity::Fatal);
    }

    #[test]
    fn empty_dataset_only_reports_count() {
        let eval = evaluate_tier0(&[], 512, 10, &thresholds());
        assert!(eval.items.is_empty());
        assert_eq!(eval.dataset.len(), 1);
        assert_eq!(eval.dataset[0].check, QualityCheck::Count);
    }

    #[test]
    fn flags_round_trip_as_camelcase_json() {
        let flag = QualityFlag {
            check: QualityCheck::NearDuplicate,
            severity: Severity::Warn,
            value: Some(2.0),
            threshold: Some(6.0),
            peers: vec!["other".to_owned()],
            acknowledged: false,
        };
        let json = serde_json::to_string(&flag).expect("serialize");
        // Field names are camelCase; the check enum value is the snake_case string the rest of
        // the contract crate uses for string enums.
        assert!(json.contains("\"check\":\"near_duplicate\""));
        assert!(json.contains("\"peers\":[\"other\"]"));
        let back: QualityFlag = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, flag);
    }

    // --- sc-6533 readiness report ---

    fn sharp(id: &str, phash: u8) -> ItemQualityInput {
        let mut it = item(id);
        it.scalars = Some(scalars(5000.0, 0.0, 0.0, vec![phash; 8]));
        it
    }

    #[test]
    fn gate_is_ready_when_every_item_is_clean() {
        let eval = evaluate_tier0(&[sharp("a", 0xAA), sharp("b", 0x55)], 512, 1, &thresholds());
        let report = build_readiness_report(eval);
        assert_eq!(report.gate, ReadinessGate::Ready);
        assert!((report.sub_scores.technical - 1.0).abs() < 1e-9);
        assert!(report.sub_scores.diversity.is_none()); // Tier-1 not computed yet
        assert_eq!(report.counts, SeverityCounts::default());
        assert!(report.items.iter().all(|i| i.severity.is_none()));
    }

    #[test]
    fn gate_needs_attention_and_technical_share_drops_on_warning() {
        let mut soft = item("soft");
        soft.scalars = Some(scalars(10.0, 0.0, 0.0, vec![0x0F; 8]));
        let eval = evaluate_tier0(&[soft, sharp("ok", 0xF0)], 512, 1, &thresholds());
        let report = build_readiness_report(eval);
        assert_eq!(report.gate, ReadinessGate::NeedsAttention);
        assert!(report.counts.warn >= 1);
        assert!((report.sub_scores.technical - 0.5).abs() < 1e-9);
        let soft_item = report.items.iter().find(|i| i.item_id == "soft").unwrap();
        assert_eq!(soft_item.severity, Some(Severity::Warn));
        let ok_item = report.items.iter().find(|i| i.item_id == "ok").unwrap();
        assert_eq!(ok_item.severity, None);
    }

    #[test]
    fn gate_blocks_when_too_few_images() {
        let eval = evaluate_tier0(&[item("a"), item("b")], 512, 12, &thresholds());
        let report = build_readiness_report(eval);
        assert_eq!(report.gate, ReadinessGate::Blocked);
        assert!(report.counts.fatal >= 1);
        assert!(report
            .dataset_flags
            .iter()
            .any(|f| f.check == QualityCheck::Count));
    }

    #[test]
    fn decode_failure_flag_keeps_item_out_of_the_technical_share() {
        // An undecodable item carries a Decode warning (injected upstream) and must not count as
        // "fine" even though it has no scalar-derived flags.
        let mut broken = item("broken");
        broken.scalars = None;
        let mut eval = evaluate_tier0(&[broken, sharp("ok", 0x3C)], 512, 1, &thresholds());
        eval.items
            .iter_mut()
            .find(|e| e.item_id == "broken")
            .unwrap()
            .flags
            .push(QualityFlag {
                check: QualityCheck::Decode,
                severity: Severity::Warn,
                value: None,
                threshold: None,
                peers: Vec::new(),
                acknowledged: false,
            });
        let report = build_readiness_report(eval);
        assert!((report.sub_scores.technical - 0.5).abs() < 1e-9);
        assert_eq!(report.gate, ReadinessGate::NeedsAttention);
    }

    #[test]
    fn readiness_context_resolves_kind_bucket_and_min_items() {
        let person = readiness_context(Some(1024), &["character".to_owned()], Some("person"), None);
        assert_eq!(person.kind, DatasetKind::Person);
        assert_eq!(person.bucket_edge, 1024);
        assert_eq!(person.min_items, 15);

        // 1000 floors to a multiple of 32; style raises the default minimum.
        let style = readiness_context(Some(1000), &["style".to_owned()], None, None);
        assert_eq!(style.kind, DatasetKind::Style);
        assert_eq!(style.bucket_edge, 992);
        assert_eq!(style.min_items, 20);

        // No target → default bucket; explicit min wins over the per-kind default.
        let object = readiness_context(None, &[], None, Some(8));
        assert_eq!(object.kind, DatasetKind::Object);
        assert_eq!(object.bucket_edge, DEFAULT_BUCKET_EDGE);
        assert_eq!(object.min_items, 8);
    }

    #[test]
    fn report_round_trips_as_camelcase_json() {
        let eval = evaluate_tier0(&[sharp("a", 0x11)], 512, 1, &thresholds());
        let report = build_readiness_report(eval);
        let json = serde_json::to_string(&report).expect("serialize");
        assert!(json.contains("\"subScores\""));
        assert!(json.contains("\"itemCount\""));
        let back: DatasetReadinessReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, report);
    }

    #[test]
    fn cached_scalars_valid_only_for_same_hash_and_bucket() {
        let cache = CachedTier0Scalars {
            content_hash: "abc".to_owned(),
            bucket_edge: 512,
            scalars: scalars(5000.0, 0.0, 0.0, vec![0; 8]),
        };
        assert!(cache.valid_for(Some("abc"), 512));
        assert!(
            !cache.valid_for(Some("abc"), 1024),
            "bucket change invalidates"
        );
        assert!(
            !cache.valid_for(Some("xyz"), 512),
            "content change invalidates"
        );
        assert!(!cache.valid_for(None, 512), "missing hash invalidates");
    }

    #[test]
    fn acknowledged_findings_drop_from_every_rollup() {
        // Two soft images; the user dismisses blur on `a` only. Far-apart pHashes keep them from
        // also registering as near-duplicates, so blur is the only finding in play.
        let soft = |id: &str, phash: u8, ack: &[QualityCheck]| {
            let mut it = item(id);
            it.scalars = Some(scalars(10.0, 0.0, 0.0, vec![phash; 8]));
            it.acknowledged = ack.to_vec();
            it
        };
        let eval = evaluate_tier0(
            &[soft("a", 0x00, &[QualityCheck::Blur]), soft("b", 0xFF, &[])],
            512,
            1,
            &thresholds(),
        );
        // The flag is kept on `a` for display, but marked acknowledged; `b`'s stands.
        let a_blur = flags_for(&eval, "a")
            .iter()
            .find(|f| f.check == QualityCheck::Blur)
            .expect("blur flag on a");
        assert!(a_blur.acknowledged);
        let b_blur = flags_for(&eval, "b")
            .iter()
            .find(|f| f.check == QualityCheck::Blur)
            .expect("blur flag on b");
        assert!(!b_blur.acknowledged);

        let report = build_readiness_report(eval);
        assert_eq!(report.counts.warn, 1, "only b's warning counts");
        assert!(
            (report.sub_scores.technical - 0.5).abs() < 1e-9,
            "a counts as clean once dismissed"
        );
        let a = report.items.iter().find(|i| i.item_id == "a").expect("a");
        assert_eq!(a.severity, None, "a badges clean");
        assert_eq!(report.gate, ReadinessGate::NeedsAttention);
    }

    #[test]
    fn all_findings_acknowledged_reads_ready() {
        let mut soft = item("solo");
        soft.scalars = Some(scalars(10.0, 0.0, 0.0, vec![0; 8]));
        soft.acknowledged = vec![QualityCheck::Blur];
        let report = build_readiness_report(evaluate_tier0(
            std::slice::from_ref(&soft),
            512,
            1,
            &thresholds(),
        ));
        assert_eq!(report.gate, ReadinessGate::Ready);
        assert_eq!(report.counts.warn, 0);
        assert!((report.sub_scores.technical - 1.0).abs() < 1e-9);
        // The dismissed flag is still in the payload so the UI can show it struck-through.
        assert!(report.items[0]
            .flags
            .iter()
            .any(|f| f.check == QualityCheck::Blur && f.acknowledged));
    }

    #[test]
    fn dismissing_a_duplicate_is_asymmetric() {
        // a and b are near-dupes (1 bit apart); the user dismisses only on a, so b still counts.
        let near = |id: &str, hash: u8, ack: &[QualityCheck]| {
            let mut it = item(id);
            it.scalars = Some(scalars(5000.0, 0.0, 0.0, vec![hash, 0, 0, 0, 0, 0, 0, 0]));
            it.acknowledged = ack.to_vec();
            it
        };
        let eval = evaluate_tier0(
            &[
                near("a", 0b0000_0000, &[QualityCheck::NearDuplicate]),
                near("b", 0b0000_0001, &[]),
            ],
            512,
            2,
            &thresholds(),
        );
        let a = flags_for(&eval, "a")
            .iter()
            .find(|f| f.check == QualityCheck::NearDuplicate)
            .expect("near-dup on a");
        assert!(a.acknowledged, "a's dup is dismissed");
        let b = flags_for(&eval, "b")
            .iter()
            .find(|f| f.check == QualityCheck::NearDuplicate)
            .expect("near-dup on b");
        assert!(!b.acknowledged, "b is still a duplicate of something");
    }

    #[test]
    fn quality_ack_voids_when_content_hash_changes() {
        let ack = QualityAck {
            content_hash: "abc".to_owned(),
            checks: vec![QualityCheck::Blur],
        };
        assert!(ack.valid_for(Some("abc")));
        assert_eq!(ack.effective_checks(Some("abc")), vec![QualityCheck::Blur]);
        // Replaced image (new hash) or missing hash → the dismissal no longer applies.
        assert!(!ack.valid_for(Some("xyz")));
        assert!(ack.effective_checks(Some("xyz")).is_empty());
        assert!(ack.effective_checks(None).is_empty());
    }
}
