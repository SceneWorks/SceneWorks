//! SC-20686 persistent-KV attribution observer for the **MLX Metal lane**.
//!
//! This is the Metal counterpart of the Candle/CUDA observers in `candle-gen-wan` and
//! `candle-gen-flux2`. It writes the same JSONL transcript schema the campaign adapter
//! (`scripts/sc20686_campaign_adapter.py`) validates, adds `"backend": "mlx-metal"` to every event,
//! and adds `phase-window` events that attribute allocator/process memory per generation phase.
//!
//! **Inert by default.** Nothing here runs unless a campaign entrypoint armed an output request on
//! the rendering thread ([`request_output`] + [`CampaignRequest::arm`]) *and* the product provider
//! then entered [`observe_generation`]. Every hook first checks a thread-local that is `None` for
//! ordinary generation, so the production path performs no evaluation, allocation, string
//! formatting or allocator reset.
//!
//! # Memory attribution method
//!
//! MLX exposes process-global allocator counters: `active` (bytes held by live arrays), `cache`
//! (freed-but-retained buffers) and `peak` (active high-water since the last reset). The observer
//! owns every reset while a campaign is armed and folds the counter into three nested high-waters
//! before each reset:
//!
//! * the **run** high-water (`peak_bytes` on every event; the admission-domain peak the reducer
//!   compares against), which also folds every sampled `active + cache` reservation;
//! * the current **phase window** (`encode`, `load`, `prepare-cache`, `denoise-step`, `decode`,
//!   `post-denoise`), emitted as a `phase-window` event with Darwin `phys_footprint`;
//! * one **read window** around each cross-attention read.
//!
//! MLX is lazy, so a read only allocates when its graph is evaluated. While a campaign is armed a
//! read window therefore evaluates its inputs first (so their production is outside the window),
//! resets the peak, runs the attention, evaluates the output and records
//! `high - before` as the physical read transient. Those extra evaluation boundaries exist only in
//! campaign mode; they are a documented measurement perturbation of the lazy schedule, never a
//! product behaviour.
//!
//! Persistent cache bytes are the exact `nbytes` of the retained K/V arrays (not a shape estimate),
//! registered when the product creates the cache and released when the product drops it.
//! Recomputed K/V (FLUX.2 edit reference slices, MLX Wan-VACE text K/V) is reported as transient
//! with zero persistent bytes, mirroring the Candle FLUX contract.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::{Array, Dtype};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{CancelFlag, Error, GenerationRequest, LoadPhase, Progress, Result};

/// Backend identity every Metal-lane event carries. The adapter rejects a Metal-lane transcript
/// without it and a CUDA-lane transcript that claims it.
pub const BACKEND: &str = "mlx-metal";
/// SC-20675 packed group-affine group size used for the compatibility projection.
pub const GROUP_SIZE: u64 = 32;

/// Phase-window names the adapter accepts.
pub const PHASE_WINDOWS: [&str; 6] = [
    "encode",
    "load",
    "prepare-cache",
    "denoise-step",
    "post-denoise",
    "decode",
];

/// MLX allocator counters. The production implementation reads the Metal allocator; tests inject a
/// scripted implementation so attribution is exact and independent of concurrently running tests.
pub trait AllocatorCounters {
    fn active(&self) -> u64;
    fn peak(&self) -> u64;
    fn cache(&self) -> u64;
    fn reset_peak(&self);
}

struct MlxCounters;

impl AllocatorCounters for MlxCounters {
    fn active(&self) -> u64 {
        mlx_rs::memory::get_active_memory() as u64
    }
    fn peak(&self) -> u64 {
        mlx_rs::memory::get_peak_memory() as u64
    }
    fn cache(&self) -> u64 {
        mlx_rs::memory::get_cache_memory() as u64
    }
    fn reset_peak(&self) {
        mlx_rs::memory::reset_peak_memory();
    }
}

/// Darwin process footprint `(phys_footprint, phys_footprint_peak)`.
pub trait FootprintProbe {
    fn sample(&self) -> Option<(u64, u64)>;
}

struct DarwinFootprint;

impl FootprintProbe for DarwinFootprint {
    fn sample(&self) -> Option<(u64, u64)> {
        own_phys_footprint()
    }
}

/// Own `(phys_footprint, lifetime peak)` via `proc_pid_rusage`, not a footprint subprocess.
#[cfg(target_os = "macos")]
fn own_phys_footprint() -> Option<(u64, u64)> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: the RUSAGE_INFO_V4 flavor writes at most one `rusage_info_v4` into the zeroed buffer.
    let status = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if status != 0 {
        return None;
    }
    // SAFETY: the successful call initialized the buffer (and it was zeroed before).
    let info = unsafe { info.assume_init() };
    footprint_pair(info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
}

#[cfg(not(target_os = "macos"))]
fn own_phys_footprint() -> Option<(u64, u64)> {
    None
}

/// A footprint sample is usable only when it is positive and its peak covers it.
pub fn footprint_pair(current: u64, peak: u64) -> Option<(u64, u64)> {
    (current > 0 && peak >= current).then_some((current, peak))
}

/// Destination of transcript events.
pub trait EventSink {
    fn emit(&mut self, event: Value);
}

struct JsonlSink(Box<dyn Write>);

impl EventSink for JsonlSink {
    fn emit(&mut self, event: Value) {
        let line = serde_json::to_string(&event).unwrap_or_else(|_| "{}".into()) + "\n";
        let _ = self.0.write_all(line.as_bytes());
        let _ = self.0.flush();
    }
}

/// Request-derived facts. Built lazily by [`observe_generation`] only when a campaign is armed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestFacts {
    pub batch: u32,
    pub frames: u32,
    pub width: u32,
    pub height: u32,
    pub prompt_sha256: String,
    pub guidance: String,
    pub reference_count: u32,
}

impl RequestFacts {
    /// The facts SceneWorks' request carries: batch = `count`, frames (1 for images), geometry,
    /// prompt digest, guidance, and the still-image reference count.
    pub fn from_request(req: &GenerationRequest) -> Self {
        Self {
            batch: req.count.max(1),
            frames: req.frames.unwrap_or(1).max(1),
            width: req.width,
            height: req.height,
            prompt_sha256: format!("{:x}", Sha256::digest(req.prompt.as_bytes())),
            guidance: req
                .guidance
                .map_or_else(|| "none".into(), |value| value.to_string()),
            reference_count: req.image_reference_count(),
        }
    }
}

/// Live attention geometry bound by the product hook that owns the K/V.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvGeometry {
    /// Attention layers that project this K/V per forward.
    pub layers: u32,
    pub heads: u32,
    pub head_dimension: u32,
    /// Query tokens (latent/target tokens).
    pub sq: u64,
    /// K/V tokens attributed to the cache (text tokens for Wan, reference tokens for FLUX).
    pub skv: u64,
}

#[derive(Clone, Debug)]
struct Context {
    source_ref: String,
    model_snapshot_revision: String,
    residency_strategy: String,
    snapshot_sha256: String,
    snapshot_bytes: u64,
    variant: String,
    facts: RequestFacts,
    geometry: KvGeometry,
    dtype: String,
    mask: String,
    rope: String,
    real_weights: Option<bool>, // None until a product projection confirms it; gates metadata.
}

impl Context {
    fn geometry_ready(&self) -> bool {
        let g = &self.geometry;
        g.layers != 0
            && g.heads != 0
            && g.head_dimension != 0
            && g.sq != 0
            && g.skv != 0
            && !self.dtype.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CacheAccounting {
    dense_bytes: u64,
    candidate_bytes: u64,
    reads: u64,
}

/// Identity of one product-owned cache: an owner plus a stream and slot. Wan's per-block cross-K/V
/// (stream 0) is owned by the stable heap buffer of its per-generate `StepCache` set, so every read
/// resolves by the address of the `(k, v)` tuple it receives; FLUX.2's reference-K/V cache
/// (streams 1 = double, 2 = single) by a process-unique cache id, which survives moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheKey {
    pub owner: usize,
    pub stream: u8,
    pub slot: usize,
}

struct PhaseWindow {
    name: &'static str,
    index: u32,
    started: Instant,
    before: u64,
    high: u64,
}

struct State {
    sink: Box<dyn EventSink>,
    counters: Box<dyn AllocatorCounters>,
    footprint: Box<dyn FootprintProbe>,
    context: Context,
    cancel: CancelFlag,
    cancellation_armed: bool,
    schedule_control: bool,
    started: Instant,
    metadata_emitted: bool,
    start_emitted: bool,
    terminal_emitted: bool,
    live_read_seen: bool,
    cancel_triggered: bool,
    run_high: u64,
    phase: Option<PhaseWindow>,
    last_step: u32,
    next_cache_id: u64,
    keys: BTreeMap<CacheKey, u64>,
    caches: BTreeMap<u64, CacheAccounting>,
    completed_reads: Vec<u64>,
    live_dense: u64,
    peak_dense: u64,
    live_candidate: u64,
    peak_candidate: u64,
    recomputed_projections: u64,
    recomputed_dense: u64,
    recomputed_candidate: u64,
    max_read_transient: u64,
    reads: u64,
    attributable_read_ns: u128,
    joint_read_ns: u128,
    non_attributable_reads: u64,
    reference_forward: bool,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
    static PENDING: RefCell<Option<Pending>> = const { RefCell::new(None) };
    static LAST_CANCELLED: Cell<bool> = const { Cell::new(false) };
}

struct Pending {
    path: PathBuf,
    source_ref: String,
    residency_strategy: String,
    cancellation: bool,
    schedule_control: bool,
}

/// True only while an armed campaign is observing this thread's generation. Product hooks test this
/// before doing any campaign-only work.
pub fn active() -> bool {
    STATE.with(|state| state.borrow().is_some())
}

/// Whether the most recent armed campaign was cancelled by the product after its first live read.
pub fn campaign_cancelled() -> bool {
    LAST_CANCELLED.with(Cell::get)
}

fn is_lowercase_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Entrypoint-side request. The adapter supplies the verified inference revision and the frozen
/// product residency; model identity and geometry are derived by the runtime.
pub struct CampaignRequest {
    path: PathBuf,
    source_ref: String,
    residency_strategy: String,
    cancellation: bool,
    schedule_control: bool,
}

pub fn request_output(
    path: impl Into<PathBuf>,
    source_ref: impl Into<String>,
    residency_strategy: impl Into<String>,
) -> io::Result<CampaignRequest> {
    let path = path.into();
    let source_ref = source_ref.into();
    let residency_strategy = residency_strategy.into();
    if path.as_os_str().is_empty() || path == Path::new("-") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SC-20686 campaign requires a dedicated event file",
        ));
    }
    if !is_lowercase_revision(&source_ref) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "inference source revision must be lowercase 40-hex",
        ));
    }
    if !matches!(residency_strategy.as_str(), "resident" | "sequential") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "residency strategy must be resident or sequential",
        ));
    }
    Ok(CampaignRequest {
        path,
        source_ref,
        residency_strategy,
        cancellation: false,
        schedule_control: false,
    })
}

impl CampaignRequest {
    pub fn arm(self) -> Self {
        PENDING.with(|slot| {
            *slot.borrow_mut() = Some(Pending {
                path: self.path.clone(),
                source_ref: self.source_ref.clone(),
                residency_strategy: self.residency_strategy.clone(),
                cancellation: self.cancellation,
                schedule_control: self.schedule_control,
            })
        });
        self
    }

    /// Arm the deliberate cancellation arm: the product is cancelled after its first live read.
    pub fn arm_cancellation(mut self) -> Self {
        self.cancellation = true;
        self.arm()
    }

    /// Arm the **schedule-control** arm: phase windows, cache creation/release and footprint only,
    /// with no per-read evaluation windows, so run and phase peaks follow the product's own lazy
    /// schedule. Read attribution comes from the normal arm; this arm is its schedule control.
    pub fn arm_schedule_control(mut self) -> Self {
        self.schedule_control = true;
        self.arm()
    }
}

impl Drop for CampaignRequest {
    fn drop(&mut self) {
        PENDING.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot
                .as_ref()
                .is_some_and(|pending| pending.path == self.path)
            {
                *slot = None;
            }
        });
    }
}

fn snapshot_identity(root: &Path) -> io::Result<(String, u64)> {
    fn collect(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.file_name().is_some_and(|name| name == ".git") {
                continue;
            }
            if path.is_dir() {
                collect(root, &path, out)?;
            } else if path.is_file() {
                out.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
            }
        }
        Ok(())
    }
    let mut names = Vec::new();
    collect(root, root, &mut names)?;
    if names.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "snapshot inventory is empty",
        ));
    }
    names.sort();
    let mut aggregate = Sha256::new();
    let mut total = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    for name in names {
        let mut file = File::open(root.join(&name))?;
        let mut digest = Sha256::new();
        let mut bytes = 0u64;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            bytes = bytes.checked_add(read as u64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "snapshot byte count overflow")
            })?;
            digest.update(&buffer[..read]);
        }
        total = total.checked_add(bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "snapshot byte count overflow")
        })?;
        aggregate.update(name.to_string_lossy().as_bytes());
        aggregate.update([0]);
        aggregate.update(bytes.to_le_bytes());
        aggregate.update([0]);
        aggregate.update(digest.finalize());
        aggregate.update([b'\n']);
    }
    Ok((format!("{:x}", aggregate.finalize()), total))
}

/// The immutable model revision of `root`: the root or one of its two nearest ancestors is a
/// 40-hex revision directory (or carries `.snapshot-revision`), so nested tiers such as
/// `<revision>/q4` resolve without widening the content hash beyond `root`.
fn model_snapshot_revision(root: &Path) -> io::Result<String> {
    for candidate in root.ancestors().take(3) {
        if let Some(name) = candidate
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| is_lowercase_revision(name))
        {
            return Ok(name.to_owned());
        }
        if let Ok(marker) = std::fs::read_to_string(candidate.join(".snapshot-revision")) {
            let marker = marker.trim();
            if is_lowercase_revision(marker) {
                return Ok(marker.to_owned());
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "snapshot root or parent has no lowercase 40-hex immutable model revision",
    ))
}

/// Components injected by tests; production uses the Metal allocator, `proc_pid_rusage` and a
/// JSONL file.
pub struct Instruments {
    pub sink: Box<dyn EventSink>,
    pub counters: Box<dyn AllocatorCounters>,
    pub footprint: Box<dyn FootprintProbe>,
}

/// Observed scope. Dropping it emits `invalidated` and the post-release allocator remnant, then
/// deactivates the observer.
pub struct Scope(());

impl Drop for Scope {
    fn drop(&mut self) {
        with_state(|state| {
            close_phase(state);
            emit(state, "invalidated", "invalidated", EventFields::default());
            let remnant = remnant(state);
            emit(state, "released", "release-remnant", remnant);
        });
        STATE.with(|slot| *slot.borrow_mut() = None);
    }
}

fn activate_pending(
    snapshot_root: &Path,
    cancel: &CancelFlag,
    variant: &str,
    facts: impl FnOnce() -> RequestFacts,
    instruments: impl FnOnce(&Path) -> io::Result<Instruments>,
) -> io::Result<Option<Scope>> {
    let Some(pending) = PENDING.with(|slot| slot.borrow_mut().take()) else {
        return Ok(None);
    };
    if active() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "an SC-20686 campaign is already observing this thread",
        ));
    }
    let (snapshot_sha256, snapshot_bytes) = snapshot_identity(snapshot_root)?;
    let model_snapshot_revision = model_snapshot_revision(snapshot_root)?;
    let facts = facts();
    if facts.batch == 0 || facts.frames == 0 || facts.width == 0 || facts.height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "runtime campaign facts are incomplete",
        ));
    }
    let instruments = instruments(&pending.path)?;
    let context = Context {
        source_ref: pending.source_ref,
        model_snapshot_revision,
        residency_strategy: pending.residency_strategy,
        snapshot_sha256,
        snapshot_bytes,
        variant: variant.to_owned(),
        facts,
        geometry: KvGeometry::default(),
        dtype: String::new(),
        mask: "none".into(),
        rope: "none".into(),
        real_weights: None,
    };
    let run_high = instruments
        .counters
        .peak()
        .max(instruments.counters.active() + instruments.counters.cache());
    let state = State {
        sink: instruments.sink,
        counters: instruments.counters,
        footprint: instruments.footprint,
        context,
        cancel: cancel.clone(),
        cancellation_armed: pending.cancellation,
        schedule_control: pending.schedule_control,
        started: Instant::now(),
        metadata_emitted: false,
        start_emitted: false,
        terminal_emitted: false,
        live_read_seen: false,
        cancel_triggered: false,
        run_high,
        phase: None,
        last_step: 0,
        next_cache_id: 1,
        keys: BTreeMap::new(),
        caches: BTreeMap::new(),
        completed_reads: Vec::new(),
        live_dense: 0,
        peak_dense: 0,
        live_candidate: 0,
        peak_candidate: 0,
        recomputed_projections: 0,
        recomputed_dense: 0,
        recomputed_candidate: 0,
        max_read_transient: 0,
        reads: 0,
        attributable_read_ns: 0,
        joint_read_ns: 0,
        non_attributable_reads: 0,
        reference_forward: false,
    };
    LAST_CANCELLED.with(|slot| slot.set(false));
    STATE.with(|slot| *slot.borrow_mut() = Some(state));
    with_state(|state| open_phase(state, "encode", 0));
    Ok(Some(Scope(())))
}

fn production_instruments(path: &Path) -> io::Result<Instruments> {
    Ok(Instruments {
        sink: Box::new(JsonlSink(Box::new(File::create(path)?))),
        counters: Box::new(MlxCounters),
        footprint: Box::new(DarwinFootprint),
    })
}

/// Test/diagnostic activation with injected instruments. Still requires an armed request, so it
/// cannot observe an ordinary generation.
pub fn activate_with_instruments(
    snapshot_root: &Path,
    cancel: &CancelFlag,
    variant: &str,
    facts: RequestFacts,
    instruments: Instruments,
) -> io::Result<Option<Scope>> {
    activate_pending(
        snapshot_root,
        cancel,
        variant,
        || facts,
        |_| Ok(instruments),
    )
}

/// Run one product generation under the observer when (and only when) a campaign is armed. The
/// provider's own progress stream drives the phase windows; the terminal event is emitted before
/// the scope's invalidation/release so the transcript keeps product order.
pub fn observe_generation<T>(
    snapshot_root: &Path,
    cancel: &CancelFlag,
    variant: &str,
    facts: impl FnOnce() -> RequestFacts,
    on_progress: &mut dyn FnMut(Progress),
    run: impl FnOnce(&mut dyn FnMut(Progress)) -> Result<T>,
) -> Result<T> {
    observe_generation_with(
        snapshot_root,
        cancel,
        variant,
        facts,
        production_instruments,
        on_progress,
        run,
    )
}

/// [`observe_generation`] with injected instruments (tests drive the full wrapper with scripted
/// allocator counters and a capturing sink). Still inert unless a request is armed.
pub fn observe_generation_with<T>(
    snapshot_root: &Path,
    cancel: &CancelFlag,
    variant: &str,
    facts: impl FnOnce() -> RequestFacts,
    instruments: impl FnOnce(&Path) -> io::Result<Instruments>,
    on_progress: &mut dyn FnMut(Progress),
    run: impl FnOnce(&mut dyn FnMut(Progress)) -> Result<T>,
) -> Result<T> {
    if PENDING.with(|slot| slot.borrow().is_none()) {
        return run(on_progress);
    }
    let scope = activate_pending(snapshot_root, cancel, variant, facts, instruments)
        .map_err(|error| Error::Msg(format!("{variant}: arm SC-20686 campaign: {error}")))?;
    let Some(scope) = scope else {
        return run(on_progress);
    };
    let result = {
        let mut forward = |progress: Progress| {
            observe_progress(&progress);
            on_progress(progress);
        };
        run(&mut forward)
    };
    match &result {
        Ok(_) => observe_generation_end(),
        Err(Error::Canceled) => observe_cancelled(),
        Err(_) => {}
    }
    drop(scope);
    result
}

/// Record a normal completion (`generation-end` + `metrics`). Idempotent.
pub fn observe_generation_end() {
    with_state(|state| {
        if state.terminal_emitted {
            return;
        }
        close_phase(state);
        state.terminal_emitted = true;
        emit(state, "generation-end", "terminal", EventFields::default());
        emit_metrics(state);
    });
}

/// Record the product cancellation of an armed cancellation arm. Only a cancellation after a live
/// K/V read counts: a preflight refusal can never masquerade as the cancellation oracle.
pub fn observe_cancelled() {
    with_state(|state| {
        if !state.cancellation_armed || !state.live_read_seen || state.terminal_emitted {
            return;
        }
        close_phase(state);
        state.terminal_emitted = true;
        LAST_CANCELLED.with(|slot| slot.set(true));
        emit(state, "cancelled", "terminal", EventFields::default());
        emit_metrics(state);
    });
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    STATE.with(|slot| slot.borrow_mut().as_mut().map(f))
}

/// Advance phase windows from the provider's progress stream.
pub fn observe_progress(progress: &Progress) {
    with_state(|state| match *progress {
        Progress::Step { current, total } => {
            state.last_step = current;
            if current < total {
                open_phase(state, "denoise-step", current + 1);
            } else {
                open_phase(state, "post-denoise", current);
            }
        }
        Progress::Decoding => open_phase(state, "decode", state.last_step),
        // The prompt encoder's load is part of the conditioning encode: a staged-residency provider
        // (gen-core `StagedResidency`, e.g. FLUX.2 Klein) announces it before loading and encoding,
        // and mapping it to `load` left the generation-start `encode` window empty -- an allocator
        // sample with a zero peak, which the adapter (rightly) refuses as missing evidence -- while
        // the text encode was misattributed to `load`. The render bundle's load stays `load`.
        Progress::Loading(LoadPhase::TextEncoder) => enter_phase(state, "encode"),
        Progress::Loading(LoadPhase::Renderer) => enter_phase(state, "load"),
    });
}

/// Mark the start of a product phase (`prepare-cache` before a cache build, `load` before a
/// mid-denoise component load).
pub fn mark_phase(name: &'static str) {
    with_state(|state| enter_phase(state, name));
}

/// Enter `name`. A mid-denoise load or cache build (the MoE expert swap) interrupts a
/// `denoise-step` window that has not run its step yet: that window is relabelled rather than closed,
/// so the swap work is attributed to `name`; re-entering the open window is a no-op. Every
/// `(window, index)` therefore stays unique.
fn enter_phase(state: &mut State, name: &'static str) {
    let index = state.last_step;
    match state.phase.as_mut() {
        Some(window) if window.name == "denoise-step" && window.index > state.last_step => {
            window.name = name;
            window.index = index;
        }
        // A load announced twice (an explicit mark and the provider's `Progress::Loading`) is one
        // window.
        Some(window) if window.name == name && window.index == index => {}
        _ => open_phase(state, name, index),
    }
}

/// Mark that the next evaluated work is the next denoise step.
pub fn mark_denoise() {
    with_state(|state| {
        let index = state.last_step + 1;
        open_phase(state, "denoise-step", index)
    });
}

fn fold_peak(state: &mut State) -> u64 {
    let peak = state.counters.peak().max(state.counters.active());
    if let Some(phase) = state.phase.as_mut() {
        phase.high = phase.high.max(peak);
    }
    let reserved = state.counters.active() + state.counters.cache();
    state.run_high = state.run_high.max(peak).max(reserved);
    peak
}

fn close_phase(state: &mut State) {
    let Some(mut window) = state.phase.take() else {
        return;
    };
    let peak = state.counters.peak().max(state.counters.active());
    window.high = window.high.max(peak);
    let after = state.counters.active();
    let reserved = after + state.counters.cache();
    state.run_high = state.run_high.max(window.high).max(reserved);
    let (footprint, footprint_peak) = state.footprint.sample().unwrap_or((0, 0));
    let fields = EventFields {
        allocator: Some(AllocatorWindow {
            before: window.before,
            after,
            high: window.high.max(window.before).max(after),
            reserved,
        }),
        extra: vec![
            ("window", json!(window.name)),
            ("window_index", json!(window.index)),
            (
                "duration_ms",
                json!(window.started.elapsed().as_secs_f64() * 1000.0),
            ),
            ("phys_footprint_bytes", json!(footprint)),
            ("phys_footprint_peak_bytes", json!(footprint_peak)),
        ],
        ..EventFields::default()
    };
    emit(state, "phase-window", window.name, fields);
}

fn open_phase(state: &mut State, name: &'static str, index: u32) {
    close_phase(state);
    fold_peak(state);
    state.counters.reset_peak();
    let before = state.counters.active();
    state.phase = Some(PhaseWindow {
        name,
        index,
        started: Instant::now(),
        before,
        high: before,
    });
}

#[derive(Clone, Copy, Debug, Default)]
struct AllocatorWindow {
    before: u64,
    after: u64,
    high: u64,
    reserved: u64,
}

#[derive(Default)]
struct EventFields {
    persistent_bytes: u64,
    transient_bytes: u64,
    reused: u64,
    cache_id: u64,
    candidate_persistent_bytes: u64,
    tensor_shape: String,
    allocator: Option<AllocatorWindow>,
    extra: Vec<(&'static str, Value)>,
}

fn remnant(state: &mut State) -> EventFields {
    let active = state.counters.active();
    let reserved = active + state.counters.cache();
    state.run_high = state.run_high.max(reserved).max(active);
    EventFields {
        allocator: Some(AllocatorWindow {
            before: active,
            after: active,
            high: active,
            reserved,
        }),
        ..EventFields::default()
    }
}

fn emit(state: &mut State, phase: &'static str, operation: &str, fields: EventFields) {
    if let Some(window) = fields.allocator {
        state.run_high = state
            .run_high
            .max(window.high)
            .max(window.reserved)
            .max(window.after);
    }
    let allocator = fields.allocator.unwrap_or_default();
    let mut value = json!({
        "phase": phase,
        "backend": BACKEND,
        "attention": "cross",
        "operation": operation,
        "tensor_shape": fields.tensor_shape,
        "dtype": state.context.dtype,
        "mask": state.context.mask,
        "rope": state.context.rope,
        "persistent_bytes": fields.persistent_bytes,
        "transient_bytes": fields.transient_bytes,
        "peak_bytes": state.run_high,
        "reused": fields.reused,
        "elapsed_ms": state.started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        "at_ns": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        "sample_kind": "allocator",
        "cache_id": fields.cache_id,
        "candidate_persistent_bytes": fields.candidate_persistent_bytes,
        "allocator_before_bytes": allocator.before,
        "allocator_after_bytes": allocator.after,
        "allocator_high_bytes": allocator.high,
        "allocator_reserved_bytes": allocator.reserved,
        "allocator_measurement_available": fields.allocator.is_some(),
    });
    for (key, extra) in fields.extra {
        value[key] = extra;
    }
    if phase == "metadata" {
        let context = &state.context;
        let geometry = &context.geometry;
        value["source_ref"] = json!(context.source_ref);
        value["model_snapshot_revision"] = json!(context.model_snapshot_revision);
        value["residency_strategy"] = json!(context.residency_strategy);
        value["snapshot_sha256"] = json!(context.snapshot_sha256);
        value["snapshot_bytes"] = json!(context.snapshot_bytes);
        value["variant"] = json!(context.variant);
        value["geometry"] = json!({
            "batch": context.facts.batch,
            "resolution": format!("{}x{}", context.facts.width, context.facts.height),
            "reference_count": context.facts.reference_count,
            "frames": context.facts.frames,
            "prompt": context.facts.prompt_sha256,
            "guidance": context.facts.guidance,
            "layers": geometry.layers,
            "heads": geometry.heads,
            "head_dimension": geometry.head_dimension,
            "sq": geometry.sq,
            "skv": geometry.skv,
            "dtype": context.dtype,
            "mask": context.mask,
            "rope": context.rope,
        });
        value["cancellation_armed"] = json!(state.cancellation_armed);
        if state.cancellation_armed {
            value["cancellation_arm_id"] = json!(format!(
                "{}:{}:{}",
                BACKEND, context.source_ref, context.variant
            ));
        }
        value["real_weights"] = json!(context.real_weights == Some(true));
        value["full_generation"] = json!(!state.cancellation_armed);
        value["schedule_control"] = json!(state.schedule_control);
        value["attention_kind"] = json!("cross");
    }
    state.sink.emit(value);
}

fn emit_metrics(state: &mut State) {
    let persistent_route = !state.caches.is_empty() || !state.completed_reads.is_empty();
    let layers = u64::from(state.context.geometry.layers);
    let generation_ms = (state.started.elapsed().as_secs_f64() * 1000.0).max(0.001);
    let attributable = state.reads > 0 && state.non_attributable_reads == 0;
    let (current_persistent, candidate_persistent, current_transient, reused, minimum) =
        if persistent_route {
            let minimum = state
                .caches
                .values()
                .map(|cache| cache.reads)
                .chain(state.completed_reads.iter().copied())
                .min()
                .unwrap_or(0);
            (
                state.peak_dense,
                state.peak_candidate,
                state.max_read_transient,
                state.reads,
                minimum,
            )
        } else {
            let reused = state
                .recomputed_projections
                .checked_div(layers)
                .unwrap_or(0);
            (
                0,
                state.recomputed_candidate,
                state.recomputed_dense,
                reused,
                reused,
            )
        };
    let read_ms = state.attributable_read_ns as f64 / 1_000_000.0;
    let joint_ms = state.joint_read_ns as f64 / 1_000_000.0;
    let fields = EventFields {
        extra: vec![
            ("current_persistent_bytes", json!(current_persistent)),
            ("current_read_transient_bytes", json!(current_transient)),
            ("candidate_persistent_bytes", json!(candidate_persistent)),
            // No packed reader is wired on either lane, so the candidate read materializes the
            // same dense K/V the current read does.
            ("candidate_read_transient_bytes", json!(current_transient)),
            ("generation_duration_ms", json!(generation_ms)),
            (
                "cache_read_duration_ms",
                json!(if attributable {
                    read_ms.min(generation_ms)
                } else {
                    0.0
                }),
            ),
            (
                "joint_attention_context_duration_ms",
                json!(if attributable {
                    0.0
                } else {
                    (joint_ms + read_ms).max(0.001).min(generation_ms)
                }),
            ),
            (
                "reference_runtime_attribution_available",
                json!(attributable),
            ),
            ("reused_requests", json!(reused)),
            ("minimum_cache_reads", json!(minimum)),
            (
                "real_weights",
                json!(state.context.real_weights == Some(true)),
            ),
            ("full_generation", json!(!state.cancellation_armed)),
            ("attention_kind", json!("cross")),
        ],
        ..EventFields::default()
    };
    emit(state, "metrics", "metrics", fields);
}

/// Bind live attention geometry. Zero/empty fields leave the current value unchanged so a route can
/// bind what it knows at each hook. Metadata is emitted once geometry is complete and a real
/// projection was confirmed.
pub fn bind_geometry(geometry: KvGeometry, dtype: Option<Dtype>, mask: &str, rope: &str) {
    with_state(|state| {
        let bound = &mut state.context.geometry;
        if geometry.layers != 0 {
            bound.layers = geometry.layers;
        }
        if geometry.heads != 0 {
            bound.heads = geometry.heads;
        }
        if geometry.head_dimension != 0 {
            bound.head_dimension = geometry.head_dimension;
        }
        if geometry.sq != 0 {
            bound.sq = geometry.sq;
        }
        if geometry.skv != 0 {
            bound.skv = geometry.skv;
        }
        if let Some(dtype) = dtype {
            state.context.dtype = dtype_name(dtype).into();
        }
        if !mask.is_empty() {
            state.context.mask = mask.into();
        }
        if !rope.is_empty() {
            state.context.rope = rope.into();
        }
        try_emit_metadata(state);
    });
}

/// Confirm that a product projection ran on the loaded snapshot's weights. Unit tests of this
/// crate can never produce a promotable (`real_weights: true`) transcript.
pub fn confirm_real_weights() {
    with_state(|state| {
        state.context.real_weights = Some(!cfg!(test));
        try_emit_metadata(state);
    });
}

fn try_emit_metadata(state: &mut State) {
    if state.metadata_emitted
        || state.context.real_weights.is_none()
        || !state.context.geometry_ready()
    {
        return;
    }
    state.metadata_emitted = true;
    emit(state, "metadata", "metadata", EventFields::default());
    if !state.start_emitted {
        state.start_emitted = true;
        emit(
            state,
            "generation-start",
            "generation-start",
            EventFields::default(),
        );
    }
}

/// `F16`/`BF16`/`F32` as the reducer's dtype table names them.
pub fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Bfloat16 => "BF16",
        Dtype::Float16 => "F16",
        Dtype::Float32 => "F32",
        _ => "UNSUPPORTED",
    }
}

/// Exact SC-20675 v2 packed group-affine bytes for one `[B,H,S,D]` K/V pair (keys grouped on the
/// token axis with a dense f32 pending tail; values grouped on the channel axis; f16 scale+zero per
/// completed group). Identical to the Candle observers and the reducer.
pub fn packed_group_affine_kv_bytes(
    batch: u64,
    heads: u64,
    tokens: u64,
    width: u64,
) -> Option<u64> {
    if batch == 0 || heads == 0 || tokens == 0 || width == 0 {
        return None;
    }
    let group = GROUP_SIZE;
    let rows = batch.checked_mul(heads)?;
    let complete = tokens / group;
    let pending = tokens % group;
    let key_codes = rows
        .checked_mul(complete)?
        .checked_mul(group.checked_mul(width)?.checked_add(3)? / 4)?;
    let key_metadata = rows
        .checked_mul(complete)?
        .checked_mul(width)?
        .checked_mul(4)?;
    let key_pending = rows
        .checked_mul(pending)?
        .checked_mul(width)?
        .checked_mul(4)?;
    let value_rows = rows.checked_mul(tokens)?;
    let value_codes = value_rows.checked_mul(width.checked_add(3)? / 4)?;
    let value_metadata = value_rows
        .checked_mul(width.checked_add(group - 1)? / group)?
        .checked_mul(4)?;
    key_codes
        .checked_add(key_metadata)?
        .checked_add(key_pending)?
        .checked_add(value_codes)?
        .checked_add(value_metadata)
}

fn shape_of(array: &Array) -> Option<[u64; 4]> {
    let shape = array.shape();
    if shape.len() != 4 || shape.iter().any(|&dim| dim <= 0) {
        return None;
    }
    Some([
        shape[0] as u64,
        shape[1] as u64,
        shape[2] as u64,
        shape[3] as u64,
    ])
}

fn campaign_error(message: impl Into<String>) -> Error {
    Error::Msg(format!("SC-20686: {}", message.into()))
}

/// Register one product-owned **persistent** K/V cache (e.g. one Wan block's cross-K/V or one FLUX.2
/// kv-edit reference slot). Bytes are the exact `nbytes` of the retained arrays. Re-registering a
/// live key is a **rebuild**: the previous cache is released first (with `rebuild_operation`).
pub fn register_cache(
    key: CacheKey,
    key_array: &Array,
    value_array: &Array,
    operation: &'static str,
    rebuild_operation: &'static str,
) -> Result<u64> {
    let Some(result) = with_state(|state| -> Result<u64> {
        let shape = shape_of(key_array)
            .filter(|shape| Some(*shape) == shape_of(value_array))
            .ok_or_else(|| campaign_error("cache K/V must share one [B,H,S,D] shape"))?;
        if key_array.dtype() != value_array.dtype() {
            return Err(campaign_error("cache K/V dtypes differ"));
        }
        let dense = (key_array.nbytes() + value_array.nbytes()) as u64;
        let candidate = packed_group_affine_kv_bytes(shape[0], shape[1], shape[2], shape[3])
            .ok_or_else(|| campaign_error("packed projection overflow"))?;
        if let Some(previous) = state.keys.remove(&key) {
            release_id(state, previous, rebuild_operation);
        }
        if state.context.dtype.is_empty() {
            state.context.dtype = dtype_name(key_array.dtype()).into();
        }
        state.context.real_weights = Some(!cfg!(test));
        try_emit_metadata(state);
        let id = state.next_cache_id;
        state.next_cache_id += 1;
        state.keys.insert(key, id);
        state.caches.insert(
            id,
            CacheAccounting {
                dense_bytes: dense,
                candidate_bytes: candidate,
                reads: 0,
            },
        );
        state.live_dense += dense;
        state.peak_dense = state.peak_dense.max(state.live_dense);
        state.live_candidate += candidate;
        state.peak_candidate = state.peak_candidate.max(state.live_candidate);
        fold_peak(state);
        let fields = EventFields {
            persistent_bytes: dense,
            cache_id: id,
            candidate_persistent_bytes: candidate,
            tensor_shape: format!("k={:?};v={:?}", key_array.shape(), value_array.shape()),
            extra: vec![("kv_batch", json!(shape[0]))],
            ..EventFields::default()
        };
        emit(state, "cross-kv-created", operation, fields);
        Ok(id)
    }) else {
        return Ok(0);
    };
    result
}

/// The live cache id for `key`, or `None` outside a campaign / for an unregistered key.
pub fn cache_id(key: CacheKey) -> Option<u64> {
    STATE.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|state| state.keys.get(&key).copied())
    })
}

fn release_id(state: &mut State, id: u64, operation: &str) {
    let Some(cache) = state.caches.remove(&id) else {
        return;
    };
    state.completed_reads.push(cache.reads);
    state.live_dense = state.live_dense.saturating_sub(cache.dense_bytes);
    state.live_candidate = state.live_candidate.saturating_sub(cache.candidate_bytes);
    let mut fields = remnant(state);
    fields.persistent_bytes = cache.dense_bytes;
    fields.candidate_persistent_bytes = cache.candidate_bytes;
    fields.cache_id = id;
    emit(state, "cross-kv-released", operation, fields);
}

/// Release every cache owned by `owner` in `streams` (the product dropped its cache container).
pub fn release_owner(owner: usize, streams: &[u8], operation: &'static str) {
    with_state(|state| {
        let keys: Vec<CacheKey> = state
            .keys
            .keys()
            .filter(|key| key.owner == owner && streams.contains(&key.stream))
            .copied()
            .collect();
        for key in keys {
            if let Some(id) = state.keys.remove(&key) {
                release_id(state, id, operation);
            }
        }
    });
}

/// RAII release of a registered cache set (Wan's per-generate `StepCache`). Declare it after the
/// arrays it covers so the tensors are freed before the post-release remnant is sampled.
pub struct CacheSetGuard {
    owner: usize,
    operation: &'static str,
}

impl Drop for CacheSetGuard {
    fn drop(&mut self) {
        release_owner(self.owner, &[0], self.operation);
    }
}

/// Register a Wan-style per-block `(k, v)` cross-attention cache set. Keys are the stable addresses
/// of the tuples inside `set`, which the product then passes by reference to every read. Returns
/// `None` outside a campaign (zero work).
pub fn register_cross_kv_set(
    set: &[(Array, Array)],
    query_tokens: u64,
    operation: &'static str,
    release_operation: &'static str,
) -> Result<Option<CacheSetGuard>> {
    if !active() {
        return Ok(None);
    }
    let Some((first_k, _)) = set.first() else {
        return Err(campaign_error("empty cross-K/V set"));
    };
    let shape = shape_of(first_k).ok_or_else(|| campaign_error("cross-K/V is not [B,H,S,D]"))?;
    bind_geometry(
        KvGeometry {
            layers: set.len() as u32,
            heads: shape[1] as u32,
            head_dimension: shape[3] as u32,
            sq: query_tokens,
            skv: shape[2],
        },
        Some(first_k.dtype()),
        "none",
        "none",
    );
    confirm_real_weights();
    let owner = set.as_ptr() as usize;
    for (slot, (k, v)) in set.iter().enumerate() {
        register_cache(
            CacheKey {
                owner,
                stream: 0,
                slot,
            },
            k,
            v,
            operation,
            release_operation,
        )?;
    }
    Ok(Some(CacheSetGuard {
        owner,
        operation: release_operation,
    }))
}

/// The cache key of one `(k, v)` tuple of a registered set, if it is registered.
pub fn cross_kv_cache_id(kv: &(Array, Array)) -> Option<u64> {
    if !active() {
        return None;
    }
    STATE.with(|slot| {
        let state = slot.borrow();
        let state = state.as_ref()?;
        let address = kv as *const (Array, Array) as usize;
        let size = std::mem::size_of::<(Array, Array)>();
        state.keys.iter().find_map(|(key, id)| {
            (key.stream == 0 && key.owner + key.slot * size == address).then_some(*id)
        })
    })
}

/// Record a **recomputed** K/V projection (no persistent cache): the FLUX.2 edit reference slice or
/// MLX Wan-VACE's per-call text K/V. `tokens` is the attributed token count (the trailing reference
/// slice for FLUX, the whole context for VACE). Returns the dense bytes of that slice.
pub fn record_recomputed_kv(
    key_array: &Array,
    value_array: &Array,
    tokens: u64,
    operation: &'static str,
) -> Result<Option<u64>> {
    let Some(result) = with_state(|state| -> Result<u64> {
        let shape = shape_of(key_array)
            .filter(|shape| Some(*shape) == shape_of(value_array))
            .ok_or_else(|| campaign_error("recomputed K/V must share one [B,H,S,D] shape"))?;
        if tokens == 0 || tokens > shape[2] || key_array.dtype() != value_array.dtype() {
            return Err(campaign_error(
                "recomputed K/V slice exceeds its projection",
            ));
        }
        let itemsize = key_array.item_size() as u64;
        let dense = 2 * shape[0] * shape[1] * tokens * shape[3] * itemsize;
        let layers = u64::from(state.context.geometry.layers);
        let candidate = packed_group_affine_kv_bytes(shape[0], shape[1], tokens, shape[3])
            .and_then(|bytes| bytes.checked_mul(layers))
            .ok_or_else(|| campaign_error("packed projection overflow or unbound layers"))?;
        if state.context.dtype.is_empty() {
            state.context.dtype = dtype_name(key_array.dtype()).into();
        }
        state.context.real_weights = Some(!cfg!(test));
        try_emit_metadata(state);
        state.recomputed_projections += 1;
        state.recomputed_dense = state.recomputed_dense.max(dense);
        // A route may recompute slices of different lengths (Wan-VACE projects each CFG branch over
        // its own unpadded context): the candidate is the largest live slice's packed layer stack,
        // like the dense read workspace is the largest live slice.
        state.recomputed_candidate = state.recomputed_candidate.max(candidate);
        let fields = EventFields {
            transient_bytes: dense,
            tensor_shape: format!(
                "k=[{},{},{},{}];v=[{},{},{},{}]",
                shape[0], shape[1], tokens, shape[3], shape[0], shape[1], tokens, shape[3]
            ),
            // The projection's own live dtype: a route may mix them (the Wan2.1-VACE-1.3B checkpoint
            // ships its main blocks F32 and its VACE blocks BF16), so the event-level `dtype`
            // (the bound context's) cannot account for every slice's bytes.
            extra: vec![
                ("kv_batch", json!(shape[0])),
                ("kv_dtype", json!(dtype_name(key_array.dtype()))),
            ],
            ..EventFields::default()
        };
        emit(state, "cross-kv-created", operation, fields);
        Ok(dense)
    }) else {
        return Ok(None);
    };
    result.map(Some)
}

/// What a read window reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadTarget {
    /// A registered persistent cache.
    Cache(u64),
    /// A recomputed projection of the given dense bytes.
    Recomputed(u64),
}

/// An open read window. Created only in campaign mode.
pub struct ReadWindow {
    target: ReadTarget,
    before: u64,
    started: Instant,
}

/// Open a read window: evaluate the inputs (so their production is outside the window), fold and
/// reset the allocator peak, and start timing. `Ok(None)` outside a campaign — no evaluation.
pub fn begin_read(target: Option<ReadTarget>, inputs: &[&Array]) -> Result<Option<ReadWindow>> {
    let Some(target) = target else {
        return Ok(None);
    };
    let reads_observed = STATE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|state| !state.schedule_control)
    });
    if !reads_observed {
        return Ok(None);
    }
    mlx_rs::transforms::eval(inputs.iter().copied())?;
    let before = with_state(|state| {
        fold_peak(state);
        state.counters.reset_peak();
        state.counters.active()
    });
    Ok(before.map(|before| ReadWindow {
        target,
        before,
        started: Instant::now(),
    }))
}

/// Close a read window: evaluate the attention output, record the active high-water above the
/// window's start as the physical read transient, and (for the cancellation arm) cancel the
/// product after its first live read. `attributable` is false when the attention also covers
/// non-cache tokens (FLUX joint attention), so its duration is context, not cache-read runtime.
pub fn finish_read(
    window: Option<ReadWindow>,
    output: &Array,
    operation: &'static str,
    attributable: bool,
) -> Result<()> {
    let Some(window) = window else {
        return Ok(());
    };
    mlx_rs::transforms::eval([output])?;
    let elapsed = window.started.elapsed().as_nanos();
    with_state(|state| {
        let after = state.counters.active();
        let high = state.counters.peak().max(window.before).max(after);
        let reserved = after + state.counters.cache();
        if let Some(phase) = state.phase.as_mut() {
            phase.high = phase.high.max(high);
        }
        state.counters.reset_peak();
        let allocator = AllocatorWindow {
            before: window.before,
            after,
            high,
            reserved,
        };
        let (cache_id, transient) = match window.target {
            ReadTarget::Cache(id) => {
                if let Some(cache) = state.caches.get_mut(&id) {
                    cache.reads += 1;
                }
                (id, high - window.before)
            }
            ReadTarget::Recomputed(dense) => (0, dense),
        };
        if matches!(window.target, ReadTarget::Cache(_)) {
            state.max_read_transient = state.max_read_transient.max(transient);
        }
        state.reads += 1;
        if attributable {
            state.attributable_read_ns += elapsed;
        } else {
            state.non_attributable_reads += 1;
            state.joint_read_ns += elapsed;
        }
        state.live_read_seen = true;
        let fields = EventFields {
            transient_bytes: transient,
            reused: 1,
            cache_id,
            allocator: Some(allocator),
            ..EventFields::default()
        };
        emit(state, "cross-kv-read", operation, fields);
        if state.cancellation_armed && !state.cancel_triggered {
            state.cancel_triggered = true;
            state.cancel.cancel();
        }
    });
    Ok(())
}

/// Scope the FLUX.2 edit forwards that carry the trailing reference tokens. Returns `None` (no
/// state) outside a campaign.
pub struct ReferenceForward(bool);

pub fn reference_forward(includes_reference: bool) -> Option<ReferenceForward> {
    with_state(|state| {
        let previous = state.reference_forward;
        state.reference_forward = includes_reference;
        ReferenceForward(previous)
    })
}

impl Drop for ReferenceForward {
    fn drop(&mut self) {
        let previous = self.0;
        with_state(|state| state.reference_forward = previous);
    }
}

/// Whether the current forward carries the trailing reference tokens (campaign mode only).
pub fn in_reference_forward() -> bool {
    STATE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|state| state.reference_forward)
    })
}

/// Bound `skv` (the attributed reference/context tokens), or 0 outside a campaign.
pub fn bound_skv() -> u64 {
    STATE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(0, |state| state.context.geometry.skv)
    })
}

/// Test support for provider crates' hook tests: the production Metal allocator counters, a fixed
/// footprint probe and an in-memory capturing sink. Activation still requires an armed request.
#[doc(hidden)]
pub fn capture_instruments() -> (Instruments, std::rc::Rc<RefCell<Vec<Value>>>) {
    struct Capture(std::rc::Rc<RefCell<Vec<Value>>>);
    impl EventSink for Capture {
        fn emit(&mut self, event: Value) {
            self.0.borrow_mut().push(event);
        }
    }
    struct Fixed;
    impl FootprintProbe for Fixed {
        fn sample(&self) -> Option<(u64, u64)> {
            Some((1, 1))
        }
    }
    let events = std::rc::Rc::new(RefCell::new(Vec::new()));
    (
        Instruments {
            sink: Box::new(Capture(events.clone())),
            counters: Box::new(MlxCounters),
            footprint: Box::new(Fixed),
        },
        events,
    )
}

#[cfg(test)]
mod tests;
