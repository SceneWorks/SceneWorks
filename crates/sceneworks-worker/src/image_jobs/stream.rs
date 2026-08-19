enum GenEvent {
    Step {
        index: usize,
        current: u32,
        total: u32,
    },
    Decoding {
        index: usize,
    },
    Loading {
        index: usize,
        phase: LoadPhase,
    },
    /// One latent-resolution preview frame of image `index`'s developing render (epic 16624,
    /// sc-16904). Forwarded from a [`gen_core::PreviewSink`] by [`preview_sink_for`]; engines
    /// that don't emit previews simply never produce this variant.
    Preview {
        index: usize,
        frame: gen_core::PreviewFrame,
    },
    /// Typed, provider-authored account of the prompt that actually reached image `index`'s
    /// renderer. Only an active request-local sink can produce this event.
    PromptEnhancement {
        index: usize,
        expected_prompt: String,
        report: gen_core::PromptEnhancementReport,
    },
    Image {
        index: usize,
        seed: i64,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        /// Pre-built `faceLikeness` sidecar block (epic 4406, sc-4409) for this image, or `None`
        /// when the producing path did not score it. `consume_gen_events` inserts a `Some` block
        /// verbatim into the per-image `rawAdapterSettings` under
        /// [`face_likeness::FACE_LIKENESS_FACT_KEY`] — the omit-when-absent persistence seam, so a
        /// path that doesn't score (every non-angle-set path) is untouched.
        face_likeness: Option<JsonObject>,
    },
}

type GeneratedImage = (i64, u32, u32, Vec<u8>);

/// A generated image plus its optional pre-built `faceLikeness` sidecar block (sc-4409). Returned by
/// the per-item closure of [`drive_gen_items_scored`] so the identity-likeness post-pass (used by all
/// four angle-set lanes — InstantID, FLUX.2 edit, Qwen-Edit, SenseNova-U1) can attach a per-image
/// score without disturbing the shared [`GeneratedImage`] tuple every other generator returns.
type ScoredGeneratedImage = (i64, u32, u32, Vec<u8>, Option<JsonObject>);

/// Per-image preview sink for a [`GenerationRequest`](gen_core::GenerationRequest) (sc-16904).
///
/// `PreviewSink::emit` runs synchronously on the denoise thread, so the closure must never block:
/// `try_send` drops the frame when the channel is momentarily full. The consumer keeps only the
/// latest frame per job (single-slot, latest-wins), so a dropped intermediate frame is invisible.
/// Contrast with [`send_gen_progress`]'s `blocking_send`, which is correct for `Progress` events
/// (they are load-bearing for cancel polling) but would stall the GPU here.
fn preview_sink_for(
    tx: &tokio::sync::mpsc::Sender<GenEvent>,
    index: usize,
) -> gen_core::PreviewSink {
    let tx = tx.clone();
    gen_core::PreviewSink::new(move |frame| {
        let _ = tx.try_send(GenEvent::Preview { index, frame });
    })
}

#[derive(Clone)]
struct PromptEnhancementEventSink {
    tx: tokio::sync::mpsc::Sender<GenEvent>,
    index: usize,
}

impl PromptEnhancementEventSink {
    fn for_prompt(&self, prompt: &str) -> gen_core::PromptEnhancementSink {
        let tx = self.tx.clone();
        let index = self.index;
        let expected_prompt = prompt.to_owned();
        gen_core::PromptEnhancementSink::new(move |report| {
            // One small load-bearing provenance event per image. Unlike decorative previews this
            // must not be dropped under channel pressure: a missing report fails the enabled
            // request closed.
            let _ = tx.blocking_send(GenEvent::PromptEnhancement {
                index,
                expected_prompt: expected_prompt.clone(),
                report,
            });
        })
    }
}

fn prompt_enhancement_event_sink_for(
    tx: &tokio::sync::mpsc::Sender<GenEvent>,
    index: usize,
) -> PromptEnhancementEventSink {
    PromptEnhancementEventSink {
        tx: tx.clone(),
        index,
    }
}

fn send_gen_progress(tx: &tokio::sync::mpsc::Sender<GenEvent>, index: usize, progress: Progress) {
    let event = match progress {
        Progress::Step { current, total } => GenEvent::Step {
            index,
            current,
            total,
        },
        Progress::Decoding => GenEvent::Decoding { index },
        Progress::Loading(phase) => GenEvent::Loading { index, phase },
    };
    let _ = tx.blocking_send(event);
}

fn send_generated_image(
    tx: &tokio::sync::mpsc::Sender<GenEvent>,
    index: usize,
    image: GeneratedImage,
) -> bool {
    let (seed, width, height, pixels) = image;
    tx.blocking_send(GenEvent::Image {
        index,
        seed,
        width,
        height,
        pixels,
        face_likeness: None,
    })
    .is_ok()
}

/// Like [`send_generated_image`] but carries the optional pre-built `faceLikeness` block (sc-4409).
fn send_scored_generated_image(
    tx: &tokio::sync::mpsc::Sender<GenEvent>,
    index: usize,
    image: ScoredGeneratedImage,
) -> bool {
    let (seed, width, height, pixels, face_likeness) = image;
    tx.blocking_send(GenEvent::Image {
        index,
        seed,
        width,
        height,
        pixels,
        face_likeness,
    })
    .is_ok()
}

fn drive_gen_items<I, Item, F>(
    tx: tokio::sync::mpsc::Sender<GenEvent>,
    items: I,
    mut generate: F,
) -> WorkerResult<()>
where
    I: IntoIterator<Item = Item>,
    F: FnMut(
        usize,
        Item,
        gen_core::PreviewSink,
        &mut dyn FnMut(Progress),
    ) -> WorkerResult<Option<GeneratedImage>>,
{
    for (index, item) in items.into_iter().enumerate() {
        let _cache_release = RequestCacheRelease;
        let mut on_progress = |progress| send_gen_progress(&tx, index, progress);
        let Some(image) = generate(index, item, preview_sink_for(&tx, index), &mut on_progress)?
        else {
            break;
        };
        if !send_generated_image(&tx, index, image) {
            break;
        }
        // Return image N's retained Metal buffer cache to the system before image N+1
        // allocates, so a multi-image batch doesn't stack each image's transient working
        // set on top of the already-resident model weights and cross the unified-memory
        // ceiling — an OS memory-pressure SIGKILL (Jetsam) that the dense SenseNova-U1 8B
        // family hits first (sc-5567). Frees only freed/retained buffers; the cached
        // generator's live weight arrays are untouched.
    }
    Ok(())
}

/// Prompt-reporting sibling of [`drive_gen_items`]. Kept separate so the additive inference
/// contract changes only the FLUX.2-dev-capable generic lanes; every other producer remains source-
/// and behavior-identical.
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
fn drive_gen_items_reported<I, Item, F>(
    tx: tokio::sync::mpsc::Sender<GenEvent>,
    items: I,
    mut generate: F,
) -> WorkerResult<()>
where
    I: IntoIterator<Item = Item>,
    F: FnMut(
        usize,
        Item,
        gen_core::PreviewSink,
        PromptEnhancementEventSink,
        &mut dyn FnMut(Progress),
    ) -> WorkerResult<Option<GeneratedImage>>,
{
    for (index, item) in items.into_iter().enumerate() {
        let _cache_release = RequestCacheRelease;
        let mut on_progress = |progress| send_gen_progress(&tx, index, progress);
        let Some(image) = generate(
            index,
            item,
            preview_sink_for(&tx, index),
            prompt_enhancement_event_sink_for(&tx, index),
            &mut on_progress,
        )?
        else {
            break;
        };
        if !send_generated_image(&tx, index, image) {
            break;
        }
    }
    Ok(())
}

/// Like [`drive_gen_items`] but the per-item closure additionally returns an optional pre-built
/// `faceLikeness` sidecar block (sc-4409), carried through to `consume_gen_events` for per-image
/// persistence. Used by all four angle-set lanes — InstantID, FLUX.2 edit, Qwen-Edit, and
/// SenseNova-U1 — each of which scores every finished view against the per-job cached source identity
/// embedding on its generation thread (the `!Send` face stack lives there). Every non-scoring path
/// keeps using [`drive_gen_items`].
//
// The scored producers are all face-backend paths; off-Mac they compile only with the candle backend
// (the angle-set scorer's backend legs are cfg-gated the same way), so allow this dead when neither
// face backend is present.
#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
fn drive_gen_items_scored<I, Item, F>(
    tx: tokio::sync::mpsc::Sender<GenEvent>,
    items: I,
    mut generate: F,
) -> WorkerResult<()>
where
    I: IntoIterator<Item = Item>,
    F: FnMut(
        usize,
        Item,
        gen_core::PreviewSink,
        &mut dyn FnMut(Progress),
    ) -> WorkerResult<Option<ScoredGeneratedImage>>,
{
    for (index, item) in items.into_iter().enumerate() {
        let _cache_release = RequestCacheRelease;
        let mut on_progress = |progress| send_gen_progress(&tx, index, progress);
        let Some(image) = generate(index, item, preview_sink_for(&tx, index), &mut on_progress)?
        else {
            break;
        };
        if !send_scored_generated_image(&tx, index, image) {
            break;
        }
    }
    Ok(())
}

/// [`drive_gen_items_scored`] plus the request-local prompt-report sink used by the shared MLX lane.
/// Face likeness and prompt provenance remain independent per-image facts. The Candle FLUX.2-dev
/// edit route does not admit the character-image mode, so it has no scored/reporting combination.
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn drive_gen_items_scored_reported<I, Item, F>(
    tx: tokio::sync::mpsc::Sender<GenEvent>,
    items: I,
    mut generate: F,
) -> WorkerResult<()>
where
    I: IntoIterator<Item = Item>,
    F: FnMut(
        usize,
        Item,
        gen_core::PreviewSink,
        PromptEnhancementEventSink,
        &mut dyn FnMut(Progress),
    ) -> WorkerResult<Option<ScoredGeneratedImage>>,
{
    for (index, item) in items.into_iter().enumerate() {
        let _cache_release = RequestCacheRelease;
        let mut on_progress = |progress| send_gen_progress(&tx, index, progress);
        let Some(image) = generate(
            index,
            item,
            preview_sink_for(&tx, index),
            prompt_enhancement_event_sink_for(&tx, index),
            &mut on_progress,
        )?
        else {
            break;
        };
        if !send_scored_generated_image(&tx, index, image) {
            break;
        }
    }
    Ok(())
}

/// Release MLX's freed-buffer cache between batch images so peak memory doesn't carry
/// forward across a `drive_gen_items` loop (sc-5567). `clear_cache()` returns only the
/// retained-for-reuse buffers to the OS — live arrays (the cached model weights) are not
/// touched — so the one-time reallocation cost on the next image is negligible against a
/// tens-of-seconds generation, and far cheaper than an OOM kill. No-op off macOS: the
/// Windows/CUDA candle lane shares this loop but has no `mlx_rs` dependency.
#[cfg(target_os = "macos")]
fn release_gen_cache_between_items() {
    #[cfg(test)]
    if SUPPRESS_TEST_MLX_CACHE_RELEASE.with(std::cell::Cell::get) {
        return;
    }
    mlx_rs::memory::clear_cache();
}

#[cfg(all(test, target_os = "macos"))]
std::thread_local! {
    /// Headless unit-test escape hatch for drivers whose provider work is fully faked. The real
    /// allocator call hard-exits when no Metal device exists, so those tests suppress only this
    /// ancillary release while still exercising the exact production item loop. Hardware tests do
    /// not enter this scope and continue to verify the real `clear_cache` behavior.
    static SUPPRESS_TEST_MLX_CACHE_RELEASE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(all(test, target_os = "macos"))]
fn without_mlx_cache_release_for_headless_test<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SUPPRESS_TEST_MLX_CACHE_RELEASE.with(|suppressed| suppressed.set(self.0));
        }
    }

    let previous = SUPPRESS_TEST_MLX_CACHE_RELEASE.with(|suppressed| suppressed.replace(true));
    let _restore = Restore(previous);
    run()
}

#[cfg(not(target_os = "macos"))]
fn release_gen_cache_between_items() {}

/// Always release allocator-cache buffers at the end of an item, including cancellation and error
/// exits. Provider scopes synchronize/release active graphs and windows first; this guard then
/// removes freed scratch so the next warm request observes an independent cache state.
struct RequestCacheRelease;

impl Drop for RequestCacheRelease {
    fn drop(&mut self) {
        release_gen_cache_between_items();
    }
}

// Shared by the macOS MLX paths and the Windows/CUDA candle InstantID lane (sc-5491): both load a
// `!Send` engine on the blocking thread and stream per-item events back. `G` is the loaded model
// (MLX `Box<dyn Generator>` or candle `InstantId`) — created and consumed inside the one
// `spawn_blocking`, so it never needs to be `Send`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn start_gen_stream<G, L, D>(
    job_id: String,
    engine_id: &'static str,
    adapter_count: usize,
    load: L,
    drive: D,
) -> (
    CancelFlag,
    tokio::sync::mpsc::Receiver<GenEvent>,
    tokio::task::JoinHandle<WorkerResult<()>>,
)
where
    L: FnOnce() -> WorkerResult<G> + Send + 'static,
    D: FnOnce(G, tokio::sync::mpsc::Sender<GenEvent>, CancelFlag) -> WorkerResult<()>
        + Send
        + 'static,
{
    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);
    let blocking_cancel = cancel.clone();
    let blocking = tokio::task::spawn_blocking(move || -> WorkerResult<()> {
        emit_load_event(
            "image_pipeline_load_start",
            &job_id,
            engine_id,
            adapter_count,
        );
        let generator = load()?;
        emit_load_event(
            "image_pipeline_load_complete",
            &job_id,
            engine_id,
            adapter_count,
        );
        drive(generator, tx, blocking_cancel)
    });
    (cancel, rx, blocking)
}

fn start_cached_gen_stream<D>(
    job_id: String,
    engine_id: &'static str,
    adapter_count: usize,
    spec: LoadSpec,
    load_error_context: String,
    drive: D,
) -> (
    CancelFlag,
    tokio::sync::mpsc::Receiver<GenEvent>,
    tokio::task::JoinHandle<WorkerResult<()>>,
)
where
    D: FnOnce(&dyn Generator, tokio::sync::mpsc::Sender<GenEvent>, CancelFlag) -> WorkerResult<()>
        + Send
        + 'static,
{
    start_cached_gen_stream_with_request_state(
        job_id,
        engine_id,
        adapter_count,
        spec,
        load_error_context,
        move |generator,
              _cache_state,
              _loaded_policy,
              warm_policy: crate::execution_planner::WarmPolicyProposal,
              _external_committed_bytes,
              tx,
              cancel| {
            // `start_cached_gen_stream` exposes only the generator, so this route has no
            // request-scoped memory block a policy switch could act through. Decline truthfully.
            warm_policy.decline(
                crate::execution_planner::ServedAsIsReason::RouteHasNoRequestScopedMemory,
            );
            drive(generator, tx, cancel)
        },
    )
}

/// The reclaimable-byte estimate and the gate that consumes resident-entry credit for one cold load.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
struct ColdLoadAdmission<A> {
    incoming_reclaimable_weight_bytes: u64,
    admit: A,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl<A> ColdLoadAdmission<A> {
    fn new(incoming_reclaimable_weight_bytes: u64, admit: A) -> Self {
        Self {
            incoming_reclaimable_weight_bytes,
            admit,
        }
    }
}

/// Cached stream whose pre-load admission belongs to the cache miss rather than the route preamble.
/// The admission closure is invoked by the cache loader only for a cold/different-key load; an exact
/// warm hit goes straight to `drive` without re-gating or evicting itself.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn start_cached_gen_stream_after_cold_admission<A, D>(
    job_id: String,
    engine_id: &'static str,
    adapter_count: usize,
    spec: LoadSpec,
    load_error_context: String,
    cold_admission: ColdLoadAdmission<A>,
    drive: D,
) -> (
    CancelFlag,
    tokio::sync::mpsc::Receiver<GenEvent>,
    tokio::task::JoinHandle<WorkerResult<()>>,
)
where
    A: FnOnce(u64) -> WorkerResult<()> + Send + 'static,
    D: FnOnce(&dyn Generator, tokio::sync::mpsc::Sender<GenEvent>, CancelFlag) -> WorkerResult<()>
        + Send
        + 'static,
{
    let ColdLoadAdmission {
        incoming_reclaimable_weight_bytes,
        admit: cold_admission,
    } = cold_admission;
    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);
    let blocking_cancel = cancel.clone();
    let blocking = tokio::spawn(async move {
        emit_load_event(
            "image_pipeline_load_start",
            &job_id,
            engine_id,
            adapter_count,
        );
        crate::generator_cache::with_cached_generator_for_request_after_cold_admission(
            engine_id,
            spec,
            load_error_context,
            incoming_reclaimable_weight_bytes,
            cold_admission,
            move |generator,
                  _cache_state,
                  _loaded_policy,
                  warm_policy: crate::execution_planner::WarmPolicyProposal,
                  _external_committed_bytes,
                  _provider_resident_bytes| {
                warm_policy.decline(
                    crate::execution_planner::ServedAsIsReason::RouteHasNoRequestScopedMemory,
                );
                emit_load_event(
                    "image_pipeline_load_complete",
                    &job_id,
                    engine_id,
                    adapter_count,
                );
                drive(generator, tx, blocking_cancel)
            },
        )
        .await
    });
    (cancel, rx, blocking)
}

/// Cached stream seam that exposes cold/warm state, the LOADED execution policy, and this request's
/// vetted [`crate::execution_planner::WarmPolicyProposal`] to the callback. Geometry and request
/// strategy remain absent from the generator load identity.
///
/// The two policy slots have different jobs and must not be swapped: `loaded_policy` is what the
/// resident weights actually are, so it is the ADMISSION input; the proposal carries this request's
/// own policy intent after vetting, and the callback owes it exactly one settlement — thread it into
/// `mlx_fit_gate::evaluate_request` (which floors the memory ladder with it and reports what the
/// selection did) or `decline` it. It is `#[must_use]`, so forgetting is a compiler warning.
fn start_cached_gen_stream_with_request_state<D>(
    job_id: String,
    engine_id: &'static str,
    adapter_count: usize,
    spec: LoadSpec,
    load_error_context: String,
    drive: D,
) -> (
    CancelFlag,
    tokio::sync::mpsc::Receiver<GenEvent>,
    tokio::task::JoinHandle<WorkerResult<()>>,
)
where
    D: FnOnce(
            &dyn Generator,
            gen_core::MemoryCacheState,
            crate::generator_cache::ExecutionPolicy,
            crate::execution_planner::WarmPolicyProposal,
            u64,
            tokio::sync::mpsc::Sender<GenEvent>,
            CancelFlag,
        ) -> WorkerResult<()>
        + Send
        + 'static,
{
    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);
    let blocking_cancel = cancel.clone();
    let blocking = tokio::spawn(async move {
        emit_load_event(
            "image_pipeline_load_start",
            &job_id,
            engine_id,
            adapter_count,
        );
        crate::generator_cache::with_cached_generator_for_request(
            engine_id,
            spec,
            load_error_context,
            move |generator,
                  cache_state,
                  loaded_policy,
                  warm_policy,
                  external_committed_bytes,
                  _provider_resident_bytes| {
                emit_load_event(
                    "image_pipeline_load_complete",
                    &job_id,
                    engine_id,
                    adapter_count,
                );
                drive(
                    generator,
                    cache_state,
                    loaded_policy,
                    warm_policy,
                    external_committed_bytes,
                    tx,
                    blocking_cancel,
                )
            },
        )
        .await
    });
    (cancel, rx, blocking)
}

/// True when this job can run real in-process inference: the model is a linked,
/// engine-backed family and its weights resolve locally.
/// Fail-loud gate for the stub fallback (sc-4176): Some(message) when the
/// requested model id is a known MLX engine model but its weights snapshot
/// can't be resolved (partially deleted HF cache, stale refs, missing
/// modelPath). None when the model isn't engine-backed (the stub is its
/// intended path) or the weights resolve. MLX-only (uses `mlx_model` + the macOS
/// `resolve_weights_dir`); the candle lane's narrower twin is `candle_weights_gap`
/// (base.rs, sc-20529), which covers the unconverted convert-at-install class only —
/// candle has no general tier-completeness probe to mirror the arm above.
#[cfg(target_os = "macos")]
pub(crate) fn mlx_weights_gap(request: &ImageRequest, settings: &Settings) -> Option<String> {
    let model = mlx_model(&request.model)?;
    match resolve_weights_dir(request, settings) {
        Ok(Some(dir)) => {
            // The resolvers fall back to a COMPLETE sibling tier whenever one exists (sc-12279
            // generalized), so reaching here with an incomplete `dir` means NO complete tier is
            // installed for this model — the load would die mid-generation on the first missing file.
            // Turn that into an actionable pre-flight message naming the tier, directing the user to
            // Model Manager or to pick an installed tier, instead of a raw "No such file or directory".
            if !resolved_tier_is_complete(request, &dir) {
                let tier = dir.file_name().and_then(|s| s.to_str()).unwrap_or("selected");
                return Some(format!(
                    "{}: the '{tier}' quant tier isn't fully installed on this machine (some weight \
                     files are missing). Re-download or repair it in Model Manager, or pick a tier \
                     you have already installed from the studio's Quant tier menu, then retry.",
                    request.model,
                ));
            }
            return None;
        }
        Err(error) => return Some(error.to_string()),
        Ok(None) => {}
    }
    Some(format!(
        "{}: MLX weights not found or incomplete (Hugging Face repo {}). \
         Re-download the model in Model Manager, then retry.",
        request.model,
        model_repo(request, &model),
    ))
}
