//! SC-20686 Metal-lane observer tests on tiny synthetic arrays with scripted allocator counters.
//! Scripted counters keep attribution exact while other tests share the process-global MLX
//! allocator; the arrays themselves are real MLX arrays so byte accounting uses live `nbytes`.

use super::*;
use std::rc::Rc;

#[derive(Default)]
struct Script {
    active: Cell<u64>,
    peak: Cell<u64>,
    cache: Cell<u64>,
    resets: Cell<u32>,
}

impl Script {
    fn alloc(&self, bytes: u64) {
        self.active.set(self.active.get() + bytes);
        self.peak.set(self.peak.get().max(self.active.get()));
    }
    fn free(&self, bytes: u64) {
        self.active.set(self.active.get() - bytes);
        self.cache.set(self.cache.get() + bytes);
    }
}

struct Scripted(Rc<Script>);

impl AllocatorCounters for Scripted {
    fn active(&self) -> u64 {
        self.0.active.get()
    }
    fn peak(&self) -> u64 {
        self.0.peak.get()
    }
    fn cache(&self) -> u64 {
        self.0.cache.get()
    }
    fn reset_peak(&self) {
        // MLX resets the high-water to zero; the observer must still fold `active`.
        self.0.peak.set(0);
        self.0.resets.set(self.0.resets.get() + 1);
    }
}

struct FixedFootprint;

impl FootprintProbe for FixedFootprint {
    fn sample(&self) -> Option<(u64, u64)> {
        Some((4096, 8192))
    }
}

struct Capture(Rc<RefCell<Vec<Value>>>);

impl EventSink for Capture {
    fn emit(&mut self, event: Value) {
        self.0.borrow_mut().push(event);
    }
}

struct Harness {
    root: tempfile::TempDir,
    snapshot: PathBuf,
    events: Rc<RefCell<Vec<Value>>>,
    script: Rc<Script>,
    cancel: CancelFlag,
}

const SOURCE_REF: &str = "fedcba9876543210fedcba9876543210fedcba98";
const MODEL_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn harness() -> Harness {
    let root = tempfile::tempdir().unwrap();
    let snapshot = root.path().join(MODEL_REVISION).join("q4");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("config.json"), b"{\"layers\":2}").unwrap();
    Harness {
        root,
        snapshot,
        events: Rc::new(RefCell::new(Vec::new())),
        script: Rc::new(Script::default()),
        cancel: CancelFlag::default(),
    }
}

fn facts() -> RequestFacts {
    RequestFacts {
        batch: 1,
        frames: 17,
        width: 64,
        height: 64,
        prompt_sha256: "a".repeat(64),
        guidance: "5".into(),
        reference_count: 0,
    }
}

impl Harness {
    fn instruments(&self) -> Instruments {
        Instruments {
            sink: Box::new(Capture(self.events.clone())),
            counters: Box::new(Scripted(self.script.clone())),
            footprint: Box::new(FixedFootprint),
        }
    }

    fn activate(&self, cancellation: bool) -> Scope {
        self.activate_arm(if cancellation { "cancel" } else { "normal" })
    }

    fn activate_arm(&self, arm: &str) -> Scope {
        let request = request_output(
            self.root.path().join("unused.jsonl"),
            SOURCE_REF,
            "sequential",
        )
        .unwrap();
        let request = match arm {
            "cancel" => request.arm_cancellation(),
            "control" => request.arm_schedule_control(),
            _ => request.arm(),
        };
        let scope = activate_with_instruments(
            &self.snapshot,
            &self.cancel,
            "wan2_2_t2v_14b",
            facts(),
            self.instruments(),
        )
        .unwrap()
        .expect("armed request activates");
        drop(request);
        scope
    }

    fn phases(&self) -> Vec<String> {
        self.events
            .borrow()
            .iter()
            .map(|event| event["phase"].as_str().unwrap().to_owned())
            .collect()
    }

    fn of(&self, phase: &str) -> Vec<Value> {
        self.events
            .borrow()
            .iter()
            .filter(|event| event["phase"] == phase)
            .cloned()
            .collect()
    }
}

fn kv(batch: i32, tokens: i32) -> (Array, Array) {
    let shape = [batch, 2, tokens, 8];
    let count = (batch * 2 * tokens * 8) as usize;
    (
        Array::from_slice(&vec![1.0f32; count], &shape),
        Array::from_slice(&vec![2.0f32; count], &shape),
    )
}

#[test]
fn observer_off_is_inert_and_never_builds_facts() {
    assert!(!active());
    let facts_built = Cell::new(0);
    let mut progress = |_: Progress| {};
    let result = observe_generation(
        Path::new("/nonexistent"),
        &CancelFlag::default(),
        "wan2_2_t2v_14b",
        || {
            facts_built.set(facts_built.get() + 1);
            facts()
        },
        &mut progress,
        |_| {
            let set = vec![kv(1, 4)];
            assert!(!active());
            assert!(register_cross_kv_set(&set, 16, "create", "release")
                .unwrap()
                .is_none());
            assert_eq!(cross_kv_cache_id(&set[0]), None);
            assert!(record_recomputed_kv(&set[0].0, &set[0].1, 4, "recompute")
                .unwrap()
                .is_none());
            assert!(begin_read(Some(ReadTarget::Recomputed(1)), &[&set[0].0])
                .unwrap()
                .is_none());
            assert!(reference_forward(true).is_none());
            assert_eq!(bound_skv(), 0);
            Ok(7)
        },
    );
    assert_eq!(result.unwrap(), 7);
    assert_eq!(
        facts_built.get(),
        0,
        "unarmed generation must not derive campaign facts"
    );
}

#[test]
fn persistent_cache_bytes_and_read_transient_are_separate_domains() {
    let h = harness();
    let scope = h.activate(false);
    h.script.alloc(10_000); // weights
    let set = vec![kv(2, 5), kv(2, 5)];
    let guard = register_cross_kv_set(&set, 48, "create", "release")
        .unwrap()
        .expect("campaign registers the set");
    let dense_per_cache = (set[0].0.nbytes() + set[0].1.nbytes()) as u64;
    let created = h.of("cross-kv-created");
    assert_eq!(created.len(), 2);
    for event in &created {
        assert_eq!(event["persistent_bytes"], dense_per_cache);
        assert_eq!(event["transient_bytes"], 0);
        assert_eq!(event["kv_batch"], 2);
        assert_eq!(
            event["candidate_persistent_bytes"],
            packed_group_affine_kv_bytes(2, 2, 5, 8).unwrap()
        );
    }

    let id = cross_kv_cache_id(&set[1]).expect("tuple address resolves to its cache");
    assert_eq!(id, created[1]["cache_id"].as_u64().unwrap());
    let window = begin_read(Some(ReadTarget::Cache(id)), &[&set[1].0]).unwrap();
    let before = h.script.active.get();
    h.script.alloc(700); // attention workspace + output
    h.script.free(600);
    finish_read(window, &set[1].0, "read", true).unwrap();
    let read = h.of("cross-kv-read").pop().unwrap();
    assert_eq!(read["allocator_before_bytes"], before);
    assert_eq!(read["allocator_high_bytes"], before + 700);
    assert_eq!(read["allocator_after_bytes"], before + 100);
    assert_eq!(read["transient_bytes"], 700);
    assert_eq!(read["cache_id"], id);
    assert!(
        read["peak_bytes"].as_u64().unwrap() >= read["allocator_reserved_bytes"].as_u64().unwrap()
    );

    observe_generation_end();
    let metrics = h.of("metrics").pop().unwrap();
    // Persistent = simultaneous residency of both retained caches; the read transient is its own
    // physical high-water and is never added to (or taken from) the persistent figure.
    assert_eq!(metrics["current_persistent_bytes"], 2 * dense_per_cache);
    assert_eq!(metrics["current_read_transient_bytes"], 700);
    assert_eq!(metrics["candidate_read_transient_bytes"], 700);
    assert_eq!(metrics["reused_requests"], 1);
    assert_eq!(metrics["minimum_cache_reads"], 0, "cache 1 was never read");
    assert_eq!(metrics["reference_runtime_attribution_available"], true);
    drop(guard);
    drop(scope);
    let released = h.of("cross-kv-released");
    assert_eq!(released.len(), 2);
    assert!(released
        .iter()
        .all(|event| event["persistent_bytes"] == dense_per_cache
            && event["allocator_measurement_available"] == true));
}

#[test]
fn phase_windows_reset_the_peak_and_nest_read_windows() {
    let h = harness();
    let scope = h.activate(false);
    // Encode: a large transient that must NOT leak into the following windows.
    h.script.alloc(5_000);
    h.script.free(5_000);
    mark_phase("prepare-cache");
    h.script.alloc(300);
    mark_denoise();
    // Denoise step 1 with a nested read window whose reset must not lose the step high-water.
    h.script.alloc(2_000);
    h.script.free(2_000);
    let set = vec![kv(1, 3)];
    let guard = register_cross_kv_set(&set, 9, "create", "release")
        .unwrap()
        .unwrap();
    let id = cross_kv_cache_id(&set[0]).unwrap();
    let window = begin_read(Some(ReadTarget::Cache(id)), &[&set[0].0]).unwrap();
    // The read's own high-water exceeds everything else in the step: its reset must fold it into
    // the enclosing step window, not drop it.
    h.script.alloc(3_000);
    h.script.free(3_000);
    finish_read(window, &set[0].0, "read", true).unwrap();
    observe_progress(&Progress::Step {
        current: 1,
        total: 1,
    });
    observe_progress(&Progress::Decoding);
    h.script.alloc(900);
    h.script.free(900);
    drop(guard);
    observe_generation_end();
    drop(scope);

    let windows = h.of("phase-window");
    let names: Vec<_> = windows
        .iter()
        .map(|w| {
            (
                w["window"].as_str().unwrap().to_owned(),
                w["window_index"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        names,
        vec![
            ("encode".into(), 0),
            ("prepare-cache".into(), 0),
            ("denoise-step".into(), 1),
            ("post-denoise".into(), 1),
            ("decode".into(), 1),
        ]
    );
    let high = |index: usize| {
        windows[index]["allocator_high_bytes"].as_u64().unwrap()
            - windows[index]["allocator_before_bytes"].as_u64().unwrap()
    };
    assert_eq!(high(0), 5_000, "encode high-water");
    assert_eq!(
        high(1),
        300,
        "prepare-cache is not charged the encode transient"
    );
    assert_eq!(
        high(2),
        3_000,
        "the nested read reset keeps the step high-water"
    );
    assert_eq!(high(4), 900, "decode high-water");
    assert!(windows.iter().all(|w| w["phys_footprint_bytes"] == 4096
        && w["phys_footprint_peak_bytes"] == 8192
        && w["allocator_measurement_available"] == true));
    // Every window and read resets the process-global high-water it just folded.
    assert!(h.script.resets.get() >= 6);
    let run_peak = h.of("metrics")[0]["peak_bytes"].as_u64().unwrap();
    assert!(run_peak >= 5_000, "the run high-water folds every window");
    // Phase windows close before the terminal event.
    let phases = h.phases();
    let last_window = phases.iter().rposition(|p| p == "phase-window").unwrap();
    let terminal = phases.iter().position(|p| p == "generation-end").unwrap();
    assert!(last_window < terminal);
}

#[test]
fn lifecycle_is_in_product_order_with_backend_and_runtime_identity() {
    let h = harness();
    let mut progress_seen = 0;
    let request = request_output(h.root.path().join("unused.jsonl"), SOURCE_REF, "sequential")
        .unwrap()
        .arm();
    let result = observe_generation_with(
        &h.snapshot,
        &h.cancel,
        "wan2_2_t2v_14b",
        facts,
        |_| Ok(h.instruments()),
        &mut |_| progress_seen += 1,
        |on_progress| {
            let set = vec![kv(2, 4)];
            let _guard = register_cross_kv_set(&set, 32, "create", "release")?;
            for step in 1..=2 {
                let id = cross_kv_cache_id(&set[0]).unwrap();
                let window = begin_read(Some(ReadTarget::Cache(id)), &[&set[0].0])?;
                finish_read(window, &set[0].0, "read", true)?;
                on_progress(Progress::Step {
                    current: step,
                    total: 2,
                });
            }
            on_progress(Progress::Decoding);
            Ok(())
        },
    );
    drop(request);
    result.unwrap();
    assert_eq!(progress_seen, 3, "product progress is still forwarded");
    let phases = h.phases();
    let index = |phase: &str| phases.iter().position(|p| p == phase).unwrap();
    assert!(index("metadata") < index("generation-start"));
    assert!(index("generation-start") < index("cross-kv-created"));
    assert!(index("cross-kv-created") < index("cross-kv-read"));
    assert!(index("cross-kv-released") < index("generation-end"));
    assert!(index("generation-end") < index("metrics"));
    assert!(index("metrics") < index("invalidated"));
    assert!(index("invalidated") < index("released"));
    assert_eq!(phases.iter().filter(|p| *p == "metadata").count(), 1);
    assert!(h
        .events
        .borrow()
        .iter()
        .all(|event| event["backend"] == BACKEND && event["attention"] == "cross"));
    let metadata = h.of("metadata").pop().unwrap();
    assert_eq!(metadata["source_ref"], SOURCE_REF);
    assert_eq!(metadata["model_snapshot_revision"], MODEL_REVISION);
    assert_eq!(metadata["residency_strategy"], "sequential");
    assert_eq!(metadata["variant"], "wan2_2_t2v_14b");
    assert_eq!(metadata["geometry"]["skv"], 4);
    assert_eq!(metadata["geometry"]["sq"], 32);
    assert_eq!(metadata["geometry"]["heads"], 2);
    assert_eq!(metadata["geometry"]["dtype"], "F32");
    assert_eq!(
        metadata["real_weights"], false,
        "unit tests never produce promotable evidence"
    );
    assert_eq!(metadata["cancellation_armed"], false);
    let metrics = h.of("metrics").pop().unwrap();
    assert_eq!(metrics["minimum_cache_reads"], 2);
    assert_eq!(metrics["reused_requests"], 2);
    assert!(!active(), "the scope deactivates the observer");
}

#[test]
fn cancellation_arm_cancels_after_the_first_live_read_only() {
    let h = harness();
    let scope = h.activate(true);
    observe_cancelled();
    assert!(
        h.of("cancelled").is_empty(),
        "no terminal before a live read"
    );
    let set = vec![kv(1, 4)];
    let guard = register_cross_kv_set(&set, 8, "create", "release")
        .unwrap()
        .unwrap();
    assert!(!h.cancel.is_cancelled());
    let id = cross_kv_cache_id(&set[0]).unwrap();
    let window = begin_read(Some(ReadTarget::Cache(id)), &[&set[0].0]).unwrap();
    finish_read(window, &set[0].0, "read", true).unwrap();
    assert!(
        h.cancel.is_cancelled(),
        "the product is cancelled after its first read"
    );
    drop(guard);
    observe_cancelled();
    observe_cancelled();
    drop(scope);
    assert!(campaign_cancelled());
    let phases = h.phases();
    assert_eq!(phases.iter().filter(|p| *p == "cancelled").count(), 1);
    assert!(!phases.iter().any(|p| p == "generation-end"));
    let metadata = h.of("metadata").pop().unwrap();
    assert_eq!(metadata["cancellation_armed"], true);
    assert_eq!(metadata["full_generation"], false);
    assert!(metadata["cancellation_arm_id"]
        .as_str()
        .unwrap()
        .starts_with(BACKEND));
}

#[test]
fn re_extracting_a_live_slot_is_a_rebuild_release_then_create() {
    let h = harness();
    let scope = h.activate(false);
    bind_geometry(
        KvGeometry {
            layers: 1,
            heads: 2,
            head_dimension: 8,
            sq: 16,
            skv: 4,
        },
        None,
        "joint-unmasked",
        "flux2-4-axis",
    );
    let key = CacheKey {
        owner: 0xABC0,
        stream: 1,
        slot: 0,
    };
    let (k, v) = kv(1, 4);
    let first = register_cache(key, &k, &v, "extract", "extract(rebuild)").unwrap();
    let second = register_cache(key, &k, &v, "extract", "extract(rebuild)").unwrap();
    assert_ne!(first, second);
    assert_eq!(cache_id(key), Some(second));
    let window = begin_read(Some(ReadTarget::Cache(second)), &[&k]).unwrap();
    finish_read(window, &k, "cached", false).unwrap();
    release_owner(0xABC0, &[1, 2], "drop");
    assert_eq!(cache_id(key), None);
    observe_generation_end();
    drop(scope);
    let released = h.of("cross-kv-released");
    assert_eq!(released.len(), 2);
    assert_eq!(released[0]["operation"], "extract(rebuild)");
    assert_eq!(released[0]["cache_id"], first);
    assert_eq!(released[1]["operation"], "drop");
    let metrics = h.of("metrics").pop().unwrap();
    // The rebuilt cache was never read: the per-cache minimum is honest about that.
    assert_eq!(metrics["minimum_cache_reads"], 0);
    assert_eq!(
        metrics["current_persistent_bytes"],
        (k.nbytes() + v.nbytes()) as u64
    );
    assert_eq!(metrics["reference_runtime_attribution_available"], false);
    let metadata = h.of("metadata").pop().unwrap();
    assert_eq!(metadata["geometry"]["mask"], "joint-unmasked");
    assert_eq!(metadata["geometry"]["rope"], "flux2-4-axis");
}

#[test]
fn recomputed_kv_reports_zero_persistence_and_per_layer_candidate() {
    let h = harness();
    let scope = h.activate(false);
    bind_geometry(
        KvGeometry {
            layers: 2,
            heads: 2,
            head_dimension: 8,
            sq: 16,
            skv: 3,
        },
        None,
        "",
        "",
    );
    let (k, v) = kv(1, 7); // [txt+target+ref]; the trailing 3 tokens are the reference slice
    let dense = 2 * 2 * 3 * 8 * 4;
    for _forward in 0..3 {
        for _layer in 0..2 {
            let recorded = record_recomputed_kv(&k, &v, bound_skv(), "slice").unwrap();
            assert_eq!(recorded, Some(dense));
            let window = begin_read(recorded.map(ReadTarget::Recomputed), &[&k]).unwrap();
            finish_read(window, &k, "joint", false).unwrap();
        }
    }
    assert!(
        record_recomputed_kv(&k, &v, 8, "slice").is_err(),
        "slice beyond projection"
    );
    observe_generation_end();
    drop(scope);
    let created = h.of("cross-kv-created");
    assert!(created
        .iter()
        .all(|e| e["persistent_bytes"] == 0 && e["transient_bytes"] == dense));
    let reads = h.of("cross-kv-read");
    assert!(reads
        .iter()
        .all(|e| e["transient_bytes"] == dense && e["cache_id"] == 0));
    let metrics = h.of("metrics").pop().unwrap();
    assert_eq!(metrics["current_persistent_bytes"], 0);
    assert_eq!(metrics["current_read_transient_bytes"], dense);
    assert_eq!(
        metrics["candidate_persistent_bytes"],
        packed_group_affine_kv_bytes(1, 2, 3, 8).unwrap() * 2
    );
    assert_eq!(metrics["reused_requests"], 3);
    assert_eq!(metrics["minimum_cache_reads"], 3);
}

/// The SC-20686 W2 `wan_vace` evidence: Wan-VACE recomputes each CFG branch over its own unpadded
/// context (cond 13 tokens, uncond 126) and the 1.3B checkpoint's main blocks project in F32 while
/// its VACE blocks project in BF16. Every projection records its own live dtype, and the candidate
/// is the largest live slice's packed layer stack (as the dense read workspace is the largest slice).
#[test]
fn recomputed_kv_records_each_projections_live_dtype_and_largest_slice() {
    let h = harness();
    let scope = h.activate(false);
    bind_geometry(
        KvGeometry {
            layers: 2,
            heads: 2,
            head_dimension: 8,
            sq: 16,
            skv: 7,
        },
        None,
        "",
        "",
    );
    let (cond_k, cond_v) = kv(1, 3);
    let (uncond_k, uncond_v) = kv(1, 7);
    let bf16 = |a: &Array| a.as_dtype(Dtype::Bfloat16).unwrap();
    // Per branch: one BF16 (VACE-block) and one F32 (main-block) projection.
    // The longer slice is not the last one: the candidate must still be the largest.
    let projections = [
        (bf16(&uncond_k), bf16(&uncond_v), 7, 2),
        (uncond_k.clone(), uncond_v.clone(), 7, 4),
        (bf16(&cond_k), bf16(&cond_v), 3, 2),
        (cond_k.clone(), cond_v.clone(), 3, 4),
    ];
    for (k, v, tokens, itemsize) in &projections {
        let recorded = record_recomputed_kv(k, v, *tokens, "slice").unwrap();
        assert_eq!(recorded, Some(2 * 2 * tokens * 8 * itemsize));
    }
    observe_generation_end();
    drop(scope);
    let created = h.of("cross-kv-created");
    let dtypes: Vec<_> = created.iter().map(|e| e["kv_dtype"].clone()).collect();
    assert_eq!(
        dtypes,
        [json!("BF16"), json!("F32"), json!("BF16"), json!("F32")]
    );
    let metrics = h.of("metrics").pop().unwrap();
    assert_eq!(metrics["current_read_transient_bytes"], 2 * 2 * 7 * 8 * 4);
    assert_eq!(
        metrics["candidate_persistent_bytes"],
        packed_group_affine_kv_bytes(1, 2, 7, 8).unwrap() * 2
    );
}

#[test]
fn reference_forward_scope_restores_and_packed_projection_matches_candle() {
    let h = harness();
    let scope = h.activate(false);
    assert!(!in_reference_forward());
    {
        let _outer = reference_forward(true);
        assert!(in_reference_forward());
        {
            let _inner = reference_forward(false);
            assert!(!in_reference_forward());
        }
        assert!(in_reference_forward());
    }
    assert!(!in_reference_forward());
    drop(scope);
    // Same fixture as the Candle observers: B=1,H=2,Skv=65,D=64 → 6_704 bytes.
    assert_eq!(packed_group_affine_kv_bytes(1, 2, 65, 64), Some(6_704));
    assert_eq!(packed_group_affine_kv_bytes(1, 2, 0, 64), None);
    assert_eq!(packed_group_affine_kv_bytes(u64::MAX, 2, 65, 64), None);
}

#[test]
fn request_and_footprint_inputs_fail_closed() {
    assert!(request_output("-", SOURCE_REF, "sequential").is_err());
    assert!(request_output("e.jsonl", "HEAD", "sequential").is_err());
    assert!(request_output("e.jsonl", SOURCE_REF, "offloaded").is_err());
    assert_eq!(
        footprint_pair(2_228_560, 2_277_712),
        Some((2_228_560, 2_277_712))
    );
    assert_eq!(footprint_pair(10, 10), Some((10, 10)));
    assert_eq!(footprint_pair(0, 10), None);
    assert_eq!(footprint_pair(20, 10), None);
    #[cfg(target_os = "macos")]
    {
        let (current, peak) = DarwinFootprint
            .sample()
            .expect("own proc_pid_rusage footprint");
        assert!(current > 0 && peak >= current);
    }
}

#[test]
fn a_snapshot_without_an_immutable_revision_refuses_activation() {
    let root = tempfile::tempdir().unwrap();
    let snapshot = root.path().join("main").join("q4");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("config.json"), b"{}").unwrap();
    let request = request_output(root.path().join("e.jsonl"), SOURCE_REF, "sequential")
        .unwrap()
        .arm();
    let events = Rc::new(RefCell::new(Vec::new()));
    let result = activate_with_instruments(
        &snapshot,
        &CancelFlag::default(),
        "wan2_2_t2v_14b",
        facts(),
        Instruments {
            sink: Box::new(Capture(events.clone())),
            counters: Box::new(Scripted(Rc::new(Script::default()))),
            footprint: Box::new(FixedFootprint),
        },
    );
    drop(request);
    assert!(result.is_err());
    assert!(!active());
    assert!(events.borrow().is_empty());
}

/// D run 37023146895: FLUX.2 Klein's staged residency announces `Loading(TextEncoder)` before its
/// prompt encode. The text-encoder load and encode belong to the `encode` window, so that window
/// carries the conditioning peak instead of closing empty (a zero-peak allocator sample the adapter
/// refuses); the render bundle's `Loading(Renderer)` opens `load`.
#[test]
fn a_text_encoder_load_is_attributed_to_the_encode_window() {
    let h = harness();
    let scope = h.activate(false);
    observe_progress(&Progress::Loading(crate::LoadPhase::TextEncoder));
    h.script.alloc(4096); // the prompt encoder's weights and encode
    h.script.free(4096);
    observe_progress(&Progress::Loading(crate::LoadPhase::Renderer));
    h.script.alloc(8192); // the transformer + VAE
    mark_denoise();
    observe_progress(&Progress::Step {
        current: 1,
        total: 1,
    });
    observe_progress(&Progress::Decoding);
    observe_generation_end();
    drop(scope);
    let windows = h.of("phase-window");
    let names: Vec<&str> = windows
        .iter()
        .map(|w| w["window"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ["encode", "load", "denoise-step", "post-denoise", "decode"]
    );
    assert_eq!(windows[0]["allocator_high_bytes"], 4096, "{:?}", windows[0]);
    assert_eq!(windows[1]["allocator_high_bytes"], 8192, "{:?}", windows[1]);
    for window in &windows {
        assert!(
            window["peak_bytes"].as_u64().unwrap() > 0,
            "every allocator sample carries a positive peak: {window:?}"
        );
    }
}

#[test]
fn a_mid_denoise_expert_swap_keeps_every_window_unique() {
    let h = harness();
    let scope = h.activate(false);
    mark_phase("prepare-cache");
    mark_denoise();
    observe_progress(&Progress::Step {
        current: 1,
        total: 3,
    });
    // Sequential MoE swap before step 2: the curated path marks `load`, the native path emits
    // `Progress::Loading`; either way the pending step-2 window becomes the swap, then the next
    // expert's cache build and step 2 follow.
    mark_phase("load");
    observe_progress(&Progress::Loading(crate::LoadPhase::Renderer));
    mark_phase("prepare-cache");
    mark_denoise();
    observe_progress(&Progress::Step {
        current: 2,
        total: 3,
    });
    observe_progress(&Progress::Step {
        current: 3,
        total: 3,
    });
    observe_progress(&Progress::Decoding);
    observe_generation_end();
    drop(scope);
    let windows: Vec<(String, u64)> = h
        .of("phase-window")
        .iter()
        .map(|w| {
            (
                w["window"].as_str().unwrap().to_owned(),
                w["window_index"].as_u64().unwrap(),
            )
        })
        .collect();
    let unique: std::collections::BTreeSet<_> = windows.iter().cloned().collect();
    assert_eq!(
        unique.len(),
        windows.len(),
        "duplicate phase windows: {windows:?}"
    );
    let steps: Vec<u64> = windows
        .iter()
        .filter(|(name, _)| name == "denoise-step")
        .map(|(_, index)| *index)
        .collect();
    assert_eq!(steps, vec![1, 2, 3]);
    assert!(windows.contains(&("load".to_owned(), 1)));
}

#[test]
fn the_schedule_control_arm_opens_no_read_windows() {
    let h = harness();
    let scope = h.activate_arm("control");
    let set = vec![kv(1, 4)];
    let guard = register_cross_kv_set(&set, 8, "create", "release")
        .unwrap()
        .expect("caches are still attributed on the control arm");
    let id = cross_kv_cache_id(&set[0]).unwrap();
    let resets = h.script.resets.get();
    assert!(
        begin_read(Some(ReadTarget::Cache(id)), &[&set[0].0])
            .unwrap()
            .is_none(),
        "the control arm must not evaluate or reset around reads"
    );
    assert_eq!(h.script.resets.get(), resets);
    mark_denoise();
    observe_progress(&Progress::Step {
        current: 1,
        total: 1,
    });
    observe_progress(&Progress::Decoding);
    drop(guard);
    observe_generation_end();
    drop(scope);
    assert!(h.of("cross-kv-read").is_empty());
    assert_eq!(h.of("cross-kv-created").len(), 1);
    assert!(!h.of("phase-window").is_empty());
    let metadata = h.of("metadata").pop().unwrap();
    assert_eq!(metadata["schedule_control"], true);
    assert_eq!(metadata["full_generation"], true);
}

/// SC-20686 D2: a route that binds its complete geometry (dtype included) before its first
/// projection — Wan-VACE's `sc20686_bind`, and `register_cross_kv_set`'s own bind — must not emit
/// metadata until the projection is confirmed, or the transcript's single metadata event claims
/// `real_weights: false` for a real-weight run and the adapter refuses the row.
#[test]
fn metadata_waits_for_the_confirmed_projection_after_a_complete_bind() {
    let h = harness();
    let scope = h.activate(false);
    bind_geometry(
        KvGeometry {
            layers: 1,
            heads: 2,
            head_dimension: 8,
            sq: 16,
            skv: 4,
        },
        Some(Dtype::Float32),
        "none",
        "none",
    );
    assert!(
        h.of("metadata").is_empty(),
        "complete geometry alone is not a confirmed product projection"
    );
    let set = vec![kv(1, 4)];
    let guard = register_cross_kv_set(&set, 16, "create", "release")
        .unwrap()
        .unwrap();
    let phases = h.phases();
    let metadata = phases.iter().position(|p| p == "metadata").unwrap();
    let start = phases.iter().position(|p| p == "generation-start").unwrap();
    let create = phases.iter().position(|p| p == "cross-kv-created").unwrap();
    assert!(metadata < start && start < create, "{phases:?}");
    assert_eq!(phases.iter().filter(|p| *p == "metadata").count(), 1);
    drop(guard);
    drop(scope);
}
