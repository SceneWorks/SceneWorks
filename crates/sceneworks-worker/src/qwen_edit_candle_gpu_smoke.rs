//! Local real-weight GPU smoke for the candle **Qwen-Image-Edit** worker lane (sc-13534 — the
//! hardware-gated acceptance for the tier/gate reconciliation). `#[ignore]`d — run by hand on the RTX
//! PRO 6000 (Blackwell sm_120). It drives the ACTUAL worker seam this story fixed —
//! [`crate::image_jobs::resolve_qwen_edit_candle_base`] — then loads the bespoke
//! `runtime_cuda::providers::qwen_image::QwenEdit` provider at the resolved dir and renders one real
//! reference-conditioned edit.
//!
//! ## Why this smoke exists
//!
//! Before sc-13534 the lane resolved the UPSTREAM `Qwen/Qwen-Image-Edit(-2511)` snapshot — a repo no
//! download flow fetches. So on a normal install the lane was dead, and on a dev box that happened to
//! have the dense upstream snapshot cached it loaded DENSE bf16 while the fit-gate had sized q4. This
//! smoke is the "the packed turnkey tier actually resolves + loads + renders" evidence:
//!
//!   1. The worker resolver lands on the `q4/` tier subdir of `SceneWorks/qwen-image-edit-2511-mlx`
//!      (NOT the upstream repo, NOT the turnkey root that holds no `transformer/`).
//!   2. `QwenEdit::load` packed-detects that tier from `transformer/config.json` and renders a coherent,
//!      non-degenerate edit — proving the packed q4 path the sc-11019 `vramGbByTier` rows were measured
//!      against is the path that runs.
//!
//! Build with `CUDA_COMPUTE_CAP=120` (native Blackwell sm_120). The HF cache env points at the real
//! snapshot — no explicit weights path (the worker resolver finds it and, on Windows, hardlink-repairs
//! the `hf`-populated blob symlinks candle can't traverse):
//! ```text
//! $env:HF_HOME="E:\huggingface"                # or HF_HUB_CACHE="E:\huggingface\hub"
//! $env:QWEN_EDIT_OUT_DIR="D:\sceneworks-qwen-edit-validate"
//! # optional: QWEN_EDIT_STEPS=8  QWEN_EDIT_GUIDANCE=4.0  QWEN_EDIT_SEED=42  QWEN_EDIT_W=768  QWEN_EDIT_H=768
//! #           QWEN_EDIT_MODEL=qwen_image_edit_2511  QWEN_EDIT_PROMPT="..."
//! cargo test -p sceneworks-worker --no-default-features --features backend-candle --release \
//!   qwen_edit_worker_lane_gpu_smoke -- --ignored --nocapture
//! ```

use gen_core::{CancelFlag, Image, OffloadPolicy, Progress};
use runtime_cuda::providers::qwen_image::{QwenEdit, QwenEditPaths, QwenEditRequest};

use super::smoke_support::{env_or, image_std, save_png, DEGENERATE_STD_FLOOR_DEFAULT};

/// sc-13534 worker-lane E2E: drive the ACTUAL worker base resolution + `QwenEdit::load` + render for the
/// candle Qwen-Image-Edit lane on real CUDA. Calls the WORKER's
/// [`crate::image_jobs::resolve_qwen_edit_candle_base`] the way `generate_candle_qwen_edit_stream` does,
/// asserts it resolves the `q4/` tier subdir of the `SceneWorks/qwen-image-edit-2511-mlx` turnkey (NOT
/// the upstream snapshot the pre-fix code reached, NOT the tier-less root), then packed-loads q4 and
/// renders a coherent true-CFG edit against a synthetic reference.
#[test]
#[ignore = "real-weight worker-lane GPU smoke; needs the SceneWorks/qwen-image-edit-2511-mlx turnkey in \
            the HF cache (HF_HUB_CACHE/HF_HOME) + a CUDA device (cap=120)"]
fn qwen_edit_worker_lane_gpu_smoke() {
    use crate::image_jobs::resolve_qwen_edit_candle_base;
    use crate::settings::Settings;
    use sceneworks_core::image_request::ImageRequest;
    use serde_json::json;

    let out_dir = std::path::PathBuf::from(env_or("QWEN_EDIT_OUT_DIR", "/tmp/qwen_edit_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let settings = Settings::from_env();

    let model = env_or("QWEN_EDIT_MODEL", "qwen_image_edit_2511");
    let steps: usize = env_or("QWEN_EDIT_STEPS", "8")
        .parse()
        .expect("QWEN_EDIT_STEPS");
    let guidance: f32 = env_or("QWEN_EDIT_GUIDANCE", "4.0")
        .parse()
        .expect("QWEN_EDIT_GUIDANCE");
    let seed: u64 = env_or("QWEN_EDIT_SEED", "42")
        .parse()
        .expect("QWEN_EDIT_SEED");
    let w: u32 = env_or("QWEN_EDIT_W", "768").parse().expect("QWEN_EDIT_W");
    let h: u32 = env_or("QWEN_EDIT_H", "768").parse().expect("QWEN_EDIT_H");
    let prompt = env_or(
        "QWEN_EDIT_PROMPT",
        "turn the plain gray card into a lush watercolor painting of a mountain lake at sunrise",
    );
    let negative = env_or(
        "QWEN_EDIT_NEG",
        "blurry, low quality, jpeg artifacts, deformed",
    );

    // The minimal job payload the router hands `generate_candle_qwen_edit_stream`: an `edit_image` job on
    // a Qwen-Image-Edit id with a source asset. `mlxQuantize: 4` pins the q4 tier deterministically —
    // both the receipt-variant step and the `standard_tier_subdir` descent resolve q4 regardless of
    // whether an app download receipt exists in this cache (an `hf`-populated cache has none). No
    // manifest `repo`, so `qwen_edit_candle_repo` reads the turnkey off the MODEL_TABLE row — the exact
    // production default this story restored.
    let payload = json!({
        "model": model,
        "mode": "edit_image",
        "sourceAssetId": "smoke-src",
        "prompt": prompt,
        "width": w,
        "height": h,
        "advanced": { "mlxQuantize": 4 },
    });
    let request = ImageRequest::from_payload(payload.as_object().unwrap());

    // THE worker resolver (the sc-13534 change): edit lane → the q4 tier subdir of the packed turnkey.
    let dir = resolve_qwen_edit_candle_base(&request, &settings)
        .expect("resolve_qwen_edit_candle_base")
        .unwrap_or_else(|| {
            panic!(
                "{model}: SceneWorks/qwen-image-edit-2511-mlx turnkey not in the HF cache — set \
                 HF_HUB_CACHE/HF_HOME"
            )
        });
    assert!(
        dir.ends_with("q4"),
        "{model}: worker must descend into the q4 tier subdir, got {}",
        dir.display()
    );
    assert!(
        dir.join("transformer").is_dir(),
        "{model}: resolved tier dir must hold transformer/ (packed-detect reads its config.json), got {}",
        dir.display()
    );
    // The exact defect this story fixed: never the upstream snapshot, never the tier-less turnkey root.
    let dir_str = dir.to_string_lossy().replace('\\', "/");
    assert!(
        dir_str.contains("qwen-image-edit-2511-mlx"),
        "{model}: must resolve the SceneWorks turnkey, not an upstream snapshot, got {dir_str}"
    );
    assert!(
        !dir_str.contains("models--Qwen--Qwen-Image-Edit"),
        "{model}: resolved the UPSTREAM repo — the pre-sc-13534 bug, got {dir_str}"
    );
    println!(
        "[worker-smoke] {model}: resolve_qwen_edit_candle_base -> {}",
        dir.display()
    );

    // Packed q4 load, resident residency (the default cross-request-cached path). No adapters — this is
    // the production (non-distilled) true-CFG edit path; QwenEdit packed-detects q4 from the tier's
    // transformer/config.json (group_size 64).
    let model_engine = QwenEdit::load(&QwenEditPaths {
        root: dir.clone(),
        adapters: Vec::new(),
        offload_policy: OffloadPolicy::Resident,
    })
    .unwrap_or_else(|e| panic!("QwenEdit::load({}): {e}", dir.display()));

    // A synthetic reference: a flat mid-gray card. The prompt asks for a strong stylization, so a live
    // edit must diverge from the flat input (a degenerate/black decode stays flat and fails the floor).
    let reference = Image {
        width: w,
        height: h,
        pixels: vec![128u8; (w * h * 3) as usize],
    };

    let cancel = CancelFlag::new();
    let req = QwenEditRequest {
        prompt: prompt.clone(),
        negative: negative.clone(),
        width: w,
        height: h,
        steps,
        guidance,
        seed,
        lightning: false,
        cancel: cancel.clone(),
    };
    println!(
        "[worker-smoke] {model}: rendering {w}x{h} @ {steps} steps, guidance {guidance}, seed {seed} ..."
    );
    let started = std::time::Instant::now();
    let mut steps_seen = 0u32;
    let out = model_engine
        .generate(&req, std::slice::from_ref(&reference), &mut |p| {
            if let Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        })
        .unwrap_or_else(|e| panic!("{model} generate: {e}"));
    let elapsed = started.elapsed();

    assert_eq!((out.width, out.height), (w, h), "output dims");
    assert_eq!(out.pixels.len(), (w * h * 3) as usize, "RGB8 pixel count");
    assert!(steps_seen >= 1, "expected denoise step progress");

    // Degenerate-decode floor: a coherent edit clears the general per-pixel std-dev bar by a wide
    // margin; a NaN / all-black / flat collapse pulls it toward 0.
    let std = image_std(&out);
    assert!(
        std >= DEGENERATE_STD_FLOOR_DEFAULT,
        "{model}: decode looks degenerate (per-pixel std {std:.3} < {DEGENERATE_STD_FLOOR_DEFAULT}) — \
         the packed q4 tier did not render a live image"
    );

    let png = out_dir.join(format!("{model}_q4_{w}x{h}_s{seed}.png"));
    save_png(&out, &png);
    println!(
        "[worker-smoke] {model}: q4 edit OK in {:.1}s (std {std:.2}) -> {}",
        elapsed.as_secs_f32(),
        png.display()
    );
}

/// sc-13960 warm-2nd-run acceptance on real CUDA. The bespoke edit lane loads off the UNcached
/// `start_gen_stream` and never touches the single-slot generator cache, so a co-resident generator's
/// VRAM stays held while the bespoke gate reads RAW free — and a naive gate against that reduced free
/// needlessly downtiers (or rejects) a render whose room the co-resident's imminent eviction frees.
/// This smoke validates the fix end-to-end on the RTX PRO 6000, using LIVE `nvidia-smi` readings — the
/// facts the unit tests cannot see: that a resident generator really occupies VRAM, and that
/// evicting/dropping it really frees that VRAM back for the next load.
///
///   1. Load a real packed-q4 `QwenEdit`, render one edit; while it is RESIDENT, measure how much VRAM
///      it occupies (this is what a bespoke gate's raw free is reduced by when a generator is cached).
///   2. DROP it and confirm `nvidia-smi` free RISES back — evicting the resident frees its VRAM for the
///      incoming render (on this hardware a full drop returns it to the driver; the shared-device
///      sequential-offload path instead pools it — either way it becomes available, the same property
///      the shipped video `with_uncached_generator` lane relies on). This is the fix's whole mechanism.
///   3. Confirm the gate's reclaim credit PREDICTS that freed budget: crediting the resident's peak onto
///      the RESIDENT (reduced) reading — [`vram_gate::with_reclaimable`] over `note_loaded_peak` — lands
///      within a couple GB of the measured post-drop free. That is what lets the gate admit resident
///      BEFORE the evict has happened.
///   4. Assert the gate's plan FLIPS: on a budget whose raw free sits just below the resident peak, the
///      plan is a non-resident downtier (the bug); crediting the peak the evict will free readmits it
///      resident (the fix). Uses the REAL occupied peak, forced constrained so the flip shows on any card.
///   5. Reload + render a WARM second edit into the reclaimed room — proving the reclaimed residency is
///      safe (no OOM), the literal "warm 2nd-run" the acceptance names.
///
/// Run (needs the turnkey in the HF cache + a CUDA device via SCENEWORKS_GPU_ID, `--release` for a
/// realistic peak):
/// ```text
/// $env:SCENEWORKS_GPU_ID="0"; $env:HF_HOME="E:\huggingface"
/// $env:QWEN_EDIT_OUT_DIR="D:\sceneworks-qwen-edit-validate"
/// cargo test -p sceneworks-worker --no-default-features --features backend-candle --release \
///   qwen_edit_warm_reclaim_gpu_smoke -- --ignored --nocapture
/// ```
#[test]
#[ignore = "sc-13960 real-weight warm-reclaim GPU smoke; needs the SceneWorks/qwen-image-edit-2511-mlx \
            turnkey in the HF cache + a CUDA device (cap=120)"]
fn qwen_edit_warm_reclaim_gpu_smoke() {
    use crate::image_jobs::resolve_qwen_edit_candle_base;
    use crate::settings::Settings;
    use crate::vram_gate::{
        load_plan, note_loaded_peak, reclaimable_pool_gb, with_reclaimable, LoadPlan, VramBudget,
    };
    use sceneworks_core::image_request::ImageRequest;
    use serde_json::json;

    let out_dir = std::path::PathBuf::from(env_or("QWEN_EDIT_OUT_DIR", "/tmp/qwen_edit_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let settings = Settings::from_env();
    let gpu_id = settings.gpu_id.clone();

    // A live VRAM budget read (blocking, since this is a sync test); `None` means no NVIDIA GPU visible,
    // which fails the smoke loudly rather than pretending it validated hardware it never touched.
    let budget = |label: &str| -> VramBudget {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        let b = rt
            .block_on(crate::gpu::nvidia_vram_budget_gb(&gpu_id))
            .unwrap_or_else(|| panic!("no live NVIDIA VRAM budget for gpu {gpu_id} at {label}"));
        println!(
            "[reclaim-smoke] {label}: free={:.2} GB / total={:.2} GB",
            b.free_gb, b.total_gb
        );
        b
    };

    // Small + fast: this smoke measures VRAM mechanics, not image fidelity. The packed-q4 resident
    // weights dominate the pool regardless of resolution, so 512² / 4 steps is plenty.
    let (w, h): (u32, u32) = (512, 512);
    let steps: usize = 4;
    let guidance: f32 = 4.0;
    let model = env_or("QWEN_EDIT_MODEL", "qwen_image_edit_2511");
    let prompt = "turn the plain gray card into a lush watercolor of a mountain lake at sunrise";
    let negative = "blurry, low quality, jpeg artifacts, deformed";

    let payload = json!({
        "model": model,
        "mode": "edit_image",
        "sourceAssetId": "smoke-src",
        "prompt": prompt,
        "width": w,
        "height": h,
        "advanced": { "mlxQuantize": 4 },
    });
    let request = ImageRequest::from_payload(payload.as_object().unwrap());
    let dir = resolve_qwen_edit_candle_base(&request, &settings)
        .expect("resolve_qwen_edit_candle_base")
        .unwrap_or_else(|| {
            panic!("{model}: SceneWorks/qwen-image-edit-2511-mlx turnkey not cached")
        });

    let render = |tag: &str| {
        let engine = QwenEdit::load(&QwenEditPaths {
            root: dir.clone(),
            adapters: Vec::new(),
            offload_policy: OffloadPolicy::Resident,
        })
        .unwrap_or_else(|e| panic!("QwenEdit::load ({tag}): {e}"));
        let reference = Image {
            width: w,
            height: h,
            pixels: vec![128u8; (w * h * 3) as usize],
        };
        let req = QwenEditRequest {
            prompt: prompt.to_owned(),
            negative: negative.to_owned(),
            width: w,
            height: h,
            steps,
            guidance,
            seed: 42,
            lightning: false,
            cancel: CancelFlag::new(),
        };
        let out = engine
            .generate(&req, std::slice::from_ref(&reference), &mut |_| {})
            .unwrap_or_else(|e| panic!("{model} generate ({tag}): {e}"));
        (out, engine) // caller controls the drop point
    };

    // ---- (1) baseline → load+render #1 → measure occupied → DROP ----
    let baseline = budget("baseline");
    let (out1, engine1) = render("render#1");
    let loaded = budget("after load+render #1 (engine still resident)");
    let occupied = (baseline.free_gb - loaded.free_gb).max(0.0);
    println!("[reclaim-smoke] render #1 occupied ~{occupied:.2} GB of VRAM");
    assert!(
        occupied > 4.0,
        "render #1 must occupy real VRAM (got {occupied:.2} GB) — the pool premise is untestable otherwise"
    );
    let std1 = image_std(&out1);
    assert!(
        std1 >= DEGENERATE_STD_FLOOR_DEFAULT,
        "render #1 degenerate (std {std1:.3})"
    );
    drop(engine1); // evict: free the resident generator's VRAM back for the next load

    // ---- (2) the crux: evicting/dropping the resident FREES its VRAM back (nvidia-smi free RISES) ----
    // This is the mechanism the fix depends on. The bespoke gate reads raw free WHILE a generator is
    // resident (reduced by `occupied`); admitting the render requires that evicting the resident will
    // free that VRAM. Confirm it does — most of `occupied` comes back.
    let after_drop = budget("after DROP of engine #1");
    let freed = after_drop.free_gb - loaded.free_gb; // VRAM the evict returned
    println!(
        "[reclaim-smoke] evicting engine #1 freed {freed:.2} GB back (of {occupied:.2} occupied)"
    );
    assert!(
        freed > occupied * 0.5,
        "evicting the resident generator must free most of its {occupied:.2} GB back for the incoming \
         render (only {freed:.2} GB came back) — the reclaim credit would be dishonest otherwise"
    );

    // ---- (3) the gate's reclaim credit PREDICTS that freed budget from the RESIDENT (reduced) reading.
    // While the generator is resident the bespoke gate sees `loaded` (free = total − occupied). Crediting
    // the resident's peak (recorded via note_loaded_peak — what the imminent evict frees) must land near
    // the ACTUAL post-evict free, so the gate can admit resident before the evict has run.
    note_loaded_peak(&gpu_id, occupied);
    let pool = reclaimable_pool_gb(&gpu_id);
    assert!(
        pool >= occupied - 1e-6,
        "note_loaded_peak/reclaimable_pool must record the occupied peak"
    );
    let predicted = with_reclaimable(loaded, pool); // the gate's view: reduced reading + reclaim credit
    println!(
        "[reclaim-smoke] gate credit: resident free {:.2} + peak {:.2} -> predicted {:.2} GB \
         (measured post-evict {:.2})",
        loaded.free_gb, pool, predicted.free_gb, after_drop.free_gb
    );
    assert!(
        (predicted.free_gb - after_drop.free_gb).abs() < 3.0,
        "the reclaim credit must predict the post-evict free within a few GB (predicted {:.2} vs \
         measured {:.2}) — otherwise the gate admits against a budget the evict never delivers",
        predicted.free_gb,
        after_drop.free_gb
    );
    // For the flip below, budget the render against the actual reclaimed free (measured post-evict).
    let reclaimed = after_drop;

    // ---- (4) the gate plan FLIPS: raw free below the resident peak downtiers; crediting the pooled
    // pages readmits it resident. Uses the REAL occupied value as the resident peak. The "warm 2nd
    // render" budget is forced deterministically (raw free just below the peak — what the retained pool
    // does on a card sized ~2x the model — pooling `occupied` GB) so the flip is demonstrated regardless
    // of how roomy THIS particular card is; the untouched post-drop nvidia budget is also checked when
    // the card is constrained enough to exhibit the flip on its own reading.
    let peak = Some(occupied);
    let seq = Some(occupied * 0.6); // a plausible staged peak, safely below the resident peak
    let constrained = VramBudget {
        free_gb: occupied - 1.0,
        total_gb: baseline.total_gb,
    };
    let raw_plan = load_plan(peak, seq, Some(constrained), true);
    let fixed_plan = load_plan(
        peak,
        seq,
        Some(with_reclaimable(constrained, occupied)),
        true,
    );
    println!(
        "[reclaim-smoke] deterministic flip (real peak {occupied:.2} GB, free {:.2}): raw={raw_plan:?} \
         -> reclaimed={fixed_plan:?}",
        constrained.free_gb
    );
    assert_ne!(
        raw_plan,
        LoadPlan::Resident,
        "raw free below the resident peak must NOT admit resident — that is the needless downtier \
         sc-13960 fixes"
    );
    assert_eq!(
        fixed_plan,
        LoadPlan::Resident,
        "reclaiming the pool must readmit the warm render at full residency"
    );

    // If this card's real post-drop free is already below the peak, the same flip holds on the
    // untouched nvidia-smi reading — the fully-natural warm-2nd-run condition.
    if after_drop.free_gb < occupied {
        assert_ne!(
            load_plan(peak, seq, Some(after_drop), true),
            LoadPlan::Resident
        );
        assert_eq!(
            load_plan(peak, seq, Some(reclaimed), true),
            LoadPlan::Resident
        );
        println!("[reclaim-smoke] natural flip on the real post-drop budget ALSO verified");
    } else {
        println!(
            "[reclaim-smoke] (card too roomy — post-drop free {:.2} >= peak {occupied:.2} — for a \
             fully-natural downtier; the deterministic flip above used the real occupied peak)",
            after_drop.free_gb
        );
    }

    // ---- (5) warm 2nd run: reload (reusing the pooled pages) + render resident, no OOM ----
    let (out2, engine2) = render("render#2 (warm)");
    let std2 = image_std(&out2);
    assert!(
        std2 >= DEGENERATE_STD_FLOOR_DEFAULT,
        "warm render #2 degenerate (std {std2:.3})"
    );
    drop(engine2);
    save_png(&out2, &out_dir.join(format!("{model}_warm2nd_{w}x{h}.png")));
    println!(
        "[reclaim-smoke] warm 2nd-run rendered resident OK (std {std2:.2}) — sc-13960 validated"
    );
}
