# sc-4422 / sc-4424 — InstantID angle-set kps reframing (validation record)

The Character-Studio InstantID angle set was reframed so the subject **fills the frame**
(head-and-shoulders, shoulders retained) instead of the original small, low-in-frame face —
the outputs are LoRA training data, so more character pixels = more training signal.

## How the presets were derived

1. `extract_kps.py` — ran SCRFD (insightface antelopev2) over an 11-view reference set to
   recover the per-angle 5-point landmarks (`extracted_kps_from_qwen_refs.json`). The reference
   captured the head *orientations*; the *framing* was retuned next.
2. `tighten_preview.py` — scaled/centered each cloud to a head-and-shoulders crop
   (`tightened_kps_validated.json`).
3. `instantid_exp.py` / `render_all11.py` — rendered the real InstantID stack
   (RealVisXL_V5.0 + InstantX/InstantID) with the tightened kps. `measure_out.py` measured the
   rendered framing.

## Findings

- Face height 38% → **52–54%**, dead headroom removed (eye-line 0.52 → 0.34), consistent across
  the set. See `instantid_current_vs_tightened.png` and `montage_all11.png`.
- **InstantID renders to the kps within ~0.005** → deterministic framing control.
- The three down-tilt views clipped the crown; a +0.06 downward shift fixed them
  (`instantid_downfix/before_after.png`). Final values are in `tightened_kps_validated.json`.

## Where the values live

`tightened_kps_validated.json` is the source of truth, committed verbatim into:
- `crates/sceneworks-worker/src/image_jobs.rs` → `INSTANTID_ANGLE_KPS` (Mac MLX path, passed to
  the engine via `generate_with_kps`).
- `apps/worker/scene_worker/instantid_adapter.py` → `VIEW_ANGLE_KPS` (Win/Linux torch path).

Scripts use absolute local paths and a cached `~/mlx-flux-venv`; they are kept for reproducibility,
not as part of the build.
