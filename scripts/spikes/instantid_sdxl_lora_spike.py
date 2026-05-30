"""sc-2222 spike — InstantID SDXL LoRA stacking on MPS (GO/NO-GO gate for sc-2224).

Question: does the existing SDXL LoRA merge path
(``lora_adapters.apply_loras_to_pipeline`` -> diffusers ``load_lora_weights``) stack
cleanly onto the InstantID pipeline — a ``StableDiffusionXLControlNetPipeline``
(IdentityNet, plus an OpenPose ControlNet in pose mode) with the InstantID
IP-Adapter loaded — in **bf16 on MPS**, without breaking ControlNet/IP-Adapter
conditioning or NaN-ing, and does the merge visibly change the output, preserve
identity, and remove cleanly between jobs?

It drives the REAL adapter code (``InstantIDAdapter._load_pipeline`` / ``_run_pipeline``
/ ``_run_pose``) and the REAL ``apply_loras_to_pipeline`` — the exact seam sc-2224 will
wire — so the findings transfer directly. The only monkeypatch is
``load_reference_image``, swapped to read a face PNG straight off disk (the asset-sidecar
resolution is orthogonal to what the spike tests).

Probe LoRA: ``latent-consistency/lcm-lora-sdxl`` — a genuine SDXL UNet LoRA whose merge
produces a large, unambiguous pixel delta at full weight. It is a *sampling-regime* LoRA,
not a character LoRA, so it is a MECHANICAL probe of the merge path; a real character
SDXL LoRA belongs in the sc-2226 e2e.

Run with the SceneWorks desktop venv (has torch/diffusers/insightface/peft):
  "/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python" \
      scripts/spikes/instantid_sdxl_lora_spike.py

Outputs PNGs + a JSON summary under --out (default /tmp/sc2222_instantid_lora).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --- resolve repo paths + defaults BEFORE importing scene_worker -----------------
_REPO = Path(__file__).resolve().parents[2]
for _pkg in (_REPO / "apps" / "worker", _REPO / "packages" / "shared"):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

_DEFAULT_DATA = Path.home() / "Library" / "Application Support" / "SceneWorks" / "data"
_DEFAULT_REF = (
    _DEFAULT_DATA
    / "projects" / "test.sceneworks" / "assets" / "uploads" / "kelsie-0009-5b1ebbc9.png"
)
_DEFAULT_LORA = (
    Path.home()
    / ".cache" / "huggingface" / "hub" / "models--latent-consistency--lcm-lora-sdxl"
    / "snapshots" / "a18548dd4956b174ec5b0d78d340c8dae0a129cd"
    / "pytorch_lora_weights.safetensors"
)
_POSE_INDEX = _REPO / "apps" / "web" / "public" / "poses" / "index.json"

os.environ.setdefault("SCENEWORKS_GPU_ID", "mps")
os.environ.setdefault("SCENEWORKS_DATA_DIR", str(_DEFAULT_DATA))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from scene_worker import instantid_adapter as ia  # noqa: E402
from scene_worker.image_adapters import ImageRequest, MODEL_TARGETS  # noqa: E402
from scene_worker.instantid_adapter import InstantIDAdapter  # noqa: E402
from scene_worker.lora_adapters import apply_loras_to_pipeline  # noqa: E402
from scene_worker.openpose_skeleton import normalize_keypoints  # noqa: E402
from scene_worker.settings import WorkerSettings  # noqa: E402

import torch  # noqa: E402

MODEL = "instantid_realvisxl"
ADAPTER_ID = "instantid_sdxl"
SEED = 12345


def _noop_progress(*_args, **_kwargs) -> None:
    return None


def _mps_mem_gb() -> dict[str, float]:
    try:
        return {
            "current_gb": round(torch.mps.current_allocated_memory() / 1e9, 2),
            "driver_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2),
        }
    except Exception:
        return {}


def _img_stats(img: Image.Image) -> dict[str, float]:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return {"mean": round(float(arr.mean()), 2), "min": float(arr.min()), "max": float(arr.max())}


def _pixel_delta(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute pixel difference (0-255 scale) between two same-size RGB images."""
    aa = np.asarray(a.convert("RGB"), dtype=np.float32)
    bb = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float32)
    return round(float(np.abs(aa - bb).mean()), 3)


def _build_request(advanced: dict) -> ImageRequest:
    return ImageRequest(
        project_id="spike",
        mode="character_image",
        prompt="a woman, portrait, natural light, photorealistic",
        negative_prompt="blurry, lowres, deformed",
        model=MODEL,
        count=1,
        seed=SEED,
        seeds=[],
        width=768,
        height=768,
        style_preset="cinematic",
        loras=[],
        character_id=None,
        character_look_id=None,
        source_asset_id=None,
        reference_asset_id="spike-ref",
        advanced=advanced,
        model_manifest_entry={},
    )


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return round(float(np.dot(u, v) / (nu * nv)), 4)


def _face_embedding(adapter: InstantIDAdapter, img: Image.Image) -> np.ndarray | None:
    """ArcFace embedding of the largest face in img, or None if no face detected."""
    import cv2

    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = adapter._face_analysis().get(bgr)
    if not faces:
        return None
    face = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))[-1]
    return np.asarray(face["embedding"], dtype=np.float32)


def _active_adapters(pipe) -> list[str]:
    try:
        active = pipe.get_active_adapters()
        return list(active) if active else []
    except Exception:
        try:
            return list(getattr(pipe.unet, "active_adapters", lambda: [])() or [])
        except Exception:
            return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=_DEFAULT_REF)
    ap.add_argument("--lora", type=Path, default=_DEFAULT_LORA)
    ap.add_argument("--out", type=Path, default=Path("/tmp/sc2222_instantid_lora"))
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--skip-pose", action="store_true")
    args = ap.parse_args()

    if not args.reference.exists():
        print(f"FATAL: reference not found: {args.reference}", file=sys.stderr)
        return 2
    if not args.lora.exists():
        print(f"FATAL: lora not found: {args.lora}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    # Monkeypatch reference resolution to read the PNG directly (sidecar lookup is
    # orthogonal to the merge path this spike exercises).
    reference_img = Image.open(args.reference).convert("RGB")
    ia.load_reference_image = lambda *_a, **_k: reference_img  # type: ignore[assignment]

    lora_spec = {
        "id": "lcm-lora-sdxl",
        "path": str(args.lora),
        "weight": args.weight,
        "families": ["sdxl"],
    }

    settings = WorkerSettings()
    adapter = InstantIDAdapter()
    model_target = MODEL_TARGETS[MODEL]
    project_path = _DEFAULT_DATA / "projects" / "test.sceneworks"
    advanced = {"steps": args.steps, "viewAngle": "front", "ipAdapterScale": 0.8}
    request = _build_request(advanced)

    findings: dict = {
        "device": "mps",
        "dtype": "bfloat16",
        "model": MODEL,
        "lora": lora_spec["id"],
        "steps": args.steps,
        "weight": args.weight,
        "torch": torch.__version__,
    }

    # ---- reference identity baseline ------------------------------------------
    ref_emb = _face_embedding(adapter, reference_img)
    findings["reference_face_detected"] = ref_emb is not None

    # ============================ PROBE A: identity / angle mode ===============
    print(">>> PROBE A: identity-mode (IdentityNet + IP-Adapter), angle=front")
    t0 = time.time()
    pipe = adapter._load_pipeline(
        settings, request, model_target, progress=_noop_progress, job_id="spike-a", pose_set=False
    )
    findings["load_seconds"] = round(time.time() - t0, 1)
    findings["mem_after_load"] = _mps_mem_gb()

    a = {}
    try:
        t0 = time.time()
        base_img = adapter._run_pipeline(settings, pipe, request, SEED, project_path, view_angle_override="front")
        a["baseline_seconds"] = round(time.time() - t0, 1)
        base_img.save(args.out / "A_baseline.png")
        a["baseline_stats"] = _img_stats(base_img)

        state = apply_loras_to_pipeline(
            pipe, [lora_spec], adapter_id=ADAPTER_ID, model_family="sdxl"
        )
        a["lora_applied_adapters"] = list(state.adapter_names)
        a["active_after_apply"] = _active_adapters(pipe)
        a["mem_after_lora"] = _mps_mem_gb()

        t0 = time.time()
        lora_img = adapter._run_pipeline(settings, pipe, request, SEED, project_path, view_angle_override="front")
        a["lora_seconds"] = round(time.time() - t0, 1)
        lora_img.save(args.out / "A_with_lora.png")
        a["lora_stats"] = _img_stats(lora_img)
        a["delta_baseline_vs_lora"] = _pixel_delta(base_img, lora_img)

        # Clean removal: clear LoRA, re-render the same seed -> should match baseline.
        cleared_state = apply_loras_to_pipeline(
            pipe, [], adapter_id=ADAPTER_ID, model_family="sdxl", previous_state=state
        )
        a["active_after_clear"] = _active_adapters(pipe)
        cleared_img = adapter._run_pipeline(settings, pipe, request, SEED, project_path, view_angle_override="front")
        cleared_img.save(args.out / "A_cleared.png")
        a["delta_baseline_vs_cleared"] = _pixel_delta(base_img, cleared_img)

        # Identity preservation: ArcFace cosine vs the reference for base vs LoRA.
        if ref_emb is not None:
            be = _face_embedding(adapter, base_img)
            le = _face_embedding(adapter, lora_img)
            a["cosine_baseline_vs_ref"] = _cosine(be, ref_emb) if be is not None else None
            a["cosine_lora_vs_ref"] = _cosine(le, ref_emb) if le is not None else None
        a["ok"] = True
    except Exception as exc:  # noqa: BLE001
        a["ok"] = False
        a["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        traceback.print_exc()
    findings["probe_a_identity"] = a

    # ============================ PROBE B: pose / multi-controlnet =============
    if not args.skip_pose:
        print(">>> PROBE B: pose-mode (IdentityNet + OpenPose + IP-Adapter) + faceRestore")
        b = {}
        try:
            keypoints = _load_sample_pose()
            b["pose_keypoints_loaded"] = keypoints is not None
            pose_advanced = {"steps": args.steps, "ipAdapterScale": 0.8, "faceRestore": True}
            pose_request = _build_request(pose_advanced)
            pose_pipe = adapter._load_pipeline(
                settings, pose_request, model_target, progress=_noop_progress,
                job_id="spike-b", pose_set=True,
            )
            b["mem_after_pose_load"] = _mps_mem_gb()
            state = apply_loras_to_pipeline(
                pose_pipe, [lora_spec], adapter_id=ADAPTER_ID, model_family="sdxl"
            )
            b["active_before_gen"] = _active_adapters(pose_pipe)
            t0 = time.time()
            # faceRestore=True => a SECOND pipe() call (_restore_face); confirms the merged
            # LoRA persists across calls without re-applying (the sc-2222 _restore_face Q).
            pose_img = adapter._run_pose(
                settings, pose_pipe, pose_request, SEED, project_path, normalize_keypoints(keypoints)
            )
            b["pose_seconds"] = round(time.time() - t0, 1)
            pose_img.save(args.out / "B_pose_with_lora.png")
            b["pose_stats"] = _img_stats(pose_img)
            b["active_after_gen"] = _active_adapters(pose_pipe)
            b["lora_persisted_across_restore"] = bool(b["active_after_gen"])
            b["mem_peak"] = _mps_mem_gb()
            b["ok"] = True
        except Exception as exc:  # noqa: BLE001
            b["ok"] = False
            b["error"] = f"{type(exc).__name__}: {exc}"
            import traceback

            traceback.print_exc()
        findings["probe_b_pose"] = b

    # ---- GO/NO-GO heuristic ----------------------------------------------------
    a = findings.get("probe_a_identity", {})
    # NB: compare against explicit values, not `x or default` — a perfectly clean
    # removal yields delta 0.0, which is falsy and would wrongly fall through to the
    # default with `or`.
    delta_lora = a.get("delta_baseline_vs_lora")
    delta_cleared = a.get("delta_baseline_vs_cleared")
    base_mean = a.get("baseline_stats", {}).get("mean")
    lora_mean = a.get("lora_stats", {}).get("mean")
    go = bool(
        a.get("ok")
        and delta_lora is not None and delta_lora > 3.0       # LoRA visibly changed output
        and base_mean is not None and base_mean > 5.0         # not all-black (no NaN)
        and lora_mean is not None and lora_mean > 5.0
        and delta_cleared is not None and delta_cleared < 2.0  # clean removal
    )
    findings["verdict"] = "GO" if go else "REVIEW"

    print("\n================ sc-2222 SPIKE SUMMARY ================")
    print(json.dumps(findings, indent=2))
    (args.out / "summary.json").write_text(json.dumps(findings, indent=2))
    print(f"\nImages + summary.json written to: {args.out}")
    print(f"VERDICT: {findings['verdict']}")
    return 0


def _load_sample_pose():
    """First library pose with a visible head (so IdentityNet + face-restore engage)."""
    try:
        data = json.loads(_POSE_INDEX.read_text())
    except Exception:
        return None
    poses = data.get("poses", []) if isinstance(data, dict) else []
    for pose in poses:
        kp = pose.get("keypoints")
        if isinstance(kp, list) and len(kp) >= 16 and kp[0] is not None:
            return kp
    return poses[0].get("keypoints") if poses else None


if __name__ == "__main__":
    raise SystemExit(main())
