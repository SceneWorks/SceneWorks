"""SCRFD 5-point face-landmark extraction for the Key Point Library (epic 4422, sc-4433).

The Windows/Linux InsightFace counterpart of the macOS native-MLX path
(``crates/sceneworks-worker/src/kps_jobs.rs``). Given one image, detect the largest face
and return its 5 landmarks (``[left_eye, right_eye, nose, mouth_left, mouth_right]``)
normalized to a square ``[0,1]`` canvas — directly consumable as an InstantID
angle/framing preset by the engine pass-in path (``generate_with_kps``, sc-4425). The
reference supplies *identity*; these kps supply the *pose/framing*.

Detection-only: reuses the antelopev2 SCRFD detector InstantID already provisions
(``instantid_adapter._ensure_antelopev2``); no ArcFace embedding or SDXL stack. The
worker advertises ``kps_extract`` only when the InsightFace backend is installed.

Normalization mirrors the engine's centered square letterbox (``kps::letterbox`` /
``instantid_adapter._letterbox``): a detected pixel ``(px, py)`` in a ``w x h`` image maps
to ``((1 - w/M)/2 + px/M, (1 - h/M)/2 + py/M)`` with ``M = max(w, h)`` — scale-free, so the
result round-trips through ``generate_with_kps`` (which draws ``kps_norm * side`` on a
square canvas after letterboxing the reference the same way). Kept byte-identical in
spirit to the Rust ``normalize_to_square`` so both platforms produce the same preset.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

ProgressCallback = Callable[[str, str, float, str], None]
CancelCallback = Callable[[], bool]

# Landmark order, surfaced on the result so a consumer never has to assume it. Matches
# the Rust ``KPS_ORDER`` and the InstantID ``VIEW_ANGLE_KPS`` ordering.
KPS_ORDER = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
# SCRFD detector input (square), matches the macOS native path + InstantID.
DET_SIZE = (640, 640)
# Below this detection score the landmarks are returned but flagged ``lowConfidence`` so
# the caller can warn the captured angle may be unreliable (matches the Rust threshold).
LOW_CONF_THRESH = 0.65
_DETECTOR = {"id": "antelopev2_scrfd", "backend": "insightface"}


class KpsExtractError(RuntimeError):
    """Raised for an unusable extraction request (no resolvable source image)."""


def kps_extractor_backend_available() -> bool:
    """True when InsightFace + onnxruntime + OpenCV are importable (the SCRFD path)."""
    return all(
        importlib.util.find_spec(mod) is not None
        for mod in ("insightface", "onnxruntime", "cv2")
    )


def _normalize_to_square(px: float, py: float, w: int, h: int) -> list[float]:
    """Map a detected pixel coordinate into square-normalized ``[0,1]`` preset space,
    applying the engine's centered-letterbox geometry analytically (no image resize)."""
    m = float(max(w, h))
    return [(1.0 - w / m) / 2.0 + px / m, (1.0 - h / m) / 2.0 + py / m]


def _face_app(settings: Any) -> Any:
    """Load the antelopev2 FaceAnalysis app (CPU EP), reusing InstantID's provisioning."""
    from insightface.app import FaceAnalysis

    from .instantid_adapter import _ensure_antelopev2

    root = _ensure_antelopev2()
    app = FaceAnalysis(
        name="antelopev2", root=str(root), providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    return app


def _resolve_source_path(settings: Any, job: dict[str, Any], payload: dict[str, Any]) -> str:
    """Resolve the source image: a staged ``sourcePath`` or a project ``sourceAssetId``
    (+ ``projectId``). Mirrors the dual contract the Rust handler accepts."""
    raw = payload.get("sourcePath")
    if raw and str(raw).strip():
        return str(raw)

    asset_id = payload.get("sourceAssetId")
    project_id = payload.get("projectId") or job.get("projectId")
    if asset_id and str(asset_id).strip() and project_id:
        from sceneworks_shared import find_project_path, load_asset_with_media

        project_path = find_project_path(
            Path(getattr(settings, "data_dir", ".")) / "recent-projects.json",
            str(project_id),
        )
        if project_path is not None:
            _record, media_path = load_asset_with_media(project_path, str(asset_id))
            if media_path is not None:
                return str(media_path)

    raise KpsExtractError("kps extraction needs a sourceAssetId or sourcePath")


def run_kps_extract(
    settings: Any,
    job: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
    cancel_requested: CancelCallback | None = None,
) -> dict[str, Any]:
    """Detect the largest face in one image and return its normalized 5-point kps.

    payload: ``{"projectId"?, "sourceAssetId"?, "sourcePath"?}``
    result:  ``{"detected", "kps"?, "kpsOrder"?, "bbox"?, "detScore"?, "lowConfidence"?,
                "sourceWidth", "sourceHeight", "detector"}``

    ``detected: false`` (with ``reason: "no_face"``) is an explicit, well-formed result —
    the acceptance "clear failure, not silent bad data" — not an exception.
    """
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    payload = job.get("payload") or {}
    image_path = _resolve_source_path(settings, job, payload)
    pil = PILImage.open(image_path).convert("RGB")
    w, h = pil.width, pil.height

    if progress is not None:
        progress("running", "running", 0.5, "Detecting face landmarks.")

    app = _face_app(settings)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return {
            "detected": False,
            "reason": "no_face",
            "sourceWidth": w,
            "sourceHeight": h,
            "detector": dict(_DETECTOR),
        }

    face = sorted(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )[-1]
    kps = [_normalize_to_square(float(x), float(y), w, h) for x, y in face.kps[:5]]
    bbox = _normalize_to_square(
        float(face.bbox[0]), float(face.bbox[1]), w, h
    ) + _normalize_to_square(float(face.bbox[2]), float(face.bbox[3]), w, h)
    score = float(getattr(face, "det_score", 0.0))
    return {
        "detected": True,
        "kps": kps,
        "kpsOrder": list(KPS_ORDER),
        "bbox": bbox,
        "detScore": score,
        "lowConfidence": score < LOW_CONF_THRESH,
        "sourceWidth": w,
        "sourceHeight": h,
        "detector": dict(_DETECTOR),
    }
