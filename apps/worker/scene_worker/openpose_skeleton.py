"""OpenPose (COCO-18) skeleton helpers for the InstantID pose library (sc-2064).

The pose library (apps/web/public/poses/) ships normalized 18-point skeletons; the web
sends the selected poses' keypoints in the job. This module renders an OpenPose control
image from those keypoints (matching the controlnet_aux draw_bodypose format the xinsir
SDXL OpenPose ControlNet was trained on — no controlnet_aux dependency) and derives a
small face box from the head keypoints so InstantID can anchor the face before the
face-restoration pass.

Keypoint order (COCO-18):
 0 nose 1 neck 2 r_sho 3 r_elb 4 r_wri 5 l_sho 6 l_elb 7 l_wri
 8 r_hip 9 r_kne 10 r_ank 11 l_hip 12 l_kne 13 l_ank 14 r_eye 15 l_eye 16 r_ear 17 l_ear
"""
from __future__ import annotations

import math

import numpy as np

LIMB_SEQ: tuple[tuple[int, int], ...] = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)
COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
)

Keypoint = tuple[float, float] | None


def normalize_keypoints(raw: object) -> list[Keypoint]:
    """Coerce a job-payload keypoint list into exactly 18 normalized (x, y) | None
    points. Accepts [x, y], [x, y, conf] (conf<=0 -> dropped), or None per entry."""
    points: list[Keypoint] = []
    items = raw if isinstance(raw, (list, tuple)) else []
    for entry in items:
        if entry is None or not isinstance(entry, (list, tuple)) or len(entry) < 2:
            points.append(None)
            continue
        if len(entry) >= 3 and entry[2] is not None and float(entry[2]) <= 0:
            points.append(None)
            continue
        try:
            points.append((float(entry[0]), float(entry[1])))
        except (TypeError, ValueError):
            points.append(None)
    points = points[:18] + [None] * max(0, 18 - len(points))
    return points


def draw_bodypose(canvas_w: int, canvas_h: int, keypoints: list[Keypoint], stickwidth: int = 4) -> np.ndarray:
    """Render an OpenPose (COCO-18) skeleton (black background, colored sticks + joints)
    matching the controlnet_aux format. Returns an RGB uint8 array."""
    import cv2

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    pts = [None if p is None else (float(p[0]) * canvas_w, float(p[1]) * canvas_h) for p in keypoints]

    for i, (a, b) in enumerate(LIMB_SEQ):
        if a >= len(pts) or b >= len(pts) or pts[a] is None or pts[b] is None:
            continue
        xa, ya = pts[a]
        xb, yb = pts[b]
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        length = math.hypot(xa - xb, ya - yb)
        angle = math.degrees(math.atan2(ya - yb, xa - xb))
        poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, COLORS[i])

    for i in range(min(18, len(pts))):
        if pts[i] is None:
            continue
        x, y = pts[i]
        cv2.circle(canvas, (int(x), int(y)), stickwidth, COLORS[i], thickness=-1)
    return canvas


def face_box_from_keypoints(keypoints: list[Keypoint]) -> tuple[float, float, float] | None:
    """(cx, cy, height_frac) for placing the InstantID face kps, derived from the head
    keypoints (nose / eyes / neck). Returns None when the head is not visible (e.g. a
    back view or a pose where the face is occluded), so the adapter disables IdentityNet
    + the face-restoration pass and lets the shared seed carry continuity there."""
    nose = keypoints[0] if len(keypoints) > 0 else None
    r_eye = keypoints[14] if len(keypoints) > 14 else None
    l_eye = keypoints[15] if len(keypoints) > 15 else None
    neck = keypoints[1] if len(keypoints) > 1 else None
    eyes = [e for e in (r_eye, l_eye) if e is not None]
    if nose is None and not eyes:
        return None  # no usable face landmarks

    cx = nose[0] if nose is not None else sum(e[0] for e in eyes) / len(eyes)
    head_ys = [p[1] for p in (nose, r_eye, l_eye) if p is not None]
    top_y = min(head_ys)
    # Estimate face height from the neck->nose span when available (head is ~1.4x that
    # vertical run), else a sensible default; clamp to a small full-body face fraction.
    if neck is not None and nose is not None:
        face_h = abs(neck[1] - nose[1]) * 1.4
    else:
        face_h = 0.09
    face_h = max(0.045, min(0.20, face_h))
    cy = top_y + face_h * 0.45
    return (cx, cy, face_h)
