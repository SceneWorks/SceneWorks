"""Unit tests for the InsightFace kps-extraction adapter (sc-4433).

Cover the pure square-normalization (kept in parity with the Rust
``kps_jobs::normalize_to_square``), the backend-availability gate, and the
``run_kps_extract`` orchestration with a fake detector so coverage needs neither
the antelopev2 onnx weights nor a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scene_worker import kps_extract_adapters as ka


def test_normalize_square_image_is_plain_fraction():
    # Square image: side cancels, normalized = px/side, py/side (Rust parity).
    assert ka._normalize_to_square(512.0, 256.0, 1024, 1024) == pytest.approx([0.5, 0.25])
    assert ka._normalize_to_square(0.0, 0.0, 800, 800) == pytest.approx([0.0, 0.0])


def test_normalize_landscape_centers_vertically():
    # 1000x500 landscape (M=1000): x = px/1000; y letterboxed into the centered band.
    x, y = ka._normalize_to_square(500.0, 250.0, 1000, 500)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)
    # Top edge of the source sits at 0.25 (the top pad band).
    assert ka._normalize_to_square(0.0, 0.0, 1000, 500)[1] == pytest.approx(0.25)


def test_normalize_portrait_centers_horizontally():
    # 500x1000 portrait (M=1000): y = py/1000; x centered into [0.25, 0.75].
    x, y = ka._normalize_to_square(250.0, 500.0, 500, 1000)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)
    assert ka._normalize_to_square(0.0, 0.0, 500, 1000)[0] == pytest.approx(0.25)


def test_backend_available_reflects_optional_deps(monkeypatch):
    monkeypatch.setattr(ka.importlib.util, "find_spec", lambda _name: object())
    assert ka.kps_extractor_backend_available() is True
    monkeypatch.setattr(
        ka.importlib.util,
        "find_spec",
        lambda name: None if name == "insightface" else object(),
    )
    assert ka.kps_extractor_backend_available() is False


def _square_png(tmp_path, side: int = 400):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "face.png"
    Image.new("RGB", (side, side), (128, 128, 128)).save(path)
    return str(path)


def test_run_kps_extract_normalizes_largest_face(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    path = _square_png(tmp_path, 400)

    # A fake SCRFD face: bbox + 5 kps in pixels on a 400x400 square → normalized = px/400.
    face = SimpleNamespace(
        bbox=[100.0, 100.0, 300.0, 300.0],
        kps=[[150, 160], [250, 160], [200, 210], [160, 260], [240, 260]],
        det_score=0.91,
    )
    monkeypatch.setattr(ka, "_face_app", lambda _s: SimpleNamespace(get=lambda _bgr: [face]))
    monkeypatch.setattr(ka, "_resolve_source_path", lambda _s, _j, _p: path)

    result = ka.run_kps_extract(None, {"payload": {"sourcePath": path}})
    assert result["detected"] is True
    assert result["kpsOrder"][2] == "nose"
    assert result["kps"][0] == pytest.approx([0.375, 0.4])  # 150/400, 160/400
    assert result["kps"][2] == pytest.approx([0.5, 0.525])  # nose 200/400, 210/400
    assert result["bbox"] == pytest.approx([0.25, 0.25, 0.75, 0.75])
    assert result["lowConfidence"] is False
    assert result["detector"]["backend"] == "insightface"


def test_run_kps_extract_no_face_is_explicit(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    path = _square_png(tmp_path, 320)
    monkeypatch.setattr(ka, "_face_app", lambda _s: SimpleNamespace(get=lambda _bgr: []))
    monkeypatch.setattr(ka, "_resolve_source_path", lambda _s, _j, _p: path)

    result = ka.run_kps_extract(None, {"payload": {"sourcePath": path}})
    assert result["detected"] is False
    assert result["reason"] == "no_face"
    assert "kps" not in result


def test_resolve_source_requires_a_source():
    with pytest.raises(ka.KpsExtractError):
        ka._resolve_source_path(None, {}, {})
