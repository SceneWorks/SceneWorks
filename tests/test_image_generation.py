from __future__ import annotations

from sceneworks_api import image_generation
from sceneworks_api.image_generation import ImageJobRequest, image_job_payload


def test_blank_image_seed_gets_independent_batch_seeds(monkeypatch):
    values = iter([101, 202, 303, 404])
    monkeypatch.setattr(image_generation.secrets, "randbits", lambda _bits: next(values))

    request = ImageJobRequest(projectId="project-1", prompt="city at night", seed=None, count=4)
    payload = image_job_payload(request)

    assert payload["seed"] is None
    assert payload["seeds"] == [101, 202, 303, 404]


def test_explicit_image_seed_is_preserved():
    request = ImageJobRequest(projectId="project-1", prompt="city at night", seed=1234)

    payload = image_job_payload(request)

    assert payload["seed"] == 1234
    assert "seeds" not in payload
