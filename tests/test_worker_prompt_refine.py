from __future__ import annotations

from worker_runtime_shared import *

def test_build_system_prompt_uses_workflow_medium_and_embeds_guide():
    image = build_system_prompt("# Z-Image Guide\n\nUse short prompts.", "image")
    assert "generative image model" in image
    assert "Z-Image Guide" in image

    video = build_system_prompt(None, "video")
    assert "generative video model" in video
    assert "# Model prompt guide" not in video

def test_clean_output_strips_reasoning_and_quoting():
    assert clean_output("<think>plan</think>A vivid sunset over hills.") == "A vivid sunset over hills."
    assert clean_output('"A vivid sunset over hills."') == "A vivid sunset over hills."
    assert clean_output("```\nA vivid sunset over hills.\n```") == "A vivid sunset over hills."

def test_prompt_refiner_load_unavailable_without_backend():
    # torch/transformers aren't installed in the CI worker-test env, so load()
    # surfaces a clear PromptRefineUnavailable rather than an opaque ImportError.
    with pytest.raises(PromptRefineUnavailable):
        PromptRefiner(model_name_or_path="some/model", gpu_id="cpu").load()

def test_prompt_refiner_load_is_idempotent_when_already_resident(monkeypatch):
    # sc-4191: a second load() for the same checkpoint must not re-import torch /
    # re-run from_pretrained — it returns immediately while the model is resident.
    refiner = PromptRefiner(model_name_or_path="some/model", gpu_id="cpu")
    refiner.model = object()  # pretend the checkpoint is resident
    refiner._loaded_model_name = "some/model"

    def _boom(*_args, **_kwargs):
        raise AssertionError("load() must not reload while the same model is resident")

    monkeypatch.setattr("scene_worker.prompt_refine.require_inference_backend_for_gpu_worker", _boom)
    refiner.load()  # no exception → early-returned without touching the backend

def test_prompt_refiner_unload_frees_and_is_safe_when_empty():
    # sc-4191: unload() drops the resident model and reports it; a no-op when
    # nothing is loaded so evict_other_image_adapters can call it unconditionally.
    refiner = PromptRefiner(model_name_or_path="some/model", gpu_id="cpu")
    assert refiner.unload() is False  # nothing loaded yet

    released = {}

    refiner.model = object()
    refiner.tokenizer = object()
    refiner.torch = object()
    refiner._loaded_model_name = "some/model"
    import scene_worker.prompt_refine as pr

    orig = pr.release_inference_memory
    pr.release_inference_memory = lambda torch: released.setdefault("called", True)
    try:
        assert refiner.unload() is True
    finally:
        pr.release_inference_memory = orig

    assert released.get("called") is True
    assert refiner.model is None
    assert refiner.tokenizer is None
    assert refiner.loaded_models() == []

def test_prompt_refiner_refine_applies_chat_template_and_cleans():
    # Drive refine() with a fake tokenizer/model so we exercise the chat-template
    # → generate → decode → clean path without real weights.
    refiner = PromptRefiner(model_name_or_path="some/model", gpu_id="cpu")

    class _Ids:
        shape = (1, 3)

        def to(self, _device):
            return self

    captured = {}

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            return _Ids()

        def decode(self, tokens, **kwargs):
            return "<think>scheming</think>A vivid neon street at midnight."

    class _Model:
        def generate(self, **kwargs):
            captured["generate"] = kwargs
            return [[101, 102, 103, 201, 202]]

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    refiner.tokenizer = _Tokenizer()
    refiner.model = _Model()
    refiner.device = "cpu"
    refiner.torch = SimpleNamespace(no_grad=lambda: _NoGrad())

    out = refiner.refine("neon street", guide="# Guide", workflow="image")

    assert out == "A vivid neon street at midnight."  # think-block stripped
    # Instruction + guide folded into a single user turn (portable across templates).
    assert captured["messages"][0]["role"] == "user"
    assert "# Guide" in captured["messages"][0]["content"]
    assert "neon street" in captured["messages"][0]["content"]

def test_run_prompt_refine_job_writes_refined_result(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    _FakeRefiner.instances = []
    adapters = _refine_adapters()
    api = _DryRunApi()
    job = {"id": "job-refine-1", "type": "prompt_refine", "payload": {"prompt": "dog in park", "workflow": "image"}}

    run_prompt_refine_job(api, _refine_settings(), job, adapters)

    terminal = api.progress[-1]
    assert terminal["status"] == "completed"
    assert terminal["result"]["refinedPrompt"] == "Refined (image): dog in park"
    assert terminal["result"]["originalPrompt"] == "dog in park"
    # Loaded the model before refining; emitted a loading_model stage.
    assert adapters["prompt_refiner"].loaded is True
    assert any(entry["stage"] == "loading_model" for entry in api.progress)

def test_run_prompt_refine_job_reuses_resident_refiner(monkeypatch):
    """sc-4191: repeat prompt_refine jobs must reuse the one resident refiner and
    its already-loaded model, never construct a new instance or reload per job."""
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    _FakeRefiner.instances = []
    adapters = _refine_adapters()
    refiner = adapters["prompt_refiner"]
    api = _DryRunApi()

    for n in range(3):
        job = {"id": f"job-refine-reuse-{n}", "type": "prompt_refine", "payload": {"prompt": "dog", "workflow": "image"}}
        run_prompt_refine_job(api, _refine_settings(), job, adapters)

    # Exactly one refiner ever constructed across three jobs.
    assert len(_FakeRefiner.instances) == 1
    assert _FakeRefiner.instances[0] is refiner
    # load() is called each job but stays cheap (idempotent in the real adapter);
    # the fake just records it. The instance is reused, not rebuilt.
    assert refiner.load_calls == 3
    assert refiner.loaded is True

def test_run_prompt_refine_job_reports_failure(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)

    class _BoomRefiner(_FakeRefiner):
        def refine(self, prompt, *, guide, workflow):
            raise PromptRefineUnavailable("Could not load the prompt-refinement model.")

    api = _DryRunApi()
    adapters = {"prompt_refiner": _BoomRefiner(model_name_or_path="", gpu_id="cpu", max_new_tokens=512)}
    job = {"id": "job-refine-2", "type": "prompt_refine", "payload": {"prompt": "dog", "workflow": "image"}}

    run_prompt_refine_job(api, _refine_settings(), job, adapters)

    terminal = api.progress[-1]
    assert terminal["status"] == "failed"
    assert "Could not load" in terminal["message"]

