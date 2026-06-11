from __future__ import annotations

from worker_runtime_shared import *

def test_cpu_worker_does_not_advertise_gpu_generation_capabilities():
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})

    assert capabilities == ["cpu"]
    assert "placeholder" not in capabilities

def test_gpu_worker_advertises_generation_capabilities(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})

    assert "image_generate" in capabilities
    assert "image_vqa" in capabilities
    assert "video_generate" in capabilities
    assert "training_caption" in capabilities
    assert "person_replace" in capabilities
    assert "placeholder" not in capabilities

def test_gpu_worker_without_cuda_torch_does_not_claim_generation_jobs(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: False)
    # pose_detect and kps_extract are torch-independent, so they are advertised whenever
    # their backends are installed — force them off here so this torch-gating assertion
    # stays deterministic across local and CI optional dependency sets.
    monkeypatch.setattr("scene_worker.runtime.pose_detector_backend_available", lambda: False)
    monkeypatch.setattr("scene_worker.runtime.kps_extractor_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu", "nvidia"]})

    # lora_train dry-run validation needs no inference backend, so it is
    # advertised even without torch; generation job types are not.
    assert capabilities == ["gpu", "lora_train", "nvidia"]
    for job_type in ("image_generate", "image_edit", "image_vqa", "video_generate", "training_caption", "prompt_refine"):
        assert job_type not in capabilities

def test_python_worker_only_advertises_inference_job_capabilities(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    # Detection backends (pose/kps) are advertised only when their optional deps are
    # installed; force them off so this torch-inference list stays deterministic.
    monkeypatch.setattr("scene_worker.runtime.pose_detector_backend_available", lambda: False)
    monkeypatch.setattr("scene_worker.runtime.kps_extractor_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    job_capabilities = [capability for capability in capabilities if capability != "gpu"]

    assert job_capabilities == [
        "image_detail",
        "image_edit",
        "image_generate",
        "image_interleave",
        "image_upscale",
        "image_vqa",
        "lora_train",
        "lora_train_execute",
        "person_replace",
        "prompt_refine",
        "training_caption",
        "video_bridge",
        "video_extend",
        "video_generate",
    ]

def test_python_cpu_worker_does_not_advertise_person_tracking_jobs():
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})

    assert capabilities == ["cpu"]

def test_gpu_worker_advertises_real_person_jobs_when_backends_installed(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.detector_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.tracker_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.segmenter_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "person_detect" in capabilities
    assert "person_track" in capabilities
    assert "person_segment" in capabilities

def test_gpu_worker_advertises_kps_extract_when_backend_installed(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.kps_extractor_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "kps_extract" in capabilities

def test_gpu_worker_omits_kps_extract_without_backend(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.kps_extractor_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "kps_extract" not in capabilities

def test_gpu_worker_omits_person_jobs_without_detector_backend(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.detector_backend_available", lambda: False)
    monkeypatch.setattr("scene_worker.runtime.tracker_backend_available", lambda: False)
    monkeypatch.setattr("scene_worker.runtime.segmenter_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "person_detect" not in capabilities
    assert "person_track" not in capabilities

def test_cpu_worker_never_advertises_real_person_jobs_even_with_backends(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.detector_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.tracker_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})
    assert capabilities == ["cpu"]

def test_python_cpu_child_disables_cuda():
    env = child_environment(SimpleNamespace(), worker_id="python-inference-worker-cpu", gpu_id="cpu")

    assert env["CUDA_VISIBLE_DEVICES"] == ""

def test_python_gpu_child_selects_cuda_device():
    env = child_environment(SimpleNamespace(), worker_id="worker-gpu-0", gpu_id="0")

    assert env["CUDA_VISIBLE_DEVICES"] == "0"

def test_select_torch_device_uses_assigned_gpu_when_multiple_cuda_devices_are_visible():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 2

        class backends:
            mps = None

    assert select_torch_device(Torch, "1") == "cuda:1"

def test_select_torch_device_uses_visible_cuda_default_when_child_process_is_narrowed():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

        class backends:
            mps = None

    assert select_torch_device(Torch, "1") == "cuda"

def test_gpu_worker_fails_fast_when_torch_cuda_is_unavailable():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        require_inference_backend_for_gpu_worker(Torch, "0")

def test_auto_gpu_worker_fails_fast_when_inference_backend_is_unavailable():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            mps = None

    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        require_inference_backend_for_gpu_worker(Torch, "auto")

def test_mps_worker_can_advertise_generation_capabilities(monkeypatch):
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return True

    monkeypatch.setattr("scene_worker.image_adapters.importlib.import_module", lambda name: Torch if name == "torch" else None)

    capabilities = worker_capabilities({"id": "mps", "name": "Apple GPU", "capabilities": ["placeholder", "gpu"]})

    assert "image_generate" in capabilities
    assert "video_generate" in capabilities

def test_gpu_worker_accepts_mps_inference_backend():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return True

    require_inference_backend_for_gpu_worker(Torch, "mps")

def test_friendly_failure_identifies_gpu_oom():
    message, error = friendly_failure("Image generation", RuntimeError("CUDA error: out of memory"))

    assert message == "Image generation failed because the GPU ran out of memory."
    assert "lower resolution" in error
    assert "Technical detail" in error

def test_friendly_failure_identifies_cpu_only_torch_worker():
    message, error = friendly_failure("Image generation", RuntimeError("CUDA-enabled PyTorch is not available."))

    assert message == "Image generation failed because the worker is missing CUDA-enabled PyTorch."
    assert "Rebuild the worker image" in error
    assert "Technical detail" in error

def test_friendly_failure_identifies_missing_model_files():
    message, error = friendly_failure("Image generation", RuntimeError("Repository not found: owner/model"))

    assert message == "Image generation failed because required model files were not available."
    assert "Model Manager" in error
    assert "Rust utility worker" in error
    assert "HF_TOKEN" in error

def test_friendly_failure_identifies_missing_diffusers_model_index():
    message, error = friendly_failure(
        "Video generation",
        RuntimeError(
            "404 Client Error. Entry Not Found for url: "
            "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/model_index.json."
        ),
    )

    assert message == "Video generation failed because required model files were not available."
    assert "model_index.json" in error
    assert "Technical detail" in error

def test_friendly_failure_identifies_missing_peft_backend():
    message, error = friendly_failure(
        "Image generation",
        RuntimeError("LoRA style requires the PEFT backend for z_image_diffusers."),
    )

    assert message == "Image generation failed because the selected preset or LoRA needs PEFT support."
    assert "pip install -r apps/worker/requirements.txt" in error
    assert "docker compose build worker --no-cache" in error
    assert "Technical detail" in error

def test_friendly_failure_identifies_missing_sentencepiece_backend():
    message, error = friendly_failure(
        "Video generation",
        RuntimeError(
            "The component <class 'transformers.models.t5.tokenization_t5."
            "_LazyModule.__getattr__.<locals>.Placeholder'> of <class "
            "'diffusers.pipelines.ltx.pipeline_ltx_image2video.LTXImageToVideoPipeline'> "
            "cannot be loaded as it does not seem to have any of the loading methods defined."
        ),
    )

    assert message == "Video generation failed because the worker is missing a tokenizer backend."
    assert "tokenizer support libraries" in error
    assert "pip install -r apps/worker/requirements.txt" in error
    assert "docker compose build worker --no-cache" in error
    assert "Technical detail" in error

def test_friendly_failure_identifies_missing_protobuf_backend():
    message, error = friendly_failure(
        "Training captioning",
        RuntimeError("requires the protobuf library but it was not found in your environment"),
    )

    assert message == "Training captioning failed because the worker is missing a tokenizer backend."
    assert "tokenizer support libraries" in error
    assert "pip install -r apps/worker/requirements.txt" in error

def test_friendly_failure_identifies_disk_full_by_message():
    message, error = friendly_failure(
        "LoRA training", RuntimeError("OSError: [Errno 28] No space left on device")
    )

    assert message == "LoRA training failed because the disk ran out of space."
    assert "Free up disk space" in error
    assert "Technical detail" in error

def test_friendly_failure_identifies_disk_full_by_oserror_errno():
    message, error = friendly_failure(
        "LoRA training", OSError(28, "No space left on device")
    )

    assert message == "LoRA training failed because the disk ran out of space."
    assert "Free up disk space" in error

def test_is_cuda_oom_detects_oom_by_type_and_message():
    class OutOfMemoryError(RuntimeError):
        pass

    assert is_cuda_oom(OutOfMemoryError("allocator failed"))
    assert is_cuda_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_cuda_oom(RuntimeError("some other failure"))

def test_worker_check_reports_inference_sidecar_capabilities(monkeypatch):
    events = []
    monkeypatch.setattr("scene_worker.runtime.emit", events.append)
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.detector_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.tracker_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.segmenter_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.pose_detector_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.kps_extractor_backend_available", lambda: True)
    monkeypatch.setattr(
        "scene_worker.runtime.discover_gpu",
        lambda _gpu_id: {"id": "0", "name": "GPU 0", "capabilities": ["gpu"]},
    )

    run_check(SimpleNamespace(worker_id="worker-1", gpu_id="0"))

    assert events[0]["event"] == "worker_check"
    assert events[0]["jobTypes"] == [
        "image_generate",
        "image_edit",
        "video_generate",
        "video_extend",
        "video_bridge",
        "person_replace",
        "person_detect",
        "person_track",
        "pose_detect",
        # sc-4433: SCRFD 5-point landmark extraction (Key Point Library).
        "kps_extract",
        "lora_train",
        "training_caption",
        # sc-1635: VQA + interleave are advertised and dispatched, so the check
        # must report them too.
        "image_vqa",
        "image_interleave",
        # sc-2041: prompt refinement is dispatched on every Python worker.
        "prompt_refine",
        # sc-2431: standalone image upscale (Image Editor).
        "image_upscale",
        # sc-2438: standalone tile-ControlNet detail refine (Image Editor).
        "image_detail",
    ]
    assert events[0]["supportedJobTypes"] == events[0]["jobTypes"]

def test_main_check_exits_without_api_loop(monkeypatch):
    calls = []
    monkeypatch.setattr("scene_worker.runtime.run_check", lambda settings: calls.append(settings.worker_id))

    main(["--check"])

    assert calls == ["worker-local-0"]

def test_heartbeat_loaded_models_are_not_sent_as_current_job(monkeypatch):
    class Api:
        def __init__(self):
            self.path = None
            self.payload = None

        def post(self, path, payload):
            self.path = path
            self.payload = payload
            return {}

    class Settings:
        worker_id = "worker-1"
        gpu_id = "0"

    monkeypatch.setattr("scene_worker.runtime.gpu_utilization", lambda _gpu_id: None)
    api = Api()
    heartbeat(api, Settings(), "idle", loaded_models=["model-a"])

    assert api.path == "/api/v1/workers/worker-1/heartbeat"
    assert api.payload == {"status": "idle", "currentJobId": None, "loadedModels": ["model-a"]}

def test_heartbeat_reports_gpu_utilization_when_available(monkeypatch):
    class Api:
        def __init__(self):
            self.payload = None

        def post(self, _path, payload):
            self.payload = payload
            return {}

    class Settings:
        worker_id = "worker-1"
        gpu_id = "0"

    monkeypatch.setattr(
        "scene_worker.runtime.gpu_utilization",
        lambda _gpu_id: {"memoryTotalMb": 24576, "memoryUsedMb": 4096, "memoryFreeMb": 20480, "gpuLoadPercent": 12},
    )

    api = Api()
    heartbeat(api, Settings(), "idle")

    assert api.payload["utilization"] == {
        "memoryTotalMb": 24576,
        "memoryUsedMb": 4096,
        "memoryFreeMb": 20480,
        "gpuLoadPercent": 12,
    }

def test_keepalive_heartbeat_reports_current_loaded_models_each_tick(monkeypatch):
    calls = []
    models_by_tick = [["model-a"], ["model-b"]]
    loaded_model_calls = []

    class StopAfterTwoHeartbeats:
        def __init__(self):
            self.calls = 0

        def wait(self, _interval):
            self.calls += 1
            return self.calls > 2

    class Settings:
        worker_id = "worker-1"
        heartbeat_seconds = 1

    def capture_heartbeat(api, settings, status, current_job_id=None, loaded_models=None):
        calls.append(
            {
                "api": api,
                "settings": settings,
                "status": status,
                "current_job_id": current_job_id,
                "loaded_models": loaded_models,
            }
        )

    monkeypatch.setattr("scene_worker.runtime.heartbeat", capture_heartbeat)

    def loaded_models():
        loaded_model_calls.append("tick")
        return models_by_tick[len(loaded_model_calls) - 1]

    keep_job_alive(
        api=object(),
        settings=Settings(),
        job_id="job-1",
        status="busy",
        stop_event=StopAfterTwoHeartbeats(),
        loaded_models=loaded_models,
    )

    assert [call["status"] for call in calls] == ["busy", "busy"]
    assert [call["current_job_id"] for call in calls] == ["job-1", "job-1"]
    assert [call["loaded_models"] for call in calls] == [["model-a"], ["model-b"]]
    assert len(loaded_model_calls) == 2

def test_loaded_model_resolution_failure_keeps_heartbeat_alive(monkeypatch):
    events = []

    def failing_loaded_models():
        raise RuntimeError("cache is mid-load")

    monkeypatch.setattr("scene_worker.runtime.emit", events.append)

    assert resolve_loaded_models(failing_loaded_models, job_id="job-1") == []
    assert events == [
        {
            "event": "loaded_models_failed",
            "error": "cache is mid-load",
            "jobId": "job-1",
            "reportedAt": events[0]["reportedAt"],
        }
    ]

def test_cancel_monitor_caches_cancel_state(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    api = _CancelPollApi(cancel_requested=True)
    settings = SimpleNamespace(worker_id="worker-1", force_cancel_seconds=0)
    monitor = JobCancelMonitor(api, settings, "job-1", poll_interval=0.05, force_cancel_seconds=0)
    monitor.start()
    try:
        assert _wait_until(monitor.requested)
    finally:
        monitor.stop()

def test_cancel_monitor_does_not_escalate_without_cancel(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    exits = []
    monkeypatch.setattr("scene_worker.runtime.os._exit", lambda code: exits.append(code))
    api = _CancelPollApi(cancel_requested=False)
    settings = SimpleNamespace(worker_id="worker-1", force_cancel_seconds=0.1)
    monitor = JobCancelMonitor(api, settings, "job-1", poll_interval=0.02, force_cancel_seconds=0.1)
    monitor.start()
    try:
        time.sleep(0.3)
        assert exits == []
        assert monitor.requested() is False
    finally:
        monitor.stop()

def test_cancel_monitor_force_terminates_after_deadline(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    exited = threading.Event()
    exits = []

    def fake_exit(code):
        exits.append(code)
        exited.set()

    monkeypatch.setattr("scene_worker.runtime.os._exit", fake_exit)
    api = _CancelPollApi(cancel_requested=True)
    settings = SimpleNamespace(worker_id="worker-1", force_cancel_seconds=0.15)
    monitor = JobCancelMonitor(api, settings, "job-9", poll_interval=0.02, force_cancel_seconds=0.15)
    monitor.start()
    try:
        assert exited.wait(timeout=2.0)
    finally:
        monitor.stop()
    assert exits == [FORCED_CANCEL_EXIT_CODE]
    # The job was marked canceled before the (mocked) process exit, so the UI
    # resolves immediately instead of waiting for a stale-worker reaper.
    terminal = api.progress[-1]
    assert terminal["status"] == "canceled"
    assert terminal["stage"] == "canceled"
    assert "force-stopped" in terminal["message"].lower()

def test_cancel_monitor_runs_on_force_terminate_hook_before_exit(monkeypatch):
    # The force-cancel backstop reaps tracked temp files via on_force_terminate
    # before the hard os._exit (which skips cooperative adapter cleanup) — the
    # image/training runners rely on this to drop their scratch dirs (sc-1719).
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    events: list[tuple[str, object]] = []
    exited = threading.Event()

    def fake_exit(code):
        events.append(("exit", code))
        exited.set()

    monkeypatch.setattr("scene_worker.runtime.os._exit", fake_exit)
    api = _CancelPollApi(cancel_requested=True)
    settings = SimpleNamespace(worker_id="worker-1", force_cancel_seconds=0.1)
    monitor = JobCancelMonitor(
        api,
        settings,
        "job-hook",
        poll_interval=0.02,
        force_cancel_seconds=0.1,
        on_force_terminate=lambda: events.append(("hook", None)),
    )
    monitor.start()
    try:
        assert exited.wait(timeout=2.0)
    finally:
        monitor.stop()
    assert ("hook", None) in events
    # The reap must run before the process exits.
    assert events.index(("hook", None)) < events.index(("exit", FORCED_CANCEL_EXIT_CODE))

def test_cancel_monitor_skips_escalation_when_stopped(monkeypatch):
    # A cooperative cancel that finishes before the deadline must not force-kill.
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    exits = []
    monkeypatch.setattr("scene_worker.runtime.os._exit", lambda code: exits.append(code))
    api = _CancelPollApi(cancel_requested=True)
    settings = SimpleNamespace(worker_id="worker-1", force_cancel_seconds=5)
    monitor = JobCancelMonitor(api, settings, "job-2", poll_interval=0.02, force_cancel_seconds=5)
    monitor.start()
    assert _wait_until(monitor.requested)  # cancel observed
    monitor.stop()  # cooperative path finished the job before the deadline
    time.sleep(0.1)
    assert exits == []

def test_cancel_step_callback_sets_interrupt_when_canceled():
    pipe = _FakeStepPipe()
    callback = cancel_step_callback(pipe, lambda: True)
    assert callback is not None
    returned = callback(pipe, 0, None, {"latents": "x"})
    assert returned == {"latents": "x"}  # callback must pass kwargs through
    assert pipe._interrupt is True

def test_cancel_step_callback_leaves_interrupt_when_not_canceled():
    pipe = _FakeStepPipe()
    callback = cancel_step_callback(pipe, lambda: False)
    assert callback is not None
    callback(pipe, 0, None, {})
    assert pipe._interrupt is False

def test_cancel_step_callback_none_without_predicate():
    assert cancel_step_callback(_FakeStepPipe(), None) is None

def test_cancel_step_callback_none_when_pipe_lacks_support():
    assert cancel_step_callback(_NoCallbackPipe(), lambda: True) is None

def test_worker_settings_force_cancel_seconds_default(monkeypatch):
    monkeypatch.delenv("SCENEWORKS_WORKER_FORCE_CANCEL_SECONDS", raising=False)
    from scene_worker.settings import WorkerSettings

    assert WorkerSettings().force_cancel_seconds == 30

def test_worker_settings_force_cancel_seconds_env_override(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_WORKER_FORCE_CANCEL_SECONDS", "5")
    from scene_worker.settings import WorkerSettings

    assert WorkerSettings().force_cancel_seconds == 5

def test_repo_slug_functions_match_cross_language_contract():
    # sc-1667: the repo->directory slug string ops are duplicated in the Rust
    # API, the Rust CPU worker, and here. They must stay byte-identical so a
    # repo resolves to the same managed dir / HF cache dir in every language.
    # This fixture is the shared contract; matching Rust tests live in
    # apps/rust-api and crates/sceneworks-worker.
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "rust_migration_contracts" / "repo_slugs.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
    assert cases, "repo_slugs fixture has no cases"
    for case in cases:
        repo = case["repo"]
        assert safe_download_dir(repo) == case["safeDownloadDir"], f"safe_download_dir drift for {repo!r}"
        assert safe_repo_dir_name(repo) == case["safeRepoDirName"], f"safe_repo_dir_name drift for {repo!r}"
