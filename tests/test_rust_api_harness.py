import os
from pathlib import Path

from rust_api_harness import enable_api_listening_log, listening_url_from_log, spawn_env


def test_listening_url_from_text_and_json_tracing_output():
    assert (
        listening_url_from_log(
            "INFO sceneworks_rust_api: SceneWorks API listening "
            "event=\"api_listening\" address=127.0.0.1:43125"
        )
        == "http://127.0.0.1:43125"
    )
    assert (
        listening_url_from_log(
            '{"event":"api_listening","address":"127.0.0.1:51234"}'
        )
        == "http://127.0.0.1:51234"
    )


def test_api_listening_event_survives_restrictive_ambient_logging():
    env = {"RUST_LOG": "warn"}
    enable_api_listening_log(env)
    assert env["RUST_LOG"] == "warn,sceneworks_rust_api::server=info"


def test_spawn_env_pins_the_huggingface_cache_inside_the_test_root(monkeypatch, tmp_path):
    """A spawned binary must not resolve its Hugging Face cache from the ambient
    environment. `huggingface_hub_cache_dir` reads HF_HUB_CACHE /
    HUGGINGFACE_HUB_CACHE / HF_HOME AHEAD of SCENEWORKS_DATA_DIR, so an inherited
    value silently re-points the model catalog's install-state sweep at the HOST's
    cache: "installed" then reflects whatever the developer downloaded rather than
    what the fixture staged. It is green on an empty CI runner and red only on a real
    workstation -- measured, a 22.6s sweep that blew the e2e LoRA test's 5s request
    timeout, against 0.43s pinned."""
    host_cache = tmp_path / "host-cache"
    for inherited in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.setenv(inherited, str(host_cache))

    root = tmp_path / "root"
    env = spawn_env(root)

    # Positive pin, not merely an unset: with all three cleared the binary's
    # `ensure_default_huggingface_home()` defaults HF_HOME to ~/.cache/huggingface,
    # which is still a shared host directory. This value is the same
    # `<data_dir>/cache/huggingface/hub` the binary falls back to on a clean env.
    assert env["HF_HUB_CACHE"] == str(root / "data" / "cache" / "huggingface" / "hub")
    # The lower-precedence vars must be absent rather than overridden -- the worker
    # and the hf tooling it shells out to read HF_HOME directly.
    assert "HUGGINGFACE_HUB_CACHE" not in env
    assert "HF_HOME" not in env
    # Still the ambient environment otherwise: the spawn needs PATH, the binary-path
    # exports, and the rest of the developer's/CI's env.
    assert env.get("PATH") == os.environ.get("PATH")


def test_live_binary_spawn_sites_do_not_inherit_the_ambient_environment():
    """Both live-binary harnesses must build their spawn env through `spawn_env`.

    This is the drift that produced the bug: the parity harness pinned the Hugging
    Face cache (sc-19708, so a golden could not record WHOSE machine ran it) while
    the e2e harness kept a bare ambient copy, leaving one suite hermetic and the
    other host-dependent. A new spawn site that reaches for the ambient environment
    reintroduces it, and CI -- with an empty HF cache -- cannot see that happen."""
    for module in ("test_rust_api_worker_smoke.py", "test_rust_api_contract_snapshots.py"):
        source = (Path(__file__).parent / module).read_text(encoding="utf-8")
        # Held as booleans so a failure reports the claim, not a 200-line dump of the
        # module under inspection.
        inherits_ambient_env = "os.environ.copy()" in source
        spawns_through_helper = "spawn_env(" in source
        assert not inherits_ambient_env, (
            f"{module} builds a spawn environment from the ambient env; use "
            "rust_api_harness.spawn_env(root) so the Hugging Face cache stays pinned"
        )
        assert spawns_through_helper, f"{module} no longer spawns through spawn_env"
