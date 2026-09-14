"""Shared helpers for the live Rust API test files (sc-8934 / F-132).

`test_rust_api_worker_smoke.py` (e2e) and `test_rust_api_contract_snapshots.py`
(parity) both spawn the real `sceneworks-rust-api` binary and drive it over HTTP.
They previously kept copy-pasted `free_port` / `wait_for_health` / PNG / safetensors
builders that had already drifted -- one copy carried a corrupt `PNG_1X1` (a stray
`\\x01` before the IHDR CRC made it undecodable). This module is the single source
of truth for those fixtures so they can't diverge again.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import httpx

# A minimal, fully valid 1x1 8-bit RGB PNG. Every chunk CRC is correct
# (IHDR 907753de, IDAT 33129514, IEND ae426082), unlike the corrupt copy this
# replaces. The Rust API does not currently decode uploaded images, but pinning
# the upload contract to a decodable PNG keeps the fixture honest if validation
# is ever added.
PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\xdac\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfe3\x12\x95\x14\x00\x00\x00\x00IEND\xaeB`\x82"
)


LISTENING_ADDRESS_RE = re.compile(
    r"""api_listening.*?address(?:\\?["']?\s*[:=]\s*\\?["']?)"""
    r"""(?P<address>\[[0-9a-fA-F:]+\]|[0-9.]+):(?P<port>[0-9]+)"""
)


def listening_url_from_log(log: str) -> str | None:
    """Extract the API's authoritative port-0 bound address from tracing output."""
    match = LISTENING_ADDRESS_RE.search(log)
    if not match:
        return None
    host = match.group("address")
    if host in {"0.0.0.0", "[::]"}:
        host = "127.0.0.1"
    return f"http://{host}:{match.group('port')}"


def enable_api_listening_log(env: dict[str, str]) -> None:
    """Keep the authoritative port-0 event visible under restrictive ambient logs."""
    directive = "sceneworks_rust_api::server=info"
    ambient = env.get("RUST_LOG", "").strip()
    env["RUST_LOG"] = f"{ambient},{directive}" if ambient else directive


def safetensors_bytes() -> bytes:
    """A valid safetensors payload: 8-byte little-endian header length + a JSON
    header + a small tensor body. The LoRA import path inspects the header for
    architecture detection, so a stub like ``b"lora"`` is rejected with an
    invalid-header 400. The trailing body lets copy round-trip assertions compare
    against the exact bytes."""
    header = json.dumps({"__metadata__": {"format": "pt"}}, separators=(",", ":")).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + b"tensor-bytes"


# Historical alias used by the e2e smoke file; kept so both files share one impl.
minimal_safetensors = safetensors_bytes


class SpawnedProcess:
    """A spawned API/worker binary whose pipes can't deadlock the test (F-042).

    The children emit `tracing` output the whole time a test drives them. When a
    spawn keeps ``stdout``/``stderr`` as ``PIPE`` and nobody reads them until the
    child exits, the child blocks on ``write`` the moment either OS pipe buffer
    fills (~64 KB) -- the test then hangs until its deadline and is mis-reported
    as "job did not reach status" rather than the real cause.

    This wrapper drains both ``stdout`` and ``stderr`` on background daemon
    threads into one in-memory buffer, so the child can always make progress.
    SceneWorks tracing writes to stdout, while compiler/process failures use
    stderr; port discovery and diagnostics therefore need both streams.

    It delegates the small slice of the ``Popen`` surface the tests use
    (``poll``/``returncode``/``terminate``/``wait``/``kill``) so it drops in for
    the old ``subprocess.Popen`` handles.
    """

    def __init__(self, popen: subprocess.Popen) -> None:
        self._popen = popen
        self._chunks: list[str] = []
        self._lock = threading.Lock()
        self._threads = [
            threading.Thread(
                target=self._drain,
                args=(stream,),
                name=f"spawned-{name}-drain",
                daemon=True,
            )
            for name, stream in (
                ("stdout", self._popen.stdout),
                ("stderr", self._popen.stderr),
            )
        ]
        for thread in self._threads:
            thread.start()

    def _drain(self, stream) -> None:
        if stream is None:
            return
        # Iterating a text-mode pipe yields lines until EOF (child exit/close),
        # so this loop drains continuously and ends on its own.
        for line in stream:
            with self._lock:
                self._chunks.append(line)

    def stderr_text(self) -> str:
        """Return the child's captured output. If the child has already exited,
        wait briefly for both drain threads to consume their final buffered bytes."""
        if self._popen.poll() is not None:
            for thread in self._threads:
                thread.join(timeout=5)
        with self._lock:
            return "".join(self._chunks)

    def wait_for_listening_url(self, timeout_seconds: float = 30.0) -> str:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if url := listening_url_from_log(self.stderr_text()):
                return url
            if self.poll() is not None:
                raise AssertionError(
                    f"Rust API exited early with code {self.returncode}: {self.stderr_text()}"
                )
            time.sleep(0.05)
        raise AssertionError(
            "Rust API did not report its bound address within "
            f"{timeout_seconds:.0f}s: {self.stderr_text()}"
        )

    # -- Popen delegation (only what the tests touch) ------------------------
    def poll(self) -> int | None:
        return self._popen.poll()

    @property
    def returncode(self) -> int | None:
        return self._popen.returncode

    def terminate(self) -> None:
        self._popen.terminate()

    def kill(self) -> None:
        self._popen.kill()

    def wait(self, timeout: float | None = None) -> int:
        return self._popen.wait(timeout=timeout)


def spawn_env(root: Path) -> dict[str, str]:
    """The base environment for a spawned API/worker binary, with the Hugging Face
    cache pinned inside ``root``.

    A live-binary spawn inherits the developer's ambient environment, and a real
    workstation carries `HF_HOME` (or `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE`)
    pointing at a multi-terabyte Hugging Face cache. The spawned binary resolves
    those three vars AHEAD of `SCENEWORKS_DATA_DIR`
    (`sceneworks_core::hf_home::huggingface_hub_cache_dir`), so without a pin the
    model catalog's install-state sweep probes the HOST's cache instead of the
    test's temp data dir -- install state stops being a property of the fixture and
    becomes a property of whatever the developer happens to have downloaded.

    Unsetting the three is NOT sufficient: `ensure_default_huggingface_home()` then
    defaults `HF_HOME` to the OS Hugging Face home (`~/.cache/huggingface`), which
    is still a shared, host-owned directory. Pinning `HF_HUB_CACHE` (top precedence)
    at `<root>/data/cache/huggingface/hub` instead reproduces exactly the
    `<data_dir>/cache/huggingface/hub` fallback the binary uses under a clean
    environment, so the spawn is hermetic rather than merely env-less.

    CI runners have an empty HF cache, so an unpinned spawn is green there and red
    only on a developer box. Measured on Windows against a 2.2 TB ambient cache, the
    first `POST /api/v1/image/jobs` in the e2e LoRA test spent 22.6s in that sweep
    and blew the request's 5s timeout; pinned, the same call takes 0.43s. The parity
    harness has pinned it since sc-19708 (an unpinned golden recorded WHOSE machine
    ran it) while the e2e harness did not -- this is the one implementation both use
    so the isolation cannot drift back out of either.
    """
    env = os.environ.copy()
    env["HF_HUB_CACHE"] = str(Path(root) / "data" / "cache" / "huggingface" / "hub")
    for inherited in ("HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        env.pop(inherited, None)
    return env


def spawn_process(command, *, cwd, env) -> SpawnedProcess:
    """Spawn an API/worker binary with non-blocking pipe handling (F-042).

    ``stdout`` and ``stderr`` are drained by background threads -- neither can
    fill and block the child, so a chatty binary can no longer hang the test. Returns a
    :class:`SpawnedProcess` that stands in for the old raw ``Popen`` handle."""
    popen = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return SpawnedProcess(popen)


def wait_for_health(
    base_url: str,
    process: SpawnedProcess,
    runtime: str = "rust",
    timeout_seconds: float = 30.0,
) -> None:
    """Poll ``/api/v1/health`` until the spawned API answers 200, surfacing an
    early exit (with captured stderr) instead of hanging until the deadline."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr_text()
            raise AssertionError(
                f"{runtime} API exited early with code {process.returncode}: {stderr}"
            )
        try:
            response = httpx.get(f"{base_url}/api/v1/health", timeout=1)
            if response.status_code == 200:
                body = response.json()
                readiness = body.get("readiness")
                if body.get("status") == "ok" and (
                    readiness is None or readiness.get("status") == "ready"
                ):
                    return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.25)
    raise AssertionError(
        f"{runtime} API did not become healthy within {timeout_seconds:.0f}s: {last_error}"
    )
