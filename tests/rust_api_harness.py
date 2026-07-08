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
import socket
import subprocess
import time

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


def free_port() -> int:
    """Bind an ephemeral loopback port and hand back its number for the API."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def wait_for_health(
    base_url: str,
    process: subprocess.Popen,
    runtime: str = "rust",
    timeout_seconds: float = 30.0,
) -> None:
    """Poll ``/api/v1/health`` until the spawned API answers 200, surfacing an
    early exit (with captured stderr) instead of hanging until the deadline."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(
                f"{runtime} API exited early with code {process.returncode}: {stderr}"
            )
        try:
            response = httpx.get(f"{base_url}/api/v1/health", timeout=1)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.25)
    raise AssertionError(
        f"{runtime} API did not become healthy within {timeout_seconds:.0f}s: {last_error}"
    )
