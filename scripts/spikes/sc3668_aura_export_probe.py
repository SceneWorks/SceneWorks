#!/usr/bin/env python
"""sc-3668 — AuraSR (GigaGAN UnetUpsampler) ONNX/CoreML viability probe.

Mirrors the sc-3489 Real-ESRGAN path-selection spike, but for the *second*
upscaler engine (engine="aura-sr", fal/AuraSR-v2). The sc-3489 ONNX+CoreML path
worked because RRDBNet is a pure static-weight conv stack that folds onto the
CoreML EP. AuraSR's defining op (AdaptiveConv2DMod) computes conv weights at
RUNTIME from a style vector and runs them through F.conv2d with a *dynamic
(non-initializer) weight input*. CoreML's conv requires static weights, so those
nodes fall back to CPU.

This probe measures, empirically:
  1. Does the UnetUpsampler export to ONNX at all?
  2. How many Conv nodes have a DYNAMIC weight (CoreML-ineligible) vs a STATIC
     initializer weight (CoreML-eligible)? -> the decisive structural metric.
  3. CoreML EP vs CPU EP latency for one 64x64 -> 256x256 tile.
  4. Tiles-per-image blowup (AuraSR tiles at 64px vs Real-ESRGAN at 512px).

Run with the SceneWorks app venv (has torch + onnx + onnxruntime + aura_sr):
  "$VENV/bin/python" scripts/spikes/sc3668_aura_export_probe.py
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

import aura_sr

SNAP = Path(
    "~/.cache/huggingface/hub/models--fal--AuraSR-v2/"
    "snapshots/ff452185a7c8b51206dd62c21c292e7baad5c3a3"
)
OUT = Path("/tmp/sc3668")
OUT.mkdir(parents=True, exist_ok=True)
ONNX_PATH = OUT / "aura_upsampler_b1_64.onnx"

INPUT_TILE = 64  # config.json input_image_size
SCALE = 4
STYLE_DIM = 128  # style_network.dim_in


def load_model() -> aura_sr.AuraSR:
    # AuraSR.from_pretrained hardcodes device="cuda"; build on CPU manually.
    from safetensors.torch import load_file

    config = json.loads((SNAP / "config.json").read_text())
    model = aura_sr.AuraSR(config, device="cpu")
    checkpoint = load_file(str(SNAP / "model.safetensors"))
    model.upsampler.load_state_dict(checkpoint, strict=True)
    model.upsampler.eval()
    # The TorchScript tracer chokes on AdaptiveConv2DMod's shape-dependent dynamic
    # padding (get_same_padding(h, ...)). For every adaptive conv kernel=3/stride=1/
    # dilation=1 -> padding is the constant 1, so pin it (faithful) to let the trace
    # through. (That the stock model needs this patch at all is itself a signal.)
    aura_sr.get_same_padding = lambda *a, **k: 1
    # Make attention explicit (einsum+softmax) instead of flash SDPA so it has a
    # chance of being CoreML-eligible; flash SDPA would fall back anyway.
    n_flash = 0
    for m in model.upsampler.modules():
        if isinstance(m, aura_sr.Attend):
            m.flash = False
            n_flash += 1
    print(f"  set flash=False on {n_flash} Attend modules")
    return model


class ExportWrap(nn.Module):
    """Fixed (lowres, noise) -> rgb signature for ONNX export."""

    def __init__(self, upsampler: nn.Module):
        super().__init__()
        self.m = upsampler

    def forward(self, lowres: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.m(lowres_image=lowres, noise=noise)


@contextlib.contextmanager
def deterministic_forward():
    """Zero the internal noise_aug (randn_like) so torch ref + ONNX are comparable
    and so the exported graph has no RandomNormal node."""
    real = torch.randn_like

    def zeros_like(x, *a, **k):
        return torch.zeros_like(x)

    torch.randn_like = zeros_like
    try:
        yield
    finally:
        torch.randn_like = real


def export_onnx(model: aura_sr.AuraSR) -> None:
    wrap = ExportWrap(model.upsampler).eval()
    lowres = torch.rand(1, 3, INPUT_TILE, INPUT_TILE)
    noise = torch.randn(1, STYLE_DIM)
    print(f"exporting ONNX -> {ONNX_PATH} (this loads/traces a 617M-param UNet, be patient)...")
    t0 = time.time()
    with deterministic_forward(), torch.no_grad():
        torch.onnx.export(
            wrap,
            (lowres, noise),
            str(ONNX_PATH),
            input_names=["lowres", "noise"],
            output_names=["rgb"],
            opset_version=17,
            dynamo=False,
        )
    print(f"  export OK in {time.time()-t0:.1f}s, file={ONNX_PATH.stat().st_size/1e6:.0f} MB")


def analyze_conv_weights() -> dict:
    m = onnx.load(str(ONNX_PATH))
    g = m.graph
    init_names = {i.name for i in g.initializer}
    convs_static = 0
    convs_dynamic = 0
    op_counts: dict[str, int] = {}
    for node in g.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        if node.op_type == "Conv":
            w = node.input[1] if len(node.input) > 1 else None
            if w in init_names:
                convs_static += 1
            else:
                convs_dynamic += 1
    return {
        "convs_static_weight": convs_static,
        "convs_dynamic_weight": convs_dynamic,
        "total_nodes": len(g.node),
        "op_counts": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
    }


def ort_session(providers):
    so = ort.SessionOptions()
    so.log_severity_level = 1  # capture EP partition info on stderr
    return ort.InferenceSession(str(ONNX_PATH), sess_options=so, providers=providers)


def faithfulness_and_latency(model: aura_sr.AuraSR) -> dict:
    lowres = torch.rand(1, 3, INPUT_TILE, INPUT_TILE)
    noise = torch.randn(1, STYLE_DIM)
    with deterministic_forward(), torch.no_grad():
        ref = model.upsampler(lowres_image=lowres, noise=noise).clamp(0, 1).cpu().numpy()

    feeds = {"lowres": lowres.numpy(), "noise": noise.numpy()}
    res: dict = {}

    # CPU EP — faithfulness + latency
    sess_cpu = ort_session(["CPUExecutionProvider"])
    out_cpu = sess_cpu.run(None, feeds)[0]
    res["cpu_max_abs_diff_vs_torch"] = float(np.max(np.abs(out_cpu - ref)))
    n = 3
    t0 = time.time()
    for _ in range(n):
        sess_cpu.run(None, feeds)
    res["cpu_latency_s_per_tile"] = (time.time() - t0) / n

    # CoreML EP — node placement + latency
    avail = ort.get_available_providers()
    res["available_providers"] = avail
    if "CoreMLExecutionProvider" in avail:
        try:
            sess_cml = ort_session([("CoreMLExecutionProvider", {}), "CPUExecutionProvider"])
            res["coreml_providers_in_session"] = sess_cml.get_providers()
            sess_cml.run(None, feeds)  # warmup / compile
            t0 = time.time()
            for _ in range(n):
                sess_cml.run(None, feeds)
            res["coreml_latency_s_per_tile"] = (time.time() - t0) / n
        except Exception as e:  # noqa: BLE001
            res["coreml_error"] = f"{type(e).__name__}: {e}"
    else:
        res["coreml_error"] = "CoreMLExecutionProvider not available in this build"

    # torch MPS latency (the path AuraSR runs on today)
    if torch.backends.mps.is_available():
        ups = model.upsampler.to("mps")
        lr = lowres.to("mps")
        nz = noise.to("mps")
        with deterministic_forward(), torch.no_grad():
            ups(lowres_image=lr, noise=nz)  # warmup
            torch.mps.synchronize()
            t0 = time.time()
            for _ in range(n):
                ups(lowres_image=lr, noise=nz)
            torch.mps.synchronize()
        res["torch_mps_latency_s_per_tile"] = (time.time() - t0) / n
        model.upsampler.to("cpu")
    return res


def tile_blowup() -> dict:
    def n_tiles(side: int, tile: int) -> int:
        return math.ceil(side / tile) ** 2

    return {
        "768x768": {"aura_64px": n_tiles(768, 64), "esrgan_512px": n_tiles(768, 512)},
        "1024x1024": {"aura_64px": n_tiles(1024, 64), "esrgan_512px": n_tiles(1024, 512)},
        "note": "AuraSR overlapped mode runs TWO passes -> double the aura tile count.",
    }


def main() -> None:
    print("=== sc-3668 AuraSR ONNX/CoreML viability probe ===")
    print(f"onnxruntime {ort.__version__}, torch {torch.__version__}")
    model = load_model()
    export_onnx(model)
    weights = analyze_conv_weights()
    runtime = faithfulness_and_latency(model)
    blow = tile_blowup()
    report = {"conv_weight_analysis": weights, "runtime": runtime, "tile_blowup": blow}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\n=== REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
