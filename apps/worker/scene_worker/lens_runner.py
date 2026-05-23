"""Out-of-process Lens / Lens-Turbo image generation runner.

Executed by `scene_worker.image_adapters.LensTurboAdapter` via the dedicated
Lens sidecar venv (/opt/lens-venv) — NOT the main worker venv. Lens needs
transformers 5.x + diffusers 0.38, which conflict with the main worker stack
(transformers 4.x) that native LTX-2.3 requires, so Lens runs isolated here.

Contract: argv[1] is a path to a JSON spec; the runner writes one PNG per seed
into spec["outDir"] and prints a single result JSON object to stdout:
    {"images": ["<outDir>/lens_0000.png", ...]}
Progress and diagnostics go to stderr (captured into the worker log). A non-zero
exit code with an "error" JSON on stdout signals failure to the adapter.

The runner is intentionally dependency-light at module scope (json/sys/os/path)
so a bad spec fails cleanly before importing torch.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _log(message: str) -> None:
    sys.stderr.write(f"[lens_runner] {message}\n")
    sys.stderr.flush()


def _apply_loras(transformer, loras) -> None:
    """Inject + scale trained `lens` LoRAs on the transformer (PeftAdapterMixin).

    Each ``loras`` entry is ``{"path", "weight", "name"}``, already resolved to a
    concrete .safetensors file by the adapter. ``save_lora_adapter`` (training
    kernel) and ``load_lora_adapter`` are the symmetric PeftAdapterMixin pair; the
    ``prefix=None`` retry covers builds that saved the adapter without a
    ``transformer.`` key prefix.
    """
    names: list[str] = []
    weights: list[float] = []
    for index, lora in enumerate(loras):
        name = str(lora.get("name") or f"lora_{index}")
        path = str(lora["path"])
        try:
            transformer.load_lora_adapter(path, adapter_name=name)
        except Exception:  # noqa: BLE001 - retry with no key prefix before failing
            transformer.load_lora_adapter(path, adapter_name=name, prefix=None)
        names.append(name)
        try:
            weights.append(float(lora.get("weight", 1.0)))
        except (TypeError, ValueError):
            weights.append(1.0)
    if hasattr(transformer, "set_adapters"):
        transformer.set_adapters(names, weights=weights)
    elif names and hasattr(transformer, "set_adapter"):
        transformer.set_adapter(names[0])


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "lens_runner expects exactly one argument: the spec JSON path"}))
        return 2
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    # The vendored `lens` package lives next to this file in _vendor/. Importing
    # it registers LensPipeline/LensTransformer2DModel/LensGptOssEncoder into the
    # diffusers/transformers namespaces that model_index.json references.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))

    import torch  # noqa: E402  (heavy import deferred until the spec is valid)
    import transformers  # noqa: E402
    from lens import LensGptOssEncoder, LensPipeline  # noqa: E402

    repo = spec["repo"]
    seeds = [int(seed) for seed in spec.get("seeds", [])] or [0]
    out_dir = Path(spec["outDir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"

    requested_device = str(spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Lens sidecar requested a CUDA device but torch.cuda.is_available() is False in the "
            "lens venv. Rebuild the worker image with a CUDA (cu128) torch in /opt/lens-venv."
        )
    if requested_device == "mps":
        # Route the few ops without an MPS kernel (in the mxfp4-dequantized
        # gpt-oss / Flux.2 VAE paths) to CPU instead of erroring. The adapter
        # sets this too via select_torch_device; set it here so a standalone
        # runner invocation is safe on Apple Silicon as well.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }.get(spec.get("dtype"), torch.float32 if requested_device == "cpu" else torch.bfloat16)
    disable_mxfp4 = bool(spec.get("disableMxfp4", False))
    cpu_offload = bool(spec.get("cpuOffload", False))

    _log(f"torch {torch.__version__} transformers {transformers.__version__} device={requested_device} dtype={dtype}")

    text_encoder_kwargs = {"subfolder": "text_encoder", "dtype": dtype}
    mxfp4_config = getattr(transformers, "Mxfp4Config", None)
    if mxfp4_config is not None:
        text_encoder_kwargs["quantization_config"] = mxfp4_config(dequantize=disable_mxfp4)
    text_encoder = LensGptOssEncoder.from_pretrained(repo, **text_encoder_kwargs)
    pipe = LensPipeline.from_pretrained(repo, text_encoder=text_encoder, torch_dtype=dtype)
    if cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(requested_device)
    _log("pipeline loaded")

    # LensPipeline has no LoRA loader mixin, but LensTransformer2DModel inherits
    # diffusers' PeftAdapterMixin, so trained `lens` LoRAs (sc-1587) load directly
    # on the transformer. The adapter resolved each entry to a concrete file path
    # + weight in the main venv; here we only inject and scale them. A LoRA trained
    # on the base microsoft/Lens applies cleanly to Lens-Turbo (same architecture).
    loras = spec.get("loras") or []
    if loras:
        _apply_loras(pipe.transformer, loras)
        _log(f"applied {len(loras)} LoRA(s)")

    generator_device = requested_device if requested_device.startswith("cuda") else "cpu"
    base_resolution = int(spec.get("baseResolution", 1024))
    aspect_ratio = str(spec.get("aspectRatio", "1:1"))
    steps = int(spec.get("numInferenceSteps", 4))
    guidance_scale = float(spec.get("guidanceScale", 1.0))
    prompt = spec.get("prompt", "")
    negative_prompt = spec.get("negativePrompt") or ""

    images: list[str] = []
    for index, seed in enumerate(seeds):
        generator = torch.Generator(generator_device).manual_seed(int(seed))
        kwargs = {
            "prompt": prompt,
            "base_resolution": base_resolution,
            "aspect_ratio": aspect_ratio,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
            "generator": generator,
            "enable_reasoner": False,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        image = pipe(**kwargs).images[0].convert("RGB")
        path = out_dir / f"lens_{index:04d}.png"
        image.save(path, "PNG")
        images.append(str(path))
        _log(f"generated image {index + 1}/{len(seeds)} -> {path}")

    result = {"images": images}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - surface any failure as structured JSON
        import traceback

        traceback.print_exc()
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        # Best-effort: persist the error next to where images would have gone so
        # the adapter can surface it even if stdout was lost.
        try:
            spec_arg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
            out_dir = Path(spec_arg["outDir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass
        print(json.dumps(payload))
        raise SystemExit(1)
