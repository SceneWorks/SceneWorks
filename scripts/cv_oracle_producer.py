#!/usr/bin/env python3
"""Explicit manual-only CPU oracle capture from pinned upstream Ultralytics.

No torch/numpy/model imports occur until dispatch, source, pin and input admission.
Expected outputs come from upstream modules, never from the Rust CV implementation.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_provision import MANIFEST, ORACLES, ROOT, admit, digest, load_manifest, verify_asset


def validate_shapes(tensors, expected):
    if set(tensors) != set(expected):
        raise ValueError("oracle tensor inventory differs from the frozen contract")
    for name, shape in expected.items():
        value = tensors[name]
        if list(value.shape) != shape or str(value.dtype) != "float32":
            raise ValueError(f"oracle tensor shape/dtype mismatch: {name}")
        # Works for NumPy arrays; invoked only inside manual generation.
        if not bool((value == value).all()) or not bool((abs(value) < float("inf")).all()):
            raise ValueError(f"nonfinite oracle tensor: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--inference-sha", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--install-report", type=Path, required=True)
    args = parser.parse_args()
    admit(args.source_sha, args.inference_sha, "mlx")
    if sys.platform != "darwin" or sys.version_info[:2] != (3, 12) or os.environ.get("CV_ORACLE_EXECUTION") != "true":
        raise ValueError("oracle generation requires the explicit native Mac Python 3.12 manual CPU job")
    manifest = load_manifest()
    weight = next(item for item in manifest["models"] if item["name"] == "yolo11m_fused_mlx.safetensors")
    verify_asset(args.fixtures / weight["name"], weight)
    verify_asset(args.fixtures / "people.jpg", manifest["photo"])
    install = json.loads(args.install_report.read_text())
    installed = {item["metadata"]["name"].lower().replace("_", "-"): item for item in install["install"]}
    for package in manifest["oracle"]["packages"]:
        item = installed[package["name"].lower()]
        if (importlib.metadata.version(package["name"]) != package["version"] or
                item["download_info"]["archive_info"]["hashes"]["sha256"] != package["sha256"]):
            raise ValueError(f"oracle tool identity mismatch: {package['name']}")
    upstream = Path(importlib.metadata.distribution("ultralytics").locate_file("ultralytics"))
    source_files = {name: digest(upstream / name) for name in manifest["oracle"]["sourceFiles"]}
    if source_files != manifest["oracle"]["sourceFiles"]:
        raise ValueError("installed upstream architecture/source differs from the pinned commit")
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("oracle output must be outside checkout")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "running", "sourceSha": args.source_sha, "inferenceSha": args.inference_sha,
               "upstreamCommit": manifest["oracle"]["upstreamCommit"], "upstreamTag": manifest["oracle"]["upstreamTag"],
               "weightsSha256": weight["sha256"], "photoSha256": manifest["photo"]["sha256"],
               "manifestSha256": digest(MANIFEST), "producerSha256": digest(Path(__file__)),
               "device": "cpu", "dtype": "float32", "threads": 1, "sourceFiles": source_files, "outputs": {}}
    try:
        # This is the first model-capable import. The isolated job has already passed admission.
        import cv2
        import numpy as np
        import torch
        import ultralytics
        import yaml
        from PIL import Image
        from safetensors.numpy import load_file, save_file
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import ops

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        if Path(ultralytics.__file__).resolve().parent != upstream.resolve():
            raise ValueError("unexpected upstream import path")
        architecture_file = upstream / "cfg/models/11/yolo11.yaml"
        cfg = yaml.safe_load(architecture_file.read_text())
        cfg["scale"] = "m"
        if cfg["nc"] != 80 or cfg["scales"]["m"] != [0.5, 1.0, 512]:
            raise ValueError("pinned upstream YOLO11m architecture changed")
        # Upstream construction uses its own stride probe. All construction and forward work
        # occurs only here, on CPU, within the explicitly dispatched oracle workflow.
        model = DetectionModel(cfg, ch=3, nc=80, verbose=False).cpu().eval().fuse(verbose=False)
        if len(model.model) != 24 or model.stride.tolist() != [8, 16, 32] or model.model[23].nc != 80:
            raise ValueError("upstream graph/stride/class contract changed")
        arrays = load_file(str(args.fixtures / weight["name"]))
        if len(arrays) != 225 or set(arrays) != set(model.state_dict()):
            raise ValueError("upstream fused architecture does not match all 225 pinned tensors")
        state = {}
        for key, value in arrays.items():
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid fused weight dtype/data: {key}")
            # The published weights transpose every OIHW tensor to OHWI, including depthwise
            # and the fixed DFL conv. Restore every tensor; no partial/non-strict loading.
            restored = np.ascontiguousarray(value.transpose(0, 3, 1, 2) if value.ndim == 4 else value)
            state[key] = torch.from_numpy(restored)
            if state[key].shape != model.state_dict()[key].shape:
                raise ValueError(f"upstream weight shape mismatch: {key}")
        model.load_state_dict(state, strict=True)
        if any(not torch.equal(value, state[key]) for key, value in model.state_dict().items()):
            raise ValueError("upstream weights differ after strict loading")
        if any(value.device.type != "cpu" for value in model.parameters()):
            raise ValueError("oracle model must execute on CPU")
        if any(isinstance(value, torch.nn.BatchNorm2d) for value in model.modules()):
            raise ValueError("oracle architecture is not fully fused")
        receipt["tensorCount"] = len(state)
        receipt["toolVersions"] = {package["name"]: importlib.metadata.version(package["name"]) for package in manifest["oracle"]["packages"]}
        receipt["installReportSha256"] = digest(args.install_report)

        rgb = np.asarray(Image.open(args.fixtures / "people.jpg").convert("RGB"))
        if rgb.shape != (1080, 810, 3):
            raise ValueError("pinned photo must be 810x1080 RGB")
        ratio = min(640 / rgb.shape[1], 640 / rgb.shape[0])
        size = (round(rgb.shape[1] * ratio), round(rgb.shape[0] * ratio))
        pad_x, pad_y = (640 - size[0]) // 2, (640 - size[1]) // 2
        letterboxed = np.full((640, 640, 3), 114, dtype=np.uint8)
        letterboxed[pad_y:pad_y + size[1], pad_x:pad_x + size[0]] = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.ascontiguousarray(letterboxed.transpose(2, 0, 1)[None])).float() / 255
        captures, hooks, pools = {}, [], []

        def capture(name):
            def hook(_module, _inputs, value):
                captures[name] = value.detach().clone()
            return hook

        def before(name):
            def hook(_module, inputs):
                captures[name] = inputs[0].detach().clone()
            return hook

        for index in (4, 5, 6, 7, 8, 9, 10, 16, 19, 22):
            hooks.append(model.model[index].register_forward_hook(capture(f"block{index}")))
        sppf = model.model[9]
        hooks.append(sppf.cv1.register_forward_hook(capture("sppf_cv1")))
        hooks.append(sppf.m.register_forward_hook(lambda _module, _inputs, value: pools.append(value.detach().clone())))
        psa = model.model[10].m[0]
        attn = psa.attn
        if (attn.num_heads, attn.key_dim, attn.head_dim) != (4, 32, 64):
            raise ValueError("upstream attention contract changed")
        hooks.append(attn.qkv.register_forward_hook(capture("qkv")))
        hooks.append(attn.pe.register_forward_hook(capture("attn_pe")))
        hooks.append(attn.proj.register_forward_pre_hook(before("attn_projection_input")))
        hooks.append(psa.ffn.register_forward_pre_hook(before("psa_after_attn")))
        with torch.inference_mode():
            final, _raw = model(image)
            if final.shape != (1, 84, 8400) or not torch.isfinite(final).all():
                raise ValueError("upstream full output contract failed")
            if not bool(((final[:, 4:] >= 0) & (final[:, 4:] <= 1)).all()):
                raise ValueError("class outputs are not probabilities")
            if len(pools) != 3:
                raise ValueError("upstream SPPF must perform exactly three pool calls")
            captures.update({f"sppf_p{i + 1}": value for i, value in enumerate(pools)})
            # Derive the otherwise unexposed matmul intermediate from upstream-captured QKV,
            # using the pinned upstream Attention formula, then assert it reconstructs the
            # actual upstream projection input. No Rust implementation is imported or invoked.
            q, k, v = captures["qkv"].view(1, 4, 128, 400).split([32, 32, 64], dim=2)
            scores = ((q.transpose(-2, -1) @ k) * attn.scale).softmax(dim=-1)
            captures["attn_xattn"] = (v @ scores.transpose(-2, -1)).reshape(1, 256, 20, 20)
            delta = (captures["attn_xattn"] + captures["attn_pe"] - captures["attn_projection_input"]).abs().max().item()
            if delta >= 1e-5:
                raise ValueError(f"upstream attention decomposition failed: {delta}")
            # Upstream NMS and box scaling are the independent photo reference.
            detections = ops.non_max_suppression(final.clone(), conf_thres=0.25, iou_thres=0.45, classes=[0], max_det=300)[0]
            detections[:, :4] = ops.scale_boxes((640, 640), detections[:, :4], rgb.shape[:2], ratio_pad=((ratio, ratio), (pad_x, pad_y)))
        for hook in hooks:
            hook.remove()
        if len(detections) != 4 or not torch.isfinite(detections).all() or not bool((detections[:, 5] == 0).all()):
            raise ValueError("documented bus photo must produce exactly four finite person detections")
        base_keys = ("block4", "block10", "block16", "block19", "block22")
        extended_keys = tuple(f"block{i}" for i in range(4, 11)) + ("sppf_cv1", "sppf_p1", "sppf_p2", "sppf_p3", "attn_xattn", "attn_pe", "psa_after_attn")
        base = {name: captures[name].numpy() for name in base_keys}
        base.update(input=image.numpy(), final=final.numpy())
        extended = {name: captures[name].numpy() for name in extended_keys}
        extended["input"] = image.numpy()
        block_shapes = {"block4": [1, 512, 80, 80], "block5": [1, 512, 40, 40], "block6": [1, 512, 40, 40],
                        "block7": [1, 512, 20, 20], "block8": [1, 512, 20, 20], "block9": [1, 512, 20, 20],
                        "block10": [1, 512, 20, 20], "block16": [1, 256, 80, 80], "block19": [1, 512, 40, 40], "block22": [1, 512, 20, 20]}
        validate_shapes(base, {**{name: block_shapes[name] for name in base_keys}, "input": [1, 3, 640, 640], "final": [1, 84, 8400]})
        validate_shapes(extended, {**{name: block_shapes[name] for name in extended_keys if name.startswith("block")},
                                  **{name: [1, 256, 20, 20] for name in extended_keys if not name.startswith("block")}, "input": [1, 3, 640, 640]})
        save_file(base, str(output / "refs.safetensors"))
        save_file(extended, str(output / "refs_ext.safetensors"))
        (output / "ref_people.json").write_text(json.dumps({"dets": [{"xyxy": row[:4].tolist(), "conf": float(row[4]), "cls": int(row[5])} for row in detections],
                                                            "settings": {"imgsz": 640, "conf": 0.25, "iou": 0.45, "classes": [0], "padding": 114}}, indent=2) + "\n")
        receipt.update(personCount=4, attentionReconstructionMaxAbs=delta,
                       shapes={"base": {k: list(v.shape) for k, v in base.items()}, "extended": {k: list(v.shape) for k, v in extended.items()}},
                       outputs={name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size} for name in ORACLES})
        if sum(item["bytes"] for item in receipt["outputs"].values()) > manifest["budgets"]["oracleOutputBytes"]:
            raise ValueError("oracle output budget exceeded")
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        (output / "oracle.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
