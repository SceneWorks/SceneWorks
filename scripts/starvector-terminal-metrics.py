#!/usr/bin/env python3
"""No-download, source-owned standard metric runner for SC-22261.

The real campaign bundle maps every product attachment to one case record.  A
runner can provide an interpreter, but cannot replace this script or provide a
metric command which invents terminal aggregates.
"""
import argparse
import hashlib
import json
import pathlib

_LPIPS = None
_CLIP = {}


def fail(message):
    raise SystemExit("starvector terminal metrics: " + message)


def sha256_file(item):
    digest = hashlib.sha256()
    with open(item, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb512(item):
    from PIL import Image
    import numpy
    image = Image.open(item).convert("RGBA")
    if image.size != (512, 512):
        fail("metric attachment is not a 512x512 preview")
    canvas = Image.new("RGBA", image.size, "white")
    canvas.alpha_composite(image)
    return numpy.asarray(canvas.convert("RGB"), dtype=numpy.uint8)


def compare(reference, preview):
    import lpips
    import numpy
    import torch
    from skimage.metrics import structural_similarity
    left, right = rgb512(reference), rgb512(preview)
    ssim = structural_similarity(
        left, right, data_range=255, channel_axis=2, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False,
    )
    global _LPIPS
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="alex", version="0.1").eval()
    def tensor(image):
        return torch.from_numpy(image.astype(numpy.float32).transpose(2, 0, 1)).unsqueeze(0).mul(2.0 / 255.0).sub(1.0)
    with torch.no_grad():
        distance = float(_LPIPS(tensor(left), tensor(right)).item())
    return {"ssim": ssim, "lpips": distance}


def event_evidence(event, label):
    value = event.get("job", {}).get("terminal_evidence")
    if not isinstance(value, dict):
        fail(label + " product job is missing typed terminal evidence")
    return value


def prompt_cosine(prompt, image_path, clip):
    """Exact local CLIP comparison; the bundle names a pre-provisioned file."""
    import open_clip
    import torch
    from PIL import Image
    key = (clip["model"], clip["checkpoint"])
    if key not in _CLIP:
        if not pathlib.Path(clip["checkpoint"]).is_file() or sha256_file(clip["checkpoint"]) != clip["checkpoint_sha256"]:
            fail("pre-provisioned CLIP checkpoint hash mismatch")
        _CLIP[key] = (*open_clip.create_model_and_transforms(clip["model"], pretrained=clip["checkpoint"], device="cpu"), open_clip.get_tokenizer(clip["model"]))
    model, _, preprocess, tokenizer = _CLIP[key]
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image); text_features = model.encode_text(tokenizer([prompt]))
        image_features /= image_features.norm(dim=-1, keepdim=True); text_features /= text_features.norm(dim=-1, keepdim=True)
    return float((image_features @ text_features.T).item())


def terminal_suites(bundle, events):
    identity = bundle.get("terminal_suite_identity")
    if not isinstance(identity, dict):
        fail("bundle lacks terminal suite identity")
    hostile_cases = []
    for index, (case, event) in enumerate(zip(bundle["hostile_sanitizer"], events["hostile_sanitizer"])):
        observed = event_evidence(event, "hostile")
        # All publication/staging and inline facts are emitted by the actual
        # product worker, never inferred from an expected-disposition flag.
        hostile_cases.append({"case_index": index, "case_id": case["case_id"], "input_sha256": case["input_sha256"], "expected_policy": "reject_or_sanitize_inert", "outcome": observed["outcome"], "error_code": observed["error_code"], "canonical_svg_sha256": observed.get("canonical_svg_sha256"), "preview_png_sha256": observed.get("preview_png_sha256"), "published_paths": observed["published_paths"], "staging_residue": observed["staging_residue"], "result_contains_inline_svg": observed["result_contains_inline_svg"]})
    prompt_cases = []
    clip = identity.get("clip")
    for index, (case, event) in enumerate(zip(bundle["prompt_composition"], events["prompt_composition"])):
        observed = event_evidence(event, "prompt")
        accepted = observed.get("accepted") is True
        raster_cosine = preview_cosine = loss = None
        if accepted:
            for item in (case["raster_png"], observed["preview_png"]):
                if not pathlib.Path(item).is_file(): fail("prompt product attachment missing")
            raster_cosine = prompt_cosine(case["prompt"], case["raster_png"], clip)
            preview_cosine = prompt_cosine(case["prompt"], observed["preview_png"], clip)
            loss = raster_cosine - preview_cosine
        prompt_cases.append({"case_index": index, "case_id": case["case_id"], "prompt_sha256": case["prompt_sha256"], "raster_png_sha256": case["raster_png_sha256"], "vector_provider_transcript_sha256": observed["provider_transcript_sha256"], "canonical_svg_sha256": observed.get("canonical_svg_sha256") if accepted else None, "preview_png_sha256": observed.get("preview_png_sha256") if accepted else None, "accepted": accepted, "raster_prompt_cosine": raster_cosine, "preview_prompt_cosine": preview_cosine, "alignment_loss": loss})
    return {"execution": identity["execution"], "producer": identity["producer"], "metric_identity": identity["metric_identity"], "inference_preflight": identity["inference_preflight"], "hostile_sanitizer": {"corpus_sha256": identity["hostile_corpus_sha256"], "sanitizer_version": identity["sanitizer_version"], "cases": hostile_cases}, "prompt_composition": {"raster_provider_id": identity["raster_provider_id"], "raster_model": identity["raster_model"], "raster_revision": identity["raster_revision"], "raster_inventory_sha256": identity["raster_inventory_sha256"], "clip_provider_id": identity["clip_provider_id"], "clip_model": identity["clip_model"], "clip_revision": identity["clip_revision"], "clip_inventory_sha256": identity["clip_inventory_sha256"], "metric_transcript_sha256": identity["metric_transcript_sha256"], "corpus_sha256": identity["prompt_corpus_sha256"], "cases": prompt_cases}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["measure"])
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tuple", required=True)
    parser.add_argument("--transcript-sha256", required=True)
    args = parser.parse_args()
    import os
    if os.environ.get("STARVECTOR_TERMINAL_NO_JOB_DOWNLOADS") != "1":
        fail("no-job-downloads guard is required")
    bundle = json.loads(pathlib.Path(args.bundle).read_text())
    events = json.loads(pathlib.Path(args.events).read_text())
    if events.get("tuple") != args.tuple:
        fail("route events tuple mismatch")
    cases = bundle["tuples"][args.tuple]["image_quality"]
    if len(cases) != 120 or len(events.get("image_quality", [])) != 120:
        fail("exactly 120 product quality cases required")
    facts = []
    for case, event in zip(cases, events["image_quality"]):
        if case.get("case_id") != event.get("case_id"):
            fail("route event order does not match the immutable bundle")
        reference, preview = pathlib.Path(case["reference_png"]), pathlib.Path(case["preview_png"])
        if not reference.is_file() or not preview.is_file():
            fail("product metric attachment missing")
        if sha256_file(reference) != case["reference_png_sha256"] or sha256_file(preview) != case["preview_png_sha256"]:
            fail("product metric attachment hash mismatch")
        facts.append({"case_id": case["case_id"], **compare(reference, preview)})
    parity_facts = []
    parity_records = bundle["tuples"][args.tuple]["deterministic_parity"]
    if len(parity_records) != 20 or len(events.get("deterministic_parity", [])) != 20:
        fail("exactly 20 deterministic reruns required")
    for case, event in zip(parity_records, events["deterministic_parity"]):
        if case.get("case_id") != event.get("case_id"):
            fail("deterministic route event order mismatch")
        first, second = event_evidence({"job": event.get("first")}, "parity first"), event_evidence({"job": event.get("second")}, "parity second")
        first_file, second_file = pathlib.Path(first["preview_png"]), pathlib.Path(second["preview_png"])
        if not first_file.is_file() or not second_file.is_file() or sha256_file(first_file) != first["preview_png_sha256"] or sha256_file(second_file) != second["preview_png_sha256"]:
            fail("deterministic preview artifact mismatch")
        parity_facts.append({"case_id": case["case_id"], "rendered_ssim": compare(first_file, second_file)["ssim"]})
    # The JS producer owns exact inference-schema assembly.  This script emits
    # raw per-case values only, never a pass/fail or trusted aggregate.
    result = {"tuple": args.tuple, "route_transcript_sha256": args.transcript_sha256, "image_quality_facts": facts, "deterministic_parity_facts": parity_facts}
    if args.tuple == "candle-cuda:8b":
        if len(events.get("hostile_sanitizer", [])) != 200 or len(events.get("prompt_composition", [])) != 60:
            fail("final tuple is missing a complete hostile or prompt suite")
        result["terminal_suites"] = terminal_suites(bundle, events)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
