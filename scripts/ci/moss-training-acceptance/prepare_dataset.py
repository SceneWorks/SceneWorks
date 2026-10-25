#!/usr/bin/env python3
"""Prepare the fixed three-shape image dataset once; never load model weights.

The RGB recipe is the frozen adapter_acceptance.rs lines 21..58 named by the
campaign. PNG encoding is lossless and performed once in the hosted preparation
job; every native cell consumes the very same archived image/caption bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

MANIFEST_SHA256 = "cad2d90a22ccda3f69f74103386efa49e9d2c89d76e65d8b92195e201a4c8296"
RECIPE_SHA256 = "32032a0200ed4d96518d29042e0e8a7604819a396eca0f0866fbcf131aac7941"
CAPTIONS = (
    "sksmoss red circle centered on a pale background",
    "sksmoss blue square centered on a pale background",
    "sksmoss green triangle centered on a pale background",
)
COLORS = ((220, 30, 30), (30, 60, 220), (30, 180, 60))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pixel(index, x, y):
    if index == 0:
        inside = (x - 128) ** 2 + (y - 128) ** 2 < 70 * 70
    elif index == 1:
        inside = 58 <= x < 198 and 58 <= y < 198
    elif index == 2:
        inside = 45 < y < 210 and abs(x - 128) < (y - 45) // 2
    else:
        raise ValueError("only the three frozen shapes are defined")
    return COLORS[index] if inside else (240, 240, 224)


def chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def shape_png(index):
    rows = bytearray()
    for y in range(256):
        rows.append(0)  # Lossless PNG filter None; no machine-dependent image metadata.
        for x in range(256):
            rows.extend(pixel(index, x, y))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def prepare(manifest_file, output):
    manifest_file, output = Path(manifest_file), Path(output)
    encoded = manifest_file.read_bytes()
    if sha256(encoded) != MANIFEST_SHA256:
        raise ValueError("fixed dataset preparation requires the exact sealed campaign")
    manifest = json.loads(encoded)
    dataset = manifest["dataset"]
    if (dataset["id"], dataset["recipe"]["sourceSha256"], dataset["width"], dataset["height"]) != (
            "synthetic-moss-v1", RECIPE_SHA256, 256, 256):
        raise ValueError("dataset recipe changed; document/reseal the complete campaign")
    if output.exists() and any(output.iterdir()):
        raise ValueError("dataset output must be new/empty; never carry prior campaign data")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, caption in enumerate(CAPTIONS):
        image = shape_png(index)
        image_name, caption_name = f"shape-{index}.png", f"shape-{index}.txt"
        if dataset["items"][index] != {"image": image_name, "caption": caption, "mask": None}:
            raise ValueError("frozen image/caption/mask recipe mismatch")
        caption_bytes = caption.encode("utf-8")
        (output / image_name).write_bytes(image)
        (output / caption_name).write_bytes(caption_bytes)
        records.append({"image": image_name, "imageSha256": sha256(image),
                        "captionFile": caption_name, "captionSha256": sha256(caption_bytes),
                        "caption": caption, "mask": None, "width": 256, "height": 256})
    document = {"schemaVersion": 1, "kind": "moss-fixed-image-dataset",
                "datasetId": dataset["id"], "manifestSha256": MANIFEST_SHA256,
                "recipeSourceSha256": RECIPE_SHA256,
                "generatorSha256": sha256(Path(__file__).read_bytes()),
                "imageCount": 3, "items": records,
                "modelWeightsLoaded": False,
                "evaluationManifestHash": None,
                "evaluationHashNote": "Native EvaluationManifest::new computes the evaluation identity before weights load from these exact bytes; this preparation step does not fabricate it."}
    payload = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")
    (output / "dataset-manifest.json").write_bytes(payload)
    return sha256(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    print(json.dumps({"preparedDatasetManifestSha256": prepare(arguments.manifest, arguments.output),
                      "modelExecution": False}))
