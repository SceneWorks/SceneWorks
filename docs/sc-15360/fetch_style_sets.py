#!/usr/bin/env python3
"""Pull public style datasets from the HF datasets-server into a CALIB_DIR layout.

Writes <out>/<set-name>/NNN.jpg plus a sibling NNN.txt caption, which is the layout
both calibrate_thresholds (images) and sweep_caption_alignment (image + sibling .txt)
expect.
"""
import json
import os
import sys
import urllib.request

SERVER = "https://datasets-server.huggingface.co"
# (dataset, config, split, how many rows to take)
SETS = [
    ("Norod78/Yarn-art-style", "default", "train", 18),
    ("Norod78/PaperCutoutStyle", "default", "train", 14),
    ("Norod78/WeirdOutfitStyle", "default", "train", 40),
    ("Norod78/cartoon-blip-captions", "default", "train", 40),
]
OUT = sys.argv[1] if len(sys.argv) > 1 else "./style-calib"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sceneworks-calib"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def rows(dataset, config, split, offset, length):
    url = f"{SERVER}/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
    return json.loads(get(url))


for dataset, config, split, want in SETS:
    name = dataset.split("/")[-1]
    dest = os.path.join(OUT, name)
    os.makedirs(dest, exist_ok=True)
    got = 0
    offset = 0
    while got < want:
        batch = min(100, want - got)
        try:
            payload = rows(dataset, config, split, offset, batch)
        except Exception as exc:
            print(f"  {name}: rows failed at offset {offset}: {str(exc)[:90]}")
            break
        items = payload.get("rows", [])
        if not items:
            break
        # Find the image column and a text/caption column from the feature list.
        feats = {f["name"]: f["type"].get("_type") for f in payload.get("features", [])}
        img_col = next((k for k, v in feats.items() if v == "Image"), None)
        txt_col = next((k for k, v in feats.items() if v == "Value" and k != img_col), None)
        if img_col is None:
            print(f"  {name}: no image column in {feats}")
            break
        for entry in items:
            row = entry["row"]
            src = row.get(img_col, {}).get("src") if isinstance(row.get(img_col), dict) else None
            if not src:
                continue
            try:
                blob = get(src)
            except Exception as exc:
                print(f"  {name}: image fetch failed: {str(exc)[:70]}")
                continue
            stem = os.path.join(dest, f"{got:03d}")
            with open(stem + ".jpg", "wb") as fh:
                fh.write(blob)
            caption = (row.get(txt_col) or "").strip() if txt_col else ""
            if caption:
                with open(stem + ".txt", "w") as fh:
                    fh.write(caption)
            got += 1
            if got >= want:
                break
        offset += len(items)
    caps = len([f for f in os.listdir(dest) if f.endswith(".txt")])
    print(f"{name}: {got} images, {caps} captions -> {dest}")

# Reproduce (sc-15360 style-floor calibration):
#   python3 fetch_style_sets.py ./style-calib
#   CALIB_DIR=./style-calib RUST_TEST_THREADS=1 \
#     cargo test -p sceneworks-worker calibrate_thresholds -- --ignored --nocapture
#
# The datasets-server `/rows` endpoint returns image URLs AND a caption column, so the
# same corpus also feeds `sweep_caption_alignment` (sc-15388) once captions are written
# as sibling .txt files — which this script does.
