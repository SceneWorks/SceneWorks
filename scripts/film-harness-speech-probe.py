#!/usr/bin/env python3
"""Probe a film-harness export for the SPOKEN dialogue lines the run synthesized (sc-23404).

The phase-1 evaluation probed the export's tones with a Goertzel filter at a known frequency, which
is exactly what a synthesized line does not have: speech is broadband and its content is not known
in advance.  So this asks the question the tone probe was standing in for — *is there something in
the dialogue's slot that is not there just outside it* — by comparing the decoded mix's RMS inside
each line's window against a control window of the same length in the same shot.

The beds run continuously under everything, so the control window is the honest baseline: whatever
the beds contribute is in BOTH numbers, and only the line is in one of them.  A line that landed
lifts its window well above the control; a line that was dropped (the failure mode that made
`hand-film-FINAL-30s.mp4` silent) leaves the two equal.

It reads the run record for where each line is supposed to be, so the slots are the ones the
harness actually placed rather than ones re-derived from the plan by hand.

Usage:
    scripts/film-harness-speech-probe.py --run RUN_DIR/run.json --export FILE.mp4 \
        [--out probe.json] [--ffmpeg /path/to/ffmpeg]

Exit codes: 0 every placed line is present, 1 at least one is not, 2 the inputs are unusable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

# A line has to be at least this much louder (in RMS ratio) than the control window beside it to
# count as present.  Speech at gain 1.0 over beds at 0.35 / 0.2 clears this by a wide margin; the
# threshold is set where it is so a line that is merely a little louder than the beds does not pass
# as one that was actually spoken.
PRESENT_RATIO = 1.5
# How far from the line's own window the control window is taken, so a fade does not leak into it.
CONTROL_GAP_SECONDS = 0.3


def decode_pcm16(ffmpeg: str, path: str, rate: int = 16000) -> list[float]:
    """The export's audio as mono float samples, decoded by ffmpeg."""
    with tempfile.TemporaryDirectory() as work:
        out = os.path.join(work, "mix.wav")
        result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", path,
             "-vn", "-ac", "1", "-ar", str(rate), "-f", "wav", "-acodec", "pcm_s16le", out],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"ffmpeg could not decode {path}: {result.stderr.strip()}")
        with open(out, "rb") as handle:
            data = handle.read()
    start = data.find(b"data")
    if start < 0:
        raise SystemExit(f"decoded {path} has no data chunk")
    body = data[start + 8 :]
    count = len(body) // 2
    return [value / 32768.0 for value in struct.unpack("<%dh" % count, body[: count * 2])]


def rms(samples: list[float], start: float, end: float, rate: int) -> float:
    first = max(0, int(start * rate))
    last = min(len(samples), int(end * rate))
    if last <= first:
        return 0.0
    total = sum(value * value for value in samples[first:last])
    return math.sqrt(total / (last - first))


def dialogue_slots(record: dict) -> list[dict]:
    """Every dialogue item the run PLACED, from the timeline the record read back off the save.

    Timeline seconds are picture seconds only when no shot carries a crossfade (the fixture carries
    none); with transitions the export is shorter than the timeline and these windows would need the
    same offset the mix applies.  The report says which timeline it read so that is checkable.
    """
    timeline = record.get("timeline") or {}
    slots = []
    for track in timeline.get("tracks") or []:
        if track.get("role") != "dialogue":
            continue
        for item in track.get("items") or []:
            start = item.get("timelineStart")
            end = item.get("timelineEnd")
            if start is None or end is None or end <= start:
                continue
            slots.append(
                {
                    "itemId": item.get("itemId"),
                    "shotId": item.get("shotId"),
                    "startSeconds": float(start),
                    "endSeconds": float(end),
                }
            )
    return slots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="path to the run record (run.json)")
    parser.add_argument(
        "--export",
        help="path to the exported MP4; default: the record's own projectPath + export.renderPath "
        "(renderPath is project-RELATIVE, which is the mistake this default exists to prevent)",
    )
    parser.add_argument("--out", help="write the probe JSON here as well as to stdout")
    parser.add_argument("--ffmpeg", default=os.environ.get("SCENEWORKS_FFMPEG") or shutil.which("ffmpeg"))
    args = parser.parse_args()
    if not args.ffmpeg:
        print("no ffmpeg on PATH and SCENEWORKS_FFMPEG is unset", file=sys.stderr)
        return 2

    with open(args.run) as handle:
        record = json.load(handle)
    export_path = args.export
    if not export_path:
        rendered = (record.get("export") or {}).get("renderPath")
        project = record.get("projectPath")
        if not rendered or not project:
            print(
                "the record names no export (projectPath + export.renderPath); pass --export",
                file=sys.stderr,
            )
            return 2
        export_path = os.path.join(project, rendered)
    if not os.path.isfile(export_path):
        print(f"no such export: {export_path}", file=sys.stderr)
        return 2
    args.export = export_path
    rate = 16000
    samples = decode_pcm16(args.ffmpeg, args.export, rate)
    duration = len(samples) / rate

    slots = dialogue_slots(record)
    spoken = {line["role"]: line for line in record.get("synthesizedSound") or []}
    results = []
    for slot in slots:
        length = slot["endSeconds"] - slot["startSeconds"]
        inside = rms(samples, slot["startSeconds"], slot["endSeconds"], rate)
        # The control sits AFTER the line where there is room, otherwise before it.
        after = slot["endSeconds"] + CONTROL_GAP_SECONDS
        if after + length <= duration:
            control_start = after
        else:
            control_start = max(0.0, slot["startSeconds"] - CONTROL_GAP_SECONDS - length)
        control = rms(samples, control_start, control_start + length, rate)
        ratio = (inside / control) if control > 0 else (math.inf if inside > 0 else 0.0)
        results.append(
            {
                **slot,
                "insideRms": round(inside, 6),
                "controlStartSeconds": round(control_start, 4),
                "controlRms": round(control, 6),
                "ratio": None if ratio == math.inf else round(ratio, 3),
                "present": ratio >= PRESENT_RATIO,
            }
        )

    report = {
        "export": os.path.abspath(args.export),
        "run": os.path.abspath(args.run),
        "decodedSeconds": round(duration, 3),
        "presentRatioThreshold": PRESENT_RATIO,
        "synthesizedLines": [
            {
                "role": line.get("role"),
                "text": line.get("text"),
                "model": line.get("model"),
                "voice": line.get("voice"),
                "status": line.get("status"),
                "jobId": line.get("jobId"),
                "assetId": line.get("assetId"),
                "file": line.get("file"),
            }
            for line in spoken.values()
        ],
        "dialogueSlots": results,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    if not results:
        print("no dialogue items were placed in this export", file=sys.stderr)
        return 1
    return 0 if all(slot["present"] for slot in results) else 1


if __name__ == "__main__":
    sys.exit(main())
