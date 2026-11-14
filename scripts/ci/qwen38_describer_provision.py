"""Provision the pinned Candle q4/q8 tiers of the Qwen3.8 27B describer (SW-7120).

Run only by the dispatch-only `qwen38-describer-provision.yml` workflow on the CUDA real-weights
runner. Downloads into the standard hub layout under HF_HUB_CACHE, anonymously, at one exact
revision. Does nothing when every file is already present at its published size; refuses before
downloading when the cache volume has less than MIN_FREE_BYTES free; checks every file's size after
downloading. Runs no model.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

REPO = "SceneWorks/qwen3.8-27b-abliterated-candle"
REVISION = "accaccc109912b9b194f7d5a47f2123fdc70672e"
# Tier folder -> total bytes of its files, as the catalog's download rows record them
# (config/manifests/builtin.models.jsonc, `footprint.diskSizeBytes`).
TIER_BYTES = {"q4": 19_995_370_331, "q8": 32_371_188_675}
# Both tiers are ~52.4 GB; keep headroom for the volume's other tenants.
MIN_FREE_BYTES = 70 * 10**9


def expected_files(entries, tier_bytes=TIER_BYTES):
    """The repo's files under each tier folder, as {path: size}, checked against the tier totals."""
    expected = {}
    for tier, total in tier_bytes.items():
        files = {name: size for name, size in entries if name.startswith(f"{tier}/")}
        if not files or any(not isinstance(size, int) or size <= 0 for size in files.values()):
            raise RuntimeError(f"{REPO}@{REVISION} lists no sized files under {tier}/")
        if sum(files.values()) != total:
            raise RuntimeError(
                f"{tier}/ at {REVISION} totals {sum(files.values())} bytes, the catalog says {total}")
        expected.update(files)
    return expected


def snapshot_dir(cache: Path) -> Path:
    return cache / ("models--" + REPO.replace("/", "--")) / "snapshots" / REVISION


def mismatched_files(snapshot: Path, expected):
    """Files absent from the snapshot or of the wrong size, as [(path, expected, found-or-None)]."""
    result = []
    for name, size in sorted(expected.items()):
        path = snapshot.joinpath(*name.split("/"))
        found = path.stat().st_size if path.is_file() else None
        if found != size:
            result.append((name, size, found))
    return result


def require_free_space(free_bytes: int, minimum: int = MIN_FREE_BYTES):
    if free_bytes < minimum:
        raise RuntimeError(
            f"the cache volume has {free_bytes / 1e9:.1f} GB free; provisioning needs at least "
            f"{minimum / 1e9:.0f} GB. Free space on it and dispatch again.")


def report(lines):
    text = "\n".join(lines) + "\n"
    print(text, end="")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as file:
            file.write("## Qwen3.8 27B describer provisioning\n\n" + text)


def main():
    cache = Path(os.environ.get("HF_HUB_CACHE", ""))
    if not cache.is_absolute() or not cache.is_dir():
        raise RuntimeError(f"HF_HUB_CACHE must name an existing absolute hub directory, got {str(cache)!r}")
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION:
        raise RuntimeError(f"{REPO} resolved {REVISION} to {info.sha}")
    expected = expected_files([(sibling.rfilename, sibling.size) for sibling in info.siblings])
    snapshot = snapshot_dir(cache)
    missing = mismatched_files(snapshot, expected)
    if not missing:
        report([f"Already present: {REPO}@{REVISION} tiers {', '.join(TIER_BYTES)} "
                f"({len(expected)} files) at `{snapshot}`. Nothing downloaded."])
        return 0
    free = shutil.disk_usage(cache).free
    require_free_space(free)
    print(f"{len(missing)} of {len(expected)} files missing or incomplete; {free / 1e9:.1f} GB free. Downloading.")
    snapshot_download(
        repo_id=REPO,
        repo_type="model",
        revision=REVISION,
        allow_patterns=[f"{tier}/*" for tier in TIER_BYTES],
        cache_dir=str(cache),
        token=False,
        max_workers=4,
    )
    still = mismatched_files(snapshot, expected)
    if still:
        raise RuntimeError("files have the wrong size after download: "
                           + "; ".join(f"{name} expected {size}, found {found}" for name, size, found in still))
    report([f"Provisioned {REPO}@{REVISION} tiers {', '.join(TIER_BYTES)} at `{snapshot}`.",
            *(f"- {tier}/: {total} bytes" for tier, total in TIER_BYTES.items()),
            f"- {len(expected)} files checked at their published sizes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
