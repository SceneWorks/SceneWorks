"""Provision the pinned Candle tiers of the Qwen3.8 27B describer (SW-7120).

`SCENEWORKS_QWEN38_DESCRIBER_PROVISION` picks what is provisioned: `quantised` (the default) is the
re-host's q4/ and q8/ folders; `bf16` is the full-precision upstream snapshot, the files the
catalog's bf16 row names. Run only by the dispatch-only `qwen38-describer-provision.yml` workflow on the CUDA real-weights
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

# Full precision (bf16): the upstream repository at the catalog's pinned revision, the files its
# bf16 download row names (config/manifests/builtin.models.jsonc). The row's size is an estimate,
# so the listed total must fall within BF16_TOLERANCE of it; each file is then checked at its
# published size.
BF16_REPO = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"
BF16_REVISION = "739e3c5b89849f6c238ce1e5b70008612ae42cdd"
BF16_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)
BF16_SHARD_PATTERN = "model-*.safetensors"
BF16_ESTIMATED_BYTES = 55_575_900_000
BF16_TOLERANCE = 0.01
# ~55.6 GB of files; about the same ~20 GB of headroom as the quantised tiers.
BF16_MIN_FREE_BYTES = 75 * 10**9
MODES = ("quantised", "bf16")


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


def bf16_expected_files(entries, estimated=BF16_ESTIMATED_BYTES, tolerance=BF16_TOLERANCE):
    """The upstream files the bf16 row names, as {path: size}: every named file and every
    top-level `model-*.safetensors` shard, sized, with a total within `tolerance` of the estimate."""
    sized = {name: size for name, size in entries}
    missing = [name for name in BF16_FILES if not isinstance(sized.get(name), int) or sized[name] <= 0]
    if missing:
        raise RuntimeError(f"{BF16_REPO}@{BF16_REVISION} lists no sized {', '.join(missing)}")
    shards = {name: size for name, size in sized.items()
              if "/" not in name and name.startswith("model-") and name.endswith(".safetensors")}
    if not shards or any(not isinstance(size, int) or size <= 0 for size in shards.values()):
        raise RuntimeError(f"{BF16_REPO}@{BF16_REVISION} lists no sized {BF16_SHARD_PATTERN} shards")
    expected = {name: sized[name] for name in BF16_FILES} | shards
    total = sum(expected.values())
    if abs(total - estimated) > estimated * tolerance:
        raise RuntimeError(f"bf16 files at {BF16_REVISION} total {total} bytes, the catalog estimates {estimated}")
    return expected


def snapshot_dir(cache: Path, repo: str = REPO, revision: str = REVISION) -> Path:
    return cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


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


def plan(mode):
    """What `mode` provisions: (repo, revision, label, expected-files reader, download patterns,
    free-space minimum)."""
    if mode == "quantised":
        return (REPO, REVISION, f"tiers {', '.join(TIER_BYTES)}", expected_files,
                [f"{tier}/*" for tier in TIER_BYTES], MIN_FREE_BYTES)
    if mode == "bf16":
        return (BF16_REPO, BF16_REVISION, "full precision (bf16)", bf16_expected_files,
                [*BF16_FILES, BF16_SHARD_PATTERN], BF16_MIN_FREE_BYTES)
    raise RuntimeError(f"unknown provisioning mode {mode!r}; use one of {', '.join(MODES)}")


def main():
    cache = Path(os.environ.get("HF_HUB_CACHE", ""))
    if not cache.is_absolute() or not cache.is_dir():
        raise RuntimeError(f"HF_HUB_CACHE must name an existing absolute hub directory, got {str(cache)!r}")
    mode = os.environ.get("SCENEWORKS_QWEN38_DESCRIBER_PROVISION", "").strip() or "quantised"
    repo, revision, label, read_expected, patterns, min_free = plan(mode)
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=False)
    if info.sha != revision:
        raise RuntimeError(f"{repo} resolved {revision} to {info.sha}")
    expected = read_expected([(sibling.rfilename, sibling.size) for sibling in info.siblings])
    snapshot = snapshot_dir(cache, repo, revision)
    missing = mismatched_files(snapshot, expected)
    if not missing:
        report([f"Already present: {repo}@{revision} {label} "
                f"({len(expected)} files) at `{snapshot}`. Nothing downloaded."])
        return 0
    free = shutil.disk_usage(cache).free
    require_free_space(free, min_free)
    print(f"{len(missing)} of {len(expected)} files missing or incomplete; {free / 1e9:.1f} GB free. Downloading.")
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        revision=revision,
        allow_patterns=patterns,
        cache_dir=str(cache),
        token=False,
        max_workers=4,
    )
    still = mismatched_files(snapshot, expected)
    if still:
        raise RuntimeError("files have the wrong size after download: "
                           + "; ".join(f"{name} expected {size}, found {found}" for name, size, found in still))
    report([f"Provisioned {repo}@{revision} {label} at `{snapshot}`.",
            f"- {sum(expected.values())} bytes in {len(expected)} files, checked at their published sizes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
