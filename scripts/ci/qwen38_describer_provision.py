"""Provision the pinned Candle tiers of the Qwen3.8 27B describer (SW-7120) and of the
MiniMax-H3-Image prompt expander (SW-7202).

`SCENEWORKS_QWEN38_DESCRIBER_PROVISION` picks what is provisioned: `quantised` (the default) is the
files the catalog's q4 and q8 rows name in the re-host's fully quantized `q4-full/` and `q8-full/`
folders (SW-7202: config, tokenizer and template sidecars, the shard index and every shard);
`bf16` is the full-precision upstream snapshot, the files the
catalog's bf16 row names. `expander-q4` / `expander-q8` is ONE expander tier: every file of its
Candle folder (`candle-q4-full/` or `candle-q8-full/`, the folder its catalog row names) plus the
five upstream `prompt/` files the catalog's `prompt` co-requisite names, each at its own pinned
revision. Only the selected tier is fetched, so a cache provisioned this way holds no other
expander tier. `expander-abliterated-q4` / `expander-abliterated-q8` (SW-7208) is the expander run by
the abliterated describer instead: ONE describer tier (the files its catalog row names in `q4-full/`
or `q8-full/`) plus the same five prompt files, and no MiniMax-H3-Image-PE weights. Every mode
also provisions the two Qwen3.8-27B block drafts (SW-7204; DFlash2 and DSpine, the soft
co-requisites both catalog entries install with every tier), each at its pinned revision; `drafts`
provisions only them. Run only by the dispatch-only `qwen38-describer-provision.yml` workflow on the CUDA real-weights
runner. Downloads into the standard hub layout under HF_HUB_CACHE, anonymously, at one exact
revision. Does nothing when every file is already present at its published size; refuses before
downloading when the cache volume has less than MIN_FREE_BYTES free; checks every file's size after
downloading. Runs no model.
"""
from __future__ import annotations

from fnmatch import fnmatchcase
import os
from pathlib import Path
import shutil
import sys

REPO = "SceneWorks/qwen3.8-27b-abliterated-candle"
# The revision and totals come from the inference `qwen38-full-tiers.yml` (model describer, backend
# candle) publication report; they must equal the catalog's rows.
REVISION = "f76e0088020724953fd129a0004fa78bd9fee8cc"
# Tier (the catalog row's `variant`) -> the re-host folder its row names.
TIER_FOLDERS = {"q4": "q4-full", "q8": "q8-full"}
# The selectors each tier's catalog row names inside its folder. A Candle tier is sharded like its
# bf16 source: candle-llm `prepare_snapshot` writes one shard per source shard under the source's
# name plus a fresh `model.safetensors.index.json`, and copies the source's sidecars. Only the
# files these selectors match are downloaded and totalled, exactly what the app's tier download
# fetches; any other sidecar the preparer copied into the folder is neither.
TIER_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
)
# Tier -> total bytes of the files its row's selectors match, as the catalog's download rows record
# them (config/manifests/builtin.models.jsonc, `footprint.diskSizeBytes`).
TIER_BYTES = {"q4": 17_589_495_082, "q8": 29_965_305_376}
# Both tiers are ~47.6 GB; keep headroom for the volume's other tenants.
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

# The MiniMax-H3-Image prompt expander (catalog `minimax_h3_image_pe`): its Candle tiers' re-host
# and revision, and per tier the folder its row names (`candle-q4-full/*`: every file of it) and
# that folder's total, the row's `footprint.diskSizeBytes`.
EXPANDER_REPO = "SceneWorks/minimax-h3-image-pe"
EXPANDER_REVISION = "b0efd86df40968dcdcaebfbf4267ae1bce9431b5"
EXPANDER_TIER_FOLDERS = {"q4": "candle-q4-full", "q8": "candle-q8-full"}
EXPANDER_TIER_BYTES = {"q4": 17_357_728_848, "q8": 29_521_202_308}
# The tier plus about 20 GB of headroom, as for the describer's tiers.
EXPANDER_MIN_FREE_BYTES = {"q4": 40 * 10**9, "q8": 52 * 10**9}
# The upstream `prompt/` files the worker reads at run time and verifies against
# `prompt/manifest.json`: the catalog's `prompt` co-requisite row and its exact total.
PROMPT_REPO = "MiniMaxAI/MiniMax-H3-Image-PE"
PROMPT_REVISION = "35603d4b63949afa12cb098227fffefedf081afa"
PROMPT_FILES = (
    "prompt/manifest.json",
    "prompt/t2i/system_prompt.txt",
    "prompt/t2i/task_prompt.txt",
    "prompt/t2i/request_params.json",
    "prompt/edit/task_prompt.txt",
)
PROMPT_BYTES = 56_282
PROMPT_MIN_FREE_BYTES = 1 * 10**9
# The Qwen3.8-27B block drafts (SW-7204): the catalog's soft co-requisite rows of both the describer
# and the prompt expander (config/manifests/builtin.models.jsonc), each repo's pinned revision, the
# three files its row names and their exact total (`footprint.diskSizeBytes`).
DRAFT_FILES = ("config.json", "model.safetensors", "LICENSE")
DRAFTS = (
    ("SceneWorks/qwen3.8-27b-dflash2", "9030fabfb5e3ca0c5c180f84927af86aa9f9a2e9", 3_848_830_679),
    ("SceneWorks/qwen3.8-27b-dspine-b16", "8e450d7faf38238b64580163a7d3dd7e44af759d", 4_051_845_726),
)
# Both drafts are ~7.9 GB; keep headroom for the volume's other tenants.
DRAFT_MIN_FREE_BYTES = 20 * 10**9
MODES = ("quantised", "bf16", "expander-q4", "expander-q8", "expander-abliterated-q4",
         "expander-abliterated-q8", "drafts")


def tier_patterns(folder, tier_files=TIER_FILES):
    """The catalog row's selectors for the tier in `folder`."""
    return [f"{folder}/{name}" for name in tier_files]


def expected_files(entries, tier_bytes=TIER_BYTES, tier_folders=TIER_FOLDERS, tier_files=TIER_FILES):
    """The repo's files each tier's selectors match, as {path: size}: every named file present and
    sized, at least one shard, and each tier's total equal to the catalog's."""
    expected = {}
    for tier, total in tier_bytes.items():
        folder = tier_folders[tier]
        patterns = tier_patterns(folder, tier_files)
        files = {name: size for name, size in entries
                 if any(fnmatchcase(name, pattern) for pattern in patterns)}
        unmatched = [pattern for pattern in patterns if not any(fnmatchcase(name, pattern) for name in files)]
        if not files or unmatched or any(not isinstance(size, int) or size <= 0 for size in files.values()):
            raise RuntimeError(f"{REPO}@{REVISION} lists no sized files under {folder}/"
                               + (f" for {', '.join(unmatched)}" if unmatched else ""))
        if sum(files.values()) != total:
            raise RuntimeError(
                f"{folder}/ at {REVISION} totals {sum(files.values())} bytes, the catalog says {total}")
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


def expander_expected_files(entries, tier, tier_bytes=EXPANDER_TIER_BYTES,
                            tier_folders=EXPANDER_TIER_FOLDERS):
    """Every file of the expander tier's folder, as {path: size}, with the folder's total equal to
    the catalog's."""
    folder = tier_folders[tier]
    files = {name: size for name, size in entries if name.startswith(f"{folder}/")}
    if not files or any(not isinstance(size, int) or size <= 0 for size in files.values()):
        raise RuntimeError(f"{EXPANDER_REPO}@{EXPANDER_REVISION} lists no sized files under {folder}/")
    if sum(files.values()) != tier_bytes[tier]:
        raise RuntimeError(f"{folder}/ at {EXPANDER_REVISION} totals {sum(files.values())} bytes, "
                           f"the catalog says {tier_bytes[tier]}")
    return files


def prompt_expected_files(entries, names=PROMPT_FILES, total=PROMPT_BYTES):
    """The upstream prompt files the co-requisite names, as {path: size}, totalling the catalog's."""
    sized = {name: size for name, size in entries}
    missing = [name for name in names if not isinstance(sized.get(name), int) or sized[name] <= 0]
    if missing:
        raise RuntimeError(f"{PROMPT_REPO}@{PROMPT_REVISION} lists no sized {', '.join(missing)}")
    expected = {name: sized[name] for name in names}
    if sum(expected.values()) != total:
        raise RuntimeError(f"prompt files at {PROMPT_REVISION} total {sum(expected.values())} bytes, "
                           f"the catalog says {total}")
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
                [pattern for tier in TIER_BYTES for pattern in tier_patterns(TIER_FOLDERS[tier])],
                MIN_FREE_BYTES)
    if mode == "bf16":
        return (BF16_REPO, BF16_REVISION, "full precision (bf16)", bf16_expected_files,
                [*BF16_FILES, BF16_SHARD_PATTERN], BF16_MIN_FREE_BYTES)
    raise RuntimeError(f"unknown provisioning mode {mode!r}; use one of {', '.join(MODES)}")


def draft_expected_files(entries, repo, total, names=DRAFT_FILES):
    """The draft files its catalog row names, as {path: size}, totalling the catalog's."""
    sized = {name: size for name, size in entries}
    missing = [name for name in names if not isinstance(sized.get(name), int) or sized[name] <= 0]
    if missing:
        raise RuntimeError(f"{repo} lists no sized {', '.join(missing)}")
    expected = {name: sized[name] for name in names}
    if sum(expected.values()) != total:
        raise RuntimeError(f"{repo} files total {sum(expected.values())} bytes, the catalog says {total}")
    return expected


def draft_plans():
    """The two block drafts every mode provisions, each in `plan`'s shape."""
    return [
        (repo, revision, f"block draft {repo.split('/')[-1]}",
         lambda entries, repo=repo, total=total: draft_expected_files(entries, repo, total),
         list(DRAFT_FILES), DRAFT_MIN_FREE_BYTES)
        for repo, revision, total in DRAFTS
    ]


def prompt_plan():
    """The upstream prompt files every expander mode provisions, in `plan`'s shape."""
    return (PROMPT_REPO, PROMPT_REVISION, "expander prompt files (prompt/)", prompt_expected_files,
            list(PROMPT_FILES), PROMPT_MIN_FREE_BYTES)


def plans(mode):
    """Every snapshot `mode` provisions, each in `plan`'s shape, then the two block drafts
    (SW-7204): a describer mode is one snapshot; an expander tier is its Candle folder, then the
    upstream prompt files; an abliterated expander tier is ONE describer tier folder, then the same
    prompt files; `drafts` is the drafts alone."""
    if mode == "drafts":
        return draft_plans()
    return weight_plans(mode) + draft_plans()


def weight_plans(mode):
    """The weights and prompt files `mode` provisions, without the drafts."""
    described = mode.removeprefix("expander-abliterated-")
    if mode.startswith("expander-abliterated-") and described in TIER_BYTES:
        folder = TIER_FOLDERS[described]
        return [
            (REPO, REVISION, f"abliterated expander tier {described} ({folder}/)",
             lambda entries: expected_files(entries, {described: TIER_BYTES[described]}),
             tier_patterns(folder), EXPANDER_MIN_FREE_BYTES[described]),
            prompt_plan(),
        ]
    tier = mode.removeprefix("expander-")
    if mode.startswith("expander-") and tier in EXPANDER_TIER_FOLDERS:
        folder = EXPANDER_TIER_FOLDERS[tier]
        return [
            (EXPANDER_REPO, EXPANDER_REVISION, f"expander tier {tier} ({folder}/)",
             lambda entries: expander_expected_files(entries, tier), [f"{folder}/*"],
             EXPANDER_MIN_FREE_BYTES[tier]),
            prompt_plan(),
        ]
    return [plan(mode)]


def main():
    cache = Path(os.environ.get("HF_HUB_CACHE", ""))
    if not cache.is_absolute() or not cache.is_dir():
        raise RuntimeError(f"HF_HUB_CACHE must name an existing absolute hub directory, got {str(cache)!r}")
    mode = os.environ.get("SCENEWORKS_QWEN38_DESCRIBER_PROVISION", "").strip() or "quantised"
    for part in plans(mode):
        provision_snapshot(cache, *part)
    return 0


def provision_snapshot(cache, repo, revision, label, read_expected, patterns, min_free):
    """Provision one pinned snapshot; nothing is downloaded when every file is present at its
    published size."""
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
        return
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


if __name__ == "__main__":
    sys.exit(main())
