"""Read-only Hugging Face cache metadata inventory (sc-26584 safety-impact diagnostic).

Records lstat metadata for a FIXED, source-derived allowlist of fixture-named paths that the failed
canonical run 39687007308 (source f203ec37) API tests could have touched when they ran without a
CacheRootGuard. It never writes inside the cache, never follows an intermediate symlink, never
lists, recurses, reads or hashes any file, and prints no environment beyond the three HF root
variables. Metadata only; it is not product evidence and says nothing about what the old test
process actually did.

Usage: python3.12 -I cache_inventory.py --output-dir <NEW directory under $RUNNER_TEMP>
"""

import argparse
import datetime
import errno
import json
import os
import re
import stat
import sys

SCHEMA_VERSION = 1
FAILED_RUN_ID = "39687007308"
FAILED_SOURCE_SHA = "f203ec37f49386fd5043c62333adb87fbb4c1550"
REPORT_NAME = "cache-inventory.json"
SAFE_LINK_TEXT = re.compile(r"[A-Za-z0-9._/-]+")  # no scheme, query, userinfo or spaces
MAX_LINK_TEXT = 256  # in-root link text only; longer or outside-root text is redacted

# Root resolution mirrors sceneworks_core::hf_home::huggingface_hub_cache_dir at f203, minus the
# cfg(test-support) CacheRootGuard and the final `<data_dir>/cache/huggingface/hub` fallback: that
# fallback is a per-test temporary directory, not a persistent cache, so it resolves to UNKNOWN.
ROOT_ENV_ORDER = (
    ("HF_HUB_CACHE", ""),
    ("HUGGINGFACE_HUB_CACHE", ""),
    ("HF_HOME", "hub"),
)

# Fixture repos (apps/rust-api/src/models.rs variant_delete_tests at f203) and the exact fixture
# names their `seed`/`blob`/`link` helpers create. Nothing else is ever inspected.
IRIS_REPO = "models--speridlabs--iris-3b"
QWEN_REPO = "models--Qwen--Qwen3-VL-4B-Instruct"
# sc-25682 seeds blob0..blob4 (5 files); sc-25684 seeds blob0..3,5,6 + one shared empty-prompt-blob
# (8 files). The union is blob0..blob6 + empty-prompt-blob; blob7 is never named by either helper.
IRIS_BLOBS = tuple(f"blob{i}" for i in range(7)) + ("empty-prompt-blob",)
IRIS_SNAPSHOT_FILES = (
    "config.yaml",
    "model.safetensors",
    "depth/config.yaml",
    "depth/model.safetensors",
    "depth/empty_prompt.safetensors",
    "upscaler/config.yaml",
    "upscaler/model.safetensors",
    "upscaler/empty_prompt.safetensors",
)


# f8c46 variant_delete_tests fixture repos, from the tests' own `repo_name` /
# `huggingface_repo_cache_path` literals. The adapter repo is a REAL Hugging Face repo name; only
# the tests' fixture-named blobs/links (literal snapshots/rev) are listed, never a real revision.
MINIMAX_REPO = "models--SceneWorks--minimax-h3-mlx"
ADAPTER_REPO = "models--ostris--zimage_turbo_training_adapter"
SOLO_REPO = "models--Org--solo-model"
PAIR_REPO = "models--SceneWorks--whole-repo-pair"
# catalog.rs fixtures (f8c46): literal repos and the literal revisions the tests write.
LORA_PROBE_REPO = "models--Test--lora-install-probe"
LORA_PROBE_REVISION = "0123456789abcdef0123456789abcdef01234567"
OWNER_MODEL_REPO = "models--owner--model"
PARTIAL_MODEL_REPO = "models--partial--model"
CACHED_MODEL_REPO = "models--owner--cached-model"
TIERED_MODEL_REPO = "models--owner--tiered-model"


def _repo_tree(repo, dirs, blobs, links):
    """Fixed repo-level directories, blob names and snapshots/rev links for one fixture repo."""
    paths = [repo, f"{repo}/blobs", f"{repo}/snapshots", f"{repo}/snapshots/rev"]
    paths += [f"{repo}/snapshots/rev/{name}" for name in dirs]
    paths += [f"{repo}/blobs/{name}" for name in blobs]
    paths += [f"{repo}/snapshots/rev/{name}" for name in links]
    return tuple(paths)


def _build_groups():
    f203 = []
    for repo in (IRIS_REPO, QWEN_REPO):
        f203 += [repo, f"{repo}/blobs", f"{repo}/snapshots", f"{repo}/snapshots/rev"]
    f203 += [f"{IRIS_REPO}/snapshots/rev/depth", f"{IRIS_REPO}/snapshots/rev/upscaler"]
    f203 += [f"{IRIS_REPO}/blobs/{name}" for name in IRIS_BLOBS]
    f203 += [f"{IRIS_REPO}/snapshots/rev/{name}" for name in IRIS_SNAPSHOT_FILES]
    f203 += [f"{QWEN_REPO}/blobs/encoder-config", f"{QWEN_REPO}/snapshots/rev/config.json"]

    # seed_minimax_tier(repo, tier, base, ref, size) is called with ("q4","q4dit","q4refdit") and
    # ("bf16","bf16dit","bf16refdit"): a shared `<tier>-config` blob, `<etag>-index` and `<etag>`
    # blobs per partition, and three snapshot links per tier/partition.
    tiers = (("q4", "q4dit", "q4refdit"), ("bf16", "bf16dit", "bf16refdit"))
    partitions = ("transformer", "transformer_ref")
    dirs, blobs, links = [], [], []
    for tier, base_etag, ref_etag in tiers:
        dirs += [tier] + [f"{tier}/{part}" for part in partitions]
        blobs += [f"{tier}-config", f"{base_etag}-index", base_etag, f"{ref_etag}-index", ref_etag]
        for part in partitions:
            links += [
                f"{tier}/{part}/config.json",
                f"{tier}/{part}/diffusion_pytorch_model.safetensors.index.json",
                f"{tier}/{part}/diffusion_pytorch_model-00001-of-00001.safetensors",
            ]
    return (
        ("f203_iris_qwen", tuple(f203)),
        ("f8c46_minimax_h3_shared_repo", _repo_tree(MINIMAX_REPO, dirs, blobs, links)),
        (
            "f8c46_zimage_training_adapter",
            _repo_tree(
                ADAPTER_REPO,
                [],
                ["v1blob", "v2blob"],
                [
                    "zimage_turbo_training_adapter_v1.safetensors",
                    "zimage_turbo_training_adapter_v2.safetensors",
                ],
            ),
        ),
        (
            "f8c46_exclusive_solo_model",
            _repo_tree(
                SOLO_REPO,
                ["q4", "q4/transformer"],
                ["q4dit", "readme"],
                ["q4/transformer/model.safetensors", "README.md"],
            ),
        ),
        (
            "f8c46_unscoped_whole_repo_pair",
            _repo_tree(PAIR_REPO, [], ["dit"], ["model.safetensors"]),
        ),
        # apps/rust-api/src/tests/catalog.rs at f8c46. These write plain files/dirs (no blobs/).
        (
            "f8c46_catalog_lora_install_probe",
            (
                LORA_PROBE_REPO,
                f"{LORA_PROBE_REPO}/refs",
                f"{LORA_PROBE_REPO}/refs/main",  # WRITTEN with the revision literal; lstat only
                f"{LORA_PROBE_REPO}/snapshots",
                f"{LORA_PROBE_REPO}/snapshots/{LORA_PROBE_REVISION}",
                f"{LORA_PROBE_REPO}/snapshots/{LORA_PROBE_REVISION}/cached.safetensors",
            ),
        ),
        # DELETE base-model?permanent=true resolves default download owner/model; nothing written.
        ("f8c46_catalog_overlay_delete_owner_model", (OWNER_MODEL_REPO,)),
        # DELETE partial_model resolves partial/model; nothing written into the cache.
        ("f8c46_catalog_partial_trash_model", (PARTIAL_MODEL_REPO,)),
        (
            "f8c46_catalog_resolved_cache_cached_model",
            (
                CACHED_MODEL_REPO,
                f"{CACHED_MODEL_REPO}/snapshots",
                f"{CACHED_MODEL_REPO}/snapshots/{'a' * 40}",
                f"{CACHED_MODEL_REPO}/snapshots/{'a' * 40}/model.safetensors",
            ),
        ),
        (
            "f8c46_catalog_resolved_cache_tiered_model",
            (
                TIERED_MODEL_REPO,
                f"{TIERED_MODEL_REPO}/snapshots",
                f"{TIERED_MODEL_REPO}/snapshots/{'b' * 40}",
                f"{TIERED_MODEL_REPO}/snapshots/{'b' * 40}/q8.safetensors",
                f"{TIERED_MODEL_REPO}/snapshots/{'b' * 40}/q4.safetensors",
            ),
        ),
    )


GROUPS = _build_groups()
ALLOWLIST = tuple(path for _, paths in GROUPS for path in paths)
GROUP_OF = {path: name for name, paths in GROUPS for path in paths}
GROUP_COUNTS = {name: len(paths) for name, paths in GROUPS}
assert len(ALLOWLIST) == len(set(ALLOWLIST)) == 101, len(ALLOWLIST)
assert GROUP_COUNTS["f203_iris_qwen"] == 28
MAX_REPORT_BYTES = 32 * 1024

LIMITATIONS = (
    "Metadata only (lstat/readlink); not product evidence and no host-impact claim by itself.",
    "This job's ambient HF_* values may differ from the transient environment of the old test "
    "process; the old process root, iteration and changed files were never logged.",
    "Absence of a path is not evidence of deletion, and presence is not evidence of integrity: "
    "no file contents or hashes are read.",
    "Only fixture-named paths (literal snapshots/rev, fixed blob names) are checked; a deletion "
    "against a real non-'rev' snapshot of the same repos is outside this allowlist.",
    "Provenance: groups f203_* come from the f203ec37 Iris/Qwen tests and f8c46_* from the f8c46 "
    "variant_delete_tests fixtures (minimax shared repo, zimage adapter, solo model, whole-repo "
    "pair) plus the five catalog.rs tests (lora probe, overlay delete, partial trash, two resolved-cache "
    "deletes). Behavior is source-derived; nothing proves the old test processes used this root.",
    "The zimage adapter and minimax repo names are real Hugging Face repos: real revisions or "
    "files in them are NOT inventoried, so real-install state and pre-state are unknown.",
    "Without HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE or HF_HOME the old tests resolve to a temporary "
    "data_dir; no persistent root is resolvable here and the result is UNKNOWN (nothing scanned).",
)


def _iso(ns):
    return datetime.datetime.fromtimestamp(ns / 1e9, datetime.timezone.utc).isoformat()


def _kind(mode):
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


def _errname(exc):
    return errno.errorcode.get(exc.errno, "OSERROR")


def _classify(exc):
    if exc.errno == errno.ENOENT:
        return "absent"
    if exc.errno in (errno.EACCES, errno.EPERM):
        return "inaccessible"
    return "unknown"


def _root_problem(root, home):
    if "\x00" in root:
        return "NUL in root"
    if not os.path.isabs(root):
        return "root is not absolute"
    if ".." in root.split(os.sep):
        return "root contains '..'"
    normal = os.path.normpath(root)
    if len([p for p in normal.split(os.sep) if p]) < 3:
        return "overbroad root (fewer than 3 path components)"
    if home and normal == os.path.normpath(home):
        return "overbroad root (equals home directory)"
    return None


def resolve_root(env, home=None):
    """Return (root_or_None, provenance). Never touches the filesystem."""
    sources = []
    chosen = None
    for name, suffix in ROOT_ENV_ORDER:
        value = (env.get(name) or "").strip()
        sources.append({"variable": name, "set": bool(value)})  # never the raw value
        if value and chosen is None:
            chosen = (name, value, suffix)
    provenance = {
        "precedence": [name for name, _ in ROOT_ENV_ORDER],
        "sources": sources,
        "excluded": "CacheRootGuard (test-only) and the temporary <data_dir>/cache fallback",
    }
    if chosen is None:
        provenance.update(chosen_variable=None, status="UNKNOWN", reason="no HF root variable set")
        return None, provenance
    name, value, suffix = chosen
    root = os.path.join(value, suffix) if suffix else value
    provenance.update(chosen_variable=name)
    problem = _root_problem(root, home)
    if problem:
        provenance.update(status="REJECTED", reason=problem)
        return None, provenance
    provenance["status"] = "RESOLVED"
    normal = os.path.normpath(root)
    provenance["root"] = normal  # only a valid absolute filesystem path is ever echoed
    if os.path.basename(normal) != "hub":
        provenance["warning"] = "root basename is not 'hub'"
    return normal, provenance


def _meta(st):
    return {
        "type": _kind(st.st_mode),
        "mode": oct(stat.S_IMODE(st.st_mode)),
        "size": st.st_size,
        "mtime": _iso(st.st_mtime_ns),
    }


def inspect_path(resolved_root, relative):
    """lstat one allowlisted relative path without following any symlink."""
    entry = {"path": relative, "group": GROUP_OF.get(relative)}
    current = resolved_root
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = os.path.join(current, part)
        last = index == len(parts) - 1
        try:
            st = os.lstat(current)
        except OSError as exc:
            entry["status"] = _classify(exc)
            entry["error"] = _errname(exc)
            if not last:
                entry["stopped_at"] = "/".join(parts[: index + 1])
            return entry
        kind = _kind(st.st_mode)
        if not last:
            if kind != "directory":
                entry["status"] = (
                    "blocked_by_symlink" if kind == "symlink" else "blocked_not_directory"
                )
                entry["stopped_at"] = "/".join(parts[: index + 1])
                return entry
            continue
        entry["status"] = "present"
        entry.update(_meta(st))
        if kind == "symlink":
            try:
                target = os.readlink(current)
            except OSError as exc:
                entry["readlink_error"] = _errname(exc)
            else:
                lexical = os.path.normpath(os.path.join(os.path.dirname(current), target))
                within = lexical == resolved_root or lexical.startswith(resolved_root + os.sep)
                entry["target_lexically_within_root"] = within
                # Target text outside the cache root may carry arbitrary URI/credential text.
                if within and len(target) <= MAX_LINK_TEXT and SAFE_LINK_TEXT.fullmatch(target):
                    entry["readlink"] = target
                else:
                    entry["readlink_redacted"] = True
    return entry


def inspect_root(root):
    """Return (root_record, resolved_root_or_None). Symlinked roots are resolved once, here."""
    record = {"path": root}
    try:
        st = os.lstat(root)
    except OSError as exc:
        record["status"] = _classify(exc)
        record["error"] = _errname(exc)
        return record, None
    record.update(_meta(st))
    if _kind(st.st_mode) == "symlink":
        record["readlink_redacted"] = True
    resolved = os.path.realpath(root)
    record["resolved_root"] = resolved
    problem = _root_problem(resolved, None)
    if problem:
        record["status"] = "REJECTED"
        record["reason"] = f"resolved {problem}"
        return record, None
    if not os.path.isdir(resolved):
        record["status"] = "not_directory"
        return record, None
    record["status"] = "present"
    return record, resolved


def build_report(env, now=None, home=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    root, provenance = resolve_root(env, home)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sceneworks-cache-metadata-inventory",
        "generated_at": now.isoformat(),
        "diagnosed_failed_run_id": FAILED_RUN_ID,
        "diagnosed_failed_source_sha": FAILED_SOURCE_SHA,
        "diagnostic": {
            "repository": env.get("GITHUB_REPOSITORY"),
            "run_id": env.get("GITHUB_RUN_ID"),
            "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
            "sha": env.get("GITHUB_SHA"),
            "ref": env.get("GITHUB_REF"),
            "runner_name": env.get("RUNNER_NAME"),
        },
        "resolver": provenance,
        "allowlist_count": len(ALLOWLIST),
        "limitations": list(LIMITATIONS),
        "root": None,
        "entries": [],
    }
    if root is None:
        report["root_status"] = provenance["status"]
        return report
    record, resolved = inspect_root(root)
    report["root"] = record
    if resolved is None:
        report["root_status"] = record["status"]
        return report
    report["root_status"] = "RESOLVED"
    report["entries"] = [inspect_path(resolved, rel) for rel in ALLOWLIST]
    groups = {name: {"count": count} for name, count in GROUP_COUNTS.items()}
    for entry in report["entries"]:
        tally = groups[entry["group"]]
        tally[entry["status"]] = tally.get(entry["status"], 0) + 1
    report["groups"] = groups
    return report


def validate_output_dir(output_dir, runner_temp, cache_root):
    """The report directory must be NEW, directly under RUNNER_TEMP, and disjoint from the cache."""
    if not runner_temp or not os.path.isabs(runner_temp):
        raise SystemExit("RUNNER_TEMP must be an absolute path")
    if not os.path.isabs(output_dir):
        raise SystemExit("output directory must be absolute")
    temp = os.path.realpath(runner_temp)
    target = os.path.normpath(output_dir)
    parent = os.path.realpath(os.path.dirname(target))
    if parent != temp and not parent.startswith(temp + os.sep):
        raise SystemExit("output directory must be under RUNNER_TEMP")
    final = os.path.join(parent, os.path.basename(target))
    if cache_root:
        real_cache = os.path.realpath(cache_root)
        if (
            final == real_cache
            or final.startswith(real_cache + os.sep)
            or real_cache.startswith(final + os.sep)
        ):
            raise SystemExit("output directory overlaps the cache root")
    return final


def _serialize(report, enforce_cap=True):
    text = json.dumps(report, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    if enforce_cap and len(text.encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("report exceeds the size cap")
    return text


def main(argv=None, env=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    env = os.environ if env is None else env
    root, _ = resolve_root(env, env.get("HOME"))
    output = validate_output_dir(args.output_dir, env.get("RUNNER_TEMP"), root)
    os.mkdir(output, 0o700)  # exclusive: fails if it already exists
    try:
        report = build_report(env, home=env.get("HOME"))
        status = 0 if report["root_status"] in ("RESOLVED", "UNKNOWN") else 3
        text = _serialize(report)
    except Exception as exc:  # the artifact must exist even when the diagnostic fails
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "sceneworks-cache-metadata-inventory",
            "root_status": "ERROR",
            "error_type": type(exc).__name__,
            "diagnosed_failed_run_id": FAILED_RUN_ID,
            "diagnosed_failed_source_sha": FAILED_SOURCE_SHA,
            "limitations": list(LIMITATIONS),
            "entries": [],
        }
        status = 1
        text = _serialize(report, enforce_cap=False)  # fixed small error shape
    with open(os.path.join(output, REPORT_NAME), "x", encoding="utf-8") as handle:
        handle.write(text)
    print(f"{REPORT_NAME}: root_status={report['root_status']} entries={len(report['entries'])}")
    return status


if __name__ == "__main__":
    sys.exit(main())
