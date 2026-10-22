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
import stat
import sys

SCHEMA_VERSION = 1
FAILED_RUN_ID = "39687007308"
FAILED_SOURCE_SHA = "f203ec37f49386fd5043c62333adb87fbb4c1550"
REPORT_NAME = "cache-inventory.json"

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


def _build_allowlist():
    paths = []
    for repo in (IRIS_REPO, QWEN_REPO):
        paths += [repo, f"{repo}/blobs", f"{repo}/snapshots", f"{repo}/snapshots/rev"]
    paths += [f"{IRIS_REPO}/snapshots/rev/depth", f"{IRIS_REPO}/snapshots/rev/upscaler"]
    paths += [f"{IRIS_REPO}/blobs/{name}" for name in IRIS_BLOBS]
    paths += [f"{IRIS_REPO}/snapshots/rev/{name}" for name in IRIS_SNAPSHOT_FILES]
    paths += [f"{QWEN_REPO}/blobs/encoder-config", f"{QWEN_REPO}/snapshots/rev/config.json"]
    return tuple(paths)


ALLOWLIST = _build_allowlist()
assert len(ALLOWLIST) == 28 and len(set(ALLOWLIST)) == 28

LIMITATIONS = (
    "Metadata only (lstat/readlink); not product evidence and no host-impact claim by itself.",
    "This job's ambient HF_* values may differ from the transient environment of the old test "
    "process; the old process root, iteration and changed files were never logged.",
    "Absence of a path is not evidence of deletion, and presence is not evidence of integrity: "
    "no file contents or hashes are read.",
    "Only fixture-named paths (literal snapshots/rev, fixed blob names) are checked; a deletion "
    "against a real non-'rev' snapshot of the same repos is outside this allowlist.",
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
        sources.append({"variable": name, "set": bool(value), "value": value or None})
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
    provenance.update(chosen_variable=name, unresolved_root=root)
    problem = _root_problem(root, home)
    if problem:
        provenance.update(status="REJECTED", reason=problem)
        return None, provenance
    provenance["status"] = "RESOLVED"
    normal = os.path.normpath(root)
    if os.path.basename(normal) != "hub":
        provenance["warning"] = "root basename is not 'hub'"
    return normal, provenance


def _meta(st):
    return {
        "type": _kind(st.st_mode),
        "mode": oct(stat.S_IMODE(st.st_mode)),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "mtime": _iso(st.st_mtime_ns),
    }


def inspect_path(resolved_root, relative):
    """lstat one allowlisted relative path without following any symlink."""
    entry = {"path": relative}
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
                entry["readlink"] = target
                lexical = os.path.normpath(os.path.join(os.path.dirname(current), target))
                entry["target_lexically_within_root"] = lexical == resolved_root or lexical.startswith(
                    resolved_root + os.sep
                )
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
        try:
            record["readlink"] = os.readlink(root)
        except OSError:
            pass
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
    with open(os.path.join(output, REPORT_NAME), "x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"{REPORT_NAME}: root_status={report['root_status']} entries={len(report['entries'])}")
    return status


if __name__ == "__main__":
    sys.exit(main())
