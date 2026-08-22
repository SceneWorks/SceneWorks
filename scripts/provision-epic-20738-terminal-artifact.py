#!/usr/bin/env python3
"""Audit, stage, and resolve immutable epic-20738 artifacts.

The trusted runner cache is always audited first. Each distinct reviewed authority is then copied
once into a campaign-owned staging tree and reused by every cell. Network access is impossible
except while staging the one frozen missing FLUX Schnell q8 file; that download names one exact
repo/revision/filename and lands only in fresh campaign staging. Runtime resolution is fully offline.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path

SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
NETWORK_FILL_AUTHORITY = (
    "SceneWorks/flux1-schnell-mlx",
    "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
    "q8/transformer/model.safetensors",
)


def parse_request(path: Path) -> dict:
    request = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "repository", "revision", "subdirectory", "allowPatterns"}
    if set(request) != required:
        raise ValueError(f"artifact request fields must be exactly {sorted(required)}")
    if not REPOSITORY.fullmatch(request["repository"]):
        raise ValueError("repository must be an exact owner/name identity")
    if not SHA40.fullmatch(request["revision"]):
        raise ValueError("revision must be an exact lowercase 40-hex commit")
    patterns = request["allowPatterns"]
    if not isinstance(patterns, list) or not patterns or not all(
        isinstance(pattern, str) and pattern for pattern in patterns
    ):
        raise ValueError("allowPatterns must be a non-empty string array")
    if any(
        ".." in Path(pattern).parts or Path(pattern).is_absolute() for pattern in patterns
    ):
        raise ValueError("allowPatterns cannot escape the exact snapshot")
    subdirectory = Path(request["subdirectory"])
    if subdirectory.is_absolute() or ".." in subdirectory.parts:
        raise ValueError("subdirectory cannot escape the exact snapshot")
    return request


def _is_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & REPARSE_POINT
    )


def _inside(root: Path, candidate: Path, *, allow_equal: bool = False) -> bool:
    if allow_equal and candidate == root:
        return True
    try:
        candidate.relative_to(root)
        return candidate != root
    except ValueError:
        return False


def _ordinary_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} is missing: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode) or _is_reparse(metadata):
        raise RuntimeError(
            f"{label} must be an ordinary directory, not a symlink/reparse point: {path}"
        )
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise RuntimeError(f"{label} traverses a symlink/reparse point: {path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context(request: dict, cache_root: Path) -> tuple[Path, Path, Path]:
    if not cache_root.is_absolute():
        raise ValueError("cache root must be an absolute path")
    trusted_root = _ordinary_directory(cache_root, "trusted cache root")
    owner, name = request["repository"].split("/", 1)
    repository_root = _ordinary_directory(
        trusted_root / f"models--{owner}--{name}", "exact cached repository"
    )
    if not _inside(trusted_root, repository_root):
        raise RuntimeError("cached repository escaped the trusted cache root")
    snapshots = _ordinary_directory(repository_root / "snapshots", "cached snapshots directory")
    snapshot = _ordinary_directory(
        snapshots / request["revision"],
        f"exact cached revision {request['revision']}",
    )
    if snapshot.parent != snapshots or not _inside(trusted_root, snapshot):
        raise RuntimeError("exact cached revision escaped the trusted cache root")
    selected = _ordinary_directory(
        snapshot / request["subdirectory"], "selected cached artifact subdirectory"
    )
    if not _inside(snapshot, selected, allow_equal=True):
        raise RuntimeError("selected artifact subdirectory escaped the exact snapshot")
    return trusted_root, snapshot, selected


def _reviewed_missing(request: dict) -> list[str]:
    repository, revision, filename = NETWORK_FILL_AUTHORITY
    if (
        request["repository"] == repository
        and request["revision"] == revision
        and request["subdirectory"] == "q8"
        and request["allowPatterns"] == ["q8/*"]
    ):
        return [filename]
    return []


def _walk_files(
    selected: Path,
    snapshot: Path,
    trusted_root: Path,
    allowed_broken: set[str],
) -> tuple[list[tuple[Path, str, int]], set[str]]:
    files: list[tuple[Path, str, int]] = []
    broken: set[str] = set()
    pending = [selected]
    while pending:
        directory = pending.pop()
        if not _inside(snapshot, directory, allow_equal=True):
            raise RuntimeError(f"cached artifact directory escaped the exact snapshot: {directory}")
        for entry in sorted(os.scandir(directory), key=lambda candidate: candidate.name):
            lexical = Path(entry.path)
            relative_snapshot = lexical.relative_to(snapshot).as_posix()
            if lexical.name.endswith(".incomplete"):
                raise RuntimeError(
                    f"cached artifact contains an untrusted incomplete file: {lexical}"
                )
            metadata = lexical.lstat()
            try:
                resolved = lexical.resolve(strict=True)
            except FileNotFoundError as error:
                if relative_snapshot in allowed_broken and _is_reparse(metadata):
                    broken.add(relative_snapshot)
                    continue
                raise RuntimeError(
                    f"cached artifact contains a broken symlink/reparse point: {lexical}"
                ) from error
            resolved_metadata = resolved.stat()
            if stat.S_ISDIR(metadata.st_mode) and not _is_reparse(metadata):
                if not _inside(snapshot, resolved):
                    raise RuntimeError(
                        f"cached artifact directory escaped the exact snapshot: {lexical}"
                    )
                pending.append(resolved)
            elif stat.S_ISREG(metadata.st_mode) or _is_reparse(metadata):
                if not stat.S_ISREG(resolved_metadata.st_mode):
                    raise RuntimeError(
                        f"cached artifact reparse target is not a regular file: {lexical}"
                    )
                if not _inside(trusted_root, resolved):
                    raise RuntimeError(
                        f"cached artifact file escaped the trusted cache root: {lexical}"
                    )
                if resolved_metadata.st_size < 1:
                    raise RuntimeError(f"cached artifact contains an empty file: {lexical}")
                files.append((lexical, relative_snapshot, resolved_metadata.st_size))
            else:
                raise RuntimeError(f"cached artifact contains a non-regular entry: {lexical}")
    return sorted(files, key=lambda item: item[1]), broken


def audit_cached_artifact(request: dict, cache_root: Path) -> dict:
    trusted_root, snapshot, selected = _context(request, cache_root)
    reviewed_missing = _reviewed_missing(request)
    files, broken = _walk_files(
        selected, snapshot, trusted_root, set(reviewed_missing)
    )
    matched_by_pattern = {
        pattern: [
            relative
            for _, relative, _ in files
            if fnmatch.fnmatchcase(relative, pattern)
        ]
        for pattern in request["allowPatterns"]
    }
    missing_patterns = [
        pattern for pattern, matches in matched_by_pattern.items() if not matches
    ]
    if missing_patterns:
        raise RuntimeError(
            "exact cached revision is incomplete for allow-pattern(s): "
            + ", ".join(missing_patterns)
        )
    present = {relative for _, relative, _ in files}
    missing = sorted(
        relative
        for relative in reviewed_missing
        if relative not in present or relative in broken
    )
    matched_records = []
    for lexical, relative, size in files:
        if any(
            fnmatch.fnmatchcase(relative, pattern)
            for pattern in request["allowPatterns"]
        ):
            matched_records.append(
                {
                    "path": lexical.relative_to(selected).as_posix(),
                    "bytes": size,
                    "sha256": _sha256(lexical.resolve(strict=True)),
                }
            )
    matched_records.sort(key=lambda record: record["path"])
    return {
        "id": request["id"],
        "cacheRoot": str(trusted_root),
        "snapshotRoot": str(snapshot),
        "selectedRoot": str(selected),
        "complete": not missing,
        "missingFiles": missing,
        "matchedFiles": [record["path"] for record in matched_records],
        "selectedFiles": sorted(
            lexical.relative_to(selected).as_posix() for lexical, _, _ in files
        ),
        "reusedFiles": matched_records,
        "downloadedFiles": [],
    }


def resolve_cached_artifact(request: dict, cache_root: Path) -> dict:
    result = audit_cached_artifact(request, cache_root)
    if not result["complete"]:
        raise RuntimeError(
            "exact cached revision is missing reviewed required file(s): "
            + ", ".join(result["missingFiles"])
        )
    return result


def _stage_root(staging_root: Path) -> Path:
    if not staging_root.is_absolute():
        raise ValueError("staging root must be an absolute path")
    return _ordinary_directory(staging_root, "campaign artifact staging root")


def stage_artifact(
    request: dict,
    cache_root: Path,
    staging_root: Path,
    allow_reviewed_download: bool,
) -> dict:
    audit = audit_cached_artifact(request, cache_root)
    reviewed = list(NETWORK_FILL_AUTHORITY)
    if audit["missingFiles"] and (
        not allow_reviewed_download
        or [
            request["repository"],
            request["revision"],
            *audit["missingFiles"],
        ]
        != reviewed
    ):
        raise RuntimeError(
            "network staging is not approved for missing set: "
            + ", ".join(audit["missingFiles"])
        )
    if allow_reviewed_download and audit["missingFiles"] != [NETWORK_FILL_AUTHORITY[2]]:
        raise RuntimeError("network staging requires exactly the sole reviewed missing filename")

    destination_root = _stage_root(staging_root)
    owner, name = request["repository"].split("/", 1)
    destination_snapshot = (
        destination_root
        / f"models--{owner}--{name}"
        / "snapshots"
        / request["revision"]
    )
    destination_selected = destination_snapshot / request["subdirectory"]
    destination_selected.mkdir(parents=True, exist_ok=True)

    for record in audit["reusedFiles"]:
        source = Path(audit["selectedRoot"]) / record["path"]
        destination = destination_selected / record["path"]
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"refusing to overwrite staged cache file: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.resolve(strict=True), destination)
        if destination.stat().st_size != record["bytes"] or _sha256(destination) != record["sha256"]:
            raise RuntimeError(f"staged cache copy failed byte verification: {destination}")

    downloaded_files = []
    if audit["missingFiles"]:
        repository, revision, filename = NETWORK_FILL_AUTHORITY
        destination = destination_snapshot / filename
        if destination.exists() or destination.is_symlink():
            raise RuntimeError("refusing to overwrite the reviewed missing staging target")
        destination.parent.mkdir(parents=True, exist_ok=True)

        from huggingface_hub import (
            __version__,
            get_hf_file_metadata,
            hf_hub_download,
            hf_hub_url,
        )

        if __version__ != "0.36.0":
            raise RuntimeError(
                f"network staging requires huggingface_hub 0.36.0, got {__version__}"
            )
        url = hf_hub_url(repo_id=repository, filename=filename, revision=revision)
        metadata = get_hf_file_metadata(url, token=False)
        etag = str(metadata.etag or "").strip('"')
        if (
            metadata.commit_hash != revision
            or not SHA256.fullmatch(etag)
            or not isinstance(metadata.size, int)
            or metadata.size < 1
        ):
            raise RuntimeError("exact missing-file metadata did not bind commit, LFS SHA, and size")
        downloaded = Path(
            hf_hub_download(
                repo_id=repository,
                filename=filename,
                revision=revision,
                repo_type="model",
                local_dir=destination_snapshot,
                cache_dir=destination_root / ".hf-network-staging",
                token=False,
                force_download=False,
                local_files_only=False,
            )
        )
        if downloaded != destination or not destination.is_file():
            raise RuntimeError(
                "pinned missing-file download did not materialize the exact staging path"
            )
        actual_sha = _sha256(destination)
        if destination.stat().st_size != metadata.size or actual_sha != etag:
            raise RuntimeError("downloaded missing file failed exact size/LFS SHA verification")
        downloaded_files.append(
            {
                "path": destination.relative_to(destination_selected).as_posix(),
                "bytes": metadata.size,
                "sha256": actual_sha,
                "lfsSha256": etag,
                "commitSha": metadata.commit_hash,
            }
        )

    staged = resolve_cached_artifact(request, destination_root)
    staged["sourceCacheRoot"] = audit["cacheRoot"]
    staged["reusedFiles"] = audit["reusedFiles"]
    staged["downloadedFiles"] = downloaded_files
    return staged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--allow-reviewed-download", action="store_true")
    args = parser.parse_args()
    if args.audit and args.stage_root:
        raise ValueError("--audit and --stage-root are mutually exclusive")
    if args.allow_reviewed_download and not args.stage_root:
        raise ValueError("--allow-reviewed-download requires --stage-root")
    request = parse_request(args.request.resolve())
    if args.audit:
        result = audit_cached_artifact(request, args.cache_root)
    elif args.stage_root:
        result = stage_artifact(
            request,
            args.cache_root,
            args.stage_root,
            args.allow_reviewed_download,
        )
    else:
        result = resolve_cached_artifact(request, args.cache_root)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
