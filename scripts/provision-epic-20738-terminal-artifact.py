#!/usr/bin/env python3
"""Audit and just-in-time stage immutable epic-20738 artifacts.

The trusted runner cache is fully audited before GPU work.  The sole reviewed missing file can be
downloaded once into an isolated campaign-owned store.  Individual authorities are subsequently
copied into fresh, short-lived staging roots with network access disabled.
"""

from __future__ import annotations

import argparse
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
    required = {
        "id", "repository", "revision", "subdirectory", "allowPatterns", "expectedFiles"
    }
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
    expected_files = request["expectedFiles"]
    if (
        not isinstance(expected_files, list)
        or not expected_files
        or not all(isinstance(file, str) and file for file in expected_files)
        or len(set(expected_files)) != len(expected_files)
        or expected_files != sorted(expected_files)
        or any(Path(file).is_absolute() or ".." in Path(file).parts for file in expected_files)
    ):
        raise ValueError("expectedFiles must be a sorted unique confined non-empty file array")
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


def _expected_file(
    selected: Path,
    snapshot: Path,
    trusted_root: Path,
    relative_selected: str,
) -> tuple[Path, str, int]:
    lexical = selected / Path(relative_selected)
    current = selected
    for segment in Path(relative_selected).parts[:-1]:
        current = current / segment
        metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or _is_reparse(metadata):
            raise RuntimeError(
                f"expected artifact parent must be an ordinary directory: {current}"
            )
        resolved_parent = current.resolve(strict=True)
        if not _inside(snapshot, resolved_parent, allow_equal=True):
            raise RuntimeError(f"expected artifact parent escaped snapshot: {current}")
    metadata = lexical.lstat()
    resolved = lexical.resolve(strict=True)
    resolved_metadata = resolved.stat()
    if not (stat.S_ISREG(metadata.st_mode) or _is_reparse(metadata)):
        raise RuntimeError(f"expected artifact is not a regular file: {lexical}")
    if not stat.S_ISREG(resolved_metadata.st_mode):
        raise RuntimeError(f"expected artifact target is not a regular file: {lexical}")
    if not _inside(trusted_root, resolved):
        raise RuntimeError(f"expected artifact file escaped trusted cache root: {lexical}")
    if resolved_metadata.st_size < 1:
        raise RuntimeError(f"expected artifact contains an empty file: {lexical}")
    return lexical, lexical.relative_to(snapshot).as_posix(), resolved_metadata.st_size


def audit_cached_artifact(request: dict, cache_root: Path) -> dict:
    trusted_root, snapshot, selected = _context(request, cache_root)
    reviewed_missing = _reviewed_missing(request)
    files = []
    missing = []
    for relative_selected in request["expectedFiles"]:
        relative_snapshot = (
            Path(request["subdirectory"]) / relative_selected
        ).as_posix() if request["subdirectory"] != "." else relative_selected
        try:
            files.append(
                _expected_file(
                    selected, snapshot, trusted_root, relative_selected
                )
            )
        except (FileNotFoundError, RuntimeError) as error:
            if relative_snapshot in reviewed_missing and (
                isinstance(error, FileNotFoundError)
                or "broken" in str(error).lower()
            ):
                missing.append(relative_snapshot)
                continue
            raise RuntimeError(
                f"authoritative expected file is missing or invalid: {relative_snapshot}: {error}"
            ) from error
    matched_records = []
    for lexical, relative, size in files:
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
        "selectedFiles": [record["path"] for record in matched_records],
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


def _missing_store_root(missing_store: Path) -> Path:
    if not missing_store.is_absolute():
        raise ValueError("missing-file store must be an absolute path")
    return _ordinary_directory(missing_store, "campaign missing-file store")


def download_reviewed_missing(request: dict, cache_root: Path, missing_store: Path) -> dict:
    audit = audit_cached_artifact(request, cache_root)
    reviewed = list(NETWORK_FILL_AUTHORITY)
    if [request["repository"], request["revision"], *audit["missingFiles"]] != reviewed:
        raise RuntimeError("network fill requires exactly the sole reviewed missing filename")

    destination_root = _missing_store_root(missing_store)
    repository, revision, filename = NETWORK_FILL_AUTHORITY
    owner, name = repository.split("/", 1)
    destination_snapshot = (
        destination_root / f"models--{owner}--{name}" / "snapshots" / revision
    )
    destination = destination_snapshot / filename
    receipt_path = destination_root / "download-receipt.json"
    if destination.exists() or destination.is_symlink() or receipt_path.exists():
        raise RuntimeError("refusing to overwrite the campaign missing-file store")
    destination.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import (
        __version__,
        get_hf_file_metadata,
        hf_hub_download,
        hf_hub_url,
    )

    if __version__ != "0.36.0":
        raise RuntimeError(
            f"network fill requires huggingface_hub 0.36.0, got {__version__}"
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
    network_cache = destination_root / ".hf-network-staging"
    downloaded = Path(
        hf_hub_download(
            repo_id=repository,
            filename=filename,
            revision=revision,
            repo_type="model",
            local_dir=destination_snapshot,
            cache_dir=network_cache,
            token=False,
            force_download=False,
            local_files_only=False,
        )
    )
    if downloaded != destination or not destination.is_file():
        raise RuntimeError("pinned missing-file download did not materialize the exact store path")
    if destination.is_symlink():
        raise RuntimeError("pinned missing-file download produced a reparse target")
    actual_sha = _sha256(destination)
    if destination.stat().st_size != metadata.size or actual_sha != etag:
        raise RuntimeError("downloaded missing file failed exact size/LFS SHA verification")
    if network_cache.exists():
        shutil.rmtree(network_cache, ignore_errors=False)
    local_metadata = destination_snapshot / ".cache"
    if local_metadata.exists():
        shutil.rmtree(local_metadata, ignore_errors=False)
    if not destination.is_file() or destination.stat().st_size != metadata.size or _sha256(destination) != etag:
        raise RuntimeError("downloaded missing file did not remain durable after network-cache cleanup")
    record = {
        "path": Path(filename).relative_to(request["subdirectory"]).as_posix(),
        "bytes": metadata.size,
        "sha256": actual_sha,
        "lfsSha256": etag,
        "commitSha": metadata.commit_hash,
    }
    receipt_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "id": request["id"],
        "storeRoot": str(destination_root),
        "downloadedFiles": [record],
    }


def _stored_missing_file(request: dict, missing_store: Path) -> dict:
    root = _missing_store_root(missing_store)
    if _reviewed_missing(request) != [NETWORK_FILL_AUTHORITY[2]]:
        raise RuntimeError("missing-file store is not approved for this authority")
    receipt_path = root / "download-receipt.json"
    try:
        record = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("campaign missing-file receipt is absent or invalid") from error
    if set(record) != {"path", "bytes", "sha256", "lfsSha256", "commitSha"}:
        raise RuntimeError("campaign missing-file receipt fields drifted")
    expected = NETWORK_FILL_AUTHORITY[2]
    expected_selected = Path(expected).relative_to(request["subdirectory"]).as_posix()
    owner, name = NETWORK_FILL_AUTHORITY[0].split("/", 1)
    stored = (
        root / f"models--{owner}--{name}" / "snapshots"
        / NETWORK_FILL_AUTHORITY[1] / expected
    )
    if (
        record["path"] != expected_selected
        or record["commitSha"] != NETWORK_FILL_AUTHORITY[1]
        or record["sha256"] != record["lfsSha256"]
        or not SHA256.fullmatch(record["sha256"])
        or not isinstance(record["bytes"], int)
        or record["bytes"] < 1
        or not stored.is_file()
        or stored.is_symlink()
        or stored.stat().st_size != record["bytes"]
        or _sha256(stored) != record["sha256"]
    ):
        raise RuntimeError("campaign missing-file store failed exact identity verification")
    return record


def stage_artifact(
    request: dict,
    cache_root: Path,
    staging_root: Path,
    missing_store: Path | None,
) -> dict:
    audit = audit_cached_artifact(request, cache_root)
    stored_missing = None
    if audit["missingFiles"]:
        if audit["missingFiles"] != [NETWORK_FILL_AUTHORITY[2]] or missing_store is None:
            raise RuntimeError(
                "offline JIT staging lacks the exact reviewed missing file: "
                + ", ".join(audit["missingFiles"])
            )
        stored_missing = _stored_missing_file(request, missing_store)

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
            if not destination.is_symlink() and destination.is_file() and destination.stat().st_size == record["bytes"] and _sha256(destination) == record["sha256"]:
                continue
            raise RuntimeError(f"refusing to overwrite non-identical staged cache file: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.resolve(strict=True), destination)
        if destination.stat().st_size != record["bytes"] or _sha256(destination) != record["sha256"]:
            raise RuntimeError(f"staged cache copy failed byte verification: {destination}")

    downloaded_files = []
    if stored_missing:
        owner, name = NETWORK_FILL_AUTHORITY[0].split("/", 1)
        source = (
            _missing_store_root(missing_store)
            / f"models--{owner}--{name}" / "snapshots"
            / NETWORK_FILL_AUTHORITY[1] / NETWORK_FILL_AUTHORITY[2]
        )
        destination = destination_selected / stored_missing["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if not (not destination.is_symlink() and destination.is_file() and destination.stat().st_size == stored_missing["bytes"] and _sha256(destination) == stored_missing["sha256"]):
                raise RuntimeError("refusing to overwrite non-identical staged missing file")
        else:
            shutil.copyfile(source, destination)
        if destination.stat().st_size != stored_missing["bytes"] or _sha256(destination) != stored_missing["sha256"]:
            raise RuntimeError("JIT copy of campaign missing file failed byte verification")
        downloaded_files.append(
            {
                **stored_missing,
                "path": destination.relative_to(destination_selected).as_posix(),
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
    parser.add_argument("--missing-store", type=Path)
    parser.add_argument("--download-reviewed-missing", action="store_true")
    args = parser.parse_args()
    if args.audit and args.stage_root:
        raise ValueError("--audit and --stage-root are mutually exclusive")
    if args.download_reviewed_missing and (args.audit or args.stage_root or not args.missing_store):
        raise ValueError("--download-reviewed-missing requires only --missing-store")
    request = parse_request(args.request.resolve())
    if args.download_reviewed_missing:
        result = download_reviewed_missing(request, args.cache_root, args.missing_store)
    elif args.audit:
        result = audit_cached_artifact(request, args.cache_root)
    elif args.stage_root:
        result = stage_artifact(
            request,
            args.cache_root,
            args.stage_root,
            args.missing_store,
        )
    else:
        result = resolve_cached_artifact(request, args.cache_root)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
