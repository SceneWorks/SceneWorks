#!/usr/bin/env python3
"""Audit and just-in-time stage immutable epic-20738 artifacts.

The trusted runner cache is fully audited before GPU work. Reviewed missing files can be downloaded
once into an isolated campaign-owned store. Individual authorities are subsequently copied into
fresh, short-lived staging roots with network access disabled.
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

# Exact published q4 inventories reviewed for sc-21306. A current Illustrious snapshot may be
# absent or incomplete, but every present and downloaded byte is bound here before terminal use.
_SHARED_ILLUSTRIOUS_Q4 = {
    "model_index.json": (708, "f15de0dfcd39efb54005fe7110fc07580dc7b90e73b2a7fb2ad4500381592e96"),
    "scheduler/scheduler_config.json": (526, "4276ecb13fc42b6667fe47bac67a376e99e4c26d477e189c4630dd7037a25a9a"),
    "text_encoder/config.json": (649, "f87b89e4249e027632236caba75d1140e14fd4c2ce4b4e554f2912b234e72cf9"),
    "text_encoder_2/config.json": (659, "3b96bc14843360d24e864f7d1ac6d83e95cad8f68209e7e503cefa9a4f65b18b"),
    "tokenizer/merges.txt": (573514, "0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052"),
    "tokenizer/special_tokens_map.json": (496, "c8227988aadac941e93b5ffef08b4add8ccbdb26404a62b3e41f61eef119b252"),
    "tokenizer/tokenizer_config.json": (734, "2edbd316072af0382bb999f82b007ad8af159d6277648e29b7747d72c3add791"),
    "tokenizer/vocab.json": (1109372, "fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e"),
    "tokenizer_2/merges.txt": (573514, "0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052"),
    "tokenizer_2/special_tokens_map.json": (484, "a83f5831a70d1d21c057186d35aaa504894103d24ee905baa99bc7e83ceb70ee"),
    "tokenizer_2/tokenizer_config.json": (893, "6e6c80bd367c7b39df2d7add74577ffd79d0840f3aa96e3b5893b95230743bcf"),
    "tokenizer_2/vocab.json": (1109372, "fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e"),
    "unet/config.json": (1915, "aeb34c12f61f1edd9f7e17d8332f91197bacad70754bfaa450836137c40c8c4d"),
    "vae/config.json": (807, "5d301324288c14277bab2b284004b91b81faffc800202ddba59e401c82683959"),
}
REVIEWED_HYDRATION_AUTHORITIES = {
    ("SceneWorks/illustrious-xl-v1-mlx", "778c3f02b7703b0c2755d0c0447592897193c6b5"): {
        **_SHARED_ILLUSTRIOUS_Q4,
        "text_encoder/model.safetensors": (200319411, "13e30e7c01f6c629a91732835d5fc6bd112ac8c949b7ac0f9bc5924336a4163a"),
        "text_encoder_2/model.safetensors": (610429154, "0fd5337422027e90951c2f7e1aa277817baafa2bb23ad0173154d57fcd1470ba"),
        "unet/diffusion_pytorch_model.safetensors": (2595556100, "f04b0c61537821e6057d420295ee9a4e94b45c497055823d38a6d9cd613c68b8"),
        "vae/diffusion_pytorch_model.fp16.safetensors": (167335384, "3304e1f0f639f7ad9b3c7ac96ff65db855f8e841cc039f232896e6910b5a9591"),
        "vae/diffusion_pytorch_model.safetensors": (334643294, "a91793eaf162d3531017f59decf47890e2abb6a376e709bb33542ebe8d4cfe0f"),
    },
    ("SceneWorks/illustrious-xl-v2-mlx", "672e9851ede4dc856fa945649b6691975c9d74a3"): {
        **_SHARED_ILLUSTRIOUS_Q4,
        "text_encoder/model.safetensors": (200319411, "20988f339be7d4851b9dbda25d614143c02d3f0485d3f4225b5d222463f33e2e"),
        "text_encoder_2/model.safetensors": (610429154, "7337dc56d6801e727c825b8699ba9f35c043909ba818ded018bc5f81acff3a95"),
        "unet/diffusion_pytorch_model.safetensors": (2595555776, "f3548787d6760d8eab11d1eb644b946b1eb90dbc08f697f5f919a83e03d4557f"),
        "vae/diffusion_pytorch_model.fp16.safetensors": (167335384, "8858c565cc9f13282b028069e92a6a4b4ebc1693a98e30176bbff16d7a230f51"),
        "vae/diffusion_pytorch_model.safetensors": (334643294, "b0a960def5b0ff715be74a7a82202a80300f8b06a3bc08e364af3647b6d7a653"),
    },
}


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


def _hydration_inventory(request: dict) -> dict[str, tuple[int, str]] | None:
    inventory = REVIEWED_HYDRATION_AUTHORITIES.get(
        (request["repository"], request["revision"])
    )
    if inventory is None:
        return None
    if request["subdirectory"] != "q4" or request["allowPatterns"] != ["q4/*"]:
        raise RuntimeError("reviewed hydration subdirectory/allow-pattern authority drifted")
    if request["expectedFiles"] != sorted(inventory):
        raise RuntimeError("reviewed hydration exact file-list authority drifted")
    return inventory


def _optional_ordinary_directory(path: Path, label: str) -> Path | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    return _ordinary_directory(path, label)


def _context(
    request: dict, cache_root: Path, *, allow_absent: bool = False
) -> tuple[Path, Path, Path, bool]:
    if not cache_root.is_absolute():
        raise ValueError("cache root must be an absolute path")
    trusted_root = _ordinary_directory(cache_root, "trusted cache root")
    owner, name = request["repository"].split("/", 1)
    repository_path = trusted_root / f"models--{owner}--{name}"
    repository_root = _optional_ordinary_directory(repository_path, "exact cached repository")
    if repository_root is None:
        if not allow_absent:
            _ordinary_directory(repository_path, "exact cached repository")
        snapshot = repository_path / "snapshots" / request["revision"]
        return trusted_root, snapshot, snapshot / request["subdirectory"], False
    if not _inside(trusted_root, repository_root):
        raise RuntimeError("cached repository escaped the trusted cache root")
    snapshots_path = repository_root / "snapshots"
    snapshots = _optional_ordinary_directory(snapshots_path, "cached snapshots directory")
    if snapshots is None:
        if not allow_absent:
            _ordinary_directory(snapshots_path, "cached snapshots directory")
        snapshot = snapshots_path / request["revision"]
        return trusted_root, snapshot, snapshot / request["subdirectory"], False
    snapshot_path = snapshots / request["revision"]
    snapshot = _optional_ordinary_directory(
        snapshot_path, f"exact cached revision {request['revision']}"
    )
    if snapshot is None:
        if not allow_absent:
            _ordinary_directory(snapshot_path, f"exact cached revision {request['revision']}")
        return trusted_root, snapshot_path, snapshot_path / request["subdirectory"], False
    if snapshot.parent != snapshots or not _inside(trusted_root, snapshot):
        raise RuntimeError("exact cached revision escaped the trusted cache root")
    selected_path = snapshot / request["subdirectory"]
    selected = _optional_ordinary_directory(selected_path, "selected cached artifact subdirectory")
    if selected is None:
        if not allow_absent:
            _ordinary_directory(selected_path, "selected cached artifact subdirectory")
        return trusted_root, snapshot, selected_path, False
    if not _inside(snapshot, selected, allow_equal=True):
        raise RuntimeError("selected artifact subdirectory escaped the exact snapshot")
    return trusted_root, snapshot, selected, True


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


def _strict_selected_census(selected: Path, expected_files: list[str]) -> None:
    allowed_files = set(expected_files)
    allowed_directories = {
        Path(*Path(filename).parts[:index]).as_posix()
        for filename in expected_files
        for index in range(1, len(Path(filename).parts))
    }
    pending = [selected]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                lexical = Path(entry.path)
                relative = lexical.relative_to(selected).as_posix()
                metadata = entry.stat(follow_symlinks=False)
                if _is_reparse(metadata):
                    if relative not in allowed_files:
                        raise RuntimeError(f"unexpected symlink/reparse entry in reviewed snapshot: {lexical}")
                elif stat.S_ISDIR(metadata.st_mode):
                    if relative not in allowed_directories:
                        raise RuntimeError(f"unexpected directory in reviewed snapshot: {lexical}")
                    pending.append(lexical)
                elif stat.S_ISREG(metadata.st_mode):
                    if relative not in allowed_files:
                        raise RuntimeError(f"unexpected file in reviewed snapshot: {lexical}")
                else:
                    raise RuntimeError(f"unsafe filesystem entry in reviewed snapshot: {lexical}")


def _lexically_absent(selected: Path, relative_selected: str) -> bool:
    current = selected
    try:
        current.lstat()
    except FileNotFoundError:
        return True
    parts = Path(relative_selected).parts
    for index, segment in enumerate(parts):
        current = current / segment
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return True
        if index < len(parts) - 1 and (
            not stat.S_ISDIR(metadata.st_mode) or _is_reparse(metadata)
        ):
            raise RuntimeError(f"expected artifact parent is unsafe: {current}")
    return False


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
    hydration_inventory = _hydration_inventory(request)
    trusted_root, snapshot, selected, selected_exists = _context(
        request, cache_root, allow_absent=hydration_inventory is not None
    )
    if hydration_inventory is not None and selected_exists:
        _strict_selected_census(selected, request["expectedFiles"])
    reviewed_missing = _reviewed_missing(request)
    files = []
    missing = []
    for relative_selected in request["expectedFiles"]:
        relative_snapshot = (
            Path(request["subdirectory"]) / relative_selected
        ).as_posix() if request["subdirectory"] != "." else relative_selected
        if hydration_inventory is not None and _lexically_absent(selected, relative_selected):
            missing.append(relative_snapshot)
            continue
        try:
            files.append(
                _expected_file(
                    selected, snapshot, trusted_root, relative_selected
                )
            )
        except (FileNotFoundError, RuntimeError) as error:
            if relative_snapshot in reviewed_missing and isinstance(error, FileNotFoundError):
                missing.append(relative_snapshot)
                continue
            raise RuntimeError(
                f"authoritative expected file is missing or invalid: {relative_snapshot}: {error}"
            ) from error
    matched_records = []
    for lexical, relative, size in files:
        selected_path = lexical.relative_to(selected).as_posix()
        digest = _sha256(lexical.resolve(strict=True))
        if hydration_inventory is not None and hydration_inventory[selected_path] != (size, digest):
            raise RuntimeError(f"reviewed hydration source byte identity drifted: {relative}")
        matched_records.append({"path": selected_path, "bytes": size, "sha256": digest})
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


def _mkdir_confined(root: Path, relative: Path, label: str) -> Path:
    current = root
    for segment in relative.parts:
        current = current / segment
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or _is_reparse(metadata):
            raise RuntimeError(f"{label} contains an unsafe directory: {current}")
        resolved = current.resolve(strict=True)
        if not _inside(root, resolved):
            raise RuntimeError(f"{label} escaped its campaign root: {current}")
    return current


def _receipt_path(root: Path, request: dict) -> Path:
    owner, name = request["repository"].split("/", 1)
    return root / "download-receipts" / f"{owner}--{name}@{request['revision']}.json"


def _approved_download_inventory(
    request: dict, missing_files: list[str]
) -> dict[str, tuple[int | None, str | None]]:
    hydration = _hydration_inventory(request)
    prefix = "" if request["subdirectory"] == "." else f"{request['subdirectory']}/"
    if hydration is not None:
        approved = {
            f"{prefix}{filename}": identity for filename, identity in hydration.items()
        }
        if not missing_files or any(filename not in approved for filename in missing_files):
            raise RuntimeError("network fill escaped the reviewed hydration inventory")
        return {filename: approved[filename] for filename in missing_files}
    if [request["repository"], request["revision"], *missing_files] != list(
        NETWORK_FILL_AUTHORITY
    ):
        raise RuntimeError("network fill requires an exact reviewed missing-file plan")
    return {NETWORK_FILL_AUTHORITY[2]: (None, None)}


def download_reviewed_missing(request: dict, cache_root: Path, missing_store: Path) -> dict:
    audit = audit_cached_artifact(request, cache_root)
    approved = _approved_download_inventory(request, audit["missingFiles"])

    destination_root = _missing_store_root(missing_store)
    repository, revision = request["repository"], request["revision"]
    owner, name = repository.split("/", 1)
    destination_snapshot = (
        destination_root / f"models--{owner}--{name}" / "snapshots" / revision
    )
    receipt_path = _receipt_path(destination_root, request)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError("refusing to overwrite the campaign missing-file store")
    if destination_snapshot.exists() or destination_snapshot.is_symlink():
        raise RuntimeError("campaign missing-file authority store is ambiguous or already populated")
    _mkdir_confined(
        destination_root,
        destination_snapshot.relative_to(destination_root),
        "campaign missing-file authority store",
    )

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
    network_cache = destination_root / ".hf-network-staging"
    records = []
    for filename, (reviewed_size, reviewed_sha) in approved.items():
        destination = destination_snapshot / filename
        url = hf_hub_url(repo_id=repository, filename=filename, revision=revision)
        metadata = get_hf_file_metadata(url, token=False)
        etag = str(metadata.etag or "").strip('"')
        if (
            metadata.commit_hash != revision
            or not isinstance(metadata.size, int)
            or metadata.size < 1
            or (reviewed_size is not None and metadata.size != reviewed_size)
            or (reviewed_sha is None and not SHA256.fullmatch(etag))
            or (reviewed_sha is not None and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", etag))
            or (reviewed_sha is not None and len(etag) == 64 and etag != reviewed_sha)
        ):
            raise RuntimeError("exact missing-file metadata did not bind reviewed commit/etag/size")
        _mkdir_confined(
            destination_snapshot,
            destination.parent.relative_to(destination_snapshot),
            "campaign missing-file destination",
        )
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
        if destination.is_symlink() or _is_reparse(destination.lstat()):
            raise RuntimeError("pinned missing-file download produced a reparse target")
        actual_sha = _sha256(destination)
        expected_sha = reviewed_sha or etag
        if destination.stat().st_size != metadata.size or actual_sha != expected_sha:
            raise RuntimeError("downloaded missing file failed exact size/content SHA verification")
        records.append({
            "path": Path(filename).relative_to(request["subdirectory"]).as_posix(),
            "bytes": metadata.size,
            "sha256": actual_sha,
            # The legacy receipt field carries the audited content SHA. For LFS files it is also
            # the remote LFS etag; ordinary Git files are independently bound by commit and size.
            "lfsSha256": actual_sha,
            "commitSha": metadata.commit_hash,
        })
    if network_cache.exists():
        shutil.rmtree(network_cache, ignore_errors=False)
    local_metadata = destination_snapshot / ".cache"
    if local_metadata.exists():
        shutil.rmtree(local_metadata, ignore_errors=False)
    for record in records:
        destination = destination_snapshot / request["subdirectory"] / record["path"]
        if (
            not destination.is_file()
            or destination.is_symlink()
            or destination.stat().st_size != record["bytes"]
            or _sha256(destination) != record["sha256"]
        ):
            raise RuntimeError("downloaded missing file did not remain durable after cleanup")
    _mkdir_confined(
        destination_root,
        receipt_path.parent.relative_to(destination_root),
        "campaign download receipt store",
    )
    receipt_path.write_text(json.dumps(records, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "id": request["id"],
        "storeRoot": str(destination_root),
        "downloadedFiles": records,
    }


def _stored_missing_files(
    request: dict, missing_store: Path, missing_files: list[str]
) -> list[dict]:
    root = _missing_store_root(missing_store)
    approved = _approved_download_inventory(request, missing_files)
    receipt_path = _receipt_path(root, request)
    try:
        records = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("campaign missing-file receipt is absent or invalid") from error
    if not isinstance(records, list) or len(records) != len(approved):
        raise RuntimeError("campaign missing-file receipt census drifted")
    owner, name = request["repository"].split("/", 1)
    expected_records = []
    for (filename, (reviewed_size, reviewed_sha)), record in zip(approved.items(), records):
        expected_selected = Path(filename).relative_to(request["subdirectory"]).as_posix()
        stored = root / f"models--{owner}--{name}" / "snapshots" / request["revision"] / filename
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "bytes", "sha256", "lfsSha256", "commitSha"}
            or record["path"] != expected_selected
            or record["commitSha"] != request["revision"]
            or record["sha256"] != record["lfsSha256"]
            or not SHA256.fullmatch(record["sha256"])
            or not isinstance(record["bytes"], int)
            or record["bytes"] < 1
            or (reviewed_size is not None and record["bytes"] != reviewed_size)
            or (reviewed_sha is not None and record["sha256"] != reviewed_sha)
            or not stored.is_file()
            or stored.is_symlink()
            or _is_reparse(stored.lstat())
            or stored.stat().st_size != record["bytes"]
            or _sha256(stored) != record["sha256"]
        ):
            raise RuntimeError("campaign missing-file store failed exact identity verification")
        expected_records.append(record)
    return expected_records


def stage_artifact(
    request: dict,
    cache_root: Path,
    staging_root: Path,
    missing_store: Path | None,
) -> dict:
    audit = audit_cached_artifact(request, cache_root)
    stored_missing = []
    if audit["missingFiles"]:
        if missing_store is None:
            raise RuntimeError(
                "offline JIT staging lacks the exact reviewed missing file(s): "
                + ", ".join(audit["missingFiles"])
            )
        stored_missing = _stored_missing_files(request, missing_store, audit["missingFiles"])

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
    for stored_record, missing_snapshot_path in zip(stored_missing, audit["missingFiles"]):
        owner, name = request["repository"].split("/", 1)
        source = _missing_store_root(missing_store) / f"models--{owner}--{name}" / "snapshots" / request["revision"] / missing_snapshot_path
        stored_missing_record = stored_record
        destination = destination_selected / stored_missing_record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if not (not destination.is_symlink() and destination.is_file() and destination.stat().st_size == stored_missing_record["bytes"] and _sha256(destination) == stored_missing_record["sha256"]):
                raise RuntimeError("refusing to overwrite non-identical staged missing file")
        else:
            shutil.copyfile(source, destination)
        if destination.stat().st_size != stored_missing_record["bytes"] or _sha256(destination) != stored_missing_record["sha256"]:
            raise RuntimeError("JIT copy of campaign missing file failed byte verification")
        downloaded_files.append(
            {
                **stored_missing_record,
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
