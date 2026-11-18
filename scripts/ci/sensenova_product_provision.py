"""Fill only absent files in the exact manifest-selected public SenseNova snapshot."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ENDPOINT = "https://huggingface.co"


def provision(plan, cache, api, download):
    repo, revision, names = plan["repo"], plan["revision"], plan["files"]
    if not (re.fullmatch(r"[\w-]+/[\w.-]+", repo) and re.fullmatch(r"[a-f0-9]{40}", revision)
            and cache.is_absolute() and cache.is_dir() and names and len(set(names)) == len(names)):
        raise ValueError("invalid public snapshot plan")
    for name in names:
        if not all(re.fullmatch(r"[\w.-]+", part) and part not in (".", "..") for part in name.split("/")):
            raise ValueError(f"invalid enumerated file {name}")
    info = api.model_info(repo, revision=revision, files_metadata=True, token=False)
    if info.sha != revision:
        raise ValueError("publisher revision differs from manifest")
    published = {entry.rfilename: entry for entry in info.siblings}
    snapshot = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    missing_bytes = 0
    for name in names:
        entry = published[name]
        digest = entry.lfs.sha256 if entry.lfs else entry.blob_id
        if not isinstance(entry.size, int) or entry.size <= 0 or not re.fullmatch(
                r"[a-f0-9]{64}" if entry.lfs else r"[a-f0-9]{40}", digest or ""):
            raise ValueError(f"missing publisher identity for {name}")
        if not (snapshot / name).is_file():
            missing_bytes += entry.size
    if shutil.disk_usage(cache).free < missing_bytes:
        raise OSError(f"cache requires space for {missing_bytes} absent bytes")
    files = []
    for name in names:
        entry = published[name]
        target = snapshot / name
        missing = not target.is_file()
        if missing:
            download(repo_id=repo, revision=revision, filename=name, cache_dir=str(cache),
                     endpoint=ENDPOINT, token=False)
        if target.stat().st_size != entry.size:
            raise ValueError(f"publisher size mismatch for {name}")
        digest = hashlib.sha256() if entry.lfs else hashlib.sha1(f"blob {entry.size}\0".encode())
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != (entry.lfs.sha256 if entry.lfs else entry.blob_id):
            raise ValueError(f"publisher hash mismatch for {name}")
        files.append({"file": name, "bytes": entry.size, "publisherHash": digest.hexdigest(),
                      "downloaded": missing})
    return {"repo": repo, "revision": revision, "files": files}


if __name__ == "__main__":
    from huggingface_hub import HfApi, hf_hub_download

    plan, receipt = map(Path, sys.argv[1:])
    result = provision(json.loads(plan.read_text()), Path(os.environ["HF_HUB_CACHE"]),
                       HfApi(endpoint=ENDPOINT, token=False), hf_hub_download)
    receipt.write_text(json.dumps(result, indent=2) + "\n")
