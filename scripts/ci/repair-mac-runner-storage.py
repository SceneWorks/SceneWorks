"""Install the pinned storage allocator and persist one runner's verified configuration."""
import hashlib
import json
import os
import pathlib
import plistlib
import re
import shutil
import subprocess
import urllib.request

assert os.environ.get("GITHUB_ACTIONS") == "true"
assert os.environ.get("RUNNER_NAME") == "nax-macos-2"
root = pathlib.Path("/Volumes/Work/dev-host")  # dev-host: allow-path established shared host root; never create its mount or marker
home = pathlib.Path.home()  # dev-host: allow-path installed binary and runner configuration only
runner = pathlib.Path(os.environ["RUNNER_WORKSPACE"]).parent.parent
assert (runner / ".runner").is_file()
mounts = subprocess.check_output(["/sbin/mount"], text=True)
for mount in (root.parent, pathlib.Path("/Volumes/Models")):  # dev-host: allow-path established required NAS mounts
    assert any(f" on {mount} (smbfs," in line for line in mounts.splitlines()), f"Missing SMB mount: {mount}"
assert (root / ".dev-host-root").is_file(), "Missing existing dev-host root marker"
assert root.stat().st_dev != pathlib.Path("/").stat().st_dev

version = "0.3.0"
asset = f"dev-host-{version}-aarch64-apple-darwin"
url = f"https://github.com/SceneWorks/DevWorks/releases/download/dev-host-v{version}/{asset}"
checksum = urllib.request.urlopen(url + ".sha256", timeout=30).read().decode().split()
assert len(checksum) == 2 and checksum[1].lstrip("*") == asset
assert re.fullmatch(r"[a-fA-F0-9]{64}", checksum[0])
payload = urllib.request.urlopen(url, timeout=60).read()
assert hashlib.sha256(payload).hexdigest() == checksum[0].lower(), "Release checksum mismatch"
binary = home / ".local/bin/dev-host"
binary.parent.mkdir(parents=True, exist_ok=True)
staged = binary.with_name(f".dev-host-{os.environ['GITHUB_RUN_ID']}")
staged.write_bytes(payload)
staged.chmod(0o755)
assert subprocess.check_output([str(staged), "--version"], text=True, timeout=8).strip() == f"dev-host {version}"
staged.replace(binary)
env = dict(os.environ, DEV_HOST_ROOT=str(root), DEV_HOST_BIN=str(binary))
subprocess.run([str(binary), "init-host"], env=env, check=True, timeout=20)
check = subprocess.check_output([str(binary), "check"], env=env, text=True, timeout=20).strip()
allocated = subprocess.check_output([
    str(binary), "env", "--format", "github", "--project", "SceneWorks",
    "--worktree", os.environ["GITHUB_WORKSPACE"], "--session", f"storage-repair-{os.environ['GITHUB_RUN_ID']}",
], env=env, text=True, timeout=20)
assignments = dict(line.split("=", 1) for line in allocated.splitlines() if "=" in line)
task = pathlib.Path(assignments["TMPDIR"])
assert task.resolve().is_relative_to(root.resolve())

hook = home / ".local/share/dev-host/job-started.sh"
hook.parent.mkdir(parents=True, exist_ok=True)
hook.write_text('#!/bin/bash\nset -euo pipefail\nexec "${DEV_HOST_BIN}" check\n')
hook.chmod(0o755)
values = {"DEV_HOST_ROOT": str(root), "DEV_HOST_BIN": str(binary), "ACTIONS_RUNNER_HOOK_JOB_STARTED": str(hook)}
path = runner / ".env"
original = path.read_text() if path.exists() else ""
backup = task / "runner-env.before-repair"
backup.write_text(original)
backup.chmod(0o600)
lines = [line for line in original.splitlines() if line.partition("=")[0] not in values]
lines.extend(f"{key}={value}" for key, value in values.items())
temporary = path.with_name(".env.storage-repair")
temporary.write_text("\n".join(lines) + "\n")
temporary.chmod(0o600)
temporary.replace(path)
service = (runner / ".service").read_text().strip()
plist = home / "Library/LaunchAgents" / f"{service}.plist"
assert plist.is_file(), f"Missing service definition: {plist}"
service_config = plistlib.loads(plist.read_bytes())
assert service_config["Label"] == service
receipt = {"runner": os.environ["RUNNER_NAME"], "root": str(root), "binary": str(binary),
           "version": version, "sha256": checksum[0].lower(), "check": check,
           "savedConfig": values, "service": service, "task": str(task), "restartRequired": True}
(task / "storage-repair.json").write_text(json.dumps(receipt, indent=2))
shutil.copyfile(task / "storage-repair.json", pathlib.Path(os.environ["GITHUB_WORKSPACE"]) / "storage-repair.json")
print(json.dumps(receipt, indent=2))
