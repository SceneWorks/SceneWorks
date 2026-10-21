"""Build the encoder-free HF hub for the Iris depth / restoration smokes (sc-25690).

    clone_hub.py <destination root> <hub repo dir>...

Each `models--org--name` directory is rebuilt under the destination as a real directory tree whose
files are APFS CLONES (`cp -c`, copy-on-write) of the resolved hub blobs: no bytes are copied, and
— unlike a hard link, which shares the inode — nothing a job writes through the encoder-free hub
can reach `~/sceneworks-rw-weights/hub`. (A symlink is not an option either: the worker
canonicalizes a training base path and requires it inside the app-managed hub.)

Every clone is checked: it must be its own inode, and the shared blob's link count must be what it
was. `cp -c` falls back to a full copy across volumes; the destination must therefore sit on the
hub's volume, which is checked up front so a 30 GB copy never starts silently.
"""

import os
import subprocess
import sys


def clone_repo(repo, dest_root):
    base = os.path.join(dest_root, os.path.basename(os.path.normpath(repo)))
    cloned = 0
    for here, _dirs, files in os.walk(repo, followlinks=True):
        rel = os.path.relpath(here, repo)
        os.makedirs(os.path.join(base, rel), exist_ok=True)
        for name in files:
            source = os.path.realpath(os.path.join(here, name))
            target = os.path.join(base, rel, name)
            before = os.stat(source)
            subprocess.run(["/bin/cp", "-c", source, target], check=True)
            after, made = os.stat(source), os.stat(target)
            if made.st_ino == before.st_ino or after.st_nlink != before.st_nlink:
                raise SystemExit(f"{target} shares its inode with the shared hub blob {source}")
            if made.st_size != before.st_size:
                raise SystemExit(f"{target} is not a whole clone of {source}")
            cloned += 1
    return cloned


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    dest_root, repos = argv[0], argv[1:]
    os.makedirs(dest_root, exist_ok=True)
    for repo in repos:
        if os.stat(repo).st_dev != os.stat(dest_root).st_dev:
            raise SystemExit(
                f"{dest_root} is not on the volume of {repo}: an APFS clone would become a full copy"
            )
    for repo in repos:
        print(f"cloned {clone_repo(repo, dest_root)} files of {os.path.basename(repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
