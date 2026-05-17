import argparse
import json
from pathlib import Path

from sceneworks_shared import reindex_project

from .settings import get_settings
from .server import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m sceneworks_api")
    subparsers = parser.add_subparsers(dest="command")
    reindex_parser = subparsers.add_parser("reindex", help="Rebuild a project's DB index from sidecar files.")
    reindex_parser.add_argument("--project", required=True, help="Path to a .sceneworks project folder.")
    args = parser.parse_args()

    if args.command == "reindex":
        counts = reindex_project(Path(args.project).expanduser().resolve())
        print(json.dumps(counts, sort_keys=True))
    else:
        run(get_settings())
