#!/usr/bin/env python3
"""Validate one terminal evidence document with its checked-in Draft 2020-12 schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--document", required=True, type=Path)
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    document = json.loads(args.document.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        lines = []
        for error in errors:
            location = "/" + "/".join(str(part) for part in error.absolute_path)
            lines.append(f"{location or '/'}: {error.message}")
        raise SystemExit("\n".join(lines))
    print(f"Draft 2020-12 validation passed: {args.document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
