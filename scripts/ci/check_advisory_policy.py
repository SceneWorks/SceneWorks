#!/usr/bin/env python3
"""Validate advisory coverage and require every cargo-deny ignore to be owned and expiring."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and earlier; CI is pinned to 3.12.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DENY = ROOT / "deny.toml"
DEFAULT_METADATA = ROOT / "advisory-ignores.toml"
ADVISORY_RE = re.compile(r"RUSTSEC-[0-9]{4}-[0-9]{4}")
OWNER_RE = re.compile(
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})(?:/[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))?"
)
REQUIRED_FIELDS = frozenset({"advisory", "reason", "reachability", "owner", "expires"})
REQUIRED_GRAPH_FIELDS = frozenset({"all-features", "targets"})
REQUIRED_ADVISORY_FIELDS = frozenset({"ignore", "unmaintained", "unsound", "yanked"})
FORBIDDEN_ROOT_GRAPH_FIELDS = frozenset(
    {
        "all-features",
        "exclude",
        "exclude-dev",
        "exclude-unpublished",
        "features",
        "no-default-features",
        "targets",
    }
)
REQUIRED_TARGETS = frozenset(
    {
        "x86_64-unknown-linux-gnu",
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc",
    }
)
MINIMUM_RATIONALE_LENGTH = 20
MAXIMUM_EXCEPTION_LIFETIME_DAYS = 90
PLACEHOLDERS = frozenset({"n/a", "none", "tbd", "todo", "unknown"})


class PolicyError(ValueError):
    """The advisory policy is incomplete, malformed, or expired."""


@dataclass(frozen=True)
class AdvisoryIgnore:
    advisory: str
    reason: str
    reachability: str
    owner: str
    expires: date


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{field} must be a table")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PolicyError(f"{field} must be a non-empty string without surrounding whitespace")
    return value


def _rationale(value: Any, field: str) -> str:
    rationale = _string(value, field)
    if len(rationale) < MINIMUM_RATIONALE_LENGTH or rationale.casefold() in PLACEHOLDERS:
        raise PolicyError(
            f"{field} must contain at least {MINIMUM_RATIONALE_LENGTH} characters of rationale"
        )
    return rationale


def _advisory(value: Any, field: str) -> str:
    advisory = _string(value, field)
    if ADVISORY_RE.fullmatch(advisory) is None:
        raise PolicyError(f"{field} must match RUSTSEC-YYYY-NNNN")
    return advisory


def _load_toml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PolicyError(f"could not read {path}: {error}") from error


def _require_exact_fields(
    document: Mapping[str, Any],
    required: frozenset[str],
    field: str,
) -> None:
    missing = required - set(document)
    unexpected = set(document) - required
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("noncanonical " + ", ".join(sorted(unexpected)))
        raise PolicyError(
            f"{field} must contain exactly the coverage-preserving settings: "
            + "; ".join(details)
        )


def _validate_graph(deny_document: Mapping[str, Any]) -> None:
    graph = _mapping(deny_document.get("graph"), "deny.toml [graph]")
    _require_exact_fields(graph, REQUIRED_GRAPH_FIELDS, "deny.toml [graph]")
    if graph.get("all-features") is not True:
        raise PolicyError("deny.toml [graph].all-features must be true")
    raw_targets = graph.get("targets")
    if not isinstance(raw_targets, list):
        raise PolicyError("deny.toml [graph].targets must be an array of target tables")
    targets = []
    for index, raw_target in enumerate(raw_targets):
        target = _mapping(raw_target, f"deny.toml [graph].targets[{index}]")
        _require_exact_fields(
            target,
            frozenset({"triple"}),
            f"deny.toml [graph].targets[{index}]",
        )
        targets.append(_string(target["triple"], f"deny.toml [graph].targets[{index}].triple"))
    if len(set(targets)) != len(targets):
        raise PolicyError("deny.toml [graph].targets contains a duplicate target")
    if set(targets) != REQUIRED_TARGETS:
        missing = sorted(REQUIRED_TARGETS - set(targets))
        unexpected = sorted(set(targets) - REQUIRED_TARGETS)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise PolicyError(
            "deny.toml [graph].targets must cover shipped platforms: " + "; ".join(details)
        )


def _validate_advisories(deny_document: Mapping[str, Any]) -> Mapping[str, Any]:
    advisories = _mapping(deny_document.get("advisories"), "deny.toml [advisories]")
    _require_exact_fields(advisories, REQUIRED_ADVISORY_FIELDS, "deny.toml [advisories]")
    canonical = {
        "unmaintained": "workspace",
        "unsound": "workspace",
        "yanked": "warn",
    }
    for setting, expected in canonical.items():
        if advisories.get(setting) != expected:
            raise PolicyError(
                f"deny.toml [advisories].{setting} must be {expected!r}, "
                f"found {advisories.get(setting)!r}"
            )
    return advisories


def validate_documents(
    deny_document: Mapping[str, Any],
    metadata_document: Mapping[str, Any],
    *,
    today: date | None = None,
) -> tuple[AdvisoryIgnore, ...]:
    """Return validated ignores, or raise ``PolicyError`` on any policy gap."""

    today = today or date.today()
    deny_document = _mapping(deny_document, "deny.toml")
    forbidden_root = FORBIDDEN_ROOT_GRAPH_FIELDS & set(deny_document)
    if forbidden_root:
        raise PolicyError(
            "deny.toml uses deprecated root graph settings that can override [graph]: "
            + ", ".join(sorted(forbidden_root))
        )
    _validate_graph(deny_document)
    advisories = _validate_advisories(deny_document)
    raw_ignored = advisories.get("ignore")
    if not isinstance(raw_ignored, list):
        raise PolicyError("deny.toml [advisories].ignore must be an array")

    ignored = [
        _advisory(value, f"deny.toml [advisories].ignore[{index}]")
        for index, value in enumerate(raw_ignored)
    ]
    if len(set(ignored)) != len(ignored):
        raise PolicyError("deny.toml [advisories].ignore contains a duplicate advisory")

    metadata_document = _mapping(metadata_document, "advisory-ignores.toml")
    unknown_root = set(metadata_document) - {"version", "ignore"}
    if unknown_root:
        raise PolicyError(
            "advisory-ignores.toml has unknown root fields: " + ", ".join(sorted(unknown_root))
        )
    if type(metadata_document.get("version")) is not int or metadata_document["version"] != 1:
        raise PolicyError("advisory-ignores.toml version must be the integer 1")

    raw_entries = metadata_document.get("ignore")
    if not isinstance(raw_entries, list):
        raise PolicyError("advisory-ignores.toml ignore must be an array of tables")

    entries = []
    for index, raw_entry in enumerate(raw_entries):
        field = f"advisory-ignores.toml ignore[{index}]"
        entry = _mapping(raw_entry, field)
        missing = REQUIRED_FIELDS - set(entry)
        unknown = set(entry) - REQUIRED_FIELDS
        if missing:
            raise PolicyError(f"{field} is missing fields: {', '.join(sorted(missing))}")
        if unknown:
            raise PolicyError(f"{field} has unknown fields: {', '.join(sorted(unknown))}")

        advisory = _advisory(entry["advisory"], f"{field}.advisory")
        owner = _string(entry["owner"], f"{field}.owner")
        if OWNER_RE.fullmatch(owner) is None:
            raise PolicyError(f"{field}.owner must be an accountable @user or @org/team")
        expires = entry["expires"]
        if type(expires) is not date:
            raise PolicyError(f"{field}.expires must be an unquoted TOML local date")
        if expires <= today:
            raise PolicyError(
                f"{field}.expires must be after {today.isoformat()}, found {expires.isoformat()}"
            )
        latest_expiry = today + timedelta(days=MAXIMUM_EXCEPTION_LIFETIME_DAYS)
        if expires > latest_expiry:
            raise PolicyError(
                f"{field}.expires must be within {MAXIMUM_EXCEPTION_LIFETIME_DAYS} days of "
                f"{today.isoformat()}, found {expires.isoformat()}"
            )
        entries.append(
            AdvisoryIgnore(
                advisory=advisory,
                reason=_rationale(entry["reason"], f"{field}.reason"),
                reachability=_rationale(entry["reachability"], f"{field}.reachability"),
                owner=owner,
                expires=expires,
            )
        )

    governed = [entry.advisory for entry in entries]
    if len(set(governed)) != len(governed):
        raise PolicyError("advisory-ignores.toml contains a duplicate advisory")
    ungoverned = sorted(set(ignored) - set(governed))
    stale = sorted(set(governed) - set(ignored))
    if ungoverned or stale:
        details = []
        if ungoverned:
            details.append("missing metadata for " + ", ".join(ungoverned))
        if stale:
            details.append("metadata without a deny.toml ignore for " + ", ".join(stale))
        raise PolicyError("advisory ignore sets differ: " + "; ".join(details))
    return tuple(entries)


def validate_policy(
    deny_path: Path = DEFAULT_DENY,
    metadata_path: Path = DEFAULT_METADATA,
    *,
    today: date | None = None,
) -> tuple[AdvisoryIgnore, ...]:
    return validate_documents(
        _load_toml(deny_path),
        _load_toml(metadata_path),
        today=today,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deny", type=Path, default=DEFAULT_DENY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    try:
        entries = validate_policy(args.deny, args.metadata)
    except PolicyError as error:
        print(f"advisory policy: ERROR: {error}", file=sys.stderr)
        return 1
    print(f"advisory policy: OK ({len(entries)} governed ignores)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
