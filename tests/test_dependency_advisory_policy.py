from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and earlier; CI is pinned to 3.12.
    import tomli as tomllib

from scripts.ci.check_advisory_policy import (
    MAXIMUM_EXCEPTION_LIFETIME_DAYS,
    PolicyError,
    validate_documents,
    validate_policy,
)


ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 8, 27)
ADVISORY = "RUSTSEC-2026-9999"
REPRESENTATIVE_PLATFORM_PACKAGES = frozenset(
    {
        "crossbeam-epoch",
        "cudarc",
        "metal",
        "mlx-gen",
        "objc2-metal",
        "plist",
        "quick-xml",
        "runtime-cuda",
        "runtime-macos",
    }
)


def documents() -> tuple[dict, dict]:
    deny = {
        "graph": {
            "all-features": True,
            "targets": [
                {"triple": "x86_64-unknown-linux-gnu"},
                {"triple": "aarch64-apple-darwin"},
                {"triple": "x86_64-pc-windows-msvc"},
            ],
        },
        "advisories": {
            "ignore": [ADVISORY],
            "unmaintained": "workspace",
            "unsound": "workspace",
            "yanked": "warn",
        },
    }
    metadata = {
        "version": 1,
        "ignore": [
            {
                "advisory": ADVISORY,
                "reason": "A compatible upstream replacement is not available yet.",
                "reachability": "The affected API is not reachable from shipped application paths.",
                "owner": "@michaeltrefry",
                "expires": TODAY + timedelta(days=MAXIMUM_EXCEPTION_LIFETIME_DAYS),
            }
        ],
    }
    return deny, metadata


def test_committed_policy_is_current_and_has_no_exceptions() -> None:
    assert validate_policy(ROOT / "deny.toml", ROOT / "advisory-ignores.toml", today=TODAY) == ()


def test_policy_requires_all_features_and_every_shipped_target() -> None:
    deny, metadata = documents()
    deny["graph"]["all-features"] = False
    with pytest.raises(PolicyError, match="all-features must be true"):
        validate_documents(deny, metadata, today=TODAY)

    deny, metadata = documents()
    deny["graph"]["targets"].pop()
    with pytest.raises(PolicyError, match="missing x86_64-pc-windows-msvc"):
        validate_documents(deny, metadata, today=TODAY)

    deny, metadata = documents()
    deny["graph"]["targets"][0]["features"] = ["backend-candle"]
    with pytest.raises(PolicyError, match="noncanonical features"):
        validate_documents(deny, metadata, today=TODAY)


@pytest.mark.parametrize(
    ("section", "setting", "value"),
    [
        ("graph", "exclude", ["sceneworks-desktop"]),
        ("graph", "exclude-dev", True),
        ("graph", "exclude-unpublished", True),
        ("graph", "no-default-features", True),
        ("graph", "features", ["backend-candle"]),
        ("advisories", "db-urls", ["https://example.invalid/advisory-db"]),
        ("advisories", "db-path", "/tmp/partial-advisory-db"),
        ("advisories", "disable-yank-checking", True),
        ("advisories", "git-fetch-with-cli", True),
        ("advisories", "ignore-yanked", ["spin@0.9.8"]),
        ("advisories", "maximum-db-staleness", "P9999D"),
        ("advisories", "unused-ignored-advisory", "allow"),
        ("advisories", "version", 2),
    ],
)
def test_rejects_every_noncanonical_graph_or_advisory_setting(
    section: str, setting: str, value: object
) -> None:
    deny, metadata = documents()
    deny[section][setting] = value
    with pytest.raises(PolicyError, match=setting):
        validate_documents(deny, metadata, today=TODAY)


@pytest.mark.parametrize(
    "setting",
    [
        "all-features",
        "exclude",
        "exclude-dev",
        "exclude-unpublished",
        "features",
        "no-default-features",
        "targets",
    ],
)
def test_rejects_deprecated_root_graph_overrides(setting: str) -> None:
    deny, metadata = documents()
    deny[setting] = True
    with pytest.raises(PolicyError, match=setting):
        validate_documents(deny, metadata, today=TODAY)


@pytest.mark.parametrize(
    ("setting", "value", "expected"),
    [
        ("unmaintained", "allow", "workspace"),
        ("unsound", "allow", "workspace"),
        ("unsound", "all", "workspace"),
        ("yanked", "allow", "warn"),
    ],
)
def test_advisory_lint_scopes_must_match_canonical_policy(
    setting: str, value: str, expected: str
) -> None:
    deny, metadata = documents()
    deny["advisories"][setting] = value
    with pytest.raises(PolicyError, match=rf"{setting} must be '{expected}'"):
        validate_documents(deny, metadata, today=TODAY)


@pytest.mark.parametrize(
    ("section", "setting"),
    [
        ("graph", "all-features"),
        ("graph", "targets"),
        ("advisories", "ignore"),
        ("advisories", "unmaintained"),
        ("advisories", "unsound"),
        ("advisories", "yanked"),
    ],
)
def test_rejects_missing_canonical_policy_settings(section: str, setting: str) -> None:
    deny, metadata = documents()
    del deny[section][setting]
    with pytest.raises(PolicyError, match=rf"missing .*{setting}"):
        validate_documents(deny, metadata, today=TODAY)


def test_all_feature_metadata_resolves_representative_platform_packages() -> None:
    result = subprocess.run(
        ["cargo", "metadata", "--locked", "--all-features", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    metadata = json.loads(result.stdout)
    package_names = {package["id"]: package["name"] for package in metadata["packages"]}
    resolved = {package_names[node["id"]] for node in metadata["resolve"]["nodes"]}
    assert not REPRESENTATIVE_PLATFORM_PACKAGES - resolved


def test_requires_reason_reachability_owner_and_expiry() -> None:
    for missing in ("reason", "reachability", "owner", "expires"):
        deny, metadata = documents()
        del metadata["ignore"][0][missing]
        with pytest.raises(PolicyError, match=f"missing fields: {missing}"):
            validate_documents(deny, metadata, today=TODAY)

    for field in ("reason", "reachability"):
        deny, metadata = documents()
        metadata["ignore"][0][field] = "todo"
        with pytest.raises(PolicyError, match="at least 20 characters"):
            validate_documents(deny, metadata, today=TODAY)

    deny, metadata = documents()
    metadata["ignore"][0]["owner"] = "someone"
    with pytest.raises(PolicyError, match="accountable @user"):
        validate_documents(deny, metadata, today=TODAY)


@pytest.mark.parametrize("expiry", [TODAY, date(2026, 8, 26), "2026-11-30"])
def test_rejects_expired_or_non_date_expiry(expiry: date | str) -> None:
    deny, metadata = documents()
    metadata["ignore"][0]["expires"] = expiry
    with pytest.raises(PolicyError, match="must be after|unquoted TOML local date"):
        validate_documents(deny, metadata, today=TODAY)


@pytest.mark.parametrize(
    "expiry",
    [
        TODAY + timedelta(days=MAXIMUM_EXCEPTION_LIFETIME_DAYS + 1),
        date(9999, 12, 31),
    ],
)
def test_rejects_overlong_advisory_exceptions(expiry: date) -> None:
    deny, metadata = documents()
    metadata["ignore"][0]["expires"] = expiry
    with pytest.raises(PolicyError, match=f"within {MAXIMUM_EXCEPTION_LIFETIME_DAYS} days"):
        validate_documents(deny, metadata, today=TODAY)


def test_deny_and_metadata_sets_must_match_exactly() -> None:
    deny, metadata = documents()
    deny["advisories"]["ignore"].append("RUSTSEC-2026-9998")
    with pytest.raises(PolicyError, match="missing metadata"):
        validate_documents(deny, metadata, today=TODAY)

    deny, metadata = documents()
    deny["advisories"]["ignore"] = []
    with pytest.raises(PolicyError, match="metadata without a deny.toml ignore"):
        validate_documents(deny, metadata, today=TODAY)


def test_rejects_duplicates_and_unknown_fields() -> None:
    deny, metadata = documents()
    deny["advisories"]["ignore"].append(ADVISORY)
    with pytest.raises(PolicyError, match="duplicate advisory"):
        validate_documents(deny, metadata, today=TODAY)

    deny, metadata = documents()
    metadata["ignore"].append(deepcopy(metadata["ignore"][0]))
    with pytest.raises(PolicyError, match="duplicate advisory"):
        validate_documents(deny, metadata, today=TODAY)

    deny, metadata = documents()
    metadata["ignore"][0]["ticket"] = "SC-11295"
    with pytest.raises(PolicyError, match="unknown fields: ticket"):
        validate_documents(deny, metadata, today=TODAY)


def test_committed_policy_versions_clear_target_advisories() -> None:
    with (ROOT / "Cargo.lock").open("rb") as source:
        lock = tomllib.load(source)
    versions = {package["name"]: package["version"] for package in lock["package"]}
    assert versions["anyhow"] == "1.0.103"
    assert versions["memmap2"] == "0.9.11"
    assert versions["plist"] == "1.10.0"
    assert versions["quick-xml"] == "0.41.0"
    assert versions["crossbeam-epoch"] == "0.9.20"
