"""sc-17139: the MiniMax-H3 hosted-artifact layout decision is schema-expressible TODAY.

``docs/sc-17139-minimax-h3-hosting-layout.md`` decides how the MiniMax-H3 family is
re-hosted on the SceneWorks HF org: only the DiT and the text encoder are tiered, both
VAEs are shared dense co-requisites, and ``transformer_ref`` becomes a second manifest
entry sharing one components mirror.

That decision is worth nothing if authoring it later turns out to need a schema change,
because a new manifest key sends the ``parity`` lane red (the live catalog is validated
against ``model-manifest.schema.json`` by
``test_builtin_models_manifest_satisfies_authoring_schema``). This module validates the
SKETCHED ``downloads[]`` shape against the real schema now, so sc-17158 inherits a
proven shape instead of discovering a blocked one.

It deliberately does NOT add the model to the catalog — that is sc-17158's job — and it
makes no assertion about whether ``minimax_h3`` is present in the live manifest, so it
neither blocks nor breaks when that story lands.

Three things are guarded:

  * the sketch validates (every key used really exists, every ``allOf`` conditional is
    satisfied);
  * the guard has TEETH — mutations the decision depends on being illegal are rejected;
  * the document and this fixture cannot drift apart, because the repo names, component
    ids and the two byte constants are asserted to appear verbatim in the doc.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "packages" / "schemas" / "model-manifest.schema.json"
DECISION_DOC_PATH = ROOT / "docs" / "sc-17139-minimax-h3-hosting-layout.md"

DIT_REPO = "SceneWorks/minimax-h3-mlx"
REF_DIT_REPO = "SceneWorks/minimax-h3-ref-mlx"
COMPONENTS_REPO = "SceneWorks/minimax-h3-components-mlx"

# Placeholder 40-hex pins. The real upload commits are backfilled by sc-17150; the
# schema pattern and the co-requisite/first-party pin audits only care about the SHAPE,
# and authoring the sketch with `main` or a short SHA is exactly the mistake those
# audits exist to catch.
DIT_REVISION = "0" * 40
REF_DIT_REVISION = "1" * 40
COMPONENTS_REVISION = "2" * 40

TIERS = ("q4", "q8", "bf16")
DEFAULT_TIER = "q4"
PLATFORMS = ["macos", "windows", "linux"]

# Exact upstream byte totals, read from the safetensors index manifests of
# MiniMaxAI/MiniMax-H3 at snapshot 939557dc319dd91227e30195a763f272ba7f8765. These two
# components are shipped DENSE in every tier, so unlike the tiered rows their sizes are
# known now and are not estimates. See §2 and §9 of the decision document.
VIDEO_VAE_BYTES = 10_415_475_936
AUDIO_VAE_BYTES = 605_306_340

# Tiered rows carry placeholder sizes: sc-17150 replaces them with measured hosted
# directory sizes. Any positive integer exercises the schema identically.
PLACEHOLDER_SIZE = 1


def _tier_rows(repo: str, revision: str) -> list[dict]:
    """The DiT tier matrix: one selectable, non-co-requisite row per tier, one repo."""
    return [
        {
            "provider": "huggingface",
            "repo": repo,
            "revision": revision,
            "variant": tier,
            **({"default": True} if tier == DEFAULT_TIER else {}),
            "files": [f"{tier}/*"],
            "platforms": list(PLATFORMS),
            "estimatedSizeBytes": PLACEHOLDER_SIZE,
            "footprint": {
                "diskSizeBytes": PLACEHOLDER_SIZE,
                "residentMemoryBytes": None,
                "peakMemoryBytes": None,
            },
        }
        for tier in TIERS
    ]


def _shared_component_rows() -> list[dict]:
    """The shared floor: a PER-TIER text encoder plus two TIER-AGNOSTIC dense VAEs.

    These rows are byte-for-byte identical between the two manifest entries — that
    identity is what makes the floor download once rather than twice.
    """
    rows = [
        {
            "provider": "huggingface",
            "repo": COMPONENTS_REPO,
            "revision": COMPONENTS_REVISION,
            "coRequisite": True,
            "componentId": "text_encoder",
            "variant": tier,
            "subdir": f"{tier}/text_encoder",
            "files": [f"{tier}/text_encoder/*"],
            "platforms": list(PLATFORMS),
            "estimatedSizeBytes": PLACEHOLDER_SIZE,
        }
        for tier in TIERS
    ]
    for component_id, size in (
        ("video_vae", VIDEO_VAE_BYTES),
        ("audio_vae", AUDIO_VAE_BYTES),
    ):
        rows.append(
            {
                "provider": "huggingface",
                "repo": COMPONENTS_REPO,
                "revision": COMPONENTS_REVISION,
                "coRequisite": True,
                "componentId": component_id,
                # No `variant`: a co-requisite without one is tier-agnostic and always
                # applies. Both VAEs stay dense bf16 in every tier.
                "subdir": component_id,
                "files": [f"{component_id}/*"],
                "platforms": list(PLATFORMS),
                "estimatedSizeBytes": size,
            }
        )
    return rows


def _sketched_downloads(repo: str, revision: str) -> list[dict]:
    return _tier_rows(repo, revision) + _shared_component_rows()


def _model_entry(model_id: str, name: str, downloads: list[dict]) -> dict:
    """A minimal schema-valid video model carrying the sketched downloads.

    Only the keys the schema REQUIRES are set (plus the `ui.promptGuide` the schema
    demands of a non-utility picker type). Capabilities, limits, defaults and the rest
    of the entry are sc-17158's work and are deliberately absent — this fixture proves
    the DOWNLOAD shape, nothing else.
    """
    return {
        "id": model_id,
        "name": name,
        "family": "minimax-h3",
        "type": "video",
        "ui": {
            "promptGuide": {
                "title": "MiniMax-H3 Prompt Guide",
                "path": "/prompt-guides/minimax-h3.md",
            }
        },
        "downloads": downloads,
    }


def _sketched_manifest() -> dict:
    return {
        "schemaVersion": 1,
        "models": [
            _model_entry("minimax_h3", "MiniMax-H3", _sketched_downloads(DIT_REPO, DIT_REVISION)),
            _model_entry(
                "minimax_h3_ref",
                "MiniMax-H3 (References)",
                _sketched_downloads(REF_DIT_REPO, REF_DIT_REVISION),
            ),
        ],
    }


def _validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _errors(manifest: dict) -> list:
    return sorted(_validator().iter_errors(manifest), key=lambda error: list(error.absolute_path))


def test_sketched_h3_download_shape_needs_no_schema_change():
    """sc-17139 acceptance: the decided layout is expressible with TODAY's schema.

    A failure here means the layout decision requires a `model-manifest.schema.json`
    change, and sc-17158 must land that change in the same PR as the manifest entry or
    the `parity` lane goes red.
    """
    errors = _errors(_sketched_manifest())
    assert not errors, (
        "the sc-17139 MiniMax-H3 download sketch violates model-manifest.schema.json — "
        "the layout needs a schema change:\n"
        + "\n".join(
            f"- {'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
    )


def test_sketched_shape_uses_only_keys_the_schema_declares():
    """`downloads.items` is `additionalProperties: false`, so an undeclared key is a hard
    schema error rather than a silently ignored field. Assert the intersection explicitly
    so a future reader can see WHICH keys the decision depends on.
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    item_schema = schema["properties"]["models"]["items"]["properties"]["downloads"]["items"]
    assert item_schema.get("additionalProperties") is False, (
        "this guard assumes downloads.items rejects undeclared keys; if that changed, an "
        "unknown key would no longer be caught at authoring time"
    )
    declared = set(item_schema["properties"])
    used: set[str] = set()
    for row in _sketched_downloads(DIT_REPO, DIT_REVISION):
        used.update(row)
    undeclared = sorted(used - declared)
    assert not undeclared, f"sketch uses download keys the schema does not declare: {undeclared}"
    # The four keys that carry the decision's weight, beyond the ordinary tier matrix.
    assert {"coRequisite", "componentId", "subdir", "variant"} <= used


def test_sketch_guard_has_teeth():
    """Mutation guard. Each mutation below breaks an invariant the layout RELIES on; if
    any of them validated, the positive test above would be decoration.
    """
    # `subdir` is valid ONLY on a coRequisite: true row (sc-14980). The layout uses it to
    # narrow the shared mirror to one component directory.
    no_flag = copy.deepcopy(_sketched_manifest())
    del no_flag["models"][0]["downloads"][3]["coRequisite"]
    assert any(
        error.validator == "required" and "coRequisite" in error.message
        for error in _errors(no_flag)
    ), "a subdir/componentId row without coRequisite: true must be rejected"

    # `variant` is a closed enum; the tier keys are not free-form.
    bad_variant = copy.deepcopy(_sketched_manifest())
    bad_variant["models"][0]["downloads"][0]["variant"] = "q6"
    assert any(
        error.validator == "enum" for error in _errors(bad_variant)
    ), "an off-enum quant tier must be rejected"

    # First-party rows MUST pin an immutable 40-hex commit; `main` resolves online and
    # hard-fails the offline pinned-SHA resolver.
    floating = copy.deepcopy(_sketched_manifest())
    floating["models"][0]["downloads"][0]["revision"] = "main"
    assert any(
        error.validator == "pattern" for error in _errors(floating)
    ), "a floating revision must be rejected by the 40-hex pattern"

    # An ABSOLUTE subdir is rejected by the pattern (it requires a leading path char).
    absolute = copy.deepcopy(_sketched_manifest())
    absolute["models"][0]["downloads"][3]["subdir"] = "/q8/text_encoder"
    assert any(
        error.validator == "pattern" for error in _errors(absolute)
    ), "an absolute subdir must be rejected by the subdir pattern"


def test_subdir_pattern_does_not_stop_traversal_only_safe_join_does():
    """Recorded because it is a trap, not because the sketch depends on it.

    `..` is spelled entirely from characters the `subdir` pattern allows
    (`^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$`), so `../q8/text_encoder` VALIDATES. The
    schema is not the traversal guard; `safe_join` in
    `crates/sceneworks-worker/src/model_jobs.rs` (`component_dir`) is, and it is the only
    one — the value arrives from the payload `modelManifestEntry`, unvalidated over the
    LAN jobs API. Anyone reading the pattern as a containment guarantee is wrong, and
    this test says so out loud.
    """
    traversing = copy.deepcopy(_sketched_manifest())
    traversing["models"][0]["downloads"][3]["subdir"] = "../q8/text_encoder"
    assert not _errors(traversing), (
        "if the schema learned to reject `..` in a subdir, tighten this test and the "
        "corresponding note in docs/sc-17139-minimax-h3-hosting-layout.md"
    )


def test_sketch_satisfies_the_audits_the_schema_cannot_express():
    """Three live invariants enforced by tests/test_builtin_manifest_audit.py and
    crates/sceneworks-core/src/builtin_manifests.rs rather than by JSON Schema. Checking
    them on the sketch means sc-17158 does not discover them after authoring.
    """
    for entry in _sketched_manifest()["models"]:
        downloads = entry["downloads"]
        primary = [row for row in downloads if not row.get("coRequisite")]
        co_reqs = [row for row in downloads if row.get("coRequisite")]

        # (1) At most one applicable non-co-requisite default per OS. Every row here
        # declares the same three platforms, so "at most one" is absolute.
        defaults = [row for row in primary if row.get("default")]
        assert len(defaults) == 1, f"{entry['id']}: exactly one default tier"
        assert defaults[0]["variant"] == DEFAULT_TIER

        # (2) Every SceneWorks-org row pins a 40-hex revision, and every co-requisite
        # pins one regardless of org (F-029, sc-13659).
        for row in downloads:
            if row["repo"].startswith("SceneWorks/") or row.get("coRequisite"):
                assert re.fullmatch(r"[0-9a-f]{40}", row["revision"]), (
                    f"{entry['id']}: {row['repo']} must pin a full 40-hex commit"
                )

        # (3) Tiers own DISJOINT file predicates, so a per-tier delete reclaims exactly
        # that tier's bytes instead of silently reclaiming nothing.
        predicates = [tuple(row["files"]) for row in primary]
        assert len(set(predicates)) == len(predicates), (
            f"{entry['id']}: tier file predicates must be disjoint"
        )

        # Rows sharing a componentId must be distinguishable by variant, or
        # resolve_co_requisites_for_tier refuses to guess at job time.
        by_component: dict[str, list] = {}
        for row in co_reqs:
            by_component.setdefault(row["componentId"], []).append(row.get("variant"))
        for component_id, variants in by_component.items():
            if len(variants) > 1:
                assert len(set(variants)) == len(variants) and None not in variants, (
                    f"{entry['id']}: {component_id} has multiple rows, so each needs a "
                    f"distinct variant; got {variants}"
                )


def test_shared_floor_is_identical_across_both_entries():
    """The whole point of the two-entry decision: `minimax_h3` and `minimax_h3_ref` share
    ONE components mirror, so the text encoder and both VAEs download once. If the
    co-requisite rows ever diverge in repo or revision the floor is paid twice — 11.04 GB
    at minimum, up to 38.5 GB with the bf16 text encoder.
    """
    base, ref = _sketched_manifest()["models"]
    base_co = [row for row in base["downloads"] if row.get("coRequisite")]
    ref_co = [row for row in ref["downloads"] if row.get("coRequisite")]
    assert base_co == ref_co, (
        "the co-requisite rows must be byte-identical across both MiniMax-H3 entries"
    )
    assert {row["componentId"] for row in base_co} == {"text_encoder", "video_vae", "audio_vae"}
    assert all(row["repo"] == COMPONENTS_REPO for row in base_co)
    # The two entries' PRIMARY sides must NOT share a repo: one repo per entry is what
    # keeps install state, stale detection and whole-model delete correct.
    base_primary = {row["repo"] for row in base["downloads"] if not row.get("coRequisite")}
    ref_primary = {row["repo"] for row in ref["downloads"] if not row.get("coRequisite")}
    assert base_primary == {DIT_REPO} and ref_primary == {REF_DIT_REPO}
    assert not base_primary & ref_primary


def test_decision_document_and_this_fixture_cannot_drift():
    """The document is the deliverable; this fixture is its executable half. Assert the
    load-bearing constants appear verbatim in the doc so neither can be edited alone.
    """
    doc = DECISION_DOC_PATH.read_text(encoding="utf-8")
    for token in (
        DIT_REPO,
        REF_DIT_REPO,
        COMPONENTS_REPO,
        "minimax_h3_ref",
        "text_encoder",
        "video_vae",
        "audio_vae",
        f"{VIDEO_VAE_BYTES:,}",
        f"{AUDIO_VAE_BYTES:,}",
    ):
        assert token in doc, f"{DECISION_DOC_PATH.name} no longer mentions {token!r}"
