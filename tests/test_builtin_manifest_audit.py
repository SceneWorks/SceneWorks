"""Structural audits of the live ``config/manifests/builtin.models.jsonc``.

These assertions guard the shipped catalog config, NOT worker runtime behaviour.
They were originally embedded in ``tests/test_worker_image_adapters.py`` (the
retired Python worker's adapter suite), which meant a future ``apps/worker``
deletion (epic 8283, Python eradication) would silently take the live-catalog
gates down with it. sc-8861 (F-059) extracts them here so the coverage survives
that deletion: this module parses the manifest file DIRECTLY and imports no
``scene_worker`` symbol at module scope.

The manifest is a JSONC file (the Rust API owns the canonical parser). Two
self-contained readers are inlined below rather than imported from
``tests/worker_runtime_shared.py`` (that helper module top-imports
``scene_worker``, which would re-couple these audits to the retired worker):

  * ``_strip_jsonc_comments`` + ``_load_builtin_models_manifest`` parse the file
    to a dict for the capability / UI-wiring audits.
  * ``_manifest_brace_walker`` walks balanced braces so a URL containing ``//``
    inside an entry doesn't trip a naive comment strip; used by the per-model
    ``mlx`` block audits.

The three character_image ENGINE-WIRING guards that used to live here additionally
cross-referenced the retired Python worker's ``MODEL_TARGETS`` table via a lazy
``importorskip``, so they degraded to a clean SKIP once ``apps/worker`` was deleted
(epic 8283) — losing their coverage. sc-9513 (F-059 follow-up) reimplemented them
against the Rust worker's own character-image engine wiring, reading this SAME
embedded manifest, in ``crates/sceneworks-worker/src/engines.rs`` (the tests
``character_image_capability_implies_engine_or_tuning_declaration`` /
``kolors_declares_strict_pose_controlnet`` /
``models_with_engine_block_advertise_character_image``). This module now imports no
``scene_worker`` symbol at all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "manifests" / "builtin.models.jsonc"
SCHEMA_PATH = ROOT / "packages" / "schemas" / "model-manifest.schema.json"


def _strip_jsonc_comments(body: str) -> str:
    """Mirror scripts/check-scaffold.mjs::stripJsoncComments so the audit reads
    the real `config/manifests/builtin.models.jsonc` without a JSONC dependency.
    Walks the body char-by-char, suppressing // line and /* block */ comments
    but leaving them intact when they appear inside string literals.
    """
    result: list[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(body):
        char = body[i]
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue
        if char == "/" and nxt == "/":
            while i < len(body) and body[i] != "\n":
                i += 1
            result.append("\n")
            continue
        if char == "/" and nxt == "*":
            i += 2
            while i < len(body) - 1 and not (body[i] == "*" and body[i + 1] == "/"):
                i += 1
            i += 2
            continue
        result.append(char)
        i += 1
    return "".join(result)


def _load_builtin_models_manifest() -> dict:
    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    return json.loads(_strip_jsonc_comments(raw))


def test_builtin_models_manifest_satisfies_authoring_schema():
    """sc-12338: the builtin catalog's $schema is an enforced CI contract."""
    manifest = _load_builtin_models_manifest()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: list(error.absolute_path),
    )
    assert not errors, "builtin.models.jsonc violates model-manifest.schema.json:\n" + "\n".join(
        f"- {'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in errors
    )


def test_builtin_schema_rejects_an_unknown_closed_model_key():
    """Mutation guard: a typo/decorative builtin key must make the CI gate fail."""
    manifest = _load_builtin_models_manifest()
    manifest["models"][0]["recommendded"] = True
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    assert any("recommendded" in error.message for error in errors)


def _sample_audio_model_entry() -> dict:
    """A representative `type: "audio"` entry exercising every field of the new
    `audio` capability sub-block (sc-13401, epic 13400). Not a shipped model —
    real audio entries land in sc-13402 — so it lives in the test, not the
    builtin manifest.
    """
    return {
        "id": "sample_audio_speech",
        "name": "Sample Audio Speech",
        "type": "audio",
        "audio": {
            "voices": [
                {
                    "id": "af_heart",
                    "label": "Heart",
                    "gender": "female",
                    "accent": "american",
                    "language": "en-US",
                },
                {"id": "bm_george", "gender": "male", "accent": "british"},
            ],
            "languages": ["en-US", "en-GB"],
            "sampleRates": [24000, 48000],
            "maxDurationSecs": 30.0,
            "editModes": ["extend", "inpaint", "cover"],
            "supportsMultiSpeaker": True,
            "maxSpeakers": 2,
            "conditioning": ["AudioEdit", "ReferenceAudio", "VoiceEmbedding"],
        },
    }


def test_schema_accepts_audio_type_and_audio_sub_block():
    """sc-13401: a `type: "audio"` entry with a populated `audio` sub-block
    validates against the authoring schema (the new sibling of mlx/candle)."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    manifest = {"schemaVersion": 1, "models": [_sample_audio_model_entry()]}
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    assert not errors, "sample audio entry must satisfy the schema:\n" + "\n".join(
        f"- {'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in errors
    )


def test_schema_rejects_unknown_field_under_audio_sub_block():
    """Mutation guard: the `audio` block is additionalProperties:false, so a typo
    / undeclared field under it must fail validation."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    entry = _sample_audio_model_entry()
    entry["audio"]["bogusField"] = True
    manifest = {"schemaVersion": 1, "models": [entry]}
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    assert any("bogusField" in error.message for error in errors), (
        "an unknown key under `audio` must be rejected by additionalProperties:false"
    )


def test_schema_rejects_audio_voice_without_id():
    """A voice object requires `id` so the picker always has a backend key."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    entry = _sample_audio_model_entry()
    entry["audio"]["voices"] = [{"label": "No Id", "gender": "female"}]
    manifest = {"schemaVersion": 1, "models": [entry]}
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    # Discriminate on the jsonschema error's shape, not a substring of its
    # message: a loose `"id" in error.message` incidentally matches unrelated
    # errors (e.g. a type-enum error listing "video", which contains "id"), so
    # it could false-green under a full schema revert. Pin the `required`
    # keyword, its `["id"]` value, and the path at the voice item instead — this
    # only holds while the voice object's `required: ["id"]` is present.
    assert any(
        error.validator == "required"
        and error.validator_value == ["id"]
        and list(error.absolute_path) == ["models", 0, "audio", "voices", 0]
        for error in errors
    ), (
        "a voice entry without `id` must be rejected by the voice object's "
        "required:['id'] (a `required` error at models/0/audio/voices/0)"
    )


def test_schema_rejects_unknown_model_type():
    """Negative control: the `type` enum still rejects an out-of-set value even
    after `audio` was added, so the enum is not accidentally open."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    entry = _sample_audio_model_entry()
    entry["type"] = "hologram"
    manifest = {"schemaVersion": 1, "models": [entry]}
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    assert any("hologram" in error.message or "enum" in error.message for error in errors)


# The seeded audio catalog (sc-13402, epic 13400). Each id is a live candle-audio
# provider registered in `crates/audio/candle-audio-catalog`; the second element is
# the `audio` capability sub-block key that MUST be populated for the Audio Studio to
# build its pickers/mode gates without probing the backend.
_SEEDED_AUDIO_MODELS = {
    "kokoro_82m": "voices",
    "moss_sfx_v2": "sampleRates",
    "acestep_v15_turbo": "editModes",
    "openvoice_v2": "conditioning",
    "chatterbox_ve": "conditioning",
}


def test_builtin_manifest_ships_the_seeded_audio_models():
    """sc-13402: the five live audio providers (Kokoro, MOSS-SFX, ACE-Step,
    OpenVoice V2, Chatterbox-VE) are seeded as `type: "audio"` entries, each
    carrying a populated `audio` capability sub-block (not just the schema-legal
    shape proven by sc-13401, but the ACTUAL shipped entries). Kokoro is the
    recommended Speech model."""
    manifest = _load_builtin_models_manifest()
    by_id = {m.get("id"): m for m in manifest["models"]}

    for model_id, required_cap_key in _SEEDED_AUDIO_MODELS.items():
        entry = by_id.get(model_id)
        assert entry is not None, f"seeded audio model {model_id} is missing from the manifest"
        assert entry.get("type") == "audio", f"{model_id} must be type:audio"
        audio = entry.get("audio")
        assert isinstance(audio, dict) and audio, f"{model_id} must carry a populated `audio` block"
        assert required_cap_key in audio, (
            f"{model_id}.audio must advertise `{required_cap_key}` (populated from backend "
            f"Capabilities, not an empty stub)"
        )
        # Every audio entry must be installable/downloadable like image/video models.
        downloads = entry.get("downloads") or []
        assert downloads and downloads[0].get("repo"), (
            f"{model_id} must define a download entry with a repo (install/download parity)"
        )
        assert entry.get("paths", {}).get("model"), f"{model_id} must define paths.model"

    # Kokoro's real voice surface: the 28 English packs the pinned snapshot ships,
    # each an object with an `id` (discriminates against an empty/placeholder list).
    kokoro_voices = by_id["kokoro_82m"]["audio"]["voices"]
    assert len(kokoro_voices) == 28, "Kokoro advertises its 28 shipped English voices"
    assert all(isinstance(v, dict) and v.get("id") for v in kokoro_voices)
    assert by_id["kokoro_82m"].get("recommended") is True, "Kokoro is the recommended Speech model"

    # ACE-Step's real edit surface (Conditioning::AudioEdit → repaint task modes).
    assert set(by_id["acestep_v15_turbo"]["audio"]["editModes"]) == {
        "inpaint",
        "repaint",
        "extend",
    }
    assert "AudioEdit" in by_id["acestep_v15_turbo"]["audio"]["conditioning"]


def _duplicate_default_downloads(manifest: dict) -> list[str]:
    """Return model/platform pairs with ambiguous primary download selection."""
    ambiguous: list[str] = []
    for model in manifest["models"]:
        downloads = model.get("downloads", [])
        for platform in ("macos", "windows", "linux"):
            defaults = [
                download
                for download in downloads
                if download.get("default") is True
                and download.get("coRequisite") is not True
                and (
                    "platforms" not in download
                    or platform in download.get("platforms", [])
                )
            ]
            if len(defaults) > 1:
                ambiguous.append(f"{model['id']}:{platform}")
    return ambiguous


def test_builtin_download_defaults_are_unique_per_platform():
    """A model may have one primary default per OS, never two applicable defaults."""
    assert not _duplicate_default_downloads(_load_builtin_models_manifest())


def test_download_default_guard_rejects_an_ambiguous_platform_mutation():
    """Mutation guard for the platform-aware replacement of schema maxContains."""
    manifest = _load_builtin_models_manifest()
    model = next(model for model in manifest["models"] if model["id"] == "wan_2_2")
    windows_download = next(
        download
        for download in model["downloads"]
        if download.get("variant") == "q8" and "windows" in download.get("platforms", [])
    )
    windows_download["default"] = True
    assert _duplicate_default_downloads(manifest) == ["wan_2_2:windows", "wan_2_2:linux"]


# ---------------------------------------------------------------------------
# F-029 download-revision pin authority (sc-13659).
#
# A download entry's optional `revision` is the immutable-commit pin the worker
# fetches into `snapshots/<sha>/`; absent means the worker resolves `main`. The
# JSON Schema constrains its FORMAT (`^[0-9a-f]{40}$`), but the
# "coRequisite: true REQUIRES a revision" invariant lives here (and in the Rust
# builtin_manifests.rs backstop) because JSON Schema cannot grandfather the
# sc-13591 pin migration still in flight.
# ---------------------------------------------------------------------------

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# `(model_id, repo)` co-requisite download pairs whose F-029 pin migration is
# still IN FLIGHT under sc-13591. Each is a KNOWN, tracked gap: the immutable
# commit SHA lives in the sc-13591 inventory but is applied by a later per-family
# story, not sc-13659 (schema + plumbing + enforcement only — it must not add
# real pins). A brand-new co-requisite may NOT join this list; pin its `revision`
# instead. Kept in lockstep with the identical Rust allowlist in
# crates/sceneworks-core/src/builtin_manifests.rs.
_COREQUISITE_REVISION_MIGRATION_PENDING: frozenset[tuple[str, str]] = frozenset(
    {
        ("qwen_image", "SceneWorks/qwen-image-2512-fun-controlnet-union"),
        # ("ltx_2_3", "SceneWorks/ltx-2.3-mlx") pinned in sc-13683 (the gemma coRequisite now carries
        # the full 40-hex LTX_BUNDLE_REVISION); removed here + in the Rust twin to keep both green.
        ("ltx_2_3_eros", "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments"),
        ("wan_2_2_t2v_14b", "lightx2v/Wan2.2-Lightning"),
        ("wan_2_2_i2v_14b", "lightx2v/Wan2.2-Lightning"),
        ("pid_qwenimage", "SceneWorks/gemma-2-2b-it"),
        ("pid_flux", "SceneWorks/gemma-2-2b-it"),
        ("pid_flux2", "SceneWorks/gemma-2-2b-it"),
        ("pid_sdxl", "SceneWorks/gemma-2-2b-it"),
    }
)


def _corequisite_revision_gaps(manifest: dict) -> set[tuple[str, str]]:
    """`(model_id, repo)` co-requisite pairs NOT pinned to a full 40-hex SHA."""
    gaps: set[tuple[str, str]] = set()
    for model in manifest["models"]:
        for download in model.get("downloads", []):
            if download.get("coRequisite") is not True:
                continue
            revision = download.get("revision")
            if not (isinstance(revision, str) and _FULL_SHA_RE.match(revision)):
                gaps.add((model["id"], download.get("repo", "")))
    return gaps


def test_corequisite_downloads_pin_a_full_sha_revision():
    """F-029 (sc-13659): every coRequisite download pins an immutable 40-hex commit.

    A co-requisite is a FETCH-ALL companion the runtime resolves offline via a
    pinned-SHA `hf_get_pinned` reading `snapshots/<sha>/`; leaving it on `main`
    lands the wrong snapshot and hard-fails offline. The only tolerated gaps are
    the sc-13591 pins still being migrated by later stories.
    """
    manifest = _load_builtin_models_manifest()
    unexpected = _corequisite_revision_gaps(manifest) - _COREQUISITE_REVISION_MIGRATION_PENDING
    assert not unexpected, (
        "co-requisite downloads must pin a 40-hex commit SHA (F-029, sc-13659); these are "
        f"unpinned and NOT tracked for the sc-13591 migration: {sorted(unexpected)}"
    )


def test_corequisite_revision_migration_allowlist_has_no_stale_entries():
    """Self-cleaning guard: an allowlist row that no longer names an unpinned
    co-requisite must be deleted, so pinning one in a later sc-13591 story forces
    its removal instead of the allowlist silently excusing an already-compliant
    entry (a test asserting a default is a false green — the allowlist must shrink).
    """
    manifest = _load_builtin_models_manifest()
    stale = _COREQUISITE_REVISION_MIGRATION_PENDING - _corequisite_revision_gaps(manifest)
    assert not stale, (
        "stale F-029 migration allowlist entries (now pinned or removed) must be deleted from "
        f"_COREQUISITE_REVISION_MIGRATION_PENDING: {sorted(stale)}"
    )


def test_corequisite_revision_guard_flags_a_new_unpinned_corequisite():
    """Mutation guard: the rule is LIVE for new entries, not decoration. A brand-new
    co-requisite with no revision (and not on the migration allowlist) is caught.
    """
    manifest = _load_builtin_models_manifest()
    kokoro = next(model for model in manifest["models"] if model["id"] == "kokoro_82m")
    kokoro.setdefault("downloads", []).append(
        {"provider": "huggingface", "repo": "example/new-corequisite", "coRequisite": True}
    )
    new_pair = ("kokoro_82m", "example/new-corequisite")
    assert new_pair in _corequisite_revision_gaps(manifest)
    assert new_pair not in _COREQUISITE_REVISION_MIGRATION_PENDING
    unexpected = _corequisite_revision_gaps(manifest) - _COREQUISITE_REVISION_MIGRATION_PENDING
    assert new_pair in unexpected


def _model_entry_with_download(download: dict) -> dict:
    """A minimal schema-valid model entry carrying a single `downloads` entry, for
    exercising the download-item schema in isolation."""
    return {
        "id": "sample_pinned_model",
        "name": "Sample Pinned Model",
        "type": "image",
        "downloads": [download],
    }


def test_schema_pins_download_revision_to_a_40hex_sha():
    """sc-13659: the authoring schema constrains `revision` to a full 40-hex commit
    (the F-029 pin authority), accepting a valid SHA and rejecting a branch/tag/
    short/uppercase/wrong-length value via the `pattern` keyword.
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    def revision_errors(revision: str) -> list:
        manifest = {
            "schemaVersion": 1,
            "models": [
                _model_entry_with_download(
                    {
                        "provider": "huggingface",
                        "repo": "namespace/model",
                        "files": [],
                        "revision": revision,
                    }
                )
            ],
        }
        return list(validator.iter_errors(manifest))

    assert not revision_errors("a" * 40), "a full 40-hex SHA must satisfy the schema"
    for bad in ("main", "v1.0", "abc123", "A" * 40, "g" * 40, "a" * 39, "a" * 41):
        errors = revision_errors(bad)
        # Discriminate on the failing keyword so a full schema revert (dropping the
        # pattern) turns this red rather than passing on some unrelated error.
        assert any(error.validator == "pattern" for error in errors), (
            f"revision {bad!r} must be rejected by the 40-hex pattern"
        )


def test_schema_accepts_a_component_id_on_a_corequisite_download():
    """sc-13679: a coRequisite download may carry a `componentId` — the explicit repo→component
    mapping the worker's `resolve_co_requisites` seam reads to stage `LoadSpec::components`. The
    authoring schema constrains it to lowercase snake_case (same shape as a descriptor id), accepting
    a valid id and rejecting capitals / hyphens / a leading digit / empty via the `pattern` keyword.
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    def component_errors(component_id: str) -> list:
        manifest = {
            "schemaVersion": 1,
            "models": [
                _model_entry_with_download(
                    {
                        "provider": "huggingface",
                        "repo": "ResembleAI/chatterbox",
                        "files": ["ve.safetensors"],
                        "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
                        "coRequisite": True,
                        "componentId": component_id,
                    }
                )
            ],
        }
        return list(validator.iter_errors(manifest))

    assert not component_errors("voice_embedding"), "a lowercase snake_case componentId must validate"
    for bad in ("Perth", "voice-embedding", "1codec", ""):
        errors = component_errors(bad)
        # Discriminate on the failing keyword so dropping the pattern turns this red rather than
        # passing on some unrelated error.
        assert any(error.validator == "pattern" for error in errors), (
            f"componentId {bad!r} must be rejected by the snake_case pattern"
        )


def test_manifest_constraint_contract_registry_is_complete_and_live():
    """sc-12304: constraint declarations may not silently become decoration.

    The schema is the author-facing registry; this test makes its custom contract
    annotations a CI gate. It checks both directions (manifest -> registry and
    registry -> manifest), and binding entries must name production readers that
    contain the exact key. Advisory/descriptive entries are explicitly allowed not
    to reject requests, which is materially different from an accidental dead key.
    """
    manifest = _load_builtin_models_manifest()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    model_properties = schema["properties"]["models"]["items"]["properties"]

    declared: set[str] = set()
    for model in manifest["models"]:
        declared.update(f"limits.{key}" for key in model.get("limits", {}))
        for backend in ("mlx", "candle"):
            block = model.get(backend, {})
            if "minMemoryGb" in block:
                declared.add(f"{backend}.minMemoryGb")
            declared.update(f"{backend}.limits.{key}" for key in block.get("limits", {}))

    limits_properties = model_properties["limits"]["properties"]
    registry = {f"limits.{key}": value for key, value in limits_properties.items()}
    for backend in ("mlx", "candle"):
        backend_properties = model_properties[backend]["properties"]
        if "minMemoryGb" in backend_properties:
            registry[f"{backend}.minMemoryGb"] = backend_properties["minMemoryGb"]
        backend_limits = backend_properties.get("limits", {})
        if "$ref" in backend_limits:
            sampler_properties = schema["$defs"]["samplerLimits"]["properties"]
            for key, value in sampler_properties.items():
                registry[f"{backend}.limits.{key}"] = value

    allowed_undeclared = {
        path
        for path, contract in registry.items()
        if contract.get("x-sceneworks-allow-undeclared")
        or (
            path.startswith("candle.limits.")
            and model_properties["candle"]["properties"]["limits"].get(
                "x-sceneworks-allow-undeclared"
            )
        )
    }
    assert declared <= set(registry) and set(registry) - declared <= allowed_undeclared, (
        "constraint contract drift: every declared constraint must be registered, "
        "and every registry entry must be exercised by the builtin manifest; "
        f"unregistered={sorted(declared - set(registry))}, "
        f"undeclared={sorted(set(registry) - declared)}"
    )

    allowed_classes = {"binding", "advisory", "descriptive"}
    for path, contract in registry.items():
        classification = contract.get("x-sceneworks-contract")
        assert classification in allowed_classes, f"{path}: missing/invalid contract classification"
        readers = contract.get("x-sceneworks-readers", [])
        exemption = contract.get("x-sceneworks-reader-exemption")
        assert readers or exemption, (
            f"{path}: every contract needs anchored production readers or an explicit tracked exemption"
        )
        if exemption:
            assert re.search(r"\bsc-\d+\b", exemption), (
                f"{path}: reader exemption must cite a tracked Shortcut story"
            )
        for reader in readers:
            assert set(reader) == {"path", "anchor"}, f"{path}: malformed reader metadata"
            reader_path = ROOT / reader["path"]
            assert reader_path.is_file(), f"{path}: reader does not exist: {reader['path']}"
            assert reader["anchor"] in reader_path.read_text(encoding="utf-8"), (
                f"{path}: reader {reader['path']} no longer contains anchor {reader['anchor']!r}"
            )


def _manifest_brace_walker():
    # Helper for the mlx-block manifest tests. Returns (raw, find_entry_block,
    # find_mlx_block) that walk balanced braces so a URL containing `//` (in
    # the entry text) doesn't trip a naive jsonc strip.
    raw = MANIFEST_PATH.read_text(encoding="utf-8")

    def find_balanced_block(start_index: int) -> str:
        depth = 0
        for index in range(start_index, len(raw)):
            ch = raw[index]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start_index : index + 1]
        raise AssertionError(f"unterminated brace block from index {start_index}")

    def find_entry_block(model_id: str) -> str:
        anchor = raw.index(f'"id": "{model_id}"')
        start = raw.rfind("{", 0, anchor)
        assert start != -1, f"entry start brace for {model_id} not found"
        return find_balanced_block(start)

    def find_mlx_block(entry_block: str) -> str:
        match = re.search(r'"mlx"\s*:\s*\{', entry_block)
        assert match, "entry block has no mlx block"
        # Resolve the entry block's position in the raw manifest, then walk
        # balanced braces from the actual opening brace so nested limits {...}
        # are captured (Qwen carries a sampler/scheduler limits override, FLUX
        # does not).
        entry_start = raw.index(entry_block)
        mlx_open = entry_start + match.end() - 1
        return find_balanced_block(mlx_open)

    return raw, find_entry_block, find_mlx_block


# ---------------------------------------------------------------------------
# Per-model `mlx` block structural audits (pure manifest; brace-walker based).
# Extracted from tests/test_worker_image_adapters.py (sc-8861 / F-059).
# ---------------------------------------------------------------------------


def test_flux_manifest_has_mlx_block():
    # Manifest-driven auto-dispatch + Model Manager memory tier (sc-1970).
    # The Rust API owns the canonical jsonc parser; here we just confirm both
    # FLUX entries carry an `mlx` block and the contents look right.
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()

    for model_id in ("flux_schnell", "flux_dev"):
        block = find_entry_block(model_id)
        mlx_block = find_mlx_block(block)
        quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
        mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
        assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
            f"{model_id} mlx.quantize must be a supported quant level (sc-1970)"
        )
        assert mem_match and int(mem_match.group(1)) > 0, (
            f"{model_id} mlx.minMemoryGb must be a positive int (sc-1970)"
        )


def test_qwen_image_manifest_has_mlx_block():
    # sc-1972: qwen_image carries an mlx block + sampler/scheduler limits
    # override (mflux's loop is sealed on "linear" — match the wan_2_2
    # precedent of restricting the menu to default-only when the MLX path is
    # the active backend, epic 1753 §14).
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("qwen_image")
    mlx_block = find_mlx_block(block)
    quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
        "qwen_image mlx.quantize must be a supported quant level (sc-1972)"
    )
    assert mem_match and int(mem_match.group(1)) > 0, (
        "qwen_image mlx.minMemoryGb must be a positive int (sc-1972)"
    )
    # MLX sampler/scheduler menu (epic 7114 P5, sc-7126): the native MLX engine now
    # routes through the unified curated sampler/scheduler framework (the old mflux
    # linear-only loop is gone), so the mlx block advertises the curated menu.
    assert '"dpmpp_2m"' in mlx_block and '"uni_pc"' in mlx_block, (
        "qwen_image mlx must advertise the curated sampler menu (epic 7114)"
    )
    assert '"sgm_uniform"' in mlx_block, (
        "qwen_image mlx must advertise the curated scheduler menu (epic 7114)"
    )


def test_flux2_true_v2_manifest_install_time_conversion():
    # sc-2235: the entry must declare the install-time conversion contract the
    # Rust convert job + adapter rely on.
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("flux2_klein_9b_true_v2")
    assert '"macOnly": true' in block
    assert '"adapter": "mlx_flux2"' in block
    # Only the bf16 single-file is pulled (not the whole 73 GB repo).
    assert "Flux2-Klein-9B-True-v2-bf16.safetensors" in block
    # Undistilled defaults differ from the 4-step distill.
    assert re.search(r'"steps"\s*:\s*24', block)

    mlx_block = find_mlx_block(block)
    assert '"requiresConversion": true' in mlx_block
    assert '"converter": "flux2_klein_diffusers"' in mlx_block
    assert '"convertSourceRepo": "wikeeyang/Flux2-Klein-9B-True-V2"' in mlx_block
    assert '"convertBaseRepo": "black-forest-labs/FLUX.2-klein-9B"' in mlx_block
    assert re.search(r'"quantize"\s*:\s*8', mlx_block)


def test_flux2_klein_manifest_entries_present():
    # Both flux2_klein_9b and flux2_klein_9b_kv must be present in the
    # builtin manifest with the expected adapter + family + mlx block.
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    # Both ids expose the same capability set: -kv is no longer gated to
    # character_image only — it runs plain txt2img on par with the base 9B,
    # the cache just doesn't engage without a reference (sc-2173).
    for model_id in ("flux2_klein_9b", "flux2_klein_9b_kv"):
        block = find_entry_block(model_id)
        assert '"adapter": "mlx_flux2"' in block, model_id
        assert '"family": "flux2-klein"' in block, model_id
        assert '"macOnly": true' in block, model_id
        # sc-8711 (epic 8506): re-hosted as a public, ungated SceneWorks MLX quant-matrix
        # turnkey (q4/q8/bf16), so the entry is `gated: false` with no credentialHost — the
        # FLUX Non-Commercial LICENSE.md travels with the weights.
        assert '"gated": false' in block, model_id
        mlx_block = find_mlx_block(block)
        quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
        assert quant_match is not None, f"{model_id}: mlx.quantize missing"
        # quantize records the DEFAULT tier (q4); the load Quant is forced to None so the
        # dense bf16 Qwen3 TE is preserved (DENSE_TE_TIER_MODELS).
        assert int(quant_match.group(1)) == 4, f"{model_id}: default tier should be q4 (sc-8711)"
        assert '"text_to_image"' in block, model_id
        assert '"character_image"' in block, model_id


def test_z_image_turbo_manifest_has_mlx_block():
    # sc-2145: z_image_turbo carries an mlx block + sampler/scheduler limits
    # override (mflux's loop is sealed on "linear" — match the wan_2_2 /
    # qwen_image precedents of restricting the menu to default-only when the
    # MLX path is the active backend, epic 1753 §14).
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("z_image_turbo")
    mlx_block = find_mlx_block(block)
    quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
        "z_image_turbo mlx.quantize must be a supported quant level (sc-2145)"
    )
    assert mem_match and int(mem_match.group(1)) > 0, (
        "z_image_turbo mlx.minMemoryGb must be a positive int (sc-2145)"
    )
    # epic 7114 P5 (sc-7126): the native MLX engine adopted the unified curated
    # sampler/scheduler framework, so the mflux linear-only restriction is gone.
    assert '"dpmpp_2m"' in mlx_block and '"uni_pc"' in mlx_block, (
        "z_image_turbo mlx must advertise the curated sampler menu (epic 7114)"
    )
    assert '"sgm_uniform"' in mlx_block, (
        "z_image_turbo mlx must advertise the curated scheduler menu (epic 7114)"
    )


def test_krea_2_turbo_candle_vram_tiers_match_measured_peaks():
    """sc-12126/sc-13108: never regress the directly measured standard-tier peaks."""
    manifest = _load_builtin_models_manifest()
    krea = next(model for model in manifest["models"] if model["id"] == "krea_2_turbo")
    measured_tiers = {
        tier: krea["candle"]["vramGbByTier"][tier] for tier in ("q4", "q8", "bf16")
    }

    assert measured_tiers == {
        "q4": 25.7,
        "q8": 35.2,
        "bf16": 47.2,
    }


def test_boogu_candle_vram_tiers_cover_and_pin_the_default_q8_tier():
    """sc-13533: both Boogu entries must carry a MEASURED q8 row — the tier they default to.

    `mlx.quantize: 8` makes the image-lane resolvers derive q8 for a no-pick request, and the shipped
    `base/`/`turbo/` Q8 turnkey is the ONLY variant `downloads` pulls. The candle blocks originally
    shipped only {q4, bf16} (sc-13108 measured only those two), so
    `vram_gate::predicted_peak_gb(entry, "q8")` found no row and fell through to the flat `minMemoryGb`
    floor, sizing the default tier ~2 GB UNDER its real peak and without the fit gate's 2 GB headroom —
    the permissive direction. The q8 rows below are the direct CUDA measurements (RTX PRO 6000
    Blackwell, exclusive GPU, 1024², seed 42, native path) that close it. Never regress them, and never
    drop the q8 key. Pairs with the Rust coverage lint
    `every_image_model_budgets_its_default_tier_against_a_measured_row`.
    """
    manifest = _load_builtin_models_manifest()
    expected = {
        "boogu_image": {"q4": 31.7, "q8": 42.0, "bf16": 54.4},
        "boogu_image_turbo": {"q4": 31.6, "q8": 42.1, "bf16": 54.5},
    }
    for model_id, tiers in expected.items():
        entry = next(model for model in manifest["models"] if model["id"] == model_id)
        assert {
            tier: entry["candle"]["vramGbByTier"][tier] for tier in ("q4", "q8", "bf16")
        } == tiers, model_id
        # The coarse `minMemoryGb` floor must not sit BELOW the DEFAULT (q8) tier's measured peak —
        # that under-floor (turbo shipped 40 < 42.1) was the second face of this bug, exposed whenever
        # `predicted_peak_gb` falls back to `minMemoryGb`.
        assert entry["candle"]["minMemoryGb"] >= tiers["q8"], model_id


def test_wan_2_2_candle_vram_tiers_match_measured_peaks():
    """sc-13175: never regress the measured 5B SEQUENTIAL peaks (or slide back to the resident ones).

    Re-dropped onto the sequential-offload path (sc-12757 flushes the UMT5 TE + z48 VAE off-GPU around
    the dense denoise), so these SUPERSEDE the resident numbers sc-12631 shipped (q4 46.1 / q8 48.7 /
    bf16 54.0, minMemoryGb 48). Measured on an idle RTX PRO 6000 at wan_2_2's own shipped default
    (832x480, 121 frames, 20 steps, CFG on, CANDLE_GEN_OFFLOAD=sequential), each tier in its own process.
    The peak is the tier-blind denoise attention transient, not the weights -- so q4 and q8 land on the
    SAME pool high-water and only the dense bf16 DiT is heavier; the z48 vae22 decode is the lower phase,
    which makes these card-independent. The numbers are the nvidia-smi POOL high-water (the real max
    device footprint, since cudarc never frees the pool), NOT the lower USED_MEM_HIGH concurrent-live
    (10.61/10.61/11.67 GiB) -- gating at the pool bound is the conservative answer to the sc-13174
    pool-vs-USED_MEM_HIGH caveat, so all three ship `measured: true` with no small-card packdown assumption.
    """
    manifest = _load_builtin_models_manifest()
    wan = next(model for model in manifest["models"] if model["id"] == "wan_2_2")
    candle = wan["candle"]

    assert candle["measured"] is True
    assert {tier: candle["vramGbByTier"][tier] for tier in ("q4", "q8", "bf16")} == {
        "q4": 12.1,
        "q8": 12.1,
        "bf16": 14.5,
    }
    # minMemoryGb gates the default/lightest (q4) tier + the fit gate's 2 GB headroom (12.1 + ~2).
    assert candle["minMemoryGb"] == 14
    # The re-drop's whole point: the heaviest tier's peak + the gate's 2 GB headroom still clears a 24 GB
    # card (the resident 46.1 needed ~48). If this regresses, the 5B silently walls off the card it targets.
    assert candle["vramGbByTier"]["bf16"] + 2 < 24


def test_wan_a14b_candle_all_tiers_measured_q8_admits_32gb():
    """sc-13174 (completing sc-12631): the A14B q4/q8/bf16 candle peaks are ALL MEASURED, and q8 now
    admits a 32 GB card.

    After the sequential-offload / expert-swap / bf16-TE / free-aware-tiling / finer-sdpa rework (epic
    sc-12732), the A14B renders one 14B expert at a time. Its measured `USED_MEM_HIGH` peaks at the
    1280x720/81f/4-step Lightning default are ~22 (q4) / ~28 (q8) / ~39 (bf16) GiB -- not the ~386 GiB
    OOM-floor these blocks used to carry. sc-12631 shipped q4 measured but DEFERRED q8/bf16; sc-13174
    completes them:
      * q8's live peak is ~28 GiB, but its nvidia-smi pool high-water (~34-36, which cudarc never frees)
        left it unproven whether a <=32 GB card packs down to the live peak. A GPU-memory-balloon
        emulation (64 GiB balloon -> ~31 GiB free) reproduced the SAME ~28 live peak at full GPU util with
        no spill, so q8 is gated at its live peak and now ADMITS a 32 GB RTX 5090 -- the epic goal.
      * bf16 was staged (dense fp32 diffusers, after downloading the missing transformer_2 shards) and
        measured at ~39 GiB (one bf16 expert + activations), REPLACING the old conservative derived 56
        bound: the real number admits a 48 GB card but stays refused on 32.
    Pinning the exact values (not just measured:true) mutation-checks the flip -- ripping a tier out or
    regressing q8 back to its pool bound goes RED here. This is the inverse of the sc-12631
    `..._q4_measured_admits_32gb_q8_bf16_deferred` tripwire it replaces.
    """
    manifest = _load_builtin_models_manifest()
    expected = {
        "wan_2_2_t2v_14b": {"q4": 22.13, "q8": 27.95, "bf16": 38.56},
        "wan_2_2_i2v_14b": {"q4": 22.20, "q8": 28.02, "bf16": 38.62},
    }
    for model_id, tiers in expected.items():
        entry = next(m for m in manifest["models"] if m["id"] == model_id)
        candle = entry["candle"]
        # q4/q8/bf16 are all measured now, so the block is honestly measured:true.
        assert candle["measured"] is True, f"{model_id}: q8+bf16 now measured, so measured flips to true"
        assert candle["vramGbByTier"] == tiers, (
            f"{model_id}: the measured q4/q8/bf16 peaks must not regress, got {candle['vramGbByTier']}"
        )
        assert candle["minMemoryGb"] == 24, f"{model_id}: minMemoryGb should gate q4 (~22 + 2)"
        # q4 AND q8 now fit a 32 GB card (each + the fit gate's 2 GB headroom); bf16 does not.
        assert tiers["q4"] + 2 < 32, f"{model_id}: q4 (+headroom) must fit a 32 GB card, got {tiers['q4']}"
        assert tiers["q8"] + 2 < 32, (
            f"{model_id}: q8 (+headroom) now fits a 32 GB card after the <=32 GB balloon validation, "
            f"got {tiers['q8']}"
        )
        # bf16 stays refused on a 32 GB card, but its measured peak now admits a 48 GB card (the derived
        # 56 bound refused 48).
        assert tiers["bf16"] + 2 > 32, f"{model_id}: bf16 must stay refused on a 32 GB card, got {tiers['bf16']}"
        assert tiers["bf16"] + 2 <= 48, (
            f"{model_id}: the measured bf16 peak must now admit a 48 GB card, got {tiers['bf16']}"
        )
        # Heavier tier => heavier peak (ordering sanity).
        assert tiers["q4"] < tiers["q8"] < tiers["bf16"], f"{model_id}: heavier tier => heavier peak"


def test_sdxl_manifest_has_mlx_block():
    # sdxl carries an mlx block (no `limits` override here — the MLX SDXL schedule
    # matches the torch EulerDiscrete default, and there's no per-model sampler menu
    # in the sdxl manifest entry to limit).
    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("sdxl")
    mlx_block = find_mlx_block(block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert mem_match and int(mem_match.group(1)) > 0, (
        "sdxl mlx.minMemoryGb must be a positive int"
    )


# ---------------------------------------------------------------------------
# character_image capability / UI-wiring audits (manifest-parsed dict).
# Extracted from tests/test_worker_image_adapters.py (sc-8861 / F-059).
#
# The three character_image ENGINE-WIRING guards that used to live here
# (test_character_image_capability_implies_engine_or_tuning_declaration /
# test_kolors_declares_strict_pose_controlnet /
# test_models_with_engine_block_advertise_character_image) cross-referenced the
# retired Python worker's MODEL_TARGETS table, so they were reimplemented against
# the Rust worker's own character-image engine wiring in
# crates/sceneworks-worker/src/engines.rs (sc-9513). The manifest-only symmetry
# guard below has no worker dependency and stays here.
# ---------------------------------------------------------------------------


def test_hide_reference_strength_models_declare_a_variation_knob():
    """Symmetry guard for the sc-2017 picker UX. A model that opts out of the
    IP-Adapter reference-strength slider via `ui.hideReferenceStrength` MUST
    also declare `ui.variationStrength` — otherwise the picker shows no tuning
    control at all, and the worker silently runs at default true_cfg_scale.
    """
    manifest = _load_builtin_models_manifest()
    unbalanced: list[str] = []
    for model in manifest.get("models", []):
        ui = model.get("ui") or {}
        if not ui.get("hideReferenceStrength"):
            continue
        if not ui.get("variationStrength"):
            unbalanced.append(model["id"])
    assert not unbalanced, (
        f"Models hide the Reference-strength slider without declaring "
        f"`ui.variationStrength`: {unbalanced}. The picker would leave the user "
        f"with NO identity tuning control. Add `variationStrength` or drop "
        f"`hideReferenceStrength`."
    )
