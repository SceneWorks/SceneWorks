import copy
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import uuid
from collections import Counter

import jsonschema
import pytest
from referencing import Registry, Resource


ROOT = pathlib.Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "generated" / "memory-matrix.json"
SCHEMA = ROOT / "packages" / "schemas" / "memory-matrix.schema.json"
CALIBRATION_SCHEMA = ROOT / "packages" / "schemas" / "memory-calibration.schema.json"

# The full strategy ladder (StrategyRung::ALL in sceneworks-core). Used for coverage-shape
# assertions — "one cell per rung of the ladder" — in place of pinned population counts.
STRATEGY_RUNGS = (
    "resident",
    "staged_residency",
    "bounded_decode",
    "bounded_attention",
    "bounded_transformer_residency",
)


def load_matrix():
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def matrix_validator():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    calibration_schema = json.loads(CALIBRATION_SCHEMA.read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        calibration_schema["$id"], Resource.from_contents(calibration_schema)
    )
    return jsonschema.Draft202012Validator(schema, registry=registry)


def test_generated_memory_matrix_is_schema_valid():
    matrix_validator().validate(load_matrix())


def test_matrix_accounts_for_all_models_and_pinned_mlx_staged_coverage():
    matrix = load_matrix()
    # sc-18815: the universe is modality-aware, so the census is too. `imageModels` was REMOVED
    # rather than left holding the whole-universe total under a one-modality name — a field called
    # `imageModels` reading 63 is worse than the image-only universe it replaced.
    assert "imageModels" not in matrix["summary"]
    assert matrix["summary"]["catalogEntries"] == 63
    assert matrix["summary"]["catalogEntriesByModality"] == {"image": 53, "video": 10}
    image_ids = {model["id"] for model in matrix["models"] if model["modality"] == "image"}
    video_ids = {model["id"] for model in matrix["models"] if model["modality"] == "video"}
    assert len(image_ids) == 53
    assert len(video_ids) == 10
    # SC-18218 closes FLUX.2-dev to its measured Resident-only provider contract, so its former
    # generic staged-route claim is intentionally absent from this census.
    # sc-18815 keeps this as exactly the IMAGE-lane claim its denominator says it is. The separate
    # video census consumes the provider-owned staged-residency contracts at the frozen b4 pin;
    # only SVD remains without an MLX staged implementation. The image-lane numerator is asserted
    # structurally against the census below rather than as a pinned population (the SC-18218
    # shape-over-population ruling); the denominators are the modality totals asserted above.
    assert matrix["summary"]["mlxStagedStaticCoverageDenominator"] == 53
    assert matrix["summary"]["videoMlxStagedStaticCoverage"] == 9
    assert matrix["summary"]["videoMlxStagedStaticCoverageDenominator"] == 10
    assert len(matrix["models"]) == len(matrix["modelSlices"]) == 63
    assert {model["id"] for model in matrix["models"]} == set(matrix["modelSlices"])
    # SC-18218 closes FLUX.2-dev to its measured Resident-only provider contract, so its former
    # generic staged-route claim is intentionally absent from this census — the coverage number is
    # a strict subset of the denominator, and the denominator is the whole census. The number
    # itself is recomputed from the coverage rows below rather than pinned here.
    assert (
        0
        < matrix["summary"]["mlxStagedStaticCoverage"]
        < matrix["summary"]["mlxStagedStaticCoverageDenominator"]
    )

    # SC-18826 closes the only wholly-unrouted defect: VACE-Fun already had a manifest entry and real
    # native engine, and now has the missing MLX-only VideoModelCaps row. It may still have no
    # published cells while every coordinate is unplanned Missing, but the model-level axes and lane
    # must be present — publication elision is not routing.
    assert matrix["summary"]["unroutedEntries"] == []
    vace = next(model for model in matrix["models"] if model["id"] == "wan_2_2_vace_fun_14b")
    assert vace["backends"] == ["mlx"]
    assert vace["resolvedRoutes"] == {"mlx": "wan2_2_vace_fun_14b"}
    assert vace["axes"]
    assert matrix["modelSlices"]["wan_2_2_vace_fun_14b"] == []
    assert all(model["backends"] for model in matrix["models"])

    # Video providers are genuinely per-backend, and a scalar route cannot express that: getting it
    # wrong binds a cell's calibration evidence, plan row and closure digest to a provider that never
    # ran it. `mlx:ltx_2_3` is the exact key sc-18808 already committed to the closure table.
    ltx = next(model for model in matrix["models"] if model["id"] == "ltx_2_3")
    assert ltx["resolvedRoutes"] == {"mlx": "ltx_2_3", "candle": "ltx_2_3_distilled"}
    assert {
        (cell["backend"], cell["provider"])
        for cell in matrix["cells"]
        if cell["modelId"] == "ltx_2_3"
    } <= {("mlx", "ltx_2_3"), ("candle", "ltx_2_3_distilled")}

    # Video entries name NO per-entry ownership story, because epic 18803 filed none. An integer
    # there would name a story that cannot close the cell — SC-15812's defect from the other side.
    for model in matrix["models"]:
        for backend in model["backends"]:
            assert isinstance(model["owningFamilyStories"][backend], int), model["id"]
            assert isinstance(model["owningModelStories"][backend], int) == (
                model["modality"] == "image"
            ), model["id"]

    # sc-18099: `cells` is a SUBSET, and the artifact has to say so in its own numbers. `summary`
    # keeps the RESOLVED coordinate total, which is no longer `len(cells)`, and partitions it —
    # a slim that quietly capped coverage instead of counting what it dropped is the failure this
    # pins. Nothing here is a hardcoded population: the relations hold at any catalog size.
    assert matrix["summary"]["publishedCells"] == len(matrix["cells"])
    assert (
        matrix["summary"]["publishedCells"] + matrix["summary"]["elidedCells"]
        == matrix["summary"]["cells"]
    )
    assert matrix["summary"]["elidedCells"] > 0
    assert sum(matrix["summary"]["elidedByState"].values()) == matrix["summary"]["elidedCells"]
    assert matrix["summary"]["publicationPredicate"]

    # The census covers every resolved coordinate, so a coverage claim never reads off the sample.
    assert sum(row["coordinates"] for row in matrix["coverage"]) == matrix["summary"]["cells"]
    assert sum(row["published"] for row in matrix["coverage"]) == len(matrix["cells"])
    assert sum(row["elided"] for row in matrix["coverage"]) == matrix["summary"]["elidedCells"]
    # `mlxStagedStaticCoverage` is a claim about all 53 image entries. Recomputed from the census
    # rather than from `cells`, which would silently shrink it to whatever the slim happened to
    # publish — and scoped to the image lane, which is what its denominator claims to cover
    # (sc-18815). The video lane's numerator is recomputed the same way against its own summary
    # field.
    mlx_staged = {
        row["modelId"]
        for row in matrix["coverage"]
        if row["backend"] == "mlx"
        and row["rung"] == "staged_residency"
        and row["implemented"]
        and row["modelId"] in image_ids
    }
    assert len(mlx_staged) == matrix["summary"]["mlxStagedStaticCoverage"]
    assert (
        len(
            {
                row["modelId"]
                for row in matrix["coverage"]
                if row["backend"] == "mlx"
                and row["rung"] == "staged_residency"
                and row["implemented"]
                and row["modelId"] in video_ids
            }
        )
        == matrix["summary"]["videoMlxStagedStaticCoverage"]
    )
    # SC-18218 closes FLUX.2-dev to its measured Resident-only provider contract, and the current
    # inference pin deliberately omits sequential offload from Bernini's descriptor. Neither lane may
    # contribute a generic staged-route claim to this census.
    #
    # Stated as structure, not as a population. The count moved 37 -> 38 when inference sc-18609 made
    # bernini_image's DECLARED MLX rung-4 ladder actually reachable, and renewing the constant would
    # only have re-frozen the corpus at the next catalog change.
    #
    # The replacement is DELIBERATELY WEAKER than the count, and the honest scope is worth stating so
    # nobody reads more into it. Guarded: the two lanes below by name, bespoke routes never claiming the
    # generic ladder, per-route drift where entries sharing a resolved route disagree with each other,
    # and the census being neither empty nor the whole catalog. NOT guarded: uniform drift on a route
    # nothing else shares — 35 of the 41 resolved routes are singletons, so a singleton lane silently
    # dropping out of the census, or silently claiming staged coverage it has not implemented, passes
    # everything here where the old count reddened. A whole shared family drifting uniformly passes for
    # the same reason.
    #
    # That is the accepted shape-over-population tradeoff rather than an oversight. The count did catch
    # those cases, and charged a hand-edit for every unrelated catalog or reachability change to do it;
    # runtime catching is the chosen tradeoff for the residue. The generator carries the same note beside
    # `assertMlxStagedCoverageIsStructurallyConsistent`, which is where the mirror of these assertions
    # runs against the pre-publication document.
    assert "flux2_dev" in mlx_staged, (
        "sc-20799 retired the SC-18218 resident-only pin: at inference ebcdc7da7 all three Dev MLX "
        "providers declare selectable Sequential staged residency, so flux2_dev is census-required"
    )
    assert "bernini_image" in mlx_staged, (
        "inference sc-18609 made bernini_image's declared MLX rung-4 ladder reachable"
    )
    assert 0 < len(mlx_staged) < len(image_ids), (
        "staged coverage is partial by construction; a total census would mean the exclusions vanished"
    )
    # Staged coverage is a property of the RESOLVED ROUTE, so every entry sharing a route agrees.
    # An entry drifting away from its own route siblings is exactly what a bumped count cannot see.
    #
    # EXEMPT, 2026-08-17: entries whose MLX tier axis is a single synthetic `default` — no advertised
    # tier ladder at all. Kept in lockstep with `assertMlxStagedCoverageIsStructurallyConsistent` in
    # scripts/generate-memory-matrix.mjs, which carries the full reasoning and the coordinator decision;
    # this is that assertion's mirror against the PUBLISHED document, so the two must scope alike or the
    # published artifact and the pre-publication census would disagree about the same catalog.
    #
    # `flux2_klein_9b_true_v2` is the instance: a convert-at-install entry whose transformer is a fixed
    # dense BF16 artifact (`mlx.quantize: 0`, the only truthful encoding under `resolve_quant`), sharing
    # route `flux2_klein_9b` with two tiered siblings. Its verdict is structurally fixed at "not staged"
    # because the tiers its contract declares do not exist for it, so including it reports a
    # disagreement no declaration change can resolve.
    #
    # Narrow on purpose: it removes these entries from the CROSS-ENTRY comparison only, and the
    # comparison keeps full force among tiered entries sharing a route. The JS side carries the mutation
    # control proving a drifting tiered route-mate still reds, and that an exempt entry on the route does
    # not suppress drift between its tiered siblings.
    def has_single_dense_tier_axis(model: dict) -> bool:
        return model.get("axes", {}).get("mlx", {}).get("tiers") == ["default"]

    route_verdicts: dict[str, set[bool]] = {}
    for model in matrix["models"]:
        if model["modality"] != "image" or has_single_dense_tier_axis(model):
            continue
        route_verdicts.setdefault(model["resolvedRoute"], set()).add(
            model["id"] in mlx_staged
        )
    assert all(len(verdicts) == 1 for verdicts in route_verdicts.values()), sorted(
        route for route, verdicts in route_verdicts.items() if len(verdicts) > 1
    )
    # The exemption must stay narrow: it may never empty the comparison. If every entry became
    # single-dense the loop above would pass vacuously, so assert it still has routes to compare.
    assert route_verdicts, "the per-route comparison must not be emptied by the exemption"
    # A bespoke route carries its own pipeline and never advertises the GENERIC staged ladder.
    #
    # "Generic" is enforced as written, 2026-08-17, in lockstep with the generator-side assertion (which
    # carries the reasoning). `routeKind: "bespoke"` means only "no row in engines.rs's MODEL_TABLE" — a
    # worker dispatch fact — not "the engine registers no contract". PuLID is that split: bespoke in
    # dispatch, unregistered on candle, but the MLX registry publishes a real `pulid_flux` contract at
    # pin 931366f62. This document agrees: its staged coverage row reads
    # `implementedBy.overlay = {identity: 3, lora: 0, none: 0}` — its own closed identity contract and
    # nothing generic.
    #
    # So the claim to reject is implemented coverage on a generic overlay. Teeth intact: if a bespoke
    # route ever implements `none` or `lora` staged coverage, this reds.
    generic_overlays = ("none", "lora")
    bespoke_ids = {
        model["id"] for model in matrix["models"] if model["routeKind"] == "bespoke"
    }
    assert not [
        (row["modelId"], overlay)
        for row in matrix["coverage"]
        if row["backend"] == "mlx"
        and row["rung"] == "staged_residency"
        and row["modelId"] in bespoke_ids
        for overlay in generic_overlays
        if row.get("implementedBy", {}).get("overlay", {}).get(overlay, 0)
    ]
    # Every entry keeps a slice key even when it publishes nothing, and no slice may name a cell the
    # slim dropped.
    published_ids = {cell["id"] for cell in matrix["cells"]}
    assert any(not slice_ for slice_ in matrix["modelSlices"].values()), (
        "some entry must publish nothing, or the empty-slice case is untested"
    )
    for model_id, slice_ in matrix["modelSlices"].items():
        assert set(slice_) <= published_ids, model_id
    assert {run["cellId"] for run in matrix["calibrationRuns"]} <= published_ids


def test_aliases_bespoke_routes_and_evidence_dimensions_are_explicit():
    matrix = load_matrix()
    models = {model["id"]: model for model in matrix["models"]}
    assert models["z_image_edit"]["resolvedRoute"] == "z_image_turbo"
    assert models["instantid_realvisxl"]["routeKind"] == "bespoke"
    assert models["pulid_flux_dev"]["routeKind"] == "bespoke"
    # sc-18099: read the published AXES, not `cells`. The matrix now publishes only planned-or-
    # evidenced coordinates, and InstantID's Candle lane publishes none — reading its tiers off
    # `cells` returns the empty set and this assertion would have to be deleted rather than moved.
    # `models[].axes` is the resolved cross-product, reconciled against it at generation time, and is
    # what keeps an unmeasured lane visible instead of indistinguishable from an absent one.
    models_by_id = {model["id"]: model for model in matrix["models"]}
    assert models_by_id["instantid_realvisxl"]["axes"]["candle"]["tiers"] == ["bf16"]
    assert matrix["evidenceDimensions"] == [
        "staticImplementation",
        "declaredCalibration",
        "historicalVerification",
        "currentEnvironmentVerification",
        "loadability",
        "strategyParameterVerification",
    ]


def test_calibration_evidence_is_schema_valid_and_matrix_ingested():
    calibration_schema = json.loads(CALIBRATION_SCHEMA.read_text(encoding="utf-8"))
    calibration = json.loads(
        (ROOT / "docs/generated/memory-calibration-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(
        calibration_schema, format_checker=jsonschema.FormatChecker()
    ).validate(calibration)
    matrix = load_matrix()
    evidence_ids = {record["id"] for record in calibration["records"]}
    # Well-formedness stands in for the retired id pins: every record id is an immutable
    # `imc-`-prefixed 20-hex-digit capture id, so a fabricated or truncated id cannot ride into
    # any of the derived sets below.
    assert all(re.fullmatch(r"imc-[0-9a-f]{20}", record_id) for record_id in evidence_ids)
    evidence_by_id = {record["id"]: record for record in calibration["records"]}
    records_by_status = Counter(record["status"] for record in calibration["records"])
    # Population SIZES are never pinned here: they have only ever grown (complete 33 -> 50 -> 52 -> 65
    # -> 70 across sc-16915 / SC-18237 / SC-18353 / SC-19753; runtime_complete 15 -> 19 at SC-18218)
    # and renewing the pair each time re-froze the corpus without ever asserting a property.
    # Everything downstream is derived from `records_by_status` rather than from a second
    # transcription of it, so the matrix and the bundle cannot disagree.
    #
    # sc-21715: this used to read `set(records_by_status) == {"complete", "runtime_complete"}`,
    # because a third status DID slip past every partition below — `calibrationRunsByStatus` named
    # only those two while `summary.calibrationRuns` counted the whole bundle. The tally now
    # partitions the bundle over the schema's full status enum, so the corpus no longer has to be
    # held two-status to keep the derived counts honest. What must still hold is that no record
    # carries a status the schema does not admit, and that both certifying populations are non-empty.
    record_statuses = calibration_schema["$defs"]["record"]["properties"]["status"]["enum"]
    assert set(records_by_status) <= set(record_statuses)
    assert records_by_status["complete"] > 0
    assert records_by_status["runtime_complete"] > 0
    assert len(evidence_ids) == len(calibration["records"]) == sum(
        records_by_status.values()
    )
    assert {run["record"]["id"] for run in matrix["calibrationRuns"]} == evidence_ids
    runs_by_status = Counter(run["record"]["status"] for run in matrix["calibrationRuns"])
    assert runs_by_status == records_by_status
    assert matrix["summary"]["calibrationRuns"] == len(calibration["records"])
    # A key per admitted status, zeros included, summing to the total above — derived from the enum
    # so a status added to the schema reds here instead of quietly falling outside the tally.
    assert matrix["summary"]["calibrationRunsByStatus"] == {
        re.sub(r"_([a-z])", lambda m: m.group(1).upper(), status): records_by_status[status]
        for status in record_statuses
    }
    assert sum(matrix["summary"]["calibrationRunsByStatus"].values()) == matrix["summary"][
        "calibrationRuns"
    ]

    # `current` vs `historical` is decided against the shipped inference pin plus exact audited
    # compatibility. SC-15833 certifies the Candle FLUX.2 closure across an exact window -- captured
    # at 5ffd, compatible through `audited_live_revision` -- so only its five records may authorize
    # the later runtime, and only while that revision is still the live pin. The audit is a WINDOW,
    # not a permanent grant: once the pin moves past it those five join every other retained record
    # as historical, which is the fail-closed rule SC-15833's own "refuses to authorize capture
    # promotion against a newer live inference pin" test asserts directly.
    #
    # Derived from the live pin rather than hardcoded, so a bump does not make this test wrong; it
    # degrades to `{"historical"}` on its own and comes back when the window is re-audited. sc-17524
    # did exactly that for a4f409ae -- `Cargo.lock` and `candle-gen` both moved, and the
    # compiled measurement binary was byte-identical at both ends. Only a measurement re-opens this
    # window; editing the constant below does not.
    audited_live_revision = "a4f409ae8ce73eda2ee8117b89b5f479666606b8"
    worker_manifest = (
        ROOT / "crates" / "sceneworks-worker" / "Cargo.toml"
    ).read_text(encoding="utf-8")
    live_pin_match = re.search(
        r'github\.com/SceneWorks/inference"[^}\n]*\brev\s*=\s*"([0-9a-f]{40})"',
        worker_manifest,
    )
    assert live_pin_match, "could not read the pinned inference revision from the worker manifest"
    within_audited_window = live_pin_match.group(1) == audited_live_revision
    expected_flux2_semantics = {"current"} if within_audited_window else {"historical"}
    full_runs = [
        run for run in matrix["calibrationRuns"] if run["record"]["status"] == "complete"
    ]
    runtime_complete_runs = [
        run
        for run in matrix["calibrationRuns"]
        if run["record"]["status"] == "runtime_complete"
    ]
    # Currency is the provider's COMPILE CLOSURE, not the pin (sc-17774): a record is current
    # exactly when the digest it captured is still the live digest for ITS provider lane. Derived
    # here, once, the same way the generator derives it, and legitimately EMPTY in the window
    # between a pin bump and its re-capture — which is where the sc-17137 main sync leaves the
    # corpus (both the SC-18218 captures and sc-19721's 75d66db5 re-captures predate the sc-20523
    # pin's closures).
    live_closures = json.loads(
        (ROOT / "config/inference-provider-closures.json").read_text(encoding="utf-8")
    )["providers"]
    current_by_closure = {
        record["id"]
        for record in calibration["records"]
        if live_closures.get(
            f"{record['backend']}:{record['target']['provider']}", {}
        ).get("digest")
        == record["repositories"]["inference"]["closureDigest"]
    }
    # Currency is DERIVED, never pinned. A record is current exactly when the closure digest it
    # captured is still the live digest for its provider lane, so the invariant worth asserting is
    # that the matrix's `semantics` agrees with the ledger — for every record, in both directions.
    #
    # This used to pin the identity of the single current Full-complete record (SC-21714's Candle
    # Krea capture). That assertion could not fail on a bug and was guaranteed to fail on the next
    # closure change: it red on a `gen-core` pin bump that memoized a SHA-256 digest, which cannot
    # move any model's memory footprint and therefore cannot invalidate a memory measurement. That
    # is a gate on measurement wearing a test's clothes — the frozen-corpus class, rewritten to
    # shape here rather than hand-updated to the next id (which would only re-freeze it).
    #
    # An empty `current_full_runs` is legitimate and deliberately allowed: it is exactly the window
    # between a pin bump and its re-capture, which the currency derivation above already documents.
    current_full_runs = [run for run in full_runs if run["semantics"] == "current"]
    assert {run["semantics"] for run in full_runs} <= {"current", "historical"}
    assert {run["record"]["id"] for run in current_full_runs} == {
        run["record"]["id"] for run in full_runs if run["record"]["id"] in current_by_closure
    }
    assert all(
        (run["semantics"] == "current") == (run["record"]["id"] in current_by_closure)
        for run in full_runs
    )
    # Per-record shape, over whichever Full-complete captures the corpus holds: every one names a
    # real provider lane, a real strategy rung, and a geometry the matrix can join on. Asserted for
    # all of them rather than for one pinned id, so a re-capture extends the coverage instead of
    # breaking the test.
    for run in full_runs:
        record = evidence_by_id[run["record"]["id"]]
        assert f"{record['backend']}:{record['target']['provider']}" in live_closures
        assert record["target"]["modelId"]
        assert record["target"]["tier"]
        assert record["strategy"]["rung"]
        assert record["target"]["geometry"]["width"] > 0
    expected_candle_flux2_runtime = {
        "imc-998b89c5d76dbcc84332": "bounded_attention",
        "imc-b4113eedf503e409ad1b": "resident",
        "imc-b62adbfca64f277414e1": "bounded_decode",
        "imc-bfb890dff959eaf09183": "staged_residency",
        "imc-f5c3d06f30ebf3723f13": "bounded_transformer_residency",
    }
    # The four MLX FLUX.2 production coordinates. A pin bump stales the current rows and the
    # re-capture adds one new record per pair while the historical rows stay, so the POPULATION
    # grows at every bump — which is why the rows are held to this SHAPE rather than pinned by id:
    # exactly one CURRENT row per pair (current = captured at the live closure), and retained
    # historical rows covering the same pairs. A ragged shape still means a partial or duplicated
    # ingest.
    expected_mlx_flux2_pairs = {("q8", 768), ("q8", 1024), ("q4", 768), ("q4", 1024)}
    candle_flux2_runtime = [
        run
        for run in runtime_complete_runs
        if run["record"]["target"]["modelId"] == "flux2_dev"
        and run["record"]["backend"] == "candle"
    ]
    assert {
        run["record"]["id"]: run["record"]["strategy"]["rung"]
        for run in candle_flux2_runtime
    } == expected_candle_flux2_runtime
    assert {run["semantics"] for run in candle_flux2_runtime} == expected_flux2_semantics
    assert {
        run["record"]["repositories"]["inference"]["revision"]
        for run in candle_flux2_runtime
    } == {"5ffd7612e7de4e76b6db00a7148ed3d9c15b4c0d"}

    mlx_flux2_runtime = [
        run
        for run in runtime_complete_runs
        if run["record"]["target"]["modelId"] == "flux2_dev"
        and run["record"]["backend"] == "mlx"
    ]
    assert mlx_flux2_runtime
    # Current is DERIVED from the closure, and the split is exhaustive: exactly one current row
    # per production pair, and the retained history also spans every pair — a demotion on one
    # geometry cannot hide behind a promotion on another, and a partial re-capture (three of the
    # four pairs) still fails.
    mlx_flux2_by_semantics = {"current": [], "historical": []}
    for run in mlx_flux2_runtime:
        mlx_flux2_by_semantics[run["semantics"]].append(run)
    assert {
        run["record"]["id"] for run in mlx_flux2_by_semantics["current"]
    } == {
        run["record"]["id"]
        for run in mlx_flux2_runtime
        if run["record"]["id"] in current_by_closure
    }
    current_pairs = {
        (run["record"]["target"]["tier"], run["record"]["target"]["geometry"]["width"])
        for run in mlx_flux2_by_semantics["current"]
    }
    # When any rows ARE current they must be exactly one per production pair; between a pin bump
    # and its re-capture the current half is legitimately empty and the retained history still has
    # to span every pair.
    if mlx_flux2_by_semantics["current"]:
        assert current_pairs == expected_mlx_flux2_pairs
        assert len(mlx_flux2_by_semantics["current"]) == len(expected_mlx_flux2_pairs)
    assert {
        (run["record"]["target"]["tier"], run["record"]["target"]["geometry"]["width"])
        for run in mlx_flux2_by_semantics["historical"]
    } == expected_mlx_flux2_pairs
    # Only the current rows carry admission; the historical rows must not.
    assert all(run["binding"]["eligible"] for run in mlx_flux2_by_semantics["current"])
    # Two retained cohorts, and only these two: SC-18218's originals at 10831e4ca and sc-19721's
    # re-captures at the epic's 75d66db5 pin (kept by the sc-17137 main sync; the sc-20523 pin
    # moved the closure past both, so both read historical until the bump-time re-capture).
    expected_mlx_flux2_cohorts = {
        "10831e4ca5b8bf780319a8ee7f21427175075448": (
            "355749219c38b37af5054df047b0f44b65ecd8f822fc258243eee9d09c1d0247"
        ),
        "75d66db50543ac288deb278853d0f0b432f92c5c": (
            "6f6cebb6ba86dd630bc0d3fea9da8960ea0f4acf3bbeba28cb4d3665d3b0d1c6"
        ),
    }
    for run in mlx_flux2_runtime:
        record = evidence_by_id[run["record"]["id"]]
        revision = record["repositories"]["inference"]["revision"]
        assert revision in expected_mlx_flux2_cohorts
        assert record["repositories"]["inference"]["closureDigest"] == (
            expected_mlx_flux2_cohorts[revision]
        )
        if run["semantics"] == "historical":
            assert record["repositories"]["inference"]["closureDigest"] != live_closures[
                "mlx:flux2_dev"
            ]["digest"], "a retained FLUX.2 capture must not be re-dated to the live closure"
        assert record["status"] == "runtime_complete"
        assert record["loadShape"] == "eager_materialization"
        assert record["calibrationFingerprint"] == (
            "sc-18218-flux2-dev-t2i-resident-evidence-v1"
        )
        target = record["target"]
        assert target["provider"] == "flux2_dev" and target["modelId"] == "flux2_dev"
        assert target["mode"] == "text_to_image" and target["overlay"] == "none"
        geometry = target["geometry"]
        assert (target["tier"], geometry["width"]) in expected_mlx_flux2_pairs
        assert geometry["height"] == geometry["width"]
        assert geometry["batch"] == 1 and geometry["frames"] == 1
        assert record["strategy"] == {
            "rung": "resident",
            "parameters": {},
            "engagedRungs": ["resident"],
        }
        assert record["artifact"] == {
            "repository": "SceneWorks/flux2-dev-mlx",
            "resolvedRevision": "2868b1461b2b6e6e05d84e52534df3632b4c7d5d",
            "variant": target["tier"],
        }
        assert record["quality"]["result"] == "passed"
        assert record["loadability"]["result"] == "passed"
        assert record["observedMemory"]["overall"]["allocatorBytes"] <= record[
            "predictedPeakBytes"
        ]["overall"]
        scenarios = {scenario["name"]: scenario for scenario in record["scenarios"]}
        assert {
            scenarios[name]["result"]
            for name in ("exact_fit", "unknown_budget", "stale_evidence", "loadability")
        } == {"passed"}
        assert {scenarios[name]["result"] for name in ("warm_repeat", "cancel", "error")} == {
            "not_run"
        }
        assert scenarios["overlay"]["result"] == "not_applicable"

    flux2_runtime = candle_flux2_runtime + mlx_flux2_runtime

    historical_flux1_runtime = [
        run
        for run in runtime_complete_runs
        if run["record"]["target"]["modelId"] in {"flux_schnell", "flux_dev"}
    ]
    assert Counter(
        (
            run["record"]["target"]["modelId"],
            run["record"]["strategy"]["rung"],
        )
        for run in historical_flux1_runtime
    ) == Counter(
        (model_id, rung)
        for model_id in ("flux_schnell", "flux_dev")
        for rung in (
            "resident",
            "staged_residency",
            "bounded_decode",
            "bounded_attention",
            "bounded_transformer_residency",
        )
    )
    assert {run["semantics"] for run in historical_flux1_runtime} == {"historical"}
    assert len(runtime_complete_runs) == len(flux2_runtime) + len(
        historical_flux1_runtime
    )
    # SC-18306 moves the pin beyond the MLX FLUX.2 capture closure as well as the older FLUX.1
    # captures. The Candle FLUX.2 window independently becomes current only at its audited pin.
    assert {run["semantics"] for run in runtime_complete_runs} == {"historical"}
    assert all(run["binding"]["eligible"] for run in runtime_complete_runs)
    current_eligible = [
        run
        for run in matrix["calibrationRuns"]
        if run["semantics"] == "current" and run["binding"]["eligible"]
    ]
    # Independent sources of "current", kept separate so one lane cannot mask another:
    #
    #   - SC-18353 records measured at the live pin;
    #   - records whose captured provider closure still matches the live provider closure
    #     (SC-18237's two Qwen q8 rows);
    #   - the audited FLUX.2 window, current only while its audited revision IS the live pin.
    measured_at_live_pin = {
        record["id"]
        for record in calibration["records"]
        if record["repositories"]["inference"]["revision"] == live_pin_match.group(1)
    }
    # This set is derived from the PIN, which sc-17774 retired as the currency term in favour of
    # the provider's compile closure. The two coincided while nothing had moved; they no longer
    # do, but one direction survives every bump: a record captured at the live pin was compiled
    # from the live closure, so live-pin evidence is always a SUBSET of closure-current evidence.
    # (Not pinned by id: the live-pin set turns over wholesale at every re-capture — and it is
    # legitimately EMPTY in the window between a pin bump and its re-capture, so its emptiness is
    # not asserted either; every id in the corpus is already shape-checked above.)
    assert measured_at_live_pin <= current_by_closure
    # Measured at the live pin means CURRENT, without exception — a record may not be measured here
    # and dated elsewhere. Stated as a subset so the implication survives the set being empty: with
    # nothing measured at the live pin there is nothing to classify, and the moment a record does
    # appear there it must be `current` or this fails.
    assert {
        run["semantics"]
        for run in matrix["calibrationRuns"]
        if run["record"]["id"] in measured_at_live_pin
    } <= {"current"}
    # Current is necessary but not sufficient for eligible: a record must also BIND a declared cell.
    # sc-16915 swept seven decode tile edges and the manifest binds only the production point
    # (512/64), so the six off-point edges are current-but-ineligible by design — they widen the
    # published range without certifying a cell of their own.
    unbound_decode_edges = {
        record["id"]
        for record in calibration["records"]
        if live_closures.get(
            f"{record['backend']}:{record['target']['provider']}", {}
        ).get("digest")
        == record["repositories"]["inference"]["closureDigest"]
        and record["strategy"]["rung"] == "bounded_decode"
        and record["sweep"]["cases"][0]["parameters"].get("decodeTileEdge") != 512
    }
    # The unbound edges are CHARACTERIZATION, not bound cells: only the production 512 edge binds
    # a matrix cell, and the sweep exists to show the shape either side of it. Their ids turn over
    # at every re-capture, so the set is held to its SHAPE instead of an id list: every member is
    # a current bounded-decode sweep row whose first case sits OFF the production point, and no
    # two members re-measure the same coordinate — a duplicated or mislabeled ingest still fails.
    # The set is a subset of the current Qwen corpus. SC-21714 makes Candle Krea current without
    # creating Qwen characterization edges, so only a current Qwen cohort requires this sweep.
    current_qwen_ids = {
        record["id"]
        for record in calibration["records"]
        if record["id"] in current_by_closure
        and record["backend"] == "mlx"
        and record["target"]["provider"] == "qwen_image"
    }
    if current_qwen_ids:
        assert unbound_decode_edges
    assert unbound_decode_edges <= current_qwen_ids
    unbound_coordinates = [
        (
            record["backend"],
            record["target"]["provider"],
            record["target"]["tier"],
            record["sweep"]["cases"][0]["parameters"].get("decodeTileEdge"),
        )
        for record in calibration["records"]
        if record["id"] in unbound_decode_edges
    ]
    assert len(unbound_coordinates) == len(set(unbound_coordinates))
    for backend, provider, tier, edge in unbound_coordinates:
        assert isinstance(edge, int) and edge > 0 and edge != 512, (
            backend,
            provider,
            tier,
            edge,
        )
    assert {run["record"]["id"] for run in current_eligible} == (
        current_by_closure - unbound_decode_edges
    ) | (
        set(expected_candle_flux2_runtime) if within_audited_window else set()
    )
    # The four records that WERE runtime-current before the pin moved are still present and still
    # bind cleanly — superseded by revision, not rejected. Anything else would mean the bump damaged
    # the bundle rather than re-dating it.
    superseded_rung4 = [
        run
        for run in matrix["calibrationRuns"]
        if run["record"]["id"]
        in {
            "imc-12f3ccbb72de78cea931",
            "imc-4426a6e84c4d39d9bff3",
            "imc-8f041bead8a9346cd1e6",
            "imc-8f110511b0f85d15f72f",
        }
    ]
    assert len(superseded_rung4) == 4
    assert all(run["binding"]["eligible"] for run in superseded_rung4)
    assert all(
        run["record"]["target"]["modelId"] != "z_image"
        for run in matrix["calibrationRuns"]
    ), "captures without a measured loadShape must remain outside the schema-v4 bundle"
    # The long-standing bf16 captures remain in the bundle. Only resident/staged rows still bind to
    # cells without an independent current static fingerprint; the older bounded fingerprints must
    # not mask the provider contract, even as historical characterization.
    # These four used to bind cleanly, because the shipped opt-in named the same suffixed
    # fingerprint they carry. sc-16915 repointed the opt-in at the identity the provider actually
    # declares — `mlx-gen-qwen-image` collapsed its `-v1-eager`/`-v1-deferred` pair into a single
    # `qwen-image-mlx-shared-ladder-2026-08-01-v1` — so records still carrying the old suffixed
    # spelling no longer describe the shipped provider and bind nothing.
    #
    # They are RETAINED, not deleted: the assertion moved from "eligible with no reasons" to
    # "present, historical, and ineligible for exactly one stated reason". Dropping the block
    # entirely would have hidden the change; asserting the reason keeps it legible and would fail
    # if these rows started binding again or disappeared.
    historical_qwen = [
        run
        for run in matrix["calibrationRuns"]
        if run["semantics"] == "historical"
        and run["record"]["target"]["modelId"] == "qwen_image"
        and run["record"]["target"]["tier"] == "bf16"
        and run["record"]["strategy"]["rung"] in {"resident", "staged_residency"}
    ]
    # These are two distinct populations, and flattening them would lose the distinction that matters.
    # One carries the retired `-eager` fingerprint and still cannot bind. The other carries the shared
    # fingerprint and binds the bf16 cells restored by earlier captures, but remains historical and
    # therefore cannot authorize current verification.
    #
    # Pinned as STRUCTURE, not as populations. The totals moved 6 -> 8 and 2 -> 4 when SC-18306 pushed
    # the closure past the SC-18353 resident/staged captures — currency moved, nothing about the
    # records changed — and the numbers say nothing about why. The properties below do: the two buckets
    # PARTITION the set with no third outcome, the fingerprint decides eligibility in both directions,
    # and the two rungs stay symmetric both overall and bucket by bucket. Each still fails on the
    # original defect (a retired fingerprint starting to bind, or one rung quietly losing a record) at
    # any retained-capture count.
    assert historical_qwen
    rejected = [
        run for run in historical_qwen if run["binding"]["reasons"] == ["fingerprint-mismatch"]
    ]
    retained_bindings = [run for run in historical_qwen if run["binding"]["eligible"]]
    assert rejected and retained_bindings
    assert len(rejected) + len(retained_bindings) == len(historical_qwen), (
        "every historical Qwen bf16 resident/staged run is either fingerprint-rejected or a retained "
        "binding; a third outcome would go uninspected"
    )
    assert all(not run["binding"]["eligible"] for run in rejected)
    assert all(run["binding"]["reasons"] == [] for run in retained_bindings)
    # Eligibility is decided by the fingerprint, and by nothing else, in both directions.
    assert {run["record"]["calibrationFingerprint"] for run in rejected} == {
        "qwen-image-mlx-shared-ladder-2026-08-01-v1-eager",
    }
    assert {
        run["record"]["calibrationFingerprint"] for run in retained_bindings
    } == {"qwen-image-mlx-shared-ladder-2026-08-01-v1"}
    # ...and the two populations are a PARTITION of the whole: a historical row that is neither
    # cleanly bound nor rejected for exactly the stated reason is a new failure mode, not noise.
    assert len(historical_qwen) == len(rejected) + len(retained_bindings)
    assert {
        (
            run["record"]["backend"],
            run["record"]["target"]["tier"],
            run["record"]["target"]["mode"],
            run["record"]["target"]["overlay"],
        )
        for run in historical_qwen
    } == {("mlx", "bf16", "text_to_image", "none")}
    # Symmetry is the property, not the per-rung total: each rung carries the same number of retained
    # shared-fingerprint captures beside the same number of rejected `-eager` rows. An asymmetry means
    # one rung lost a record rather than currency moving, and holding it bucket by bucket also catches
    # a rung losing a record on one side while gaining one on the other.
    for bucket in (historical_qwen, rejected, retained_bindings):
        by_rung = Counter(run["record"]["strategy"]["rung"] for run in bucket)
        assert set(by_rung) == {"resident", "staged_residency"}, by_rung
        assert len(set(by_rung.values())) == 1, by_rung


def test_complete_calibration_schema_fails_closed_on_adversarial_mutations():
    tmp_path = ROOT / ".tmp" / f"calibration-schema-{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "docs" / "generated").mkdir(parents=True)
    (repo / "docs" / "generated" / "memory-matrix.json").write_text(
        json.dumps({"generatedFrom": {"sceneWorksRevision": f"source-tree:{'1' * 64}"}}),
        encoding="utf-8",
    )
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=SceneWorks test",
            "-c",
            "user.email=test@sceneworks.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    config = {
        "providers": [
            {
                "evidenceScope": "fixture",
                "backend": "candle",
                "loadShape": "eager_materialization",
                "target": {
                    "provider": "krea_2_turbo",
                    "modelId": "krea_2_turbo",
                    "tier": "q4",
                    "mode": "text_to_image",
                    "overlay": "none",
                    "geometry": {"width": 1024, "height": 1024, "batch": 1, "frames": 1},
                },
                "rung": "bounded_decode",
                "engagedRungs": ["resident", "bounded_decode"],
                "calibrationFingerprint": "fixture-formula-v2",
                "fixture": "fixture-seed42",
                "cases": [
                    {
                        "parameters": {"decodeTileEdge": 512, "decodeOverlap": 128},
                        "expectedResult": "passed",
                    }
                ],
            }
        ]
    }
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "evidence.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    provider = ROOT / "scripts/fixtures/memory-provider-fixture.mjs"
    subprocess.run(
        [
            "node",
            "scripts/memory-calibration-harness.mjs",
            "run",
            "--config",
            str(config_path),
            "--provider-command",
            json.dumps(["node", str(provider)]),
            "--sceneworks-repo",
            str(repo),
            "--inference-repo",
            str(repo),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
    )
    schema = json.loads(
        (ROOT / "packages/schemas/memory-calibration.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    bundle = json.loads(output_path.read_text(encoding="utf-8"))
    validator.validate(bundle)
    def scenario(record, name):
        return next(item for item in record["scenarios"] if item["name"] == name)

    mutations = [
        lambda record: record["repositories"]["sceneWorks"].update(dirty=True),
        lambda record: scenario(record, "warm_repeat").update(result="not_run"),
        lambda record: scenario(record, "cancel").update(cleanupVerified=False),
        lambda record: record["quality"].update(result="not_run"),
        lambda record: record["negativeMutation"].update(measured=False),
        lambda record: record["loadability"].update(resolvedPathFingerprint=""),
        lambda record: record.update(predictedPeakBytes=None),
        lambda record: record["observedMemory"]["decode"].update(activeBytes=-1),
        lambda record: record["repositories"]["sceneWorks"].pop("matrixSourceRevision"),
        lambda record: record["sweep"].update(cases=[]),
        lambda record: record["sweep"]["axes"].append(copy.deepcopy(record["sweep"]["axes"][0])),
        lambda record: record["sweep"]["cases"].append(copy.deepcopy(record["sweep"]["cases"][0])),
        lambda record: record["quality"].update(contract=""),
        lambda record: record["hardware"].update(unexpected="closed"),
        lambda record: record["scenarios"][0].update(unexpected="closed"),
        # sc-18864: schema v5 dropped `deviceBytes`/`wiredBytes`, which both adapters emitted as
        # verbatim copies of `allocatorBytes`. That aliasing is what let every committed MLX record
        # claim wired residency above its own probed ceiling. Each alias is reintroduced on its own
        # so the schema is shown to close both, not merely one of the pair.
        lambda record: record["observedMemory"]["overall"].update(
            deviceBytes=record["observedMemory"]["overall"]["allocatorBytes"]
        ),
        lambda record: record["observedMemory"]["overall"].update(
            wiredBytes=record["observedMemory"]["overall"]["allocatorBytes"]
        ),
    ]
    for index, mutate in enumerate(mutations):
        invalid = copy.deepcopy(bundle)
        mutate(invalid["records"][0])
        assert list(validator.iter_errors(invalid)), (
            f"adversarial schema mutation {index} was accepted"
        )
    mutating_provider = ROOT / "scripts/fixtures/memory-provider-mutates-repo-fixture.mjs"
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                "node",
                "scripts/memory-calibration-harness.mjs",
                "run",
                "--config",
                str(config_path),
                "--provider-command",
                json.dumps(["node", str(mutating_provider)]),
                "--sceneworks-repo",
                str(repo),
                "--inference-repo",
                str(repo),
                "--output",
                str(tmp_path / "mutated-evidence.json"),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    def remove_readonly(function, path, _error):
        os.chmod(path, stat.S_IWRITE)
        function(path)

    shutil.rmtree(tmp_path, onexc=remove_readonly)


def test_verified_cells_are_exactly_those_carrying_live_closure_evidence():
    """`Verified` is DERIVED from live-closure evidence — asserted as that rule, not as a roster.

    This used to pin the exact set of Verified coordinates, the exact evidence record id, and
    `fullModels == 0`. None of those can fail on a bug, and all of them are guaranteed to fail on
    the next legitimate closure change or re-capture — the frozen-corpus class. They red on a pin
    bump that memoized a SHA-256 digest, which cannot move any model's memory footprint and so
    cannot invalidate a memory measurement.

    The rule below is what those assertions were reaching for, and it holds across bumps and
    captures alike: a cell is Verified exactly when it carries current-environment evidence, and
    every such piece of evidence resolves to a calibration record whose captured closure digest is
    still live for its provider lane. A demotion still fails this (the cell would claim Verified
    with evidence the ledger no longer calls current); so does a promotion with no evidence behind
    it.
    """
    matrix = load_matrix()
    calibration = json.loads(
        (ROOT / "docs/generated/memory-calibration-evidence.json").read_text(encoding="utf-8")
    )
    live_closures = json.loads(
        (ROOT / "config/inference-provider-closures.json").read_text(encoding="utf-8")
    )["providers"]
    current_ids = {
        record["id"]
        for record in calibration["records"]
        if live_closures.get(
            f"{record['backend']}:{record['target']['provider']}", {}
        ).get("digest")
        == record["repositories"]["inference"]["closureDigest"]
    }

    def key(cell):
        return (cell["modelId"], cell["backend"], cell["tier"], cell["rung"])

    verified = {key(cell) for cell in matrix["cells"] if cell["state"] == "Verified"}
    carries_current = {
        key(cell)
        for cell in matrix["cells"]
        if cell["evidence"]["currentEnvironmentVerification"]
    }
    assert verified == carries_current
    # Every current-environment citation resolves to a record the ledger still calls current. This
    # is the assertion that actually catches a stale matrix: a record demoted by a closure move
    # cannot keep authorizing a Verified cell.
    for cell in matrix["cells"]:
        for item in cell["evidence"]["currentEnvironmentVerification"]:
            assert item["source"].startswith(
                "docs/generated/memory-calibration-evidence.json#"
            )
            assert item["source"].split("#", 1)[1] in current_ids, (
                f"{key(cell)} cites {item['source']}, which the live closures no longer call current"
            )
    # Historical evidence must never be read as verification, whatever the corpus holds.
    assert all(
        cell["state"] != "Verified"
        for cell in matrix["cells"]
        if cell["evidence"]["historicalVerification"]
        and not cell["evidence"]["currentEnvironmentVerification"]
    )
    # `fullModels` is a DERIVED count — assert the derivation, not the number it happens to be.
    fully_verified_models = {
        model_id
        for model_id in {cell["modelId"] for cell in matrix["cells"]}
        if all(
            cell["state"] == "Verified"
            for cell in matrix["cells"]
            if cell["modelId"] == model_id
        )
    }
    assert matrix["summary"]["fullModels"] == len(fully_verified_models)
    current_z_image_turbo = [
        cell
        for cell in matrix["cells"]
        if cell["modelId"] == "z_image_turbo"
        and cell["backend"] == "mlx"
        and cell["tier"] == "q4"
        and cell["mode"] == "text_to_image"
        and cell["overlay"] == "none"
        and cell["evidence"]["currentEnvironmentVerification"]
    ]
    assert current_z_image_turbo == []
    historical_z_image = [
        cell
        for cell in matrix["cells"]
        if cell["modelId"] == "z_image"
        and cell["backend"] == "candle"
        and cell["evidence"]["historicalVerification"]
    ]
    assert historical_z_image == []
    # sc-16915's eager ladder and SC-18353's deferred physical captures both remain historical
    # after the provider closure moved (SC-18306 moved the closure again). The five bf16
    # coordinates stay implemented, retain their characterization and strategy parameters, and
    # fail closed as unverified.
    historical_qwen_cells = [
        cell
        for cell in matrix["cells"]
        if cell["modelId"] == "qwen_image"
        and cell["backend"] == "mlx"
        and cell["tier"] == "bf16"
        and cell["mode"] == "text_to_image"
        and cell["overlay"] == "none"
    ]
    assert len(historical_qwen_cells) == 5
    assert all(cell["state"] == "Implemented/unverified" for cell in historical_qwen_cells)
    assert all(
        not cell["evidence"]["currentEnvironmentVerification"]
        and cell["evidence"]["historicalVerification"]
        for cell in historical_qwen_cells
    ), "frozen-closure Qwen cells must retain history without claiming current verification"
    assert {
        (cell["modelId"], cell["backend"], cell["tier"], cell["mode"], cell["overlay"])
        for cell in historical_qwen_cells
    } == {("qwen_image", "mlx", "bf16", "text_to_image", "none")}
    assert {cell["rung"] for cell in historical_qwen_cells} == {
        "resident",
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
    }
    # The MLX Qwen ladder sc-16915 recaptured, bound at the production point. Kept separate from
    # `expected_parameters` below, which belongs to the CANDLE Krea cells and still names the
    # 128 / 128 MiB point — the two backends are not required to agree.
    expected_qwen_parameters = {
        "resident": {},
        "staged_residency": {},
        "bounded_decode": {"decodeTileEdge": 512, "decodeOverlap": 64},
        "bounded_attention": {
            "decodeTileEdge": 512,
            "decodeOverlap": 64,
            "attentionChunkSize": 67_108_864,
        },
        "bounded_transformer_residency": {
            "decodeTileEdge": 512,
            "decodeOverlap": 64,
            "attentionChunkSize": 67_108_864,
            "transformerWindowSize": 1,
            "transformerWindowComponent": "Dit",
        },
    }
    for cell in historical_qwen_cells:
        assert {
            key: value
            for key, value in cell["strategyParameters"].items()
            if key not in {"manifestRung", "formula", "publishedRanges"}
        } == expected_qwen_parameters[cell["rung"]]

    expected_parameters = {
        "resident": {},
        "staged_residency": {},
        "bounded_decode": {"decodeTileEdge": 512, "decodeOverlap": 128},
        "bounded_attention": {
            "decodeTileEdge": 512,
            "decodeOverlap": 128,
            "attentionChunkSize": 134_217_728,
        },
        "bounded_transformer_residency": {
            "decodeTileEdge": 512,
            "decodeOverlap": 128,
            "attentionChunkSize": 134_217_728,
            "transformerWindowSize": 1,
        },
    }
    krea_cells = [
        cell
        for cell in matrix["cells"]
        if cell["modelId"] == "krea_2_turbo"
        and cell["backend"] == "candle"
        and cell["mode"] == "text_to_image"
        and cell["overlay"] == "none"
    ]
    assert krea_cells
    # Which Krea coordinates are Verified is a property of the CORPUS, not of this lane's contract,
    # so it is derived rather than pinned (same frozen-corpus repair as the currency assertions
    # above). What this lane must guarantee: a cell is Verified exactly when it carries current
    # evidence, and every other cell fails closed to implemented-but-unverified rather than to some
    # third state.
    assert all(
        cell["state"]
        == (
            "Verified"
            if cell["evidence"]["currentEnvironmentVerification"]
            else "Implemented/unverified"
        )
        for cell in krea_cells
    )
    for cell in krea_cells:
        parameters = cell["strategyParameters"]
        assert {
            key: value
            for key, value in parameters.items()
            if key not in {"manifestRung", "formula"}
        } == expected_parameters[cell["rung"]]
        for record in cell["evidence"]["strategyParameterVerification"]:
            assert record["exactParameters"] == expected_parameters[cell["rung"]]
    adapter_streaming = [
        cell
        for cell in matrix["cells"]
        if cell["modelId"] == "krea_2_turbo"
        and cell["backend"] == "candle"
        and cell["mode"] == "text_to_image"
        and cell["overlay"] == "lora"
        and cell["rung"] == "bounded_transformer_residency"
    ]
    assert adapter_streaming
    assert all(
        cell["state"] == "Structurally N/A" and cell["evidence"]["structural"]
        for cell in adapter_streaming
    )
    for cell in matrix["cells"]:
        if cell["state"] != "Missing":
            assert cell["evidence"]["staticImplementation"]
            assert cell["calibrationFingerprint"]


def test_matrix_schema_rejects_malformed_evidence_records():
    matrix = load_matrix()
    validator = matrix_validator()
    krea_index = next(
        index
        for index, cell in enumerate(matrix["cells"])
        if cell["modelId"] == "krea_2_turbo"
        and cell["backend"] == "candle"
        and cell["tier"] == "q4"
        and cell["mode"] == "text_to_image"
        and cell["rung"] == "staged_residency"
        and cell["overlay"] == "none"
    )

    def rejected(mutate):
        candidate = copy.deepcopy(matrix)
        mutate(candidate["cells"][krea_index]["evidence"])
        assert list(validator.iter_errors(candidate))

    rejected(lambda evidence: evidence["historicalVerification"][0].pop("source"))
    rejected(
        lambda evidence: evidence["historicalVerification"][0].__setitem__(
            "observedPeakGb", "unknown"
        )
    )
    rejected(
        lambda evidence: evidence["historicalVerification"][0].__setitem__(
            "runtimeAdmission", "unknown"
        )
    )
    rejected(
        lambda evidence: evidence["historicalVerification"][0].__setitem__(
            "runtimeAdmission", False
        )
    )
    rejected(
        lambda evidence: evidence["historicalVerification"][0].pop("parity")
    )
    rejected(
        lambda evidence: evidence["historicalVerification"][0].pop(
            "runtimeAdmission"
        )
    )
    rejected(
        lambda evidence: evidence["historicalVerification"][0].pop(
            "evidenceScope"
        )
    )

    def phase_fit_with_runtime_admission(evidence):
        record = evidence["historicalVerification"][0]
        record["evidenceScope"] = "phase_fit_only"
        record.pop("parity")

    rejected(phase_fit_with_runtime_admission)

    def phase_fit_with_parity(evidence):
        record = evidence["historicalVerification"][0]
        record["evidenceScope"] = "phase_fit_only"
        record["runtimeAdmission"] = False

    rejected(phase_fit_with_parity)
    rejected(
        lambda evidence: evidence["strategyParameterVerification"][0].__setitem__(
            "geometry", "up to 1024"
        )
    )
    # sc-18099 hoisted `loadability` (and `declaredCalibration`) out of the cell and into the
    # document's `manifestScopes` map — they are functions of (entry, backend, tier) alone. The
    # schema must still fail closed on a malformed entry there, so the mutation follows the data
    # rather than being dropped: dropping it is how a hoist quietly retires a gate.
    def rejected_scope(mutate):
        candidate = copy.deepcopy(matrix)
        scope_key = candidate["cells"][krea_index]["evidence"]["manifestScope"]
        mutate(candidate["manifestScopes"][scope_key])
        assert list(validator.iter_errors(candidate))

    rejected_scope(
        lambda scope: scope["loadability"][0].__setitem__("unchecked", True)
    )
    rejected_scope(lambda scope: scope["loadability"][0].pop("repository"))
    rejected_scope(
        lambda scope: scope["declaredCalibration"][0].__setitem__("unchecked", True)
    )
    rejected_scope(lambda scope: scope["declaredCalibration"][0].pop("tier"))
    # ...and the cell's end of the join is required, so a cell cannot silently lose its scope.
    rejected(lambda evidence: evidence.pop("manifestScope"))


def test_rung4_survey_covers_every_family_and_rides_only_its_own_cells():
    """SC-15969: the per-family rung-4 applicability survey reaches the generated cells.

    The survey exists so a rung-4 verdict is a generated cell rather than prose. Two things make it
    real rather than decorative: it covers every (family, backend) the CATALOG advertises, and its
    verdict is attached to exactly the rung-4 cells — a cell that escaped the survey, or a verdict
    that drifted onto another rung, is a hole a consumer only finds at runtime.
    """
    matrix = load_matrix()
    rung4 = [
        cell
        for cell in matrix["cells"]
        if cell["rung"] == "bounded_transformer_residency"
    ]
    assert rung4
    assert all(cell["rung4Survey"]["story"] == 15969 for cell in rung4)
    assert not [
        cell
        for cell in matrix["cells"]
        if cell["rung"] != "bounded_transformer_residency" and "rung4Survey" in cell
    ]

    rows = matrix["rung4SurveyRows"]
    assert len(rows) == matrix["summary"]["rung4Survey"]["surveyedFamilyBackends"]
    # `familyStory` is the family GROUP key — the family's MLX story id for the epic-15448 image
    # families, and the family's own NAME for the video families sc-18815 admits (sc-18828 owns three
    # unrelated families at once, so no assignment of story ids to them is both distinct and true).
    # It is the same key on both backends, unlike `cells[].owningFamilyStory`, which is the
    # backend-scoped owner. Joined through the published `models[].familyGroup` rather than through
    # `owningFamilyStories["mlx"]`: that identity holds for image families only.
    advertised = {
        (model["familyGroup"], backend)
        for model in matrix["models"]
        for backend in model["backends"]
    }
    surveyed = {(row["familyStory"], row["backend"]) for row in rows}
    # SC-18826/SC-18828 close the current survey debt. The partition machinery remains published for
    # future families, but at this revision every advertised family/backend has a real verdict.
    pending = {
        (row["family"], row["backend"])
        for row in matrix["summary"]["rung4Survey"]["pendingFamilyBackends"]
    }
    assert not (surveyed & pending)
    assert surveyed | pending == advertised
    assert pending == set()
    assert all(
        isinstance(row["pendingSurveyStory"], int)
        for row in matrix["summary"]["rung4Survey"]["pendingFamilyBackends"]
    )

    # The discriminator is on every rung-4 cell and agrees with the rows, so "not surveyed yet" can
    # never be read as "surveyed and found to have no implementation". Both directions, or the flag
    # could be published as a constant and still satisfy the partition above.
    family_of = {model["id"]: model["familyGroup"] for model in matrix["models"]}
    for cell in rung4:
        key = (family_of[cell["modelId"]], cell["backend"])
        assert cell["rung4Survey"]["surveyed"] is (key in surveyed), cell["id"]
        if not cell["rung4Survey"]["surveyed"]:
            assert cell["state"] == "Missing", cell["id"]
            assert cell["rung4Survey"]["requestPeak"] == "unsurveyed"
            assert cell["rung4Survey"]["structuralApplicability"] is None
            assert cell["rung4Survey"]["implementation"] is None
            assert isinstance(cell["rung4Survey"]["pendingSurveyStory"], int)
    assert any(cell["rung4Survey"]["surveyed"] for cell in rung4)
    # Today every published rung-4 cell is surveyed and the summary confirms there is no hidden debt.
    assert all(cell["rung4Survey"]["surveyed"] for cell in rung4)
    assert not pending

    # The two findings stay separate: an architecture that CAN be windowed is never, by itself,
    # evidence that windowing it moves the request peak.
    assert {row["requestPeak"] for row in rows} <= {"moves", "does-not-move", "unmeasured"}
    assert [
        (row["familyStory"], row["backend"])
        for row in rows
        if row["requestPeak"] == "moves"
    ] == [
        (15510, "candle"),
        (15510, "mlx"),
        (15511, "mlx"),
        (15512, "candle"),
        (15512, "mlx"),
        (15517, "candle"),
        (15517, "mlx"),
        (15519, "candle"),
        # SC-15520 Chroma1 lands its MLX ladder: rung 4 at `Dit` scope moves the staged request
        # peak 19.2065 -> 14.6932 GiB (-23.50%) on Chroma1-Base q4 at 1024^2, byte-identical
        # output at every cadence in [1, 2, 5, 10]. The measured scope is exactly one cell
        # (chroma1_base/q4/text_to_image/none); every sibling entry, tier, mode and overlay
        # stays `unmeasured`.
        (15520, "mlx"),
        # SC-15521 Kolors, SC-15524 Anima and SC-15525 SDXL + derivatives land their MLX ladders
        # with measured request peaks: Anima 5.229 -> 4.151 GiB at window 1; SDXL -6.97% (q4) to
        # -21.40% (bf16) per entry per tier; Kolors -7.21% / -12.72% / -21.37% by tier, plus the
        # ladder's first three-valued scope axis (`Dit` / `TextEncoder` / `Both`) reading
        # 11.3644 / 8.8396 / 4.5436 GiB at bf16/512.
        (15521, "mlx"),
        (15524, "mlx"),
        (15525, "mlx"),
    ]
    flux2_candle = next(
        row
        for row in rows
        if row["familyStory"] == 15519 and row["backend"] == "candle"
    )
    # sc-18099 moved the family-level `summary`/`blockStacks`/`findings` onto these rows, so the
    # verdict survives every one of the family's cells being elided. Still an EXACT comparison of the
    # verdict fields — a bare subset check would stop noticing a field going missing — with the three
    # prose/inventory fields asserted as present-and-non-trivial rather than transcribed here.
    assert {
        key: value
        for key, value in flux2_candle.items()
        if key not in {"summary", "blockStacks", "findings"}
    } == {
        "familyStory": 15519,
        "backend": "candle",
        "structuralApplicability": "partial",
        "requestPeak": "moves",
        "implementation": "shared-primitive",
    }
    assert flux2_candle["summary"]
    assert flux2_candle["blockStacks"]
    # Every surveyed family/backend carries its inventory here, not only the ones that publish a
    # cell. That is the whole reason these fields moved: none of the rows can go dark.
    assert all(row["summary"] and row["blockStacks"] for row in rows)
    assert all(
        row["implementation"] != "none"
        for row in rows
        if row["requestPeak"] != "unmeasured"
    )


def test_rung4_partial_applicability_and_structural_verdicts_carry_their_evidence():
    """Partial applicability is recorded, and a Structurally N/A cell always cites why."""
    matrix = load_matrix()
    coverage = {
        (row["modelId"], row["backend"], row["rung"]): row for row in matrix["coverage"]
    }
    models_by_id = {model["id"]: model for model in matrix["models"]}

    # sc-18099 slimmed the artifact to planned-or-evidenced cells, and SDXL's rung-4 coordinates are
    # neither, so this entry now publishes no rung-4 cell at all. Nothing below was dropped: the
    # family verdict moved to `rung4SurveyRows` and the per-lane state distribution to `coverage`,
    # both derived from EVERY resolved coordinate. The per-coordinate spellings of these same claims
    # are asserted at full reach against the pre-publication document in
    # scripts/generate-memory-matrix.test.mjs.
    #
    # The story's named trap: a U-Net is not automatically Structurally N/A. SDXL's lowest level is
    # a genuine 10-deep transformer stack, so the verdict is `partial` — applicable, and now
    # partially IMPLEMENTED (SC-15525 / SC-16355 shipped the per-Transformer2D stream) rather than
    # exempt from the ladder. `partial` survives implementation: it describes the ARCHITECTURE (a
    # non-windowable conv/resnet trunk around eleven windowable Transformer2D sub-stacks), not the
    # delivery state, so it must not collapse to `full` just because the rung now ships.
    sdxl_rows = [
        row
        for row in matrix["rung4SurveyRows"]
        if row["familyStory"] == 15525
    ]
    assert sdxl_rows
    assert {row["structuralApplicability"] for row in sdxl_rows} == {"partial"}
    # Coverage is per entry per tier per overlay, never family-wide: the base `sdxl` entry publishes
    # rung 4 on bf16/overlay-none only, so both states must be present across this entry's lane.
    sdxl_lane = [
        row
        for key, row in coverage.items()
        if key[0] == "sdxl" and key[2] == "bounded_transformer_residency"
    ]
    assert sdxl_lane
    assert set().union(*(row["states"].keys() for row in sdxl_lane)) == {
        "Missing",
        "Implemented/unverified",
    }
    # Where it IS implemented it is on exactly the production coordinates the manifest declares
    # (sc-21609): text_to_image and edit_image at clean base, character_image at the identity
    # overlay — image_inpaint canonicalizes to edit_image before routing and image_detail always
    # loads a tile ControlNet (control overlay, no cell), so neither can carry the rung. Derived as
    # a mode COUNT rather than a hardcoded number so a legitimate tier drift does not stale it; the
    # exact mode/overlay split is asserted per-coordinate by the sibling JS test.
    SDXL_RUNG4_MODES = {"text_to_image", "edit_image", "character_image"}
    implementing = [row for row in sdxl_lane if row["implemented"]]
    assert implementing, "SDXL's rung-4 coverage must not vanish"
    for row in sdxl_lane:
        axes = models_by_id["sdxl"]["axes"][row["backend"]]
        assert "bf16" in axes["tiers"] and "none" in axes["overlays"]
        assert row["coordinates"] == len(axes["tiers"]) * len(axes["modes"]) * len(
            axes["overlays"]
        )
        assert row["implemented"] in (
            0,
            len(SDXL_RUNG4_MODES & set(axes["modes"])),
        ), "SDXL publishes rung 4 on exactly its declared production modes, or not at all"
    # Rung 4 is Missing OUTRIGHT on both Illustrious entries: q8 is their only advertised tier and
    # its snapshot omits the `quantization` marker, so `streamable` refuses (inference sc-17522).
    # A partially-implemented family must not carry its siblings' coverage onto them.
    illustrious = [
        row
        for key, row in coverage.items()
        if key[0] in {"illustrious_xl_v1", "illustrious_xl_v2"}
        and key[2] == "bounded_transformer_residency"
    ]
    assert illustrious
    assert set().union(*(row["states"].keys() for row in illustrious)) == {"Missing"}
    assert all(row["implemented"] == 0 for row in illustrious)
    stacks = sdxl_rows[0]["blockStacks"]
    assert any(stack["windowable"] for stack in stacks)
    assert any(not stack["windowable"] for stack in stacks)

    # SC-18828's SVD correction exercises the same distinction on video: the U-Net is
    # heterogeneous, but each lane contains 16 separately indexed one-block spatial stacks and 16
    # one-block temporal stacks. The conv/ResNet/skip trunk stays resident, so both twins are
    # `partial`; with no materializer implemented, their resolved rung-4 coordinates are Missing.
    svd_rows = [
        row for row in matrix["rung4SurveyRows"] if row["familyStory"] == "svd"
    ]
    assert {row["backend"] for row in svd_rows} == {"mlx", "candle"}
    assert all(row["structuralApplicability"] == "partial" for row in svd_rows)
    assert all(row["implementation"] == "none" for row in svd_rows)
    for row in svd_rows:
        assert [
            (stack["blocks"], stack["windowable"]) for stack in row["blockStacks"]
        ] == [
            ("16 × 1 BasicBlock", True),
            ("16 × 1 TemporalBlock", True),
            ("4 down + 1 mid + 4 up stages", False),
        ]
    svd_rung4 = [
        row
        for key, row in coverage.items()
        if key[0] == "svd" and key[2] == "bounded_transformer_residency"
    ]
    assert {row["backend"] for row in svd_rung4} == {"mlx", "candle"}
    assert all(row["states"] == {"Missing": row["coordinates"]} for row in svd_rung4)
    assert all(row["implemented"] == 0 for row in svd_rung4)

    for cell in matrix["cells"]:
        if cell["rung"] != "bounded_transformer_residency":
            continue
        survey = cell["rung4Survey"]
        if cell["state"] == "Structurally N/A":
            assert cell["evidence"]["structural"], cell["id"]
            # Exempt for exactly one of two reasons, and the cell says which: the ARCHITECTURE has
            # nothing to window, or the provider's adapter mechanism cannot carry an overlay onto a
            # rebuilt block. Conflating them would publish `none` for a family whose stack is
            # perfectly windowable.
            assert (
                survey["structuralApplicability"] == "none"
                or survey["overlayIncompatible"]
            ), cell["id"]
        else:
            assert survey["structuralApplicability"] != "none", cell["id"]
            assert not survey["overlayIncompatible"], cell["id"]

    # An overlay exemption is only honest where the streaming path exists: on an entry with no such
    # path the rung is Missing for the ordinary reason, and exempting it would presuppose a path
    # that does not exist AND silently drop the cell from the calibration workload.
    exempt_overlay = [
        cell
        for cell in matrix["cells"]
        if cell["rung"] == "bounded_transformer_residency"
        and cell["rung4Survey"]["overlayIncompatible"]
    ]
    assert exempt_overlay
    assert all(cell["overlay"] != "none" for cell in exempt_overlay)
    assert all(
        cell["rung4Survey"]["implementation"] != "none" for cell in exempt_overlay
    )
