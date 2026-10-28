import copy
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import traceback
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


def test_matrix_accounts_for_all_models_and_mlx_staged_coverage_is_consistent():
    matrix = load_matrix()
    # sc-18815: the universe is modality-aware, so the census is too. `imageModels` was REMOVED
    # rather than left holding the whole-universe total under a one-modality name. Derive the
    # populations from the published rows so adding a catalog entry cannot stale a pinned count.
    assert "imageModels" not in matrix["summary"]
    image_ids = {model["id"] for model in matrix["models"] if model["modality"] == "image"}
    video_ids = {model["id"] for model in matrix["models"] if model["modality"] == "video"}
    assert matrix["summary"]["catalogEntries"] == len(matrix["models"])
    assert matrix["summary"]["catalogEntriesByModality"] == {
        "image": len(image_ids),
        "video": len(video_ids),
    }
    # SC-18218 closes FLUX.2-dev to its measured Resident-only provider contract, so its former
    # generic staged-route claim is intentionally absent from this census.
    # sc-18815 keeps this as exactly the IMAGE-lane claim its denominator says it is. The separate
    # video census consumes the provider-owned staged-residency contracts. Both numerators are
    # asserted structurally rather than as pinned populations (the SC-18218
    # shape-over-population ruling); the denominators are the modality totals derived above.
    assert matrix["summary"]["mlxStagedStaticCoverageDenominator"] == len(image_ids)
    # sc-22512: a coverage NUMERATOR is a count of declarations that happen to exist, so it may
    # legitimately be zero — a catalog whose video lane declares no staged residency is a catalog
    # nobody has measured yet, not a broken document. Only the containment relation is asserted.
    assert 0 <= matrix["summary"]["videoMlxStagedStaticCoverage"] <= len(video_ids)
    assert matrix["summary"]["videoMlxStagedStaticCoverageDenominator"] == len(video_ids)
    assert len(matrix["models"]) == len(matrix["modelSlices"])
    assert {model["id"] for model in matrix["models"]} == set(matrix["modelSlices"])
    # SC-18218 closes FLUX.2-dev to its measured Resident-only provider contract, so its former
    # generic staged-route claim is intentionally absent from this census — the coverage number is
    # a strict subset of the denominator, and the denominator is the whole census. The number
    # itself is recomputed from the coverage rows below rather than pinned here.
    assert (
        0
        <= matrix["summary"]["mlxStagedStaticCoverage"]
        <= matrix["summary"]["mlxStagedStaticCoverageDenominator"]
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
    # sc-22512 removed the two named-entry census requirements (`flux2_dev` and `bernini_image` were
    # each asserted INTO `mlx_staged`) and the "partial by construction" band. All three reddened on
    # the ABSENCE of a declaration: an inference pin that stops advertising selectable Sequential
    # residency for one provider, or a catalog whose whole image lane declares none, is a lane
    # nobody has measured — not a defect in this document. What survives is the containment
    # relation, which holds at any coverage level including zero.
    assert mlx_staged <= image_ids
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
    # sc-22513: the root `calibrationRuns` join is gone. What must stay closed is the anchor
    # inventory — every anchor a published cell cites is a published row.
    assert {cell["anchor"]["id"] for cell in matrix["cells"] if cell["anchor"]} <= {
        row["id"] for row in matrix["anchors"]
    }


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
    # sc-22513: three dimensions, and the four record-derived ones are gone with the per-record
    # join. `anchor` is the only memory evidence a cell now carries.
    assert matrix["evidenceDimensions"] == [
        "staticImplementation",
        "structural",
        "anchor",
    ]


def test_retained_calibration_corpus_stays_schema_valid():
    """The historical calibration corpus is RETAINED as validation data for the anchor derivation
    (epic 22505, E5) — never as a gate, and no longer joined to the matrix. What must still hold is
    that the retained bundle is well-formed: every record schema-valid, every id an immutable
    `imc-` capture id, every status one the schema admits. The matrix-ingestion half of this test
    went with the per-record join sc-22513 deleted."""
    calibration_schema = json.loads(CALIBRATION_SCHEMA.read_text(encoding="utf-8"))
    calibration = json.loads(
        (ROOT / "docs/generated/memory-calibration-evidence.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(
        calibration_schema, format_checker=jsonschema.FormatChecker()
    ).validate(calibration)
    evidence_ids = {record["id"] for record in calibration["records"]}
    assert all(re.fullmatch(r"imc-[0-9a-f]{20}", record_id) for record_id in evidence_ids)
    assert len(evidence_ids) == len(calibration["records"])
    records_by_status = Counter(record["status"] for record in calibration["records"])
    record_statuses = calibration_schema["$defs"]["record"]["properties"]["status"]["enum"]
    assert set(records_by_status) <= set(record_statuses)
    assert records_by_status["complete"] > 0
    assert records_by_status["runtime_complete"] > 0



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
    # sc-22514: the anchor plan format. One entry, keyed `<modelId>:<tier>:<backend>`; the
    # composition and every constant the adapter reads are attached by the harness, not planned.
    anchor_key = "krea_2_turbo:q4:candle"
    plan = {
        "schemaVersion": 1,
        "anchors": {
            anchor_key: {
                "provider": "krea_2_turbo",
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": {"width": 1024, "height": 1024, "batch": 1, "frames": 1},
                "evidenceScope": "fixture",
                "loadShape": "eager_materialization",
                "calibrationFingerprint": "fixture-formula-v2",
                "fixture": "fixture-seed42",
            }
        },
    }
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "evidence.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    provider = ROOT / "scripts/fixtures/memory-provider-fixture.mjs"
    subprocess.run(
        [
            "node",
            "scripts/memory-calibration-harness.mjs",
            "capture",
            "--plan",
            str(plan_path),
            "--anchor",
            anchor_key,
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
        # sc-22514: an ANCHOR sweeps nothing, so its truthful sweep is degenerate (no axes, one
        # unparameterized passed case). The duplicate-axis mutation is therefore written out rather
        # than cloned from an axis that no longer exists; the schema uniqueItems closure is what is
        # under test either way.
        lambda record: record["sweep"].update(
            axes=[
                {"parameter": "decodeTileEdge", "testedValues": [512]},
                {"parameter": "decodeTileEdge", "testedValues": [512]},
            ]
        ),
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
                "capture",
                "--plan",
                str(plan_path),
                "--anchor",
                anchor_key,
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

def test_collapsed_cell_state_is_a_pure_function_of_the_published_facts():
    """sc-22513 (epic 22505, E5): `state = f(implementation, anchor present, derivation defined,
    anchor derivable)` — the fourth fact is anchor-LEVEL derivability (feature-end fix round):
    an anchor the lane's law refuses (the store marks it `underivedReason`) publishes
    Anchored/underived, not Anchored.

    Asserted three ways, because "pure" needs all three to mean anything:

    1. The vocabulary is the collapsed one. `Verified`, `Runtime verified` and
       `Implemented/unverified` are GONE, and no cell carries per-geometry evidence bookkeeping.
    2. The function is reproduced independently here, from the three facts each cell publishes.
       A state that started depending on a record, a plan row or a geometry would have to disagree
       with this table on some cell.
    3. The mapping is single-valued: every cell sharing the fact triple holds the same state. That
       is the half a re-derivation cannot fake, because an outside input would split one triple
       into two states.
    """
    matrix = load_matrix()

    def expected(implementation, anchored, derivation_defined, anchor_derivable):
        if implementation == "missing":
            return "Missing"
        if implementation == "structurally-na":
            return "Structurally N/A"
        assert implementation == "implemented"
        if not anchored:
            return "Implemented"
        return (
            "Anchored"
            if derivation_defined and anchor_derivable
            else "Anchored/underived"
        )

    states = {state["state"] for state in matrix["conformanceStates"]}
    assert states == {
        "Anchored",
        "Anchored/underived",
        "Implemented",
        "Structurally N/A",
        "Missing",
    }
    retired = {"Verified", "Runtime verified", "Implemented/unverified"}
    assert not states & retired
    assert "memoryCharacterizationStates" not in matrix
    assert "calibrationRuns" not in matrix
    assert "rung4SurveyRows" not in matrix
    assert "manifestScopes" not in matrix
    assert "inferenceRevision" not in matrix["generatedFrom"]

    by_facts: dict[tuple, set[str]] = {}
    for cell in matrix["cells"]:
        # No cell may carry per-geometry evidence bookkeeping any more.
        for retired_field in (
            "memoryCharacterization",
            "calibrationFingerprint",
            "engagedRungs",
            "plannedPipelineIdentities",
            "pipelineCharacterizations",
            "rung4Survey",
        ):
            assert retired_field not in cell, cell["id"]
        assert set(cell["evidence"]) == {"staticImplementation", "structural", "anchor"}
        assert cell["state"] not in retired
        facts = (
            cell["implementation"],
            cell["anchor"] is not None,
            cell["derivationDefined"],
            cell["anchor"] is not None and cell["anchor"]["derivable"],
        )
        assert cell["state"] == expected(*facts), cell["id"]
        by_facts.setdefault(facts, set()).add(cell["state"])
    assert all(len(seen) == 1 for seen in by_facts.values()), {
        facts: sorted(seen) for facts, seen in by_facts.items() if len(seen) > 1
    }
    # The published population must actually exercise the interesting corners, or the table above
    # is asserted vacuously.
    assert {"Anchored", "Structurally N/A"} <= {cell["state"] for cell in matrix["cells"]}
    assert any(cell["anchor"] is not None for cell in matrix["cells"])


def test_anchor_currency_is_reported_beside_the_state_and_never_moves_it():
    """sc-22511: currency reports capture provenance and cannot partition the states."""
    matrix = load_matrix()
    anchored = [cell for cell in matrix["cells"] if cell["anchor"] is not None]
    assert anchored
    by_current: dict[bool, set[str]] = {}
    for cell in anchored:
        by_current.setdefault(cell["anchor"]["current"], set()).add(cell["state"])
    # The currency flag is a boolean report and nothing else. Which values the checked-in store
    # carries is provenance at this pin, not a distribution this test may freeze. The assertions
    # below therefore re-derive every state without a currency term.
    assert by_current and set(by_current) <= {True, False}, sorted(by_current)
    # Every anchored cell's state must re-derive from the three facts WITHOUT a currency term. An
    # anchor may land on a coordinate the architecture rules out (sc-22509 measured krea_2_turbo
    # candle q4, whose streamed-blocks overlay coordinates are structurally exempt), so the
    # allowed vocabulary is not `Anchored`/`Anchored/underived` alone — the point is that flipping
    # `current` could not have produced any of these states.
    for cell in anchored:
        if cell["implementation"] == "missing":
            expected = "Missing"
        elif cell["implementation"] == "structurally-na":
            expected = "Structurally N/A"
        elif cell["derivationDefined"] and cell["anchor"]["derivable"]:
            expected = "Anchored"
        else:
            expected = "Anchored/underived"
        assert cell["state"] == expected, cell["id"]
    # Teeth: the anchored population must span more than one state, or the loop above is vacuous.
    assert len({cell["state"] for cell in anchored}) > 1
    # A cell's reported currency is its inventory row's, not an independent claim: two copies of one
    # fact that can disagree is how a stale anchor reads as current on the cell that cites it.
    inventory = {row["id"]: row for row in matrix["anchors"]}
    for cell in anchored:
        assert cell["anchor"]["current"] == inventory[cell["anchor"]["id"]]["current"], cell["id"]
    assert matrix["summary"]["staleAnchors"] == sum(
        1 for row in matrix["anchors"] if not row["current"]
    )
    # Historical captures keep their own currency. Judge today's generator against today's
    # ledger separately, so a pin change cannot require a new matrix or measurement campaign.
    live = json.loads(
        subprocess.run(
            ["node", "--input-type=module", "-e",
             'import { buildMatrix } from "./scripts/generate-memory-matrix.mjs"; '
             'const matrix = await buildMatrix(); '
             'console.log(JSON.stringify({anchors:matrix.anchors, '
             'cells:matrix.cells.filter(c => c.anchor).map(c => ({id:c.id,anchor:c.anchor})), '
             'staleAnchors:matrix.summary.staleAnchors}));'],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
    )
    live_anchors = live["anchors"]
    assert live_anchors and live["cells"], "live currency validation must exercise an anchored cell"
    live_inventory = {row["id"]: row for row in live_anchors}
    for cell in live["cells"]:
        assert cell["anchor"]["current"] == live_inventory[cell["anchor"]["id"]]["current"], cell["id"]
    assert live["staleAnchors"] == sum(1 for row in live_anchors if not row["current"])
    closures = json.loads(
        (ROOT / "config/anchor-loader-closures.json").read_text(encoding="utf-8")
    )["models"]
    store = json.loads((ROOT / "config/memory-anchors.json").read_text(encoding="utf-8"))
    by_id = {anchor["id"]: anchor for anchor in store["anchors"]}
    for row in live_anchors:
        anchor = by_id[row["id"]]
        assert row["current"] == (
            closures.get(f"{anchor['modelId']}:{anchor['backend']}", {}).get("digest")
            == anchor["source"]["loaderClosureDigest"]
        ), row["id"]


def test_the_fingerprint_covers_only_the_anchor_and_catalog_sources():
    """sc-22513: the matrix's source set is the anchor store, its currency declarations, the
    derivation/extraction sources, and the routing/catalog sources the population derives from.

    The calibration plan, the calibration evidence bundle, the provider closure ledger, the rung-4
    survey artifacts and the Cargo pin LEFT — none of them can move a cell any more, and a source
    that cannot move a cell must not rotate the artifact's revision. Asserted here against the
    PUBLISHED source set, which is the copy a consumer reads."""
    matrix = load_matrix()
    sources = {name: entry["path"] for name, entry in matrix["generatedFrom"]["sources"].items()}
    assert sources["anchorStore"] == "config/memory-anchors.json"
    assert sources["anchorLoaderClosures"] == "config/anchor-loader-closures.json"
    assert sources["anchorDerivation"] == "crates/sceneworks-core/src/memory_anchor.rs"
    assert sources["anchorAdmission"] == "crates/sceneworks-worker/src/video_admission.rs"
    # Epic 22505 feature-end fix round (E5): each modality's REAL admission sources joined the
    # fingerprint — image cells derive through vram_gate/candle_memory_strategy (candle) and
    # mlx_fit_gate (mlx), so an unwiring edit there must move `derivationDefined`.
    assert sources["anchorAdmissionImageVram"] == "crates/sceneworks-worker/src/vram_gate.rs"
    assert (
        sources["anchorAdmissionImageCandle"]
        == "crates/sceneworks-worker/src/candle_memory_strategy.rs"
    )
    assert sources["anchorExtractor"] == "scripts/extract-memory-anchors.mjs"
    assert sources["manifest"] == "config/manifests/builtin.models.jsonc"
    removed = {
        "config/memory-calibration-plan.json",
        "docs/generated/memory-calibration-evidence.json",
        "config/inference-provider-closures.json",
        "config/rung4-applicability-survey.json",
        "config/rung4-contract-prerequisites.json",
        "config/engine-capabilities/capabilities.mlx.json",
        "config/engine-capabilities/capabilities.candle.json",
        "docs/generated/video-memory-curves.json",
        "Cargo.toml",
    }
    assert not removed & set(sources.values()), sorted(removed & set(sources.values()))
    # Every remaining source is an anchor source or a routing/catalog source. Stated as the whole
    # set rather than a spot check, so a source re-joining the fingerprint has to be justified here.
    assert set(sources) == {
        "manifest",
        "routingCatalog",
        "routingCandle",
        "routingMlx",
        "engines",
        "imageRouting",
        "videoRouteWan",
        "videoRouteLtx",
        "videoRouteSvd",
        "videoRouteBernini",
        "videoRouteScail2",
        "videoRouteKreaRealtime",
        "videoRouteCandle",
        "mlxFitGate",
        "memoryRouteRegistry",
        "instantId",
        "anchorStore",
        "anchorLoaderClosures",
        "anchorDerivation",
        "anchorAdmission",
        "anchorAdmissionImageVram",
        "anchorAdmissionImageCandle",
        "anchorExtractor",
    }


def test_the_anchor_inventory_is_closed_against_the_store_and_the_cells():
    """The anchor inventory replaces `calibrationRuns`: the store IS the evidence join now. Both
    directions, because a dangling citation and an anchor that reaches no cell are the same defect
    seen from two sides."""
    matrix = load_matrix()
    store = json.loads((ROOT / "config/memory-anchors.json").read_text(encoding="utf-8"))
    store_ids = {anchor["id"] for anchor in store["anchors"]}
    inventory = {row["id"] for row in matrix["anchors"]}
    assert inventory <= store_ids
    assert matrix["summary"]["anchors"] == len(matrix["anchors"])
    assert matrix["summary"]["analyticOnlyCells"] == len(store["analyticOnly"])
    cited = {cell["anchor"]["id"] for cell in matrix["cells"] if cell["anchor"]}
    assert cited <= inventory
    for row in matrix["anchors"]:
        covering = [
            cell
            for cell in matrix["cells"]
            if cell["anchor"] and cell["anchor"]["id"] == row["id"]
        ]
        # A row's `cells` count is over the RESOLVED coordinates it covers, which is a superset of
        # the published ones — so the published coverage may be smaller, never larger, and never
        # non-zero for a row claiming none.
        assert row["cells"] >= len(covering), row["id"]
        assert (row["cells"] > 0) or not covering, row["id"]
        # An anchor may only be cited on the coordinate it was measured at.
        for cell in covering:
            assert (cell["modelId"], cell["backend"], cell["tier"]) == (
                row["modelId"],
                row["backend"],
                row["tier"],
            ), cell["id"]


def test_matrix_schema_rejects_a_malformed_collapsed_cell():
    """The schema must fail closed on the fields the collapse introduced, or the state's three
    inputs could be published in any shape at all."""
    matrix = load_matrix()
    validator = matrix_validator()
    index = next(
        index for index, cell in enumerate(matrix["cells"]) if cell["anchor"] is not None
    )

    def rejected(mutate):
        candidate = copy.deepcopy(matrix)
        mutate(candidate["cells"][index])
        assert list(validator.iter_errors(candidate))

    rejected(lambda cell: cell.__setitem__("state", "Verified"))
    rejected(lambda cell: cell.__setitem__("state", "Implemented/unverified"))
    rejected(lambda cell: cell.__setitem__("implementation", "unverified"))
    rejected(lambda cell: cell.__setitem__("derivationDefined", "yes"))
    rejected(lambda cell: cell.pop("implementation"))
    rejected(lambda cell: cell.pop("derivationDefined"))
    rejected(lambda cell: cell.pop("anchor"))
    rejected(lambda cell: cell["anchor"].pop("current"))
    rejected(lambda cell: cell["anchor"].__setitem__("current", "stale"))
    rejected(lambda cell: cell["anchor"].pop("id"))
    rejected(lambda cell: cell.__setitem__("memoryCharacterization", {"status": "fitted"}))
    rejected(lambda cell: cell["evidence"].__setitem__("historicalVerification", []))
    rejected(lambda cell: cell["evidence"].pop("anchor"))

    def rejected_document(mutate):
        candidate = copy.deepcopy(matrix)
        mutate(candidate)
        assert list(validator.iter_errors(candidate))

    rejected_document(lambda doc: doc.__setitem__("schemaVersion", 10))
    rejected_document(lambda doc: doc.pop("anchors"))
    rejected_document(lambda doc: doc["anchors"][0].pop("current"))
    rejected_document(lambda doc: doc["generatedFrom"].__setitem__("inferenceRevision", "x" * 40))
    rejected_document(lambda doc: doc["summary"].pop("anchors"))
    rejected_document(
        lambda doc: doc["claims"]["state"].__setitem__("geometrySensitive", True)
    )


def implementation_axis(state):
    """Project a cell state onto the IMPLEMENTATION axis.

    `Implemented`, `Anchored` and `Anchored/underived` all assert the same thing about the code —
    the rung exists on this route — and differ only in whether an anchor prices it. Collapsing them
    is what makes this census a claim about implementation alone, so an anchor landing or staling
    can never move it.
    """
    if state == "Missing":
        return "missing"
    if state == "Structurally N/A":
        return "structurally-na"
    assert state in {"Implemented", "Anchored", "Anchored/underived"}, state
    return "implemented"


_MOSS_FACTS_SCRIPT = """
import { readFileSync } from "node:fs";
import { stripJsoncComments } from "./scripts/lib/jsonc.mjs";
import { buildMatrix } from "./scripts/generate-memory-matrix.mjs";
const manifest = JSON.parse(
  stripJsoncComments(readFileSync("config/manifests/builtin.models.jsonc", "utf8")),
);
const models = manifest.models
  .filter((model) => model.family === "moss_image" && model.type === "image")
  .map((model) => ({
    id: model.id,
    capabilities: model.capabilities,
    loraFamilies: model.loraCompatibility?.families ?? [],
    backends: ["mlx", "candle"].filter((backend) => model[backend] !== undefined),
  }));
// The pure, write-free metadata builder: the CURRENT authoritative-source expectation for the
// MOSS lane's axes and per-(backend, rung) coverage. Nothing is read from the checked-in matrix.
const built = await buildMatrix({ publish: false });
const ids = new Set(models.map((model) => model.id));
const expected = {
  models: built.models
    .filter((model) => ids.has(model.id))
    .map((model) => ({ id: model.id, backends: model.backends, axes: model.axes })),
  coverage: built.coverage
    .filter((row) => ids.has(row.modelId))
    .map(({ modelId, backend, rung, coordinates, states }) => ({
      modelId, backend, rung, coordinates, states,
    })),
};
console.log(JSON.stringify({ models, expected }));
"""


def fold_implementation_states(states):
    folded = {"implemented": 0, "structurally-na": 0, "missing": 0}
    for state, count in states.items():
        folded[implementation_axis(state)] += count
    return folded


def moss_source_facts():
    """The MOSS image lane as the SOURCES declare it: the builtin manifest's `moss_image` family
    (read through the repo's own JSONC helper), the routing catalog's `ModelCaps` rows, and the
    runtime-dumped generator capabilities. Nothing here reads the generated matrix, so the matrix
    cannot vouch for itself."""
    out = subprocess.run(
        ["node", "--input-type=module", "-e", _MOSS_FACTS_SCRIPT],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    payload = json.loads(out)
    manifest_models = payload["models"]
    catalog = (
        ROOT / "crates/sceneworks-core/src/jobs_store/routing/catalog.rs"
    ).read_text(encoding="utf-8")
    dumps = {
        backend: json.loads(
            (ROOT / f"config/engine-capabilities/runtime/capabilities.{backend}.json").read_text(
                encoding="utf-8"
            )
        )["snapshot"]["generator_capabilities"]
        for backend in ("mlx", "candle")
    }
    facts = {}
    for entry in manifest_models:
        row = re.search(
            r'ModelCaps::new\(\s*"%s",\s*(true|false),\s*(true|false),\s*(true|false),'
            r"\s*(true|false),\s*(true|false)\s*\)" % re.escape(entry["id"]),
            catalog,
        )
        assert row, f"{entry['id']} has no routing-catalog ModelCaps row"
        mlx_routed, candle_routed, _quant, candle_lora, _both = (v == "true" for v in row.groups())
        runtime = {
            backend: next((c for c in dumps[backend] if c["id"] == entry["id"]), None)
            for backend in ("mlx", "candle")
        }
        routed = {"mlx": mlx_routed, "candle": candle_routed}
        backends = [b for b in ("mlx", "candle") if routed[b]]
        # The independent declarations of "this backend serves the model" must agree.
        assert backends == entry["backends"], (entry["id"], backends, entry["backends"])
        assert all(runtime[b] is not None for b in backends), (entry["id"], runtime)
        # Refusal coherence: the advertised adapter bit must equal the native descriptor's bit.
        for backend in backends:
            assert runtime[backend]["supports_lora"] == runtime[backend]["supports_lokr"], entry["id"]
        assert candle_lora == runtime["candle"]["supports_lora"], entry["id"]
        # An ADVERTISED adapter family (manifest `loraCompatibility.families`) must be backed by
        # BOTH native LoRA and LoKr on every routed backend plus the Candle routing bit. A false
        # source bit is refused here, so it can never be paired with a matrix that dropped the
        # overlay to hide the advertisement.
        advertised = bool(entry["loraFamilies"])
        if advertised:
            assert candle_routed and candle_lora, (entry["id"], "advertised but not Candle-routed")
            for backend in backends:
                assert runtime[backend]["supports_lora"], (entry["id"], backend)
                assert runtime[backend]["supports_lokr"], (entry["id"], backend)
        facts[entry["id"]] = {
            "advertised": advertised,
            "modes": entry["capabilities"],
            "backends": {b: {"lora": runtime[b]["supports_lora"]} for b in backends},
        }
    return {"models": facts, "expected": payload["expected"]}


def assert_moss_census_shape(matrix, facts):
    """Source-derived SHAPE of the MOSS image lane in the generated matrix. Pins no count: the
    coordinate total per row is recomputed from the row's own axes, and the axes are checked
    against the source facts."""
    expected = facts["expected"]
    facts = facts["models"]
    assert facts, "no MOSS image models derived from the manifest"
    models = {}
    for model in matrix["models"]:
        if model["family"] == "moss_image" or model["id"] in facts:
            assert model["id"] not in models, f"duplicate model {model['id']}"
            models[model["id"]] = model
    assert models.keys() == facts.keys(), (sorted(models), sorted(facts))
    valid_states = {state["state"] for state in matrix["conformanceStates"]}
    rows = {}
    for row in matrix["coverage"]:
        if row["modelId"] in facts:
            key = (row["modelId"], row["backend"], row["rung"])
            assert key not in rows, f"duplicate coverage row {key}"
            rows[key] = row
    for model_id, fact in facts.items():
        model = models[model_id]
        assert sorted(model["backends"]) == sorted(fact["backends"]), model_id
        assert model["axes"].keys() == fact["backends"].keys(), model_id
        for backend, backend_fact in fact["backends"].items():
            axes = model["axes"][backend]
            assert axes["modes"] == fact["modes"], (model_id, backend)
            expected_overlays = {"none", "lora"} if backend_fact["lora"] else {"none"}
            if fact["advertised"]:
                assert backend_fact["lora"], (model_id, backend, "advertised but source-false")
                assert "lora" in axes["overlays"], (model_id, backend, "advertised lora overlay missing")
            built_model = next(x for x in expected["models"] if x["id"] == model_id)
            built_axes = built_model["axes"][backend]
            assert sorted(built_model["backends"]) == sorted(fact["backends"]), model_id
            # Compared to the CURRENT builder output, not to a pin: an invented tier, mode or
            # overlay fails, and a legitimate source change recomputes the expectation.
            for axis in ("tiers", "modes", "rungs"):
                assert axes[axis] == built_axes[axis], (model_id, backend, axis)
            assert set(axes["overlays"]) == set(built_axes["overlays"]), (model_id, backend)
            assert len(axes["overlays"]) == len(set(axes["overlays"])), (model_id, backend)
            assert set(axes["overlays"]) == expected_overlays, (model_id, backend, axes["overlays"])
            assert axes["tiers"] and len(axes["tiers"]) == len(set(axes["tiers"])), (model_id, backend)
            assert tuple(axes["rungs"]) == STRATEGY_RUNGS, (model_id, backend)
            coordinates = len(axes["tiers"]) * len(axes["modes"]) * len(axes["overlays"])
            for rung in STRATEGY_RUNGS:
                where = (model_id, backend, rung)
                row = rows.pop(where, None)
                assert row is not None, f"missing coverage row {where}"
                assert row["coordinates"] == coordinates, where
                assert row["published"] + row["elided"] == coordinates, where
                assert row["states"] and set(row["states"]) <= valid_states, where
                assert all(
                    isinstance(n, int) and not isinstance(n, bool) and n > 0
                    for n in row["states"].values()
                ), where
                assert sum(row["states"].values()) == coordinates, where
                tally = sum(
                    n
                    for state, n in row["states"].items()
                    if implementation_axis(state) == "implemented"
                )
                assert row["implemented"] == tally, where
                built_row = next(
                    r
                    for r in expected["coverage"]
                    if (r["modelId"], r["backend"], r["rung"]) == where
                )
                assert row["coordinates"] == built_row["coordinates"], where
                assert fold_implementation_states(row["states"]) == fold_implementation_states(
                    built_row["states"]
                ), where
                if rung == "resident":
                    # A routed backend implements the plain resident rung on every coordinate.
                    assert tally == coordinates, where
    assert not rows, f"coverage rows for undeclared MOSS lanes: {sorted(rows)}"


def test_moss_image_census_shape_is_derived_from_the_sources():
    assert_moss_census_shape(load_matrix(), moss_source_facts())


def test_moss_image_census_shape_rejects_in_memory_mutations():
    facts = moss_source_facts()
    matrix = load_matrix()
    assert_moss_census_shape(matrix, facts)  # baseline GREEN
    model_id = sorted(facts["models"])[0]
    backend = sorted(facts["models"][model_id]["backends"])[0]

    def lane_rows(m):
        return [r for r in m["coverage"] if r["modelId"] == model_id and r["backend"] == backend]

    def model_of(m):
        return next(x for x in m["models"] if x["id"] == model_id)

    def missing_moss(m):
        m["models"] = [x for x in m["models"] if x["family"] != "moss_image"]
        m["coverage"] = [r for r in m["coverage"] if r["modelId"] not in facts["models"]]

    def dropped_model(m):
        m["models"] = [x for x in m["models"] if x["id"] != model_id]

    def drop_overlay(m):
        # A self-consistent overlay deletion: axes and every row shrink together.
        model_of(m)["axes"][backend]["overlays"] = ["none"]
        for r in lane_rows(m):
            r["coordinates"] //= 2
            r["elided"] = r["coordinates"] - r["published"]
            r["states"] = {k: v // 2 for k, v in r["states"].items()}
            r["implemented"] //= 2

    def unsupported_overlay(m):
        model_of(m)["axes"][backend]["overlays"] = ["none", "lora", "lokr"]

    def missing_row(m):
        m["coverage"].remove(lane_rows(m)[0])

    def duplicate_row(m):
        m["coverage"].append(copy.deepcopy(lane_rows(m)[0]))

    def duplicate_model(m):
        m["models"].append(copy.deepcopy(model_of(m)))

    def bad_state(m):
        lane_rows(m)[0]["states"] = {"Bogus": lane_rows(m)[0]["coordinates"]}

    def bad_count(m):
        lane_rows(m)[0]["states"] = {"Missing": 0}

    def bad_sum(m):
        lane_rows(m)[0]["coordinates"] += 1

    def bad_implemented(m):
        lane_rows(m)[0]["implemented"] += 1

    def resident_not_implemented(m):
        row = next(r for r in lane_rows(m) if r["rung"] == "resident")
        row["states"] = {"Missing": row["coordinates"]}
        row["implemented"] = 0

    def bad_axes(m):
        model_of(m)["axes"][backend]["rungs"] = list(STRATEGY_RUNGS[:-1])

    def invented_tier(m):
        # Self-consistent: the axis and every row of the lane grow together.
        model_of(m)["axes"][backend]["tiers"] = ["default", "q4"]
        for r in lane_rows(m):
            r["coordinates"] *= 2
            r["elided"] = r["coordinates"] - r["published"]
            r["states"] = {k: v * 2 for k, v in r["states"].items()}
            r["implemented"] *= 2

    def fake_higher_rung_implemented(m):
        # Self-consistent: states, implemented and the partition all agree on a rung the sources
        # do not implement.
        row = next(r for r in lane_rows(m) if r["rung"] == "bounded_decode")
        row["states"] = {"Implemented": row["coordinates"]}
        row["implemented"] = row["coordinates"]

    def stray_lane(m):
        row = copy.deepcopy(lane_rows(m)[0])
        row["backend"] = "bogus"
        m["coverage"].append(row)

    mutations = {
        "missing-moss": missing_moss,
        "dropped-model": dropped_model,
        "drop-overlay": drop_overlay,
        "unsupported-overlay": unsupported_overlay,
        "missing-row": missing_row,
        "duplicate-row": duplicate_row,
        "duplicate-model": duplicate_model,
        "invalid-state": bad_state,
        "invalid-count": bad_count,
        "invalid-sum": bad_sum,
        "invalid-implemented": bad_implemented,
        "resident-not-implemented": resident_not_implemented,
        "invalid-axes": bad_axes,
        "stray-lane": stray_lane,
        "invented-tier": invented_tier,
        "fake-higher-rung-implemented": fake_higher_rung_implemented,
    }

    def failed_check(m, f):
        """The source text of the assertion that rejected `m` (and that it was one)."""
        try:
            assert_moss_census_shape(m, f)
        except AssertionError as error:
            return traceback.extract_tb(error.__traceback__)[-1].line
        raise AssertionError("mutant was accepted")

    intended = {
        "missing-moss": "models.keys() == facts.keys()",
        "dropped-model": "models.keys() == facts.keys()",
        "drop-overlay": "advertised lora overlay missing",
        "unsupported-overlay": 'set(axes["overlays"]) == set(built_axes["overlays"])',
        "missing-row": "row is not None",
        "duplicate-row": "key not in rows",
        "duplicate-model": 'model["id"] not in models',
        "invalid-state": 'set(row["states"]) <= valid_states',
        "invalid-count": "assert all(",
        "invalid-sum": 'row["coordinates"] == coordinates',
        "invalid-implemented": 'row["implemented"] == tally',
        "resident-not-implemented": "fold_implementation_states(row",
        "invalid-axes": "axes[axis] == built_axes[axis]",
        "stray-lane": "not rows",
        "invented-tier": "axes[axis] == built_axes[axis]",
        "fake-higher-rung-implemented": "fold_implementation_states(row",
    }
    for name, mutate in mutations.items():
        mutated = copy.deepcopy(matrix)
        mutate(mutated)
        line = failed_check(mutated, facts)
        assert intended[name] in line, (name, line)
    assert "assert facts" in failed_check(matrix, {"models": {}, "expected": facts["expected"]})
    # A false adapter bit in the sources is refused, not absorbed into a smaller expected shape,
    # and not hidden by a matrix that dropped the overlay to agree with it (manifest advertisement
    # intact).
    smaller = copy.deepcopy(facts)
    smaller["models"][model_id]["backends"][backend]["lora"] = False
    assert smaller["models"][model_id]["advertised"]
    assert "advertised but source-false" in failed_check(matrix, smaller)
    hidden = copy.deepcopy(matrix)
    drop_overlay(hidden)
    assert "advertised but source-false" in failed_check(hidden, smaller)
    assert_moss_census_shape(matrix, facts)  # baseline restored GREEN


def assert_implementation_census_shape(matrix, pinned):
    """Preserve each lane's implementation partition, independently of axis cardinality.

    The historical census fixture supplies the expected state proportions, not a population
    ceiling. Adding supported catalog tiers may scale a lane without changing its classification.
    Declared axis products still require every coordinate, and fixture/declaration key equality
    still rejects missing lanes. Anchors and currency cannot change this implementation projection.
    Why it exists: the collapse dropped the rung-4 applicability survey out of the generator's read
    set, and that silently moved the implementation axis on nine lanes — a real MLX rung-4 ladder
    became Missing on Bernini and Krea, six real structural exemptions on Krea's Candle lane became
    Missing, and four Candle lanes newly claimed Implemented. None of it was visible in a review of
    the generator diff. This census is derived from `coverage[].states`, which counts EVERY resolved
    coordinate (published and elided), so an implementation-axis move cannot hide behind elision.

    What it pins is a claim about CODE and CATALOG — does this route implement this rung — read off
    the provider contracts, the manifest declarations and the routing tables. It deliberately pins
    no measurement, no anchor and no currency: `implementation_axis` folds the three implemented
    states together precisely so anchor movement cannot red it.

    Changing a count is legitimate; changing it SILENTLY is not. Update the fixture in the same
    commit that changes the declaration, and say in the commit body which lanes moved and why.

    Last moved: sc-24214 (epic 24212) routes `supra2_img` to Candle under the same engine id. It
    adds exactly five NEW lanes (`supra2_img:candle:*`) and moves nothing else; the profile is the
    MLX one: `resident` [2,0,0] (`candle-gen-supra2` is resident-only and the entry now declares a
    `candle` floor), every other rung [0,0,2] — the provider publishes a resident-only contract, so
    no staged residency, bounded decode, attention or transformer residency.

    Previously: sc-24213 (epic 24212) adds the MLX-only `supra2_img` catalog entry (one untagged
    artifact, so a single tier; 2 coordinates per lane). It contributes exactly five NEW lanes and
    moves nothing else: `resident` [2,0,0] (the engine loads resident and eager); every other rung
    [0,0,2] — its descriptor advertises no staged residency, and the provider implements no bounded
    decode, attention or transformer residency.

    Previously: sc-25907 (epic 25899) adds the `qwen_image_2_1_turbo` catalog entry, the distilled
    sibling of `qwen_image_2_1`, routed on BOTH lanes. It adds exactly ten NEW lanes
    (`qwen_image_2_1_turbo:<backend>:<rung>`, 18 coordinates each = 3 tiers x 3 catalog modes x
    {none, lora}) and moves NOTHING else. Each is base's lane profile verbatim — including
    `mlx:staged_residency` [18,0,0], because the same commit adds the Turbo id to the worker's
    `mlx_fit_gate.rs#engine_engages_staged_residency_is_derived_from_the_registered_capability`
    sweep (the generator's MLX staged-residency authority). This is an IMPLEMENTATION claim only:
    no anchor and no measurement is involved, and every Turbo cell is analytic (E10) — base's
    `mlx` lanes read Anchored where Turbo's read Implemented, and both fold to the same axis here.

    Previously: the sc-25680 review round gives `iris_3b` a `candle` block (declared
    `minMemoryGb` + `sequentialPeakGb` floors, so the Candle fit gate has a prediction). Exactly two
    lanes move, each [0,0,2] -> [2,0,0]: `iris_3b:candle:resident` (a lane with a per-backend block
    is no longer the generator's "routed but untriaged" all-Missing slice) and
    `iris_3b:candle:staged_residency` (the block declares a sequential peak, and `candle-gen-iris`
    does advertise sequential offload). The three bounded rungs stay [0,0,2] (no ladder at this
    pin) and nothing else moves.

    Previously: sc-25680 routes `iris_3b` to Candle/CUDA. It adds exactly five NEW lanes
    (`iris_3b:candle:*`), each [0,0,2]: the catalog entry carries no `candle` block yet, so the
    whole Candle slice is the generator's "routed but untriaged" Missing (its memory declaration is
    sc-25684's, S6), and nothing else moves.

    Previously: sc-25679 adds the MLX-only `iris_3b` catalog entry (one untagged artifact, so a
    single tier; text_to_image x {none, lora} = 2 coordinates per lane). It contributes exactly
    five NEW lanes and moves nothing else: `resident` [2,0,0]; `staged_residency` [2,0,0] because
    the descriptor advertises sequential offload and the engine joined the fit gate's
    staged-residency sweep; the three bounded rungs [0,0,2] because the provider publishes no
    memory-strategy ladder at this pin.

    Previously: sc-24163, epic 24107's terminal pin bump (inference 457f1a37), where both 2.1
    providers apply LoRA/LoKr. The five hand-authored `qwen_image_2_1` contract rows gain the
    `lora` overlay and load profile, and the worker's memory-route rows move PLAIN -> PLAIN_LORA
    with the re-dump. Three `qwen_image_2_1` lanes move and nothing else — the `lora` half of each
    declared lane becomes implemented, 9 -> 18 of 18 (3 tiers x 3 catalog modes x {none, lora}):
    * `candle:staged_residency` and `candle:bounded_decode` [9,0,9] -> [18,0,0].
    * `mlx:bounded_decode` [9,0,9] -> [18,0,0]. `mlx:staged_residency` was already 18 (the fit
      gate's staged sweep) and `resident` already 18 on both lanes.

    Previously: sc-24114 (the feature-end review round) gives `qwen_image_2_1` a HAND-AUTHORED,
    request-owned `memoryStrategyContract` on BOTH lanes whose `modes` cover the reference faces
    (`edit_image` / `image_to_image`, plus `character_image` which the catalog axis does not
    carry) as well as `text_to_image`, all over `none`. Three `qwen_image_2_1` lanes move and
    nothing else — each to 9 implemented of 18 (3 tiers x 3 catalog modes x `none`; the `lora`
    half stays missing because the provider refuses adapters):
    * `candle:staged_residency` and `candle:bounded_decode` 3 -> 9 — the declaration now covers
      the reference modes, where the generated block covered `text_to_image` alone.
    * `mlx:bounded_decode` 0 -> 9 — the MLX block now declares the rung. `mlx:staged_residency`
      does NOT move: it stays the fit gate's staged sweep (18), and the MLX block deliberately
      declares no staged row (an MLX staged row forces every load onto `Sequential`).

    Previously: sc-24112's integration on the epic's terminal pin (inference b12c632b4), where
    the 2.1 providers register on both lanes. Three `qwen_image_2_1` lanes move and nothing else:
    * `mlx:staged_residency` 0 -> 18 implemented — the engine now registers, its descriptor
      advertises sequential offload, and it joined the fit gate's staged-residency sweep (the
      generator's MLX authority), so every coordinate of the lane is implemented.
    * `candle:staged_residency` and `candle:bounded_decode` 0 -> 3 implemented (15 still missing) —
      the Candle block gained the `memoryStrategyContract` generated from the re-dumped
      `capabilities.candle.json`, which declares exactly `text_to_image` x `none` x {bf16,q8,q4}.
    The MLX block declares no contract (a non-legacy MLX lane needs hand-authored
    `requestContexts`), so no MLX rung beyond staged residency moves.

    Previously: sc-24112 gives `qwen_image_2_1` a real TIER AXIS — bf16/q8/q4 `variant` rows on
    both backends, where sc-24108/sc-24109 had one untagged artifact. `tiersFor` unions the
    declared download variants, so the entry's tier axis goes 1 -> 3 and every one of its ten lanes
    multiplies by 3.

    The same branch carries sc-24113/sc-24110, which widen the entry's `capabilities` from
    `["text_to_image"]` to `["text_to_image", "image_to_image", "edit_image"]`, so the MODE axis
    goes 2 -> 3 on top of that. The two multipliers compose: 4 -> 12 (tiers) -> 18 (modes) per
    lane, i.e. `resident` 4 -> 18 implemented and each of the other four rungs 4 -> 18 missing, on
    each backend. 18 = 3 tiers x 3 modes x {none, lora}.

    No other model is touched and no rung's CLASSIFICATION changes: the ladder is still unpublished
    on both lanes until the epic's terminal pin bump re-dumps the capability files, so the new
    coordinates are Missing for exactly the reason the old ones were. The per-tier memory floors
    ride `minMemoryGbByTier`, which the matrix does not read, so they move nothing here; nor does
    the `admissionGeometry` block.

    Previously: sc-24109 adds the native Candle/CUDA lane for `qwen_image_2_1` (routing-catalog
    `candle_routed: true` + the manifest `candle` block), so the entry contributes five NEW lanes
    (`qwen_image_2_1:candle:<rung>`) and moves NOTHING else — no existing lane gained, lost or
    moved a coordinate. The Candle profile is identical to the MLX one recorded below, and for the
    same two reasons: one bf16 artifact and no `variant`, so the tier axis is a single tier; and no
    memory-strategy ladder published at this pin (the Candle contract block is generated from
    `capabilities.candle.json`, which cannot be re-dumped until the epic's terminal pin bump), so
    only `resident` is implemented (2 coordinates: one tier x text_to_image x {none, lora}) and the
    other four rungs are Missing.

    Before that: sc-24108 adds the `qwen_image_2_1` catalog entry, which is MLX-only and ships one
    bf16 artifact with no `variant`, so it contributes exactly five NEW lanes
    (`qwen_image_2_1:mlx:<rung>`) and moves nothing else. Its provider publishes no
    memory-strategy ladder at this pin, so only `resident` is implemented (2 coordinates: the
    single tier x text_to_image x {none, lora} overlay pair) and the other four rungs are Missing.
    No existing lane gained, lost or moved a coordinate.

    Previously: sc-23648 restores missing MLX streaming declarations: SDXL q4/q8 add
    five supported mode/overlay coordinates per tier (implemented 3 -> 13), and
    RealVisXL Lightning bf16/q8 add two per tier (implemented 2 -> 6). These are
    provider-validated declarations, with no new measurements or other census moves.

    Previously: sc-22738, on top of `6968692a3` removing the candle-only short-circuit in
    `tiersFor` (`backend === "candle" && backendTiers.length ? backendTiers : [...]`). That
    short-circuit read `vramGbByTier` — a MEASUREMENT block — as a routing ceiling, so the moment a
    candle lane declared any peak the axis collapsed to those keys alone and `downloadTiers` /
    `inferred` were discarded. Six route-declared, download-shipped coordinates came back onto the
    published axis, each widening its lane's tier axis by exactly one tier:

      * `flux_dev`, `flux_schnell`, `flux2_dev` at bf16 (candle) — `["q4","q8"]` -> `["bf16","q4","q8"]`
      * `sd3_5_large`, `sd3_5_large_turbo`, `sd3_5_medium` at q8 (candle) — `["bf16","q4"]` -> `["bf16","q4","q8"]`

    That is 6 lanes x 5 rungs = 30 fixture entries, and NOTHING else: no lane gained or lost a key,
    no seventh lane moved, and every moved entry is exactly one more copy of that lane's own
    pre-existing per-tier profile (so no rung flipped state for a tier that was already on the
    axis). Matrix totals moved with it: cells 9265 -> 9390, publishedCells 2178 -> 2216,
    anchoredCells 1935 -> 1973 (the 38 newly published cells are the flux_dev/flux_schnell candle
    bf16 anchors binding).

    sc-24113 moves exactly the ten `qwen_image_2_1` lanes (5 rungs x 2 backends), each DOUBLING:
    declaring `image_to_image` on that model gives it a second mode, and a mode is a coordinate
    axis. `[0,0,2]` -> `[0,0,4]` on every rung but `resident`, and `[2,0,0]` -> `[4,0,0]` there.
    No other lane moved, and no rung changed STATE — the model gained coordinates, not an
    implementation. That is the shape to check when re-pinning: a lane outside qwen_image_2_1
    moving here would be a real regression, not a re-pin.
    """
    # MOSS advertises feature-owned adapter axes; its separate source-derived
    # census above checks every coordinate rather than pinning old refusal counts.
    moss_ids = set(moss_source_facts()["models"])
    assert moss_ids
    pinned = {key: value for key, value in pinned.items() if key.split(":")[0] not in moss_ids}
    axes_by_lane = {
        (model["id"], backend): axes
        for model in matrix["models"]
        if model["id"] not in moss_ids
        for backend, axes in model["axes"].items()
    }
    declared_keys = {
        f"{model_id}:{backend}:{rung}"
        for (model_id, backend), axes in axes_by_lane.items()
        for rung in axes["rungs"]
    }
    census = {}
    for row in matrix["coverage"]:
        if row["modelId"] in moss_ids:
            continue
        tally = {"implemented": 0, "structurally-na": 0, "missing": 0}
        for state, count in row["states"].items():
            assert isinstance(count, int) and count >= 0, row
            tally[implementation_axis(state)] += count
        key = f"{row['modelId']}:{row['backend']}:{row['rung']}"
        assert key not in census, key
        counts = [tally["implemented"], tally["structurally-na"], tally["missing"]]
        census[key] = counts
        axes = axes_by_lane[(row["modelId"], row["backend"])]
        coordinates = len(axes["tiers"]) * len(axes["modes"]) * len(axes["overlays"])
        assert coordinates > 0, key
        assert sum(counts) == row["coordinates"] == coordinates, key
        assert counts[0] == row["implemented"], key

    assert census.keys() == declared_keys == pinned.keys(), {
        "undeclared lanes": sorted(census.keys() - declared_keys),
        "missing declared lanes": sorted(declared_keys - census.keys()),
        "unpinned lanes": sorted(census.keys() - pinned.keys()),
        "vanished lanes": sorted(pinned.keys() - census.keys()),
    }
    for key, counts in census.items():
        expected = pinned[key]
        assert len(expected) == 3 and sum(expected) > 0, key
        # Exact rational comparison: preserve classification, not a pinned tier population.
        assert [count * sum(expected) for count in counts] == [
            count * sum(counts) for count in expected
        ], {"lane": key, "expected partition": expected, "actual partition": counts}


def implementation_census_fixture():
    return json.loads(
        (ROOT / "tests/fixtures/memory-matrix-implementation-census.json").read_text(
            encoding="utf-8"
        )
    )


def test_the_implementation_axis_census_preserves_per_lane_state_partition():
    assert_implementation_census_shape(load_matrix(), implementation_census_fixture())


@pytest.mark.parametrize("mutation", ["drop_lane", "drop_tier", "change_classification"])
def test_implementation_census_rejects_lost_lane_tier_and_classification(mutation):
    matrix = load_matrix()
    if mutation == "drop_lane":
        matrix["coverage"].pop()
    elif mutation == "drop_tier":
        axes = next(
            axes
            for model in matrix["models"]
            for axes in model["axes"].values()
            if len(axes["tiers"]) > 1
        )
        axes["tiers"].pop()
    else:
        row = next(row for row in matrix["coverage"] if row["states"].get("Implemented", 0))
        row["states"]["Implemented"] -= 1
        row["states"]["Missing"] = row["states"].get("Missing", 0) + 1
        row["implemented"] -= 1
    with pytest.raises(AssertionError):
        assert_implementation_census_shape(matrix, implementation_census_fixture())
