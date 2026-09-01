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

def test_collapsed_cell_state_is_a_pure_function_of_the_published_facts():
    """sc-22513 (epic 22505, E5): `state = f(implementation, anchor present, derivation defined)`.

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

    def expected(implementation, anchored, derivation_defined):
        if implementation == "missing":
            return "Missing"
        if implementation == "structurally-na":
            return "Structurally N/A"
        assert implementation == "implemented"
        if not anchored:
            return "Implemented"
        return "Anchored" if derivation_defined else "Anchored/underived"

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
    """sc-22511: currency is a REPORT. A staled loader closure means the anchor needs re-extraction,
    not that the rung stopped existing — so `anchor.current` may not partition the states."""
    matrix = load_matrix()
    anchored = [cell for cell in matrix["cells"] if cell["anchor"] is not None]
    assert anchored
    by_current: dict[bool, set[str]] = {}
    for cell in anchored:
        by_current.setdefault(cell["anchor"]["current"], set()).add(cell["state"])
    # Both currency values must be represented among the anchored cells for this to have teeth; the
    # shipped store carries stale and current anchors alike.
    assert set(by_current) == {True, False}, sorted(by_current)
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
        else:
            expected = "Anchored" if cell["derivationDefined"] else "Anchored/underived"
        assert cell["state"] == expected, cell["id"]
    # Teeth: the anchored population must span more than one state, or the loop above is vacuous.
    assert len({cell["state"] for cell in anchored}) > 1
    # A cell's reported currency is its inventory row's, not an independent claim: two copies of one
    # fact that can disagree is how a stale anchor reads as current on the cell that cites it.
    inventory = {row["id"]: row for row in matrix["anchors"]}
    for cell in anchored:
        assert cell["anchor"]["current"] == inventory[cell["anchor"]["id"]]["current"], cell["id"]
    # And the reported staleness is the store's own, recomputed from the closure ledger.
    closures = json.loads(
        (ROOT / "config/anchor-loader-closures.json").read_text(encoding="utf-8")
    )["models"]
    store = json.loads((ROOT / "config/memory-anchors.json").read_text(encoding="utf-8"))
    by_id = {anchor["id"]: anchor for anchor in store["anchors"]}
    for row in matrix["anchors"]:
        anchor = by_id[row["id"]]
        assert row["current"] == (
            closures.get(f"{anchor['modelId']}:{anchor['backend']}", {}).get("digest")
            == anchor["source"]["loaderClosureDigest"]
        ), row["id"]
    assert matrix["summary"]["staleAnchors"] == sum(
        1 for row in matrix["anchors"] if not row["current"]
    )


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


def test_the_implementation_axis_census_is_pinned_per_model_backend_rung():
    """sc-22513: the per-(modelId, backend, rung) census of implemented / structurally-N/A / missing
    coordinates, pinned against a fixture NO script regenerates.

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
    """
    matrix = load_matrix()
    census = {}
    for row in matrix["coverage"]:
        tally = {"implemented": 0, "structurally-na": 0, "missing": 0}
        for state, count in row["states"].items():
            tally[implementation_axis(state)] += count
        key = f"{row['modelId']}:{row['backend']}:{row['rung']}"
        assert key not in census, key
        census[key] = [tally["implemented"], tally["structurally-na"], tally["missing"]]

    pinned = json.loads(
        (ROOT / "tests/fixtures/memory-matrix-implementation-census.json").read_text(
            encoding="utf-8"
        )
    )
    assert census.keys() == pinned.keys(), {
        "unpinned lanes": sorted(census.keys() - pinned.keys()),
        "vanished lanes": sorted(pinned.keys() - census.keys()),
    }
    moved = {
        key: {"pinned": pinned[key], "generated": census[key]}
        for key in sorted(census)
        if census[key] != pinned[key]
    }
    assert not moved, moved
    # The census is a partition of the resolved coordinates, so it also has to add up.
    for row in matrix["coverage"]:
        key = f"{row['modelId']}:{row['backend']}:{row['rung']}"
        assert sum(pinned[key]) == row["coordinates"], key
        assert pinned[key][0] == row["implemented"], key
