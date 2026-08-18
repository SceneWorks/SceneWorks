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


def load_matrix():
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def matrix_validator():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    calibration_schema = json.loads(CALIBRATION_SCHEMA.read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        calibration_schema["$id"], Resource.from_contents(calibration_schema)
    )
    return jsonschema.Draft202012Validator(schema, registry=registry)


def test_generated_memory_matrix_is_current_and_schema_valid():
    subprocess.run(
        ["node", "scripts/generate-memory-matrix.mjs", "--check"],
        cwd=ROOT,
        check=True,
    )
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
    assert "flux2_dev" not in mlx_staged, (
        "SC-18218: the pinned MLX FLUX.2 provider is Resident-only and owes no staged-route claim"
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
    evidence_by_id = {record["id"]: record for record in calibration["records"]}
    records_by_status = Counter(record["status"] for record in calibration["records"])
    # The bundle carries exactly two statuses. A third would slip straight past every partition below,
    # which is the defect this owns — not the size of either population. Those sizes have only ever
    # grown (complete 33 -> 50 -> 52 -> 65 -> 70 across sc-16915 / SC-18237 / SC-18353 / SC-19753;
    # runtime_complete 15 -> 19 at SC-18218), and renewing the pair each time re-froze the corpus
    # without ever asserting a property. Both populations must be non-empty and must partition the
    # bundle exactly; everything downstream is then derived from `records_by_status` rather than from a
    # second transcription of it, so the matrix and the bundle cannot disagree.
    assert set(records_by_status) == {"complete", "runtime_complete"}
    assert all(count > 0 for count in records_by_status.values())
    assert len(evidence_ids) == len(calibration["records"]) == sum(
        records_by_status.values()
    )
    assert {run["record"]["id"] for run in matrix["calibrationRuns"]} == evidence_ids
    runs_by_status = Counter(run["record"]["status"] for run in matrix["calibrationRuns"])
    assert runs_by_status == records_by_status
    assert matrix["summary"]["calibrationRuns"] == sum(runs_by_status.values())
    assert matrix["summary"]["calibrationRunsByStatus"] == {
        "complete": records_by_status["complete"],
        "runtimeComplete": records_by_status["runtime_complete"],
    }

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
    # sc-16915 measured seventeen Full-complete runs at the then-live closure; SC-18237 and SC-18353
    # later added fifteen Qwen records, and SC-19753 five Z-Image records, each at the closure live
    # when it ran. The epic's pin advance moved past all of them, so every Full-complete run is now
    # historical — an accepted floor, not a re-capture work order.
    #
    # Still pinned as an exact set AND an exact count, the same way it was when runs were current: a
    # bare `<= {"current", "historical"}` would accept any mixture, and a count alone would let one
    # family's promotion mask another's demotion. Holding the current count at exactly 0 is what
    # makes a record silently surviving the closure change fail here.
    assert {run["semantics"] for run in full_runs} == {"historical"}
    assert sum(1 for run in full_runs if run["semantics"] == "current") == 0
    expected_candle_flux2_runtime = {
        "imc-998b89c5d76dbcc84332": "bounded_attention",
        "imc-b4113eedf503e409ad1b": "resident",
        "imc-b62adbfca64f277414e1": "bounded_decode",
        "imc-bfb890dff959eaf09183": "staged_residency",
        "imc-f5c3d06f30ebf3723f13": "bounded_transformer_residency",
    }
    expected_mlx_flux2_runtime = {
        "imc-747f54e1be89e30da943": ("q8", 768),
        "imc-9b235419ecbe0710da06": ("q8", 1024),
        "imc-b6537074420d51413b38": ("q4", 1024),
        "imc-f3badcb841c8707fd971": ("q4", 768),
    }
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
    assert {
        run["record"]["id"]: (
            run["record"]["target"]["tier"],
            run["record"]["target"]["geometry"]["width"],
        )
        for run in mlx_flux2_runtime
    } == expected_mlx_flux2_runtime
    assert all(run["semantics"] == "historical" for run in mlx_flux2_runtime)
    assert all(run["binding"]["eligible"] for run in mlx_flux2_runtime)
    live_closures = json.loads(
        (ROOT / "config/inference-provider-closures.json").read_text(encoding="utf-8")
    )["providers"]
    for run in mlx_flux2_runtime:
        record = evidence_by_id[run["record"]["id"]]
        tier, width = expected_mlx_flux2_runtime[record["id"]]
        assert record["repositories"]["inference"]["revision"] == (
            "10831e4ca5b8bf780319a8ee7f21427175075448"
        )
        assert record["repositories"]["inference"]["closureDigest"] == (
            "355749219c38b37af5054df047b0f44b65ecd8f822fc258243eee9d09c1d0247"
        )
        assert record["repositories"]["inference"]["closureDigest"] != live_closures[
            "mlx:flux2_dev"
        ]["digest"], "the retained FLUX.2 capture must not be re-dated to the frozen b4 closure"
        assert record["status"] == "runtime_complete"
        assert record["loadShape"] == "eager_materialization"
        assert record["calibrationFingerprint"] == (
            "sc-18218-flux2-dev-t2i-resident-evidence-v1"
        )
        assert record["target"] == {
            "provider": "flux2_dev",
            "modelId": "flux2_dev",
            "tier": tier,
            "mode": "text_to_image",
            "overlay": "none",
            "geometry": {
                "width": width,
                "height": width,
                "batch": 1,
                "frames": 1,
            },
        }
        assert record["strategy"] == {
            "rung": "resident",
            "parameters": {},
            "engagedRungs": ["resident"],
        }
        assert record["artifact"] == {
            "repository": "SceneWorks/flux2-dev-mlx",
            "resolvedRevision": "2868b1461b2b6e6e05d84e52534df3632b4c7d5d",
            "variant": tier,
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
    # This set is derived from the PIN, which sc-17774 retired as the currency term in favour of the
    # provider's compile closure. The two coincided while nothing had moved; they no longer do.
    #
    # No calibration was captured at the capability-snapshot-only pin introduced by sc-18473.
    # Pin bumps must not re-date physical evidence; provider closure, checked below, determines
    # whether the older captures remain current.
    assert measured_at_live_pin == set()

    # SC-18353 ran thirteen exact physical Qwen bf16/q4 records at 014134e3. Pin the immutable ids and
    # their capture revision so unrelated evidence cannot silently enter this closed capture set;
    # SC-18237's q8 pair remains current by provider closure despite an older capture revision.
    sc_18353_capture_revision = "014134e3035ad7e4eca5c2ed7bded2375dc3c071"
    sc_18353_capture_ids = {
        record["id"]
        for record in calibration["records"]
        if record["repositories"]["inference"]["revision"]
        == sc_18353_capture_revision
    }
    assert sc_18353_capture_ids == {
        "imc-08e925c50d9c290ed53d",
        "imc-0e00924d96eeaf12be17",
        "imc-277c04656961710d29e0",
        "imc-3c1d70abfcccd95ea119",
        "imc-50508c995a8d49b70aa2",
        "imc-5ea462dfe3101260a9b1",
        "imc-8ca170a7a9c901993007",
        "imc-8fce887b31583e05f5b5",
        "imc-91c4f21972905626cbb2",
        "imc-93989adacdb7a35156a7",
        "imc-b0527097758ac66f381e",
        "imc-b072c9b116a6a40d00e1",
        "imc-ea87169a3ea1fd340791",
    }
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
    # The physical bf16 tile edges still characterize the historical sweep, but none matches the
    # current provider closure and therefore none is a current-but-unbound coordinate.
    assert unbound_decode_edges == set()
    # Currency is the provider's COMPILE CLOSURE, not the pin (sc-17774). While nothing had moved the
    # two coincided and this assertion could be written off `measured_at_live_pin`; a pin bump that
    # leaves a provider's closure untouched separates them, which is exactly what the 014134e3 bump
    # does — it moves `mlx-gen-krea` (and, via main, `mlx-gen-z-image`) and leaves `mlx:qwen_image`
    # byte-identical. So derive the expectation from the closure the same way the generator does,
    # rather than from the pin: a record is current when the digest it captured is still the live
    # digest for ITS provider lane. Written this way it keeps falling away on its own the next time a
    # closure genuinely moves, instead of needing a hand-edit per bump.
    current_by_closure = {
        record["id"]
        for record in calibration["records"]
        if live_closures.get(
            f"{record['backend']}:{record['target']['provider']}", {}
        ).get("digest")
        == record["repositories"]["inference"]["closureDigest"]
    }
    assert current_by_closure == set()
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


def test_historical_records_remain_unverified_after_the_provider_contract_advance():
    matrix = load_matrix()
    assert matrix["summary"]["fullModels"] == 0
    # sc-16915 recaptured the Qwen and Krea MLX evidence at its then-current pin, and SC-19753
    # captured the five Z-Image q4 coordinates at the closure live when it ran. The epic's pin has
    # since advanced past all of them, so every shipped capture is now an ACCEPTED FLOOR rather than
    # current verification — a pin bump staling calibration records is the fail-closed design
    # working, not a re-capture work order.
    #
    # Still stated as the exact SET rather than a count: a count would let one model's promotion
    # silently cover another's regression, and an exact empty set still fails the moment any record
    # survives the closure change as current.
    verified = {
        (cell["modelId"], cell["backend"], cell["tier"], cell["rung"])
        for cell in matrix["cells"]
        if cell["state"] == "Verified"
    }
    # No family may remain Verified merely because it was current at an older pin.
    assert verified == set()
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
    assert all(cell["state"] == "Implemented/unverified" for cell in krea_cells)
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
    # Where it IS implemented it is on exactly the (bf16, overlay-none) slice, so the count is one
    # per MODE — derived from the published axes rather than pinned, because a hardcoded number would
    # go stale on any legitimate mode or tier drift and would stop meaning "bf16/none only". The
    # backend that implements it is not asserted here; that is the sibling JS test's subject.
    implementing = [row for row in sdxl_lane if row["implemented"]]
    assert implementing, "SDXL's rung-4 coverage must not vanish"
    for row in sdxl_lane:
        axes = models_by_id["sdxl"]["axes"][row["backend"]]
        assert "bf16" in axes["tiers"] and "none" in axes["overlays"]
        assert row["coordinates"] == len(axes["tiers"]) * len(axes["modes"]) * len(
            axes["overlays"]
        )
        assert row["implemented"] in (0, len(axes["modes"])), (
            "SDXL publishes rung 4 on bf16/overlay-none only, across every mode, or not at all"
        )
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
