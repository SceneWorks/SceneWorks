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
    assert matrix["summary"]["imageModels"] == 53
    assert matrix["summary"]["mlxStagedStaticCoverage"] == 39
    assert matrix["summary"]["mlxStagedStaticCoverageDenominator"] == 53
    assert len(matrix["models"]) == len(matrix["modelSlices"]) == 53
    assert {model["id"] for model in matrix["models"]} == set(matrix["modelSlices"])


def test_aliases_bespoke_routes_and_evidence_dimensions_are_explicit():
    matrix = load_matrix()
    models = {model["id"]: model for model in matrix["models"]}
    assert models["z_image_edit"]["resolvedRoute"] == "z_image_turbo"
    assert models["instantid_realvisxl"]["routeKind"] == "bespoke"
    assert models["pulid_flux_dev"]["routeKind"] == "bespoke"
    instantid_candle_tiers = {
        cell["tier"]
        for cell in matrix["cells"]
        if cell["modelId"] == "instantid_realvisxl" and cell["backend"] == "candle"
    }
    assert instantid_candle_tiers == {"bf16"}
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
    records_by_status = Counter(record["status"] for record in calibration["records"])
    # sc-16915 added seventeen Full-complete MLX records (qwen_image x15,
    # krea_2_turbo_control x2) measured at the current pin: complete 33 -> 50. The
    # base-only runtime-complete FLUX population is untouched.
    assert records_by_status == {"complete": 50, "runtime_complete": 15}
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
    # sc-16915 measured seventeen Full-complete runs AT the live pin, so the complete population is
    # no longer uniformly historical. Both semantics must be present and the current count pinned:
    # asserting only the set would pass if a single run stayed current.
    assert {run["semantics"] for run in full_runs} == {"current", "historical"}
    assert sum(1 for run in full_runs if run["semantics"] == "current") == 17
    expected_flux2_runtime = {
        "imc-998b89c5d76dbcc84332": "bounded_attention",
        "imc-b4113eedf503e409ad1b": "resident",
        "imc-b62adbfca64f277414e1": "bounded_decode",
        "imc-bfb890dff959eaf09183": "staged_residency",
        "imc-f5c3d06f30ebf3723f13": "bounded_transformer_residency",
    }
    flux2_runtime = [
        run
        for run in runtime_complete_runs
        if run["record"]["target"]["modelId"] == "flux2_dev"
    ]
    assert {
        run["record"]["id"]: run["record"]["strategy"]["rung"]
        for run in flux2_runtime
    } == expected_flux2_runtime
    assert {run["semantics"] for run in flux2_runtime} == expected_flux2_semantics
    assert {
        run["record"]["repositories"]["inference"]["revision"]
        for run in flux2_runtime
    } == {"5ffd7612e7de4e76b6db00a7148ed3d9c15b4c0d"}

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
    # Inside the audited window the runtime-complete population is mixed (FLUX.2 current, FLUX.1
    # historical); outside it every record is historical. `eligible` is unaffected either way —
    # ageing past the window re-dates a record, it does not reject it.
    assert {run["semantics"] for run in runtime_complete_runs} == (
        {"current", "historical"} if within_audited_window else {"historical"}
    )
    assert all(run["binding"]["eligible"] for run in runtime_complete_runs)
    current_eligible = [
        run
        for run in matrix["calibrationRuns"]
        if run["semantics"] == "current" and run["binding"]["eligible"]
    ]
    # Two independent sources of "current", kept separate and derived from the live provider
    # closures rather than the repository-wide pin:
    #
    #   - Full-complete records whose captured provider closure still matches the live closure;
    #   - the audited FLUX.2 window, current only while its audited revision IS the live pin.
    #
    # A global pin bump may leave a provider's compile closure byte-identical. Requiring the capture
    # revision itself to equal the live pin would contradict the closure-digest currency contract
    # and falsely report that calibrated admission fell back to the legacy estimator.
    current_complete_ids = {
        run["record"]["id"]
        for run in full_runs
        if run["semantics"] == "current"
    }
    assert len(current_complete_ids) == 17, (
        "the shipped bundle must retain the 17 closure-current complete records; an empty or "
        "reduced set means calibrated admission has silently fallen back toward the legacy estimator"
    )
    # Current is necessary but not sufficient for eligible: a record must also BIND a declared cell.
    # sc-16915 swept seven decode tile edges and the manifest binds only the production point
    # (512/64), so the six off-point edges are current-but-ineligible by design — they widen the
    # published range without certifying a cell of their own.
    unbound_decode_edges = {
        record["id"]
        for record in calibration["records"]
        if record["id"] in current_complete_ids
        and record["strategy"]["rung"] == "bounded_decode"
        and record["sweep"]["cases"][0]["parameters"].get("decodeTileEdge") != 512
    }
    assert len(unbound_decode_edges) == 6
    assert {run["record"]["id"] for run in current_eligible} == (
        current_complete_ids - unbound_decode_edges
    ) | (set(expected_flux2_runtime) if within_audited_window else set())
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
    assert len(historical_qwen) == 4
    assert all(not run["binding"]["eligible"] for run in historical_qwen)
    assert all(
        run["binding"]["reasons"] == ["fingerprint-mismatch"] for run in historical_qwen
    )
    assert all(
        run["record"]["calibrationFingerprint"].endswith("-eager")
        for run in historical_qwen
    ), "the mismatch must be the collapsed load-shape suffix, not some other drift"
    assert {
        (
            run["record"]["backend"],
            run["record"]["target"]["tier"],
            run["record"]["target"]["mode"],
            run["record"]["target"]["overlay"],
        )
        for run in historical_qwen
    } == {("mlx", "bf16", "text_to_image", "none")}
    assert Counter(
        run["record"]["strategy"]["rung"] for run in historical_qwen
    ) == {
        "resident": 2,
        "staged_residency": 2,
    }


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


def test_historical_records_remain_unverified_after_the_z_image_pin_advance():
    matrix = load_matrix()
    assert matrix["summary"]["fullModels"] == 0
    # sc-16915 recaptured the Qwen and Krea MLX evidence at the current pin, which is what
    # "until the records are recaptured" was waiting for, so those cells now promote.
    # Z-Image was NOT recaptured and is the subject of this test: its history must still fail
    # closed, which the per-model assertions below pin directly.
    #
    # Stated as the exact verified SET rather than as `== []`. A bare emptiness check stopped
    # being meaningful the moment anything was verified, and a count would let one model's
    # promotion silently cover another's regression.
    verified = {
        (cell["modelId"], cell["backend"], cell["tier"], cell["rung"])
        for cell in matrix["cells"]
        if cell["state"] == "Verified"
    }
    assert verified == {
        ("qwen_image", "mlx", "bf16", "resident"),
        ("qwen_image", "mlx", "bf16", "staged_residency"),
        ("qwen_image", "mlx", "bf16", "bounded_decode"),
        ("qwen_image", "mlx", "bf16", "bounded_attention"),
        ("qwen_image", "mlx", "bf16", "bounded_transformer_residency"),
        ("qwen_image", "mlx", "q8", "bounded_attention"),
        ("qwen_image", "mlx", "q8", "bounded_transformer_residency"),
        ("qwen_image", "mlx", "q4", "bounded_attention"),
        ("qwen_image", "mlx", "q4", "bounded_transformer_residency"),
        ("krea_2_turbo", "mlx", "q4", "bounded_decode"),
    }
    assert not [
        cell
        for cell in matrix["cells"]
        if cell["state"] == "Verified" and cell["modelId"].startswith("z_image")
    ], "Z-Image was not recaptured, so none of its cells may promote"
    historical_z_image_turbo = [
        cell
        for cell in matrix["cells"]
        if cell["modelId"] == "z_image_turbo"
        and cell["backend"] == "mlx"
        and cell["tier"] == "q4"
        and cell["mode"] == "text_to_image"
        and cell["overlay"] == "none"
        and cell["evidence"]["historicalVerification"]
    ]
    assert len(historical_z_image_turbo) == 5
    assert all(
        cell["state"] == "Implemented/unverified"
        and cell["evidence"]["currentEnvironmentVerification"] == []
        for cell in historical_z_image_turbo
    )
    historical_z_image = [
        cell
        for cell in matrix["cells"]
        if cell["modelId"] == "z_image"
        and cell["backend"] == "candle"
        and cell["evidence"]["historicalVerification"]
    ]
    assert historical_z_image == []
    # sc-16915 recaptured this ladder, so these five are Verified rather than
    # Implemented/unverified, and their parameters are the ones the promoted bindings name
    # (overlap 64, attention chunk 64 MiB) rather than the previous 128 / 128 MiB point.
    recaptured_qwen_cells = [
        cell
        for cell in matrix["cells"]
        if cell["modelId"] == "qwen_image"
        and cell["backend"] == "mlx"
        and cell["tier"] == "bf16"
        and cell["mode"] == "text_to_image"
        and cell["overlay"] == "none"
    ]
    assert len(recaptured_qwen_cells) == 5
    assert all(cell["state"] == "Verified" for cell in recaptured_qwen_cells)
    assert all(
        cell["evidence"]["currentEnvironmentVerification"]
        for cell in recaptured_qwen_cells
    ), "a Verified cell must carry the current-environment evidence its guard requires"
    assert {
        (cell["modelId"], cell["backend"], cell["tier"], cell["mode"], cell["overlay"])
        for cell in recaptured_qwen_cells
    } == {("qwen_image", "mlx", "bf16", "text_to_image", "none")}
    assert {cell["rung"] for cell in recaptured_qwen_cells} == {
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
    for cell in recaptured_qwen_cells:
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
    rejected(
        lambda evidence: evidence["loadability"][0].__setitem__("unchecked", True)
    )


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
    # `familyStory` is the family GROUP key, which is the family's MLX story id — the same key on
    # both backends, unlike `cells[].owningFamilyStory`, which is the backend-scoped owner.
    assert {(row["familyStory"], row["backend"]) for row in rows} == {
        (model["owningFamilyStories"]["mlx"], backend)
        for model in matrix["models"]
        for backend in model["backends"]
    }

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
    ]
    assert next(
        row
        for row in rows
        if row["familyStory"] == 15519 and row["backend"] == "candle"
    ) == {
        "familyStory": 15519,
        "backend": "candle",
        "structuralApplicability": "partial",
        "requestPeak": "moves",
        "implementation": "shared-primitive",
    }
    assert all(
        row["implementation"] != "none"
        for row in rows
        if row["requestPeak"] != "unmeasured"
    )


def test_rung4_partial_applicability_and_structural_verdicts_carry_their_evidence():
    """Partial applicability is recorded, and a Structurally N/A cell always cites why."""
    matrix = load_matrix()

    # The story's named trap: a U-Net is not automatically Structurally N/A. SDXL's lowest level is
    # a genuine 10-deep transformer stack, so the verdict is `partial` and the cell is Missing —
    # applicable but unimplemented — rather than exempt from the ladder.
    sdxl = [
        cell
        for cell in matrix["cells"]
        if cell["modelId"] == "sdxl"
        and cell["rung"] == "bounded_transformer_residency"
    ]
    assert sdxl
    assert {cell["rung4Survey"]["structuralApplicability"] for cell in sdxl} == {"partial"}
    assert {cell["state"] for cell in sdxl} == {"Missing"}
    stacks = sdxl[0]["rung4Survey"]["blockStacks"]
    assert any(stack["windowable"] for stack in stacks)
    assert any(not stack["windowable"] for stack in stacks)

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
