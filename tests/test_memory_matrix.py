import copy
import json
import os
import pathlib
import shutil
import stat
import subprocess
import uuid

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
    expected_ids = {
        "imc-ae60c3eb464c2133a553",
        "imc-ea17789b97699dcb0352",
    }
    assert {record["id"] for record in calibration["records"]} == expected_ids
    assert {run["record"]["id"] for run in matrix["calibrationRuns"]} == expected_ids
    assert all(run["binding"] == {"eligible": True, "reasons": []} for run in matrix["calibrationRuns"])
    assert all(run["semantics"] == "current" for run in matrix["calibrationRuns"])


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
    mutations = [
        lambda record: record["repositories"]["sceneWorks"].update(dirty=True),
        lambda record: record["scenarios"][3].update(result="not_run"),
        lambda record: record["scenarios"][4].update(cleanupVerified=False),
        lambda record: record["quality"].update(result="not_run"),
        lambda record: record["negativeMutation"].update(measured=False),
        lambda record: record["loadability"].update(resolvedPathFingerprint=""),
        lambda record: record.update(predictedPeakBytes=None),
        lambda record: record["observedMemory"]["decode"].update(activeBytes=-1),
        lambda record: record["repositories"]["sceneWorks"].pop("matrixSourceRevision"),
        lambda record: record["sweep"].update(axes=[]),
        lambda record: record["sweep"]["axes"].append(copy.deepcopy(record["sweep"]["axes"][0])),
        lambda record: record["sweep"]["cases"].append(copy.deepcopy(record["sweep"]["cases"][0])),
        lambda record: record["quality"].update(contract=""),
        lambda record: record["hardware"].update(unexpected="closed"),
        lambda record: record["scenarios"][0].update(unexpected="closed"),
    ]
    for mutate in mutations:
        invalid = copy.deepcopy(bundle)
        mutate(invalid["records"][0])
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(invalid)
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


def test_aggregate_cells_do_not_promote_exact_records_to_dynamic_verification():
    matrix = load_matrix()
    assert matrix["summary"]["fullModels"] == 0
    verified = [cell for cell in matrix["cells"] if cell["state"] == "Verified"]
    assert verified == []
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
        lambda evidence: evidence["strategyParameterVerification"][0].__setitem__(
            "geometry", "up to 1024"
        )
    )
    rejected(
        lambda evidence: evidence["loadability"][0].__setitem__("unchecked", True)
    )
