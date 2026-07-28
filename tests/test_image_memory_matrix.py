import json
import pathlib
import subprocess

import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "generated" / "image-memory-matrix.json"
SCHEMA = ROOT / "packages" / "schemas" / "image-memory-matrix.schema.json"


def load_matrix():
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_generated_image_memory_matrix_is_current_and_schema_valid():
    subprocess.run(
        ["node", "scripts/generate-image-memory-matrix.mjs", "--check"],
        cwd=ROOT,
        check=True,
    )
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(
        load_matrix()
    )


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


def test_only_explicit_revision_bound_calibration_claims_dynamic_verification():
    matrix = load_matrix()
    assert matrix["summary"]["fullModels"] == 0
    verified = [cell for cell in matrix["cells"] if cell["state"] == "Verified"]
    assert verified
    assert all(
        cell["modelId"] == "krea_2_turbo"
        and cell["backend"] == "candle"
        and cell["mode"] == "text_to_image"
        and cell["overlay"] == "none"
        and cell["rung"]
        in {
            "resident",
            "staged_residency",
            "bounded_decode",
            "bounded_attention",
            "bounded_transformer_residency",
        }
        and cell["calibrationFingerprint"] == "krea-turbo-cuda-phase-curves-v1"
        and cell["evidence"]["currentEnvironmentVerification"]
        and cell["evidence"]["strategyParameterVerification"]
        for cell in verified
    )
    assert all(
        all(
            int(resolution.split("x")[0]) * int(resolution.split("x")[1]) <= 1_048_576
            for resolution in cell["geometryEnvelope"]["resolutions"]
        )
        for cell in verified
    )
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
    for cell in verified:
        parameters = cell["strategyParameters"]
        assert {
            key: value
            for key, value in parameters.items()
            if key not in {"manifestRung", "formula"}
        } == expected_parameters[cell["rung"]]
        assert (
            cell["evidence"]["strategyParameterVerification"][0]["exactParameters"]
            == expected_parameters[cell["rung"]]
        )
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
