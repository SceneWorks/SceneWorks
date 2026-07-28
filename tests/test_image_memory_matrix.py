import copy
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
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
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
