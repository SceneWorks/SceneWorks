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
    assert matrix["evidenceDimensions"] == [
        "staticImplementation",
        "declaredCalibration",
        "historicalVerification",
        "currentEnvironmentVerification",
        "loadability",
        "strategyParameterVerification",
    ]


def test_static_inventory_never_claims_dynamic_verification_or_full():
    matrix = load_matrix()
    assert matrix["summary"]["fullModels"] == 0
    assert not any(cell["state"] == "Verified" for cell in matrix["cells"])
    for cell in matrix["cells"]:
        if cell["state"] != "Missing":
            assert cell["evidence"]["staticImplementation"]
            assert cell["calibrationFingerprint"]
