"""Pure verifier contract tests. Synthetic bytes only; NEVER campaign evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parents[2] / ".claude" / "tmp" / "moss-semantic-contract-tests"


def load_local(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load_local("moss_static_verifier", "verify_evidence.py")
fixtures = load_local("moss_schema_test_only", "schema_test_fixtures.py")


def tiny_export(file, values, metadata=None):
    payload = struct.pack("<" + "f" * len(values), *values)
    header = {"factor": {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(payload)]}}
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    Path(file).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def optional_diagnostic_fixture(mask=False, content_hash="a" * 64):
    """SCHEMA TEST ONLY: descriptors are invented tiny test data, not observations."""
    descriptor = {"shape": [1, 2], "strides": [2, 1], "dtype": "F32",
                  "content_encoding": "logical-row-major-f32-le-v1",
                  "content_sha256": content_hash}
    tensors = {name: copy.deepcopy(descriptor) for name in ("latent", "context", "noise")}
    if mask:
        tensors["mask"] = copy.deepcopy(descriptor)
    return {"recordKind": "schema-test-fixture", "isCampaignEvidence": False,
            "fixtureLabel": "SCHEMA TEST ONLY - NOT CAMPAIGN EVIDENCE",
            "prepared_evaluation": {
                "before": {"loss": 0.75, "sigma": 0.375, "tensors": copy.deepcopy(tensors)},
                "after": {"loss": 0.50, "sigma": 0.375, "tensors": copy.deepcopy(tensors)}}}


class StaticContractTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="schema-test-only-", dir=SCRATCH)
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_normalizer_is_loaded_under_isolated_python_from_trusted_sibling(self):
        self.assertEqual(Path(verifier._normalizer.__file__).resolve(), HERE / "normalize_native.py")

    def test_normalizer_preserves_actual_descriptor_strides_and_hash_encoding(self):
        raw = optional_diagnostic_fixture(mask=True)["prepared_evaluation"]["before"]
        point = verifier._normalizer._point(raw)
        self.assertEqual(point["prepared"]["latent"]["layout"]["strides"], [2, 1])
        self.assertEqual(point["prepared"]["latent"]["hashEncoding"], "logical-row-major-f32-le-v1")
        self.assertEqual(point["prepared"]["mask"]["sha256"], "a" * 64)
        self.assertNotIn("packageId", point["prepared"])

    def test_normalizer_rejects_missing_or_extra_prepared_tensor_population(self):
        for name in ("missing", "extra"):
            point = optional_diagnostic_fixture()["prepared_evaluation"]["before"]
            if name == "missing":
                del point["tensors"]["context"]
            else:
                point["tensors"]["unobserved"] = copy.deepcopy(point["tensors"]["latent"])
            with self.assertRaises(verifier._normalizer.NormalizationError):
                verifier._normalizer._point(point)

    def test_normalizer_never_promotes_explicit_schema_fixture(self):
        raw = {"rawFormat": verifier._normalizer.RAW_FORMAT, "manifestSha256": verifier.MANIFEST_SHA256,
               "recordKind": "schema-test-fixture", "isCampaignEvidence": False,
               "fixtureLabel": "SCHEMA TEST ONLY - NOT CAMPAIGN EVIDENCE"}
        with self.assertRaisesRegex(verifier._normalizer.NormalizationError, "fixture evidence cannot be promoted"):
            verifier._normalizer.normalize_capture(raw, self.root, {}, {}, {"caseId": "mlx/base/lora"})

    def test_normalizer_derivatives_are_owned_and_never_overwrite_raw_observations(self):
        context = verifier._normalizer._Context(self.root)
        reference = context.put("synthetic.json", {"fixtureLabel": "SCHEMA TEST ONLY - NOT CAMPAIGN EVIDENCE"})
        self.assertTrue(reference["path"].startswith("normalized-native/"))
        self.assertEqual(reference, context.put("synthetic.json", context.read(reference)))
        with self.assertRaises(verifier._normalizer.NormalizationError):
            context.put("synthetic.json", {"changed": True})
        for path in ("../escaped.json", "/tmp/escaped.json", "C:\\escaped.json"):
            with self.assertRaises(verifier._normalizer.NormalizationError):
                context.path(path, exists=False)

    def test_weight_proof_references_rebase_and_remain_verified_artifacts(self):
        file = self.root / "proof.json"
        file.write_text('{"fixtureLabel":"SCHEMA TEST ONLY - NOT CAMPAIGN EVIDENCE"}')
        value = {"identity": {"weightSources": [{"files": [{"path": "proof.json", "sha256": verifier.digest(file)}]}]}}
        self.assertEqual(len(list(verifier._artifact_references(value))), 1)
        verifier._rebase_artifacts(value, "capture-mlx/base/lora")
        self.assertEqual(next(verifier._artifact_references(value))["path"], "capture-mlx/base/lora/proof.json")

    def test_native_descriptor_normal_form_matches_schema_without_invented_fields(self):
        from jsonschema import Draft202012Validator
        schema = verifier.read_json(HERE / "evidence.schema.json")
        schema = {"$defs": schema["$defs"], "$ref": "#/$defs/evaluationPoint"}
        for mask in (False, True):
            native = optional_diagnostic_fixture(mask=mask)["prepared_evaluation"]["before"]
            Draft202012Validator(schema).validate(verifier._normalizer._point(native))

    def test_schema_requires_nonempty_repository_groups_not_a_fabricated_population(self):
        from jsonschema import Draft202012Validator
        schema = verifier.read_json(HERE / "evidence.schema.json")
        record = copy.deepcopy(fixtures.record)
        record["identity"]["weightSources"] = record["identity"]["weightSources"][:1]
        Draft202012Validator(schema).validate(record)
        self.assertEqual(record["recordKind"], "schema-test-fixture")
        record["identity"]["weightSources"] = []
        self.assertFalse(Draft202012Validator(schema).is_valid(record))

    def test_workflow_provenance_refuses_mixed_run_attempt_or_source(self):
        receipt = {"fixtureLabel": "SCHEMA TEST ONLY - NOT CAMPAIGN EVIDENCE",
                   "source": {"workflowSource": "a" * 40, "workflowSha256": "b" * 64},
                   "workflow": {"runId": "123", "attempt": 1, "sha": "a" * 40,
                                "fileSha256": "b" * 64, "ref": "refs/heads/SCHEMA_TEST_ONLY", "jobId": "synthetic"}}
        self.assertEqual(verifier.verify_workflow_provenance(receipt, "123"), 1)
        for field, value in (("runId", "124"), ("attempt", 2), ("attempt", "1"), ("sha", "c" * 40), ("fileSha256", "c" * 64)):
            broken = copy.deepcopy(receipt)
            broken["workflow"][field] = value
            with self.assertRaises(verifier.EvidenceError):
                verifier.verify_workflow_provenance(broken, "123", 1)

    def test_native_consumer_uses_optional_mapping_not_phantom_vector(self):
        source = (HERE / "verify_evidence.py").read_text()
        self.assertIn('raw.get("prepared_evaluation")', source)
        self.assertNotIn('raw.get("prepared_evaluations")', source)

    def test_future_inspection_rejected_from_environment_and_file_envelopes(self):
        clock = verifier.timestamp("2026-10-25T12:00:00Z")
        inspection = {"reviewer": "SCHEMA TEST ONLY", "completedAt": "2026-10-25T12:10:00Z",
                      "findings": ["SCHEMA TEST ONLY - NOT CAMPAIGN EVIDENCE"],
                      "reviewedArtifact": {"path": "SCHEMA_TEST_ONLY.png", "sha256": "a" * 64}}
        document = {"fixtureLabel": "SCHEMA TEST ONLY - NOT CAMPAIGN EVIDENCE",
                    "manifestSha256": verifier.MANIFEST_SHA256, "captureRunId": "123",
                    "cases": [{"caseId": case, "inspections": [copy.deepcopy(inspection) for _ in range(3)]}
                              for case in verifier.CELLS]}
        encoded = json.dumps(document)
        (self.root / "inspections.json").write_text(encoded)
        for source in ("environment", "file"):
            with self.subTest(source=source), patch.dict(verifier.os.environ, {
                    "MOSS_INSPECTION_JSON": encoded if source == "environment" else "",
                    "MOSS_CAPTURE_RUN_ID": "123"}):
                with self.assertRaisesRegex(verifier.EvidenceError, "verification future"):
                    verifier.inspection_set(self.root, clock)

    def test_completed_post_execution_inspection_remains_legal_within_original_case_window(self):
        started = verifier.timestamp("2026-10-25T12:00:00Z")
        ended = verifier.timestamp("2026-10-25T12:01:00Z")
        clock = verifier.timestamp("2026-10-25T12:03:00Z")
        at = verifier.verify_inspection_clock("2026-10-25T12:02:00Z", clock)
        self.assertGreater(at, ended)
        self.assertTrue(started <= at <= started + 3600)
        self.assertEqual(verifier.verify_inspection_clock("2026-10-25T12:03:00Z", clock), clock)
        with self.assertRaisesRegex(verifier.EvidenceError, "verification future"):
            verifier.verify_inspection_clock("2026-10-25T12:03:00.001Z", clock)

    def test_optional_diagnostic_complete_keys_validate_with_and_without_mask(self):
        manifest = verifier.read_json(HERE / "campaign.json")
        for mask in (False, True):
            record = optional_diagnostic_fixture(mask=mask)
            result = verifier.verify_optional_prepared_evaluation(record, manifest)
            self.assertEqual(set(result["before"]["tensors"]), set(result["after"]["tensors"]))
            self.assertEqual(record["recordKind"], "schema-test-fixture")
            self.assertFalse(record["isCampaignEvidence"])

    def test_optional_diagnostic_mask_before_only_is_rejected(self):
        manifest = verifier.read_json(HERE / "campaign.json")
        record = optional_diagnostic_fixture(mask=True)
        del record["prepared_evaluation"]["after"]["tensors"]["mask"]
        with self.assertRaisesRegex(verifier.EvidenceError, "tensor key sets changed"):
            verifier.verify_optional_prepared_evaluation(record, manifest)

    def test_optional_diagnostic_absent_or_null_refuses_when_manifest_requires_it(self):
        manifest = verifier.read_json(HERE / "campaign.json")
        for absent in (True, False):
            record = optional_diagnostic_fixture()
            if absent:
                del record["prepared_evaluation"]
            else:
                record["prepared_evaluation"] = None
            with self.assertRaisesRegex(verifier.EvidenceError, "required prepared_evaluation diagnostic is absent; fail-closed refusal"):
                verifier.verify_optional_prepared_evaluation(record, manifest)

    def test_optional_diagnostic_incomplete_descriptor_or_required_tensor_refuses(self):
        manifest = verifier.read_json(HERE / "campaign.json")
        for missing in ("content_encoding", "context"):
            record = optional_diagnostic_fixture()
            if missing == "context":
                del record["prepared_evaluation"]["before"]["tensors"]["context"]
                del record["prepared_evaluation"]["after"]["tensors"]["context"]
            else:
                del record["prepared_evaluation"]["before"]["tensors"]["latent"][missing]
            with self.assertRaises(verifier.EvidenceError):
                verifier.verify_optional_prepared_evaluation(record, manifest)

    def test_optional_diagnostic_present_but_incomplete_never_skips_validation(self):
        manifest = verifier.read_json(HERE / "campaign.json")
        manifest["evaluation"]["withinNativeRun"]["recordBeforeAndAfterTensorShapesDtypesLayoutsAndHashes"] = False
        record = optional_diagnostic_fixture()
        record["prepared_evaluation"]["after"] = None
        with self.assertRaisesRegex(verifier.EvidenceError, "incomplete observation keys"):
            verifier.verify_optional_prepared_evaluation(record, manifest)

    def test_optional_diagnostic_changed_hash_wrong_encoding_or_sigma_refuses(self):
        manifest = verifier.read_json(HERE / "campaign.json")
        for change in ("hash", "encoding", "sigma"):
            record = optional_diagnostic_fixture()
            after = record["prepared_evaluation"]["after"]
            if change == "hash":
                after["tensors"]["noise"]["content_sha256"] = "b" * 64
            elif change == "encoding":
                after["tensors"]["noise"]["content_encoding"] = "SCHEMA_TEST_ONLY_WRONG_ENCODING"
            else:
                after["sigma"] = 0.5
            with self.assertRaises(verifier.EvidenceError):
                verifier.verify_optional_prepared_evaluation(record, manifest)

    def test_optional_diagnostic_cross_backend_hashes_may_differ(self):
        manifest = verifier.read_json(HERE / "campaign.json")
        left = optional_diagnostic_fixture(content_hash="a" * 64)
        right = optional_diagnostic_fixture(content_hash="b" * 64)
        verifier.verify_optional_prepared_evaluation(left, manifest)
        verifier.verify_optional_prepared_evaluation(right, manifest)
        self.assertNotEqual(left["prepared_evaluation"]["before"]["tensors"], right["prepared_evaluation"]["before"]["tensors"])

    def test_valid_learning_pair_is_strict_and_fixed(self):
        self.assertEqual(verifier.verify_learning(fixtures.learning("only-test"), fixtures.EVALUATION), (0.75, 0.5))

    def test_nonlearning_pair_rejected(self):
        value = fixtures.learning("only-test")
        value["after"]["loss"] = value["before"]["loss"]
        with self.assertRaises(verifier.EvidenceError):
            verifier.verify_learning(value, fixtures.EVALUATION)

    def test_changed_actual_prepared_tensor_hash_rejected(self):
        value = fixtures.learning("only-test")
        value["after"]["prepared"]["noise"]["sha256"] = "9" * 64
        with self.assertRaises(verifier.EvidenceError):
            verifier.verify_learning(value, fixtures.EVALUATION)

    def test_wrong_evaluation_identity_rejected(self):
        with self.assertRaises(verifier.EvidenceError):
            verifier.verify_learning(fixtures.learning("only-test"), "e" * 64)

    def test_actual_numeric_factor_delta_not_file_hash_difference(self):
        left, right = self.root / "a.safetensors", self.root / "b.safetensors"
        tiny_export(left, [0.0, 0.0, 0.0])
        tiny_export(right, [1.0, 0.0, -0.5])
        delta = verifier.recompute_factor_delta(left, right)
        self.assertEqual((delta["changedElements"], delta["maxAbsDelta"], delta["squaredL2Delta"]), (2, 1.0, 1.25))

    def test_metadata_only_export_change_is_not_parameter_update(self):
        left, right = self.root / "a.safetensors", self.root / "b.safetensors"
        tiny_export(left, [0.25], {"job": "schema-test-only-left"})
        tiny_export(right, [0.25], {"job": "schema-test-only-right"})
        self.assertNotEqual(verifier.digest(left), verifier.digest(right))
        with self.assertRaises(verifier.EvidenceError):
            verifier.recompute_factor_delta(left, right)

    def test_recovery_compares_payload_not_job_metadata(self):
        left, right = self.root / "a.safetensors", self.root / "b.safetensors"
        tiny_export(left, [0.25], {"job": "schema-test-only-left"})
        tiny_export(right, [0.25], {"job": "schema-test-only-right"})
        verifier.tensor_payloads_equal(left, right)
        tiny_export(right, [0.5])
        with self.assertRaises(verifier.EvidenceError):
            verifier.tensor_payloads_equal(left, right)

    def test_nonfinite_factor_rejected(self):
        left, right = self.root / "a.safetensors", self.root / "b.safetensors"
        tiny_export(left, [0.0])
        tiny_export(right, [float("nan")])
        with self.assertRaises(verifier.EvidenceError):
            verifier.recompute_factor_delta(left, right)

    def test_missing_and_zero_rss_fail_closed(self):
        guard = copy.deepcopy(fixtures.record["cleanup"])
        start = verifier.timestamp(guard["caseStartedAt"])
        end = verifier.timestamp(guard["caseEndedAt"])
        verifier.verify_rss(guard, 600, start, end)
        for mutation in ("empty", "zero", "missing"):
            broken = copy.deepcopy(guard)
            if mutation == "empty":
                broken["aggregateRssSamples"] = []
            elif mutation == "zero":
                broken["aggregateRssSamples"][0]["bytes"] = 0
            else:
                broken["missingMeasurementCount"] = 1
            with self.assertRaises(verifier.EvidenceError):
                verifier.verify_rss(broken, 600, start, end)

    def test_device_fraction_preserves_product_f32_representation(self):
        manifest = verifier.read_json(HERE / "campaign.json")
        device = copy.deepcopy(fixtures.record["identity"]["device"])
        device["totalCapacityBytes"] = 128 * 1024 ** 3
        fraction = struct.unpack("<f", struct.pack("<f", 0.60))[0]
        device["resolvedCeilingBytes"] = int(device["totalCapacityBytes"] * fraction)
        device["projectedDemandBytes"] = 500
        self.assertEqual(verifier.verify_device(device, "mlx", manifest), device["resolvedCeilingBytes"])

    def test_fixture_record_cannot_be_campaign_acceptance(self):
        manifest = verifier.read_json(HERE / "campaign.json")
        with self.assertRaisesRegex(verifier.EvidenceError, "schema-test/missing/unproven"):
            verifier.verify_record(copy.deepcopy(fixtures.record), self.root, manifest, {})

    def test_cross_load_has_independent_job_clock(self):
        clock = copy.deepcopy(fixtures.record["crossBackendLoad"]["phaseTiming"])
        start, end = verifier.verify_cross_load_clock(clock, 3600)
        old_training_start = verifier.timestamp(fixtures.record["cleanup"]["caseStartedAt"])
        self.assertGreater(start - old_training_start, 3600)
        self.assertEqual(end - start, 2)
        clock["completedAt"] = "2000-01-02T01:00:01Z"
        clock["wallSeconds"] = 3601
        with self.assertRaises(verifier.EvidenceError):
            verifier.verify_cross_load_clock(clock, 3600)

    def test_strict_json_rejects_nan_and_duplicate_keys(self):
        file = self.root / "schema-test-only.json"
        for encoded in ('{"loss":NaN}', '{"same":1,"same":2}'):
            file.write_text(encoded)
            with self.assertRaises(verifier.EvidenceError):
                verifier.read_json(file)

    def test_png_control_compares_actual_pixels_not_job_metadata(self):
        from PIL import Image, PngImagePlugin
        left, right = self.root / "schema-test-only-a.png", self.root / "schema-test-only-b.png"
        image = Image.new("RGB", (256, 256), (12, 34, 56))
        for file, label in ((left, "schema-test-only-left"), (right, "schema-test-only-right")):
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("job", label)
            image.save(file, pnginfo=metadata)
        self.assertNotEqual(verifier.digest(left), verifier.digest(right))
        self.assertEqual(verifier.decoded_png(left), verifier.decoded_png(right))
        Image.new("RGB", (256, 256), (13, 34, 56)).save(right)
        self.assertNotEqual(verifier.decoded_png(left), verifier.decoded_png(right))

    def test_artifact_traversal_and_symlink_rejected(self):
        file = self.root / "schema-test-only.txt"
        file.write_text("SCHEMA TEST ONLY")
        with self.assertRaises(verifier.EvidenceError):
            verifier.member_path(self.root, "../outside")
        link = self.root / "schema-test-only-link"
        link.symlink_to(file)
        with self.assertRaises(verifier.EvidenceError):
            verifier.member_path(self.root, link.name)


if __name__ == "__main__":
    unittest.main()
