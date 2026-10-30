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

    def refusal_fixture(self, name="refusal", output_dir="/owned/loras/lora_low",
                        inventory_path=None, lora_id="lora_low"):
        # Literal JSON/files only; never campaign evidence or model execution.
        root = self.root / name
        root.mkdir()
        ctx = verifier._normalizer._Context(root)
        plan = {"jobId": "low-job", "config": {"steps": 60, "cacheLatents": True},
                "output": {"outputDir": output_dir, "loraId": lora_id}}
        job = {"id": "low-job", "type": "lora_train", "status": "failed", "result": {},
               "payload": {"plan": copy.deepcopy(plan)}}
        snapshot = root / "runs/low-cap/training-config.json"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text(json.dumps(plan))
        diagnostic = ctx.put("low-diagnostic.json", {"nativeMetadata": {
            "refusal": "structural working-set bound 200 bytes exceeds the 100-byte device/trainer budget",
            "weightsLoaded": False}, "nativeReceipts": []})
        event = ctx.put("low-submission.json", {"method": "POST", "httpStatus": 201,
            "route": "/api/v1/projects/owned/training/jobs", "response": copy.deepcopy(job)})
        (root / "devicepreflight.json").write_text("{}")
        low = {"jobId": "low-job", "job": job, "diagnostic": diagnostic,
               "preparedNativeConfig": {"memoryLimitBytes": 100},
               "publishedOutputInventory": [inventory_path or output_dir + "/training-config.json"],
               "submissionConfigSnapshot": ctx.reference(snapshot)}
        raw = {"raw": {"subruns": {"low-cap": low}, "api": [event]}}
        return ctx, raw

    def test_refusal_config_is_authenticated_information_not_model_output(self):
        paths = [("/owned/loras/lora_low", "/owned/loras/lora_low/training-config.json"),
                 ("/var/owned/loras/lora_low", "/private/var/owned/loras/lora_low/training-config.json"),
                 ("/private/var/owned/loras/lora_low", "/var/owned/loras/lora_low/training-config.json"),
                 (r"C:\Users\Michael\loras\lora_low", r"\\?\C:\Users\Michael\loras\lora_low\training-config.json"),
                 ("C:/Users/Michael/loras/lora_low", r"C:\Users\Michael\loras\lora_low\training-config.json"),
                 ("/owned/loras/lora_low", None)]
        for output_dir, inventory_path in paths:
            ctx, raw = self.refusal_fixture(str(len(list(self.root.iterdir()))), output_dir, inventory_path)
            low = raw["raw"]["subruns"]["low-cap"]
            if inventory_path is None:
                low["publishedOutputInventory"] = []
                del low["submissionConfigSnapshot"]
            before = copy.deepcopy(raw)
            result = verifier._normalizer._memory_refusal(ctx, raw)
            self.assertEqual(raw, before)
            self.assertFalse(result["publishedOutput"])
            self.assertEqual(result["publishedOutputInventory"], [])
            self.assertEqual(result["receipt"], low["diagnostic"])
            if inventory_path is not None:
                self.assertEqual(ctx.read(low["submissionConfigSnapshot"]), low["job"]["payload"]["plan"])
            # The existing downstream contract remains strict and requires no schema relaxation.
            from jsonschema import Draft202012Validator
            schema = verifier.read_json(HERE / "evidence.schema.json")
            contract = {"$defs": schema["$defs"], "$ref": "#/$defs/memoryRefusal"}
            validator = Draft202012Validator(contract)
            validator.validate(result)
            result["publishedOutputInventory"] = ["training-config.json"]
            self.assertTrue(list(validator.iter_errors(result)))

    def test_refusal_config_path_is_bound_to_submitted_directory_and_lora_id(self):
        values = [("/owned/loras/lora_low", "/foreign/training-config.json", "lora_low"),
                  ("/owned/loras/lora_low", "/owned/loras/lora_other/training-config.json", "lora_low"),
                  ("/owned/loras/lora_other", "/owned/loras/lora_other/training-config.json", "lora_low"),
                  ("/var/owned/loras/lora_low", "/private/other/owned/loras/lora_low/training-config.json", "lora_low"),
                  (r"C:\owned\loras\lora_low", r"\\?\C:\foreign\loras\lora_low\training-config.json", "lora_low"),
                  (r"C:\owned\loras\lora_low", r"\\?\D:\owned\loras\lora_low\training-config.json", "lora_low"),
                  (r"C:\owned\loras\lora_low", r"\\?\UNC\server\owned\loras\lora_low\training-config.json", "lora_low"),
                  (r"C:\owned\loras\lora_low", r"\\.\C:\owned\loras\lora_low\training-config.json", "lora_low"),
                  (r"C:\owned\loras\lora_low", r"\\?\C:owned\loras\lora_low\training-config.json", "lora_low"),
                  ("/owned/loras/lora_low", "/owned/loras/lora_low/training-config.json", "../lora_low")]
        for index, (output_dir, inventory_path, lora_id) in enumerate(values):
            ctx, raw = self.refusal_fixture(str(index), output_dir, inventory_path, lora_id)
            with self.subTest(path=inventory_path, lora_id=lora_id), self.assertRaises(ValueError):
                verifier._normalizer._memory_refusal(ctx, raw)

    def test_refusal_outputs_refuse_extra_unknown_and_malformed_inventory(self):
        values = [None, {}, "training-config.json", [None], [1], ["training-config.json"],
                  ["/owned/adapter.safetensors"], ["/owned/step-20"], ["/owned/preview.png"],
                  ["/owned/training-config.json", "/owned/training-config.json"],
                  ["/owned/../training-config.json"], ["/owned/training-config.json/"]]
        for index, value in enumerate(values):
            ctx, raw = self.refusal_fixture(str(index))
            raw["raw"]["subruns"]["low-cap"]["publishedOutputInventory"] = value
            with self.subTest(inventory=value), self.assertRaises(verifier._normalizer.NormalizationError):
                verifier._normalizer._memory_refusal(ctx, raw)

    def test_refusal_config_refuses_missing_tampered_wrong_plan_or_symlink(self):
        for variant in ("missing", "hash", "wrong-plan", "wrong-type", "malformed-json", "symlink",
                        "wrong-reference", "unlisted-snapshot", "missing-submission", "wrong-api-plan"):
            ctx, raw = self.refusal_fixture(variant)
            low = raw["raw"]["subruns"]["low-cap"]
            snapshot = ctx.path(low["submissionConfigSnapshot"]["path"])
            if variant == "missing":
                del low["submissionConfigSnapshot"]
            elif variant == "hash":
                snapshot.write_text("{}")
            elif variant in ("wrong-plan", "wrong-type", "malformed-json"):
                plan = copy.deepcopy(low["job"]["payload"]["plan"])
                plan["config"]["cacheLatents"] = False if variant == "wrong-plan" else 1
                snapshot.write_text("not JSON" if variant == "malformed-json" else json.dumps(plan))
                low["submissionConfigSnapshot"] = ctx.reference(snapshot)
            elif variant == "symlink":
                target = ctx.root / "other.json"
                target.write_bytes(snapshot.read_bytes())
                snapshot.unlink()
                snapshot.symlink_to(target)
            elif variant == "wrong-reference":
                low["submissionConfigSnapshot"] = ctx.put("training-config.json", low["job"]["payload"]["plan"])
            elif variant == "unlisted-snapshot":
                low["publishedOutputInventory"] = []
            elif variant == "missing-submission":
                raw["raw"]["api"] = []
            else:
                event = ctx.read(raw["raw"]["api"][0])
                event["response"]["payload"]["plan"]["config"]["steps"] = 61
                raw["raw"]["api"] = [ctx.put("wrong-api-plan.json", event)]
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                verifier._normalizer._memory_refusal(ctx, raw)

    def test_refusal_config_does_not_excuse_load_receipt_or_budget_failure(self):
        for variant in ("loaded", "receipt", "budget", "job", "export"):
            ctx, raw = self.refusal_fixture(variant)
            low = raw["raw"]["subruns"]["low-cap"]
            diagnostic = ctx.read(low["diagnostic"])
            if variant == "loaded": diagnostic["nativeMetadata"]["weightsLoaded"] = True
            elif variant == "receipt": diagnostic["nativeReceipts"] = [{"steps": 1}]
            elif variant == "budget": low["preparedNativeConfig"]["memoryLimitBytes"] = 99
            elif variant == "job": low["job"]["id"] = "wrong-job"
            else: low["job"]["result"]["trainedArtifactExported"] = True
            low["diagnostic"] = ctx.put("changed-diagnostic.json", diagnostic)
            with self.subTest(variant=variant), self.assertRaises(verifier._normalizer.NormalizationError):
                verifier._normalizer._memory_refusal(ctx, raw)

    def test_normalizer_is_loaded_under_isolated_python_from_trusted_sibling(self):
        self.assertEqual(Path(verifier._normalizer.__file__).resolve(), HERE / "normalize_native.py")

    def test_dmd_component_guard_distinguishes_merged_and_unmerged_receipts(self):
        # Literal metadata only, never a campaign receipt or model execution.
        def report(mode, roles):
            return {"resolved": {"acceleration": mode}, "assets": [{"role": role} for role in roles]}
        merged = ["transformer", "dmd_dataspace"]
        unmerged = ["transformer", "dmd_dataspace", "dmd_time", "dmd_lora"]
        check = verifier._normalizer._dmd_components_retained
        self.assertTrue(check(report("full_fp32", merged), "full_fp32"))
        self.assertTrue(check(report("unmerged_bf16", unmerged), "unmerged_bf16"))
        invalid = [("unmerged_bf16", merged), ("full_fp32", ["transformer"])]
        invalid += [("unmerged_bf16", [role for role in unmerged if role != missing])
                    for missing in ("dmd_dataspace", "dmd_time", "dmd_lora")]
        for mode, roles in invalid:
            with self.subTest(mode=mode, roles=roles), self.assertRaisesRegex(
                    verifier._normalizer.NormalizationError, "components were dropped"):
                check(report(mode, roles), mode)
        with self.assertRaisesRegex(verifier._normalizer.NormalizationError, "mode differs"):
            check(report("unmerged_bf16", unmerged), "full_fp32")
        with self.assertRaisesRegex(verifier._normalizer.NormalizationError, "unknown DMD"):
            check(report("unknown", unmerged), "unknown")

    def test_normalizer_preserves_actual_descriptor_strides_and_hash_encoding(self):
        raw = optional_diagnostic_fixture(mask=True)["prepared_evaluation"]["before"]
        point = verifier._normalizer._point(raw)
        self.assertEqual(point["prepared"]["latent"]["layout"]["strides"], [2, 1])
        self.assertEqual(point["prepared"]["latent"]["hashEncoding"], "logical-row-major-f32-le-v1")
        self.assertEqual(point["prepared"]["mask"]["sha256"], "a" * 64)
        self.assertNotIn("packageId", point["prepared"])

    def application_fixture(self, name="application", dmd=False):
        # SYNTHETIC JSON ONLY: direct helper inputs, never promoted campaign evidence.
        root = self.root / name
        root.mkdir()
        ctx = verifier._normalizer._Context(root)
        sha = "a" * 64
        parent, method = ("sft", "lokr") if dmd else ("base", "lora")
        model = f"moss_image_{parent}"
        report = {"adapter_reports": [{"adapter_path":"accelerator.safetensors","applied":252,"skipped":[]}] if dmd else []}
        row = {"id":"custom", "path":"selected.safetensors", "sha256":sha,
               "scale":1.0, "kind":method, "applied":252, "skipped":[]}
        job = {"id":"image-job", "type":"image_generate", "status":"completed",
               "payload":{"model":model,"loras":[{"id":"custom","weight":1.0,
                          "baseModel":model,"networkType":method}]},
               "result":{"assets":[{"recipe":{"model":model,"rawAdapterSettings":{
                   "mossImageExecutionReport":report,"adapterApply":[row]}}}]}}
        preload = {"jobId":"image-job", "error":None, "resolvedCapBytes":100,
                   "actualSelectedHeaderDemand":{"requiredBytes":50,"adapters":[{
                       "path":"selected.safetensors","sha256":sha,"scale":1.0}]}}
        raw = {"caseId":f"cuda/{parent}/{method}","raw":{"devicePreflight":{"resolvedCapBytes":100},
               "imagePreloads":{"selected":ctx.put("preload.json",preload)},
               "images":{"selected":{"executionReceipt":ctx.put("execution.json",report),
                                         "jobReceipt":ctx.put("job.json",job)}}}}
        return ctx, raw, job, sha

    def test_custom_application_uses_authenticated_provider_join_separate_from_dmd(self):
        for dmd in (False, True):
            ctx, raw, _, sha = self.application_fixture(str(dmd), dmd)
            application, _ = verifier._normalizer._application(ctx, raw, "selected", sha)
            self.assertEqual(application["appliedCount"], 252)
            self.assertEqual(application["receipt"], raw["raw"]["images"]["selected"]["jobReceipt"])

    def test_custom_application_rejects_dmd_only_wrong_join_and_incomplete_reports(self):
        changes = [("id","accelerator"),("path","wrong.safetensors"),("sha256","b"*64),
                   ("scale",0.0),("scale",True),("kind","lokr"),("applied",0),("applied",True),
                   ("skipped",["projection"]),("skipped",None)]
        for index, (key, value) in enumerate(changes):
            ctx, raw, job, sha = self.application_fixture(str(index))
            job["result"]["assets"][0]["recipe"]["rawAdapterSettings"]["adapterApply"][0][key] = value
            raw["raw"]["images"]["selected"]["jobReceipt"] = ctx.put("changed.json",job)
            with self.assertRaises(verifier._normalizer.NormalizationError):
                verifier._normalizer._application(ctx,raw,"selected",sha)
        for variant in ("dmd-only", "duplicate", "truncated", "wrong-job", "wrong-payload", "wrong-sealed"):
            ctx, raw, job, sha = self.application_fixture(variant,True)
            facts = job["result"]["assets"][0]["recipe"]["rawAdapterSettings"]
            if variant == "dmd-only": del facts["adapterApply"]
            elif variant == "duplicate": facts["adapterApply"] *= 2
            elif variant == "truncated": del facts["adapterApply"][0]["skipped"]
            elif variant == "wrong-job": job["id"] = "other"
            elif variant == "wrong-payload": job["payload"]["loras"][0]["id"] = "other"
            else: facts["mossImageExecutionReport"] = {"changed":True}
            raw["raw"]["images"]["selected"]["jobReceipt"] = ctx.put("changed.json",job)
            with self.assertRaises(verifier._normalizer.NormalizationError):
                verifier._normalizer._application(ctx,raw,"selected",sha)

    def test_custom_application_still_rejects_tampered_raw_job_hash(self):
        ctx, raw, job, sha = self.application_fixture()
        reference = raw["raw"]["images"]["selected"]["jobReceipt"]
        job["status"] = "failed"
        (ctx.root / reference["path"]).write_text(json.dumps(job))
        with self.assertRaisesRegex(verifier._normalizer.NormalizationError, "artifact bytes changed"):
            verifier._normalizer._application(ctx,raw,"selected",sha)

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
        rebased = verifier._rebase_artifacts(value, "capture-mlx/base/lora")
        self.assertEqual(next(verifier._artifact_references(rebased))["path"], "capture-mlx/base/lora/proof.json")
        self.assertEqual(next(verifier._artifact_references(value))["path"], "proof.json")

    def test_rebase_shared_upload_and_api_references_once_without_mutating_producer(self):
        prefix = "capture-mlx/base/lora"
        file = self.root / prefix / "dataset/upload-0.json"
        file.parent.mkdir(parents=True)
        file.write_text('{"fixtureLabel":"SCHEMA TEST ONLY - NOT CAMPAIGN EVIDENCE"}')
        reference = {"path": "dataset/upload-0.json", "sha256": verifier.digest(file)}
        upload = {"uploadReceipt": reference}
        value = {"datasetIdentity": {"uploadedItems": [upload]},
                 "productPath": {"datasetUpload": {"requestArtifact": reference,
                                                   "responseArtifact": reference}}}
        original = copy.deepcopy(value)
        rebased = verifier._rebase_artifacts(value, prefix)
        references = list(verifier._artifact_references(rebased))
        self.assertEqual(len(references), 3)
        self.assertEqual(len({id(item) for item in references}), 3)
        for item in references:
            self.assertEqual(item, {"path": f"{prefix}/dataset/upload-0.json", "sha256": reference["sha256"]})
            self.assertEqual(verifier.checked_artifact(self.root, item), file)
        self.assertEqual(value, original)
        self.assertEqual(upload["uploadReceipt"], reference)
        self.assertEqual(reference["path"], "dataset/upload-0.json")
        cross = verifier._rebase_artifacts(value, "cross-load-cuda/base/lora")
        self.assertTrue(all(item["path"] == "cross-load-cuda/base/lora/dataset/upload-0.json"
                            for item in verifier._artifact_references(cross)))
        self.assertEqual(value, original)

    def test_rebased_artifacts_still_reject_unsafe_paths_and_changed_bytes(self):
        prefix = "capture-mlx/base/lora"
        for unsafe in ("../../../../escape.json", "..\\escape.json", "C:/escape.json"):
            with self.subTest(path=unsafe), self.assertRaises(verifier.EvidenceError):
                verifier.checked_artifact(self.root, verifier._rebase_artifacts(
                    {"path": unsafe, "sha256": "a" * 64}, prefix))
        context = verifier._normalizer._Context(self.root)
        with self.assertRaises(verifier._normalizer.NormalizationError):
            context.checked({"path": "/absolute.json", "sha256": "a" * 64})
        file = self.root / prefix / "proof.json"
        file.parent.mkdir(parents=True)
        file.write_text("original")
        rebased = verifier._rebase_artifacts({"path": "proof.json", "sha256": verifier.digest(file)}, prefix)
        file.write_text("changed")
        with self.assertRaisesRegex(verifier.EvidenceError, "artifact hash mismatch"):
            verifier.checked_artifact(self.root, rebased)

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

    def test_workflow_provenance_preserves_same_run_attempts_and_refuses_other_run_or_source(self):
        receipt = {"fixtureLabel": "SCHEMA TEST ONLY - NOT CAMPAIGN EVIDENCE",
                   "source": {"workflowSource": "a" * 40, "workflowSha256": "b" * 64},
                   "workflow": {"runId": "123", "attempt": 1, "sha": "a" * 40,
                                "fileSha256": "b" * 64, "ref": "refs/heads/SCHEMA_TEST_ONLY", "jobId": "synthetic"}}
        self.assertEqual(verifier.verify_workflow_provenance(receipt, "123"), 1)
        retry = copy.deepcopy(receipt)
        retry["workflow"]["attempt"] = 2
        self.assertEqual(verifier.verify_workflow_provenance(retry, "123"), 2)
        for field, value in (("runId", "124"), ("attempt", 0), ("attempt", "1"), ("sha", "c" * 40), ("fileSha256", "c" * 64)):
            broken = copy.deepcopy(receipt)
            broken["workflow"][field] = value
            with self.assertRaises(verifier.EvidenceError):
                verifier.verify_workflow_provenance(broken, "123")

    def retry_bundle_fixture(self):
        bundles, jobs = {}, []
        for index, (phase, backend, attempt) in enumerate([
                ("capture", "mlx", 1), ("capture", "cuda", 2),
                ("cross-load", "mlx", 2), ("cross-load", "cuda", 2)]):
            directory = self.root / f"{phase}-{backend}"
            directory.mkdir(exist_ok=True)
            receipt = {"source": {"workflowSource": "a" * 40, "workflowSha256": "b" * 64},
                       "workflow": {"runId": "123", "attempt": attempt, "sha": "a" * 40,
                                    "fileSha256": "b" * 64, "ref": "refs/heads/SCHEMA_TEST_ONLY", "jobId": phase},
                       "runnerName": f"SCHEMA_TEST_ONLY-{backend}"}
            (directory / "driver.json").write_text(json.dumps(receipt))
            bundles[(phase, backend)] = (directory, receipt)
            jobs.append({"id": 100 + index, "run_id": 123, "run_attempt": attempt,
                         "head_sha": "a" * 40, "name": f"known caller / MOSS {phase} {backend}",
                         "status": "completed", "conclusion": "success", "runner_name": receipt["runnerName"]})
        inventory = {"repository": "SceneWorks/SceneWorks", "runId": "123", "attempts": [
            {"attempt": attempt, "jobs": [row for row in jobs if row["run_attempt"] == attempt]}
            for attempt in [1, 2]]}
        for backend in ["mlx", "cuda"]:
            opposite = "cuda" if backend == "mlx" else "mlx"
            bundles[("cross-load", backend)][1]["sourceCapture"] = {
                "backend": opposite, "driverSha256": verifier.digest(bundles[("capture", opposite)][0] / "driver.json")}
        return bundles, inventory

    def test_same_run_mixed_attempts_authenticate_and_preserve_each_actual_job(self):
        bundles, inventory = self.retry_bundle_fixture()
        jobs = verifier.authenticate_bundle_jobs(bundles, "123", inventory)
        self.assertEqual([item["attempt"] for item in jobs], [1, 2, 2, 2])
        self.assertEqual([item["jobId"] for item in jobs], [100, 101, 102, 103])
        verifier.verify_capture_dependencies(bundles)
        for _, receipt in bundles.values():
            verifier.verify_inspection_capture_binding({"captureWorkflow": copy.deepcopy(receipt["workflow"])}, receipt)

    def test_job_authentication_rejects_wrong_attempt_job_head_failure_and_runner(self):
        bundles, inventory = self.retry_bundle_fixture()
        for field, value in [("run_id", 124), ("run_attempt", 2), ("head_sha", "c" * 40),
                             ("name", "arbitrary MOSS capture mlx suffix"), ("conclusion", "failure"),
                             ("status", "in_progress"), ("runner_name", "other"), ("id", 0)]:
            with self.subTest(field=field):
                broken = copy.deepcopy(inventory)
                broken["attempts"][0]["jobs"][0][field] = value
                with self.assertRaises(verifier.EvidenceError):
                    verifier.authenticate_bundle_jobs(bundles, "123", broken)
        duplicate = copy.deepcopy(inventory)
        duplicate["attempts"][0]["jobs"].append(copy.deepcopy(duplicate["attempts"][0]["jobs"][0]))
        with self.assertRaisesRegex(verifier.EvidenceError, "ambiguous"):
            verifier.authenticate_bundle_jobs(bundles, "123", duplicate)
        for field, value in [("attempt", 3), ("jobId", "other")]:
            broken = copy.deepcopy(bundles)
            broken[("capture", "mlx")][1]["workflow"][field] = value
            with self.assertRaises(verifier.EvidenceError):
                verifier.authenticate_bundle_jobs(broken, "123", inventory)

    def test_cross_load_rejects_replaced_capture_even_if_adapter_bytes_were_reused(self):
        bundles, _ = self.retry_bundle_fixture()
        verifier.verify_capture_dependencies(bundles)
        driver = bundles[("capture", "mlx")][0] / "driver.json"
        driver.write_text(driver.read_text() + "\n")
        with self.assertRaisesRegex(verifier.EvidenceError, "replaced capture"):
            verifier.verify_capture_dependencies(bundles)
        # The opposite transfer only consumed the unchanged CUDA capture.
        verifier.verify_capture_dependencies({key: value for key, value in bundles.items() if key != ("cross-load", "cuda")})

    def test_inspection_must_name_exact_capture_attempt_job_and_source(self):
        bundles, _ = self.retry_bundle_fixture()
        capture = bundles[("capture", "mlx")][1]
        for field, value in [("attempt", 2), ("jobId", "cross-load"), ("runId", "124"), ("sha", "c" * 40)]:
            review = {"captureWorkflow": copy.deepcopy(capture["workflow"])}
            review["captureWorkflow"][field] = value
            with self.assertRaises(verifier.EvidenceError):
                verifier.verify_inspection_capture_binding(review, capture)
        with self.assertRaises(verifier.EvidenceError):
            verifier.verify_inspection_capture_binding({"attempt": 1}, capture)

    def test_job_retrieval_paginates_all_attempts_and_retains_artifact_metadata(self):
        calls = []
        def request(url):
            calls.append(url)
            if "/artifacts?" in url:
                return {"total_count": 0, "artifacts": []}
            page = int(url.rsplit("=", 1)[1])
            rows = [{"id": index, "run_attempt": 1 + index % 2} for index in range(100)] if page == 1 else []
            return {"total_count": 100, "jobs": rows}
        inventory = verifier.retrieve_workflow_jobs("123", request)
        self.assertEqual(len(calls), 3)
        self.assertEqual([item["attempt"] for item in inventory["attempts"]], [1, 2])
        self.assertTrue(all("filter=all&" in url for url in calls if "/jobs?" in url))
        with self.assertRaisesRegex(verifier.EvidenceError, "incomplete"):
            verifier.retrieve_workflow_jobs("123", lambda _: {"total_count": 1, "jobs": []})
        with self.assertRaisesRegex(verifier.EvidenceError, "ten-page"):
            verifier.retrieve_workflow_jobs("123", lambda _: {"total_count": 1001, "jobs": [{"id": i} for i in range(100)]})

    def test_later_actual_producer_failure_without_artifact_rejects_old_good_bundle(self):
        bundles, inventory = self.retry_bundle_fixture()
        newer = copy.deepcopy(inventory["attempts"][0]["jobs"][0])
        newer.update({"id": 999, "run_attempt": 2, "conclusion": "failure"})
        inventory["attempts"][1]["jobs"].append(newer)
        with self.assertRaisesRegex(verifier.EvidenceError, "newer actual"):
            verifier.authenticate_bundle_jobs(bundles, "123", inventory)
        newer["conclusion"] = "skipped"  # A skipped backend did not replace its successful execution.
        self.assertEqual(len(verifier.authenticate_bundle_jobs(bundles, "123", inventory)), 4)

    def test_selected_artifact_metadata_binds_name_run_head_attempt_and_driver_hash(self):
        bundles, inventory = self.retry_bundle_fixture()
        selection = {"runId": "123", "bundles": []}
        inventory["artifacts"] = []
        for index, ((phase, backend), (directory, receipt)) in enumerate(bundles.items()):
            attempt = receipt["workflow"]["attempt"]
            name = f"{phase}-{backend}-attempt-{attempt}"
            selection["bundles"].append({"bundle": f"{phase}-{backend}", "artifactName": name,
                "attempt": attempt, "driverSha256": verifier.digest(directory / "driver.json")})
            inventory["artifacts"].append({"id": index + 1000, "name": name, "expired": False,
                "digest": "sha256:" + "d" * 64, "workflow_run": {"id": 123, "head_sha": "a" * 40}})
        verified = verifier.authenticate_bundle_artifacts(bundles, "123", inventory, selection)
        self.assertEqual(len(verified), 4)
        self.assertEqual(verified[0]["artifactId"], 1000)
        self.assertEqual(verified[0]["digest"], "sha256:" + "d" * 64)
        for field, value in [("artifactName", "capture-mlx-attempt-2"), ("attempt", 2), ("driverSha256", "0" * 64)]:
            broken = copy.deepcopy(selection)
            broken["bundles"][0][field] = value
            with self.assertRaises(verifier.EvidenceError):
                verifier.authenticate_bundle_artifacts(bundles, "123", inventory, broken)
        for changes in [{"expired": True}, {"workflow_run": {"id": 124, "head_sha": "a" * 40}},
                        {"workflow_run": {"id": 123, "head_sha": "b" * 40}}]:
            broken = copy.deepcopy(inventory)
            broken["artifacts"][0].update(changes)
            with self.assertRaises(verifier.EvidenceError):
                verifier.authenticate_bundle_artifacts(bundles, "123", broken, selection)

    def test_bundle_checks_complete_cases_exact_source_and_member_hashes_before_retry_reuse(self):
        directory = self.root / "capture-mlx"
        directory.mkdir()
        raw = directory / "raw.json"
        raw.write_text("SCHEMA TEST ONLY")
        source = {"manifestSha256": verifier.MANIFEST_SHA256, "finalMergedSource": True,
                  "inferencePin": "a" * 40}
        receipt = {"ok": True, "phase": "capture", "backend": "mlx", "source": source,
                   "cases": [{"caseId": case, "executionPassed": True} for case in verifier.CELLS if case.startswith("mlx/")],
                   "members": [{"path": "raw.json", "sha256": verifier.digest(raw)}]}
        driver = directory / "driver.json"
        driver.write_text(json.dumps(receipt))
        verifier.verify_bundle(self.root, "capture", "mlx", source)
        with self.assertRaisesRegex(verifier.EvidenceError, "source binding"):
            verifier.verify_bundle(self.root, "capture", "mlx", {**source, "inferencePin": "b" * 40})
        receipt["cases"][0]["executionPassed"] = False
        driver.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(verifier.EvidenceError, "completed"):
            verifier.verify_bundle(self.root, "capture", "mlx", source)
        receipt["cases"][0]["executionPassed"] = True
        driver.write_text(json.dumps(receipt))
        raw.write_text("changed")
        with self.assertRaisesRegex(verifier.EvidenceError, "hash mismatch"):
            verifier.verify_bundle(self.root, "capture", "mlx", source)

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

    def test_actual_post_deadline_inspection_remains_legal_before_final_verification(self):
        started = verifier.timestamp("2026-10-25T12:00:00Z")
        ended = verifier.timestamp("2026-10-25T12:01:00Z")
        clock = verifier.timestamp("2026-10-25T14:03:00Z")
        at = verifier.verify_inspection_clock("2026-10-25T14:02:00Z", clock)
        self.assertGreater(at, ended)
        self.assertGreater(at, started + 3600)
        self.assertEqual(verifier.verify_inspection_clock("2026-10-25T14:03:00Z", clock), clock)
        with self.assertRaisesRegex(verifier.EvidenceError, "verification future"):
            verifier.verify_inspection_clock("2026-10-25T14:03:00.001Z", clock)

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

    def test_recovery_exact_content_ignores_only_representation_order(self):
        def write(path, header, payload):
            encoded = json.dumps(header, separators=(",", ":")).encode()
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        # Include NaN payload and signed-zero bits: comparison must stay bitwise.
        a, b = bytes.fromhex("0100c07f"), bytes.fromhex("00000080")
        for role in ("optimizer", "export"):
            with self.subTest(role=role):
                left, right = self.root / (role + "-a"), self.root / (role + "-b")
                header = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                          "b": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
                          "__metadata__": {"optimizer": "adamw", "optim.step": "60"}}
                reordered = {"__metadata__": {"optim.step": "60", "optimizer": "adamw"},
                             "b": {"shape": [1], "data_offsets": [0, 4], "dtype": "F32"},
                             "a": {"shape": [1], "data_offsets": [4, 8], "dtype": "F32"}}
                write(left, header, a + b)
                write(right, reordered, b + a)
                self.assertNotEqual(verifier.digest(left), verifier.digest(right))
                verifier.tensor_payloads_equal(left, right)
                verifier._normalizer.tensor_payloads_equal(left, right)
                for mutation in ("metadata-value", "metadata-key", "metadata-absent", "metadata-extra",
                                 "shape", "dtype", "missing", "extra", "payload"):
                    with self.subTest(mutation=mutation):
                        changed, payload = copy.deepcopy(header), a + b
                        if mutation == "metadata-value":
                            changed["__metadata__"]["optim.step"] = "59"
                        elif mutation == "metadata-key":
                            del changed["__metadata__"]["optimizer"]
                        elif mutation == "metadata-absent":
                            del changed["__metadata__"]
                        elif mutation == "metadata-extra":
                            changed["__metadata__"]["unbound"] = "extra"
                        elif mutation == "shape":
                            changed["a"]["shape"] = [1, 1]
                        elif mutation == "dtype":
                            changed["a"]["dtype"] = "I32"
                        elif mutation == "missing":
                            del changed["b"]
                            payload = a
                        elif mutation == "extra":
                            changed["extra"] = {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]}
                            payload += b
                        else:
                            payload = a + bytes.fromhex("00000000")
                        write(right, changed, payload)
                        with self.assertRaises(verifier.EvidenceError):
                            verifier.tensor_payloads_equal(left, right)
                        with self.assertRaises(verifier._normalizer.NormalizationError):
                            verifier._normalizer.tensor_payloads_equal(left, right)

    def test_recovery_original_bundle_manifests_and_inventory_remain_authenticated(self):
        ctx = verifier._normalizer._Context(self.root)
        for role in ("reference", "resumed"):
            directory = self.root / role
            directory.mkdir()
            (directory / "run.json").write_text('{"step":60,"final_loss":0.013737086}')
            tiny_export(directory / "optimizer.safetensors", [0.25], {"optim.step": "60"})
            tiny_export(directory / "tensors.safetensors", [0.5])
            manifest = {"format": "moss-complete-state-v2", "files": {
                name: [path.stat().st_size, verifier.digest(path)]
                for path in directory.iterdir() for name in [path.name]}}
            (directory / "complete.json").write_text(json.dumps(manifest))
            members = {path.name: ctx.reference(path) for path in directory.iterdir()}
            inventory = {name: ref["sha256"] for name, ref in members.items()}
            checkpoint = {"members": members, "state": {"step": 60, "final_loss": 0.013737086206674576}}
            verifier._normalizer._checked_recovery_bundle(ctx, checkpoint, inventory)
            # Even refreshed outer references cannot conceal a stale native
            # complete.json after one original optimizer payload bit changes.
            path = directory / "optimizer.safetensors"
            changed = bytearray(path.read_bytes())
            changed[-1] ^= 1
            path.write_bytes(changed)
            with self.assertRaises(verifier._normalizer.NormalizationError):
                verifier._normalizer._checked_recovery_bundle(ctx, checkpoint, inventory)
            members[path.name] = ctx.reference(path)
            inventory[path.name] = members[path.name]["sha256"]
            with self.assertRaisesRegex(verifier._normalizer.NormalizationError, "manifest member changed"):
                verifier._normalizer._checked_recovery_bundle(ctx, checkpoint, inventory)

    def test_run_state_loss_compares_exact_native_f32_bits_only(self):
        normalizer = verifier._normalizer
        raw = {"final_loss": 0.013737086, "step": 60, "update": 60, "rng": {"cursor": 60}}
        promoted = copy.deepcopy(raw)
        promoted["final_loss"] = 0.013737086206674576
        self.assertNotEqual(raw, promoted)
        normalizer._require_same_run_state(raw, promoted)
        for sign in (1, -1):
            normalizer._require_same_run_state(
                {"final_loss": sign * 3.4028235e38},
                {"final_loss": sign * 3.4028234663852886e38})
        for changed in (0.013737087, True, float("nan"), float("inf"),
                        3.4028236e38, -3.4028236e38, 1e300):
            promoted["final_loss"] = changed
            with self.assertRaises(normalizer.NormalizationError):
                normalizer._require_same_run_state(raw, promoted)
        with self.assertRaises(normalizer.NormalizationError):
            normalizer._require_same_run_state({"final_loss": 0.0}, {"final_loss": -0.0})
        with self.assertRaises(normalizer.NormalizationError):
            normalizer._require_same_run_state({"final_loss": 1.0}, {"final_loss": struct.unpack("<f", bytes.fromhex("0100803f"))[0]})
        for key, changed in (("step", 59), ("rng", {"cursor": 59}), ("extra", True)):
            promoted = copy.deepcopy(raw)
            promoted[key] = changed
            with self.assertRaises(normalizer.NormalizationError):
                normalizer._require_same_run_state(raw, promoted)

    def test_library_keeps_actual_reference_file_hash_after_logical_recovery_comparison(self):
        runs = {name: {"factorDelta": {"finalExport": {"sha256": value}}}
                for name, value in (("uninterrupted", "a" * 64), ("resumed", "b" * 64))}
        normalizer = verifier._normalizer
        self.assertEqual(normalizer._registered_reference_sha(runs, {"sha256": "a" * 64}), "a" * 64)
        with self.assertRaises(normalizer.NormalizationError):
            normalizer._registered_reference_sha(runs, {"sha256": "b" * 64})

    def test_recovery_exact_comparison_checks_payload_past_streaming_chunk(self):
        left, right = self.root / "large-a", self.root / "large-b"
        payload = b"\x00" * (verifier.CHUNK_BYTES + 4)
        header = {"factor": {"dtype": "F32", "shape": [len(payload) // 4],
                             "data_offsets": [0, len(payload)]}}
        encoded = json.dumps(header).encode()
        container = struct.pack("<Q", len(encoded)) + encoded + payload
        left.write_bytes(container)
        right.write_bytes(container)
        verifier.tensor_payloads_equal(left, right)
        changed = bytearray(container)
        changed[-1] ^= 1
        right.write_bytes(changed)
        with self.assertRaises(verifier.EvidenceError):
            verifier.tensor_payloads_equal(left, right)

    def test_nonfinite_factor_rejected(self):
        left, right = self.root / "a.safetensors", self.root / "b.safetensors"
        tiny_export(left, [0.0])
        tiny_export(right, [float("nan")])
        with self.assertRaises(verifier.EvidenceError):
            verifier.recompute_factor_delta(left, right)

    def test_native_root_controller_receipt_binds_pid_binary_joins_and_closed_stdio(self):
        # Literal files only: never starts an API, process query or model.
        case = self.root / "case"
        case.mkdir()
        entry = {"caseId":"mlx/base/lora", "rawEvidence":"case/evidence.json",
                 "counts":{"passed":1,"failed":0,"ignored":0,"exit_code":0},
                 "caseStartedAt":"2000-01-01T00:00:00Z"}
        identity = {"scope":"native-api-worker-root","pid":123,"executable":"literal-api",
                    "binarySha256":"a"*64}
        state = {"rootProcess":identity,"cleanup":{"scope":"native-api-worker-root",
                 "released":True,"stdioClosed":True,"metadataHelperAbsenceClaim":False}}
        receipt = {"backend":"mlx","executable":{"path":"literal-api","sha256":"a"*64},
                   "source":{"workflowSource":"b"*40,"inferencePin":"c"*40,"manifestSha256":verifier.MANIFEST_SHA256},
                   "preparationStartedAt":"2000-01-01T00:00:00Z","preparationFinishedAt":"2000-01-01T00:00:00Z",
                   "commands":[{"log":"case-base-lora.log","exitCode":0,"supervision":state,
                                "finishedAt":"2000-01-01T00:00:02Z"}]}
        preflight = {**receipt["source"],"caseId":entry["caseId"],"backend":"mlx","phase":"capture",
                     "verdict":"admitted","refusalReason":None,"offline":True,"weightsLoaded":False,
                     "modelRevisions":{"literal":"d"*40},"resolvedCapProvenance":"literal",
                     "headerDemandProvenance":"literal","weights":[{"present":True}],"resolvedCapBytes":600}
        native = {"raw":{"serverJoined":True,"nativeBodyReturnedAtUnixMs":946684801000,
                  "rootProcess":{"scope":"native-api-worker-root","pid":123,
                  "measurement":"native-current-process-rss-sampled-maximum","metadataHelperAbsenceClaim":False,
                  "firstFailure":None,"missingMeasurementCount":0,"resolvedCeilingBytes":600,
                  "sampledPeakBytes":250,"samples":copy.deepcopy(fixtures.record["cleanup"]["rootRssSamples"])}}}
        (case / "devicepreflight.json").write_text(json.dumps(preflight))
        for mutation in ("valid","pid","binary","join","stdio","cap","missing","claim","canceled"):
            r, n = copy.deepcopy(receipt), copy.deepcopy(native)
            if mutation == "pid": n["raw"]["rootProcess"]["pid"] = 456
            if mutation == "binary": r["commands"][0]["supervision"]["rootProcess"]["binarySha256"] = "f"*64
            if mutation == "join": n["raw"]["serverJoined"] = False
            if mutation == "stdio": r["commands"][0]["supervision"]["cleanup"]["stdioClosed"] = False
            if mutation == "cap": n["raw"]["rootProcess"]["resolvedCeilingBytes"] = 601
            if mutation == "missing": n["raw"]["rootProcess"]["missingMeasurementCount"] = 1
            if mutation == "claim": n["raw"]["rootProcess"]["metadataHelperAbsenceClaim"] = True
            if mutation == "canceled": r["commands"][0]["supervision"]["cancelled"] = True
            (case / "evidence.json").write_text(json.dumps(n))
            (self.root / "driver.json").write_text(json.dumps(r))
            if mutation == "valid":
                guard, _, _ = verifier.controller_observations(self.root, r, entry, "capture")
                verifier.verify_rss(guard, 600, verifier.timestamp(entry["caseStartedAt"]), verifier.timestamp(r["commands"][0]["finishedAt"]))
            else:
                with self.assertRaises(verifier.EvidenceError, msg=mutation):
                    verifier.controller_observations(self.root, r, entry, "capture")

    def test_missing_and_zero_rss_fail_closed(self):
        guard = copy.deepcopy(fixtures.record["cleanup"])
        start = verifier.timestamp(guard["caseStartedAt"])
        end = verifier.timestamp(guard["caseEndedAt"])
        verifier.verify_rss(guard, 600, start, end)
        for mutation in ("empty", "zero", "missing"):
            broken = copy.deepcopy(guard)
            if mutation == "empty":
                broken["rootRssSamples"] = []
            elif mutation == "zero":
                broken["rootRssSamples"][0]["bytes"] = 0
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

    def test_cross_load_has_independent_import_clock(self):
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
