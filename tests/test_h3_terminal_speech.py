"""Stdlib-only tests for scripts/h3-finetune-terminal-speech.py.

The receipt under test is the REAL `h3-finetune-stage-receipt` v1 shape produced by
`scripts/h3-finetune-terminal-stage.mjs` (settings.privateRoot, files[].repoType/
repo/revision/file/status/actualBytes/observedSha256/finalPath/expectedLegal/
explicitAcknowledgment, root licenseAuthority). No pyarrow install, no real audio,
ffmpeg/ffprobe/network/listeners/GPU: the reader, subprocess, tool resolver, probe
metadata and byte outputs are injected. Fake receipts are unit_fake and are rejected as
campaign proof.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "h3-finetune-terminal-speech.py"
_spec = importlib.util.spec_from_file_location("h3_finetune_terminal_speech", MODULE_PATH)
speech = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(speech)

ROW_ID = "1272-128104-0000"
DATASET_REPO = "hf-internal-testing/librispeech_asr_dummy"
DATASET_REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
PARQUET = "clean/validation-00000-of-00001.parquet"
FAKE_PARQUET_BYTES = b"PARQUET-fixture"  # 15 bytes; the fake receipt declares this exact size
FAKE_PARQUET_SHA = hashlib.sha256(FAKE_PARQUET_BYTES).hexdigest()
FAKE_AUDIO = b"\x01\x02\x03\x04fake-flac"
FAKE_WAV = b"RIFF\x00\x00fake-pcm16"
ATTRIBUTION = "LibriSpeech (OpenSLR), Panayotov et al., CC BY 4.0"
EXPECTED_COLUMNS = ["id", "audio", "text", "speaker_id"]


def cache_relative() -> str:
    return "/".join(
        ["hf", "hub", f"datasets--{DATASET_REPO.replace('/', '--')}", "snapshots", DATASET_REVISION, *PARQUET.split("/")]
    )


def fake_expected() -> dict:
    expected = copy.deepcopy(speech.EXPECTED)
    expected["parquet_bytes"] = len(FAKE_PARQUET_BYTES)
    expected["parquet_sha256"] = FAKE_PARQUET_SHA
    return expected


def base_receipt(stage_root: str, *, proof_kind: str = "frozen_stage_content", authority_kind: str = "user-approved") -> dict:
    acknowledgment = {
        "sha256": FAKE_PARQUET_SHA,
        "acknowledged": True,
        "licenseId": "CC-BY-4.0",
        "attribution": ATTRIBUTION,
    }
    return {
        "version": 1,
        "kind": "h3-finetune-stage-receipt",
        "proof_kind": proof_kind,
        "campaign_proof": False,
        "producer": {"script": "scripts/h3-finetune-terminal-stage.mjs", "runId": "1", "runAttempt": "1"},
        "settings": {"privateRoot": stage_root},
        "licenseAuthority": {
            "kind": authority_kind,
            "approvedBy": "coordinator",
            "approvedAt": "2026-10-25T00:00:00Z",
            "reference": "slack://approved-ccby4",
            "subjects": [{"sha256": FAKE_PARQUET_SHA, "acknowledged": True, "licenseId": "CC-BY-4.0", "attribution": ATTRIBUTION}],
            "acknowledged": True,
        },
        "status": "completed",
        "files": [
            {
                "repoType": "dataset",
                "repo": DATASET_REPO,
                "revision": DATASET_REVISION,
                "file": PARQUET,
                "url": "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/x/clean/validation-00000-of-00001.parquet",
                "status": "completed",
                "actualBytes": len(FAKE_PARQUET_BYTES),
                "observedSha256": FAKE_PARQUET_SHA,
                "finalPath": str(Path(stage_root) / cache_relative()),
                "expectedLegal": {"licenseId": "CC-BY-4.0", "attributionRequired": ATTRIBUTION},
                "explicitAcknowledgment": acknowledgment,
            }
        ],
    }


def rows_fixture(count: int = 1, mutate=None) -> list[dict]:
    def row(index: int) -> dict:
        return {
            "id": ROW_ID if index == 0 else f"0000-999999-999{index}",
            "audio": {"bytes": FAKE_AUDIO, "path": "1272-128104-0000.flac"},
            "text": "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES",
            "speaker_id": 1272,
        }

    rows = [row(i) for i in range(count)]
    if mutate is not None:
        mutate(rows)
    return rows


def reader_fixture(rows):
    seen = {}

    def reader(path, columns):
        seen["columns"] = list(columns)
        return [copy.deepcopy(row) for row in rows]

    reader.columns_seen = seen
    return reader


def probe_fixture(*, reversed_ts=None, omit_timing=False, reversed_duration=None, **overrides):
    def prober(path):
        is_reversed = Path(path).name == "speech-reversed.wav"
        duration_ts = 48_000
        if is_reversed and reversed_ts is not None:
            duration_ts = reversed_ts
        duration = reversed_duration if (is_reversed and reversed_duration is not None) else duration_ts / 32_000
        stream = {
            "codec_name": "pcm_s16le",
            "sample_rate": 32_000,
            "channels": 1,
            "duration": duration,
            "nb_read_frames": 48,
        }
        if not omit_timing:
            stream["duration_ts"] = duration_ts
            stream["time_base"] = "1/32000"
        stream.update(overrides)
        return stream

    return prober


def runner_fixture(fail_on=None):
    calls = []

    def runner(argv, cwd=None):
        assert isinstance(argv, list) and argv
        assert all(isinstance(item, str) for item in argv)
        calls.append(list(argv))
        if fail_on is not None and any(fail_on in item for item in argv):
            return SimpleNamespace(returncode=1, stdout="", stderr="injected failure")
        if "-version" in argv:
            tool = "ffprobe" if "ffprobe" in argv[0] else "ffmpeg"
            return SimpleNamespace(returncode=0, stdout=f"{tool} version 7.1-fake\n", stderr="")
        if "-f" in argv and "null" in argv:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        target = Path(argv[-1])
        target.write_bytes(FAKE_WAV + b"reversed" if target.name == "speech-reversed.wav" else FAKE_WAV)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    runner.calls = calls
    return runner


def resolver_fixture():
    def resolver(name, root):
        return os.path.join(os.sep, "trusted", "bin", name)

    return resolver


def stage(tmp: tempfile.TemporaryDirectory, *, rows=None, receipt=None, output_dir=None, expected=None, staged_bytes=None, prepare=True, **overrides):
    stage_root = Path(tmp.name)
    staged = Path(stage_root) / cache_relative()
    staged.parent.mkdir(parents=True, exist_ok=True)
    if prepare:
        staged.write_bytes(FAKE_PARQUET_BYTES if staged_bytes is None else staged_bytes)
    defaults = {
        "expected": expected or fake_expected(),
        "reader": reader_fixture(rows_fixture() if rows is None else rows),
        "tool_resolver": resolver_fixture(),
        "runner": runner_fixture(),
        "prober": probe_fixture(),
        "allow_unit_fake": True,
    }
    defaults.update(overrides)
    return speech.stage_speech(
        receipt or base_receipt(str(stage_root)),
        str(stage_root),
        output_dir,
        **defaults,
    )


class SpeechStageTests(unittest.TestCase):
    def test_valid_real_receipt_produces_complete_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name))
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["proof_kind"], "unit_fake")
            self.assertEqual(receipt["method"], "generated-not-model-proof")
            self.assertEqual(receipt["asset"]["id"], ROW_ID)
            self.assertEqual(receipt["source"]["parquet"]["finalPath"], cache_relative())
            self.assertEqual(receipt["source"]["parquet"]["sha256"], FAKE_PARQUET_SHA)
            self.assertEqual(receipt["source"]["row"], ROW_ID)
            self.assertTrue(receipt["source"]["transcript"])
            self.assertEqual(receipt["source"]["speakerId"], 1272)
            self.assertEqual(receipt["source"]["sourceAudio"]["sha256"], hashlib.sha256(FAKE_AUDIO).hexdigest())
            self.assertEqual(receipt["license"]["expectedLegal"]["licenseId"], "CC-BY-4.0")
            self.assertEqual(receipt["license"]["explicitAcknowledgment"]["attribution"], ATTRIBUTION)
            self.assertEqual(receipt["license"]["licenseAuthority"]["kind"], "user-approved")
            self.assertTrue(receipt["license"]["explicitInput"])
            self.assertFalse(receipt["license"]["mirrorLicenseDeclared"])
            self.assertIsNone(receipt["tools"]["pyarrow"]["version"])
            self.assertEqual(receipt["tools"]["pyarrow"]["source"], "injected-reader")
            roles = [output["role"] for output in receipt["outputs"]]
            self.assertEqual(roles, ["speech-source", "diagnostic-reversed-not-speaker"])
            self.assertEqual(receipt["outputs"][0]["samples"], 48_000)
            self.assertEqual(receipt["durationSeconds"], 1.5)
            receipt_path = Path(name) / "conditioning" / f"speech-{ROW_ID}" / "speech-receipt.json"
            self.assertEqual(json.loads(receipt_path.read_text()), receipt)

    def test_reader_receives_exact_columns(self):
        with tempfile.TemporaryDirectory() as name:
            reader = reader_fixture(rows_fixture())
            stage(SimpleNamespace(name=name), reader=reader)
            self.assertEqual(reader.columns_seen["columns"], EXPECTED_COLUMNS)

    def test_unit_fake_cannot_be_campaign_proof(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name))
            with self.assertRaises(speech.SpeechStageError) as caught:
                speech.assert_campaign_receipt(receipt)
            self.assertEqual(caught.exception.code, "proof")
            self.assertIn("unit_fake", caught.exception.message)

    def test_campaign_native_receipt_passes(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name), proof_kind="campaign_native")
            self.assertEqual(speech.assert_campaign_receipt(receipt), receipt)

    def test_unit_fake_receipt_refused_by_default(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = base_receipt(name, proof_kind="unit_fake", authority_kind="unit_fake")
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), receipt=receipt, allow_unit_fake=False)
            self.assertEqual(caught.exception.code, "proof")

    def test_unit_fake_authority_refused_without_test_flag(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = base_receipt(name, authority_kind="unit_fake")
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), receipt=receipt, allow_unit_fake=False)
            self.assertEqual(caught.exception.code, "license")

    def test_missing_license_approval_fields_fail(self):
        for path in (("kind",), ("approvedBy",), ("approvedAt",), ("reference",), ("subjects",), ("acknowledged",)):
            with tempfile.TemporaryDirectory() as name:
                receipt = copy.deepcopy(base_receipt(name))
                node = receipt["licenseAuthority"]
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = None
                with self.assertRaises(speech.SpeechStageError) as caught:
                    stage(SimpleNamespace(name=name), receipt=receipt)
                self.assertEqual(caught.exception.code, "license")

    def test_wrong_receipt_kind_version_or_status_fails(self):
        for mutate in (
            lambda r: r.__setitem__("kind", "invented"),
            lambda r: r.__setitem__("version", 2),
            lambda r: r.__setitem__("status", "running"),
        ):
            with tempfile.TemporaryDirectory() as name:
                receipt = copy.deepcopy(base_receipt(name))
                mutate(receipt)
                with self.assertRaises(speech.SpeechStageError) as caught:
                    stage(SimpleNamespace(name=name), receipt=receipt)
                self.assertEqual(caught.exception.code, "receipt")

    def test_wrong_private_root_fails(self):
        with tempfile.TemporaryDirectory() as name, tempfile.TemporaryDirectory() as other:
            receipt = base_receipt(other)
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), receipt=receipt)
            self.assertEqual(caught.exception.code, "receipt")

    def test_missing_exact_row_fails(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), rows=rows_fixture(mutate=lambda rows: rows[0].__setitem__("id", "9999-999999-9999")))
            self.assertEqual(caught.exception.code, "row")

    def test_duplicate_row_fails(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), rows=rows_fixture() + rows_fixture())
            self.assertEqual(caught.exception.code, "row")

    def test_empty_audio_fails(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), rows=rows_fixture(mutate=lambda rows: rows[0]["audio"].__setitem__("bytes", b"")))
            self.assertEqual(caught.exception.code, "audio")

    def test_missing_transcript_or_speaker_fails(self):
        for mutate in (
            lambda rows: rows[0].__setitem__("text", "  "),
            lambda rows: rows[0].__setitem__("speaker_id", None),
        ):
            with tempfile.TemporaryDirectory() as name:
                with self.assertRaises(speech.SpeechStageError) as caught:
                    stage(SimpleNamespace(name=name), rows=rows_fixture(mutate=mutate))
                self.assertEqual(caught.exception.code, "row")

    def test_missing_or_wrong_dataset_row_fails(self):
        for mutate in (
            lambda row: row.__setitem__("repoType", "model"),
            lambda row: row.__setitem__("repo", "other/repo"),
            lambda row: row.__setitem__("revision", "0" * 40),
            lambda row: row.__setitem__("file", "clean/other.parquet"),
            lambda row: row.__setitem__("status", "pending"),
            lambda row: row.__setitem__("actualBytes", len(FAKE_PARQUET_BYTES) + 1),
            lambda row: row.__setitem__("observedSha256", "0" * 64),
        ):
            with tempfile.TemporaryDirectory() as name:
                receipt = copy.deepcopy(base_receipt(name))
                mutate(receipt["files"][0])
                with self.assertRaises(speech.SpeechStageError):
                    stage(SimpleNamespace(name=name), receipt=receipt)

    def test_expected_legal_requires_ccby4_and_raw_credit(self):
        for mutate in (
            lambda legal: legal.__setitem__("licenseId", "MIT"),
            lambda legal: legal.__setitem__("attributionRequired", ""),
            lambda legal: legal.pop("attributionRequired"),
        ):
            with tempfile.TemporaryDirectory() as name:
                receipt = copy.deepcopy(base_receipt(name))
                mutate(receipt["files"][0]["expectedLegal"])
                with self.assertRaises(speech.SpeechStageError) as caught:
                    stage(SimpleNamespace(name=name), receipt=receipt)
                self.assertEqual(caught.exception.code, "license")

    def test_explicit_acknowledgment_must_match_and_credit(self):
        mutations = (
            lambda ack: ack.__setitem__("acknowledged", False),
            lambda ack: ack.__setitem__("sha256", "0" * 64),
            lambda ack: ack.__setitem__("licenseId", "MIT"),
            lambda ack: ack.__setitem__("attribution", "unrelated"),
            lambda ack: ack.pop("attribution"),
        )
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as name:
                receipt = copy.deepcopy(base_receipt(name))
                mutate(receipt["files"][0]["explicitAcknowledgment"])
                with self.assertRaises(speech.SpeechStageError) as caught:
                    stage(SimpleNamespace(name=name), receipt=receipt)
                self.assertEqual(caught.exception.code, "license")

    def test_inherited_license_requires_inherited_acknowledgment(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = copy.deepcopy(base_receipt(name))
            receipt["files"][0]["expectedLegal"]["inheritedLicenseId"] = "CC-BY-4.0"
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), receipt=receipt)
            self.assertEqual(caught.exception.code, "license")
        with tempfile.TemporaryDirectory() as name:
            receipt = copy.deepcopy(base_receipt(name))
            receipt["files"][0]["expectedLegal"]["inheritedLicenseId"] = "CC-BY-4.0"
            receipt["files"][0]["explicitAcknowledgment"]["inheritedLicenseId"] = "CC-BY-4.0"
            receipt["files"][0]["explicitAcknowledgment"]["inheritedAcknowledged"] = True
            self.assertEqual(stage(SimpleNamespace(name=name), receipt=receipt)["status"], "completed")

    def test_final_path_must_be_under_root_and_pinned_cache_layout(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = copy.deepcopy(base_receipt(name))
            receipt["files"][0]["finalPath"] = str(Path(name) / "clean" / "validation-00000-of-00001.parquet")
            alt = Path(receipt["files"][0]["finalPath"])
            alt.parent.mkdir(parents=True, exist_ok=True)
            alt.write_bytes(FAKE_PARQUET_BYTES)
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), receipt=receipt)
            self.assertEqual(caught.exception.code, "path")

    def test_final_path_outside_root_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as name, tempfile.TemporaryDirectory() as outside:
            receipt = copy.deepcopy(base_receipt(name))
            receipt["files"][0]["finalPath"] = str(Path(outside) / "validation.parquet")
            Path(receipt["files"][0]["finalPath"]).write_bytes(FAKE_PARQUET_BYTES)
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), receipt=receipt)
            self.assertEqual(caught.exception.code, "path")
        with tempfile.TemporaryDirectory() as name, tempfile.TemporaryDirectory() as outside:
            outside_file = Path(outside) / "validation.parquet"
            outside_file.write_bytes(FAKE_PARQUET_BYTES)
            staged = Path(name) / cache_relative()
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.symlink_to(outside_file)
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), prepare=False)
            self.assertEqual(caught.exception.code, "path")

    def test_traversal_refused(self):
        with tempfile.TemporaryDirectory() as name:
            for relative in ("../escape.parquet", "clean/../../escape.parquet", "/etc/passwd"):
                with self.assertRaises(speech.SpeechStageError) as caught:
                    speech._confined_existing(name, relative, "test path")
                self.assertEqual(caught.exception.code, "path")

    def test_wrong_actual_bytes_fails(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), staged_bytes=FAKE_PARQUET_BYTES + b"tampered")
            self.assertEqual(caught.exception.code, "input")

    def test_existing_output_dir_refused(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name) / "conditioning" / f"speech-{ROW_ID}"
            output.mkdir(parents=True)
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), output_dir=str(output))
            self.assertEqual(caught.exception.code, "output")

    def test_output_dir_outside_stage_root_refused(self):
        with tempfile.TemporaryDirectory() as outer, tempfile.TemporaryDirectory() as inner:
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=inner), output_dir=os.path.join(outer, "conditioning"))
            self.assertEqual(caught.exception.code, "path")

    def test_ffmpeg_failure_writes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            runner = runner_fixture(fail_on="speech-32000.wav")
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), runner=runner)
            self.assertEqual(caught.exception.code, "tool")
            receipt_path = Path(name) / "conditioning" / f"speech-{ROW_ID}" / "speech-receipt.json"
            written = json.loads(receipt_path.read_text())
            self.assertEqual(written["status"], "failed")
            self.assertEqual(written["failure"]["code"], "tool")
            self.assertEqual(written["proof_kind"], "unit_fake")

    def test_probe_wrong_sample_rate_channels_or_duration_fails(self):
        for overrides in ({"sample_rate": 44100}, {"channels": 2}, {"duration": 0.0}):
            with tempfile.TemporaryDirectory() as name:
                with self.assertRaises(speech.SpeechStageError) as caught:
                    stage(SimpleNamespace(name=name), prober=probe_fixture(**overrides))
                self.assertEqual(caught.exception.code, "probe")

    def test_areverse_must_preserve_exact_sample_count(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), prober=probe_fixture(reversed_ts=47_999))
            self.assertEqual(caught.exception.code, "probe")
            self.assertIn("sample count", caught.exception.message)

    def test_duration_fallback_allows_one_sample_but_not_two(self):
        with tempfile.TemporaryDirectory() as name:
            one_sample = probe_fixture(omit_timing=True, reversed_duration=1.5 + 1 / 32_000)
            self.assertEqual(stage(SimpleNamespace(name=name), prober=one_sample)["status"], "completed")
        with tempfile.TemporaryDirectory() as name:
            two_samples = probe_fixture(omit_timing=True, reversed_duration=1.5 + 2 / 32_000)
            with self.assertRaises(speech.SpeechStageError) as caught:
                stage(SimpleNamespace(name=name), prober=two_samples)
            self.assertEqual(caught.exception.code, "probe")

    def test_full_decode_pass_runs_for_both_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            runner = runner_fixture()
            stage(SimpleNamespace(name=name), runner=runner)
            decode_passes = [argv for argv in runner.calls if "-f" in argv and "null" in argv]
            self.assertEqual(len(decode_passes), 2)

    def test_reversed_output_is_a_diagnostic_not_a_speaker(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name))
            source_output, reversed_output = receipt["outputs"]
            self.assertEqual(source_output["role"], "speech-source")
            self.assertEqual(reversed_output["role"], "diagnostic-reversed-not-speaker")
            self.assertNotEqual(reversed_output["sha256"], source_output["sha256"])

    def test_areverse_only_single_stream(self):
        with tempfile.TemporaryDirectory() as name:
            runner = runner_fixture()
            stage(SimpleNamespace(name=name), runner=runner)
            areverse = [argv for argv in runner.calls if "areverse" in argv]
            self.assertEqual(len(areverse), 1)
            self.assertTrue(areverse[0][-1].endswith("speech-reversed.wav"))

    def test_production_guard_accepts_approved_runners(self):
        for env in (
            {"RUNNER_OS": "macOS", "RUNNER_NAME": "nax-macos-2"},
            {"RUNNER_OS": "Windows", "RUNNER_NAME": "cuda-windows"},
            {"RUNNER_OS": "Windows", "RUNNER_NAME": "cuda-windows-2"},
        ):
            full = {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "workflow_dispatch", "RUNNER_TEMP": "/tmp", **env}
            self.assertEqual(speech.require_production_context(env=full)["runner"], env["RUNNER_NAME"])

    def test_production_guard_rejects_local_and_dev(self):
        base = {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "workflow_dispatch", "RUNNER_TEMP": "/tmp"}
        cases = [
            {"RUNNER_OS": "macOS", "RUNNER_NAME": "michaels-dev-mac"},
            {"RUNNER_OS": "Windows", "RUNNER_NAME": "cuda-windows-3"},
            {"RUNNER_OS": "Linux", "RUNNER_NAME": "ubuntu-latest"},
            {},
        ]
        for runner in cases:
            with self.assertRaises(speech.SpeechStageError) as caught:
                speech.require_production_context(env={**base, **runner})
            self.assertEqual(caught.exception.code, "guard")
        for env in ({**base, "GITHUB_EVENT_NAME": "push"}, {**base, "GITHUB_ACTIONS": "false"}):
            with self.assertRaises(speech.SpeechStageError) as caught:
                speech.require_production_context(env=env)
            self.assertEqual(caught.exception.code, "guard")

    def test_pyarrow_is_imported_only_inside_reader(self):
        self.assertNotIn("pyarrow", sys.modules)
        self.assertNotIn("pyarrow.parquet", sys.modules)
        with mock.patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}):
            with self.assertRaises(speech.SpeechStageError) as caught:
                speech._pyarrow_rows("/unused", speech.EXPECTED["columns"])
            self.assertEqual(caught.exception.code, "tool")
            self.assertIn(speech.PYARROW_REQUIRED, caught.exception.message)

    def test_pyarrow_version_must_be_exact(self):
        fake_parquet = SimpleNamespace(read_table=lambda path, columns: SimpleNamespace(to_pylist=lambda: rows_fixture()))
        wrong = SimpleNamespace(__version__="19.1.0")
        right = SimpleNamespace(__version__="19.0.1", parquet=fake_parquet)
        with mock.patch.dict(sys.modules, {"pyarrow": wrong, "pyarrow.parquet": fake_parquet}):
            with self.assertRaises(speech.SpeechStageError) as caught:
                speech._pyarrow_rows("/unused", speech.EXPECTED["columns"])
            self.assertEqual(caught.exception.code, "tool")
            self.assertIn(speech.PYARROW_REQUIRED, caught.exception.message)
        with mock.patch.dict(sys.modules, {"pyarrow": right, "pyarrow.parquet": fake_parquet}):
            rows, version = speech._pyarrow_rows("/unused", speech.EXPECTED["columns"])
            self.assertEqual(version, "19.0.1")
            self.assertEqual(rows[0]["id"], ROW_ID)

    def test_injected_reader_does_not_claim_installed_pyarrow_version(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name))
            self.assertEqual(receipt["tools"]["pyarrow"], {"version": None, "source": "injected-reader"})

    def test_production_run_uses_no_shell(self):
        captured = {}

        def fake_run(argv, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.object(speech.subprocess, "run", side_effect=fake_run) as patched:
            speech._run(["/usr/bin/true", "arg"])
        patched.assert_called_once()
        self.assertIs(captured.get("shell"), False)

    def test_trusted_tool_refusals(self):
        with tempfile.TemporaryDirectory() as name:
            staged_tool = Path(name) / "ffmpeg"
            staged_tool.write_bytes(b"")
            with mock.patch.object(speech.shutil, "which", return_value=str(staged_tool)):
                with self.assertRaises(speech.SpeechStageError) as caught:
                    speech.trusted_tool("ffmpeg", name)
            self.assertEqual(caught.exception.code, "tool")
        with tempfile.TemporaryDirectory() as name:
            with mock.patch.object(speech.shutil, "which", return_value=None):
                with self.assertRaises(speech.SpeechStageError) as caught:
                    speech.trusted_tool("ffmpeg", name)
            self.assertEqual(caught.exception.code, "tool")


if __name__ == "__main__":
    unittest.main()
