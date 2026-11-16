"""Metadata-only checks for the Docker FlashAttention fetcher."""

import importlib.util
import errno
import pathlib
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).with_name("fetch-flash-prebuilt.py")
SPEC = importlib.util.spec_from_file_location("fetch_flash_prebuilt", SCRIPT)
FETCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCHER)


class FlashPrebuiltTests(unittest.TestCase):
    def test_package_validation_rejects_corruption_and_source_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            archive = directory / "libflashattention.a"
            archive.write_bytes(b"archive fixture")
            trusted = FETCHER.sha256(archive)
            for name in FETCHER.FILES - {"libflashattention.a", "flash-attn-prebuilt.txt"}:
                (directory / name).write_text("license fixture")
            source = "a" * 64
            (directory / "flash-attn-prebuilt.txt").write_text(
                f"format=1\nsource_sha256={source}\ntarget={FETCHER.TARGET}\n"
                f"archive_sha256={trusted}\n"
            )
            FETCHER.validate(directory, source, trusted)
            with mock.patch.object(pathlib.Path, "rename", side_effect=OSError(errno.ENOTEMPTY, "winner published")):
                FETCHER.publish(directory.parent / "candidate", directory, source, trusted)
            with mock.patch.object(pathlib.Path, "rename", side_effect=OSError(errno.EACCES, "permission denied")):
                with self.assertRaises(OSError):
                    FETCHER.publish(directory.parent / "candidate", directory, source, trusted)
            with self.assertRaisesRegex(ValueError, "source or target mismatch"):
                FETCHER.validate(directory, "b" * 64, trusted)
            archive.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                FETCHER.validate(directory, source, trusted)
            (directory / "flash-attn-prebuilt.txt").write_text(
                f"format=1\nsource_sha256={source}\ntarget={FETCHER.TARGET}\n"
                f"archive_sha256={FETCHER.sha256(archive)}\n"
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                FETCHER.validate(directory, source, trusted)


if __name__ == "__main__":
    unittest.main()
