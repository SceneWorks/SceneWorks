from __future__ import annotations

from sceneworks_api.models import (
    download_size_from_siblings,
    format_bytes,
    load_manifest,
    model_is_installed,
    model_install_marker,
    strip_jsonc_comments,
)


def test_jsonc_comment_stripping_preserves_url_strings():
    payload = '{"repo":"https://example.test/model", "name":"ok"} // trailing comment\n'

    assert strip_jsonc_comments(payload) == '{"repo":"https://example.test/model", "name":"ok"} \n'


def test_manifest_cache_reloads_when_mtime_changes(tmp_path):
    manifest = tmp_path / "models.jsonc"
    manifest.write_text('{"models":[{"id":"first"}]}', encoding="utf-8")

    assert load_manifest(manifest) == [{"id": "first"}]

    manifest.write_text('{"models":[{"id":"second"}]}', encoding="utf-8")
    assert load_manifest(manifest) == [{"id": "second"}]


def test_partial_model_directory_is_not_installed(tmp_path):
    model_dir = tmp_path / "models" / "owner__model"
    model_dir.mkdir(parents=True)
    (model_dir / "partial.bin").write_bytes(b"partial")

    assert not model_is_installed(model_dir)


def test_model_directory_with_completion_marker_is_installed(tmp_path):
    model_dir = tmp_path / "models" / "owner__model"
    model_dir.mkdir(parents=True)
    model_install_marker(model_dir).write_text("{}", encoding="utf-8")

    assert model_is_installed(model_dir)


def test_download_size_from_siblings_respects_allow_patterns():
    siblings = [
        {"rfilename": "model-00001.safetensors", "size": 100},
        {"rfilename": "model-00002.safetensors", "size": 200},
        {"rfilename": "README.md", "size": 50},
    ]

    assert download_size_from_siblings(siblings, ["*.safetensors"]) == 300


def test_download_size_from_siblings_returns_none_when_unknown():
    assert download_size_from_siblings([{"rfilename": "model.bin"}]) is None


def test_format_bytes_for_model_catalog():
    assert format_bytes(None) is None
    assert format_bytes(0) == "0 B"
    assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
