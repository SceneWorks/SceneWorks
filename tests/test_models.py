from __future__ import annotations

from sceneworks_api.models import load_manifest, strip_jsonc_comments


def test_jsonc_comment_stripping_preserves_url_strings():
    payload = '{"repo":"https://example.test/model", "name":"ok"} // trailing comment\n'

    assert strip_jsonc_comments(payload) == '{"repo":"https://example.test/model", "name":"ok"} \n'


def test_manifest_cache_reloads_when_mtime_changes(tmp_path):
    manifest = tmp_path / "models.jsonc"
    manifest.write_text('{"models":[{"id":"first"}]}', encoding="utf-8")

    assert load_manifest(manifest) == [{"id": "first"}]

    manifest.write_text('{"models":[{"id":"second"}]}', encoding="utf-8")
    assert load_manifest(manifest) == [{"id": "second"}]
