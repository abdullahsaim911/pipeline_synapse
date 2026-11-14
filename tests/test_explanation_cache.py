import json

from explanation_cache import cache_is_current, generation_manifest, manifest_path


def test_cache_requires_matching_version_text_audio_evidence_and_inputs(tmp_path):
    audio = tmp_path / "brief.mp3"
    snapshot = tmp_path / "vlm_snapshot.json"
    audio.write_bytes(b"ID3first audio")
    snapshot.write_text('{"evidence": "first"}')
    assert not cache_is_current("First", audio, snapshot)
    manifest = generation_manifest("First", audio, snapshot, input_hash="inputs")
    manifest_path(audio).write_text(json.dumps(manifest))
    assert cache_is_current("First", audio, snapshot)
    assert cache_is_current("First", audio, snapshot, input_hash="inputs")
    assert not cache_is_current("First", audio, snapshot, input_hash="changed")
    assert not cache_is_current("Other", audio, snapshot)
    audio.write_bytes(b"ID3other audio")
    assert not cache_is_current("First", audio, snapshot)
    audio.write_bytes(b"ID3first audio")
    snapshot.write_text('{"evidence": "changed"}')
    assert not cache_is_current("First", audio, snapshot)
    snapshot.write_text('{"evidence": "first"}')
    manifest["quality_version"] = "old"
    manifest_path(audio).write_text(json.dumps(manifest))
    assert not cache_is_current("First", audio, snapshot)


def test_corrupt_manifest_is_a_cache_miss(tmp_path):
    audio = tmp_path / "brief.mp3"
    audio.write_bytes(b"ID3audio")
    for contents in ('{', '[]', 'null'):
        manifest_path(audio).write_text(contents)
        assert not cache_is_current("Text", audio)


def test_provider_or_model_change_invalidates_completed_cache(tmp_path, monkeypatch):
    audio = tmp_path / "brief.mp3"
    audio.write_bytes(b"ID3audio")
    monkeypatch.setenv("SYNAPSE_EXPLANATION_PROVIDER", "ollama")
    manifest_path(audio).write_text(json.dumps(generation_manifest("Text", audio)))
    assert cache_is_current("Text", audio)
    monkeypatch.setenv("SYNAPSE_EXPLANATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "first-configured-model")
    assert not cache_is_current("Text", audio)
    manifest_path(audio).write_text(json.dumps(generation_manifest("Text", audio)))
    assert cache_is_current("Text", audio)
    monkeypatch.setenv("GEMINI_MODEL", "second-configured-model")
    assert not cache_is_current("Text", audio)
