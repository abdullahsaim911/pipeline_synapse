"""Versioned provenance for shared generated artifacts (personal saves are separate)."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


QUALITY_VERSION = "2026-11-14-grounded-v6"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_path(audio_path):
    return Path(audio_path).with_suffix(".generation.json")


def write_manifest(audio_path, provenance):
    path = manifest_path(audio_path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(provenance, handle, ensure_ascii=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generation_manifest(text, audio_path, snapshot_path=None, *, input_hash=None):
    from synthesizer.configuration import explanation_profile
    from tts_engine.providers import tts_profile
    return {
        "quality_version": QUALITY_VERSION,
        "text_generator": explanation_profile(),
        "tts": tts_profile(),
        "text_hash": fingerprint(text.strip()),
        "audio_hash": _file_hash(audio_path),
        "snapshot_hash": _file_hash(snapshot_path) if snapshot_path and Path(snapshot_path).is_file() else None,
        "input_hash": input_hash,
    }


def cache_is_current(text, audio_path, snapshot_path=None, *, input_hash=None):
    try:
        stored = json.loads(manifest_path(audio_path).read_text(encoding="utf-8"))
        expected = generation_manifest(text, audio_path, snapshot_path, input_hash=input_hash)
        # API callers validate the artifacts; the synthesis owner also validates
        # its complete prompt inputs before reusing text or audio.
        if input_hash is None:
            expected["input_hash"] = stored.get("input_hash")
        return stored == expected
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def text_cache_is_current(text, audio_path, snapshot_path=None, *, input_hash=None):
    """Reuse grounded text when only speech needs regeneration."""
    try:
        checkpoint = Path(audio_path).with_suffix('.text-generation.json')
        source = checkpoint if checkpoint.is_file() else manifest_path(audio_path)
        stored = json.loads(source.read_text(encoding='utf-8'))
        from synthesizer.configuration import explanation_profile
        return (stored.get('quality_version') == QUALITY_VERSION
                and stored.get('text_generator') == explanation_profile()
                and stored.get('text_hash') == fingerprint(text.strip())
                and stored.get('input_hash') == input_hash
                and stored.get('snapshot_hash') == (
                    _file_hash(snapshot_path) if snapshot_path and Path(snapshot_path).is_file() else None))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def write_text_checkpoint(text, audio_path, snapshot_path=None, *, input_hash=None):
    """Preserve validated synthesis across TTS failures without accepting stale text."""
    from synthesizer.configuration import explanation_profile
    provenance = {'quality_version': QUALITY_VERSION, 'text_generator': explanation_profile(),
                  'text_hash': fingerprint(text.strip()), 'input_hash': input_hash,
                  'snapshot_hash': _file_hash(snapshot_path) if snapshot_path and Path(snapshot_path).is_file() else None}
    path = Path(audio_path).with_suffix('.text-generation.json')
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(provenance, handle)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
