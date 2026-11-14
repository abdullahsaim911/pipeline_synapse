"""Local Ollama vocabulary accounting without loading inference weights.

Reserve the entire template/system byte length as an upper bound on wrapper
tokens. If the local vocabulary is unavailable, UTF-8 bytes bound prompt tokens
conservatively for the supported byte-fallback Mistral tokenizer.
"""

from functools import lru_cache
import json
import os
from pathlib import Path
import re
from threading import Lock

_lock = Lock()


@lru_cache(maxsize=2)
def _local_vocabulary(root: str, model: str):
    # Synapse's local explanation model uses the ordinary library/name:tag layout.
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?", model):
        return None, 1024
    name, _, tag = model.partition(":")
    directory = Path(root)
    try:
        manifest = json.loads((directory / "manifests/registry.ollama.ai/library" / name / (tag or "latest")).read_text())
        model_path = None
        wrapper_bytes = 64  # BOS, separators and terminal markers.
        for layer in manifest["layers"]:
            digest = layer["digest"]
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
                raise ValueError("Invalid Ollama blob digest")
            path = directory / "blobs" / digest.replace(":", "-")
            kind = layer["mediaType"].rsplit(".", 1)[-1]
            if kind == "model":
                model_path = path
            elif kind in {"template", "system"}:
                wrapper_bytes += path.stat().st_size
        if model_path is None:
            return None, wrapper_bytes
        from llama_cpp import Llama
        vocabulary = Llama(model_path=str(model_path), vocab_only=True, n_gpu_layers=0, verbose=False)
        return vocabulary, wrapper_bytes
    except (OSError, ValueError, KeyError, ImportError, RuntimeError):
        return None, 1024


def prompt_input_tokens(prompt: str, model: str) -> int:
    root = os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama/models"))
    # llama.cpp tokenization and lazy vocabulary construction share one lock.
    with _lock:
        vocabulary, wrapper_reserve = _local_vocabulary(root, model)
        encoded = prompt.encode("utf-8")
        count = len(vocabulary.tokenize(encoded, add_bos=True, special=False)) if vocabulary else len(encoded)
    return count + wrapper_reserve
