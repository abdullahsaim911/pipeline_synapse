"""Provision speech assets once, under the pipeline repository on its drive."""
from pathlib import Path
import importlib.resources
import json
import os
import subprocess
import sys

from .providers import LAB


def prepare_assets():
    # These settings are confined to this one-time provisioning process. They
    # do not redirect caches for the other runtimes started by the launcher.
    for name, folder in (('TEMP', 'temp'), ('TMP', 'temp'), ('PIP_CACHE_DIR', 'pip-cache'),
                         ('HF_HOME', 'hf-cache'), ('HF_XET_CACHE', 'hf-cache/xet')):
        path = LAB / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    import yaml
    from huggingface_hub import hf_hub_download
    cache = LAB / 'hf-cache' / 'hub'

    def fetch(repo, filename, revision):
        return hf_hub_download(repo, filename, revision=revision, cache_dir=str(cache))

    pocket = LAB / 'models' / 'pocket'
    pocket.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(importlib.resources.files('pocket_tts').joinpath('config/english.yaml').read_text(encoding='utf-8'))
    repo = 'kyutai/pocket-tts-without-voice-cloning'
    revision = 'a7056f505a4a8ca72d8b5fb1440071d6f1a35739'
    weights = fetch(repo, 'languages/english/model.safetensors', revision)
    tokenizer = fetch(repo, 'languages/english/tokenizer.json', '00eac05ed3d16bdc3f6b5d598874019c34a89214')
    voice = fetch(repo, 'languages/english/embeddings/alba.safetensors', revision)
    config['weights_path'] = config['weights_path_without_voice_cloning'] = weights
    config['flow_lm']['lookup_table']['tokenizer_path'] = tokenizer
    (pocket / 'english.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    (pocket / 'voice.json').write_text(json.dumps({'path': str(Path(voice).relative_to(LAB))}), encoding='utf-8')
    kokoro = LAB / 'models' / 'kokoro'
    import shutil
    for filename in ('config.json', 'kokoro-v1_0.pth', 'voices/af_heart.pt'):
        target = kokoro / filename
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fetch('hexgrad/Kokoro-82M', filename,
                                 'f3ff3571791e39611d31c381e3a41a3af07b4987'), target)
    import spacy
    if not spacy.util.is_package('en_core_web_sm'):
        env = dict(os.environ)
        for name, folder in (('TEMP', 'temp'), ('TMP', 'temp'), ('PIP_CACHE_DIR', 'pip-cache')):
            path = LAB / folder
            path.mkdir(parents=True, exist_ok=True)
            env[name] = str(path)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'],
                       env=env, check=True)


if __name__ == '__main__':
    prepare_assets()
