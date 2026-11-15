"""Local CPU speech providers. Imports and model loading are deliberately lazy."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / '.tts-lab'
SAMPLE_RATE = 24000


def tts_profile():
    return {'version': 1, 'primary': 'pocket-tts-4.0.1-english',
            'voice': 'alba', 'fallback': 'kokoro-82m-v1.0',
            'fallback_voice': 'af_heart', 'sample_rate': SAMPLE_RATE}


class PocketTTSProvider:
    name = 'PocketTTS'

    def __init__(self):
        self.model = self.state = None

    @property
    def loaded(self):
        return self.model is not None and self.state is not None

    def warm(self):
        if self.model is None:
            import os
            import json
            os.environ['KPOCKET_TTS_ERROR_WITHOUT_EOS'] = '1'
            from pocket_tts import TTSModel
            folder = LAB / 'models' / 'pocket'
            model = TTSModel.load_model(config=folder / 'english.yaml')
            voice = Path(json.loads((folder / 'voice.json').read_text())['path'])
            state = model.get_state_for_audio_prompt(voice if voice.is_absolute() else LAB / voice)
            self.model, self.state = model, state

    def chunks(self, text, stop):
        self.warm()
        # Use Pocket's tokenizer-aware text splitting; hand splitting would
        # damage decimal numbers, abbreviations and technical punctuation.
        source = self.model.generate_audio_stream(self.state, text, max_tokens=50, stop=stop)
        try:
            for chunk in source:
                if stop.is_set():
                    return
                yield chunk.detach().cpu().numpy().reshape(-1)
        finally:
            # Pocket owns decoder/generation threads. Drain their sentinel before
            # releasing the inference lock, including on lease loss or shutdown.
            stop.set()
            for _ in source:
                pass

    def unload_models(self):
        self.model = self.state = None


class KokoroProvider:
    name = 'Kokoro'

    def __init__(self):
        self.pipeline = self.voice = None

    @property
    def loaded(self):
        return self.pipeline is not None and self.voice is not None

    def warm(self):
        if self.pipeline is None:
            import spacy
            if not spacy.util.is_package('en_core_web_sm'):
                raise RuntimeError('Kokoro language assets are missing; run API model setup')
            import torch
            from kokoro import KPipeline
            from kokoro.model import KModel
            folder = LAB / 'models' / 'kokoro'
            model = KModel(repo_id='hexgrad/Kokoro-82M',
                           config=str(folder / 'config.json'),
                           model=str(folder / 'kokoro-v1_0.pth')).cpu().eval()
            self.pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=model, device='cpu')
            self.voice = torch.load(folder / 'voices' / 'af_heart.pt',
                                    map_location='cpu', weights_only=True)

    def chunks(self, text, stop):
        self.warm()
        for result in self.pipeline(text, voice=self.voice):
            if stop.is_set():
                return
            yield result.audio.detach().cpu().numpy().reshape(-1)

    def unload_models(self):
        self.pipeline = self.voice = None


def get_provider(name, **_kwargs):
    if name.lower() == 'pocket':
        return PocketTTSProvider()
    if name.lower() == 'kokoro':
        return KokoroProvider()
    raise ValueError('Available TTS providers: pocket, kokoro')
