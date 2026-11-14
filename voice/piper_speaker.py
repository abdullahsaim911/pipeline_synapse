"""Local Piper text-to-speech for spoken voice-command replies (/voice/speak).

Voice replies are short and frequent, so they use a small offline Piper voice
instead of the pipeline's explanation TTS. Nothing is downloaded at runtime:
install the voice once with

    python -m voice.piper_speaker download

Configuration (all optional):
    SYNAPSE_PIPER_VOICE         voice name, default en_US-lessac-medium
    SYNAPSE_PIPER_VOICE_DIR     folder holding <voice>.onnx and <voice>.onnx.json,
                                default <project>/models/piper
    SYNAPSE_PIPER_LENGTH_SCALE  speaking pace; above 1.0 is slower, default 1.4
    SYNAPSE_PIPER_SPEAKER       speaker id for multi-speaker voices, default the voice's own

The default pace was measured, not guessed: lessac at 1.4 speaks about
164 words/minute, close to clear conversational speech (150-160). Its
natural 1.0 pace is about 209 words/minute, which learners found too fast.
Voices differ: en_US-libritts_r-medium stays near 200 words/minute even at 1.7.
"""

import argparse
import logging
import os
import threading
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOICE = "en_US-lessac-medium"
DEFAULT_VOICE_DIR = PROJECT_ROOT / "models" / "piper"
DEFAULT_LENGTH_SCALE = 1.4
MIN_LENGTH_SCALE, MAX_LENGTH_SCALE = 0.5, 2.0


class PiperVoiceMissingError(RuntimeError):
    """The configured Piper voice files are not installed."""


def _length_scale(raw: Optional[str]) -> float:
    try:
        value = float(raw) if raw not in (None, "") else DEFAULT_LENGTH_SCALE
    except ValueError:
        logger.warning("Ignoring invalid SYNAPSE_PIPER_LENGTH_SCALE")
        return DEFAULT_LENGTH_SCALE
    return min(MAX_LENGTH_SCALE, max(MIN_LENGTH_SCALE, value))


def _speaker_id(raw: Optional[str]) -> Optional[int]:
    if raw in (None, ""):
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid SYNAPSE_PIPER_SPEAKER")
        return None


class PiperSpeaker:
    """Thread-safe, lazily loaded Piper voice that writes WAV replies."""

    file_extension = ".wav"
    media_type = "audio/wav"

    def __init__(self, voice: Optional[str] = None, voice_dir: Optional[Path] = None,
                 length_scale: Optional[float] = None, speaker_id: Optional[int] = None) -> None:
        self.voice_name = voice or os.environ.get("SYNAPSE_PIPER_VOICE", "").strip() or DEFAULT_VOICE
        configured_dir = os.environ.get("SYNAPSE_PIPER_VOICE_DIR", "").strip()
        self.voice_dir = Path(voice_dir or configured_dir or DEFAULT_VOICE_DIR)
        self.length_scale = (_length_scale(str(length_scale)) if length_scale is not None
                             else _length_scale(os.environ.get("SYNAPSE_PIPER_LENGTH_SCALE")))
        self.speaker_id = (speaker_id if speaker_id is not None
                           else _speaker_id(os.environ.get("SYNAPSE_PIPER_SPEAKER")))
        self._voice = None
        self._config = None
        self._lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        return self.voice_dir / f"{self.voice_name}.onnx"

    @property
    def cache_identity(self) -> str:
        """Everything that changes the sound, so cached replies never cross settings."""
        return f"piper|{self.voice_name}|{self.length_scale:.3f}|{self.speaker_id}"

    def _load(self):
        with self._lock:
            if self._voice is None:
                model = self.model_path
                if not model.is_file() or not Path(f"{model}.json").is_file():
                    raise PiperVoiceMissingError(
                        f"Piper voice '{self.voice_name}' is not installed in {self.voice_dir}. "
                        "Run: python -m voice.piper_speaker download"
                    )
                from piper import PiperVoice, SynthesisConfig

                self._voice = PiperVoice.load(str(model))
                self._config = SynthesisConfig(length_scale=self.length_scale, speaker_id=self.speaker_id)
                logger.info("Piper voice loaded: %s", self.voice_name)
            return self._voice, self._config

    def warm(self) -> None:
        self._load()

    def generate(self, text: str, output_path: str) -> str:
        voice, config = self._load()
        with wave.open(output_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=config)
        return output_path

    def cleanup(self) -> None:
        with self._lock:
            self._voice = self._config = None


def download(voice: Optional[str] = None, voice_dir: Optional[Path] = None) -> Path:
    """Install the configured voice once; the API itself never downloads."""
    from piper.download_voices import download_voice

    speaker = PiperSpeaker(voice=voice, voice_dir=voice_dir)
    speaker.voice_dir.mkdir(parents=True, exist_ok=True)
    download_voice(speaker.voice_name, speaker.voice_dir)
    return speaker.model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the Piper voice used for spoken voice replies.")
    parser.add_argument("command", choices=["download"])
    parser.add_argument("--voice", help=f"voice name (default: SYNAPSE_PIPER_VOICE or {DEFAULT_VOICE})")
    parser.add_argument("--dir", type=Path, help=f"install folder (default: {DEFAULT_VOICE_DIR})")
    args = parser.parse_args()
    print(f"Installed: {download(args.voice, args.dir)}")


if __name__ == "__main__":
    main()
