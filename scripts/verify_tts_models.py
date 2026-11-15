"""Measure real local speech in the startup-selected API interpreter, offline."""
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tts_engine import TTSEngine
from tts_engine.providers import LAB
from tts_engine.streaming import AudioAttempt


def main():
    import psutil
    import soundfile as sf
    import torch
    root = LAB / 'reports' / 'production'
    root.mkdir(parents=True, exist_ok=True)
    text = 'The graph shows velocity increasing over time. The slope represents acceleration.'
    results = []
    for provider in ('pocket', 'kokoro'):
        engine = TTSEngine(preferred_provider=provider)
        process = psutil.Process()
        for run in ('cold', 'warm'):
            started = time.perf_counter()
            first = []
            peaks = [process.memory_info().rss]
            stopped = threading.Event()

            def sample_memory():
                while not stopped.wait(.02):
                    peaks.append(process.memory_info().rss)

            sampler = threading.Thread(target=sample_memory, daemon=True)
            sampler.start()
            attempt = AudioAttempt(root=root)
            publish = attempt.audio

            def measured_audio(*args):
                if not first:
                    first.append(time.perf_counter() - started)
                publish(*args)

            attempt.audio = measured_audio
            output = root / f'{provider}-{run}.mp3'
            try:
                with attempt.bind():
                    engine.generate(text, output)
                elapsed = time.perf_counter() - started
                info = sf.info(output)
                if info.frames <= 0 or info.samplerate != 24000:
                    raise RuntimeError('Invalid saved MP3')
                attempt.complete()
                results.append({'provider': engine.last_provider_used, 'run': run,
                                'first_audio_seconds': round(first[0], 3),
                                'generation_seconds': round(elapsed, 3),
                                'audio_seconds': round(info.duration, 3),
                                'peak_process_rss_mb': round(max(peaks) / 1048576, 1),
                                'file_bytes': output.stat().st_size})
            finally:
                stopped.set()
                sampler.join()
                attempt.dispose()
        engine.cleanup()
    result = {'python': sys.executable, 'torch': torch.__version__, 'text': text, 'results': results}
    (root / 'measurements.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    main()
