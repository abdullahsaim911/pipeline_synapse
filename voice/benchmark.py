"""
Benchmark for the voice-control models on the local CPU (module 0).

Two sub-commands:

  make-clips  Generate clean synthetic test clips with the project's Edge-TTS
              (needs internet) plus a manifest of reference texts.
  run         Time faster-whisper (CPU, int8) and the embedding model and print
              load time, latency, peak memory and word accuracy.

Synthetic clips are clean TTS speech, so accuracy here is optimistic compared
with a real microphone. Point --clips at a folder holding your own recordings
and a manifest.json ([{"file": ..., "text": ...}]) to get realistic numbers.

Usage (from the repository root):
  python -m voice.benchmark make-clips --out <dir>
  python -m voice.benchmark run --clips <dir> --whisper tiny base
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time

import psutil

PHRASES = [
    "Open my bookmarks",
    "Take me back to the home screen",
    "Pause the video",
    "Resume the lecture",
    "Go to the next explanation",
    "Skip forward thirty seconds",
    "Read that explanation again",
    "Save this explanation for later",
    "Switch to study mode",
    "Make the text bigger",
    "Find lectures about eigenvectors",
    "Open the second result",
    "What does the x axis represent in this graph",
    "Can you explain that more simply",
]
VOICES = ["en-US-AriaNeural", "en-GB-RyanNeural"]
SENTENCES_FOR_EMBEDDING = PHRASES * 8  # ~110 short sentences


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _normalize(text: str) -> list:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("-", " ")).split()


def _word_errors(reference: str, hypothesis: str) -> tuple:
    """Word-level edit distance. Returns (errors, reference_word_count)."""
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    rows = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, rows[0] = rows[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(rows[j] + 1, rows[j - 1] + 1, prev + (r != h))
            prev, rows[j] = rows[j], cur
    return rows[-1], len(ref)


def make_clips(out_dir: str) -> None:
    import edge_tts

    os.makedirs(out_dir, exist_ok=True)
    manifest = []

    async def generate():
        for v_index, voice in enumerate(VOICES):
            for p_index, phrase in enumerate(PHRASES):
                name = f"v{v_index}_p{p_index:02d}.mp3"
                path = os.path.join(out_dir, name)
                await edge_tts.Communicate(phrase, voice).save(path)
                manifest.append({"file": name, "text": phrase, "voice": voice})

    asyncio.run(generate())
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} clips to {out_dir}")


def bench_whisper(model_size: str, clips_dir: str, manifest: list, runs: int) -> dict:
    from faster_whisper import WhisperModel

    before = _rss_mb()
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    load_s = time.perf_counter() - t0
    after_load = _rss_mb()

    # Warm-up so first-call overhead is not counted.
    warm_segments, _ = model.transcribe(
        os.path.join(clips_dir, manifest[0]["file"]), language="en"
    )
    list(warm_segments)

    latencies, errors, words, peak = [], 0, 0, after_load
    results = []
    for item in manifest:
        path = os.path.join(clips_dir, item["file"])
        best, text = None, ""
        for _ in range(runs):
            t = time.perf_counter()
            segments, _info = model.transcribe(
                path, language="en", beam_size=1, vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            elapsed = time.perf_counter() - t
            best = elapsed if best is None else min(best, elapsed)
        latencies.append(best)
        e, n = _word_errors(item["text"], text)
        errors, words = errors + e, words + n
        peak = max(peak, _rss_mb())
        results.append({"expected": item["text"], "heard": text, "seconds": round(best, 2)})

    latencies.sort()
    return {
        "model": model_size,
        "load_seconds": round(load_s, 2),
        "rss_added_mb": round(after_load - before, 0),
        "peak_rss_mb": round(peak, 0),
        "latency_median_s": round(statistics.median(latencies), 2),
        "latency_p95_s": round(latencies[int(0.95 * (len(latencies) - 1))], 2),
        "word_error_rate": round(errors / max(words, 1), 3),
        "clips": len(manifest),
        "details": results,
    }


def bench_embeddings(model_name: str) -> dict:
    from sentence_transformers import SentenceTransformer

    before = _rss_mb()
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device="cpu")
    load_s = time.perf_counter() - t0
    after_load = _rss_mb()

    model.encode(["warm up"])
    single = []
    for sentence in PHRASES:
        t = time.perf_counter()
        model.encode([sentence])
        single.append(time.perf_counter() - t)
    t = time.perf_counter()
    model.encode(SENTENCES_FOR_EMBEDDING, batch_size=32)
    batch_s = time.perf_counter() - t
    return {
        "model": model_name,
        "load_seconds": round(load_s, 2),
        "rss_added_mb": round(after_load - before, 0),
        "peak_rss_mb": round(_rss_mb(), 0),
        "single_sentence_median_ms": round(statistics.median(single) * 1000, 1),
        "batch_sentences": len(SENTENCES_FOR_EMBEDDING),
        "batch_seconds": round(batch_s, 2),
    }


def run(args) -> None:
    with open(os.path.join(args.clips, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    memory = psutil.virtual_memory()
    report = {
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(memory.total / 1024 ** 3, 1),
        "ram_available_gb_at_start": round(memory.available / 1024 ** 3, 2),
        "whisper": [],
    }
    for size in args.whisper:
        print(f"== faster-whisper {size} (cpu, int8)")
        result = bench_whisper(size, args.clips, manifest, args.runs)
        report["whisper"].append(result)
        print({k: v for k, v in result.items() if k != "details"})
    if not args.skip_embeddings:
        print(f"== embeddings {args.embedding_model}")
        report["embeddings"] = bench_embeddings(args.embedding_model)
        print(report["embeddings"])
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.report}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    mk = sub.add_parser("make-clips")
    mk.add_argument("--out", required=True)

    rn = sub.add_parser("run")
    rn.add_argument("--clips", required=True)
    rn.add_argument("--whisper", nargs="+", default=["tiny", "base"])
    rn.add_argument("--runs", type=int, default=1)
    rn.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    rn.add_argument("--skip-embeddings", action="store_true")
    rn.add_argument("--report")

    args = parser.parse_args(argv)
    if args.command == "make-clips":
        make_clips(args.out)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
