"""Run real model checks in an isolated directory, preserving lecture caches.

Example: python scripts/evaluate_explanation_quality.py --video Tf4A93763Bw
         --intervention a0a2adaa --extract --output .runtime/quality-fields
Inspect the saved image, observation and scripts against the quality contract.
This is an evaluation runner, not an automatic semantic quality judge.
"""

import argparse
import hashlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video")
    parser.add_argument("--intervention")
    parser.add_argument("--reference-case", type=Path, help="Authored reference image/lecture fixture; kept separate from saved videos.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract", action="store_true", help="Calls the remote VLM once (may incur provider charges).")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--model", default="mistral:latest", help="Text model to evaluate; does not change the VLM or application defaults.")
    parser.add_argument("--provider", choices=["ollama", "gemini", "modal_mistral"], default="ollama")
    parser.add_argument("--modes", nargs="+", default=["brief", "explanatory", "detailed"],
                        choices=["brief", "explanatory", "detailed"])
    args = parser.parse_args()
    if args.reference_case:
        if args.video or args.intervention or not (args.extract or args.snapshot):
            parser.error("A reference case needs --extract or --snapshot, without --video/--intervention")
        fixture = json.loads(args.reference_case.read_text(encoding="utf-8"))
        target, transcript_records = fixture["target"], fixture["transcript"]["entries"]
        source = None
    else:
        if not args.video or not args.intervention:
            parser.error("Supply --video and --intervention, or --reference-case")
        source = ROOT / "data" / args.video
        records = json.loads((source / "interventions.json").read_text(encoding="utf-8"))["interventions"]
        matches = [item for item in records if item["id"].startswith(args.intervention)]
        if len(matches) != 1:
            parser.error("Intervention prefix must identify exactly one saved intervention")
        target = matches[0]
        transcript_records = json.loads((source / "transcript.json").read_text(encoding="utf-8"))["entries"]
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
        (args.output / name).write_text(text, encoding="utf-8")

    frame = Path(target["frame_path"])
    if not frame.is_absolute():
        frame = ROOT / frame
    save("target.json", target)
    from explanation_cache import QUALITY_VERSION
    source_files = ["vlm_engine/snapshot_engine.py", "vlm_engine/evidence_refinement.py",
                    "visual_evidence/compact_schema.py", "visual_evidence/assembler.py",
                    "vlm_interface/modal_vlm_interface.py", "synthesizer/llm_synthesizer.py",
                    "synthesizer/context_manager.py", "synchronizer/synchronizer.py",
                    "modal_inference/config.py", "modal_inference/launch_config.py",
                    "llm_interface/ollama_client.py", "llm_interface/prompt_budget.py", "explanation_cache.py",
                    "llm_interface/modal_mistral_provider.py", "modal_inference/mistral_config.py",
                    "synthesizer/context_manager.py", "vlm_engine/region_grounding.py"]
    from llm_interface.gemini_provider import DEFAULT_MODEL
    text_model = (os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL) if args.provider == "gemini" else args.model
    if args.provider == "modal_mistral":
        from modal_inference.mistral_config import MODEL_ID
        text_model = MODEL_ID
    save("run-metadata.json", {"quality_version": QUALITY_VERSION, "text_model": text_model,
         "text_provider": args.provider,
         "source_kind": target.get("source_kind", "saved_lecture"),
         "snapshot_source": str(args.snapshot) if args.snapshot else None,
         "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}})
    if args.extract:
        from vlm_engine.snapshot_engine import SnapshotEngine
        engine = SnapshotEngine(provider="modal", max_retries=0)
        snapshot = engine.analyze_frame(
            str(frame), target["content_type"], frame_id=target["id"],
            timestamp_seconds=target["timestamp"],
            canonical_category_ids=target.get("canonical_categories"),
            raw_category_labels=target.get("raw_category_labels"),
        )
        engine.cleanup()
    else:
        snapshot_path = args.snapshot or source / "interventions" / target["id"][:8] / "vlm_snapshot.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        from visual_evidence import COMPACT_OBSERVATION_SCHEMA, assemble_compact_qwen_observation
        if snapshot.get("schema") == COMPACT_OBSERVATION_SCHEMA:
            evidence = assemble_compact_qwen_observation(snapshot, frame_id=target["id"], timestamp_seconds=target["timestamp"])
            snapshot = {"_error": False, "_structured_observation": snapshot,
                        "visual_evidence": evidence.to_schema_dict()}
    save("snapshot.json", snapshot)
    from visual_evidence import assess_snapshot
    assessment = assess_snapshot(snapshot)
    save("snapshot-assessment.json", asdict(assessment))
    if not assessment.usable:
        print("Extraction failed; inspect snapshot.json. No explanation was generated.", flush=True)
        return 1
    print("Extraction available:", (snapshot.get("visual_evidence") or {}).get("category_ids"), flush=True)
    if args.extract_only:
        return 0

    from synchronizer import TranscriptEntry, reconcile_visual_evidence
    from synthesizer import LLMSynthesizer
    from synthesizer.context_manager import ContextManager
    entries = [TranscriptEntry(start=item["start"], end=item.get("end", item["start"] + item.get("duration", 0)), text=item["text"])
               for item in transcript_records]
    snapshot["_lecture_context"] = ContextManager(entries).get_aligned_context(target["timestamp"])
    aligned = " ".join(item["text"] for item in snapshot["_lecture_context"]["aligned_segments"])
    current = next((item.text for item in entries if item.start <= target["timestamp"] <= item.end), "")
    snapshot["coverage_plan"] = reconcile_visual_evidence(snapshot, aligned)
    save("synthesis-input.json", snapshot)
    save("settings.json", {key: os.environ.get(key) for key in [
        "SYNAPSE_OLLAMA_NUM_CTX", "SYNAPSE_OLLAMA_NUM_GPU", "SYNAPSE_VLM_PROVIDER",
    ]})
    provider = None
    if args.provider == "gemini":
        from llm_interface.gemini_provider import GeminiProvider
        provider = GeminiProvider.from_env()
    elif args.provider == "modal_mistral":
        from llm_interface.modal_mistral_provider import ModalMistralProvider
        provider = ModalMistralProvider.from_env()
    synth = LLMSynthesizer("http://localhost:11434", args.model, 600, 1, entries, "explanatory", text_provider=provider)
    generation_owner = provider or synth.client
    if args.provider == "modal_mistral":
        post = provider._post
        def archive_raw(path, payload):
            response = post(path, payload)
            if path == "/v1/chat/completions":
                save(f"{mode}.call-{len(calls) + 1:02d}.raw.json", response)
            return response
        provider._post = archive_raw
    generate = generation_owner.generate
    failures = 0
    for mode in args.modes:
        calls = []
        def record(prompt, *positional, **kwargs):
            call_index = len(calls) + 1
            save(f"{mode}.call-{call_index:02d}.prompt.txt", prompt)
            if call_index == 1:
                save(f"{mode}.prompt.txt", prompt)
            if args.provider == "gemini":
                # Evaluation is intentionally slower than interactive use to
                # avoid turning a batch comparison into a free-tier burst.
                time.sleep(6)
            response = generate(prompt, *positional, **kwargs)
            save(f"{mode}.call-{call_index:02d}.response.json", asdict(response))
            if call_index == 1:
                save(f"{mode}.response.json", asdict(response))
            calls.append({"call": call_index, "latency_ms": getattr(response, "latency_ms", None),
                          "finish_reason": getattr(response, "finish_reason", None)})
            save(f"{mode}.calls.json", calls)
            return response
        generation_owner.generate = record
        try:
            text = synth.weave(current, snapshot, target["content_type"], "quality evaluation", mode,
                               require_visual_explanation=True)
            save(f"{mode}.txt", text)
            print(mode, len(text.split()), "words", flush=True)
        except Exception as exc:
            from llm_interface.providers import LLMProviderError
            cause = exc.__cause__
            save(f"{mode}.error.json", {"error_type": type(exc).__name__, "error": str(exc),
                 "provider_error": str(cause) if isinstance(cause, LLMProviderError) else None})
            print(mode, "failed:", type(exc).__name__, flush=True)
            failures += 1
    synth.cleanup()
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
