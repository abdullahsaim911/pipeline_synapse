"""
Comprehensive Test Script for Qwen2-VL-7B on 6GB GPU

Tests all optimization approaches to find the best for your setup.

Usage:
    python test_all_7b_approaches.py --approach awq         # Best (recommended)
    python test_all_7b_approaches.py --approach 4bit         # Good
    python test_all_7b_approaches.py --approach vllm         # Fast but experimental
    python test_all_7b_approaches.py --approach all           # Test all
"""

import argparse
import torch
import sys
import time
from pathlib import Path

# Import VLM interface
sys.path.insert(0, str(Path(__file__).parent))
from vlm_interface.vlm_interface import Qwen2VLInterface


def test_approach(approach: str, model_path: str):
    """Test a specific optimization approach."""
    print(f"\n{'='*60}")
    print(f"Testing: {approach.upper()}")
    print(f"{'='*60}\n")

    try:
        # Clear GPU cache before each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        start_load = time.time()

        if approach == "awq":
            # AutoAWQ - BEST for 6GB GPU
            vlm = Qwen2VLInterface(
                model_name=model_path,
                use_awq=True,
                device_map="auto",
            )
        elif approach == "4bit":
            # 4-bit with CPU offloading
            vlm = Qwen2VLInterface(
                model_name=model_path,
                use_4bit=True,
                device_map="auto",
            )
        elif approach == "vllm":
            # vLLM - Experimental for Qwen2-VL
            print("[WARNING] vLLM has experimental Qwen2-VL support")
            try:
                from vllm import LLM

                llm = LLM(
                    model=model_path,
                    quantization="bitsandbytes",
                    gpu_memory_utilization=0.85,
                    max_model_len=2048,
                )

                # Test inference with vLLM
                prompts = ["What is 2+2?", "Hello, how are you?"]
                outputs = llm.generate(prompts, max_tokens=10)

                print(f"vLLM Result: {outputs[0].outputs[0].text}")
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Peak GPU Memory: {peak_mem:.2f} GB")
                print(f"Load Time: {time.time() - start_load:.2f}s")
                del llm
                return True
            except Exception as e:
                print(f"[ERROR] vLLM failed: {e}")
                return False
        else:
            print(f"[ERROR] Unknown approach: {approach}")
            return False

        load_time = time.time() - start_load
        print(f"\nLoad Time: {load_time:.2f}s")

        # Test inference
        print("\n--- Test 1: Text-only ---")
        test_start = time.time()
        response = vlm.analyze(
            image_path=None,
            prompt="What is 2+2? Answer in 3 words.",
            generation_params={"max_new_tokens": 20, "do_sample": False}
        )
        infer_time = time.time() - test_start
        print(f"Response: {response.text}")
        print(f"Inference Time: {infer_time:.2f}s")
        print(f"Total Duration: {response.total_duration_ms}ms")

        # Check GPU memory
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            current_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"\nPeak GPU Memory: {peak_mem:.2f} GB")
            print(f"Current GPU Memory: {current_mem:.2f} GB")

        vlm.cleanup()

        # Final status
        if peak_mem < 6.0:
            print(f"\n✅ SUCCESS: Fits in 6GB GPU ({peak_mem:.2f}GB)")
            return True
        else:
            print(f"\n❌ FAIL: Exceeds 6GB GPU ({peak_mem:.2f}GB)")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)[:200]}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test all Qwen2-VL-7B optimization approaches on 6GB GPU"
    )
    parser.add_argument(
        "--approach",
        choices=["awq", "4bit", "vllm", "all"],
        default="all",
        help="Which approach to test (default: all)",
    )
    parser.add_argument(
        "--model-path",
        default="models/Qwen2-VL-7B-Instruct",
        help="Path to standard model",
    )
    parser.add_argument(
        "--awq-model-path",
        default="models/Qwen2-VL-7B-Instruct-AWQ",
        help="Path to AWQ-quantized model",
    )

    args = parser.parse_args()

    print("="*60)
    print("Qwen2-VL-7B Optimization Test Suite")
    print("="*60)

    if not torch.cuda.is_available():
        print("\n[ERROR] No CUDA GPU detected. This test requires GPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU: {gpu_name}")
    print(f"Total Memory: {gpu_memory:.1f} GB")
    print(f"Target Memory: 6.0 GB\n")

    approaches = {
        "awq": {
            "model": args.awq_model_path,
            "desc": "AutoAWQ - FASTEST, ~3.5GB, Excellent quality"
        },
        "4bit": {
            "model": args.model_path,
            "desc": "4-bit + Offload - Medium speed, ~4GB, Good quality"
        },
        "vllm": {
            "model": args.model_path,
            "desc": "vLLM - Fast, ~4GB, Experimental Qwen2-VL support"
        },
    }

    if args.approach == "all":
        results = {}
        for approach in ["awq", "4bit", "vllm"]:
            results[approach] = test_approach(approach, approaches[approach]["model"])

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for approach, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{approach.upper()}: {status} - {approaches[approach]['desc']}")

        # Recommendation
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        if results.get("awq"):
            print("✅ Use AutoAWQ - Best for your 6GB GPU")
        elif results.get("4bit"):
            print("✅ Use 4-bit + Offloading")
        elif results.get("vllm"):
            print("⚠️  vLLM works but is experimental for Qwen2-VL")
        else:
            print("❌ No approach worked. Consider using 2B model instead.")

    else:
        test_approach(args.approach, approaches[args.approach]["model"])


if __name__ == "__main__":
    main()
