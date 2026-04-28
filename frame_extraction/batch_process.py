"""
SYNAPSE - Batch Video Processing Pipeline

Processes YouTube videos to extract pedagogically significant keyframes
for visually impaired STEM learners.

Extraction Methods:
  - pedagogical (default): Concept-focused extraction with:
      - CLIP classification for STEM content
      - Concept completion detection (equations, diagrams)
      - Multi-layer redundancy filtering (SSIM + histogram)
      - 5-20 frames per 10-minute lecture
  - semantic: Adaptive clustering-based extraction
  - scene: PySceneDetect visual change detection
  - ssim: Legacy SSIM-based frame comparison

Usage:
    python batch_process.py                          # Pedagogical (recommended)
    python batch_process.py --extractor semantic     # Adaptive clustering
    python batch_process.py --extractor scene        # Visual scene detection
    python batch_process.py --start 0 --end 5        # Process subset of URLs
"""

import os
import time
import json


# =============================================================================
# EXTRACTOR IMPORTS
# =============================================================================

# Pedagogical v2 extractor (default - robust, generalized)
try:
    from pedagogical_extractor_v2 import extract_pedagogical_keyframes as extract_v2

    PEDAGOGICAL_V2_AVAILABLE = True
except ImportError:
    PEDAGOGICAL_V2_AVAILABLE = False

# Pedagogical extractor (fallback)


# SSIM extractor (legacy)
try:
    from stream import stream_and_extract

    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


# =============================================================================
# UTILITIES
# =============================================================================


def load_urls(filepath="urls.txt"):
    """Load YouTube URLs from file, skipping comments and blank lines."""
    if not os.path.exists(filepath):
        print(f"  Error: '{filepath}' not found.")
        return []

    urls = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    import re

    patterns = [r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})", r"([a-zA-Z0-9_-]{11})"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url


# ============================================================================
# MAIN BATCH PROCESSOR
# =============================================================================


def run_batch(
    filepath="urls.txt",
    extractor="pedagogical",
    method="hybrid",
    url_range=None,
):
    """
    Process a batch of YouTube videos.

    Args:
        filepath: Path to file containing YouTube URLs
        extractor: Extraction method
            - "pedagogical": Concept-focused for STEM education (default)
            - "semantic": Adaptive clustering-based
            - "scene": PySceneDetect visual detection
            - "ssim": Legacy SSIM-based
        method: For semantic extractor - "hybrid", "cluster", or "temporal"
        url_range: Optional (start, end) tuple to process subset of URLs
    """
    urls = load_urls(filepath)[13:14]

    if url_range:
        start = url_range[0] or 0
        end = url_range[1] or len(urls)
        urls = urls[start:end]

    if not urls:
        print("  No URLs found. Nothing to process.")
        return

    total = len(urls)
    results = []

    # Determine extractor
    use_v2 = False
    use_pedagogical = False

    if extractor == "pedagogical":
        if PEDAGOGICAL_V2_AVAILABLE:
            extractor_name = "Pedagogical v2 (robust, adaptive)"
            use_v2 = True
        elif PEDAGOGICAL_AVAILABLE:
            extractor_name = "Pedagogical (STEM-focused)"
            use_pedagogical = True
        else:
            extractor_name = "SSIM (fallback)"
    elif extractor == "semantic" and SEMANTIC_AVAILABLE:
        extractor_name = f"Semantic ({method})"
    elif extractor == "scene" and SCENE_AVAILABLE:
        extractor_name = "PySceneDetect"
    else:
        extractor_name = "SSIM (legacy)"

    # Print header
    print("\n" + "=" * 65)
    print("  SYNAPSE — Pedagogical Keyframe Extraction for STEM Education")
    print("=" * 65)
    print(f"  URLs to process: {total}")
    print(f"  Extractor: {extractor_name}")
    print("=" * 65)

    for index, url in enumerate(urls, start=1):
        print(f"\n\n{'─'*65}")
        print(f"  VIDEO {index} of {total}")
        print(f"  URL: {url}")
        print("─" * 65)

        start_time = time.time()

        try:
            video_id = extract_video_id(url)
            output_dir = f"data/{video_id}/keyframes"

            # Extract keyframes based on selected method
            if use_v2:
                keyframes = extract_v2(video_id, output_dir, verbose=True)
                keyframe_count = len(keyframes)
            elif use_pedagogical:
                keyframes = extract_pedagogical_keyframes(video_id, output_dir)
                keyframe_count = len(keyframes)
            elif extractor == "semantic" and SEMANTIC_AVAILABLE:
                keyframes = extract_semantic_keyframes_from_url(
                    video_id, output_dir, method=method
                )
                keyframe_count = len(keyframes)
            elif extractor == "scene" and SCENE_AVAILABLE:
                video_id, keyframe_count = extract_keyframes_hybrid(url)
                keyframes = []
            else:
                video_id, keyframe_count = stream_and_extract(url)
                keyframes = []

            elapsed = round(time.time() - start_time, 1)

            results.append(
                {
                    "index": index,
                    "url": url,
                    "video_id": video_id,
                    "status": "success",
                    "keyframes": keyframe_count,
                    "time_sec": elapsed,
                }
            )

            print(f"\n  ✓ Extracted {keyframe_count} keyframes in {elapsed}s")

        except Exception as e:
            elapsed = round(time.time() - start_time, 1)
            print(f"\n  ✗ Error: {e}")
            import traceback

            traceback.print_exc()

            results.append(
                {
                    "index": index,
                    "url": url,
                    "video_id": "unknown",
                    "status": "failed",
                    "keyframes": 0,
                    "time_sec": elapsed,
                    "error": str(e),
                }
            )

    # Print summary
    print("\n\n" + "=" * 65)
    print("  BATCH COMPLETE — Summary")
    print("=" * 65)

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    total_frames = sum(r["keyframes"] for r in success)
    total_time = sum(r["time_sec"] for r in results)

    for r in results:
        icon = "✓" if r["status"] == "success" else "✗"
        vid = r["video_id"][:15] if len(r["video_id"]) > 15 else r["video_id"]
        print(
            f"  {icon} [{r['index']:02d}] {vid:15s}  {r['keyframes']:3d} frames  {r['time_sec']:6.1f}s"
        )

    print("─" * 65)
    print(f"  Success: {len(success)}/{total} videos")
    print(f"  Total frames: {total_frames}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 65)

    # Save batch report
    report_path = "data/batch_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved: {report_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SYNAPSE - Pedagogical Keyframe Extraction for STEM Education"
    )
    parser.add_argument(
        "--urls", "-u", default="urls.txt", help="Path to file with YouTube URLs"
    )
    parser.add_argument(
        "--extractor",
        "-e",
        choices=["pedagogical", "semantic", "scene", "ssim"],
        default="pedagogical",
        help="Extraction method (default: pedagogical)",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["hybrid", "cluster", "temporal"],
        default="hybrid",
        help="For semantic extractor: hybrid, cluster, or temporal",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=None,
        help="Start index for URL processing (0-based)",
    )
    parser.add_argument(
        "--end",
        "-n",
        type=int,
        default=None,
        help="End index for URL processing (exclusive)",
    )

    args = parser.parse_args()

    url_range = None
    if args.start is not None or args.end is not None:
        url_range = (args.start, args.end)

    run_batch(
        filepath=args.urls,
        extractor=args.extractor,
        method=args.method,
        url_range=url_range,
    )
