"""
SYNAPSE - Main Entry Point

Pedagogical keyframe extraction from STEM lecture videos
for visually impaired learners.

Features:
- Extracts only conceptually important frames
- CLIP-based STEM content classification
- Concept completion detection (equations, diagrams)
- Multi-layer redundancy filtering
- 5-20 frames per 10-minute lecture

Usage:
    python main.py                              # Pedagogical (default, recommended)
    python main.py --extractor semantic         # Adaptive clustering
    python main.py --extractor scene            # PySceneDetect
    python main.py --start 0 --end 5            # Process only first 5 URLs
"""

import os
import sys

# Add ffmpeg to PATH
os.environ["PATH"] += ";" + os.path.join(os.getcwd(), "ffmpeg", "bin")

if __name__ == "__main__":
    from batch_process import run_batch
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
        help="Extraction method: 'pedagogical' (STEM-focused, default), 'semantic', 'scene', or 'ssim'",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["hybrid", "cluster", "temporal"],
        default="hybrid",
        help="For semantic extractor: 'hybrid', 'cluster', or 'temporal'",
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
