"""Prepare demo videos in one go.

    python -m synapse_interaction.prepare URL1 URL2 URL3

For every YouTube URL: downloads the video (video-only, <=720p) to demo_videos\\ID.mp4, runs the EXISTING pipeline
in detection-only mode (M0 + M0b + M1, creates data\\ID\\transcript.json and keyframes), then runs the interaction
detector (data\\ID\\interactions.json). Steps already done are skipped.
"""
import os
import re
import sys


def _vid(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url


def prepare_video(url: str) -> str:
    """Download + existing pipeline (M0, M0b, M1) + interaction detector. Returns data/<id>. Skips finished steps."""
    from .config import Config
    from .detector import detect_interactions

    os.makedirs("demo_videos", exist_ok=True)
    vid = _vid(url)
    mp4 = os.path.join("demo_videos", vid + ".mp4")
    vdir = os.path.join("data", vid)
    print(f"\n=== {vid} ===")

    if not os.path.exists(mp4):
        print("[1/3] downloading video ...")
        import yt_dlp
        with yt_dlp.YoutubeDL({"format": "bv*[height<=720]", "outtmpl": mp4, "quiet": True, "no_warnings": True}) as ydl:
            ydl.download([url])
    else:
        print("[1/3] video already downloaded")

    if not os.path.exists(os.path.join(vdir, "transcript.json")):
        print("[2/3] running existing pipeline (transcription + keyframes + M1), this is slow on CPU ...")
        from orchestrator import PipelineOrchestrator
        PipelineOrchestrator(base_dir="data").process_detection_only(youtube_url=url)
    else:
        print("[2/3] pipeline outputs already exist")

    print("[3/3] detecting interactions ...")
    res = detect_interactions(mp4, os.path.join(vdir, "interactions.json"), Config(enable_hand=False))
    print(f"      {len(res)} interaction episodes -> {vdir}\\interactions.json")
    return vdir


def main():
    urls = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not urls:
        print("usage: python -m synapse_interaction.prepare URL [URL ...]")
        return
    for url in urls:
        prepare_video(url)
    print("\nNext: for each video run  python -m synapse_interaction.review data\\ID   (about 10 min each)")
    print("Then: python -m synapse_interaction.before_after data\\ID1 data\\ID2 ...")


if __name__ == "__main__":
    main()
