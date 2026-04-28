import os
import shutil
import subprocess
import numpy as np
import cv2
import yt_dlp
from skimage.metrics import structural_similarity as ssim


SCALE_WIDTH = 640

SSIM_THRESHOLD = 0.50  # Lower = more sensitive to changes (was 0.65)
MIN_GAP_SEC = 1.5  # Minimum gap between keyframes
MAX_GAP_SEC = 15.0  # Force save if no keyframe for this long
JPEG_QUALITY = 88
DATA_DIR = "data"


def _check_ffmpeg():
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"Required tool(s) not found in PATH: {', '.join(missing)}")


def get_stream_info(youtube_url):
    print("  Resolving stream URL via yt-dlp...")
    ydl_opts = {
        # Prefer H.264 (avc1) over AV1 — AV1 requires special decoder support
        "format": "bestvideo[vcodec^=avc1][height<=720]/bestvideo[vcodec^=avc][height<=720]/bestvideo[ext=mp4][height<=720]/best",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
    video_data = info["entries"][0] if "entries" in info else info
    return (
        video_data.get("url"),
        video_data.get("id", "vid"),
        video_data.get("title", "Title"),
        video_data.get("duration", 0),
    )


def probe_video(stream_url):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "csv=p=0",
        stream_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    line = next(
        (l.strip() for l in result.stdout.splitlines() if l.strip()), "1280,720,30/1"
    )
    parts = line.split(",")
    return (
        int(parts[0]),
        int(parts[1]),
        float(parts[2].split("/")[0]) / float(parts[2].split("/")[1]),
    )


def stream_and_extract(youtube_url):
    _check_ffmpeg()
    stream_url, video_id, title, duration = get_stream_info(youtube_url)
    orig_w, orig_h, fps = probe_video(stream_url)

    scale_h = int(orig_h * (SCALE_WIDTH / orig_w))
    scale_h += scale_h % 2
    bytes_per_frame = SCALE_WIDTH * scale_h * 3
    # Sample at ~1 fps instead of full framerate (huge speedup)
    sample_fps = 1.0
    min_gap_frames = max(1, int(sample_fps * MIN_GAP_SEC))
    max_gap_frames = max(1, int(sample_fps * MAX_GAP_SEC))

    print(f"\n  Title: {title}")
    print(f"  Duration: {duration}s | Sampling at {sample_fps} fps")
    print(f"  Logic: Structural Similarity (SSIM) - Best for STEM/Text")
    print("-" * 55)

    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        stream_url,
        "-vf",
        f"fps={sample_fps},scale={SCALE_WIDTH}:{scale_h}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=bytes_per_frame * 16
    )

    # State variables
    last_saved_ssim_frame = None
    last_saved_idx = -min_gap_frames
    frame_idx = 0
    keyframe_count = 0

    try:
        while True:
            raw = proc.stdout.read(bytes_per_frame)
            if len(raw) < bytes_per_frame:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (scale_h, SCALE_WIDTH, 3)
            )

            # 1. Convert to Grayscale for structural math (ignores color noise)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # 2. Skip empty/garbage frames
            # - Purely black: max pixel value <= 5
            # - Purely white: min pixel value >= 250
            # - Near-empty black: mean < 0.1 (almost all black, just noise)
            # - Near-empty white: mean > 254
            max_val = int(np.max(gray))
            min_val = int(np.min(gray))
            mean_val = float(np.mean(gray))

            is_garbage = False
            if max_val <= 5:
                is_garbage = True  # Purely black
            elif min_val >= 250:
                is_garbage = True  # Purely white
            elif mean_val < 0.1:
                is_garbage = True  # Near-black garbage (noise pixels only)
            elif mean_val > 254.0:
                is_garbage = True  # Near-white

            if is_garbage:
                frame_idx += 1
                continue

            # 3. Decision Logic
            gap_ok = (frame_idx - last_saved_idx) >= min_gap_frames
            gap_too_long = (frame_idx - last_saved_idx) >= max_gap_frames
            is_keyframe = False
            similarity_score = 0.0

            if last_saved_ssim_frame is None:
                is_keyframe = True  # Save the very first frame
            elif gap_too_long:
                is_keyframe = True  # Force save - too long without a keyframe
                similarity_score = ssim(last_saved_ssim_frame, gray)
            elif gap_ok:
                # Compare current frame structure to the last SAVED keyframe
                # SSIM returns 1.0 for perfect match, 0.0 for no match.
                similarity_score = ssim(last_saved_ssim_frame, gray)
                is_keyframe = similarity_score < SSIM_THRESHOLD

            if is_keyframe:
                keyframe_count += 1
                last_saved_idx = frame_idx
                last_saved_ssim_frame = gray.copy()  # Update reference

                # Save Logic
                ts = frame_idx / sample_fps
                h, m, s = int(ts // 3600), int((ts % 3600) // 60), int(ts % 60)
                folder = os.path.join(DATA_DIR, video_id, "keyframes")
                os.makedirs(folder, exist_ok=True)

                name = f"frame_{keyframe_count:04d}_{h:02d}_{m:02d}_{s:02d}.jpg"
                filepath = os.path.join(folder, name)

                # Convert back to BGR for OpenCV saving
                cv2.imwrite(
                    filepath,
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )

                print(
                    f"  [{keyframe_count:04d}] @ {h:02d}:{m:02d}:{s:02d} | Similarity: {similarity_score:.3f}"
                )

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    finally:
        proc.stdout.close()
        proc.terminate()

    print("-" * 55)
    print(f"  Extraction Complete. Saved {keyframe_count} keyframes.")
    print("=" * 55)
    return video_id, keyframe_count


if __name__ == "__main__":
    url = input("\nPaste a YouTube URL: ").strip()
    if "youtube.com" in url or "youtu.be" in url:
        stream_and_extract(url)
