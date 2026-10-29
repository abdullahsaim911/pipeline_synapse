"""Frame-fed detection: the module works on frames it is GIVEN and never opens a video itself."""
import math
import os
import tempfile

import numpy as np
import pytest

from synapse_interaction import Config, InteractionStream, WRITING_DRAWING, MOUSE_POINTING, HAND_POINTING
from synapse_interaction.hand import HandScanner, index_pointing, interactions_from_hits
from synapse_interaction.tests.synth import make_video
from synapse_interaction.video_io import iter_frames


def _decoded_frames(fps):
    """Frames as the frame extractor would hand them over (decoded once, here from the synthetic clip)."""
    d = tempfile.mkdtemp()
    v = os.path.join(d, "v.mp4")
    make_video(v)
    return list(iter_frames(v, fps, 640))


def test_frames_pushed_at_2fps_give_the_same_events_as_the_video_file_path():
    frames = _decoded_frames(2.0)
    d = tempfile.mkdtemp()
    out = os.path.join(d, "interactions.json")
    stream = InteractionStream(Config(enable_hand=False), out_path=out)
    for t, fr in frames:
        stream.feed(t, fr)
    res = stream.finish()
    kinds = {r.type for r in res}
    assert WRITING_DRAWING in kinds and MOUSE_POINTING in kinds, [(r.type, r.start, r.end) for r in res]
    w = next(r for r in res if r.type == WRITING_DRAWING)
    m = next(r for r in res if r.type == MOUSE_POINTING)
    assert 3.0 <= w.start <= 5.0 and 8.0 <= w.end <= 10.5      # the pen writes from 4 s to 9 s
    assert 10.0 <= m.start <= 12.0 and 14.0 <= m.end <= 16.5    # the pointer moves from 11 s to 15 s
    assert os.path.exists(out)


def test_event_pictures_come_from_the_pushed_frames_and_the_spool_is_removed():
    frames = _decoded_frames(2.0)
    d = tempfile.mkdtemp()
    out = os.path.join(d, "interactions.json")
    stream = InteractionStream(Config(enable_hand=False), out_path=out)
    for t, fr in frames:
        stream.feed(t, fr)
    res = stream.finish()
    assert res and all(r.frame_path and os.path.exists(r.frame_path) for r in res)
    assert not os.path.exists(os.path.join(d, "interaction_frames", "_spool"))


def test_feed_never_raises_on_bad_frames():
    stream = InteractionStream(Config(enable_hand=False, save_frames=False))
    for bad in (None, np.zeros((10, 10), np.uint8), "frame", np.zeros((0, 0, 3), np.uint8)):
        stream.feed(0.0, bad)       # must not raise into the frame extractor
    good = np.zeros((360, 640, 3), np.uint8)
    stream.feed(0.0, good)
    stream.feed(0.5, good)
    assert stream.finish() == []


def test_nothing_is_opened_or_downloaded(monkeypatch):
    """The streaming path must not touch cv2.VideoCapture or yt-dlp."""
    import cv2
    def boom(*a, **k):
        raise AssertionError("the interaction module opened a video")
    monkeypatch.setattr(cv2, "VideoCapture", boom)
    stream = InteractionStream(Config(enable_hand=False, save_frames=False))
    frame = np.zeros((360, 640, 3), np.uint8)
    for k in range(6):
        stream.feed(k * 0.5, frame)
    stream.finish()


# ---------------------------------------------------------------------------- hand ----------------
def _pointing_hand(angle_deg, W=1280, H=720, L=108.0):
    """Synthetic hand in PIXELS: index straight along `angle_deg` (0 = right, 90 = up), other fingers curled."""
    wx, wy = 640.0, 500.0
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), -math.sin(a)
    f = lambda k: (wx + ux * k * L, wy + uy * k * L)
    pts = {0: (wx, wy), 5: f(.6), 6: f(.85), 7: f(1.05), 8: f(1.25)}
    for mcp, pip, dip, tip in ((9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        pts[mcp], pts[pip], pts[dip], pts[tip] = f(.6), f(.8), f(.7), f(.55)
    px = [pts.get(i, (wx, wy)) for i in range(21)]
    return [(x / W, y / H, 0.0) for x, y in px]       # MediaPipe-style: x/width, y/height


@pytest.mark.parametrize("angle", [90, 60, 30, 0, -45, -90])
def test_pointing_is_recognised_in_every_direction_on_a_wide_frame(angle):
    lm = _pointing_hand(angle)
    assert index_pointing(lm, aspect=1280 / 720)[0] is True


class FakeHands:
    """Stands in for MediaPipe: reports a pointing hand in the frames listed in `at`."""
    def __init__(self, at):
        self.at = set(at)
        self.calls = 0

    def hand_hits(self, bgr):
        self.calls += 1
        t = round(float(bgr[0, 0, 0]) / 10.0, 1)      # the test paints the time into pixel [0,0]
        return (True, [((0.4, 0.3), (1.0, 0.0))]) if t in self.at else (False, [])

    def close(self):
        pass


def _timed_frame(t):
    fr = np.zeros((360, 640, 3), np.uint8)
    fr[0, 0, 0] = int(t * 10)
    return fr


def test_hand_scanner_merges_frames_into_one_pointing_event_and_rate_limits():
    cfg = Config(enable_hand=True, save_frames=False, hand_fps=2.0)
    fake = FakeHands(at=[2.0, 2.5, 3.0, 3.5])
    sc = HandScanner(cfg, mp=fake)
    for k in range(40):                    # frames arrive at 4 per second; the hand scan only needs 2
        sc.push(k * 0.25, _timed_frame(k * 0.25))
    events = sc.interactions()
    assert fake.calls == 20                # 2 per second over 10 s, not 40
    assert len(events) == 1 and events[0].type == HAND_POINTING
    assert 1.5 <= events[0].start <= 2.3 and 3.5 <= events[0].end <= 4.2


def test_a_single_flash_or_a_waving_finger_is_not_pointing():
    cfg = Config()
    one = [(5.0, 0.4, 0.3, 1.0, 0.0)]
    assert interactions_from_hits(one, cfg) == []            # shorter than hand_min_dur
    waving = [(t, 0.2 + 0.6 * (k % 2), 0.3, 1.0, 0.0) for k, t in enumerate([5.0, 5.5, 6.0, 6.5, 7.0])]
    assert interactions_from_hits(waving, cfg) == []         # fingertip speed far above the limit


def test_hand_stage_in_a_stream_with_no_mediapipe_just_skips():
    stream = InteractionStream(Config(enable_hand=True, save_frames=False))   # mediapipe not installed here
    frame = np.zeros((360, 640, 3), np.uint8)
    stream.feed(0.0, frame)
    stream.feed(0.5, frame)
    assert stream.finish() == []


def test_the_module_cannot_download_or_stream_video():
    """Architecture guard: no yt-dlp anywhere, and only video_io.py (file-only helpers) may open a video."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        src = open(os.path.join(root, name), encoding="utf-8").read()
        assert "import yt_dlp" not in src, name
        if name != "video_io.py":
            assert "VideoCapture" not in src, name


def test_urls_and_youtube_ids_are_refused():
    from synapse_interaction.video_io import resolve_stream
    for source in ("https://www.youtube.com/watch?v=aircAruvnKk", "aircAruvnKk"):
        with pytest.raises(ValueError):
            resolve_stream(source)
