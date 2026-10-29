"""The optional frame tap on the frame extractor must not change how the extractor works.

Off (the default): the extractor runs exactly as before. On: the tap sees every decoded frame, and the extractor
itself is handed the same frames, with the same timestamps and pixels, as without the tap.

The real extract_pedagogical_keyframes is run here. torch/PIL are replaced by empty stubs (only the frame-streaming
part, phase 2, is exercised): the function is stopped right after it by making the CLIP loader raise.
"""
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Stop(Exception):
    pass


def _video_frame(t):
    """The picture the fake video shows at time t: its pixels encode the time, so identical frames can be told."""
    fr = np.zeros((8, 8, 3), np.uint8)
    fr[:, :, 0] = int(round(t * 2)) % 250
    fr[0, 0, 1] = 7
    return fr


@pytest.fixture()
def ex(monkeypatch):
    class Anything(types.ModuleType):
        def __getattr__(self, item):           # torch.Tensor, Image.Image, ... : any attribute is fine
            return object
    for name in ("torch", "PIL", "PIL.Image"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, Anything(name))
    monkeypatch.syspath_prepend(ROOT)
    sys.modules.pop("frame_extraction.pedagogical_extractor_v2", None)
    module = pytest.importorskip("frame_extraction.pedagogical_extractor_v2")

    def fake_stream(url, width, height, fps=0.5, **kw):
        fake_stream.fps_asked.append(fps)
        for i in range(int(20 * fps)):                      # 20 seconds of video
            yield i / fps, _video_frame(i / fps).copy()
    fake_stream.fps_asked = []

    monkeypatch.setattr(module, "get_video_info", lambda vid: ("url", 20, "t", 8, 8))
    monkeypatch.setattr(module, "stream_frames", fake_stream)
    monkeypatch.setattr(module, "is_mostly_blank", lambda fr: False)
    monkeypatch.setattr(module, "get_clip", lambda: (_ for _ in ()).throw(_Stop()))
    module.fake_stream = fake_stream
    module.real_FrameData = module.FrameData
    return module


def _run(module, monkeypatch, **kw):
    """Run the real function until phase 2 is over; return what the extractor itself received."""
    got = []
    original = module.real_FrameData
    monkeypatch.setattr(module, "FrameData",
                        lambda timestamp, frame: got.append((timestamp, frame.tobytes())) or original(timestamp=timestamp, frame=frame))
    with pytest.raises(_Stop):
        module.extract_pedagogical_keyframes("vid", "out_unused", **kw)
    return got


def test_without_a_tap_extraction_is_exactly_as_before(ex, monkeypatch):
    got = _run(ex, monkeypatch)
    assert [t for t, _ in got] == [float(t) for t in range(0, 20, 2)]       # one frame every 2 s
    assert ex.fake_stream.fps_asked == [0.5]                                  # decoded at the extractor's own rate


def test_with_a_tap_the_extractor_gets_the_same_frames_and_the_tap_gets_all_of_them(ex, monkeypatch):
    plain = _run(ex, monkeypatch)
    tapped_times = []
    ex.fake_stream.fps_asked.clear()
    tapped = _run(ex, monkeypatch, frame_tap=lambda t, fr: tapped_times.append(t), tap_fps=2.0)
    assert tapped == plain                                                    # same timestamps AND same pixels
    assert len(tapped_times) == 40 and tapped_times[:3] == [0.0, 0.5, 1.0]   # the tap saw 2 frames per second
    assert ex.fake_stream.fps_asked == [2.0]                                  # the video was decoded once


def test_a_failing_tap_cannot_break_extraction(ex, monkeypatch):
    plain = _run(ex, monkeypatch)

    def broken(t, fr):
        raise RuntimeError("tap bug")

    assert _run(ex, monkeypatch, frame_tap=broken, tap_fps=2.0) == plain


def test_a_tap_slower_than_the_extractor_changes_nothing(ex, monkeypatch):
    plain = _run(ex, monkeypatch)
    seen = []
    assert _run(ex, monkeypatch, frame_tap=lambda t, fr: seen.append(t), tap_fps=0.25) == plain
    assert len(seen) == 10                                                    # it still sees every frame the extractor sees


def test_an_extractor_rate_that_does_not_divide_the_tap_rate_still_gets_its_usual_frames(ex, monkeypatch):
    """sample_fps 0.3 (a frame every 3.33 s): the tap rate is rounded up to a whole multiple so the grid is the same."""
    plain = _run(ex, monkeypatch, sample_fps=0.3)
    seen = []
    ex.fake_stream.fps_asked.clear()
    tapped = _run(ex, monkeypatch, sample_fps=0.3, frame_tap=lambda t, fr: seen.append(t), tap_fps=2.0)
    assert [round(t, 6) for t, _ in tapped] == [round(t, 6) for t, _ in plain]
    assert ex.fake_stream.fps_asked == [pytest.approx(2.1)]                   # 0.3 x 7
