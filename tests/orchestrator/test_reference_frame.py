"""The reference image saved for an intervention must be the picture the point was described from."""
import json
import os
import tempfile

from orchestrator.orchestrator import PipelineOrchestrator


def _orchestrator():
    return object.__new__(PipelineOrchestrator)      # no models or services are needed for this lookup


def _video_dir():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "keyframes"))
    for name in ("kf_0.jpg", "kf_20.jpg"):
        open(os.path.join(d, "keyframes", name), "wb").write(b"jpg")
    with open(os.path.join(d, "keyframes", "keyframes.json"), "w") as f:
        json.dump([{"timestamp_seconds": 0.0, "frame_path": os.path.join(d, "keyframes", "kf_0.jpg")},
                   {"timestamp_seconds": 20.0, "frame_path": os.path.join(d, "keyframes", "kf_20.jpg")}], f)
    return d


class _Point:
    def __init__(self, timestamp, frame_path):
        self.timestamp, self.frame_path = timestamp, frame_path


def test_without_remembered_points_the_nearest_keyframe_is_used_as_before():
    d = _video_dir()
    o = _orchestrator()
    assert o._find_frame_for_timestamp(18.0, d) == os.path.join(d, "keyframes", "kf_20.jpg")


def test_a_point_added_by_interaction_fusion_keeps_its_own_picture():
    d = _video_dir()
    own = os.path.join(d, "interaction_frames", "000_WRITING_DRAWING_0017.00.jpg")
    os.makedirs(os.path.dirname(own))
    open(own, "wb").write(b"jpg")
    o = _orchestrator()
    o._point_frames = {(os.path.normpath(d), 17.0): own}
    assert o._find_frame_for_timestamp(17.0, d) == own                                   # its own picture
    assert o._find_frame_for_timestamp(18.0, d) == os.path.join(d, "keyframes", "kf_20.jpg")   # other timestamps unchanged


def test_pictures_of_one_video_are_not_used_for_another():
    d1, d2 = _video_dir(), _video_dir()
    own = os.path.join(d1, "interaction_frames", "x.jpg")
    os.makedirs(os.path.dirname(own))
    open(own, "wb").write(b"jpg")
    o = _orchestrator()
    o._point_frames = {(os.path.normpath(d1), 17.0): own}
    assert o._find_frame_for_timestamp(17.0, d2) == os.path.join(d2, "keyframes", "kf_20.jpg")


def test_a_missing_file_falls_back_to_the_keyframe():
    d = _video_dir()
    o = _orchestrator()
    o._point_frames = {(os.path.normpath(d), 17.0): os.path.join(d, "gone.jpg")}
    assert o._find_frame_for_timestamp(17.0, d) == os.path.join(d, "keyframes", "kf_20.jpg")


def test_analysing_the_points_remembers_each_points_own_frame(monkeypatch):
    """_analyze_vlm_snapshots records the frames before any model is loaded (an invalid provider stops it right after)."""
    import pytest
    monkeypatch.setenv("SYNAPSE_VLM_PROVIDER", "nonsense")
    d = _video_dir()
    o = _orchestrator()
    points = [_Point(2.0, "a.jpg"), _Point(17.0, "interaction_frames/b.jpg")]
    with pytest.raises(ValueError):
        o._analyze_vlm_snapshots(points, d)
    assert o._point_frames == {(os.path.normpath(d), 2.0): "a.jpg", (os.path.normpath(d), 17.0): "interaction_frames/b.jpg"}
