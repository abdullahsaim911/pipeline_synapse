"""The whole path the pipeline takes: decoded frames -> InteractionStream -> interactions.json ->
Synchronizer's suffer points -> fuse_interactions. Uses the REAL Synchronizer and InterventionPoint."""
import json
import os
import tempfile

import cv2
import numpy as np

from synapse_interaction import Config, InteractionStream, fuse_interactions, load_interactions
from synapse_interaction.tests.synth import make_pages_video
from synapse_interaction.video_io import iter_frames
from synchronizer import Synchronizer, InterventionPoint


def _build(tmp):
    video = os.path.join(tmp, "v.mp4")
    make_pages_video(video, pages=1, lines=3, pause=3.0)            # teacher writes three lines, pausing between
    frames = list(iter_frames(video, 2.0, 640))                      # what the extractor would hand over
    vdir = os.path.join(tmp, "data", "VID")
    os.makedirs(os.path.join(vdir, "keyframes"))
    out = os.path.join(vdir, "interactions.json")
    stream = InteractionStream(Config(enable_hand=False), out_path=out)
    for t, fr in frames:
        stream.feed(t, fr)
    events = stream.finish()

    # What the extractor and the transcription would have produced for the same video (sparse speech)
    keyframes = []
    for ts in (2.0, 20.0):
        t, fr = min(frames, key=lambda f: abs(f[0] - ts))
        p = os.path.join(vdir, "keyframes", f"frame_{int(ts):04d}.jpg")
        cv2.imwrite(p, fr)
        keyframes.append({"timestamp": "00:00", "timestamp_seconds": ts, "frame_path": p,
                          "content_type": "text", "complexity_score": 0.35, "clip_confidence": 0.6})
    with open(os.path.join(vdir, "keyframes", "keyframes.json"), "w") as f:
        json.dump(keyframes, f)
    with open(os.path.join(vdir, "transcript.json"), "w") as f:
        json.dump({"entries": [{"start": 0.0, "duration": 6.0, "text": "okay so let us begin with the first example"},
                               {"start": 6.0, "duration": 30.0, "text": "uh"}]}, f)   # then the teacher goes quiet
    return vdir, out, events


def test_fusion_receives_the_detected_events_and_the_pipeline_contract_holds():
    tmp = tempfile.mkdtemp()
    vdir, out, events = _build(tmp)
    assert events, "the writing on the synthetic page should be detected"
    assert os.path.exists(out)

    sync = Synchronizer(os.path.join(vdir, "transcript.json"), os.path.join(vdir, "keyframes", "keyframes.json"))
    points = sync.detect_suffer_points()
    before = [(p.timestamp, p.frame_path, p.transcript_context) for p in points]

    fused = fuse_interactions(points, out, os.path.join(vdir, "transcript.json"),
                              os.path.join(vdir, "keyframes", "keyframes.json"))

    assert all(isinstance(p, InterventionPoint) for p in fused)          # same type M2/M3 already expect
    assert [p.timestamp for p in fused] == sorted(p.timestamp for p in fused)
    kept = {(p.timestamp, p.transcript_context) for p in fused}
    assert {(t, c) for t, _, c in before} <= kept                        # no point removed, no transcript text rewritten
    for t, original_picture, _ in before:                                # the picture is the keyframe or a better-scored gesture picture
        now = next(p.frame_path for p in fused if p.timestamp == t)
        assert now == original_picture or "interaction_frames" in now.replace("\\", "/")
    assert all(p.frame_path and os.path.exists(p.frame_path) for p in fused)   # every point has a real picture
    assert all(p.confidence in ("low", "medium", "high") for p in fused)
    assert any("WRITING_DRAWING" in (p.trigger_reason or "") or (p.trigger_reason or "").startswith("New visual")
               for p in fused), [(p.timestamp, p.trigger_reason) for p in fused]


def test_new_points_use_the_picture_taken_from_the_pushed_frames():
    tmp = tempfile.mkdtemp()
    vdir, out, _ = _build(tmp)
    sync = Synchronizer(os.path.join(vdir, "transcript.json"), os.path.join(vdir, "keyframes", "keyframes.json"))
    points = sync.detect_suffer_points()
    fused = fuse_interactions(points, out, os.path.join(vdir, "transcript.json"),
                              os.path.join(vdir, "keyframes", "keyframes.json"))
    added = [p for p in fused if (p.trigger_reason or "").startswith("New visual")]
    for p in added:
        assert "interaction_frames" in p.frame_path.replace("\\", "/")
        img = cv2.imread(p.frame_path)
        assert img is not None and img.shape[1] <= 1280


def test_a_rerun_does_not_leave_old_pictures_behind():
    tmp = tempfile.mkdtemp()
    vdir, out, _ = _build(tmp)
    folder = os.path.join(vdir, "interaction_frames")
    stale = os.path.join(folder, "999_WRITING_DRAWING_0001.00.jpg")
    cv2.imwrite(stale, np.zeros((10, 10, 3), np.uint8))
    InteractionStream(Config(enable_hand=False), out_path=out)           # starting a new run clears the folder
    assert not os.path.exists(stale)


def test_hand_pointing_events_flow_through_fusion_like_the_others():
    """A pointing hand while the teacher says 'look at this graph' upgrades a nearby point and, far from any,
    creates a new one with its own picture."""
    from synapse_interaction import Interaction, HAND_POINTING, save_interactions
    tmp = tempfile.mkdtemp()
    vdir, out, _ = _build(tmp)
    pic = os.path.join(vdir, "interaction_frames", "hand.jpg")
    os.makedirs(os.path.dirname(pic), exist_ok=True)
    canvas = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(canvas, "y = x^2", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    cv2.imwrite(pic, canvas)
    hand = [Interaction(HAND_POINTING, 1.5, 3.0, 0.8, source="mediapipe", frame_path=None),         # near the point at 2 s
            Interaction(HAND_POINTING, 40.0, 43.0, 0.8, source="mediapipe", frame_path=pic)]       # alone, deictic speech
    save_interactions(out, hand, video="x", duration=60.0)
    tp = os.path.join(vdir, "transcript.json")
    with open(tp, "w") as f:
        json.dump({"entries": [{"start": 0.0, "duration": 6.0, "text": "okay so let us begin"},
                               {"start": 38.0, "duration": 8.0, "text": "now look at this graph right here"}]}, f)
    kp = os.path.join(vdir, "keyframes", "keyframes.json")
    points = Synchronizer(tp, kp).detect_suffer_points()
    fused = fuse_interactions(points, out, tp, kp)
    near = next(p for p in fused if p.timestamp == 2.0)
    assert "HAND_POINTING" in near.trigger_reason
    added = [p for p in fused if p.timestamp > 30]
    assert added and added[0].frame_path == pic
    assert "pointing" in added[0].trigger_reason and added[0].confidence == "high"


# --------------------------------------------------------------- which picture the VLM describes -------------
def _scene(keyframe_conf, event_conf, event_at=5.0, keyframe_at=2.0, event_type="WRITING_DRAWING"):
    """One extractor keyframe (given clip_confidence) and one writing event with its own picture."""
    from synapse_interaction import Interaction, save_interactions
    tmp = tempfile.mkdtemp()
    vdir = os.path.join(tmp, "data", "VID")
    os.makedirs(os.path.join(vdir, "keyframes"))
    os.makedirs(os.path.join(vdir, "interaction_frames"))
    page = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(page, "y = x^2", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    kf_pic = os.path.join(vdir, "keyframes", "kf.jpg")
    ev_pic = os.path.join(vdir, "interaction_frames", "ev.jpg")
    cv2.imwrite(kf_pic, page)
    cv2.imwrite(ev_pic, page)
    other = np.zeros((360, 640, 3), np.uint8)                          # a different page, for an unrelated point
    cv2.putText(other, "F = ma", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    cv2.line(other, (20, 300), (620, 340), (255, 255, 255), 6)
    other_pic = os.path.join(vdir, "keyframes", "other.jpg")
    cv2.imwrite(other_pic, other)
    kp = os.path.join(vdir, "keyframes", "keyframes.json")
    json.dump([{"timestamp_seconds": keyframe_at, "frame_path": kf_pic, "content_type": "equation",
                "complexity_score": 0.4, "clip_confidence": keyframe_conf}], open(kp, "w"))
    tp = os.path.join(vdir, "transcript.json")
    json.dump({"entries": [{"start": 0.0, "duration": 60.0, "text": "uh"}]}, open(tp, "w"))   # silent teacher
    out = os.path.join(vdir, "interactions.json")
    ev = Interaction(event_type, event_at - 1.0, event_at, event_conf,
                     source="motion", frame_path=ev_pic)
    save_interactions(out, [ev], video="x", duration=60.0)
    _scene.other_pic = other_pic
    return tp, kp, out, kf_pic, ev_pic


def _points(timestamp, frame_path, conf="medium"):
    return [InterventionPoint(timestamp=timestamp, frame_path=frame_path, content_type="equation",
                              complexity_score=0.4, clip_confidence=0.0, transcript_context="uh",
                              trigger_reason="High Complexity", confidence=conf)]


def test_existing_point_describes_the_gesture_picture_when_it_scores_higher():
    tp, kp, out, kf_pic, ev_pic = _scene(keyframe_conf=0.5, event_conf=0.9)
    pts = _points(2.0, kf_pic)
    pts[0].clip_confidence = 0.5
    fused = fuse_interactions(pts, out, tp, kp)
    assert len(fused) == 1 and fused[0].frame_path == ev_pic          # 0.9 beats 0.5


def test_existing_point_keeps_its_keyframe_when_the_keyframe_scores_higher_or_equal():
    for kf_conf in (0.95, 0.9):                                       # higher, and a tie
        tp, kp, out, kf_pic, ev_pic = _scene(keyframe_conf=kf_conf, event_conf=0.9)
        pts = _points(2.0, kf_pic)
        pts[0].clip_confidence = kf_conf
        fused = fuse_interactions(pts, out, tp, kp)
        assert fused[0].frame_path == kf_pic, kf_conf


def test_new_point_uses_the_higher_scoring_picture():
    # no existing point near the event, so fusion creates one; a keyframe 3 s before it competes with the gesture picture
    for kf_conf, expect_gesture in ((0.5, True), (0.95, False)):
        tp, kp, out, kf_pic, ev_pic = _scene(keyframe_conf=kf_conf, event_conf=0.9, event_at=40.0, keyframe_at=37.0)
        far = _points(2.0, _scene.other_pic)                          # some other, distant point showing a different page
        fused = fuse_interactions(far, out, tp, kp)
        new = [p for p in fused if p.timestamp > 30]
        assert new, "a point should have been created for the event"
        assert new[0].frame_path == (ev_pic if expect_gesture else kf_pic), kf_conf


def test_a_distant_keyframe_does_not_compete():
    tp, kp, out, kf_pic, ev_pic = _scene(keyframe_conf=0.99, event_conf=0.9, event_at=40.0, keyframe_at=10.0)
    fused = fuse_interactions(_points(2.0, _scene.other_pic), out, tp, kp)
    new = [p for p in fused if p.timestamp > 30]
    assert new and new[0].frame_path == ev_pic                        # the 0.99 keyframe is 30 s away: ignored


def test_a_keyframe_taken_during_a_pointing_event_is_a_gesture_frame_and_keeps_its_picture():
    # a hand points from 4.0 to 5.0 s and the extractor's keyframe was taken at 4.5 s: the same moment
    tp, kp, out, kf_pic, ev_pic = _scene(keyframe_conf=0.3, event_conf=0.95, event_at=5.0, keyframe_at=4.5,
                                         event_type="HAND_POINTING")
    pts = _points(4.5, kf_pic)
    pts[0].clip_confidence = 0.3
    fused = fuse_interactions(pts, out, tp, kp)
    assert fused[0].frame_path == kf_pic                              # 0.95 > 0.3, but the keyframe IS the gesture frame
    assert "HAND_POINTING" in fused[0].trigger_reason                 # and the point is still marked as a gesture point


def test_a_new_pointing_point_uses_a_keyframe_taken_during_the_event():
    tp, kp, out, kf_pic, ev_pic = _scene(keyframe_conf=0.3, event_conf=0.95, event_at=40.0, keyframe_at=39.5,
                                         event_type="HAND_POINTING")
    fused = fuse_interactions(_points(2.0, _scene.other_pic), out, tp, kp)
    new = [p for p in fused if p.timestamp > 30]
    assert new and new[0].frame_path == kf_pic


def test_a_keyframe_from_the_middle_of_writing_does_not_beat_the_finished_board():
    # writing runs 4.0 to 5.0 s; its picture is the finished board at 5.3 s. A keyframe at 4.5 s shows half a board,
    # so it competes on score (0.95 > 0.3) instead of winning automatically.
    tp, kp, out, kf_pic, ev_pic = _scene(keyframe_conf=0.3, event_conf=0.95, event_at=5.0, keyframe_at=4.5)
    pts = _points(4.5, kf_pic)
    pts[0].clip_confidence = 0.3
    assert fuse_interactions(pts, out, tp, kp)[0].frame_path == ev_pic


def test_a_keyframe_taken_when_the_writing_is_finished_is_the_same_picture():
    tp, kp, out, kf_pic, ev_pic = _scene(keyframe_conf=0.3, event_conf=0.95, event_at=5.0, keyframe_at=5.2)
    pts = _points(5.2, kf_pic)
    pts[0].clip_confidence = 0.3
    assert fuse_interactions(pts, out, tp, kp)[0].frame_path == kf_pic
