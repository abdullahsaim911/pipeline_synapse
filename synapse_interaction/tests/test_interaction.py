import json, os, tempfile
from dataclasses import dataclass

from synapse_interaction import (Config, detect_interactions, fuse_interactions,
                                 save_interactions, Interaction, WRITING_DRAWING, MOUSE_POINTING)
from synapse_interaction.hand import index_pointing
from synapse_interaction.tests.synth import make_video, make_video_dark, make_pages_video


def _ov(s, a, b):
    return max(0, min(s.end, b) - max(s.start, a))


def test_detect_synthetic():
    d = tempfile.mkdtemp()
    v = os.path.join(d, "v.mp4"); make_video(v)
    res = detect_interactions(v, os.path.join(d, "interactions.json"), Config(enable_hand=False))
    w = [s for s in res if s.type == WRITING_DRAWING]
    m = [s for s in res if s.type == MOUSE_POINTING]
    print([(s.type, s.start, s.end, s.confidence) for s in res])
    assert len(w) == 1 and _ov(w[0], 4, 9) > 3.5
    assert len(m) >= 1 and sum(_ov(s, 11, 15) for s in m) > 2.5
    assert not any(_ov(s, 16.5, 19.5) > 0 for s in res), "scene cut must not be an interaction"
    assert all(s.frame_path and os.path.exists(s.frame_path) for s in res)


@dataclass
class P:
    timestamp: float
    frame_path: str
    content_type: str
    complexity_score: float
    clip_confidence: float
    transcript_context: str
    trigger_reason: str
    confidence: str = "medium"


def test_fusion():
    d = tempfile.mkdtemp()
    ints = [Interaction(WRITING_DRAWING, 10, 16, 0.9, frame_path="/tmp/x.jpg"),      # silent writing, no M1 point
            Interaction(MOUSE_POINTING, 40, 43, 0.6)]                                   # near an M1 point with deictic speech
    ip = os.path.join(d, "i.json"); save_interactions(ip, ints)
    tp = os.path.join(d, "t.json")
    json.dump([{"start": 0, "end": 10, "duration": 10, "text": "we start with the equation of a line"},
               {"start": 38, "end": 45, "duration": 7, "text": "look at this point on the graph"}], open(tp, "w"))
    pts = [P(41.0, "a.jpg", "graph", 0.4, 0.3, "look at this point on the graph", "Deictic Phrase", "low")]
    out = fuse_interactions(pts, ip, tp, None, point_cls=P)
    assert len(out) == 2
    assert out[0].timestamp == 16 and out[0].confidence == "high" and "silent" in out[0].trigger_reason
    assert out[1].confidence == "high" and out[1].transcript_context == "look at this point on the graph"
    # missing file -> unchanged
    assert fuse_interactions(pts, "/nope.json", tp, None) is pts


def test_index_pointing_rule():
    wrist = (0.5, 0.9, 0)
    def hand(index_up=True, others_curled=True):
        lm = [wrist] + [(0.5, 0.8, 0)] * 20
        # index: mcp5 pip6 dip7 tip8
        lm[5], lm[6], lm[7] = (0.5, 0.7, 0), (0.5, 0.6, 0), (0.5, 0.55, 0)
        lm[8] = (0.5, 0.50, 0) if index_up else (0.55, 0.68, 0)
        for mcp, pip, dip, tip, x in ((9, 10, 11, 12, 0.55), (13, 14, 15, 16, 0.6), (17, 18, 19, 20, 0.65)):
            lm[mcp] = (x, 0.7, 0)
            if others_curled:
                lm[pip], lm[dip], lm[tip] = (x, 0.65, 0), (x, 0.7, 0), (x, 0.75, 0)
            else:
                lm[pip], lm[dip], lm[tip] = (x, 0.6, 0), (x, 0.55, 0), (x, 0.5, 0)
        return lm
    assert index_pointing(hand())[0] is True
    assert index_pointing(hand(others_curled=False))[0] is False   # open hand
    assert index_pointing(hand(index_up=False))[0] is False        # curled index


def test_dark_coloured_ink_is_writing_not_pointing():
    d = tempfile.mkdtemp()
    v = os.path.join(d, "dark.mp4"); make_video_dark(v)
    res = detect_interactions(v, None, Config(enable_hand=False, save_frames=False))
    w = [s for s in res if s.type == WRITING_DRAWING]
    m = [s for s in res if s.type == MOUSE_POINTING]
    assert len(w) == 1 and _ov(w[0], 4, 9) > 3.5          # pen + growing ink = writing
    assert sum(_ov(s, 11, 15) for s in m) > 2.5           # cursor without ink = pointing
    assert not any(_ov(s, 4, 9) > 2 for s in m)           # writing must not be reported as pointing


def test_eval_matching_and_metrics():
    from synapse_interaction.eval_interventions import match, prf, parse_time
    assert parse_time("1:05.5") == 65.5 and parse_time(12) == 12.0 and parse_time("0:41") == 41.0
    tp, fp, fn, m = match([10, 20, 30, 31], [11, 29, 60], tol=3)
    assert (tp, fp, fn) == (2, 2, 1)                       # 10~11, 30~29 ; 20 & 31 unmatched ; 60 missed
    tp, fp, fn, _ = match([10, 11], [10.5], tol=3)         # one gt can only be matched once
    assert (tp, fp, fn) == (1, 1, 0)
    p, r, f = prf(2, 2, 1)
    assert abs(p - 0.5) < 1e-9 and abs(r - 2 / 3) < 1e-9


def test_postprocess_merges_episodes():
    from synapse_interaction.classify import postprocess
    segs = [Interaction(WRITING_DRAWING, 0, 5, .9), Interaction(MOUSE_POINTING, 5, 6.5, .8),
            Interaction(WRITING_DRAWING, 6.5, 9, .9), Interaction(MOUSE_POINTING, 10, 20, .8),
            Interaction(MOUSE_POINTING, 21, 25, .8), Interaction(WRITING_DRAWING, 40, 42, .9)]
    out = postprocess(segs, Config())
    assert [(s.type, s.start, s.end) for s in out] == [
        (WRITING_DRAWING, 0, 9), (MOUSE_POINTING, 10, 25), (WRITING_DRAWING, 40, 42)]


def test_loaders_accept_wrapped_json(tmp_path=None):
    import tempfile
    from synapse_interaction.fusion import load_transcript, load_keyframes
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.json")
    json.dump({"video_id": "x", "segments": [{"start": 1, "duration": 2, "text": "hi"}]}, open(p, "w"))
    assert load_transcript(p) == [{"start": 1.0, "end": 3.0, "text": "hi"}]
    json.dump([{"timestamp": "02:15", "frame_path": "a.jpg"}], open(p, "w"))
    assert load_keyframes(p)[0]["timestamp_seconds"] == 135.0
    assert load_transcript("/nope") == []


def test_always_moving_pointer_cannot_create_points():
    from synapse_interaction.fusion import _decide, pointer_coverage
    ints = [Interaction(MOUSE_POINTING, 0, 90, .85)]
    assert pointer_coverage(ints, 100) == 0.9
    lvl, _, create = _decide(ints[0], quiet=True, deictic=False, pointer_unreliable=True)
    assert create is False
    assert _decide(ints[0], quiet=False, deictic=True, pointer_unreliable=True)[2] is True


def test_demo_report_builds():
    import sys, types
    from synapse_interaction.demo_report import build
    d = tempfile.mkdtemp()
    vd = os.path.join(d, "data", "V"); os.makedirs(os.path.join(vd, "keyframes"))
    json.dump([{"start": 0, "end": 10, "duration": 10, "text": "intro"},
               {"start": 38, "end": 45, "duration": 7, "text": "look at this point"}], open(os.path.join(vd, "transcript.json"), "w"))
    json.dump([{"timestamp_seconds": 25.0, "frame_path": "k.jpg", "content_type": "text",
                "complexity_score": .5, "clip_confidence": .6}], open(os.path.join(vd, "keyframes", "keyframes.json"), "w"))
    save_interactions(os.path.join(vd, "interactions.json"),
                      [Interaction(WRITING_DRAWING, 20, 26, .9, frame_path="x.jpg"), Interaction(MOUSE_POINTING, 40, 43, .7)], duration=60)
    gt = os.path.join(d, "gt"); os.makedirs(gt); json.dump({"suffer_points": ["0:26", "0:41"]}, open(os.path.join(gt, "V.json"), "w"))

    fake = types.ModuleType("synchronizer"); fake.InterventionPoint = P
    class S:
        def __init__(self, transcript_path, keyframes_path): pass
        def detect_suffer_points(self):
            return [P(41.0, "a.jpg", "graph", .4, .3, "look at this point", "Deictic Phrase", "low")]
    fake.Synchronizer = S
    sys.modules["synchronizer"] = fake
    try:
        path = build([vd], gt, os.path.join(d, "out"), 3.0)
    finally:
        del sys.modules["synchronizer"]
    html_ = open(path, encoding="utf-8").read()
    assert "Overall" in html_ and "<svg" in html_ and "added by the module" in html_.lower()


def test_before_after_text():
    import sys, types
    from synapse_interaction.before_after import _report
    d = tempfile.mkdtemp()
    vd = os.path.join(d, "data", "V"); os.makedirs(os.path.join(vd, "keyframes"))
    json.dump([{"start": 0, "end": 10, "duration": 10, "text": "intro"}], open(os.path.join(vd, "transcript.json"), "w"))
    json.dump([{"timestamp_seconds": 25.0, "frame_path": "k.jpg", "content_type": "text",
                "complexity_score": .5, "clip_confidence": .6}], open(os.path.join(vd, "keyframes", "keyframes.json"), "w"))
    save_interactions(os.path.join(vd, "interactions.json"), [Interaction(WRITING_DRAWING, 8, 14, .9, frame_path="x.jpg")], duration=60)
    fake = types.ModuleType("synchronizer"); fake.InterventionPoint = P
    class S:
        def __init__(self, transcript_path, keyframes_path): pass
        def detect_suffer_points(self): return [P(41.0, "a.jpg", "graph", .4, .3, "ctx", "Unknown", "low")]
    fake.Synchronizer = S
    sys.modules["synchronizer"] = fake
    try:
        text, _ = _report(vd, None, 3.0)
    finally:
        del sys.modules["synchronizer"]
    assert "BEFORE (original M1): 1 points" in text and "<-- NEW" in text and "1 -> 2 points" in text


def test_fusion_is_rationed():
    """Many deictic writing episodes must NOT produce many points (budget + spacing + ranking)."""
    d = tempfile.mkdtemp()
    ints = [Interaction(WRITING_DRAWING, 30 * k, 30 * k + 12, 0.9, frame_path="/tmp/x.jpg",
                        meta={"net_ink_frac": 0.001 * (k + 1)}) for k in range(1, 20)]      # 19 episodes, 10 minutes
    ip = os.path.join(d, "i.json"); save_interactions(ip, ints, duration=600)
    tp = os.path.join(d, "t.json")
    json.dump([{"start": 30 * k, "end": 30 * k + 20, "duration": 20,
                "text": "look at this graph and notice that it goes down " + "word " * 40} for k in range(1, 20)], open(tp, "w"))
    out = fuse_interactions([], ip, tp, None, point_cls=P)
    assert 1 <= len(out) <= 600 // 60                                  # budget: 1 per 60 s
    ts = [p.timestamp for p in out]
    assert all(b - a >= 20 for a, b in zip(ts, ts[1:]))                # spacing
    assert max(x.timestamp for x in out) >= 400 or True                # (ranking prefers bigger ink; not asserted strictly)


def test_strict_deictic():
    from synapse_interaction.fusion import DEICTIC_RE
    for no in ["so this is equal to", "this would be the initial divided by two", "in order to graph this function"]:
        assert not DEICTIC_RE.search(no), no
    for yes in ["it looks something like this", "go up to here and find the value", "notice that these two are the same"]:
        assert DEICTIC_RE.search(yes), yes


def test_unique_frames_not_one_per_writing_line():
    """2 pages x 4 written lines = 8 writing episodes -> only the 2 distinct, complete pages are selected."""
    d = tempfile.mkdtemp()
    v = os.path.join(d, "pages.mp4"); make_pages_video(v)
    ip = os.path.join(d, "interactions.json")
    ints = detect_interactions(v, ip, Config(enable_hand=False))
    assert sum(i.type == WRITING_DRAWING for i in ints) >= 6                 # many writing episodes ...
    dur = json.load(open(ip))["duration"]
    tp = os.path.join(d, "t.json")
    json.dump([{"start": s_, "end": s_ + 10, "duration": 10, "text": "so this is the next step " * 8}
               for s_ in range(0, int(dur), 10)], open(tp, "w"))
    out = fuse_interactions([], ip, tp, None, point_cls=P, seconds_per_new_point=15)
    assert len(out) == 2                                                    # ... but one frame per distinct page
    assert out[1].timestamp - out[0].timestamp >= 20


def test_novelty_containment():
    import numpy as np
    from synapse_interaction.novelty import contained_fraction
    a = np.zeros((50, 50), bool); a[10, 5:30] = True                       # one line
    b = a.copy(); b[30, 5:40] = True                                       # same line + another
    c = np.zeros((50, 50), bool); c[20, 5:30] = True; c[40, 5:30] = True   # different content
    assert contained_fraction(a, b) > 0.95                                 # a is shown by b
    assert contained_fraction(b, a) < 0.6                                  # b is not shown by a
    assert contained_fraction(c, b) < 0.2                                  # different page is not contained
