"""M1b - Interaction Fusion.

Takes M1's List[InterventionPoint] (UNCHANGED) + interactions.json and returns a list of the same
type. Existing points are never removed: they can be upgraded (confidence / trigger_reason) and
new points can be added for interactions M1 missed. M2/M3 see the same contract as before.

Do NOT edit `transcript_context`: M3 locates the current transcript entry by matching that string.
"""
from __future__ import annotations
import json
import os
import re
from typing import List, Optional

from .schema import Interaction, load_interactions, WRITING_DRAWING, MOUSE_POINTING

LEVELS = ["low", "medium", "high"]
# STRICT: only phrases where the teacher points at something visual. Plain "this"/"here" appear in almost
# every sentence ("this is equal to...") so they are NOT used. TIP: swap in M1's own deictic list to stay consistent.
DEICTIC_RE = re.compile(
    r"\b(look at (this|that|these|those|here|the)|take a look|as you can see|you can see|can see (that|how|here)|"
    r"see (how|that|here|what)|notice (that|how|these|this)|"
    r"over here|right here|up here|down here|up to here|down to here|over to here|go (up|down|over) to|"
    r"(look|looks|looking) (something )?like this|like this|(this|that) one|(these|those) two|"
    r"(this|that) (graph|point|line|curve|axis|part|side|value|region|area|slope|intercept|diagram|picture|figure|shape))\b",
    re.I)

def _lvl(x: str) -> int:
    return LEVELS.index(x) if x in LEVELS else 1


def _first_list(data):
    """transcript.json / keyframes.json may be a bare list or wrapped in a dict."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("segments", "entries", "transcript", "transcripts", "keyframes", "frames", "data", "items"):
            if isinstance(data.get(k), list):
                return data[k]
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _raw_json(path: Optional[str]):
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _mmss(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    sec = 0.0
    for p in str(x).split(":"):
        sec = sec * 60 + float(p)
    return sec


def load_transcript(path: Optional[str]) -> list:
    """-> [{"start": s, "end": e, "text": str}, ...] whatever the file layout is."""
    out = []
    for e in _first_list(_raw_json(path)):
        if not isinstance(e, dict):
            continue
        try:
            s = _mmss(e.get("start", e.get("start_time", e.get("timestamp", 0.0))))
            end = e.get("end", e.get("end_time"))
            end = _mmss(end) if end is not None else s + float(e.get("duration", 0.0))
        except Exception:
            continue
        out.append({"start": s, "end": end, "text": str(e.get("text", ""))})
    return out


def load_keyframes(path: Optional[str]) -> list:
    out = []
    for e in _first_list(_raw_json(path)):
        if not isinstance(e, dict):
            continue
        try:
            ts = e.get("timestamp_seconds", e.get("timestamp", 0.0))
            e = dict(e); e["timestamp_seconds"] = _mmss(ts)
        except Exception:
            continue
        out.append(e)
    return out


def pointer_coverage(interactions: List[Interaction], duration: Optional[float] = None) -> float:
    """Fraction of the video covered by MOUSE_POINTING. A pointer that is ALWAYS moving carries no
    information about *where the teacher wants attention*, so pointing is then treated as weak."""
    if not interactions:
        return 0.0
    total = duration or max(i.end for i in interactions)
    pt = sum(i.duration for i in interactions if i.type == MOUSE_POINTING)
    return pt / total if total else 0.0


def _overlapping(transcripts, a: float, b: float):
    return [t for t in transcripts if t["end"] > a and t["start"] < b]


def _words_per_sec(transcripts, a: float, b: float) -> float:
    dur = max(1e-6, b - a)
    words = 0.0
    for t in _overlapping(transcripts, a, b):
        seg = max(1e-6, t["end"] - t["start"])
        ov = min(b, t["end"]) - max(a, t["start"])
        words += len(t["text"].split()) * max(0.0, ov) / seg
    return words / dur


def _decide(seg: Interaction, quiet: bool, deictic: bool, pointer_unreliable: bool = False):
    """-> (level, reason, may_create_new_point)"""
    if seg.type == WRITING_DRAWING:
        if quiet:
            return "high", "WRITING_DRAWING (silent)", True
        if deictic:
            return "high", "WRITING_DRAWING + deictic", True
        return "medium", "WRITING_DRAWING", False
    # pointing (hand or mouse)
    if deictic:
        return "high", f"{seg.type} + deictic", True
    if pointer_unreliable and seg.type == MOUSE_POINTING:
        return "medium", seg.type, False       # pointer moves all the time: weak evidence on its own
    if quiet:
        return "high", f"{seg.type} (silent)", True
    return "medium", seg.type, seg.confidence >= 0.8


def _is_gesture_frame(keyframe_time: float, seg: Interaction, tol: float = 0.25) -> bool:
    """True when the extractor's keyframe shows the SAME moment as the event's picture, so there is nothing to choose.

    Pointing (hand or mouse): the pointer is visible during the whole event, so any keyframe taken inside it is a
    gesture frame. Writing: an episode can last minutes and its picture is the COMPLETE board taken just after the last
    stroke, so only a keyframe at that moment is the same picture; a keyframe from the middle of the writing shows a
    half-written board and has to compete on confidence like any other."""
    if seg.type == WRITING_DRAWING:
        return abs(keyframe_time - (seg.end + 0.3)) <= tol
    return seg.start - tol <= keyframe_time <= seg.end + tol


def _gesture_picture_wins(point_conf: float, seg: Interaction, keyframe_time: Optional[float] = None) -> bool:
    """The picture with the higher confidence is the one the VLM should describe. A tie keeps the extractor's keyframe
    (the behaviour before this rule existed). A keyframe that is itself a gesture frame always keeps its picture."""
    if keyframe_time is not None and _is_gesture_frame(keyframe_time, seg):
        return False
    return bool(seg.frame_path) and os.path.exists(seg.frame_path) and seg.confidence > point_conf


def _default_point_cls(points):
    for mod in ("synchronizer", "synchronizer.synchronizer"):
        try:
            return getattr(__import__(mod, fromlist=["InterventionPoint"]), "InterventionPoint")
        except Exception:
            continue
    return type(points[0]) if points else None


def _video_duration(interactions_path: str, interactions) -> float:
    try:
        d = _raw_json(interactions_path).get("duration")
        if d:
            return float(d)
    except Exception:
        pass
    return max((i.end for i in interactions), default=0.0)


def fuse_interactions(points: list, interactions_path: str, transcript_path: Optional[str] = None,
                      keyframes_path: Optional[str] = None, point_cls=None,
                      match_slack: float = 2.0, seconds_per_new_point: float = 60.0,
                      min_new_gap: float = 20.0, min_new_score: float = 1.0,
                      min_ink_px: int = 300, redundant_thr: float = 0.80, pic_window: float = 4.0) -> list:
    """Never raises: on any problem the original list is returned.

    Upgrades: an existing M1 point near an interaction gets a higher confidence + a reason.
    New points are UNIQUE, content-rich frames (not one per writing line):
      * richest first : candidates are ranked by how much is on screen (a complete page beats its partial versions)
      * novelty       : skipped if >= `redundant_thr` of the frame's ink is already shown by a selected frame
                        (M1's own frames count too) -> only visually different frames survive
      * content       : needs >= `min_ink_px` ink pixels (at 480 px width) so an empty page is never picked
      * spacing/budget: >= `min_new_gap` s from every other point; at most 1 new point per `seconds_per_new_point` s

    Which PICTURE the VLM describes: the one with the higher confidence. A point's keyframe is scored by its
    `clip_confidence`, a gesture picture by the event's `confidence`. For an existing point near an event the gesture
    picture replaces the keyframe only if it scores higher; for a new point the nearest keyframe (within `pic_window`
    seconds) competes with the gesture picture in the same way. Ties keep the keyframe. A keyframe taken while the event
    was happening is itself a gesture frame: it keeps its picture without any comparison.
    """
    try:
        interactions = load_interactions(interactions_path)
    except Exception:
        return points
    try:
        return _fuse(points, interactions, interactions_path, transcript_path, keyframes_path, point_cls,
                     match_slack, seconds_per_new_point, min_new_gap, min_new_score, min_ink_px, redundant_thr,
                     pic_window)
    except Exception:
        return points


def _fuse(points, interactions, interactions_path, transcript_path, keyframes_path, point_cls,
          match_slack, seconds_per_new_point, min_new_gap, min_new_score, min_ink_px, redundant_thr,
          pic_window=4.0):
    from .novelty import load_mask, contained_fraction
    transcripts = load_transcript(transcript_path)
    keyframes = load_keyframes(keyframes_path)
    point_cls = point_cls or _default_point_cls(points)
    duration = _video_duration(interactions_path, interactions)
    unreliable = pointer_coverage(interactions, duration or None) > 0.40
    out = list(points)
    pic_conf = {id(p): float(getattr(p, "clip_confidence", 0.0) or 0.0) for p in out}   # confidence of each point's picture

    cands = []        # dicts: seg, grid, ink, quiet, deictic
    for seg in interactions:
        lo, hi = seg.start - match_slack, seg.end + match_slack
        quiet = _words_per_sec(transcripts, seg.start, seg.end) < 1.0
        deictic = any(DEICTIC_RE.search(t["text"]) for t in _overlapping(transcripts, lo, hi))
        level, reason, _ = _decide(seg, quiet, deictic, unreliable)

        near = [p for p in out if lo <= p.timestamp <= hi]
        if near:                                       # upgrade an existing point, never duplicate
            for p in near:
                if _lvl(level) > _lvl(p.confidence):
                    p.confidence = level
                if reason not in (p.trigger_reason or ""):
                    p.trigger_reason = f"{p.trigger_reason} + {reason}" if p.trigger_reason else reason
                if _gesture_picture_wins(pic_conf.get(id(p), 0.0), seg, p.timestamp):
                    p.frame_path = seg.frame_path          # the better-scored picture is the one to describe
                    pic_conf[id(p)] = seg.confidence
            continue
        if point_cls is None or (unreliable and seg.type == MOUSE_POINTING):
            continue
        grid = load_mask(seg.frame_path)
        cands.append({"seg": seg, "grid": grid, "ink": int(grid.sum()) if grid is not None else 0,
                      "quiet": quiet, "deictic": deictic})

    # ---- rank: richest content first (a complete page beats its half-written versions) ----
    max_ink = max((c["ink"] for c in cands), default=0) or 1
    for c in cands:
        c["score"] = (c["ink"] / max_ink) + 0.3 * c["quiet"] + 0.3 * c["deictic"] + 0.2 * min(1.0, c["seg"].duration / 8.0)
    cands.sort(key=lambda c: -c["score"])

    refs = [g for g in (load_mask(getattr(p, "frame_path", None)) for p in out) if g is not None]   # M1's own frames
    budget = max(1, int(duration // seconds_per_new_point)) if duration else 1
    added = 0
    for c in cands:
        if added >= budget:
            break
        seg, grid = c["seg"], c["grid"]
        if grid is not None:
            if c["ink"] < min_ink_px:               # not enough on screen to be worth describing
                continue
            if any(contained_fraction(grid, r) >= redundant_thr for r in refs):
                continue                               # same content already covered by another selected frame
        elif not (c["quiet"] or c["deictic"]):         # no image to judge -> fall back to the speech cues
            continue
        t = seg.end if seg.type == WRITING_DRAWING else (seg.start + seg.end) / 2
        if any(abs(t - p.timestamp) < min_new_gap for p in out):
            continue
        kf = min(keyframes, key=lambda k: abs(k.get("timestamp_seconds", 0) - t)) if keyframes else {}
        frame_path = seg.frame_path or kf.get("frame_path", "")
        kf_near = bool(kf) and abs(kf.get("timestamp_seconds", 0) - t) <= pic_window
        if kf_near and kf.get("frame_path") and not _gesture_picture_wins(
                float(kf.get("clip_confidence", 0.0) or 0.0), seg, kf.get("timestamp_seconds", 0)):
            frame_path = kf["frame_path"]                  # the keyframe scores at least as high: describe that one
        if not frame_path:
            continue                                   # M2 needs an image
        ctx = ""
        if transcripts:
            hit = _overlapping(transcripts, t - 0.01, t + 0.01) or \
                  [min(transcripts, key=lambda x: min(abs(x["start"] - t), abs(x["end"] - t)))]
            ctx = hit[0]["text"]
        why = "New visual content (" + ("writing" if seg.type == WRITING_DRAWING else "pointing") + \
              (", silent" if c["quiet"] else "") + (", deictic" if c["deictic"] else "") + ")"
        out.append(point_cls(timestamp=round(t, 2), frame_path=frame_path,
                             content_type=kf.get("content_type", "text"),
                             complexity_score=kf.get("complexity_score", 0.0),
                             clip_confidence=kf.get("clip_confidence", 0.0),
                             transcript_context=ctx, trigger_reason=why,
                             confidence="high" if (c["quiet"] or c["deictic"]) else "medium"))
        if grid is not None:
            refs.append(grid)
        added += 1
    return sorted(out, key=lambda p: p.timestamp)
