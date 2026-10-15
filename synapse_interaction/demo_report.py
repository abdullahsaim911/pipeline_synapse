"""One-page HTML demo for your supervisor: does the interaction module improve intervention points?

    python -m synapse_interaction.demo_report data\\ID1 data\\ID2 data\\ID3 --gt-dir ground_truth --out demo_report

Needs, for every video folder: transcript.json, keyframes\\keyframes.json (from the existing pipeline) and
interactions.json (from `synapse_interaction.run`/`prepare`). Answer keys (ground_truth\\ID.json) are optional but
needed for precision / recall / F1. The original M1 is NOT modified: the "after" list is M1 + fusion layer.
"""
from __future__ import annotations
import argparse
import base64
import html
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2

from .eval_interventions import load_ground_truth, match, prf
from .fusion import fuse_interactions, load_transcript
from .schema import load_interactions, WRITING_DRAWING, MOUSE_POINTING, HAND_POINTING

COLORS = {WRITING_DRAWING: "#2e9e5b", MOUSE_POINTING: "#e8890c", HAND_POINTING: "#2f6fdb"}
NAMES = {WRITING_DRAWING: "Writing / drawing", MOUSE_POINTING: "Mouse pointing", HAND_POINTING: "Hand pointing"}


def _m1_and_fused(video_dir: str):
    from synchronizer import Synchronizer          # the project's own, unmodified M1
    tp = os.path.join(video_dir, "transcript.json")
    kp = os.path.join(video_dir, "keyframes", "keyframes.json")
    ip = os.path.join(video_dir, "interactions.json")
    base = Synchronizer(transcript_path=tp, keyframes_path=kp).detect_suffer_points()
    base_info = [(p.timestamp, p.confidence, p.trigger_reason) for p in base]
    fused = fuse_interactions(
        Synchronizer(transcript_path=tp, keyframes_path=kp).detect_suffer_points(), ip, tp, kp)
    return base_info, fused


def _duration(video_dir: str, interactions, transcript) -> float:
    try:
        d = json.load(open(os.path.join(video_dir, "interactions.json"), encoding="utf-8")).get("duration")
        if d:
            return float(d)
    except Exception:
        pass
    ends = [i.end for i in interactions] + [t["end"] for t in transcript]
    return max(ends) if ends else 60.0


def _thumb(path: Optional[str], width: int = 340) -> str:
    if not path or not os.path.exists(path):
        return ""
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return '<img src="data:image/jpeg;base64,%s" alt="frame">' % base64.b64encode(buf).decode()


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _timeline(dur, speech, ints, base_t, added_t, upgraded_t, gt) -> str:
    W, L, R = 1000, 140, 20
    x = lambda t: L + (min(max(t, 0), dur) / dur) * (W - L - R)
    rows = [("Teacher speaking", 30), ("Writing / drawing", 62), ("Mouse pointing", 90), ("Hand pointing", 118),
            ("M1 alone", 160), ("M1 + module", 198), ("Answer key", 236)]
    H = 270
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="12">']
    for name, y in rows:
        s.append(f'<text x="6" y="{y + 4}" fill="#555">{html.escape(name)}</text>'
                 f'<line x1="{L}" x2="{W - R}" y1="{y}" y2="{y}" stroke="#e3e3e3"/>')
    step = 30 if dur <= 420 else 60
    t = 0
    while t <= dur:
        s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="14" y2="{H - 22}" stroke="#f0f0f0"/>'
                 f'<text x="{x(t):.1f}" y="{H - 6}" text-anchor="middle" fill="#888">{_mmss(t)[:-2]}</text>')
        t += step
    for e in speech:
        s.append(f'<rect x="{x(e["start"]):.1f}" y="22" width="{max(1, x(e["end"]) - x(e["start"])):.1f}" height="16" fill="#c9d3e0"/>')
    ypos = {WRITING_DRAWING: 62, MOUSE_POINTING: 90, HAND_POINTING: 118}
    for i in ints:
        y = ypos.get(i.type, 62)
        s.append(f'<rect x="{x(i.start):.1f}" y="{y - 8}" width="{max(2, x(i.end) - x(i.start)):.1f}" height="16" rx="3" fill="{COLORS.get(i.type, "#777")}"/>')
    for t in base_t:
        s.append(f'<circle cx="{x(t):.1f}" cy="160" r="6" fill="#555"/>')
    for t in base_t:
        if not any(abs(t - u) < 0.01 for u in upgraded_t):
            s.append(f'<circle cx="{x(t):.1f}" cy="198" r="6" fill="#555"/>')
    for t in upgraded_t:
        s.append(f'<circle cx="{x(t):.1f}" cy="198" r="7" fill="#fff" stroke="#7a3fd1" stroke-width="3"/>')
    for t in added_t:
        s.append(f'<circle cx="{x(t):.1f}" cy="198" r="8" fill="#7a3fd1"/>')
    for t in gt:
        s.append(f'<path d="M{x(t):.1f} 227 l7 9 l-7 9 l-7 -9 z" fill="#d6293e"/>')
    s.append("</svg>")
    return "".join(s)


def _metrics_row(name, pts, gt, tol):
    tp, fp, fn, _ = match(pts, gt, tol)
    p, r, f = prf(tp, fp, fn)
    return name, len(pts), tp, fp, fn, p, r, f


def build(video_dirs: List[str], gt_dir: Optional[str], out_dir: str, tol: float) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sections, totals = [], {"base": [0, 0, 0], "fused": [0, 0, 0]}
    per_video_rows = []
    for vd in video_dirs:
        vid = os.path.basename(vd.rstrip("\\/"))
        ints = load_interactions(os.path.join(vd, "interactions.json"))
        tr = load_transcript(os.path.join(vd, "transcript.json"))
        base_info, fused = _m1_and_fused(vd)
        base_t = [b[0] for b in base_info]
        base_map = {round(b[0], 2): b for b in base_info}
        added, upgraded = [], []
        for p in fused:
            b = base_map.get(round(p.timestamp, 2))
            if b is None:
                added.append(p)
            elif (p.confidence, p.trigger_reason) != (b[1], b[2]):
                upgraded.append((p, b))
        fused_t = [p.timestamp for p in fused]
        dur = _duration(vd, ints, tr)

        gt = []
        gpath = os.path.join(gt_dir, vid + ".json") if gt_dir else None
        if gpath and os.path.exists(gpath):
            gt = load_ground_truth(gpath)

        metrics_html = '<p class="muted">No answer key for this video, so no precision / recall.</p>'
        if gt:
            rows = [_metrics_row("M1 alone (original)", base_t, gt, tol), _metrics_row("M1 + interaction module", fused_t, gt, tol)]
            for key, row in (("base", rows[0]), ("fused", rows[1])):
                for k, v in zip(range(3), row[2:5]):
                    totals[key][k] += v
            per_video_rows.append((vid, rows))
            body = "".join(
                f"<tr><td>{html.escape(n)}</td><td>{c}</td><td>{tp}</td><td>{fp}</td><td>{fn}</td>"
                f"<td>{p:.2f}</td><td>{r:.2f}</td><td><b>{f:.2f}</b></td></tr>" for n, c, tp, fp, fn, p, r, f in rows)
            metrics_html = ("<table><tr><th>Version</th><th>Points</th><th>Correct</th><th>Wrong</th><th>Missed</th>"
                            "<th>Precision</th><th>Recall</th><th>F1</th></tr>" + body + "</table>"
                            f'<p class="muted">A point is correct if it is within &plusmn;{tol:.0f} s of a moment in your answer key '
                            f'({len(gt)} moments).</p>')

        added_rows = ""
        for p in added:
            near = " ".join(t["text"] for t in tr if t["end"] > p.timestamp - 3 and t["start"] < p.timestamp + 3)[:260]
            added_rows += (f"<tr><td>{_mmss(p.timestamp)}</td><td>{html.escape(p.trigger_reason)}</td>"
                           f"<td>{html.escape(near) or '<i>(silence)</i>'}</td><td>{_thumb(getattr(p, 'frame_path', None))}</td></tr>")
        upg_rows = "".join(
            f"<tr><td>{_mmss(p.timestamp)}</td><td>{html.escape(str(b[1]))} &rarr; {html.escape(str(p.confidence))}</td>"
            f"<td>{html.escape(p.trigger_reason)}</td></tr>" for p, b in upgraded)

        sections.append(f"""
<section><h2>Video: {html.escape(vid)} <span class="muted">({_mmss(dur)[:-2]} min)</span></h2>
<p>Module detected <b>{len(ints)}</b> interaction episodes
 (writing {sum(i.type == WRITING_DRAWING for i in ints)}, mouse {sum(i.type == MOUSE_POINTING for i in ints)},
 hand {sum(i.type == HAND_POINTING for i in ints)}).
 Original M1 found <b>{len(base_info)}</b> intervention points; with the module <b>{len(fused)}</b>
 (<span style="color:#7a3fd1"><b>{len(added)} added</b></span>, {len(upgraded)} upgraded).</p>
{_timeline(dur, tr, ints, base_t, [p.timestamp for p in added], [p.timestamp for p, _ in upgraded], gt)}
<p class="legend"><span style="color:#2e9e5b">&#9632;</span> writing &nbsp; <span style="color:#e8890c">&#9632;</span> mouse pointing &nbsp;
<span style="color:#2f6fdb">&#9632;</span> hand pointing &nbsp; <span style="color:#555">&#9679;</span> M1 point &nbsp;
<span style="color:#7a3fd1">&#9679;</span> added by module &nbsp; <span style="color:#7a3fd1">&#9711;</span> upgraded &nbsp;
<span style="color:#d6293e">&#9670;</span> answer key</p>
<h3>Accuracy</h3>{metrics_html}
<h3>Points added by the module</h3>
{('<table><tr><th>Time</th><th>Why</th><th>Teacher says</th><th>Frame</th></tr>' + added_rows + '</table>') if added_rows else '<p class="muted">None.</p>'}
<h3>Points upgraded by the module</h3>
{('<table><tr><th>Time</th><th>Confidence</th><th>Reason</th></tr>' + upg_rows + '</table>') if upg_rows else '<p class="muted">None.</p>'}
</section>""")

    overall = ""
    if per_video_rows:
        def agg(key):
            tp, fp, fn = totals[key]
            p, r, f = prf(tp, fp, fn)
            return tp, fp, fn, p, r, f
        a, b = agg("base"), agg("fused")
        overall = f"""
<section><h2>Overall ({len(per_video_rows)} video{'s' if len(per_video_rows) != 1 else ''} with answer keys)</h2>
<table><tr><th>Version</th><th>Correct</th><th>Wrong</th><th>Missed</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr><td>M1 alone (original)</td><td>{a[0]}</td><td>{a[1]}</td><td>{a[2]}</td><td>{a[3]:.2f}</td><td>{a[4]:.2f}</td><td><b>{a[5]:.2f}</b></td></tr>
<tr><td>M1 + interaction module</td><td>{b[0]}</td><td>{b[1]}</td><td>{b[2]}</td><td>{b[3]:.2f}</td><td>{b[4]:.2f}</td><td><b>{b[5]:.2f}</b></td></tr>
<tr><td><i>change</i></td><td>{b[0] - a[0]:+d}</td><td>{b[1] - a[1]:+d}</td><td>{b[2] - a[2]:+d}</td><td>{b[3] - a[3]:+.2f}</td><td>{b[4] - a[4]:+.2f}</td><td><b>{b[5] - a[5]:+.2f}</b></td></tr></table>
<p class="muted">Numbers are summed over all videos. Answer keys were made by a person who watched the videos
(blind review: candidates from both versions were mixed, source hidden). Small sample: indicative, not final.</p></section>"""

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>SYNAPSE - interaction module demo</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1040px;margin:24px auto;padding:0 16px;color:#222;background:#fff}}
h1{{margin-bottom:4px}} h2{{margin-top:36px;border-bottom:2px solid #eee;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;margin:8px 0}} th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top;font-size:14px}}
th{{background:#f6f6f6}} .muted{{color:#777;font-size:13px}} .legend{{font-size:13px}} img{{border-radius:4px;border:1px solid #ddd}}
@media (prefers-color-scheme: dark){{body{{background:#171717;color:#e6e6e6}} th{{background:#262626}} td,th{{border-color:#3a3a3a}} h2{{border-color:#333}}}}
</style></head><body>
<h1>SYNAPSE: visual interaction module</h1>
<p class="muted">Compares the original intervention-point detector (M1, unchanged) with M1 plus the visual-interaction module
(writing, mouse pointing, hand pointing). Purple = what the module adds.</p>
{overall}{''.join(sections)}
</body></html>"""
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dirs", nargs="+")
    ap.add_argument("--gt-dir", default="ground_truth")
    ap.add_argument("--out", default="demo_report")
    ap.add_argument("--tol", type=float, default=3.0)
    a = ap.parse_args()
    path = build(a.video_dirs, a.gt_dir, a.out, a.tol)
    print("Report written:", os.path.abspath(path))


if __name__ == "__main__":
    main()
