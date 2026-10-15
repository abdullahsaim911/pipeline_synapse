"""python -m synapse_interaction.preview data\\VIDEO_ID
Shows, for every detected segment, what the fusion rules would do with your transcript."""
import sys
from .schema import load_interactions
from .fusion import (load_transcript, _words_per_sec, _overlapping, DEICTIC_RE, _decide,
                     pointer_coverage)


def main():
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "data/VIDEO_ID"
    ints = load_interactions(f"{video_dir}/interactions.json")
    tr = load_transcript(f"{video_dir}/transcript.json")
    cov = pointer_coverage(ints)
    unreliable = cov > 0.40
    print(f"transcript entries: {len(tr)} | pointer coverage: {cov:.0%}"
          + (" (pointer always moving -> pointing treated as weak)" if unreliable else ""))
    print(f"{'type':16s} {'start':>7s} {'end':>7s} {'words/s':>8s} {'quiet':>6s} {'deictic':>8s}  -> level / creates new point?")
    n_new = 0
    for s in ints:
        wps = _words_per_sec(tr, s.start, s.end)
        quiet = wps < 1.0
        deictic = any(DEICTIC_RE.search(t["text"]) for t in _overlapping(tr, s.start - 2, s.end + 2))
        level, reason, may_create = _decide(s, quiet, deictic, unreliable)
        n_new += may_create
        print(f"{s.type:16s} {s.start:7.1f} {s.end:7.1f} {wps:8.2f} {str(quiet):>6s} {str(deictic):>8s}  -> {level:6s} {'CREATE' if may_create else '-'}")
        if may_create and "--text" in sys.argv:
            hits = [t["text"] for t in _overlapping(tr, s.start - 2, s.end + 2) if DEICTIC_RE.search(t["text"])]
            snippet = " | ".join(hits)[:230] if hits else " | ".join(t["text"] for t in _overlapping(tr, s.start, s.end))[:230]
            print(f"      said: {snippet}")
    print(f"\n{len(ints)} segments, {n_new} could create a new intervention point (before the 5 s redundancy rule)")


if __name__ == "__main__":
    main()
