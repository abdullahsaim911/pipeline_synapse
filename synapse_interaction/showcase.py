"""Run the whole demo on ANY YouTube video with one command.

    python -m synapse_interaction.showcase "https://www.youtube.com/watch?v=VIDEO_ID"

1) downloads the video, 2) runs your existing pipeline (frame extraction, transcription, M1) = BEFORE,
3) runs the interaction module, 4) prints BEFORE / AFTER and what the module found.
Finished steps are skipped, so run it once before presenting and again (instantly) while presenting.
"""
import os
import sys

from .before_after import _report
from .prepare import prepare_video, _vid
from .schema import load_interactions
from .demo_report import _mmss


def main():
    urls = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not urls:
        print('usage: python -m synapse_interaction.showcase "https://www.youtube.com/watch?v=VIDEO_ID"')
        return
    url = urls[0]
    vdir = prepare_video(url)
    vid = _vid(url)

    ints = load_interactions(os.path.join(vdir, "interactions.json"))
    lines = ["", "WHAT THE INTERACTION MODULE SAW", "=" * 60]
    if not ints:
        lines.append("  nothing detected (video may have no writing or pointer)")
    for i in ints:
        lines.append(f"  {_mmss(i.start):>7s} - {_mmss(i.end):>7s}  {i.type}")
    text, _ = _report(vdir, "ground_truth", 3.0, show_text=True)
    out = "\n".join(lines) + "\n\n" + text
    print(out)
    path = f"showcase_{vid}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"\nsaved to {path}")


if __name__ == "__main__":
    main()
