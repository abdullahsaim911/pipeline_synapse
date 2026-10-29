"""
Measure intent accuracy on the held-out sentences (module 8):  python -m voice.evaluate [--sweep]

wrong_exec is the number that matters most: a command RUN that the learner did not mean.
"""

import argparse
import itertools
import sys
from collections import Counter

from voice.catalog import ActionCatalog
from voice.interpret import Interpreter
from voice.schemas import InterpretRequest, VoiceContext


def run(interp: Interpreter, lang: str = "en", verbose: bool = False) -> Counter:
    tally, misses = Counter(), []
    for item in interp.catalog.eval_items(lang):
        ctx = VoiceContext(screen=item.screen, playing=item.playing, pending_confirmation=item.pending,
                           video_id="aircAruvnKk" if item.screen == "explanation" else None)
        res = interp.interpret(InterpretRequest(text=item.text, context=ctx))
        ran = res.decision in ("execute", "ask_question")
        right_action = res.action == item.expect and all(res.args.get(k) == v for k, v in item.args.items())
        if item.expect == "unsupported":
            outcome = "wrong_exec" if ran else "ok_refused"
        elif ran and right_action:
            outcome = "ok_exec"
        elif ran:
            outcome = "wrong_exec"
        elif res.decision in ("confirm", "clarify") and (right_action or item.expect in [c.action for c in res.candidates]):
            outcome = "ok_asked"
        else:
            outcome = "missed"
        tally[outcome] += 1
        if verbose and outcome in ("wrong_exec", "missed"):
            misses.append(f"  {outcome:10} '{item.text}' [{item.screen}] expected {item.expect} {item.args or ''} -> {res.decision} {res.action} {res.args} {res.confidence:.2f}")
    if verbose:
        print("\n".join(misses))
    return tally


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    catalog = ActionCatalog.load()
    interp = Interpreter(catalog)
    if args.sweep:
        print("execute_min confirm_min margin | ok_exec ok_asked ok_refused missed wrong_exec")
        for e, c, m in itertools.product((0.5, 0.55, 0.6, 0.65, 0.7), (0.35, 0.4, 0.45), (0.03, 0.05, 0.08)):
            interp.execute_min, interp.confirm_min, interp.margin = e, c, m
            t = run(interp)
            print(f"{e:11} {c:11} {m:6} | {t['ok_exec']:7} {t['ok_asked']:8} {t['ok_refused']:10} {t['missed']:6} {t['wrong_exec']:10}")
        return 0
    t = run(interp, verbose=args.verbose)
    total = sum(t.values())
    print(f"{total} held-out sentences: " + ", ".join(f"{k}={v}" for k, v in sorted(t.items())))
    print(f"ran correctly {t['ok_exec'] / total:.1%}; asked or refused safely {(t['ok_asked'] + t['ok_refused']) / total:.1%}; "
          f"missed {t['missed'] / total:.1%}; WRONG command run {t['wrong_exec'] / total:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
