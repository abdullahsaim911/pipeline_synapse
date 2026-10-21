"""
Shared fixtures for quiz tests.

LECTURE is a small, clearly SYNTHETIC lecture written for these tests (it is not
real transcript data and makes no claim about any real lecture). Each of the first
eight passages carries one topic with a well-formed MCQ and true/false question
that its own text supports; passage 9 is a short review of topic 1.
"""

import copy
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qa.tokens import estimate_tokens  # noqa: E402
from qa.transcript import TranscriptSegment  # noqa: E402
from quiz.passages import Passage, PassageSet, build_passages  # noqa: E402

QUIZ_ID = "3f2b8c1e-5d4a-4e7b-9a10-2c6d8e9f0a1b"

PASSAGE_TEXTS = [
    # S1
    "today we learn integration by parts the formula says the integral of u dv equals u v minus the "
    "integral of v du we choose u using the rule called liate where logarithmic functions come first and "
    "exponential functions come last for example with x times e to the x we pick u equal to x",
    # S2
    "when a ball is thrown straight up gravity pulls it down with an acceleration of nine point eight meters "
    "per second squared at the top of its path the velocity is zero but the acceleration is still nine point "
    "eight the ball takes the same time to rise as it takes to fall back to the same height",
    # S3
    "plants make glucose in photosynthesis using light energy water and carbon dioxide the light reactions "
    "happen in the thylakoid membranes while the calvin cycle takes place in the stroma oxygen is released "
    "as a byproduct when water is split during the light reactions",
    # S4
    "during mitosis a cell divides into two identical daughter cells the phases in order are prophase "
    "metaphase anaphase and telophase chromosomes line up in the middle of the cell during metaphase and "
    "the sister chromatids are pulled apart during anaphase",
    # S5
    "ohms law says that voltage equals current times resistance so if a circuit has a current of two amperes "
    "and a resistance of fifty ohms the voltage is one hundred volts doubling the resistance while keeping "
    "the voltage the same cuts the current in half",
    # S6
    "in a neutralization reaction an acid reacts with a base to form a salt and water a solution with a ph of "
    "seven is neutral while a ph below seven is acidic and a ph above seven is basic",
    # S7
    "binary search works on a sorted list by comparing the target with the middle element and discarding half "
    "of the list each time so it runs in logarithmic time while linear search may check every element",
    # S8
    "newtons second law says that force equals mass times acceleration so a force of twenty newtons on a mass "
    "of four kilograms gives an acceleration of five meters per second squared and if the mass doubles the "
    "acceleration halves for the same force",
    # S9 (review of S1)
    "to review integration by parts remember the liate rule for choosing u with logarithmic functions first "
    "and exponential functions last a good habit is to write down u and dv before computing anything",
]

TOPICS = [
    {"mcq": ("Which rule helps you choose u in integration by parts?",
             ["The LIATE rule", "The chain rule", "The quotient rule", "The ratio test"], 0),
     "tf": ("Integration by parts says the integral of u dv equals u v minus the integral of v du", True)},
    {"mcq": ("What is the acceleration of the ball at the top of its path?",
             ["Still nine point eight meters per second squared", "Zero because the ball stops",
              "Twice the value at launch", "Pointing upward and very large"], 0),
     "tf": ("A ball thrown straight up has zero velocity at the top of its path", True)},
    {"mcq": ("Which gas is released as a byproduct of the light reactions?",
             ["Oxygen", "Nitrogen", "Methane", "Hydrogen sulfide"], 0),
     "tf": ("The Calvin cycle takes place in the stroma", True)},
    {"mcq": ("How many identical daughter cells does mitosis produce from one cell?",
             ["Two", "Four", "Eight", "Six"], 0),
     "tf": ("Chromosomes line up in the middle of the cell during metaphase", True)},
    {"mcq": ("A circuit has a current of two amperes and a resistance of fifty ohms. What is the voltage?",
             ["One hundred volts", "Twenty five volts", "Fifty two volts", "Four volts"], 0),
     "tf": ("Doubling the resistance while keeping the voltage the same cuts the current in half", True)},
    {"mcq": ("What is the pH of a neutral solution?",
             ["Seven", "Zero", "Fourteen", "Ten"], 0),
     "tf": ("A solution with a pH below seven is acidic", True)},
    {"mcq": ("What must be true about the list before binary search can be used?",
             ["It must be sorted", "It must contain only integers", "It must have an even length",
              "It must be stored in a hash table"], 0),
     "tf": ("Binary search discards half of the list each time", True)},
    {"mcq": ("What acceleration results from a force of twenty newtons on a mass of four kilograms?",
             ["Five meters per second squared", "Eighty meters per second squared",
              "Sixteen meters per second squared", "Zero point two meters per second squared"], 0),
     "tf": ("If the mass doubles the acceleration halves for the same force", True)},
]
REVIEW_TF = ("A good habit is to write down u and dv before computing anything", True)  # supported by S9 only


def lecture_from(texts, step=100.0):
    passages = tuple(
        Passage(id=f"S{i}", index=i, start=(i - 1) * step, end=i * step, text=t, tokens=estimate_tokens(t))
        for i, t in enumerate(texts, start=1)
    )
    return PassageSet(passages, len(passages) * step, False, 0)


LECTURE = lecture_from(PASSAGE_TEXTS)


def make_segments(count=120, step=6.0):
    """Filler captions for passage-building tests (they are NOT meant to support questions)."""
    return [
        TranscriptSegment(
            index=i, start=i * step, end=(i + 1) * step,
            text=f"synthetic caption number {i} talks about topic {i % 7} using ordinary filler words here",
        )
        for i in range(count)
    ]


def make_passages(count=120):
    return build_passages(make_segments(count))


def mcq(n=0, **overrides):
    """A well-formed MCQ candidate for topic n; cites passage S(n+1)."""
    prompt, choices, key = TOPICS[n]["mcq"]
    candidate = {
        "type": "mcq", "prompt": prompt, "choices": list(choices), "correct_index": key,
        "explanation": "The speaker states this directly in the cited passage.",
        "hints": [], "concept": {"id": f"concept-{n}", "label": f"Concept {n}"},
        "evidence_ids": [f"S{n + 1}"], "relies_on_exact_figure": False, "depends_on_visual": False,
    }
    candidate.update(overrides)
    return candidate


def concept(tag, n=0):
    """A distinct concept for questions that deliberately share a topic (the model is told to use a new concept per question)."""
    return {"id": f"concept-{n}-{tag}", "label": f"Concept {n} {tag}"}


def tf(n=0, **overrides):
    prompt, answer = TOPICS[n]["tf"]
    candidate = {
        "type": "tf", "prompt": prompt, "answer": answer,
        "explanation": "The speaker states this directly in the cited passage.",
        "hints": [], "concept": {"id": f"concept-{n}-tf", "label": f"Concept {n} tf"},
        "evidence_ids": [f"S{n + 1}"], "relies_on_exact_figure": False, "depends_on_visual": False,
    }
    candidate.update(overrides)
    return candidate


def without(candidate, *keys):
    out = copy.deepcopy(candidate)
    for key in keys:
        out.pop(key, None)
    return out
