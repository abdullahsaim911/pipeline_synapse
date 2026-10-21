"""Moment context: ownership validation and filtering of unreliable visual material."""

import json
import os
import tempfile
import unittest

from tests.qa.helpers import LECTURE_LINES, make_segments
from qa.moment import (
    SNAPSHOT_FAILED,
    SNAPSHOT_INVALID,
    SNAPSHOT_MISSING,
    SNAPSHOT_VALID,
    InterventionNotFoundError,
    assess_snapshot,
    clean_visual_snapshot,
    is_transcript_echo,
    load_moment,
    strip_explanation_scaffolding,
)

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from api.database import Base
    from api import models
    HAVE_SQLALCHEMY = True
except ImportError:  # pragma: no cover
    HAVE_SQLALCHEMY = False

VIDEO_A = "AAAAAAAAAAA"
VIDEO_B = "BBBBBBBBBBB"
SEGMENTS = make_segments()

GOOD_SNAPSHOT = {
    "content_type": "graph",
    "visual_analysis": {
        "structural_description": "A curve rising steeply to the right of the y axis.",
        "layout": "One plot filling the frame.",
        "text_readout": "y equals x e to the x",
    },
    "conceptual_hints": "",
    "missing_elements": None,
}
ERROR_SNAPSHOT = {
    "content_type": "biology", "visual_analysis": {},
    "structural_description": "Error: [WinError -529697949] Windows Error 0xe06d7363",
    "layout": "[WinError -529697949] Windows Error 0xe06d7363",
    "text_readout": "[WinError -529697949] Windows Error 0xe06d7363",
    "conceptual_hints": "", "missing_elements": None,
    "_error": "[WinError -529697949] Windows Error 0xe06d7363",
}


class VisualFilterTests(unittest.TestCase):
    def test_good_snapshot_with_nested_fields(self):
        pairs = dict(clean_visual_snapshot(GOOD_SNAPSHOT))
        self.assertIn("rising steeply", pairs["Structure"])
        self.assertEqual(pairs["Text read from the screen"], "y equals x e to the x")
        self.assertNotIn("Concept hints", pairs)

    def test_failed_snapshot_is_discarded(self):
        self.assertIsNone(clean_visual_snapshot(ERROR_SNAPSHOT))

    def test_error_text_without_error_key_is_discarded(self):
        snap = {"structural_description": "Error: model crashed", "layout": "[WinError 5] x"}
        self.assertIsNone(clean_visual_snapshot(snap))

    def test_missing_elements_marker_discards_snapshot(self):
        snap = dict(GOOD_SNAPSHOT, missing_elements="the image is blurry")
        self.assertIsNone(clean_visual_snapshot(snap))

    def test_unusable_inputs(self):
        for bad in (None, [], "text", {}, {"structural_description": "null"}):
            self.assertIsNone(clean_visual_snapshot(bad), bad)


class AssessSnapshotTests(unittest.TestCase):
    def test_states(self):
        self.assertEqual(assess_snapshot(None)[0], SNAPSHOT_MISSING)
        self.assertEqual(assess_snapshot(GOOD_SNAPSHOT)[0], SNAPSHOT_VALID)
        self.assertEqual(assess_snapshot(ERROR_SNAPSHOT)[0], SNAPSHOT_FAILED)
        self.assertEqual(assess_snapshot(dict(GOOD_SNAPSHOT, missing_elements="blurry"))[0], SNAPSHOT_FAILED)
        for bad in ([], "text", 5, {}, {"structural_description": "null"}):
            self.assertEqual(assess_snapshot(bad)[0], SNAPSHOT_INVALID, bad)

    def test_error_text_filter_is_narrow(self):
        code_slide = {"text_readout": "catch (NullPointerException e) { log(e); }",
                      "structural_description": "Error analysis of the circuit with a 5% tolerance."}
        state, pairs = assess_snapshot(code_slide)
        self.assertEqual(state, SNAPSHOT_VALID)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(assess_snapshot({"text_readout": "Error: [WinError 5] Access is denied"})[0],
                         SNAPSHOT_INVALID)


class ScaffoldingTests(unittest.TestCase):
    def test_removes_template_headings_and_markdown_only(self):
        raw = ("YOUR EXPLANATION - FOCUS ON THESE ELEMENTS:\n"
               "1. EXHAUSTIVE VISUAL DESCRIPTION:\n"
               "## What happens\n"
               "- The **rate** r = 2 * x costs $5 and ATP is produced.\n"
               "RNA POLYMERASE\n"
               "3) Second law: F = m a\n"
               "6. RICH ACCESSIBILITY TECHNIQUES (Spatial):\n")
        text = strip_explanation_scaffolding(raw)
        self.assertNotIn("YOUR EXPLANATION", text)
        self.assertNotIn("EXHAUSTIVE", text)
        self.assertNotIn("RICH ACCESSIBILITY", text)
        self.assertNotIn("**", text)
        self.assertNotIn("##", text)
        self.assertIn("The rate r = 2 * x costs $5 and ATP is produced.", text)  # math and symbols intact
        self.assertIn("RNA POLYMERASE", text)                                    # unrelated upper case kept
        self.assertIn("3) Second law: F = m a", text)                            # list items kept

    def test_inline_prefix(self):
        self.assertEqual(strip_explanation_scaffolding("YOUR EXPLANATION: The slope is rise over run."),
                         "The slope is rise over run.")

    def test_plain_text_is_unchanged(self):
        text = "The slope is rise over run. A steeper line means a larger slope."
        self.assertEqual(strip_explanation_scaffolding(text), text)

    def test_non_string(self):
        self.assertEqual(strip_explanation_scaffolding(None), "")


class EchoTests(unittest.TestCase):
    def test_caption_echo_is_detected(self):
        self.assertTrue(is_transcript_echo("pick u as the function that comes first in", SEGMENTS))
        self.assertTrue(is_transcript_echo("The remaining integral IS easy", SEGMENTS, caption=""))

    def test_real_explanation_is_not_an_echo(self):
        self.assertFalse(is_transcript_echo(
            "The instructor is showing a graph that explains how to choose u using LIATE.", SEGMENTS))


@unittest.skipUnless(HAVE_SQLALCHEMY, "SQLAlchemy not installed")
class LoadMomentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = self.tmp.name
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                               poolclass=StaticPool)
        Base.metadata.create_all(bind=engine)
        self.addCleanup(engine.dispose)
        self.db = sessionmaker(bind=engine)()
        self.addCleanup(self.db.close)
        for vid in (VIDEO_A, VIDEO_B):
            self.db.add(models.Video(id=vid, title=vid, source_url="http://x"))
        self.db.commit()

    def add_intervention(self, iid, video_id=VIDEO_A, ts=120.0, caption="u equals x so du equals dx"):
        self.db.add(models.Intervention(id=iid, video_id=video_id, timestamp=ts, frame_path="f.jpg",
                                        content_type="graph", transcript_context=caption))
        self.db.commit()

    def add_explanation(self, iid, text, mode="explanatory", snapshot=None, video_id=VIDEO_A,
                        snapshot_path=None):
        path = None
        if snapshot is not None or snapshot_path:
            if snapshot_path:
                path = snapshot_path
            else:
                folder = os.path.join(self.data, video_id, "interventions", iid[:8])
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, "vlm_snapshot.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(snapshot, handle)
                path = os.path.relpath(path, self.data)  # stored relative, like the pipeline does
        self.db.add(models.Explanation(intervention_id=iid, text_explanation=text,
                                       output_mode=mode, vlm_snapshot_path=path))
        self.db.commit()

    def load(self, iid, video_id=VIDEO_A):
        return load_moment(self.db, video_id, iid, SEGMENTS, self.data)

    def test_valid_intervention_with_explanation_and_visual(self):
        self.add_intervention("int-1")
        self.add_explanation("int-1", "The graph shows exponential growth multiplied by x.", snapshot=GOOD_SNAPSHOT)
        moment = self.load("int-1")
        self.assertEqual(moment.timestamp, 120.0)
        self.assertIn("exponential growth", moment.explanation)
        self.assertIsNotNone(moment.visual)
        self.assertTrue(moment.has_generated_content)

    def test_unknown_intervention(self):
        with self.assertRaises(InterventionNotFoundError):
            self.load("nope")

    def test_intervention_from_another_video_is_rejected(self):
        self.add_intervention("int-b", video_id=VIDEO_B)
        with self.assertRaises(InterventionNotFoundError):
            self.load("int-b", video_id=VIDEO_A)
        self.assertEqual(self.load("int-b", video_id=VIDEO_B).intervention_id, "int-b")

    def test_invalid_video_id_is_rejected(self):
        self.add_intervention("int-1")
        with self.assertRaises(InterventionNotFoundError):
            self.load("int-1", video_id="../x")

    def test_failed_snapshot_withholds_its_explanation_but_keeps_the_caption(self):
        self.add_intervention("int-1")
        self.add_explanation("int-1", "A fluent explanation of the idea behind choosing parts, "
                                      "describing arrows and graphs on the screen.",
                             snapshot=ERROR_SNAPSHOT)
        moment = self.load("int-1")
        self.assertEqual(moment.visual_status, SNAPSHOT_FAILED)
        self.assertIsNone(moment.visual)
        self.assertIsNone(moment.explanation)
        self.assertFalse(moment.has_generated_content)
        self.assertEqual(moment.caption, "u equals x so du equals dx")  # transcript evidence survives

    def test_missing_snapshot_record_withholds_the_explanation(self):
        self.add_intervention("int-1")
        self.add_explanation("int-1", "An explanation we cannot check against any visual analysis.")
        moment = self.load("int-1")
        self.assertEqual((moment.visual_status, moment.explanation, moment.visual),
                         (SNAPSHOT_MISSING, None, None))

    def test_snapshot_file_that_does_not_exist_counts_as_missing(self):
        self.add_intervention("int-1")
        self.add_explanation("int-1", "An explanation whose snapshot file was deleted.",
                             snapshot_path=os.path.join(VIDEO_A, "interventions", "gone", "vlm_snapshot.json"))
        moment = self.load("int-1")
        self.assertEqual((moment.visual_status, moment.explanation), (SNAPSHOT_MISSING, None))

    def test_valid_snapshot_keeps_explanation_and_visual(self):
        self.add_intervention("int-1")
        self.add_explanation("int-1", "The graph shows x times e to the x growing quickly.",
                             snapshot=GOOD_SNAPSHOT)
        moment = self.load("int-1")
        self.assertEqual(moment.visual_status, SNAPSHOT_VALID)
        self.assertIn("growing quickly", moment.explanation)
        self.assertIsNotNone(moment.visual)

    def test_scaffolding_is_removed_from_an_eligible_explanation(self):
        raw = ("YOUR EXPLANATION:\n\n1. COMPLETE VISUAL DESCRIPTION:\n"
               "   - **Curve**: it rises steeply to the right of the y axis.\n\n"
               "2. WHAT THE TEACHER IS EXPLAINING:\nChoosing u with LIATE.\n\n"
               "4. HOW THE VISUAL LINKS TO THE CONCEPT (Multiple Connections):\n"
               "The steep growth matches the x times e to the x term.")
        self.add_intervention("int-1")
        self.add_explanation("int-1", raw, snapshot=GOOD_SNAPSHOT)
        text = self.load("int-1").explanation
        for artifact in ("YOUR EXPLANATION", "COMPLETE VISUAL DESCRIPTION", "WHAT THE TEACHER IS EXPLAINING",
                         "HOW THE VISUAL LINKS", "**", "   - "):
            self.assertNotIn(artifact, text)
        for content in ("Curve: it rises steeply", "Choosing u with LIATE.", "steep growth matches"):
            self.assertIn(content, text)

    def test_explanation_that_is_only_scaffolding_is_dropped(self):
        self.add_intervention("int-1")
        self.add_explanation("int-1", "YOUR EXPLANATION:\n\n1. COMPLETE VISUAL DESCRIPTION:\n",
                             snapshot=GOOD_SNAPSHOT)
        moment = self.load("int-1")
        self.assertIsNone(moment.explanation)
        self.assertIsNotNone(moment.visual)   # the valid visual fields are still usable

    def test_falls_through_to_another_mode_when_the_preferred_one_is_unusable(self):
        self.add_intervention("int-1")
        self.add_explanation("int-1", "YOUR EXPLANATION:", mode="explanatory", snapshot=GOOD_SNAPSHOT)
        self.add_explanation("int-1", "Brief take on choosing parts for products.", mode="brief")
        self.assertIn("Brief take", self.load("int-1").explanation)

    def test_caption_only_explanation_is_not_treated_as_understanding(self):
        self.add_intervention("int-1", caption="plug into the formula to get x e to the x")
        self.add_explanation("int-1", "plug into the formula to get x e to the x", snapshot=ERROR_SNAPSHOT)
        moment = self.load("int-1")
        self.assertIsNone(moment.explanation)
        self.assertIsNone(moment.visual)
        self.assertFalse(moment.has_generated_content)
        self.assertEqual(moment.caption, "plug into the formula to get x e to the x")

    def test_intervention_without_explanation(self):
        self.add_intervention("int-1")
        moment = self.load("int-1")
        self.assertEqual((moment.explanation, moment.visual), (None, None))

    def test_prefers_explanatory_mode(self):
        self.add_intervention("int-1")
        self.add_explanation("int-1", "Brief take on integrating products carefully.", mode="brief",
                             snapshot=GOOD_SNAPSHOT)
        self.add_explanation("int-1", "Explanatory take on integrating products carefully.", mode="explanatory")
        self.assertIn("Explanatory take", self.load("int-1").explanation)

    def test_snapshot_path_outside_the_video_folder_is_ignored(self):
        self.add_intervention("int-1")
        other = os.path.join(self.data, VIDEO_B, "interventions", "zzzzzzzz")
        os.makedirs(other)
        with open(os.path.join(other, "vlm_snapshot.json"), "w") as handle:
            json.dump({"structural_description": "SECRET content of another lecture"}, handle)
        self.add_explanation("int-1", "Some explanation of the worked example.",
                             snapshot_path=os.path.join(VIDEO_B, "interventions", "zzzzzzzz", "vlm_snapshot.json"))
        moment = self.load("int-1")
        self.assertIsNone(moment.visual)
        self.assertIsNone(moment.explanation)

    def test_traversal_in_snapshot_path_is_ignored(self):
        self.add_intervention("int-1")
        os.makedirs(os.path.join(self.data, VIDEO_A))
        with open(os.path.join(self.data, "outside.json"), "w") as handle:
            json.dump({"structural_description": "outside"}, handle)
        self.add_explanation("int-1", "Some explanation of the worked example.",
                             snapshot_path=os.path.join(VIDEO_A, "..", "outside.json"))
        self.assertIsNone(self.load("int-1").visual)

    def test_corrupt_snapshot_file_is_ignored(self):
        self.add_intervention("int-1")
        self.add_explanation("int-1", "Some explanation of the worked example.", snapshot={})
        path = os.path.join(self.data, VIDEO_A, "interventions", "int-1", "vlm_snapshot.json")
        with open(path, "w") as handle:
            handle.write("{broken")
        moment = self.load("int-1")
        self.assertEqual((moment.visual_status, moment.visual, moment.explanation),
                         (SNAPSHOT_INVALID, None, None))


if __name__ == "__main__":
    unittest.main()
