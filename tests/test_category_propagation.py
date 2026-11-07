"""Regression tests for visual-category and confidence propagation."""

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base
from api.explanation_service import _load_target
from api.models import Intervention, Video
from api.processing_jobs import _build_interventions
from category_normalization import normalize_category_labels
from vlm_engine.snapshot_engine import SnapshotEngine


class CategoryNormalizationTests(unittest.TestCase):
    def test_multiple_categories_are_retained_and_ambiguous_labels_are_reported(self):
        normalized = normalize_category_labels(["graph", "equation", "diagram"])

        self.assertEqual(normalized.canonical_categories, ("CT-01", "CT-02"))
        self.assertEqual(normalized.ambiguous_labels, ("diagram",))
        self.assertEqual(normalized.raw_labels, ("graph", "equation", "diagram"))
        self.assertEqual(normalized.status, "ambiguous")

    def test_unknown_labels_are_not_promoted_to_an_approved_category(self):
        normalized = normalize_category_labels(["slide", "new-label"])

        self.assertEqual(normalized.canonical_categories, ())
        self.assertEqual(normalized.ambiguous_labels, ("slide",))
        self.assertEqual(normalized.unmapped_labels, ("new-label",))


class ConfidencePropagationTests(unittest.TestCase):
    def test_new_interventions_keep_classifier_and_complexity_scores_separate(self):
        records = _build_interventions(
            "video-1",
            {
                "intervention_points": [{
                    "timestamp": 12.0,
                    "frame_path": "data/video-1/keyframes/frame.jpg",
                    "content_type": "graph",
                    "complexity_score": 0.21,
                    "clip_confidence": 0.87,
                    "classification_confidence": 0.87,
                    "confidence_semantics": "classifier",
                    "raw_category_labels": ["graph", "equation"],
                }],
            },
        )

        row = records[0]
        self.assertEqual(row.complexity_score, 0.21)
        self.assertEqual(row.clip_confidence, 0.87)
        self.assertEqual(row.classification_confidence, 0.87)
        self.assertEqual(row.confidence_semantics, "classifier")
        self.assertEqual(row.canonical_categories, ["CT-01", "CT-02"])

    def test_missing_confidence_is_marked_legacy_without_fabrication(self):
        row = _build_interventions(
            "video-1",
            {"intervention_points": [{
                "timestamp": 12.0,
                "frame_path": "frame.jpg",
                "content_type": "graph",
                "complexity_score": 0.21,
                "clip_confidence": 0.21,
            }]},
        )[0]

        self.assertIsNone(row.classification_confidence)
        self.assertEqual(row.confidence_semantics, "legacy_complexity")
        self.assertEqual(row.complexity_score, 0.21)

    def test_low_new_confidence_does_not_erase_the_category_hint(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        with sessions() as db:
            db.add(Video(id="video-1", title="Test", source_url="https://example.test"))
            db.add(Intervention(
                id="intervention-1",
                video_id="video-1",
                timestamp=1.0,
                frame_path="frame.jpg",
                content_type="graph",
                clip_confidence=0.12,
                classification_confidence=0.12,
                confidence_semantics="classifier",
                trigger_reason="Deictic",
            ))
            db.commit()

            target = _load_target(db, "video-1", "intervention-1")
            self.assertEqual(target.content_type, "graph")
            self.assertIn("low category confidence", target.trigger_reason)

    def test_legacy_complexity_does_not_erase_existing_category(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        with sessions() as db:
            db.add(Video(id="video-1", title="Test", source_url="https://example.test"))
            db.add(Intervention(
                id="intervention-1",
                video_id="video-1",
                timestamp=1.0,
                frame_path="frame.jpg",
                content_type="diagram",
                clip_confidence=0.12,
                confidence_semantics="legacy_complexity",
            ))
            db.commit()

            target = _load_target(db, "video-1", "intervention-1")
            self.assertEqual(target.content_type, "diagram")


class PromptPropagationTests(unittest.TestCase):
    def test_notes_alias_uses_existing_handwritten_directive(self):
        engine = object.__new__(SnapshotEngine)
        prompt = engine._build_prompt("handwritten_notes")
        self.assertIn("handwritten_notes", prompt)

    def test_alias_is_used_when_selecting_the_modal_prompt(self):
        class FakeVLM:
            def __init__(self):
                self.prompt = None

            def analyze(self, image_path_or_pil, prompt, generation_params):
                self.prompt = prompt
                return type("Response", (), {"text": '{"content_type":"handwritten_notes"}'})()

        engine = object.__new__(SnapshotEngine)
        engine.provider = "modal"
        engine.use_gguf = False
        engine.vlm_interface = FakeVLM()
        result = engine.analyze_frame("unused.jpg", "notes")

        self.assertEqual(result["content_type"], "handwritten_notes")
        self.assertIn("CRITICAL: OCR handwritten content", engine.vlm_interface.prompt)

    def test_low_classifier_confidence_is_explicitly_sent_to_qwen_prompt(self):
        class FakeVLM:
            def __init__(self):
                self.prompt = None

            def analyze(self, image_path_or_pil, prompt, generation_params):
                self.prompt = prompt
                return type("Response", (), {"text": '{"content_type":"graph"}'})()

        engine = object.__new__(SnapshotEngine)
        engine.provider = "modal"
        engine.use_gguf = False
        engine.vlm_interface = FakeVLM()
        engine.analyze_frame(
            "unused.jpg",
            "graph",
            classification_confidence=0.12,
            confidence_semantics="classifier",
        )

        self.assertIn("CATEGORY HINT UNCERTAINTY", engine.vlm_interface.prompt)
        self.assertIn("0.120", engine.vlm_interface.prompt)


if __name__ == "__main__":
    unittest.main()

