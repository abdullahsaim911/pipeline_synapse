"""
Synchronizer Module (M1)

Detect "suffer points" where blind students lose visual access.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class TranscriptEntry:
    """Transcript segment with timing information."""
    start: float
    end: float
    text: str


@dataclass
class InterventionPoint:
    """Represents a detected suffer point requiring intervention."""
    timestamp: float
    frame_path: str
    content_type: str
    complexity_score: float
    clip_confidence: float
    transcript_context: str
    trigger_reason: str
    confidence: str = "medium"


class Synchronizer:
    """
    Detect suffer points where blind students lose visual access.

    Uses a scoring system with multiple factors:
    - Deictic phrases (+50)
    - Silent drawing (+80)
    - High complexity (+30)
    - Negative deictic (-50)
    """

    # Deictic phrases to detect
    DEICTIC_PHRASES = [
        "look at", "here we see", "this graph", "notice", "observe",
        "check this", "see this", "here's", "look here"
    ]

    # Negative deictic phrases (indicate moving on)
    NEGATIVE_DEICTIC_PHRASES = [
        "next slide", "moving on", "let's continue", "moving forward",
        "now let's", "next we'll", "moving to", "next topic"
    ]

    # Scoring thresholds
    DEICTIC_SCORE = 50
    SILENT_DRAWING_SCORE = 80
    HIGH_COMPLEXITY_SCORE = 50
    HIGH_COMPLEXITY_THRESHOLD = 0.30
    NEGATIVE_DEICTIC_PENALTY = -50
    SUFFER_POINT_THRESHOLD = 50

    # Matching window (seconds) - aligns with context window
    MATCHING_WINDOW = 30.0

    # Redundancy window (seconds)
    REDUNDANCY_WINDOW = 5.0

    def __init__(
        self,
        transcript_path: str,
        keyframes_path: str
    ):
        """
        Initialize Synchronizer with transcript and keyframes data.

        Args:
            transcript_path: Path to transcript.json
            keyframes_path: Path to keyframes.json
        """
        # Load transcript
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)

        self.transcripts: List[TranscriptEntry] = []
        for entry in transcript_data.get("entries", []):
            self.transcripts.append(TranscriptEntry(
                start=entry.get("start", 0.0),
                end=entry.get("start", 0.0) + entry.get("duration", 0.0),
                text=entry.get("text", "")
            ))

        # Load keyframes
        with open(keyframes_path, 'r') as f:
            keyframes_data = json.load(f)

        # Handle both formats: list directly or dict with "keyframes" key
        if isinstance(keyframes_data, list):
            self.keyframes: List[Dict] = keyframes_data
        else:
            self.keyframes: List[Dict] = keyframes_data.get("keyframes", [])

    def detect_suffer_points(self) -> List[InterventionPoint]:
        """
        Detect suffer points using the scoring system.

        Returns:
            List of InterventionPoint objects
        """
        interventions = []
        last_intervention_time = None

        for frame in self.keyframes:
            # _calculate_score returns (score, trigger_reason, matched_segment) —
            # eliminates a redundant _match_segment call and gives accurate reasons
            score, trigger_reason, matched_segment = self._calculate_score(frame)

            # Check if score meets threshold OR has meaningful visual content
            complexity = frame.get("complexity_score", 0.0)

            # High confidence: score >= 100 (multiple triggers)
            # Medium confidence: score >= 50 (single trigger)
            # Low confidence: score < 50 but complexity > 0.25 (visual-only)
            meets_threshold = score >= self.SUFFER_POINT_THRESHOLD
            visual_only = complexity > 0.25 and not meets_threshold

            if meets_threshold or visual_only:
                # Check redundancy (skip if within 5s of previous)
                current_time = frame.get("timestamp_seconds", 0.0)

                if last_intervention_time is not None:
                    if abs(current_time - last_intervention_time) < self.REDUNDANCY_WINDOW:
                        continue  # Skip redundant intervention

                transcript_context = matched_segment.text if matched_segment else ""

                # Determine confidence level
                if score >= 100:
                    confidence = "high"
                elif meets_threshold:
                    confidence = "medium"
                else:
                    confidence = "low"

                interventions.append(InterventionPoint(
                    timestamp=current_time,
                    frame_path=frame.get("frame_path", ""),
                    content_type=frame.get("content_type", "text"),
                    complexity_score=complexity,
                    clip_confidence=frame.get("clip_confidence", 0.0),
                    transcript_context=transcript_context,
                    trigger_reason=trigger_reason,
                    confidence=confidence
                ))

                last_intervention_time = current_time

        return interventions

    def _calculate_score(self, frame: Dict) -> tuple:
        """
        Calculate intervention score for a frame.

        Returns (score, trigger_reason, matched_segment) to avoid a redundant
        _match_segment call in detect_suffer_points and to report accurate
        component-based trigger reasons instead of inaccurate score-based guesses.
        """
        score = 0
        triggered = []

        frame_time = frame.get("timestamp_seconds", 0.0)
        matched_segment = self._match_segment(frame_time)

        if matched_segment:
            transcript_text = matched_segment.text.lower()

            for phrase in self.DEICTIC_PHRASES:
                if phrase in transcript_text:
                    score += self.DEICTIC_SCORE
                    triggered.append("Deictic")
                    break

            for phrase in self.NEGATIVE_DEICTIC_PHRASES:
                if phrase in transcript_text:
                    score += self.NEGATIVE_DEICTIC_PENALTY
                    triggered.append("Negative Deictic")
                    break

        if matched_segment and len(matched_segment.text.strip()) < 10:
            complexity = frame.get("complexity_score", 0.0)
            if complexity > 0.3:
                score += self.SILENT_DRAWING_SCORE
                triggered.append("Silent Drawing")

        complexity = frame.get("complexity_score", 0.0)
        if complexity > self.HIGH_COMPLEXITY_THRESHOLD:
            score += self.HIGH_COMPLEXITY_SCORE
            triggered.append("High Complexity")

        trigger_reason = " + ".join(triggered) if triggered else "Unknown"
        return score, trigger_reason, matched_segment

    def _match_segment(self, timestamp: float) -> Optional[TranscriptEntry]:
        """
        Find transcript segment containing timestamp, or nearest within 30s.

        Falls back to nearest segment endpoint when timestamp falls in a gap
        between segments (e.g., pauses between sentences).
        """
        for segment in self.transcripts:
            if segment.start <= timestamp <= segment.end:
                return segment

        # Fallback: nearest segment endpoint within 30s (handles inter-segment gaps)
        if not self.transcripts:
            return None
        nearest = min(
            self.transcripts,
            key=lambda s: min(abs(s.start - timestamp), abs(s.end - timestamp))
        )
        if min(abs(nearest.start - timestamp), abs(nearest.end - timestamp)) <= self.MATCHING_WINDOW:
            return nearest
        return None


