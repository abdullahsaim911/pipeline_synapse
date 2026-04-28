"""
Context Manager Module

Gathers context for explanation generation.
Handles edge cases and provides robust error handling.
"""

import logging
from typing import List, Dict, Optional, Any
import re

logger = logging.getLogger(__name__)


class ContextManager:
    """Gathers and manages context for different explanation modes."""

    # Context window sizes for each mode (in seconds, ±) - Enhanced for deeper explanations
    CONTEXT_WINDOWS = {
        "brief": 30,      # Increased from 10s - need more context for why this visual appears
        "explanatory": 60, # Increased from 30s - connect to prior concepts, build narrative
        "detailed": 120,  # Increased from 50s - full lecture segment for deep connections
        "standard": 30
    }

    # Keywords for concept extraction
    CONCEPT_KEYWORDS = [
        "derivative", "integral", "slope", "rate", "function",
        "velocity", "acceleration", "force", "energy", "power",
        "equation", "graph", "linear", "proportion", "ratio",
        "theorem", "principle", "law", "formula", "variable",
        "exponential", "logarithm", "probability", "matrix", "vector",
        "optimization", "symmetry", "equilibrium", "wave", "entropy",
        "limit", "continuity", "differentiation", "integration",
        "area", "volume", "surface", "curve", "line", "point",
        "angle", "triangle", "circle", "polygon", "geometry"
    ]

    # Concept to principle mappings
    CONCEPT_TO_PRINCIPLE = {
        "derivative": ["rate of change", "slope", "instantaneous rate", "tangent line"],
        "integral": ["accumulation", "area", "total", "antiderivative", "summation"],
        "slope": ["rate of change", "derivative", "tangent", "gradient", "steepness"],
        "velocity": ["speed", "rate of motion", "derivative of position", "directional speed"],
        "acceleration": ["rate of velocity change", "derivative of velocity", "second derivative of position"],
        "force": ["mass times acceleration", "newton's second law", "push or pull", "interaction"],
        "energy": ["work", "power", "capacity", "potential", "kinetic energy"],
        "linear": ["proportional", "direct relationship", "constant rate", "straight line"],
        "exponential": ["rapid growth", "compound growth", "multiplicative change"],
        "logarithm": ["inverse of exponential", "scale compression", "orders of magnitude"],
        "probability": ["likelihood", "chance", "random event", "distribution"],
        "matrix": ["array", "linear transformation", "system of equations", "data structure"],
        "vector": ["direction and magnitude", "multi-dimensional quantity", "arrow representation"],
        "equilibrium": ["balance point", "stable state", "forces balance", "no net change"],
        "optimization": ["maximization", "minimization", "best solution", "extremum finding"],
        "symmetry": ["invariance", "reflection", "rotation", "mirror image", "balance"],
        "wave": ["oscillation", "periodic", "frequency", "amplitude", "propagation"],
        "entropy": ["disorder", "randomness", "uncertainty", "information content"],
        "limit": ["approach value", "convergence", "boundary value", "tends to"],
        "continuity": ["no breaks", "smooth", "connected", "uninterrupted"],
        "differentiation": ["finding derivative", "rate of change", "slope finding"],
        "integration": ["finding integral", "accumulation", "area calculation"],
        "area": ["space covered", "region", "surface measure", "planar extent"],
        "volume": ["3D space", "capacity", "cubic measure", "space occupied"],
        "angle": ["rotation", "opening", "geometric figure", "vertex measure"]
    }

    # Topic detection keywords
    TOPIC_KEYWORDS = {
        "equations": ["equation", "solve", "equals", "formula", "variable", "unknown"],
        "graphs": ["graph", "plot", "curve", "axis", "coordinate", "intersection", "slope"],
        "theorems": ["theorem", "principle", "law", "proof", "corollary", "lemma"],
        "functions": ["function", "mapping", "input", "output", "domain", "range"],
        "calculus": ["derivative", "integral", "limit", "differentiate", "integrate"],
        "geometry": ["triangle", "circle", "angle", "polygon", "area", "volume", "perimeter"],
        "probability": ["probability", "chance", "random", "distribution", "expected value"],
        "linear_algebra": ["matrix", "vector", "determinant", "eigenvalue", "transformation"],
        "statistics": ["mean", "median", "mode", "standard deviation", "variance", "distribution"],
        "physics": ["force", "velocity", "acceleration", "energy", "momentum", "work"],
        "unknown": []  # Default topic
    }

    def __init__(self, transcript_entries: Optional[List] = None):
        """
        Initialize Context Manager.

        Args:
            transcript_entries: List of TranscriptEntry objects or dictionaries
        """
        self.transcripts = self._validate_transcripts(transcript_entries)

        if not self.transcripts:
            logger.warning("ContextManager initialized with no valid transcripts")

    def _validate_transcripts(self, transcript_entries: Optional[List]) -> List[Any]:
        """
        Validate and normalize transcript entries.

        Args:
            transcript_entries: Raw transcript data

        Returns:
            List of validated transcript entries
        """
        if not transcript_entries:
            return []

        validated = []
        for i, entry in enumerate(transcript_entries):
            try:
                # Handle dictionary entries
                if isinstance(entry, dict):
                    # Ensure required fields exist
                    if "text" not in entry:
                        logger.warning(f"Transcript entry {i} missing 'text' field, skipping")
                        continue

                    # Convert to a consistent structure
                    validated.append(type('TranscriptEntry', (), {
                        'text': entry.get('text', ''),
                        'start': float(entry.get('start', 0.0)),
                        'end': float(entry.get('end', entry.get('start', 0.0)) + 1.0)
                    })())
                # Handle object entries (already have required attributes)
                elif hasattr(entry, 'text') and hasattr(entry, 'start'):
                    validated.append(entry)
                else:
                    logger.warning(f"Transcript entry {i} has invalid structure, skipping")

            except Exception as e:
                logger.error(f"Error validating transcript entry {i}: {e}", exc_info=True)
                continue

        return validated

    def get_context(self, current_index: int, mode: str) -> Dict:
        """
        Get context based on mode.

        Args:
            current_index: Index of current transcript segment
            mode: "brief", "explanatory", "detailed", or "standard"

        Returns:
            Dictionary with context data including:
            - context_text: Combined text from context window
            - previous_concepts: List of concepts mentioned before current point
            - related_principles: Related mathematical/physical principles
            - topic_timeline: Timeline of topics discussed

        Examples:
            >>> cm = ContextManager(transcripts)
            >>> context = cm.get_context(0, "brief")
            >>> len(context["context_text"])
            150  # Approximately 10 seconds of speech
        """
        try:
            # Validate and normalize mode
            mode = mode.lower() if mode else "standard"
            if mode not in self.CONTEXT_WINDOWS:
                logger.warning(f"Unknown mode '{mode}', using 'standard'")
                mode = "standard"

            window = self.CONTEXT_WINDOWS[mode]

            # Get current segment safely
            current_segment = self._get_current_segment(current_index)
            current_time = current_segment.start if current_segment else 0.0

            # Get context segments within window
            context_segments = self._get_segments_in_window(current_time, window)

            # Extract concepts (only from segments before current time)
            previous_concepts = self._extract_concepts(
                [s for s in context_segments if s.start < current_time]
            )

            # Find related principles
            related_principles = self._find_related_principles(previous_concepts)

            # Build topic timeline
            topic_timeline = self._build_topic_timeline(context_segments)

            return {
                "context_text": " ".join(entry.text for entry in context_segments),
                "previous_concepts": previous_concepts,
                "related_principles": related_principles,
                "topic_timeline": topic_timeline,
                "window_size": window,
                "mode": mode
            }

        except Exception as e:
            logger.error(f"Error getting context: {e}", exc_info=True)
            return {
                "context_text": "",
                "previous_concepts": [],
                "related_principles": [],
                "topic_timeline": {},
                "window_size": 30,
                "mode": mode if mode else "standard"
            }

    def _get_current_segment(self, current_index: int) -> Any:
        """Safely get current transcript segment."""
        if not self.transcripts:
            return None

        try:
            index = max(0, min(current_index, len(self.transcripts) - 1))
            return self.transcripts[index]
        except (IndexError, TypeError) as e:
            logger.error(f"Error getting current segment at index {current_index}: {e}")
            return self.transcripts[0] if self.transcripts else None

    def _get_segments_in_window(self, timestamp: float, window_seconds: float) -> List[Any]:
        """
        Get transcript segments within time window.

        Args:
            timestamp: Center timestamp
            window_seconds: Window size (±)

        Returns:
            List of transcript entries within the window
        """
        if not self.transcripts:
            return []

        try:
            segments = []
            start_time = timestamp - window_seconds
            end_time = timestamp + window_seconds

            for entry in self.transcripts:
                # Check if segment overlaps with window
                if (entry.start <= end_time) and (entry.end >= start_time):
                    segments.append(entry)

            # Sort by timestamp
            segments.sort(key=lambda x: x.start)

            return segments

        except Exception as e:
            logger.error(f"Error getting segments in window: {e}", exc_info=True)
            return []

    def _extract_concepts(self, segments: List[Any]) -> List[str]:
        """
        Extract key concepts from transcript segments.

        Args:
            segments: List of transcript segments

        Returns:
            List of unique concept names
        """
        concepts = []

        if not segments:
            return concepts

        try:
            # Combine all text
            text = " ".join(
                entry.text.lower() if hasattr(entry, 'text') else str(entry).lower()
                for entry in segments
            )

            # Extract concepts based on keywords
            for keyword in self.CONCEPT_KEYWORDS:
                if keyword in text and keyword not in concepts:
                    concepts.append(keyword)

            return concepts

        except Exception as e:
            logger.error(f"Error extracting concepts: {e}", exc_info=True)
            return []

    def _find_related_principles(self, concepts: List[str]) -> List[str]:
        """
        Find related mathematical/physical principles.

        Args:
            concepts: List of concept names

        Returns:
            List of related principle names
        """
        principles = []

        if not concepts:
            return principles

        try:
            for concept in concepts:
                concept_lower = concept.lower()
                if concept_lower in self.CONCEPT_TO_PRINCIPLE:
                    for principle in self.CONCEPT_TO_PRINCIPLE[concept_lower]:
                        if principle not in principles:
                            principles.append(principle)

            return principles

        except Exception as e:
            logger.error(f"Error finding related principles: {e}", exc_info=True)
            return []

    def _build_topic_timeline(self, segments: List[Any]) -> Dict[str, str]:
        """
        Build timeline of topics discussed.

        Args:
            segments: List of transcript segments

        Returns:
            Dictionary mapping timestamps to topics
        """
        if not segments:
            return {}

        timeline = {}
        current_topic = "unknown"

        try:
            for entry in segments:
                if not hasattr(entry, 'text'):
                    continue

                text = entry.text.lower()

                # Detect topic based on keywords
                detected_topic = None
                for topic, keywords in self.TOPIC_KEYWORDS.items():
                    if topic == "unknown":
                        continue
                    if any(keyword in text for keyword in keywords):
                        detected_topic = topic
                        break

                # Update topic if changed
                if detected_topic and detected_topic != current_topic:
                    current_topic = detected_topic

                # Add to timeline
                try:
                    timestamp = float(entry.start)
                    timeline[f"{timestamp:.1f}"] = current_topic
                except (ValueError, AttributeError, TypeError) as e:
                    logger.debug(f"Error converting timestamp to float: {e}")
                    continue

            return timeline

        except Exception as e:
            logger.error(f"Error building topic timeline: {e}", exc_info=True)
            return {}

    def get_surrounding_text(self, timestamp: float, before_seconds: float,
                           after_seconds: float) -> str:
        """
        Get text surrounding a specific timestamp.

        Args:
            timestamp: Center timestamp
            before_seconds: Seconds before timestamp
            after_seconds: Seconds after timestamp

        Returns:
            Combined text from surrounding segments
        """
        if not self.transcripts:
            return ""

        try:
            segments = self._get_segments_in_window(timestamp, max(before_seconds, after_seconds))

            # Filter segments based on direction
            before_text = []
            after_text = []

            for entry in segments:
                if entry.start < timestamp:
                    before_text.append(entry.text)
                elif entry.start >= timestamp:
                    after_text.append(entry.text)

            return " ".join(before_text + after_text)

        except Exception as e:
            logger.error(f"Error getting surrounding text: {e}", exc_info=True)
            return ""

    def get_concept_history(self, concept: str, lookback_seconds: float = 300) -> List[Dict]:
        """
        Get history of when a concept was mentioned.

        Args:
            concept: Concept to search for
            lookback_seconds: How far back to look (default: 5 minutes)

        Returns:
            List of dictionaries with timestamp and context
        """
        if not concept or not self.transcripts:
            return []

        try:
            history = []
            concept_lower = concept.lower()

            for entry in self.transcripts:
                if concept_lower in entry.text.lower():
                    history.append({
                        "timestamp": entry.start,
                        "text": entry.text,
                        "context": self.get_surrounding_text(entry.start, 5, 5)
                    })

            # Filter by lookback time
            if self.transcripts:
                current_time = max(e.start for e in self.transcripts)
                cutoff_time = current_time - lookback_seconds
                history = [h for h in history if h["timestamp"] >= cutoff_time]

            return history

        except Exception as e:
            logger.error(f"Error getting concept history: {e}", exc_info=True)
            return []
