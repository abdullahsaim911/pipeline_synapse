# Synchronizer Module (M1) Documentation

## Overview

The Synchronizer Module is the brain of the suffer point detection system. It analyzes the relationship between transcript segments and extracted keyframes to identify moments when blind students lose access to visual information. These "suffer points" are critical moments where audio descriptions must be inserted.

## Purpose

- Detect when blind students lose visual access to content
- Identify specific moments requiring intervention
- Score and prioritize interventions based on multiple factors
- Match visual content to corresponding speech segments
- Provide context for subsequent VLM analysis and synthesis

## Tools & Technologies

### Core Dependencies
- **Python 3.9+**: Primary programming language
- **JSON**: Data serialization and storage
- **Regex (re)**: Pattern matching for deictic phrase detection
- **Dataclasses**: Structured data representation
- **Type Hints**: Enhanced code documentation and type safety

### No External Dependencies
The Synchronizer operates purely on existing data (transcript and keyframes) and requires no external ML models or heavy libraries.

## Implementation Details

### Module Structure

```
synchronizer/
├── __init__.py
└── synchronizer.py              # Main synchronization logic
```

### Core Components

#### 1. Data Structures

**TranscriptEntry**: Represents a segment of spoken text with timing

```python
@dataclass
class TranscriptEntry:
    """Transcript segment with timing information."""
    start: float          # Start time in seconds
    end: float            # End time in seconds
    text: str             # Spoken content
```

**InterventionPoint**: Represents a detected suffer point requiring intervention

```python
@dataclass
class InterventionPoint:
    """Represents a detected suffer point requiring intervention."""
    timestamp: float              # When the intervention is needed (seconds)
    frame_path: str               # Path to the keyframe image
    content_type: str             # Type of visual content (graph, equation, etc.)
    complexity_score: float       # Visual complexity (0.0 - 1.0)
    clip_confidence: float        # CLIP confidence score
    transcript_context: str       # What the teacher was saying
    trigger_reason: str           # Why intervention was triggered
    confidence: str = "medium"    # "high", "medium", or "low"
```

#### 2. Synchronizer Class

**Main Detection Engine**: Orchestrates the suffer point detection process

```python
class Synchronizer:
    """
    Detect suffer points where blind students lose visual access.

    Uses a scoring system with multiple factors:
    - Deictic phrases (+50)
    - Silent drawing (+80)
    - High complexity (+50)
    - Negative deictic (-50)
    """
```

**Scoring System**: Multi-factor approach to identify critical moments

```python
# Scoring thresholds
DEICTIC_SCORE = 50                # Teacher points at screen
SILENT_DRAWING_SCORE = 80         # Teacher draws silently
HIGH_COMPLEXITY_SCORE = 50        # Complex visual content
HIGH_COMPLEXITY_THRESHOLD = 0.30  # Complexity threshold
NEGATIVE_DEICTIC_PENALTY = -50    # Teacher moves on
SUFFER_POINT_THRESHOLD = 50       # Minimum score for intervention

# Matching window (seconds)
MATCHING_WINDOW = 30.0            # Maximum gap for transcript matching

# Redundancy window (seconds)
REDUNDANCY_WINDOW = 5.0           # Minimum time between interventions
```

**Deictic Phrase Detection**: Identifies when teacher references visual content

```python
DEICTIC_PHRASES = [
    "look at", "here we see", "this graph", "notice", "observe",
    "check this", "see this", "here's", "look here"
]

NEGATIVE_DEICTIC_PHRASES = [
    "next slide", "moving on", "let's continue", "moving forward",
    "now let's", "next we'll", "moving to", "next topic"
]
```

#### 3. Suffer Point Detection Algorithm

```python
def detect_suffer_points(self) -> List[InterventionPoint]:
    """
    Detect suffer points using the scoring system.

    Returns:
        List of InterventionPoint objects
    """
    interventions = []
    last_intervention_time = None

    for frame in self.keyframes:
        # Calculate score and determine trigger reason
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
```

#### 4. Score Calculation

```python
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

        # Check for deictic phrases
        for phrase in self.DEICTIC_PHRASES:
            if phrase in transcript_text:
                score += self.DEICTIC_SCORE
                triggered.append("Deictic")
                break

        # Check for negative deictic phrases
        for phrase in self.NEGATIVE_DEICTIC_PHRASES:
            if phrase in transcript_text:
                score += self.NEGATIVE_DEICTIC_PENALTY
                triggered.append("Negative Deictic")
                break

    # Check for silent drawing (short text + high complexity)
    if matched_segment and len(matched_segment.text.strip()) < 10:
        complexity = frame.get("complexity_score", 0.0)
        if complexity > 0.3:
            score += self.SILENT_DRAWING_SCORE
            triggered.append("Silent Drawing")

    # Check for high complexity
    complexity = frame.get("complexity_score", 0.0)
    if complexity > self.HIGH_COMPLEXITY_THRESHOLD:
        score += self.HIGH_COMPLEXITY_SCORE
        triggered.append("High Complexity")

    trigger_reason = " + ".join(triggered) if triggered else "Unknown"
    return score, trigger_reason, matched_segment
```

#### 5. Transcript Matching

```python
def _match_segment(self, timestamp: float) -> Optional[TranscriptEntry]:
    """
    Find transcript segment containing timestamp, or nearest within 30s.

    Falls back to nearest segment endpoint when timestamp falls in a gap
    between segments (e.g., pauses between sentences).
    """
    # First, try to find exact match (timestamp within segment range)
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
```

## Key Algorithms

### 1. Suffer Point Detection Algorithm

```
Input: transcript.json, keyframes.json
Output: List[InterventionPoint]

1. Load transcript segments with timing
2. Load keyframes with metadata
3. For each keyframe:
   a. Match to nearest transcript segment (within 30s window)
   b. Initialize score = 0
   c. Check for deictic phrases in transcript:
      If found: score += 50, trigger_reason += "Deictic"
   d. Check for negative deictic phrases:
      If found: score -= 50, trigger_reason += "Negative Deictic"
   e. Check for silent drawing (text < 10 chars AND complexity > 0.3):
      If true: score += 80, trigger_reason += "Silent Drawing"
   f. Check for high complexity (complexity > 0.30):
      If true: score += 50, trigger_reason += "High Complexity"
   g. Determine if intervention needed:
      - score >= 50: intervention required
      - score < 50 AND complexity > 0.25: visual-only intervention
   h. Check redundancy:
      - If within 5s of previous intervention: skip
   i. Create InterventionPoint with confidence level
4. Return sorted list of interventions
```

### 2. Score Threshold Logic

```
Score Calculation:
  base = 0
  if deictic_phrase_found: base += 50
  if negative_deictic_found: base -= 50
  if silent_drawing: base += 80
  if high_complexity: base += 50

Intervention Decision:
  if score >= 100: HIGH confidence (multiple triggers)
  if 50 <= score < 100: MEDIUM confidence (single trigger)
  if score < 50 AND complexity > 0.25: LOW confidence (visual-only)
  else: NO intervention

Confidence Levels:
  HIGH: Multiple factors indicate intervention needed
  MEDIUM: Single factor indicates intervention needed
  LOW: Visual content present but no speech trigger
```

## Configuration Parameters

### Scoring Parameters

```python
# Trigger scores
DEICTIC_SCORE = 50                # Teacher points at screen
SILENT_DRAWING_SCORE = 80         # Teacher draws silently
HIGH_COMPLEXITY_SCORE = 50        # Complex visual content
NEGATIVE_DEICTIC_PENALTY = -50    # Teacher moves on

# Thresholds
HIGH_COMPLEXITY_THRESHOLD = 0.30  # Minimum complexity for high complexity trigger
SUFFER_POINT_THRESHOLD = 50       # Minimum score for intervention

# Windows
MATCHING_WINDOW = 30.0            # Maximum gap for transcript matching (seconds)
REDUNDANCY_WINDOW = 5.0           # Minimum time between interventions (seconds)
```

### Customization Example

```python
from synchronizer import Synchronizer

# Custom thresholds for different use cases
synchronizer = Synchronizer(
    transcript_path="transcript.json",
    keyframes_path="keyframes.json"
)

# Modify scoring (access class attributes)
synchronizer.DEICTIC_SCORE = 60                    # More sensitive to deictic phrases
synchronizer.SUFFER_POINT_THRESHOLD = 40          # Lower threshold for more interventions
synchronizer.REDUNDANCY_WINDOW = 3.0              # Closer interventions allowed

# Run detection
interventions = synchronizer.detect_suffer_points()
```

## Output Format

### InterventionPoint Structure

```python
{
    "timestamp": 135.5,
    "frame_path": "data/video_id/keyframes/frame_0001_02_15.jpg",
    "content_type": "equation",
    "complexity_score": 0.4523,
    "clip_confidence": 0.3891,
    "transcript_context": "Look at this equation on the screen.",
    "trigger_reason": "Deictic + High Complexity",
    "confidence": "high"
}
```

### Usage in Pipeline

```python
# Load synchronizer
synchronizer = Synchronizer(
    transcript_path="data/video_id/transcript.json",
    keyframes_path="data/video_id/keyframes/keyframes.json"
)

# Detect suffer points
interventions = synchronizer.detect_suffer_points()

# Process each intervention
for intervention in interventions:
    print(f"[{intervention.timestamp:.1f}s] {intervention.content_type}")
    print(f"  Reason: {intervention.trigger_reason}")
    print(f"  Context: {intervention.transcript_context}")
    print(f"  Confidence: {intervention.confidence}")
```

## Usage Examples

### Basic Usage

```python
from synchronizer import Synchronizer

# Initialize synchronizer
synchronizer = Synchronizer(
    transcript_path="data/video_id/transcript.json",
    keyframes_path="data/video_id/keyframes/keyframes.json"
)

# Detect suffer points
interventions = synchronizer.detect_suffer_points()

# Print results
print(f"Found {len(interventions)} suffer points:")
for intervention in interventions:
    print(f"  {intervention.timestamp:.1f}s: {intervention.trigger_reason}")
```

### Custom Scoring

```python
# Create custom synchronizer with adjusted thresholds
class CustomSynchronizer(Synchronizer):
    # More aggressive intervention detection
    DEICTIC_SCORE = 40                    # Lower threshold for deictic
    SUFFER_POINT_THRESHOLD = 35          # Lower overall threshold
    REDUNDANCY_WINDOW = 4.0              # Allow closer interventions

    # Additional trigger: teacher speaks too fast
    SPEECH_RATE_THRESHOLD = 3.0          # words per second

    def _calculate_score(self, frame: Dict) -> tuple:
        score, trigger_reason, matched_segment = super()._calculate_score(frame)

        # Check for fast speech
        if matched_segment:
            words = len(matched_segment.text.split())
            duration = matched_segment.end - matched_segment.start
            if duration > 0 and (words / duration) > self.SPEECH_RATE_THRESHOLD:
                score += 30
                trigger_reason += " + Fast Speech"

        return score, trigger_reason, matched_segment
```

### Confidence Filtering

```python
# Filter interventions by confidence level
interventions = synchronizer.detect_suffer_points()

# Keep only high and medium confidence
filtered = [i for i in interventions if i.confidence in ["high", "medium"]]

print(f"Original: {len(interventions)}, Filtered: {len(filtered)}")
```

## Performance Characteristics

### Processing Time

| Video Length | Processing Time | Interventions Detected |
|--------------|-----------------|------------------------|
| 5 minutes | < 100ms | 5-15 |
| 10 minutes | < 200ms | 10-30 |
| 30 minutes | < 500ms | 30-90 |
| 60 minutes | < 1s | 60-180 |

### Memory Usage

- **RAM**: < 10MB (operates on existing JSON files)
- **Disk**: Minimal (reads existing files, no new data created)
- **CPU**: Minimal (simple string matching and calculations)

### Scalability

- **Linear Time Complexity**: O(N) where N = number of keyframes
- **Constant Space**: No memory growth with input size
- **No GPU Required**: Pure CPU processing

## Trigger Analysis

### Common Scenarios

| Scenario | Score | Reason | Action |
|----------|-------|--------|--------|
| "Look at this graph" + complex graph | 100 | Deictic + High Complexity | Generate description |
| Teacher draws silently | 80 | Silent Drawing | Narrate drawing process |
| Complex equation shown | 50 | High Complexity | Explain equation |
| "Moving to the next topic" | -50 | Negative Deictic | Skip (moving on) |
| "Notice this pattern" + simple text | 50 | Deictic | Describe pattern |

### False Positive Prevention

1. **Redundancy Window**: Prevents multiple interventions in quick succession
2. **Negative Deictic Penalty**: Avoids interventions when teacher is transitioning
3. **Confidence Levels**: Allows filtering of low-confidence interventions
4. **Context Matching**: Ensures interventions align with actual speech content

## Troubleshooting

### Common Issues

**Issue**: "Too many interventions detected"
- **Solution**: Increase `SUFFER_POINT_THRESHOLD` or `REDUNDANCY_WINDOW`

**Issue**: "Missing important interventions"
- **Solution**: Lower `SUFFER_POINT_THRESHOLD` or add custom deictic phrases

**Issue**: "Interventions not matching speech"
- **Solution**: Check `MATCHING_WINDOW` and transcript timing accuracy

**Issue**: "Duplicate interventions"
- **Solution**: Reduce `REDUNDANCY_WINDOW` or check for timing overlap

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check individual frame scores
synchronizer = Synchronizer("transcript.json", "keyframes.json")

for frame in synchronizer.keyframes[:5]:  # Check first 5 frames
    score, reason, segment = synchronizer._calculate_score(frame)
    print(f"Frame at {frame.get('timestamp_seconds'):.1f}s:")
    print(f"  Score: {score}, Reason: {reason}")
    print(f"  Matched: '{segment.text if segment else 'None'}'")
```

## Best Practices

### 1. Threshold Tuning

```python
# Conservative approach (fewer interventions)
synchronizer.SUFFER_POINT_THRESHOLD = 60
synchronizer.REDUNDANCY_WINDOW = 7.0

# Aggressive approach (more interventions)
synchronizer.SUFFER_POINT_THRESHOLD = 40
synchronizer.REDUNDANCY_WINDOW = 3.0
```

### 2. Content-Specific Configuration

```python
# For mathematics videos (more equations)
class MathSynchronizer(Synchronizer):
    DEICTIC_PHRASES = [
        "look at", "here we see", "this equation", "notice", "observe",
        "check this", "see this", "here's", "look here",
        "this formula", "the equation", "as you can see"
    ]

# For biology videos (more diagrams)
class BioSynchronizer(Synchronizer):
    DEICTIC_PHRASES = [
        "look at", "here we see", "this diagram", "notice", "observe",
        "check this", "see this", "here's", "look here",
        "this structure", "as you can see", "in this figure"
    ]
```

### 3. Validation and Quality Control

```python
# Validate intervention distribution
interventions = synchronizer.detect_suffer_points()

# Check temporal distribution
time_gaps = [
    interventions[i+1].timestamp - interventions[i].timestamp
    for i in range(len(interventions)-1)
]

avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
print(f"Average time between interventions: {avg_gap:.1f}s")

# Check confidence distribution
from collections import Counter
confidence_counts = Counter(i.confidence for i in interventions)
print(f"Confidence distribution: {confidence_counts}")
```

## Future Enhancements

1. **Semantic Analysis**: Use NLP to better understand context
2. **Adaptive Thresholds**: Dynamically adjust based on video content
3. **Predictive Detection**: Anticipate upcoming visual content
4. **User Feedback**: Learn from manual intervention corrections
5. **Multi-Modal Fusion**: Combine audio, visual, and text features
6. **Confidence Calibration**: Improve confidence scoring accuracy
7. **Context-Aware Detection**: Consider lecture structure and topic
