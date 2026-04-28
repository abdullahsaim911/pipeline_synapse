"""Debug script for deictic detection issue."""

import json
from pathlib import Path

# Paths
transcript_path = Path("F:/Prototyping/Synapse/data/YG15m2VwSjA/transcript.json")
keyframes_path = Path("F:/Prototyping/Synapse/data/YG15m2VwSjA/keyframes/keyframes.json")

# Load transcript
with open(transcript_path) as f:
    transcript_data = json.load(f)

# Load keyframes
with open(keyframes_path) as f:
    keyframes_data = json.load(f)

# Deictic phrases from synchronizer.py
DEICTIC_PHRASES = [
    "look at", "here we see", "this graph", "notice", "observe",
    "check this", "see this", "here's", "look here"
]

print("=" * 60)
print("DETECTING DEICTIC PHRASES IN TRANSCRIPT")
print("=" * 60)

deictic_segments = []
for entry in transcript_data["entries"]:
    text_lower = entry["text"].lower()
    for phrase in DEICTIC_PHRASES:
        if phrase in text_lower:
            deictic_segments.append({
                "start": entry["start"],
                "end": entry["end"],
                "text": entry["text"],
                "phrase": phrase
            })
            break

print(f"\nFound {len(deictic_segments)} segments with deictic phrases:")
for i, seg in enumerate(deictic_segments[:10], 1):
    print(f"{i}. [{seg['start']:.2f}s - {seg['end']:.2f}s] '{seg['phrase']}' in: {seg['text'][:60]}...")

if len(deictic_segments) > 10:
    print(f"... and {len(deictic_segments) - 10} more")

print("\n" + "=" * 60)
print("KEYFRAME TIMESTAMPS")
print("=" * 60)

print("\nKeyframes (first 10):")
for i, kf in enumerate(keyframes_data[:10], 1):
    ts = kf["timestamp_seconds"]
    print(f"{i}. {ts}s")

print("\n" + "=" * 60)
print("MATCHING DEICTIC SEGMENTS TO KEYFRAMES (5s window)")
print("=" * 60)

def find_matching_keyframe(timestamp, keyframes):
    """Find keyframe within 5s of timestamp."""
    for kf in keyframes:
        ts = kf["timestamp_seconds"]
        if abs(ts - timestamp) <= 5.0:
            return kf
    return None

matches = 0
for seg in deictic_segments:
    # Check midpoint of segment
    midpoint = (seg["start"] + seg["end"]) / 2
    kf = find_matching_keyframe(midpoint, keyframes_data)
    if kf:
        matches += 1
        print(f"✓ Match: {seg['phrase']} at {seg['start']:.2f}s -> keyframe at {kf['timestamp_seconds']}s")

print(f"\n{matches}/{len(deictic_segments)} deictic segments matched to keyframes within 5s window")

print("\n" + "=" * 60)
print("GAPS BETWEEN KEYFRAMES")
print("=" * 60)

gaps = []
for i in range(len(keyframes_data) - 1):
    gap = keyframes_data[i+1]["timestamp_seconds"] - keyframes_data[i]["timestamp_seconds"]
    gaps.append(gap)
    if gap > 10:
        print(f"Large gap: {gap:.1f}s between {keyframes_data[i]['timestamp_seconds']}s and {keyframes_data[i+1]['timestamp_seconds']}s")

print(f"\nAverage gap: {sum(gaps)/len(gaps):.1f}s")
print(f"Max gap: {max(gaps):.1f}s")

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
print(f"\nKeyframes: {len(keyframes_data)}")
print(f"Deictic segments found: {len(deictic_segments)}")
print(f"Matches within 5s: {matches}")

print("\nKeyframe timestamps where deictic phrases appear nearby:")
for seg in deictic_segments[:10]:
    midpoint = (seg["start"] + seg["end"]) / 2
    # Find nearest keyframe
    nearest = min(keyframes_data, key=lambda kf: abs(kf["timestamp_seconds"] - midpoint))
    distance = abs(nearest["timestamp_seconds"] - midpoint)
    print(f"  Deictic at {seg['start']:.1f}s -> Nearest keyframe at {nearest['timestamp_seconds']}s (distance: {distance:.1f}s)")
