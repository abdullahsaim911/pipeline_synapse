# synapse_interaction

Detects WRITING_DRAWING, MOUSE_POINTING and HAND_POINTING in lecture videos (CPU only) and writes
`interactions.json`. Everything for this feature lives in this folder.

## Commands (run from the project root, environment activated)
    python -m synapse_interaction.run  <video.mp4|url> --out <dir> [--debug] [--dump-steps] [--no-hand]
    python -m synapse_interaction.baseline "<youtube url>"             # existing pipeline up to M1 -> M1's "before" points
    python -m synapse_interaction.preview data\<video_id> [--text]     # what the fusion rules would do
    python -m synapse_interaction.review  data\<video_id>              # build the answer key (blind y/n review)
    python -m synapse_interaction.eval_interventions --video-dir data\<video_id>   # M1 vs M1 + interactions
    pytest synapse_interaction/tests -q

## Layout
    run.py detector.py motion.py classify.py hand.py   detection
    fusion.py                                          merge with M1's points (until M1 itself is changed)
    eval_interventions.py review.py baseline.py        evaluation tools
    eval_data/ground_truth, eval_data/baseline         answer keys and M1 baselines (small JSON, commit these)
    tests/                                             unit + synthetic-video tests

## HAND_POINTING (optional)
Needs mediapipe plus two model files in `models/` (hand_landmarker.task, pose_landmarker_lite.task).
Without them the hand stage skips itself.
