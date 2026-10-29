# synapse_interaction

Detects WRITING_DRAWING, MOUSE_POINTING and HAND_POINTING in lecture videos (CPU only) and writes
`interactions.json`. Everything for this feature lives in this folder.

## Commands (run from the project root, environment activated)
    python -m synapse_interaction.run <video.mp4> --out <dir> [--dump-steps] [--no-hand]   # a file already on disk
    python -m synapse_interaction.get_models                                                # one-off hand-model download
    pytest synapse_interaction/tests -q

## Layout
    stream.py                                          InteractionStream: the pipeline entry point (frames in, interactions.json out)
    motion.py classify.py hand.py                      detection (ink/motion scan, writing vs pointer, MediaPipe hands)
    fusion.py novelty.py                               merge the events into M1's intervention points
    schema.py config.py video_io.py                    data classes, thresholds, file helpers
    detector.py run.py                                 video-file entry point for the command line and tests
    tests/                                             unit, synthetic-video, extractor-tap and pipeline-integration tests

## How it gets its frames (no video is opened or downloaded)
In the pipeline the frame extractor decodes the lecture once and hands every decoded frame to this module:

    stream = InteractionStream(Config(), out_path=".../interactions.json")
    extract_pedagogical_keyframes(..., frame_tap=stream.feed)   # optional hook, off by default
    stream.finish()                                             # writes interactions.json

With the hook the video is decoded once at 2 fps and the extractor still receives every 4th frame (same frames, same
timestamps and pixels), so its own keyframes are exactly the same as without it. A test runs the real extractor both ways.
`detect_interactions(video)` (CLI / tests) opens a file and feeds the same code.
Pictures for the detected events are taken from the pushed frames (spooled as small JPEGs, the unused ones deleted).

## HAND_POINTING (optional)
Needs `pip install mediapipe` and, with current mediapipe, one model file: run `python -m synapse_interaction.get_models`
(saved in `synapse_interaction/models/`). Without them the hand stage skips itself and says so in the log.
Set `SYNAPSE_HAND=0` to switch it off in the pipeline.
