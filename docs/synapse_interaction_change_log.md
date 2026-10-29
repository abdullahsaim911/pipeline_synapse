# Synapse Interaction: change log (what was changed, in which file, and why)

This is the file-by-file record of the gesture / interaction work done on 2026-10-29. The overview, audit findings
and open items are in [`synapse_interaction.md`](synapse_interaction.md). Voice control is a separate piece of work
and is documented in [`voice-control.md`](voice-control.md).

Nothing described here is committed yet. Every change can be inspected with `git diff` and undone with git (the
deleted files are in commit `c59d8c8`).

## 1. Why the work was needed

The gesture / interaction module (`synapse_interaction`) had been added to the pipeline last. Three things were wrong
or unwanted:

1. In the pipeline it never ran its hand-pointing part (`Config(enable_hand=False)`), and its dependencies were
   missing (no `mediapipe`, no model file; `opencv-python` absent from the backend `.venv`).
2. It opened and, when OpenCV failed, **downloaded** the lecture again (yt-dlp), after the frame extractor had
   already decoded it. The requirement is: take the frames from the pipeline, never download or stream again,
   detect, then hand the result to fusion.
3. Its folder was full of evaluation and demo tools that the pipeline never uses.

## 2. Rules that shaped the work

Stated by the project owner during the work, in this order:

| Rule | Consequence |
|---|---|
| Use the frame extractor's frames; do not download or stream the video again | New frame-fed design (section 3) |
| Gesture must feed the fusion step that creates intervention / suffer points | Integration test with the real `Synchronizer` (section 5) |
| Do not change the working of other modules (VLM engine, frame extraction) | `vlm_*`, `synchronizer`, `synthesizer` untouched; the orchestrator reference-image fix was reverted, then done again when explicitly approved (5b) |
| The frame extractor MAY be changed to provide input, but its working must not change | Optional, off-by-default hook; tested against the real function (section 3.3) |
| Remove extra files of the module | Eight tools deleted (section 4) |
| Add the module's requirements to `requirements.txt` | Section 3.5 |

History of one decision, so nobody repeats it: first an extractor hook was added; then, when told not to touch the
extractor, it was reverted and replaced by `tap.py` (a run-time wrapper around `stream_frames`); then, when told the
extractor may be changed as long as its working is unchanged, the hook came back and `tap.py` was deleted. The hook
is the final design: it is explicit, per call and thread-safe, whereas the wrapper patched a global function.

## 3. Changes, file by file

### 3.1 New files in `synapse_interaction/`

| File | What it is | Why |
|---|---|---|
| `stream.py` | `InteractionStream`: `feed(t, bgr_frame)` per frame, `finish()` to build events, choose pictures and write `interactions.json`. `feed` never raises. Event pictures are spooled as small JPEGs (at most one per `spool_interval`, 1 s), the nearest one per event is copied to `interaction_frames/NNN_TYPE_time.jpg`, then the spool is deleted. | The core of the requirement: the module receives frames instead of opening a video. Event pictures cannot be re-read by seeking in a video any more, so they are taken from the pushed frames. |
| `get_models.py` | `python -m synapse_interaction.get_models` downloads `hand_landmarker.task` into `synapse_interaction/models/`. Downloads only when run. | Current MediaPipe needs a model file that is not in the repo; nothing should download silently. |
| `tests/test_stream.py` | 15 tests: frames pushed at 2 fps give the same events as the file path; pictures exist and the spool is removed; `feed` survives bad input; `cv2.VideoCapture` is never called; hand scanner merges hits / rate-limits / rejects a flash and waving; hand stage skips without MediaPipe; architecture guard (no `yt_dlp`, only `video_io.py` opens video); URLs are refused. | Proves the new behaviour and locks the "never download or stream" rule. |
| `tests/test_extractor_tap.py` | 4 tests that run the **real** `extract_pedagogical_keyframes` (torch / PIL stubbed, stopped after the frame-sampling phase) with no tap, with a tap, with a crashing tap and with a slow tap, and compare what the extractor receives (timestamps and pixels). | Proves the extractor hook does not change the extractor. |
| `tests/test_pipeline_integration.py` | 4 tests: synthetic video through `InteractionStream`, the real `Synchronizer`, then `fuse_interactions`; nearby points upgraded, nothing removed, every point has a real picture; new points use the pushed-frame pictures; a re-run clears old pictures; hand-pointing events flow through fusion (upgrade nearby, new point when alone and deictic). | The end-to-end check of "gesture feeds intervention points". |

### 3.2 Modified files in `synapse_interaction/`

| File | Change | Why |
|---|---|---|
| `motion.py` | The scan is now a stateful `Scanner.push(t, frame)`; `scan(stream)` (file path) is a thin wrapper. Maths unchanged. | Frames arrive one at a time from the extractor; the old code pulled them from a video. Results stay identical (the 14 original tests still pass). |
| `hand.py` | Rewritten around pushed frames: `HandScanner.push(t, frame)` (rate-limited to `hand_fps`), `interactions_from_hits()` (pure, testable), `detect()` kept for files. Only the hand model is needed (the pose "is there a person" pre-check, which seeked 16 times, was removed). `index_pointing(lm, aspect)` measures in square units; path and speed use frame-heights. | No video access; fewer files and no seeking; geometry correct on 16:9 frames (a cheap correctness fix, not a proven cause of misses). |
| `config.py` | `hand_model_path` is relative to the package (was the working directory); `pose_model_path` and `hand_probe_frames` removed; `spool_interval` added. | The server can start from any directory; settings that no longer exist are gone. |
| `detector.py` | `detect_interactions(file)` now just feeds an `InteractionStream` from a local file. | One detection code path for the pipeline, the command line and the tests. |
| `video_io.py` | `resolve_stream` accepts only a file on disk; a URL or YouTube id raises a clear error. yt-dlp removed. | The module must never download or stream video. |
| `run.py` | Local files only; `--debug` removed. | Followed from the two rows above (the debug renderer is deleted). |
| `fusion.py` | Which picture the VLM describes is now decided by confidence (section 5c). `_gesture_picture_wins()`, new `pic_window` argument (4 s), docstring. Nothing else in fusion changed. | Requested: the higher-scoring picture should be the one explained, not always the gesture picture. |
| `__init__.py` | Exports `InteractionStream`; docstring updated. | Public entry point of the new design. |
| `README.md` | Commands, layout, "how it gets its frames", hand-model instructions. | Documentation matched the code. |
| `requirements-interaction.txt` | Dropped `yt-dlp`; `mediapipe>=0.10.14`; comment on the model step. | The module no longer needs yt-dlp. |
| `tests/test_interaction.py` | Removed three tests that exercised the deleted tools; the other tests are unchanged and pass. | The tools they tested are deleted. |

### 3.3 Frame extractor (one file, one optional hook)

`frame_extraction/pedagogical_extractor_v2.py`: `extract_pedagogical_keyframes(..., frame_tap=None, tap_fps=2.0)`.
24 lines added, 1 changed (a signature line, a comment block, and a small block before / inside the frame loop).

- `frame_tap=None` (the default): the code path is the original one. Same stream rate, same frames.
- With a tap: the stream is decoded **once** at `tap_fps` (2 fps instead of 0.5). Every decoded frame is passed to
  the tap. The extractor keeps every `tap_fps / sample_fps`-th frame (every 4th), which are the same frames with
  the same timestamps it would have used (0, 2, 4, ... s). A tap that raises is ignored.
- The tap rate is rounded up to a whole multiple of `sample_fps` (0.5 x 4 = 2 fps by default; 0.3 x 7 = 2.1 fps), so the
  extractor's frames stay exactly the same for any `sample_fps` (tested).
- Why a hook at all: the gesture and writing detectors were built for 2 to 4 frames per second, and the extractor
  samples only 0.5 per second. Without the hook the video would have to be decoded a second time, which is exactly
  what must not happen.
- Proof it does not change the extractor: `tests/test_extractor_tap.py` (section 3.1).

### 3.4 Orchestrator (`orchestrator/orchestrator.py`)

Only the lines that were already the interaction module's were edited, inside `_execute_parallel_extraction`:

- **Removed:** the block that ran after extraction and called `detect_interactions(youtube_url, ...)` with
  `Config(enable_hand=False)`; when OpenCV could not read the stream it downloaded `video.mp4` with yt-dlp and ran
  again.
- **Added:** an `InteractionStream` is created (unless `SYNAPSE_INTERACTIONS=0`; hand pointing is on unless
  `SYNAPSE_HAND=0`), passed to the extractor as `frame_tap=interaction_stream.feed`, and `finish()` is called when
  extraction ends, inside the same video thread. Errors are printed and never stop the pipeline.
- **Unchanged:** `_synchronize` still calls `fuse_interactions(...)` on `interactions.json`. The three entry points
  (full, detection-only, intervention-only) all go through this path.

### 3.5 Other files

| File | Change | Why |
|---|---|---|
| `requirements.txt` | New section "Synapse Interaction Module": `mediapipe>=0.10.14` plus comments (no yt-dlp needed, model download step, `SYNAPSE_HAND=0`, ffmpeg must be on PATH). `numpy` and `opencv-python` were already listed. | Requested; hand pointing could not run without `mediapipe`. |
| `.gitignore` | Added `synapse_interaction/models/`. | The downloaded model file must not be committed. |
| `docs/SYNAPSE_COMPLETE_PIPELINE_DOCUMENTATION.md` | M7 item 4 described the deleted `before_after.py`; it now describes `stream.py`. | Documentation must not describe files that do not exist. |
| `docs/synapse_interaction.md` | New: overview, audit, status, open items. | Requested. |
| `docs/synapse_interaction_change_log.md` | This file. | Requested. |

## 4. Deleted files (extra, never imported by the pipeline)

`before_after.py`, `debug_render.py`, `demo_report.py`, `eval_interventions.py`, `preview.py`, `review.py`,
`showcase.py`, `prepare.py`, and the empty `eval_data/baseline` and `eval_data/ground_truth` folders.

Why: they were evaluation and demo tooling (answer-key review, before / after reports, debug video, a downloader).
A search showed they were referenced only by each other, the module README, one documentation paragraph and three
tests. `prepare.py` and `video_io.resolve_stream` also downloaded video, which is now forbidden. Recover any of them
with `git show c59d8c8:synapse_interaction/<file>`.

## 5. Integration check (decoded frames to intervention points)

`tests/test_pipeline_integration.py` runs the path the pipeline takes with the real `Synchronizer` and
`InterventionPoint`. Findings:

- Fusion works as designed: nearby points are upgraded (for example `High Complexity` becomes
  `High Complexity + WRITING_DRAWING (silent)` and confidence rises), nothing is removed or rewritten, new points
  are created for events far from any point, hand-pointing events behave like the others.
- **Fixed, in this module:** event pictures were stored as absolute paths, unlike the extractor's relative
  `data\<id>\keyframes\...`. They now follow `out_path` as given (`data/<id>/interaction_frames/...`). A re-run also
  left the previous run's pictures behind; they are now cleared.
- **Fixed afterwards, with permission (orchestrator):** `Orchestrator._synthesize_interventions_only` saved the nearest
  *extractor keyframe* as each intervention's reference `frame.jpg`. For a point added by fusion the VLM describes the
  interaction picture, but the saved reference image was a different keyframe. See section 5b.
- **Left alone:** `fusion._decide` returns a "may create a new point" flag that `_fuse` never reads. Using or
  deleting it changes which points are created, so it needs the fusion owner's decision.

## 5b. Reference-image fix (orchestrator only)

File: `orchestrator/orchestrator.py`. Two small edits, no other module touched:
- `_analyze_vlm_snapshots` records `(video folder, timestamp) -> point.frame_path` for every point it is about to
  describe. The key includes the video folder because the API shares one orchestrator instance between videos.
- `_find_frame_for_timestamp` returns that remembered picture first (if the file exists); otherwise it falls back to the
  nearest keyframe exactly as before.
Effect: for points from the extractor nothing changes (their own frame is the keyframe anyway); for points added by
interaction fusion the saved `frame.jpg` is now the picture the VLM described. Tests: `tests/orchestrator/test_reference_frame.py`
(5 tests: fallback unchanged; own picture kept while other timestamps are unchanged; no mix-up between videos; a missing
file falls back; frames are remembered before any model loads). The VLM engine, VLM interface, synchronizer, synthesizer and API are
unchanged.

## 5c. The higher-scoring picture wins (fusion)

File: `synapse_interaction/fusion.py`. Before, a gesture picture was always used for points that fusion created, and an
existing point always kept its keyframe. Now the picture with the higher confidence is the one the VLM describes:

| Situation | Candidates | Rule |
|---|---|---|
| An existing point (from the synchronizer) is near an event | the point's keyframe (`clip_confidence`) against the event's picture (`Interaction.confidence`) | the event picture replaces the keyframe only if it scores **higher**; a tie keeps the keyframe |
| Fusion creates a new point | the nearest extractor keyframe **within `pic_window` = 4 s** (`clip_confidence`) against the event's picture | the higher score wins; a tie keeps the keyframe; a keyframe further away than 4 s does not compete |

Details:
- **A keyframe can itself be a gesture frame.** `_is_gesture_frame()` in `fusion.py`:
  - *Pointing (hand or mouse):* a keyframe taken anywhere inside the event (0.25 s tolerance) shows the pointer, so it is the
    gesture frame and keeps its picture without any comparison. The point is still marked as a gesture point.
  - *Writing:* an episode can last minutes and its picture is the finished board just after the last stroke, so only a
    keyframe within 0.25 s of that moment counts as the same picture. A keyframe from the middle of the writing shows a
    half-written board and competes on confidence like any other. (An earlier version treated every keyframe inside a
    writing episode as the gesture frame; the audit showed that would have hidden the finished board.)
- Only `frame_path` can change. The point's timestamp, `transcript_context`, `content_type` and `complexity_score` stay as
  they were (a new point still takes `content_type` and `complexity_score` from the nearest keyframe, as before).
- If several events are near the same point, the best-scoring picture among them wins (the running score is kept per
  point).
- The orchestrator fix in 5b makes the saved reference `frame.jpg` match whichever picture won.
- Caveat to decide on: the two scores come from different sources and are not calibrated against each other.
  `clip_confidence` is the extractor's CLIP score (about 0.3 to 0.6 in the sample data) while an event's confidence is a
  heuristic from 0.45 to 0.95 that grows with the number of detections. In practice an event picture will usually win. If the
  team wants a different balance, change `_gesture_picture_wins()` (one function) or scale one of the scores.
- Tests (`tests/test_pipeline_integration.py`): a keyframe taken during a pointing event keeps its picture even against a
  higher score, for an existing point and for a new one; a mid-writing keyframe does not beat the finished board; a keyframe
  from the moment writing finished is the same picture; gesture wins when higher; keyframe kept when higher and on a tie; a new
  point uses the higher-scoring picture both ways; a keyframe 30 s away does not compete.

## 6. Reverted, so it is not in the diff

- The first version of the orchestrator reference-image change (it was re-done properly, see 5b).
- A first version of the extractor hook (replaced by `tap.py`, then restored in its final form).
- `tap.py` and `tests/test_tap.py` (a run-time wrapper around `stream_frames`; deleted).

## 7. How it was verified, and what that does not prove

- 34 tests in `synapse_interaction/tests` pass (14 original, 3 removed with their tools, 23 added), run in a scratch
  virtual environment with `opencv-python-headless`, `numpy`, `scikit-image` and `pytest`.
- The 73 of 74 API tests that run pass; the one failure is a Windows path-encoding test (`C:` against `C%3A`) that
  fails the same way without these changes.
- **Not verified:** nothing has run on a real lecture. `ffmpeg`, `opencv-python` and `mediapipe` are not installed on
  the development machine's backend environment, so real hand detection, real timing, the extractor hook inside the
  real pipeline, and the speed cost of running the detectors inside the frame loop are untested. The detector
  thresholds were set for 4 fps and not tuned for 2 fps. The pointing rule is unchanged (strict about the fingers,
  yet it fires on any raised index finger).

## 8. Environment switches

| Setting | Effect |
|---|---|
| `SYNAPSE_INTERACTIONS=0` | No interaction detection at all. |
| `SYNAPSE_HAND=0` | Writing / mouse-pointer detection only. |
| `python -m synapse_interaction.get_models` | One-off download of the hand model (needed with current MediaPipe). |

## 9. Committing

`*.md` files are ignored by this repository's `.gitignore`, so add the documents explicitly:
`git add -f docs/synapse_interaction.md docs/synapse_interaction_change_log.md`.
