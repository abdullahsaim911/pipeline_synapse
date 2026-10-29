# Synapse Interaction (gesture / writing / pointer detection)

Status document for the `synapse_interaction` module: what it does, what the audit found, what has been
changed, what is still open, and how to run and test it. Dates are 2026-10-29 unless stated.

See also: [`synapse_interaction_change_log.md`](synapse_interaction_change_log.md) for the file-by-file record of what
was changed and why.

## 1. What the module does

`synapse_interaction` looks at the lecture video and finds moments where the teacher **shows** something
instead of saying it, so those moments can become intervention points for blind learners.

| Event | Meaning | Detected by |
|---|---|---|
| `WRITING_DRAWING` | Pen strokes / drawing that stay on the page | Ink and motion scan (OpenCV, NumPy only) |
| `MOUSE_POINTING` | A cursor moving over content without adding ink | Same scan (small moving blob) |
| `HAND_POINTING` | A person pointing a finger | MediaPipe Hands (optional) |

Output is `interactions.json` per video. `fusion.py` then merges it into the synchronizer's intervention
points: it upgrades the confidence of existing points near an event and adds a few new, visually different
points. It never removes existing points and never edits `transcript_context`.

## 2. Where it sits in the pipeline

```
orchestrator
  ├─ audio thread  → transcription
  └─ video thread  → frame extractor (decodes the lecture once)
                       └─ every decoded frame ──► InteractionStream.feed()   (frame_tap hook)
                     InteractionStream.finish() → interactions.json
synchronizer → fusion.fuse_interactions(points, interactions.json) → VLM → ...
```

Before this work the orchestrator ran interaction detection **after** extraction and opened the video a second
time (and, when OpenCV could not read the stream, downloaded a copy with yt-dlp). Now the video is decoded
once, by the frame extractor, and the interaction module only receives frames.

## 3. Audit findings (before the changes)

Found by reading the code and checking the environment. Items marked *not verified* were not run.

1. **Gesture detection never ran in the pipeline.** `orchestrator.py` built the detector with
   `Config(enable_hand=False)`. HAND_POINTING could not appear in `interactions.json`.
2. **It would have skipped itself anyway.**
   - `mediapipe` was not in `requirements.txt` (only in `synapse_interaction/requirements-interaction.txt`).
   - The backend `.venv` has no `mediapipe`, `opencv-python` or `yt-dlp`. Without `opencv-python` the whole
     package fails to import, and the orchestrator only prints "Interaction detection failed".
   - The two MediaPipe model files (`models/*.task`) were not in the repo, and their paths were relative to the
     working directory.
3. **Representative frames were fetched by seeking** (`CAP_PROP_POS_MSEC`) on a possibly remote stream. On
   H.264/VP9 streams this lands on the previous keyframe, so an event picture could be seconds off. *Not
   verified on real video.*
4. **The pointing rule is strict and also loose.** It needs the index finger extended and the other three curled
   (misses a relaxed point, a finger aimed at the camera, a pen in hand) and it fires on any raised index finger
   (for example someone counting "one"). With two hands only the first is used.
5. **Hand pointing rarely creates a new intervention point.** Hand confidence is `min(0.9, 0.45 + 0.07 × hits)`,
   and `fusion.py` needs confidence ≥ 0.8 to add a point when the speech is not silent or deictic. That takes five
   hits, about 2.5 s of steady pointing at 2 fps.
6. **Wrong suspicion, corrected.** It was suspected that measuring landmarks as x/width, y/height (not square
   units) breaks sideways pointing. A synthetic test showed a straight sideways finger passes either way. The
   correction was still added because it is cheap and correct, but it was not a cause.

Frame extraction (`frame_extraction/stream.py`, the SSIM extractor) was also read. It is not what the
orchestrator uses, but:
- `min_gap_frames = int(1.0 × 1.5)` is 1, so the intended 1.5 s minimum gap is 1 s.
- Keyframe folders are not cleaned between runs: `data/7ukDKVHnac4/keyframes/` contains two different frames with
  the same index, so old and new frames are mixed.
- `data/` with thousands of JPEGs is committed to git.

`pedagogical_extractor_v2.py` (about 1,000 lines, the one the orchestrator uses) has **not** been reviewed beyond
its frame-streaming part.

## 4. What has been done

### 4.1 Frame-fed detection (the main change)
- **`stream.py` (new): `InteractionStream`.** `feed(t, bgr_frame)` per frame, `finish()` to build events, pick
  pictures and write `interactions.json`. `feed` never raises, so a bad frame can never break extraction.
- **`motion.py`: `Scanner`.** The ink/motion scan is now a frame-by-frame class. `scan()` (video file) is a thin
  wrapper over it.
- **`hand.py`:**
  - `HandScanner.push(t, frame)` replaces reading a video. It rate-limits to `hand_fps`.
  - `interactions_from_hits()` is a pure function (testable without MediaPipe).
  - The "is there a person" pre-check (pose model, 16 seeks) was removed. Only the hand model is needed.
  - `index_pointing(lm, aspect=...)` measures in square units.
  - Distances and speed use frame-heights.
- **Event pictures** come from the pushed frames. A small JPEG is spooled to
  `interaction_frames/_spool/` at most once per `spool_interval` (1.0 s), the nearest one is copied per event, and
  the spool is deleted. Roughly 0.5 GB per hour of video, transient.
- **`detector.py`:** `detect_interactions(video)` (CLI/tests) now feeds the same `InteractionStream`, so there is a
  single detection code path.
- **`config.py`:** `hand_model_path` is relative to the package; `pose_model_path` and `hand_probe_frames` removed;
  `spool_interval` added.
- **`get_models.py` (new):** `python -m synapse_interaction.get_models` downloads the hand model. Nothing downloads
  unless it is run. `synapse_interaction/models/` is in `.gitignore`.

### 4.2 Optional hook in the frame extractor (its behaviour is unchanged)
`extract_pedagogical_keyframes(..., frame_tap=None, tap_fps=2.0)` in `frame_extraction/pedagogical_extractor_v2.py`
(about 20 lines: three comment lines of intent, one `if` before and one inside the frame loop):
- **Off by default.** With `frame_tap=None` the code path is the original one: same stream rate, same frames.
- **With a tap:** the stream is decoded once at `tap_fps` (2 fps), every frame goes to the tap, and the extractor keeps
  every `tap_fps / sample_fps`-th frame (every 4th) for its own steps, the same frames with the same timestamps
  (0, 2, 4, ... s) and pixels as without the tap. Tested against the real function, comparing timestamps and pixels.
- Exceptions raised by the tap are swallowed, so the tap cannot break extraction.
- Nothing else in the extractor was changed; `vlm_engine/`, `vlm_interface/`, `synchronizer/` and `synthesizer/` are
  untouched.

### 4.2b No download or streaming inside the module
- `video_io.resolve_stream` accepts only a file that exists on disk; a URL or YouTube id raises a clear error.
- `prepare.py` no longer downloads a video with yt-dlp; it runs the pipeline, whose frame extractor feeds the detector.
- A test fails if any file in the package imports `yt_dlp`, or if any file other than `video_io.py` uses
  `cv2.VideoCapture`.
- The only remaining file-reading code is the CLI/test path (`detect_interactions(file)`), which is not used by the
  pipeline.

### 4.3 Orchestrator
- Creates an `InteractionStream` (unless `SYNAPSE_INTERACTIONS=0`), passes `frame_tap=stream.feed` to the extractor,
  calls `finish()` after extraction. Only the lines that were already the interaction module's (the old
  detect-after-extraction block) were edited.
- Hand detection is on unless `SYNAPSE_HAND=0`; it skips itself with a log line if `mediapipe` or the model is
  missing.
- The old block that re-opened the video and downloaded a copy with yt-dlp was removed.

### 4.4 Tests
`synapse_interaction/tests` now has 34 tests (11 original ones, `test_stream.py`, `test_extractor_tap.py`, `test_pipeline_integration.py`):
- Frames pushed at 2 fps give the same event types and times as the video-file path (writing 4–9 s, pointer
  11–15 s on the synthetic clip).
- Event pictures exist and the spool is removed.
- `feed` never raises on bad input.
- The streaming path never calls `cv2.VideoCapture` (checked by making it raise).
- `index_pointing` accepts every direction on a 16:9 frame.
- `HandScanner` merges hits into one event and rate-limits; one flash and a waving finger are rejected.
- The hand stage skips cleanly without MediaPipe.
- Extractor tap: with and without it the real extractor function receives identical frames (timestamps and pixels);
  the tap sees 2 frames per second; a failing tap cannot break extraction.

The 28 API tests that touch the orchestrator also pass. The 34 tests ran in a scratch virtual environment with
`opencv-python-headless`, `numpy`, `scikit-image` and `pytest`, because the backend `.venv` has no OpenCV.

## 5. How to run

Pipeline: nothing to call. Needs `opencv-python` and `ffmpeg` in the server's environment.

```
# hand pointing (optional)
pip install mediapipe
python -m synapse_interaction.get_models        # downloads synapse_interaction/models/hand_landmarker.task
# switch parts off
SYNAPSE_HAND=0          # no hand pointing
SYNAPSE_INTERACTIONS=0  # no interaction detection at all
```

Command line (a video file already on disk; URLs and YouTube ids are refused, the module never downloads or streams):

```
python -m synapse_interaction.run <video.mp4> --out <dir> [--no-hand] [--debug] [--dump-steps]
pytest synapse_interaction/tests -q
```

## 6. In progress / still open

Not done, in the order I would do them:

1. **Run the real pipeline once** on a demo video (`demo_videos/*.mp4`) with `ffmpeg`, `opencv-python` and
   `mediapipe` installed, and read the log: frames fed, hand hits, events. Nothing has been run end to end yet.
2. **Tune for 2 fps.** The thresholds were set for the older 4 fps scan. The synthetic clip passes, but real
   handwriting and real pointers have not been measured at 2 fps.
3. **Measure hand pointing on real footage.** The rule is unchanged (items 4 and 5 in section 3). Options: loosen the
   curl/extension test, require the fingertip to be over content, handle two hands, and let a short confident point
   create a point.
4. **Review `pedagogical_extractor_v2.py`**  (extraction quality, stale keyframes, redundancy filtering).
5. **Clean up `frame_extraction/stream.py` findings** (1.5 s gap, stale keyframe folder, committed `data/` JPEGs).
6. **Requirements:** done - `mediapipe>=0.10.14` is in `requirements.txt` (section "Synapse Interaction Module"); the model file
   still needs the one-off `get_models` command and `ffmpeg` must be on PATH.
7. **Commit and push.** All changes are uncommitted on `voice-control-e`.

## 7. Known limits

- The motion scan can only see ink and cursor-sized motion. A hovering, motionless cursor is not detected.
- Writing detection assumes a plain page with a dominant background colour (Khan-style); busy slides are labelled
  "busy" or "cut" and skipped.
- At 2 fps a pen stroke can move a long way between frames; very fast writing is merged into one event.
- Hand detection costs a MediaPipe call per sampled frame (about 2 per second) on CPU.
- Event pictures are limited to one per second of video.

## 8. Files

| File | Role |
|---|---|
| `stream.py` | `InteractionStream`: feed frames, finish, write JSON |
| `motion.py` | `Scanner`: ink/motion features per frame |
| `classify.py` | Turns per-frame features into writing / pointer segments |
| `hand.py` | Hand pointing: MediaPipe wrapper, `HandScanner`, `index_pointing`, `interactions_from_hits` |
| `fusion.py`, `novelty.py` | Merge events into intervention points; picture-novelty test |
| `schema.py`, `config.py`, `video_io.py` | Data classes, all thresholds, file-only video helpers |
| `detector.py`, `run.py` | Video-file entry point for the command line and tests |
| `get_models.py` | Downloads the hand model on request |
| `tests/` | Synthetic-video, unit, extractor-tap and pipeline-integration tests |

## 9. Integration fixes and clean-up (second pass)

Checked the path decoded frames → `InteractionStream` → `interactions.json` → `Synchronizer` → `fuse_interactions`
with the real `InterventionPoint` (new `tests/test_pipeline_integration.py`). Fusion itself worked: nearby points
are upgraded, far-away events create new points, and hand-pointing events flow through the same way. Two real
problems in this module were found and fixed:

1. **Absolute picture paths.** Event pictures were stored as absolute paths, unlike keyframes
   (`data\<id>\keyframes\...`, relative to the working directory) and unlike the database column "relative path".
   They are now built from `out_path` as given, e.g. `data/<id>/interaction_frames/000_..jpg`.
2. **Stale pictures.** A re-run left the previous run's pictures in `interaction_frames/`. They are now cleared when a
   new run starts.

Removed as extra (evaluation and demo tooling that the pipeline never imports; still in git history, commit
`c59d8c8`): `before_after.py`, `debug_render.py`, `demo_report.py`, `eval_interventions.py`, `preview.py`,
`review.py`, `showcase.py`, `prepare.py`, the empty `eval_data/` folders, and the three tests that only exercised
them. `run.py` lost its `--debug` option (it used `debug_render.py`).

**Which picture the VLM describes (fusion, `fusion.py`).** The picture with the higher confidence wins: the event's picture
against the point's keyframe (`clip_confidence`), or for a new point against the nearest keyframe within 4 s. A tie keeps the
keyframe, and a keyframe taken while a *pointing* event was happening (so it is itself a gesture frame) always keeps its picture; for
writing, only a keyframe at the moment the writing finished counts as the same picture.
See the change log, section 5c, including the caveat that the two scores are not calibrated against each other.

**Reference image (fixed, orchestrator).** For a point that fusion added, the saved reference `frame.jpg` used to be the
nearest extractor keyframe instead of the picture the VLM described. `orchestrator/orchestrator.py` now remembers each
point's own frame and uses it first; details and tests in the change log, section 5b.

Left alone on purpose: `fusion._decide` returns a "may create a new point" flag that `_fuse` never reads. New
points are decided by ranking, a budget of one per 60 s and a 20 s spacing instead. Deleting the flag or using it
would change which points are created, so it needs a decision from whoever owns the fusion rules.
