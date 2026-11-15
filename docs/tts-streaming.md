# Pocket TTS and Kokoro streaming

Production speech runs inside the Python API environment selected by
`scripts/synapse.ps1`. Pocket TTS 4.0.1 (English, Alba) is primary; Kokoro 0.9.4
(Kokoro-82M v1.0, af_heart) is fallback. Both use CPU. Pocket now loads in a
background startup warm-up after the voice/STT and video-transcription warm-ups
settle. It uses the same shared speech engine and inference gate as requests.
Kokoro remains lazy-loaded as fallback. No startup audio is generated or streamed.

Short voice-command replies use the merged Piper voice, preserving its configured
voice, pace, and WAV cache. Both `/voice/speak` and `/voice/speak/stream` use Piper.
Piper shares the non-preemptive CPU speech gate with explanation TTS: the active
generation finishes, then pending voice replies take priority over explanations.
Streaming converts Piper PCM to the existing mono 24 kHz protocol and bounded
chunks while saving the native WAV; it does not run a second inference.
API model setup provisions Piper as well as Pocket/Kokoro. Existing installations
can install `requirements/runtime-api.lock` and run
`python -m voice.piper_speaker download` before restarting the backend.

Pocket startup loading waits for at least 2 GiB available RAM (and honors the
existing processing headroom guard), retrying every second while memory is low.
`/health` remains responsive. `/ready` reports a `tts` component and stays 503
until the primary is ready, including `waiting_for_local_models` and
`waiting_for_memory` states. The launcher includes TTS in its all-models-ready
message. Existing model inputs, polling intervals and database schema are unchanged.

Both Modal deployments configure a seven-minute idle window (420 seconds).
Redeploy `modal_inference/app.py` and `modal_inference/mistral_app.py` to apply
the change remotely; a local backend restart alone does not update Modal.
Both existing apps were redeployed successfully with this setting on 15 November
2026. Restart the local backend to activate the Pocket startup warm-up.

The frontend is `synapse-web` **inside this Synapse repository**. It is a
separate Git repository. Both working trees contain part of this change.

## Setup and start input as `/voice/speak`.
- Existing endpoints remain available for completed files and cached playback.
- Stream responses are `application/x-ndjson`: `start`, ordered `audio`,
  `complete`, `error`, and `keepalive` events. Explanation streams additionally
  publish `text_ready` immediately after validated LLM text is saved, including
  while it waits for CPU speech. The event identifies the generation, intervention,
  output mode and text. Audio is base64 24 kHz mono signed
  16-bit little-endian PCM. `sequence` and `sample_offset` detect lost/reordered
  chunks. A completed cache hit sends `complete` with `cached: true`.
- The same samples feed a continuous FFmpeg MP3 encoder. Existing ownership
  checks, MP3 validation, atomic publication, provenance and SQL publication
  remain responsible for durable `ready`. Receiving audio never means saved.
- Journal files under `.tts-lab/streams` let late subscribers start from the
  beginning with bounded memory. Each journal is limited to 64 MiB; recent
  terminal attempts are retired on subsequent generation and at shutdown.
  A disconnected subscriber does not cancel shared durable generation.
- Models share one inference lock. API requests and existing heartbeat/status
  threads remain available while background generation runs.
- Pocket failure before publication switches to Kokoro. Once chunks have been
  published, failure terminates delivery and fails the preparation. An explicit
  preparation retry uses Kokoro. This decision is preserved in the existing
  failure-reason field, including across backend restarts.
- Valid text gets its own input/provenance checkpoint so an audio retry does not
  require another explanation-model call. Shared speech caches include the new
  TTS profile; personal saved audio remains untouched.

## Worker scheduling

The existing preparation manager owns three workers: visual evidence, LLM text,
and CPU speech. Each stage processes one intervention at a time. When intervention
1 reaches speech, intervention 2 can generate text while intervention 3 extracts
visual evidence. Both handoff queues hold at most four waiting items; full queues
apply backpressure while saved checkpoints and heartbeat leases remain valid.

Explicit requests take priority over background work at each stage. Background
work follows lecture timestamps. Every production explanation endpoint submits or
joins this same pipeline, including the compatible synchronous endpoint; HTTP
subscribers do not run inference. Foreground work already handed to speech does
not reserve the LLM worker. Queued speech always drains to avoid circular waits.

The shared CPU speech gate completes its active generation, then admits pending
user-facing voice questions before explanation speech. Foreground explanation
speech precedes background speech, with stable order inside each priority.
Pocket and Kokoro still share one inference slot. Playback continues through the
existing conversation flow; text availability alone does not trigger listening.

`text_ready` displays text immediately; `start` configures the audio player and
the first PCM chunk makes live playback available. `complete` is emitted only
after the MP3, provenance, SQL explanation, and durable ready state are committed.
If speech fails, the frontend keeps the text. A retry reuses valid text without
initializing the LLM provider. An interrupted-stream fallback choice survives
retry admission and recovery, and cancellation terminates queued journals.

No schema migration is required. Public statuses and polling intervals remain
unchanged; internal stages include `text_ready`. Deploy backend and `synapse-web`
together because the stream parser must understand the additional event. Existing
model prompts, frame/intervention selection, and transcript alignment are retained.

The frontend retains the existing Listen/question flow and polling. It adds an
AudioWorklet player with an eight-second ring buffer, sample-rate conversion,
pause/resume, listening speed, and account/navigation cancellation. Playback
does not finish on a temporary buffer underrun: it finishes only after a
successful terminal event and all queued samples have played. Existing
conversation satisfaction and lecture-resume rules still apply. Seeking uses
the completed MP3 player after live playback; complete MP3s remain available for
export and saved-content playback. Browsers without streaming support use the
existing completed-file path.

## Verification

Three-worker scheduling verification: 218 backend tests and 15 subtests passed,
including overlap, bounded handoffs, voice priority/promotion, manual joins,
checkpoint retries, restart fallback, queued cancellation, late subscribers,
and completion ordering. Six existing evidence-region cases were deselected
after confirming their documented three-versus-four-region fixture mismatch.
The audio retry expectation now explicitly checks reuse of validated text;
corrupted text has separate regeneration coverage. The offline integration
fixture pins its mocked Ollama provider instead of inheriting `.env`.

The full frontend suite passed 1,354 tests. A subsequent focused run passed
64 tests including an additional check that live audio is offered as streaming,
without announcing the explanation as fully ready. The production build passed;
focused ESLint reported no errors and four existing hook-dependency warnings.
These checks use controlled inference boundaries; no new full lecture with live
remote models or human listening assessment was performed for the scheduling change.

Automated checks cover provider failures before/after publication, retry,
lease loss, invalid samples, MP3 encoding, late subscribers, origin validation,
durable status/persistence, fragmented event parsing, sequence validation,
worklet resampling, pause/drain/completion, and real-mode frontend integration.
The full frontend suite passed 1,334 tests including the streaming protocol and
live-player integration tests; the final account-cancellation change also passed
50 focused frontend tests. The broad backend regression run passed 331 tests plus 15 subtests with
the existing visual-evidence integration fixture suite excluded.

The visual-evidence fixture test at
`tests/api/test_final_pipeline_integration.py:153` expects four regions but its
current source produces three. The same failure was reproduced after loading
the original preparation/orchestrator/cache code from Git HEAD in memory. This
change does not alter the evidence fixture or its expectations.

Real API-environment CPU model measurements (a short two-sentence sample):

| Model/run | First PCM | Full generation | Audio duration | Peak process RSS |
|---|---:|---:|---:|---:|
| Pocket, cold | 2.943 s | 6.584 s | 4.72 s | 1,301 MiB |
| Pocket, warm | 0.109 s | 1.382 s | 4.80 s | 1,276 MiB |
| Kokoro, cold | 9.771 s | 9.869 s | 5.73 s | 1,962 MiB |
| Kokoro, warm | 3.898 s | 3.925 s | 5.73 s | 1,911 MiB |

These are total process RSS peaks, not incremental model-memory requirements.
Kokoro ran after Pocket in the same process. Timing varies with machine load.
The original API Torch 2.6.0+cu124 and Transformers versions were retained.

Real headless Edge consumed the real backend speech stream, paused/resumed the
worklet and drained it successfully: cold first audio about 4.4 s and warm
first audio 184 ms. This verifies streamed playback, not human listening quality
or a full real lecture with remote inference.

Reproduce model measurements with `scripts/verify_tts_models.py` using the API
interpreter; results and MP3s are written to `.tts-lab/reports/production`.
`scripts/verify_tts_browser.cjs` checks actual browser playback against a Vite
server on 5178 configured for a speech API on 8767; it uses the local scratch
Playwright installation and installed Microsoft Edge. Test servers are separate
from the normal backend and should be stopped after testing.
