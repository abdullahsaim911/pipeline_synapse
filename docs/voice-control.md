# Voice control

Hands-free, multilingual control of Synapse for blind and low-vision learners.
This file is updated at the end of every module. Frontend files live in the
`synapse-web` repository; the short frontend note is `docs/voice-control.md`
there.

## Status

| # | Module | Status |
|---|---|---|
| 0 | Readiness and benchmark | Done (see results and limits below) |
| 1 | Contract and action catalog (English) | Done |
| 2 | Audio arbiter and capture (frontend) | Done (behind a flag; see limits below) |
| 3 | Speech recognition service | Built (see limits) |
| 4 | Intent interpretation service | Built (see limits) |
| 5 | Validation and dispatch (frontend) | Built (see limits) |
| 6 | Spoken feedback | Built (see limits) |
| 7 | Questions (reuse `/video/ask`) | Built (see limits) |
| 8 | Evaluation and user testing | Built (see limits) |

Work proceeds one module at a time, each tested and reviewed before the next.

## Decisions

| Topic | Decision |
|---|---|
| Languages | English first. Urdu, Hindi and Roman Urdu/Hindi later, only after recognition and intent accuracy are measured per language. No claim of universal support. |
| Speech recognition | faster-whisper, CPU, int8. `base` first, `tiny` as fallback. Runs locally; audio is not sent anywhere. |
| Intent | Multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) matched against an action catalog. No large LLM for ordinary commands. |
| Hard questions | Existing `POST /video/ask` (see "What already exists"). |
| Spoken feedback | Existing `TTSEngine`. Kept behind one interface so a screen-reader preference can replace it later. |
| Push to talk | **Alt+T toggles** (press to start, press again to stop; nothing to hold). **Space held** on the focused Talk button is the hold-to-talk alternative. No wake word, no always-on microphone. |
| Out of scope | Login, a settings screen, a depth selector, the quiz screen's own controls. |

### Push-to-talk keys (implemented in module 2)

**Alt+T toggles listening.** The app's convention is Alt+letter (Alt+P/L/N/B/R/M/K
on the lecture screen, Alt+S on Home). Alt+T was unused. It is not an Electron
menu mnemonic (those are F/E/V/W/H, which is why Alt+V is not used). Like Alt+S
it works while typing in a text field. Keys with Ctrl, Meta or Shift are
ignored, which also keeps AltGr layouts (reported as Ctrl+Alt) from triggering
it. Alt+Shift is avoided because Windows uses it to switch input language, which
matters for Urdu and Hindi users.

**Space, held, on the focused Talk button** is the hold-to-talk alternative. It
is deliberately not a global key: Space already belongs to buttons, text fields
and screen readers. If listening was started with Alt+T, pressing Space on the
button stops it instead.

Because a toggle can be forgotten, listening also ends by itself (see module 2).

Not verified: behaviour with NVDA, JAWS and Narrator, and on non-US layouts.

## What already exists (verified in the code)

Frontend (`synapse-web`)
- Lecture-screen keyboard shortcuts in `Explanation.jsx` call named handlers:
  `handleLectureToggle`, `togglePlay`, `selectPointManually`, `stepRate`,
  `voiceControlRef.current.toggle()`. Voice reuses these handlers.
- `pauseLectureAudio()` pauses the video and the explanation audio and sets
  `pauseOwner = "user"`, which stops the auto-resume and the 300 ms watcher.
  The voice arbiter reuses it.
- Voice Notes (`useVoiceRecorder.js`) already opens the microphone after pausing
  everything, with a settle delay and a tone. The mic is a shared resource: a
  voice command must not start while a note is recording.
- `AskLecture.jsx` and `askApi.js` already call `POST /video/ask`.
- `useAnnouncer` and the lecture screen's `announce()` speak through
  `aria-live`; the app deliberately avoids speaking the same text twice.
- Electron grants only audio microphone capture (`electron/permissions.cjs`).
- Vitest suite (857 tests) needs **Node 22.12 or newer**; see "Running the tests".

Backend (`pipeline_synapse`)
- `POST /video/ask` answers lecture questions. It uses the Gemini provider by
  default (`llm_interface.get_text_provider`) and needs a Gemini API key. Only
  the question text is sent, never audio.
- `TTSEngine`: Edge-TTS (needs internet), SpeechT5 fallback (English only,
  heavy). It writes mp3 files; there is no "speak this text" endpoint yet.
- `api/security.py`: CORS limited to the configured origins and an
  `enforce_allowed_origin` dependency. Voice endpoints will use it.
- `tests/` holds pytest suites; voice tests follow the same layout.

## Architecture

Six stages, each its own module, with one job:

```
Frontend                                   Backend
1 Capture    hold-to-talk, audio arbiter
2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ 16 kHz mono WAV â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º  Recognition    faster-whisper
3 â—„â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ text + language â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  Interpretation  embeddings + catalog
4 Validation   schema, screen, risk/confirm
5 Dispatch     calls the existing handlers
6 Feedback     â—„â”€â”€â”€â”€â”€ cached TTS audio â”€â”€  /voice/speak    existing TTSEngine
Questions      â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º  existing /video/ask
```

Interpretation is semantic: the utterance is compared by meaning with prototype
sentences per action, and arguments such as "the second one" are matched against
what is on screen. Sentences are data, not patterns. The only deterministic
parsing is for argument values (numbers, directions), never for choosing an action.

### Keeping the voices apart (audio arbiter, module 2)

One arbiter owns the microphone and all assistant speech.

1. On press: stop assistant speech, snapshot what was playing, call
   `pauseLectureAudio()`, confirm both are paused, then open the mic. If a pause
   cannot be confirmed, say so and suggest headphones; do not record over audio.
2. On release: close the mic before anything is spoken.
3. While a voice turn is active, the intervention auto-pause and the
   auto-resume at the end of the explanation audio are suppressed.
4. Afterwards resume only what was playing, and only if the command did not
   change playback (pause, play, replay, seek, navigation all clear the restore).
5. Assistant speech plays only after the mic is closed, so it cannot be recorded.
6. A screen reader's speech cannot be paused by the app; feedback is behind one
   interface so it can be handed to the screen reader via `aria-live`.

## Module 0: readiness and benchmark

### Environment

An isolated, git-ignored `.venv` in the repository root (reuses the installed
CPU torch). Added to it: `faster-whisper`, `sentence-transformers`, `edge-tts`,
`psutil`, `pytest`, `pydantic`, and `av==16.1.0`.

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install faster-whisper sentence-transformers edge-tts psutil pytest "pydantic>=2.10" "av==16.1.0"
```

**Compatibility finding:** faster-whisper 1.2.1 does not pin PyAV, and PyAV 19
breaks its audio decoding (`open() got an unexpected keyword argument
'metadata_errors'`). `av==16.1.0` works. The voice endpoint will avoid the
problem entirely by receiving 16 kHz WAV and passing raw samples to Whisper.

Machine: Intel i5-1135G7 (4 cores / 8 threads), 7.8 GB RAM (about 0.8 GB free
while measuring), no CUDA, Iris Xe graphics.

### Results (CPU, int8, 28 synthetic English clips, two voices)

| Model | Load | RAM added | Peak RAM | Median latency | p95 | Word error rate |
|---|---|---|---|---|---|---|
| Whisper `tiny` | 1.8 s | 98 MB | 192 MB | 0.42 s | 0.49 s | 3.8 % |
| Whisper `base` | 1.8 s | 142 MB | 240 MB | 0.74 s | 0.90 s | 3.1 % |
| MiniLM multilingual (embeddings) | 5.5 s cached (257 s first run, includes download) | 531 MB | 811 MB* | 57 ms per sentence | | |

\* Includes importing torch. The 112-sentence batch took 1.1 s.

Both Whisper sizes met the proposed targets (about 2 s per short clip, under
about 1 GB added). `base` is recommended: slightly better accuracy and still
under a second. `tiny` stays as the fallback if RAM is tight.

Whisper wrote "thirty" as "30", so interpretation must accept digits as well as
words. Both sizes mis-heard the domain word "eigenvectors" ("e-gen vectors",
"EGEN vectors"); module 3 should try Whisper's `initial_prompt`/`hotwords`
with lecture vocabulary.

### Limits of these numbers (do not over-read them)

- The clips are clean synthetic speech from Edge-TTS, so accuracy is optimistic
  compared with a real microphone. Re-run with your own recordings (below).
- Whisper and the embedding model were measured separately. Together they add
  roughly 670 MB on top of Python and torch. This was not measured combined,
  and free RAM was about 0.8 GB, so the models must load lazily and unload when idle.
- Not measured while a lecture-processing job was running (it competes for CPU).
- Only English was tested.

### Re-running

```powershell
# generate clean test clips (needs internet)
.\.venv\Scripts\python.exe -m voice.benchmark make-clips --out <folder>
# time the models; add --runs 3 to take the best of three
.\.venv\Scripts\python.exe -m voice.benchmark run --clips <folder> --whisper tiny base --report report.json
```

For realistic numbers, put your own recordings in a folder with a
`manifest.json` of `[{"file": "a.wav", "text": "open my bookmarks"}]`.

## Module 1: contract and action catalog

### Files

Backend
- `voice/schemas.py` â€” `VoiceContext`, `InterpretRequest`, `InterpretResponse`,
  `TranscribeResponse`. Decisions are `execute`, `confirm`, `clarify`,
  `ask_question`, `unsupported`, `no_speech`, and each decision is checked for
  consistency (for example `execute` must carry an action).
- `voice/catalog/actions.json` â€” 38 language-neutral actions: id, description,
  screens, typed parameters, risk, and the frontend handler it binds to.
- `voice/catalog/examples/en.json` â€” 228 prototype sentences (at least 5 per action).
- `voice/catalog/eval/en.json` â€” 138 held-out sentences with the screen and
  playback context they are spoken in. None may appear in the examples.
- `voice/catalog/catalog.py` â€” loads and validates the data; refuses
  inconsistent data (duplicate sentences, unknown actions, an eval item on a
  screen where its action is unavailable, bad arguments).
- `voice/catalog/__main__.py` â€” `check` and `export-js` commands.
- `tests/voice/` â€” `test_schemas.py`, `test_catalog.py`, `test_benchmark.py`.

Frontend (`synapse-web/src/voice/`)
- `actions.generated.js` â€” generated from the catalog. Never edit by hand.
- `contract.js` â€” validates any backend response before the app may act on it,
  and builds the (bounded) context object sent with each utterance.
- `contract.test.js` â€” 48 tests.

### Action catalog

Binding status of each action: `existing` means the frontend handler is already
there; `wrapper` means a small wrapper around existing state or player calls is
needed (module 5); `voice` means the voice layer handles it itself.

| Area | Actions |
|---|---|
| Navigation | navigate_home, navigate_library, navigate_bookmarks, navigate_notes, open_timeline, start_practice |
| Settings | set_theme |
| Discovery | search_lectures, select_result, start_lecture (confirm), explore_subject, continue_lecture |
| Lecture | play_lecture*, pause_lecture*, seek_lecture*, play_explanation*, pause_explanation*, replay_explanation*, next_moment, previous_moment, go_to_moment, change_listening_speed, toggle_voice_note, toggle_bookmark, export_pdf, set_presentation_mode*, change_text_size*, toggle_high_contrast* |
| Collections | open_item, play_item_audio, remove_item (confirm) |
| Questions | ask_question |
| Voice layer | stop_speaking, repeat_last_response, cancel, help, confirm_yes, confirm_no |

\* needs a small wrapper.

### Regenerating the frontend copy

```powershell
.\.venv\Scripts\python.exe -m voice.catalog export-js --out "<synapse-web>\src\voice\actions.generated.js"
```

### Running the tests

Backend (from the repository root):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/voice -q
# also verify the frontend copy is current:
$env:SYNAPSE_WEB_PATH = "<synapse-web>"
.\.venv\Scripts\python.exe -m pytest tests/voice -q
.\.venv\Scripts\python.exe -m voice.catalog check
```

Frontend (from `synapse-web`):

```powershell
# the frontend's Vitest 5 / jsdom 30 need Node 22.12+; the machine default is Node 20
& "C:\Users\noorf\AppData\Local\nvm\v22.14.0\node.exe" node_modules\vitest\vitest.mjs run src/voice
npx eslint src/voice
```

`npm test` with Node 20 fails to start (jsdom/undici). This is an existing
environment mismatch, not caused by this feature. `npm ci --ignore-scripts` was
used to install the declared dependencies without downloading Electron.

### Results

- Backend: 74 passed (`tests/voice`), including the frontend-sync check.
- Frontend: 48 passed for `src/voice`; the full suite is 857 passed in 35 files.
- ESLint: clean on the new files.

### Limits

- The catalog covers English only. Each new language needs its own `examples`
  and `eval` file; no code changes.
- The held-out set is written by the developer, so it can be easier than real
  speech. Module 4 will report accuracy on it, and module 8 on recorded speech.
- Bindings marked `wrapper` do not exist yet.
- Quiz-screen voice control is out of scope.

## Module 2: audio arbiter and capture (frontend only)

What it does: when the learner presses **Alt+T**, everything Synapse is playing
is paused and confirmed silent, the microphone listens for one short utterance,
and afterwards only the right things are resumed. It records and converts the
speech to a 16 kHz WAV clip and hands it to a callback. Nothing is sent to the
backend yet (module 3), so today the control just reports how many seconds it
recorded.

### How one turn works

1. **Refuse if the microphone is taken.** If a voice note is being recorded (or an
   unsaved note is waiting), voice control does not start and says why.
2. **Pause and prove it.** The lecture screen's existing `pauseLectureAudio()` is
   called through a small "port". The control then waits (up to 1.5 s) until the
   port reports the lecture video and the explanation audio are both silent. If
   they are not, it does not open the microphone, restores what was playing, and
   suggests headphones.
3. **Open the microphone, measure the room.** For 250 ms the microphone is open but
   not recording; the quietest level seen is the room's noise floor.
4. **Tone, then record.** A tone plays (880 Hz), then recording starts. The tone is
   never recorded.
5. **Listening.** Nothing is announced while the microphone is open (it would hear
   the screen reader). If the learner starts the lecture or the explanation audio
   meanwhile, the turn is dropped.
6. **Ends by itself** after 1.4 s of silence once speech was heard, after 6 s with
   no speech, or at 12 s. A lower tone (587 Hz) marks the end.
7. **Close first, then speak.** The microphone is released before any message is
   announced or any audio is restored.
8. **Restore.** Only what was playing is resumed. The lecture screen's pause owner
   is put back exactly as it was, so an explanation that was holding the lecture
   paused still resumes it when it finishes.
9. **Never resume** after a spoken playback command (the callback returns
   `{ playbackCommand }`), after the learner left the lecture screen, or after the
   learner started audio themselves.

### Files (frontend, `synapse-web`)

New
- `src/voice/voiceTurn.js` â€” the rules as pure functions: states, limits,
  `decideRestore`, `isQuiet`.
- `src/voice/wav.js` â€” mix to mono, resample to 16 kHz, encode WAV.
- `src/voice/micCapture.js` â€” microphone, noise-floor measurement, automatic stops.
- `src/voice/useVoiceTurn.js` â€” `VoiceTurnController` (no React) and the thin hook.
- `src/voice/voiceAudio.js` â€” the registry and `useVoiceAudioPort` the lecture
  screen uses to offer pause, check and restore.
- `src/voice/VoiceAudioProvider.jsx`, `src/voice/VoiceControl.jsx` â€” provider and
  the Talk button with the Alt+T and Space handling and one polite live region.
- `src/lib/voiceFlag.js` â€” feature flag (see below).
- Tests: `voiceTurn.test.js`, `wav.test.js`, `micCapture.test.js`,
  `useVoiceTurn.test.js`, `VoiceControl.test.jsx`,
  `src/components/Explanation.voicePort.test.jsx`.

Changed (small, existing files)
- `src/components/Explanation.jsx` â€” registers the audio port; ignores Alt+M while
  voice control is listening.
- `src/components/VoiceNotes.jsx` â€” the control object gains `isBusy()`.
- `src/hooks/useVoiceRecorder.js` â€” `playTone` accepts a pitch (default unchanged).
- `src/components/Navbar.jsx` â€” optional `leading` slot for the Talk button.
- `src/App.jsx` â€” shows the Talk button when the flag is on.
- `src/main.jsx` â€” wraps the app in the registry provider.
- `.env.example`, `src/components/VoiceNotes.test.jsx` (one new test).

### Feature flag

`VITE_VOICE_CONTROL_ENABLED`, read by `src/lib/voiceFlag.js` the same way as the
quiz flag: on in development, **off in a production build and in tests** unless
set to `true` or `1`. The Talk button therefore does not appear in a production
build until the feature is complete.

### Results

- Frontend: 124 new tests (33 rules and WAV, 16 capture, 35 controller, 26
  component, 13 lecture-screen integration, 1 Voice Notes), full suite 981
  passed in 41 files (was 857 in 35).
- Mutation check: five deliberate bugs were introduced one at a time (restoring
  after leaving the screen, not giving the pause owner back, never pausing,
  resuming after a playback command, no guard when audio starts); the tests
  failed 2, 5, 24, 3 and 3 tests respectively, then passed again after reverting.
- ESLint clean on all new files; the existing files have no new findings
  (`Explanation.jsx` keeps its 6 known ones).
- A production `vite build` compiles.
- axe finds no violations on the Talk control, idle or listening.

### Limits (what is NOT verified)

- **Never run with a real microphone, the real YouTube player, Electron, or a
  screen reader.** All of the above uses fakes. The existing Voice Notes feature
  was verified by hand with a real microphone and NVDA, but voice control has not
  been.
- The silence and noise thresholds (`LIMITS` in `voiceTurn.js`) are starting
  values, not tuned on real rooms or real speech.
- `getUserMedia({ audio: true })` and the permission policy are the same as Voice
  Notes, so Electron should allow it, but this was not exercised.
- Decoding the browser's webm to PCM relies on `AudioContext.decodeAudioData`;
  tests replace it.
- Space-hold behaviour with NVDA's browse mode (which may handle Space itself)
  is untested; Alt+T and the button click work without it.
- Alt+T is not yet in the lecture screen's spoken shortcut list (Alt+K) or its
  on-page list; that waits until voice control is switched on for everyone.
- Quiz and Timeline screens have no audio port (nothing of Synapse's plays
  there), so a turn there simply records.

### Running the tests

```powershell
cd <synapse-web>
$node22 = "C:\Users\noorf\AppData\Local\nvm\v22.14.0\node.exe"
& $node22 node_modules\vitest\vitest.mjs run src/voice src/components/Explanation.voicePort.test.jsx src/components/VoiceNotes.test.jsx
& $node22 node_modules\vitest\vitest.mjs run      # whole suite
npx eslint src/voice
```

### Module 2 follow-up: races found in a real browser

Driving real Chrome (fake microphone, real YouTube player) found three bugs the unit tests could not,
all fixed and tested: (1) the player lags an automatic pause, so a pause owner now counts as paused;
(2) a new turn right after a resume could collide, so the next turn waits until the resume is
confirmed; (3) YouTube **ignores a pause while buffering** (about 2 s after a resume), so the quiet check
keeps asking while the player buffers (up to 8 s). Two edge cases remain open: a turn started within about
2 s of the previous one can still give up with "could not pause" (it fails safely), and Space-hold failed once
inside a long scripted sequence although it works on its own.

## Modules 3 to 8: speech, understanding, actions, replies, questions, measurement

Everything below was built in one pass. The loop: **Alt+T -> record -> `POST /voice/transcribe` ->
`POST /voice/interpret` -> validate -> speak the reply -> run the existing handler -> restore audio.**

### Backend (`voice/`)

| File | Job |
|---|---|
| `stt.py` | faster-whisper `base`, CPU int8, loaded on first use and unloaded after 5 idle minutes. Takes the 16 kHz mono WAV (limits: 2 MB, 30 s). Silence never reaches the model, and segments Whisper marks as probably-not-speech are dropped. A lecture-vocabulary prompt helps terms such as "eigenvector". |
| `interpret.py` | Compares the sentence **by meaning** with the catalog's examples (multilingual MiniLM embeddings), restricted to the screen, the actions the frontend can run, and what is playing. Then picks the best match whose values can be found, applies the policy (execute / confirm / clarify / refuse / ask a question), and extracts argument values (numbers, "faster", "night"). |
| `routes.py` | `/voice/transcribe`, `/voice/interpret`, `/voice/speak` (the existing `TTSEngine`, cached on disk), `/voice/warmup`. All use the existing origin check. |
| `evaluate.py` | `python -m voice.evaluate [--verbose|--sweep]` measures intent accuracy on the held-out sentences. |

### Frontend (`src/voice/`)

| File | Job |
|---|---|
| `voiceApi.js` | Client for the four endpoints. |
| `voiceActions.js` | Screens register the handlers they already have (`useVoiceActions`). Nothing is duplicated. |
| `voiceCommands.js` | The handler: transcribe, interpret, validate (`contract.js`), confirm risky actions, run, and report playback commands. |
| `voiceSpeech.js`, `voiceMessages.js` | Spoken replies: the backend TTS, falling back to the screen reader; the preference `synapse_voice_feedback` (`tts` default, or `screenreader`). English text only. |

Screens wired: app-level navigation, theme, subject browsing (`App.jsx`); search, pick a result, start
(`useLectureSearch`, so Home and the subject pages); continue (`Home`); open and remove (`LibraryScreen`,
`BookmarksScreen`); and on the lecture screen play, pause, seek, explanation play/pause/replay, next, previous,
go to moment, speed, voice note, bookmark, PDF, mode, text size, contrast, timeline and practice.
Questions go to the **existing** `/video/ask` with the lecture, the playback time and the current moment.

### Measured results

| Check | Result |
|---|---|
| Intent accuracy, 138 held-out English sentences | 69.6 % run correctly, 24.6 % asked or refused safely, 3.6 % missed, **2.2 % wrong command run** |
| Real speech (13 synthetic clips) through real Whisper `base` and the embeddings | 9 exactly right; 2 safely asked about; 1 recognition error ("study" heard as "steady"); 1 with no matching command |
| Speed on this laptop (4 cores, CPU only) | recognition about 2 s per short clip; interpretation about 50 ms; first load 32 s (the app asks the server to warm up at start) |
| Spoken reply | 2.4 s the first time, 10 ms when cached |
| Tests | backend `tests/voice` 101 passed; frontend 1016 passed (42 files) |

### Limits (read these)

- **English only.** Urdu, Hindi and Roman Urdu/Hindi are **not built or measured**: they need `examples/<lang>.json`
  and `eval/<lang>.json`, then `python -m voice.evaluate` per language. The embedding model is multilingual, but nothing
  non-English was tested, so no claim is made.
- **The accuracy figures are optimistic.** The examples were extended once after seeing the first evaluation, so the held-out set is
  no longer fully unseen. The clips were synthetic speech, not a real microphone.
- **Never run with a real microphone, NVDA or Electron.** The real-browser checks used a fake microphone and were run before modules 3 to 8.
- Whisper can mishear words; a confident wrong transcript can still reach the interpreter. Risky actions always ask first.
- Question answering uses the existing `/video/ask`, which needs a Gemini key (the question text, never the audio, is sent).
  Spoken replies use Edge-TTS (needs internet); if it is unavailable the reply is spoken by the screen reader instead.
- Not wired: `open_item` on the Voice notes screen, the quiz screen, "Settings". Alt+T is not yet in the Alt+K spoken list.
- `tts_engine` needs `nest_asyncio`, which was missing from `requirements.txt` (added). It patches asyncio process-wide.
- First command after starting the server is slow until the models load (about 30 s); the app triggers a warm-up on start.

### Running it

```powershell
# backend (voice models are already cached in this environment)
cd C:\Users\noorf\Documents\pipeline_synapse
.\.venv\Scripts\python.exe -m alembic upgrade head       # first time only
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# frontend, in a second window (mock mode needs no real lecture processing)
cd <synapse-web>; $env:VITE_API_MOCK_MODE = "true"; npm run dev
# tests and measurements
.\.venv\Scripts\python.exe -m pytest tests/voice -q
.\.venv\Scripts\python.exe -m voice.evaluate --verbose
```

Try, with headphones: open the app, press **Alt+T**, say "open my bookmarks", press **Alt+T** again (or pause for a second). Then on a lecture:
"pause", "go forward thirty seconds", "next moment", "bookmark this", "why is the curve flat" (a question), "delete the first one" (asks to confirm).

## Module 9: the remaining screens (Voice notes, Timeline, Quiz)

A screen-by-screen audit found three screens with no voice actions. Now wired:

| Screen | By voice | Reuses |
|---|---|---|
| Voice notes | `open_item` ("open the second note"; numbered in list order) | `onOpenNote` |
| Timeline | `go_to_moment` ("go to moment 3": selects it, as pressing its row does) | row selection |
| Quiz | 10 new actions: `quiz_start`, `quiz_answer` (option 2 / answer B), `quiz_check`, `quiz_hint`, `quiz_read`, `quiz_next`, `quiz_previous`, `quiz_review`, `quiz_submit` (asks yes/no first), `quiz_back`; also text size and contrast | `useQuizSession` methods |

Rules: quiz actions are offered only where the matching button exists (check and hint only in Practice, submit only on the review page, previous not on question 1). Answering only selects; it never checks or submits. Screens with no voice by design: login, settings, depth selector.

Measured: catalog 48 actions, 168 held-out sentences: 68.5% run correctly, 25.6% asked or refused safely, 4.2% missed, 1.8% wrong command run. Backend 103 tests, frontend 1019 tests pass. Not tested with a real microphone. English only.

## Module 10: real-browser fixes and full-app coverage

Found by running a real Chrome with real speech (fake mic fed with a spoken clip), not by unit tests:

- **Slow, silent turns.** The spoken reply was played before the action, so nothing happened for ~10 s. Now the action runs first and the reply follows. The reply is also shown as a caption beside the Talk button.
- **Cold start.** The first command waited for the models to load. The server now loads them in the background at startup.
- **Alt+T lost after clicking the video.** Keys go to the YouTube frame, not the page. Focus is taken back when it moves into a frame, and the shortcut listens in the capture phase.
- **"Play the video"** was a tie with "show the video" (study mode). More examples fixed it.
- **Starting a lecture no longer asks yes/no** (`start_lecture` is now low risk, so "start the lecture" with a pasted link just starts).

New actions (catalog now 58): `paste_url` (clipboard into the link field; needs clipboard permission), `continue_lecture` on every screen ("play my most recent explanation"), "latest/recent" for `open_item`/`play_item_audio`, `skip_explanation`, `retry_explanation`, `set_visual_tab`, `toggle_focus`, `export_audio`, `open_settings`, `close_settings`, `where_am_i`, plus `app_question` answers. 205 held-out sentences: 68.8% run correctly, 24.9% safely asked, 5.4% missed, 1.0% wrong command.

Checked end to end in Chrome with a spoken clip: open bookmarks, paste link, start lecture, play, pause, play most recent, where am I. Not checked with a real microphone.

## Module 11: the full timeline screen

New actions on the timeline: filter_moments (show only graphs / equations / charts / diagrams / code / tables, or all), open_moment (selected, or by number), read_moments, describe_moment. next_moment / previous_moment / go_to_moment now work there and count the list as shown after the filter. If the lecture has none of the kind asked for, it says what it does have. Next/previous direction is settled by the word (previous, back, before), and a number in the sentence favours actions that need one (moment 3). 221 held-out sentences: 74.2% run correctly, 20.8% safely asked, 2.7% missed, 2.3% wrong command run.

