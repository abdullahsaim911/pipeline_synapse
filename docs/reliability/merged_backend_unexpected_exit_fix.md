# Merged Backend Unexpected Exit Fix

**Date:** 2026-11-09  
**Verdict:** **FIXED — RUNTIME VERIFIED**

## Revision and worktree protection

- Backend branch/revision: `merged/VC` at `bb91e72`.
- Frontend branch/revision: `merged/VC` at `2f87de0`.
- The frontend worktree was clean and was not changed.
- Pre-existing backend changes were preserved: modified `.gitignore`, untracked `.claude/`, and untracked reliability documents. This task added only the narrow `.gitignore` exception required to expose this report.
- No configured database, user data, video, Modal inference, Ollama, Mistral, or live TTS provider was used.

## Components inspected

The investigation traced `api/main.py`, `api/model_warmup.py`, `api/processing_jobs.py`, `api/explanation_preparation.py`, `voice/lifecycle.py`, `voice/routes.py`, `voice/stt.py`, `voice/interpret.py`, the voice catalog and schemas, database migration/startup behavior, and the related voice and B1–B5 tests.

The output after FastAPI startup was identified as Voice Control initialization:

- `Loading weights: 199/199`: `sentence-transformers` loading `paraphrase-multilingual-MiniLM-L12-v2`.
- `Batches: 8/8`: `Interpreter._build()` encoding the action-catalog examples.
- The initialization was invoked by `VoiceWarmup` in the `synapse-voice-warmup` daemon thread after FastAPI reported ready.

The warm-up thread did not block the asyncio event loop. It did, however, execute native sentence-transformers/PyTorch/PyArrow and faster-whisper/CTranslate2 initialization outside the process main thread.

## Exact failure reproduction

All application runs used a freshly migrated temporary SQLite database and these safety settings:

```powershell
$env:DATABASE_URL = "sqlite:///C:/.../temporary/diagnostic.db"
$env:SYNAPSE_MODAL_WARMUP_ON_STARTUP = "0"
$env:SYNAPSE_VOICE_WARMUP_ENABLED = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -X faulthandler -m uvicorn api.main:app --host 127.0.0.1 --port 8765
```

Before the fix, the isolated backend:

- completed FastAPI startup;
- initially served `/openapi.json` successfully;
- stopped responding while Voice Control native initialization ran;
- exited after 112.9 seconds without lifespan shutdown logs;
- returned unsigned Windows status `3221225477`, hexadecimal `0xC0000005` (access violation).

No matching Application Error/WER event was available in the Windows Application log for the captured PID/time.

The component matrix established:

| Probe | Result |
|---|---|
| Intent model only, process main thread | Exit `0` |
| STT model only, process main thread | Exit `0` |
| Intent then STT, retained for 120 seconds on main thread | Exit `0` |
| Original STT-then-intent order | Fatal `0xC0000005` in `pyarrow.dataset` import |
| Reordered models in managed warm-up thread | Fatal `0xC0000005` after model initialization |
| `VoiceWarmup` thread alone, without FastAPI | Fatal `0xC0000005` at about 30 seconds |

This proves the terminating failure was caused by native Voice Control model initialization/retention in a Python background thread. Reordering imports reduced one immediate failure mode but did not make the thread safe, so import reordering alone was rejected as an incomplete fix.

## Root cause and relation to the native diagnostic

The merged Voice Control warm-up owned a Python thread correctly at the Python lifecycle level, but the native dependency combination was not safe in that execution context on this Windows/Python environment:

- Python `3.13.6` AMD64
- `sentence-transformers 5.3.0`
- `faster-whisper 1.2.1`
- `ctranslate2 4.7.1`
- `pyarrow 24.0.0`
- `numpy 2.4.4`
- `scikit-learn 1.8.0`
- `transformers 5.6.2`
- `torch 2.6.0+cu124`

`0xC0000005` is process-fatal and cannot be contained by a Python `try/except`. The permanent containment boundary therefore has to be a process, not another thread.

The previously observed PyArrow/Transformers diagnostic is related at the native-library level but is not identical to the backend termination. The standalone probe:

```powershell
python -X faulthandler -c "from transformers import SpeechT5Processor; print('IMPORT_OK')"
```

still prints a first-chance access-violation traceback through PyArrow/Pandas/scikit-learn/Transformers, then prints `IMPORT_OK` and exits `0` (`0x00000000`). The original Voice Control path instead terminated the Uvicorn process with `0xC0000005`. No dependency versions were changed because the standalone diagnostic is handled and no narrowly verified replacement version was established.

## Fix

`voice/process_runtime.py` now provides one owned, restartable subprocess for the two native Voice Control models.

- Native models are initialized and used on the worker process main thread.
- Intent and STT share one persistent worker, so models are reused rather than loaded per request.
- Parent/worker communication uses private framed stdio IPC; no network listener or external service was added.
- The existing `/voice/transcribe`, `/voice/interpret`, and `/voice/warmup` contracts remain unchanged through small proxy objects.
- The action catalog remains available in the API process for existing validation.
- A crashed or timed-out worker is discarded; the next request can start a fresh worker.
- Child exception details are not exposed to clients or logs. Diagnostics include only the operation and worker exit status.
- The IPC reader thread is owned, joined on cleanup, and only one worker can be created.
- Shutdown can preempt an in-flight warm-up/request, terminates only the owned worker if the grace expires, and remains bounded.
- The safe intent-before-STT warm-up order is retained inside the isolated worker.

`voice/lifecycle.py` starts the worker before its background warm-up thread, reports worker startup failure as `unavailable`, and records successful readiness. `voice/routes.py` now obtains the process-backed recognizer/interpreter proxies and always stops the worker during application shutdown, even if a request still owns the voice turn gate.

The process boundary is deliberately limited to STT and intent models. Voice speech/TTS caching and the existing TTSEngine remain unchanged.

## Files changed

- `.gitignore` — expose this required report; preserves the pre-existing B6 exception.
- `voice/process_runtime.py` — new owned native-model worker and compatible proxies.
- `voice/routes.py` — use the shared worker, safe warm-up order, and bounded worker cleanup.
- `voice/lifecycle.py` — start/observe the worker and retain existing single warm-up ownership.
- `tests/voice/test_process_runtime.py` — mocked IPC, diagnostic-redaction, and forced-stop tests.
- `tests/voice/test_lifecycle.py` — worker startup success/failure lifecycle coverage.
- `tests/voice/test_voice_backend.py` — warm-up ordering coverage for background and HTTP entry points.
- `docs/reliability/merged_backend_unexpected_exit_fix.md` — this report.

No frontend, schema, migration, pipeline, model prompt, Modal, Ollama/Mistral, TTS provider, cancellation, recovery, or B1–B5 manager file was changed.

## Runtime verification

The final test used the real already-cached voice models, offline mode, a temporary migrated SQLite database, Modal warm-up disabled, and port `8765`.

- FastAPI startup completed.
- `/openapi.json` returned HTTP 200 throughout model preparation.
- `/voice/warmup` returned 429 while the single startup worker owned the gate, then HTTP 204 when both models were ready.
- Repeated `/voice/warmup` calls returned 204, proving initialized resources were reused.
- One bounded observation recorded 48 successful OpenAPI responses out of 49 attempts; the one miss occurred before the listener was ready.
- The first run reached voice readiness at 30.45 seconds and stayed healthy through 100 seconds.
- The programmatic graceful run reached readiness at 46.90 seconds, served 34 successful OpenAPI requests, remained healthy for another 20 seconds, and then shut down normally.
- Shutdown logs included `Application shutdown complete`; backend shutdown took 1.07 seconds and the process exited `0`.
- No unexpected Uvicorn exit occurred after model initialization.

Warm-up duration varied with local memory pressure. During diagnosis, the model worker reached about 1.26 GB working set and the machine had as little as 448 MB available. The single-worker design bounds duplication but does not change the model dependencies' intrinsic memory requirement.

## Tests

Focused Voice Control suite:

```text
100 passed, 1 skipped in 2.54s
```

B1–B5 regression selection covering explanation leases, processing recovery, admission, processing jobs/shutdown, cancellation/resume, explanation preparation, final lifecycle, Modal warm-up, pipeline overlap, and explanation audio/TTS checkpoints:

```text
131 passed, 15 subtests passed in 84.80s
```

The regression process emitted the known handled PyArrow first-chance diagnostic and still exited `0`; it did not terminate pytest.

## Remaining limitations

- The standalone PyArrow/Transformers first-chance access-violation diagnostic remains. It is separated from Uvicorn by the Voice Control process boundary, but should be addressed as a dependency-compatibility task only after a compatible version matrix is proven.
- Real microphone transcription content was not exercised; STT model initialization and the existing mocked request-contract tests were exercised.
- No live provider, configured user database, video pipeline, or end-to-end frontend workflow was run.
- Warm-up time depends on host memory pressure; failure remains observable/recoverable and no longer terminates Uvicorn.

## Safe local start

Apply migrations as usual, then start without reload:

```powershell
.\synapse-env\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

For a safe diagnostic run, point `DATABASE_URL` to a freshly migrated disposable SQLite file, set `SYNAPSE_MODAL_WARMUP_ON_STARTUP=0`, and set `HF_HUB_OFFLINE=1` plus `TRANSFORMERS_OFFLINE=1` before starting Uvicorn.

## Readiness verdict

**FIXED — RUNTIME VERIFIED**

The formerly fatal model warm-up completed with real cached models, HTTP remained responsive, the process stayed alive after readiness, and graceful shutdown completed with exit code 0.
