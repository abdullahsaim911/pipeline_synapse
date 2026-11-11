# Task B6 — Comprehensive Production Readiness Verification

**Assessment date:** 2026-11-08  
**Code reviewed:** `bb91e72` (`merged/VC`), including the clean committed B1–B5 baseline  
**Assessment status:** verification complete; production code unchanged  
**Production verdict:** **NO-GO**  
**Controlled live-test verdict:** **CONDITIONAL GO**, subject to the preconditions in this report

## 1. Executive verdict

The B1–B5 lifecycle fixes are present and their deterministic regressions pass. The strongest verified properties are durable lease fencing, single-video admission, no READY state without a valid explanation/audio cache, cancellation fencing, bounded manager shutdown, and exact retry/fallback limits for the mocked providers.

That is not sufficient for production approval. No real YouTube, Modal, Ollama/Mistral, Edge TTS, SpeechT5, configured application database, packaged Electron, or forced operating-system termination was exercised during B6. The supported production process model is also not documented or enforced. A standard executor worker can outlive the processing manager's shutdown deadline, Edge/SpeechT5 calls do not have a proven wall-clock bound, and packaged Electron's `Origin: null` conflicts with endpoints protected by the default origin policy.

The current code is suitable for one carefully controlled live-validation session on a backed-up, single-process Windows installation. It is not yet production-certified.

### Evidence levels used below

- **Code:** static inspection proves the implemented branch/guard, but not behavior against real infrastructure.
- **Deterministic test:** temporary SQLite, temporary files, mocks, controlled events, or isolated subprocesses.
- **Runtime:** an actual local subprocess/import behavior measured during B6. This does **not** mean provider or end-to-end runtime validation.
- **Not verified:** requires a separately approved live session or deployment exercise.

## 2. Verification results

### Test commands and results

The comprehensive backend/provider command covered the B1–B5 suites plus mocked Modal transport, Ollama lifecycle/options, visual-evidence validation, synthesis, TTS and audio delivery:

```text
393 passed, 6 warnings, 15 subtests passed in 87.20s
```

Focused frontend contract tests covered processing polling, reconnect, Cancel/Resume races, explanation status/retry, cached GET behavior and audio/text fallback:

```text
6 test files passed
60 tests passed in 39.14s
```

Additional deterministic probes:

- Mocked Ollama HTTP 500: 3 requests, 2 backoffs, final `OllamaServerError`.
- Mocked Ollama HTTP 400: 1 request and no retry; the raw provider body remained present in the exception (a defect recorded below).
- Controlled snapshot disk-write failure: propagated before Ollama dispatch, with zero final text/audio artifacts.
- Backend module import with `DATABASE_URL=sqlite:///:memory:`: exit code 0.
- Native SpeechT5 import probe: handled access-violation diagnostic reproduced 3/3 times; all imports completed and exited 0.

No test touched the configured application database. No migrations, videos, live HTTP providers, model inference, TTS, deployment or frontend behavior changes were performed.

## 3. REL-001 through REL-011 matrix

| Finding | Original root cause | Implemented evidence | Regression evidence | Remaining limitation | Result |
|---|---|---|---|---|---|
| REL-001 automatic processing recovery | A fresh foreign lease ignored at startup was never reconsidered after expiry. | Periodic recovery owns one maintenance thread and conditionally claims only expired work (`api/processing_jobs.py:499`, `:975`, `:935`). | `test_processing_recovery.py`: fresh-before-expiry, eventual recovery, two-manager winner, old-generation fencing and scan failure recovery. | Actual forced process death was represented by durable rows, not a killed real pipeline process. | Verified by code + deterministic tests |
| REL-002 fresh explanation leases | Startup treated every generating preparation as abandoned. | Startup and maintenance route recovery through stale timestamp/owner/attempt conditions (`api/explanation_preparation.py:1132`, `:1229`). | `test_explanation_leases.py:133`, `:156`, `:181`, `:341`. | Multi-host clock behavior was not tested; current deployment is one Windows host. | Verified by code + deterministic tests |
| REL-003 explanation heartbeat | Stage changes were incorrectly used as liveness. | Tracked leases are renewed independently; owner/status/attempt fence every renewal (`api/explanation_preparation.py:1172`, `:1183`). | Blocked VLM, Mistral and TTS heartbeat tests plus transient/sustained database-error tests (`test_explanation_leases.py:236-341`). | A provider can still consume compute after local ownership is lost; it cannot commit. | Verified by code + deterministic tests |
| REL-004 atomic slot acquisition | Existing-row activation did not consistently translate uniqueness races into a 409. | The unique slot is committed through `commit_active_slot`; the winner is reloaded and different-video conflicts become `ProcessingJobBusyError` (`api/processing_jobs.py:392`). | New row, existing row, cached activation, resume, same-video join and HTTP 409 races (`test_admission.py:162-329`). | SQLite lock errors outside heartbeat paths are not generally retried. | Verified by code + deterministic tests |
| REL-005 processing heartbeat errors | One database exception killed the heartbeat and allowed stale persistence. | Operational errors roll back and retry with bounded backoff; sustained inability to prove ownership fences persistence (`api/processing_jobs.py:1183-1260`). | `test_processing_shutdown.py:178-299`. | Real sustained disk/I/O failure was not induced against the configured database. | Verified by code + deterministic tests |
| REL-006 bounded processing shutdown | `ThreadPoolExecutor.shutdown(wait=True)` could block forever. | Stop is linearized with submit, queued futures are cancelled, and waiting uses one deadline (`api/processing_jobs.py:545-620`). | Blocked/cooperative calls, recovery after expiry, late-result fencing, idempotence and subprocess exit semantics (`test_processing_shutdown.py:301-391`). | The manager returns on time, but a Python non-daemon executor thread can still delay interpreter exit. | Manager verified; process-exit risk remains |
| REL-007 manual global admission | Manual explanation generation bypassed the global video slot. | Manual work acquires/joins the same durable `ProcessingJob` slot before provider dispatch (`api/explanation_preparation.py:524`, `:556`). | Other-video denial before provider call, legacy rows, manual failure/cancel release and concurrent same-video deduplication (`test_admission.py:353-513`). | No live cross-provider resource measurement. | Verified by code + deterministic tests |
| REL-008 immediate slot release | Terminal explanation transitions relied on a later status/submission reconciliation. | One conditional update requires terminal processing plus no pending/generating sibling (`api/processing_jobs.py:416`); ready/fail/cancel/recovery paths invoke it. Resume reasserts acquisition in its update (`:846-885`). | All terminal states, pending/generating retention, running/queued retention, simultaneous terminal transitions, legacy/manual release and resume race (`test_final_lifecycle.py`, `test_admission.py`). | Slot metadata remains denormalized; `_release_completed_slots()` is still needed as repair fallback. | Verified by code + deterministic tests |
| REL-009 preparation shutdown | Bounded joins did not fence still-running daemon workers. | `begin_shutdown()` stops dispatch/heartbeat and marks local leases lost; `shutdown()` shares one deadline and is idempotent (`api/explanation_preparation.py:625-682`). | Blocked VLM/Mistral/TTS, queued work, late result rejection and one-winner post-expiry recovery (`test_final_lifecycle.py:177-315`). | Synchronous calls cannot be force-killed and may run after lifespan shutdown until they return. | Ownership verified; compute lifetime remains bounded only by provider behavior |
| REL-010 Modal warm-up lifecycle | Stop signalled an event but never joined the warm-up thread. | `request_stop()` plus bounded `stop()`; no new probes after stop; filesystem lock released by the owner (`api/model_warmup.py:108-180`). | Idle/repeated stop, blocked probe, no redispatch and eventual lock cleanup (`test_model_warmup.py:168-201`). | A current `urlopen` can outlive the 1-second join grace until its request timeout. | Verified by code + deterministic tests |
| REL-011 TTS classification | Edge retried arbitrary exceptions and diagnostics did not distinguish permanent/transient failures. | Stable retryable/permanent types, two-attempt Edge bound, bounded backoff, partial cleanup, one fallback traversal and sanitized final error (`tts_engine/exceptions.py:8-36`, `tts_engine/providers.py:87-213`, `tts_engine/tts_engine.py:110-141`). | Seven focused classification/call-count tests plus invalid-audio and text-checkpoint suites. | The Edge call and SpeechT5 model loading/generation have no explicit outer wall-clock deadline. | Classification verified; real provider timing not verified |

## 4. Active pipeline and invariants

| Stage | Success contract | Failure/checkpoint contract | Evidence |
|---|---|---|---|
| YouTube admission | A single `ProcessingJob.active_slot=1` globally; same-video calls join. | Different video receives a durable winner in 409; no orchestrator call after denial. | `api/routers/video.py:365`; `api/processing_jobs.py:622`; admission tests |
| ASR/frame extraction/intervention selection | `process_detection_only()` returns transcript entries and intervention points. | Invalid/error result marks the fenced processing attempt failed; no partial DB cache is accepted. | `orchestrator/orchestrator.py:918`; `api/processing_jobs.py:173-227` |
| Database persistence | Video, transcript and interventions commit in one transaction, conditional on generation ownership. | Rollback becomes sanitized `persistence_failed`; old generation cannot commit. | `api/processing_jobs.py:173-227`; processing persistence/recovery tests |
| Modal Qwen3-VL | Response must pass transport, parse and evidence usability validation. | Auth/4xx stop; timeout/selected 5xx retry; invalid/truncated evidence is saved diagnostically but does not proceed. | `vlm_interface/modal_vlm_interface.py:198`, `:391-520`; `api/explanation_service.py:210-271` |
| Evidence checkpoint | Snapshot is atomically replaced under ownership fencing. | Write failure propagates; the deterministic B6 probe proved Ollama was not invoked and no final artifacts appeared. | `orchestrator/orchestrator.py:1357-1370`, `:1411-1542` |
| Reconciliation | Valid visual facts are combined with transcript coverage; cached canonical evidence is not silently rewritten. | Missing/unusable evidence raises before Mistral. | `orchestrator/orchestrator.py:1544-1617`; evidence-delivery tests |
| Local Mistral | CPU defaults are sent as `num_gpu=0`, `num_ctx=2048`; non-empty strict visual explanation is required. | 5xx/connection/timeout receive at most 3 attempts; 4xx stops; failure cannot create an Explanation row or READY state. | `llm_interface/ollama_client.py:128-175`, `:360-423`; Ollama and pipeline-overlap tests |
| Text checkpoint | Generated text is written to `.tmp`, ownership-checked and atomically replaced. | TTS failure leaves valid text reusable; cancellation cannot commit late text. | `orchestrator/orchestrator.py:1620-1680`; evidence/cancellation tests |
| TTS | A generated temporary file must have an ID3 or MPEG frame header before atomic replacement. | Failure preserves text; invalid/late audio is removed and cannot become READY. | `orchestrator/orchestrator.py:1690-1725`; TTS, cancellation and audio tests |
| Explanation cache/READY | DB text plus a servable MP3 inside `data_dir` are both required; final lease owner/attempt must still match. | Missing/corrupt/out-of-root audio makes cache invalid and prevents READY. | `api/explanation_service.py:105-164`; `api/explanation_preparation.py:1082-1106` |
| Frontend playback | UI fetches prepared explanation only after backend status is READY and displays text even if browser playback fails. | Failed preparation exposes retry/skip; no mock explanation is fabricated in real mode. | `synapse-web/src/components/Explanation.jsx:471-533`, `:718-813`, `:1334-1362` |

### Pipeline conclusions

1. No inspected failure branch silently promotes work to READY.
2. Snapshot and text checkpoints can survive later-stage failure and are reused.
3. Invalid evidence does not reach Mistral.
4. Mistral failure cannot produce a misleading cached explanation.
5. TTS failure preserves text but prevents READY until valid audio exists.
6. Cancellation and generation/attempt fencing protect DB rows and final file replacement.
7. Attempts are durable and terminal failures are not automatically retried.
8. Expensive calls are deduplicated while ownership is healthy; stale provider calls can continue consuming compute after losing ownership but cannot commit.
9. One video owns admission globally. Within that video, remote VLM extraction and local synthesis intentionally overlap as two single-consumer lanes (`test_pipeline_overlap.py:193-340`).
10. Legacy cache compatibility is covered for prior snapshots and DB-backed video records. Compatibility with the configured real database was intentionally not tested.

## 5. Fault-injection matrix

All database states below were temporary SQLite states.

| Scenario | Expected transition / invocation count | Observed final state, checkpoint and recovery | Result |
|---|---|---|---|
| Unexpected processing-owner interruption | Fresh `running` remains owned; after expiry exactly one new claim | Seeded abandoned row remained untouched before expiry, then attempt incremented once and completed | Pass; simulated durable interruption, not OS kill |
| Restart before/after lease expiry | No early steal; one recovery after expiry | Both processing and preparation suites observed this behavior without a new request | Pass |
| Two backend managers/processes | One conditional claim and one provider execution | One durable owner; losing manager performed no expensive call | Pass |
| Duplicate same-video submissions | Join one row/future | One orchestrator call; requested modes merged | Pass |
| Manual explanation during other video | 409 before any provider | Provider invocation count 0; other video's slot unchanged | Pass |
| Cancel during VLM | Pending/generating becomes cancelled; late evidence discarded | No Explanation row or final checkpoint from stale owner | Pass |
| Cancel during Mistral | Text/final result fenced | Late generation did not commit or strand resumed attempt | Pass |
| Cancel during TTS | Keep text; remove temporary audio | Text checkpoint present; final and temporary MP3 absent | Pass |
| Cancel immediately followed by Resume | New generation owns attempt; old result rejected | Fresh attempt completed; old response could not overwrite it | Pass |
| Temporary processing heartbeat DB errors | Retry then renew same owner | Lease advanced; provider count remained one | Pass |
| Sustained processing heartbeat DB errors | Mark ownership unsafe; reject late persistence | Old result discarded; row recoverable by a later owner | Pass |
| Temporary/sustained preparation heartbeat errors | Retry transiently; fence after safe timeout | Healthy owner retained for transient error; sustained loss prevented commit | Pass |
| SQLite writer/transaction errors | Roll back failed session before retry | Mocked `OperationalError("locked"/"busy")` paths rolled back and recovered or fenced | Pass for heartbeat paths |
| Modal timeout/5xx | Bounded retry count/backoff | Mocked transport exhausted configured attempts with classified error | Pass |
| Modal auth/invalid JSON/truncation | No retry for permanent auth; invalid evidence never reaches synthesis | Mistral invocation count 0 for unusable evidence; failure metadata retained | Pass |
| Mistral HTTP 500 | 3 total attempts and 2 backoffs, then fail | B6 probe observed exact counts and final `OllamaServerError` | Pass |
| Mistral HTTP 400 | No retry | One call; however raw provider body survived in exception text | Behavioral pass; diagnostic defect B6-06 |
| TTS transient/permanent failures | Retry transient only; permanent skips Edge retry; fallback once | Exact Edge and fallback call counts matched; final errors sanitized | Pass |
| Invalid audio | Never READY | Missing/header-invalid/out-of-root MP3 rejected; text remains readable where available | Pass |
| Disk checkpoint write failure | Stop before downstream provider and final artifacts | B6 controlled failure propagated, Ollama call count 0, text/audio artifact count 0 | Pass for checkpoint path |
| Missing/corrupt cache | Recompute or remain non-ready | Corrupt snapshot not reused; missing MP3 not READY; complete cache skipped providers | Pass |
| Shutdown during blocked processing | Manager returns by grace and late result is fenced | `shutdown()` returned false within bound; durable row later recovered | Pass |
| Shutdown during blocked VLM/Mistral/TTS | Manager returns by grace; old lease cannot commit | Generating row remained recoverable; late result created no Explanation | Pass |
| Simultaneous last terminal preparations | Slot retained until final durable terminal state | First transition returned no release; last transition cleared exactly one slot | Pass |
| Python exit with uninterruptible processing call | Manager deadline must not be mistaken for process exit | Child reported bounded manager return but stayed alive until executor call returned | Pass; confirmed deployment risk |

### Coverage limitations

- No actual `TerminateProcess`, machine restart, power loss, filesystem-full condition, or configured-database lock was induced.
- Processing DB contention was fault-injected primarily at heartbeat/claim boundaries; arbitrary request transactions do not have a universal SQLite retry policy.
- The disk probe covered checkpoint failure, not a full-disk test across every video/frame/transcript writer.

## 6. Confirmed defects and risks found in B6

### P0

No new P0 data-loss or stale-commit defect was confirmed in the isolated verification.

### P1 — production blockers

#### B6-01: bounded manager shutdown is not bounded process exit

`ProcessingJobManager.shutdown()` correctly returns by its deadline, but `ThreadPoolExecutor` uses non-daemon workers. The subprocess test proved the child remained alive after the manager returned (`tests/api/test_processing_shutdown.py:376-391`). A detection/provider call that never returns can therefore prevent clean process termination.

**Minimal corrective action:** define a supervised process-level termination policy and ensure every detection-stage external call has a finite timeout/cooperative cancellation point. Do not force-kill Python threads.

#### B6-02: TTS and fallback loading have no verified wall-clock deadline

Edge `Communicate.save()` is retried a bounded number of times, but no outer timeout encloses an individual attempt (`tts_engine/providers.py:187-213`). SpeechT5 `from_pretrained()` and generation also have no operation deadline (`:232-433`). A blocked call can outlive preparation shutdown indefinitely, even though late commits are fenced.

**Minimal corrective action:** first determine supported timeout/cancellation mechanisms in the installed Edge and Hugging Face versions; then add a provider-level bound or isolate the fallback in a supervised process. Do not add unsafe thread termination.

#### B6-03: no production-safe launcher contract is checked in

Running `python api/main.py` invokes Uvicorn with `reload=True` (`api/main.py:119-127`), and `api.setup` also advertises `--reload`. Reload can overlap lifespans during development and is not a production process model.

**Minimal corrective action:** document and use one Windows command with `--workers 1` and no reload for production-like runs. Keep reload development-only.

#### B6-04: multiple Uvicorn workers are not supported/certified

Database uniqueness and lease fencing handle several races, but each process owns independent queues, executors, maintenance threads and warm-up state. SQLite has serialized writes and no explicit busy timeout in `api/database.py:43-60`. The test suite proves selected races, not arbitrary multi-worker load or graceful orchestration.

**Required deployment rule:** one FastAPI/Uvicorn worker and one application instance per SQLite database. PostgreSQL is a future option only if real multi-process requirements arise.

#### B6-05: packaged Electron origin policy is unresolved

The packaged UI loads from `file://` (`synapse-web/electron.cjs:35-43`) and therefore uses `Origin: null`. The API intentionally rejects `null` by default (`api/security.py:15-27`). Retry, Ask, Quiz and learning-history endpoints use `enforce_allowed_origin`; those features will fail unless deployment explicitly accepts and configures `null`, or the Electron main process proxies the calls.

**Minimal corrective action:** choose and test one packaged transport design. Do not silently add `null` without accepting that sandboxed web content can share that origin value.

### P2 — important hardening/operational gaps

#### B6-06: Ollama 4xx exception retains raw provider response text

`OllamaClient._make_request()` includes `response.text` in `OllamaClientError` (`llm_interface/ollama_client.py:242-247`). The B6 probe confirmed the body survived in the exception. Later traceback logging can expose a provider payload, potentially including prompt-derived content.

**Minimal corrective action:** retain status and a stable code, not the raw body; log only a sanitized classification.

#### B6-07: explanation polling stops automatically after about 132.5 seconds

The frontend performs at most 30 polls with delays capped at five seconds (`synapse-web/src/components/Explanation.jsx:471-524`). A legitimate Modal cold start plus Mistral/TTS can exceed that. The UI then requires “Check again.” No data is lost, but readiness is no longer observed automatically.

**Minimal corrective action:** use a durable, low-frequency poll while the backend remains pending/generating, or clearly document the manual refresh behavior.

#### B6-08: backend restart reconnect is session-limited

Processing reconnect survives a page refresh through `sessionStorage` (`synapse-web/src/App.jsx:312-346`) and tolerates five temporary poll failures (`synapse-web/src/api.js:186-207`). It does not guarantee recovery after a full Electron/browser session closes, and a longer backend outage becomes an error requiring user action.

#### B6-09: native dependency environment is internally inconsistent

`pip check` reports `autoawq 0.2.9 requires triton, which is not installed`. The maintained pipeline uses remote Modal by default, so this did not break the reviewed path, but the environment is not dependency-clean.

#### B6-10: operational health and capacity are shallow

`/health` returns a static success response (`api/main.py:113-116`); it does not prove schema accessibility, disk capacity, queue health, Ollama, or Modal readiness. No disk-space guard, artifact-retention policy, RAM ceiling or production alert threshold was found.

## 7. Windows native access-violation investigation

### Environment

```text
Python:          3.13.6, MSC v.1944, 64-bit AMD64
pyarrow:         24.0.0
pandas:          3.0.2
scikit-learn:    1.8.0
transformers:    5.6.2
numpy:           2.4.4
torch:           2.6.0+cu124
```

### Reproduction and import order

The minimal reproducer was:

```powershell
python -X faulthandler -c "from transformers import SpeechT5Processor; print(1)"
```

Observed in 3/3 isolated runs:

```text
Windows fatal exception: access violation
... pyarrow.__init__.py:71 -> pyarrow.lib ...
1
exit code 0
```

Important distinctions:

- `import pyarrow`, `import pandas`, `import sklearn`, `import transformers`, `import transformers.generation.utils`, `import torch; import pyarrow`, and `import torch; import sklearn` all completed with exit 0 and no diagnostic in the tested order.
- Importing `SpeechT5Processor` first reproduced the diagnostic.
- Importing `SpeechT5HifiGan` or `SpeechT5ForTextToSpeech` before `SpeechT5Processor` avoided it in the tested subprocesses.
- Importing `tts_engine.providers`, which imports `SpeechT5Processor` first (`tts_engine/providers.py:43-49`), reproduced the diagnostic with faulthandler but exited 0.
- The same TTS import without faulthandler exited 0 without visible output.
- Importing `api.main` against an in-memory database exited 0 without the diagnostic; backend module startup does not eagerly import the TTS stack.

### Determination

This is a **handled native access violation reported by faulthandler**, not a terminating process crash in the observed runs. The stack reaches `pyarrow.lib` through Transformers' lazy SpeechT5 processor import, sklearn and pandas. The import-order dependence points to a native wheel/lazy-import interaction, but B6 did not establish which binary owns the first-chance exception. There is no evidence here of environment corruption, and `pip check`'s unrelated AutoAWQ/Triton gap does not explain this stack.

### Risk and remediation

Risk is **P2 for backend startup** because the backend imports cleanly and every observed subprocess exited 0. Risk remains **P1 for unvalidated SpeechT5 fallback use**, because a real fallback model load/generation was prohibited and native behavior has not been exercised under load.

Smallest safe next step:

1. Reproduce in a fresh disposable environment using the project's intended Python version and a recorded dependency set.
2. Compare the same three import orders and real SpeechT5 construction without changing this environment.
3. Check upstream compatibility/release notes for Python 3.13 across Transformers, sklearn, pandas and PyArrow.
4. Only then approve a pin/order workaround. Blind reinstall, upgrade or downgrade is not supported by current evidence.

## 8. Deployment, shutdown and SQLite assessment

### Supported deployment envelope

- Windows, loopback-only FastAPI, one Uvicorn worker, no reload.
- One SQLite database and one application instance.
- Remote Modal VLM and local Ollama/Mistral CPU mode.
- Development Vite origin is supported by default; packaged Electron needs an explicit origin/transport decision.

### Shutdown deadlines

Lifespan starts all stop signals before sequential joins (`api/main.py:61-72`): preparation is fenced, warm-up is signalled, processing gets its grace, preparation gets its grace, then warm-up gets its grace. With defaults, shutdown can consume approximately 5 + 5 + 1 seconds before other cleanup, because these are separate deadlines rather than one global deadline. An outstanding processing executor call may delay actual interpreter exit beyond all three.

### Recovery after forced termination

Code/test evidence shows:

- processing recovery after the 120-second default lease;
- preparation recovery after the 1800-second default lease;
- fresh leases are not stolen;
- cached checkpoints are reused;
- old generations cannot persist.

This was not validated with a force-killed real backend. Operators should expect a preparation interrupted immediately after heartbeat to remain generating for up to 30 minutes before automatic recovery.

### SQLite suitability

SQLite is acceptable for the intended single-user, single-process, one-video workload. Transactions are generally short, foreign keys are enabled, and uniqueness is the concurrency arbiter. It is not certified for multiple Uvicorn workers. No `busy_timeout` is configured, so lock contention can surface immediately; heartbeat paths recover/fence, while arbitrary API transactions may return errors.

PostgreSQL is not required for the supported envelope. Reconsider it only if multi-process/multi-host concurrency becomes a real requirement.

## 9. Provider, performance and resource assessment

### Modal

- Per-request default timeout: 120 seconds.
- Default retry setting produces up to four total attempts with exponential backoff (`vlm_interface/modal_vlm_interface.py:391-456`).
- Startup warm-up total bound: 600 seconds; individual attempt bound: 60 seconds; shutdown join: 1 second (`api/model_warmup.py:31-37`).
- Cold-start time and GPU cost were not measured.

### Ollama/Mistral

- CPU configuration is code/mock verified: `num_gpu=0`, `num_ctx=2048`.
- Generation allows three attempts, each with a 120-second read timeout plus bounded backoff (`llm_interface/ollama_client.py:128-175`, `:360-423`).
- CPU utilization, RAM high-water mark, tokens/second, model residency and actual timeout behavior were not measured.

### VLM/synthesis concurrency and local GPU

- Remote VLM extraction can overlap local Mistral for the preceding intervention; deterministic timing tests verify the overlap and bounded queue.
- Local VLM mode holds `VLM_EXTRACTION_LOCK` and the test proves it does not overlap Mistral on the shared GPU (`test_pipeline_overlap.py:640-658`).
- The supported configuration uses Modal plus CPU Ollama, so Ollama should not contend for CUDA with CLIP. Actual GPU memory was not measured.

### TTS

- Edge attempts are bounded in count, not proven in wall-clock duration.
- SpeechT5 is lazy-loaded once per provider instance and cleaned up, but model download/load/RAM/CPU behavior was not measured.
- The native import diagnostic makes SpeechT5 fallback a required live-test observation.

### Disk and polling

- Text/audio writes use temporary files and atomic replacement; cancellation/failure cleanup is covered.
- Snapshot replacement retries Windows sharing violations, but a final failure can leave its `.tmp` file because `_write_json_atomic()` has no cleanup `finally` (`orchestrator/orchestrator.py:1357-1370`).
- Downloaded video, frames, transcripts, snapshots and explanations have no quota/retention monitor.
- Backend lease scans default to at most every 30 seconds; frontend explanation polling caps at five seconds and stops after 30 polls. No excessive tight polling was found.

## 10. Security and diagnostic observations

Strengths:

- Modal credentials are redacted from configuration and error bodies.
- Structured stage logging excludes prompts, transcripts, images and recognized secret fields (`stage_tracing.py`, `api/logging_setup.py`).
- TTS classifications do not return raw provider payloads.
- Audio delivery constrains files to `data_dir` and validates MP3 headers.
- The API binds to `127.0.0.1` in the checked-in direct launcher.

Gaps:

- Ollama 4xx response bodies survive in exception text (B6-06).
- CORS is not authentication. Any local process can call the API without an Origin header (`api/security.py:8-18`).
- Processing Cancel/Resume POST routes do not use `enforce_allowed_origin` (`api/routers/video.py:446-496`). Job IDs are hard to guess, but origin enforcement is inconsistent.
- Packaged Electron uses `webSecurity: false` (`synapse-web/electron.cjs:30`), expanding renderer consequences if untrusted content executes. No redesign was attempted in B6.
- `.env` and real logs were deliberately not read, so secret-at-rest practices and actual log content remain unverified.

## 11. Frontend/backend contract assessment

| Contract | Assessment |
|---|---|
| Processing status/stage | Matches backend enums and displays stage text; no fabricated percentage. |
| Progress | Honest stage-based progress only. |
| Cancel | Sends one durable request, accepts `cancelling`, continues authoritative polling, and rejects stale running responses. |
| Resume | Reuses the same job ID and retains cancelled UI state if Resume fails. |
| Backend restart | Poll retries five network failures; page refresh reconnects using `sessionStorage`. Full application-session restart is not guaranteed. |
| Explanation readiness | Prepared GET is called only after status READY. |
| Failed explanation | Safe message, Retry/Skip/Resume controls, retry exhaustion disables retry. |
| Audio unavailable | Text remains readable; playback/export errors are accessible. |
| Long preparation | Polling stops after about 132.5 seconds and requires “Check again” (B6-07). |
| Structured View | Not assessed or modified beyond confirming that B6 did not touch it. |

## 12. Controlled live-test checklist — do not execute without approval

### Preconditions for all tests

1. Make and verify a recoverable backup of the configured SQLite database; do not migrate it during this session.
2. Start exactly one Uvicorn worker, without reload, bound to loopback.
3. Confirm `SYNAPSE_OLLAMA_NUM_GPU=0` and `SYNAPSE_OLLAMA_NUM_CTX=2048` or the equivalent defaults.
4. Resolve the packaged Electron origin decision, or use the supported Vite development origin for this session.
5. Confirm sufficient disk space and capture redacted stage logs.
6. Record current process IDs, database job/preparation states and artifact paths without logging credentials or transcript contents.
7. Obtain explicit approval for Modal cost and live TTS/network calls.

### Test 1 — reopen one already-processed video

- **Precondition:** choose an existing video whose DB rows, snapshot, text and MP3 validate.
- **Expected provider calls:** 0 Modal, 0 Mistral, 0 TTS.
- **Expected DB changes:** none beyond harmless timestamps, ideally none.
- **Expected frontend:** interventions and cached explanation load; audio plays.
- **Success:** no provider dispatch in stage logs and cache remains valid.
- **Cost exposure:** none expected.
- **Stop if:** any cache validation unexpectedly schedules paid inference.

### Test 2 — one intervention through VLM → Mistral → TTS

- **Precondition:** one approved intervention with no valid explanation cache; Ollama ready on CPU.
- **Expected provider calls:** normally 1 Modal inference, 1 Mistral HTTP request, up to 2 Edge attempts; SpeechT5 only if existing fallback policy is triggered. Transport failures can increase Modal to 4 and Mistral to 3 total attempts.
- **Expected DB:** preparation `pending → generating → ready`, one durable attempt increment, one valid Explanation row.
- **Expected artifacts:** usable snapshot, text checkpoint, valid MP3; no `.tmp` remains.
- **Expected frontend:** waiting state, then text/audio without manual generation POST.
- **Success:** one logical inference per stage, valid checkpoints and READY only after MP3 validation.
- **Cost exposure:** at least one paid Modal request; exact currency cannot be estimated from repository evidence. Local CPU/RAM and TTS network use are unpriced here.
- **Stop if:** retries amplify unexpectedly, secrets/payloads appear in logs, RAM pressure becomes unsafe, or duplicate inference begins.

### Test 3 — controlled backend restart

- **Precondition:** use the completed video from Test 2; do not restart during a paid call initially.
- **Expected calls:** 0 providers after restart because valid checkpoints exist.
- **Expected DB:** terminal states unchanged; no attempt increment.
- **Expected frontend:** active page reconnects if outage stays inside poll tolerance; page refresh reconnects via session storage.
- **Success:** no duplicate work and cached playback returns.
- **Stop if:** restart requeues terminal work or creates a second owner.

### Test 4 — cancellation and immediate resume

- **Precondition:** explicit approval for possible duplicate paid exposure if cancellation occurs after Modal dispatch; use one intervention only.
- **Expected calls:** cancelled owner may finish its in-flight request but cannot commit; resume creates exactly one fresh attempt. Up to two logical Modal calls may therefore be billed.
- **Expected DB:** unfinished preparation becomes cancelled, then pending/generating under the new attempt; old owner/generation cannot overwrite it.
- **Expected frontend:** `cancelling → cancelled`, one Resume action, then normal status.
- **Success:** no stale artifact commit, temporary audio removed, valid checkpoints reused.
- **Stop if:** old result changes DB/files after resume or more than one fresh owner dispatches.

### Test 5 — second-video admission

- **Precondition:** run while Test 4's first video durably owns the slot.
- **Expected calls for second video:** 0 provider/detection calls.
- **Expected response:** HTTP 409 with the active video/job identifiers.
- **Expected DB:** exactly one `active_slot=1`; no active row for the second video.
- **Expected frontend:** actionable “another video is being processed” message.
- **Cost exposure:** none for the rejected video.
- **Stop if:** any second-video resource-intensive stage starts.

## 13. Minimal corrective actions in severity order

1. **P1:** establish and document the supported one-worker/no-reload Windows launch command.
2. **P1:** define a process-supervision policy for uninterruptible processing workers and prove actual forced-restart recovery.
3. **P1:** add or establish real wall-clock bounds for Edge and SpeechT5 operations without unsafe thread termination.
4. **P1:** resolve and test packaged Electron's origin/transport contract.
5. **P1 validation:** run the approved five-step live checklist and record real provider counts, latency, CPU/RAM/GPU and cost.
6. **P2:** sanitize Ollama 4xx exceptions so response bodies cannot enter tracebacks.
7. **P2:** decide whether explanation polling should remain active beyond 132.5 seconds.
8. **P2:** reproduce the SpeechT5/PyArrow diagnostic in a fresh disposable environment and approve an evidence-based version/import-order remedy.
9. **P2:** add snapshot temporary-file cleanup, disk-capacity checks and basic retention guidance.
10. **P2:** add a readiness endpoint/diagnostic that distinguishes process health, schema access and optional provider readiness without causing inference.

## 14. Final go/no-go decisions

### Controlled live testing: CONDITIONAL GO

The isolated evidence is strong enough for one deliberately bounded live-validation session after backup, single-process/no-reload startup, origin configuration and cost approval. The session must follow the stopping conditions above and start with cache reuse before any paid call.

### Production deployment: NO-GO

Production deployment is blocked by the unresolved process-exit bound, unbounded TTS/fallback operations, absent production launcher contract, packaged Electron origin mismatch, unclean/native-warning dependency environment, and lack of controlled live provider evidence and measured resource behavior.

No B6 production fix was applied. Review and explicit approval are required before any corrective implementation or live inference.
