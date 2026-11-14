# App startup and remote model readiness

Verified on 2026-11-14 using the actual PowerShell launcher and API runtime.

## Start and inspect

From the repository root, run the backend supervisor:

```powershell
.\scripts\synapse.ps1 start
```

Inspect the same component states from another terminal:

```powershell
.\scripts\synapse.ps1 status
```

The normal launcher opens the existing log viewer. `-NoLogWindow` suppresses that
window for background/automated startup; it does not disable logging or warmup.

The active frontend is a separate process in `synapse-web/`. `npm run dev` starts Vite;
`npm run electron:dev` starts Vite and Electron. Electron loads the UI and does
not launch Python, load models, or replace the backend supervisor.

## Actual startup sequence

1. Select the existing API, voice and pipeline runtimes, then start Uvicorn.
2. Load `.env` and verify database schema compatibility before starting workers.
3. Start independent background warmups for Qwen VLM and remote Mistral, plus the
   existing voice and transcription warmups. A cold remote GPU does not block
   the backend HTTP server from starting.
4. Each remote warmup probes its own `/health` endpoint, then authenticated
   `/v1/models`. Mistral must serve the pinned Ministral 3 14B Instruct model;
   serving Qwen or another model does not qualify as ready.
5. Publish per-component states through `/ready` and log transitions. The
   supervisor prints “All configured startup models are ready” only after the
   local workers and both configured remote workers are ready or disabled.

The readiness response includes:

```json
{
  "status": "ready",
  "components": {
    "api": "ready",
    "voice": "ready",
    "transcription": "ready",
    "modal": "ready",
    "mistral": "ready"
  }
}
```

`modal` remains the existing Qwen status key for compatibility. `mistral` is
independent, with `warming`, `ready`, `unavailable` or `disabled`. Its process
coordination lock is separate from Qwen's; both can warm simultaneously.

The existing top-level HTTP readiness contract covers the latency-critical local
workers. Therefore check the remote component states too: top-level `ready` alone
does not prove both remote models are warm. The launcher handles this distinction.
The launcher also reads component JSON from HTTP 503 responses while local models
are loading, so those transitions remain visible.

## Configuration and limits

- `SYNAPSE_EXPLANATION_PROVIDER=modal_mistral` selects remote Mistral.
- `SYNAPSE_MISTRAL_MODAL_URL` is its separate deployment URL.
- `SYNAPSE_MODAL_API_KEY` authenticates the existing private model endpoints.
- `SYNAPSE_MISTRAL_WARMUP_ON_STARTUP=0` disables Mistral startup warmup independently.
- `SYNAPSE_MODAL_WARMUP_ON_STARTUP` retains its existing Qwen behavior.

Warmups share the existing bounded timeout/backoff mechanism but maintain separate
status, verification and shutdown state. They make no generation call. Invalid
credentials, a missing endpoint or a wrong served model reports unavailable.

Ready records the last successful probe, not permanent GPU allocation: Modal may
scale the worker to zero after its idle window. A later request can still incur a
cold start. Warm the app before a demonstration. No silent provider fallback or
Gemini explanation generation is introduced.

## Verification

The backend was reloaded through `scripts/synapse.ps1 start -NoLogWindow`. Its
launcher log showed Mistral moving from warming to ready, and `/ready` subsequently
reported all five components ready. The frontend returned HTTP 200.

24 focused startup/readiness/runtime tests pass, covering independent endpoints,
model identity, opt-out, startup/shutdown, and the API status field. This proves
startup behavior, not universal explanation accuracy. That broader quality work
has separate acceptance evidence and remaining limitations.
