# On-demand explanations with voice control

Reviewed on 2026-11-13.

This document records the complete implementation plan after the learner's decisions. It extends the existing explanation conversation and personal-storage work. Where older explanation plans differ, the decisions recorded here govern this feature.

The request-to-generation-to-playback workflow is implemented through the existing workers. Migration `20261113_0011` adds durable session interests, private requests and review progress, manual-target metadata and preparation priority. It was applied to the quiet local database after a verified SQLite backup. Live microphone, remote-provider/media retrieval and NVDA verification remain manual acceptance work.

Implementation owners added during delivery are `api/on_demand.py`, `api/routers/explanation_requests.py`, and `synapse-web/src/lib/lectureRequests.js`. The existing preparation manager, processing manager, explanation service, orchestrator, synthesizer, conversation hook, player component, account lifecycle and voice catalog were extended at their existing ownership boundaries.

## 1. Agreed product behavior

- Admit heavy processing for one video at a time.
- Multiple interested sessions may use that video. Leaving one session preserves work another session needs. After the final interested session leaves or expires, stop unfinished work and fence late results, subject to the sign-out continuation decision below.
- Sign-out preserves already-admitted video processing and explanation jobs. Clear private views, playback and callbacks. Record continuation for the current admitted job generation so a later session-expiry scan cannot cancel it because the learner signed out. Clear this continuation when that generation settles.
- While a user request is deferred because its explanation provider is unavailable, continue runnable background work with prepared evidence. Restore the user's priority when its required stage becomes eligible. Ordinary foreground preparation still holds priority over starting another background explanation.
- Keep learner requests, review, and satisfaction private to the owning profile. Reuse generic evidence and generated content where their source and generation contracts match.
- Persist requests and heard/unheard progress across refresh and cache rebuilding. Retain completed saved versions.
- Saved Library/Bookmark views offer an explicit transition into active processing when a requested depth is missing.
- Permit useful background knowledge and examples. Identify them in both speech and text as additional explanation. Claims about the lecture and its visual content must remain source-grounded.
- A fresh explanation request defaults to Standard. Brief and Detailed are available on explicit request.
- Retain the existing English voice contract, keyboard-accessible controls, automatic encounters, clarity question, reminder behavior, Help, and voice-note guards.

Current personal-storage APIs resolve identity through `api/auth.py`, which supports signed-in local accounts and a local-profile compatibility path. Bind private requests and review to that resolved owner. Preserve local transport, account revocation and origin restrictions. Remote account deployment remains a separate requirement.

## 2. Verified architecture and integration constraints

| Existing owner | Current responsibility | Feature integration |
| --- | --- | --- |
| `api/processing_jobs.py` | Detection, durable video admission, cache replacement, cancellation, recovery, requested lecture-wide modes | Retain one-video admission. Wake waiting user requests after source context commits. Keep targeted depth requests separate from lecture-wide mode scheduling. |
| `api/explanation_preparation.py` | Evidence and synthesis workers, bounded handoff, priority, leases, retries, provider deferral | Reuse for every requested explanation. Restore durable request priority and coordinate cancellation with live session interest. |
| `api/explanation_service.py` | Target loading, cache validation, evidence preparation, artifact persistence | Resolve appropriate evidence kind and publish matching artifacts with source/generation identity. |
| `orchestrator/orchestrator.py` | Visual extraction, reconciliation, synthesis and TTS checkpoints | Retain these stages and add explicit transcript-supported synthesis. |
| `synthesizer/llm_synthesizer.py` | Depth-specific evidence-grounded prompts | Add clearly identified educational enrichment while preserving source claims and uncertainty. |
| `api/models.py`, `api/history_schemas.py`, Alembic migrations | Shared inference data and personal ownership | Add request/session records and manual-target metadata with safe cache-rebuild relationships. |
| `api/personal_storage.py` | Immutable owned snapshots and Library capture | Preserve exact completed versions for requesting owners. Filter automatic subscriber capture by target origin. |
| `api/routers/video.py` | Shared processing and explanation endpoints | Retain existing targeted endpoints and introduce timestamp resolution through the same admission owner. |
| `api/routers/learning_history.py`, personal router and schema modules | Owned learning state | Expose session interests, private requests and review through the established profile dependency and validation conventions. |
| `synapse-web/src/App.jsx` | Navigation and same-lecture conversation lifetime | Register/renew/release session interest. Rehydrate private progress. Coordinate saved-to-active transitions. |
| `synapse-web/src/hooks/useExplanationConversation.js` | Questions, waiting, selected versions and review | Own listening intent and conversation transitions. Persist meaningful progress and reject stale work. |
| `synapse-web/src/components/Explanation.jsx` | Both players, preparation UI, saved playback and voice handlers | Retain exclusive media ownership and expose one request handler to voice and controls. |
| Existing frontend API and personal-storage clients | Requests, mode mapping, saved assets | Carry stable request/target/artifact identities and coordinate status observation. |
| `voice/catalog`, `voice/interpret.py`, `voice/schemas.py` | Allowed actions and contextual English interpretation | Add standalone explanation/depth requests and preserve contextual answers. |
| `synapse-web/src/voice` | Recognition transport, contract validation, dispatch and playback restoration | Capture the full pre-turn context and route actions to the existing conversation owner. |

There are two existing `mode` concepts. Explanation depth maps Brief to `brief`, Standard to backend `explanatory`, and Detailed to `detailed`. Study/accessibility presentation mode remains independent. Saved artifact version is also distinct from depth.

## 3. Request flow

1. At voice recording start, capture session ID, question ID/version, playback timestamp, explicitly selected target if applicable, current explanation depth, and current playback artifact.
2. Confirm that application-controlled audio is quiet before opening the microphone. Close it before feedback or playback.
3. Interpret the utterance through the existing action catalog and validate it against the current screen and session.
4. A fresh request uses the captured playback context. An explicitly selected moment or an explicit moment number uses its resolved stable ID. Ask a short clarification if the referent remains ambiguous.
5. The conversation pauses the lecture and records its return context once. It requests the selected depth, defaulting to Standard.
6. Backend admission returns the durable request ID, resolved or waiting target, selected depth, effective state, retry eligibility, and artifact identity when available.
7. A ready result may play after this explicit request. Queued or preparing work offers Wait or Resume. Failed work offers eligible Retry or Resume.
8. Wait retains permission to play upon readiness while the intent remains current. Resume clears that permission and offers Listen/Later when ready.
9. Natural completion of the current artifact records heard-to-end and asks whether it was clear. Satisfaction, replay, and depth choices follow the existing conversation.

Persisted readiness never grants playback permission after refresh. Re-entry rehydrates progress and offers explicit playback.

### Completion announcement after choosing to listen later

- During preparation, "listen later" means resume the lecture while retaining the request and its completion notification. It does not mark the explanation heard or suppress the ready announcement.
- When that requested target/depth has validated text and servable audio, pause the lecture at a safe interaction boundary and announce, "Your Standard explanation for this moment is ready. Would you like to listen now? Say listen or later." Name the actual depth and moment where needed.
- This announces completion of the requested explanation. Detection finishing or another explanation becoming ready does not satisfy this request's completion condition.
- Listen plays the matching artifact. Later continues the lecture and keeps the item in private unheard review, without repeating the same ready notification. An unanswered question follows the existing reminder behavior.
- Deduplicate the notification for that request/artifact and defer it behind recording, Help, voice-note playback and another active question. Navigation, sign-out and stale sessions cannot resurrect the old prompt.

## 4. Scheduling and priority contract

The request path must join the existing video owner and preparation manager. It must not invoke another orchestrator, rerun video detection for a valid target, or use a direct provider executor.

### Non-preemptive processing boundary

- Finish the explanation currently executing in the synthesis worker, including its TTS and validated publication.
- Let an already executing evidence/provider call return and checkpoint safely.
- Before starting another background explanation, dispatch the learner's eligible request.
- After eligible user work completes, continue background preparation from its checkpoints.
- A request for the same work already executing joins it. It does not create a second attempt.

The evidence and synthesis workers operate independently. Priority applies at each worker boundary. A completed background evidence checkpoint can wait while foreground evidence is prepared. An active synchronous provider cannot be safely suspended in the middle of a call.

### Dispatch order

1. Required unfinished stage for admitted user requests, ordered by the first explicit request time among eligible work.
2. Background work ordered by the existing educational timestamp order.

Repeated requests join the same generation key and preserve its first priority position. A second learner requesting that key adds interest without adding executable work. A depth change creates or joins only that target/depth key.

Persist request ordering so restart reconstruction uses it. Current `_priority_keys` provides membership promotion, while stable user-request ordering and priority restoration require the durable request integration.

### Concrete scheduling cases

| Arrival condition | Required behavior |
| --- | --- |
| Background A is synthesizing and B/C are queued | Finish A through TTS. Run the requested U before B/C. Resume B/C afterward. |
| Evidence for background B is running | Finish/checkpoint that evidence call. Prepare U next. Hold B's synthesis behind eligible U. |
| Synthesis handoff is full | Reuse the existing displacement/checkpoint mechanism to make room for U without losing background leases or evidence. |
| U is already in either queue | Promote its existing entry. |
| U is the explanation currently executing | Join it and return its real state. |
| U is cached | Return the validated artifact without occupying another generation slot. |
| U has a retryable failure | Admit an explicit retry within the existing attempt limit and promote it. |
| U's provider is deferred/unavailable | Return the truthful wait state. Release the active attempt according to existing deferral rules. Dispatch other eligible user work first. Allow runnable checkpointed background work while all foreground work is deferred. Reassert foreground priority when it becomes eligible. |
| U1 and U2 arrive together | Admit/deduplicate transactionally. Preserve durable ordering and finish the executing explanation before the next eligible request. |
| A different video owns capacity | Return the established busy outcome. Retain private user intent for explicit retry. |
| The learner requests during detection/transcription | Persist a waiting-for-context request and join the current video job. Resolve it after valid transcript/target context commits. Drain eligible user requests before normal background dispatch. |
| Detection fails or is cancelled | Transition dependent requests to a recoverable or terminal source state. Never leave them reporting active generation. |

Current detection runs as a blocking pipeline operation and publishes its usable context at completion. The initial implementation waits for that checkpoint. Earlier streaming context publication would require a separate change to detection's persistence contract.

## 5. Durable request and session ownership

Use additive Alembic migrations and the current SQLAlchemy ownership conventions. Proposed records are below. Final field names follow the repository schemas during implementation.

### Session interest

- Server-issued session ID, owner profile, canonical video ID, created time, heartbeat time, expiry and released time.
- Validate that requests belong to an active session for that video.
- Use a 15-second heartbeat and a 90-second expiry as initial fixed lifecycle values. Renew when returning from suspension. Do not infer ownership solely from a browser unload event.
- Coordinate admission, renewal, final-interest cancellation and reopening through the same atomic lifecycle owner. A stale release/expiry must not cancel a newer interested session.
- Distinguish ordinary departure, expiry and sign-out in this lifecycle. Preserve a server-owned continuation decision for the current admitted job generation on sign-out. Account revocation still rejects private API mutations and clears private UI state. Shared workers complete under their job leases, and completion remains attached to the original request owner.
- Retain the existing one-video processing constraint. Logical cancellation does not free physical capacity for a second video's heavy work until tracked in-flight provider work has settled.

### Private request and review record

- Owner, canonical video ID, request ID, idempotency key, resolved context locator, requested depth, first explicit request time and last update.
- An optional mutable intervention reference, with safe nulling/rebinding during cache replacement.
- Source identity, effective request lifecycle and a reference to the current shared preparation where available. Provider stage/attempt state stays with the existing preparation owner.
- Selected completed saved-version reference, heard-to-end, partial/interrupted progress and satisfaction. These are independent fields.
- Request identity and review survive cache deletion. Completed saved versions remain independently readable.
- Store normalized target intent and supported question context only when required. Retain the existing policy against storing raw voice recordings for command history.

### Manual-target metadata and cache identity

- Mark target origin and evidence kind explicitly so manual moments can be filtered from automatic scheduling, public moment lists and general subscriber capture.
- Use a source-context identity to join equivalent manual targets safely. A formatted timestamp alone is insufficient around slide transitions or source changes.
- Add source/generation-contract identity to cache validation and checkpoints. It must distinguish changed input or synthesis rules from an existing compatible artifact.
- Backfill existing targets as detected visual moments and preserve existing mode values. Inspect data before imposing new uniqueness. Keep SQLite/PostgreSQL-compatible constraints and indexes.

An added migration follows the repository's current migration head. Migration verification uses a fresh database and a populated copy. Applying migrations to user data requires the established verified database/media backup procedure.

## 6. Cache rebuild and personal-storage compatibility

`persist_processing_result` can replace the video's mutable cache. New private records must not cascade away with those rows or prevent legitimate replacement.

Before replacement, fence affected preparation attempts. Preserve request locators and completed saved references. After the new detection result commits, resolve unfinished requests against the new source identity. Reuse a result only when its context still matches. Otherwise show the request as requiring preparation or clarification.

Update cache validation for legitimate transcript-supported targets. Missing media for a visual target remains a failure. Preserve the existing validity rules for detected visual moments.

Automatic Library capture must distinguish general detected content from privately requested content. Save a requested completed version for its subscribed requesting owners, while compatible generic artifacts remain reusable. A personal or question-specific answer must never overwrite the generic moment/depth cache.

## 7. Target resolution and evidence

- Validate canonical video IDs, owner/session membership, finite timestamps, available duration, supported depths and idempotency keys at the backend boundary.
- Prefer an explicitly identified matching target. For a timestamp request, use the aligned transcript segment and relevant visual/scene context.
- Reuse an existing moment only when it corresponds to the requested content. The existing nearest-keyframe helper needs relevance checking before use here.
- Resolve media through existing backend media handling. Do not accept arbitrary filesystem paths from the client.
- With useful visual evidence, reuse the current extraction and transcript reconciliation stages.
- With useful transcript and no useful visual, use an explicit transcript-supported clarification path. Reuse Q&A context selection where it fits and retain the preparation/TTS/persistence owner.
- If neither source establishes the learner's referent, return insufficient context and offer clarification or Resume. Outside knowledge cannot establish what an unknown lecture segment meant.

## 8. Explanation generation and artifact consistency

- Standard explains the concept and essential relationships at clear speech-first pacing. Brief retains indispensable facts while reducing optional detail. Detailed adds useful supported structure and clarification.
- Give the source-based explanation first where useful. Identify added background or examples naturally in speech and text, such as “Here is an additional example.”
- Additional material must not be attributed to the teacher or image. Preserve uncertainty, exact mathematical relationships, and the distinction between extracted facts and inferred educational context.
- Use the existing text model and TTS stack. No additional provider or research dependency is needed for this scope.
- Carry source kind, source identity, depth, generation-contract version and artifact identity through synthesis and storage. Internal provenance need not become a verbose spoken disclaimer.
- Generate audio from the exact published text. Validate and atomically publish files with existing checkpoint/lease checks.
- If text succeeds and TTS fails, retain the valid text checkpoint for retry. Do not publish the normal text-plus-audio artifact as READY prematurely.
- Browser audio failure can expose accessible text while preserving truthful backend preparation state.
- Regeneration cannot rewrite an immutable saved version. Display, audio, export, bookmarks and note attachments must use the same selected artifact.

## 9. API integration

Retain existing process, targeted request, retry, prepared-content and status contracts. Introduce owner-scoped session-interest and private-request operations through the established profile API patterns.

Required operations are session registration/renewal/release, explanation request admission by explicit target or timestamp, bounded private request/status listing, meaningful review updates and request retry. Admission returns a durable ID immediately after commit and never waits for inference. Duplicate operation IDs with different payloads produce a conflict.

Effective responses distinguish waiting for source context, queued, preparing, ready, failed, cancelled and unavailable context. Preserve the existing preparation status enum and use a compatible request-level lifecycle rather than changing old clients' meaning silently.

GET/status calls remain observational. Recheck state after uncertain admission/network failure before retrying. Keep sanitised errors, origin restrictions, ownership checks and same-video validation.

## 10. Frontend and voice implementation

- Add a dedicated `request_explanation` action, with an optional requested depth and the supported target-reference shape. Fresh “explain this” defaults to Standard.
- Map natural standalone depth requests through the same owner. Preserve contextual `answer_explanation` for active questions.
- Map "listen later" during preparation to Resume with a pending completion notification. Map Later after the ready offer to deferral of the ready artifact. Validate these meanings against the active question.
- Resolve numbered targets to stable IDs before asynchronous handling. A refreshed list must not change the eventual target of a confirmed command.
- Bind transcript interpretation and execution to the captured session/question. Extend the current pre-turn snapshot beyond playback state so timestamp context cannot drift during recognition.
- Contextualize yes/no, reject ambiguous negation, and support clear correction. “I understand, but replay it” chooses replay. “Brief or Detailed?” followed by “yes” requires clarification.
- Keep help, repeat, location and explicit navigation available. A content question uses Q&A and preserves the pending explanation decision. Distinguish stop speech, pause audio, resume lecture and cancellation of an interaction.
- Register new request/depth actions with the proper playback outcome so a finished voice turn cannot restore the lecture over an explanation request.
- Regenerate `actions.generated.js` using `python -m voice.catalog export-js` and verify schema/catalog/frontend consistency.
- Consolidate preparation observation at the existing lecture/conversation seam. Observe requested work without starting one unbounded polling loop per utterance. Reconnect and finite polling exhaustion must retain recoverable intent.
- Keep concise truthful waiting messages and accessible answer controls in both presentation modes. Reminders suspend during Help, recording, note playback and voice turns. Repeated reminders never steal focus.

## 11. Saved view, review and re-entry

Saved Library/Bookmark reads remain independent of current preparation status. Missing depth offers an explicit switch to active lecture processing. That transition obeys interest registration, busy state, admission and unsaved voice-note guards.

Persist per-owner requests and heard/partial/satisfaction independently. End review includes skipped/partial detected items and requested unheard alternatives in timestamp order. Generated, heard and understood are separate outcomes.

Refresh/reopen rehydrates owned progress, reconciles source/target identities and asks before playback. Same-lecture Timeline navigation preserves the current conversation. Leaving releases session interest and clears local callbacks, timers and playback intents.

Sign-out clears private playback and callbacks while preserving the admitted processing generation. Subsequent sign-in restores only the signed-in owner's requests and review. A previous account's late responses cannot populate the new account's view.

## 12. Concurrency, scale and recovery

- Keep one evidence worker and one explanation worker under the current resource model. Preserve bounded handoff, checkpoints, retry limits, provider backoff, leases and heartbeats.
- Admission/deduplication and cancellation ownership must be transactional. Keep database transactions short and outside provider calls.
- Durable request order and lifecycle state are authoritative. In-memory queues are dispatch aids rebuilt from that state.
- Coalesce repeated requests and bound request/status pages using existing API patterns. Do not expand voice context indiscriminately.
- An API process restart must recover source-waiting and queued user requests before new background dispatch. Existing attempt fencing continues to prevent stale publication.
- Multi-process leases protect target ownership. Cross-process dispatch/resource coordination must be proven before enabling multiple heavy-worker owners. More API workers alone do not establish global priority.
- Keep the current single-video busy behavior. Cross-video queues and simultaneous heavy-video execution are outside the agreed first implementation.

## 13. Implementation order and completion gates

1. Capture current baselines and preserve unrelated working-tree changes. Add focused regression coverage for the observed direct-listen admission and stale-completion bugs.
2. Implement additive owner/session/request metadata and migration compatibility. Prove source-cache replacement preserves private requests and saved artifacts.
3. Integrate atomic session interest, last-interest cancellation, physical capacity release and reopening. Prove concurrent leave/open and expiry races.
4. Implement target resolution and waiting-for-context admission. Persist user ordering and bind targeted requests to existing preparation workers. Keep lecture-wide modes separate.
5. Extend dispatch/recovery priority at both stages. Prove running work completes, user work runs next, displaced background work resumes, deferred work stays recoverable and duplicate requests join.
6. Add transcript-supported synthesis and labelled educational enrichment, source/contract cache validation and consistent text/TTS publication.
7. Wire voice actions, snapshot context, conversation transitions, session registration, status observation and durable progress. Preserve saved-view transitions and accessibility.
8. Run focused integration, migration and build checks. Exercise real microphone, media/providers and NVDA. Report unexercised paths explicitly.

Each step must leave the existing application runnable. Additive endpoints/fields keep existing callers compatible. Do not activate partial private-target behavior through legacy shared-list or cancellation routes before ownership integration is ready.

## 14. Small compatibility fix delivered with this plan

`useExplanationConversation.requestListen` previously entered playback waiting for a non-ready version without calling request admission. That could strand an unrequested version and could not promote already queued work. It also retained old playback identity when another direct request superseded it.

The direct-listen path now reuses READY content, presents existing failed/cancelled recovery choices, and calls the current request endpoint for missing/pending work. It clears the old playback identity and waiting intent before switching. Saved read-only handling retains its existing branch.

Four focused regression cases cover unrequested admission, promotion of requested pending work, failure without automatic retry, and ignoring an old completion after a new request. These initial fixes are now integrated with durable timestamp requests, shared session interests, private progress, standalone voice/depth mapping, completion announcements and transcript-supported synthesis.

Additional delivery checks cover startup dispatch gating, provider deferral, chronological request pages, failed release retry, expired-session reconnection, account-change guards, last-interest cancellation, private Library capture, source-change detection and cache-independent saved progress. Request reconciliation failures are isolated from the original video job's completion and background scheduling.

Session interest uses the proposed 15-second heartbeat and 90-second expiry. If a suspended/offline tab returns after expiry, the frontend offers reconnection through its retry control and the existing retry-explanation voice command. An authenticated logout preserves linked admitted work on the server. Account-transition cleanup runs after successful guards/authentication so a rejected account change cannot detach a live lecture.

The local migration backup is `backups/synapse-pre-migration-20261113T120211397754Z.db.backup`. Existing media was untouched by the additive migration. The missing Argon2 account dependency was installed in the API runtime and pinned in `requirements/runtime-api.lock`.

Final delivery verification passed the full frontend suite with 1,296 tests in 64 files, the 336-test backend/pipeline regression run, and subsequent focused job/lifecycle, account-isolation, priority, recovery and review checks after the final changes. These runs overlap and their counts should not be added. The production build, catalog validation and local schema/model verification passed. Focused frontend lint has zero errors and four hook-dependency warnings. The build retains its Browserslist-age and large-chunk advisories.

Verification for this compatibility patch passed 87 tests across five affected frontend suites, followed by the full frontend suite with 1,269 tests across 61 files. All 47 backend request/lease/pipeline-overlap tests passed. The frontend production build passed. Three new regression cases failed against the previous implementation and passed after the fix. Focused lint reported zero errors and three hook-dependency warnings in existing effects. The build reported outdated Browserslist data and a large-chunk advisory. Real provider, microphone and NVDA checks remain outstanding.

## 15. Acceptance and proof

### Jobs and pipeline

- Block background A, queue B/C, request U and verify execution A, U, B/C after A's complete publication.
- Exercise request arrival during evidence, synthesis, TTS, checkpoint commit and full handoff. Finish the current executing explanation and preserve background checkpoints.
- Promote a key already in each queue, join a running key, coalesce concurrent requests, and preserve eligible user FIFO across restart.
- Admit a request during detection, defer target resolution, then prioritise it after context commits. Propagate source failure/cancellation honestly.
- Exercise provider deferral, exhausted/permanent failure and retry after uncertain network outcomes without duplicate effective publication.
- Prove another video cannot start heavy processing while a logically cancelled provider is still active.

### Database and ownership

- Upgrade clean and populated database copies without loss of existing jobs, personal records or media references.
- Rebuild invalid cache while requests exist. Preserve review and saved artifacts, null/rebind mutable references safely and reject mismatched source reuse.
- Prove private request visibility and targeted subscriber capture under different local profile fixtures.
- Leave one of two interested sessions, expire the last session and race departure with a new registration. Prevent stale cancellation of new work.
- Sign out during detection, evidence preparation and synthesis. Verify admitted work continues through session expiry, private writes still require valid account authorization, completion retains the original request owner, and continuation is cleared when that generation settles.
- Defer a foreground request on provider unavailability while ready background work exists. Verify background progress during the delay and user priority at the next eligible stage after recovery.

### Frontend and voice

- Exercise fresh Standard requests, all depth mappings, explicit target selection, correction, negation, ambiguity and stale recognition.
- Prove Wait permits current-intent playback, Resume requires a later offer, and old completions/readiness cannot change a newer interaction.
- Say "listen later" while preparation is pending. Verify preparation continues, that request's READY event announces completion and offers Listen/Later, and choosing Later then preserves unheard review without another duplicate ready announcement.
- Verify seeking, playback pause/replay, Help, notes, review, refresh and saved-to-active transitions.
- Verify matching text/audio/export/version identity and correct added-explanation labelling.
- Run affected existing suites and the production build. Exercise actual browser/Electron microphone and playback behavior and NVDA in both presentation modes.

Completion requires the entire owned request-to-generation-to-playback path and its recovery behavior. Passing the small compatibility patch's tests establishes only that patch's scope.
