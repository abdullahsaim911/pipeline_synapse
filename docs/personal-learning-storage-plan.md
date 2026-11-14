# Personal learning storage: implementation plan

Date: 2026-11-13. Status: local-backend implementation completed; real microphone, NVDA, and installed-package migration verification remain pending.

## Scope and agreed decisions

- The active frontend is `synapse-web`, not `frontend`.
- Phase 1 uses the existing backend on this computer, SQLite, managed files, and the backend-controlled local development profile. It does not introduce remote accounts or cross-device sync.
- Permanent voice notes move to backend files plus database metadata. IndexedDB remains a recoverable draft/upload store until backend confirmation.
- Library is grouped by lecture, with all its successfully processed explanations available. Recent explanations track actual viewing separately. Bookmarks are explicit only.
- Opening a saved explanation seeks to its time and plays its audio while the video stays paused. Opening a lecture or a voice note's lecture position seeks and stays paused. Retain an explicit Read text action.
- Saving/playing a voice note never automatically resumes the lecture. Only one Synapse media source plays at once.
- Match voice-note context to ready explanations from the same lecture within an inclusive 60 seconds before/after its recording-start timestamp. A note with no matches still saves. This operation never starts inference.
- Preserve the exact saved text, mode, audio, frame, and available presentation context. Saved content must not change after regeneration.
- Every new user operation has a keyboard/button path and a voice mapping using the same handler. Follow existing screen-reader focus and announcement conventions.
- Existing browser saves are imported explicitly, with original data retained until successful verification.
- Phase 2 adds real online accounts. The user wants both Google and email/password sign-in. Supabase Auth is recommended, not yet connected or configured.

## Evidence and conflicts to resolve

| Evidence | Conflict | Resolution |
| --- | --- | --- |
| `synapse-web/src/storage.js`; Explanation bookmark/library IDs use video + point | Different explanation depths overwrite the same local entry | Use canonical mode plus saved-version identity. Import surviving legacy entries without claiming overwritten versions can be recovered. |
| `api/explanation_service.py` updates a cache row; orchestrator writes mode-specific paths | An ID or path can resolve to newer content | Publish a durable saved version with immutable text/context and immutable media keys. Do not duplicate media for each personal reference. |
| `synapse-web/src/api.js` replaces audio paths with blob URLs for playback | Blob URLs expire after restart | Persist only server storage/version IDs and resource URLs. Create/revoke blob URLs at playback time. |
| `GenerateExplanationResponse` lacks explanation ID | Frontend cannot reliably identify the exact backend explanation viewed | Add stable explanation/version identity to the API and retain it through frontend parsing. Do not infer identity from formatted text. |
| App sends `/video/{id}/explanations/open` and `/cancel` on navigation | Reopening saved content can resume Standard preparation; navigation and generation are coupled | A saved-content open must have a read-only route. Generate/resume only through the existing explicit processing workflow. Retain current single-window cancellation for that workflow. |
| `getPreparedExplanation` is gated on preparation status | A valid preserved save could be hidden by a failed/cancelled current preparation | Saved-version reads use their own asset availability; background preparation status is independent. |
| Voice note opens derive mode from the first Library entry | Multiple depths or removing Library can change how a note opens | Save context/version references on the note. Its recording timestamp remains authoritative. |
| `useVoiceRecorder` currently keeps a failed draft in memory | A reload during failed upload can lose the recording | Persist the draft before upload, keep a stable upload operation ID, and remove it only after backend confirmation. Recovered drafts do not upload silently. |
| Recorder has a five-minute product limit; storage allows a larger sanity bound | Server might accept recordings the UI would reject | Enforce the product limit, bounded bytes, actual decodable type/duration, finite timestamps, and same-lecture context on the backend. Legacy migration may report invalid historical notes rather than truncate them. |
| Files and SQL cannot commit as one transaction | Crash can leave an orphan file or a row with no file | Stage and validate files, publish immutable files, commit ready rows, recover uncommitted files. Queue deletion durably and retry after failure. Never report successful save before the durable record is ready. |
| Upload response can be lost after commit | Retry can save two copies | User-scoped idempotency key plus media hash. Same operation/payload returns the existing result; a different payload conflicts. |
| Existing history expires after 30 days | Expiring a recent entry must not destroy a permanent save | Separate history retention from Library, bookmark, and note attachment retention. No cascade from history deletion to saved assets. |
| Shared legacy `Intervention.is_bookmarked` | It cannot represent private bookmark ownership | New personal bookmark relation. Keep the legacy field compatibility-only. |
| Library/Bookmarks are synchronous local reads/writes | Switching directly to fetch could show false empty states or false save success | Explicit loading, ready, empty, saving, and error states. Confirm mutations before updating saved state, or roll back reliably. Handle aborts and late responses. |
| Async text can outlive a moment/mode switch | Save effect could combine previous text with a newly selected point | Carry video, intervention, mode, and version in the loaded result. Save/announce only when they match the active request. |
| `useVoiceActions` uses current visible positions; pending confirmation stores arguments | A refreshed list can change what item N means before confirmation | Resolve to a stable item ID when requesting confirmation. Validate screen and target again before executing. |
| Voice context includes at most 60 items, while collection indices allow up to 500 | Later visible items lack voice context | Use one bounded page of at most 60 items, stable numbering for the visible page, and voice/keyboard next/previous-page actions. Do not merely enlarge prompts. |
| `play_item_audio` is absent from playback/leaving outcome sets | Command acknowledgement or restoration can overlap playback it just started | Classify all new playback/navigation actions correctly and await announcements before playback. Test both command acknowledgement and restoration. |
| Global Notes screen registers opening but not note play/delete | Visible controls do not have full voice parity | Add shared handlers for play/stop, open time, delete/cancel/confirm; expose draft save/retry/discard only when available. |
| Audio registry assumes non-lecture screens have no audio | Voice command capture can hear a recording played on Notes | Register note playback on the audio registry; pause/stop before command capture and prevent inappropriate restoration. |
| Lecture conversation asks permission to listen and asks clarity questions | Saved-content autoplay can compete with conversation prompts | Saved-play entry counts as explicit intent to listen. Coordinate using the existing conversation and pause-owner logic; no duplicate listen prompt. Explicit resume remains required after saved playback. |
| Local storage/privacy text refers to browser-only data | Moving storage to a backend, especially a configurable remote base URL, changes privacy claims | Phase 1 note upload requires a loopback backend. State that recordings are saved by Synapse on this computer. Remote note upload belongs to the later authenticated phase. |
| Local profile is not authentication | Exposing local `/me` endpoints remotely would expose private data | Preserve local-development environment restrictions and origin checks; verify loopback transport. Do not describe phase 1 as secure multi-user accounts. |
| Electron uses `file://`, `webSecurity: false`; backend rejects `Origin: null` by default | Packaged note uploads can fail; simply allowing null origins weakens browser isolation | Verify a trusted packaged transport/origin while retaining YouTube playback and microphone policy. Do not globally allow null or wildcard origins. This is a packaged-release gate. |
| Existing theme names/settings differ from backend preferences | Blindly syncing every preference could reset appearance/accessibility settings | Keep preference/quiz/Q&A storage outside this feature. Do not expand to general settings sync. |
| Database backup alone omits MP3s/recordings | Restoring SQLite can leave broken recordings | Document and exercise a coordinated database + managed-media backup/restore procedure. |

The local-backend resolutions are implemented. Online-account prerequisites remain phase 2. Automated checks and the remaining manual verification boundaries are described in `synapse-web/docs/PERSONAL_STORAGE.md`.

## Data ownership and schema

Reuse `User`, `Video`, `Intervention`, `Explanation`, `LibraryItem`, `LectureActivity`, `PlaybackProgress`, and `RecentExplanation`.

Add only the relations required by this feature:

- Saved explanation versions: stable identity, source explanation, video/intervention identity, canonical mode, full text, frame/transcript/presentation snapshot, generation time, and durable media references. Snapshot persistence must not depend on future existence of a mutable cache row.
- Personal saved explanation references: local profile, lecture/library membership, version, first saved and last viewed times. Keep recent viewing distinct from durable membership.
- Bookmarks: owner, saved explanation version, creation time. At most one bookmark per owner/version; saving is idempotent.
- Voice notes: owner, video ID, exact lecture seconds, optional label, audio storage key, actual MIME/size/duration/hash, recorded/imported time, saved time, and any existing silent-warning flag.
- Voice-note explanation links: note and saved version, unique pair. Link ready versions at note save time, without automatic inference or silent later relinking.
- Durable media deletion records: storage key and enough lifecycle state to retry deletion. Files referenced elsewhere must not be removed.
- Import operation records where needed for retry/deduplication; do not mutate or silently erase the original browser data.

Use Alembic migrations with named constraints, user-first indexes, same-video validation, and SQLite/PostgreSQL-compatible types. Inspect duplicates and validate existing data before imposing new uniqueness. No automatic downgrade that discards new recordings; rollback should preserve tables/files for recovery.

Store canonical backend modes (`brief`, `explanatory`, `detailed`). Map UI labels/aliases at the API boundary. Never confuse depth with study/accessibility presentation mode.

## Media ownership

Use the existing backend data root. Personal recordings use server-generated paths beneath `users/{user_id}/voice_notes/{note_id}/`. Preserve the actual codec/container instead of renaming WebM to MP3.

Saved explanation assets use immutable version paths beneath a saved-content area. Hashes verify identity/integrity. Do not make a new copy for every bookmark/attachment.

Resolve keys inside the managed root, reject traversal/symlinks escaping it, reject arbitrary client paths, and stream files through owner-scoped endpoints. Do not put private recordings on a public static mount. Playback should support the browser's required media requests.

Cancellation/failed generation must not publish partial assets as ready. Ordinary removal of a library reference must not delete a bookmark, voice note, or shared generation cache. Personal media deletion requires confirmation and coordinated metadata/file cleanup.

## API and frontend implementation order

### 1. Durable content and migrations

Implement the saved-version and media lifecycle owner first. Exercise save, regeneration, restart, and corrupt/missing-file behaviour using isolated temporary storage. Validate migrations against both an empty database and a populated legacy fixture. Back up the application database and media before applying to user data.

### 2. Personal APIs

Extend `/me` with lecture detail/saved explanation reads, lecture-filtered recent activity, personal bookmarks, voice-note upload/list/read/audio/delete, and note explanation attachments. Reuse the local-profile dependency and mutation origin checks. Use finite timestamp bounds and deterministic pagination. Response contracts distinguish missing content from empty personal lists and return durable resource IDs, not filesystem paths or blob URLs.

Do not auto-save the whole shared cache to the local profile. Record this profile's lecture membership when processing/opening a lecture; within that membership, preserve its successfully prepared explanations. Define the success hook in the responsible backend processing layer so preservation does not depend on which explanation the UI happens to open.

### 3. Voice-note storage integration

Add the backend notes client using the existing recorder/store seam. Keep recorder lifecycle and pause behaviour. Persist drafts and operation IDs locally, upload on explicit save/stop, retain on failure, offer recovery, then discard the local draft after durable success.

Update `VoiceNotes`, `NotesScreen`, `VoiceNoteList`, and `useNotePlayer` to read metadata/media from the backend. Keep one-player behaviour, request cancellation, error announcements, and focus restoration. Show attached explanations with timestamp/mode and one shared saved-explanation opening callback.

### 4. Library, Recents, bookmarks, playback

Replace localStorage as the authority for new personal saves. Use a lecture list plus saved explanation entries inside each lecture. Keep Read text as a paused action; main saved-explanation opening plays, as agreed. Bind Sidebar/Home summaries to the same authoritative data.

Use an explicit saved-open intent in App/Explanation so read-only reopening does not call generation/resume. Load related lecture points independently. Save playback progress at a bounded interval and when pausing/leaving, avoiding older responses overwriting newer positions. Keep the note's exact time independent of the selected nearby explanation.

### 5. Voice parity and accessibility

Reuse action registry, catalog, command validator, audio registry, and existing dialog/focus conventions. Update backend catalog/examples and generated frontend contract together. Cover open/play, filter, pagination, note playback, draft save/retry/discard, import/retry, and deletion confirmation.

Use the same handler for button, keyboard, and voice. Resolve destructive targets before confirmation. Await actual persistence before reporting success. Late voice/network responses must not act on a different screen/item.

Navigation focuses the correct heading without cold-start focus theft. Loading/error/retry states are labelled. Announce short state changes once, not every progress tick; substantive text stays outside live regions. No announcements while recording. Keep stable toggle names/pressed states, visible focus, skip links, high contrast/text-size behaviour, and help-dialog focus return.

Do not promise detection of when an external screen reader has finished speaking: browsers provide no general reliable API for that. Coordinate Synapse's own speech explicitly, keep navigation announcements short, and verify the selected saved-play behaviour manually with NVDA. Retain an immediate pause/read-text alternative.

### 6. Existing-data import and packaging

Provide an explicit preview/import flow for localStorage Library/bookmarks and IndexedDB recordings. Surviving entries retain their original text/time/mode; unresolved backend identities remain recoverable rather than silently dropped or substituted. Import stable keys prevent duplicate recordings on retry. Do not erase legacy data automatically.

Update local-only wording accurately, including onboarding/help. Verify dev browser, dev Electron, and packaged Electron transport. Keep the current audio-only microphone permission policy. Update storage/schema/voice documentation with actual verified behaviour.

## Focused acceptance checks

1. A lecture and all its ready explanations survive frontend/backend restart with matching text/audio/frame/time/mode.
2. Standard and detailed versions of the same point remain distinct.
3. Reprocessing a cache entry does not change a preserved explanation; failed/cancelled preparation cannot hide it.
4. Saved open fetches related points, seeks correctly, plays only explanation audio, and starts no inference. Lecture/note-time opening stays paused.
5. Recording captures the lecture time before pausing, has a five-minute limit, and never automatically resumes any audio.
6. Upload failure/reload retains a recoverable draft. A lost response plus retry creates exactly one confirmed note.
7. Voice-note context includes same-lecture ready versions at both inclusive +/-60 boundaries, excludes others, and allows zero matches.
8. Invalid audio/path/time/context and cross-owner identifiers are rejected. Local endpoints remain disabled outside local environments.
9. File/DB failure injection leaves no broken ready rows. Failed deletion is durable and retried. Backup/restore includes recordings.
10. Removing a Library item/history entry does not erase bookmarks, notes, attached versions, or shared cache.
11. Visible list numbering matches voice context/pagination; a refreshed list never changes an already confirmed deletion target.
12. Notes playback is quiet before a voice command; command acknowledgement/restoration never overlaps newly started media.
13. Keyboard/screen-reader loading, errors, retry, import, and deletion retain sensible focus and short announcements. Test with NVDA and a real microphone/player in addition to fakes.
14. Interrupted import can resume without duplicates or loss of original browser data.
15. Dev and packaged media requests work without wildcard/null-origin bypasses.

Use focused backend API/migration/media tests and frontend recorder/store/reopen/voice/accessibility tests, then touched-file lint and production build. Existing unrelated work must remain intact. Do not report browser mocks, automated axe checks, or previous human reports as proof of current real hardware/NVDA behaviour.

## Later online-account phase

Recommended: Supabase Auth with Google plus email/password, preserving application UUID ownership and mapping verified external identities to it. The frontend must never supply an authoritative user ID. Validate tokens server-side, handle account linking and logout, and partition/clear personal caches when identities change.

Before enabling online accounts: resolve Electron's `webSecurity: false`, trusted OAuth redirects and credential storage; add authenticated media delivery; replace the single-window shared cancellation policy with per-consumer interest/leases so one user or tab cannot stop another; protect every processing/control endpoint, not only `/me`; add deployment TLS, storage backups, and resource/admission limits. Treat these as prerequisites, not supported phase-1 behaviour.

Official references checked 2026-11-13:

- https://supabase.com/pricing — Free currently includes 50,000 MAU, 500 MB database, 1 GB file storage; inactive free projects can pause. Free Auth does not mean unlimited media hosting or inference.
- https://supabase.com/docs/guides/auth/social-sign-in/auth-google — Google sign-in configuration.
- https://supabase.com/docs/guides/auth/passwords — email/password flows.
- https://supabase.com/docs/guides/auth/auth-smtp — the default email sender is restricted; public verification/recovery needs custom SMTP. Select the mail service when the online phase starts.

## Review outcome

The local-backend phase is implemented. Migration `20261113_0009` was applied after verifying a coordinated database/media backup at `backups/learning-storage-20261113T070955118965Z`. The setup command also created its normal pre-migration SQLite backup. Existing shared videos, interventions, transcripts and explanations were retained; browser saves are not silently imported.

The implementation uses owner-scoped immutable saved-explanation rows with bookmark/view timestamps rather than a separate single-purpose bookmark table. Media bytes are deduplicated by content hash. Library membership now stores lecture metadata and survives shared-cache deletion. Saved-content requests never start inference. IndexedDB remains for drafts/import recovery, while confirmed recordings use backend files and SQL metadata.

Visual browser inspection was blocked by the computer-use tool because it could not reliably verify the browser URL. Automated tests are not a substitute for an installed-package, real microphone or NVDA check. A broader backend test run encountered a native PyArrow access violation; the focused storage/history/migration/preparation/audio/voice checks were run separately. Online accounts and cross-device sync are intentionally deferred, as agreed.
