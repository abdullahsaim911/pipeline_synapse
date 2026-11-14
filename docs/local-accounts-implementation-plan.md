# Local accounts without account UI

Status: the original UI-free phase was implemented with migration `20261113_0010`. Login/signup and a development-only local-profile bypass have since been added. The scope and sequence below describe the original phase; see `synapse-web/docs/LOCAL_ACCOUNTS.md` for the current access policy, launcher configuration and validation limits. Real microphone, NVDA and installed-package checks remain manual release gates.

## Scope and current evidence

- Active frontend: `synapse-web`. Add no login/signup pages, dialogs, navigation controls, or new voice commands requiring those controls.
- Keep the existing local application usable with its existing development profile when no account session has been selected.
- Add local username/password accounts, persistent sessions, logout, and private-data authorization. No remote identity provider, Google login, email delivery, or cross-device sync in this phase.
- `User` and personal tables already contain ownership. Both personal-storage and learning-history routers currently resolve the fixed development profile through `get_local_development_user`.
- `api/security.py` currently allows Content-Type only and disables credentialed CORS. Frontend and backend may use different loopback hostnames. Authentication transport must work with the existing topology.
- `personalApi.js` owns many personal requests, but `backendNotes.js` fetches recording audio directly. Saved audio/frame URLs also need authenticated loading.
- `explanation_service.py` captures completed explanations for Library subscribers after inference commits. Preserve this subscriber-based behavior; never resolve a background job's owner from the currently signed-in browser.
- IndexedDB contains recording drafts and quiz storage. These require an ownership audit before claiming account isolation. Existing appearance and accessibility preferences remain device settings.

## Required invariants

1. Existing local-profile data, files, identifiers, and pipeline work survive migration unchanged.
2. The backend determines ownership from a verified session, never a request-body user ID.
3. An absent credential may select the development profile only in explicit local compatibility mode. Invalid, expired, or revoked credentials return 401; never silently retry as the development profile.
4. Logout revokes the account session. It cannot restore access to that account through the compatibility profile.
5. Accounts are private within the application. Local compatibility mode is not a security boundary against other programs or administrators on the computer.
6. Shared lecture processing/cache remains shared. Personal saves, bookmarks, recordings, viewing history and progress remain user-owned. Shared processing is not duplicated per account.
7. Account changes cannot retarget an existing save, draft, import, upload retry, or asynchronous result to a different user.
8. Authentication changes do not start, cancel, restart, or change generation jobs. Existing pipeline cancellation and pause semantics remain intact.

## Implementation sequence

### 1. Baseline and ownership map

- Inspect applicable repository instructions and preserve pre-existing work in both repositories.
- Inventory all personal endpoints, media paths, browser stores, request helpers, and background persistence callbacks. Classify each as account-owned or device/shared.
- Record existing frontend/backend test results and pipeline lifecycle tests before editing. Do not rely on a whole-suite run that crashes in native PyArrow.
- Keep quiz generation, Q&A, transcription, extraction, model calls, processing queues, voice recognition and playback algorithms outside this change. If browser quiz records contain personal history, scope their storage boundary without altering quiz behavior.

### 2. Additive database migration

- Reuse `users`; keep the fixed development user and all existing foreign keys unchanged.
- Add an account-credential table linked uniquely to a user: normalized unique username, salted password hash, created/changed timestamps. Keep display name separate. Initially use a restricted, case-insensitive ASCII username to avoid ambiguous normalization.
- Add server-side sessions: user, unique digest of a cryptographically random opaque token, created/expiry timestamps and revocation state. Never store the raw session token in SQLite.
- Use Argon2id through a maintained password-hashing dependency. Bound input length before hashing; never trim passwords, log secrets, or invent a hash scheme.
- Use database uniqueness for concurrent signup and session-token uniqueness. Add indexes for session lookup and user ownership as required by actual queries.
- Take and verify a coordinated database/media backup before applying the migration. Test upgrading an existing database with populated local saves. Refuse a destructive downgrade while account data remains.

### 3. Backend authentication and session lifecycle

- Add `/auth/signup`, `/auth/login`, `/auth/logout`, and `/auth/session`. Successful signup signs into the new account; login issues a new session; logout is idempotent for an already revoked session.
- Use a 30-day persistent session initially, without silent sliding extension. Restarting the API must not invalidate an otherwise valid session. Session expiry is returned to callers.
- Use an opaque bearer token in the Authorization header for this local HTTP phase. This avoids cross-site cookie failures between localhost and 127.0.0.1 and supports the existing API consumers. Allow Authorization only for the existing explicit CORS origins; preserve origin and loopback restrictions.
- Persist the frontend token in guarded browser storage so reopening works; acknowledge that JavaScript can read it. Never place it in a URL, analytics, logs, or media paths. Do not represent this as the transport for a future public deployment.
- Add one current-user dependency shared by personal-storage and history routes. Keep the development-profile lookup as an explicit compatibility branch, rather than duplicating authorization logic in endpoints.
- Bound signup/login request sizes and password-hashing concurrency; rate-limit attempts by normalized account and local client using bounded state. Use generic invalid-login errors and avoid permanent account lockouts. Never log credentials.
- Return minimal session/user metadata, with no credential hashes or token digests. Do not allow request-controlled role/profile flags or registration as the reserved development profile.
- No password-recovery claim in this phase: document that email reset is unavailable. A recovery/reset workflow is separate work, not a hidden password bypass.

### 4. Complete the authorization boundary

- Apply the shared resolver to all existing personal/history routes and any account-owned preferences endpoints found during inventory.
- Keep every read, update, delete, list, media request and note-context attachment scoped to the resolved user. Return 404 for another user's object.
- Keep shared generation endpoints and shared source media behavior unchanged. Personal URLs must not become an alternate path to shared static files or another user's recording.
- Revalidate session/ownership before committing a long-running upload or import. Define and test commit-versus-logout ordering: already committed work remains with its original owner; rejected work remains recoverable; no write can be reassigned to the next account.
- Preserve user-scoped retry IDs and media reference counting. Deleting one account's note/bookmark must not remove another account's referenced media.
- Leave subscriber snapshot capture based on persisted Library membership. Auth failure must not turn successful inference into a failed pipeline job.

### 5. Frontend integration without account UI

- Add an account/session module exposing initialize, signup, login, logout and current-session operations for future pages. Include typed/status-bearing errors compatible with existing JavaScript conventions.
- Initialize identity before private reads or writes. While no credential exists, use the development profile in compatibility mode. Do not auto-create a password account or auto-transfer data.
- Route personal JSON requests and authenticated audio/frame fetches through the session-aware request boundary. Use authenticated blob loading for protected media and release object URLs when identity changes or playback ends.
- Keep shared pipeline request behavior unchanged. Do not globally rewrite unrelated fetches or automatically retry failed mutations after authentication changes.
- Track an identity generation number: abort old personal requests, discard late responses, clear private in-memory summaries/confirmations, stop private playback and load only the new identity's data.
- Synchronize account changes across tabs. A late logout response must not clear a newer login. Do not let an old request's 401 invalidate a newer session.
- Block an explicit local switch during active recording until the existing draft is safely preserved. An external tab switch preserves the draft under the original owner and prevents upload under the new identity. Preserve existing leave guards and screen-reader status conventions.
- On expired/revoked sessions, clear private presentation and report the existing accessible error state. Do not silently switch to development data or silently upload pending work there. Explicit logout can deliberately return to the development profile in compatibility mode.
- If persistent browser storage is unavailable, keep a session in memory and return an explicit persistence limitation; never falsely promise survival across restart.

### 6. Browser data and legacy ownership

- Namespace drafts/retry records and any personal quiz/session records by stable backend user ID. Record the owner at creation, not retry time. Isolate processing-screen restoration metadata from another user's browsing history without changing the processing job itself.
- Treat existing unowned drafts and browser saves as belonging to the existing development profile. Keep originals; no automatic import into a registered account.
- Keep device-level theme, text/display accessibility, speech preferences and onboarding behavior unchanged unless the ownership audit finds actual private data.
- Do not build a data-transfer feature into signup. Existing server data stays with the development profile; any later transfer is explicit, transactional, collision-aware and separately reviewed.

### 7. Proof and release gates

- Backend: signup validation/duplicate races, wrong passwords, bounded failed attempts, token digest storage, expiry, restart persistence, logout/revocation, CORS/origin checks, missing versus invalid credential behavior and compatibility mode disabled.
- Two-user matrix: Library, recents, bookmarks, history/progress, notes, attached explanations and every private audio/frame URL. Attempt cross-user read/update/delete and retry-key reuse.
- Race tests: account change during list fetch, snapshot save, upload, failed retry, import and audio load; logout concurrent with commit; stale 401/logout responses; multiple tabs; owner-tagged offline drafts.
- Migration: seeded existing database upgrade, unchanged local records/media, backup verification and rollback safety.
- Frontend: no account UI added, initial development workflow unchanged, persisted session restore, expired-session error behavior, cache clearing, authenticated media cleanup, browser-storage failure and legacy-store compatibility.
- Pipeline regression: explicit processing, prepared/reused explanation, cancellation, generation completion during account change, subscriber snapshot capture, saved reopening without inference and voice-note playback/pause behavior. Account transitions must not trigger pipeline open/cancel calls.
- Run focused tests first, then the existing frontend full suite/build/lint and relevant backend pipeline suites. Record unrelated pre-existing failures separately; do not modify unrelated systems to force a green run.
- Manual gates remain real microphone, NVDA, packaged Electron startup/session restore/media playback and the localhost/127.0.0.1 development combinations. Automated checks do not substitute for these.

## Completion criteria

The app opens and works with existing local data and no account interface. API signup/login/logout work; sessions survive restart; two accounts cannot access each other's application data. Existing generation and playback flows retain their behavior. The future pages need only call the account module. Document local compatibility mode and how to disable it before introducing a normal signed-out user experience or remote deployment.

Account deletion, password reset, online authentication, public deployment, cross-device sync and development-profile data transfer are not included in this implementation.
