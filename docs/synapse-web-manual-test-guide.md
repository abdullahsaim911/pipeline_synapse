# Synapse: complete manual test guide

Based on the active `synapse-web` implementation, reviewed November 14, 2026. These are expected behaviors to verify, not a claim that live screen-reader, microphone, YouTube, or model testing has passed.

## 1. Start the correct app

In a PowerShell terminal:

```powershell
Set-Location F:\Prototyping\Synapse
.\scripts\synapse.ps1 start
```

In another terminal:

```powershell
Set-Location F:\Prototyping\Synapse
.\scripts\synapse.ps1 status
Invoke-RestMethod http://127.0.0.1:8000/ready | ConvertTo-Json -Depth 5
```

Before measuring performance, check `components.api`, `voice`, `transcription`, `modal` (Qwen), and `mistral`. For this configured demonstration, both remote models should be `ready`. Top-level `ready` alone does not prove that the remote models are warm. `unavailable` needs investigation; `disabled` is not proof of a working model. Warmup checks health/model identity, not generation quality. A GPU can become cold again after an idle period.

Start the active frontend separately:

```powershell
Set-Location F:\Prototyping\Synapse\synapse-web
npm run dev -- --port 5173 --strictPort
```

Open the URL printed by Vite, normally `http://localhost:5173`. If port 5173 is occupied, stop the old frontend in its own terminal before starting this one. An older `frontend` server was previously running on this port; an HTTP 200 does not identify which UI you are testing. `--strictPort` prevents silently switching to another port.

For Electron, keep the correct Vite server running and use a third terminal:

```powershell
Set-Location F:\Prototyping\Synapse\synapse-web
npm run electron
```

Alternatively, `npm run electron:dev` starts both UI processes when port 5173 is free. Neither Electron nor Vite starts the Python backend.

Current `synapse-web/.env` selects the real API at `http://127.0.0.1:8000`, sets `VITE_API_MOCK_MODE=false`, and enables quiz with `VITE_QUIZ_ENABLED=true`. Restart Vite after changing these values. Do not accept mock results or explicitly labelled structured-view demo data as evidence of model accuracy. Personal storage requires the local backend.

## 2. Prepare a useful test set

- Use a disposable account and headphones. Keep the lecture URL and a notebook for failures.
- Choose at least three different lectures: a clear diagram/process, mathematics with a graph, and code or tables. Include one previously unprocessed lecture and one saved lecture.
- First do the complete journey with ordinary controls, then repeat with keyboard and NVDA. Finally repeat essential operations in Electron if that is how you will demonstrate the app. Browser success does not establish Electron success.
- Record browser/Electron version, NVDA version, app mode, URL, lecture timestamp, explanation depth, exact input, expected result, actual result, and time taken.
- Do not clear browser storage or delete real recordings to test recovery. Use disposable data.

## 3. Keyboard and NVDA baseline

Start NVDA before opening the app. Use Tab/Shift+Tab to move, Enter to activate, and Space for focused buttons/checkboxes. Use NVDA browse commands H for headings, D for landmarks, B for buttons, and F for fields; NVDA+F7 opens the elements list. NVDA+Space changes browse/focus mode when necessary. Ctrl stops NVDA speech.

| Shortcut | Current action | Where |
| --- | --- | --- |
| Alt+T | Start/stop voice command | Where Talk is available |
| Alt+S | Focus lecture link/search | Home and subject results |
| Alt+P | Play/pause/resume lecture | Lecture |
| Alt+L | Listen/pause explanation | Lecture |
| Alt+N / Alt+B | Next/previous moment | Lecture |
| Alt+R | Focus start of explanation | Lecture |
| Alt+. / Alt+, | Faster/slower explanation audio | Lecture |
| Alt+M | Start/stop voice note | Lecture |
| Alt+K | Speak shortcut list | Lecture |

Alt+B does not bookmark; Alt+R does not restart audio. Lecture shortcuts are ignored while typing in fields; Alt+T intentionally remains available. Help blocks the background shortcuts. If a shortcut fails, record focus location and NVDA mode, then try the corresponding visible button to isolate the failure.

Across every screen check:

- The skip link reaches main content; headings and navigation landmarks make sense.
- Each control has a meaningful name, role, and state. No essential action needs a mouse.
- Focus is visible, follows navigation sensibly, and is not lost after content disappears.
- Loading, errors, selected states, and confirmations are understandable without sight. Updates should not repeatedly interrupt reading or move focus on each poll.
- Reading the explanation does not unexpectedly play audio. No two lecture/explanation/note audio sources play together.

## 4. Account, onboarding, Help, and appearance

1. Create a disposable account. Try a short username and password first: focus should reach the invalid field and the error should be readable. Username is 3–64 characters, starts with a letter/number, and permits letters, numbers, dot, dash, underscore. Password is 8–128 characters. This is a local account, not an email login, and there is no password-recovery flow.
2. Test Show password, incorrect login, successful login, reload/session restore, sign out, and sign back in. A restore failure should offer recovery rather than show private data. Development can allow a local profile; use real accounts to test isolation.
3. Follow onboarding, choose a presentation mode, and try Skip for now where offered. No lecture should start before you select one.
4. Open Help while idle and while playing audio. NVDA should announce a dialog; focus moves inside, Tab/Shift+Tab stay inside, and background controls cannot be activated. Escape/Close returns focus to the opener. Audio paused/stopped for Help should not restart simply because Help closes. Help cannot open during active voice-note recording; stop the recording first.
5. Open Settings, select Paper, Night, and Sepia; verify selected state and readable contrast. Escape closes Settings and returns focus to its button. Settings is a disclosure, not the Help modal.
6. Test text enlargement, high contrast, browser zoom at 200%, and keyboard scrolling. Check that content and essential controls remain reachable; record clipping, overlap, or unreadable combinations.
7. Sign into a second disposable account. The first account's library, bookmarks, notes, and quiz work must not appear. Return to the first account and confirm its saved work remains.

## 5. Find and process a lecture

1. On Home, press Alt+S. Submit an empty field: receive a clear validation message. Search a topic, wait for results, and select a result. It should start the same lecture flow as pasting a valid YouTube URL.
2. Explore a subject and open a result. Verify title/channel metadata and that the chosen lecture, not another result, opens. Test no results, a malformed link, and an unavailable/private video; failures must be understandable.
3. Submit a fresh supported lecture once. Watch processing status and first available moments. Processing and explanation preparation are separate; a lecture can open while some explanations are still preparing.
4. Try cancellation on a disposable fresh job. Wait for the backend's cancelled state. Completed explanations should remain saved; a failed cancellation must not falsely announce success. Use retry/reconnect controls when offered.
5. Reload during processing, then reconnect/resume if offered. Confirm the same job/lecture is tracked and there is no misleading completion or duplicated work from repeated clicks.

## 6. Lecture and explanation flow — highest priority

1. Start the lecture explicitly with its control or Alt+P. Verify play/pause and that the displayed status matches actual sound.
2. Let playback reach an eligible visual moment. Synapse should pause and present an explanation question. It should not speak the explanation over the teacher.
3. If ready, choose **listen** or **skip**. If preparing, choose **wait** to remain paused, or **resume** to continue while it prepares. Use the visible answer buttons first, then test the same choices by voice.
4. After resuming while preparation continues, verify a ready notification offers **listen** or **later**. Later should let the lecture continue and defer the explanation.
5. After explanation audio ends, expect: **yes** (clear, continue), **no** (choose another depth), or **replay**. Do not expect unconditional automatic lecture resume.
6. Choose no, then Brief, Standard, and Detailed on different trials. A missing depth should be requested, not replaced by unrelated text. Each depth must keep the same essential facts.
7. Pause explanation with Alt+L; change speed with Alt+. and Alt+,. Check audible speed and selected control state. Alt+R should let NVDA read the text from its start without starting TTS.
8. Navigate next/previous moments and open a moment through the timeline. The title, timestamp, category, text, and audio must all belong to that selection. Fast switching must not attach a late response to the wrong moment.
9. Let the lecture end, including a trial with deferred explanations. The review should offer choices appropriate to each item's ready/preparing/failed state, allow next/finish, and announce completion rather than loop forever.
10. When preparation fails, use retry if offered, or resume. A nonretryable/exhausted request should not offer unlimited retries. If audio fails, text should remain readable and the user should be able to continue.

Accessibility mode hides the video and removes its iframe from keyboard/screen-reader navigation while lecture audio remains usable. Study mode shows it. Switching modes should preserve the lecture and playback position. In Study mode, Structured View appears only when structured rows resolve. Test arrow keys/Home/End on the visual tabs, selected-state announcements, and Focus important area if available. In Accessibility mode, read the corresponding Accessible visual information section. Missing data is an unavailable-data case, not permission to invent it. Explicitly labelled demo rows are examples, not extracted facts.

## 7. Voice control

See [the complete voice command and announcement reference](synapse-web-voice-reference.md) for every catalog action, contextual restrictions, and the implemented feedback messages. In particular, selecting a search result by voice only selects it; say “Start learning” afterwards. Voice quiz start currently creates 5 Practice questions.

Use English commands. Press Alt+T once, wait for the listening tone, say one command, then pause or press Alt+T to finish. Alternatively focus Talk and hold Space to speak, releasing to finish. Do not hold Alt+T. Typical states are Getting ready, Listening, Working, and Talk. No-speech timeout is about 6 seconds, utterance limit about 12 seconds, and silence after speech about 1.4 seconds; processing adds time.

Use headphones and stop NVDA speech with Ctrl before speaking. Synapse can pause its own media but cannot guarantee that external screen-reader sound will not enter your microphone.

| Context | Example commands to try | Expected result |
| --- | --- | --- |
| Global study navigation | “Open my library”, “Open bookmarks”, “Open voice notes”, “Go home” | Correct destination; previous screen's audio stops |
| Orientation/help | “Where am I?”, “Help”, “Repeat the last response”, “Stop speaking” | Relevant response/action without unintended navigation |
| Home | “Search for electric fields”, “Open result one” | Search and choose current result, not stale results |
| Lecture | “Pause lecture”, “Play lecture”, “Go forward ten seconds” | Correct playback operation |
| Explanation | “Listen to the explanation”, “Next moment”, “Previous moment”, “Read the explanation”, “Listen faster” | Correct selected explanation/control |
| Explanation question | “Listen”, “Wait”, “Resume”, “Yes”, “No”, “Replay”, “Detailed” | Only choices valid for the current question act |
| Timeline | “Show equations”, “Read the moments”, “Go to moment two”, “Open this moment” | Filter/select/preview/open remain distinct |
| Notes | “Play note one”, “Stop note audio”, “Delete note one” | Correct current note; deletion confirmation |
| Quiz | Navigate questions, choose an answer, check answer in Practice, review | Same result as equivalent visible controls |

Examples are contextual, not guaranteed exact-string matches. If the app asks for confirmation, start a new Talk turn and answer yes/no. Test cancel/no first on disposable data. Commands without a valid current target should clarify or fail safely.

Additional voice checks:

- Deny microphone permission, then restore it and retry. A useful error should replace an endless Listening state.
- Speak nothing, give an ambiguous command, request an absent item, and try a non-English command. No unintended action should occur; voice currently requests English.
- Start Talk while lecture/explanation audio plays: media must pause before capture. A harmless command may restore previously playing media; an explicit pause command must leave it paused. Audio that was already paused must not start.
- Open Help or navigate while a voice request is processing. A late response must not execute behind the dialog or on the wrong screen.
- Start voice-note recording, then try Talk. Both microphone recordings must never run simultaneously; stopping a note can require a separate subsequent Talk action.

## 8. Voice notes and saved collections

1. During a lecture, press Alt+M, wait for the tone, and record 5–10 seconds. Lecture and explanation audio pause. Press Alt+M again: a normal explicit stop **automatically saves** the note and leaves the lecture paused. Find its timestamp and play it.
2. Open global Voice Notes, play/stop the same note, open its lecture timestamp, and open a nearby saved explanation if listed. Reload and confirm persistence. Saved audio/metadata live in the local backend; account-scoped drafts/recovery also use browser IndexedDB.
3. Start another recording and navigate away. This interruption should offer Save/Discard/Stay rather than silently discard it. Device loss also holds a draft where possible. On save failure, retry/discard should remain available. Do not assume unsaved in-memory audio survives closing the app.
4. A recording shorter than 0.5 seconds is rejected. Silence may produce a warning. Optional boundary test: at 5 minutes recording stops and saves, with advance warning.
5. Delete a disposable note: Cancel receives initial confirmation focus; cancelling keeps it. Confirm deletion and verify sensible focus on a neighbouring item or empty-state heading. Reload to verify deletion. Deletion is permanent.
6. Bookmark a prepared explanation. In Bookmarks, check lecture, moment, depth, text, and audio. Remove the bookmark after cancelling once; the underlying recording should remain.
7. In Library test Saved lectures, Recent explanations, search/filter, and pagination if enough data exists. **Open and play explanation** can start saved audio; **Read text** should open without auto-playing.
8. Remove a disposable lecture from Library: bookmarks and recordings should be retained, as the confirmation/status describes. Verify them separately.
9. Open a saved version whose other depth is missing: expect a choice to **process** or **later**. Saved-only viewing must not silently pretend an unsaved depth exists. If Recover browser saves is shown, import once and verify content/duplicates before retrying.

## 9. Ask and quiz

Ask: open Ask this lecture, ask a short lecture-related question, verify the answer against the teacher, and follow any evidence timestamp. Try something outside the lecture: the app should report not-in-lecture/insufficient context instead of inventing support. Empty or over-500-character input should be rejected. Check loading, retryable failure, keyboard focus, and that typing does not trigger lecture shortcuts. Spoken lecture questions should use this same question-answering feature.

Quiz: the current frontend configuration enables it, but the backend has a separate feature flag. A disabled/unavailable quiz is not evidence that Mistral explanation generation failed. Ask/quiz have separate model dependencies from the explanation pipeline.

1. From a processed lecture with usable transcript, generate Practice with the default 6 questions. Allowed requested count is 5–10. Insufficient transcript/material should give an explicit message rather than fabricated questions.
2. Select an answer, Check answer, read feedback, use a hint, change the answer, and check again. Practice score uses the **first checked answer**. An unchecked selection counts as unanswered.
3. Flag a question, navigate next/previous, open review, and verify selected/unanswered/flagged summaries. Use keyboard radio controls and verify group labels with NVDA.
4. Leave/reload and resume. Answers, flags, current position, and saved state should match. Starting a new set should preserve the prior attempt until the new questions are validated; heed save/conflict errors.
5. Submit with an unanswered question. Confirm the warning, try Go back, then submit deliberately. Correct/incorrect/unanswered totals must add to the question count. Source lecture links should open the correct timestamp without unexpectedly starting playback.
6. Repeat in Assessment: no per-question Check answer/hints; scoring uses the final selection on submission. Verify post-submission review and source evidence.
7. If storage fails, the app should explain that work cannot be safely saved; do not accept a success message that loses answers. Test multiple tabs only with disposable attempts, checking conflict handling.

## 10. Explanation accuracy: your agreed framework

Functional success and educational correctness are separate. For each sampled moment compare the actual video frame, teacher narration around the timestamp, and all three explanation depths. Score each component 0 (missing/wrong), 1 (partial), or 2 (correct/useful):

| Component | Question |
| --- | --- |
| Orientation | Does it identify the visual and topic? |
| Structure | Can I reconstruct arrangement, grouping, order, and connections? |
| Salient information | Are important labels, symbols, values, units, and directions correct? |
| Educational meaning | Does it explain the concept and why the relationships matter? |
| Lecture connection | Does it connect to what the teacher actually says? |

An invented number, reversed arrow, wrong operator, invented code, or contradictory essential fact is a failure even if the prose sounds fluent. Unreadable evidence should be acknowledged. Brief/Standard/Detailed may differ in depth but must agree on facts. Spoken audio and exported text must match the selected moment/version; check pronunciation of important symbols and units.

Cover these categories across several videos, marking any untested category explicitly:

| Category | Check especially |
| --- | --- |
| Charts/graphs | Axes, scales, units, legend, trends, values |
| Mathematics | Exact symbols, signs, operators, transformations |
| Scientific structures | Parts, labels, physical relationships |
| Processes/networks | Arrow direction, branches, sequence, dependencies |
| Tables/matrices | Headers, row/column associations, exact cells |
| Instructional text | Headings, bullets, grouping, reading order |
| Geometry | Shapes, angles, coordinates, dimensions |
| Code | Syntax, indentation, control flow |
| Software interfaces | Components, hierarchy, controls, states |
| Maps/spatial data | Regions, routes, relative positions |
| Observational images | Visible features without unsupported mechanisms |
| Physical scenes/illustrations | Arrangement, annotations, interactions |

Include a mixed frame and a dense/unreadable frame. A still image alone does not justify claims about motion. Do not conclude that all videos are accurate from one successful lecture.

## 11. Exports and failure recovery

- Export PDF and audio where available. Open the actual downloaded files: correct title/moment/depth, complete readable text, working audio, and no previous lecture's content. Inspect PDF reading order if it will be used with a screen reader.
- On disposable work, interrupt internet or stop the backend through its normal launcher, then try an operation. Expect clear failure and a way to retry; restore services and verify recovery. Do not stop a shared backend during someone else's work.
- Separate a cold GPU delay, generation failure, and TTS failure. Text may be usable while audio is unavailable. Prior live runs found slow fallback speech generation; record time rather than repeatedly submitting.
- Refresh/reopen and check session, library, bookmarks, notes, and quiz persistence. Saved text/audio still require the local backend; the YouTube video and cloud-backed operations require their own connectivity.

## 12. Decide whether the app passed

Use Pass / Fail / Blocked / Not tested for each section. A practical first pass is: account → search → fresh lecture → one complete explanation conversation → voice playback command → save/play note → bookmark/reopen → Ask → Practice → sign out/in. Then repeat essential paths using only keyboard and NVDA, and expand accuracy coverage across the other videos.

Block a demonstration for wrong lecture/answer targets, unsupported essential facts, lost recordings/quiz answers, cross-account leaks, inaccessible essential controls, microphone stuck on, or overlapping teacher/explanation audio. Report preparation/audio delays separately from factual errors.

Bug report template:

```text
Screen and browser/Electron + NVDA versions:
Lecture URL, timestamp, depth, presentation mode:
Steps / exact spoken command / keyboard focus:
Expected:
Actual, including NVDA announcement:
Time taken and readiness/error status:
Reproduces after reload? Evidence file/screenshot/log location:
```

## Evidence and limits of this review

Read the active App, account/session and personal-storage clients, search/lecture components, explanation conversation hook, structured-view resolver, shortcuts, voice controller/recorder/action handlers, notes/collections, Ask/quiz clients and scoring, Electron startup, and readiness documentation.

Validation on November 14: `synapse-web` production build passed; all 67 frontend test files and 1,323 tests passed. The run emitted React test-environment warnings and a backend connection-refused log in a passing test. Build warnings concerned browser-data age and bundle size. These checks do not establish live backend availability, real microphone behavior, actual NVDA accessibility, or explanation accuracy. Those remain the manual checks above.
