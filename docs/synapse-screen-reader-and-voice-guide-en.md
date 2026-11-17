# Synapse: Complete Screen Reader and Voice Control Guide

Updated November 16, 2026. This guide covers the active `synapse-web` application: using Windows and NVDA, controlling the app by voice, and combining both methods. Instructions were checked against the current source code; every step has not been tested with a live microphone and screen reader.

Speak voice commands **in English**. The example phrases describe supported intentions rather than a rigid command language. A command also needs the appropriate screen, content, and application state.

## 1. Understand the three ways of interacting

- **Screen reader:** NVDA reads page text, control names, selections, focus changes, and accessible updates. You can navigate with the keyboard without using a microphone.
- **Voice control:** You speak an instruction such as “Play the lecture,” and Synapse performs the corresponding action.
- **Voice notes:** You record your own study notes. Words spoken while recording a note are saved as audio rather than executed as app commands.

The teacher’s lecture, Synapse’s explanation, and your recorded notes are separate audio sources. Before listening to an explanation, check the selected lecture and moment. A **moment** is a timestamp associated with important visual material, such as a diagram, equation, or graph.

## 2. Prepare the app and equipment

1. Open the running Synapse app. A typical local address is `http://localhost:5173`; use the address supplied by your installation.
2. Start NVDA if you want screen reader support. Use a voice that can read the English interface.
3. Connect a microphone for voice commands and voice notes. Allow microphone access when the correct site or app requests it.
4. Wear headphones so the teacher, Synapse, and NVDA are less likely to be recorded by your microphone.
5. Before giving a voice command, press Ctrl to stop any current NVDA speech. This does not stop Synapse speech or lecture playback.
6. Make sure the Synapse server is available. An open interface does not establish that search, speech recognition, explanations, or quizzes are ready.

If you need to start the application yourself, see the [startup instructions](synapse-web-manual-test-guide.md#1-start-the-correct-app). Otherwise, continue with the steps below.

## 3. Navigate with a screen reader

### 3.1 Essential NVDA and keyboard commands

These commands assume a typical Windows NVDA configuration. The NVDA modifier is usually Insert or, depending on your settings, Caps Lock.

| Key | Purpose |
| --- | --- |
| Tab | Move focus to the next available control |
| Shift+Tab | Move focus to the previous control |
| Enter | Activate a focused button or link |
| H / Shift+H | Next / previous heading in Browse mode |
| D / Shift+D | Next / previous landmark in Browse mode |
| B / Shift+B | Next / previous button in Browse mode |
| F / Shift+F | Next / previous form field in Browse mode |
| NVDA+F7 | Open the Elements List |
| NVDA+Space | Switch between Browse mode and Focus mode |
| Ctrl | Stop current NVDA speech |
| Arrow keys | Read text or move within controls such as radio groups and tabs |

When you reach **Skip to main content**, press Enter to move past repeated navigation. Read the page heading to identify your current screen: Home, Lecture, Timeline, Library, or another destination.

**Important Space behavior:** When Talk is available, Space outside a text field can start or stop voice control, including when another button is focused. Use **Enter to activate buttons and links**, and Arrow keys to select radio options. NVDA Browse mode may handle Space itself, so **Alt+T is the clearest way to start voice control while using a screen reader**.

### 3.2 Synapse shortcuts

| Shortcut | Action | Where it applies |
| --- | --- | --- |
| Alt+T | Start or stop a voice command | Wherever Talk is available, including while typing |
| Alt+S | Focus the lecture link or search field | Home and subject results |
| Alt+P | Play, pause, or resume the teacher’s lecture | Lecture |
| Alt+L | Listen to or pause explanation audio | Lecture |
| Alt+N | Next moment | Lecture |
| Alt+B | Previous moment | Lecture |
| Alt+R | Focus the beginning of the explanation text | Lecture |
| Alt+. | Increase explanation listening speed | Lecture |
| Alt+, | Decrease explanation listening speed | Lecture |
| Alt+M | Start or stop recording a voice note | Lecture |
| Alt+K | Hear the lecture shortcut list | Lecture |

Alt+B does not create a bookmark. Alt+R moves focus to readable text; it does not replay audio. Lecture shortcuts are generally ignored while typing in text fields. Help blocks shortcuts that would operate the background screen.

If a shortcut does not work, check your screen and focus location. Then use Tab to find the corresponding visible control and press Enter.

## 4. Give voice commands

### 4.1 Start and finish a command

1. If NVDA is speaking, press Ctrl to stop it.
2. Press and release Alt+T. Do not hold it down.
3. Wait for the listening tone. Do not start speaking during Getting ready.
4. Speak one clear English instruction, such as “Open my library.”
5. Stop speaking, or press Alt+T again to finish recording.
6. Wait through Working, listen to the response, and check the result.

Outside text fields, you can also tap Space to start listening and tap it again to stop. Alternatively, hold Space while speaking and release it to finish. Enter on the Talk button toggles recording too. Prefer Alt+T when using NVDA.

The Talk button normally changes through these states:

`Talk → Getting ready… → Listening… → Working… → Talk`

A turn may end after about six seconds without speech. A single utterance is limited to about twelve seconds. A short silence after speaking can also finish capture. Keep questions concise.

### 4.2 Automatic listening after a question

Auto-listen is normally enabled in the current app. After Synapse speaks a clarification, confirmation, or explanation question, it may open the microphone automatically for your answer. **Wait for the tone and Listening state before answering.**

This is not continuous listening or a wake-word system. If listening does not start automatically, press Alt+T to begin a new turn. Auto-listen is disabled when the stored screen-reader-only feedback preference is active, to avoid recording the screen reader. There is currently no ordinary Settings selector for that preference.

### 4.3 Useful general commands

| Say | Result |
| --- | --- |
| “Where am I?” | Hear your current screen and context |
| “Help” | Hear a short guide to commands for this screen |
| “Say that again” | Repeat the assistant’s previous response |
| “Stop talking” | Stop assistant speech |
| “Cancel” | Clear a pending voice request or cancel a supported confirmation |
| “Go to the home screen” | Open Home |
| “Open my library” | Open Library |
| “Open bookmarks” | Open Bookmarks |
| “Open my voice notes” | Open Voice Notes |

“Stop talking” does not stop NVDA. “Cancel” is not a universal command for cancelling a processing job. Confirm success by checking the destination, selection, or actual playback as well as the spoken reply.

## 5. Sign in and complete Welcome

### With a screen reader and keyboard

1. Read the Sign in or Create account heading.
2. Tab to Username and Password and enter your details.
3. Use a username of 3–64 characters, beginning with a letter or number. Letters, numbers, dots, dashes, and underscores are allowed.
4. Use a password of 8–128 characters. Use Show password if needed.
5. Activate the appropriate submit button. If an error appears, read it and correct the indicated field.
6. On Welcome, choose Accessibility mode or Study mode, then activate Get started or Continue. You can use Skip for now when offered.

This is a local account rather than an email login. There is currently no standard password-recovery flow.

### Available voice assistance

- Say “Create an account” to open signup.
- Say “I already have an account” to return to sign-in.
- Say “My username is maha22” to fill your username. Listen to the spelling read back and correct it if needed.
- **Type your password**, then press Enter or use the submit button. Passwords are not accepted through voice control.
- On Welcome, say “Accessibility mode” or “Study mode,” followed by “Get started.” You can also say “Skip the introduction.”

Accessibility mode hides the video while keeping lecture audio available. Study mode shows the video. You can change presentation mode later on the lecture screen.

## 6. Find and open a lecture

### With a screen reader

1. Open Home and press Alt+S.
2. Enter a topic, such as `electric fields`, or paste a YouTube lecture link.
3. Activate the appropriate Search or Start learning control.
4. For search results, read titles and available channel details. Activate the correct result’s open control. A visible result control may begin processing directly.
5. Read processing updates. Opening the lecture and preparing its explanations are separate stages.
6. Use Retry, reconnect, or cancellation controls when offered. To cancel processing, use the available processing cancellation button.

### By voice

1. “Go to the home screen.”
2. “Search for lectures about electric fields.”
3. After results arrive, say “Read the results.” Use “Read more” for additional entries.
4. “Choose the first result.”
5. **Then say “Start learning.”** Selecting a result by voice can populate its URL without starting the lecture.
6. Once the lecture screen opens, say “Play the lecture.”

If you have already copied a YouTube link, say “Paste the link,” then “Start learning.” If clipboard access fails, use Alt+S and Ctrl+V instead.

On Home, “Show me physics lectures” browses a subject. “Continue where I left off” resumes an available previous lecture.

Result numbers refer to the current list. After a new search or page change, read the list again before choosing a number.

## 7. Control the lecture and explanations

### 7.1 Teacher playback and moment navigation

| Keyboard or control | Voice example | Action |
| --- | --- | --- |
| Alt+P or Play lecture | “Play the lecture” | Start or resume the teacher |
| Alt+P or Pause lecture | “Pause the lecture” | Pause the teacher |
| Available seek controls | “Skip forward thirty seconds” | Move forward in the lecture |
| Available seek controls | “Go back ten seconds” | Move backward in the lecture |
| Alt+N | “Go to the next moment” | Select the next explanation moment |
| Alt+B | “Go back to the previous moment” | Select the previous moment |

Before microphone capture, Synapse attempts to pause its own media. A harmless command may restore media that was previously playing. An explicit pause or navigation action should preserve the resulting playback state. Check what you actually hear.

### 7.2 Answer explanation questions

At an important visual moment, the lecture can pause and ask whether you want an explanation. Listen to the complete question and choose from its offered options. Speak after the listening tone; if listening has not started, press Alt+T. You can also Tab to the corresponding answer button and press Enter.

| Current state | Answer | Result |
| --- | --- | --- |
| Explanation ready | “Listen” | Play explanation audio |
| Explanation ready | “Skip” | Skip it and continue the lecture |
| Explanation preparing | “Wait” | Keep the lecture paused while waiting |
| Explanation preparing | “Resume” | Continue the lecture during preparation |
| Ready notification after continuing | “Listen” / “Later” | Listen now / defer it |
| Explanation finished; asked whether it was clear | “Yes” | Continue the lecture or review |
| Same question | “No” | Choose another explanation depth |
| Same question | “Replay” | Hear the explanation again |
| Asked to choose another depth | “Brief,” “Standard,” or “Detailed” | Request an offered version |
| Preparation failed | “Retry,” when offered / “Resume” | Retry preparation / continue |
| Chosen saved depth is missing | “Process” / “Later” | Prepare the missing version / keep using saved content |
| Explanation audio failed | “Listen,” when offered / “Resume” | Retry audio / continue |
| End-of-lecture review | “Next” / “Finish,” when offered | Move to the next review item / end review |

“Yes” takes its meaning from the current question or confirmation. Do not assume it always means play. After an explanation ends, answer the question instead of expecting the teacher to resume unconditionally.

### 7.3 Request, listen to, or read an explanation

- “Please explain this part” or “I do not understand this, explain it” requests an explanation using the current lecture context. Standard is the default depth.
- “Give me a brief explanation” or “Give me the detailed version” requests a different depth.
- “Read the explanation to me” or “Listen to the explanation” **plays explanation audio**.
- “Pause the explanation” pauses explanation audio.
- “Read that explanation again” replays it from the beginning.
- “Speak faster” and “Speak slower” change explanation listening speed, rather than the teacher’s video speed.

**To read the text with NVDA:** Pause teacher and explanation audio, press Alt+R, and use NVDA reading commands. Check the selected moment title and Brief, Standard, or Detailed version. Use the readable explanation for long content rather than relying on a short assistant reply.

## 8. Use the Timeline and visual information

The Timeline lists lecture moments with their titles, categories, and timestamps.

### With a screen reader

1. Open Timeline from the lecture.
2. Read the heading and moment list.
3. Select a category filter if needed.
4. Select a moment and read its preview or description.
5. Activate its Open control to open it in the lecture. Selecting and opening can be separate actions.

### By voice

| Say | Action |
| --- | --- |
| “Show the timeline” | Open Timeline from the lecture |
| “Read the list” | Hear moment names, categories, and times |
| “Show only the graphs” | Filter to graph moments |
| “Show all moments” | Restore all categories |
| “Go to moment two” | Select a moment from the current list |
| “What is selected?” | Hear the selected moment’s details |
| “Open this moment” | Open the selected moment in the lecture |

Filters such as graphs, equations, charts, diagrams, code, and tables depend on the lecture’s available content. Filtering can change item numbers.

In Study mode, Picture and Structured View tabs appear when the required content is available. Focus the tabs and use Arrow keys, Home, or End. You can also say “Show me the picture” or “Show the structured view” in the appropriate context. “Highlight the important part” requires an available focus region.

In Accessibility mode, read **Accessible visual information**. Missing structured information does not establish that the image contains no information. Clearly labelled demo content is an example rather than extracted lecture evidence.

## 9. Record and manage voice notes

### 9.1 Record and save with the keyboard

1. Open the lecture at the time you want to annotate.
2. Press Alt+M. Teacher and explanation audio pause.
3. Wait for the recording tone, then speak your note.
4. Press Alt+M again or activate Stop recording. A normal explicit stop automatically saves the note.
5. Read or listen for the saved status, timestamp, and duration.
6. The teacher remains paused. Press Alt+P when you want to continue.

### 9.2 Start a note by voice

Press Alt+T and say “Record a voice note.” Wait for the note recording tone, then dictate your note.

The record action is a toggle: repeating it stops recording. However, command capture and note recording cannot use the microphone simultaneously. **Use Alt+M or Stop recording to finish reliably.** Pressing Talk may first stop the note; begin a separate turn for your next command.

A recording shorter than half a second may be rejected. The maximum duration is five minutes, with a warning near the limit. Silence may produce a warning. Check that the note was saved after recording stops.

### 9.3 Handle a draft or save failure

Navigation during recording, a disconnected microphone, or a save error can produce Save, Retry, Discard, or Stay options.

- To keep the recording, choose Save or Retry. If a draft is available, you can say “Save my recording draft.”
- To abandon it, choose Discard or say “Discard my voice note draft.”
- An unsaved draft held only in memory may not survive closing the app. Verify successful saving first.

### 9.4 Play, open, and delete notes

On Voice Notes, use Tab and Enter to operate note controls, or use these commands:

| Say | Action |
| --- | --- |
| “Open my voice notes” | Open the note list |
| “Play voice note two” | Play the second visible note; repeating the action can stop it |
| “Stop my voice note audio” | Stop the current note playback |
| “Open voice note two at its time” | Open the lecture paused at that note’s timestamp |
| “Open explanation one attached to note two” | Open and play the first listed attached explanation |
| “Delete voice note two” | Open a deletion confirmation |
| “Cancel voice note deletion” | Keep the note |
| “Confirm voice note deletion” | Permanently delete the named note |

Check the note’s name and timestamp before confirming. Do not assume deleted recordings can be restored.

## 10. Use Bookmarks and Library

### 10.1 Bookmark an explanation

On the lecture screen, activate Bookmark or say “Bookmark this explanation.” If the explanation is still saving, wait and try again.

Bookmarking is a toggle. Repeating the action can remove the bookmark. Listen to the resulting state, including any “Bookmark removed” message.

### 10.2 Open saved content

1. Use navigation controls, or say “Open my library” or “Open bookmarks.”
2. Say “Read my saved items” to hear the list. “Read more” can read more entries; it is different from opening another page.
3. Check the lecture title, moment, timestamp, and depth.
4. Say “Read saved item two” to open its text without automatically playing audio.
5. Say “Play the audio of the first item” to open and listen to a saved explanation.

Additional commands include:

- “Show recent explanations” — select the Library’s recent explanations view.
- “Filter saved items by calculus” — filter by topic or lecture title.
- “Clear the saved items filter” — clear the filter.
- “Next page of saved items” / “Previous page of saved items” — change pages when available.
- “Retry loading saved content” — retry after a loading failure.
- “Import my browser saves” — import available older browser saves into local backend storage.

### 10.3 Remove saved entries

Use Remove or a command such as “Delete the first bookmark.” Listen carefully to the target. The voice layer may first request Yes or No. For an existing on-screen removal prompt, “Confirm saved item removal” and “Cancel saved item removal” may also be available. The voice and button flows do not necessarily use an identical sequence of prompts.

Removing a lecture from Library keeps its bookmarks and recordings. Removing a bookmark and deleting a personal recording are separate operations.

If a saved explanation lacks the requested depth, choose Process or Later when prompted. Saved content does not guarantee that every depth and audio version is already available.

## 11. Ask questions about a lecture

### With a screen reader

1. Open the correct lecture.
2. Find Ask this lecture.
3. Enter a question and activate its submit control. Questions must be nonempty and no longer than 500 characters.
4. Read the answer and any source evidence.
5. Open an evidence timestamp if useful. The lecture may open paused at that time; press play to hear the teacher.

### By voice

Ask a concise question such as “What does the axis on this graph represent?” Synapse attempts to answer using the available lecture context. Open a lecture first if none is active.

An app usage question has a different purpose: “How do I bookmark this?” asks for instructions rather than directly bookmarking the explanation.

If the answer reports insufficient context, open the relevant moment and ask a more specific question.

## 12. Complete Practice and Assessment quizzes

Quiz availability depends on the lecture, backend, and feature settings.

### 12.1 Start a quiz

On the lecture screen, activate Practice this lecture or say “Start the practice quiz” to open setup. Then:

- Say “Start the practice questions” for a default Practice set. The current default is **six questions**.
- Say “Start an assessment with eight questions” to specify mode and count. Supported counts are 5–10.
- Alternatively, select mode and count using setup controls, then activate the generate button.
- Voice start uses its own defaults when mode or count is omitted; do not assume it uses a selection previously changed in the setup form.
- If a previous attempt is available, use its Resume control. There is no dedicated voice command for resuming an old attempt.

### 12.2 Answer questions

With NVDA, read the question heading. Use Arrow keys within the answer radio group. Activate Check answer, Hint, Next, Previous, or Review with Enter.

| Say | Action and restriction |
| --- | --- |
| “Read the question” | Read the current question and choices |
| “Option two” | Select the second choice; does not check or submit |
| “Check my answer” | Check the selected answer in Practice |
| “Give me a hint” | Request an available Practice hint |
| “Next question” | Move to the next question when one exists; voice navigation reads it |
| “Previous question” | Move to the previous question when one exists |
| “Review my answers” | Open the pre-submission review |
| “Submit my answers” | Submit from the review screen |
| “Go back to the lecture” | Return from quiz setup or results |

Practice scoring uses the **first checked answer**. A selected answer that was never checked can count as unanswered. Assessment has no per-question Check answer or hints; it scores the final selections on submission.

Use available interface controls for Flag, Reveal answer, resuming an older attempt, and direct question-number navigation. The dedicated quiz “Go back to the lecture” command is not registered while answering; use the available navigation controls.

### 12.3 Review and submit

1. Read unanswered and flagged summaries on Review.
2. Return to individual questions through the available controls if you want to change answers.
3. Activate Submit. Read any unanswered-question warning before proceeding.
4. Submitted answers cannot be changed.
5. Read results, feedback, and evidence timestamps. If a save or conflict error appears, follow the recovery instructions rather than assuming your work was saved.

## 13. Change appearance, export content, and open Help

### Appearance

- Activate Settings or say “Open settings” to find Paper, Night, and Sepia themes. A direct command such as “Switch to night mode” also works in the supported context.
- Press Escape or say “Close settings” to close the panel.
- On the lecture screen, use Presentation mode controls or say “Switch to accessibility mode” or “Switch to study mode.”
- Where display controls are available, use “Make the text bigger,” “Make the text smaller,” or “Turn on high contrast.” Contrast is a toggle; repeating the action can turn it off. Check the final state.
- Browser zoom shortcuts such as Ctrl with plus, Ctrl with minus, and Ctrl+0 can also help.

### Export

Select the correct moment and depth, then say “Export this as a PDF” or “Download the audio.” You can use the corresponding visible controls when available.

Audio must be available before you can download it. If a browser or Windows save dialog opens, complete that dialog separately with the keyboard and NVDA.

### Help dialog

Saying “Help” speaks a short command guide. To open the full Help dialog, focus the **Help button** and press Enter.

Focus moves inside the dialog. Use Tab and Shift+Tab to move through its controls, then Escape or Close to dismiss it. Talk and background shortcuts are blocked while it is open.

Audio paused for Help does not automatically resume when Help closes. Use playback controls to continue. If a voice note is recording, stop it with Alt+M before opening Help.

## 14. Understand spoken feedback

Synapse can introduce screens and speak some status and error updates. You may therefore hear both Synapse and NVDA.

- **Synapse speech:** Assistant replies, questions, screen introductions, and some status messages. If backend speech fails, it can fall back to available system speech, then to the screen reader’s live region.
- **NVDA speech:** Focused headings, button names, radio selections, dialogs, text, and accessible live updates.
- **Media playback:** Teacher audio, explanation audio, and recorded notes use their own playback controls.

A stored screen-reader-only feedback preference is supported, but there is currently no ordinary Settings or voice selector for changing it.

When NVDA says “selected,” “pressed,” “expanded,” or “disabled,” it is reporting a control’s state. A successful action does not always produce a separate success sentence. Read long content from the page rather than relying on short assistant replies.

## 15. Troubleshoot common problems

| Problem or message | What to do |
| --- | --- |
| Alt+T does nothing | Close Help, check Talk availability, focus the app, and try Enter on Talk |
| Space behaves unexpectedly | Check whether you are in a text field and which NVDA mode is active; use Alt+T for voice and Enter for buttons |
| Microphone permission blocked | Check site permissions and Windows Settings → Privacy & security → Microphone, including applicable desktop app access; retry |
| No microphone found | Connect or enable the correct device and check the Windows input selection |
| Microphone busy | Release the microphone in other recording or calling apps; stop any voice note |
| “I didn’t hear anything” | Wait for the tone, speak clearly, and check microphone selection and input level |
| “I didn’t catch that” | Speak a slightly fuller, clearer phrase |
| “Please answer in English” | Speak the command or offered option in English |
| “That isn’t available on this screen” | Ask Where am I or Help; open the correct screen or use its visible control |
| “Which one?” | Give a number from the current list, such as “The second one” |
| “The list has changed” | Read the updated list and select the item again |
| “There is nothing to confirm” | There is no pending confirmation; select the intended action first |
| Server unavailable | Use available keyboard controls, check the backend connection, and retry voice after service returns |
| Explanation audio does not play | Use Alt+R to read text; try Listen or Retry when offered, or Resume |
| Explanation seems to reference the wrong moment | Check the title, timestamp, and depth; select the right moment before requesting again |
| Recording did not save | Read the draft or Retry prompt and verify saving before closing the app |
| Teacher or NVDA is audible during capture | Use headphones, Ctrl for NVDA, and Pause for teacher playback |
| “Sorry, that didn’t work” | Check the target and prerequisites, try its visible control, and read any error text |

Recognizing an instruction and completing the operation are separate steps. After hearing “Okay,” check the actual screen, selection, or playback.

## 16. Follow a complete study session

### Screen reader and keyboard workflow

1. Start NVDA and open Synapse.
2. Sign in and choose Accessibility mode on Welcome.
3. On Home, press Alt+S, enter `electric fields`, and search.
4. Read the results and activate the correct result’s open control.
5. When the lecture opens, press Alt+P to hear the teacher.
6. When an explanation question appears, Tab to Listen and press Enter.
7. To read text, pause audio, press Alt+R, and use NVDA reading commands.
8. Answer Yes if the explanation was clear, or No to choose another offered depth.
9. Press Alt+M to record a note, speak after the tone, then press Alt+M again. Check the saved message.
10. Press Alt+P to continue. Use Bookmark when you want to save an explanation for later.
11. Return to Library, Bookmarks, or Voice Notes to read or listen to your saved work.

### Combined voice and screen reader workflow

Start each new command with Alt+T. If Synapse has already started listening automatically, answer after the tone. Use the keyboard for passwords and controls without a voice action.

1. “Go to the home screen.”
2. “Search for lectures about electric fields.”
3. “Read the results.”
4. “Choose the first result.”
5. “Start learning.”
6. After the lecture opens, say “Play the lecture.”
7. When offered an explanation, answer “Listen” after the tone.
8. After the explanation, answer “Yes,” or “No” followed by an offered depth such as “Detailed.”
9. Say “Bookmark this explanation” and check the resulting bookmark state.
10. Say “Record a voice note,” speak after its recording tone, and press Alt+M to stop.
11. After the saved message, start a new voice turn and say “Play the lecture.”
12. Later, say “Open bookmarks,” then “Read saved item one.” Read the opened text with NVDA.

## 17. Current limitations and references

- Voice input currently expects English.
- Type passwords. Use the relevant interface controls for sign-out, processing cancellation, and opening the full Help dialog.
- Commands depend on the active screen and state. Empty lists, unavailable explanations, and changed confirmations need to be addressed before an action can proceed.
- Library, bookmarks, and recordings depend on this local installation and account storage. Do not assume automatic synchronization to another computer. Older browser data can have separate import or recovery options.
- Check AI explanations and quiz answers against the teacher and source timestamps when accuracy matters, especially numbers, equations, labels, and directions.

This guide prioritizes source behavior reviewed on November 16, 2026. Older references may describe manual follow-ups, fewer account voice actions, or a fixed five-question quiz. The current code supports auto-listen, username and Welcome actions, and spoken quiz mode and count.

Related references:

- [Voice command and announcement reference](synapse-web-voice-reference.md)
- [Complete manual verification guide](synapse-web-manual-test-guide.md)
- [Keyboard shortcut definitions](../synapse-web/src/lib/shortcuts.js)
- [Talk, Space, and auto-listen behavior](../synapse-web/src/voice/VoiceControl.jsx)
- [Account form](../synapse-web/src/components/AccountForm.jsx)
- [Quiz voice handlers](../synapse-web/src/components/QuizScreen.jsx)
