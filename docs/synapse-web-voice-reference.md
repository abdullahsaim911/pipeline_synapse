# Voice commands and announcements: current implementation

Reviewed November 14, 2026 against `synapse-web`, its registered handlers, `voice/catalog/actions.json`, and English examples. This describes source behavior, not a claim that every phrase has passed a live microphone test. Recognition matches meaning; these are example sentences, not an exact-string command language.

## Operating rules

Press Alt+T, wait for the tone, speak English, then stop speaking or press Alt+T again. Alternatively hold Space on the focused Talk button and release to finish. Each follow-up answer needs a new voice turn. The app is not continuously listening and has no wake-word flow. Button states: Talk → Getting ready… → Listening… → Working… → Talk. Start/stop tones distinguish microphone capture boundaries. No-speech timeout is about 6 seconds, maximum utterance 12 seconds, silence after speech about 1.4 seconds.

Commands are limited to registered actions on the current screen/state. Item numbers refer to the current visible result/filter/page. Unsupported targets should fail or clarify. Account forms/onboarding are not a full voice-operated workflow. When Help is open, background voice control is disabled. Starting Talk stops the app's spoken reply. Teacher/explanation audio must pause before capture; paused media restores only when appropriate. Explicit pause, playback changes, and navigation must not be undone by automatic restoration.

## Important differences from the visible controls

- **Search:** “Choose the first result” selects/populates its URL and clears results. Then say “Start learning.” Clicking a result through the visible result control can start processing directly.
- **Help:** “Help” speaks a short command guide. It does not open the Help dialog. Open the dialog using the Help button with keyboard/mouse.
- **Quiz start:** “Start the practice questions” currently calls Practice with **5 questions**. It does not use the setup form's selected mode/count. To create Assessment or a chosen count, use the setup controls.
- **Quiz availability:** Back to lecture is registered in setup/results, not while answering. Next/previous are available only when that neighbour exists. Submit is registered on review, not the question page. Check/hint exist only in Practice. No dedicated voice mapping exists for flag question, reveal answer, resume an old attempt, selecting quiz mode/count, or a direct question-number jump.
- **Read versus listen:** “Read the explanation to me” means play explanation audio. Alt+R moves focus to text. “Read saved item two” opens saved text without auto-playing.
- **Notes:** “Record a voice note” is a toggle, not an idempotent start. Repeating it stops the recording and normally saves automatically. Playing the same note again is also a toggle that stops it. Explicit stop keeps the lecture paused. Save/discard draft commands exist only when a draft/save-failure prompt exists.
- **Removal:** Note deletion opens an inline confirmation. Collection removal is a risky interpreted action that can first receive a voice-layer yes/no confirmation; its handler can then remove directly. Do not expect the exact same inline prompt sequence as clicking Remove. Confirm/cancel saved removal commands also support an already-open on-screen removal prompt.
- **View controls:** Structured View requires available rows to be visible, and visual tabs require Study mode. Focus highlighting needs a focus region. Setting a hidden tab does not create structured evidence. High contrast and bookmark commands toggle state; repeating “turn on”/“bookmark this” can turn it off/remove it.
- **Global actions:** There are no registered voice commands for signing in/out, editing account credentials, opening the Help modal, cancelling a processing job, setting an exact text percentage or exact playback rate, or selecting feedback mode. Use the corresponding existing controls where present.
- **General guidance:** The spoken “Help” list is short, not exhaustive. Some handler errors use “Sorry, that didn't work.” rather than explain the exact missing prerequisite. Recognition of a phrase and completion of its operation are separate checks.

## Explanation conversation: exact choices

| Current prompt/state | Say | Effect |
| --- | --- | --- |
| Explanation ready at a visual | listen / skip | Listen now / defer and resume |
| Explanation requested or preparing | wait / resume | Keep lecture paused / continue while preparing |
| Ready notification after continuing | listen / later | Listen now / defer and continue |
| Explanation audio completed | yes / no / replay | Continue lecture (or review) / choose another depth / replay |
| Choose another depth | brief / standard / detailed, as offered; resume | Request offered version / continue |
| Failed preparation | retry, when offered; resume | Retry / continue |
| Saved version missing chosen depth | process / later | Activate processing / remain with saved content |
| Audio failure | listen, when offered; resume | Try audio again / continue |
| End-of-lecture review | listen / wait / retry, as offered; next / finish | Handle current item / next review item / end review |

You can also request: “Please explain this part,” “Give me a brief explanation,” “Give me the standard version,” or “Give me the detailed version.” The target uses current lecture/selected-moment context. Choices accepted by an active explanation question are not arbitrary global commands; “yes” outside a question/confirmation has no implicit meaning.

## How announcements reach you

1. **App speech:** Voice replies and explanation questions use backend TTS by default. If unavailable, they fall back to a polite screen-reader live region. An existing stored preference can select screen-reader-only feedback; there is no Settings/voice selector for it in this reviewed UI. Reply audio is capped at 1,500 characters; use displayed/readable content for longer material.
2. **Screen-reader announcements:** Many search, playback, note, timeline, and quiz updates update `role=status`/live regions. Without a screen reader these are not guaranteed audible. Ordinary visible statuses and changing button labels are not automatically spoken sentences.
3. **Focus announcements:** Some outcomes move focus to a heading, feedback, result, or dialog; NVDA reads that element. The app need not also say a separate success sentence.

Screen readers choose timing and wording for role/state/focus announcements. Backend errors and content replies can vary. Bracketed fields below are dynamic placeholders, not literal words.

## Voice interpretation and microphone announcements

| Event | Implemented message/template |
| --- | --- |
| Normal non-playback command succeeds | “Okay, [action phrase].” |
| Playback command | No generic Okay; operation/media/prompt provides feedback where implemented |
| Needs confirmation | “Do you want to [action phrase]? Press Alt+T and say yes or no.” |
| Ambiguous command | “Did you mean [action A], or [action B]?” |
| No | “Okay, cancelled.” |
| Cancel | “Okay.”; pending voice request cleared, current UI confirmation cancelled where mapped |
| Yes without a pending confirmation | “There is nothing to confirm.” |
| Nothing spoken | “I didn't hear anything.” |
| Clip too short | “I didn't catch that.” |
| Unrecognised response | “Sorry, I didn't understand that. Say help to hear what you can ask.” |
| Unsupported here | “That isn't available on this screen.” |
| Handler fails without a specific spoken error | “Sorry, that didn't work.” |
| Server/transcription/interpretation unavailable | “I can't reach the Synapse server, so voice commands aren't available right now.” |
| Confident non-English input | “Please answer in English.”; can append current explanation prompt |
| Unclear explanation choice | “I am not sure which option you mean. [current prompt]” |
| Missing number | “Which one? Say a number, like the second one.” |
| Missing query | “What should I search for?” |
| Missing time/direction | “How many seconds?” / “Which direction?” |
| Missing theme | “Which theme: paper, night or sepia?” |
| Missing presentation mode | “Accessibility mode or study mode?” |
| Missing subject | “Which subject?” |
| Missing category | “Which kind: graphs, equations, charts, diagrams, code or tables?” |
| Where am I | “You are on [screen description]. [N items are listed…] Say help to hear what you can ask.” |
| Repeat before any reply | “I haven't said anything yet.” |
| Question with no lecture | “Open a lecture first, then ask your question.” |
| Spoken lecture question | “Let me check the lecture.”, then answer or API error |
| App how-to | “Yes, you can do that by voice. Press Alt+T and ask me to [action].” or states the required screen |
| Unknown app how-to | “I'm not sure how to do that by voice. Say help to hear what you can ask on this screen.” |
| List changes before confirmation | “The list has changed. Please select the item again.” |
| Draft changes | “The recording draft has changed. Please select it again.” |
| Confirmation changes | “The confirmation has changed. Please confirm the current selection again.” |
| Voice turn cancelled | “Voice control cancelled.” |
| App audio starts during capture | “Voice control stopped because audio started playing.” |
| Could not pause audio | “Voice control could not pause the lecture audio, so it did not start. Try again, or use headphones.” |
| Note microphone busy | “A voice note is being recorded. Stop it before using voice control.” (first Talk press can instead stop the note) |
| Device lost | “The microphone stopped working.” |
| Other voice failure | “Something went wrong with voice control.” |

## Search, explanations, and timeline announcements

| Event | Implemented message/template or exposure |
| --- | --- |
| Search begins | “Searching.” |
| Search results | “[N] lectures found.” |
| No results | “No lectures found. Try another search term or paste a YouTube link.” |
| Empty search | “Please enter a lecture link or a search term.” |
| Valid link detected | “YouTube lecture detected.” |
| Clipboard not a recognised video URL | “Pasted from the clipboard.” |
| Clipboard blocked | “I couldn't read the clipboard. Allow clipboard access for this site, or paste with Control V.” |
| Clipboard empty | “The clipboard is empty. Copy a YouTube link first.” |
| Ready visual | “Lecture paused. [title] at [time]. The [depth] explanation is ready. Would you like to listen? Say listen or skip.” |
| Preparing | “Lecture paused. The [depth] explanation for [moment] is still preparing. Say wait to stay here, or resume to continue the lecture. I will let you know when it is ready.” |
| Explicit request | “The [depth] explanation for [moment] has been requested. Say wait to stay here, or resume to continue the lecture while it prepares.” |
| Ready after deferred preparation | Ready prompt asks “Would you like to listen now? Say listen or later to continue the lecture.” |
| Finished explanation | “The explanation has finished. Was it clear? Say yes to resume the lecture, no to choose another version, or replay to listen again.” |
| No / choose depth | “Choose [other depths]. Brief is shorter; Detailed gives a fuller explanation. Say [choices], or resume.” |
| Failed preparation | “The [depth] explanation for [moment] could not be prepared. Say [valid choices].” or backend error followed by retry/resume options |
| Audio could not play | “The explanation audio could not be played. The text is available. Say listen to try again, or resume.” |
| Saved depth absent | “The [depth] version has not been saved. Say process to prepare it, or later to keep using saved content.” |
| Saved playback fails | “Audio is unavailable. You can read the saved text.” |
| End review | “The lecture has ended. [N] explanations remain in this review. [depth/moment/status]. Say [choices].” |
| Review finished | “Lecture review finished.” |
| Skip | “Explanation skipped. Resuming the lecture.” |
| Retry requested | “Retry requested. The explanation is being prepared.” |
| Check preparation | “Checking explanation preparation again.” |
| Listening speed | “Listening speed [rate] times.” / “Already at the fastest speed.” / “Already at the slowest speed.” |
| First/last moment | “This is the first moment.” / “This is the last moment.” |
| Invalid moment number | “There is no moment [N].” |
| Bookmark | “Explanation bookmarked.” / “Bookmark removed.” |
| Bookmark before save completes | “Wait for this explanation to finish saving, then try Bookmark again.” |
| Presentation mode | “Study mode on. The lecture video is now shown.” / “Accessibility mode on. The video is hidden; the lecture audio still plays.” |
| Focus visual area | “Showing focused view.” / “Showing full frame.” |
| Evidence timestamp opened | “Jumped to [time]. Lecture paused.” |
| Timeline selection | “Selected [title], [category], at [time]. Preview updated.” |
| Timeline filter | “Showing [category/all] moments: [N].” |
| Missing category | “This lecture has no [category] moments. It has: [categories].” |
| Read moments | Numbered titles/categories/timestamps, or “There are no moments in this list.” |
| Describe selection | “Selected: [title], [category], at [time].” / “Nothing is selected yet.” |

Explanation question prompts append **“Press Alt plus T to answer in English after the tone.”** Unanswered prompts repeat 15 seconds after speech finishes, then at 30-second intervals, while the same question remains active and the app is not guarded by Help/notes. Reminder repetitions do not repeatedly move focus.

Processing status, storage errors, and preparation errors also appear in their own status/alert regions. They are state/backend-dependent, not one fixed universal announcement. The remote Qwen/Mistral warmup states are exposed by backend readiness/launcher; do not expect automatic frontend speech saying “Mistral ready.” Voice play/pause handlers do not guarantee the same explicit announcement path as every keyboard/button handler; test actual playback rather than treating Okay as proof.

## Notes, collections, Ask, and quiz announcements

| Event | Implemented message/template or exposure |
| --- | --- |
| Note recording starts | “Lecture paused. Recording starts after the tone.” |
| Note saved | “Voice note saved at [time], [duration]. The lecture is still paused.”; optional silent-warning suffix |
| Too short | “That recording was too short and was not saved. The lecture is still paused.” |
| Empty capture | “Nothing was recorded. Please try again. The lecture is still paused.” |
| Recording interrupted | “Recording stopped. Choose Save or Discard.” |
| Device lost while recording | “The microphone stopped working. Choose Save or Discard for what was recorded.” |
| Draft only in memory | “The draft is still in memory. Save or discard it before closing Synapse.” |
| Save fails | Error followed by “Choose Retry or Discard.” |
| Cancel recording | “Recording cancelled. The lecture is still paused.” |
| Discard draft | “Recording discarded.”; failure: “The local draft could not be discarded. Please try again.” |
| Recording nearing limit | “30 seconds left in this recording.” |
| Limit reached | “Maximum length of 5 minutes reached. Recording stopped.” |
| Lecture starts during note recording | “The lecture was paused again because it can't play while you record.” |
| Note playback finishes | “Finished playing voice note.” |
| Note audio fails | “The recording for this voice note is missing.” / “This voice note could not be loaded.” / “This voice note could not be played.” |
| Note delete | “Deleted [note]. [N] left.” / “No voice notes left.”; errors include note name |
| Library removal | “Lecture removed from Library. Bookmarks and recordings were kept.” |
| Bookmark removal | “Bookmark removed.” |
| Import/recovery | Import result/error text with actual counts; browser/account data determines wording |
| Typed Ask loading | “Looking for the answer...” |
| Typed Ask result | Focus/result heading: “Answer”, “I couldn't find that”, or “Not covered in this lecture”; answer/evidence readable below |
| Quiz generation | “Creating your questions. This can take up to a minute.” |
| Quiz generation cancellation | “Cancelled. No questions were made.” |
| Generated quiz cannot be saved | “The new questions could not be saved.” |
| Save recovers | “Progress saved again.” |
| Voice answer selection | “Selected [choice].” |
| Check without selection | “Choose an answer first, then check it.” |
| Read question | “Question [N] of [total]. [prompt and choices].” |
| Hint | “Hint [N]: [hint text]” |
| Quiz text size | “Text size [percentage] percent.” |
| Quiz contrast | “High contrast on.” / “High contrast off.” |
| Quiz next/previous | Focus moves to question heading; not necessarily an extra live sentence |
| Check answer/review/results | Focus moves to feedback/review/results heading, with content and score to read |

Permission errors share microphone descriptions: blocked access gives Windows microphone-permission guidance; no device says “No microphone was found. Connect or enable one, then try again.”; busy device says another app is using it or it is not responding. A visible control's native checked/pressed/expanded/disabled state is screen-reader feedback, not a separate recorded app announcement.

## Complete catalog mapping

The table below lists every current catalog action. Use the restrictions above: a catalog entry alone is not proof that its handler is available in the current state. English examples are the actual first example from the repository, with additional parameter details where needed.

| Action | Screen | Say, for example | Behavior |
| --- | --- | --- | --- |
| `navigate_home` | * | “go to the home screen” | Go to the Home screen where a new lecture can be searched for or started. |
| `navigate_library` | * | “open my library” | Open the Library, the list of explanations the learner has read or listened to before. |
| `navigate_bookmarks` | * | “open bookmarks” | Open Bookmarks, the explanations the learner saved to come back to. |
| `navigate_notes` | * | “open my voice notes” | Open Voice notes, the audio notes the learner recorded during lectures. |
| `open_timeline` | explanation | “show the timeline” | From the lecture, show the timeline of all the moments found in this lecture. |
| `start_practice` | explanation | “start the practice quiz” | From the lecture, open the practice quiz that tests the learner on this lecture. |
| `set_theme` | * | “switch to night mode” | Change the colour theme of the app to paper, night or sepia. |
| `search_lectures` | home, subject | “search for lectures about eigenvalues” | Search for lectures about a topic that the learner names. |
| `select_result` | home, subject | “choose the first result” | Pick one of the lecture search results by its position in the list. |
| `start_lecture` | home, subject | “start learning” | Start learning the selected or pasted lecture, which prepares it for explanations. |
| `explore_subject` | home | “show me physics lectures” | Browse lectures for one school subject such as physics or chemistry. |
| `continue_lecture` | * | “continue where I left off” | Continue the lecture the learner was last studying. |
| `play_lecture` | explanation | “play the lecture” | Start or resume playing the lecture video. |
| `pause_lecture` | explanation | “pause the lecture” | Pause the lecture video. |
| `seek_lecture` | explanation | “skip forward thirty seconds” | Jump the lecture video forward or backward by a number of seconds. |
| `play_explanation` | explanation | “read the explanation to me” | Listen to the spoken explanation of the current moment. |
| `pause_explanation` | explanation | “pause the explanation” | Pause the spoken explanation audio. |
| `replay_explanation` | explanation | “read that explanation again” | Play the spoken explanation of the current moment again from the beginning. |
| `next_moment` | explanation, timeline | “go to the next moment” | Go to the next moment in the lecture that has an explanation. |
| `previous_moment` | explanation, timeline | “go back to the previous moment” | Go back to the previous moment in the lecture that has an explanation. |
| `go_to_moment` | explanation, timeline | “go to moment five” | Jump to a particular moment of the lecture by its number in the list of moments. |
| `change_listening_speed` | explanation | “speak faster” | Make the spoken explanation faster or slower. |
| `toggle_voice_note` | explanation | “record a voice note” | Start or stop recording a personal voice note at this point of the lecture. |
| `toggle_bookmark` | explanation | “bookmark this explanation” | Save the current explanation as a bookmark, or remove it from the bookmarks. |
| `export_pdf` | explanation | “export this as a PDF” | Download the current explanation as a PDF document. |
| `set_presentation_mode` | explanation | “switch to study mode” | Switch the lecture screen between accessibility mode, which hides the video, and study mode, which shows it. |
| `change_text_size` | explanation, quiz | “make the text bigger” | Make the text on the lecture screen larger or smaller. |
| `toggle_high_contrast` | explanation, quiz | “turn on high contrast” | Turn the high contrast display on or off on the lecture screen. |
| `open_item` | bookmarks, library, notes | “open the first one” | Open one saved item, bookmark or voice note by its position in the list. |
| `play_item_audio` | library, bookmarks | “play the audio of the first item” | Open a Library item and play its spoken explanation right away. |
| `remove_item` | bookmarks, library | “delete the first bookmark” | Delete one saved item or bookmark from its list. |
| `ask_question` | * | “what does the axis on this graph represent” | Ask a question about a lecture or about the topic being studied. Used for anything that is a question about the content and not a command for the app. |
| `stop_speaking` | * | “stop talking” | Stop whatever the assistant is saying out loud right now. |
| `repeat_last_response` | * | “say that again” | Say the assistant's last reply again. |
| `cancel` | * | “cancel” | Forget the current request and do nothing. |
| `help` | * | “help” | Explain what the learner can ask for by voice on this screen. |
| `confirm_yes` | * | “yes” | Answer yes to the question the assistant just asked. |
| `confirm_no` | * | “no” | Answer no to the question the assistant just asked. |
| `quiz_start` | quiz | “start the practice questions” | On the practice setup, create a practice set of questions for this lecture. |
| `quiz_answer` | quiz | “option two” | Choose one answer for the current practice question by its number or letter (option 1 or A, 2 or B, ...). Does not check or submit. |
| `quiz_check` | quiz | “check my answer” | Check whether the chosen answer to the current practice question is right. |
| `quiz_hint` | quiz | “give me a hint” | Give a hint for the current practice question. |
| `quiz_read` | quiz | “read the question” | Read the current practice question and its answer choices aloud. |
| `quiz_next` | quiz | “next question” | Go to the next practice question. |
| `quiz_previous` | quiz | “previous question” | Go back to the previous practice question. |
| `quiz_review` | quiz | “review my answers” | Open the review page that lists unanswered questions before finishing. |
| `quiz_submit` | quiz | “submit my answers” | Finish the practice set and submit all answers. Answers cannot be changed afterwards. |
| `quiz_back` | quiz | “go back to the lecture” | Leave the practice screen and go back to the lecture. |
| `app_question` | * | “how do I bookmark this” | Ask HOW to do something in the app or whether the app can do something (for example 'how do I bookmark this', 'can I change the text size'). Answered from the list of actions. |
| `paste_url` | home, subject | “paste the link” | Paste the link that was copied to the clipboard into the lecture link field. |
| `skip_explanation` | explanation | “skip this explanation” | Skip the explanation that is being prepared or shown and carry on with the lecture. |
| `retry_explanation` | explanation | “try the explanation again” | Try again to prepare the explanation for this moment when it failed or is not ready. |
| `set_visual_tab` | explanation | “show me the picture” | In study mode, switch between the picture of the moment and its structured text description. |
| `toggle_focus` | explanation | “highlight the important part” | Highlight or stop highlighting the important region of the picture. |
| `export_audio` | explanation | “download the audio” | Download the explanation as an audio file. |
| `open_settings` | * | “open settings” | Open the settings panel. |
| `close_settings` | * | “close settings” | Close the settings panel. |
| `where_am_i` | * | “where am I” | Say which screen the learner is on and what is on it. |
| `filter_moments` | timeline | “show only the graphs” | On the timeline, show only the moments of one kind (graphs, equations, charts, diagrams, code, tables), or all of them. |
| `open_moment` | timeline | “open this moment” | On the timeline, open a moment in the lecture: the selected one, or the one with the given number in the list. |
| `read_moments` | timeline | “read the list” | On the timeline, read out the moments in the list with their kind and time. |
| `describe_moment` | timeline | “what is selected” | On the timeline, say which moment is selected: its title, kind and time. |
| `answer_explanation` | explanation | “answer the explanation question” | Answer the active explanation question using one of its available choices. |
| `play_note` | notes, explanation | “play voice note two” | Play or stop the user's recorded voice note by its visible list number. |
| `stop_note_audio` | notes, explanation | “stop my voice note audio” | Stop playback of the personal voice recording currently being heard. |
| `delete_note` | notes, explanation | “delete voice note two” | Show the confirmation for deleting a personal voice recording by list number. |
| `confirm_note_deletion` | notes, explanation | “confirm voice note deletion” | Confirm deletion of the voice recording named in the on-screen deletion prompt. |
| `cancel_note_deletion` | notes, explanation | “cancel voice note deletion” | Cancel the pending voice note deletion and keep the personal recording. |
| `open_note_explanation` | notes, explanation | “open explanation one attached to note two” | Open and play a nearby saved explanation attached to a numbered voice note. |
| `save_note_draft` | explanation | “save my recording draft” | Save or retry uploading the personal voice recording currently held as a draft. |
| `discard_note_draft` | explanation | “discard my voice note draft” | Discard the unsaved personal recording currently held in the draft prompt. |
| `import_local_saves` | library, bookmarks, notes | “import my browser saves” | Explicitly import browser saves and retry recoverable recordings into local backend storage. |
| `filter_saved_items` | library, bookmarks | “filter saved items by calculus” | Filter the visible saved collection by a topic or lecture title. |
| `clear_saved_filter` | library, bookmarks | “clear the saved items filter” | Clear the current saved collection search and show all matching saved entries. |
| `next_saved_page` | library, bookmarks, notes, explanation | “next page of saved items” | Show the next page of saved items or recorded personal voice notes. |
| `previous_saved_page` | library, bookmarks, notes, explanation | “previous page of saved items” | Show the preceding page of saved items or recorded personal voice notes. |
| `retry_saved_load` | library, bookmarks, notes, explanation | “retry loading saved content” | Retry reading the saved collection or voice notes after a storage request failure. |
| `set_saved_view` | library | “show recent explanations” | Choose saved lectures or recently viewed explanations within the Library screen. |
| `confirm_saved_removal` | library, bookmarks | “confirm saved item removal” | Confirm the saved lecture or bookmark named in the on-screen removal prompt. |
| `cancel_saved_removal` | library, bookmarks | “cancel saved item removal” | Cancel the pending Library or bookmark removal and keep the saved entry. |
| `read_saved_item` | library, bookmarks | “read saved item two” | Open a saved explanation as text at its lecture timestamp without starting audio. |
| `open_note` | notes, explanation | “open voice note two at its time” | Open the lecture paused at the exact timestamp of a numbered personal voice note. |
| `request_explanation` | explanation | “I do not understand this, explain it” | Request an explanation of this lecture part, choosing Brief, Standard or Detailed. Standard is the default. |
