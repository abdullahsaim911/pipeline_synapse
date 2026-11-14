# Cross-video explanation accuracy review

This work targets all lectures, not only the energy-field counterexample. Qwen3-VL-8B remains unchanged. The workspace selects local Mistral for explanation text using `SYNAPSE_EXPLANATION_PROVIDER=ollama`, as requested by the user. Gemini was evaluated but is not selected for explanations. Qwen3-VL-8B remains unchanged. A downloaded Qwen3-4B-Instruct text-model comparison did not establish a clear enough improvement to use it.

## Evaluation scope

- Inventory: 19 saved videos and 201 available intervention frames.
- Diverse sample: one educational moment per video plus remaining classifier-hint types, totaling 24 cases.
- Source-image expectations and risk checks: `qa/fixtures/lecture_quality_reviewed.json` (24 manually inspected source images; some teacher-connection checks remain pending).
- The sample contains ten reviewed categories. Ordinary numerical tables and maps were absent from this particular sample, so explicitly authored synthetic references were added for CT-05 and CT-10. These are not saved-video successes.
- No exact percentage of general accuracy is claimed. Successful extraction is a technical result; it does not establish educational truth.

## General repairs retained

- The native serving `/tokenize` endpoint counts the same image-first message used for generation. This removes local Transformers/PyTorch processor loading and its large RAM cost. A real two-request check used approximately 43 MB client RAM, loaded neither library, and matched preflight/inference input counts at 907/907.
- vLLM 0.19 processor cache was disabled after real logs demonstrated that token-count requests populated its frontend cache without populating the generation receiver cache, causing HTTP 500 `Expected a cached item` assertions. Final deployment tag: `explanation-quality-native-budget-20261113`.
- Sequential region/element/relation IDs and mild overview repetition control address duplicate cross-region refs and runaway repeated records. Focused exact reads do not receive a repetition penalty.
- Invalid integer-grid boxes with inverted/zero area are discarded with diagnostics; no coordinate swapping or invented crop bounds. Observed content and textual positions remain usable with partial status.
- Focused checks explicitly receive their JSON contract and distinguish character legibility from scientific interpretation. They verify a named source record and prioritize consequential expressions over incidental labels.
- Exact-text disagreements retain both raw readings in the audit. Neither conflicting literal is promoted as verified. Rejected text is excluded from labels, descriptions, category-detail duplicates and spoken-math hints.
- Selected descriptions are nested with their elements and regions in the speech prompt, avoiding duplicate inventories and distant joins. All modes pack against the same worst-case context reserve, preserving the same selected source records.
- A cache join must not clear another request?s active publication lease. A deterministic two-caller test covers files becoming available before the owner finishes publication.

## Real results and limits

The first native-count experiment exposed a serving-cache defect; its failures are retained as failed artifacts. After repair, the 24-case run produced 19 usable extractions on first attempt. Targeted repairs/retries subsequently produced usable extractions for all five failed cases. These are recovery counts, not an accuracy rate. Individual runs record source hashes so experiments from different revisions are not silently treated as one controlled benchmark.

The source-image review still finds important semantic failures. Examples include missing vector marks, an incorrect hand-gesture description, a cursor identified as an oxygen atom, misread small book/document labels, omitted dense-layout details and unclear teacher-claim interpretation. Mistral also confuses angular momentum with angular velocity in one script and incorrectly endorses a displayed misconception that the teacher is refuting. These remain failed acceptance cases.

| Case | Extraction review | Finding |
| --- | --- | --- |
| 1 | needs_repair | Vector arrow marks missing from exact angular-momentum notation. Hand is described as thumb down although the visible thumb points right. Indexed moments of inertia incorrectly asserted to differ; teacher calls wheels identical. |
| 2 | core_facts_retained_with_limits | Formula and example are present, but representation-only changes were labeled disagreements. Instructional question must not be turned into a teacher counterclaim. |
| 3 | needs_repair | Invented or misread book title The Art of Focus; observed book header is Deep Work. Book headings and mug/hand arrangement broadly retained. |
| 4 | needs_repair | Invented single fixed axis/tick scheme conflates changing coordinate origin. A focused reading changed formula grouping, which cannot be promoted as verified. |
| 5 | core_facts_retained_with_limits | Main first-law text and portraits retained. Sketch leader/text target direction needs comparison with original image. |
| 6 | needs_repair | Yellow cursor/highlight incorrectly labeled an oxygen sphere in the molecular model. |
| 7 | core_facts_retained_with_limits | Repeated identical labels truncated the baseline output. With overview repetition control, source completed in 1670 tokens. Recovered source retains action/reaction equation; source quality still needs endpoint and role review. |
| 8 | needs_repair | Missing two readable time labels after focused checks failed. Dense left repeated-panel grid omitted, damaging layout reconstruction. Classifier table label should not establish actual table cells. |
| 9 | needs_repair | Reaction coefficients and proton/neutron counts retained. Carbon isotope notation omits displayed atomic number 6. |
| 10 | needs_repair | Duplicate e_lines IDs across three regions rejected in the baseline. Sequential-ID prompt recovery produced usable evidence. Inward/outward direction and unlettered small magnet remain image-level acceptance checks. |
| 11 | core_facts_retained_with_limits | Invalid box rejected baseline. Current decoder discards only inverted crop anchors and reports partial evidence without guessing geometry. Recovered extraction retains displayed calculations; exact direction/value review remains required. |
| 12 | core_facts_retained_with_limits | Core rhombus/triangle dimensions and perimeter working retained. Exact formulas source only a region, so focused checks did not run. |
| 13 | core_facts_retained_with_limits | Reactant/product masses and paired totals retained in their visible order. Printed formula references are unanchored, so no independent crop confirmation. |
| 14 | needs_repair | Chemical label C1a is an unsupported/misread literal in a small decorative document. The person/document/magnifier layout is retained. |
| 15 | needs_repair | Both kinetic-energy formulas retained. A small marker is named with an invented character; colors of equation text also incorrect. |
| 16 | core_facts_retained_with_limits | Both columns and the displayed misconception are retained. Explanation must explicitly use the teacher counterexample to avoid endorsing the misconception. |
| 17 | needs_repair | Label inventory retained but central tissue boundary, cell groups and human-arm magnification relation are underdescribed. |
| 18 | needs_repair | Position/velocity topics retained. Focused check reduced the worked velocity expression to its final result, losing arithmetic and unit scope. |
| 19 | needs_repair | Core signed momentum relations retained. Two essential formulas lost after focused timeouts; still-image descriptions overstate motion. |
| 20 | review_pending | Initial transport failures recovered. Classifier hints must not turn loose components into a completed circuit, or code-editor imagery into the teacher teaching the displayed algorithm. |
| 21 | review_pending | Initial transport failures recovered. Classifier hints must not turn loose components into a completed circuit, or code-editor imagery into the teacher teaching the displayed algorithm. |
| 22 | needs_repair | Mass 6 kg incorrectly becomes unreadable after focused failure, though source is legible. |
| 23 | core_facts_retained_with_limits | Main problem, force components and triangle captured. No source anchors for exact formulas; cannot treat valid extraction as exact-reading proof. |
| 24 | core_facts_retained_with_limits | Correctly leaves product area unfilled in current image. Teacher-predicted products should be distinguished from visible products. |

## Reproducible artifacts

- `.runtime/quality-all-lectures-inventory-1/report.json`: full inventory and sample selection.
- `.runtime/quality-all-lectures-extraction-3/report.json`: all 24 initial cases.
- `.runtime/quality-all-lectures-recovery-1/report.json` and `quality-all-lectures-recovery-2/report.json`: isolated recovery attempts; original failures remain.
- `.runtime/quality-cross-video-extraction-review.json`: source-image review findings.
- `.runtime/quality-cross-video-synthesis-pilot-1/`: Mistral, all three modes on three videos.
- `.runtime/quality-cross-video-mistral-standard-1/`: wider Standard-mode checks.
- `.runtime/quality-cross-video-qwen-text-pilot-1/`, `quality-cross-video-qwen-text-pilot-2/`, and `quality-cross-video-short-prompt-probe-1/`: isolated text-model experiments, not application-model changes.
- `.runtime/quality-native-budget-live.json`: native-count/inference identity check.
- `qa/fixtures/reference_table.json` and `reference_map.json`: explicitly synthetic category-gap fixtures.

All evaluations preserve shared lecture caches and personal saved versions. No broad data deletion, database migration or bulk rewrite was performed.

## Remaining acceptance work

1. Fix extraction semantics without guessing missing structure or treating second readings as authoritative.
2. Improve scientific explanation and explicit teacher-counterclaim handling. Prompt changes and the tested alternative text model have not yet demonstrated reliable accuracy.
3. Re-run a frozen revision across the cross-video set and all three modes; retain paired per-case reviews against the original image and speech.
4. Expand beyond the 24-case sample and synthetic category gaps. Passing this sample alone cannot establish accuracy on every future video.

## Final verification in this continuation

- 467 core tests passed (plus 9 subtests): `.runtime/quality-core32.log`.
- 279 API tests passed (plus 15 subtests): `.runtime/quality-api30.log`.
- The wider Standard-mode run completed all 18 selected additional cases; all
  returned technically complete scripts. Their manual review is saved in
  `.runtime/quality-cross-video-synthesis-review.json`. Technical completion is
  not semantic acceptance. The three earlier videos were evaluated in all modes.
- Explicitly synthetic table and map extractions were usable. The table's three
  text-mode scripts still omitted central numerical comparisons and invented
  uncertainty. This is a failed acceptance result, not category coverage proof.
- A live exact-read check retained the integration-by-parts formula but initially
  marked a Unicode/LaTeX spelling difference as a conflict. A narrow spelling
  comparator now recognizes conventional commands and token-separating whitespace,
  while preserving signs, brackets, vector marks, subscripts, units and identifiers.
  It does not solve equations or use algebraic equivalence. Regression tests cover
  meaningful near-miss operators, grouping and units.
- A Qwen3-4B text-only audit probe detected the incorrect opposite-charge
  repulsion claim and unsupported pole configuration in a saved bad script.
  Its suggested corrections also inferred additional field behavior, so neither
  those corrections nor a model audit was added to production without validation.
- An isolated text-only probe on the existing remote Qwen3-VL-8B model completed,
  but retained unsupported details from the extracted source. It did not establish
  a production-model switch. Both production model identities remain unchanged.
- Current generation cache version: `2026-11-13-grounded-v4`. Old shared artifacts
  regenerate on request. Personal saved versions remain immutable.

The task is not complete: reliable scientific reasoning and visual reconstruction
across all videos have not passed semantic acceptance. The earlier 2?4 hour estimate
was too optimistic given the demonstrated model and source-quality failures.

## Configured cloud comparison and selected explanation provider

The existing Gemini integration for Ask/Quiz was configured and reachable. An
isolated table test preserved every row value, header and unit, unlike the local
Mistral scripts. The same 24-case cross-video set was then run in all three modes
using the existing `gemini-3.5-flash-lite` setting. This does not change the VLM.
The initial burst hit HTTP 429 rate limits; failed requests remain recorded and
paced recovery evaluations distinguish provider availability from semantic quality.

Compared cases materially improved scientific reasoning: the mass-conservation
reaction was explained without carbon dioxide being a reactant; tension-vector
components replaced the original vector without incorrectly deleting a component;
and the force/motion lesson was described as a teacher counterexample rather than
endorsing the displayed misconception. These are reviewed examples, not a claim
of zero errors on every video. Incorrect visual source facts, such as the cursor
in the triglyceride illustration, can still propagate and remain unresolved.

Production integration reuses `TextGenerationProvider`, `GenerationSettings` and
the existing Gemini provider. `LLMSynthesizer` now accepts that provider explicitly;
all three modes use the same evidence/context path. The orchestrator skips local
Ollama readiness/prewarm when cloud explanations are selected. Cache manifests
include the selected text provider and model, so changing either invalidates shared
generated artifacts. No silent local fallback is used after a cloud error.
Normal completion is required before text/audio publication; missing completion,
length limits, safety stops and provider failures are rejected. Tests cover the
cloud production path through persistence/TTS/cache plus the unchanged remote VLM.

`LLM_PROVIDER` for Ask/Quiz remains independent. To select the older local
explanation path explicitly, set `SYNAPSE_EXPLANATION_PROVIDER=ollama`; Gemini uses
the existing `GEMINI_API_KEY` and `GEMINI_MODEL`. No key was printed or changed.
Cloud calls remain subject to the configured account's quota and availability.

One unrelated archival test in `tests/quiz/test_provider.py` cannot run because
`experiments/pipeline_flags_v1/config_pf.json` is absent from this checkout. The
remaining provider tests passed; no archival fixture or expected hash was invented.


### Current provider decision

The user requested Mistral rather than Gemini. The workspace explanation setting is now `SYNAPSE_EXPLANATION_PROVIDER=ollama` (`mistral:latest`). Gemini results above are comparison evidence only. A running backend must reload its configuration before this setting takes effect. Remaining Mistral semantic errors and visual-source errors are unresolved; switching back does not establish accuracy.
