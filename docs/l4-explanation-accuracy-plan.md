# L4 explanation accuracy plan

Date: 2026-11-14. Status: implementation in progress; isolated worker deployed, activation pending live acceptance.

## Decision and scope

Keep `Qwen/Qwen3-VL-8B-Instruct-FP8` for visual extraction. Use Mistral AI's
`mistralai/Ministral-3-14B-Instruct-2512` in official FP8 precision for explanations on a separate Modal
worker, preferring L4 with A10G fallback. No Gemini for explanations. This changes
the old Mistral 7B checkpoint to a newer 14B instruction model in the same vendor
family; it is not merely moving the existing model to another GPU.

Use one candidate and one bounded pipeline improvement. Do not build a collection
of agents, a new segmentation service, or a model-comparison framework. Deployment
and region checks address different problems: hosting provides resources and
availability; better observations and reasoning training target content accuracy.

## What the implementation already does

The following code was inspected before writing this plan:

| Owner | Existing behavior | Gap relevant to this work |
| --- | --- | --- |
| `vlm_engine/snapshot_engine.py` | Schema-guided full-frame extraction; regions, elements, positions, hierarchy, exact content and relationships; diagnostics; calls focused refinement on the Modal path. | A valid schema does not establish visual correctness. Missing details may never reach refinement. |
| `vlm_engine/evidence_refinement.py` | At most three checks; magnified, bounded crops; radial arrow geometry and precision-sensitive text; audit records and explicit disagreement. | Target selection is narrow and depends on overview-generated descriptions and anchors. Non-radial arrows, small ordinary labels, and some structural relationships are not checked. A failed verification can remove otherwise legible original content. |
| `visual_evidence/assembler.py`, `models.py`, `notation.py` | Canonical structure, validated references, box checks and conservative notation equivalence. | Preserve these trust boundaries while adding check types; do not silently fix conflicting source content. |
| `synchronizer/synchronizer.py` | `reconcile_visual_evidence` builds a conservative transcript-coverage plan over atomic visual facts, with a default 64-fact bound. | This is a coverage matcher, not a scientific reasoning engine or proof of teacher intent. |
| `synthesizer/llm_synthesizer.py` | Five explanation components, 12 category directives, three modes, uncertainty rules, attention groups, whole-record packing and aligned teacher context. | The current prompt is substantial. Local Ollama has context checks, but the remote text-provider path does not receive equivalent preflight budgeting. Old output ceilings of 768/1024/1536 tokens are unsuitable to reuse blindly for a reasoning model. |
| `synthesizer/configuration.py`, `llm_interface/providers.py` | Explanation-owned selection and a text-provider contract with result, usage and finish metadata. | Selection currently supports Ollama and Gemini; add the remote Mistral implementation here, not a second orchestration pipeline. |
| `orchestrator/orchestrator.py` | Extraction, reconciliation, synthesis, audio, cache reuse and cancellation/ownership checks; skips local readiness when a text provider is supplied. | Both synthesis entry points and predictive warmup must consistently use the new profile. |
| `modal_inference/config.py`, `launch_config.py`, `app.py` | Authenticated Qwen serving, pinned vLLM, model caches, scale-to-zero, one worker and L4/A10G scheduling. | Qwen-specific quantization and multimodal flags cannot be copied unchanged into a Mistral worker. |
| `explanation_cache.py`, API preparation/service code | Provider/model-bound artifact manifests, matching text/audio pairs and request leases. | Include the new model and changed quality version; retain atomic publication and cancellation behavior. |
| Existing evaluation scripts and `qa/fixtures/lecture_quality_reviewed.json` | Saved-video inventory, isolated outputs, source review and three-mode evaluation. | Add the remote provider to existing scripts; no duplicate benchmark runner. |

Current workspace explanations remain configured for local `mistral:latest` until
the new worker passes acceptance. This plan changes no application setting.

## Chain-of-Region: what it means here

The ICLR 2025 Chain-of-Region paper decomposes scientific diagrams and assembles
visual metadata, using OpenCV region identification and shape detection/merging.
Our region hierarchy and crop checks are related ideas, not a complete
implementation of that paper. Its results do not establish accuracy on all lecture
categories, video frames, or our specific Qwen model.

Adopt a bounded coarse-to-fine adaptation:

1. Keep one whole-frame overview so panel positions and the global teaching
   structure remain available.
2. Select up to three consequential checks across distinct regions, using actual
   source boxes. Prioritize ambiguity that could change the meaning: arrowheads
   and endpoints, formula scope/signs, table header-to-value associations, small
   essential labels, and cross-region links. Do not spend all checks on similar
   incidental numbers. A category hint alone is never evidence.
3. Ask narrow observation questions on crops. Exact reads should first transcribe
   visible content without seeing the candidate answer, then compare in code;
   this reduces anchoring to an incorrect overview. Preserve a block's full
   expression, code indentation, and neighboring headers or endpoints needed to
   understand it.
4. Merge only records identified by stable source IDs. Translate any crop-local
   coordinates back into full-frame coordinates before use. Keep both endpoint
   anchors when checking a relationship; a crop that cuts off an endpoint cannot
   establish a connection or direction.
5. Preserve conflicts rather than choosing the more plausible reading. Transport
   errors, timeouts and malformed responses mean verification failed, not that
   the source is unreadable. Keep original evidence with an explicit unverified
   status/audit; genuine visual disagreement remains ambiguous. Never silently
   promote an unverified literal to independently verified evidence.

Extend `evidence_refinement.py` and its existing models/tests. No recursive region
splitting, arbitrary quadrant crops, majority voting or new detector dependency
in the first delivery. A conservative NumPy image check can locate a narrow,
high-contrast divider spanning at least 85% of frame height. If it establishes a
nonempty panel not covered by the overview, one of the three existing check slots
is used to describe that panel; its content remains partially supported. If an essential item has no trustworthy anchor, record
the limitation and use a valid parent-region box only when the named target can
be uniquely identified within it. Otherwise do not guess. This bounded design
cannot guarantee recovery of content omitted entirely by the overview.

## Mistral worker and integration

1. Add a separate Modal text application and launch configuration. Reuse fitting
   deployment/security conventions and persistent weight caching, while keeping
   Qwen's deployed application unchanged. One active sequence and one container;
   L4/A10G preference; scale-to-zero outside demo preparation.
2. Use official FP8 weights with BF16 computation and a 12288-token total context.
   Weight size alone does not establish fit: KV cache and runtime overhead require
   live validation on each allowed GPU. Serve text-only and avoid allocating vision capacity using the
   pinned runtime's supported mechanism. Do not assume maximum advertised context
   fits on an L4. If this profile cannot start safely, resolve memory usage before
   activation; do not silently select a different checkpoint or GPU tier.
3. Use the model's supported Mistral tokenizer/template with a low generation
   temperature (0.1). This Instruct checkpoint does not use a reasoning parser or
   private-thinking template. The isolated worker uses vLLM 0.19.1 with
   Transformers 5.5.3 pinned after finding an incompatible latest dependency. Do not add tool-calling features: this use case needs none.
4. Add a Modal Mistral `TextGenerationProvider` adapter returning final answer
   text, finish status and token usage. Keep reasoning output separate from the
   final answer; never send it to TTS. Missing final text or length termination is
   failure, even if reasoning content exists. Reuse typed safe errors, bearer
   authentication, bounded network retries and cancellation boundaries.
5. Extend explanation configuration and profile metadata, independent of
   Ask/Quiz. Update both orchestrator synthesis paths, warmup, evaluation CLI
   selection and cache identity. No silent fallback to Gemini or old Mistral.
6. Reserve up to 1536 completion tokens for the final explanation within the
   12288 total. This leaves more input room for teacher speech and visual records. This is a starting ceiling, not a guaranteed sufficient
   reasoning allowance or a latency target. Count the exact served chat template
   before inference; repack complete source records with existing dependency
   handling. If essentials cannot fit, fail explicitly. All modes receive the
   same essential evidence and completion allowance; mode instructions
   control the length/depth of the final explanation.

## Make the synthesis task easier without routine extra calls

Reuse the existing canonical evidence and coverage plan as the intermediate
evidence packet. Group information by meaningful region and reading order; retain
source IDs, consequential values/relationships and status. Supply the complete
aligned teacher thought and clearly separate it from visible facts.

Keep one normal Mistral generation call per requested mode. Its instructions
require the five existing components: orientation, structure, salient details,
educational meaning and lecture connection. Explicitly distinguish teacher
assertions, hypothetical examples, questions and refuted misconceptions; do not
relabel all negative sentences through a keyword rule. Ask the reasoning model
to resolve those distinctions before writing its final spoken answer, without
requesting a second free-form outline call or publishing its reasoning.

For the first delivery, preserve the synthesizer's existing plain-string return
contract. Do not require a new JSON explanation envelope, coverage declarations,
a separate review model, or an automatic content-repair loop. Those introduce
formatting and semantic-check failure modes before the model integration is proven.

The immediate before-audio check is small and belongs inside synthesis: require
successful completion, nonempty final-answer text, and correct separation of
reasoning from the answer. These extend the existing `_call_llm` response checks.
Do not expose internal reasoning if the parser fails. Reject missing/incomplete
final answers through `TextSynthesisUnavailableError`, before text persistence or
TTS. Existing source validation, MP3 validation and cache checks remain in place.

Evaluate exact values, essential coverage and conceptual correctness against the
reviewed sources in the offline acceptance run. They are not new automatic
blocking rules in the first production delivery. Correct spoken notation can
differ from literal source strings; a missing keyword is not proof of an omission.
Only add runtime content gates later when they have demonstrated adequate
precision. No automatic check described here proves scientific correctness.

## Latency, availability and data safety

- Bound visual calls to one overview plus at most three crops; reuse saved valid
  evidence across modes. One normal text call, no new content-review/repair call.
- Keep existing bounded network retries and apply an overall request
  deadline. Audit the existing 120-second synthesis timeout and API lease behavior
  against live timings; keep ownership checks before publication and after calls.
- Record cold startup, warm extraction, crop time, reasoning/final generation,
  audio generation, queue wait and cache reuse separately. Do not promise a
  seconds-per-answer figure before measuring the selected L4 worker.
- Warm both workers before a scheduled demo and retain them for the demo window.
  A warm worker incurs cost; fallback scheduling reduces capacity risk but cannot
  guarantee instantaneous allocation.
- Bump the relevant quality version and bind the exact provider/model revision,
  prompt/evidence policy and input hashes. Regenerate stale shared artifacts on
  request. Preserve personal saves and avoid bulk data rewrites or migrations.
- Keep model adapters, credential validation and server tests offline where
  possible. Never log credentials or expose raw reasoning in app responses.

## Delivery order and smallest sufficient proof

1. **Worker proof:** start the selected model on one allowed 24GB GPU; verify
   auth, actual memory headroom, token counting, reasoning/final separation,
   normal completion and a complete spoken answer. Verify the alternate GPU
   before advertising that fallback as working. Keep application selection local.
2. **Pipeline integration:** implement the provider, shared budgets, bounded
   region refinements and final-response validity checks in the responsible owners.
   Add focused regression tests for the new behavior, not a second framework.
3. **Fast content gate:** use six existing reviewed examples spanning arrows,
   formulas, tables, teacher counterclaims, vector decomposition and mixed
   speech/frame content. Require zero critical factual errors and preservation
   of essential details. Compare against saved baseline outputs, not new model
   candidates. A failure blocks default activation; do not rerun until it looks good.
4. **Breadth gate:** run the existing 24-case/19-video sample in all three modes
   from one frozen source revision, plus the clearly marked synthetic table/map
   fixtures. Review the actual image and aligned speech for each case. Track
   visual-source errors separately from synthesis errors, five-component coverage,
   exact details, uncertainty, cross-mode consistency and latency. Essential facts
   must agree across modes; no critical error is acceptable in the acceptance set.
   Add a few previously unused moments for a generalization check, without tuning
   prompts to their answers. This still does not prove every future video correct.
5. **App gate:** verify one fresh video-to-explanation-to-audio run, all modes,
   matching cache reuse, stale-model regeneration, duplicate-request joining,
   cancellation, remote failure and recovery. Check existing API/lease regression
   tests affected by the changes; do not rebuild unrelated application features.
6. **Activation:** select the remote Mistral profile, restart the backend, confirm
   the actual served model and cache manifest, warm the workers and repeat an app
   smoke test. Retain the explicit local profile for rollback; never change to it
   silently during a failed demonstration.

The definition of done is the gates above passing with recorded output and timing
evidence. No accuracy percentage, delivery duration or latency guarantee is
established by this planning exercise. If the selected model still fails with
verified inputs, report that limitation rather than adding unbounded retries.

## Research and what it supports

- [Official Ministral 3 14B Instruct model card](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512):
  current model and official FP8 checkpoint. Hardware fit and Synapse accuracy
  require separate live checks. The 8B Reasoning candidate was evaluated first
  and rejected for slow generations and intermittent missing final answers.
- [vLLM Ministral Instruct recipe](https://github.com/vllm-project/recipes/blob/main/Mistral/Ministral-3-Instruct.md):
  Mistral loading/tokenization, FP8 serving and memory/context controls.
  Published examples are not proof of our L4 configuration's memory or latency.
- [Chain-of-Region, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/464012c83279e19be4cd42c25f341c92-Abstract-Conference.html):
  region decomposition and metadata assembly for scientific diagram analysis.
  We adapt that principle; the first delivery does not reproduce its full method.
- [Official Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL):
  resolution/pixel-budget controls and spatial grounding conventions. More pixels
  alone do not resolve the arrow counterexample observed in our saved tests.

The initial planning turn changed no runtime behavior. Implementation has now begun; see the execution record below.

## Flow compatibility clarification

After tracing `synthesize_intervention_explanation`,
`generate_explanation_from_evidence` and `_generate_enhanced_explanation`, the
first-delivery plan is deliberately narrower than the earlier proposal for a
structured explanation envelope and content-repair call.

The current flow remains:

`frame -> Qwen snapshot/refinement -> transcript reconciliation -> prompt -> text generation -> text checkpoint -> TTS -> valid MP3 -> generation manifest -> API database commit -> ready`

The clearer evidence packet is only formatting of the existing validated visual
records, coverage plan and aligned transcript inside prompt construction. It is
not another external service, model call, database object, or public API field.
For example, a displayed claim that force is needed to keep moving stays a
visual fact; nearby teacher speech explaining why that claim is wrong stays a
separate teacher statement. The prompt instructs the model to explain the
teacher's correction instead of treating the displayed claim as the conclusion.
The format does not itself classify the teacher's meaning reliably.

The final-answer validity check runs before the synthesizer returns its existing
string. Therefore orchestrator file paths, text checkpoints, TTS input, returned
dictionary, API schemas, database transactions, cancellation ownership and
personal-save behavior do not require redesign. An invalid answer follows the
existing typed failure path; successful answers continue through the same stages.
The new provider/parser still requires focused compatibility and lifecycle tests
before activation. This is a design constraint, not a claim of zero regression risk.


## Execution record

- Added a separate pinned Ministral 3 8B Reasoning deployment, provider adapter, exact serving-token budget, final-answer separation, and configuration/evaluation integration.
- Extended aligned teacher speech from 15 seconds per side to 30 seconds, with bounded sentence-boundary extensions to 45 seconds and no bridging of caption gaps. Prompt packing preserves whole nearby caption records.
- Extended existing crop checks to directed endpoint relationships and small source labels; reads no longer receive the overview candidate. Transport failures preserve unverified source readings instead of declaring them unreadable.
- Offline regression: 500 core tests plus 9 subtests, and 281 API tests plus 15 subtests passed before the additional remote-provider lifecycle cases.
- Initial worker startup found an incompatible latest Transformers dependency. Pinned Transformers 5.5.3 and redeployed. Live worker acceptance and app activation remain pending.


### Updated execution status

- The 8B Reasoning candidate was rejected: long generations and intermittent
  reasoning-only responses did not meet app reliability requirements. The attempted
  native thinking-budget control also required server configuration absent from
  that profile; it was removed, not left active. No third-party runtime patch or
  manual conversion of private reasoning into an answer was introduced.
- Current isolated deployment: `mistralai/Ministral-3-14B-Instruct-2512`, revision
  `29439f81c2be264d8d393273f99e7db9c0961120`, official FP8 weights,
  12288 total tokens, 1536 completion ceiling, text-only, one sequence/container,
  L4 preferred and A10G fallback. Modal deployment tag: `ministral14-fp8-l4-20261114`.
  The app name `synapse-ministral-reasoning` is retained from the first candidate
  to preserve the endpoint; it does not identify the current model variant.
- Current local application selection is still `ollama`. No Gemini explanation
  calls are made. New provider activation is pending acceptance.
- Teacher context is now 30 seconds before / 60 seconds after the frame, extending
  toward sentence boundaries to at most 45 seconds before / 90 seconds after.
  Caption gaps still terminate selection; complete records are packed nearest
  the frame first and restored to chronological order. All modes share context.
- A real region-recovery call recovered the previously omitted dense left panel.
  It is marked partial; its small details are not established as exact content.
- Current core regression: 505 tests plus 9 subtests passed. API regression:
  283 tests plus 15 subtests passed. These prove contracts, not semantic accuracy.
- An isolated full model/TTS/cache run with the earlier 8B candidate created a
  decodable 52.78-second MP3 and verified matching cache reuse. Initial end-to-end
  time was 165 seconds because Edge TTS rejected the request and SpeechT5 generated
  the audio. This does not establish the new 14B model's end-to-end timing.
- Real API startup against isolated database/media roots passed `/health` and
  served 57 OpenAPI paths. Voice/transcription/model warmup were deliberately
  disabled, so `/ready` correctly returned 503; full application readiness is
  not claimed. The isolated test server was stopped.
- The first 14B small generation completed in 7.3 seconds. It still speculated
  about unseen visual details in a sparse probe, so startup and speed are not
  semantic acceptance. The source-grounded six-case/all-mode evaluation is running
  in `.runtime/ministral14-six-case-gate1/`, followed by the table fixture.

### Acceptance decision

The 14B candidate is not activated. Early source-grounded outputs still contain a
spoken differential error and unsupported interpretations of the dense time-scale
diagram. The recovered left panel improves structural coverage but does not prove
the intended meaning of ruler markers. See `.runtime/ministral14-acceptance-review1.json`.
The current application remains on local Mistral. Further broad evaluation cannot
substitute for resolving these failed cases.


## Final implementation changes after acceptance failures

The execution notes above are chronological. The current implementation supersedes
both the plain-string-only initial proposal and the rejected rewrite loop:

- The remote Mistral call internally returns four speech fields. The synthesizer
  validates them and still returns one plain string to the existing orchestrator.
  There are no API schema or database migrations for this change.
- A conditional source review checks questionable visual interpretations. It labels
  each complete candidate sentence using stable named IDs. Code retains supported
  sentences in original order, without rewriting them. Unattributed speculative
  mappings are excluded even if the reviewer returns an inconsistent flag. This is
  at most one extra model call, not an unbounded review or repair loop.
- A model reviewer can still misjudge support and omit useful material. The content
  gate must be evaluated against the original source; valid JSON and a passing
  unit test do not prove an accurate or complete explanation.
- Exact source integral hints identify terminal differentials and correct the
  specific speech error 'with respect to du' to 'with respect to u' only when that
  differential is explicitly established in the source. Unknown notation is not
  guessed or algebraically simplified.
- The total served context remains 12288 tokens. The final answer ceiling is now
  3072 tokens to permit the complete internal response, with input headroom also
  reserved for the bounded review. Failed/incomplete output is never published.
- Native constrained decoding disables unnecessary JSON formatting whitespace.
  Because xgrammar ignores length when mixed with a character pattern, the wire
  schema uses length bounds and native forbidden speech-markup tokens; local
  validation retains the full format contract. A small repetition penalty reduces
  repeated prose. The exact template is counted before every generation.
- Ownership is checked before and after every remote generation/review so canceled
  requests cannot publish late text, audio or database rows.
- A real 14B run generated a decodable 31.67-second MP3 and verified exact cache
  reuse, recorded in `.runtime/activated-live-proof2/result.json`. It took 127.5
  seconds including the existing SpeechT5 fallback after Edge TTS rejection.
- Core proof at this stage: 520 tests and 9 subtests; the subsequent decoder-bound
  regression brings it to 521. API proof: 283 tests and 15 subtests. Focused latest
  provider/integration tests pass. Full app readiness has not been established.
- Latest source-grounded validation is `.runtime/release-six-standard/`. Activation
  must reflect its observed results; no all-video accuracy claim is justified.


## Current outcome (supersedes earlier completion estimates)

The requested accuracy goal is NOT complete. The Mistral 14B FP8 worker is deployed
and the adapter/API/text/audio/cache path is implemented, but final source-quality
acceptance has not passed. `.env` remains `SYNAPSE_EXPLANATION_PROVIDER=ollama`.
No backend was switched to the new provider. The endpoint is retained for isolated
validation; it scales to zero when idle.

A real 14B run generated a decodable MP3 and reused its matching cache. This proves
integration, not explanatory quality: review showed that filtering may omit the
actual teacher correction. Dense-diagram explanations can still infer meanings
that are not supplied. The serving decoder also occasionally returns symbols
outside its requested speech pattern; local validation rejects those responses.
Do not silently remove or substitute mathematical content to obtain a passing run.

See `.runtime/explanation-task-current-status.json` for the current acceptance
status, and `.runtime/complete-math-and-diagram-proof` and
`.runtime/release-six-standard` for reproducible failed outputs. Evaluation scripts
archive each generation/review call separately. No all-video accuracy claim or
full app readiness claim is supported. Current changes are uncommitted; unrelated
pre-existing workspace edits have been preserved.


## Latest startup and activation status

Remote Mistral is now selected in `.env`. The compact prompt, source-based integral
readout and supported-sentence review resolved specific tested failures, but do not
establish universal accuracy. Actual 14B audio generation and matching cache reuse
passed; the latest run produced 105.592 seconds of audio in 161.1 seconds including
SpeechT5 fallback after Edge TTS rejection. Brief and Detailed calls completed for
the three critical cases in `.runtime/activated-acceptance-modes`.

The official backend launcher was restarted and verified. Separate Qwen and Mistral
warmups ran concurrently; `/ready` reports both remote models plus API, voice and
transcription ready. See `docs/app-startup-and-model-readiness.md` for commands,
configuration, status semantics and verification. This supersedes historical notes
above saying that the app still selects Ollama. Full all-video quality acceptance
remains a separate, unproven claim.
