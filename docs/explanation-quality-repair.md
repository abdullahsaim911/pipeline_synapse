# Explanation quality repair - 13 November 2026

The later [cross-video review](cross-video-explanation-quality-review.md) supersedes
this report's evaluation scope and serving-profile details. It covers all 19 saved
videos through a diverse 24-case sample, records unsuccessful semantic acceptance,
and documents the native token-budget/cache repair. The older experiments below
remain historical evidence, not current success claims.

The supplied [quality framework](explanation-quality-contract.md) remains the
acceptance contract. The GPU availability repair is complete. Educational output
accuracy is **not yet demonstrated across all twelve categories**.

## Completed and retained

- Modal uses native ordered fallback **L4 -> A10G**, with one GPU/container and the
  existing single-worker limit. Both choices have 24 GB VRAM; A10G uses vLLM's
  FP8-weight Marlin kernel with 16-bit activations. No unrestricted `any` or
  premium GPU fallback was added. Authentication, persistent caches and scale-to-zero
  are retained. The selected worker stays warm for the existing five-minute idle window.
- Live logs confirmed allocation of NVIDIA A10G, successful Qwen startup, HTTP 200
  inference, and the configured 12,288-token context. First startup took 124 seconds;
  subsequent single-frame evaluations took approximately 15-31 seconds. Availability
  fallback does not eliminate cold startup. Existing backend startup warmup remains enabled.
- Verified local model identity: `mistral:latest` is **7.2B Q4_K_M**, not 4B. CPU-only
  loading failed on host/KV memory allocation. Generation initially succeeded with 24 GPU layers, but subsequent live runs
  exhausted memory. Memory-mapped loading with 20 GPU layers succeeded.
  The local setting is now 20 layers with an 8,192-token context. Memory mapping is enabled and batch size is 128
  for warmup and generation to reduce allocation pressure. The launcher no longer
  shadows the `.env` GPU setting with a CPU-only environment assignment.
- Synthesis gets a bounded, timestamp-aligned nearby passage, independent of OCR
  matching. Adjacent caption fragments are joined into readable speech. Explicit
  transcript ends/durations are respected. All depths get the same essential context;
  temporal proximity is not treated as proof of teacher intent.
- Synthesis requires actual visual structure, grounded educational meaning and a
  supported lecture connection. A deterministic region guide keeps selected labels
  and descriptions adjacent to their positions. Generation temperature is 0.2.
- Prompt packing counts the installed local model vocabulary and reserves completion
  capacity plus conservative template/system overhead. Unavailable vocabulary uses
  a conservative UTF-8 byte bound for the supported byte-fallback Mistral tokenizer.
  Whole source records and dependencies are selected; oversized indispensable input
  fails explicitly. Output ceilings are Brief 768, Standard 1,024, Detailed 1,536.
  These are ceilings, not target lengths or semantic quality scores.
- Missing completion, explicit incomplete responses and length-limited output are
  rejected before publication. Provider failures retain their cause. The real Brief
  evaluation that reached 384 tokens was rejected rather than saved as completed;
  Brief's ceiling was increased while its prompt still requests concise essentials.
- Qwen's wire message is image-first, matching the official processor's preflight.
  Actual server and client input counts matched in live checks (e.g. 2,199/2,199).
- The structured extraction contract uses Qwen's integer 0-1000 coordinate grid.
  The backend converts declared coordinates to canonical 0-1 boxes. Historical
  observations without that declaration retain their old normalized contract;
  pixel coordinates are never silently guessed or clamped.
- Canonical categories include observed region/element categories; classifier hints
  remain provenance and cannot reintroduce a category rejected by the observation.
- Each generated element must carry a description. A semantic name embedded only
  in an opaque model-local ref is insufficient because canonical IDs replace refs.
- Hierarchy generation uses parent pointers instead of a redundant child inventory.
  Canonical parent/child links are made reciprocal. Containment cycles are rejected;
  legitimate flow/network cycles remain valid.
- Conflicting opposite/negated teacher relationship statements retain uncertain
  coverage rather than being suppressed as already explained by an earlier match.
- Canonical math receives exact notation and scope guidance, not an unsafe second
  expression produced by the legacy symbol replacer. Exact operators and grouping
  remain authoritative.
- One overview extraction is followed by up to three focused checks of grounded
  radial-arrow regions or exact mathematical/code/numeric content. All calls use
  the same Qwen model. Original outputs and focused checks remain inspectable;
  disagreements retain uncertainty. Transport retries remain; the isolated runner
  disables extraction retries. Region, element and relation refs use distinct `r_`, `e_` and `rel_`
  prefixes to avoid collisions with each other and the reserved `scene` target. Invalid reference graphs still fail validation.
- Versioned generation manifests bind the text, MP3 and evidence hashes. Old or
  unversioned shared caches regenerate on request; selected text and audio are
  reused together. Old snapshots are re-extracted by the extraction owner.
  Publication uses existing ownership/checkpoint guards. Personal saves are untouched.
- Prompt packing keeps selected elements and their descriptions together. Positions
  take precedence over duplicate coordinates in the speech-only projection. The
  supplemental coverage index is capped at 3,000 characters so it cannot displace
  the visual meanings it references.

## What real evaluation established

Historical inventory: 79 saved snapshots, including 34 flagged errors. Four recent
lecture groups contain 31 snapshots with 13 flagged failures. These span different
configurations and are not a production failure-rate measurement.

The energy frame at `data/Tf4A93763Bw/keyframes/frame_0004_06_46.jpg` has three
panels. Its green gravitational arrows point **inward** toward Earth. Historical
Qwen output says they emanate outward, and the historical explanation mostly
summarizes basketball dribbling instead of describing the panels.

A fixture reviewed against the actual image isolates Mistral from Qwen. Mistral
now describes the panels and connects them to the adjacent teacher passage, but
some runs still add unsupported interpretations or commentary. This is improvement,
not proof that every explanation meets the framework.

Focused real Qwen probes are particularly informative:

- Full-frame probe: incorrectly reports green arrowheads at the outer ends.
- Magnified left-panel probe: correctly reports arrowheads near the globe and
  direction toward its center.
- Full structured extraction still reverses this relationship, despite technically
  valid JSON and sufficient context. It must not be declared educationally correct.

Globally enlarged frames and same-call overview/detail views were evaluated. They
**did not reliably correct the semantic error** and were removed from production.
The final production path retains one image/call. Their artifacts remain isolated
under `.runtime/quality-*`; they are not successful quality acceptance results.

The final serving profile is deployed with tag `explanation-quality-focused-20261113`.
The local regional endpoint and 600-second request timeout match the deployment.
No credential was changed. No production migration, shared-cache regeneration or
personal saved-version modification was performed.

## Same-model focused evaluation

The user chose to retain Qwen3-VL-8B. The live extraction in
`.runtime/quality-fields-focused-2/snapshot.json` completed with that exact model.
Its overview again described the green arrows as outward. The automatic left-panel
check read arrowheads near the globe and direction toward its center. Canonical
arrow description/direction were corrected, while the conflict and original
observation remain recorded. The electric-panel check independently read outward
arrowheads. Both checks completed normally, taking about 10.9 seconds together.
The overview request took 275.6 seconds including deployment/cold startup.
This demonstrates this counterexample, not universal accuracy across categories.

A live Mistral check then exposed descriptions being omitted during prompt packing.
This was fixed by retaining element-description dependencies and reducing repeated
coverage metadata. The final live three-mode run is saved in
`.runtime/quality-fields-focused-synthesis-6/`. All responses stopped normally and
contained the panels and relevant teacher passage. However, the Detailed response
incorrectly said opposite charges repel and inferred unsupported magnet interactions.
The earlier Brief/Standard runs also showed inconsistent uncertainty handling.
**These are failed semantic acceptance results; explanation accuracy is not repaired.**
A focused same-Mistral editing probe (`.runtime/quality-review-probe.txt`) repeated
the wrong charge claim even with a direct correction instruction. It was not added
to production, because it added latency without demonstrated improvement.

An independent integration-by-parts frame preserved the visible expression
`integral 2x sin x dx` in two successful real extractions
(`.runtime/quality-math-focused-2/` and `quality-math-focused-3/`). One preceding
response used the reserved `scene` as an entity/parent; distinct ref prefixes now
prevent declaring it. A stricter experiment requiring every exact record to supply
a source ref produced an undeclared `e_2` reference and was rejected. That extra
requirement was removed; supplied refs remain strictly validated. Exact focused
checks require grounded source refs and boxes and therefore do not run when the
model omits the source anchor. This remains an extraction limitation.

Regression proof: 441 core tests and 278 API tests passed in separate processes,
with 9 and 15 additional subtests respectively. The combined process hit an
existing Windows native access violation importing pyarrow through transformers;
separate runs completed. Controlled-provider scheduling tests now stub real memory
admission so unrelated live-model pressure does not change their outcome.
The new integration tests cover stale-generation regeneration of both text and
MP3 and required re-extraction of unversioned evidence. Byte-identical current
replacement evidence can safely reuse matching current generated artifacts.

No bulk deletion, personal-version rewrite or production database migration was
performed. Shared artifacts regenerate through existing request/queue owners.

## Reproduction and artifacts

The isolated runner saves prompts, snapshots, raw text responses, completion metadata
and scripts without overwriting lecture caches. It refuses an existing output folder.

```powershell
synapse-env/Scripts/python.exe scripts/evaluate_explanation_quality.py `
  --video Tf4A93763Bw --intervention a0a2adaa `
  --snapshot qa/fixtures/energy_fields_reviewed.json `
  --output .runtime/quality-fields-new
```

For real remote extraction, omit `--snapshot` and add `--extract`. This can start a
billed GPU worker. The reviewed fixture is **not** a fresh Qwen success. This runner
does not generate or assess TTS audio. Automated category/schema tests verify code
contracts and preservation; they do not certify visual or pedagogical truth.
