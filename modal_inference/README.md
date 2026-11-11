# Synapse Qwen3-VL Modal profile

The initial single-L4 profile is defined once in `modal_inference/config.py`:

- `Qwen/Qwen3-VL-8B-Instruct-FP8`
- vLLM `0.19.1`
- one Modal NVIDIA L4 and one scheduled sequence
- 12,288 total model tokens
- 8,192 default and server-wide maximum completion tokens
- 0.90 vLLM GPU-memory utilization
- one image, no video, with a 1,344-pixel client-side maximum dimension
- temperature 0.1

These are configuration targets, not measured performance claims. A 12,288-token
context budget does not prove that model weights, KV cache, activations, image
processing, and vLLM workspaces fit in 24 GB. A live Modal startup must succeed
before this profile is considered deployment-validated. The 8,192-token
completion setting remains bounded by the 12,288-token total context and must be
validated with the exact processor budget before deployment.

## Context accounting

The remote adapter lazily uses the official Qwen3-VL Hugging Face processor to
expand the chat template and image tokens before dispatch. Processor files are
local-only by default:

```text
SYNAPSE_QWEN_PROCESSOR_LOCAL_FILES_ONLY=1
```

Set the value to `0` only when processor download is intentionally allowed. The
adapter fails closed by default if the processor is unavailable; the snapshot
records that exact accounting was not possible and never substitutes character
counts. This policy is controlled with:

```text
SYNAPSE_VLM_REQUIRE_EXACT_CONTEXT_BUDGET=1
```

Setting that value to `0` is an explicit compatibility fallback: requests stay
bounded to one resized image and the output ceiling, but exact context
feasibility is then unverified and recorded as such.

Generate the five-case offline feasibility report without inference:

```text
python modal_inference/token_budget_report.py
python modal_inference/token_budget_report.py --allow-download
```

## Live validation

Deployment and live inference are intentionally separate from unit tests. After
authorized deployment, validate server startup, a single structured request,
finish reason, prompt/completion counts, latency, and peak GPU memory from Modal
and vLLM diagnostics. Never treat valid JSON alone as extraction completeness.
