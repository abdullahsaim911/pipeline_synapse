# Synapse Qwen3-VL Modal profile

The single-worker profile is defined once in `modal_inference/config.py`:

- `Qwen/Qwen3-VL-8B-Instruct-FP8`
- vLLM `0.19.1`
- one Modal GPU, preferring L4 and falling back to A10G, and one scheduled sequence
- 12,288 total model tokens
- 8,192 default and server-wide maximum completion tokens
- 0.90 vLLM GPU-memory utilization
- one frame, no video, with a 1,344-pixel maximum dimension
- temperature 0.1

Modal uses its native ordered GPU fallback scheduling: if L4 capacity is unavailable,
it tries A10G instead of keeping an L4-only request queued. Both options have 24 GB
VRAM. vLLM supports the FP8 checkpoint on A10G through weight-only FP8 Marlin with
16-bit activations; L4 retains native FP8 execution. Model weights and context limits
stay the same, but numerical results and latency may differ between kernels. A10G
has its own Modal rate. This avoids an unrestricted `any` request and excludes T4's
smaller memory and larger premium GPU classes. Availability and cold-start time
still apply if both GPU pools are busy.

Extraction uses one structured overview followed by up to three focused checks of grounded precision-sensitive regions, using the same model and one image per call. Bounding boxes use Qwen's integer 0-1000 grid and are converted to canonical 0-1 coordinates. Invalid crop anchors are discarded with a partial-evidence diagnostic; coordinates are never guessed.

References: [Modal GPU fallbacks](https://modal.com/docs/examples/gpu_fallbacks),
[vLLM 0.19.1 FP8 support](https://docs.vllm.ai/en/v0.19.1/features/quantization/fp8/).

These are configuration targets, not measured performance claims. A 12,288-token
context budget does not prove that model weights, KV cache, activations, image
processing, and vLLM workspaces fit in 24 GB. A live Modal startup must succeed
before this profile is considered deployment-validated. The 8,192-token
completion setting remains bounded by the 12,288-token total context and must be
validated with the exact processor budget before deployment.

## Context accounting

The remote adapter calls the serving vLLM `/tokenize` endpoint with exactly the
same image-first message used for inference. It does not load a Hugging Face
processor, PyTorch or model weights on the local machine. Counts are validated
against the configured 12,288-token context, with a 32-token safety reserve.
Invalid/missing counts or mismatched server context fail before inference.

The vLLM 0.19 image processor cache is disabled with
`--mm-processor-cache-gb 0`: a token-count request otherwise populates the frontend
processor cache without populating EngineCore's receiver cache, making a following
generation fail with `Expected a cached item`. Live repeated-image checks with the
cache disabled completed normally and matched preflight/inference input counts.

The injected local processor estimator remains available for isolated offline
feasibility reports and unit tests. `SYNAPSE_QWEN_PROCESSOR_LOCAL_FILES_ONLY`
applies to that offline tooling, not to the production remote adapter.

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
