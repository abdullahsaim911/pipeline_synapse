#!/usr/bin/env python3
"""pipeline_flags_v1 - do the optional Ask pipeline stages (hybrid + rerank, ASR clean-up) improve retrieval, and how does the
faithfulness gate score? NO language-model (Gemini) call is made anywhere in this file.

    python -B pipeline_flags_eval.py register                 # writes config_pf.json (once), then pin its hash below
    python -B pipeline_flags_eval.py verify                   # offline: pins, models, environment
    python -B pipeline_flags_eval.py run-a --out DIR          # items 1-3: ranking metrics, evidence metrics, latency
    python -B pipeline_flags_eval.py faith-pilot --out DIR    # item 4 pilot on the 17 existing answers (no new calls)

Run it with an interpreter that has sentence-transformers (the isolated experiment environment); models load OFFLINE from the cache.

Two families of metrics, both on the frozen benchmark questions and gold labels:
  * RANKING level (chunk lists, whole lecture, no relevance gate): Hit Rate@5, MRR@10 (+ the other frozen metrics) for
      R0 BM25 (frozen, reproduces the saved baseline), R1 BM25 on ASR-cleaned captions, R2 BM25+dense fused with RRF,
      R3 RRF then cross-encoder rerank.
  * EVIDENCE level (what the model would really receive from AskService, gates and padding included, via a recording stand-in
      for the model): A0 default, A1 hybrid+rerank, A2 ASR clean-up, A12 both.
Latency is the pre-model pipeline time (AskService.answer with an instant stand-in model); Gemini latency is NOT included.
"""
import collections, csv, glob, hashlib, importlib.util, json, os, platform, statistics, sys, time, types

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REG_PATH = os.path.join(HERE, "config_pf.json")
REG_SHA256 = "90579e4e5ba8682da4732f1bb47bca9172f59129c94366b118bd7983ed14f996"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
try:                                   # llm_interface imports the unrelated Ollama client, which needs requests
    import requests  # noqa: F401
except ImportError:
    sys.modules["requests"] = types.ModuleType("requests")

HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RANK_ARMS = ("R0_bm25", "R1_bm25_asr", "R2_rrf", "R3_rrf_rerank")
EVID_ARMS = ("A0_baseline", "A1_hybrid", "A2_asr", "A12_hybrid_asr")
EV_METRICS = ("evidence_hit", "interval_coverage", "full_interval_coverage", "strict_containment", "gold_cue_fraction")
RANK_REPORT = ("hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5", "mrr_at_10", "interval_recall_at_5", "full_evidence_coverage_at_5")
PASSES = 3
LATENCY_BUDGET_MS = {"A1_hybrid": 500.0, "A2_asr": 50.0, "A12_hybrid_asr": 500.0}     # added median pre-model time vs A0, warm
COLD_START_BUDGET_S = 5.0
FABRICATED_SENTENCE = "Photosynthesis converts sunlight into chemical energy inside chloroplasts."
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class EvalError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- environment, models, pins
def snapshot_dir(model_name):
    base = os.path.join(HF_HUB, "models--" + model_name.replace("/", "--"), "snapshots")
    snaps = sorted(os.listdir(base)) if os.path.isdir(base) else []
    if len(snaps) != 1:
        raise EvalError(f"expected exactly one cached snapshot for {model_name}, found {len(snaps)}")
    return os.path.join(base, snaps[0])


def tree_fingerprint(snapshot):
    files = {}
    for root, _, names in os.walk(snapshot):
        for n in names:
            p = os.path.join(root, n)
            files[os.path.relpath(p, snapshot).replace(os.sep, "/")] = sha256_file(p)
    return hashlib.sha256("\n".join(f"{k}\t{v}" for k, v in sorted(files.items())).encode()).hexdigest()


def code_pins(root=REPO):
    paths = sorted(glob.glob(os.path.join(root, "qa", "*.py")) + glob.glob(os.path.join(root, "llm_interface", "*.py")))
    return {os.path.relpath(p, root).replace(os.sep, "/"): sha256_file(p) for p in paths}


def build_registration(root=REPO):
    with open(os.path.join(root, "eval/config_bm25_v1.json"), encoding="utf-8") as f:
        frozen = json.load(f)
    ds = frozen["dataset"]
    return {
        "experiment_id": "pipeline_flags_v1", "status": "REGISTERED - NOT RUN", "created": "2026-10-15",
        "purpose": "Measure whether the optional Ask pipeline stages improve retrieval on the frozen benchmark, and how the faithfulness gate scores, BEFORE any flag is enabled.",
        "no_model_calls": "No Gemini or other language-model API call is made in this experiment (stage A and the faithfulness pilot).",
        "stage_b_not_included": "A shadow run of the faithfulness gate on freshly generated answers (about 139 Gemini calls) is NOT part of this registration and needs separate approval and a quota check.",
        "questions": {"headline": "PRIMARY_ANSWERABLE (133)", "descriptive_only": "SENSITIVITY_ANSWERABLE (6)", "not_run": "unanswerable and excluded questions",
                      "gold": "frozen bm25_v1.0 gold cue sets (baseline preprocessing) for ALL arms; a cue removed by cleaning counts as not covered"},
        "arms": {"ranking": {"R0_bm25": "frozen raw BM25 ranking (must reproduce the saved baseline exactly)", "R1_bm25_asr": "the same BM25 on ASR-cleaned captions (qa.asr_cleanup.sanitize_segments)",
                             "R2_rrf": "BM25 (positive scores) + dense (MiniLM) fused with RRF k=60, all chunks ranked (HybridSelector.rank_chunks)",
                             "R3_rrf_rerank": "top 12 fused chunks reordered by the cross-encoder, then the remaining fused chunks"},
                 "evidence": {"A0_baseline": "AskService defaults", "A1_hybrid": "AskService with HybridSelector (12 candidates, keep 4, offline models)",
                              "A2_asr": "AskConfig(asr_sanitize=True)", "A12_hybrid_asr": "both (exploratory)"},
                 "evidence_method": "AskService.answer with a recording stand-in model; the final EvidencePlan is captured; no prompt tuning, no thresholds changed"},
        "metrics": {"ranking": "frozen: Hit Rate@1/3/5, Interval Recall@1/3/5, Full-evidence coverage@1/3/5, MRR@10 (headline: Hit Rate@5 and MRR@10)",
                    "evidence": "evidence_hit, interval_coverage, full_interval_coverage, strict_containment, gold_cue_fraction, gate_failure, evidence size, estimated prompt tokens"},
        "statistics": {"interval": "question-level percentile bootstrap 95%, B=10000, seed 20261013 (frozen function); paired differences; lecture-clustered check for primary contrasts",
                       "decision_rule": "improvement iff the WHOLE paired 95% interval is above 0; decline iff the whole interval is below 0; otherwise no demonstrated difference (a bound equal to 0 counts as containing 0)",
                       "multiplicity": "none applied; 2 primary contrasts x 2 metrics",
                       "power_note": "BM25 Hit@5 is already about 0.93 (9 misses of 133), so at most 9 questions can improve; only a large gain could reach the decision rule. This benchmark can show harm or a large gain, not a small one."},
        "contrasts": {"ranking_primary": ["C1: R3_rrf_rerank - R0_bm25", "C2: R1_bm25_asr - R0_bm25"], "ranking_secondary": ["C3: R2_rrf - R0_bm25", "C4: R3_rrf_rerank - R2_rrf"],
                      "evidence": ["E1: A1_hybrid - A0_baseline", "E2: A2_asr - A0_baseline", "E3: A12_hybrid_asr - A0_baseline (exploratory)"], "metrics_for_decision": ["hit_rate_at_5", "mrr_at_10"],
                      "evidence_metrics_for_decision": ["evidence_hit", "strict_containment"]},
        "enable_rule_preregistered": {
            "for_each_flag": "RECOMMEND only if ALL hold: (1) its ranking primary contrast shows 'improvement' on Hit Rate@5 or MRR@10 and 'decline' on neither; (2) its evidence contrast shows 'decline' on neither evidence_hit nor strict_containment, and gate failures do not increase; (3) latency: warm median pre-model time added versus A0 is within budget (hybrid 500 ms, ASR clean-up 50 ms) and, for hybrid, cold start is within 5 s",
            "otherwise": "DO NOT ENABLE YET, and the failing criteria are listed",
            "even_if_recommended": "the benchmark was already used for the earlier BM25/hybrid comparison and for production-evidence checks, so a recommendation must be confirmed on new held-out questions before permanent enabling"},
        "latency_protocol": {"passes": PASSES, "pass_1": "cold: fresh selector (model load and chunk-vector cache misses)", "passes_2_3": "warm", "measured": "AskService.answer wall time with an instant stand-in model, and the selector call alone",
                             "excluded": "Gemini request time; memory use"},
        "faithfulness_pilot": {
            "data": "the 17 existing run_1 answers of ask_prompt_b_v1 (no new calls); evidence passages rebuilt with the production code and checked against the registered prompt hashes",
            "presumed_faithful_label": "the AI-assisted scores of those answers (grounding = ok for all 17); NOT human-verified",
            "synthetic_negatives": ["mismatch: the answer scored against another case's passages", "append: the real answer plus one fabricated unrelated sentence", "number: every number in the answer replaced by a different number (only answers with numbers)"],
            "reported": "score per answer, share blocked at thresholds 0.5-1.0 (false-block rate on the 17), detection rate per negative type; NOT a precision on real hallucinations, which cannot be computed without labelled unfaithful answers"},
        "determinism": "evidence plans must be identical across passes; R0 must reproduce the saved baseline metrics; R2 is compared with the saved dense_hybrid_v1 M3 rankings (agreement rate reported)",
        "limitations": ["Benchmark labels are machine-drafted; 1 of 176 primary questions is video-verified.", "Ranking metrics ignore the production relevance gate; evidence metrics include it.",
                        "Latency is single-machine CPU timing, pre-model only.", "The cross-encoder and the faithfulness gate had not been evaluated before this experiment."],
        "pins": {"frozen_config": {"eval/config_bm25_v1.json": sha256_file(os.path.join(root, "eval/config_bm25_v1.json"))},
                 "frozen_harness": {"eval/bm25_eval.py": sha256_file(os.path.join(root, "eval/bm25_eval.py"))},
                 "saved_baseline": {"eval/results/bm25_v1.0/per_question_results.csv": sha256_file(os.path.join(root, "eval/results/bm25_v1.0/per_question_results.csv"))},
                 "saved_m3": {"experiments/dense_hybrid_v1/results/m3_hybrid_v1/final/full_rankings_m3.csv":
                              sha256_file(os.path.join(root, "experiments/dense_hybrid_v1/results/m3_hybrid_v1/final/full_rankings_m3.csv"))},
                 "code": code_pins(root), "dataset": ds["file_sha256"]},
        "models": {"embed": {"name": EMBED_MODEL, "tree_fingerprint": tree_fingerprint(snapshot_dir(EMBED_MODEL))},
                   "rerank": {"name": RERANK_MODEL, "tree_fingerprint": tree_fingerprint(snapshot_dir(RERANK_MODEL))}},
    }


def load_registration(path=REG_PATH, expected=None):
    expected = REG_SHA256 if expected is None else expected
    if sha256_file(path) != expected:
        raise EvalError("config_pf.json differs from the hash pinned in the harness")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_all(reg, root=REPO, check_models=True):
    problems = []
    for group in ("frozen_config", "frozen_harness", "saved_baseline", "saved_m3", "code"):
        for rel, expected in reg["pins"][group].items():
            path = os.path.join(root, rel)
            actual = sha256_file(path) if os.path.exists(path) else None
            if actual != expected:
                problems.append(f"{rel}: expected {expected[:12]}..., got {'MISSING' if actual is None else actual[:12] + '...'}")
    for rel, expected in reg["pins"]["dataset"].items():
        path = os.path.join(root, "dataset/synapse_stem25_reviewed_v1", rel)
        if not os.path.exists(path) or sha256_file(path) != expected:
            problems.append(f"dataset/{rel} differs")
    if check_models:
        for key in ("embed", "rerank"):
            m = reg["models"][key]
            if tree_fingerprint(snapshot_dir(m["name"])) != m["tree_fingerprint"]:
                problems.append(f"model snapshot {m['name']} differs from the registration")
    ask_vars = sorted(k for k in os.environ if k.startswith("ASK_"))
    if ask_vars:
        problems.append(f"ASK_* environment variables are set: {ask_vars}")
    if problems:
        raise EvalError("verification failed:\n  " + "\n  ".join(problems))
    return {"pins_checked": sum(len(reg["pins"][g]) for g in reg["pins"]), "models_checked": check_models}


def _qa():
    from qa import asr_cleanup, faithfulness, hybrid, retrieval, service
    from qa.transcript import normalize_segments
    from llm_interface.providers import GenerationResult
    return types.SimpleNamespace(asr=asr_cleanup, faith=faithfulness, hybrid=hybrid, R=retrieval, S=service, normalize=normalize_segments, Result=GenerationResult)


def load_frozen(reg, root=REPO):
    rel = "eval/bm25_eval.py"
    path = os.path.join(root, rel)
    if sha256_file(path) != reg["pins"]["frozen_harness"][rel]:
        raise EvalError("eval/bm25_eval.py differs from the registration; not importing it")
    spec = importlib.util.spec_from_file_location("frozen_bm25_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------- data and contexts
def _rd(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_data(bm25, cfg, root=REPO):
    ds = os.path.join(root, cfg["dataset"]["path"])
    manifest = _rd(os.path.join(root, cfg["question_manifest"]["path"]))
    questions = {r["question_id"]: r for r in _rd(os.path.join(ds, "questions.csv"))}
    with open(os.path.join(ds, "gold_evidence.json"), encoding="utf-8") as f:
        gold = {g["question_id"]: g for g in json.load(f)["questions"]}
    cues = collections.defaultdict(list)
    with open(os.path.join(ds, "transcript_segments.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cues[r["lecture_id"]].append(r)
    groups = bm25.split_groups(manifest)
    rows = [r for g in ("PRIMARY_ANSWERABLE", "SENSITIVITY_ANSWERABLE") for r in groups[g]]
    return rows, questions, gold, cues


def asr_context(Q, base_ctx, evidence_config):
    """Chunks and cue sets for the ASR-cleaned captions of one lecture. Start times are preserved, so cue ids map by start time."""
    segs = Q.asr.sanitize_segments(base_ctx["segs"])
    chunks = Q.R._group(range(len(segs)), segs, evidence_config)
    chunk_cues = [{base_ctx["start_to_id"][segs[i].start] for i in range(a, b + 1)} for a, b in chunks]
    return {"segs": segs, "chunks": chunks, "chunk_cues": chunk_cues, "start_to_id": base_ctx["start_to_id"]}


def cue_ids_of_plan(plan, segs, start_to_id):
    """Cue ids inside the passages of an EvidencePlan. The block text must match the segments used (guards against index mix-ups)."""
    ids = set()
    if plan is None or not plan.ok:
        return ids
    for b in plan.blocks:
        try:
            joined = " ".join(segs[i].text for i in range(b.first_index, b.last_index + 1))
        except IndexError:
            raise EvalError("evidence block indexes fall outside the segment list used for mapping")
        if joined != b.text:
            raise EvalError("evidence block text does not match the segment list used for mapping")
        ids |= {start_to_id[segs[i].start] for i in range(b.first_index, b.last_index + 1)}
    return ids


def evidence_metrics(evidence, intervals):
    if not intervals:
        return None
    gold = set().union(*intervals)
    covered = sum(1 for iv in intervals if evidence & iv)
    return {"evidence_hit": 1.0 if covered else 0.0, "interval_coverage": covered / len(intervals),
            "full_interval_coverage": 1.0 if covered == len(intervals) else 0.0, "strict_containment": 1.0 if gold <= evidence else 0.0,
            "gold_cue_fraction": len(gold & evidence) / len(gold)}


# ----------------------------------------------------------------------------------------------- statistics
def decide(lo, hi):
    if lo > 0:
        return "improvement"
    if hi < 0:
        return "decline"
    return "no_demonstrated_difference"


def paired(bm25, first, second, metrics, B, seed, level):
    if [r["question_id"] for r in first] != [r["question_id"] for r in second]:
        raise EvalError("paired arms are not aligned")
    diffs = {m: [a[m] - b[m] for a, b in zip(first, second)] for m in metrics}
    ci = bm25.bootstrap_ci(diffs, B, seed, level)
    n, out = len(first), {}
    for m in metrics:
        lo, hi = ci[m]
        out[m] = {"difference": sum(diffs[m]) / n, "ci_low": lo, "ci_high": hi, "decision": decide(lo, hi), "n": n,
                  "first_better": sum(1 for d in diffs[m] if d > 0), "second_better": sum(1 for d in diffs[m] if d < 0), "tied": sum(1 for d in diffs[m] if d == 0)}
    return out


def clustered(bm25, first, second, metrics, B, seed, level):
    by = collections.defaultdict(lambda: {m: [] for m in metrics})
    for a, b in zip(first, second):
        for m in metrics:
            by[a["lecture_id"]][m].append(a[m] - b[m])
    ci = bm25.clustered_bootstrap_ci(dict(by), B, seed, level)
    return {m: {"ci_low": ci[m][0], "ci_high": ci[m][1], "decision": decide(*ci[m])} for m in metrics}


def summarize(bm25, records, metrics, B, seed, level, with_ci=True):
    n = len(records)
    vals = {m: [r[m] for r in records] for m in metrics}
    ci = bm25.bootstrap_ci(vals, B, seed, level) if (with_ci and n >= 10) else None
    return {"n": n, "metrics": {m: {"estimate": sum(vals[m]) / n, "ci_low": ci[m][0] if ci else None, "ci_high": ci[m][1] if ci else None} for m in metrics}}


def latency_stats(values_ms):
    xs = sorted(values_ms)
    if not xs:
        return None
    return {"n": len(xs), "median": statistics.median(xs), "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))], "max": xs[-1], "mean": sum(xs) / len(xs)}


def recommend(flag, rank_contrast, evid_contrast, latency, gate_failure_delta, budget_ms, cold_start_s=None):
    """Apply the preregistered enable rule to computed results. Returns criteria booleans, reasons and the verdict string."""
    rank_decisions = [rank_contrast[m]["decision"] for m in ("hit_rate_at_5", "mrr_at_10")]
    ev_decisions = [evid_contrast[m]["decision"] for m in ("evidence_hit", "strict_containment")]
    c1 = "improvement" in rank_decisions and "decline" not in rank_decisions
    c2 = "decline" not in ev_decisions and gate_failure_delta <= 0
    c3a = latency is not None and latency <= budget_ms
    c3b = cold_start_s is None or cold_start_s <= COLD_START_BUDGET_S
    reasons = []
    if not c1:
        reasons.append(f"ranking: no demonstrated improvement ({rank_decisions})")
    if not c2:
        reasons.append(f"evidence: decline or more gate failures ({ev_decisions}, gate_failure_delta={gate_failure_delta})")
    if not c3a:
        reasons.append(f"latency: warm added median {latency if latency is None else round(latency, 1)} ms exceeds {budget_ms} ms")
    if not c3b:
        reasons.append(f"cold start {round(cold_start_s, 1)} s exceeds {COLD_START_BUDGET_S} s (needs a start-up warm-up)")
    ok = c1 and c2 and c3a and c3b
    return {"flag": flag, "criteria": {"ranking_improvement": c1, "evidence_no_decline": c2, "warm_latency_within_budget": c3a, "cold_start_ok": c3b},
            "verdict": "RECOMMEND (confirm on held-out questions first)" if ok else "DO NOT ENABLE YET", "reasons": reasons}


# ----------------------------------------------------------------------------------------------- stage A
class Capture:
    """Wraps an evidence selector: remembers the last plan and the total time spent in the selector."""

    def __init__(self, inner):
        self.inner, self.plan, self.seconds, self.calls = inner, None, 0.0, 0

    def reset(self):
        self.plan, self.seconds, self.calls = None, 0.0, 0

    def __call__(self, *args, **kwargs):
        t = time.perf_counter()
        plan = self.inner(*args, **kwargs)
        self.seconds += time.perf_counter() - t
        self.calls += 1
        self.plan = plan
        return plan


class InstantModel:
    name, model = "instant-standin", "none"

    def __init__(self, result_cls):
        self._cls, self.calls = result_cls, 0

    def generate(self, prompt, settings=None):
        self.calls += 1
        return self._cls(text="The instructor says this.", provider=self.name, model=self.model, latency_ms=0, usage=None)


def make_evidence_arms(Q, hybrid_cfg=None):
    """arm -> (AskService, Capture, uses_asr). A fresh selector (and cache) per arm."""
    arms = {}
    for arm, hybrid_on, asr_on in (("A0_baseline", False, False), ("A1_hybrid", True, False), ("A2_asr", False, True), ("A12_hybrid_asr", True, True)):
        inner = Q.hybrid.HybridSelector(hybrid_cfg or Q.hybrid.HybridConfig(enabled=True)) if hybrid_on else Q.R.select_evidence
        cap = Capture(inner)
        svc = Q.S.AskService(provider=InstantModel(Q.Result), config=Q.S.AskConfig(asr_sanitize=asr_on), evidence_selector=cap)
        arms[arm] = (svc, cap, asr_on)
    return arms


def run_stage_a(reg, root=REPO):
    bm25 = load_frozen(reg, root)
    with open(os.path.join(root, "eval/config_bm25_v1.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    bm25.verify_pinned_hashes(cfg, root)
    Q = _qa()
    ec = Q.R.EvidenceConfig()
    rows, questions, gold, cues = load_data(bm25, cfg, root)
    base, asr = {}, {}
    for r in rows:
        lid = r["lecture_id"]
        if lid not in base:
            base[lid] = bm25.build_lecture_context(cues[lid], Q.R, Q.normalize, ec, cfg["text_field"]["value"])
            asr[lid] = asr_context(Q, base[lid], ec)
    selector = Q.hybrid.HybridSelector(Q.hybrid.HybridConfig(enabled=True))
    saved = {r["question_id"]: r for r in _rd(os.path.join(root, "eval/results/bm25_v1.0/per_question_results.csv"))}
    m3 = collections.defaultdict(list)
    for r in _rd(os.path.join(root, "experiments/dense_hybrid_v1/results/m3_hybrid_v1/final/full_rankings_m3.csv")):
        m3[r["question_id"]].append((int(r["rank"]), int(r["chunk_index"])))

    # ---- ranking level
    rank_records = {a: [] for a in RANK_ARMS}
    base_mismatch, m3_agree, m3_total, no_trace = [], 0, 0, []
    traces = {}
    for r in rows:
        qid, lid = r["question_id"], r["lecture_id"]
        q, ctx, actx = questions[qid]["question_text"], base[lid], asr[lid]
        ivs = bm25.gold_intervals(gold[qid], ctx["dropped"])
        info = {"question_id": qid, "lecture_id": lid, "group": r["group"], "subject": questions[qid]["subject"], "question_category": questions[qid]["question_category"]}
        ranked0, _, _ = bm25.rank_question(Q.R, q, ctx)
        ranked1, _, _ = bm25.rank_question(Q.R, q, actx)
        trace = selector.rank_chunks(q, ctx["segs"], ec)
        if trace is not None and [tuple(c) for c in trace.chunks] != [tuple(c) for c in ctx["chunks"]]:
            raise EvalError(f"{qid}: hybrid chunking differs from the frozen chunking")
        if trace is None:                                    # no content terms: the pipeline would use BM25 evidence, so ranks equal BM25
            no_trace.append(qid)
        traces[qid] = trace
        fused_rank = [i for i, _ in trace.fused] if trace else ranked0
        rerank_rank = trace.reranked_full() if trace else ranked0
        for arm, ranked, cc in (("R0_bm25", ranked0, ctx["chunk_cues"]), ("R1_bm25_asr", ranked1, actx["chunk_cues"]),
                                ("R2_rrf", fused_rank, ctx["chunk_cues"]), ("R3_rrf_rerank", rerank_rank, ctx["chunk_cues"])):
            m = bm25.question_metrics(ranked, cc, ivs)
            if m is not None:
                rank_records[arm].append({**info, **m})
        r0 = rank_records["R0_bm25"][-1] if rank_records["R0_bm25"] and rank_records["R0_bm25"][-1]["question_id"] == qid else None
        if r0 is not None and any(str(r0[k]) != saved[qid][k] for k in ("hit_rate_at_5", "mrr_at_10", "interval_recall_at_5")):
            base_mismatch.append(qid)
        top_saved = [c for _, c in sorted(m3[qid])[:10]]
        m3_total += 1
        m3_agree += int(fused_rank[:10] == top_saved)

    # ---- evidence level (passes: 1 cold, then warm); the baseline plans of pass 1 define the reference
    arms = make_evidence_arms(Q)
    ev_records = {a: [] for a in EVID_ARMS}
    timing = {a: {p: [] for p in range(1, PASSES + 1)} for a in EVID_ARMS}
    select_timing = {a: {p: [] for p in range(1, PASSES + 1)} for a in EVID_ARMS}
    first_of_lecture = {a: [] for a in EVID_ARMS}
    cold_start = {}
    plans_pass1 = {}
    for p in range(1, PASSES + 1):
        seen = set()
        for r in rows:
            qid, lid = r["question_id"], r["lecture_id"]
            q, ctx = questions[qid]["question_text"], base[lid]
            ivs = bm25.gold_intervals(gold[qid], ctx["dropped"])
            for arm, (svc, cap, uses_asr) in arms.items():
                cap.reset()
                t = time.perf_counter()
                out = svc.answer(question=q, segments=ctx["segs"])
                total = time.perf_counter() - t
                timing[arm][p].append(total * 1000)
                select_timing[arm][p].append(cap.seconds * 1000)
                if p == 1 and arm not in cold_start and arm in ("A1_hybrid", "A12_hybrid_asr"):
                    cold_start[arm] = total
                if p == 1 and lid not in seen and arm in ("A1_hybrid", "A12_hybrid_asr"):
                    first_of_lecture[arm].append(total * 1000)
                segs = asr[lid]["segs"] if uses_asr else ctx["segs"]
                ids = cue_ids_of_plan(cap.plan, segs, ctx["start_to_id"])
                key = (arm, qid)
                signature = tuple((b.first_index, b.last_index) for b in cap.plan.blocks) if cap.plan.ok else (cap.plan.decision,)
                if p == 1:
                    plans_pass1[key] = signature
                    m = evidence_metrics(ids, ivs)
                    if m is not None:
                        ev_records[arm].append({"question_id": qid, "lecture_id": lid, "group": r["group"], **m, "gate_failure": 0.0 if cap.plan.ok else 1.0,
                                                "n_blocks": len(cap.plan.blocks) if cap.plan.ok else 0, "evidence_chars": sum(len(b.text) for b in cap.plan.blocks) if cap.plan.ok else 0,
                                                "estimated_prompt_tokens": out.estimated_prompt_tokens, "mode": (cap.plan.mode if cap.plan.ok else cap.plan.decision) or ""})
                elif plans_pass1[key] != signature:
                    raise EvalError(f"{arm} {qid}: evidence differs between passes (non-deterministic)")
            seen.add(lid)
    return {"bm25": bm25, "Q": Q, "rows": rows, "rank_records": rank_records, "ev_records": ev_records, "timing": timing, "select_timing": select_timing,
            "cold_start": cold_start, "first_of_lecture": first_of_lecture, "base_mismatch": base_mismatch, "m3_agreement": (m3_agree, m3_total), "no_trace": no_trace, "cfg": cfg}


def analyse_stage_a(reg, res):
    """Aggregates, paired contrasts, latency summary and the preregistered recommendations."""
    bm25, cfg = res["bm25"], res["cfg"]
    st = cfg["statistics"]["confidence_interval"]
    B, seed, level = st["resamples"], st["seed"], st["level"]
    prim = lambda recs: [r for r in recs if r["group"] == "PRIMARY_ANSWERABLE"]
    sens = lambda recs: [r for r in recs if r["group"] == "SENSITIVITY_ANSWERABLE"]
    out = {"ranking": {}, "evidence": {}, "contrasts": {"ranking": {}, "evidence": {}}, "latency": {}, "integrity": {}}
    for arm, recs in res["rank_records"].items():
        out["ranking"][arm] = {"PRIMARY_ANSWERABLE": summarize(bm25, prim(recs), RANK_REPORT, B, seed, level), "SENSITIVITY_ANSWERABLE_descriptive": summarize(bm25, sens(recs), RANK_REPORT, B, seed, level, with_ci=False)}
    ev_all = EV_METRICS + ("gate_failure",)
    for arm, recs in res["ev_records"].items():
        out["evidence"][arm] = {"PRIMARY_ANSWERABLE": summarize(bm25, prim(recs), ev_all, B, seed, level), "SENSITIVITY_ANSWERABLE_descriptive": summarize(bm25, sens(recs), ev_all, B, seed, level, with_ci=False),
                                "mean_blocks": statistics.mean(r["n_blocks"] for r in prim(recs)), "mean_evidence_chars": statistics.mean(r["evidence_chars"] for r in prim(recs)),
                                "mean_estimated_prompt_tokens": statistics.mean(r["estimated_prompt_tokens"] for r in prim(recs)), "mode_counts": dict(collections.Counter(r["mode"] for r in prim(recs)))}
    rm = ("hit_rate_at_5", "mrr_at_10")
    for cid, a, b in (("C1", "R3_rrf_rerank", "R0_bm25"), ("C2", "R1_bm25_asr", "R0_bm25"), ("C3", "R2_rrf", "R0_bm25"), ("C4", "R3_rrf_rerank", "R2_rrf")):
        out["contrasts"]["ranking"][cid] = {"contrast": f"{a} minus {b}", "role": "primary" if cid in ("C1", "C2") else "secondary", "n": 133,
                                            "paired": paired(bm25, prim(res["rank_records"][a]), prim(res["rank_records"][b]), RANK_REPORT, B, seed, level),
                                            "lecture_clustered": clustered(bm25, prim(res["rank_records"][a]), prim(res["rank_records"][b]), rm, B, seed, level)}
    em = ("evidence_hit", "strict_containment", "full_interval_coverage")      # gate_failure is compared by its means (higher is worse)
    for cid, a in (("E1", "A1_hybrid"), ("E2", "A2_asr"), ("E3", "A12_hybrid_asr")):
        out["contrasts"]["evidence"][cid] = {"contrast": f"{a} minus A0_baseline", "paired": paired(bm25, prim(res["ev_records"][a]), prim(res["ev_records"]["A0_baseline"]), em, B, seed, level),
                                             "lecture_clustered": clustered(bm25, prim(res["ev_records"][a]), prim(res["ev_records"]["A0_baseline"]), ("evidence_hit", "strict_containment"), B, seed, level)}
    warm = lambda d, arm: [x for p in range(2, PASSES + 1) for x in d[arm][p]]
    for arm in EVID_ARMS:
        out["latency"][arm] = {"pre_model_answer_ms_warm": latency_stats(warm(res["timing"], arm)), "selector_ms_warm": latency_stats(warm(res["select_timing"], arm)),
                               "pre_model_answer_ms_pass1_cold": latency_stats(res["timing"][arm][1]), "cold_start_first_question_s": res["cold_start"].get(arm),
                               "first_question_of_each_lecture_ms_pass1": latency_stats(res["first_of_lecture"].get(arm, []))}
    base_med = out["latency"]["A0_baseline"]["pre_model_answer_ms_warm"]["median"]
    out["recommendations"] = []
    for flag, rc, ec, arm, budget in (("ASK_HYBRID_RETRIEVAL (hybrid + rerank)", "C1", "E1", "A1_hybrid", LATENCY_BUDGET_MS["A1_hybrid"]), ("ASK_ASR_SANITIZE", "C2", "E2", "A2_asr", LATENCY_BUDGET_MS["A2_asr"])):
        added = out["latency"][arm]["pre_model_answer_ms_warm"]["median"] - base_med
        gf_delta = out["evidence"][arm]["PRIMARY_ANSWERABLE"]["metrics"]["gate_failure"]["estimate"] - out["evidence"]["A0_baseline"]["PRIMARY_ANSWERABLE"]["metrics"]["gate_failure"]["estimate"]
        rec = recommend(flag, out["contrasts"]["ranking"][rc]["paired"], out["contrasts"]["evidence"][ec]["paired"], added, gf_delta, budget,
                        res["cold_start"].get(arm) if arm == "A1_hybrid" else None)
        rec["added_warm_median_ms"] = added
        out["recommendations"].append(rec)
    out["integrity"] = {"R0_matches_saved_baseline": not res["base_mismatch"], "R0_mismatching_questions": res["base_mismatch"],
                        "R2_top10_agrees_with_saved_M3": f"{res['m3_agreement'][0]} of {res['m3_agreement'][1]} questions", "questions_without_content_terms_using_bm25_ranks": res["no_trace"], "evidence_plans_identical_across_passes": True}
    return out


def write_stage_a(out_dir, reg, res, analysis):
    os.makedirs(out_dir)
    def dump(name, obj):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=2, default=str)
    dump("analysis.json", analysis)
    with open(os.path.join(out_dir, "per_question_ranking.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id", "lecture_id", "group", "arm"] + list(RANK_REPORT))
        for arm, recs in res["rank_records"].items():
            for r in recs:
                w.writerow([r["question_id"], r["lecture_id"], r["group"], arm] + [r[m] for m in RANK_REPORT])
    with open(os.path.join(out_dir, "per_question_evidence.csv"), "w", encoding="utf-8", newline="") as f:
        cols = ["question_id", "lecture_id", "group", "arm", "mode"] + list(EV_METRICS) + ["gate_failure", "n_blocks", "evidence_chars", "estimated_prompt_tokens"]
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for arm, recs in res["ev_records"].items():
            for r in recs:
                w.writerow([r["question_id"], r["lecture_id"], r["group"], arm, r["mode"]] + [r[c] for c in cols[5:]])
    with open(os.path.join(out_dir, "latency_per_question_ms.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arm", "pass", "index", "answer_ms", "selector_ms"])
        for arm in EVID_ARMS:
            for p in range(1, PASSES + 1):
                for i, (a, s) in enumerate(zip(res["timing"][arm][p], res["select_timing"][arm][p])):
                    w.writerow([arm, p, i, round(a, 3), round(s, 3)])
    manifest = {"experiment": reg["experiment_id"], "registration_sha256": REG_SHA256, "python": platform.python_version(), "machine": platform.platform(), "cpu_count": os.cpu_count(),
                "harness_sha256": sha256_file(os.path.abspath(__file__)), "files_sha256": {n: sha256_file(os.path.join(out_dir, n)) for n in sorted(os.listdir(out_dir))},
                "no_model_calls": True}
    dump("run_manifest.json", manifest)


# ----------------------------------------------------------------------------------------------- faithfulness pilot
def perturb_numbers(text):
    """Replace every integer in the text by a different one; returns (new text, count)."""
    import re
    count = 0

    def sub(m):
        nonlocal count
        count += 1
        return str(int(m.group(0)) + 1 + (int(m.group(0)) % 3))
    return re.sub(r"\b\d+\b", sub, text), count


def faithfulness_pilot(reg, root=REPO):
    Q = _qa()
    base_dir = os.path.join(root, "experiments/ask_prompt_b_v1")
    with open(os.path.join(base_dir, "config_b.json"), encoding="utf-8") as f:
        cases = {c["id"]: c for c in json.load(f)["cases"]}
    raw = [json.loads(l) for l in open(os.path.join(base_dir, "results/run_1/raw_results.jsonl"), encoding="utf-8") if l.strip()]
    answers = [r for r in raw if r.get("status") != "ERROR" and r.get("answer_text")]
    cfg = Q.faith.FaithfulnessConfig(mode="shadow")
    checker = Q.faith.LexicalSupportChecker(cfg)
    passages, prompt_ok = {}, {}
    ds = os.path.join(root, "dataset/synapse_stem25_reviewed_v1")
    qrows = {r["question_id"]: r for r in _rd(os.path.join(ds, "questions.csv"))}
    for cid, case in cases.items():
        if case["source"] == "demo":
            with open(os.path.join(root, "data/3zLQfuMZMF4/transcript.json"), encoding="utf-8") as f:
                d = json.load(f)
            segs = Q.normalize([{"start": e["start"], "end": e["end"], "text": e["text"]} for e in d["entries"]])
        else:
            lid = qrows[case["source"].split(":", 1)[1]]["lecture_id"]
            segs = Q.normalize([{"start": c["start_seconds"], "end": c["end_seconds"], "text": c["text"]} for c in
                                (json.loads(l) for l in open(os.path.join(ds, "transcript_segments.jsonl"), encoding="utf-8")) if c["lecture_id"] == lid])
        seen = {}

        class Rec:
            name, model = "rec", "none"

            def generate(self, prompt, settings=None):
                seen["prompt"] = prompt
                return Q.Result(text="The instructor says this.", provider="rec", model="none", latency_ms=0, usage=None)
        out = Q.S.AskService(provider=Rec(), config=Q.S.AskConfig()).answer(question=case["question"], segments=segs)
        passages[cid] = out.evidence_passages
        prompt_ok[cid] = hashlib.sha256(seen["prompt"].encode("utf-8")).hexdigest() == case["user_prompt_sha256"]
    order = sorted(cases)
    real, mismatch, append, number = [], [], [], []
    for r in answers:
        cid = r["case"]
        real.append((cid, r["arm"], checker.check(r["answer_text"], passages[cid]).score))
        other = order[(order.index(cid) + 1) % len(order)]
        mismatch.append(checker.check(r["answer_text"], passages[other]).score)
        append.append(checker.check(r["answer_text"] + " " + FABRICATED_SENTENCE, passages[cid]).score)
        new_text, n = perturb_numbers(r["answer_text"])
        if n:
            number.append(checker.check(new_text, passages[cid]).score)
    share = lambda scores, t: sum(1 for s in scores if s < t) / len(scores) if scores else None
    scores_real = [s for _, _, s in real]
    return {"n_real_answers": len(real), "evidence_prompt_hashes_match_the_registered_ones": all(prompt_ok.values()),
            "real_answer_scores": [{"case": c, "arm": a, "score": round(s, 3)} for c, a, s in real],
            "mean_score_real": sum(scores_real) / len(scores_real), "min_score_real": min(scores_real),
            "by_threshold": {str(t): {"false_block_rate_on_real_answers": share(scores_real, t), "detect_mismatch": share(mismatch, t), "detect_appended_fabrication": share(append, t),
                                      "detect_wrong_numbers": share(number, t)} for t in THRESHOLDS},
            "n_negatives": {"mismatch": len(mismatch), "appended_fabrication": len(append), "wrong_numbers": len(number)},
            "mean_score_negatives": {"mismatch": sum(mismatch) / len(mismatch), "appended_fabrication": sum(append) / len(append), "wrong_numbers": (sum(number) / len(number)) if number else None},
            "caveats": ["17 answers, all scored 'grounding ok' by AI-assisted, non-blind scoring; not human-verified faithful.", "Negatives are synthetic; detection rates say nothing about real hallucinations.",
                        "One fabricated sentence added to a multi-sentence answer lowers the score by only the share of one sentence.", "A faithful copy of a wrong caption (case P3) cannot be detected."]}


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a[0] if a else ""
    if cmd == "register":
        if os.path.exists(REG_PATH):
            print("REFUSING: config_pf.json exists", file=sys.stderr)
            return 2
        reg = build_registration()
        with open(REG_PATH, "w", encoding="utf-8", newline="\n") as f:
            json.dump(reg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("written", REG_PATH, sha256_file(REG_PATH))
        return 0
    reg = load_registration()
    if cmd == "verify":
        print(verify_all(reg))
        return 0
    if cmd in ("run-a", "faith-pilot"):
        out_dir = os.path.abspath(a[a.index("--out") + 1])
        if os.path.exists(out_dir):
            print(f"REFUSING to overwrite {out_dir}", file=sys.stderr)
            return 2
        verify_all(reg)
        if cmd == "run-a":
            res = run_stage_a(reg)
            analysis = analyse_stage_a(reg, res)
            write_stage_a(out_dir, reg, res, analysis)
            print("written", out_dir)
        else:
            os.makedirs(out_dir)
            result = faithfulness_pilot(reg)
            with open(os.path.join(out_dir, "faithfulness_pilot.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(result, f, indent=2)
            print("written", out_dir)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
