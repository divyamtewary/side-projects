from __future__ import annotations
import time
import statistics
from typing import Dict, Any, List, Callable

NEUTRAL_PROMPT = (
    "Summarize the following synthetic neutral passage in one paragraph. "
    "Passage: The synthetic dataset describes neutral encyclopedic topics including "
    "photosynthesis, water cycle, and cartography. Provide a concise summary."
)

def run_performance(runtime, preset: str, on_log: Callable = None) -> Dict[str, Any]:
    """
    Runs baseline generation performance.
    Quick: 1 measured run (64 tokens) after 1 warmup.
    Standard: 3 measured runs (128 tokens) after 1 warmup, dispersion reported.
    Returns PerformanceResult dict.
    """
    def log(m): 
        if on_log: on_log(m)
    budget = 64 if preset == "quick" else 128
    num_measured = 1 if preset == "quick" else 3

    log(f"Warm-up run ({budget} tokens, excluded from measurement)...")
    try:
        runtime.generate(NEUTRAL_PROMPT, max_new_tokens=min(16, budget), do_sample=False)
    except Exception as e:
        log(f"Warm-up warning: {e}")

    results = []
    load_time = getattr(runtime, "load_time_s", 0.0)
    for i in range(num_measured):
        log(f"Measured run {i+1}/{num_measured} – generating {budget} tokens...")
        t0 = time.time()
        out = runtime.generate(NEUTRAL_PROMPT, max_new_tokens=budget, do_sample=False)
        t1 = time.time()
        # capture rss
        try:
            import psutil, os
            rss = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
        except Exception:
            rss = None
        # out already has timing, but we keep our own
        out["peak_rss_mb"] = rss
        out["run_index"] = i
        results.append(out)
        # small pause between runs
        time.sleep(0.2)

    # aggregate: if multiple, report mean; else single
    if len(results) == 1:
        r = results[0]
        # ensure fields present
        return {
            "schema_version": "1.0",
            "num_input_tokens": r.get("num_input_tokens", 0),
            "num_output_tokens": r.get("num_output_tokens", budget),
            "load_time_s": round(load_time, 3),
            "prefill_time_s": round(r.get("prefill_time_s", 0), 3),
            "decode_time_s": round(r.get("decode_time_s", 0), 3),
            "total_time_s": round(r.get("total_time_s", 0), 3),
            "prefill_tok_per_s": round(r.get("prefill_tokens_per_second", 0), 2),
            "decode_tok_per_s": round(r.get("tokens_per_second_decode", 0), 2),
            "overall_tok_per_s": round(r.get("tokens_per_second_overall", 0), 2),
            "inter_token_latency_ms_mean": round(r.get("mean_ms", 0), 2),
            "inter_token_latency_ms_median": round(r.get("median_ms", 0), 2),
            "inter_token_latency_ms_p95": round(r.get("p95_ms", 0), 2),
            "inter_token_latency_ms_p99": round(r.get("p99_ms", 0), 2),
            "inter_token_latency_ms_min": round(r.get("min_ms", 0), 2),
            "inter_token_latency_ms_max": round(r.get("max_ms", 0), 2),
            "peak_rss_mb": round(r.get("peak_rss_mb", 0), 1) if r.get("peak_rss_mb") else None,
            "token_timestamps": r.get("token_timestamps", []),
            "truth_level": "measured/derived",
            "note": "TTFT is streaming-approximation, distinct from true prefill. Fixed ordering is a stated limitation.",
            "_raw_runs": results,  # internal, not in schema but useful
        }
    else:
        # average
        def avg(key):
            vals = [rr.get(key, 0) for rr in results]
            return sum(vals)/len(vals) if vals else 0
        # take first input tokens as representative
        return {
            "schema_version": "1.0",
            "num_input_tokens": int(avg("num_input_tokens")),
            "num_output_tokens": int(avg("num_output_tokens")),
            "load_time_s": round(load_time, 3),
            "prefill_time_s": round(avg("prefill_time_s"), 3),
            "decode_time_s": round(avg("decode_time_s"), 3),
            "total_time_s": round(avg("total_time_s"), 3),
            "prefill_tok_per_s": round(avg("prefill_tokens_per_second"), 2),
            "decode_tok_per_s": round(avg("tokens_per_second_decode"), 2),
            "overall_tok_per_s": round(avg("tokens_per_second_overall"), 2),
            "inter_token_latency_ms_mean": round(avg("mean_ms"), 2),
            "inter_token_latency_ms_median": round(avg("median_ms"), 2),
            "inter_token_latency_ms_p95": round(avg("p95_ms"), 2),
            "inter_token_latency_ms_p99": round(avg("p99_ms"), 2),
            "inter_token_latency_ms_min": round(avg("min_ms"), 2),
            "inter_token_latency_ms_max": round(avg("max_ms"), 2),
            "peak_rss_mb": round(avg("peak_rss_mb"), 1) if results[0].get("peak_rss_mb") else None,
            "token_timestamps": results[0].get("token_timestamps", []),
            "truth_level": "measured/derived",
            "note": "TTFT is streaming-approximation, distinct from true prefill. Fixed ordering is a stated limitation. Standard preset reports mean over 3 runs.",
            "_raw_runs": results,
            "_dispersion": {
                "decode_tok_per_s_std": round(statistics.pstdev([r.get("tokens_per_second_decode",0) for r in results]),3) if len(results)>1 else 0,
                "total_time_s_std": round(statistics.pstdev([r.get("total_time_s",0) for r in results]),3) if len(results)>1 else 0,
            }
        }
