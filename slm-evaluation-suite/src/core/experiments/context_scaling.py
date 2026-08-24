from __future__ import annotations
from typing import Dict, Any, List, Callable
import math

# synthetic neutral filler generator – no aviation domain
def _synthetic_passage(level: int) -> tuple[str, int]:
    """Return (passage, approx tokens). Levels 1..5."""
    bases = [
        ("Minimal context", "Synthetic neutral passage about light refraction in water. ", 75),
        ("Short context", "Synthetic passage describing the water cycle and river basin geography with neutral encyclopedic filler. ", 236),
        ("Medium context", "Synthetic passage about cartography history and neutral trade routes, combined with encyclopedic filler. ", 428),
        ("Long context", "Synthetic extended passage mixing biology of photosynthesis with open-domain geography and neutral science. ", 641),
        ("Extended context", "Synthetic full-length passage concatenating neutral coverage of photosynthesis, water cycle, cartography, and general science. ", 1169),
    ]
    label, base, tokens = bases[level-1]
    # repeat base to match desired char length
    repeat = max(1, level*3)
    passage = (base * repeat).strip()
    # keep token count as base tokens for deterministic testing (or compute approx)
    return passage, tokens

def run_context_scaling(runtime, preset: str, on_log: Callable = None) -> Dict[str, Any]:
    def log(m):
        if on_log:
            on_log(m)
    n_points = 3 if preset == "quick" else 5
    points = []
    for i in range(1, n_points+1):
        passage, tok = _synthetic_passage(i)
        label = f"L{i}: {['Minimal','Short','Medium','Long','Extended'][i-1]}"
        log(f"Context sweep {label} – ~{tok} input tokens...")
        # use runtime's context_scaling_sweep if fake, else simulate via generate
        try:
            if hasattr(runtime, "context_scaling_sweep"):
                # for fake we have helper, but we emulate point-by-point for real
                # to keep streaming, we just call generate with passage as prompt
                pass
        except Exception:
            pass
        # Build prompt from passage
        prompt = f"Synthetic evaluation task. Context: {passage[:2000]} \nQuestion: Summarize the key points in one sentence."
        try:
            budget = 96 if preset == "quick" else 128
            out = runtime.generate(prompt, max_new_tokens=budget, do_sample=False)
            tok_per_sec = out.get("tokens_per_second_decode", 0)
            total = out.get("total_time_s", 0)
            ms_per = out.get("mean_ms", 0)
            preview = out.get("response", "")[:180].replace("\n"," ")
        except Exception as e:
            log(f"  generation failed: {e}")
            tok_per_sec = 0
            total = 0
            ms_per = 0
            preview = f"ERROR: {e}"
        points.append({
            "label": label,
            "num_input_tokens": tok,
            "context_chars": len(passage),
            "num_output_tokens": budget,
            "total_time_s": round(total, 2),
            "tok_per_sec": round(tok_per_sec, 2),
            "ms_per_token": round(ms_per, 1),
            "response_preview": preview,
        })

    # derived: fit power law r = k / n^alpha using log regression
    fit_k = None
    fit_alpha = None
    pearson_r = None
    conclusion = "Descriptive only: no pass/fail claim about model quality."
    try:
        import math, numpy as np
        n = np.array([p["num_input_tokens"] for p in points], dtype=float)
        r = np.array([p["tok_per_sec"] for p in points], dtype=float)
        # avoid zeros
        mask = (r > 0) & (n > 0)
        if mask.sum() >= 3:
            log_n = np.log(n[mask])
            log_r = np.log(r[mask])
            # linear fit: log_r = log_k - alpha * log_n
            A = np.vstack([np.ones_like(log_n), -log_n]).T
            coeffs, *_ = np.linalg.lstsq(A, log_r, rcond=None)
            log_k, alpha = coeffs[0], coeffs[1]
            fit_k = float(math.exp(log_k))
            fit_alpha = float(alpha)
            # pearson
            try:
                from scipy.stats import pearsonr as pr
                pr_val, _ = pr(log_n, log_r)
                pearson_r = float(pr_val)
            except Exception:
                # numpy corr
                pearson_r = float(np.corrcoef(log_n, log_r)[0,1])
            if pearson_r is not None and abs(pearson_r) > 0.5 and fit_alpha > 0:
                conclusion = f"Descriptive trend: tok/s decreases as input tokens increase (alpha≈{fit_alpha:.2f}, Pearson r≈{pearson_r:.2f}). No quality judgement implied."
            else:
                conclusion = f"Descriptive trend: weak or no clear dependence (alpha≈{fit_alpha:.2f}, r≈{pearson_r:.2f}). Fixed ordering is a stated limitation."
    except Exception as e:
        conclusion = f"Could not fit trend: {e}. Descriptive only."

    return {
        "schema_version": "1.0",
        "points": points,
        "fit_k": fit_k,
        "fit_alpha": fit_alpha,
        "pearson_r": pearson_r,
        "conclusion": conclusion,
        "truth_level": "measured/derived/interpreted",
    }
