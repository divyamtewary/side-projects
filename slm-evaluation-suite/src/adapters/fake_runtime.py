from __future__ import annotations
import time
import random
import math
from typing import Dict, Any, List, Tuple

# Synthetic neutral passages labelled as synthetic – no aviation domain
SYNTHETIC_PASSAGES = [
    ("L1: Minimal", "Synthetic passage A. " * 12, 75),
    ("L2: Short", "Synthetic passage B with neutral scientific filler about light refraction in water. " * 18, 236),
    ("L3: Medium", "Synthetic passage C describing the water cycle and neutral geography of river basins. " * 28, 428),
    ("L4: Long", "Synthetic passage D about the history of cartography and neutral trade routes. " * 42, 641),
    ("L5: Extended", "Synthetic passage E combining neutral biology of photosynthesis and open-domain encyclopedic filler. " * 75, 1169),
]

# For accuracy probe – neutral facts
SYNTHETIC_FACTS = [
    "The synthetic dataset contains 42 reference samples.",
    "Mean neutral score is 3.14 across synthetic passages.",
    "Dataset version is v0.1.0 synthetic.",
]

def simulate_generation(num_input_tokens: int, num_output_tokens: int = 128, base_tps: float = 6.0, noise: float = 0.15) -> Dict[str, Any]:
    """
    Simulate timing: tok/s decreases with input length (power law) and slightly with output position.
    Returns dict with same keys as real stream_generate.
    """
    # power law: r = k / n^alpha ; pick k approx base_tps * (100^0.35) to normalize
    alpha = 0.35
    k = base_tps * (100 ** alpha)
    r = k / (max(num_input_tokens, 10) ** alpha)
    r = max(1.0, r * random.uniform(1 - noise, 1 + noise))
    # TTFT approx prefill_time
    prefill_time = max(0.2, (num_input_tokens / max(r*3, 6)) * random.uniform(0.85, 1.15))
    decode_time = num_output_tokens / r
    total_time = prefill_time + decode_time
    # inter-token latencies: mean ~ 1000/r ms
    mean_ms = 1000.0 / r
    intervals = [random.gauss(mean_ms, mean_ms*0.12) for _ in range(num_output_tokens-1)]
    intervals = [max(10.0, x) for x in intervals]
    # compute percentiles
    import numpy as np
    arr = np.array(intervals) if intervals else np.array([mean_ms])
    # token timestamps simulated
    now = time.time()
    timestamps = []
    t = now + prefill_time
    timestamps.append((1, t))
    for i, iv in enumerate(intervals, start=2):
        t += iv/1000.0
        timestamps.append((i, t))
    return {
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "prefill_time_s": round(prefill_time, 3),
        "decode_time_s": round(decode_time, 3),
        "total_time_s": round(total_time, 3),
        "prefill_tokens_per_second": round(num_input_tokens / prefill_time, 2) if prefill_time else 0,
        "tokens_per_second_decode": round(r, 2),
        "tokens_per_second_overall": round(num_output_tokens/total_time, 2) if total_time else 0,
        "token_timestamps": timestamps,
        "intervals_ms": intervals,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr,95)),
        "p99_ms": float(np.percentile(arr,99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "response": "Synthetic model output for evaluation – neutral and deterministic filler. " * 8,
    }

class FakeRuntime:
    """
    Deterministic fake runtime that mimics LocalTransformersRuntime interface
    without loading any model. Useful for UI demo, CI, and when torch is absent.
    """
    def __init__(self, model_id: str = "fake://tiny"):
        self.model_id = model_id
        self.loaded = False
        self.load_time_s = 0.0
        self.model_info = {
            "schema_version": "1.0",
            "model_id": model_id,
            "source": "fake",
            "local_path": None,
            "architecture": "FakeDecoderForCausalLM",
            "num_parameters": 135_000_000,
            "num_layers": 12,
            "num_heads": 8,
            "num_kv_heads": 2,
            "hidden_size": 768,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
            "torch_dtype": "float32",
            "quantization": None,
            "config_raw": {"model_type": "fake", "hidden_size": 768, "num_hidden_layers": 12},
            "tokenizer_vocab_size": 32000,
            "files": [{"name": "config.json", "size_bytes": 1200, "size_mb": 0.0}],
            "truth_level": "measured",
        }

    def load(self):
        t0 = time.time()
        time.sleep(0.35 + random.uniform(0, 0.25))
        self.load_time_s = time.time() - t0
        self.loaded = True
        return self.model_info, self.load_time_s

    def generate(self, prompt: str, max_new_tokens: int = 128, do_sample: bool = False) -> Dict[str, Any]:
        approx_input = max(10, len(prompt.split()) * 1.3)
        return simulate_generation(int(approx_input), max_new_tokens)

    def context_scaling_sweep(self, preset: str = "quick"):
        points = []
        # pick subset for quick vs standard
        max_points = 3 if preset == "quick" else 5
        selected = SYNTHETIC_PASSAGES[:max_points]
        for label, passage, tok in selected:
            res = simulate_generation(tok, 128 if preset == "quick" else 150)
            points.append({
                "label": label,
                "num_input_tokens": tok,
                "context_chars": len(passage),
                "num_output_tokens": 128,
                "total_time_s": res["total_time_s"],
                "tok_per_sec": res["tokens_per_second_decode"],
                "ms_per_token": res["mean_ms"],
                "response_preview": res["response"][:120],
            })
            time.sleep(0.25)
        return points

    def decode_position_trace(self, max_tokens: int = 400):
        # simulate latencies increasing slightly with position
        base = 155
        latencies = []
        for i in range(max_tokens):
            # linear slope + noise
            lat = base + (i * 0.18) + random.gauss(0, 8)
            latencies.append(max(40, lat))
        return latencies

    def accuracy_probe(self, preset: str = "quick"):
        # synthetic facts audit
        resp = "The synthetic dataset contains 42 reference samples. Dataset version is v0.1.0 synthetic. " + ("Neutral filler. "*20)
        matched = [f for f in SYNTHETIC_FACTS if any(tok in resp for tok in f.split()[:3])]
        # fake slightly imperfect
        score = len(matched)/len(SYNTHETIC_FACTS)
        return {
            "prompt": "Synthetic QA: Summarize the synthetic dataset facts.",
            "expected_facts": SYNTHETIC_FACTS,
            "response": resp,
            "matched_facts": matched,
            "accuracy_score": round(score, 3),
            "audit_details": [{"fact": f, "matched": f in matched} for f in SYNTHETIC_FACTS],
        }
