from __future__ import annotations
from typing import Dict, Any, Callable

SYNTHETIC_FACTS = [
    "The synthetic dataset contains 42 reference samples.",
    "Mean neutral score is 3.14 across synthetic passages.",
    "Dataset version is v0.1.0 synthetic.",
]

SYNTHETIC_CONTEXT = (
    "Synthetic neutral context (labelled as synthetic, no aviation domain): "
    "The synthetic dataset v0.1.0 contains 42 reference samples collected under neutral conditions. "
    "Across all synthetic passages the mean neutral score is 3.14. "
    "The dataset is versioned as v0.1.0 synthetic and is used for evaluation calibration. "
    "Additional neutral filler about photosynthesis and water cycle is included but irrelevant to the factual probes."
)

PROMPT_TMPL = "Using only the context below, answer: What are the three key facts about the synthetic dataset? Context: {context}"

def audit_accuracy(text: str, facts: list[str]) -> tuple[float, list[dict]]:
    """
    Simple exact-substring/keyword audit (as in notebook). Returns (score, details).
    For each fact, checks if key tokens appear in text. Descriptive only.
    """
    details = []
    matched = 0
    low = text.lower()
    for fact in facts:
        # key tokens are numbers/versions
        keywords = [t for t in fact.replace("."," ").split() if any(c.isdigit() for c in t) or "v0" in t]
        if not keywords:
            keywords = fact.split()[:2]
        hit = all(k.lower().strip(".,") in low for k in keywords)
        # also try whole fact fuzzy: if 70% words present
        if not hit:
            words = [w.lower().strip(".,") for w in fact.split()]
            present = sum(1 for w in words if w in low)
            hit = present / max(len(words),1) >= 0.6
        if hit:
            matched += 1
        details.append({"fact": fact, "matched": hit, "keywords": keywords})
    score = matched / max(len(facts),1)
    return round(score,3), details

def run_accuracy(runtime, preset: str, on_log: Callable = None) -> Dict[str, Any]:
    def log(m):
        if on_log:
            on_log(m)
    log("Running accuracy / grounding probe (synthetic neutral facts)...")
    prompt = PROMPT_TMPL.format(context=SYNTHETIC_CONTEXT)
    try:
        if hasattr(runtime, "accuracy_probe") and runtime.__class__.__name__ == "FakeRuntime":
            # use fake helper but keep our audit for consistency
            probe = runtime.accuracy_probe(preset)
            resp = probe["response"]
            facts = probe["expected_facts"]
            score, details = audit_accuracy(resp, facts)
            return {
                "schema_version": "1.0",
                "prompt": prompt,
                "expected_facts": facts,
                "response": resp,
                "matched_facts": [d["fact"] for d in details if d["matched"]],
                "accuracy_score": score,
                "audit_details": details,
                "truth_level": "measured/interpreted",
            }
        else:
            out = runtime.generate(prompt, max_new_tokens=150, do_sample=False)
            resp = out.get("response","")
            score, details = audit_accuracy(resp, SYNTHETIC_FACTS)
            return {
                "schema_version": "1.0",
                "prompt": prompt,
                "expected_facts": SYNTHETIC_FACTS,
                "response": resp,
                "matched_facts": [d["fact"] for d in details if d["matched"]],
                "accuracy_score": score,
                "audit_details": details,
                "truth_level": "measured/interpreted",
            }
    except Exception as e:
        log(f"Accuracy probe failed: {e}")
        return {
            "schema_version": "1.0",
            "prompt": prompt,
            "expected_facts": SYNTHETIC_FACTS,
            "response": f"ERROR: {e}",
            "matched_facts": [],
            "accuracy_score": 0.0,
            "audit_details": [],
            "truth_level": "measured/interpreted",
        }
