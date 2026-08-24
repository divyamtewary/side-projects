from __future__ import annotations
import json
import pathlib
from typing import Dict, Any

def _md_escape(s: str) -> str:
    return s.replace("|","\\|") if s else s

def build_overview_markdown(manifest: Dict[str, Any], env: Dict[str, Any], model_info: Dict[str, Any],
                            perf: Dict[str, Any], ctx: Dict[str,Any], dec: Dict[str,Any], acc: Dict[str,Any]) -> str:
    lines = []
    lines.append(f"# SLM Evaluation – {manifest.get('model_id','unknown')}")
    lines.append("")
    lines.append(f"**Run ID:** `{manifest.get('run_id')}`  •  **Date (UTC):** {manifest.get('created_utc')}  •  **Preset:** {manifest.get('preset')}  •  **Runtime:** {manifest.get('runtime_type')}")
    lines.append("")
    lines.append("> Evidence taxonomy: **MEASURED** = direct reading, **DERIVED** = computed from measured (formula shown), **INTERPRETED** = descriptive summary – no pass/fail quality claim.")
    lines.append("")
    lines.append("---")
    lines.append("")
    # Environment
    lines.append("## 1  Environment Snapshot (MEASURED)")
    lines.append("")
    if env:
        lines.append(f"- Platform: {env.get('platform')} {env.get('platform_version')}")
        lines.append(f"- Python: {env.get('python_version')}  •  torch {env.get('torch_version') or 'not installed'}  •  transformers {env.get('transformers_version') or 'not installed'}")
        lines.append(f"- CPU: {env.get('cpu_count_logical')} logical / {env.get('cpu_count_physical')} physical  •  RAM {env.get('ram_total_gb')} GB (avail {env.get('ram_available_gb')} GB)")
        lines.append(f"- CUDA available: {env.get('cuda_available')}")
    lines.append("")
    # Model
    lines.append("## 2  Model Inspection (MEASURED)")
    lines.append("")
    if model_info:
        lines.append(f"- Model ID: `{model_info.get('model_id')}`")
        lines.append(f"- Architecture: {model_info.get('architecture')}  •  Source: {model_info.get('source')}")
        if model_info.get('num_parameters'):
            n = model_info['num_parameters']
            lines.append(f"- Parameters: {n:,} ({n/1e9:.2f} B)" if n>1e9 else f"- Parameters: {n:,} ({n/1e6:.1f} M)")
        lines.append(f"- Layers: {model_info.get('num_layers')}  •  Heads: {model_info.get('num_heads')} (KV={model_info.get('num_kv_heads')})  •  Hidden: {model_info.get('hidden_size')}  •  Vocab: {model_info.get('vocab_size')}")
        lines.append(f"- Dtype: {model_info.get('torch_dtype')}  •  Quantization: {model_info.get('quantization') or 'none'}")
        # files table
        files = model_info.get("files",[])
        if files:
            lines.append("")
            lines.append("| File | Size (MB) |")
            lines.append("|---|---|")
            for f in files[:12]:
                lines.append(f"| {f.get('name')} | {f.get('size_mb')} |")
            if len(files)>12:
                lines.append(f"| … +{len(files)-12} more | |")
    lines.append("")
    # Performance
    lines.append("## 3  Baseline Performance (MEASURED + DERIVED)")
    lines.append("")
    lines.append("> TTFT is **streaming-approximation** via TextIteratorStreamer first-token time, distinct from true prefill.  Fixed prompt ordering is a stated limitation. Auto precision selection is used (no claim that FP16 is fastest on CPU).")
    lines.append("")
    if perf and "error" not in perf:
        lines.append("| Metric | Value | Truth |")
        lines.append("|---|---|---|")
        lines.append(f"| Input tokens | {perf.get('num_input_tokens')} | MEASURED |")
        lines.append(f"| Output tokens | {perf.get('num_output_tokens')} | MEASURED |")
        lines.append(f"| Load time | {perf.get('load_time_s')} s | MEASURED |")
        lines.append(f"| Prefill (≈TTFT) | {perf.get('prefill_time_s')} s | MEASURED (≈) |")
        lines.append(f"| Decode time | {perf.get('decode_time_s')} s | MEASURED |")
        lines.append(f"| Total time | {perf.get('total_time_s')} s | MEASURED |")
        lines.append(f"| Prefill tok/s | {perf.get('prefill_tok_per_s')} | DERIVED = input / prefill |")
        lines.append(f"| **Decode tok/s** | **{perf.get('decode_tok_per_s')}** | **DERIVED = output / decode (primary)** |")
        lines.append(f"| Overall tok/s | {perf.get('overall_tok_per_s')} | DERIVED = output / total |")
        lines.append(f"| Inter-token mean | {perf.get('inter_token_latency_ms_mean')} ms | DERIVED |")
        lines.append(f"| Median | {perf.get('inter_token_latency_ms_median')} ms | DERIVED |")
        lines.append(f"| P95 | {perf.get('inter_token_latency_ms_p95')} ms | DERIVED |")
        lines.append(f"| Peak RSS | {perf.get('peak_rss_mb') or 'n/a'} MB | MEASURED |")
        if perf.get("_dispersion"):
            d=perf["_dispersion"]
            lines.append("")
            lines.append(f"_Dispersion (standard, 3 runs): decode tok/s σ={d.get('decode_tok_per_s_std')}  total σ={d.get('total_time_s_std')} s_")
    else:
        lines.append(f"_Performance data unavailable: {perf.get('error') if perf else 'no data'}_")
    lines.append("")
    lines.append("**Charts:** `charts/01_performance.png` and `charts/02_latency.png`")
    lines.append("")
    # Context scaling
    lines.append("## 4  Context Scaling H1 (MEASURED + DERIVED + INTERPRETED descriptive)")
    lines.append("")
    if ctx and "error" not in ctx and ctx.get("points"):
        lines.append("| Context | Input tok | Chars | Output tok | Total (s) | tok/s | ms/tok | Preview |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for p in ctx["points"]:
            lines.append(f"| {p['label']} | {p['num_input_tokens']} | {p['context_chars']} | {p['num_output_tokens']} | {p['total_time_s']} | {p['tok_per_sec']} | {p['ms_per_token']} | {_md_escape(p['response_preview'][:60])}… |")
        lines.append("")
        lines.append(f"- Fit: r = k / n^α  with k≈{ctx.get('fit_k'):.2f} α≈{ctx.get('fit_alpha'):.3f} Pearson r≈{ctx.get('pearson_r'):.3f} (DERIVED via log-linear least squares)." if ctx.get('fit_k') else "- Fit: insufficient points.")
        lines.append(f"- Conclusion (INTERPRETED, descriptive only): {ctx.get('conclusion')}")
    else:
        lines.append(f"_Context scaling unavailable: {ctx.get('error') if ctx else 'no data'}_")
    lines.append("")
    lines.append("**Chart:** `charts/03_context_scaling.png`")
    lines.append("")
    # Decode position
    lines.append("## 5  Decode-position H2 (MEASURED + DERIVED)")
    lines.append("")
    if dec and "error" not in dec and dec.get("latencies_ms"):
        lines.append(f"- Input tokens: {dec.get('input_tokens')}  Output tokens: {dec.get('output_tokens')} (MEASURED)")
        lines.append(f"- Slope: {dec.get('slope_ms_per_token'):.4f} ms/token (DERIVED via linear regression, binned 50-token means).")
        lines.append(f"- Conclusion: {dec.get('conclusion')}")
        lines.append(f"- Binned means (ms): {', '.join(str(x) for x in dec.get('binned_means_ms',[]))}")
    else:
        lines.append(f"_Decode-position unavailable: {dec.get('error') if dec else 'no data'}_")
    lines.append("")
    lines.append("**Chart:** `charts/04_decode_position.png`")
    lines.append("")
    # Accuracy
    lines.append("## 6  Accuracy / Grounding Probe (MEASURED + INTERPRETED)")
    lines.append("")
    lines.append("> Synthetic neutral passages are used (explicitly labelled synthetic). No domain-specific aviation text from contaminated fixtures.")
    lines.append("")
    if acc and "error" not in acc:
        lines.append(f"- Prompt: {acc.get('prompt','')[:200]}…")
        lines.append(f"- Expected facts ({len(acc.get('expected_facts',[]))}):")
        for f in acc.get("expected_facts",[]):
            lines.append(f"  - {f}")
        lines.append(f"- Response preview: {acc.get('response','')[:300].replace(chr(10),' ')}…")
        lines.append(f"- **Score: {acc.get('accuracy_score',0)*100:.0f}% ({len(acc.get('matched_facts',[]))}/{len(acc.get('expected_facts',[]))})** (DERIVED = matched/expected)")
        lines.append(f"- Audit details:")
        for d in acc.get("audit_details",[]):
            mark = "✓" if d.get("matched") else "✗"
            lines.append(f"  - {mark} {d.get('fact')}")
    else:
        lines.append(f"_Accuracy probe unavailable: {acc.get('error') if acc else 'no data'}_")
    lines.append("")
    lines.append("**Chart:** `charts/05_accuracy.png`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7  Summary (INTERPRETED, descriptive only)")
    lines.append("")
    # beginner / expert
    if perf and ctx:
        # short interpreted
        lines.append("**Beginner takeaway:** This model was measured on your machine at about "
                     f"{perf.get('decode_tok_per_s','?')} tok/s for short prompts, slowing as context grows (see H1). "
                     "The charts show where time is spent (prefill vs decode) and how consistent each token is. No quality judgement is made – these are measurements on synthetic tasks.")
        lines.append("")
        lines.append("**Expert takeaway:** "
                     f"Decode tok/s {perf.get('decode_tok_per_s')} (overall {perf.get('overall_tok_per_s')}) ; "
                     f"TTFT≈{perf.get('prefill_time_s')}s (streaming approx, prefill {perf.get('prefill_tok_per_s')} tok/s). "
                     f"H1 power-law α≈{ctx.get('fit_alpha','?')} (Pearson r≈{ctx.get('pearson_r','?')}). "
                     f"H2 slope {dec.get('slope_ms_per_token','?') if dec else '?'} ms/tok. "
                     f"Grounding {acc.get('accuracy_score',0)*100 if acc else 0:.0f}%. "
                     "Fixed ordering, warm-up excluded, synthetic passages.")
    lines.append("")
    lines.append("**Limitations (stated beside result they qualify):**")
    lines.append("- TTFT via streaming timestamps is an approximation, not true prefill kernel time.")
    lines.append("- Fixed prompt ordering; no randomized order or interleaving.")
    lines.append("- AUTO precision selection; no assertion FP16 fastest on CPU.")
    lines.append("- Single-machine measurement; not comparable across hardware without environment.json.")
    lines.append("- One-number quality score is deliberately not provided; evidence cards (sections 3-6) replace it.")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by slm-evaluation-suite v0.1.0 (schema {manifest.get('schema_version')}) • {manifest.get('created_utc')} UTC • Reproduce via run_id {manifest.get('run_id')} (config SHA {manifest.get('config_sha256')})*")
    lines.append("")
    return "\n".join(lines)

def build_summary_json(manifest, env, model_info, perf, ctx, dec, acc) -> Dict[str, Any]:
    # combine into EvaluationSummary shape – also include overall summary text
    beginner = ""
    expert = ""
    if perf and ctx:
        beginner = f"Measured {perf.get('decode_tok_per_s','?')} tok/s (decode) on this machine; speed falls as input grows (H1). See charts."
        expert = f"Decode {perf.get('decode_tok_per_s')} tok/s, TTFT≈{perf.get('prefill_time_s')}s, H1 α≈{ctx.get('fit_alpha')}, H2 slope {dec.get('slope_ms_per_token') if dec else 'n/a'}, grounding {acc.get('accuracy_score',0) if acc else 0:.0f}"
    return {
        "schema_version": manifest.get("schema_version","1.0"),
        "manifest": manifest,
        "environment": env,
        "model_info": model_info,
        "performance": perf,
        "context_scaling": ctx,
        "decode_position": dec,
        "accuracy": acc,
        "overall_summary": f"Evaluation of {manifest.get('model_id')} ({manifest.get('preset')}) completed at {manifest.get('created_utc')}. Descriptive only.",
        "limitations": [
            "TTFT is streaming-approximation",
            "Fixed ordering",
            "AUTO precision",
            "Single-machine not cross-hw comparable",
            "No single quality score – evidence cards only"
        ],
        "beginner_takeaway": beginner,
        "expert_takeaway": expert,
    }
