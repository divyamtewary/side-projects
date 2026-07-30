"""
NEURAL OBSERVATORY
Architecture-aware computation visualiser for small causal language models.

Watch a transformer think. Per-layer residual norms, attention-head entropy,
MLP sparsity, RMS-norm scale factors, and a logit lens that shows which layer
the model actually made up its mind at.

Runs on CPU. Works with any Hugging Face causal LM that has a standard decoder
stack; ships pointed at a 270M model so it fits on a laptop.

    streamlit run app.py

The model is chosen with the MODEL_ID environment variable or from the sidebar.
"""

import streamlit as st
import torch, torch.nn.functional as F
import numpy as np
import time, os, json, string, hashlib, datetime, math, io
import psutil
from collections import defaultdict
from threading import Thread
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Neural Observatory",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# MODEL SELECTION
# ======================================================================
# Any HF causal LM with a standard decoder stack works. These are the ones
# actually tested on CPU; the sidebar accepts arbitrary repo ids or local paths.
# The default is deliberately ungated so a fresh clone runs with no HF account.
MODEL_CHOICES = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "google/gemma-3-270m-it",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "google/gemma-3-1b-it",
]
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID") or MODEL_CHOICES[0]

# ======================================================================
# MODEL SPEC  (populated from model.config at load time)
# ======================================================================
SPEC = {
    "name": DEFAULT_MODEL_ID,
    "parameters": "—",
    "context_length": "—",
    "architecture": "Decoder-only transformer",
    "num_layers": 0,
    "num_heads": 0,
    "num_kv_heads": 0,
    "hidden_dim": 0,
    "head_dim": 0,
    "activation": "—",
    "normalization": "RMS Norm",
    "position_embedding": "RoPE",
    "license": "see model card",
    "release_date": "—",
}


def grid_shape(n):
    """Near-square (rows, cols) factorisation of n, for laying a hidden state out as an image."""
    r = int(math.isqrt(n))
    while r > 1 and n % r:
        r -= 1
    return (r, n // r) if r > 1 else (1, n)


def volume_shape(n):
    """Near-cubic (x, y, z) factorisation of n, for the voxel view."""
    x = round(n ** (1 / 3)) or 1
    while x > 1 and n % x:
        x -= 1
    m = n // max(x, 1)
    y = int(math.isqrt(m))
    while y > 1 and m % y:
        y -= 1
    return (max(x, 1), max(y, 1), m // max(y, 1))


DISPLAY_HEADS = 4
INPUT_ANALYSIS_TOKEN_LIMIT = 512
# ======================================================================
# THEME
# ======================================================================
BG_DEEP    = "#09090b"
BG_PRIMARY = "#111114"
BG_SURFACE = "#17181d"
BG_HOVER   = "#20232b"
BORDER     = "rgba(255,255,255,0.08)"
BORDER_LIT = "rgba(168,85,247,0.28)"
ACCENT     = "#a855f7"
ACCENT_DIM = "rgba(168,85,247,0.14)"
GREEN      = "#22c55e"
AMBER      = "#f59e0b"
RED        = "#ef4444"
PURPLE     = "#8b5cf6"
CYAN       = "#22d3ee"
PINK       = "#ec4899"
TEXT_1     = "#f5f7fb"
TEXT_2     = "#a1a1aa"
TEXT_3     = "#71717a"
WHITE      = "#ffffff"
VIVID = [ACCENT, GREEN, AMBER, RED, PURPLE, CYAN, PINK, "#14b8a6", "#f97316", "#6366f1"]

# ======================================================================
# CSS
# ======================================================================
st.markdown(f"""
<style>
.stApp {{
    background:
        radial-gradient(circle at top right, rgba(168,85,247,0.12), transparent 28%),
        radial-gradient(circle at bottom left, rgba(34,211,238,0.08), transparent 24%),
        linear-gradient(180deg, #070709 0%, {BG_DEEP} 42%, #060607 100%);
}}
header[data-testid="stHeader"] {{
    background: transparent !important;
    backdrop-filter: none !important;
}}
#MainMenu {{ visibility: hidden; }}
footer  {{ visibility: hidden; }}

div.block-container {{
    padding-top: 1.25rem;
    padding-bottom: 1.6rem;
    max-width: 1680px;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(17,17,20,0.98) 0%, rgba(12,12,14,0.98) 100%);
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {{ color: {TEXT_1}; }}
section[data-testid="stSidebar"] hr {{ border-color: {BORDER}; }}

.stChatMessage {{
    background: linear-gradient(180deg, rgba(24,24,29,0.95) 0%, rgba(18,18,22,0.95) 100%);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 10px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.22);
}}
.stChatInput > div {{
    background: rgba(23,24,29,0.98) !important;
    border: 1px solid {BORDER_LIT} !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 48px rgba(0,0,0,0.28);
}}
.stChatInput textarea {{
    background: transparent !important;
    color: {TEXT_1} !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: rgba(17,17,20,0.88);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 6px;
    gap: 6px;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 12px;
    color: {TEXT_2};
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(90deg, rgba(168,85,247,0.22) 0%, rgba(34,211,238,0.18) 100%) !important;
    border: 1px solid {BORDER_LIT} !important;
    color: {WHITE} !important;
}}

div[data-testid="stMetric"] {{
    background: linear-gradient(180deg, rgba(24,24,29,0.96) 0%, rgba(18,18,22,0.96) 100%);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 14px 32px rgba(0,0,0,0.22);
}}
div[data-testid="stMetric"] label {{ color: {TEXT_2} !important; font-size: 12px !important; }}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color: {TEXT_1} !important; font-weight: 700; }}

.stButton > button {{
    background: {ACCENT_DIM};
    border: 1px solid {BORDER_LIT};
    color: {TEXT_1};
    border-radius: 14px;
    font-weight: 600;
    transition: all 0.2s;
    box-shadow: 0 10px 24px rgba(0,0,0,0.18);
}}
.stButton > button:hover {{ background: {BG_HOVER}; border-color: {ACCENT}; }}

.stDownloadButton > button {{
    background: rgba(34,211,238,0.09);
    border: 1px solid rgba(34,211,238,0.22);
    color: {TEXT_1};
    border-radius: 14px;
}}

.streamlit-expanderHeader {{
    background: {BG_SURFACE};
    border-radius: 12px;
    color: {TEXT_1} !important;
}}
details {{ border: 1px solid {BORDER} !important; border-radius: 12px; }}

::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG_PRIMARY}; }}
::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 4px; }}
.js-plotly-plot .plotly .main-svg {{ background: transparent !important; }}

.glass {{
    background: linear-gradient(180deg, rgba(24,24,29,0.95) 0%, rgba(16,16,20,0.95) 100%);
    border: 1px solid {BORDER};
    border-radius: 22px;
    padding: 20px 22px;
    margin-bottom: 12px;
    backdrop-filter: blur(18px);
    box-shadow: 0 18px 48px rgba(0,0,0,0.25);
}}
.obs-header {{
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0 12px 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 14px;
}}
.obs-header h3 {{ margin: 0; color: {TEXT_1}; font-weight: 600; font-size: 1.05rem; }}
.obs-header .dot {{ width: 8px; height: 8px; border-radius: 50%; background: {GREEN}; display: inline-block; }}
.stat-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
.stat-chip {{
    background: rgba(255,255,255,0.03);
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 12px;
    color: {TEXT_2};
    font-family: 'Segoe UI', sans-serif;
}}
.stat-chip b {{ color: {TEXT_1}; }}
.spec-card {{
    background: linear-gradient(180deg, rgba(24,24,29,0.95) 0%, rgba(18,18,22,0.95) 100%);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 12px 16px;
    margin-bottom: 8px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.2);
}}
.spec-card .lbl {{ color: {TEXT_3}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
.spec-card .val {{ color: {TEXT_1}; font-weight: 700; font-size: 1.05rem; }}

.deck-kicker {{
    color: {PURPLE};
    font-size: 11px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.35rem;
}}
.deck-title {{
    color: {TEXT_1};
    font-size: 2rem;
    line-height: 1.08;
    font-weight: 800;
    margin: 0;
}}
.deck-subtitle {{
    color: {TEXT_2};
    font-size: 0.98rem;
    line-height: 1.65;
    margin-top: 0.55rem;
}}
.insight-card {{
    background: linear-gradient(180deg, rgba(24,24,29,0.95) 0%, rgba(17,17,20,0.95) 100%);
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.22);
}}
.insight-card .ic-title {{ color: {TEXT_1}; font-size: 0.95rem; font-weight: 700; margin-bottom: 0.2rem; }}
.insight-card .ic-value {{ color: {PURPLE}; font-size: 1.32rem; font-weight: 800; margin-bottom: 0.25rem; }}
.insight-card .ic-caption {{ color: {TEXT_2}; font-size: 0.82rem; line-height: 1.55; }}
.hero-panel {{
    padding: 26px 28px;
    border-radius: 24px;
    border: 1px solid {BORDER};
    background: linear-gradient(135deg, rgba(24,24,29,0.98) 0%, rgba(15,15,18,0.98) 100%);
    box-shadow: 0 24px 64px rgba(0,0,0,0.3);
    margin-bottom: 14px;
}}
.soft-divider {{
    height: 1px;
    margin: 0.7rem 0 1.1rem 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.12) 25%, rgba(255,255,255,0.06) 75%, transparent 100%);
}}
.empty-deck {{
    padding: 54px 24px;
    text-align: center;
    border-radius: 24px;
    border: 1px dashed {BORDER_LIT};
    background: rgba(255,255,255,0.02);
}}
.empty-deck h3 {{ color: {TEXT_1}; margin-bottom: 0.5rem; }}
.empty-deck p {{ color: {TEXT_2}; max-width: 560px; margin: 0 auto; line-height: 1.7; }}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# LOAD MODEL
# ======================================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import LogitsProcessor, LogitsProcessorList

    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(min(os.cpu_count(), 4))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        # eager is mandatory: sdpa and flash-attention never materialise the
        # attention matrix, so there is nothing for the hooks to read.
        attn_implementation="eager",
    )
    model.eval()

    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    n_heads = getattr(cfg, "num_attention_heads", 0) or 0
    hidden = getattr(cfg, "hidden_size", 0) or 0
    ctx = getattr(cfg, "max_position_embeddings", 0) or 0
    arch = (getattr(cfg, "architectures", None) or [type(model).__name__])[0]

    spec = {
        "name": model_id,
        "parameters": f"{n_params / 1e6:.0f} M" if n_params < 1e9 else f"{n_params / 1e9:.2f} B",
        "context_length": f"{ctx / 1000:.1f} K" if ctx else "—",
        "architecture": arch,
        "num_layers": getattr(cfg, "num_hidden_layers", 0),
        "num_heads": n_heads,
        "num_kv_heads": getattr(cfg, "num_key_value_heads", n_heads),
        "hidden_dim": hidden,
        "head_dim": getattr(cfg, "head_dim", None) or (hidden // n_heads if n_heads else 0),
        "activation": getattr(cfg, "hidden_activation", None) or getattr(cfg, "hidden_act", "—"),
        "normalization": "RMS Norm",
        "license": "see model card",
    }
    SPEC.update(spec)

    # Whitespace is spelled differently by different tokenizer families.
    # SentencePiece (Gemma, Llama 1/2) marks a leading space with U+2581.
    # Byte-level BPE (GPT-2, Llama 3, SmolLM2, Qwen) uses U+0120 for space
    # and U+010A for newline. Whitelisting only the SentencePiece marker bans
    # every space-prefixed token on a BPE model, and the output comes out as
    # onelongwordlikethis.
    allowed_chars = set(string.printable) | {
        "\u2581",  # SentencePiece space
        "\u0120",  # byte-level BPE space
        "\u010a",  # byte-level BPE newline
        "\u0109",  # byte-level BPE tab
        "\u010d",  # byte-level BPE carriage return
    }
    banned_ids = []
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        if tid in tokenizer.all_special_ids:
            continue
        if any(c not in allowed_chars for c in tok):
            banned_ids.append(tid)

    class EnglishOnly(LogitsProcessor):
        def __init__(self, bids, vs):
            self.mask = torch.zeros(vs, dtype=torch.bool)
            self.mask[bids] = True
        def __call__(self, input_ids, scores):
            vsz = scores.shape[-1]
            m = self.mask.to(scores.device)
            if vsz > len(m):
                m = torch.cat([m, torch.ones(vsz - len(m), dtype=torch.bool, device=scores.device)])
            elif vsz < len(m):
                m = m[:vsz]
            scores[:, m] = -float("inf")
            return scores

    logits_proc = LogitsProcessorList([EnglishOnly(banned_ids, len(vocab))])
    return model, tokenizer, logits_proc, spec


# ======================================================================
# ARCHITECTURE-AWARE OBSERVABILITY ENGINE
# ======================================================================
class ArchitectureObserver:
    """
    Forward hooks on every sub-module of a standard decoder stack:
      - input_layernorm            (RMS Norm, pre-attention)
      - self_attn                  (Multi-Head Attention)
      - post_attention_layernorm   (RMS Norm, pre-MLP)
      - mlp.gate_proj              (Linear -> captures the pre-activation)
      - mlp                        (full MLP output)
      - the layer itself           (residual stream after each block)

    One set per layer. Hooks unregister themselves if RAM crosses the safety
    threshold, because captured tensors add up faster than you expect.
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._registered = False
        self.num_hooked_layers = 0
        self.memory_limit_reached = False

        # -- data stores (reset each generation) --
        self.rms_norm_data = {}        # {layer_idx: {pre_attn: {...}, post_attn: {...}}}
        self.attention_data = {}       # {layer_idx: {output_stats, head_entropy, weights, ...}}
        self.gelu_data = {}            # {layer_idx: {pre_gelu_*, post_gelu_*, dead_neuron_frac}}
        self.mlp_data = {}             # {layer_idx: {mean, std, norm, sparsity}}
        self.residual_stream = {}      # {layer_idx: {norm, mean, std, max, min}}
        self.attn_output_data = {}     # {layer_idx: {mean, std, norm}}
        self.token_probs = []          # [{tokens, probs, entropy, top1_confidence}, ...]
        self.per_step_head_entropy = defaultdict(dict)  # {step: {layer: [16 floats]}}
        self.step_activations = defaultdict(dict)       # {step: {layer: [1024 floats]}}
        self.metadata = {}
        self._step = 0

    def check_memory(self):
        # Stop saving intermediate large structures if memory uses >85% limit
        if psutil.virtual_memory().percent > 85.0:
            if not self.memory_limit_reached:
                print("⚠️ SYSTEM MEMORY > 85% - UNREGISTERING OBSERVER HOOKS TO PREVENT OOM ⚠️")
                self.memory_limit_reached = True
                self._clear() # Unregister immediately
            return False
        return True

    # ------------------------------------------------------------------
    def register_hooks(self):
        if self._registered or self.memory_limit_reached:
            return
        self._clear()
        
        num_to_hook = min(SPEC["num_layers"], len(self.model.model.layers))
        self.num_hooked_layers = num_to_hook
        for i in range(num_to_hook):
            layer = self.model.model.layers[i]
            self.hooks.append(layer.input_layernorm.register_forward_hook(self._rms_hook(i, "pre_attn")))
            self.hooks.append(layer.self_attn.register_forward_hook(self._attn_hook(i)))
            self.hooks.append(layer.post_attention_layernorm.register_forward_hook(self._rms_hook(i, "post_attn")))
            if hasattr(layer.mlp, "gate_proj"):
                self.hooks.append(layer.mlp.gate_proj.register_forward_hook(self._gate_hook(i)))
            self.hooks.append(layer.mlp.register_forward_hook(self._mlp_hook(i)))
            self.hooks.append(layer.register_forward_hook(self._layer_hook(i)))
        self._registered = True

    def reset(self):
        self.rms_norm_data.clear()
        self.attention_data.clear()
        self.gelu_data.clear()
        self.mlp_data.clear()
        self.residual_stream.clear()
        self.attn_output_data.clear()
        self.token_probs.clear()
        self.per_step_head_entropy.clear()
        self.step_activations.clear()
        self.metadata.clear()
        self._step = 0
        self.memory_limit_reached = False

    def _clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self._registered = False
        self.num_hooked_layers = 0

    # ------------------------------------------------------------------
    # RMS NORM HOOKS
    # ------------------------------------------------------------------
    def _rms_hook(self, layer_idx, position):
        def fn(mod, inp, out):
            with torch.no_grad():
                x = inp[0].detach().cpu().float()
                y = out.detach().cpu().float()
                in_rms = x.pow(2).mean(-1).sqrt().mean().item()
                out_rms = y.pow(2).mean(-1).sqrt().mean().item()
                if layer_idx not in self.rms_norm_data:
                    self.rms_norm_data[layer_idx] = {}
                self.rms_norm_data[layer_idx][position] = dict(
                    input_rms=in_rms,
                    output_rms=out_rms,
                    scale_factor=out_rms / max(in_rms, 1e-10),
                    weight_mean=mod.weight.detach().cpu().float().mean().item(),
                    weight_std=mod.weight.detach().cpu().float().std().item(),
                )
        return fn

    # ------------------------------------------------------------------
    # ATTENTION HOOK  (captures per-head weights when available)
    # ------------------------------------------------------------------
    def _attn_hook(self, layer_idx):
        def fn(mod, inp, out):
            with torch.no_grad():
                attn_out = out[0] if isinstance(out, tuple) else out
                t = attn_out.detach().cpu().float()
                info = dict(mean=t.mean().item(), std=t.std().item(), norm=t.norm().item())

                aw = None
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    aw = out[1].detach().cpu().float()  # [B, H, Q, K]
                    # per-head entropy   (average over query positions)
                    ent = -(aw * (aw + 1e-10).log()).sum(-1).mean(dim=2)[0]  # [H]
                    info["head_entropy"] = ent.tolist()
                    # last-token attention vector per head
                    info["last_token_attn"] = aw[0, :, -1, :].numpy()  # [H, K]
                    # which position each head focuses on most (for last token)
                    info["head_focus_pos"] = aw[0, :, -1, :].argmax(-1).tolist()
                    # store per-step summary
                    self.per_step_head_entropy[self._step][layer_idx] = ent.tolist()

                self.attention_data[layer_idx] = info
        return fn

    # ------------------------------------------------------------------
    # GELU / GATE_PROJ HOOK
    # ------------------------------------------------------------------
    def _gate_hook(self, layer_idx):
        def fn(mod, inp, out):
            with torch.no_grad():
                pre = out.detach().cpu().float()
                post = F.gelu(pre)
                dead = (post.abs() < 1e-6).float().mean().item()
                flat_pre = pre.flatten()[:5000]
                flat_post = post.flatten()[:5000]
                self.gelu_data[layer_idx] = dict(
                    pre_mean=pre.mean().item(),
                    pre_std=pre.std().item(),
                    pre_min=pre.min().item(),
                    pre_max=pre.max().item(),
                    post_mean=post.mean().item(),
                    post_std=post.std().item(),
                    dead_fraction=dead,
                    pre_hist=flat_pre.numpy(),
                    post_hist=flat_post.numpy(),
                )
        return fn

    # ------------------------------------------------------------------
    # MLP HOOK
    # ------------------------------------------------------------------
    def _mlp_hook(self, layer_idx):
        def fn(mod, inp, out):
            with torch.no_grad():
                t = out.detach().cpu().float() if isinstance(out, torch.Tensor) else out[0].detach().cpu().float()
                self.mlp_data[layer_idx] = dict(
                    mean=t.mean().item(), std=t.std().item(),
                    norm=t.norm().item(), sparsity=(t.abs() < 1e-6).float().mean().item(),
                    max=t.max().item(), min=t.min().item(),
                )
        return fn

    # ------------------------------------------------------------------
    # LAYER HOOK  (residual stream)
    # ------------------------------------------------------------------
    def _layer_hook(self, layer_idx):
        def fn(mod, inp, out):
            if self.memory_limit_reached:
                return
            with torch.no_grad():
                step_idx = self._step
                t = out[0] if isinstance(out, tuple) else out
                t = t.detach().cpu().float()
                flat = t.flatten()[:10000]
                self.residual_stream[layer_idx] = dict(
                    norm=t.norm().item(), mean=t.mean().item(),
                    std=t.std().item(), max=t.max().item(),
                    min=t.min().item(),
                    kurtosis=self._kurtosis(flat),
                    histogram=flat.numpy(),
                )
                self.step_activations[step_idx][layer_idx] = t[0, -1, :].numpy()
                if self.num_hooked_layers and layer_idx == self.num_hooked_layers - 1:
                    self._step += 1
                    if self._step % 10 == 0:
                        self.check_memory() # Check memory limit every 10 steps to dodge OOM
        return fn

    @staticmethod
    def _kurtosis(t):
        m, s = t.mean(), t.std()
        if s < 1e-8: return 0.0
        return (((t - m) / s) ** 4).mean().item() - 3.0

    # ------------------------------------------------------------------
    # TOKEN PROBS
    # ------------------------------------------------------------------
    def capture_token_probs(self, logits, tokenizer, top_k=10):
        with torch.no_grad():
            probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
            tp, ti = torch.topk(probs[0], top_k)
            toks = [tokenizer.decode([idx]) for idx in ti.tolist()]
            return dict(
                tokens=toks, probabilities=tp.tolist(),
                entropy=-(probs * (probs + 1e-10).log()).sum(-1).item(),
                top1_confidence=tp[0].item(),
            )

    # ------------------------------------------------------------------
    # FULL ARTIFACT FOR DOWNLOAD
    # ------------------------------------------------------------------
    def build_artifact(self, messages, response_text):
        """Comprehensive experiment artifact."""
        def _s(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, dict): return {k: _s(v) for k, v in o.items()}
            if isinstance(o, list): return [_s(i) for i in o]
            if isinstance(o, (np.float32, np.float64)): return float(o)
            if isinstance(o, (np.int32, np.int64)): return int(o)
            return o

        # strip large arrays but keep stats
        gelu_clean = {}
        for k, v in self.gelu_data.items():
            gelu_clean[k] = {kk: vv for kk, vv in v.items() if "hist" not in kk}

        attn_clean = {}
        for k, v in self.attention_data.items():
            attn_clean[k] = {kk: vv for kk, vv in v.items() if kk not in ("last_token_attn",)}

        residual_clean = {}
        for k, v in self.residual_stream.items():
            residual_clean[k] = {kk: vv for kk, vv in v.items() if kk != "histogram"}

        return _s(dict(
            model_specs=SPEC,
            conversation=messages,
            response=response_text,
            metadata=self.metadata,
            rms_norm=self.rms_norm_data,
            attention=attn_clean,
            gelu_activations=gelu_clean,
            mlp=self.mlp_data,
            residual_stream=residual_clean,
            token_probabilities=self.token_probs,
            per_step_head_entropy=dict(self.per_step_head_entropy),
        ))

    def get_summary(self):
        return dict(
            rms_norm=dict(self.rms_norm_data),
            attention=dict(self.attention_data),
            gelu=dict(self.gelu_data),
            mlp=dict(self.mlp_data),
            residual_stream=dict(self.residual_stream),
            token_probs=list(self.token_probs),
            per_step_head_entropy=dict(self.per_step_head_entropy),
            step_activations=dict(self.step_activations),
            metadata=dict(self.metadata),
        )

    def reset(self):
        self.rms_norm_data.clear()
        self.attention_data.clear()
        self.gelu_data.clear()
        self.mlp_data.clear()
        self.residual_stream.clear()
        self.attn_output_data.clear()
        self.token_probs.clear()
        self.per_step_head_entropy.clear()
        self.step_activations.clear()
        self.metadata.clear()
        self._step = 0


# ======================================================================
# CHART STYLING
# ======================================================================
def styled(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_1, size=14), x=0.02) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,20,0.78)",
        font=dict(color=TEXT_1, size=12, family="Segoe UI, Arial"),
        legend=dict(bgcolor="rgba(17,17,20,0.82)", bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT_1, size=11)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        hoverlabel=dict(bgcolor=BG_SURFACE, font_size=12, font_color=TEXT_1, bordercolor=ACCENT),
        hovermode="closest",
        height=height,
        margin=dict(l=50, r=20, t=50, b=40),
        colorway=VIVID,
    )
    return fig

CFG = {"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ======================================================================
# ARCHITECTURE DIAGRAM  (Plotly-based block diagram)
# ======================================================================
def create_architecture_figure(obs=None):
    """Build a Plotly figure showing the transformer stack, layer by layer."""
    residual = obs.get("residual_stream", {}) if obs else {}
    rms = obs.get("rms_norm", {}) if obs else {}
    N = SPEC["num_layers"]  # 25

    fig = go.Figure()

    # -- Layout geometry --
    row_h   = 1.0       # height per layer row
    gap     = 0.2       # gap between rows
    pitch   = row_h + gap
    total_h = (N + 3) * pitch  # layers + input + final_norm + output
    W       = 10.0      # total width

    def add_block(x0, x1, y0, y1, text, fillcolor, bordercolor, textcolor=TEXT_1, fontsize=10):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=fillcolor, line=dict(color=bordercolor, width=1.5),
            layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=text,
            showarrow=False, font=dict(color=textcolor, size=fontsize, family="Segoe UI"),
            xanchor="center", yanchor="middle")

    def add_arrow(x, y0, y1):
        fig.add_annotation(x=x, y=y1, ax=x, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor=TEXT_3)

    # -- INPUT block --
    iy = 0
    add_block(1, 9, iy, iy + row_h, "INPUT  --  Tokens", BG_SURFACE, BORDER, TEXT_1, 12)
    add_arrow(5, iy + row_h, iy + row_h + gap)

    # -- Embedding block --
    ey = pitch
    add_block(0.5, 9.5, ey, ey + row_h,
        f"Token Embedding  |  dim {SPEC['hidden_dim']}",
        BG_SURFACE, PURPLE, PURPLE, 10)
    add_arrow(5, ey + row_h, ey + row_h + gap)

    # -- 25 Layer blocks --
    max_norm = max((v.get("norm", 0) for v in residual.values()), default=1.0) or 1.0

    for i in range(N):
        ly = (i + 2) * pitch
        r = residual.get(i, {})
        rm = rms.get(i, {})
        norm_val   = r.get("norm", 0)
        pre_scale  = rm.get("pre_attn", {}).get("scale_factor", 0)
        post_scale = rm.get("post_attn", {}).get("scale_factor", 0)
        pre_s  = f"{pre_scale:.2f}" if pre_scale else "--"
        post_s = f"{post_scale:.2f}" if post_scale else "--"
        
        is_global = ((i + 1) % 6 == 0)
        attn_color = PURPLE if is_global else ACCENT
        attn_label = f"Global Attn<br>{SPEC['num_heads']}H" if is_global else f"Local Attn (SW=1k)<br>{SPEC['num_heads']}H"

        # -- Layer label --
        fig.add_annotation(x=0.2, y=ly + row_h / 2,
            text=f"<b>L{i}</b>", showarrow=False,
            font=dict(color=TEXT_2, size=11, family="Segoe UI"),
            xanchor="center", yanchor="middle")

        # Sub-blocks within each layer
        # Pre-Attn RMS Norm
        add_block(0.5, 2.1, ly, ly + row_h, f"RMS<br>{pre_s}", BG_SURFACE, CYAN, CYAN, 9)
        # MHA
        add_block(2.3, 5.0, ly, ly + row_h, attn_label, BG_SURFACE, attn_color, attn_color, 10)
        # Residual +
        add_block(5.2, 5.8, ly, ly + row_h, "+", BG_SURFACE, AMBER, AMBER, 12)
        # Post-Attn RMS Norm
        add_block(6.0, 7.3, ly, ly + row_h, f"RMS<br>{post_s}", BG_SURFACE, CYAN, CYAN, 9)
        # MLP GELU
        add_block(7.5, 9.5, ly, ly + row_h, "MLP  GELU", BG_SURFACE, GREEN, GREEN, 10)

        # Norm bar on right side (if data available)
        if residual and norm_val > 0:
            bar_w = (norm_val / max_norm) * 0.4
            fig.add_shape(type="rect",
                x0=9.6, x1=9.6 + bar_w, y0=ly + 0.15, y1=ly + row_h - 0.15,
                fillcolor=ACCENT, line=dict(width=0), opacity=0.7)
            fig.add_annotation(x=9.6 + bar_w + 0.05, y=ly + row_h / 2,
                text=f"{norm_val:.1f}", showarrow=False,
                font=dict(color=TEXT_3, size=8), xanchor="left")

        # Arrow to next block
        if i < N - 1:
            add_arrow(5, ly + row_h, ly + row_h + gap)

    # -- Arrow to final norm --
    last_ly = (N + 1) * pitch
    add_arrow(5, (N - 1 + 2) * pitch + row_h, last_ly)

    # -- Final RMS Norm --
    add_block(0.5, 9.5, last_ly, last_ly + row_h,
        f"Final RMS Norm  |  dim {SPEC['hidden_dim']}",
        BG_SURFACE, CYAN, CYAN, 10)
    add_arrow(5, last_ly + row_h, last_ly + row_h + gap)

    # -- OUTPUT block --
    oy = (N + 2) * pitch
    add_block(1, 9, oy, oy + row_h, "OUTPUT  --  Logits", BG_SURFACE, BORDER, TEXT_1, 12)

    # -- Figure layout --
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,11,14,0.98)",
        font=dict(color=TEXT_1, family="Segoe UI"),
        xaxis=dict(visible=False, range=[-0.2, 10.5], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.5, oy + row_h + 0.5], fixedrange=True),
        height=max(700, (N + 4) * 55),
        margin=dict(l=0, r=0, t=10, b=10),
        showlegend=False,
    )
    return fig


# ======================================================================
# VISUALISATION FUNCTIONS
# ======================================================================

def chart_residual_stream(residual):
    if not residual: return None
    layers = sorted(residual.keys())
    norms = [residual[l]["norm"] for l in layers]
    stds  = [residual[l]["std"]  for l in layers]
    names = [f"Layer {l}" for l in layers]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residual Stream L2 Norm", "Residual Stream Std"))
    fig.add_trace(go.Bar(x=names, y=norms, marker=dict(color=norms, colorscale="Viridis", showscale=True,
        colorbar=dict(title="Norm", x=1.02, len=0.9)), hovertemplate="<b>%{x}</b><br>L2: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=names, y=stds, mode="lines+markers", marker=dict(color=AMBER, size=7),
        line=dict(color=AMBER, width=2), hovertemplate="<b>%{x}</b><br>Std: %{y:.6f}<extra></extra>"), row=1, col=2)
    styled(fig, height=350)
    fig.update_layout(showlegend=False)
    return fig


def chart_rms_norm(rms_data):
    if not rms_data: return None
    layers = sorted(rms_data.keys())
    pre_in  = [rms_data[l].get("pre_attn", {}).get("input_rms", 0)  for l in layers]
    pre_out = [rms_data[l].get("pre_attn", {}).get("output_rms", 0) for l in layers]
    post_in = [rms_data[l].get("post_attn", {}).get("input_rms", 0) for l in layers]
    post_out= [rms_data[l].get("post_attn", {}).get("output_rms", 0)for l in layers]
    names = [f"L{l}" for l in layers]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pre-Attention RMS Norm", "Post-Attention RMS Norm"))
    for vals, nm, clr, row, col in [
        (pre_in,  "Input",  ACCENT, 1, 1), (pre_out, "Output", GREEN,  1, 1),
        (post_in, "Input",  ACCENT, 1, 2), (post_out,"Output", GREEN,  1, 2),
    ]:
        fig.add_trace(go.Scatter(x=names, y=vals, name=nm, mode="lines+markers",
            marker=dict(color=clr, size=6), line=dict(color=clr, width=2),
            hovertemplate=f"<b>%{{x}}</b><br>{nm} RMS: %{{y:.4f}}<extra></extra>",
            showlegend=(col == 1)), row=row, col=col)
    styled(fig, height=320)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5))
    return fig


def chart_gelu_analysis(gelu_data):
    if not gelu_data: return None
    layers = sorted(gelu_data.keys())
    names = [f"L{l}" for l in layers]
    dead = [gelu_data[l]["dead_fraction"] * 100 for l in layers]
    pre_std = [gelu_data[l]["pre_std"] for l in layers]
    post_std= [gelu_data[l]["post_std"] for l in layers]

    fig = make_subplots(rows=1, cols=3,
        subplot_titles=("Dead Neuron Fraction (%)", "Pre/Post GELU Std", "GELU Gate Histogram (last layer)"))
    fig.add_trace(go.Bar(x=names, y=dead, marker=dict(color=dead, colorscale="Reds", showscale=False),
        hovertemplate="<b>%{x}</b><br>Dead: %{y:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=names, y=pre_std, name="Pre-GELU", mode="lines+markers",
        marker=dict(color=ACCENT, size=6), line=dict(color=ACCENT)), row=1, col=2)
    fig.add_trace(go.Scatter(x=names, y=post_std, name="Post-GELU", mode="lines+markers",
        marker=dict(color=GREEN, size=6), line=dict(color=GREEN)), row=1, col=2)

    last = layers[-1]
    ph = gelu_data[last].get("pre_hist")
    gh = gelu_data[last].get("post_hist")
    if ph is not None:
        fig.add_trace(go.Histogram(x=ph, name="Pre-GELU", marker_color=ACCENT, opacity=0.6, nbinsx=60), row=1, col=3)
    if gh is not None:
        fig.add_trace(go.Histogram(x=gh, name="Post-GELU", marker_color=GREEN, opacity=0.6, nbinsx=60), row=1, col=3)
    fig.update_layout(barmode="overlay")
    styled(fig, height=340)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5))
    return fig


def chart_attn_vs_mlp(attn, mlp):
    if not attn or not mlp: return None
    layers = sorted(set(attn.keys()) & set(mlp.keys()))
    if not layers: return None
    names = [f"L{l}" for l in layers]
    a_norm = [attn[l]["norm"] for l in layers]
    m_norm = [mlp[l]["norm"] for l in layers]
    a_std  = [attn[l]["std"]  for l in layers]
    m_std  = [mlp[l]["std"]  for l in layers]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Output Norm: Attn vs MLP", "Output Std: Attn vs MLP"))
    for vals, nm, clr, col in [(a_norm, "Attn", ACCENT, 1), (m_norm, "MLP", GREEN, 1),
                                 (a_std, "Attn", ACCENT, 2), (m_std, "MLP", GREEN, 2)]:
        fig.add_trace(go.Scatter(x=names, y=vals, name=nm, mode="lines+markers",
            marker=dict(color=clr, size=6), line=dict(color=clr, width=2),
            showlegend=(col == 1)), row=1, col=col)
    styled(fig, height=320)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5))
    return fig


def observed_head_count(attn_data, limit=DISPLAY_HEADS):
    counts = []
    for values in attn_data.values():
        if "head_entropy" in values:
            counts.append(len(values["head_entropy"]))
        elif "last_token_attn" in values and getattr(values["last_token_attn"], "shape", None) is not None:
            counts.append(int(values["last_token_attn"].shape[0]))
    if not counts:
        return limit
    return max(1, min(limit, max(counts)))


def chart_head_entropy_heatmap(attn_data):
    """Layer x active-head heatmap of attention entropy."""
    layers = sorted([l for l in attn_data if "head_entropy" in attn_data[l]])
    if not layers: return None
    head_count = observed_head_count(attn_data)
    z = [attn_data[l]["head_entropy"][:head_count] for l in layers]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Head {h}" for h in range(head_count)],
        y=[f"Layer {l}" for l in layers], colorscale="Viridis",
        hovertemplate="<b>%{y}, %{x}</b><br>Entropy: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Entropy"),
    ))
    styled(fig, title=f"Attention Head Entropy  ({len(layers)} layers x {head_count} active heads)", height=400)
    return fig


def chart_head_focus(attn_data, layer_idx):
    """Show where each active head focuses for the last generated token."""
    info = attn_data.get(layer_idx, {})
    lta = info.get("last_token_attn")   # [16, K]
    if lta is None: return None
    max_heads = min(DISPLAY_HEADS, lta.shape[0])
    lta = lta[:max_heads]
    num_heads, kv_len = lta.shape
    cols = 2 if num_heads > 1 else 1
    rows = math.ceil(num_heads / cols)
    fig = make_subplots(rows=rows, cols=cols,
        subplot_titles=[f"Head {h}" for h in range(num_heads)],
        vertical_spacing=0.12, horizontal_spacing=0.08)
    for h in range(num_heads):
        r, c = h // cols + 1, h % cols + 1
        positions = list(range(kv_len))
        fig.add_trace(go.Bar(
            x=positions, y=lta[h], marker_color=VIVID[h % len(VIVID)],
            hovertemplate=f"Head {h}<br>Pos %{{x}}: %{{y:.4f}}<extra></extra>",
            showlegend=False,
        ), row=r, col=c)
    styled(fig, title=f"Layer {layer_idx}  --  {num_heads} active heads  --  Last Token Focus", height=max(360, 220 * rows))
    fig.update_layout(margin=dict(t=70))
    return fig


def chart_token_probs(tp_list):
    if not tp_list: return None
    latest = tp_list[-1]
    tokens = latest["tokens"]
    probs = latest["probabilities"]
    colors = [ACCENT if i == 0 else PURPLE if i < 3 else "#6366f1" for i in range(len(tokens))]
    fig = go.Figure(go.Bar(
        x=probs, y=[t.strip() or " " for t in tokens], orientation="h",
        marker=dict(color=colors, line=dict(color=TEXT_1, width=0.5)),
        hovertemplate="<b>%{y}</b><br>P = %{x:.4f}<extra></extra>",
    ))
    styled(fig, title="Top-K Token Probabilities (Last Step)", height=350)
    fig.update_layout(xaxis_title="Probability")
    fig.update_yaxes(autorange="reversed")
    return fig


def chart_entropy_timeline(tp_list):
    if not tp_list or len(tp_list) < 2: return None
    ent = [t["entropy"] for t in tp_list]
    conf = [t["top1_confidence"] for t in tp_list]
    steps = list(range(1, len(ent) + 1))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=steps, y=ent, name="Entropy", mode="lines+markers",
        marker=dict(color=AMBER, size=4), line=dict(color=AMBER, width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=steps, y=conf, name="Top-1 Confidence", mode="lines+markers",
        marker=dict(color=GREEN, size=4), line=dict(color=GREEN, width=2)), secondary_y=True)
    styled(fig, title="Generation Uncertainty Timeline", height=300)
    fig.update_layout(legend=dict(orientation="h", y=1.1, xanchor="center", x=0.5))
    fig.update_yaxes(title_text="Entropy (bits)", secondary_y=False)
    fig.update_yaxes(title_text="Confidence", secondary_y=True)
    return fig


def chart_layer_radar(obs, layer_idx):
    attn = obs.get("attention", {}).get(layer_idx, {})
    mlp  = obs.get("mlp", {}).get(layer_idx, {})
    if not attn or not mlp: return None
    cats = ["Mean", "Std", "Norm"]
    av = [abs(attn.get("mean", 0)), attn.get("std", 0), attn.get("norm", 0)]
    mv = [abs(mlp.get("mean", 0)), mlp.get("std", 0), mlp.get("norm", 0)]
    mx = [max(a, b, 1e-8) for a, b in zip(av, mv)]
    an = [a / m for a, m in zip(av, mx)]
    mn = [m / mx_ for m, mx_ in zip(mv, mx)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=an + [an[0]], theta=cats + [cats[0]], fill="toself",
        name="Attention", fillcolor="rgba(59,130,246,0.15)", line=dict(color=ACCENT)))
    fig.add_trace(go.Scatterpolar(r=mn + [mn[0]], theta=cats + [cats[0]], fill="toself",
        name="MLP", fillcolor="rgba(34,197,94,0.15)", line=dict(color=GREEN)))
    fig.update_layout(
        title=dict(text=f"Layer {layer_idx}  --  Attn vs MLP Profile", font=dict(color=TEXT_1, size=14)),
        height=320, paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_1),
        polar=dict(bgcolor="rgba(10,22,40,0.6)",
                   radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.06)"),
                   angularaxis=dict(gridcolor="rgba(255,255,255,0.06)")),
        legend=dict(orientation="h", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def chart_sparsity_heatmap(mlp_data, gelu_data):
    if not mlp_data: return None
    layers = sorted(mlp_data.keys())
    names = [f"L{l}" for l in layers]
    metrics_labels = ["MLP Sparsity", "MLP Std", "MLP Max"]
    z = [
        [mlp_data[l]["sparsity"] for l in layers],
        [mlp_data[l]["std"] for l in layers],
        [mlp_data[l]["max"] for l in layers],
    ]
    if gelu_data:
        metrics_labels.append("GELU Dead Frac")
        z.append([gelu_data.get(l, {}).get("dead_fraction", 0) for l in layers])
    fig = go.Figure(go.Heatmap(z=z, x=names, y=metrics_labels, colorscale="Magma",
        hovertemplate="<b>%{y}</b> @ %{x}<br>%{z:.6f}<extra></extra>"))
    styled(fig, title="Layer Metrics Heatmap", height=280)
    return fig


def chart_activation_dist(residual):
    if not residual: return None
    layers = sorted(residual.keys())
    fig = go.Figure()
    cols = px.colors.sample_colorscale("Viridis", [i / max(len(layers) - 1, 1) for i in range(len(layers))])
    for i, l in enumerate(layers):
        h = residual[l].get("histogram")
        if h is not None and len(h) > 0:
            fig.add_trace(go.Violin(y=h, name=f"L{l}", box_visible=True, meanline_visible=True,
                fillcolor=cols[i], line_color=cols[i], opacity=0.7))
    styled(fig, title="Residual Stream Activation Distributions", height=380)
    fig.update_layout(showlegend=False, yaxis_title="Activation Value")
    return fig


def chart_architecture_dynamics(obs):
    if not obs:
        return None

    layers = list(range(SPEC["num_layers"]))
    residual = obs.get("residual_stream", {})
    attention = obs.get("attention", {})
    mlp = obs.get("mlp", {})
    rms = obs.get("rms_norm", {})
    gelu = obs.get("gelu", {})

    families = {
        "Residual": [residual.get(l, {}).get("norm", 0.0) for l in layers],
        "Attention": [attention.get(l, {}).get("norm", 0.0) for l in layers],
        "MLP": [mlp.get(l, {}).get("norm", 0.0) for l in layers],
        "RMS": [((rms.get(l, {}).get("pre_attn", {}).get("output_rms", 0.0) + rms.get(l, {}).get("post_attn", {}).get("output_rms", 0.0)) / 2.0) for l in layers],
        "GELU Alive %": [max(0.0, 100.0 * (1.0 - gelu.get(l, {}).get("dead_fraction", 0.0))) for l in layers],
    }
    colors = {"Residual": ACCENT, "Attention": CYAN, "MLP": GREEN, "RMS": PURPLE, "GELU Alive %": AMBER}
    heatmap_rows = []
    for name, values in families.items():
        vmax = max(values) if any(values) else 1.0
        heatmap_rows.append([(v / vmax) if vmax else 0.0 for v in values])

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.9, 1.1],
        subplot_titles=("Subsystem intensity by layer", "Layer-wise subsystem traces"),
        horizontal_spacing=0.1,
        specs=[[{"type": "heatmap"}, {"type": "xy"}]],
    )
    fig.add_trace(go.Heatmap(
        z=heatmap_rows,
        x=[f"L{l}" for l in layers],
        y=list(families.keys()),
        colorscale=[[0.0, "#111114"], [0.35, "#3b0764"], [0.7, "#7c3aed"], [1.0, "#22d3ee"]],
        colorbar=dict(title="Normalized"),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Intensity: %{z:.3f}<extra></extra>",
    ), row=1, col=1)
    for name, values in families.items():
        fig.add_trace(go.Scatter(
            x=[f"L{l}" for l in layers],
            y=values,
            name=name,
            mode="lines+markers",
            line=dict(color=colors[name], width=2.6),
            marker=dict(size=7, color=colors[name]),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ), row=1, col=2)
    styled(fig, title="Layer Dynamics Map", height=520)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.75))
    fig.update_yaxes(title_text="Raw magnitude", row=1, col=2)
    return fig


def estimate_kv_cache_mb(token_count, num_layers, num_heads, head_dim, bytes_per_elem=4):
    total_bytes = token_count * num_layers * num_heads * head_dim * 2 * bytes_per_elem
    return total_bytes / (1024 ** 2)


def chart_memory_profile(obs):
    if not obs:
        return None
    meta = obs.get("metadata", {})
    sequence_lengths = meta.get("sequence_lengths", [])
    kv_cache_mb = meta.get("kv_cache_mb", [])
    step_latencies = meta.get("step_latencies_s", [])
    instant_tps = meta.get("instantaneous_tps", [])
    if not sequence_lengths or not kv_cache_mb:
        return None

    steps = list(range(1, len(sequence_lengths) + 1))
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Memory growth during decode", "Latency / TPS bottleneck trace"),
        horizontal_spacing=0.12,
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
    )
    fig.add_trace(go.Scatter(
        x=steps,
        y=kv_cache_mb,
        mode="lines+markers",
        name="KV cache MB",
        line=dict(color=PURPLE, width=3),
        marker=dict(size=7, color=PURPLE),
        hovertemplate="Step %{x}<br>KV cache: %{y:.3f} MB<extra></extra>",
    ), row=1, col=1)

    if step_latencies:
        fig.add_trace(go.Bar(
            x=steps,
            y=step_latencies,
            name="Latency / token",
            marker_color="rgba(236,72,153,0.55)",
            hovertemplate="Step %{x}<br>Latency: %{y:.4f}s<extra></extra>",
        ), row=1, col=2, secondary_y=False)
    if instant_tps:
        fig.add_trace(go.Scatter(
            x=steps[:len(instant_tps)],
            y=instant_tps,
            mode="lines+markers",
            name="Instant TPS",
            line=dict(color=CYAN, width=2.6),
            marker=dict(size=6, color=CYAN),
            hovertemplate="Step %{x}<br>TPS: %{y:.2f}<extra></extra>",
        ), row=1, col=2, secondary_y=True)

    if instant_tps:
        baseline = float(np.median(instant_tps)) if instant_tps else 0.0
        bottleneck_idx = next((i for i, tps in enumerate(instant_tps) if baseline and tps < baseline * 0.8), None)
        if bottleneck_idx is not None:
            step_no = bottleneck_idx + 1
            fig.add_vline(x=step_no, line_width=1.5, line_dash="dash", line_color=RED, row=1, col=2)
            fig.add_annotation(
                x=step_no,
                y=max(instant_tps),
                text=f"Bottleneck starts ~ step {step_no}",
                showarrow=True,
                arrowcolor=RED,
                font=dict(color=TEXT_1, size=11),
                bgcolor="rgba(17,17,20,0.92)",
                row=1,
                col=2,
            )

    styled(fig, title="Memory / Latency Telemetry", height=420)
    fig.update_xaxes(title_text="Decode step", row=1, col=1)
    fig.update_xaxes(title_text="Decode step", row=1, col=2)
    fig.update_yaxes(title_text="KV cache (MB)", row=1, col=1)
    fig.update_yaxes(title_text="Latency (s)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Tokens / second", row=1, col=2, secondary_y=True)
    return fig


def chart_telemetry_melt(obs):
    if not obs:
        return None
    meta = obs.get("metadata", {})
    metrics = {
        "Metrics": meta.get("tokens_per_second", 0.0),
        "Events": len(meta.get("stream_chunks", [])),
        "Logs": len(obs.get("token_probs", [])),
        "Traces": len(obs.get("attention", {})),
    }
    fig = go.Figure(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker=dict(color=[ACCENT, PINK, AMBER, CYAN]),
        hovertemplate="<b>%{x}</b><br>Signal strength: %{y:.2f}<extra></extra>",
    ))
    styled(fig, title="MELT Signal Footprint", height=300)
    return fig


def analyze_input_interpretation(model, tokenizer, inputs, token_limit=INPUT_ANALYSIS_TOKEN_LIMIT, head_limit=DISPLAY_HEADS):
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    total_tokens = int(input_ids.shape[1])
    start_idx = max(0, total_tokens - token_limit)
    input_slice = input_ids[:, start_idx:]
    attn_slice = attention_mask[:, start_idx:] if attention_mask is not None else None

    forward_inputs = {"input_ids": input_slice}
    if attn_slice is not None:
        forward_inputs["attention_mask"] = attn_slice

    with torch.no_grad():
        outputs = model(
            **forward_inputs,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    sliced_ids = input_slice[0].detach().cpu().tolist()
    token_labels = []
    for token_id in sliced_ids:
        token_txt = tokenizer.decode([token_id])
        token_txt = token_txt.replace("\n", "⏎").replace("\t", "⇥")
        token_labels.append(token_txt if token_txt.strip() else "␠")

    embed_state = outputs.hidden_states[0][0].detach().cpu().float()
    token_norms = embed_state.norm(dim=-1).tolist()

    max_layers = min(SPEC["num_layers"], len(outputs.attentions), len(model.model.layers))
    layer_attn = {}
    qkv_norms = {}
    for layer_idx in range(max_layers):
        attn = outputs.attentions[layer_idx][0].detach().cpu().float()
        active_heads = min(head_limit, attn.shape[0])
        layer_attn[layer_idx] = attn[:active_heads].numpy()

        hidden_in = outputs.hidden_states[layer_idx][0]
        layer_mod = model.model.layers[layer_idx].self_attn
        q = layer_mod.q_proj(hidden_in)
        k = layer_mod.k_proj(hidden_in)
        v = layer_mod.v_proj(hidden_in)
        qh = q.view(q.shape[0], -1, SPEC["head_dim"])[:, :active_heads, :]
        kh = k.view(k.shape[0], -1, SPEC["head_dim"])[:, :active_heads, :]
        vh = v.view(v.shape[0], -1, SPEC["head_dim"])[:, :active_heads, :]
        qkv_norms[layer_idx] = {
            "q": qh.norm(dim=-1).detach().cpu().numpy(),
            "k": kh.norm(dim=-1).detach().cpu().numpy(),
            "v": vh.norm(dim=-1).detach().cpu().numpy(),
        }

    return {
        "total_tokens": total_tokens,
        "start_idx": start_idx,
        "analyzed_tokens": len(sliced_ids),
        "truncated": start_idx > 0,
        "token_ids": sliced_ids,
        "token_labels": token_labels,
        "token_norms": token_norms,
        "attention": layer_attn,
        "qkv_norms": qkv_norms,
        "head_count": min(head_limit, observed_head_count({k: {"head_entropy": list(range(v.shape[0]))} for k, v in layer_attn.items()})) if layer_attn else head_limit,
    }


def chart_input_token_attention(input_analysis, selected_layer, selected_head, selected_token_idx):
    if not input_analysis:
        return None
    attn = input_analysis.get("attention", {}).get(selected_layer)
    if attn is None:
        return None
    head_idx = min(selected_head, attn.shape[0] - 1)
    token_idx = min(selected_token_idx, attn.shape[1] - 1)
    weights = attn[head_idx, token_idx]
    labels = input_analysis.get("token_labels", [])
    fig = go.Figure(go.Heatmap(
        z=[weights],
        x=[f"{i}: {lbl}"[:18] for i, lbl in enumerate(labels)],
        y=[f"L{selected_layer} H{head_idx} q@{token_idx}"],
        colorscale="Magma",
        hovertemplate="Key token %{x}<br>Attention: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Attention"),
    ))
    styled(fig, title="Selected Token Attention over Input Sequence", height=240)
    fig.update_xaxes(tickangle=-45)
    return fig


def chart_qkv_token_bars(input_analysis, selected_layer, selected_token_idx):
    if not input_analysis:
        return None
    qkv = input_analysis.get("qkv_norms", {}).get(selected_layer)
    if not qkv:
        return None
    token_idx = min(selected_token_idx, qkv["q"].shape[0] - 1)
    head_count = min(DISPLAY_HEADS, qkv["q"].shape[1])
    heads = [f"Head {h}" for h in range(head_count)]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Q", x=heads, y=qkv["q"][token_idx][:head_count], marker_color=ACCENT))
    fig.add_trace(go.Bar(name="K", x=heads, y=qkv["k"][token_idx][:head_count], marker_color=CYAN))
    fig.add_trace(go.Bar(name="V", x=heads, y=qkv["v"][token_idx][:head_count], marker_color=GREEN))
    styled(fig, title=f"QKV Norms for Token {selected_token_idx} at Layer {selected_layer}", height=320)
    fig.update_layout(barmode="group")
    return fig


def chart_input_embedding_norms(input_analysis):
    if not input_analysis:
        return None
    labels = input_analysis.get("token_labels", [])
    norms = input_analysis.get("token_norms", [])
    if not norms:
        return None
    fig = go.Figure(go.Bar(
        x=[f"{i}" for i in range(len(norms))],
        y=norms,
        marker=dict(color=norms, colorscale="Plasma", showscale=True, colorbar=dict(title="Norm")),
        customdata=labels,
        hovertemplate="Token %{x}: %{customdata}<br>Embedding norm: %{y:.4f}<extra></extra>",
    ))
    styled(fig, title="Input Representation Norms After Positional Encoding", height=280)
    fig.update_xaxes(title_text="Input token index")
    fig.update_yaxes(title_text="L2 norm")
    return fig


def chart_selected_token_crosslayer_attention(input_analysis, selected_head, selected_token_idx):
    if not input_analysis:
        return None
    labels = input_analysis.get("token_labels", [])
    attn = input_analysis.get("attention", {})
    if not attn:
        return None
    layers = sorted(attn.keys())
    z = []
    for layer_idx in layers:
        layer_attn = attn[layer_idx]
        head_idx = min(selected_head, layer_attn.shape[0] - 1)
        token_idx = min(selected_token_idx, layer_attn.shape[1] - 1)
        z.append(layer_attn[head_idx, token_idx])
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{i}:{lbl}"[:16] for i, lbl in enumerate(labels)],
        y=[f"Layer {l}" for l in layers],
        colorscale="Viridis",
        hovertemplate="%{y}<br>Key %{x}<br>Attention %{z:.4f}<extra></extra>",
        colorbar=dict(title="Attention"),
    ))
    styled(fig, title="Cross-Layer Attention for Selected Query Token", height=420)
    fig.update_xaxes(tickangle=-45)
    return fig


def render_input_token_sequence(input_analysis, selected_layer, selected_head, selected_token_idx):
    if not input_analysis:
        return ""
    labels = input_analysis.get("token_labels", [])
    attn = input_analysis.get("attention", {}).get(selected_layer)
    if attn is None:
        return ""
    head_idx = min(selected_head, attn.shape[0] - 1)
    token_idx = min(selected_token_idx, attn.shape[1] - 1)
    weights = attn[head_idx, token_idx]
    max_w = float(np.max(weights)) if len(weights) else 1.0
    spans = []
    for idx, (label, weight) in enumerate(zip(labels, weights)):
        alpha = 0.12 + (0.88 * (float(weight) / max(max_w, 1e-8)))
        border = f"2px solid {ACCENT}" if idx == token_idx else "1px solid rgba(255,255,255,0.06)"
        bg = f"rgba(168,85,247,{alpha:.3f})" if weight > 0 else "rgba(255,255,255,0.03)"
        safe_label = label.replace("<", "&lt;").replace(">", "&gt;")
        spans.append(f"<span title='idx={idx} | attn={float(weight):.4f}' style='display:inline-block;margin:3px;padding:4px 8px;border-radius:10px;background:{bg};border:{border};color:{TEXT_1};font-size:12px;'>{idx}:{safe_label}</span>")
    return "".join(spans)


def chart_activation_volume(obs, step_idx):
    if not obs:
        return None
    step_data = obs.get("step_activations", {}).get(step_idx)
    if not step_data:
        return None

    xs, ys, zs, vals, texts = [], [], [], [], []
    for layer_idx in sorted(step_data.keys()):
        vec = np.asarray(step_data[layer_idx], dtype=float)
        if vec.size != SPEC["hidden_dim"]:
            continue
        volume = vec.reshape(volume_shape(vec.size))
        
        # Optimized: Only plot the top 150 absolute highest activations per layer
        flat_indices = np.argsort(np.abs(volume.flatten()))[-150:]
        
        for f_idx in flat_indices:
            v_idx = np.unravel_index(f_idx, volume.shape)
            x, y, z = v_idx[0], v_idx[1], v_idx[2]
            value = float(volume[x, y, z])
            xs.append(x)
            ys.append(y)
            zs.append(layer_idx + z * 0.08)
            vals.append(value)
            texts.append(f"Layer {layer_idx}<br>Voxel ({x},{y},{z})<br>Activation {value:.4f}")

    if not vals:
        return None

    fig = go.Figure(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(
            size=5,
            color=vals,
            colorscale="Turbo",
            opacity=0.72,
            colorbar=dict(title="Activation"),
        ),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"3D Activation Volume -- Step {step_idx}", font=dict(color=TEXT_1, size=15), x=0.03),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(17,17,20,0.75)",
            xaxis=dict(title="Grid X", color=TEXT_2, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Grid Y", color=TEXT_2, gridcolor="rgba(255,255,255,0.06)"),
            zaxis=dict(title="Layer / depth", color=TEXT_2, gridcolor="rgba(255,255,255,0.06)"),
        ),
        font=dict(color=TEXT_1),
        height=720,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ======================================================================
# LOGIT LENS  -- project each layer's hidden state through unembedding
# ======================================================================
LOGIT_LENS_TOP_K = 10

def compute_logit_lens(model, obs, tokenizer, logits_proc=None, top_k=LOGIT_LENS_TOP_K):
    """
    Mechanistic interpretability 'logit lens':
    For each generation step and each layer, take the last-token hidden state,
    apply the final RMS norm + lm_head, and softmax to see what the model
    would predict if decoding stopped at that layer.
    """
    step_data = obs.get("step_activations", {})
    if not step_data:
        return None
    final_norm = model.model.norm
    lm_head = model.lm_head
    logit_lens = {}
    with torch.no_grad():
        for step_idx in sorted(step_data.keys()):
            layer_preds = {}
            for layer_idx in sorted(step_data[step_idx].keys()):
                vec = np.asarray(step_data[step_idx][layer_idx], dtype=np.float32)
                if vec.size != SPEC["hidden_dim"]:
                    continue
                h = torch.tensor(vec).unsqueeze(0).unsqueeze(0).to(model.device, dtype=model.dtype)
                normed = final_norm(h)
                logits = lm_head(normed)[0, 0]
                
                if logits_proc:
                    logits = logits_proc(None, logits.unsqueeze(0)).squeeze(0)
                    
                probs = torch.softmax(logits.float(), dim=-1).cpu()
                top_probs, top_ids = torch.topk(probs, top_k)
                tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
                layer_preds[layer_idx] = dict(tokens=tokens, probs=top_probs.tolist())
            logit_lens[step_idx] = layer_preds
    return logit_lens


# Probability ramp for the logit lens. Every stop is dark enough that white
# text keeps a readable contrast ratio, so the colour encodes magnitude without
# fighting the label sitting on top of it.
LOGIT_LENS_RAMP = [
    (0.000, (20, 22, 34)),
    (0.020, (30, 47, 92)),
    (0.080, (33, 88, 122)),
    (0.250, (67, 56, 172)),
    (0.550, (124, 58, 210)),
    (1.000, (176, 38, 190)),
]


def render_logit_lens_html(logit_lens_step, top_k=LOGIT_LENS_TOP_K):
    """Render the per-layer top token predictions as a styled HTML table (matches reference screenshot)."""
    if not logit_lens_step:
        return f"<p style='color:{TEXT_2};'>No logit lens data for this step.</p>"

    def prob_to_bg(p):
        p = min(max(p, 0.0), 1.0)
        for (lo, c_lo), (hi, c_hi) in zip(LOGIT_LENS_RAMP, LOGIT_LENS_RAMP[1:]):
            if p <= hi:
                t = (p - lo) / (hi - lo) if hi > lo else 0.0
                r, g, b = (round(a + (z - a) * t) for a, z in zip(c_lo, c_hi))
                return f"rgb({r},{g},{b})"
        r, g, b = LOGIT_LENS_RAMP[-1][1]
        return f"rgb({r},{g},{b})"

    rows = []
    layers = sorted(logit_lens_step.keys())
    for li in layers:
        preds = logit_lens_step[li]
        toks, probs = preds["tokens"], preds["probs"]
        # Layer index cell
        top1_bg = prob_to_bg(probs[0])
        c = (f"<td style='padding:8px 14px;text-align:center;font-weight:800;color:{TEXT_1};"
             f"background:{top1_bg};border:1px solid rgba(255,255,255,0.06);min-width:46px;'>{li + 1}</td>")
        for tok, prob in zip(toks[:top_k], probs[:top_k]):
            bg = prob_to_bg(prob)
            safe = tok.replace("<", "&lt;").replace(">", "&gt;")
            if not safe.strip():
                safe = "&nbsp;"
            tcol = TEXT_1 if prob > 0.015 else "rgba(255,255,255,0.78)"
            c += (f"<td style='padding:6px 8px;text-align:center;background:{bg};"
                  f"border:1px solid rgba(255,255,255,0.035);min-width:88px;'>"
                  f"<span style='font-weight:700;font-size:13px;color:{tcol};display:block;'>{safe}</span>"
                  f"<span style='font-size:10px;color:rgba(255,255,255,0.72);'>{prob:.4f}</span></td>")
        rows.append(f"<tr>{c}</tr>")

    header_cells = (f"<th style='padding:8px 14px;background:{BG_DEEP};color:{TEXT_2};"
                    f"border:1px solid rgba(255,255,255,0.06);font-size:12px;font-weight:600;'>Layer</th>")
    header_cells += "".join(
        f"<th style='padding:6px 8px;background:{BG_DEEP};color:{TEXT_3};"
        f"border:1px solid rgba(255,255,255,0.06);font-size:11px;text-align:center;'>Top Next Token Predictions</th>"
        if i == top_k // 2 else
        f"<th style='padding:6px 8px;background:{BG_DEEP};color:transparent;"
        f"border:1px solid rgba(255,255,255,0.06);font-size:11px;'>&nbsp;</th>"
        for i in range(top_k)
    )
    return (f"<div style='overflow-x:auto;border-radius:12px;border:1px solid rgba(255,255,255,0.08);'>"
            f"<table style='width:100%;border-collapse:collapse;font-family:Segoe UI,Arial,sans-serif;'>"
            f"<tr>{header_cells}</tr>"
            f"{''.join(rows)}"
            f"</table></div>")


# ======================================================================
# REPRESENTATION GRID  --  near-square heatmap of the hidden state vector
# ======================================================================

def chart_representation_grid(obs, step_idx, layer_idx):
    """Render the hidden state vector as a near-square heatmap."""
    step_data = obs.get("step_activations", {})
    if not step_data or step_idx not in step_data or layer_idx not in step_data[step_idx]:
        return None
    vec = np.asarray(step_data[step_idx][layer_idx], dtype=float)
    if vec.size != SPEC["hidden_dim"]:
        return None
    rows, cols = grid_shape(vec.size)
    grid = vec.reshape(rows, cols)
    fig = go.Figure(go.Heatmap(
        z=grid, colorscale="Viridis",
        hovertemplate="Row %{y}, Col %{x}<br>Value: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Activation"),
    ))
    styled(fig, title=f"Layer {layer_idx}  |  {rows}×{cols} Representation Grid  (Step {step_idx})", height=500)
    fig.update_xaxes(title_text="Dimension (col)", dtick=8)
    fig.update_yaxes(title_text="Dimension (row)", dtick=4, autorange="reversed")
    return fig


def chart_representation_flow(obs, step_idx):
    """Side-by-side heatmaps for all layers at a given step."""
    step_data = obs.get("step_activations", {})
    if not step_data or step_idx not in step_data:
        return None
    layers = sorted(step_data[step_idx].keys())
    if not layers:
        return None
    n = len(layers)
    cols = min(5, n)
    rows_n = math.ceil(n / cols)
    fig = make_subplots(
        rows=rows_n, cols=cols,
        subplot_titles=[f"Layer {l}" for l in layers],
        horizontal_spacing=0.04, vertical_spacing=0.07,
    )
    for idx, layer_idx in enumerate(layers):
        r = idx // cols + 1
        c = idx % cols + 1
        vec = np.asarray(step_data[step_idx][layer_idx], dtype=float)
        if vec.size != SPEC["hidden_dim"]:
            continue
        grid = vec.reshape(grid_shape(vec.size))
        fig.add_trace(go.Heatmap(
            z=grid, colorscale="Viridis", showscale=(idx == 0),
            hovertemplate=f"Layer {layer_idx}<br>Row %{{y}}, Col %{{x}}<br>%{{z:.4f}}<extra></extra>",
        ), row=r, col=c)
    styled(fig, title=f"Layer-wise Representation Flow  (Step {step_idx})", height=220 * rows_n + 80)
    return fig


# ======================================================================
# 3D ARCHITECTURE  --  interactive 3D transformer block diagram
# ======================================================================

def create_3d_architecture_scene(obs):
    """
    Build a 3D interactive Plotly scene that shows the
    transformer architecture as stacked translucent blocks (Attention in
    blue, MLP in green, Norm in cyan) with data-driven opacity and
    connecting residual-stream flow lines.  Inspired by the classic
    '3D layered diagram' style from mechanistic interpretability research.
    """
    residual = obs.get("residual_stream", {}) if obs else {}
    rms_data = obs.get("rms_norm", {}) if obs else {}
    attn = obs.get("attention", {}) if obs else {}
    mlp_obs = obs.get("mlp", {}) if obs else {}
    N = SPEC["num_layers"]
    fig = go.Figure()

    # ---- helper: 3D box via Mesh3d + wireframe ----
    def box3d(x0, y0, z0, dx, dy, dz, color, opacity=0.35, name="", legend=False):
        vx = [x0, x0+dx, x0+dx, x0, x0, x0+dx, x0+dx, x0]
        vy = [y0, y0, y0+dy, y0+dy, y0, y0, y0+dy, y0+dy]
        vz = [z0, z0, z0, z0, z0+dz, z0+dz, z0+dz, z0+dz]
        ii = [0,0,4,4,0,0,1,1,0,0,3,3]
        jj = [1,2,5,6,1,4,2,5,3,4,2,7]
        kk = [2,3,6,7,4,5,5,6,4,7,7,6]
        fig.add_trace(go.Mesh3d(
            x=vx, y=vy, z=vz, i=ii, j=jj, k=kk,
            color=color, opacity=opacity, name=name,
            flatshading=True, showlegend=legend,
            hovertemplate=f"{name}<extra></extra>" if name else None,
        ))
        # wireframe
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        ex, ey, ez = [], [], []
        for a, b in edges:
            ex += [vx[a], vx[b], None]
            ey += [vy[a], vy[b], None]
            ez += [vz[a], vz[b], None]
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color="rgba(255,255,255,0.3)", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))

    def text3d(x, y, z, txt, size=9, color=TEXT_2):
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z], mode="text", text=[txt],
            textfont=dict(color=color, size=size), showlegend=False, hoverinfo="skip",
        ))

    def flow_line(pts, color="rgba(168,85,247,0.4)", width=3, dash=None):
        xs, ys, zs = zip(*pts)
        fig.add_trace(go.Scatter3d(
            x=list(xs), y=list(ys), z=list(zs), mode="lines",
            line=dict(color=color, width=width, dash=dash),
            showlegend=False, hoverinfo="skip",
        ))

    sp = 4.0  # layer spacing along z
    aw, ah, ad = 4.0, 2.0, 1.8     # attention box size
    mw, mh, md = 3.5, 2.0, 1.8     # MLP box size
    nw, nh, nd = 1.8, 2.0, 0.7     # norm box size
    embed_w = 9.0

    # ---- Embedding layer ----
    hdim = SPEC["hidden_dim"]
    box3d(-0.5, 0, -3, embed_w, 2.0, 0.7, "rgba(139,92,246,0.5)", 0.40, f"Token Embedding (seq × {hdim})", True)
    text3d(embed_w / 2 - 0.5, 1, -2.5, f"Input  →  Embed + Pos Enc  →  seq_len × {hdim}", 10, TEXT_1)

    # Matrix annotation on embedding
    mat_labels = [" -1.04", " -0.04", "  0.37", "  ...", " -0.41"]
    for ci, lbl in enumerate(mat_labels):
        text3d(-0.3 + ci * 0.85, -0.5, -2.3, lbl, 7, "rgba(255,255,255,0.55)")

    for i in range(N):
        zb = i * sp

        # --- Norms ---
        pre_s = rms_data.get(i, {}).get("pre_attn", {}).get("scale_factor", 0)
        post_s = rms_data.get(i, {}).get("post_attn", {}).get("scale_factor", 0)
        box3d(-0.5, 0, zb, nw, nh, nd, "rgba(34,211,238,0.35)", 0.30,
              f"L{i} Pre-Attn RMS Norm" + (f"  scale={pre_s:.2f}" if pre_s else ""))

        # --- Attention block  (blue) ---
        attn_n = attn.get(i, {}).get("norm", 0)
        a_opa = min(0.22 + attn_n * 0.012, 0.65) if attn_n else 0.30
        box3d(2.0, 0, zb, aw, ah, ad, f"rgba(59,130,246,{a_opa:.2f})", a_opa,
              f"L{i} Multi-Head Attention ({SPEC['num_heads']}h)", i == 0)
        text3d(4.0, 1, zb + ad / 2, "Attention", 9, "rgba(200,220,255,0.8)")

        # Residual add marker
        text3d(6.3, 1, zb + 0.9, "+", 12, AMBER)

        # Post-attention norm
        box3d(6.8, 0, zb, nw, nh, nd, "rgba(34,211,238,0.35)", 0.30,
              f"L{i} Post-Attn RMS Norm" + (f"  scale={post_s:.2f}" if post_s else ""))

        # --- MLP block  (green) ---
        mlp_n = mlp_obs.get(i, {}).get("norm", 0)
        m_opa = min(0.22 + mlp_n * 0.008, 0.65) if mlp_n else 0.30
        box3d(9.0, 0, zb, mw, mh, md, f"rgba(34,197,94,{m_opa:.2f})", m_opa,
              f"L{i} MLP + GELU", i == 0)
        text3d(10.75, 1, zb + md / 2, "MLP", 9, "rgba(180,255,180,0.8)")

        # Layer label
        text3d(-2.2, 1, zb + 0.9, f"Layer {i}", 10, TEXT_1)

        # --- Internal flow arrows ---
        flow_line([(-0.5 + nw, 1, zb + nd / 2), (2.0, 1, zb + ad / 2)],
                  "rgba(255,255,255,0.2)", 2)
        flow_line([(2.0 + aw, 1, zb + ad / 2), (6.8, 1, zb + nd / 2)],
                  "rgba(255,255,255,0.2)", 2)
        flow_line([(6.8 + nw, 1, zb + nd / 2), (9.0, 1, zb + md / 2)],
                  "rgba(255,255,255,0.2)", 2)

        # --- Residual to next layer ---
        if i < N - 1:
            r_norm = residual.get(i, {}).get("norm", 0)
            r_alpha = min(0.15 + r_norm * 0.015, 0.6) if r_norm else 0.2
            flow_line([(5.5, 1, zb + ad), (5.5, 1, (i + 1) * sp)],
                      f"rgba(168,85,247,{r_alpha:.2f})", 4, "dot")

    # ---- Final Norm + LM Head ----
    z_out = N * sp
    box3d(2.5, 0, z_out, 4.5, 2.0, 0.7, "rgba(34,211,238,0.4)", 0.35, "Final RMS Norm")
    box3d(2.5, 0, z_out + 1.5, 4.5, 2.0, 0.7, "rgba(236,72,153,0.45)", 0.40, "LM Head → Logits", True)
    text3d(4.75, 1, z_out + 0.3, "Final RMS Norm", 9, CYAN)
    text3d(4.75, 1, z_out + 1.8, "Unembedding → Vocabulary", 9, PINK)
    flow_line([(5.5, 1, (N - 1) * sp + ad), (5.5, 1, z_out)],
              "rgba(168,85,247,0.3)", 4, "dot")

    # ---- Floating matrix annotations from actual data ----
    step_data = obs.get("step_activations", {}) if obs else {}
    if step_data:
        last_step = max(step_data.keys())
        for li in [0, N // 2, N - 1]:
            if li in step_data.get(last_step, {}):
                vec = np.asarray(step_data[last_step][li], dtype=float)
                if vec.size == SPEC["hidden_dim"]:
                    sample_vals = vec[:6]
                    for si, sv in enumerate(sample_vals):
                        text3d(-3.5 + si * 0.9, -0.6, li * sp + 0.3,
                               f"{sv:.2f}", 7, "rgba(255,255,255,0.45)")
                    text3d(-3.5, -1.2, li * sp + 0.1,
                           f"last row (1×{SPEC['hidden_dim']})", 7, "rgba(255,255,255,0.35)")

    fig.update_layout(
        scene=dict(
            bgcolor="rgba(9,9,11,1)",
            xaxis=dict(visible=False, range=[-5, 15]),
            yaxis=dict(visible=False, range=[-2, 4]),
            zaxis=dict(visible=False, range=[-4, N * sp + 4]),
            camera=dict(
                eye=dict(x=2.0, y=-1.6, z=0.35),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
            aspectratio=dict(x=1.2, y=0.5, z=2.8),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_1),
        height=950,
        margin=dict(l=0, r=0, t=55, b=0),
        title=dict(text=f"3D Transformer Architecture  —  {SPEC['num_layers']} layers × {SPEC['num_heads']} heads × {SPEC['hidden_dim']} dim",
                   font=dict(color=TEXT_1, size=15), x=0.03),
        legend=dict(bgcolor="rgba(17,17,20,0.9)", bordercolor=BORDER,
                    font=dict(color=TEXT_1, size=11), orientation="h",
                    yanchor="bottom", y=1.0, xanchor="center", x=0.5),
    )
    return fig


def summarise_observability(obs):
    if not obs:
        return {}

    meta = obs.get("metadata", {})
    attention = obs.get("attention", {})
    mlp = obs.get("mlp", {})
    gelu = obs.get("gelu", {})
    residual = obs.get("residual_stream", {})
    rms = obs.get("rms_norm", {})
    token_probs = obs.get("token_probs", [])

    entropies = []
    for values in attention.values():
        entropies.extend(values.get("head_entropy", []))

    residual_norms = {layer: vals.get("norm", 0.0) for layer, vals in residual.items()}
    peak_layer = max(residual_norms, key=residual_norms.get) if residual_norms else None

    rms_scales = []
    for values in rms.values():
        for position in ("pre_attn", "post_attn"):
            scale = values.get(position, {}).get("scale_factor")
            if scale is not None:
                rms_scales.append(scale)

    return dict(
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
        tok_s=meta.get("tokens_per_second", 0),
        time_s=meta.get("generation_time_s", 0),
        ttft_s=meta.get("ttft_s", meta.get("time_to_first_token", 0)),
        avg_entropy=float(np.mean(entropies)) if entropies else 0.0,
        avg_dead=float(np.mean([v.get("dead_fraction", 0.0) for v in gelu.values()])) if gelu else 0.0,
        avg_mlp_norm=float(np.mean([v.get("norm", 0.0) for v in mlp.values()])) if mlp else 0.0,
        latest_conf=token_probs[-1].get("top1_confidence", 0.0) if token_probs else 0.0,
        peak_layer=peak_layer,
        peak_norm=residual_norms.get(peak_layer, 0.0) if peak_layer is not None else 0.0,
        rms_stability=float(np.mean(rms_scales)) if rms_scales else 0.0,
        decode_steps=len(token_probs),
        head_count=meta.get("estimated_head_count", observed_head_count(attention) if attention else DISPLAY_HEADS),
        max_kv_cache_mb=max(meta.get("kv_cache_mb", [0.0])) if meta.get("kv_cache_mb") else 0.0,
    )


def build_observation_insights(obs):
    s = summarise_observability(obs)
    if not s:
        return []
    peak_label = f"Layer {s['peak_layer']}" if s.get("peak_layer") is not None else "Unavailable"
    return [
        ("Peak residual energy", peak_label, f"Residual stream norm peaks at {s['peak_norm']:.2f}, marking the strongest representational amplification in the current answer."),
        ("Attention spread", f"{s['avg_entropy']:.2f}", "Average head entropy across captured heads. Lower values imply sharper focus; higher values imply broader contextual mixing."),
        ("GELU inactivity", f"{s['avg_dead'] * 100:.2f}%", "Approximate fraction of near-silent post-GELU activations averaged across hooked layers."),
        ("Decoder confidence", f"{s['latest_conf'] * 100:.1f}%", "Top-1 probability on the latest decode step, useful for spotting certainty spikes or hesitation."),
    ]


def chart_observability_sunburst(obs):
    if not obs:
        return None

    attention   = obs.get("attention", {})
    mlp = obs.get("mlp", {})
    residual = obs.get("residual_stream", {})
    rms = obs.get("rms_norm", {})
    gelu = obs.get("gelu", {})
    token_probs = obs.get("token_probs", [])

    groups = {
        "Residual Stream": ({layer: vals.get("norm", 0.0) for layer, vals in residual.items()}, ACCENT),
        "Attention": ({layer: vals.get("norm", 0.0) for layer, vals in attention.items()}, CYAN),
        "MLP": ({layer: vals.get("norm", 0.0) for layer, vals in mlp.items()}, GREEN),
        "RMS Norm": ({layer: ((vals.get("pre_attn", {}).get("output_rms", 0.0) + vals.get("post_attn", {}).get("output_rms", 0.0)) / 2.0) for layer, vals in rms.items()}, PURPLE),
        "GELU": ({layer: max(1e-4, (1.0 - vals.get("dead_fraction", 0.0)) * 100.0) for layer, vals in gelu.items()}, AMBER),
        "Decode": ({idx + 1: max(1e-4, step.get("top1_confidence", 0.0) * 100.0) for idx, step in enumerate(token_probs)}, PINK),
    }

    ids = ["response"]
    labels = ["Response"]
    parents = [""]
    values = [0.0]
    colors = [TEXT_1]
    total = 0.0

    for family, (items, color) in groups.items():
        filtered = {k: float(v) for k, v in items.items() if v and v > 0}
        if not filtered:
            continue
        family_total = sum(filtered.values())
        total += family_total
        family_id = f"family::{family}"
        ids.append(family_id)
        labels.append(family)
        parents.append("response")
        values.append(family_total)
        colors.append(color)
        for key, value in sorted(filtered.items(), key=lambda item: item[0]):
            ids.append(f"{family_id}::{key}")
            labels.append(f"Layer {key}" if family != "Decode" else f"Step {key}")
            parents.append(family_id)
            values.append(value)
            colors.append(color)

    if total <= 0:
        return None

    values[0] = total
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        maxdepth=3,
        insidetextorientation="radial",
        marker=dict(colors=colors, line=dict(color="rgba(255,255,255,0.08)", width=1)),
        hovertemplate="<b>%{label}</b><br>Magnitude: %{value:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Response Observability Sunburst", x=0.03, font=dict(color=TEXT_1, size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color=TEXT_1, family="Segoe UI"),
        height=640,
    )
    return fig


def chart_layer_attribute_sunburst(obs, layer_idx):
    if not obs:
        return None

    attn = obs.get("attention", {}).get(layer_idx, {})
    mlp = obs.get("mlp", {}).get(layer_idx, {})
    residual = obs.get("residual_stream", {}).get(layer_idx, {})
    gelu = obs.get("gelu", {}).get(layer_idx, {})
    rms = obs.get("rms_norm", {}).get(layer_idx, {})
    if not any([attn, mlp, residual, gelu, rms]):
        return None

    entropy_mean = float(np.mean(attn.get("head_entropy", [0.0]))) if attn.get("head_entropy") else 0.0
    families = {
        "Attention": {
            "Norm": max(1e-4, attn.get("norm", 0.0)),
            "Std": max(1e-4, attn.get("std", 0.0) * 100.0),
            "Entropy": max(1e-4, entropy_mean * 10.0),
        },
        "Residual": {
            "Norm": max(1e-4, residual.get("norm", 0.0)),
            "Std": max(1e-4, residual.get("std", 0.0) * 100.0),
            "Kurtosis": max(1e-4, abs(residual.get("kurtosis", 0.0)) * 10.0),
        },
        "RMS Norm": {
            "Pre Scale": max(1e-4, rms.get("pre_attn", {}).get("scale_factor", 0.0) * 50.0),
            "Post Scale": max(1e-4, rms.get("post_attn", {}).get("scale_factor", 0.0) * 50.0),
            "Post RMS": max(1e-4, rms.get("post_attn", {}).get("output_rms", 0.0) * 10.0),
        },
        "MLP": {
            "Norm": max(1e-4, mlp.get("norm", 0.0)),
            "Std": max(1e-4, mlp.get("std", 0.0) * 100.0),
            "Sparsity": max(1e-4, mlp.get("sparsity", 0.0) * 100.0),
        },
        "GELU": {
            "Dead Frac": max(1e-4, gelu.get("dead_fraction", 0.0) * 100.0),
            "Post Std": max(1e-4, gelu.get("post_std", 0.0) * 100.0),
            "Post Mean": max(1e-4, abs(gelu.get("post_mean", 0.0)) * 100.0),
        },
    }
    family_colors = {"Attention": CYAN, "Residual": ACCENT, "RMS Norm": PURPLE, "MLP": GREEN, "GELU": AMBER}

    ids = [f"layer::{layer_idx}"]
    labels = [f"Layer {layer_idx}"]
    parents = [""]
    values = [0.0]
    colors = [TEXT_1]
    total = 0.0

    for family, metrics in families.items():
        family_total = sum(metrics.values())
        if family_total <= 0:
            continue
        total += family_total
        family_id = f"layer::{layer_idx}::{family}"
        ids.append(family_id)
        labels.append(family)
        parents.append(f"layer::{layer_idx}")
        values.append(family_total)
        colors.append(family_colors[family])
        for metric, value in metrics.items():
            ids.append(f"{family_id}::{metric}")
            labels.append(metric)
            parents.append(family_id)
            values.append(value)
            colors.append(family_colors[family])

    if total <= 0:
        return None

    values[0] = total
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        maxdepth=3,
        insidetextorientation="radial",
        marker=dict(colors=colors, line=dict(color="rgba(255,255,255,0.08)", width=1)),
        hovertemplate="<b>%{label}</b><br>Scaled magnitude: %{value:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Layer {layer_idx} Attribute Sunburst", x=0.03, font=dict(color=TEXT_1, size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color=TEXT_1, family="Segoe UI"),
        height=560,
    )
    return fig


def chart_token_candidate_sunburst(tp_list, top_n=3):
    if not tp_list:
        return None

    ids = ["decode"]
    labels = ["Decode"]
    parents = [""]
    values = [0.0]
    colors = [TEXT_1]
    total = 0.0

    for idx, step in enumerate(tp_list, start=1):
        probs = step.get("probabilities", [])[:top_n]
        toks = step.get("tokens", [])[:top_n]
        if not probs:
            continue
        step_total = float(sum(probs))
        total += step_total
        step_id = f"decode::{idx}"
        ids.append(step_id)
        labels.append(f"Step {idx}")
        parents.append("decode")
        values.append(step_total)
        colors.append(PURPLE if idx % 2 else ACCENT)
        for rank, (tok, prob) in enumerate(zip(toks, probs), start=1):
            clean_tok = (tok.replace("\n", "\\n").replace(" ", "␠") or "␠")[:18]
            ids.append(f"{step_id}::{rank}")
            labels.append(clean_tok)
            parents.append(step_id)
            values.append(float(max(prob, 1e-4)))
            colors.append(CYAN if rank == 1 else PINK if rank == 2 else AMBER)

    if total <= 0:
        return None

    values[0] = total
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="remainder",
        maxdepth=3,
        marker=dict(colors=colors, line=dict(color="rgba(255,255,255,0.08)", width=1)),
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Probability mass: %{value:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Decode Candidate Sunburst", x=0.03, font=dict(color=TEXT_1, size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color=TEXT_1, family="Segoe UI"),
        height=560,
    )
    fig.update_traces(insidetextfont=dict(size=11, color=TEXT_1), outsidetextfont=dict(size=11, color=TEXT_1), sort=False)
    return fig


# ======================================================================
# GENERATION
# ======================================================================
class TelemetryStreamer:
    def __init__(self, tokenizer):
        from transformers import TextIteratorStreamer
        self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.token_times = []
        self.token_ids = []

        original_put = self.streamer.put

        def wrapped_put(value):
            is_prompt = getattr(self.streamer, "next_tokens_are_prompt", False) and getattr(self.streamer, "skip_prompt", False)
            if not is_prompt:
                values = value.tolist() if hasattr(value, "tolist") else value
                if isinstance(values, list) and values and isinstance(values[0], list):
                    values = values[0]
                if not isinstance(values, list):
                    values = [values]
                now = time.time()
                for token_id in values:
                    self.token_times.append(now)
                    self.token_ids.append(int(token_id))
            original_put(value)

        self.streamer.put = wrapped_put

    def __iter__(self):
        return iter(self.streamer)


def generate_with_observability(model, tokenizer, messages, observer, logits_proc, max_tokens=512, temperature=0.0, stream_handler=None):
    observer.reset()
    original_oa = model.config.output_attentions
    model.config.output_attentions = True
    observer.register_hooks()

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    observer.metadata = {"input_tokens": input_len, "timestamp": time.time()}

    try:
        observer.metadata["input_analysis"] = analyze_input_interpretation(model, tokenizer, inputs)
    except Exception as exc:
        observer.metadata["input_analysis_error"] = str(exc)

    gen_kw = dict(
        **inputs, max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")],
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=logits_proc,
        output_scores=True, return_dict_in_generate=True,
    )
    if temperature > 0:
        gen_kw["temperature"] = temperature

    telemetry_streamer = None
    stream_chunks = []
    streamed_text = ""
    token_times = []

    try:
        telemetry_streamer = TelemetryStreamer(tokenizer)
        gen_kw["streamer"] = telemetry_streamer.streamer

        result_box = {}
        error_box = {}

        def _run_generation():
            try:
                with torch.no_grad():
                    result_box["out"] = model.generate(**gen_kw)
            except Exception as exc:
                error_box["error"] = exc

        t0 = time.time()
        worker = Thread(target=_run_generation, daemon=True)
        worker.start()
        for chunk in telemetry_streamer:
            if chunk:
                streamed_text += chunk
                stream_chunks.append({"t": round(time.time() - t0, 4), "text": chunk})
                if stream_handler is not None:
                    stream_handler(streamed_text, chunk)
        worker.join()
        if "error" in error_box:
            raise error_box["error"]
        out = result_box["out"]
        elapsed = time.time() - t0
        token_times = telemetry_streamer.token_times[: len(out.sequences[0][input_len:])]
    except Exception:
        gen_kw.pop("streamer", None)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**gen_kw)
        elapsed = time.time() - t0
    finally:
        model.config.output_attentions = original_oa

    gen_ids = out.sequences[0][input_len:]
    decoded_text = tokenizer.decode(gen_ids, skip_special_tokens=True).replace("<end_of_turn>", "").strip()
    text = streamed_text.strip() or decoded_text

    if hasattr(out, "scores") and out.scores:
        for sl in out.scores:
            observer.token_probs.append(observer.capture_token_probs(sl.unsqueeze(0), tokenizer))

    n_out = len(gen_ids)
    ttft = (token_times[0] - t0) if token_times else elapsed / max(n_out, 1)
    step_latencies = []
    instantaneous_tps = []
    for i, token_time in enumerate(token_times):
        delta = (token_time - t0) if i == 0 else (token_time - token_times[i - 1])
        step_latencies.append(round(delta, 4))
        instantaneous_tps.append(round(1.0 / max(delta, 1e-4), 2))
    observed_heads = observed_head_count(observer.attention_data) if observer.attention_data else DISPLAY_HEADS
    sequence_lengths = list(range(input_len + 1, input_len + n_out + 1))
    kv_cache_mb = [round(estimate_kv_cache_mb(seq_len, SPEC["num_layers"], observed_heads, SPEC["head_dim"]), 4) for seq_len in sequence_lengths]
    activation_mb = round((input_len * SPEC["hidden_dim"] * SPEC["num_layers"] * 4) / (1024 ** 2), 4)
    observer.metadata.update(
        output_tokens=n_out, total_tokens=input_len + n_out,
        generation_time_s=round(elapsed, 3),
        tokens_per_second=round(n_out / max(elapsed, 0.001), 2),
        time_to_first_token=round(ttft, 4),
        ttft_s=round(ttft, 4),
        step_latencies_s=step_latencies,
        instantaneous_tps=instantaneous_tps,
        generated_token_timestamps_s=[round(t - t0, 4) for t in token_times],
        sequence_lengths=sequence_lengths,
        kv_cache_mb=kv_cache_mb,
        activation_prefill_mb=activation_mb,
        estimated_head_count=observed_heads,
        stream_chunks=stream_chunks,
    )
    return text, observer.get_summary()


# ======================================================================
# EXPERIMENT SCENARIOS
# ======================================================================
EXPERIMENTS = {
    "Factual Recall": dict(
        desc="Test ability to recall factual information accurately",
        prompts=[
            "What is the capital of Japan?",
            "Who painted the Mona Lisa?",
            "What is the chemical formula for water?",
        ]),
    "Multi-Step Reasoning": dict(
        desc="Test logical and multi-step reasoning",
        prompts=[
            "If all birds have feathers and a penguin is a bird, does a penguin have feathers? Explain step by step.",
            "A jar has 3 red and 5 blue marbles. You pick one without looking. What is the probability of red?",
        ]),
    "Creative Writing": dict(
        desc="Test creative language generation",
        prompts=[
            "Write a haiku about the ocean.",
            "Describe a thunderstorm in exactly three sentences.",
        ]),
    "Code Generation": dict(
        desc="Test code synthesis ability",
        prompts=[
            "Write a Python function to check if a string is a palindrome.",
            "Write a Python function that returns the nth Fibonacci number.",
        ]),
    "Instruction Following": dict(
        desc="Test precision in following instructions",
        prompts=[
            "List exactly 5 fruits that are red.",
            "Summarize what machine learning is in exactly 2 sentences.",
        ]),
    "Summarisation": dict(
        desc="Test text compression ability",
        prompts=[
            "Summarize the concept of photosynthesis in one sentence.",
            "Explain quantum computing in a single paragraph for a 10-year-old.",
        ]),
}


# ======================================================================
# SESSION STATE
# ======================================================================
def init_state():
    defs = dict(
        messages=[], conversations={}, active_conversation_id=None,
        obs_data=None, obs_history=[], last_messages_for_artifact=None, last_response_for_artifact=None,
        system_prompt="You are a helpful and concise AI assistant. Answer directly.",
        max_tokens=512, temperature=0.0, show_observability=True,
        uploaded_files={}, total_tokens_used=0, total_generations=0,
        generation_times=[], model_loaded=False, selected_layer=0,
        experiment_results=[], capture_attention=True, selected_response_index=None,
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

def new_conversation():
    cid = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    st.session_state.conversations[cid] = dict(
        title="New Chat", messages=[], created_at=datetime.datetime.now().isoformat(), obs_history=[])
    st.session_state.active_conversation_id = cid
    st.session_state.messages = []
    st.session_state.obs_data = None
    st.session_state.obs_history = []
    st.session_state.selected_response_index = None
    return cid

def save_active():
    cid = st.session_state.active_conversation_id
    if cid and cid in st.session_state.conversations:
        st.session_state.conversations[cid]["messages"] = list(st.session_state.messages)
        st.session_state.conversations[cid]["obs_history"] = list(st.session_state.obs_history)

def set_title(t):
    cid = st.session_state.active_conversation_id
    if cid and cid in st.session_state.conversations:
        st.session_state.conversations[cid]["title"] = (t[:40] + "...") if len(t) > 40 else t


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown(
        f"<h3 style='color:{ACCENT};font-weight:800;letter-spacing:-0.5px;margin-bottom:0;'>◉ Neural Observatory</h3>"
        f"<p style='color:{TEXT_3};font-size:11px;margin-top:2px;'>watch a transformer think</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(f"<p style='font-size:13px;color:{TEXT_2};font-weight:600;letter-spacing:0.5px;'>MODEL</p>", unsafe_allow_html=True)
    _choice = st.selectbox(
        "Model",
        MODEL_CHOICES + ["custom…"],
        index=MODEL_CHOICES.index(DEFAULT_MODEL_ID) if DEFAULT_MODEL_ID in MODEL_CHOICES else 0,
        label_visibility="collapsed",
    )
    if _choice == "custom…":
        _choice = st.text_input("HF repo id or local path", value=DEFAULT_MODEL_ID, label_visibility="collapsed")
    st.session_state.model_id = _choice
    st.caption("Weights are cached after the first load. Bigger models are slower on CPU, not impossible.")
    st.markdown("---")

    if st.button("+  New Chat", use_container_width=True, type="primary"):
        new_conversation(); st.rerun()
    st.markdown("---")

    st.markdown(f"<p style='font-size:13px;color:{TEXT_2};font-weight:600;letter-spacing:0.5px;'>CONVERSATIONS</p>", unsafe_allow_html=True)
    if st.session_state.conversations:
        for cid, conv in sorted(st.session_state.conversations.items(), key=lambda x: x[1]["created_at"], reverse=True):
            c1, c2 = st.columns([6, 1])
            with c1:
                pfx = "> " if cid == st.session_state.active_conversation_id else "  "
                if st.button(f"{pfx}{conv['title']}", key=f"c_{cid}", use_container_width=True):
                    st.session_state.active_conversation_id = cid
                    st.session_state.messages = list(conv["messages"])
                    st.session_state.obs_history = list(conv.get("obs_history", []))
                    st.session_state.obs_data = st.session_state.obs_history[-1] if st.session_state.obs_history else None
                    st.rerun()
            with c2:
                if st.button("x", key=f"d_{cid}"):
                    del st.session_state.conversations[cid]
                    if st.session_state.active_conversation_id == cid:
                        st.session_state.active_conversation_id = None; st.session_state.messages = []
                    st.rerun()
    else:
        st.caption("No conversations yet.")
    st.markdown("---")

    with st.expander("Settings", expanded=False):
        st.session_state.system_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt, height=80)
        st.session_state.max_tokens = st.slider("Max Tokens", 64, 1024, st.session_state.max_tokens, step=64)
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.5, st.session_state.temperature, step=0.05)
        st.session_state.capture_attention = st.toggle("Capture Attention Heads", value=st.session_state.capture_attention,
            help="Capture per-head attention patterns. Slight perf cost.")

    with st.expander("Context Files", expanded=False):
        uploaded = st.file_uploader("Upload context", type=["txt","md","py","json","csv"], accept_multiple_files=True)
        if uploaded:
            for uf in uploaded:
                st.session_state.uploaded_files[uf.name] = uf.getvalue().decode("utf-8", errors="replace")
        if st.session_state.uploaded_files:
            to_rm = []
            for fn in st.session_state.uploaded_files:
                fc1, fc2 = st.columns([5, 1])
                with fc1: st.caption(fn)
                with fc2:
                    if st.button("x", key=f"rm_{fn}"): to_rm.append(fn)
            for fn in to_rm: del st.session_state.uploaded_files[fn]; st.rerun()

    with st.expander("Session Stats", expanded=False):
        st.metric("Generations", st.session_state.total_generations)
        st.metric("Tokens Used", st.session_state.total_tokens_used)
        if st.session_state.generation_times:
            st.metric("Avg Time", f"{np.mean(st.session_state.generation_times):.2f}s")

    with st.expander("Export", expanded=False):
        if st.button("Export Chat (JSON)", use_container_width=True):
            if st.session_state.messages:
                st.download_button("Download Chat",
                    data=json.dumps({"messages": st.session_state.messages}, indent=2, default=str),
                    file_name=f"chat_{datetime.datetime.now():%Y%m%d_%H%M%S}.json", mime="application/json", use_container_width=True)
        if st.button("Clear All Data", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

    st.markdown("---")
    st.markdown(f'<div style="text-align:center;color:{TEXT_3};font-size:11px;">Neural Observatory<br>{st.session_state.model_id}</div>', unsafe_allow_html=True)


# ======================================================================
# MAIN
# ======================================================================
if st.session_state.active_conversation_id is None:
    new_conversation()

try:
    # load_model is cached, so its body only runs on the first load. SPEC is a
    # module-level dict that Streamlit resets to its placeholder values on every
    # rerun, so it has to be repopulated here rather than inside load_model.
    model, tokenizer, logits_proc, model_spec = load_model(st.session_state.model_id)
    SPEC.update(model_spec)
except Exception as exc:
    st.error(f"Could not load `{st.session_state.model_id}`\n\n```\n{exc}\n```")
    st.info(
        "**Gated repository?** Every `google/gemma-*` model requires accepting the licence on "
        "its Hugging Face model page, then `huggingface-cli login` with a read token.\n\n"
        "**Want to skip that?** Pick `HuggingFaceTB/SmolLM2-135M-Instruct` or "
        "`Qwen/Qwen2.5-0.5B-Instruct` in the sidebar — both are ungated and work identically here."
    )
    st.stop()
observer = ArchitectureObserver(model)
st.session_state.model_loaded = True

response_count = len(st.session_state.obs_history)
if response_count:
    if st.session_state.selected_response_index is None or st.session_state.selected_response_index >= response_count:
        st.session_state.selected_response_index = response_count - 1
    obs = st.session_state.obs_history[st.session_state.selected_response_index]
else:
    st.session_state.selected_response_index = None
    obs = st.session_state.obs_data
summary = summarise_observability(obs)

left_panel, right_panel = st.columns([0.9, 1.35], gap="large")

with left_panel:
    cid = st.session_state.active_conversation_id
    title = st.session_state.conversations.get(cid, {}).get("title", "New Chat")
    st.markdown(f"""
    <div class="hero-panel">
        <div class="deck-kicker">Conversation Console</div>
        <h1 class="deck-title">Talk to the model. Watch the stack react.</h1>
        <p class="deck-subtitle">The left side stays conversational. The right side turns every reply into a response-wise observability deck with layer anatomy, head behaviour, token confidence, and live architectural stress points.</p>
        <div class="soft-divider"></div>
        <div class="stat-row">
            <span class="stat-chip"><b>{title}</b></span>
            <span class="stat-chip"><b>{SPEC['name']}</b> local inference</span>
            <span class="stat-chip"><b>{SPEC['num_layers']}</b> layers</span>
            <span class="stat-chip"><b>{SPEC['num_heads']}</b> heads</span>
            <span class="stat-chip"><b>{model.device}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if obs:
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            st.metric("Input", summary.get("input_tokens", 0))
        with lc2:
            st.metric("Output", summary.get("output_tokens", 0))
        with lc3:
            st.metric("Tok/s", summary.get("tok_s", 0))
        with lc4:
            st.metric("Confidence", f"{summary.get('latest_conf', 0) * 100:.1f}%")

    assistant_cursor = 0
    for msg in st.session_state.messages:
        if msg["role"] == "assistant" and st.session_state.selected_response_index == assistant_cursor:
            st.markdown(f"<div class='stat-row'><span class='stat-chip'><b>Selected for observability deck</b></span></div>", unsafe_allow_html=True)
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])
            if "metadata" in msg and msg["metadata"]:
                m = msg["metadata"]
                st.markdown(f'<div class="stat-row"><span class="stat-chip"><b>{m.get("generation_time_s","?")}</b> sec</span><span class="stat-chip"><b>{m.get("output_tokens","?")}</b> tokens</span><span class="stat-chip"><b>{m.get("tokens_per_second","?")}</b> tok/s</span><span class="stat-chip"><b>{m.get("total_tokens","?")}</b> context</span></div>', unsafe_allow_html=True)
        if msg["role"] == "assistant":
            assistant_cursor += 1

    if not st.session_state.messages:
        st.markdown(f"""
        <div class="glass" style="padding:28px 24px;">
            <div class="deck-kicker">Neural Surgery</div>
            <p style="color:{TEXT_1};font-size:1.1rem;font-weight:700;margin-bottom:0.35rem;">Every answer becomes the instrument panel.</p>
            <p style="color:{TEXT_2};line-height:1.7;margin:0;">Start a prompt on the left. The right panel will expand into a full observability deck with response sunbursts, per-layer anatomy, attention focus, token trajectories, and experiment views that feel closer to an analyst workstation than a standard chat UI.</p>
        </div>
        """, unsafe_allow_html=True)

    if prompt := st.chat_input("Probe the model, inspect the reply, and open up the stack...", key="ci"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        if len(st.session_state.messages) == 1:
            set_title(prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        ctx = ""
        if st.session_state.uploaded_files:
            ctx = "\n\n".join(f"--- {fn} ---\n{c.strip()}\n--- END ---" for fn, c in st.session_state.uploaded_files.items())
        eph = [{"role": "user", "content": st.session_state.system_prompt}, {"role": "assistant", "content": "Understood."}]
        for m in st.session_state.messages[:-1]:
            eph.append({"role": m["role"], "content": m["content"]})
        ut = f"[REF]\n{ctx}\n[END]\n\nUser: {prompt}" if ctx else prompt
        eph.append({"role": "user", "content": ut})

        if not st.session_state.capture_attention:
            model.config.output_attentions = False

        with st.chat_message("assistant"):
            stream_box = st.empty()
            with st.spinner("Generating and instrumenting the response..."):
                try:
                    def _stream_handler(current_text, _chunk):
                        stream_box.markdown(current_text + "▌")

                    resp, obs_sum = generate_with_observability(
                        model, tokenizer, eph, observer, logits_proc,
                        max_tokens=st.session_state.max_tokens,
                        temperature=st.session_state.temperature,
                        stream_handler=_stream_handler,
                    )
                    stream_box.markdown(resp)
                    meta = obs_sum.get("metadata", {})
                    st.markdown(f'<div class="stat-row"><span class="stat-chip"><b>{meta.get("generation_time_s","?")}</b> sec</span><span class="stat-chip"><b>{meta.get("output_tokens","?")}</b> tokens</span><span class="stat-chip"><b>{meta.get("tokens_per_second","?")}</b> tok/s</span><span class="stat-chip"><b>{meta.get("ttft_s","?")}</b> TTFT</span></div>', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": resp, "metadata": meta})
                    st.session_state.obs_data = obs_sum
                    st.session_state.obs_history.append(obs_sum)
                    st.session_state.selected_response_index = len(st.session_state.obs_history) - 1
                    st.session_state.last_messages_for_artifact = eph
                    st.session_state.last_response_for_artifact = resp
                    st.session_state.total_generations += 1
                    st.session_state.total_tokens_used += meta.get("total_tokens", 0)
                    st.session_state.generation_times.append(meta.get("generation_time_s", 0))
                    save_active()
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.exception(e)
        st.rerun()

with right_panel:
    st.markdown(f"""
    <div class="hero-panel">
        <div class="deck-kicker">Observability Deck</div>
        <h1 class="deck-title">Response-wise model observability</h1>
        <p class="deck-subtitle">A black-mode deck inspired by modern research workspaces: layered cards, interactive charts, and tabs dedicated to how this specific response moved through the network.</p>
    </div>
    """, unsafe_allow_html=True)

    if obs:
        if st.session_state.obs_history:
            response_options = [f"Response {i + 1} -- {msg['content'][:70].replace(chr(10), ' ')}" for i, msg in enumerate([m for m in st.session_state.messages if m["role"] == "assistant"])]
            selected_label = st.selectbox("Response focus", response_options, index=st.session_state.selected_response_index or 0, key="deck_response_sel")
            st.session_state.selected_response_index = response_options.index(selected_label)
            obs = st.session_state.obs_history[st.session_state.selected_response_index]
            summary = summarise_observability(obs)

        top_metrics = st.columns(6)
        with top_metrics[0]:
            st.metric("Decode steps", summary.get("decode_steps", 0))
        with top_metrics[1]:
            st.metric("Avg entropy", f"{summary.get('avg_entropy', 0):.2f}")
        with top_metrics[2]:
            st.metric("Peak layer", f"L{summary.get('peak_layer')}" if summary.get("peak_layer") is not None else "--")
        with top_metrics[3]:
            st.metric("RMS stability", f"{summary.get('rms_stability', 0):.2f}")
        with top_metrics[4]:
            st.metric("GELU dead", f"{summary.get('avg_dead', 0) * 100:.2f}%")
        with top_metrics[5]:
            st.metric("TTFT", f"{summary.get('ttft_s', 0):.3f}s")

        deck_tabs = st.tabs(["Response Deck", "Input Lens", "Logit Lens", "Repr Grid", "3D Architecture", "Activation Volume", "Layer Anatomy", "Attention Atlas", "Token Trajectory", "Memory & Trace", "Experiment Bench"])

        with deck_tabs[0]:
            d1, d2 = st.columns([1.35, 0.95], gap="large")
            with d1:
                fig = chart_observability_sunburst(obs)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config=CFG, key="deck_sunburst")
                dyn_fig = chart_architecture_dynamics(obs)
                if dyn_fig:
                    st.plotly_chart(dyn_fig, use_container_width=True, config=CFG, key="arch_dynamics_v5")
                with st.expander("Structural architecture map", expanded=False):
                    arch_fig = create_architecture_figure(obs)
                    st.plotly_chart(arch_fig, use_container_width=True, config={"displayModeBar": False}, key="arch_diagram_v4")
            with d2:
                st.markdown(f"<div class='obs-header'><span class='dot'></span><h3>Response reading</h3></div>", unsafe_allow_html=True)
                for insight_title, insight_value, insight_caption in build_observation_insights(obs):
                    st.markdown(f"<div class='insight-card'><div class='ic-title'>{insight_title}</div><div class='ic-value'>{insight_value}</div><div class='ic-caption'>{insight_caption}</div></div>", unsafe_allow_html=True)

                st.markdown(f"<div class='obs-header' style='margin-top:8px;'><span class='dot'></span><h3>Architecture card</h3></div>", unsafe_allow_html=True)
                spec_pairs = [
                    ("Parameters", SPEC["parameters"]),
                    ("Layers", SPEC["num_layers"]),
                    ("Observed Heads", summary.get("head_count", DISPLAY_HEADS)),
                    ("Hidden", SPEC["hidden_dim"]),
                ]
                sp1, sp2 = st.columns(2)
                for idx, (label, value) in enumerate(spec_pairs):
                    with (sp1 if idx % 2 == 0 else sp2):
                        st.markdown(f"<div class='spec-card'><div class='lbl'>{label}</div><div class='val'>{value}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='glass' style='font-size:13px;color:{TEXT_2};line-height:1.75;'><b style='color:{TEXT_1};'>Decoder-only dense stack.</b><br>Each layer follows RMS Norm -> multi-head attention -> residual add -> RMS Norm -> gated GELU MLP -> residual add. The sunburst and cards above show which family dominated this answer and where activity concentrated.</div>", unsafe_allow_html=True)

        with deck_tabs[1]:
            input_analysis = obs.get("metadata", {}).get("input_analysis")
            if input_analysis:
                input_metrics = st.columns(4)
                with input_metrics[0]:
                    st.metric("Total input tokens", input_analysis.get("total_tokens", 0))
                with input_metrics[1]:
                    st.metric("Analysed window", input_analysis.get("analyzed_tokens", 0))
                with input_metrics[2]:
                    st.metric("Active heads", input_analysis.get("head_count", DISPLAY_HEADS))
                with input_metrics[3]:
                    st.metric("Window start", input_analysis.get("start_idx", 0))

                if input_analysis.get("truncated"):
                    st.info(f"Input attention analysis is showing the last {input_analysis.get('analyzed_tokens', 0)} tokens out of {input_analysis.get('total_tokens', 0)} total input tokens.")

                layer_options = sorted(input_analysis.get("attention", {}).keys())
                selected_input_layer = st.selectbox("Input analysis layer", layer_options, key="input_layer_sel")
                head_options = list(range(input_analysis.get("head_count", DISPLAY_HEADS)))
                selected_input_head = st.selectbox("Head", head_options, format_func=lambda x: f"Head {x}", key="input_head_sel")
                token_labels = input_analysis.get("token_labels", [])
                default_token = max(len(token_labels) - 1, 0)
                selected_token_idx = st.selectbox(
                    "Selected query token",
                    list(range(len(token_labels))),
                    index=default_token,
                    format_func=lambda i: f"{i}: {token_labels[i][:64]}",
                    key="input_token_sel",
                )

                il1, il2 = st.columns([1.15, 0.85], gap="large")
                with il1:
                    st.markdown(f"<div class='glass' style='padding:18px 18px 12px 18px;'><div class='deck-kicker'>Input token sequence</div><div style='line-height:2.1'>{render_input_token_sequence(input_analysis, selected_input_layer, selected_input_head, selected_token_idx)}</div></div>", unsafe_allow_html=True)
                    token_attn_fig = chart_input_token_attention(input_analysis, selected_input_layer, selected_input_head, selected_token_idx)
                    if token_attn_fig:
                        st.plotly_chart(token_attn_fig, use_container_width=True, config=CFG, key="input_token_heatmap")
                    cross_fig = chart_selected_token_crosslayer_attention(input_analysis, selected_input_head, selected_token_idx)
                    if cross_fig:
                        st.plotly_chart(cross_fig, use_container_width=True, config=CFG, key="input_crosslayer_heatmap")
                with il2:
                    qkv_fig = chart_qkv_token_bars(input_analysis, selected_input_layer, selected_token_idx)
                    if qkv_fig:
                        st.plotly_chart(qkv_fig, use_container_width=True, config=CFG, key="input_qkv_bars")
                    emb_fig = chart_input_embedding_norms(input_analysis)
                    if emb_fig:
                        st.plotly_chart(emb_fig, use_container_width=True, config=CFG, key="input_embed_norms")
                    st.markdown(f"<div class='glass' style='font-size:13px;color:{TEXT_2};line-height:1.75;'><b style='color:{TEXT_1};'>How to read this tab.</b><br>Pick a query token from the prompt window. The coloured token strip shows which input tokens received the most attention from that token for the chosen layer and head. The QKV bars show how strongly the selected token was projected into query, key, and value channels before attention mixing. This is the closest view of how the model formed its initial input-layer representation from the prompt and context block.</div>", unsafe_allow_html=True)
            else:
                st.info("No input interpretation data is available for this response. Generate a new reply to capture prompt attention and QKV telemetry.")

        with deck_tabs[2]:
            st.markdown(f"<div class='obs-header'><span class='dot'></span><h3>Logit Lens  —  Mechanistic Interpretability</h3></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{TEXT_2};font-size:13px;line-height:1.6;'>The <b style=\"color:{TEXT_1};\">logit lens</b> takes the last-token hidden state at each layer, applies the final RMS norm and unembedding head, and shows which token the model would predict if decoding stopped at that layer. Watch the predictions sharpen from generic tokens in early layers to the final answer in deeper layers.</p>", unsafe_allow_html=True)
            step_data_ll = obs.get("step_activations", {})
            if step_data_ll:
                step_keys_ll = sorted(step_data_ll.keys())
                tps = obs.get("token_probs", [])
                def format_ll_step(idx):
                    if idx < len(tps):
                        w = tps[idx].get("tokens", [""])[0]
                        return f"Step {idx}: {repr(w).strip()}"
                    return f"Step {idx}"
                ll_step = st.selectbox("Select Generated Token", options=step_keys_ll, index=len(step_keys_ll)-1, key="ll_step_sel", format_func=format_ll_step)
                with st.spinner("Projecting hidden states through unembedding..."):
                    ll_data = compute_logit_lens(model, obs, tokenizer, logits_proc=logits_proc, top_k=LOGIT_LENS_TOP_K)
                if ll_data and ll_step in ll_data:
                    st.markdown(render_logit_lens_html(ll_data[ll_step], top_k=LOGIT_LENS_TOP_K), unsafe_allow_html=True)
                    ll_metrics = st.columns(4)
                    with ll_metrics[0]:
                        st.metric("Layers", len(ll_data[ll_step]))
                    with ll_metrics[1]:
                        top1_tokens = [ll_data[ll_step][l]["tokens"][0] for l in sorted(ll_data[ll_step].keys())]
                        convergence_layer = 0
                        final_tok = top1_tokens[-1] if top1_tokens else ""
                        for ci, t in enumerate(top1_tokens):
                            if t == final_tok:
                                convergence_layer = ci + 1
                                break
                        st.metric("Converges at", f"Layer {convergence_layer}")
                    with ll_metrics[2]:
                        st.metric("Final prediction", final_tok[:20] if final_tok else "--")
                    with ll_metrics[3]:
                        if ll_data[ll_step]:
                            final_prob = ll_data[ll_step].get(max(ll_data[ll_step].keys()), {}).get("probs", [0])[0]
                        else:
                            final_prob = 0.0
                        st.metric("Final confidence", f"{final_prob:.4f}")
                    st.markdown(f"<div class='glass' style='font-size:13px;color:{TEXT_2};line-height:1.75;margin-top:12px;'><b style='color:{TEXT_1};'>Reading the logit lens.</b><br>Early layers (1-4) often predict a generic or high-frequency token. As the residual stream passes through attention and MLP blocks, the prediction refines. The layer where the top-1 prediction first matches the final output token is the <b style='color:{ACCENT};'>convergence layer</b>, the point where the model has effectively committed to its answer. Colour intensity maps to probability; bright cells indicate confident predictions.</div>", unsafe_allow_html=True)
                else:
                    st.info("Logit lens projection is not available for this step.")
            else:
                st.info("No step-wise activations captured. Generate a response to populate the logit lens.")

        with deck_tabs[3]:
            _rg_rows, _rg_cols = grid_shape(SPEC["hidden_dim"])
            st.markdown(f"<div class='obs-header'><span class='dot'></span><h3>Representation Grid  —  {_rg_rows}×{_rg_cols} Hidden State</h3></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{TEXT_2};font-size:13px;line-height:1.6;'>The {SPEC['hidden_dim']}-dimensional last-token hidden vector is reshaped into a <b style=\"color:{TEXT_1};\">{_rg_rows}×{_rg_cols} pixel grid</b> so you can visually watch the representation morph as it flows through each layer.</p>", unsafe_allow_html=True)
            step_data_rg = obs.get("step_activations", {})
            if step_data_rg:
                step_keys_rg = sorted(step_data_rg.keys())
                tps = obs.get("token_probs", [])
                def format_ll_step(idx):
                    if idx < len(tps):
                        w = tps[idx].get("tokens", [""])[0]
                        return f"[{idx}] {repr(w).strip()}"
                    return f"Step {idx}"
                rg_step = st.selectbox("Select Generated Token", options=step_keys_rg, index=len(step_keys_rg)-1, key="rg_step_sel", format_func=format_ll_step)
                rg_mode = st.radio("View mode", ["Single layer (detailed)", "All layers (flow)"], horizontal=True, key="rg_mode")
                if rg_mode == "Single layer (detailed)":
                    rg_layer = st.select_slider("Layer", options=list(range(SPEC["num_layers"])), value=0, key="rg_layer_sel")
                    grid_fig = chart_representation_grid(obs, rg_step, rg_layer)
                    if grid_fig:
                        st.plotly_chart(grid_fig, use_container_width=True, config=CFG, key="repr_grid_single")
                    # Stats for the selected grid
                    if rg_step in step_data_rg and rg_layer in step_data_rg[rg_step]:
                        vec = np.asarray(step_data_rg[rg_step][rg_layer], dtype=float)
                        rg_stats = st.columns(5)
                        with rg_stats[0]:
                            st.metric("Mean", f"{vec.mean():.4f}")
                        with rg_stats[1]:
                            st.metric("Std", f"{vec.std():.4f}")
                        with rg_stats[2]:
                            st.metric("Max", f"{vec.max():.4f}")
                        with rg_stats[3]:
                            st.metric("Min", f"{vec.min():.4f}")
                        with rg_stats[4]:
                            st.metric("L2 Norm", f"{np.linalg.norm(vec):.2f}")
                else:
                    flow_fig = chart_representation_flow(obs, rg_step)
                    if flow_fig:
                        st.plotly_chart(flow_fig, use_container_width=True, config=CFG, key="repr_grid_flow")
                st.markdown(f"<div class='glass' style='font-size:13px;color:{TEXT_2};line-height:1.75;margin-top:12px;'><b style='color:{TEXT_1};'>How the grid works.</b><br>The {SPEC['hidden_dim']} activations are laid out left-to-right, top-to-bottom in a {_rg_rows}×{_rg_cols} grid. In early layers the pattern tends to be smooth and low-contrast (the model hasn't specialised the representation yet). Deeper layers develop sharper features: bright spots and dark regions that correspond to the abstract features the model uses for its prediction. Comparing grids across layers reveals how much each transformer block reshapes the hidden state.</div>", unsafe_allow_html=True)
            else:
                st.info("No step-wise activations captured. Generate a response to populate the representation grid.")

        with deck_tabs[4]:
            st.markdown(f"<div class='obs-header'><span class='dot'></span><h3>3D Transformer Architecture</h3></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{TEXT_2};font-size:13px;line-height:1.6;'>Interactive 3D visualisation of the full stack. <b style=\"color:{TEXT_1};\">Blue</b> = Multi-Head Attention, <b style=\"color:{GREEN};\">Green</b> = MLP, <b style=\"color:{CYAN};\">Cyan</b> = RMS Norm. Block opacity is driven by the actual activation norms captured from the last response. Rotate and zoom to explore the {SPEC['num_layers']}-layer pipeline.</p>", unsafe_allow_html=True)
            arch3d_fig = create_3d_architecture_scene(obs)
            if arch3d_fig:
                st.plotly_chart(arch3d_fig, use_container_width=True, config=CFG, key="arch_3d_scene")
            arch3d_info = st.columns(4)
            with arch3d_info[0]:
                st.metric("Layers", SPEC["num_layers"])
            with arch3d_info[1]:
                st.metric("Heads", SPEC["num_heads"])
            with arch3d_info[2]:
                st.metric("Hidden dim", SPEC["hidden_dim"])
            with arch3d_info[3]:
                st.metric("Head dim", SPEC["head_dim"])
            st.markdown(f"<div class='glass' style='font-size:13px;color:{TEXT_2};line-height:1.75;margin-top:8px;'><b style='color:{TEXT_1};'>Architecture narrative.</b><br>Each of the {SPEC['num_layers']} layers contains a pre-attention RMS norm, a {SPEC['num_heads']}-head self-attention block (Local or Global based on the 5:1 interleaving), a post-attention RMS norm, and a gated GELU MLP. The residual stream (dotted purple lines) carries the hidden state between layers. Block opacity scales with the norm of the respective subsystem's output — brighter blocks did more work.</div>", unsafe_allow_html=True)

        with deck_tabs[5]:
            step_data = obs.get("step_activations", {})
            if step_data:
                step_keys = sorted(step_data.keys())
                default_step = step_keys[-1]
                tps = obs.get("token_probs", [])
                def format_ll_step(idx):
                    if idx < len(tps):
                        w = tps[idx].get("tokens", [""])[0]
                        return f"[{idx}] {repr(w).strip()}"
                    return f"Step {idx}"
                selected_step = st.selectbox("Select Generated Token", options=step_keys, index=len(step_keys)-1, format_func=format_ll_step, key="activation_step_sel")
                st.markdown(f"<div class='obs-header'><span class='dot'></span><h3>3D activation volume</h3></div>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:{TEXT_2};font-size:13px;'>Rotate the volume to inspect how last-token activations distribute through the {SPEC['num_layers']}-layer stack for the selected generation step.</p>", unsafe_allow_html=True)
                volume_fig = chart_activation_volume(obs, selected_step)
                if volume_fig:
                    st.plotly_chart(volume_fig, use_container_width=True, config=CFG, key="activation_volume_3d")
                step_meta_cols = st.columns(3)
                with step_meta_cols[0]:
                    st.metric("Stored steps", len(step_keys))
                with step_meta_cols[1]:
                    st.metric("Selected step", selected_step)
                with step_meta_cols[2]:
                    st.metric("Output tokens", obs.get("metadata", {}).get("output_tokens", 0))
            else:
                st.info("No step-wise activation volume is available for this response yet.")

        with deck_tabs[6]:
            selected_layer = st.selectbox("Inspect layer", list(range(SPEC["num_layers"])), index=0, key="layer_attr_sel")
            la1, la2 = st.columns([1.15, 0.85], gap="large")
            with la1:
                layer_sunburst = chart_layer_attribute_sunburst(obs, selected_layer)
                if layer_sunburst:
                    st.plotly_chart(layer_sunburst, use_container_width=True, config=CFG, key="layer_sunburst")
                radar = chart_layer_radar(obs, selected_layer)
                if radar:
                    st.plotly_chart(radar, use_container_width=True, config=CFG, key="cf_radar_v4")
            with la2:
                rms_chart = chart_rms_norm(obs.get("rms_norm", {}))
                if rms_chart:
                    st.plotly_chart(rms_chart, use_container_width=True, config=CFG, key="cf_rms_v4")
                attn_s = obs.get("attention", {}).get(selected_layer, {})
                mlp_s = obs.get("mlp", {}).get(selected_layer, {})
                gelu_s = obs.get("gelu", {}).get(selected_layer, {})
                residual_s = obs.get("residual_stream", {}).get(selected_layer, {})
                st.markdown(f"<div class='obs-header'><span class='dot'></span><h3>Layer {selected_layer} raw attributes</h3></div>", unsafe_allow_html=True)
                raw_rows = []
                for family, data in [("attention", attn_s), ("mlp", mlp_s), ("gelu", gelu_s), ("residual", residual_s)]:
                    for metric, value in data.items():
                        if isinstance(value, float):
                            raw_rows.append({"family": family, "metric": metric, "value": round(value, 6)})
                if raw_rows:
                    st.dataframe(raw_rows, use_container_width=True, hide_index=True, height=420)
                else:
                    st.info("No per-layer metrics captured for this layer yet.")

        with deck_tabs[7]:
            aa1, aa2 = st.columns([1.05, 0.95], gap="large")
            with aa1:
                entropy_fig = chart_head_entropy_heatmap(obs.get("attention", {}))
                if entropy_fig:
                    st.plotly_chart(entropy_fig, use_container_width=True, config=CFG, key="attn_entropy_hm_v4")
                focus_layer_candidates = [layer for layer in sorted(obs.get("attention", {}).keys()) if "last_token_attn" in obs.get("attention", {}).get(layer, {})]
                if focus_layer_candidates:
                    focus_layer = st.selectbox("Focus layer", focus_layer_candidates, format_func=lambda x: f"Layer {x}", key="attn_layer_sel_v4")
                    focus_fig = chart_head_focus(obs.get("attention", {}), focus_layer)
                    if focus_fig:
                        st.plotly_chart(focus_fig, use_container_width=True, config=CFG, key="attn_head_focus_v4")
            with aa2:
                compare_fig = chart_attn_vs_mlp(obs.get("attention", {}), obs.get("mlp", {}))
                if compare_fig:
                    st.plotly_chart(compare_fig, use_container_width=True, config=CFG, key="cf_attn_mlp_v4")
                ranking_rows = []
                for layer, data in sorted(obs.get("attention", {}).items()):
                    for head, entropy in enumerate(data.get("head_entropy", [])[:DISPLAY_HEADS]):
                        ranking_rows.append({"layer": layer, "head": head, "entropy": round(entropy, 4)})
                if ranking_rows:
                    import pandas as pd
                    ranking = pd.DataFrame(ranking_rows).sort_values("entropy", ascending=False)
                    st.dataframe(ranking, use_container_width=True, hide_index=True, height=540)
                else:
                    st.info("Attention capture is disabled or no head entropy was returned.")

        with deck_tabs[8]:
            tt1, tt2 = st.columns([1.1, 0.9], gap="large")
            with tt1:
                token_sunburst = chart_token_candidate_sunburst(obs.get("token_probs", []))
                if token_sunburst:
                    st.plotly_chart(token_sunburst, use_container_width=True, config=CFG, key="token_sunburst")
                else:
                    st.info("Decode candidate sunburst needs per-step token probabilities. Generate a fresh response to populate it.")
                entropy_timeline = chart_entropy_timeline(obs.get("token_probs", []))
                if entropy_timeline:
                    st.plotly_chart(entropy_timeline, use_container_width=True, config=CFG, key="cf_entropy_v4")
            with tt2:
                token_probs_chart = chart_token_probs(obs.get("token_probs", []))
                if token_probs_chart:
                    st.plotly_chart(token_probs_chart, use_container_width=True, config=CFG, key="cf_token_probs_v4")
                residual_fig = chart_residual_stream(obs.get("residual_stream", {}))
                if residual_fig:
                    st.plotly_chart(residual_fig, use_container_width=True, config=CFG, key="cf_residual_v4")
                dist_fig = chart_activation_dist(obs.get("residual_stream", {}))
                if dist_fig:
                    st.plotly_chart(dist_fig, use_container_width=True, config=CFG, key="cf_dist_v4")

        with deck_tabs[9]:
            m1, m2 = st.columns([1.1, 0.9], gap="large")
            with m1:
                memory_fig = chart_memory_profile(obs)
                if memory_fig:
                    st.plotly_chart(memory_fig, use_container_width=True, config=CFG, key="memory_profile")
                melt_fig = chart_telemetry_melt(obs)
                if melt_fig:
                    st.plotly_chart(melt_fig, use_container_width=True, config=CFG, key="melt_profile")
            with m2:
                meta = obs.get("metadata", {})
                st.markdown(f"<div class='obs-header'><span class='dot'></span><h3>Memory and latency telemetry</h3></div>", unsafe_allow_html=True)
                insights = [
                    ("Peak KV cache", f"{summary.get('max_kv_cache_mb', 0):.3f} MB", "Estimated decode-time cache growth from active heads, head dimension, and total context length."),
                    ("Prefill activation", f"{meta.get('activation_prefill_mb', 0):.3f} MB", "Approximate one-pass activation footprint before decode begins."),
                    ("TTFT", f"{meta.get('ttft_s', 0):.3f} s", "True time to first token, measured from streaming telemetry rather than total-time averaging."),
                    ("Stream chunks", f"{len(meta.get('stream_chunks', []))}", "Count of visible streaming updates emitted to the chat panel while the model was decoding."),
                ]
                for card_title, card_value, card_caption in insights:
                    st.markdown(f"<div class='insight-card'><div class='ic-title'>{card_title}</div><div class='ic-value'>{card_value}</div><div class='ic-caption'>{card_caption}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='glass' style='font-size:13px;color:{TEXT_2};line-height:1.75;'><b style='color:{TEXT_1};'>Instrumentation narrative.</b><br>This tab treats the chat as a telemetry source. Metrics show KV cache growth and TPS. Events come from chunked streaming updates. Logs are token-probability snapshots. Traces are layer and attention captures. Together they form a MELT-style explanation of where decode bottlenecks begin.</div>", unsafe_allow_html=True)

        with deck_tabs[10]:
            st.markdown(f'<div class="obs-header"><span class="dot"></span><h3>Experiment Bench</h3></div>', unsafe_allow_html=True)
            st.markdown(f"<p style='color:{TEXT_2};font-size:13px;'>Run structured scenarios, inspect the generated reply, and keep artifacts for later comparison.</p>", unsafe_allow_html=True)

            ex1, ex2 = st.columns([0.9, 1.1], gap="large")

            with ex1:
                scenario = st.selectbox("Scenario", list(EXPERIMENTS.keys()), key="exp_scenario")
                st.caption(EXPERIMENTS[scenario]["desc"])
                prompt_choice = st.selectbox("Prompt", EXPERIMENTS[scenario]["prompts"], key="exp_prompt")
                custom = st.text_area("Custom prompt", height=100, key="exp_custom")
                final_prompt = custom.strip() if custom.strip() else prompt_choice

                if st.button("Run experiment", type="primary", use_container_width=True, key="exp_run"):
                    eph = [
                        {"role": "user", "content": st.session_state.system_prompt},
                        {"role": "assistant", "content": "Understood."},
                        {"role": "user", "content": final_prompt},
                    ]
                    with st.spinner("Running experiment and capturing internals..."):
                        try:
                            resp, obs_sum = generate_with_observability(
                                model, tokenizer, eph, observer, logits_proc,
                                max_tokens=st.session_state.max_tokens, temperature=st.session_state.temperature)
                            result = dict(
                                scenario=scenario,
                                prompt=final_prompt,
                                response=resp,
                                obs=obs_sum,
                                timestamp=datetime.datetime.now().isoformat(),
                                artifact=observer.build_artifact(eph, resp),
                            )
                            st.session_state.experiment_results.append(result)
                            st.session_state.obs_data = obs_sum
                            st.rerun()
                        except Exception as e:
                            st.error(f"Experiment failed: {e}")

            with ex2:
                if st.session_state.experiment_results:
                    latest = st.session_state.experiment_results[-1]
                    st.markdown(f"<div class='glass'><p style='color:{TEXT_2};font-size:12px;'>LATEST SCENARIO</p><p style='color:{TEXT_1};font-size:1.05rem;font-weight:700;margin-bottom:0;'>{latest['scenario']}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='glass'><p style='color:{TEXT_2};font-size:12px;'>PROMPT</p><p style='color:{TEXT_1};margin-bottom:0;'>{latest['prompt']}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='glass'><p style='color:{TEXT_2};font-size:12px;'>RESPONSE</p><p style='color:{TEXT_1};margin-bottom:0;'>{latest['response']}</p></div>", unsafe_allow_html=True)

                    meta = latest["obs"].get("metadata", {})
                    ec = st.columns(4)
                    with ec[0]:
                        st.metric("Input", meta.get("input_tokens", "--"))
                    with ec[1]:
                        st.metric("Output", meta.get("output_tokens", "--"))
                    with ec[2]:
                        st.metric("Speed", f'{meta.get("tokens_per_second","--")} tok/s')
                    with ec[3]:
                        st.metric("Time", f'{meta.get("generation_time_s","--")}s')

                    art_json = json.dumps(latest["artifact"], indent=2, default=str)
                    st.download_button(
                        "Download experiment artifact",
                        data=art_json,
                        file_name=f"experiment_{latest['scenario'].replace(' ','_')}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                    quick_fig = chart_gelu_analysis(latest["obs"].get("gelu", {}))
                    if quick_fig:
                        st.plotly_chart(quick_fig, use_container_width=True, config=CFG, key="exp_gelu_v4")
                else:
                    st.markdown(f'<div class="empty-deck"><h3>No experiment run yet</h3><p>Pick a scenario or write a custom benchmark prompt. The resulting answer will populate this panel and also refresh the live observability deck above.</p></div>', unsafe_allow_html=True)

            if len(st.session_state.experiment_results) > 1:
                st.markdown("---")
                hist_rows = []
                for i, run in enumerate(st.session_state.experiment_results):
                    meta = run["obs"].get("metadata", {})
                    hist_rows.append({
                        "run": i + 1,
                        "scenario": run["scenario"],
                        "prompt": run["prompt"][:70] + ("..." if len(run["prompt"]) > 70 else ""),
                        "output_tokens": meta.get("output_tokens", "--"),
                        "tok_s": meta.get("tokens_per_second", "--"),
                        "time_s": meta.get("generation_time_s", "--"),
                    })
                st.dataframe(hist_rows, use_container_width=True, hide_index=True)
    else:
        st.markdown(f"""
        <div class="empty-deck">
            <h3>Observability deck is waiting for the first response.</h3>
            <p>Send a prompt from the left panel. Once a reply is generated, this space will transform into a layer-aware dashboard with sunburst views, attention atlases, token trajectories, and experiment traces tied to that response.</p>
        </div>
        """, unsafe_allow_html=True)

        preview_cols = st.columns(5)
        preview_items = [
            ("Architecture", SPEC["architecture"]),
            ("Layers", SPEC["num_layers"]),
            ("Heads", SPEC["num_heads"]),
            ("Activation", SPEC["activation"]),
            ("Context", SPEC["context_length"]),
        ]
        for i, (label, value) in enumerate(preview_items):
            with preview_cols[i]:
                st.markdown(f"<div class='spec-card'><div class='lbl'>{label}</div><div class='val'>{value}</div></div>", unsafe_allow_html=True)
