#!/usr/bin/env python3
"""
SLM Evaluation Suite — Tkinter Edition
Self-hosted, beginner + expert friendly.

Choose a local directory OR a Hugging Face model (after HF login),
watch the pipeline run live, then browse the evidence cards,
charts and descriptive summaries in the Results tab. Past runs
are saved as <model>__<date> and can be reopened or exported.

Run:
    python app.py
Requires: pip install -r requirements.txt
Optional for real models: torch + transformers
"""
import sys
import os
import pathlib
import threading
import queue
import json
import traceback
import webbrowser
import subprocess
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# Ensure src is importable when running from project root
ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.adapters.storage import RunStorage
from src.adapters.fake_runtime import FakeRuntime
from src.adapters.local_runtime import inspect_local_model
from src.adapters.hardware import capture_environment
from src.core.orchestration import Orchestrator, SECTIONS
from src.reports.export import export_pdf, export_docx

# Theme
BG_DEEP = "#0a0a0f"
BG = "#111114"
BG_CARD = "#181a20"
BG_SURFACE = "#1f242e"
BG_HOVER = "#2a3040"
BORDER = "#2a2f3a"
ACCENT = "#a78bfa"
ACCENT_DIM = "#25203a"
CYAN = "#22d3ee"
GREEN = "#34d399"
AMBER = "#fbbf24"
PINK = "#f472b6"
TEXT_1 = "#f5f7fb"
TEXT_2 = "#a1a1aa"
TEXT_3 = "#71717a"
WHITE = "#ffffff"
ERROR = "#f87171"

# Recommended HF SLMs (small, CPU-friendly)
HF_SUGGESTIONS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "google/gemma-3-270m-it",
    "google/gemma-3-1b-it",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct",
]

def _now_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

class SLMEvalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SLM Evaluation Suite  —  Measure, don't mythologise.")
        self.geometry("1180x760")
        self.minsize(1040, 640)
        self.configure(bg=BG_DEEP)

        # state
        self.storage = RunStorage(ROOT / "runs")
        self.model_source_var = tk.StringVar(value="fake")  # fake | local | hf
        self.local_path_var = tk.StringVar(value="")
        self.hf_model_var = tk.StringVar(value=HF_SUGGESTIONS[0])
        self.hf_token_var = tk.StringVar(value="")
        self.preset_var = tk.StringVar(value="quick")
        self.runtime_type_var = tk.StringVar(value="fake")
        self.hf_login_status = tk.StringVar(value="Checking …")
        self.hf_logged_in = False
        self.current_run_dir: pathlib.Path | None = None
        self.current_run_id: str | None = None
        self.orchestrator: Orchestrator | None = None
        self.worker_thread: threading.Thread | None = None
        self.event_queue: queue.Queue = queue.Queue()
        self.section_status: dict[str, str] = {k: "pending" for k, _ in SECTIONS}
        self.chart_images: list = []  # keep refs
        self.chart_labels: list = []
        self.history_selection: pathlib.Path | None = None

        self._build_style()
        self._build_layout()
        self._check_hf_login_async()
        self.after(100, self._poll_queue)
        self._refresh_history()

    # ---------- styling ----------
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=BG_DEEP, borderwidth=0, tabmargins=[8,8,0,0])
        style.configure("TNotebook.Tab", background=BG_CARD, foreground=TEXT_2, padding=[14,8], font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", BG_SURFACE)], foreground=[("selected", TEXT_1)])
        style.configure("TFrame", background=BG_DEEP)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("TLabel", background=BG_DEEP, foreground=TEXT_1, font=("Segoe UI", 9))
        style.configure("Card.TLabel", background=BG_CARD, foreground=TEXT_1)
        style.configure("Small.TLabel", background=BG_DEEP, foreground=TEXT_3, font=("Segoe UI", 8))
        style.configure("CardSmall.TLabel", background=BG_CARD, foreground=TEXT_3, font=("Segoe UI", 8))
        style.configure("Kicker.TLabel", background=BG_DEEP, foreground=ACCENT, font=("Segoe UI", 7, "bold"))
        style.configure("TButton", background=BG_SURFACE, foreground=TEXT_1, borderwidth=0, focusthickness=0, padding=[14,7], font=("Segoe UI", 9, "bold"))
        style.map("TButton", background=[("active", BG_HOVER)], foreground=[("disabled", TEXT_3)])
        style.configure("Accent.TButton", background=ACCENT, foreground=WHITE, padding=[16,8], font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#8b5cf6")])
        style.configure("Ghost.TButton", background=BG_DEEP, foreground=TEXT_2, borderwidth=1, relief="solid")
        style.configure("TProgressbar", background=ACCENT, troughcolor=BG_SURFACE, borderwidth=0)
        style.configure("TEntry", fieldbackground=BG_SURFACE, foreground=TEXT_1, borderwidth=0, padding=6)
        style.configure("TCombobox", fieldbackground=BG_SURFACE, background=BG_SURFACE, foreground=TEXT_1, arrowcolor=TEXT_2)
        style.configure("TRadiobutton", background=BG_CARD, foreground=TEXT_1, font=("Segoe UI", 9))
        style.map("TRadiobutton", background=[("active", BG_CARD)], indicatorcolor=[("selected", ACCENT)])

    # ---------- layout ----------
    def _build_layout(self):
        # top bar
        top = tk.Frame(self, bg=BG_DEEP, height=56)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)
        # banner
        left = tk.Frame(top, bg=BG_DEEP)
        left.pack(side="left", fill="y", padx=18, pady=10)
        tk.Label(left, text="SLM EVALUATION SUITE", bg=BG_DEEP, fg=ACCENT, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(left, text="Measure local model performance without pretending one number explains quality", bg=BG_DEEP, fg=TEXT_3, font=("Segoe UI", 8)).pack(anchor="w")
        # right badges
        right = tk.Frame(top, bg=BG_DEEP)
        right.pack(side="right", fill="y", padx=18, pady=14)
        self._badge(right, "LOCAL ONLY", ACCENT_DIM, ACCENT)
        self._badge(right, "EVIDENCE CARDS", BG_SURFACE, TEXT_2)
        self._badge(right, "v0.1.0", BG_SURFACE, TEXT_3)

        # notebook
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(0,10))

        self.tab_setup = ttk.Frame(self.nb, style="TFrame")
        self.tab_progress = ttk.Frame(self.nb, style="TFrame")
        self.tab_results = ttk.Frame(self.nb, style="TFrame")
        self.tab_history = ttk.Frame(self.nb, style="TFrame")

        self.nb.add(self.tab_setup, text="  ①  Setup  ")
        self.nb.add(self.tab_progress, text="  ②  Live Run  ")
        self.nb.add(self.tab_results, text="  ③  Results  ")
        self.nb.add(self.tab_history, text="  ④  History  ")

        self._build_setup()
        self._build_progress()
        self._build_results()
        self._build_history()

        # footer
        footer = tk.Frame(self, bg=BG_DEEP, height=22)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        self.footer_var = tk.StringVar(value="Ready — choose a model to begin.  •  Fake runtime requires no download.")
        tk.Label(footer, textvariable=self.footer_var, bg=BG_DEEP, fg=TEXT_3, font=("Segoe UI", 7), anchor="w").pack(side="left", padx=14)
        tk.Label(footer, text="Evidence taxonomy: MEASURED / DERIVED / INTERPRETED — descriptive only", bg=BG_DEEP, fg=TEXT_3, font=("Segoe UI", 7)).pack(side="right", padx=14)

    def _badge(self, parent, text, bg, fg):
        f = tk.Frame(parent, bg=bg, bd=1, relief="solid", highlightbackground=BORDER, highlightcolor=BORDER)
        # using Frame with custom bg
        l = tk.Label(f, text=text, bg=bg, fg=fg, font=("Segoe UI", 7, "bold"), padx=8, pady=2)
        l.pack()
        f.pack(side="left", padx=4)

    # ---- Setup tab
    def _build_setup(self):
        outer = tk.Frame(self.tab_setup, bg=BG_DEEP)
        outer.pack(fill="both", expand=True, padx=14, pady=14)
        # left column: model selection
        left = tk.Frame(outer, bg=BG_DEEP)
        left.pack(side="left", fill="both", expand=True, padx=(0,8))
        right = tk.Frame(outer, bg=BG_DEEP, width=360)
        right.pack(side="right", fill="y", padx=(8,0))
        right.pack_propagate(False)

        # Card: Model source
        card = self._card(left, "MODEL SOURCE — choose one")
        # radio row
        rb_frame = tk.Frame(card, bg=BG_CARD)
        rb_frame.pack(fill="x", pady=(6,8))
        ttk.Radiobutton(rb_frame, text="Demo (Fake — no download)", variable=self.model_source_var, value="fake", command=self._on_source_change).pack(anchor="w", pady=2)
        ttk.Radiobutton(rb_frame, text="Local directory (offline, trust_remote_code=False)", variable=self.model_source_var, value="local", command=self._on_source_change).pack(anchor="w", pady=2)
        ttk.Radiobutton(rb_frame, text="Hugging Face Hub (requires login)", variable=self.model_source_var, value="hf", command=self._on_source_change).pack(anchor="w", pady=2)
        tk.Label(card, text="Tip: leave on Fake for a full demo in < 90 s. Local/HF will load real weights if torch+transformers are installed.", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7), wraplength=540, justify="left").pack(anchor="w", pady=(4,0))

        # Local picker (hidden by default)
        self.local_frame = tk.Frame(card, bg=BG_CARD)
        # we will show/hide via pack
        tk.Label(self.local_frame, text="Local model path", bg=BG_CARD, fg=TEXT_2, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(8,2))
        row = tk.Frame(self.local_frame, bg=BG_CARD)
        row.pack(fill="x")
        ent = ttk.Entry(row, textvariable=self.local_path_var)
        ent.pack(side="left", fill="x", expand=True, padx=(0,6))
        ttk.Button(row, text="Browse…", command=self._on_browse_local).pack(side="left")
        self.local_info_var = tk.StringVar(value="No directory selected. Must contain config.json.")
        tk.Label(self.local_frame, textvariable=self.local_info_var, bg=BG_CARD, fg=TEXT_3, font=("Consolas", 7), wraplength=540, justify="left").pack(anchor="w", pady=(4,0))
        ttk.Button(self.local_frame, text="Validate local model", command=self._validate_local).pack(anchor="w", pady=(6,0))

        # HF frame
        self.hf_frame = tk.Frame(card, bg=BG_CARD)
        # login row
        tk.Label(self.hf_frame, text="Hugging Face login", bg=BG_CARD, fg=TEXT_2, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(8,4))
        # status
        status_row = tk.Frame(self.hf_frame, bg=BG_CARD)
        status_row.pack(fill="x")
        tk.Label(status_row, text="Status:", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 8)).pack(side="left")
        tk.Label(status_row, textvariable=self.hf_login_status, bg=BG_CARD, fg=AMBER, font=("Segoe UI", 8, "bold")).pack(side="left", padx=6)
        ttk.Button(status_row, text="Refresh", command=self._check_hf_login_async).pack(side="right", padx=4)
        # token entry + login
        tok_row = tk.Frame(self.hf_frame, bg=BG_CARD)
        tok_row.pack(fill="x", pady=(6,4))
        tk.Label(tok_row, text="HF token (read):", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 8)).pack(side="left")
        tok_ent = ttk.Entry(tok_row, textvariable=self.hf_token_var, show="•", width=22)
        tok_ent.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(tok_row, text="Login", command=self._on_hf_login).pack(side="left")
        hint = tk.Label(self.hf_frame, text="Get a token at huggingface.co/settings/tokens — read-only is enough. Tokens are written to HF cache, never logged.", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7), wraplength=540, justify="left")
        hint.pack(anchor="w", pady=(2,6))
        # model picker
        tk.Label(self.hf_frame, text="Model ID", bg=BG_CARD, fg=TEXT_2, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        combo_row = tk.Frame(self.hf_frame, bg=BG_CARD)
        combo_row.pack(fill="x", pady=2)
        self.hf_combo = ttk.Combobox(combo_row, textvariable=self.hf_model_var, values=HF_SUGGESTIONS, font=("Consolas", 8))
        self.hf_combo.pack(side="left", fill="x", expand=True, padx=(0,6))
        ttk.Button(combo_row, text="Use", command=lambda: self.hf_login_status.set(f"Selected {self.hf_model_var.get()}")).pack(side="left")
        # quick suggestions
        sug_frame = tk.Frame(self.hf_frame, bg=BG_CARD)
        sug_frame.pack(fill="x", pady=(4,0))
        for mid in HF_SUGGESTIONS[:4]:
            b = tk.Label(sug_frame, text=mid.split("/")[-1], bg=BG_SURFACE, fg=CYAN, font=("Segoe UI", 7, "bold"), padx=6, pady=2, cursor="hand2")
            b.pack(side="left", padx=2)
            b.bind("<Button-1>", lambda e, m=mid: self.hf_model_var.set(m))
            b.bind("<Enter>", lambda e, w=b: w.config(bg=BG_HOVER))
            b.bind("<Leave>", lambda e, w=b: w.config(bg=BG_SURFACE))
        self.hf_detail_var = tk.StringVar(value="Select a model above or type any HF repo id. Gated models (gemma-3-*) need licence acceptance + token.")
        tk.Label(self.hf_frame, textvariable=self.hf_detail_var, bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7), wraplength=540, justify="left").pack(anchor="w", pady=(6,0))

        # Preset card
        card2 = self._card(left, "EVALUATION PRESET")
        preset_row = tk.Frame(card2, bg=BG_CARD)
        preset_row.pack(fill="x", pady=4)
        ttk.Radiobutton(preset_row, text="Quick  (~90 s, 64 tok, 1 run, 3 context points)", variable=self.preset_var, value="quick").pack(anchor="w", pady=2)
        ttk.Radiobutton(preset_row, text="Standard  (~4 min, 128 tok, 3 runs with σ, 5 points)", variable=self.preset_var, value="standard").pack(anchor="w", pady=2)
        tk.Label(card2, text="Quick is default for demos. Standard repeats measurement 3× and reports dispersion.", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7), wraplength=540).pack(anchor="w", pady=(4,0))

        # How it works card
        card3 = self._card(left, "HOW IT WORKS — for beginners & experts")
        txt = (
            "Beginner: This tool measures speed, not quality. It tells you how fast THIS model runs on YOUR machine for realistic prompts, "
            "whether speed falls as context grows, and whether the model can cite neutral synthetic facts. Charts do the teaching — no ML expertise needed.\n\n"
            "Expert: Streaming TTFT via TextIteratorStreamer (approx), greedy decoding, warm-up excluded, MEASURED/DERIVED/INTERPRETED split, "
            "deterministic run_id, atomic JSON. No hand-wavy single score — evidence cards only. Limitations are printed beside the numbers they qualify."
        )
        tk.Label(card3, text=txt, bg=BG_CARD, fg=TEXT_2, font=("Segoe UI", 8), wraplength=540, justify="left").pack(anchor="w")

        # Right column: ready card + action
        rcard = self._card(right, "READY TO RUN")
        self.ready_summary_var = tk.StringVar(value="Demo (Fake) · Quick preset\nNo download required · ~90 s")
        tk.Label(rcard, textvariable=self.ready_summary_var, bg=BG_CARD, fg=TEXT_1, font=("Consolas", 8), justify="left", anchor="w").pack(anchor="w", pady=4)
        self.model_preview = ScrolledText(rcard, height=12, bg=BG_SURFACE, fg=TEXT_2, font=("Consolas", 7), bd=0, relief="flat", wrap="word")
        self.model_preview.pack(fill="both", expand=True, pady=6)
        self.model_preview.insert("1.0", "Select a model source above.\n\nFake runtime simulates a 135M-scale decoder (12 layers, 8 heads) with realistic power-law slowdown (r = k / n^α).\n\nLocal dir will show config.json allowlisted fields.\nHF will show download/cache path once resolved.\n\nAll runs are saved to runs/<model>__<date> with manifest, environment, measurements, charts and overview.md")
        self.model_preview.configure(state="disabled")

        btn_row = tk.Frame(rcard, bg=BG_CARD)
        btn_row.pack(fill="x", pady=(8,0))
        self.run_btn = ttk.Button(btn_row, text="▶  Run evaluation", style="Accent.TButton", command=self._on_run)
        self.run_btn.pack(fill="x", expand=True)
        tk.Label(rcard, text="Runs live — switch to Live Run tab to watch progress.", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7)).pack(pady=4)
        # links
        link_row = tk.Frame(rcard, bg=BG_CARD)
        link_row.pack(fill="x", pady=(8,0))
        for label, url in [("Docs", "https://github.com/divyamtewary/side-projects"), ("HF tokens", "https://huggingface.co/settings/tokens")]:
            l = tk.Label(link_row, text=label, bg=BG_CARD, fg=CYAN, font=("Segoe UI", 7, "underline"), cursor="hand2")
            l.pack(side="left", padx=6)
            l.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

        self._on_source_change()
        # watch vars for ready summary
        self.preset_var.trace_add("write", lambda *a: self._update_ready_summary())
        self.hf_model_var.trace_add("write", lambda *a: self._update_ready_summary())
        self.local_path_var.trace_add("write", lambda *a: self._update_ready_summary())

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=BORDER, bd=0)
        outer.pack(fill="x", pady=6, padx=2)
        inner = tk.Frame(outer, bg=BG_CARD)
        inner.pack(fill="x", padx=1, pady=1)
        tk.Label(inner, text=title, bg=BG_CARD, fg=ACCENT, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=10, pady=(8,2))
        body = tk.Frame(inner, bg=BG_CARD)
        body.pack(fill="x", padx=10, pady=(0,10))
        return body

    def _on_source_change(self):
        src = self.model_source_var.get()
        # hide both, show selected
        for w in (self.local_frame, self.hf_frame):
            w.pack_forget()
        if src == "local":
            self.local_frame.pack(fill="x", pady=4)
            self.runtime_type_var.set("local")
        elif src == "hf":
            self.hf_frame.pack(fill="x", pady=4)
            self.runtime_type_var.set("hf_hub")
        else:
            self.runtime_type_var.set("fake")
        self._update_ready_summary()
        self._update_model_preview()

    def _update_ready_summary(self):
        src = self.model_source_var.get()
        preset = self.preset_var.get()
        if src == "fake":
            self.ready_summary_var.set(f"Demo (Fake) · {preset.capitalize()} preset\nNo download · ~{'90 s' if preset=='quick' else '4 min'}")
        elif src == "local":
            p = self.local_path_var.get() or "(no path)"
            self.ready_summary_var.set(f"Local dir · {preset.capitalize()}\n{p[:40]}")
        else:
            mid = self.hf_model_var.get()
            self.ready_summary_var.set(f"HF Hub · {preset.capitalize()}\n{mid}")
        self._update_model_preview()

    def _update_model_preview(self):
        txt = ""
        src = self.model_source_var.get()
        if src == "fake":
            txt = (
                "FakeRuntime selected (recommended for first run).\n"
                "• 135M-scale decoder, 12 layers, 768 hidden, 32k vocab (synthetic)\n"
                "• Realistic power-law: tok/s ∝ 1/n^0.35 plus per-token MA + binned latency\n"
                "• No torch/transformers required — great for CI and screenshots\n"
                "• Switch to Local/HF once you want real weights\n"
            )
        elif src == "local":
            p = self.local_path_var.get()
            if p and pathlib.Path(p).exists():
                try:
                    info = inspect_local_model(p)
                    txt = f"Local model inspected:\n• Architecture: {info.get('architecture')}\n• Layers: {info.get('num_layers')}  Hidden: {info.get('hidden_size')}  Vocab: {info.get('vocab_size')}\n• Files: {len(info.get('files',[]))}  Config filtered ({len(info.get('config_raw',{}))} fields)\n• Path: {p}\n"
                    for f in info.get('files',[])[:5]:
                        txt += f"  - {f['name']}  {f['size_mb']} MB\n"
                except Exception as e:
                    txt = f"Local path error: {e}\nMust contain config.json and be a valid HF model directory."
            else:
                txt = f"Local directory:\n{ (p or '(empty)') }\n\nMust contain config.json. Use Browse to select folder.\n"
        else:
            mid = self.hf_model_var.get()
            txt = (
                f"HF model: {mid}\n"
                f"• After login, weights are snapshot_download() cached under %HF_HOME%\n"
                f"• Login status: {self.hf_login_status.get()}\n"
                f"• Gated repos (google/gemma-*) require licence click on model page + token with access.\n"
                f"• If torch/transformers missing, install requirements.txt then reload.\n"
            )
            # show login status more
            if self.hf_logged_in:
                txt += "✓ Logged in — you can run gated models you've accepted.\n"
            else:
                txt += "○ Not logged in — ungated models (SmolLM2, Qwen, TinyLlama, Phi-3) still work; gated will 401.\n"
        self.model_preview.configure(state="normal")
        self.model_preview.delete("1.0","end")
        self.model_preview.insert("1.0", txt)
        self.model_preview.configure(state="disabled")

    def _on_browse_local(self):
        d = filedialog.askdirectory(title="Select model directory (must contain config.json)", mustexist=True)
        if d:
            self.local_path_var.set(d)
            self._validate_local()

    def _validate_local(self):
        p = self.local_path_var.get()
        if not p:
            messagebox.showwarning("Local model", "Pick a directory first.")
            return
        try:
            info = inspect_local_model(p)
            self.local_info_var.set(f"✓ Valid: {info.get('architecture')}  •  {info.get('num_layers')} layers  •  {len(info.get('files',[]))} files")
            self._update_model_preview()
        except Exception as e:
            self.local_info_var.set(f"✗ {e}")
            messagebox.showerror("Validation failed", str(e))

    # HF login helpers
    def _check_hf_login_async(self):
        def worker():
            try:
                from src.adapters.hf_runtime import check_hf_login
                res = check_hf_login()
                logged = bool(res.get("logged_in"))
                user = res.get("user","") or ""
                err = res.get("error","")
                def upd():
                    self.hf_logged_in = logged
                    if logged:
                        self.hf_login_status.set(f"✓ Logged in as {user}")
                        # find label
                        for w in self.tab_setup.winfo_descendants():
                            if isinstance(w, tk.Label) and w.cget("textvariable") == str(self.hf_login_status):
                                w.config(fg=GREEN)
                    else:
                        self.hf_login_status.set("○ Not logged in")
                        for w in self.tab_setup.winfo_descendants():
                            if isinstance(w, tk.Label) and w.cget("textvariable") == str(self.hf_login_status):
                                w.config(fg=AMBER)
                    self._update_model_preview()
                self.after(0, upd)
            except Exception as e:
                def err_upd():
                    self.hf_login_status.set("○ Check failed")
                    self.hf_logged_in = False
                    self._update_model_preview()
                self.after(0, err_upd)
        threading.Thread(target=worker, daemon=True).start()

    def _on_hf_login(self):
        token = self.hf_token_var.get().strip()
        if not token:
            messagebox.showwarning("HF Login", "Paste a read token from huggingface.co/settings/tokens")
            return
        try:
            from huggingface_hub import login as hf_login
            hf_login(token=token, add_to_git_credential=False)
            # also try write
            messagebox.showinfo("HF Login", "Token saved to HF cache. Checking status…")
            self.hf_token_var.set("")
            self._check_hf_login_async()
        except Exception as e:
            messagebox.showerror("Login failed", f"{e}\n\nTry: pip install -U huggingface_hub  then huggingface-cli login")
            traceback.print_exc()

    # ---- Progress tab
    def _build_progress(self):
        outer = tk.Frame(self.tab_progress, bg=BG_DEEP)
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        # top: progress bar + status
        top_card_outer = tk.Frame(outer, bg=BORDER)
        top_card_outer.pack(fill="x", pady=(0,8))
        top_card = tk.Frame(top_card_outer, bg=BG_CARD)
        top_card.pack(fill="x", padx=1, pady=1)
        row = tk.Frame(top_card, bg=BG_CARD)
        row.pack(fill="x", padx=12, pady=10)
        tk.Label(row, text="PIPELINE PROGRESS", bg=BG_CARD, fg=ACCENT, font=("Segoe UI", 7, "bold")).pack(anchor="w")
        # bar
        bar_row = tk.Frame(top_card, bg=BG_CARD)
        bar_row.pack(fill="x", padx=12, pady=(4,8))
        self.progress = ttk.Progressbar(bar_row, mode="determinate", maximum=100, value=0)
        self.progress.pack(fill="x", expand=True, side="left", padx=(0,8))
        self.progress_pct = tk.StringVar(value="0%")
        tk.Label(bar_row, textvariable=self.progress_pct, bg=BG_CARD, fg=TEXT_1, font=("Consolas", 10, "bold"), width=5).pack(side="left")
        self.overall_status_var = tk.StringVar(value="Idle — press Run in Setup.")
        tk.Label(top_card, textvariable=self.overall_status_var, bg=BG_CARD, fg=TEXT_2, font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=(0,8))
        btn_row = tk.Frame(top_card, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=(0,10))
        self.cancel_btn = ttk.Button(btn_row, text="Cancel", command=self._on_cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(0,6))
        ttk.Button(btn_row, text="Open runs folder", command=self._open_runs_folder).pack(side="left")
        ttk.Button(btn_row, text="Clear log", command=lambda: self.log_text.configure(state="normal") or self.log_text.delete("1.0","end") or self.log_text.configure(state="disabled")).pack(side="right")

        # middle split: section list + log
        mid = tk.Frame(outer, bg=BG_DEEP)
        mid.pack(fill="both", expand=True)

        # sections
        sec_outer = tk.Frame(mid, bg=BORDER, width=320)
        sec_outer.pack(side="left", fill="y", padx=(0,8))
        sec_outer.pack_propagate(False)
        sec_inner = tk.Frame(sec_outer, bg=BG_CARD)
        sec_inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(sec_inner, text="SECTIONS  (what's running now)", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=10, pady=(8,4))
        self.section_frame = tk.Frame(sec_inner, bg=BG_CARD)
        self.section_frame.pack(fill="x", padx=8, pady=4)
        self.section_rows: dict[str, dict] = {}
        for key, title in SECTIONS:
            r = tk.Frame(self.section_frame, bg=BG_CARD)
            r.pack(fill="x", pady=2)
            # dot
            dot = tk.Label(r, text="○", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 10, "bold"), width=2)
            dot.pack(side="left")
            tk.Label(r, text=title, bg=BG_CARD, fg=TEXT_1, font=("Segoe UI", 8, "bold"), anchor="w").pack(side="left", fill="x", expand=True)
            # status pill
            pill = tk.Label(r, text="pending", bg=BG_SURFACE, fg=TEXT_3, font=("Segoe UI", 7, "bold"), padx=6, pady=1)
            pill.pack(side="right")
            self.section_rows[key] = {"dot": dot, "pill": pill, "title": title}
        # legend
        tk.Label(sec_inner, text="○ pending  ● running (amber)  ✓ done (green)  ✗ failed (red)", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7)).pack(anchor="w", padx=10, pady=(8,6))
        tk.Label(sec_inner, text="All sections run sequentially. Progress events are also saved to events.jsonl.", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7), wraplength=300, justify="left").pack(anchor="w", padx=10)

        # log
        log_outer = tk.Frame(mid, bg=BORDER)
        log_outer.pack(side="left", fill="both", expand=True)
        log_inner = tk.Frame(log_outer, bg=BG_CARD)
        log_inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(log_inner, text="LIVE LOG  (streaming)", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=10, pady=(8,4))
        self.log_text = ScrolledText(log_inner, bg=BG_SURFACE, fg=TEXT_2, font=("Consolas", 8), bd=0, relief="flat", wrap="word", state="disabled", height=18)
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self._log("Ready. Choose a model in Setup and press Run.\n  • Fake runs in < 90 s and needs no GPU or token.\n  • Local/HF loads real weights (may take longer).\n")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ---- Results tab
    def _build_results(self):
        outer = tk.Frame(self.tab_results, bg=BG_DEEP)
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        # top bar: summary + export
        top = tk.Frame(outer, bg=BORDER)
        top.pack(fill="x", pady=(0,8))
        top_inner = tk.Frame(top, bg=BG_CARD)
        top_inner.pack(fill="x", padx=1, pady=1)
        hdr = tk.Frame(top_inner, bg=BG_CARD)
        hdr.pack(fill="x", padx=12, pady=8)
        left = tk.Frame(hdr, bg=BG_CARD)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="RESULTS — evidence cards", bg=BG_CARD, fg=ACCENT, font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.results_title_var = tk.StringVar(value="No run yet — run an evaluation or open one from History.")
        tk.Label(left, textvariable=self.results_title_var, bg=BG_CARD, fg=TEXT_1, font=("Segoe UI", 10, "bold"), wraplength=700, justify="left").pack(anchor="w", pady=2)
        self.results_sub_var = tk.StringVar(value="Charts: baseline • latency • context scaling • decode trace • accuracy")
        tk.Label(left, textvariable=self.results_sub_var, bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7)).pack(anchor="w")
        right = tk.Frame(hdr, bg=BG_CARD)
        right.pack(side="right", padx=8)
        ttk.Button(right, text="Export PDF", command=self._export_pdf).pack(side="left", padx=4, pady=4)
        ttk.Button(right, text="Export DOCX", command=self._export_docx).pack(side="left", padx=4, pady=4)
        ttk.Button(right, text="Open folder", command=self._open_current_folder).pack(side="left", padx=4)

        # scrollable results area
        container = tk.Frame(outer, bg=BG_DEEP)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, bg=BG_DEEP, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.results_scroll = tk.Frame(canvas, bg=BG_DEEP)
        self.results_scroll.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.results_scroll, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # summary text
        self.summary_text = ScrolledText(self.results_scroll, bg=BG_CARD, fg=TEXT_2, font=("Segoe UI", 8), bd=0, relief="flat", wrap="word", height=18)
        self.summary_text.pack(fill="x", padx=4, pady=4)
        self.summary_text.insert("1.0", "No evaluation yet.\n\n1. Go to Setup → choose Fake (quick) for a 60-second demo.\n2. Watch Live Run for real-time section status.\n3. Come back here for charts, metrics table and export buttons.\n\nEvery run is saved under runs/<model>__<date> so you can compare across models and time.")
        self.summary_text.configure(state="disabled")

        # metrics frame (beginner/expert toggle)
        toggle_row = tk.Frame(self.results_scroll, bg=BG_DEEP)
        toggle_row.pack(fill="x", pady=(8,4))
        tk.Label(toggle_row, text="View:", bg=BG_DEEP, fg=TEXT_3, font=("Segoe UI", 8, "bold")).pack(side="left", padx=4)
        self.view_mode = tk.StringVar(value="beginner")
        ttk.Radiobutton(toggle_row, text="Beginner (plain English)", variable=self.view_mode, value="beginner", command=self._refresh_results_view).pack(side="left", padx=4)
        ttk.Radiobutton(toggle_row, text="Expert (full numbers)", variable=self.view_mode, value="expert", command=self._refresh_results_view).pack(side="left", padx=4)
        tk.Label(toggle_row, text="Charts are always expert-grade; the text adapts.", bg=BG_DEEP, fg=TEXT_3, font=("Segoe UI", 7)).pack(side="left", padx=8)

        # metrics table (tree)
        table_outer = tk.Frame(self.results_scroll, bg=BORDER)
        table_outer.pack(fill="x", pady=4, padx=4)
        table_inner = tk.Frame(table_outer, bg=BG_CARD)
        table_inner.pack(fill="x", padx=1, pady=1)
        tk.Label(table_inner, text="KEY METRICS", bg=BG_CARD, fg=ACCENT, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=8, pady=(6,2))
        cols = ("metric","value","truth")
        self.metrics_tree = ttk.Treeview(table_inner, columns=cols, show="headings", height=10)
        self.metrics_tree.heading("metric", text="Metric")
        self.metrics_tree.heading("value", text="Value")
        self.metrics_tree.heading("truth", text="Truth level")
        self.metrics_tree.column("metric", width=280, anchor="w")
        self.metrics_tree.column("value", width=220, anchor="w")
        self.metrics_tree.column("truth", width=140, anchor="center")
        self.metrics_tree.pack(fill="x", padx=8, pady=(0,8))

        # charts gallery
        charts_outer = tk.Frame(self.results_scroll, bg=BORDER)
        charts_outer.pack(fill="both", expand=True, pady=6, padx=4)
        charts_inner = tk.Frame(charts_outer, bg=BG_CARD)
        charts_inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(charts_inner, text="CHARTS  (deterministic matplotlib — Agg)", bg=BG_CARD, fg=ACCENT, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=8, pady=(6,4))
        tk.Label(charts_inner, text="Click any chart to open full-size PNG.", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7)).pack(anchor="w", padx=8)
        self.charts_frame = tk.Frame(charts_inner, bg=BG_CARD)
        self.charts_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.charts_placeholder = tk.Label(self.charts_frame, text="No charts yet — run an evaluation.", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 8))
        self.charts_placeholder.pack(pady=20)

        # raw path helper
        self.results_raw_var = tk.StringVar(value="")
        tk.Label(self.results_scroll, textvariable=self.results_raw_var, bg=BG_DEEP, fg=TEXT_3, font=("Consolas", 7), wraplength=900, justify="left").pack(anchor="w", padx=4, pady=4)

    # ---- History tab
    def _build_history(self):
        outer = tk.Frame(self.tab_history, bg=BG_DEEP)
        outer.pack(fill="both", expand=True, padx=14, pady=14)
        # header
        hdr = tk.Frame(outer, bg=BORDER)
        hdr.pack(fill="x", pady=(0,8))
        hinner = tk.Frame(hdr, bg=BG_CARD)
        hinner.pack(fill="x", padx=1, pady=1)
        tk.Label(hinner, text="PAST EVALUATIONS", bg=BG_CARD, fg=ACCENT, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=12, pady=(8,2))
        tk.Label(hinner, text="Saved as runs/<model-name>__<UTC-date> — each folder is self-contained (manifest, measurements, charts, overview.md).", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7), wraplength=900, justify="left").pack(anchor="w", padx=12, pady=(0,6))
        bar = tk.Frame(hinner, bg=BG_CARD)
        bar.pack(fill="x", padx=12, pady=(0,8))
        ttk.Button(bar, text="↻  Refresh", command=self._refresh_history).pack(side="left", padx=(0,6))
        ttk.Button(bar, text="Open runs folder", command=self._open_runs_folder).pack(side="left", padx=6)
        ttk.Button(bar, text="Open selected", command=self._open_history_selection).pack(side="left", padx=6)
        ttk.Button(bar, text="Delete selected", command=self._delete_history_selection).pack(side="left", padx=6)

        # list + detail split
        split = tk.Frame(outer, bg=BG_DEEP)
        split.pack(fill="both", expand=True)
        list_outer = tk.Frame(split, bg=BORDER, width=420)
        list_outer.pack(side="left", fill="y", padx=(0,8))
        list_outer.pack_propagate(False)
        list_inner = tk.Frame(list_outer, bg=BG_CARD)
        list_inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(list_inner, text="Runs (newest first)", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=8, pady=(6,4))
        # tree for history
        cols = ("run","date","preset","tok/s")
        self.history_tree = ttk.Treeview(list_inner, columns=cols, show="headings", height=18, selectmode="browse")
        self.history_tree.heading("run", text="Model / Run")
        self.history_tree.heading("date", text="Date (UTC)")
        self.history_tree.heading("preset", text="Preset")
        self.history_tree.heading("tok/s", text="tok/s")
        self.history_tree.column("run", width=200)
        self.history_tree.column("date", width=130)
        self.history_tree.column("preset", width=60, anchor="center")
        self.history_tree.column("tok/s", width=60, anchor="center")
        vsb = ttk.Scrollbar(list_inner, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=vsb.set)
        self.history_tree.pack(side="left", fill="both", expand=True, padx=(8,0), pady=(0,8))
        vsb.pack(side="left", fill="y", pady=(0,8))
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_select)
        self.history_tree.bind("<Double-1>", lambda e: self._open_history_selection())

        # detail pane
        det_outer = tk.Frame(split, bg=BORDER)
        det_outer.pack(side="left", fill="both", expand=True)
        det_inner = tk.Frame(det_outer, bg=BG_CARD)
        det_inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(det_inner, text="DETAILS", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=8, pady=(6,4))
        self.history_detail = ScrolledText(det_inner, bg=BG_SURFACE, fg=TEXT_2, font=("Consolas", 7), bd=0, relief="flat", wrap="word", height=18)
        self.history_detail.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self.history_detail.insert("1.0", "Select a run on the left to see manifest + summary.\n\nDouble-click or press Open selected to load it into Results tab.")
        self.history_detail.configure(state="disabled")
        # export from history
        exp_row = tk.Frame(det_inner, bg=BG_CARD)
        exp_row.pack(fill="x", padx=8, pady=(0,8))
        ttk.Button(exp_row, text="Load in Results tab", command=self._open_history_selection).pack(side="left", padx=(0,6))
        ttk.Button(exp_row, text="Export PDF", command=lambda: self._export_from_history("pdf")).pack(side="left", padx=4)
        ttk.Button(exp_row, text="Export DOCX", command=lambda: self._export_from_history("docx")).pack(side="left", padx=4)

    # ---------- helpers ----------
    def _open_runs_folder(self):
        p = ROOT / "runs"
        p.mkdir(exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(p))  # type: ignore
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showinfo("Runs folder", f"{p}\n\nCould not open automatically: {e}")

    def _open_current_folder(self):
        if not self.current_run_dir or not self.current_run_dir.exists():
            messagebox.showinfo("No run", "No run is currently loaded. Pick one from History or run a new evaluation.")
            return
        p = self.current_run_dir
        try:
            if sys.platform == "win32":
                os.startfile(str(p))  # type: ignore
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showinfo("Run folder", f"{p}\n\n{e}")

    # ---------- run ----------
    def _on_run(self):
        # validate selection
        src = self.model_source_var.get()
        model_id = ""
        local_path = None
        runtime_type = self.runtime_type_var.get()
        if src == "fake":
            model_id = "fake://tiny"
            runtime_type = "fake"
        elif src == "local":
            p = self.local_path_var.get().strip()
            if not p:
                messagebox.showwarning("Local model", "Select a local model directory first.")
                self.nb.select(self.tab_setup)
                return
            pp = pathlib.Path(p)
            if not pp.exists() or not (pp / "config.json").exists():
                messagebox.showerror("Local model", f"Directory does not look like an HF model:\n{p}\n\nMissing config.json.")
                return
            model_id = str(pp.resolve())
            local_path = str(pp.resolve())
            runtime_type = "local"
        else:  # hf
            mid = self.hf_model_var.get().strip()
            if not mid or "/" not in mid:
                messagebox.showwarning("HF model", "Enter a valid HF model id, e.g. HuggingFaceTB/SmolLM2-135M-Instruct")
                return
            model_id = mid
            runtime_type = "hf_hub"
            # check gated+not logged
            if mid.startswith("google/") and not self.hf_logged_in:
                if not messagebox.askokcancel("Gated model without login",
                    "This looks like a gated model (google/*). You are not logged in, so download will likely fail with 401.\n\n"
                    "Continue anyway? (You can paste a token in Setup → HF login first.)"):
                    return

        preset = self.preset_var.get()
        # confirm run
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Already running", "An evaluation is already in progress. Cancel it first.")
            return

        # reset progress UI
        self.section_status = {k: "pending" for k, _ in SECTIONS}
        for k in self.section_status:
            self._set_section_status(k, "pending")
        self.progress.configure(value=0)
        self.progress_pct.set("0%")
        self.overall_status_var.set(f"Starting {model_id} ({preset}) …")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0","end")
        self.log_text.configure(state="disabled")
        self._log(f"Run requested: model={model_id}  preset={preset}  runtime={runtime_type}")
        self.nb.select(self.tab_progress)
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.footer_var.set(f"Running {model_id} — watch Live Run…")

        # build runtime
        try:
            if runtime_type == "fake":
                runtime = FakeRuntime(model_id=model_id)
            elif runtime_type == "local":
                # for local we still try real HF runtime if transformers available, else fake
                try:
                    from src.adapters.hf_runtime import HFTransformersRuntime
                    runtime = HFTransformersRuntime(model_id=model_id, local_path=local_path)
                except Exception as e:
                    self._log(f"HF runtime not available ({e}) — falling back to Fake for local dir.")
                    runtime = FakeRuntime(model_id=model_id)
            else:  # hf_hub
                try:
                    from src.adapters.hf_runtime import HFTransformersRuntime
                    runtime = HFTransformersRuntime(model_id=model_id)
                except Exception as e:
                    messagebox.showerror("Runtime unavailable", f"Could not create HF runtime: {e}\n\nInstall: pip install transformers torch huggingface_hub")
                    self._reset_run_buttons()
                    return
        except Exception as e:
            messagebox.showerror("Runtime error", str(e))
            self._reset_run_buttons()
            return

        # orchestrator
        self.orchestrator = Orchestrator(self.storage, runtime, model_id=model_id, preset=preset, runtime_type=runtime_type, local_path=local_path)
        # start worker
        def worker():
            try:
                run_dir = self.orchestrator.run(progress_cb=self._queue_event)
                self.event_queue.put({"type": "worker_done", "run_dir": str(run_dir)})
            except Exception as e:
                self.event_queue.put({"type": "error", "message": str(e), "trace": traceback.format_exc()})
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _queue_event(self, ev: dict):
        # called from worker thread; push to queue for UI thread
        try:
            self.event_queue.put(ev)
        except Exception:
            pass

    def _poll_queue(self):
        try:
            while True:
                ev = self.event_queue.get_nowait()
                self._handle_event(ev)
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _handle_event(self, ev: dict):
        t = ev.get("type")
        if t == "log":
            self._log(ev.get("message",""))
        elif t == "section_start":
            sec = ev.get("section")
            self._set_section_status(sec, "running")
            self.overall_status_var.set(f"Running: {ev.get('title',sec)} …")
            pct = ev.get("pct", 0)
            self.progress.configure(value=pct)
            self.progress_pct.set(f"{pct}%")
            self.footer_var.set(f"Running {sec} … {pct}%")
        elif t == "section_done":
            sec = ev.get("section")
            self._set_section_status(sec, "done")
            pct = ev.get("pct", 0)
            self.progress.configure(value=pct)
            self.progress_pct.set(f"{pct}%")
            self._log(f"✓ {ev.get('title',sec)} done")
        elif t == "run_start":
            self._log(f"Run {ev.get('run_id')} → {ev.get('run_dir')}")
        elif t == "run_done":
            self.progress.configure(value=100)
            self.progress_pct.set("100%")
            self.overall_status_var.set(f"Done — {ev.get('run_dir')}")
            self.footer_var.set("Evaluation complete — see Results tab.")
            self.current_run_dir = pathlib.Path(ev.get("run_dir")) if ev.get("run_dir") else None
            self.current_run_id = ev.get("run_id")
            self._log(f"✅ Run complete: {ev.get('run_dir')}")
            self._reset_run_buttons()
            self._load_results(self.current_run_dir)
            self._refresh_history()
            self.nb.select(self.tab_results)
            messagebox.showinfo("Done", f"Evaluation finished:\n{ev.get('run_dir')}\n\nResults tab now shows charts and summary.")
        elif t == "worker_done":
            # already handled via run_done
            pass
        elif t == "error":
            self._log(f"✗ ERROR: {ev.get('message')}")
            if ev.get("trace"):
                self._log(ev["trace"][:2000])
            self.overall_status_var.set(f"Error: {ev.get('message','unknown')[:80]}")
            self._reset_run_buttons()
            messagebox.showerror("Run failed", f"{ev.get('message')}\n\nSee Live Run log and events.jsonl")
        elif t == "cancelled":
            self.overall_status_var.set("Cancelled")
            self._log("Cancelled by user.")
            self._reset_run_buttons()
        else:
            # generic
            self._log(str(ev))

    def _set_section_status(self, key: str, status: str):
        if key not in self.section_rows:
            return
        row = self.section_rows[key]
        dot, pill = row["dot"], row["pill"]
        self.section_status[key] = status
        if status == "pending":
            dot.config(text="○", fg=TEXT_3)
            pill.config(text="pending", bg=BG_SURFACE, fg=TEXT_3)
        elif status == "running":
            dot.config(text="●", fg=AMBER)
            pill.config(text="running…", bg=AMBER, fg=BG_DEEP)
        elif status == "done":
            dot.config(text="✓", fg=GREEN)
            pill.config(text="done", bg=GREEN, fg=BG_DEEP)
        elif status == "failed":
            dot.config(text="✗", fg=ERROR)
            pill.config(text="failed", bg=ERROR, fg=WHITE)
        elif status == "skipped":
            dot.config(text="–", fg=TEXT_3)
            pill.config(text="skipped", bg=BG_SURFACE, fg=TEXT_3)

    def _reset_run_buttons(self):
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

    def _on_cancel(self):
        if self.orchestrator:
            self.orchestrator.cancel()
            self._log("Cancel requested — waiting for current section to finish …")
            self.overall_status_var.set("Cancelling …")
            self.cancel_btn.configure(state="disabled")

    # ---------- results loading ----------
    def _load_results(self, run_dir: pathlib.Path | None):
        if not run_dir or not run_dir.exists():
            return
        self.current_run_dir = run_dir
        try:
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.current_run_id = manifest.get("run_id")
        except Exception:
            manifest = {"model_id": run_dir.name, "run_id": run_dir.name, "created_utc": "?", "preset": "?", "runtime_type": "?"}
            self.current_run_id = run_dir.name

        # title
        self.results_title_var.set(f"{manifest.get('model_id','?')}  •  {manifest.get('preset','')}  •  {manifest.get('created_utc','')}")
        self.results_sub_var.set(f"Run {manifest.get('run_id','')}  •  {run_dir.name}  •  click charts to enlarge")
        self.results_raw_var.set(f"Folder: {run_dir}   •   {run_dir / 'overview.md'}")

        # summary text
        overview = run_dir / "overview.md"
        summary_json = run_dir / "summary.json"
        text = ""
        if overview.exists():
            text = overview.read_text(encoding="utf-8")
        elif summary_json.exists():
            try:
                sj = json.loads(summary_json.read_text(encoding="utf-8"))
                text = sj.get("overall_summary","") + "\n\n" + json.dumps(sj, indent=2)[:6000]
            except Exception:
                text = "No overview found."
        else:
            text = "No overview found in this run."

        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0","end")
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state="disabled")

        # metrics tree
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        # try to load measurements
        perf = {}
        ctx = {}
        dec = {}
        acc = {}
        try:
            if (run_dir / "raw" / "measurements.json").exists():
                perf = json.loads((run_dir / "raw" / "measurements.json").read_text(encoding="utf-8"))
            elif (run_dir / "summary.json").exists():
                perf = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get("performance",{})
        except Exception:
            pass
        try:
            if (run_dir / "raw" / "context_scaling.json").exists():
                ctx = json.loads((run_dir / "raw" / "context_scaling.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        try:
            if (run_dir / "raw" / "decode_position.json").exists():
                dec = json.loads((run_dir / "raw" / "decode_position.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        try:
            if (run_dir / "raw" / "accuracy.json").exists():
                acc = json.loads((run_dir / "raw" / "accuracy.json").read_text(encoding="utf-8"))
        except Exception:
            pass

        self._perf_cache = perf
        self._ctx_cache = ctx
        self._dec_cache = dec
        self._acc_cache = acc
        self._manifest_cache = manifest

        self._refresh_results_view()

        # charts
        for w in self.charts_frame.winfo_children():
            w.destroy()
        self.chart_images.clear()
        charts = sorted((run_dir / "charts").glob("*.png"))
        if not charts:
            self.charts_placeholder = tk.Label(self.charts_frame, text="No charts in this run.", bg=BG_CARD, fg=TEXT_3, font=("Segoe UI", 8))
            self.charts_placeholder.pack(pady=20)
        else:
            # grid 2 per row
            try:
                from PIL import Image, ImageTk
                has_pil = True
            except Exception:
                has_pil = False
                tk.Label(self.charts_frame, text="Install Pillow to preview charts inline: pip install Pillow\nCharts are still saved as PNGs in charts/ folder.", bg=BG_CARD, fg=AMBER, font=("Segoe UI", 7)).pack()
                for ch in charts:
                    tk.Label(self.charts_frame, text=ch.name, bg=BG_CARD, fg=CYAN, font=("Consolas", 7)).pack(anchor="w")
                return
            row_frame = None
            for idx, ch in enumerate(charts):
                if idx % 2 == 0:
                    row_frame = tk.Frame(self.charts_frame, bg=BG_CARD)
                    row_frame.pack(fill="x", pady=4)
                cell = tk.Frame(row_frame, bg=BG_SURFACE, bd=1, relief="solid")
                cell.pack(side="left", fill="both", expand=True, padx=4)
                # load thumbnail
                try:
                    im = Image.open(str(ch))
                    # thumbnail size
                    w, h = im.size
                    # target width ~420
                    target_w = 420
                    scale = target_w / w
                    target_h = int(h * scale)
                    if target_h > 280:
                        target_h = 280
                        scale = target_h / h
                        target_w = int(w * scale)
                    thumb = im.copy()
                    thumb.thumbnail((420, 280), Image.LANCZOS)
                    tk_im = ImageTk.PhotoImage(thumb)
                    self.chart_images.append(tk_im)
                    lbl = tk.Label(cell, image=tk_im, bg=BG_SURFACE, cursor="hand2", bd=0)
                    lbl.pack(padx=4, pady=4)
                    lbl.bind("<Button-1>", lambda e, p=ch: self._open_image(p))
                    # caption
                    tk.Label(cell, text=ch.name, bg=BG_SURFACE, fg=TEXT_2, font=("Consolas", 7)).pack()
                    # open button
                    tk.Label(cell, text="Click to open full-size", bg=BG_SURFACE, fg=TEXT_3, font=("Segoe UI", 7)).pack(pady=(0,4))
                except Exception as e:
                    tk.Label(cell, text=f"{ch.name}\n{e}", bg=BG_SURFACE, fg=ERROR, font=("Segoe UI", 7)).pack(padx=8, pady=8)

        # keep for export
        self._last_charts = charts

    def _refresh_results_view(self):
        # fill metrics tree based on view_mode
        if not hasattr(self, "_perf_cache"):
            return
        perf = getattr(self, "_perf_cache", {})
        ctx = getattr(self, "_ctx_cache", {})
        dec = getattr(self, "_dec_cache", {})
        acc = getattr(self, "_acc_cache", {})
        manifest = getattr(self, "_manifest_cache", {})
        view = self.view_mode.get()

        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)

        if view == "beginner":
            rows = [
                ("What was measured?", f"{manifest.get('model_id','?')} on your machine", "INTERPRETED"),
                ("How fast? (main number)", f"{perf.get('decode_tok_per_s','?')} tok/s decode  ({perf.get('overall_tok_per_s','?')} overall)", "DERIVED"),
                ("How long to first token? (streaming ≈)", f"{perf.get('prefill_time_s','?')} s", "MEASURED (≈)"),
                ("Is it steady?", f"mean {perf.get('inter_token_latency_ms_mean','?')} ms  •  P95 {perf.get('inter_token_latency_ms_p95','?')} ms", "DERIVED"),
                ("Does longer context slow it?", f"Fit α≈{ctx.get('fit_alpha','?')}  r≈{ctx.get('pearson_r','?')}  — see H1 chart", "INTERPRETED"),
                ("Does long output slow per-token?", f"slope {dec.get('slope_ms_per_token','?')} ms/tok  — see H2", "INTERPRETED"),
                ("Can it cite synthetic facts?", f"{acc.get('accuracy_score',0)*100:.0f}%  ({len(acc.get('matched_facts',[]))}/{len(acc.get('expected_facts',[]))})", "INTERPRETED"),
                ("Peak memory", f"{perf.get('peak_rss_mb','?')} MB", "MEASURED"),
            ]
        else:
            # expert full
            rows = [
                ("Model", manifest.get("model_id","?"), "measured"),
                ("Run ID", manifest.get("run_id","?")[:24]+"…", "measured"),
                ("Input tokens", str(perf.get("num_input_tokens","?")), "measured"),
                ("Output tokens", str(perf.get("num_output_tokens","?")), "measured"),
                ("Load time", f"{perf.get('load_time_s','?')} s", "measured"),
                ("Prefill/TTFT≈", f"{perf.get('prefill_time_s','?')} s  ({perf.get('prefill_tok_per_s','?')} tok/s)", "measured/derived"),
                ("Decode tok/s", f"{perf.get('decode_tok_per_s','?')}  overall {perf.get('overall_tok_per_s','?')}", "derived"),
                ("Inter-token mean/median/P95/P99", f"{perf.get('inter_token_latency_ms_mean','?')}/{perf.get('inter_token_latency_ms_median','?')}/{perf.get('inter_token_latency_ms_p95','?')}/{perf.get('inter_token_latency_ms_p99','?')} ms", "derived"),
                ("Peak RSS", f"{perf.get('peak_rss_mb','?')} MB", "measured"),
                ("H1 fit k/α/r", f"k={ctx.get('fit_k','?')} α={ctx.get('fit_alpha','?')} r={ctx.get('pearson_r','?')}", "derived/interpreted"),
                ("H2 slope", f"{dec.get('slope_ms_per_token','?')} ms/tok  binned {dec.get('binned_means_ms',[])}", "derived"),
                ("Accuracy", f"{acc.get('accuracy_score','?')}  {acc.get('matched_facts',[])}", "interpreted"),
            ]
        for m, v, tr in rows:
            self.metrics_tree.insert("", "end", values=(m, str(v), tr))

        # also update summary_text header if beginner: show plain summary
        if view == "beginner" and hasattr(self, "_manifest_cache"):
            # we could inject beginner_takeaway at top of summary? For now metrics tree is the view toggle.
            pass

    def _open_image(self, path: pathlib.Path):
        try:
            if sys.platform == "win32":
                os.startfile(str(path))  # type: ignore
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            messagebox.showinfo("Chart", f"{path}\n\n{e}")

    # ---------- exports ----------
    def _export_pdf(self):
        if not self.current_run_dir or not self.current_run_dir.exists():
            messagebox.showwarning("No run", "Load a run first (History → Open selected or run a new eval).")
            return
        default = f"{self.current_run_dir.name}.pdf"
        out = filedialog.asksaveasfilename(title="Export PDF", initialfile=default, defaultextension=".pdf", filetypes=[("PDF","*.pdf"),("All","*.*")])
        if not out:
            return
        try:
            self.footer_var.set("Exporting PDF …")
            self.update_idletasks()
            export_pdf(self.current_run_dir, pathlib.Path(out))
            messagebox.showinfo("PDF exported", f"Saved to:\n{out}")
            self.footer_var.set(f"PDF saved: {out}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("PDF export failed", str(e))
            self.footer_var.set("PDF export failed")

    def _export_docx(self):
        if not self.current_run_dir or not self.current_run_dir.exists():
            messagebox.showwarning("No run", "Load a run first.")
            return
        default = f"{self.current_run_dir.name}.docx"
        out = filedialog.asksaveasfilename(title="Export Word", initialfile=default, defaultextension=".docx", filetypes=[("Word","*.docx"),("All","*.*")])
        if not out:
            return
        try:
            self.footer_var.set("Exporting DOCX …")
            self.update_idletasks()
            export_docx(self.current_run_dir, pathlib.Path(out))
            messagebox.showinfo("DOCX exported", f"Saved to:\n{out}")
            self.footer_var.set(f"DOCX saved: {out}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("DOCX export failed", str(e))
            self.footer_var.set("DOCX export failed")

    # ---------- history ----------
    def _refresh_history(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        runs = self.storage.list_runs()
        for run_dir, meta in runs:
            mid = meta.get("model_id") or meta.get("manifest",{}).get("model_id") or run_dir.name
            # shorten
            short_mid = mid.split("/")[-1] if "/" in str(mid) else str(mid)[:38]
            date = meta.get("created_utc") or meta.get("manifest",{}).get("created_utc") or "?"
            if "T" in str(date):
                date = str(date).replace("T"," ").replace("Z","").replace("-", "/")[:19]
            preset = meta.get("preset") or meta.get("manifest",{}).get("preset") or "?"
            # tok/s from summary or measurements
            tps = "?"
            try:
                sj = run_dir / "summary.json"
                if sj.exists():
                    j = json.loads(sj.read_text(encoding="utf-8"))
                    tps = str(j.get("performance",{}).get("decode_tok_per_s","?"))
                elif (run_dir / "raw" / "measurements.json").exists():
                    j = json.loads((run_dir / "raw" / "measurements.json").read_text(encoding="utf-8"))
                    tps = str(j.get("decode_tok_per_s","?"))
            except Exception:
                pass
            # tag iid is run_dir string
            self.history_tree.insert("", "end", iid=str(run_dir), values=(short_mid, date, preset, tps))
        # update detail placeholder if no runs
        if not runs:
            self.history_detail.configure(state="normal")
            self.history_detail.delete("1.0","end")
            self.history_detail.insert("1.0", "No past evaluations yet.\n\nRun your first evaluation from Setup → Run. It will appear here as runs/<model>__<date> and stay saved for comparison.")
            self.history_detail.configure(state="disabled")

    def _on_history_select(self, event):
        sel = self.history_tree.selection()
        if not sel:
            return
        p = pathlib.Path(sel[0])
        self.history_selection = p
        # show details
        detail = ""
        try:
            if (p / "manifest.json").exists():
                man = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
                detail += f"Model: {man.get('model_id')}\nRun ID: {man.get('run_id')}\nDate: {man.get('created_utc')}  Preset: {man.get('preset')}  Runtime: {man.get('runtime_type')}\nFolder: {p.name}\n\n"
            if (p / "overview.md").exists():
                txt = (p / "overview.md").read_text(encoding="utf-8")
                detail += txt[:3500]
                if len(txt) > 3500:
                    detail += "\n\n… [overview truncated, open file to see full]\n"
            elif (p / "summary.json").exists():
                sj = json.loads((p / "summary.json").read_text(encoding="utf-8"))
                detail += json.dumps(sj, indent=2)[:3500]
            else:
                detail += "(No overview or summary found in this run folder.)"
        except Exception as e:
            detail += f"\n[Error reading run: {e}]"
        self.history_detail.configure(state="normal")
        self.history_detail.delete("1.0","end")
        self.history_detail.insert("1.0", detail)
        self.history_detail.configure(state="disabled")

    def _open_history_selection(self):
        sel = self.history_tree.selection()
        if not sel:
            messagebox.showinfo("History", "Select a run first.")
            return
        p = pathlib.Path(sel[0])
        if not p.exists():
            messagebox.showerror("Not found", f"{p} no longer exists.")
            self._refresh_history()
            return
        self._load_results(p)
        self.nb.select(self.tab_results)

    def _delete_history_selection(self):
        sel = self.history_tree.selection()
        if not sel:
            messagebox.showinfo("History", "Select a run to delete.")
            return
        p = pathlib.Path(sel[0])
        if not messagebox.askokcancel("Delete run", f"Delete this evaluation?\n\n{p.name}\n\nThis will remove the folder:\n{p}\n\nThis cannot be undone."):
            return
        try:
            import shutil
            shutil.rmtree(str(p))
            messagebox.showinfo("Deleted", f"Removed {p.name}")
            self._refresh_history()
            # if current run was deleted, clear
            if self.current_run_dir and str(self.current_run_dir) == str(p):
                self.current_run_dir = None
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))

    def _export_from_history(self, kind: str):
        sel = self.history_tree.selection()
        if not sel:
            messagebox.showinfo("History", "Select a run to export.")
            return
        p = pathlib.Path(sel[0])
        if kind == "pdf":
            default = f"{p.name}.pdf"
            out = filedialog.asksaveasfilename(title="Export PDF", initialfile=default, defaultextension=".pdf", filetypes=[("PDF","*.pdf")])
            if not out:
                return
            try:
                export_pdf(p, pathlib.Path(out))
                messagebox.showinfo("Exported", f"PDF saved:\n{out}")
            except Exception as e:
                messagebox.showerror("Export failed", str(e))
        else:
            default = f"{p.name}.docx"
            out = filedialog.asksaveasfilename(title="Export DOCX", initialfile=default, defaultextension=".docx", filetypes=[("Word","*.docx")])
            if not out:
                return
            try:
                export_docx(p, pathlib.Path(out))
                messagebox.showinfo("Exported", f"DOCX saved:\n{out}")
            except Exception as e:
                messagebox.showerror("Export failed", str(e))

if __name__ == "__main__":
    # Windows DPI awareness
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = SLMEvalApp()
    app.mainloop()
