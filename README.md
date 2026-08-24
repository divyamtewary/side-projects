![side projects](docs/img/banner.svg)

<p align="center">
  <img alt="focus: local AI" src="https://img.shields.io/badge/focus-local%20AI-090A16?style=flat-square&labelColor=090A16&color=5EEAD4">
  <img alt="format: experiments and tools" src="https://img.shields.io/badge/format-experiments%20%2B%20tools-090A16?style=flat-square&labelColor=090A16&color=FF6B9A">
  <img alt="priority: reproducibility" src="https://img.shields.io/badge/priority-reproducibility-090A16?style=flat-square&labelColor=090A16&color=FFB86B">
</p>

# side-projects

A collection of focused experiments for understanding, evaluating, and building with
small language models on ordinary hardware.

These projects are deliberately narrow. Each one starts with a concrete question, builds
the smallest useful instrument, and documents what the evidence does and does not show.

---

## Project map

| Project | Status | Question or role |
|---|---|---|
| [`neural-observatory`](neural-observatory/) | Available | What is happening inside a small transformer while it generates? |
| [`slm-evaluation-suite`](slm-evaluation-suite/) | **Available** | **How can local model performance be measured without pretending one number explains quality?** — Tkinter self-hosted evaluator (local dir or HF Hub, live progress, 5 charts, PDF/DOCX) |
| [`image-craft-lab`](image-craft-lab/) | **Available** | **Play. One photo, three transformations.** The most shareable lab — upload one photo, get ASCII, Pixel and Palette studios with live thumbnails on *that* photo + Full Control. Pure Pillow, no model, no API. |
| `vocabulary-geometry-lab` | Planned | How does a model move through output-vocabulary space before a prediction stabilises? |

---

## Neural Observatory

[`neural-observatory`](neural-observatory/) is an interactive CPU visualiser for a small
language model. It captures layer-wise activations and generation telemetry and exposes
them through views such as:

- Logit lens predictions across layers.
- Attention-head entropy and focus.
- MLP sparsity and residual-stream norms.
- Token probability and entropy timelines.
- KV-cache and memory telemetry.
- A data-driven three-dimensional transformer stack.

Its purpose is observability, not speed. For a fast local runtime, see
[`llm-local-harness`](https://github.com/divyamtewary/development/tree/main/llm-local-harness).

---

## SLM Evaluation Suite

[`slm-evaluation-suite`](slm-evaluation-suite/) is a **Tkinter desktop app** for measuring small language models on your own machine. Pick a **local directory** or a **Hugging Face model** (after `huggingface-cli login`), watch the 6-section pipeline (environment → model inspection → baseline → H1 context scaling → H2 decode position → grounding probe) run with live progress bars, then browse the **evidence cards**, 5 deterministic charts and beginner/expert summaries. Every run is saved as `runs/<model>__<date>` and exportable to **PDF/DOCX**. See [`slm-evaluation-suite/README.md`](slm-evaluation-suite/README.md) for screenshots and a step-by-step usage guide.

## Image Craft Lab

[`image-craft-lab`](image-craft-lab/) is a **Streamlit lab for one photo, three transformations**. Upload one photo — every preset shows a **live thumbnail of *that* photo** — then switch between:

- **ASCII** — `6` presets + Full Control (`charset`/`cols`/`color`/`contrast`/`brightness`/`invert`), `CHAR_ASPECT=0.55` luminance → `charset[int(p/255*(n-1))]` with `BILINEAR` color sampling; exports `.txt`/`.html`/`.png`.
- **PIXEL** — `4` presets + Full Control (`grid`/`colors`/`dither`/`scale`), `NEAREST` downscale → `MEDIANCUT` quantize → `NEAREST` upscale; exports `PNG` + `CSS` `box-shadow` (single-`div` shareable sprite).
- **PALETTE** — `4` modes (`vibrant`/`muted`/`pastel`/`deep` via `HSV`) + `n_colors` `3–8`, `KMeans` or `MEDIANCUT` fallback, sorted dominant, `800×120` gradient + WCAG contrast badges; exports `CSS` (`:root` + Tailwind) + gradient `PNG`.

No model, no API, no GPU — `Pillow` is the only hard dep (`numpy` + optional `scikit-learn`). Beginners get results in 5 seconds, intermediates get full control. See [`image-craft-lab/README.md`](image-craft-lab/README.md) for screenshots and a step-by-step usage guide.

## What comes next

`vocabulary-geometry-lab` will connect the observatory's
layer-wise traces with the information-geometric ideas developed in the
side-projects repository itself.

---

## Working principles

- Prefer local, inspectable tools over opaque hosted demos.
- Use small models so experiments can be repeated.
- Keep raw measurements separate from interpretations.
- Show failures and regressions, not only successful outputs.
- Build an interactive view only when it reveals structure clearly.
- State limitations beside the result they qualify.

---

## Related repositories

- [`research`](https://github.com/divyamtewary/research) - mathematical notes and research visualisers.
- [`development`](https://github.com/divyamtewary/development) - runtimes, tools, and agent systems.
- [`blog`](https://github.com/divyamtewary/blog) - public writing about the work and the lessons behind it.

---

## Author

Divyam Tewary is building practical AI systems, exploring how intelligence works, and
turning difficult ideas into interactive tools.
