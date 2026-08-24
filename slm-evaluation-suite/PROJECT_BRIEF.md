# Project 1 — `slm-evaluation-suite`

**Repo:** `Side Projects/slm-evaluation-suite`
**Session slot:** Day 2, 19:15 → 00:00 (**4h45m, hard stop**)
**AI credit cap:** **$6.00** (+ $3 shared reserve)
**Narrative role:** *Checkpoint 1 — Measure.* "The question I started with."

> 🔴 **This project has been descoped twice and must be descoped once more.**
> The playbook specifies a multi-week build. You have under five hours.
> **Ship a headless `v0.1.0` that passes its own gates.** An honest small release
> beats an unpublished large one, and it is a better blog paragraph.

---

## 1. Resources that already exist locally

### The playbook — `NEXT PROJECTS/slm_evaluation_suite_coding_playbook/`

| File | Use it for |
|---|---|
| `00_MASTER_FIRST_PASS_PROMPT.md` | Repo tree, constraints, output discipline |
| `01_PRODUCT_CONTRACT.md` | Run-package layout, truth model (MEASURED/DERIVED/INTERPRETED), failure semantics |
| `02_ARCHITECTURE_AND_DEPENDENCIES.md` | Dependency direction, the three core protocols, approved deps |
| `03_DOMAIN_SCHEMAS.md` | **Paste this near-verbatim** — enums + every model's fields |
| `04_NOTEBOOK_REUSE_MAP.md` | What to lift from the Gemma notebook and what to correct first |
| `05_EXPERIMENT_CATALOGUE.md` | H001–H004 + B001 contracts |
| `07_TEST_AND_RELEASE_GATES.md` | The gate commands |
| `reference/AGENT_CONTEXT_HEADER.md` | **Prepend to every agent session** |
| `prompts/11_BOUNDED_REPAIR_TEMPLATE.md` | Every failure goes through this |

### The research seed — `NEXT PROJECTS/00_GEMMA3_MODEL_BREAKDOWN.ipynb`

Already-working algorithms to generalise: `stream_generate` (per-token timestamps),
TTFT / decode / total timing, inter-token mean / median / P95, `audit_accuracy`,
the H₁ context-scaling sweep and H₂ decode-position analysis, deterministic matplotlib output.

⚠️ **Its fixtures are contaminated.** `SIGNAL_BLOCK` contains "IndiGo Q3 FY2026", RASK/CASK,
load factor, fleet counts; `NOISE_BLOCKS` are aviation-industry text; prompts say
*"You are an expert aviation analyst."* Your own `04_NOTEBOOK_REUSE_MAP.md` already
forbids this: *"Avoid domain-specific fare examples. Use neutral synthetic passages
labelled as synthetic."* Replace all of it with generated filler.

---

## 2. Scope — `v0.1.0` is headless

### In

| Area | Deliverable |
|---|---|
| `domain/` | Pydantic v2 models + enums per `03_DOMAIN_SCHEMAS.md`. `schema_version="1.0"` everywhere. |
| `adapters/storage/` | Atomic JSON write, JSONL events, run dir, **deterministic run ID** = UTC timestamp + first 8 of config SHA-256 over canonical sorted-key JSON |
| `adapters/hardware/` | `EnvironmentSnapshot` via `psutil` + `platform` + torch/transformers versions |
| `adapters/local_runtime/` | Local **directory only** config inspection (allowlisted fields), `trust_remote_code=False` |
| Runtime | `LocalTransformersRuntime` + `FakeModelRuntime` |
| Experiments | **H001 only.** H002 is stretch. |
| `core/orchestration/` | Sequential runner, progress events, cancellation via `threading.Event` |
| `reports/` | Markdown overview + JSON/`measurements.json` + one PNG chart |
| CLI | `slm-eval inspect` / `run` / `report` (Typer) |
| Gates | `ruff check .` + `pytest -q` |
| Repo | README, LICENSE (MIT), `.gitignore`, one CI workflow |

### 🔴 Out — moved to `v0.2.0` roadmap, stated plainly in the README

- **Tkinter UI** (`06_UI_AND_USER_FLOW.md`, prompt 08) — the single biggest cut
- Hugging Face Hub inspection, gated/private/revision handling — **local dirs only**
- H002, H003, H004, B001
- HTML and CSV reporting
- Resume / partial-run recovery
- Cloud runtime adapter
- `mypy` in the blocking gate — run it, don't let it stop the release
- The **80% coverage gate** → target ~60%, do not measure-and-chase
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue/PR templates
- Windows/Linux launch scripts (CLI is the entry point)

**The README must exactly match what is implemented.** That rule from
`07_TEST_AND_RELEASE_GATES.md` survives every cut — it is the whole credibility of the repo.

---

## 3. H001 — the one experiment

Straight from `05_EXPERIMENT_CATALOGUE.md`:

- Fixed neutral prompt, greedy decoding
- Output budget: quick 64, standard 128
- Warm-up run **excluded** from measurement
- Quick = 1 measured run; standard = 3 measured runs with dispersion reported
- Metrics: load time, input/output tokens, TTFT, decode tok/s, overall tok/s, latency
  percentiles, peak RSS
- **Conclusion rule: descriptive only.** No pass/fail claim about model quality.

Corrections from `04_NOTEBOOK_REUSE_MAP.md` that must land in the code:

- Do **not** assert FP16 is fastest on CPU — use `AUTO` precision selection
- Label streaming-timestamp TTFT as an approximation, distinct from true prefill duration
- Record fixed ordering as a stated limitation

---

## 4. Execution plan — 4h45m, gated

Follow `README_FIRST.md`'s cost strategy: **strongest model once** for the scaffold,
cheap model for everything after. **Run the gate between every slot.** A silent failure
at slot 1 is unfixable by slot 4.

| Clock | Minutes | Prompts | Gate before moving on |
|---|---|---|---|
| 19:15 | 15 | — | Venv, `pyproject.toml`, repo skeleton, `.gitignore` |
| 19:30 | 75 | 01 + 02 combined — **strong model, one pass** | `ruff check .` and `pytest -q` exit 0 |
| 20:45 | 45 | 03 + 04 — inspection + runtimes | `slm-eval inspect --model <local-dir>` works **offline** |
| 21:30 | 15 | — | Break |
| 21:45 | 60 | 05 (H001) + 06 (orchestration) | `slm-eval run --model fake://tiny --runtime fake --preset quick --output ./tmp-runs` writes a complete run dir |
| 22:45 | 45 | 07 — Markdown + JSON + PNG | `slm-eval report --run-dir <path>` regenerates from saved artefacts alone |
| 23:30 | 30 | 09 — docs, CI, **README-matches-reality audit** | CI green |

**23:30 is the hard stop.** Whatever is not done becomes a roadmap bullet.
Truncate scope, never truncate the gate.

### Overrun triage — decide in this order

1. Drop the PNG chart (JSON + Markdown is a complete package)
2. Drop `slm-eval report` as a separate command (generate inline during `run`)
3. Drop `LocalTransformersRuntime`, ship `FakeModelRuntime` only, and say so in the README
4. Drop CI

Do **not** drop: deterministic run IDs, the atomic-write storage layer, or the
MEASURED / DERIVED / INTERPRETED separation. Those three *are* the product.

---

## 5. Agent discipline — the rules that decay after midnight

- Prepend `reference/AGENT_CONTEXT_HEADER.md` to **every** session
- **Never** paste multiple prompts at once
- Every failure → `prompts/11_BOUNDED_REPAIR_TEMPLATE.md`, one defect at a time
- If a repair starts touching unrelated files: kill it, re-scope, retry
- **Never** let an agent redesign architecture mid-implementation
- Approved deps only. No LangChain, no FastAPI, no MLflow.

---

## 6. Definition of done

- [ ] `ruff check .` exits 0
- [ ] `pytest -q` green (~10–15 tests)
- [ ] `slm-eval --help` works
- [ ] Fake quick run produces `manifest.json`, `model.json`, `environment.json`, `configuration.json`, `events.jsonl`, `raw/measurements.json`, `hypotheses/H001.md`, `overview.md`
- [ ] Run ID is reproducible for an identical configuration
- [ ] No secret, no absolute path, no company string in any artefact
- [ ] README lists implemented vs roadmap **accurately**
- [ ] `git grep -i -E "indigo|interglobe|goindigo|aviation|airline|RASK|CASK|divyam"` → **no output**
- [ ] Tagged `slm-eval-v0.1.0`, pushed

---

## 7. Blog seeds — capture as you go

- The original research question, unchanged from `README_FIRST.md`
- Why deterministic templates write the reports and **no LLM does** — a model must not
  be allowed to hallucinate its own evaluation
- MEASURED / DERIVED / INTERPRETED: the discipline that survived every descope
- Evidence cards instead of a universal "model quality" score, and why one number lies
- The things you got wrong in the first notebook: FP16-always-fastest, no warm-up,
  single repetition, TTFT conflated with prefill, correlation read as mechanism
- What descoping under a deadline taught you about release gates
