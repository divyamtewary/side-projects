from __future__ import annotations
import time
import threading
import pathlib
import json
from datetime import datetime, timezone
from typing import Callable, Dict, Any, List, Optional

from ..domain.models import SCHEMA_VERSION
from ..adapters.storage import RunStorage, atomic_write_json, atomic_write_text
from ..adapters.hardware import capture_environment
from ..core.experiments.performance import run_performance
from ..core.experiments.context_scaling import run_context_scaling
from ..core.experiments.decode_position import run_decode_position
from ..core.experiments.accuracy import run_accuracy

SECTIONS = [
    ("environment", "Environment Snapshot"),
    ("model_inspection", "Model Inspection"),
    ("performance", "Baseline Performance"),
    ("context_scaling", "Context Scaling (H1)"),
    ("decode_position", "Decode Position (H2)"),
    ("accuracy", "Accuracy Grounding Probe"),
    ("reporting", "Report & Charts"),
]

class Orchestrator:
    """
    Sequential runner, emits progress events via callback.
    callback(event: dict) where event = {"type": "section_start"|"section_done"|"log"|"progress", "section":..., "message":..., "pct":...}
    Supports cancellation via threading.Event.
    """
    def __init__(self, storage: RunStorage, runtime, model_id: str, preset: str = "quick", runtime_type: str = "fake", local_path: str | None = None):
        self.storage = storage
        self.runtime = runtime
        self.model_id = model_id
        self.preset = preset
        self.runtime_type = runtime_type
        self.local_path = local_path
        self.cancel_event = threading.Event()
        self.run_dir: pathlib.Path | None = None
        self.run_id: str | None = None
        self.manifest: Dict[str, Any] | None = None

    def cancel(self):
        self.cancel_event.set()

    def _emit(self, cb, event):
        if cb:
            try:
                cb(event)
            except Exception:
                pass
        # also persist
        if self.run_dir:
            try:
                self.storage.append_event(self.run_dir, event)
            except Exception:
                pass

    def run(self, progress_cb: Callable = None) -> pathlib.Path:
        # create run dir
        run_dir, run_id, ts = self.storage.create_run_dir(self.model_id, self.preset, self.runtime_type)
        self.run_dir = run_dir
        self.run_id = run_id
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "run_name": run_dir.name,
            "created_utc": ts,
            "preset": self.preset,
            "runtime_type": self.runtime_type,
            "model_id": self.model_id,
            "model_source": self.runtime_type,
            "sections": [s[0] for s in SECTIONS],
            "config_sha256": run_id.split("_")[-1],
        }
        self.manifest = manifest
        atomic_write_json(run_dir / "manifest.json", manifest)
        self._emit(progress_cb, {"type": "run_start", "run_id": run_id, "run_dir": str(run_dir), "ts": ts, "pct": 0})

        # helper log emitter
        def log(msg):
            self._emit(progress_cb, {"type": "log", "message": msg})

        # Track intermediate results for summary
        env_data = None
        model_info = None
        perf = None
        ctx = None
        dec = None
        acc = None

        total_steps = len(SECTIONS)

        try:
            for idx, (key, title) in enumerate(SECTIONS):
                if self.cancel_event.is_set():
                    self._emit(progress_cb, {"type": "cancelled", "message": "Cancelled by user"})
                    break
                self._emit(progress_cb, {"type": "section_start", "section": key, "title": title, "pct": int(idx/total_steps*100)})
                log(f"▶ Starting: {title}")

                if key == "environment":
                    env_data = capture_environment()
                    atomic_write_json(run_dir / "environment.json", env_data)
                    # also raw
                    atomic_write_json(run_dir / "raw" / "environment.json", env_data)

                elif key == "model_inspection":
                    # runtime.load already done before orchestrator? If not, do.
                    try:
                        if not getattr(self.runtime, "is_loaded", getattr(self.runtime, "loaded", False)):
                            log(f"Loading model: {self.model_id} (runtime={self.runtime_type}) …")
                            t0 = time.time()
                            info, load_t = self.runtime.load()
                            model_info = info
                            # ensure load_time stored
                            if perf is None:
                                # we will store separately; just keep
                                pass
                        else:
                            model_info = getattr(self.runtime, "model_info", {})
                        atomic_write_json(run_dir / "model.json", model_info)
                        atomic_write_json(run_dir / "raw" / "model.json", model_info)
                    except Exception as e:
                        # fallback: use whatever model_info we have or minimal
                        log(f"⚠ Model inspection failed: {e}")
                        if model_info is None:
                            model_info = getattr(self.runtime, "model_info", {"model_id": self.model_id, "source": self.runtime_type, "error": str(e)})
                        atomic_write_json(run_dir / "model.json", model_info)
                        # don't fail whole run; continue but mark section
                        self._emit(progress_cb, {"type": "log", "message": f"Model inspection error – continuing with fake fallback: {e}"})

                elif key == "performance":
                    # ensure model is loaded; if inspection failed, fake already
                    try:
                        perf = run_performance(self.runtime, self.preset, on_log=log)
                        atomic_write_json(run_dir / "raw" / "measurements.json", perf)
                    except Exception as e:
                        log(f"Performance experiment failed: {e}")
                        perf = {"error": str(e), "schema_version": SCHEMA_VERSION}

                elif key == "context_scaling":
                    try:
                        ctx = run_context_scaling(self.runtime, self.preset, on_log=log)
                        atomic_write_json(run_dir / "raw" / "context_scaling.json", ctx)
                    except Exception as e:
                        log(f"Context scaling failed: {e}")
                        ctx = {"error": str(e)}

                elif key == "decode_position":
                    try:
                        dec = run_decode_position(self.runtime, self.preset, on_log=log)
                        atomic_write_json(run_dir / "raw" / "decode_position.json", dec)
                    except Exception as e:
                        log(f"Decode position failed: {e}")
                        dec = {"error": str(e)}

                elif key == "accuracy":
                    try:
                        acc = run_accuracy(self.runtime, self.preset, on_log=log)
                        atomic_write_json(run_dir / "raw" / "accuracy.json", acc)
                    except Exception as e:
                        log(f"Accuracy probe failed: {e}")
                        acc = {"error": str(e)}

                elif key == "reporting":
                    # charts + markdown + summary.json
                    log("Generating charts and reports…")
                    try:
                        from ..reports.charts import generate_all_charts
                        from ..reports.markdown import build_overview_markdown, build_summary_json

                        chart_paths = generate_all_charts(run_dir, perf, ctx, dec, acc)
                        log(f"Charts written: {len(chart_paths)} PNGs")
                        overview_md = build_overview_markdown(manifest, env_data, model_info, perf, ctx, dec, acc)
                        atomic_write_text(run_dir / "overview.md", overview_md)
                        # summary.json is the unified summary
                        summary = build_summary_json(manifest, env_data, model_info, perf, ctx, dec, acc)
                        atomic_write_json(run_dir / "summary.json", summary)
                        # also hypotheses folder
                        (run_dir / "hypotheses").mkdir(exist_ok=True)
                        # H001 file
                        try:
                            h1_md = "# H1 – Context Scaling (Descriptive Only)\n\n" + (ctx.get("conclusion","") if ctx else "") + "\n\n## Points\n\n" + "\n".join([f"- {p['label']}: {p['num_input_tokens']} tokens → {p['tok_per_sec']} tok/s" for p in (ctx.get("points",[]) if ctx else [])])
                            atomic_write_text(run_dir / "hypotheses" / "H001.md", h1_md)
                        except Exception:
                            pass
                    except Exception as e:
                        log(f"Reporting failed: {e}")
                        import traceback, pathlib as _p
                        log(traceback.format_exc())

                self._emit(progress_cb, {"type": "section_done", "section": key, "title": title, "pct": int((idx+1)/total_steps*100)})
                log(f"✓ Completed: {title}")
                # small pause for UI responsiveness
                time.sleep(0.15)

            self._emit(progress_cb, {"type": "run_done", "run_dir": str(run_dir), "run_id": run_id, "pct": 100})
            return run_dir

        except Exception as e:
            self._emit(progress_cb, {"type": "error", "message": str(e)})
            raise
