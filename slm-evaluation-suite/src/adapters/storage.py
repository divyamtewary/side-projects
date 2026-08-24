from __future__ import annotations
import json
import hashlib
import pathlib
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Dict, Any

def make_run_dir_name(model_id: str, timestamp_utc: str) -> str:
    """
    Filesystem-safe name: sanitized model id + date.
    Example: HuggingFaceTB_SmolLM2-135M-Instruct__2026-08-24T14-30-22Z
    """
    safe = model_id.replace("/", "__").replace("\\", "__").replace(":", "_")
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".", "__") else "_" for c in safe)
    # keep readable length
    if len(safe) > 60:
        safe = safe[:60]
    # timestamp to file-safe
    ts_safe = timestamp_utc.replace(":", "-")
    return f"{safe}__{ts_safe}"

def atomic_write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mktemp(dir=str(path.parent)))
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # atomic replace
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

def atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mktemp(dir=str(path.parent)))
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

class RunStorage:
    """
    Manages run directory layout:
    runs/<run_name>/
      manifest.json
      environment.json
      model.json
      measurements.json
      summary.json
      overview.md
      events.jsonl
      charts/*.png
    Deterministic run_id = UTC timestamp + first 8 of config sha256 over canonical sorted-key JSON.
    """
    def __init__(self, base_dir: pathlib.Path):
        self.base_dir = pathlib.Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, model_id: str, preset: str, runtime_type: str, timestamp_utc: str = None) -> tuple[pathlib.Path, str, str]:
        if timestamp_utc is None:
            timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        config = {"model_id": model_id, "preset": preset, "runtime_type": runtime_type, "timestamp": timestamp_utc}
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
        ts_compact = timestamp_utc.replace("-", "").replace(":", "").replace("T", "T").replace("Z", "Z")
        run_id = f"{ts_compact}_{sha}"
        run_name = make_run_dir_name(model_id, timestamp_utc)
        # ensure uniqueness if collision
        run_dir = self.base_dir / run_name
        counter = 1
        while run_dir.exists():
            run_dir = self.base_dir / f"{run_name}_{counter}"
            counter += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "charts").mkdir(exist_ok=True)
        (run_dir / "raw").mkdir(exist_ok=True)
        return run_dir, run_id, timestamp_utc

    def list_runs(self):
        if not self.base_dir.exists():
            return []
        runs = []
        for p in self.base_dir.iterdir():
            if p.is_dir():
                manifest = p / "manifest.json"
                if manifest.exists():
                    try:
                        data = json.loads(manifest.read_text(encoding="utf-8"))
                        runs.append((p, data))
                    except Exception:
                        runs.append((p, {"run_id": p.name, "run_name": p.name}))
                else:
                    runs.append((p, {"run_id": p.name, "run_name": p.name}))
        # sort by created_utc desc if available
        def key(x):
            try:
                return x[1].get("created_utc", "")
            except Exception:
                return ""
        runs.sort(key=lambda x: key(x), reverse=True)
        return runs

    def append_event(self, run_dir: pathlib.Path, event: Dict[str, Any]):
        events_path = run_dir / "events.jsonl"
        line = json.dumps(event, ensure_ascii=False)
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
