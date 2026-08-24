from enum import Enum

class Preset(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"

class RuntimeType(str, Enum):
    LOCAL = "local"
    HF_HUB = "hf_hub"
    FAKE = "fake"

class ModelSource(str, Enum):
    LOCAL_DIR = "local_dir"
    HF_HUB = "hf_hub"
    FAKE = "fake"

class SectionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"

class TruthLevel(str, Enum):
    """Evidence taxonomy: see PROJECT_BRIEF MEASURED/DERIVED/INTERPRETED."""
    MEASURED = "measured"      # direct instrument reading
    DERIVED = "derived"        # computed from measured, formula shown
    INTERPRETED = "interpreted"  # human-readable summary, descriptive only
