from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
import hashlib
import json

SCHEMA_VERSION = "1.0"

class EnvironmentSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = SCHEMA_VERSION
    timestamp_utc: str
    platform: str
    platform_version: str
    python_version: str
    cpu_count_logical: int
    cpu_count_physical: Optional[int] = None
    ram_total_gb: float
    ram_available_gb: float
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None
    psutil_version: Optional[str] = None
    cuda_available: bool = False
    # truth level
    truth_level: str = "measured"

class ModelInfo(BaseModel):
    schema_version: str = SCHEMA_VERSION
    model_id: str
    source: str  # local_dir, hf_hub, fake
    local_path: Optional[str] = None
    architecture: Optional[str] = None
    num_parameters: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    torch_dtype: Optional[str] = None
    quantization: Optional[str] = None
    config_raw: Dict[str, Any] = Field(default_factory=dict)
    tokenizer_vocab_size: Optional[int] = None
    files: List[Dict[str, Any]] = Field(default_factory=list)
    truth_level: str = "measured"

class PerformanceResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    # MEASURED
    num_input_tokens: int
    num_output_tokens: int
    load_time_s: float
    prefill_time_s: float  # TTFT approx via streaming
    decode_time_s: float
    total_time_s: float
    # DERIVED
    prefill_tok_per_s: float
    decode_tok_per_s: float
    overall_tok_per_s: float
    inter_token_latency_ms_mean: float
    inter_token_latency_ms_median: float
    inter_token_latency_ms_p95: float
    inter_token_latency_ms_p99: Optional[float] = None
    inter_token_latency_ms_min: float
    inter_token_latency_ms_max: float
    peak_rss_mb: Optional[float] = None
    # raw timestamps for re-analysis
    token_timestamps: List[Any] = Field(default_factory=list)
    truth_level: str = "measured/derived"
    note: str = "TTFT is streaming-approximation, distinct from true prefill. Fixed ordering is a stated limitation."

class ContextScalingPoint(BaseModel):
    label: str
    num_input_tokens: int
    context_chars: int
    num_output_tokens: int
    total_time_s: float
    tok_per_sec: float
    ms_per_token: float
    response_preview: str = ""

class ContextScalingResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    points: List[ContextScalingPoint]
    # derived fit params
    fit_k: Optional[float] = None
    fit_alpha: Optional[float] = None
    pearson_r: Optional[float] = None
    conclusion: str = "Descriptive only: no pass/fail claim about model quality."
    truth_level: str = "measured/derived/interpreted"

class DecodePositionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    input_tokens: int
    output_tokens: int
    latencies_ms: List[float]
    binned_means_ms: List[float] = Field(default_factory=list)
    slope_ms_per_token: Optional[float] = None
    conclusion: str = ""
    truth_level: str = "measured/derived/interpreted"

class AccuracyResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    prompt: str
    expected_facts: List[str]
    response: str
    matched_facts: List[str] = Field(default_factory=list)
    accuracy_score: float = 0.0
    audit_details: List[Dict[str, Any]] = Field(default_factory=list)
    truth_level: str = "measured/interpreted"

class EvaluationManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    run_name: str  # modelname + date for filesystem
    created_utc: str
    preset: str
    runtime_type: str
    model_id: str
    model_source: str
    sections: List[str]
    config_sha256: str

    @staticmethod
    def make_run_id(config: dict, timestamp_utc: str) -> str:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
        # timestamp like 20260824T143022Z + sha
        ts = timestamp_utc.replace(":", "").replace("-", "").replace(".", "")
        # ensure format: YYYYMMDDThhmmssZ_<sha>
        # if timestamp has colons, normalize
        return f"{ts}_{sha}"

class EvaluationSummary(BaseModel):
    schema_version: str = SCHEMA_VERSION
    manifest: EvaluationManifest
    environment: EnvironmentSnapshot
    model_info: ModelInfo
    performance: Optional[PerformanceResult] = None
    context_scaling: Optional[ContextScalingResult] = None
    decode_position: Optional[DecodePositionResult] = None
    accuracy: Optional[AccuracyResult] = None
    # interpreted overall
    overall_summary: str = ""
    limitations: List[str] = Field(default_factory=list)
    beginner_takeaway: str = ""
    expert_takeaway: str = ""
