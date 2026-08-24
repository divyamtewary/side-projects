from __future__ import annotations
import json
import pathlib
from typing import Dict, Any, List

# Allowlisted config fields – never dump entire raw with secrets blindly
ALLOWLIST = {
    "architectures", "model_type", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
    "max_position_embeddings", "intermediate_size", "hidden_activation",
    "hidden_act", "rms_norm_eps", "rope_theta", "torch_dtype",
    "transformers_version", "quantization_config", "sliding_window",
    "sliding_window_pattern", "attention_bias", "tie_word_embeddings",
    "initializer_range", "use_cache", "bos_token_id", "eos_token_id", "pad_token_id"
}

def inspect_local_model(model_dir: str | pathlib.Path) -> Dict[str, Any]:
    p = pathlib.Path(model_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Model directory not found: {p}")
    config_path = p / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {p}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # allowlisted view
    filtered = {k: v for k, v in config.items() if k in ALLOWLIST}
    # files inventory
    files = []
    total_bytes = 0
    for child in p.iterdir():
        if child.is_file():
            try:
                sz = child.stat().st_size
            except Exception:
                sz = 0
            total_bytes += sz
            files.append({"name": child.name, "size_bytes": sz, "size_mb": round(sz/(1024*1024),2)})

    # tokenizer vocab size if available
    tok_vocab = None
    for cand in ["tokenizer.json", "tokenizer_config.json"]:
        try:
            tp = p / cand
            if tp.exists():
                data = json.loads(tp.read_text(encoding="utf-8"))
                # tokenizer.json has no vocab size directly, but tokenizer_config may contain
                if isinstance(data, dict) and "vocab_size" in data:
                    tok_vocab = data.get("vocab_size")
        except Exception:
            pass

    # derive num_parameters if safetensors exists? approximate via file size if needed
    # We'll not try to load weights here; just report config
    model_info = {
        "schema_version": "1.0",
        "model_id": str(p.resolve()),
        "source": "local_dir",
        "local_path": str(p.resolve()),
        "architecture": (config.get("architectures") or [config.get("model_type")])[0] if (config.get("architectures") or config.get("model_type")) else None,
        "num_parameters": None,  # filled if runtime loads model
        "num_layers": config.get("num_hidden_layers"),
        "num_heads": config.get("num_attention_heads"),
        "num_kv_heads": config.get("num_key_value_heads"),
        "hidden_size": config.get("hidden_size"),
        "vocab_size": config.get("vocab_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "torch_dtype": str(config.get("torch_dtype")) if config.get("torch_dtype") else None,
        "quantization": str(config.get("quantization_config")) if config.get("quantization_config") else None,
        "config_raw": filtered,
        "tokenizer_vocab_size": tok_vocab or config.get("vocab_size"),
        "files": sorted(files, key=lambda x: x["name"]),
        "truth_level": "measured",
    }
    return model_info
