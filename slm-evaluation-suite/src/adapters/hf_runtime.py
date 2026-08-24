from __future__ import annotations
import os
import time
import pathlib
from typing import Dict, Any, Callable, List

# Hugging Face hub helpers – import lazily so UI works without them

def check_hf_login() -> Dict[str, Any]:
    try:
        from huggingface_hub import HfApi, whoami
        # try whoami; may raise if not logged in
        try:
            info = whoami()
            return {"logged_in": True, "user": info.get("name") or info.get("fullname") or "unknown", "info": info}
        except Exception as e:
            # fallback: token exists but not valid?
            api = HfApi()
            token = api.token  # may be None
            if token:
                return {"logged_in": True, "user": "token-present", "info": {}}
            return {"logged_in": False, "error": str(e)}
    except Exception as e:
        return {"logged_in": False, "error": str(e)}

def list_user_models(limit: int = 20):
    # not trivial without search; we provide popular SLM list instead
    return [
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "google/gemma-3-270m-it",
        "google/gemma-3-1b-it",
        "microsoft/Phi-3-mini-4k-instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]

def download_hf_model(model_id: str, progress_cb: Callable = None) -> pathlib.Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub not installed: {e}")
    # use cache dir
    local_dir = snapshot_download(repo_id=model_id, local_dir_use_symlinks=False)
    if progress_cb:
        progress_cb(local_dir)
    return pathlib.Path(local_dir)

class HFTransformersRuntime:
    """
    Runtime that uses transformers AutoModelForCausalLM.
    Falls back to Fake behaviour if torch/transformers missing.
    """
    def __init__(self, model_id: str, local_path: str | None = None):
        self.model_id = model_id
        self.local_path = local_path
        self.model = None
        self.tokenizer = None
        self.load_time_s = 0.0
        self.is_loaded = False
        self.model_info: Dict[str, Any] = {}

    def load(self, trust_remote_code: bool = False) -> tuple[Dict[str, Any], float]:
        import pathlib as _p
        target = self.local_path or self.model_id
        # if local path given, validate
        if self.local_path:
            pp = _p.Path(self.local_path)
            if not pp.exists():
                raise FileNotFoundError(f"Local path not found: {pp}")
            target = str(pp)
        t0 = time.time()
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            torch.set_num_threads(os.cpu_count() or 4)
            tok = AutoTokenizer.from_pretrained(target, trust_remote_code=trust_remote_code)
            # use float32 for CPU compatibility; model may be large but we try
            model = AutoModelForCausalLM.from_pretrained(
                target,
                trust_remote_code=trust_remote_code,
                dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            model.eval()
            t1 = time.time()
            cfg = model.config
            n_params = sum(p.numel() for p in model.parameters())
            self.tokenizer = tok
            self.model = model
            self.load_time_s = t1 - t0
            self.is_loaded = True
            self.model_info = {
                "schema_version": "1.0",
                "model_id": self.model_id,
                "source": "hf_hub" if not self.local_path else "local_dir",
                "local_path": self.local_path,
                "architecture": getattr(cfg, "architectures", [type(model).__name__])[0] if getattr(cfg, "architectures", None) else type(model).__name__,
                "num_parameters": int(n_params),
                "num_layers": getattr(cfg, "num_hidden_layers", None),
                "num_heads": getattr(cfg, "num_attention_heads", None),
                "num_kv_heads": getattr(cfg, "num_key_value_heads", None),
                "hidden_size": getattr(cfg, "hidden_size", None),
                "vocab_size": getattr(cfg, "vocab_size", None),
                "max_position_embeddings": getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None),
                "torch_dtype": str(next(model.parameters()).dtype),
                "quantization": None,
                "config_raw": {k: str(v) if not isinstance(v, (str,int,float,bool, list, dict, type(None))) else v for k,v in cfg.to_dict().items()} if hasattr(cfg, "to_dict") else {},
                "tokenizer_vocab_size": len(tok) if hasattr(tok, "__len__") else getattr(tok, "vocab_size", None),
                "files": [],
                "truth_level": "measured",
            }
            return self.model_info, self.load_time_s
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_id}: {e}")

    # Real generation with streaming timestamps – simplified vs notebook's TextIteratorStreamer
    def generate(self, prompt: str, max_new_tokens: int = 128, do_sample: bool = False, temperature: float = 1.0) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        import time, threading, torch, numpy as np
        from transformers import TextIteratorStreamer
        import sys
        inputs = self.tokenizer(prompt, return_tensors="pt")
        num_input_tokens = int(inputs["input_ids"].shape[1])
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature if do_sample else 1.0, use_cache=True, streamer=streamer, pad_token_id=self.tokenizer.eos_token_id)
        t_start = time.time()
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        generated = ""
        timestamps = []
        first = None
        cnt = 0
        for chunk in streamer:
            now = time.time()
            if first is None:
                first = now
            cnt += 1
            timestamps.append((cnt, now))
            generated += chunk
        thread.join()
        t_end = time.time()
        total = t_end - t_start
        prefill = (first - t_start) if first else total
        decode = (t_end - first) if first else 0.0
        # accurate output tokens
        try:
            out_ids = self.tokenizer(generated, add_special_tokens=False)["input_ids"]
            num_out = len(out_ids)
        except Exception:
            num_out = cnt
        tps_decode = num_out / decode if decode>0 else 0
        tps_overall = num_out / total if total>0 else 0
        prefill_tps = num_input_tokens / prefill if prefill>0 else 0
        # intervals
        intervals = []
        if len(timestamps) > 1:
            intervals = [(timestamps[i][1]-timestamps[i-1][1])*1000 for i in range(1,len(timestamps))]
        import numpy as np
        arr = np.array(intervals) if intervals else np.array([0])
        return {
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_out,
            "prefill_time_s": float(prefill),
            "decode_time_s": float(decode),
            "total_time_s": float(total),
            "prefill_tokens_per_second": float(prefill_tps),
            "tokens_per_second_decode": float(tps_decode),
            "tokens_per_second_overall": float(tps_overall),
            "token_timestamps": timestamps,
            "intervals_ms": intervals,
            "mean_ms": float(arr.mean()) if len(arr) else 0,
            "median_ms": float(np.median(arr)) if len(arr) else 0,
            "p95_ms": float(np.percentile(arr,95)) if len(arr) else 0,
            "p99_ms": float(np.percentile(arr,99)) if len(arr) else 0,
            "min_ms": float(arr.min()) if len(arr) else 0,
            "max_ms": float(arr.max()) if len(arr) else 0,
            "response": generated,
        }
