"""Experiment configuration: defaults, YAML loading, overrides, validation.

A config is a nested dict. Every run stores the fully-resolved config, so a
result can always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from faults import spec as fault_spec

DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "adhoc", "seed": 0},
    "workload": {
        "model": "resnet18",            # resnet18 | resnet50 | transformer_small | cnn_tiny
        "dataset": "synthetic",         # synthetic (device-resident) | synthetic_host | cifar10
        "batch_size": 128,              # per GPU / per rank
        "grad_accum_steps": 1,
        "precision": "fp32",            # fp32 | amp (fp16 autocast + GradScaler; CUDA only)
        "optimizer": "sgd",             # sgd | adamw
        "lr": 0.05,
        "seq_len": 256,                 # language-model workloads only
        "num_workers": 2,               # host data loaders only
        "prefetch_factor": 2,           # host data loaders with workers (PyTorch default 2)
        "pin_memory": True,
        "data_root": "data",
        "data_download": False,         # allow local rank 0 to download inside the job (prefer prepare_data.py)
        "synthetic_pool": 16,           # distinct pre-generated device batches cycled through
        "cudnn_benchmark": True,
    },
    "distributed": {
        "backend": "auto",              # auto | nccl | gloo
        "bucket_cap_mb": 25,            # DDP default
        "gradient_as_bucket_view": False,
        "broadcast_buffers": True,      # DDP default
    },
    "measurement": {
        "warmup_steps": 50,
        "measured_steps": 200,
        "repeats": 1,                   # blocks per mode within one launch
        "modes": ["ddp", "nosync"],     # nosync = DDP.no_sync(): communication suppressed
        "gpu_util": True,
        "comm_hook": True,              # False: no timing hook (instrumentation-overhead control)
        "nccl_log": True,               # capture NCCL INIT logs (runtime version, transport)
        "gpu_state": True,              # nvidia-smi clocks/temp/power snapshot around each block
        "profile": {"enabled": False, "wait": 1, "warmup": 3, "active": 10},
        "fingerprint": False,           # per-tensor parameter norms at the end (optimization correctness)
    },
    "fault": dict(fault_spec.DEFAULT_FAULT),  # mechanism "none" = healthy; see faults/spec.py
}

_CHOICES = {
    ("workload", "model"): {"resnet18", "resnet50", "transformer_small", "cnn_tiny"},
    ("workload", "dataset"): {"synthetic", "synthetic_host", "cifar10"},
    ("workload", "precision"): {"fp32", "amp"},
    ("workload", "optimizer"): {"sgd", "adamw"},
    ("distributed", "backend"): {"auto", "nccl", "gloo"},
}
_MODES = {"ddp", "nosync"}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_override(s: str) -> dict:
    """'workload.batch_size=32' -> {'workload': {'batch_size': 32}} (value parsed as YAML)."""
    key, _, raw = s.partition("=")
    if not _:
        raise ValueError(f"override must be key=value: {s!r}")
    node: dict = {}
    cur = node
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = yaml.safe_load(raw)
    return node


def validate(cfg: dict) -> list[str]:
    errs = []

    def unknown(section: dict, ref: dict, path: str) -> None:
        for k, v in section.items():
            if k not in ref:
                errs.append(f"unknown key: {path}{k}")
            elif isinstance(ref[k], dict) and isinstance(v, dict):
                unknown(v, ref[k], f"{path}{k}.")

    unknown(cfg, DEFAULTS, "")
    for (sec, key), allowed in _CHOICES.items():
        if cfg[sec][key] not in allowed:
            errs.append(f"{sec}.{key}={cfg[sec][key]!r} not in {sorted(allowed)}")
    w, m = cfg["workload"], cfg["measurement"]
    for k in ("batch_size", "grad_accum_steps", "synthetic_pool", "seq_len"):
        if not isinstance(w[k], int) or w[k] < 1:
            errs.append(f"workload.{k} must be a positive int")
    for k in ("warmup_steps", "measured_steps", "repeats"):
        if not isinstance(m[k], int) or m[k] < (0 if k == "warmup_steps" else 1):
            errs.append(f"measurement.{k} invalid: {m[k]!r}")
    if not m["modes"] or set(m["modes"]) - _MODES:
        errs.append(f"measurement.modes must be a non-empty subset of {sorted(_MODES)}")
    if not isinstance(w["data_download"], bool):
        errs.append("workload.data_download must be a bool")
    errs += fault_spec.validate(cfg["fault"])
    if cfg["fault"]["mechanism"] in ("loader_sleep", "fetch_sleep", "cpu_preprocess") and w["dataset"] == "synthetic":
        errs.append("data-stall faults need a host DataLoader (workload.dataset synthetic_host or cifar10)")
    if cfg["fault"]["mechanism"] == "emulated_bandwidth" and not m["comm_hook"]:
        errs.append("emulated_bandwidth is implemented in the comm hook (measurement.comm_hook must be true)")
    for k in ("gpu_util", "comm_hook", "nccl_log", "gpu_state", "fingerprint"):
        if not isinstance(m[k], bool):
            errs.append(f"measurement.{k} must be a bool")
    if not (cfg["distributed"]["bucket_cap_mb"] > 0):
        errs.append("distributed.bucket_cap_mb must be > 0")
    return errs


def load(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for o in overrides or []:
        cfg = deep_merge(cfg, parse_override(o))
    errs = validate(cfg)
    if errs:
        raise ValueError("invalid config:\n  " + "\n  ".join(errs))
    return cfg
