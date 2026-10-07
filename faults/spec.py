"""Fault taxonomy, configuration and ground truth (spec §14-16).

Every mechanism maps to exactly one ground-truth class. Ground truth is
written to the run document's ``ground_truth`` field and is NEVER visible to
the diagnoser (see diagnose.features.observable_view).

| mechanism          | class         | how                                            | real or emulated |
|--------------------|---------------|------------------------------------------------|------------------|
| none               | HEALTHY       | -                                              | -                |
| single_bucket      | COMMUNICATION | DDP bucket_cap_mb >= model size: no overlap    | real config      |
| small_batch        | COMMUNICATION | per-GPU batch / factor: less compute per byte  | real config      |
| emulated_bandwidth | COMMUNICATION | +bytes/GBps delay per collective (comm hook)   | EMULATED         |
| shm_disable        | COMMUNICATION | NCCL_SHM_DISABLE=1 (+P2P off): other transport | real transport   |
| sleep              | STRAGGLER     | host sleep before forward on one rank          | injected delay   |
| compute            | STRAGGLER     | extra GPU matmul work on one rank              | real extra work  |
| loader_sleep       | DATA_STALL    | sleep in DataLoader worker collate             | injected delay   |
| fetch_sleep        | DATA_STALL    | sleep on the training process inside the batch | injected delay   |
|                    |               | fetch, after the prefetched batch is received  |                  |
| cpu_preprocess     | DATA_STALL    | real CPU work in DataLoader worker collate     | real extra work  |
"""

from __future__ import annotations

import copy

CLASSES = ("HEALTHY", "COMMUNICATION", "STRAGGLER", "DATA_STALL")

MECHANISMS: dict[str, dict] = {
    "none": {"class": "HEALTHY", "kind": "none", "emulated": False},
    "single_bucket": {"class": "COMMUNICATION", "kind": "config", "emulated": False},
    "small_batch": {"class": "COMMUNICATION", "kind": "config", "emulated": False},
    "emulated_bandwidth": {"class": "COMMUNICATION", "kind": "runtime", "emulated": True},
    "shm_disable": {"class": "COMMUNICATION", "kind": "environment", "emulated": False},
    "sleep": {"class": "STRAGGLER", "kind": "runtime", "emulated": False},
    "compute": {"class": "STRAGGLER", "kind": "runtime", "emulated": False},
    "loader_sleep": {"class": "DATA_STALL", "kind": "runtime", "emulated": False},
    # Synchronous, non-prefetchable fetch cost on the consumer's critical path. Unlike
    # loader_sleep (worker-side, rate-based: wait ~ delay/workers - inter-batch period),
    # prefetching cannot hide it, so the expected added data wait is delay_ms itself.
    "fetch_sleep": {"class": "DATA_STALL", "kind": "runtime", "emulated": False},
    "cpu_preprocess": {"class": "DATA_STALL", "kind": "runtime", "emulated": False},
}

DEFAULT_FAULT = {
    "mechanism": "none",
    "rank": 1,                # straggler target rank
    "delay_ms": 0.0,          # straggler / data-stall magnitude (per step / per batch)
    "throttle_GBps": 0.0,     # emulated_bandwidth: extra delay = bytes / throttle_GBps
    "batch_factor": 4,        # small_batch: batch_size // batch_factor
    "bucket_mb": 1024,        # single_bucket: bucket_cap_mb
    "level": "",              # free-form severity label recorded with ground truth
}


def validate(f: dict, world_size_hint: int = 2) -> list[str]:
    errs = []
    m = f.get("mechanism")
    if m not in MECHANISMS:
        return [f"fault.mechanism {m!r} not in {sorted(MECHANISMS)}"]
    if m in ("sleep", "compute", "loader_sleep", "fetch_sleep", "cpu_preprocess") and not f["delay_ms"] > 0:
        errs.append(f"fault.delay_ms must be > 0 for {m}")
    if m == "emulated_bandwidth" and not f["throttle_GBps"] > 0:
        errs.append("fault.throttle_GBps must be > 0 for emulated_bandwidth")
    if m == "small_batch" and not (isinstance(f["batch_factor"], int) and f["batch_factor"] >= 2):
        errs.append("fault.batch_factor must be an int >= 2")
    if m in ("sleep", "compute") and not (0 <= f["rank"] < world_size_hint):
        errs.append("fault.rank out of range")
    return errs


def apply_config_fault(cfg: dict) -> dict:
    """Return a copy of cfg with config-level fault mechanisms applied (real workload changes)."""
    cfg = copy.deepcopy(cfg)
    f = cfg["fault"]
    if f["mechanism"] == "single_bucket":
        cfg["distributed"]["bucket_cap_mb"] = f["bucket_mb"]
    elif f["mechanism"] == "small_batch":
        cfg["workload"]["batch_size"] = max(1, cfg["workload"]["batch_size"] // f["batch_factor"])
    return cfg


def ground_truth(cfg: dict) -> dict:
    f = cfg["fault"]
    m = MECHANISMS[f["mechanism"]]
    params = {k: v for k, v in f.items() if k not in ("mechanism", "level")}
    relevant = {
        "none": [], "single_bucket": ["bucket_mb"], "small_batch": ["batch_factor"],
        "emulated_bandwidth": ["throttle_GBps"], "shm_disable": [],
        "sleep": ["rank", "delay_ms"], "compute": ["rank", "delay_ms"],
        "loader_sleep": ["delay_ms"], "fetch_sleep": ["delay_ms"], "cpu_preprocess": ["delay_ms"],
    }[f["mechanism"]]
    return {
        "fault_class": m["class"],
        "fault_mechanism": f["mechanism"],
        "fault_parameters": {k: params[k] for k in relevant},
        "level": f["level"],
        "emulated": m["emulated"],
    }
