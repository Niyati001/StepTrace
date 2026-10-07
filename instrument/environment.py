"""Environment detection.

Everything here is *detected*, never assumed. Any field that cannot be
determined is recorded as ``None`` in JSON and printed as
``N/A / unavailable``. This module must import and run without PyTorch,
without a GPU and without ``nvidia-smi``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

NA = "N/A / unavailable"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Environment variables that influence distributed/CUDA behaviour.
_ENV_PREFIXES = ("NCCL_", "CUDA_", "TORCH_", "PYTORCH_", "OMP_", "GLOO_")
_ENV_EXACT = (
    "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK",
    "LOCAL_WORLD_SIZE", "USE_LIBUV", "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_URL_BASE", "COLAB_GPU", "PJRT_DEVICE", "TPU_NAME",
    "TPU_ACCELERATOR_TYPE", "XRT_TPU_CONFIG",
)


def _run(cmd: list[str], timeout: float = 20.0) -> str | None:
    """Run a command, return stripped stdout or None if unavailable/failing."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_commit() -> dict[str, Any]:
    sha = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    return {"sha": sha, "dirty": bool(dirty) if sha else None}


def _cpu_info() -> dict[str, Any]:
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore", encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    if model is None:
        model = platform.processor() or None
    return {
        "model": model,
        "logical_cores": os.cpu_count(),
        "affinity_cores": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


def _ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / 2**30, 2)
    except Exception:
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 2**20, 2)
    return None


def _nvidia_smi() -> dict[str, Any]:
    q = _run([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,pci.bus_id,compute_mode",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    if q:
        for line in q.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": int(parts[0]), "name": parts[1], "driver": parts[2],
                    "memory_total_mib": float(parts[3]), "pci_bus_id": parts[4],
                    "compute_mode": parts[5],
                })
    return {
        "available": q is not None,
        "gpus": gpus,
        "driver_version": gpus[0]["driver"] if gpus else None,
        # Text topology matrix (PIX/PHB/SYS/NV#). Recorded verbatim; not interpreted.
        "topology_matrix": _run(["nvidia-smi", "topo", "-m"]),
    }


def _tpu_presence() -> dict[str, Any]:
    """Record whether a TPU *appears* present. TPUs are not used by the MVP."""
    accel = sorted(str(p) for p in Path("/dev").glob("accel*")) if Path("/dev").exists() else []
    hints = {k: os.environ[k] for k in ("PJRT_DEVICE", "TPU_NAME", "TPU_ACCELERATOR_TYPE", "XRT_TPU_CONFIG") if k in os.environ}
    return {"device_nodes": accel, "env_hints": hints, "detected": bool(accel or hints)}


def _torch_info() -> dict[str, Any]:
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # torch missing or broken
        return {"installed": False, "error": repr(exc)}

    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
        "nccl_version": None,
        "p2p_matrix": None,
        "dist_available": dist.is_available(),
        "backends": {
            "nccl": bool(dist.is_available() and dist.is_nccl_available()),
            "gloo": bool(dist.is_available() and dist.is_gloo_available()),
        },
    }
    if info["cuda_available"]:
        for i in range(info["gpu_count"]):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "index": i, "name": p.name,
                "memory_gib": round(p.total_memory / 2**30, 2),
                "compute_capability": f"{p.major}.{p.minor}",
                "multiprocessors": p.multi_processor_count,
            })
        try:
            v = torch.cuda.nccl.version()
            info["nccl_version"] = ".".join(map(str, v)) if isinstance(v, tuple) else str(v)
        except Exception:
            pass
        n = info["gpu_count"]
        if n >= 2:
            info["p2p_matrix"] = [
                [None if i == j else bool(torch.cuda.can_device_access_peer(i, j)) for j in range(n)]
                for i in range(n)
            ]
    return info


def capabilities(env: dict[str, Any]) -> dict[str, Any]:
    """Decide which experiment families are enabled, from detected facts only."""
    t = env["torch"]
    gpus = t.get("gpu_count", 0) if t.get("installed") else 0
    nccl = bool(t.get("installed") and t["backends"]["nccl"])
    gloo = bool(t.get("installed") and t["backends"]["gloo"])
    multi_gpu_nccl = gpus >= 2 and nccl
    if multi_gpu_nccl:
        mode, backend = "multi_gpu_nccl", "nccl"
    elif gpus == 1:
        mode, backend = "single_gpu", "gloo" if gloo else None
    else:
        mode, backend = "cpu_only", "gloo" if gloo else None
    return {
        "mode": mode,
        "recommended_backend": backend,
        "multi_gpu_ddp_experiments": multi_gpu_nccl,
        "nccl_transport_experiments": multi_gpu_nccl,
        "gpu_profiling": gpus >= 1,
        "cpu_gloo_development": gloo,
        "tpu_used": False,  # by design: TPU is out of MVP scope even if present
    }


def detect() -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "os": {"system": platform.system(), "release": platform.release(), "platform": platform.platform()},
        "hostname": platform.node(),
        "cpu": _cpu_info(),
        "ram_gib": _ram_gb(),
        "torch": _torch_info(),
        "nvidia_smi": _nvidia_smi(),
        "tpu": _tpu_presence(),
        "git": git_commit(),
        "env_vars": {
            k: v for k, v in sorted(os.environ.items())
            if k.startswith(_ENV_PREFIXES) or k in _ENV_EXACT
        },
    }
    env["capabilities"] = capabilities(env)
    return env


def _fmt(v: Any) -> str:
    return NA if v is None or v == [] or v == {} else str(v)


def format_report(env: dict[str, Any]) -> str:
    t, smi, cap = env["torch"], env["nvidia_smi"], env["capabilities"]
    lines = [
        "StepTrace environment",
        "=====================",
        f"Python:              {env['python']}",
        f"PyTorch:             {_fmt(t.get('version'))}",
        f"CUDA (torch build):  {_fmt(t.get('cuda_build'))}",
        f"cuDNN:               {_fmt(t.get('cudnn'))}",
        f"Driver:              {_fmt(smi.get('driver_version'))}",
        f"NCCL:                {_fmt(t.get('nccl_version'))}",
        f"GPU count (torch):   {t.get('gpu_count', 0) if t.get('installed') else NA}",
    ]
    for g in t.get("gpus", []):
        lines.append(f"  GPU {g['index']}: {g['name']}  {g['memory_gib']} GiB  sm_{g['compute_capability']}")
    if not t.get("gpus") and smi["gpus"]:
        for g in smi["gpus"]:
            lines.append(f"  GPU {g['index']} (nvidia-smi only): {g['name']}  {g['memory_total_mib']} MiB")
    lines += [
        f"P2P access matrix:   {_fmt(t.get('p2p_matrix'))}",
        "GPU topology:        " + ("\n" + smi["topology_matrix"] if smi.get("topology_matrix") else NA),
        f"TPU detected:        {env['tpu']['detected']} (not used by the MVP)",
        f"CPU:                 {_fmt(env['cpu']['model'])}  ({_fmt(env['cpu']['logical_cores'])} logical cores)",
        f"RAM:                 {_fmt(env['ram_gib'])} GiB",
        f"OS:                  {env['os']['platform']}",
        f"Dist backends:       nccl={t.get('backends', {}).get('nccl', NA)} gloo={t.get('backends', {}).get('gloo', NA)}",
        f"Git commit:          {_fmt(env['git']['sha'])}{' (dirty)' if env['git']['dirty'] else ''}",
        "Relevant env vars:   " + (", ".join(f"{k}={v}" for k, v in env["env_vars"].items()) or NA),
        "",
        f"Capability mode:     {cap['mode']}",
        f"Recommended backend: {_fmt(cap['recommended_backend'])}",
        f"Multi-GPU DDP/NCCL experiments enabled: {cap['multi_gpu_ddp_experiments']}",
    ]
    if not t.get("installed"):
        lines.append(f"\nPyTorch import failed: {t.get('error')}")
    return "\n".join(lines)


def save(env: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    return path
