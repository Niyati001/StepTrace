"""GPU memory and coarse utilization.

Utilization comes from NVML via ``torch.cuda.utilization`` (needs pynvml /
nvidia-ml-py). NVML reports the fraction of a driver sample window (roughly
1/6 s to 1 s) in which *any* kernel was running. It says nothing about SM
occupancy or efficiency and its window is longer than one step, so it is
recorded as a coarse context signal only.
"""

from __future__ import annotations

import torch


class GpuMetrics:
    def __init__(self, device: torch.device, want_util: bool = True) -> None:
        self.device = device
        self.cuda = device.type == "cuda"
        self.util_ok = self.cuda and want_util
        if self.util_ok:
            try:
                torch.cuda.utilization(device)
            except Exception:
                self.util_ok = False

    def reset_peak(self) -> None:
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def sample(self) -> dict:
        if not self.cuda:
            return {"gpu_memory_mb": None, "gpu_memory_reserved_mb": None, "gpu_utilization_pct": None}
        util = None
        if self.util_ok:
            try:
                util = float(torch.cuda.utilization(self.device))
            except Exception:
                self.util_ok = False
        return {
            "gpu_memory_mb": torch.cuda.max_memory_allocated(self.device) / 2**20,
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved(self.device) / 2**20,
            "gpu_utilization_pct": util,
        }


_SNAPSHOT_FIELDS = ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw", "utilization.gpu")
_REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")  # new, old name


def device_snapshot(index: int) -> dict | None:
    """Clocks/temperature/power/throttle state from nvidia-smi, taken between blocks.

    Explains drift (thermal or power throttling on shared cloud GPUs) without
    perturbing timed steps. Returns None if nvidia-smi is unavailable.
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    for reason in _REASON_FIELDS:
        q = ",".join(_SNAPSHOT_FIELDS + (reason,))
        try:
            r = subprocess.run(["nvidia-smi", "-i", str(index), f"--query-gpu={q}",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode == 0 and r.stdout.strip():
            vals = [v.strip() for v in r.stdout.strip().split(",")]
            keys = ("sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_w",
                    "nvml_util_pct", "clock_event_reasons")
            out = {}
            for k, v in zip(keys, vals):
                try:
                    out[k] = float(v) if k != "clock_event_reasons" else v
                except ValueError:
                    out[k] = None
            return out
    return None
