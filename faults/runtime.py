"""Runtime fault injectors (straggler, data stall, emulated bandwidth, NCCL transport).

Each injector is calibrated on the device/host it runs on and records what it
actually did, so ground truth carries realized magnitudes, not just nominal ones.
"""

from __future__ import annotations

import os
import time

import torch


# --------------------------------------------------------------------------- straggler
class Straggler:
    """Delay one rank before its forward pass.

    sleep:   host sleep -> the GPU idles; peers wait inside the next collective.
    compute: real extra GPU matmul work of ~delay_ms (calibrated with CUDA events);
             on CPU, a calibrated matmul loop.
    """

    def __init__(self, mechanism: str, rank: int, target_rank: int, delay_ms: float,
                 device: torch.device) -> None:
        self.active = rank == target_rank
        self.mechanism, self.delay_ms, self.device = mechanism, delay_ms, device
        self.iters = 0
        self.calibration: dict = {}
        if self.active and mechanism == "compute":
            self._calibrate()

    def _burn(self, n: int) -> None:
        for _ in range(n):
            self._b = self._a @ self._a

    def _calibrate(self) -> None:
        dim = 2048 if self.device.type == "cuda" else 256
        self._a = torch.randn(dim, dim, device=self.device) / dim ** 0.5
        self._burn(3)
        n = 20
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            self._burn(n)
            e.record()
            e.synchronize()
            per = s.elapsed_time(e) / n
        else:
            t0 = time.perf_counter()
            self._burn(n)
            per = (time.perf_counter() - t0) * 1e3 / n
        self.iters = max(1, round(self.delay_ms / per))
        self.calibration = {"matmul_dim": dim, "ms_per_matmul": per, "iterations": self.iters,
                            "expected_ms": self.iters * per}

    def before_forward(self) -> None:
        if not self.active:
            return
        if self.mechanism == "sleep":
            time.sleep(self.delay_ms / 1e3)
        else:
            self._burn(self.iters)


# --------------------------------------------------------------------------- data stall
class SlowCollate:
    """DataLoader collate_fn that adds a per-batch cost inside the worker process.

    loader_sleep:   time.sleep(delay_ms)               (injected wait)
    cpu_preprocess: real CPU work of ~delay_ms          (calibrated 3x3 blur passes)
    The consumer only waits when production (workers / delay) is slower than consumption.
    """

    def __init__(self, mechanism: str, delay_ms: float, batch_size: int) -> None:
        self.mechanism, self.delay_ms, self.batch_size = mechanism, delay_ms, batch_size
        self.passes = 0
        if mechanism == "cpu_preprocess":
            self.passes = self._calibrate()

    @staticmethod
    def _blur(x: torch.Tensor, passes: int) -> torch.Tensor:
        k = torch.full((3, 1, 3, 3), 1 / 9.0)
        for _ in range(passes):
            x = torch.nn.functional.conv2d(x, k, padding=1, groups=3)
        return x

    def _calibrate(self) -> int:
        torch.set_num_threads(1)
        x = torch.randn(self.batch_size, 3, 32, 32)
        self._blur(x, 2)
        t0 = time.perf_counter()
        self._blur(x, 10)
        per = (time.perf_counter() - t0) * 1e3 / 10
        return max(1, round(self.delay_ms / per))

    def info(self) -> dict:
        return {"mechanism": self.mechanism, "delay_ms": self.delay_ms, "blur_passes": self.passes}

    def __call__(self, batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        if self.mechanism == "loader_sleep":
            time.sleep(self.delay_ms / 1e3)
        else:
            torch.set_num_threads(1)  # one core per worker, as a real preprocessing worker would
            xs = self._blur(xs, self.passes)
        return xs, ys


# --------------------------------------------------------------------------- emulated bandwidth
def gpu_sleep_cycles_per_ms(device: torch.device) -> float:
    """Calibrate torch.cuda._sleep (a spin kernel) in cycles per millisecond."""
    torch.cuda.synchronize(device)
    cycles = 5_000_000
    torch.cuda._sleep(cycles)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(cycles)
    e.record()
    e.synchronize()
    return cycles / s.elapsed_time(e)


# --------------------------------------------------------------------------- NCCL transport
NCCL_SLOW_PATH_ENV = {"NCCL_SHM_DISABLE": "1", "NCCL_P2P_DISABLE": "1"}


def apply_nccl_env(mechanism: str) -> dict:
    """Must run before communicator creation. Returns what was set (recorded in the run)."""
    if mechanism != "shm_disable":
        return {}
    for k, v in NCCL_SLOW_PATH_ENV.items():
        os.environ[k] = v
    return dict(NCCL_SLOW_PATH_ENV)
