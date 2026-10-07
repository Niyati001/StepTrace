"""Milestone 0: minimal distributed / DDP smoke test.

Proves that the detected environment can run correct data-parallel training
before any instrumentation or experiment is built on top of it.

Launch (GPU, preferred):
    torchrun --standalone --nproc_per_node=2 scripts/smoke_ddp.py --nccl-debug

Launch (local CPU / Gloo, any OS incl. Windows):
    python scripts/smoke_ddp.py --spawn 2

Exit code 0 only if every check passes. A JSON record (environment, config,
per-rank results, NCCL transport lines if available) is written to
results/raw/smoke/.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from instrument.comm_hook import CountingAllReduceHook  # noqa: E402
from instrument import nccl_log  # noqa: E402
from instrument.environment import detect  # noqa: E402
from instrument.provenance import require_clean  # noqa: E402

OUT_DIR = Path("results/raw/smoke")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class Check:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, passed: bool, **detail) -> bool:
        self.items.append({"name": name, "passed": bool(passed), **detail})
        return passed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _flat(tensors) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).float() for t in tensors])


def _make_model(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)  # identical init on every rank
    return nn.Sequential(
        nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)
    ).to(device)


def _batch(seed: int, rank: int, device: torch.device, n: int = 64):
    g = torch.Generator().manual_seed(seed * 1000 + rank)  # distinct data per rank
    x = torch.randn(n, 256, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_allreduce(chk: Check, rank: int, ws: int, device: torch.device) -> None:
    t = torch.full((1 << 20,), float(rank + 1), device=device)
    dist.all_reduce(t)
    expected = ws * (ws + 1) / 2
    err = (t - expected).abs().max().item()
    chk.add("all_reduce_sum_correct", err == 0.0, expected=expected, max_abs_err=err)


def check_ddp(chk: Check, rank: int, ws: int, device: torch.device, seed: int, steps: int) -> None:
    ref = _make_model(seed, device)                       # non-DDP reference copy
    model = _make_model(seed, device)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    hook = CountingAllReduceHook(ws)
    hook.register(ddp)
    loss_fn = nn.CrossEntropyLoss()

    # --- gradient equivalence on step 1 -------------------------------------
    x, y = _batch(seed, rank, device)
    loss_fn(ref(x), y).backward()
    ref_grad = _flat(p.grad for p in ref.parameters())
    dist.all_reduce(ref_grad)                             # independent path: explicit collective
    ref_grad /= ws

    loss_fn(ddp(x), y).backward()
    ddp_grad = _flat(p.grad for p in ddp.module.parameters())
    max_err = (ddp_grad - ref_grad).abs().max().item()
    scale = ref_grad.abs().max().item()
    chk.add("ddp_grad_equals_mean_of_local_grads",
            max_err <= 1e-6 + 1e-4 * scale, max_abs_err=max_err, grad_abs_max=scale)

    # --- comm hook accounting -----------------------------------------------
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    hook_bytes = sum(c["bytes"] for c in hook.calls)
    chk.add("comm_hook_invoked", len(hook.calls) > 0, buckets=len(hook.calls))
    chk.add("comm_hook_bytes_match_params", hook_bytes == param_bytes,
            hook_bytes=hook_bytes, param_bytes=param_bytes)
    first_step_buckets = list(hook.calls)

    # --- short training: finite, decreasing loss, ranks stay in sync ---------
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
    opt.zero_grad(set_to_none=True)
    losses = []
    for _ in range(steps):
        loss = loss_fn(ddp(x), y)                           # fixed batch: loss should fall
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    chk.add("loss_finite", all(map(lambda v: v == v and abs(v) != float("inf"), losses)))
    chk.add("loss_decreases", losses[-1] < losses[0], first=losses[0], last=losses[-1])

    flat = _flat(ddp.module.parameters())
    gathered = [torch.empty_like(flat) for _ in range(ws)]
    dist.all_gather(gathered, flat)
    div = max((g - gathered[0]).abs().max().item() for g in gathered)
    chk.add("params_identical_across_ranks", div <= 1e-6, max_abs_divergence=div)
    chk.items[-1]["first_step_buckets"] = first_step_buckets


def allreduce_sweep(rank: int, ws: int, device: torch.device, sizes_mib, warmup: int, iters: int):
    """Measured all_reduce latency per message size.

    CUDA: events on the current stream bracket a *synchronous* all_reduce; with
    NCCL the current stream waits on the NCCL stream, so elapsed time covers
    the collective. CPU: perf_counter around the blocking call. A barrier
    precedes every iteration so ranks start together.
    """
    rows = []
    for mib in sizes_mib:
        t = torch.ones(int(mib * 2**20 / 4), device=device)
        for _ in range(warmup):
            dist.all_reduce(t)
        _sync(device)
        times = []
        for _ in range(iters):
            dist.barrier()
            if device.type == "cuda":
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                dist.all_reduce(t)
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                dist.all_reduce(t)
                times.append((time.perf_counter() - t0) * 1e3)
        med = statistics.median(times)
        nbytes = t.numel() * 4
        algbw = nbytes / (med / 1e3) / 1e9
        rows.append({
            "size_mib": mib, "bytes": nbytes, "iters": iters,
            "median_ms": med, "min_ms": min(times), "max_ms": max(times),
            "algbw_GBps": algbw,
            "busbw_GBps": algbw * 2 * (ws - 1) / ws,  # nccl-tests convention for all_reduce
        })
    return rows


def parse_nccl_logs(log_dir: Path) -> dict:
    """Transport/version lines that NCCL itself reports (see instrument.nccl_log)."""
    return nccl_log.parse_dir(log_dir)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available() and torch.cuda.device_count() >= ws and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def worker(args, run_id: str, log_dir: Path) -> int:
    code = require_clean()
    rank, ws = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = resolve_backend(args.backend)

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    chk = Check()
    t0 = time.perf_counter()
    try:
        dist.init_process_group(backend=backend, device_id=device if backend == "nccl" else None)
    except TypeError:  # older torch without device_id kwarg
        dist.init_process_group(backend=backend)
    chk.add("init_process_group", True, backend=backend, seconds=time.perf_counter() - t0)

    sweep, error = [], None
    try:
        dist.barrier()
        check_allreduce(chk, rank, ws, device)
        check_ddp(chk, rank, ws, device, args.seed, args.steps)
        sizes = args.sweep_mib if device.type == "cuda" else [s for s in args.sweep_mib if s <= 16]
        sweep = allreduce_sweep(rank, ws, device, sizes, args.warmup, args.iters)
    except Exception:
        error = traceback.format_exc()
        chk.add("no_exception", False, traceback=error)

    record = {
        "rank": rank, "local_rank": local_rank, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "checks": chk.items, "allreduce_sweep": sweep,
        "peak_mem_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }
    gathered = [None] * ws
    dist.all_gather_object(gathered, record)

    passed = all(c["passed"] for r in gathered for c in r["checks"])
    if rank == 0:
        result = {
            "run_id": run_id,
            "kind": "smoke_ddp",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "passed": passed,
            "provenance": code,
            "config": {**vars(args), "backend_resolved": backend, "world_size": ws},
            "environment": detect(),
            "nccl": parse_nccl_logs(log_dir) if backend == "nccl" and args.nccl_debug else None,
            "ranks": gathered,
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"smoke_{run_id}.json"
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print_summary(result, out)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if passed else 1


def print_summary(result: dict, out: Path) -> None:
    cfg = result["config"]
    print("\nStepTrace smoke test")
    print("====================")
    print(f"backend={cfg['backend_resolved']} world_size={cfg['world_size']} "
          f"devices={[r['device_name'] for r in result['ranks']]}")
    for r in result["ranks"]:
        for c in r["checks"]:
            extra = {k: v for k, v in c.items()
                     if k not in ("name", "passed", "first_step_buckets", "traceback")}
            print(f"  rank{r['rank']} [{'PASS' if c['passed'] else 'FAIL'}] {c['name']} {extra}")
            if "traceback" in c:
                print(c["traceback"])
    r0 = result["ranks"][0]
    if r0["allreduce_sweep"]:
        print("\nall_reduce sweep (rank 0, median; measured in THIS environment only)")
        print(f"  {'MiB':>6} {'median ms':>10} {'algbw GB/s':>11} {'busbw GB/s':>11}")
        for row in r0["allreduce_sweep"]:
            print(f"  {row['size_mib']:>6} {row['median_ms']:>10.3f} "
                  f"{row['algbw_GBps']:>11.2f} {row['busbw_GBps']:>11.2f}")
    if result.get("nccl"):
        print(f"\nNCCL-reported transports: {result['nccl']['transports'] or 'none parsed'}")
    print(f"\nOVERALL: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"Saved: {out}")


# --------------------------------------------------------------------------- #
# Launchers
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_entry(rank: int, ws: int, port: int, args, run_id: str, log_dir: str) -> None:
    os.environ.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(ws),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    rc = worker(args, run_id, Path(log_dir))
    if rc:
        raise SystemExit(rc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "nccl", "gloo"], default="auto")
    ap.add_argument("--spawn", type=int, default=0,
                    help="spawn N local processes instead of using torchrun env vars")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--sweep-mib", type=float, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--nccl-debug", action="store_true",
                    help="set NCCL_DEBUG=INFO and capture per-process logs to parse transports")
    args = ap.parse_args()

    if args.spawn == 0 and "RANK" not in os.environ:
        ap.error("not launched by torchrun; use torchrun or --spawn N")
    # Every rank must agree on run_id (it names the shared NCCL log dir).
    elastic = os.environ.get("TORCHELASTIC_RUN_ID", "")
    if os.environ.get("COMMSCOPE_RUN_ID"):
        run_id = os.environ["COMMSCOPE_RUN_ID"]
    elif "RANK" in os.environ:  # torchrun: per-launch shared identifier
        run_id = elastic if elastic not in ("", "none") else f"port{os.environ['MASTER_PORT']}"
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    log_dir = OUT_DIR / f"nccl_{run_id}"
    if args.nccl_debug:
        nccl_log.enable(log_dir)  # NCCL reads these at communicator creation

    if args.spawn:
        import torch.multiprocessing as mp

        if sys.platform == "win32":
            os.environ.setdefault("USE_LIBUV", "0")  # Windows TCPStore lacks libuv
        mp.spawn(_spawn_entry, args=(args.spawn, _free_port(), args, run_id, str(log_dir)),
                 nprocs=args.spawn, join=True)
        return 0
    return worker(args, run_id, log_dir)


if __name__ == "__main__":
    raise SystemExit(main())
