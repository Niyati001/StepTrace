"""Instrumented DDP training workload (Milestone 1).

Launch (GPU):
    torchrun --standalone --nproc_per_node=2 -m workloads.train --config configs/baseline.yaml
Launch (CPU / Gloo, any OS):
    python -m workloads.train --config configs/baseline.yaml --spawn 2 --set workload.model=cnn_tiny

One launch = ``repeats`` x ``modes`` measurement blocks, mode order alternating
per repeat (A B, B A, ...) to balance slow drift. Each block: warm-up steps
(discarded) then measured steps. Modes:

* ``ddp``    - normal DDP training; every bucket all-reduce is timed by the hook.
* ``nosync`` - same step inside ``DDP.no_sync()``: no gradient collective is
               issued. A compute-only timing reference on the same hardware and
               process; ranks' weights would diverge, so rank 0's parameters,
               buffers and optimizer state are re-broadcast after each block.

The device is synchronized at the end of every step so per-step CUDA-event
timings can be resolved (see METHODOLOGY.md, "Synchronization policy").
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from faults import runtime as fault_rt
from faults import spec as fault_spec
from instrument import nccl_log
from instrument.comm_hook import ThrottledTimedAllReduceHook, TimedAllReduceHook
from instrument.environment import detect
from instrument.gpu_metrics import GpuMetrics, device_snapshot
from instrument.profiler import analyze_trace, gzip_file, load_trace, make_profiler
from instrument.provenance import require_clean
from instrument.schema import SCHEMA_VERSION, validate_run
from instrument.step_timer import Clock, StepTimer
from instrument.timeline import derive_step
from workloads import config as C
from workloads.data import build_source
from workloads.models import build_model, param_stats


# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, cfg: dict, rank: int, world: int, device: torch.device,
                 local_rank: int = 0, run_id: str = "local") -> None:
        wl, dc = cfg["workload"], cfg["distributed"]
        self.cfg, self.rank, self.world, self.device = cfg, rank, world, device
        self.K = wl["grad_accum_steps"]
        self.amp = wl["precision"] == "amp"
        if self.amp and device.type != "cuda":
            raise ValueError("precision=amp requires CUDA")
        seed = cfg["experiment"]["seed"]
        torch.backends.cudnn.benchmark = bool(wl["cudnn_benchmark"])

        torch.manual_seed(seed)  # identical initial weights on every rank
        model, self.task = build_model(wl["model"], wl["seq_len"])
        self.model_stats = param_stats(model)
        model.to(device)
        self.ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                       bucket_cap_mb=dc["bucket_cap_mb"],
                       gradient_as_bucket_view=dc["gradient_as_bucket_view"],
                       broadcast_buffers=dc["broadcast_buffers"])
        self.clock = Clock(device)
        fault = cfg["fault"]
        mech = fault["mechanism"]
        self.fault_calibration: dict = {}
        self.hook = None
        if cfg["measurement"]["comm_hook"]:
            if mech == "emulated_bandwidth":
                cyc = fault_rt.gpu_sleep_cycles_per_ms(device) if device.type == "cuda" else None
                self.fault_calibration["gpu_sleep_cycles_per_ms"] = cyc
                self.hook = ThrottledTimedAllReduceHook(world, self.clock, fault["throttle_GBps"], cyc)
            else:
                self.hook = TimedAllReduceHook(world, self.clock)
            self.hook.register(self.ddp)
        self.straggler = None
        if mech in ("sleep", "compute"):
            self.straggler = fault_rt.Straggler(mech, rank, fault["rank"], fault["delay_ms"], device)
            self.fault_calibration["straggler"] = {"active_on_this_rank": self.straggler.active,
                                                   **self.straggler.calibration}
        collate = None
        if mech in ("loader_sleep", "cpu_preprocess"):
            collate = fault_rt.SlowCollate(mech, fault["delay_ms"], wl["batch_size"])
            self.fault_calibration["collate"] = collate.info()
        if wl["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(self.ddp.parameters(), lr=wl["lr"], momentum=0.9, weight_decay=5e-4)
        else:
            self.opt = torch.optim.AdamW(self.ddp.parameters(), lr=wl["lr"])
        # set_to_none would detach grads from DDP's bucket views
        self.set_to_none = not dc["gradient_as_bucket_view"]
        self.scaler = None
        if self.amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):  # torch < 2.3
                self.scaler = torch.cuda.amp.GradScaler()
        fetch_delay = fault["delay_ms"] if mech == "fetch_sleep" else 0.0
        if fetch_delay:
            self.fault_calibration["fetch_sleep"] = {"delay_ms": fetch_delay, "where": "training process, next_batch"}
        self.source = build_source(wl, self.task, rank, world, seed, device, local_rank, run_id, collate,
                                   fetch_delay)
        self.loss_fn = nn.CrossEntropyLoss()
        self.timer = StepTimer(self.clock)
        self.gpu = GpuMetrics(device, cfg["measurement"]["gpu_util"])

    def _loss(self, x, y):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.ddp(x)
            if self.task == "lm":
                return self.loss_fn(out.reshape(-1, out.shape[-1]).float(), y.reshape(-1)) / self.K
            return self.loss_fn(out.float(), y) / self.K

    def step(self, mode: str) -> tuple[dict, list[dict]]:
        t = self.timer
        self.gpu.reset_peak()
        if self.hook:
            self.hook.begin_step()
        t.begin()
        batches, host_wait = [], 0.0
        for _ in range(self.K):
            x, y, w = self.source.next_batch()
            batches.append((x, y))
            host_wait += w
        t.mark("data_ready")
        for i, (x, y) in enumerate(batches):
            last = i == self.K - 1
            ctx = nullcontext() if (mode == "ddp" and last) else self.ddp.no_sync()
            with ctx:
                if last:
                    t.mark("accum_end")
                    if self.straggler:
                        self.straggler.before_forward()  # lands in forward_ms of the target rank
                loss = self._loss(x, y)
                if last:
                    t.mark("fwd_end")
                (self.scaler.scale(loss) if self.scaler else loss).backward()
                if last:
                    t.mark("bwd_end")
        if self.scaler:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        self.opt.zero_grad(set_to_none=self.set_to_none)
        t.mark("opt_end")

        self.clock.synchronize()
        wall_ms = (time.perf_counter_ns() - t.host_start_ns) / 1e6
        if self.hook:
            comm = self.hook.resolve(t.start)
            if mode == "nosync" and comm:
                raise RuntimeError("collective issued inside no_sync()")
        else:  # not measured in ddp mode; known zero in nosync mode
            comm = None if mode == "ddp" else []
        rec = derive_step(t.offsets(), comm)
        rec.update(
            timing_source=self.clock.source,
            wall_step_ms=wall_ms,
            host_data_wait_ms=host_wait,
            step_start_monotonic_ns=t.monotonic_start_ns,
            loss=float(loss.detach().float().item()) * self.K,
            **self.gpu.sample(),
        )
        if isinstance(self.hook, ThrottledTimedAllReduceHook):
            rec["injected_comm_delay_ms"] = self.hook.injected_ms
        return rec, comm

    @torch.no_grad()
    def fingerprint(self) -> dict:
        """Per-tensor parameter L2 norms + final loss inputs for equivalence checks."""
        return {n: float(p.detach().float().norm()) for n, p in self.ddp.module.named_parameters()}

    @torch.no_grad()
    def resync_from_rank0(self) -> None:
        tensors = list(self.ddp.module.parameters()) + list(self.ddp.module.buffers())
        for st in self.opt.state.values():
            tensors += [v for v in st.values() if torch.is_tensor(v) and v.device == self.device]
        for tns in tensors:
            dist.broadcast(tns.data, src=0)


# --------------------------------------------------------------------------- #
def run(cfg: dict, run_id: str, launch_index: int, out_root: Path, command: str) -> int:
    code = require_clean()  # fail fast: official runs only from a clean committed tree
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    requested_cfg = cfg
    cfg = fault_spec.apply_config_fault(cfg)  # config-level faults are real workload changes
    truth = fault_spec.ground_truth(requested_cfg)
    nccl_env = fault_rt.apply_nccl_env(cfg["fault"]["mechanism"])  # before communicator creation
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = cfg["distributed"]["backend"]
    if backend == "auto":
        backend = "nccl" if (torch.cuda.is_available() and torch.cuda.device_count() >= world
                             and dist.is_nccl_available()) else "gloo"
    m = cfg["measurement"]
    exp = cfg["experiment"]["name"]
    out_dir = out_root / exp
    nccl_dir = out_dir / "nccl" / run_id
    if backend == "nccl" and m["nccl_log"]:
        nccl_log.enable(nccl_dir)  # must precede communicator creation
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        try:
            dist.init_process_group("nccl", device_id=device)
        except TypeError:
            dist.init_process_group("nccl")
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo")

    tr = Trainer(cfg, rank, world, device, local_rank, run_id)
    snap = (lambda: device_snapshot(device.index)) if (device.type == "cuda" and m["gpu_state"])         else (lambda: None)

    records, blocks, gpu_state, bucket_layout = [], [], [], None
    block_id = 0
    for rep in range(m["repeats"]):
        order = m["modes"] if rep % 2 == 0 else list(reversed(m["modes"]))
        for mode in order:
            before = snap()
            dist.barrier()
            t0 = time.time()
            for _ in range(m["warmup_steps"]):
                tr.step(mode)
            for s in range(m["measured_steps"]):
                rec, comm = tr.step(mode)
                rec.update(run_id=run_id, block_id=block_id, repeat=rep, mode=mode, step=s, rank=rank)
                records.append(rec)
                if comm and bucket_layout is None:
                    bucket_layout = comm
            blocks.append({"block_id": block_id, "repeat": rep, "mode": mode,
                           "warmup_steps": m["warmup_steps"], "measured_steps": m["measured_steps"],
                           "wall_seconds": time.time() - t0})
            gpu_state.append({"rank": rank, "block_id": block_id, "before": before, "after": snap()})
            if mode == "nosync" and world > 1:
                tr.resync_from_rank0()
            block_id += 1

    fingerprint = tr.fingerprint() if m["fingerprint"] else None

    profile = None
    p = m["profile"]
    if p["enabled"]:
        dist.barrier()
        trace = out_dir / "traces" / f"{run_id}_rank{rank}.json"
        prof_recs = []
        with make_profiler(device.type == "cuda", p["wait"], p["warmup"], p["active"], trace) as prof:
            for i in range(p["wait"] + p["warmup"] + p["active"]):
                rec, _ = tr.step("ddp")
                if i >= p["wait"] + p["warmup"]:
                    prof_recs.append(rec)
                prof.step()
        analysis = analyze_trace(load_trace(trace))
        profile = {"rank": rank, "trace": str(gzip_file(trace)), "trace_analysis": analysis,
                   "hook_steps_during_profile": prof_recs}

    gathered = [None] * world
    dist.all_gather_object(gathered, {"records": records, "profile": profile, "gpu_state": gpu_state,
                                      "fault_calibration": tr.fault_calibration, "fingerprint": fingerprint,
                                      "device_name": torch.cuda.get_device_name(device)
                                      if device.type == "cuda" else "cpu"})
    rc = 0
    if rank == 0:
        steps = [r for g in gathered for r in g["records"]]
        doc = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "experiment_id": exp,
            "launch_index": launch_index,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "started_utc": started_utc,
            "command": command,
            # Provenance: exact code revision and tree state that produced this run.
            "provenance": code,
            "backend": backend,
            "world_size": world,
            "devices": [g["device_name"] for g in gathered],
            "environment": detect(),
            "config": requested_cfg,
            "effective_config": cfg,
            # Ground truth is kept apart from measurements; the diagnoser never reads it.
            "ground_truth": truth,
            "fault_runtime": {"nccl_env": nccl_env,
                              "calibration_by_rank": [g["fault_calibration"] for g in gathered]},
            "fingerprint_by_rank": [g["fingerprint"] for g in gathered] if m["fingerprint"] else None,
            "model": tr.model_stats,
            "bucket_layout_first_measured_step_rank0": bucket_layout,
            "blocks": blocks,
            "gpu_state": [x for g in gathered for x in g["gpu_state"]],
            "nccl": nccl_log.parse_dir(nccl_dir) if (backend == "nccl" and m["nccl_log"]) else None,
            "profile": [g["profile"] for g in gathered] if p["enabled"] else None,
            "steps": steps,
        }
        errs = validate_run(doc)
        doc["schema_errors"] = errs
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_id}.json"
        path.write_text(json.dumps(doc, default=str), encoding="utf-8")
        with open(out_dir / f"{run_id}_steps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(steps[0]))
            w.writeheader()
            w.writerows(steps)
        print(f"[steptrace] wrote {path} ({len(steps)} step records, schema errors: {len(errs)})")
        rc = 1 if errs else 0
    dist.barrier()
    dist.destroy_process_group()
    return rc


# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_entry(rank, world, port, cfg, run_id, launch_index, out_root, command):
    os.environ.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    rc = run(cfg, run_id, launch_index, Path(out_root), command)
    if rc:
        raise SystemExit(rc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override, e.g. --set workload.batch_size=32 (repeatable)")
    ap.add_argument("--spawn", type=int, default=0, help="spawn N local processes (CPU/Gloo dev path)")
    ap.add_argument("--out-root", default="results/raw")
    ap.add_argument("--launch-index", type=int, default=int(os.environ.get("COMMSCOPE_LAUNCH_INDEX", 0)))
    args = ap.parse_args(argv)
    cfg = C.load(args.config, args.set)
    command = " ".join([Path(sys.executable).name, "-m", "workloads.train"] + (argv or sys.argv[1:]))

    if os.environ.get("COMMSCOPE_RUN_ID"):
        run_id = os.environ["COMMSCOPE_RUN_ID"]
    elif "RANK" in os.environ:
        el = os.environ.get("TORCHELASTIC_RUN_ID", "")
        run_id = el if el not in ("", "none") else f"port{os.environ.get('MASTER_PORT')}"
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    if args.spawn:
        import torch.multiprocessing as mp

        if sys.platform == "win32":
            os.environ.setdefault("USE_LIBUV", "0")
        mp.spawn(_spawn_entry, nprocs=args.spawn, join=True,
                 args=(args.spawn, _free_port(), cfg, run_id, args.launch_index, args.out_root, command))
        return 0
    if "RANK" not in os.environ:
        ap.error("launch with torchrun, or use --spawn N")
    return run(cfg, run_id, args.launch_index, Path(args.out_root), command)


if __name__ == "__main__":
    raise SystemExit(main())
