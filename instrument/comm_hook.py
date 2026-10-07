"""DDP communication hooks. All PyTorch comm-hook API usage lives here.

API (verified against the installed torch at runtime, see scripts/smoke_ddp.py):
    DistributedDataParallel.register_comm_hook(state, hook)
    hook(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]

NOTE: do NOT add ``from __future__ import annotations`` to this module.
DDP validates the hook signature and requires the ``bucket`` annotation to be
the real ``dist.GradBucket`` class, not a string (observed with torch 2.14:
"Communication hook: bucket annotation should be dist.GradBucket").
"""

import torch
import torch.distributed as dist


class CountingAllReduceHook:
    """All-reduce-mean hook (same math as DDP's default) that records each bucket call."""

    def __init__(self, world_size: int, process_group=None) -> None:
        self.world_size = world_size
        self.process_group = process_group
        self.calls = []

    def reset(self) -> None:
        self.calls = []

    def hook(self, state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        buf = bucket.buffer()
        self.calls.append({
            "bucket_index": bucket.index(),
            "is_last": bucket.is_last(),
            "numel": buf.numel(),
            "bytes": buf.numel() * buf.element_size(),
            "dtype": str(buf.dtype),
        })
        fut = dist.all_reduce(buf, group=self.process_group, async_op=True).get_future()
        ws = self.world_size
        return fut.then(lambda f: f.value()[0].div_(ws))

    def register(self, ddp_model) -> None:
        ddp_model.register_comm_hook(state=None, hook=self.hook)


class TimedAllReduceHook:
    """All-reduce-mean hook that timestamps each bucket on the device timeline.

    * ``ready`` mark: recorded at hook entry on the current (compute) stream,
      i.e. it completes when the bucket's gradients have been produced.
    * ``end`` mark: recorded inside the future's ``then`` callback. For CUDA
      futures the callback runs with current streams that are synchronized with
      the collective's kernels (torch.futures.Future.then docs), so the mark
      completes when the all-reduce has finished on the GPU. For Gloo the
      callback runs on the host when the collective completes.

    The actual collective start is derived in ``instrument.timeline`` as
    max(ready_k, end_{k-1}) because one process group's collectives execute
    serially on one stream.
    """

    def __init__(self, world_size: int, clock, process_group=None) -> None:
        self.world_size = world_size
        self.clock = clock
        self.process_group = process_group
        self.calls = []

    def begin_step(self) -> None:
        self.calls = []

    def hook(self, state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        buf = bucket.buffer()
        rec = {"bucket_index": bucket.index(),
               "bytes": buf.numel() * buf.element_size(),
               "ready": self.clock.mark(), "end": None}
        self.calls.append(rec)
        fut = dist.all_reduce(buf, group=self.process_group, async_op=True).get_future()
        ws, clock = self.world_size, self.clock

        def _done(f):
            t = f.value()[0]
            rec["end"] = clock.mark()
            return t.div_(ws)

        return fut.then(_done)

    def resolve(self, start_mark) -> list[dict]:
        """Offsets (ms) from the step-start mark. Call after device synchronize."""
        out = []
        for c in self.calls:
            if c["end"] is None:
                raise RuntimeError(f"bucket {c['bucket_index']} collective never completed")
            out.append({"bucket_index": c["bucket_index"], "bytes": c["bytes"],
                        "ready_ms": self.clock.ms(start_mark, c["ready"]),
                        "end_ms": self.clock.ms(start_mark, c["end"])})
        return out

    def register(self, ddp_model) -> None:
        ddp_model.register_comm_hook(state=None, hook=self.hook)


class ThrottledTimedAllReduceHook(TimedAllReduceHook):
    """EMULATED communication degradation (fault injection only; not a real link limit).

    Same timing as TimedAllReduceHook, plus an extra delay of ``bytes / throttle_GBps``
    after each bucket's all-reduce, before its result is released to DDP. To behave
    like a slower serial link, the next bucket's collective is not launched before the
    previous delay has elapsed:

    * GPU: the collective is launched from a side stream that waits on the previous
      bucket's delay event (the NCCL stream waits on the launching stream); the delay
      is a ``torch.cuda._sleep`` spin kernel calibrated in cycles/ms.
    * CPU/Gloo: ``time.sleep`` in the completion callback.

    The base class (used for all healthy measurements) is unchanged.
    """

    def __init__(self, world_size: int, clock, throttle_GBps: float, sleep_cycles_per_ms=None,
                 process_group=None) -> None:
        super().__init__(world_size, clock, process_group)
        if throttle_GBps <= 0:
            raise ValueError("throttle_GBps must be > 0")
        self.throttle_GBps = throttle_GBps
        self.sleep_cycles_per_ms = sleep_cycles_per_ms
        self.launch_stream = torch.cuda.Stream(clock.device) if clock.cuda else None
        self.prev_done = None
        self.injected_ms = 0.0  # nominal delay injected in the current step

    def begin_step(self) -> None:
        super().begin_step()
        self.prev_done = None
        self.injected_ms = 0.0

    def hook(self, state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        buf = bucket.buffer()
        nbytes = buf.numel() * buf.element_size()
        rec = {"bucket_index": bucket.index(), "bytes": nbytes, "ready": self.clock.mark(), "end": None}
        self.calls.append(rec)
        delay_ms = nbytes / (self.throttle_GBps * 1e9) * 1e3
        self.injected_ms += delay_ms
        if self.launch_stream is not None:
            self.launch_stream.wait_stream(torch.cuda.current_stream())
            if self.prev_done is not None:
                self.launch_stream.wait_event(self.prev_done)
            with torch.cuda.stream(self.launch_stream):
                fut = dist.all_reduce(buf, group=self.process_group, async_op=True).get_future()
        else:
            fut = dist.all_reduce(buf, group=self.process_group, async_op=True).get_future()
        ws, clock, hk = self.world_size, self.clock, self

        def _done(f):
            t = f.value()[0]
            if clock.cuda:
                torch.cuda._sleep(int(delay_ms * hk.sleep_cycles_per_ms))
                ev = torch.cuda.Event()
                ev.record()
                hk.prev_done = ev
            else:
                import time

                time.sleep(delay_ms / 1e3)
            rec["end"] = clock.mark()
            return t.div_(ws)

        return fut.then(_done)
