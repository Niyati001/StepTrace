"""Input pipelines.

* ``synthetic``: a small pool of batches pre-generated on the device and
  cycled. No host I/O or H2D copies -> removes input-pipeline variance; used
  for the communication pilot.
* ``synthetic_host``: random tensors served through a real DataLoader
  (workers, pinning, H2D copy). Exercises the loader path without downloads.
* ``cifar10``: torchvision CIFAR-10 with standard augmentation and a
  DistributedSampler. Prepare it first with ``scripts/prepare_data.py``
  (one process, verified MD5s); see workloads/datasets.py.

Every source exposes ``next_batch() -> (x, y, host_wait_ms)`` where
``host_wait_ms`` is the host time blocked waiting for the batch.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from workloads.datasets import ensure_cifar10

VOCAB = 8192


class DevicePool:
    def __init__(self, task: str, batch: int, pool: int, seq_len: int, seed: int,
                 device: torch.device) -> None:
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(pool):
            if task == "image":
                x = torch.randn(batch, 3, 32, 32, generator=g)
                y = torch.randint(0, 10, (batch,), generator=g)
            else:
                tok = torch.randint(0, VOCAB, (batch, seq_len + 1), generator=g)
                x, y = tok[:, :-1].contiguous(), tok[:, 1:].contiguous()
            self.items.append((x.to(device), y.to(device)))
        self.i = 0

    def next_batch(self):
        x, y = self.items[self.i % len(self.items)]
        self.i += 1
        return x, y, 0.0


class LoaderSource:
    def __init__(self, dataset, batch: int, rank: int, world: int, seed: int,
                 num_workers: int, pin_memory: bool, device: torch.device, collate_fn=None,
                 prefetch_factor: int | None = None, fetch_delay_ms: float = 0.0) -> None:
        self.sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                          shuffle=True, seed=seed, drop_last=True)
        self.loader = DataLoader(dataset, batch_size=batch, sampler=self.sampler,
                                 num_workers=num_workers, drop_last=True,
                                 pin_memory=pin_memory and device.type == "cuda",
                                 persistent_workers=num_workers > 0, collate_fn=collate_fn,
                                 **({"prefetch_factor": prefetch_factor}
                                    if (prefetch_factor and num_workers > 0) else {}))
        self.device, self.epoch = device, 0
        self.fetch_delay_s = fetch_delay_ms / 1e3  # fetch_sleep fault: synchronous, on this process
        self.it = iter(self.loader)

    def next_batch(self):
        t0 = time.perf_counter()
        try:
            x, y = next(self.it)
        except StopIteration:
            self.epoch += 1
            self.sampler.set_epoch(self.epoch)
            self.it = iter(self.loader)
            x, y = next(self.it)
        if self.fetch_delay_s:
            time.sleep(self.fetch_delay_s)  # after the prefetched batch arrived: prefetching cannot hide it
        wait_ms = (time.perf_counter() - t0) * 1e3
        nb = self.device.type == "cuda"
        return x.to(self.device, non_blocking=nb), y.to(self.device, non_blocking=nb), wait_ms


def _cifar10(root: str, local_rank: int, tag: str, download: bool):
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10

    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    # No collective here: each rank verifies files itself (see workloads/datasets.py).
    # By default nothing is downloaded inside the distributed job; run
    # scripts/prepare_data.py first. With data_download=true, local rank 0 prepares
    # and the other ranks wait on file markers with a bounded timeout.
    ensure_cifar10(root, is_preparer=local_rank == 0, tag=tag, download=download)
    return CIFAR10(root, train=True, download=False, transform=tf)


def build_source(wl: dict, task: str, rank: int, world: int, seed: int, device: torch.device,
                 local_rank: int = 0, tag: str = "local", collate_fn=None, fetch_delay_ms: float = 0.0):
    ds, b = wl["dataset"], wl["batch_size"]
    if ds == "synthetic":
        return DevicePool(task, b, wl["synthetic_pool"], wl["seq_len"], seed * 1000 + rank, device)
    if task != "image":
        raise ValueError(f"dataset {ds!r} only supports image models")
    if ds == "synthetic_host":
        g = torch.Generator().manual_seed(seed)
        n = b * world * 64  # long epochs: avoid epoch-boundary refill stalls in healthy runs
        dataset = TensorDataset(torch.randn(n, 3, 32, 32, generator=g),
                                torch.randint(0, 10, (n,), generator=g))
    else:
        dataset = _cifar10(wl["data_root"], local_rank, tag, wl["data_download"])
    return LoaderSource(dataset, b, rank, world, seed, wl["num_workers"], wl["pin_memory"], device,
                        collate_fn, wl["prefetch_factor"], fetch_delay_ms)
