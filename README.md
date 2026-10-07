# StepTrace | Distributed Training Performance Diagnosis

> **Measure → Understand → Inject → Diagnose → Validate**

StepTrace is an auditable PyTorch distributed-training performance testbed. It measures where a DDP training step spends its time, injects controlled faults with known ground truth, diagnoses the resulting signature with frozen rule-based logic, and evaluates that logic on data it never saw during rule development.

![StepTrace architecture](docs/assets/architecture.png)

## Scope

All GPU evidence was measured on **one node with 2× NVIDIA Tesla T4 GPUs in Kaggle**. Collectives used NCCL `SHM/direct` transport over the host/PCIe path; topology was `PHB` and there was no NVLink.

This is **not** a multi-node network-fabric study. There are no InfiniBand, RoCE, NIC, switch, or production-cluster measurements. Communication bandwidth degradation is explicitly **emulated** unless stated otherwise. Reported performance numbers are specific to the recorded environment.

## The problem

A slow distributed step does not tell you *why* it is slow.

| Observable symptom | Possible cause | What to investigate |
|---|---|---|
| Collective communication reaches the critical path | Communication | DDP buckets, transport, overlap |
| One rank finishes compute late | Straggler | Slow rank / workload imbalance |
| GPU waits for the next batch | Data stall | Loader workers, preprocessing, prefetch |
| Several signatures appear together | Ambiguous contention | Common-mode CPU/input effects |

Because DDP is synchronous, a straggler or input stall can surface as waiting around collectives and look like a communication problem. StepTrace measures those signatures separately—and reports when they are not cleanly separable.

## What StepTrace does

- Measures per-rank step timing with **CUDA events**.
- Timestamps DDP gradient buckets with a **communication hook**.
- Cross-checks timing with `torch.profiler`.
- Uses a paired `no_sync` ablation as an independent communication-exposure estimate.
- Injects controlled **communication, straggler, and data-pipeline faults** with ground truth.
- Builds thresholds from a healthy reference collected in the **same GPU session**.
- Uses an explicit rule-based diagnoser; **no ML or LLM is used for diagnosis**.
- Freezes and hashes the rules before held-out scoring.
- Separates **fault manifestation** from diagnosis.
- Records provenance and verifies ingested evidence with SHA256 manifests.

## Key result

### 73.9% held-out accuracy — 17 / 23

![Held-out confusion matrix](docs/assets/confusion_matrix.png)

| Truth | Correct | Recall |
|---|---:|---:|
| HEALTHY | 5 / 5 | 1.00 |
| COMMUNICATION | 4 / 6 | 0.67 |
| STRAGGLER | 5 / 6 | 0.83 |
| DATA_STALL | 3 / 6 | 0.50 |

With only two launches per fault arm, per-class statistics have limited statistical power.

The held-out set was intentionally difficult: **9/9** healthy/design-mechanism controls were correct, while **8/14** mechanisms not seen before the freeze were correct. That gap is part of the result.

## Where a healthy DDP step goes

![Healthy step](docs/assets/healthy_step.png)

For the held-out reference session, a healthy ResNet-18 FP32 batch-32/GPU step had a median step time of **37.3 ms**, with approximately **4 ms of exposed communication on the critical path**.

## Why the diagnoser missed 6 runs

![Failure analysis](docs/assets/failure_analysis.png)

The misses were not discarded or retuned away.

**Severity boundary — 3 misses.** Two `comm_emulated_bw_8GBps` runs and one low-severity compute-straggler run fell close to the frozen detection boundary. The communication signal was elevated, but too little of the injected delay became exposed critical-path time to satisfy the practical detection floor consistently.

**Observable ambiguity — 3 misses.** CPU-heavy loader preprocessing competed with the four available vCPUs. That increased both input wait and per-rank compute unevenly. One physical cause therefore produced both a **DATA_STALL** and **STRAGGLER** signature. The frozen rule selected the larger observed effect.

These are findings about the detector, not reasons to change the held-out score.

## Measurement model

| Metric | Meaning |
|---|---|
| `step_time_ms` | End-to-end measured training step |
| `data_wait_ms` | GPU-side wait for batch delivery / H2D |
| `compute_time_ms` | Accumulation + forward + backward-to-last-gradient + optimizer |
| `communication_time_ms` | DDP all-reduce busy time |
| `exposed_communication_time_ms` | Communication on the critical path |
| `ddp_finalize_ms` | Post-communication DDP work |
| `compute_skew_ms` | Cross-rank compute imbalance |

Two independent communication-exposure views are retained: a critical-path timeline and paired `DDP − no_sync` ablation. A profiler window provides an additional kernel-level cross-check.

## Diagnosis

The frozen v1 diagnoser uses robust-z thresholds against a same-session healthy reference:

- **DATA_STALL** → elevated input wait
- **STRAGGLER** → elevated cross-rank compute skew
- **COMMUNICATION** → elevated exposed communication
- **HEALTHY** → none elevated

A signal must satisfy both its robust z-score and practical-effect threshold. When multiple causes are elevated, the primary verdict is the larger estimated critical-path effect; secondary causes remain evidence.

The rules were frozen before held-out scoring and were not refit after seeing the held-out failures.

## Experimental integrity

- Same-session healthy references account for variation in cloud GPU performance.
- Rules are frozen, hash-recorded, and tied to Git provenance.
- Every run records configuration, environment, seed, timestamp, Git state, and fault ground truth.
- Fault manifestation is checked independently from diagnosis.
- Raw evidence is ingested with SHA256 verification.
- Held-out failures were analyzed independently without rescoring or changing the frozen rules.
- CPU/Gloo tests provide GPU-independent regression coverage.

## What this project does **not** claim

StepTrace does **not** claim:

- multi-node performance results;
- InfiniBand/RoCE/NVSwitch/NVLink performance;
- production-scale training behavior;
- generalization to other GPU families or arbitrary models;
- that emulated bandwidth delay represents a real slow network link;
- statistically strong confidence intervals from two launches per held-out arm;
- GPU optimization speedups.

An optimization layer exists and is CPU-tested, but **no GPU optimization campaign was run**, so no speed-up is reported.

## Repository

```text
workloads/    instrumented DDP workload, models, data/config schema
instrument/   CUDA timing, communication hooks, timelines, profiler, provenance
faults/       fault taxonomy, runtime injectors, manifestation checks
diagnose/     features, thresholds, rules, evaluation, freeze logic
analysis/     statistics, validation gates, pilot selection, plots
optimize/     tuning/validation framework (CPU-tested)
scripts/      experiment and validation tooling
configs/      baseline, pilot, and public design campaign configurations
tests/        CPU-only unit and Gloo integration tests
```

Generated experiment evidence and internal development guides are intentionally **not part of the public repository**.

## Quickstart

### CPU development

```bash
pip install -r requirements.txt
python scripts/detect_environment.py
python -m pytest -q
```

Without two GPUs, the project can exercise its plumbing through CPU/Gloo. Those runs are development checks, not GPU performance evidence.

### GPU experiments

The measurement and campaign tooling is designed for a host with at least two GPUs. The reported GPU campaign was run in Kaggle on 2× Tesla T4.

See the public configurations under `configs/` and source modules under `instrument/`, `faults/`, and `diagnose/`.

## Limitations and next steps

A future version could evaluate real multi-node NCCL over a physical network fabric, real bandwidth controls, larger workloads with stronger overlap, CPU-contention-aware input signatures, and broader severity ladders.

Those are **future experiments**, not current results.
