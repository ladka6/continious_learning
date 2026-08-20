"""Uniform efficiency instrumentation shared (copied verbatim) between the
PILOT baseline repo and our TOSCA repo, so both report cost the same way.

Per task we record:
  - train/eval wall-clock seconds
  - peak GPU memory (MB) during train and during eval
  - measured inference FLOPs/sample, profiled through the model's OWN
    _eval_cnn on a single test batch -- this includes method-specific extras
    (e.g. L2P/DualPrompt's second query forward, expert routing), which
    analytic formulas would miss
  - parameter counts (total / trainable) and train-set size (so training
    FLOPs can be estimated offline as 3 x fwd_flops x samples x epochs)

Everything is flushed to one JSON per (config, seed) after every task, so a
killed job still leaves the completed tasks' metrics on disk.
"""

import json
import logging
import os
import time

import torch


def reset_peak_memory(device=None):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device=None):
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024.0**2)
    return 0.0


def profile_flops(fn, *args, **kwargs):
    """Total FLOPs of one call to ``fn`` via torch.profiler (None on failure).
    Counts CPU+CUDA matmul/conv FLOPs; small elementwise ops are ignored,
    which is fine at ViT scale."""
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, with_flops=True) as prof:
            fn(*args, **kwargs)
        flops = sum(e.flops for e in prof.key_averages() if e.flops)
        return int(flops) if flops else None
    except Exception as exc:  # profiling must never kill a training run
        logging.warning("FLOPs profiling failed: %s", exc)
        return None


def inference_flops_per_sample(model, test_dataset, batch_size=32):
    """Profile the learner's own _eval_cnn on ONE batch of the current test
    set and return FLOPs per sample."""
    from torch.utils.data import DataLoader, Subset

    n = min(batch_size, len(test_dataset))
    if n == 0:
        return None
    loader = DataLoader(
        Subset(test_dataset, list(range(n))),
        batch_size=n,
        shuffle=False,
        num_workers=0,
    )
    total = profile_flops(model._eval_cnn, loader)
    return int(total / n) if total else None


def count_parameters(network, trainable_only=False):
    params = network.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return int(sum(p.numel() for p in params))


def dir_size_mb(path):
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024.0**2)


class MetricsLogger:
    """Append-per-task metrics store, flushed to JSON after every task."""

    def __init__(self, path, args):
        self.path = path
        self.meta = {
            "model_name": args.get("model_name"),
            "dataset": args.get("dataset"),
            "seed": args.get("seed"),
            "init_cls": args.get("init_cls"),
            "increment": args.get("increment"),
            "backbone_type": args.get("backbone_type"),
            "batch_size": args.get("batch_size"),
            "epochs": args.get("epochs", args.get("tuned_epoch")),
        }
        self.tasks = []
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log_task(self, task, **metrics):
        entry = {"task": int(task)}
        entry.update(metrics)
        self.tasks.append(entry)
        with open(self.path, "w") as f:
            json.dump({"meta": self.meta, "tasks": self.tasks}, f, indent=2)
        logging.info("Efficiency metrics for task %s written to %s", task, self.path)


class PhaseTimer:
    """Context manager: wall-clock + peak GPU memory for one phase."""

    def __init__(self):
        self.seconds = 0.0
        self.peak_mb = 0.0

    def __enter__(self):
        reset_peak_memory()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._start
        self.peak_mb = peak_memory_mb()
        return False
