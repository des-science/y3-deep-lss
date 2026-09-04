# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Author: Arne Thomsen

Benchmark the GridPipeline input pipeline in isolation (no network, no GPU).

Measures the sustained throughput (examples/s) and the peak host RSS of a *single* tf.data
pipeline built exactly like run_training.py builds it for the grid dataset, for one choice of
(local_batch_size, n_readers, n_prefetch, n_workers, file_name_shuffle_buffer,
examples_shuffle_buffer). One configuration per process, so the peak RSS (VmHWM) is clean and
attributable; sweep by calling this script many times (see submissions/clariden/benchmark_dataloader.sh).

The pipeline is the one MirroredStrategy actually runs: a single, un-sharded pipeline (input_context
is None here, matching num_input_pipelines=1 on one node) that feeds all local GPUs. So the measured
throughput is what the whole node's 4 GPUs share, and the measured RSS is the whole node's dataloader
footprint (not per-GPU). Under Horovod each rank would run its own copy of this pipeline instead.

Run CPU-only: set CUDA_VISIBLE_DEVICES="" so TF never grabs a GPU (done below, before importing TF).
"""

import os
import json
import argparse
import time
import threading

# CPU-only: we benchmark the input pipeline, not the network. Must happen before TF import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


from msfm.grid_pipeline import GridPipeline
from msfm.utils import input_output, files

from deep_lss.utils import config_compose, configuration


def _vmhwm_gb():
    """Peak resident set size of this process (VmHWM from /proc), in GB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / (1024.0**2)  # kB -> GB
    except FileNotFoundError:
        pass
    return float("nan")


def _rss_gb():
    """Current resident set size of this process (VmRSS from /proc), in GB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024.0**2)
    except FileNotFoundError:
        pass
    return float("nan")


class RSSSampler(threading.Thread):
    """Background peak-RSS sampler (VmHWM already gives the peak, but a live sampler lets us also
    report the RSS *plateau* the running pipeline settles at, independent of TF's import baseline)."""

    def __init__(self, period=0.2):
        super().__init__(daemon=True)
        self.period = period
        self.peak = 0.0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_gb())
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()


def build_dataset(args, params_dset):
    """Reconstruct the exact training grid pipeline for one parameter choice."""
    net_conf = config_compose.load_composed(args.net_config)
    dlss_conf = configuration.read_split_configs(args.probes_config, args.scales_config)
    data_conf = input_output.read_yaml(args.data_config)
    msfm_conf = files.load_config(args.msfm_config)

    dset_common = dlss_conf["dset"]["common"]
    with_lensing = dset_common["with_lensing"]
    with_clustering = dset_common["with_clustering"]
    with_cross = dset_common.get("with_cross", False)

    # pipeline object (physics), mirrors run_training pipe_kwargs for the maps-only grid path
    pipeline = GridPipeline(
        conf=msfm_conf,
        params=dlss_conf["dset"]["training"]["params"],
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross=with_cross,
        apply_norm=dset_common.get("apply_norm", True),
        return_maps=True,
        return_cls=False,
    )

    # same pipeline downsampling the network config implies (None at native n_side)
    smooth_nside, _, parent_output_idx = configuration.resolve_smooth_nside(net_conf, dlss_conf, msfm_conf)

    # dset kwargs: the train/test split (data_conf) plus the performance knobs we are benchmarking
    dset_kwargs = dict(data_conf)
    dset_kwargs.update(
        dict(
            is_eval=False,
            local_batch_size=args.local_batch_size,
            n_readers=args.n_readers,
            n_prefetch=(None if args.n_prefetch < 0 else args.n_prefetch),  # -1 -> AUTOTUNE
            n_workers=(None if args.n_workers < 0 else args.n_workers),  # -1 -> AUTOTUNE
            file_name_shuffle_buffer=args.file_name_shuffle_buffer,
            examples_shuffle_buffer=args.examples_shuffle_buffer,
        )
    )

    dset = pipeline.get_dset(
        tfr_pattern=args.train_tfr_pattern,
        **dset_kwargs,
        input_context=None,  # single, un-sharded pipeline == MirroredStrategy on one node
        downsample_nside=smooth_nside if parent_output_idx is not None else None,
        parent_output_idx=parent_output_idx,
    )

    n_channels = 0
    if with_lensing:
        n_channels += pipeline.n_z_metacal
    if with_clustering:
        n_channels += pipeline.n_z_maglim
    meta = {
        "n_dv_pix": pipeline.n_dv_pix,
        "n_channels": n_channels,
        "downsample_nside": smooth_nside if parent_output_idx is not None else None,
    }
    return dset, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # configs (same as run_training)
    p.add_argument("--train_tfr_pattern", required=True)
    p.add_argument("--net_config", required=True)
    p.add_argument("--probes_config", required=True)
    p.add_argument("--scales_config", required=True)
    p.add_argument("--data_config", required=True)
    p.add_argument("--msfm_config", required=True)
    # the knobs under test
    p.add_argument("--local_batch_size", type=int, default=16)
    p.add_argument("--n_readers", type=int, default=64)
    p.add_argument("--n_prefetch", type=int, default=8, help="-1 -> tf.data.AUTOTUNE")
    p.add_argument("--n_workers", type=int, default=-1, help="-1 -> tf.data.AUTOTUNE (Null in the config)")
    p.add_argument("--file_name_shuffle_buffer", type=int, default=64)
    p.add_argument("--examples_shuffle_buffer", type=int, default=256)
    # measurement
    p.add_argument("--measure_batches", type=int, default=50)
    p.add_argument("--min_warmup_batches", type=int, default=30)
    p.add_argument(
        "--label", type=str, default="", help="free-form tag echoed into the result (e.g. probe/sweep name)"
    )
    p.add_argument("--results_file", type=str, default=None, help="append the JSON result line here")
    args = p.parse_args()

    rss_import = _rss_gb()

    dset, meta = build_dataset(args, None)
    it = iter(dset)

    # warmup: at minimum fill the examples shuffle buffer, then some, so we measure steady state
    warmup = max(args.min_warmup_batches, args.examples_shuffle_buffer // max(args.local_batch_size, 1) + 10)
    t0 = time.time()
    for _ in range(warmup):
        next(it)
    warmup_s = time.time() - t0
    rss_warm = _rss_gb()

    # measure: per-batch times (median is robust to the odd FS stall)
    sampler = RSSSampler()
    sampler.start()
    per_batch = []
    t_start = time.time()
    for _ in range(args.measure_batches):
        tb = time.time()
        next(it)
        per_batch.append(time.time() - tb)
    total_s = time.time() - t_start
    sampler.stop()

    per_batch.sort()
    n = len(per_batch)
    median_bt = per_batch[n // 2]
    p95_bt = per_batch[min(n - 1, int(0.95 * n))]
    examples = args.measure_batches * args.local_batch_size
    thrpt_mean = examples / total_s
    thrpt_median = args.local_batch_size / median_bt
    mb_per_example = meta["n_dv_pix"] * meta["n_channels"] * 4 / 1e6

    result = {
        "label": args.label,
        "local_batch_size": args.local_batch_size,
        "n_readers": args.n_readers,
        "n_prefetch": args.n_prefetch,
        "n_workers": args.n_workers,
        "file_name_shuffle_buffer": args.file_name_shuffle_buffer,
        "examples_shuffle_buffer": args.examples_shuffle_buffer,
        # geometry
        "n_dv_pix": meta["n_dv_pix"],
        "n_channels": meta["n_channels"],
        "downsample_nside": meta["downsample_nside"],
        "mb_per_example": round(mb_per_example, 2),
        # throughput
        "examples_per_s_mean": round(thrpt_mean, 1),
        "examples_per_s_median": round(thrpt_median, 1),
        "mb_per_s_mean": round(thrpt_mean * mb_per_example, 1),
        "median_batch_ms": round(median_bt * 1000, 2),
        "p95_batch_ms": round(p95_bt * 1000, 2),
        # sustainable node step rate: one MirroredStrategy step consumes 4 x local_batch_size
        "max_steps_per_s_4gpu": round(thrpt_mean / (4 * args.local_batch_size), 2),
        # memory (GB)
        "rss_after_import_gb": round(rss_import, 2),
        "rss_after_warmup_gb": round(rss_warm, 2),
        "rss_plateau_gb": round(sampler.peak, 2),
        "vmhwm_peak_gb": round(_vmhwm_gb(), 2),
        # bookkeeping
        "warmup_batches": warmup,
        "warmup_s": round(warmup_s, 1),
        "measure_batches": args.measure_batches,
    }

    line = json.dumps(result)
    print(line, flush=True)
    if args.results_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
