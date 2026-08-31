# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

Benchmark the DeepSphere/ResNet GCNN network configs (network.name == "resnet") for GPU memory fit
and step time, the GCNN analogue of ``benchmark_transformer.py``. Applies the same sizing recipe
(``project_transformer_bench_sizing_recipe``) to the ``configs/deepsphere/prod/{lensing,clustering,
combined}/{maps,maps+cls}.yaml`` lineage.

It builds each model through the *exact* construction path used by run_training.py's resnet branch
— one ``ResNetSummaryNetwork`` wrapping either a single-resolution ``HealpyGCNN`` or (combined
probes) a ``ResNetMultiResEncoder``, with the Cls branch attached iff the config has a ``cls:``
block — but feeds it synthetic random batches, so the measured peak memory and step time reflect
the architecture and per-GPU (local) batch size, isolated from the tfrecord data pipeline.

Two modes, mirroring benchmark_transformer.py:

  * driver (default): iterate over every ``*.yaml`` in a configs dir × a list of per-GPU batch
    sizes, launching one *child* subprocess per (config, batch) so each runs in a fresh process
    (GPU memory fully released between runs, an OOM is contained), then print a markdown overview
    and write a CSV.

  * child (``--single``): build + time a single (config, batch) and emit one JSON line.

Run inside the TensorFlow container on a single GPU, e.g. via an sbatch script that drives each
(config, batch) as its own ``srun --environment=tensorflow`` step, or an interactive
``srun --environment=tensorflow`` allocation for the in-process driver.

Unlike the transformer, every branch here builds one or two ``HealpyGCNN`` instances, which compute
a PyGSP sphere graph and Chebyshev sparse matrices at construction time — a real, CPU-bound,
one-time cost (seconds, scaling with n_neighbors/nside) that happens before the timed loop and is
NOT part of the per-step GPU cost being measured.
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import statistics
import csv
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("NUMBA_WARNINGS", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)

REPOS = "/users/athomsen/dlss/repos"

# Defaults mirroring submissions/clariden/training.sh (v16/rot_in_place, lensing, vmim).
DEFAULTS = {
    "msfm_config": f"{REPOS}/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml",
    "probes_config": f"{REPOS}/y3-deep-lss/configs/probes/lensing.yaml",
    "scales_config": f"{REPOS}/y3-deep-lss/configs/scales/8wl,32gc.yaml",
    "loss_config": f"{REPOS}/y3-deep-lss/configs/loss/vmim.yaml",
    "data_config": f"{REPOS}/y3-deep-lss/configs/data/default.yaml",
    "configs_dir": f"{REPOS}/y3-deep-lss/configs/deepsphere/prod/lensing",
}

# Stand-in for `n_steps: auto`, which run_training resolves from a wall-clock budget this benchmark
# does not build. Only sets the cosine's length; nothing measured here depends on it.
_BENCH_N_STEPS = 100000


# --------------------------------------------------------------------------------------
# child mode: build + time a single (config, batch)
# --------------------------------------------------------------------------------------
def run_single(args):
    import tempfile
    import time
    import numpy as np
    import tensorflow as tf

    # memory growth so get_memory_info("GPU:0")["peak"] is meaningful (cf. run_training.py)
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception:
            pass

    from msfm.utils import input_output, files
    from deep_lss.utils import configuration, optimization
    from deep_lss.models.grid_model import GridLossModel
    from deep_lss.nets import NETWORKS
    from deep_lss.nets.composite.resnet_summary import ResNetSummaryNetwork
    from deep_lss.nets.encoders.maps.gcnn.resnet_multires import ResNetMultiResEncoder
    from deep_lss.nets.layers.cls.embedding import get_cls_branch_kwargs

    net_conf = input_output.read_yaml(args.net_config)
    dlss_conf = configuration.read_split_configs(args.probes_config, args.scales_config)
    loss_conf = input_output.read_yaml(args.loss_config)
    msfm_conf = files.load_config(args.msfm_config)

    assert (
        net_conf["network"]["name"] == "resnet"
    ), f"benchmark_resnet only supports network.name=resnet, got {net_conf['network']['name']}"

    # numerical precision override, same knob as benchmark_transformer.py (resnet configs default
    # to float32 -- DeepSphere's sparse Chebyshev matmuls have no XLA/mixed-precision path).
    precision = args.precision or net_conf["network"].get("precision", "float32")
    if precision not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"Unknown precision '{precision}'; expected float32, float16 or bfloat16")
    if precision != "float32":
        tf.keras.mixed_precision.set_global_policy(f"mixed_{precision}")

    # geometry / channels -- same logic as run_training.py / benchmark_transformer.py
    n_side = msfm_conf["analysis"]["n_side"]
    smooth_nside, smooth_indices, parent_output_idx = configuration.resolve_smooth_nside(
        net_conf, dlss_conf, msfm_conf
    )

    with_lensing = dlss_conf["dset"]["common"]["with_lensing"]
    with_clustering = dlss_conf["dset"]["common"]["with_clustering"]
    with_cross = dlss_conf["dset"]["common"].get("with_cross", False)
    n_z_bins = 0
    if with_lensing:
        n_z_bins += len(msfm_conf["survey"]["metacal"]["z_bins"])
    if with_clustering:
        n_z_bins += len(msfm_conf["survey"]["maglim"]["z_bins"])
    if with_cross:
        n_z_bins += len(msfm_conf["survey"]["metacal"]["z_bins"]) * len(msfm_conf["survey"]["maglim"]["z_bins"])

    params = dlss_conf["dset"]["training"]["params"]
    n_params = len(params)
    n_output = loss_conf["mutual_info_loss"]["dim_summary_fac"] * n_params

    batch_size = args.batch_size
    tmp_dir = tempfile.mkdtemp(prefix="bench_resnet_")
    smoothing_kwargs = configuration.get_smoothing_kwargs(
        "mutual_info", msfm_conf, dlss_conf, net_conf, dir_base=tmp_dir
    )
    is_multires_gcnn = "split_probes" in smoothing_kwargs

    cls_conf = net_conf["network"].get("cls", None)
    return_cls = cls_conf is not None
    cls_transform = (cls_conf or {}).get("transform", "asinh_per_feature")

    input_norm = bool(net_conf["network"].get("input_norm", False))

    # `n_steps: auto` is resolved by run_training from a WallClockBudget, which this benchmark has
    # no equivalent of -- get_optimizer would then compute `"auto" - warmup_steps`. Substitute a
    # concrete count: it only sets the cosine's length, and nothing here depends on the LR schedule
    # (step time and peak memory are what is measured). Without this every prod and bench_v8+ config
    # is unbenchmarkable, which is most of them.
    if net_conf["training"].get("n_steps") == "auto":
        net_conf["training"]["n_steps"] = _BENCH_N_STEPS
        print(f"n_steps: auto -> {_BENCH_N_STEPS} for the benchmark (LR schedule is not measured)")

    strategy = tf.distribute.get_strategy()  # default single-device strategy
    graph_build_t0 = time.perf_counter()
    with strategy.scope():
        optimizer = optimization.get_optimizer(net_conf, "mutual_info", False)

        net_spec = NETWORKS["resnet"](
            out_features=n_output,
            smoothing_kwargs=None if is_multires_gcnn else smoothing_kwargs,
            **({"input_norm": True} if input_norm and not is_multires_gcnn else {}),
            **({"smoothing_external": True} if is_multires_gcnn else {}),
            **net_conf["network"]["kwargs"],
        )

        # the cls: block is the only thing that differs between the two paths
        cls_kwargs = get_cls_branch_kwargs(cls_conf, msfm_conf, dlss_conf, n_side, cls_transform)
        map_encoder = None
        if is_multires_gcnn:
            map_encoder = ResNetMultiResEncoder(
                smoothing_kwargs=smoothing_kwargs,
                layers=net_spec.get_conv_layers(),
                nside=smooth_nside,
                n_neighbors=net_conf["network"]["n_neighbors"],
                max_batch_size=batch_size,
                input_norm=input_norm,
                fusion=net_conf["network"].get("fusion", "concat"),
                injection_conv_layers=net_conf["network"].get("injection_conv_layers", 0),
                injection_conv_kwargs={
                    "poly_degree": net_conf["network"]["kwargs"].get("poly_degree", 5),
                    "conv_type": net_conf["network"]["kwargs"].get("conv_type", "cheby"),
                },
                # also a top-level network key, for the same reason as injection_conv_layers
                fusion_width=net_conf["network"].get("fusion_width", None),
                fuse_act=net_conf["network"].get("fuse_act", None),
            )
        network = ResNetSummaryNetwork(
            conv_layers=None if is_multires_gcnn else net_spec.get_conv_layers(),
            regression_head_layers=net_spec.get_head_layers_no_flatten(),
            n_side=None if is_multires_gcnn else smooth_nside,
            indices=None if is_multires_gcnn else smooth_indices,
            n_neighbors=net_conf["network"]["n_neighbors"],
            max_batch_size=batch_size,
            initial_Fin=None if is_multires_gcnn else n_z_bins,
            # the Cls branch, or {} for a maps-only config
            **cls_kwargs,
            map_feature_dim=net_conf["network"].get("map_feature_dim", None),
            map_encoder=map_encoder,
            map_pool=net_conf["network"].get("map_pool", None),
            map_pool_multiscale=net_conf["network"].get("map_pool_multiscale", False),
        )
        if network.gcnn is not None:
            network.gcnn.build((batch_size, len(smooth_indices), n_z_bins))
        maps_trace = tf.zeros((2, len(smooth_indices), n_z_bins))
        if return_cls:
            cls_trace = tf.zeros((2, 3 * n_side, len(cls_kwargs["l_min_per_pair"])))
            network((maps_trace, cls_trace), training=False)
        else:
            network(maps_trace, training=False)

        model = GridLossModel(
            network=network,
            n_side=None,
            indices=None,
            n_neighbors=net_conf["network"]["n_neighbors"],
            z_bank_size=net_conf["network"]["z_bank_size"],
            max_checkpoints=net_conf["network"]["max_checkpoints"],
            optimizer=optimizer,
            input_shape=None,
            max_batch_size=batch_size,
            checkpoint_dir=os.path.join(tmp_dir, "checkpoint"),
            summary_dir=os.path.join(tmp_dir, "summary"),
            restore_checkpoint=False,
            strategy=strategy,
            xla=False,
            summary_every=10**9,  # effectively never write summaries during the benchmark
        )

        mutual_info_kwargs = {
            "dim_summary": n_output,
            **loss_conf["mutual_info_loss"]["regu"],
            "mutual_info_estimator": loss_conf["mutual_info_loss"]["estimator"],
            "mutual_info_kwargs": loss_conf["mutual_info_loss"]["kwargs"],
        }
        model.setup_grid_loss_step(
            loss="mutual_info",
            batch_size=batch_size,
            dim_theta=n_params,
            dim_x=None,
            dim_channels=None,
            **mutual_info_kwargs,
            **net_conf["optimization"]["gradient_clipping"],
        )
    graph_build_s = time.perf_counter() - graph_build_t0

    n_param_weights = int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))

    # synthetic batch (content is irrelevant for memory/timing)
    if return_cls:
        x = (
            tf.random.normal((batch_size, len(smooth_indices), n_z_bins)),
            tf.random.normal((batch_size, 3 * n_side, len(cls_kwargs["l_min_per_pair"]))),
        )
    else:
        x = tf.random.normal((batch_size, len(smooth_indices), n_z_bins))
    cosmo = tf.random.uniform((batch_size, n_params))

    n_warmup, n_timed = 3, 10
    for _ in range(n_warmup):
        model.grid_train_step(x, cosmo)

    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass

    step_times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        loss = model.grid_train_step(x, cosmo)
        _ = float(loss)  # block until the step completes
        step_times.append(time.perf_counter() - t0)

    try:
        peak_gb = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9
    except Exception:
        peak_gb = float("nan")

    step_ms = statistics.median(step_times) * 1e3
    throughput = batch_size / (step_ms / 1e3)

    result = {
        "config": os.path.basename(args.net_config),
        "batch_size": batch_size,
        "precision": precision,
        "multires": is_multires_gcnn,
        "return_cls": return_cls,
        "n_pix": len(smooth_indices),
        "graph_build_s": round(graph_build_s, 1),
        "params_M": round(n_param_weights / 1e6, 3),
        "peak_gb": round(peak_gb, 2),
        "step_ms": round(step_ms, 1),
        "throughput": round(throughput, 1),
        "status": "OK",
    }
    print("BENCH_JSON " + json.dumps(result), flush=True)


# --------------------------------------------------------------------------------------
# driver mode: sweep configs × batch sizes via child subprocesses
# --------------------------------------------------------------------------------------
def run_driver(args):
    configs = sorted(glob.glob(os.path.join(args.configs_dir, "*.yaml")))
    if not configs:
        sys.exit(f"No configs found in {args.configs_dir}")
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    print(
        f"Benchmarking {len(configs)} configs × {len(batch_sizes)} batch sizes " f"{batch_sizes} on a single GPU\n",
        flush=True,
    )

    rows = []
    for cfg in configs:
        for bs in batch_sizes:
            name = os.path.basename(cfg)
            print(f">>> {name}  batch={bs} ...", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--single",
                "--net_config",
                cfg,
                "--batch_size",
                str(bs),
                "--msfm_config",
                args.msfm_config,
                "--probes_config",
                args.probes_config,
                "--scales_config",
                args.scales_config,
                "--loss_config",
                args.loss_config,
                "--data_config",
                args.data_config,
            ]
            if args.precision:
                cmd += ["--precision", args.precision]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            row = None
            for line in proc.stdout.splitlines():
                if line.startswith("BENCH_JSON "):
                    row = json.loads(line[len("BENCH_JSON ") :])
                    break
            if row is None:
                blob = (proc.stderr + proc.stdout).lower()
                if "resourceexhausted" in blob or "out of memory" in blob:
                    status = "OOM"
                elif "invalid configuration argument" in blob or "non-ok-status" in blob:
                    status = "KERNEL"
                else:
                    status = "ERROR"
                row = {
                    "config": name,
                    "batch_size": bs,
                    "precision": args.precision or "config",
                    "multires": "-",
                    "return_cls": "-",
                    "n_pix": "-",
                    "graph_build_s": "-",
                    "params_M": "-",
                    "peak_gb": "-",
                    "step_ms": "-",
                    "throughput": "-",
                    "status": status,
                }
                if status == "ERROR":
                    noise = ("+ptx85", "gpu_timer.cc")
                    clean = [ln for ln in proc.stderr.strip().splitlines() if not any(n in ln for n in noise)]
                    print("    ERROR (stderr tail):\n" + "\n".join(clean[-15:]) + "\n", flush=True)
            print(f"    {row['status']}  peak={row['peak_gb']} GB  step={row['step_ms']} ms\n", flush=True)
            rows.append(row)

    _write_outputs(rows, args.out_dir)


def _write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cols = [
        "config",
        "batch_size",
        "precision",
        "multires",
        "return_cls",
        "n_pix",
        "graph_build_s",
        "params_M",
        "peak_gb",
        "step_ms",
        "throughput",
        "status",
    ]

    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="-")
        w.writeheader()
        w.writerows(rows)

    headers = [
        "config",
        "batch",
        "prec",
        "multires",
        "cls",
        "n_pix",
        "graph(s)",
        "params(M)",
        "peak(GB)",
        "step(ms)",
        "ex/s",
        "status",
    ]
    keys = cols
    widths = [max(len(h), max(len(str(r.get(k, "-"))) for r in rows)) for h, k in zip(headers, keys)]

    def fmt(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, widths)) + " |"

    lines = [fmt(headers), "|-" + "-|-".join("-" * w for w in widths) + "-|"]
    lines += [fmt([r.get(k, "-") for k in keys]) for r in rows]
    table = "\n".join(lines)

    md_path = os.path.join(out_dir, "benchmark_results.md")
    with open(md_path, "w") as f:
        f.write("# DeepSphere/ResNet GCNN benchmark\n\n" + table + "\n")

    print("\n" + table + "\n")
    print(f"Wrote {csv_path}\nWrote {md_path}")


def run_aggregate(args):
    """Read a JSONL of per-(config, batch) results and write the CSV/markdown overview.

    Stdlib only (no TensorFlow), so it can run anywhere. Used by an sbatch script that drives each
    (config, batch) as its own ``srun --environment=tensorflow`` step (the only form that reliably
    has TF inside the CSCS container) and collects the JSON lines here.
    """
    rows = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        sys.exit(f"No result rows found in {args.jsonl}")
    _write_outputs(rows, args.out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single", action="store_true", help="child mode: benchmark one (config, batch)")
    p.add_argument("--aggregate", action="store_true", help="aggregate a JSONL of results into CSV/markdown")
    p.add_argument("--jsonl", type=str, help="path to the JSONL of results (aggregate mode)")
    p.add_argument("--net_config", type=str, help="path to a single net config (child mode)")
    p.add_argument("--batch_size", type=int, default=16, help="per-GPU (local) batch size (child mode)")
    p.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=("float32", "float16", "bfloat16"),
        help="override the config's network.precision (child mode); default uses the config",
    )
    p.add_argument("--configs_dir", type=str, default=DEFAULTS["configs_dir"], help="dir of net configs (driver)")
    p.add_argument("--batch_sizes", type=str, default="16,32,64", help="comma-separated batch sizes (driver)")
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="where to write the CSV/markdown overview",
    )
    p.add_argument("--msfm_config", type=str, default=DEFAULTS["msfm_config"])
    p.add_argument("--probes_config", type=str, default=DEFAULTS["probes_config"])
    p.add_argument("--scales_config", type=str, default=DEFAULTS["scales_config"])
    p.add_argument("--loss_config", type=str, default=DEFAULTS["loss_config"])
    p.add_argument("--data_config", type=str, default=DEFAULTS["data_config"])
    args = p.parse_args()

    if args.single:
        if not args.net_config:
            p.error("--net_config is required with --single")
        run_single(args)
    elif args.aggregate:
        if not args.jsonl:
            p.error("--jsonl is required with --aggregate")
        run_aggregate(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
