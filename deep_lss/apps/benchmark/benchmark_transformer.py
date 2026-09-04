# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Benchmark the nested hierarchical local-window transformer (TransformerSummaryNetwork)
hyperparameter configs for GPU memory fit and step time.

It builds each model through the *exact* construction path used by run_training.py (the
``is_transformer`` branch: real HealpySmoothing front-end + nested tokenizer + transformer +
regression head + GridLossModel with the variational mutual-information loss) but feeds it
synthetic random batches, so the measured peak memory and step time reflect the architecture
and per-GPU (local) batch size, isolated from the tfrecord data pipeline.

The network is always built WITHOUT the Cls branch, so ``peak_gb``/``step_ms`` measure the map
branch alone even for a config that carries a ``cls:`` block. That is the quantity the sizing
recipe wants; the Cls head is small and its cost does not scale with the batch geometry.

Two modes:

  * driver (default): iterate over every ``*.yaml`` in a hyperparameters dir × a list of
    per-GPU batch sizes, launching one *child* subprocess per (config, batch) so each runs
    in a fresh process (GPU memory fully released between runs, an OOM is contained), then
    print a markdown overview and write a CSV.

  * child (``--single``): build + time a single (config, batch) and emit one JSON line.

Run inside the TensorFlow container on a single GPU, e.g. via submissions/clariden/benchmark.sh
or an interactive ``srun --environment=tensorflow`` allocation.
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
    "configs_dir": f"{REPOS}/y3-deep-lss/configs/maps/prod/transformer/lensing",
}


# --------------------------------------------------------------------------------------
# child mode: build + time a single (config, batch)
# --------------------------------------------------------------------------------------
def run_single(args):
    import tempfile
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
    from deep_lss.nets.composite.transformer_summary import TransformerSummaryNetwork
    from deep_lss.nets.heads.regression_head import get_regression_head

    net_conf = input_output.read_yaml(args.net_config)
    dlss_conf = configuration.read_split_configs(args.probes_config, args.scales_config)
    loss_conf = input_output.read_yaml(args.loss_config)
    msfm_conf = files.load_config(args.msfm_config)

    assert net_conf["network"]["name"] == "nested_transformer", (
        f"benchmark_transformer only supports the nested_transformer net, got " f"{net_conf['network']['name']}"
    )

    # optional XLA override: fuse the tokenizer->transformer body (smoothing stays eager).
    # Lets us probe XLA without throwaway config copies.
    if args.jit_compile_body:
        net_conf["network"]["jit_compile_body"] = True

    # optional fp32-softmax override: toggle the float32 attention-softmax upcast without a
    # throwaway config copy. None -> use the config (default True inside the net).
    if args.fp32_softmax is not None:
        net_conf["network"].setdefault("kwargs", {})["fp32_softmax"] = args.fp32_softmax == "true"

    # numerical precision: --precision overrides the config's network.precision (default float32).
    # Set the global policy before building the network so the HealpySmoothing sparse kernel and
    # every layer adopt it (mirrors run_training.py).
    precision = args.precision or net_conf["network"].get("precision", "float32")
    if precision not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"Unknown precision '{precision}'; expected float32, float16 or bfloat16")
    if precision != "float32":
        tf.keras.mixed_precision.set_global_policy(f"mixed_{precision}")

    # geometry / channels — same logic as run_training.py
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)
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
    tmp_dir = tempfile.mkdtemp(prefix="bench_transformer_")
    smoothing_kwargs = configuration.get_smoothing_kwargs(
        "mutual_info", msfm_conf, dlss_conf, net_conf, dir_base=tmp_dir
    )

    strategy = tf.distribute.get_strategy()  # default single-device strategy
    with strategy.scope():
        optimizer = optimization.get_optimizer(net_conf, "mutual_info", False)

        token_nside = net_conf["network"]["token_nside"]
        transformer_kwargs = net_conf["network"]["kwargs"]
        jit_compile_body = net_conf["network"].get("jit_compile_body", False)

        head_conf = net_conf["network"].get("head", {}) or {}
        network = TransformerSummaryNetwork(
            smoothing_kwargs=smoothing_kwargs,
            smooth_indices=smooth_indices,
            nside=smooth_nside,
            token_nside=token_nside,
            in_channels=n_z_bins,
            # absent = None = no projection; see the note in run_training.py
            map_feature_dim=net_conf["network"].get("map_feature_dim", None),
            transformer_kwargs=transformer_kwargs,
            # maps-only (no Cls kwargs): the head runs straight on the map feature
            regression_head_layers=get_regression_head(
                out_features=n_output,
                head_type="dense",
                dropout_rate=head_conf.get("dropout_rate", None),
            )[1:],
            jit_compile_body=jit_compile_body,
            masked_attention=bool(net_conf["network"].get("masked_attention", False)),
        )
        network(tf.zeros((2, len(smooth_indices), n_z_bins)), training=False)

        model = GridLossModel(
            network=network,
            n_side=None,
            indices=None,
            n_neighbors=None,
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

    n_param_weights = int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))
    # The tokenizer now lives inside network.map_encoder (build_map_encoder refactor). The
    # single-resolution encoder exposes one `tokenizer`; the multi-resolution encoder
    # (HealpixMultiResMapEncoder, combined split_probes) exposes a list `tokenizers`, whose fine
    # group (index 0) sets the top-level token count. Report that primary tokenizer's geometry.
    encoder = network.map_encoder
    tokenizer = getattr(encoder, "tokenizer", None) or encoder.tokenizers[0]
    n_tokens = int(tokenizer.num_top_level_tokens)
    pix_per_token = int(tokenizer.num_pixels // max(n_tokens, 1))

    # synthetic batch (content is irrelevant for memory/timing)
    x = tf.random.normal((batch_size, len(smooth_indices), n_z_bins))
    cosmo = tf.random.uniform((batch_size, n_params))

    import time

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
        "fp32_softmax": net_conf["network"].get("kwargs", {}).get("fp32_softmax", True),
        "n_tokens": n_tokens,
        "pix_per_token": pix_per_token,
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
            if args.jit_compile_body:
                cmd.append("--jit_compile_body")
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
                    # CUDA grid-dim limit in the attention softmax kernel: the effective
                    # batch-like dim (B*N*...) is too large to launch, even though the model
                    # would fit in memory. Distinct from an OOM.
                    status = "KERNEL"
                else:
                    status = "ERROR"
                row = {
                    "config": name,
                    "batch_size": bs,
                    "precision": args.precision or "config",
                    "n_tokens": "-",
                    "pix_per_token": "-",
                    "params_M": "-",
                    "peak_gb": "-",
                    "step_ms": "-",
                    "throughput": "-",
                    "status": status,
                }
                # surface the tail of stderr for unexpected ERROR rows to aid debugging,
                # filtering the harmless '+ptx85' / gpu_timer spam that floods stderr
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
        "n_tokens",
        "pix_per_token",
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

    headers = ["config", "batch", "prec", "N", "pix/tok", "params(M)", "peak(GB)", "step(ms)", "ex/s", "status"]
    keys = [
        "config",
        "batch_size",
        "precision",
        "n_tokens",
        "pix_per_token",
        "params_M",
        "peak_gb",
        "step_ms",
        "throughput",
        "status",
    ]
    # tolerate rows from older JSONL that predate the precision column
    widths = [max(len(h), max(len(str(r.get(k, "-"))) for r in rows)) for h, k in zip(headers, keys)]

    def fmt(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, widths)) + " |"

    lines = [fmt(headers), "|-" + "-|-".join("-" * w for w in widths) + "-|"]
    lines += [fmt([r.get(k, "-") for k in keys]) for r in rows]
    table = "\n".join(lines)

    md_path = os.path.join(out_dir, "benchmark_results.md")
    with open(md_path, "w") as f:
        f.write("# Nested-transformer benchmark\n\n" + table + "\n")

    print("\n" + table + "\n")
    print(f"Wrote {csv_path}\nWrote {md_path}")


def run_aggregate(args):
    """Read a JSONL of per-(config, batch) results and write the CSV/markdown overview.

    Stdlib only (no TensorFlow), so it can run anywhere. Used by submissions/clariden/benchmark.sh,
    which drives each (config, batch) as its own ``srun --environment=tensorflow`` step (the only
    form that reliably has TF inside the CSCS container) and collects the JSON lines here.
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
        "--jit_compile_body",
        action="store_true",
        help="force XLA on the tokenizer->transformer body, overriding the config (child mode)",
    )
    p.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=("float32", "float16", "bfloat16"),
        help="override the config's network.precision (child mode); default uses the config",
    )
    p.add_argument(
        "--fp32_softmax",
        type=str,
        default=None,
        choices=("true", "false"),
        help="override network.kwargs.fp32_softmax (child mode); default uses the config (True)",
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
        # in-process driver (subprocess children). Works under a *direct* interactive
        # `srun --environment=tensorflow`; under sbatch use benchmark.sh instead, which
        # drives each child as its own srun step (see run_aggregate docstring).
        run_driver(args)


if __name__ == "__main__":
    main()
