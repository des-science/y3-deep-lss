# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Paired comparison of trained runs: the fixed recipe for "is architecture B better than A?".

This exists because the comparison kept being re-improvised per benchmark round, and the mistakes
were always in the bookkeeping rather than in the science. Two of them changed conclusions:

  * comparing MARGINAL medians instead of pairing per mock. Per-mock FoM spans ~12x across the mock
    set and that spread is common to both runs (they correlate at r~0.95), so the unpaired ratio is
    both a different functional and ~2x noisier. bench_v5 read as a win unpaired and a loss paired.
  * calling a null result on a mock set too small to resolve the effect. The legacy 16-mock
    `chain_grid_*` route has a ~2.4% floor and cannot see anything below ~7%.

So the gates below are asserts, not advice: a comparison that cannot be made validly raises instead
of silently returning a number.

Deliberately depends only on numpy/h5py/yaml (+ tensorboard, lazily, for `loss/vali_total`). It does
NOT import msi -- y3-deep-lss installs before msi, so the FoM is reimplemented here rather than taken
from msi.utils.diagnostics. The definition is the same: FoM = det(cov(p1, p2))**-0.5.

What this module does NOT answer: robustness (posterior bias on the systematics-variation mocks) and
estimator validity (SBC/TARP/HPD). Those are separate questions with separate instruments; a good FoM
ratio here is not evidence about either. DES FoM is unsigned under misspecification and is not
computed at all.

Typical use::

    python -m deep_lss.utils.run_comparison --root <runs>/v17/baseline/maps/combined \\
        --reference t2_cls bench_v5_pool_head_w64 bench_v5_convnext_droppath
"""

import argparse
import glob
import json
import os
import re

import h5py
import numpy as np

# Run-to-run (training-seed) reproducibility floor on the paired FoM ratio, measured from a near-twin
# pair: identical config, different seed, +5.5% steps -> 1.015. It is a per-RUN effect, so it does NOT
# shrink with more mocks -- the mock bootstrap CI can be 2x tighter than this and mean nothing.
SEED_FLOOR = 0.015

# Config fields that must match for a comparison to be meaningful at all. The VMIM bound's tightness
# depends on head expressiveness and theta conditioning, not only on the compression, so a flow-head
# run and a GMM-head run are not on the same scale even for the same probe.
_GATE_FIELDS = {
    "probe": ("dlss", "dset", "common"),
    "density_estimator": ("loss", "mutual_info_loss", "density_estimator"),
    "standardize_theta": ("loss", "mutual_info_loss", "standardize_theta"),
    "dim_summary_fac": ("loss", "mutual_info_loss", "dim_summary_fac"),
    "loss_function": ("loss", "loss_function"),
    "signal_indices": ("data", "signal_indices"),
}

_PROBE_KEYS = ("with_lensing", "with_clustering", "with_cross_probe", "with_cross_z")


def _dig(cfg, path, default=None):
    """Fetch a nested config key by path tuple, returning `default` if any level is missing."""
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load_run_config(run_dir):
    """Load a run's own `configs.yaml` snapshot (never the repo config -- runs drift from it).

    yaml is imported lazily so the FoM/pairing machinery stays importable in a bare environment; the
    config is still required, because both the comparability gate and the theta column order come
    from it and neither may be guessed.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - environment issue, not logic
        raise ImportError(
            "pyyaml is required to read a run's configs.yaml (it is a declared deep_lss dependency). "
            "Without it the comparability gate and the theta parameter order cannot be established, "
            "and neither may be assumed."
        )
    path = os.path.join(run_dir, "configs.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no configs.yaml in {run_dir}; cannot verify comparability")
    with open(path) as f:
        return yaml.safe_load(f)


def comparability_key(cfg):
    """Reduce a run config to the fields that must agree across a comparison."""
    key = {}
    for name, path in _GATE_FIELDS.items():
        value = _dig(cfg, path)
        if name == "probe":
            value = tuple(bool(_dig(cfg, path + (k,), False)) for k in _PROBE_KEYS)
        key[name] = value
    return key


def param_names(cfg):
    """Return the label order the run was trained with, e.g. [Om, s8, w0, Aia, n_Aia, bg1..bg4]."""
    params = _dig(cfg, ("dlss", "dset", "eval", "grid", "params"))
    if not params:
        raise KeyError("dlss.dset.eval.grid.params missing; cannot map theta columns to parameters")
    return list(params)


def _flow_dir_steps(path):
    """Training step count encoded in an `*flow_<steps>` directory name, or -1 if absent."""
    match = re.search(r"_(\d+)$", os.path.basename(os.path.dirname(path)))
    return int(match.group(1)) if match else -1


def find_flow_dir(run_dir):
    """Locate the run's inference output holding mcmc_samples.h5; return (path, all_candidates).

    Globs `*flow_*` rather than hardcoding a name: v8/v33-style runs save under `ensemble_flow_<steps>`
    and not `likelihood_flow_<steps>`.

    A run may hold SEVERAL evaluated checkpoints, and the choice is not cosmetic -- v3_cls carries
    150k/200k/310k and its paired ratio moves 0.302 -> 0.270 between them. The highest step count wins
    (the final model), but every candidate is returned so callers can surface the ambiguity instead of
    resolving it silently, which is how an arbitrary `glob(...)[0]` once produced a wrong number.
    """
    candidates = sorted(glob.glob(os.path.join(run_dir, "*flow_*", "mcmc_samples.h5")), key=_flow_dir_steps)
    if not candidates:
        raise FileNotFoundError(
            f"no */mcmc_samples.h5 under {run_dir} -- the inference stage did not run, or the job "
            f"timed out before writing it"
        )
    return candidates[-1], candidates


def _param_column(theta, params, name):
    """Extract one parameter (or the derived S8) from theta, shaped (..., n_params)."""
    if name in params:
        return theta[..., params.index(name)].astype(np.float64)
    if name == "S8":
        for needed in ("Om", "s8"):
            if needed not in params:
                raise KeyError(f"S8 needs '{needed}' among the trained params {params}")
        Om = theta[..., params.index("Om")].astype(np.float64)
        s8 = theta[..., params.index("s8")].astype(np.float64)
        return s8 * np.sqrt(Om / 0.3)
    raise KeyError(f"unknown parameter {name!r}; trained params are {params}")


def fom_per_mock(theta_sample, params, pair=("Om", "S8")):
    """FoM = det(cov(p1, p2))**-0.5 per mock, vectorised over the mock axis.

    `theta_sample` is (n_samples, n_mocks, n_params) as written by msi's coverage stage. Higher is
    better. Signed ONLY because these mocks are correctly specified -- the same statistic on a
    misspecified observation (DES) carries no sign and must not be ranked on.
    """
    x = _param_column(theta_sample, params, pair[0])
    y = _param_column(theta_sample, params, pair[1])
    n = x.shape[0]
    xc, yc = x - x.mean(0), y - y.mean(0)
    cxx = (xc * xc).sum(0) / (n - 1)
    cyy = (yc * yc).sum(0) / (n - 1)
    cxy = (xc * yc).sum(0) / (n - 1)
    det = cxx * cyy - cxy ** 2
    if np.any(det <= 0):
        raise ValueError(f"non-positive covariance determinant for {int((det <= 0).sum())} mocks")
    return det ** -0.5


def load_run(run_dir, pair=("Om", "S8")):
    """Load one run and reduce it to (real_idx, fom, config, provenance).

    Only the per-mock FoM is retained: theta_sample is ~360 MB per run, so keeping it for a table of
    ten runs is pointless. `provenance` records which checkpoint was used and how many were available.
    """
    cfg = load_run_config(run_dir)
    params = param_names(cfg)
    flow_file, candidates = find_flow_dir(run_dir)
    provenance = {
        "steps": _flow_dir_steps(flow_file),
        "n_checkpoints": len(candidates),
        "other_steps": [_flow_dir_steps(c) for c in candidates[:-1]],
    }
    with h5py.File(flow_file, "r") as h:
        if "real_idx" not in h:
            raise KeyError(
                f"{run_dir}: mcmc_samples.h5 has no real_idx -- it was written without i_sobol/i_noise, "
                f"so its mocks cannot be paired (only positional matching would be possible, which is "
                f"exactly the bug this module exists to prevent)"
            )
        real_idx = h["real_idx"][:]
        fom = fom_per_mock(h["theta_sample"][:], params, pair=pair)
    return real_idx, fom, cfg, provenance


def describe_mock_set(real_idx):
    """Summarise the mock-set index structure and say whether the bootstrap must be clustered.

    The three indices are not interchangeable (msfm/apps/run_grid_postprocessing.py):
      i_sobol  -- the cosmological Sobol point
      i_signal -- the Latin-hypercube row of ASTROPHYSICAL params, jointly with (patch, permutation);
                  the LH is seeded per cosmology, so the same i_signal at a different i_sobol is a
                  different astro draw
      i_noise  -- noise realization only
    Only rows differing in i_noise alone share both cosmological and astrophysical parameters. If a set
    holds several rows per cosmology those rows are correlated, and a naive row bootstrap understates
    the CI (100 cosmologies x 10 noise would make it ~3x too narrow).
    """
    i_sobol = real_idx[:, 0]
    _, counts = np.unique(i_sobol, return_counts=True)
    return {
        "n_mocks": int(real_idx.shape[0]),
        "n_cosmologies": int(counts.size),
        "rows_per_cosmology": (int(counts.min()), int(counts.max())),
        "n_signal": int(np.unique(real_idx[:, 1]).size),
        "n_noise": int(np.unique(real_idx[:, 2]).size),
        "clustered": bool(counts.max() > 1),
    }


def align_to(reference_idx, other_idx):
    """Return the permutation placing `other_idx`'s rows onto `reference_idx`'s order.

    Matches on the FULL (i_sobol, i_signal, i_noise) tuple. Never row position -- two runs can write
    the same mocks in different orders and nothing errors, you just compare mock 7 against mock 312
    (this has already caused two tainted result sets). Never i_sobol alone either, since i_signal
    carries the astrophysical draw.
    """
    ref_map = {tuple(row): i for i, row in enumerate(reference_idx.tolist())}
    if len(ref_map) != len(reference_idx):
        raise ValueError("reference mock set contains duplicate real_idx tuples")
    missing = [row for row in other_idx.tolist() if tuple(row) not in ref_map]
    if missing:
        raise ValueError(
            f"{len(missing)} mocks are absent from the reference set (first: {missing[0]}); the runs "
            f"were evaluated on different mocks and cannot be paired"
        )
    return np.array([ref_map[tuple(row)] for row in other_idx.tolist()])


def paired_ratio(fom_reference, fom_run, i_sobol=None, n_boot=4000, seed=0):
    """Paired FoM ratio of `fom_run` against `fom_reference`, both in reference order.

    Returns median of the per-mock ratios (NOT the ratio of medians -- a different functional), a
    bootstrap CI over mocks, and the fraction of mocks won. When `i_sobol` is given and repeats, the
    bootstrap resamples cosmologies rather than rows.
    """
    ratio = fom_run / fom_reference
    rng = np.random.default_rng(seed)
    if i_sobol is not None and np.unique(i_sobol).size < len(i_sobol):
        groups = [np.where(i_sobol == s)[0] for s in np.unique(i_sobol)]
        draws = rng.integers(0, len(groups), (n_boot, len(groups)))
        boot = np.array([np.median(ratio[np.concatenate([groups[j] for j in row])]) for row in draws])
    else:
        boot = np.median(rng.choice(ratio, (n_boot, len(ratio))), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    median = float(np.median(ratio))
    return {
        "ratio": median,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "win_frac": float((ratio > 1).mean()),
        "wash": bool(abs(median - 1) < SEED_FLOOR),
    }


def read_vali_total_wandb(run_dir, tag="loss/vali_total"):
    """Final value of a scalar from the run's wandb summary, or None if unavailable.

    Fallback for environments without tensorboard (a login node has neither tensorboard nor pip). The
    run writes `wandb/run-<ts>-<id>/files/wandb-summary.json`, a plain JSON of each logged scalar's LAST
    value -- exactly what the event-file path returns, and it agrees to all printed digits where both are
    readable.

    A chained run leaves one directory per job under the SAME wandb id, so the summaries are not
    interchangeable: an early job's file holds the value at its own stopping point. The one with the
    highest `global_step` is the final model. `steps` (from the evaluated flow directory) is compared
    against it when given, because a stale or copied wandb directory belonging to a different run is a
    real failure mode here -- t1_cls's lensing twin was created by copying a run directory wholesale and
    inherited its wandb id.
    """
    best = None
    for path in glob.glob(os.path.join(run_dir, "wandb", "run-*", "files", "wandb-summary.json")):
        try:
            with open(path) as f:
                summary = json.load(f)
        except (OSError, ValueError):
            continue
        if tag not in summary:
            continue
        if best is None or summary.get("global_step", -1) > best.get("global_step", -1):
            best = summary
    if best is None:
        return None
    return float(best[tag]), best.get("global_step", None)


def read_vali_total(run_dir, tag="loss/vali_total", steps=None):
    """Final value of a scalar from the run's TF2 event files, or None if unavailable.

    TF2 writes scalars into `value.tensor`, not `value.simple_value`, so this goes through
    tensorboard's EventAccumulator. Decoding uses tensorboard.util.tensor_util rather than
    tf.make_ndarray so it also works in environments with tensorboard but no TensorFlow (login nodes).

    Falls back to the wandb summary when tensorboard is absent, so the column is not silently `n/a` on
    a login node -- vali_total is half the ranking recipe and dropping it changes what gets promoted.
    """
    summary_dir = os.path.join(run_dir, "summary")
    if not os.path.isdir(summary_dir):
        return None
    try:
        from tensorboard.backend.event_processing import event_accumulator
        from tensorboard.util import tensor_util
    except ImportError:
        fallback = read_vali_total_wandb(run_dir, tag=tag)
        if fallback is None:
            return None
        value, wandb_step = fallback
        if steps is not None and wandb_step is not None and wandb_step != steps:
            raise ValueError(
                f"{run_dir}: wandb summary is at step {wandb_step} but the evaluated checkpoint is at "
                f"{steps}; the wandb directory does not belong to this checkpoint and its vali_total "
                f"may not be this run's"
            )
        return value
    acc = event_accumulator.EventAccumulator(summary_dir, size_guidance={event_accumulator.TENSORS: 0})
    acc.Reload()
    if tag not in acc.Tags().get("tensors", []):
        return None
    events = acc.Tensors(tag)
    if not events:
        return None
    return float(tensor_util.make_ndarray(events[-1].tensor_proto))


def compare_runs(run_dirs, reference, pair=("Om", "S8"), n_boot=4000, seed=0, strict=True):
    """Run the full recipe and return (rows, mock_set_info).

    Steps, in order, with the first two as gates:
      0. comparability -- probe, head type, theta standardization, summary-dim factor, loss, split
      1. the inference stage produced mcmc_samples.h5 with real_idx (enforced by load_run)
      2. mock-set identity across runs + index structure
      3. pair on the full real_idx tuple
      4. paired FoM ratio + vali_total
      5. flag anything inside the seed floor as a wash

    `strict=False` downgrades the comparability gate to a warning list, for the deliberate case of
    comparing across head types while knowing the number is not on one scale.
    """
    reference_dir = reference
    ordered = list(run_dirs)
    if reference_dir not in ordered:
        ordered = [reference_dir] + ordered

    loaded = {d: load_run(d, pair=pair) for d in ordered}
    ref_idx, ref_fom, ref_cfg, _ = loaded[reference_dir]

    # --- gate 0: comparability -------------------------------------------------------------------
    ref_key = comparability_key(ref_cfg)
    warnings = []
    for d, (_, _, cfg, prov) in loaded.items():
        if prov["n_checkpoints"] > 1:
            warnings.append(
                f"{os.path.basename(os.path.normpath(d))} has {prov['n_checkpoints']} evaluated "
                f"checkpoints (using {prov['steps']}, also present: {prov['other_steps']}) -- the "
                f"choice moves the ratio, so state which one the number refers to"
            )
        key = comparability_key(cfg)
        diff = {k: (ref_key[k], key[k]) for k in ref_key if ref_key[k] != key[k]}
        if diff:
            message = f"{os.path.basename(os.path.normpath(d))} differs from the reference in {diff}"
            if strict:
                raise ValueError(
                    f"comparability gate failed: {message}. These runs are not on a common scale; pass "
                    f"strict=False only if you know why that is acceptable."
                )
            warnings.append(message)

    # --- gate 2: same mocks, and a structure the bootstrap can handle ----------------------------
    info = describe_mock_set(ref_idx)
    rows = []
    for d, (idx, fom, cfg, prov) in loaded.items():
        order = align_to(ref_idx, idx)
        aligned = np.empty_like(fom)
        aligned[order] = fom
        stats = paired_ratio(
            ref_fom, aligned, i_sobol=ref_idx[:, 0] if info["clustered"] else None, n_boot=n_boot, seed=seed
        )
        stats["run"] = os.path.basename(os.path.normpath(d))
        stats["vali_total"] = read_vali_total(d, steps=prov["steps"])
        stats["is_reference"] = d == reference_dir
        stats["steps"] = prov["steps"]
        rows.append(stats)

    rows.sort(key=lambda r: -r["ratio"])
    info["warnings"] = warnings
    return rows, info


def format_table(rows, info, reference_name):
    """Render the comparison as a fixed-width table with the caveats that make it readable."""
    lines = []
    lines.append(
        f"mock set: {info['n_mocks']} mocks over {info['n_cosmologies']} cosmologies "
        f"({info['n_signal']} signal x {info['n_noise']} noise values, "
        f"{info['rows_per_cosmology'][0]}-{info['rows_per_cosmology'][1]} rows/cosmology)"
    )
    if info["clustered"]:
        lines.append("  -> repeated cosmologies: bootstrap clustered on i_sobol")
    for warning in info["warnings"]:
        lines.append(f"  WARNING {warning}")
    lines.append("")
    lines.append(f"paired FoM ratio vs {reference_name}   (seed floor {SEED_FLOOR:.3f}; '=' means indistinguishable)")
    lines.append(f"{'run':<34} {'steps':>7} {'ratio':>7} {'95% CI (mocks)':>18} {'win%':>6} {'vali_total':>11}")
    lines.append("-" * 88)
    for r in rows:
        mark = "  <- reference" if r["is_reference"] else (" =" if r["wash"] else "")
        vali = f"{r['vali_total']:.3f}" if r["vali_total"] is not None else "n/a"
        ci = "[{:.3f}, {:.3f}]".format(r["ci_low"], r["ci_high"])
        lines.append(
            f"{r['run']:<34} {r['steps']:>7d} {r['ratio']:>7.3f} {ci:>18} "
            f"{r['win_frac'] * 100:>5.0f}% {vali:>11}{mark}"
        )
    lines.append("")
    lines.append(
        "The CI is over MOCKS only. Run-to-run (seed) scatter is ~1.5% and does not shrink with more "
        "mocks; treat |ratio-1| < 0.015 as a wash regardless of the interval."
    )
    lines.append(
        "Ranks informativeness only. Robustness = posterior bias on the systematics mocks; estimator "
        "validity = SBC/TARP/HPD; real data = PPC. DES FoM is unsigned and is not computed here."
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], add_help=True)
    parser.add_argument("runs", nargs="+", help="run directory names under --root (or full paths)")
    parser.add_argument("--root", type=str, default="", help="common parent directory of the runs")
    parser.add_argument("--reference", type=str, required=True, help="run everything is paired against")
    parser.add_argument("--pair", type=str, nargs=2, default=["Om", "S8"], help="FoM parameter pair")
    parser.add_argument("--n_boot", type=int, default=4000, help="bootstrap resamples for the CI")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument(
        "--no_strict", action="store_true", help="downgrade the comparability gate from an error to a warning"
    )
    args = parser.parse_args(argv)

    names = args.runs if args.reference in args.runs else [args.reference] + args.runs
    dirs = [os.path.join(args.root, n) if args.root else n for n in names]
    reference_dir = os.path.join(args.root, args.reference) if args.root else args.reference

    rows, info = compare_runs(
        dirs,
        reference_dir,
        pair=tuple(args.pair),
        n_boot=args.n_boot,
        seed=args.seed,
        strict=not args.no_strict,
    )
    print(format_table(rows, info, os.path.basename(os.path.normpath(reference_dir))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
