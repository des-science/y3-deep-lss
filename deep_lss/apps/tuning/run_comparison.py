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

What this module does NOT answer: robustness (posterior bias on the systematics-variation mocks),
estimator validity (SBC/TARP/HPD) and the DES FoM, which is unsigned under misspecification. Those are
separate questions with separate instruments and a good FoM ratio here is not evidence about any of
them. The first and third live in `run_diagnostics`, which imports the primitives below so its numbers
are on this footing; the machinery they share (`find_inference_dir`, `run_labels`,
`checkpoint_warning`, `fom_per_mock`, `_param_column`) is defined here and used there.

Typical use::

    python -m deep_lss.apps.tuning.run_comparison --root <runs>/v17/baseline/maps/combined \\
        --reference t2_cls bench_v5_pool_head_w64 bench_v5_convnext_droppath
"""

import argparse
import glob
import json
import os
import re

import h5py
import numpy as np

# MEASURED run-to-run (training-seed) scatter of the paired FoM ratio, from a near-twin pair: identical
# config, different seed, +5.5% steps -> 1.015. It is a per-RUN effect, so it does NOT shrink with more
# mocks -- the mock bootstrap CI can be 2x tighter than this and mean nothing.
SEED_SCATTER = 0.015

# How many measured scatters a ratio must clear before the table stops calling it a wash. Same factor,
# and the same reasoning, as run_diagnostics.Q2_CONSERVATISM -- the two questions get one conservatism
# knob rather than two independently-argued numbers:
#
#   x1.3  the scatter estimate is itself uncertain -- ONE pair, and the +5.5% steps inside it are worth
#         ~+0.5% FoM on their own, so 1.015 is not a clean seed-only measurement either way.
#   x2.5  a ratio drawn from a null with that scatter exceeds 1.0x it ~32% of the time; at 2.5x it is
#         ~1% per arm, and a bench round is read by scanning 5-10 arms against one reference at once.
#
# The product is deliberately demanding: an architecture that wins by less than this is not worth
# carrying, because the simpler one is the better default when the evidence is a coin flip. Lower it
# only with a reason written down. NOTE this reclassifies older conclusions -- several bench rounds
# recorded effects of 2-4% as real against the raw 1.5% scatter, and those are washes here.
FOM_CONSERVATISM = 3.25

#: Paired-ratio deviation from 1.0 below which two runs are NOT distinguishable. Marked '=' in tables.
SEED_FLOOR = round(SEED_SCATTER * FOM_CONSERVATISM, 3)

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
    # Input modality. The `probe` key above only records which of lensing/clustering/cross are USED,
    # which is identical for a maps+cls run and a Cls-only two-point baseline -- so without this the
    # gate silently passes the one comparison where the whole point is that the inputs differ.
    "input_modality": None,
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


def input_modality(cfg):
    """Which inputs the compression saw: 'maps+cls' (a map encoder is configured) or 'cls' (two-point).

    Read from the config's own top-level shape rather than the run's path: a map run carries a `net`
    block (the map encoder), while a Cls-only baseline carries `mlp` and no `net`.
    """
    has_map_encoder = isinstance(_dig(cfg, ("net", "network")), dict)
    return "maps+cls" if has_map_encoder else "cls"


def comparability_key(cfg):
    """Reduce a run config to the fields that must agree across a comparison."""
    key = {}
    for name, path in _GATE_FIELDS.items():
        if name == "input_modality":
            key[name] = input_modality(cfg)
            continue
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


def _flow_dir_parts(path):
    """Split an inference directory name into (flow_name, steps).

    The directory is `<flow_name>_<steps>`, and flow_name is CONFIGURABLE -- `ensemble_flow` for the
    4-flow ensembles, `likelihood_flow` for a single flow, and sometimes a per-arm prefix
    (`v9_maf_convergence_maf_cosine_100_ensemble_flow`). Parsing only the trailing step count and
    treating everything that matches as "the same run at a different checkpoint" is wrong: it conflates
    flow VARIANTS with checkpoints. See `select_inference_dir`.
    """
    name = os.path.basename(os.path.dirname(path))
    match = re.search(r"^(.*?)_(\d+)$", name)
    return (match.group(1), int(match.group(2))) if match else (name, -1)


def _flow_dir_steps(path):
    """Training step count encoded in an `*flow_<steps>` directory name, or -1 if absent."""
    return _flow_dir_parts(path)[1]


def select_inference_dir(candidates, run_dir, flow_name=None):
    """Pick one inference directory from `candidates`; return (chosen, {directory: steps}).

    Two different ambiguities live in this directory listing and they need different answers:

      * SEVERAL CHECKPOINTS of one flow -- v3_cls carries 150k/200k/310k and its paired ratio moves
        0.302 -> 0.270 between them. The highest step count wins (the final model) and the others are
        returned so the caller can say which one a number refers to.
      * SEVERAL FLOWS at the same checkpoint -- a stale `likelihood_flow_300000` sitting next to the
        real `ensemble_flow_300000` (203 such collisions exist across the run roots), or five flow
        hyperparameter arms all trained to 1e6 steps. There is no "final" one here: picking by step
        count leaves the choice to filesystem order, which is exactly how a stale directory once
        silently supplied a tension chain. This RAISES unless `flow_name` says which flow to read.

    `flow_name` is not discoverable from the run's own config -- it lives in the msi/plotting runs
    config -- so it has to be passed in when a run holds more than one.
    """
    parts = {os.path.dirname(p): _flow_dir_parts(p) for p in candidates}
    names = sorted({name for name, _ in parts.values()})
    if flow_name is not None:
        parts = {d: v for d, v in parts.items() if v[0] == flow_name}
        if not parts:
            raise ValueError(f"{run_dir}: no inference directory with flow_name {flow_name!r}; found {names}")
    elif len(names) > 1:
        raise ValueError(
            f"{run_dir} holds {len(names)} DIFFERENT flows, not several checkpoints of one: {names}. "
            f"These are not interchangeable -- a stale directory next to the real one is a known "
            f"failure mode -- and the step count cannot arbitrate between them. Pass flow_name "
            f"(--flow_name) to say which flow to read; it is recorded in the msi runs config, not in "
            f"the run's own configs.yaml."
        )
    chosen = max(parts, key=lambda d: parts[d][1])
    return chosen, {d: steps for d, (_, steps) in parts.items()}


def find_inference_dir(run_dir, marker, flow_name=None):
    """Locate the run's inference output containing `marker`; return (directory, {directory: steps}).

    Globs `*flow_*` rather than hardcoding a name. The marker differs by stage and a run can genuinely
    have one without the other: `mcmc_samples.h5` is the coverage stage, `chain_*.npy` the per-mock
    sampling, and an interrupted job leaves chains but no coverage file.
    """
    candidates = glob.glob(os.path.join(run_dir, "*flow_*", marker))
    if not candidates:
        raise FileNotFoundError(
            f"no */{marker} under {run_dir} -- the inference stage did not run, or the job timed out "
            f"before writing it"
        )
    return select_inference_dir(candidates, run_dir, flow_name=flow_name)


def find_flow_dir(run_dir, flow_name=None):
    """The inference directory holding `mcmc_samples.h5` (the coverage stage)."""
    return find_inference_dir(run_dir, "mcmc_samples.h5", flow_name=flow_name)


def find_chain_dir(run_dir, flow_name=None):
    """The inference directory holding the per-mock `chain_*.npy` posteriors."""
    return find_inference_dir(run_dir, "chain_*.npy", flow_name=flow_name)


def run_labels(run_dirs):
    """Short display labels for a set of run directories, extended leftwards until they are unique.

    A basename alone collides across probes -- `lensing/bench_v7_full` and `clustering/bench_v7_full`
    are different runs with the same name -- and the Q1 and Q2 tables have to use the SAME labels or
    they cannot be read side by side.
    """
    parts = [os.path.normpath(d).split(os.sep) for d in run_dirs]
    for depth in range(1, 1 + max(len(p) for p in parts)):
        labels = ["/".join(p[-depth:]) for p in parts]
        if len(set(labels)) == len(labels):
            return labels
    return ["/".join(p) for p in parts]


def checkpoint_warning(label, steps, other_steps):
    """The one wording for "this run has more than one evaluated checkpoint", used by every table."""
    return (
        f"{label} has {len(other_steps) + 1} evaluated checkpoints (using {steps}, also present: "
        f"{other_steps}) -- the choice moves the numbers, so state which one they refer to"
    )


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
    det = cxx * cyy - cxy**2
    if np.any(det <= 0):
        raise ValueError(f"non-positive covariance determinant for {int((det <= 0).sum())} mocks")
    return det**-0.5


def load_run(run_dir, pair=("Om", "S8"), flow_name=None):
    """Load one run and reduce it to (real_idx, fom, config, provenance).

    Only the per-mock FoM is retained: theta_sample is ~360 MB per run, so keeping it for a table of
    ten runs is pointless. `provenance` records which checkpoint was used and how many were available.
    """
    cfg = load_run_config(run_dir)
    params = param_names(cfg)
    flow_dir, candidates = find_flow_dir(run_dir, flow_name=flow_name)
    provenance = {
        "steps": candidates[flow_dir],
        "n_checkpoints": len(candidates),
        "other_steps": sorted(s for d, s in candidates.items() if d != flow_dir),
    }
    with h5py.File(os.path.join(flow_dir, "mcmc_samples.h5"), "r") as h:
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
            f"were evaluated on different mocks and cannot be paired. If the runs come from different "
            f"pipelines (maps vs cls), their coverage mock sets alias differently and only overlap "
            f"partially -- use intersect=True to pair on the common subset instead."
        )
    return np.array([ref_map[tuple(row)] for row in other_idx.tolist()])


def common_mocks(idx_list):
    """Rows present in EVERY run's mock set, as an (n_common, 3) array sorted by the real_idx tuple.

    Needed because the maps and Cls pipelines pack the (i_signal, i_noise) example axis transposed, and
    the coverage selection strides by row POSITION -- so the two pipelines' 1000 mocks are different
    physical realizations of the same 1000 cosmologies and only partially coincide. Pairing on the
    intersection keeps the comparison valid (identical theta_true, identical realization) at the cost of
    a smaller n; pairing on i_sobol alone would NOT, since i_signal carries the astrophysical draw.
    """
    sets = [{tuple(row) for row in idx.tolist()} for idx in idx_list]
    common = set.intersection(*sets)
    if not common:
        raise ValueError("the runs share no mocks at all (empty real_idx intersection); they cannot be paired")
    return np.array(sorted(common), dtype=np.int64)


def gather_mocks(idx, fom, wanted):
    """Reorder one run's per-mock FoM onto the rows of `wanted` (an (n, 3) real_idx array)."""
    row_of = {tuple(row): i for i, row in enumerate(idx.tolist())}
    return fom[np.array([row_of[tuple(row)] for row in wanted.tolist()])]


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


def compare_runs(
    run_dirs,
    reference,
    pair=("Om", "S8"),
    n_boot=4000,
    seed=0,
    strict=True,
    intersect=False,
    flow_name=None,
    cross_modality=False,
):
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

    `cross_modality=True` is the DES-Y3-level comparison: a maps+cls run against the Cls-only
    two-point baseline. It permits EXACTLY ONE gate field to differ, `input_modality`, and keeps every
    other field an error -- so it is a scalpel, not `strict=False`. The paired FoM survives that one
    difference intact: it is computed from posterior samples on the same mocks, for the same
    parameters, by the same function, and it measures posterior WIDTH without reference to how the
    summary was built. That is exactly the "how much does the neural summary buy over two-point"
    question, and it is signed, because the coverage mocks are correctly specified.

    `vali_total` is kept but must be read with more care here than within one modality: the four
    conditions that make it comparable (probe, head type, theta standardization, summary dim) are
    still gated, but the VMIM bound's tightness also depends on how easy the SUMMARY DISTRIBUTION is
    for the flow to model, and two modalities are further apart on that axis than two architectures.
    """
    reference_dir = reference
    ordered = list(run_dirs)
    if reference_dir not in ordered:
        ordered = [reference_dir] + ordered

    loaded = {d: load_run(d, pair=pair, flow_name=flow_name) for d in ordered}
    ref_idx, ref_fom, ref_cfg, _ = loaded[reference_dir]
    # Same labelling rule as run_diagnostics, so the Q1 and Q2 tables name the runs identically.
    label_of = dict(zip(ordered, run_labels(ordered)))

    # --- gate 0: comparability -------------------------------------------------------------------
    ref_key = comparability_key(ref_cfg)
    warnings, crossed = [], []
    for d, (_, _, cfg, prov) in loaded.items():
        if prov["n_checkpoints"] > 1:
            warnings.append(checkpoint_warning(label_of[d], prov["steps"], prov["other_steps"]))
        key = comparability_key(cfg)
        diff = {k: (ref_key[k], key[k]) for k in ref_key if ref_key[k] != key[k]}
        if "input_modality" in diff and cross_modality:
            crossed.append(f"{label_of[d]} ({key['input_modality']} vs reference {ref_key['input_modality']})")
        # Under cross_modality the input difference IS the measurement, so it is permitted -- and only
        # it. Every other field stays an error, which is what separates this from strict=False: a
        # different head or a different holdout split would still make the ratio meaningless, and this
        # route must not become the way those get waved through.
        blocking = {k: v for k, v in diff.items() if not (cross_modality and k == "input_modality")}
        if blocking:
            message = f"{label_of[d]} differs from the reference in {blocking}"
            if strict:
                raise ValueError(
                    f"comparability gate failed: {message}. These runs are not on a common scale; pass "
                    f"strict=False only if you know why that is acceptable."
                )
            warnings.append(message)

    if cross_modality and not crossed:
        # Silent-failure guard: believing you are measuring against the two-point baseline when every
        # run is the same modality reads as "the neural gain is 1.00", which is a wrong conclusion
        # rather than a missing one.
        warnings.append(
            "cross_modality was requested but every run has the same input_modality "
            f"({ref_key['input_modality']}); this is NOT a neural-vs-two-point comparison"
        )
    elif crossed:
        warnings.append(f"CROSS-MODALITY comparison, input_modality differs by design: {crossed}")

    # --- gate 2: same mocks, and a structure the bootstrap can handle ----------------------------
    if intersect:
        # cross-pipeline case: pair on the mocks every run actually evaluated, not on the reference set
        wanted = common_mocks([idx for idx, _, _, _ in loaded.values()])
        ref_fom = gather_mocks(ref_idx, ref_fom, wanted)
        ref_idx = wanted
        dropped = {label_of[d]: len(idx) - len(wanted) for d, (idx, _, _, _) in loaded.items()}
        if any(dropped.values()):
            warnings.append(
                f"paired on the {len(wanted)}-mock intersection of the runs' coverage sets (dropped per "
                f"run: {dropped}); the CI widens as sqrt(n) but the seed floor does not"
            )
    info = describe_mock_set(ref_idx)
    rows = []
    for d, (idx, fom, cfg, prov) in loaded.items():
        if intersect:
            aligned = gather_mocks(idx, fom, ref_idx)
        else:
            order = align_to(ref_idx, idx)
            aligned = np.empty_like(fom)
            aligned[order] = fom
        stats = paired_ratio(
            ref_fom, aligned, i_sobol=ref_idx[:, 0] if info["clustered"] else None, n_boot=n_boot, seed=seed
        )
        stats["run"] = label_of[d]
        stats["vali_total"] = read_vali_total(d, steps=prov["steps"])
        stats["is_reference"] = d == reference_dir
        stats["steps"] = prov["steps"]
        rows.append(stats)

    rows.sort(key=lambda r: -r["ratio"])
    info["warnings"] = warnings
    info["cross_modality"] = bool(crossed)
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
    width = max(34, max(len(r["run"]) for r in rows))
    lines.append(f"{'run':<{width}} {'steps':>7} {'ratio':>7} {'95% CI (mocks)':>18} {'win%':>6} {'vali_total':>11}")
    lines.append("-" * (width + 54))
    for r in rows:
        mark = "  <- reference" if r["is_reference"] else (" =" if r["wash"] else "")
        vali = f"{r['vali_total']:.3f}" if r["vali_total"] is not None else "n/a"
        ci = "[{:.3f}, {:.3f}]".format(r["ci_low"], r["ci_high"])
        lines.append(
            f"{r['run']:<{width}} {r['steps']:>7d} {r['ratio']:>7.3f} {ci:>18} "
            f"{r['win_frac'] * 100:>5.0f}% {vali:>11}{mark}"
        )
    lines.append("")
    lines.append(
        f"The CI is over MOCKS only and is NOT a reproducibility interval. Measured run-to-run seed "
        f"scatter is {SEED_SCATTER:.1%} and does not shrink with more mocks; the floor above is that "
        f"scatter x{FOM_CONSERVATISM} (estimate uncertainty x multi-arm scanning), so treat "
        f"|ratio-1| < {SEED_FLOOR:.3f} as a wash however tight the CI looks."
    )
    lines.append(
        "The floor is deliberately demanding: when two architectures are inside it, the SIMPLER one "
        "wins by default -- an unresolved ratio is not a reason to carry extra machinery."
    )
    lines.append(
        "Ranks informativeness only. Robustness = posterior bias on the systematics mocks; estimator "
        "validity = SBC/TARP/HPD; real data = PPC. DES FoM is unsigned and is not computed here."
    )
    if info.get("cross_modality"):
        lines.append("")
        lines.append(
            "CROSS-MODALITY: the ratio above is the neural summary's gain over the two-point baseline "
            "on correctly-specified mocks. Point the reference at the Cls run to read it that way "
            "round. It is SIGNED (the mocks come from the model that generated the training data), so "
            "tighter genuinely is better -- but it is a gain on SIMULATIONS, and it says nothing about "
            "whether the extra information survives on real data. Q2 and the DES/fid ratio in "
            "run_diagnostics carry that half, and the seed floor still applies."
        )
        lines.append(
            "Read vali_total with extra care here: the gated conditions still hold, but the VMIM "
            "bound's tightness also tracks how hard the SUMMARY DISTRIBUTION is for the flow to model, "
            "and two modalities differ more on that axis than two architectures do."
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
    parser.add_argument(
        "--intersect",
        action="store_true",
        help="pair on the mocks common to all runs instead of requiring identical mock sets (needed "
        "only for a Cls run whose INFERENCE predates 68210dd; later ones pair 1000/1000 on their own)",
    )
    parser.add_argument(
        "--cross_modality",
        action="store_true",
        help="the DES-Y3-level comparison: permit maps+cls vs a Cls-only two-point baseline. Allows "
        "exactly one gate field (input_modality) to differ and keeps every other field an error -- "
        "use this, never --no_strict, for a neural-vs-two-point number",
    )
    parser.add_argument(
        "--flow_name",
        type=str,
        default=None,
        help="which flow to read when a run holds more than one (e.g. ensemble_flow); required only "
        "then, and taken from the msi runs config rather than the run's own configs.yaml",
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
        intersect=args.intersect,
        flow_name=args.flow_name,
        cross_modality=args.cross_modality,
    )
    print(format_table(rows, info, os.path.basename(os.path.normpath(reference_dir))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
