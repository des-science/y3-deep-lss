# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
The run-level diagnostics `run_comparison` deliberately refuses to compute: Q2 robustness, Q3
estimator validity and Q4 DES FoM.

`run_comparison` answers Q1 (informativeness) and stops there, correctly: robustness is a different
observable, coverage is compression-invariant and DES FoM is unsigned, so none of them belongs in a
ranking table. But they keep being asked
for, and the answer kept being re-improvised in a throwaway script -- six of them between 2026-08-13
and 2026-08-18. That is the same failure mode `run_comparison` was written to end, and it produced
the same class of damage:

  * THE STATISTIC DRIFTED. Three of those scripts computed the 2D (Om, S8) Mahalanobis shift, two
    computed a marginal 1D shift over (Om, S8, w0), one over (Om, s8, S8). Tables from different
    weeks were quietly not comparable. The convention recorded in the project notes -- and the one
    implemented here -- is the 2D Mahalanobis shift against each run's OWN fiducial posterior.
  * THE COLUMN ORDER CAME FROM TWO DIFFERENT CONFIG KEYS. Half the scripts read
    `dlss.dset.training.params`, half `dlss.dset.eval.grid.params`. They agree on every run checked,
    but nothing enforces it, and a mismatch mislabels every column with nothing failing. This module
    asserts they agree (`assert_param_sources_agree`) instead of trusting whichever one it read.
  * THE MOCK LIST WAS HARDCODED. Two scripts carried a literal list of contamination mocks, so a
    mock present on disk but missing from the list was silently dropped from the "all mocks" summary
    -- and the labels DID change between dataset versions (`fiducial_bench_nla_per_shell` in v17 vs
    `fiducial_bench_Aia=0.5,eta=1_shell` in v18). Mocks are discovered here, never enumerated.

Everything numerical is imported from `run_comparison` rather than restated, so the FoM here is the
same function that ranks runs, the checkpoint is chosen by the same rule, and S8 = s8*sqrt(Om/0.3)
has one definition. The only thing this module adds is what to do with chains rather than
`mcmc_samples.h5`.

Like `run_comparison` it depends only on numpy/scipy/yaml and does NOT import msi -- y3-deep-lss
installs first, and on a login node `msi.utils.mock_contamination`, `msi.utils.coverage` and
`msi.utils.diagnostics` are not importable anyway (they pull seaborn/matplotlib, and the coverage
path also wants torch/sbi/tarp). Three rules here are therefore deliberate re-statements of msi
code, and the two copies must be kept in step:

  * the mock-discovery rule, from `msi/utils/mock_contamination.py::_load_mock_chains`. That module
    draws the `unblinding_plots/*_mock_contamination_*.png` contours from exactly these chains -- it
    just never emits a number, which is why this exists.
  * `sbc_ranks`, from `msi/utils/diagnostics.py::run_sbc_precomputed`.
  * `hpd_curve`, from `msi/utils/diagnostics.py::posterior_hpd_check`.

The last two have the same "plots it, never emits it" problem, and worse: `msi/utils/coverage.py`
logs the SBC KS p-values and c2st to the inference LOG TEXT and nowhere else, and the HPD and TARP
curves are plotted and discarded without a number ever being written. So Q3 existed only as four
PNGs per run plus 59 log files, one run each -- unscannable across a benchmark round, and the exact
setup in which the TARP reading has twice been made wrong and corrected.

!! DO NOT HARVEST THE LOGGED SBC NUMBERS, AND DO NOT TRUST THREE OF THE FOUR PLOTS. !!
`tarp` 0.1.1's bootstrap wrapper MUTATES THE CALLER'S ARRAYS IN PLACE (`tarp/drp.py:174-176`):

    samples[:, idx_remove, :] = samples[:, idx_add, :]
    theta[idx_remove, :] = theta[idx_add, :]

100 iterations, cumulative, no copy taken. `run_coverage_tests` runs HPD -> TARP -> TARP marginals
-> SBC on ONE shared pair of arrays, and `run_coverage` then runs l-C2ST on the same pair, so
everything after the TARP call sees ~100 of the 1000 coverage mocks overwritten by duplicates of
other mocks. Verified 2026-08-19: sbi's own `check_sbc`, run in torch_env against the stored
`mcmc_samples.h5`, reproduces THIS module's KS p-values to every printed digit and NOT the ones in
the run logs, and replaying tarp's mutation loop moves the p-values by exactly the observed amount
(logged bg3 4.6e-02 against a pristine 2.4e-01, replay trials 1.6e-01..4.6e-01). So:

  * `mcmc_samples.h5` is written BEFORE any of the tests run and is PRISTINE. Reading it, as this
    module does, is the only reproducible route -- and every Q1/Q2/Q4 number is unaffected, since
    those come from the same h5 or from `chain_*.npy`.
  * `2_posterior_hpd.png` is computed FIRST and is therefore also pristine.
  * `2_posterior_{tarp,tarp_marginals,sbc}.png`, the logged SBC KS/c2st values and the l-C2ST
    p-value are ALL computed after the mutation. Treat them as qualitative at best.

Fixing this belongs in msi (copy before calling `get_tarp_coverage`, or reorder the tests); it is
recorded here because this is the module that reads the affected artefacts.

Typical use::

    python -m deep_lss.apps.tuning.run_diagnostics robustness \\
        --root <runs>/v18/default/maps/combined bench_v7_full bench_v9_unet_multiscale

    python -m deep_lss.apps.tuning.run_diagnostics des-fom \\
        --root <runs>/v18/default --reference cls/combined/bench_v7 maps/combined/bench_v7_full

    python -m deep_lss.apps.tuning.run_diagnostics coverage \\
        --root <runs>/v18/default/maps/lensing bench_v7_full bench_v10_mean_std_1x
"""

import argparse
import glob
import os

import numpy as np

# Deliberate reuse of run_comparison's internals, including the private helpers: this module exists
# because these were being re-implemented per session, so importing them (rather than restating them)
# is the entire point. `_param_column` carries the S8 derivation, `find_chain_dir`/`find_flow_dir` the
# flow/checkpoint rule and `run_labels` the naming, so the Q1-Q4 tables are on one footing and name
# runs alike. The two finders differ in what they look for and a run can have one without the other:
# `find_chain_dir` wants the per-mock `chain_*.npy` (Q2/Q4), `find_flow_dir` the coverage stage's
# `mcmc_samples.h5` (Q1/Q3).
from deep_lss.apps.tuning.run_comparison import (
    _dig,
    _param_column,
    align_to,
    checkpoint_warning,
    find_chain_dir,
    find_flow_dir,
    fom_per_mock,
    load_run_config,
    param_names,
    run_labels,
)

# Chain labels that are not systematics variations. The per-cosmology grid is `grid_(i,j,k)` in v18 and
# `cosmo_*` in older runs; Buzzard is a separate N-body suite. Same rule as msi's _load_mock_chains,
# which writes them. (The `_mean` suffix already excludes the grid on its own -- this is belt and
# braces, kept in step with msi rather than trimmed to what today's naming happens to need.)
EXCLUDE_PREFIXES = ("cosmo", "grid", "Buzzard")

# The correctly-specified baseline every contaminated mock is referenced against.
FIDUCIAL_LABEL = "fiducial_bench"

# Chain variants that DROP a parameter column rather than fixing it, so the file is narrower than the
# run's trained parameter list. Reading a lambdaCDM chain with the full list shifts everything after w0
# and mislabels the posterior with nothing raising. Written by msi (`likelihood_base.py:145`): the
# reduced space is exactly w0 for lambdaCDM and bta for nla, and they COMPOSE
# (`chain_DESy3_w0gt-1_nla.npy` is 9 columns against 10 params). Mirrors dlss_plot's
# runs.VARIANT_FIXED -- keep the two in step.
VARIANT_FIXED = {"lambdaCDM": ("w0",), "nla": ("bta",)}

# Qualifiers that change the SAMPLED SPACE OR THE PRIOR rather than the data vector: lambdaCDM fixes
# w0, nla fixes bta, w0gt-1 truncates the w0 prior (`likelihood_base.py:88-101`). All three tighten
# (Om, S8) on their own, and the fiducial and contaminated mocks were only ever sampled with the
# default prior -- so a DES chain carrying one of these has no like-for-like within-run reference and
# its DES/fid ratio would be measuring the prior restriction, not the data-vs-sim divergence.
# `no_sys` and `no_psi_rot` are NOT here: those change the data vector and keep the full space
# (dlss_plot/runs.py:47), so they do have a valid reference.
PRIOR_QUALIFIERS = ("lambdaCDM", "nla", "w0gt-1")

# Cross-run spread in the fiducial marginal widths beyond which "shift in units of its own sigma" is
# no longer a like-for-like comparison and the absolute shift has to be read alongside it.
WIDTH_SPREAD_WARN = 1.10

# MEASURED run-to-run scatter of the Q2 shift, in sigma, by table row: the RMS DIFFERENCE between two
# runs that should be the same run.
#
# Measured 2026-08-19 with `floor_from_pair` on bench_v4_pool_head vs bench_v5_default (v17 baseline
# maps/combined) -- the same near-twin the Q1 SEED_FLOOR came from: identical config bar a v4->v5 key
# rename, different seed, +5.5% steps. Decomposing against a first-half/second-half split of the same
# chains (which isolates sampler noise) gives sigma_seed = 0.049 and sigma_MC = 0.015 per run, so MC is
# only ~9% of the variance: the scatter is training-seed dominated and more MCMC samples will not move
# it. The three rows differ because they are different statistics of the same 7 numbers -- MEAN
# averages the scatter down by ~sqrt(7) (predicted 0.027, observed 0.025, so the per-mock differences
# behave as independent noise), while MAX is an extremum and is NOISIER than a typical single mock.
Q2_SCATTER = {"mock": 0.073, "MEAN |.|": 0.025, "MAX |.|": 0.120}

# How many measured sigmas a difference must clear before the table stops calling it a wash.
# DELIBERATELY CONSERVATIVE, because the cost of the two errors is not symmetric here: reading a real
# robustness difference as noise costs one architecture comparison, while reading noise as a real
# difference argues for carrying a more complicated network forever. Two independent factors:
#
#   x1.3  the scatter estimate is itself uncertain. ONE pair and 7 mocks give ~27% on the RMS, there is
#         no handle on pair-to-pair variation, and it is one probe at one dataset version. Use the
#         one-sigma upper end of the estimate, not its centre.
#   x2.5  a difference drawn from a null with that scatter exceeds 1.0x it ~32% of the time. At 2.5x it
#         is ~1.2% per row -- and a table is read by scanning ~8 rows at once, so the family-wise rate
#         is what matters (~9%), not the per-row one.
#
# The product is what a difference must clear to be worth acting on. Lower it only with a reason
# written down; raising it is always safe. Re-measure with `floor_from_pair` when a true same-step twin
# exists -- a better measurement is the honest way to lower the floor, not a smaller factor.
Q2_CONSERVATISM = 3.25

#: Difference below which two runs are NOT distinguishable on that row. Rows inside it print '='.
Q2_FLOOR = {row: round(scatter * Q2_CONSERVATISM, 2) for row, scatter in Q2_SCATTER.items()}

# Step-count ratio across the compared runs beyond which Q2 is confounded by training length rather
# than by architecture. Measured: the same config at 250k vs 129k steps moves 0.25 sigma RMS, several
# times the floor, with source_clustering_gatti going 0.19 -> 0.76.
STEPS_SPREAD_WARN = 1.20

# The coverage mocks' truths must be drawn from the WIDE analysis prior, not from the CosmoGrid Sobol
# grid. A grid-truth set is narrower than the prior the flow was trained against, which fakes
# overconfidence -- every SBC and HPD number in the Q3 table would then be measuring the mismatch
# instead of the estimator. `msi/utils/coverage.py::wide_prior_sobol_indices` applies the mask and the
# flow dir records what it used, so this is checkable rather than assumed: refuse, don't warn.
PRIOR_SELECTION_REQUIRED = "wide"

# How many analytic sigmas a Q3 deviation must clear before the table stops calling it calibrated.
#
# NOTE THE FACTOR DIFFERS FROM Q1/Q2 ON PURPOSE, and this is the one place in the programme where a
# smaller number is the honest one. Q1's FOM_CONSERVATISM and Q2's Q2_CONSERVATISM are both 3.25,
# which factorises as:
#
#   x1.3  the SCATTER ESTIMATE is itself uncertain -- one twin pair, 7 mocks, ~27% on the RMS.
#   x2.5  a null exceeds 1.0x its scatter ~32% of the time, and a table is read by scanning ~8 rows
#         at once, so the family-wise rate is what matters.
#
# Q3's null is not measured, it is KNOWN: calibrated SBC ranks are exactly uniform and a calibrated
# HPD curve is exactly the diagonal, so `sbc_null_scales` derives the sigma analytically from the mock
# count. There is no scatter estimate to be uncertain about, so the x1.3 term has no referent and only
# the scanning term survives. This is not a floor lowered to let a result through -- it is a floor
# that needs no twin pair, which is just as well: the v18 inventory contains none (the _1x/_2x pairs
# differ in budget, and cls/*/v1 vs cls/*/bench_v7 are the same network with a byte-identical
# preds_*.h5). Raising it to 3.25 for consistency with the other tables is always safe.
Q3_CONSERVATISM = 2.5


def assert_param_sources_agree(cfg, run_dir):
    """Gate: the training and eval parameter lists must be the same list.

    `param_names` reads `dlss.dset.eval.grid.params`, which is the column order of everything the
    inference stage writes. Ad-hoc scripts have read `dlss.dset.training.params` instead. On every run
    checked the two agree, so no published number is wrong -- but the agreement is a coincidence of
    how the configs are written, not a guarantee, and if it ever breaks the failure is silent
    mislabelling rather than an error. Check it once, here, instead of trusting whichever key a
    caller happened to reach for.
    """
    training = _dig(cfg, ("dlss", "dset", "training", "params"))
    grid = _dig(cfg, ("dlss", "dset", "eval", "grid", "params"))
    if training is not None and list(training) != list(grid):
        raise ValueError(
            f"{run_dir}: dlss.dset.training.params {list(training)} != dlss.dset.eval.grid.params "
            f"{list(grid)}. The chain column order is ambiguous for this run; resolve which list the "
            f"inference stage actually wrote before reading any chain from it."
        )


def chain_params(params, variant):
    """The column list of `chain_<variant>.npy`, which is NOT always the run's trained parameter list.

    A `lambdaCDM` variant drops w0 and an `nla` variant drops bta instead of fixing them, so the array
    is one column narrower per qualifier and they compose. Qualifiers are matched as whole
    underscore-separated tokens, not as substrings, so a longer label containing one by accident does
    not silently remove a column (same rule as dlss_plot's `runs.variant_fixed`).
    """
    tokens = set((variant or "").split("_"))
    dropped = {p for qualifier, fixed in VARIANT_FIXED.items() if qualifier in tokens for p in fixed}
    return [p for p in params if p not in dropped]


def load_chain(chain_dir, label, params):
    """Load one chain and check its width against the parameter list, or return None if absent."""
    path = os.path.join(chain_dir, f"chain_{label}.npy")
    if not os.path.isfile(path):
        return None
    chain = np.load(path)
    if chain.ndim != 2 or chain.shape[1] != len(params):
        raise ValueError(
            f"{path}: shape {chain.shape} but {len(params)} parameters {params}. The column order "
            f"cannot be established, so no parameter read out of this file would be trustworthy."
        )
    return chain


def discover_mock_labels(chain_dir):
    """Every systematics-variation mock holding a `_mean` chain here, as bare labels without `_mean`.

    Discovery, never a hardcoded list: the labels changed between dataset versions and a hardcoded
    list drops the new ones from the summary without saying so.
    """
    labels = []
    for path in sorted(glob.glob(os.path.join(chain_dir, "chain_*_mean.npy"))):
        label = os.path.basename(path)[len("chain_") : -len("_mean.npy")]
        if label.startswith(EXCLUDE_PREFIXES):
            continue
        labels.append(label)
    return labels


def _plane(chain, params, pair):
    """The (n_samples, 2) posterior projected onto the FoM parameter pair, S8 derived if needed."""
    return np.column_stack([_param_column(chain, params, name) for name in pair])


def posterior_shifts(chain_dir, params, pair=("Om", "S8"), fiducial_label=FIDUCIAL_LABEL, sl=slice(None)):
    """Shift of every contaminated mock's posterior against this run's own fiducial posterior.

    The statistic is the 2D Mahalanobis distance between the two posterior MEANS under the FIDUCIAL
    covariance -- i.e. how far the contamination moved the answer, measured in units of the
    statistical error this run itself reports. Recorded convention; the marginal 1D shifts and the
    ABSOLUTE shifts are returned alongside because a sigma-normalised shift shrinks when a run is
    simply less informative, and separating those two needs the unnormalised number.

    Referencing against the run's own fiducial posterior rather than against the true parameters is
    what makes this comparable across architectures: it isolates the response to the perturbation
    from whatever constant offset the compression already had.

    `sl` restricts every chain to the same sub-slice of its samples. Only `floor_from_pair` uses it,
    to split a chain in half and separate sampler noise from training-seed scatter.
    """
    fiducial = load_chain(chain_dir, f"{fiducial_label}_mean", params)
    if fiducial is None:
        raise FileNotFoundError(
            f"{chain_dir}: no chain_{fiducial_label}_mean.npy -- there is no correctly-specified "
            f"baseline to reference the shifts against, so no robustness number can be formed"
        )
    ref = _plane(fiducial, params, pair)[sl]
    mu = ref.mean(axis=0)
    cov = np.cov(ref, rowvar=False)
    sigma = np.sqrt(np.diag(cov))
    cov_inv = np.linalg.inv(cov)

    shifts = {}
    for label in discover_mock_labels(chain_dir):
        if label == fiducial_label:
            continue
        chain = load_chain(chain_dir, f"{label}_mean", params)
        delta = _plane(chain, params, pair)[sl].mean(axis=0) - mu
        shifts[label] = {
            "mahalanobis": float(np.sqrt(delta @ cov_inv @ delta)),
            "marginal_sigma": delta / sigma,
            "absolute": delta,
        }
    return {"mu": mu, "sigma": sigma, "n_samples": int(ref.shape[0]), "shifts": shifts}


def floor_from_pair(run_a, run_b, pair=("Om", "S8"), fiducial_label=FIDUCIAL_LABEL, flow_name=None):
    """Measure the Q2 reproducibility floor from a pair of runs that should be the same run.

    This is what produced `Q2_FLOOR`, kept so the number can be re-derived rather than trusted. Give
    it a same-config different-seed pair: the RMS of their per-mock shift differences IS the floor on
    a difference, since by construction there is nothing real to find between them.

    `mc_*` is the same quantity computed between the first and second half of each run's OWN chains.
    It isolates the sampler contribution (no retraining involved), so the difference in quadrature is
    the training-seed contribution. If `mc` ever approaches `rms`, the floor is sampling-limited and
    longer chains would lower it; measured at 2026-08-19 it does not.
    """
    per_run, halves = [], []
    for run_dir in (run_a, run_b):
        params = param_names(load_run_config(run_dir))
        chain_dir, _ = find_chain_dir(run_dir, flow_name=flow_name)
        full = posterior_shifts(chain_dir, params, pair=pair, fiducial_label=fiducial_label)
        n = full["n_samples"]
        lo = posterior_shifts(chain_dir, params, pair=pair, fiducial_label=fiducial_label, sl=slice(0, n // 2))
        hi = posterior_shifts(chain_dir, params, pair=pair, fiducial_label=fiducial_label, sl=slice(n // 2, None))
        per_run.append(full["shifts"])
        halves.append((lo["shifts"], hi["shifts"]))

    def rms(x, y):
        common = sorted(set(x) & set(y))
        return float(np.sqrt(np.mean([(x[m]["mahalanobis"] - y[m]["mahalanobis"]) ** 2 for m in common]))), common

    value, mocks = rms(*per_run)
    # A half chain has twice the variance of the full one, so a half-split difference has 4x the
    # per-run MC variance: sigma_MC = rms_half / 2.
    mc = float(np.sqrt(np.mean([rms(lo, hi)[0] ** 2 / 4 for lo, hi in halves])))
    seed = float(np.sqrt(max(value**2 / 2 - mc**2, 0.0)))
    return {
        "rms": value,
        "sigma_seed": seed,
        "sigma_mc": mc,
        "sigma_run": float(np.hypot(seed, mc)),
        "n_mocks": len(mocks),
        "mocks": mocks,
    }


def _load_runs(run_dirs, flow_name=None, finder=find_chain_dir):
    """Resolve each run to (label, run_dir, params, inference_dir, steps), plus the shared warnings.

    Every subcommand needs exactly this preamble -- config, parameter-source gate, inference directory,
    checkpoint ambiguity -- and having it more than once is how the tables drifted into naming the same
    run differently and wording the same warning two ways.

    `finder` is the only thing that varies: `find_chain_dir` for the tables that read `chain_*.npy`
    (Q2, Q4), `find_flow_dir` for the one that reads `mcmc_samples.h5` (Q3). Both return
    (directory, {directory: steps}) and both apply the same highest-checkpoint rule, so the resolved
    step count means the same thing in every table.
    """
    entries, warnings = [], []
    for label, run_dir in zip(run_labels(run_dirs), run_dirs):
        cfg = load_run_config(run_dir)
        assert_param_sources_agree(cfg, run_dir)
        inference_dir, candidates = finder(run_dir, flow_name=flow_name)
        if len(candidates) > 1:
            others = sorted(v for k, v in candidates.items() if k != inference_dir)
            warnings.append(checkpoint_warning(label, candidates[inference_dir], others))
        entries.append(
            {
                "label": label,
                "run_dir": run_dir,
                "params": param_names(cfg),
                "inference_dir": inference_dir,
                "steps": candidates[inference_dir],
            }
        )
    return entries, warnings


def robustness(run_dirs, pair=("Om", "S8"), fiducial_label=FIDUCIAL_LABEL, flow_name=None):
    """Q2 table: posterior bias on the contamination mocks for each run. Smaller is more robust."""
    loaded, warnings = _load_runs(run_dirs, flow_name=flow_name)
    runs = []
    for meta in loaded:
        entry = posterior_shifts(meta["inference_dir"], meta["params"], pair=pair, fiducial_label=fiducial_label)
        entry.update(meta)
        runs.append(entry)

    # Only mocks every run actually has can be tabulated side by side; the rest are named, not hidden.
    per_run = [set(r["shifts"]) for r in runs]
    common = sorted(set.intersection(*per_run)) if per_run else []
    dropped = sorted(set.union(*per_run) - set(common)) if per_run else []
    if not common:
        raise ValueError("the runs share no contamination mock in common; there is nothing to tabulate")
    if dropped:
        warnings.append(f"not present in every run, left out of the table: {dropped}")

    # Training length moves Q2 by more than the floor, so an uneven step count is a confound, not a
    # detail: a robustness "difference" between runs of different length is about the budget.
    steps = np.array([r["steps"] for r in runs], dtype=float)
    if len(steps) > 1 and steps.max() / steps.min() > STEPS_SPREAD_WARN:
        warnings.append(
            f"step counts span {steps.max() / steps.min():.2f}x ({sorted(int(s) for s in steps)}). Q2 is "
            f"strongly sensitive to training length -- 250k vs 129k on one config moves it 0.25 sigma "
            f"RMS, several times the floor -- so a difference here may be budget, not architecture"
        )

    # A shift in units of "its own sigma" only compares across runs while the sigmas do.
    for i, name in enumerate(pair):
        widths = np.array([r["sigma"][i] for r in runs])
        if widths.max() / widths.min() > WIDTH_SPREAD_WARN:
            warnings.append(
                f"fiducial {name} widths span {widths.max() / widths.min():.2f}x across these runs, so "
                f"the sigma-normalised shifts are not on a common scale -- read the absolute shifts "
                f"(--full) alongside them"
            )
    return {"runs": runs, "mocks": common, "pair": tuple(pair), "warnings": warnings}


def format_robustness(result, full=False):
    """Render the Q2 table: mocks down the page, runs across it."""
    runs, mocks, pair = result["runs"], result["mocks"], result["pair"]
    width = max(14, max(len(r["label"]) for r in runs) + 2)
    lines = [
        f"Q2 robustness: posterior shift on {len(mocks)} contamination mocks, each run against its OWN",
        f"{FIDUCIAL_LABEL} posterior, in units of that posterior's width. SMALLER IS MORE ROBUST.",
        "",
        f"fiducial posterior ({pair[0]}, {pair[1]})",
        f"{'run':<{width}}{'steps':>9}{'mean_' + pair[0]:>12}{'sig_' + pair[0]:>12}"
        f"{'mean_' + pair[1]:>12}{'sig_' + pair[1]:>12}{'samples':>10}",
    ]
    for r in runs:
        lines.append(
            f"{r['label']:<{width}}{r['steps']:>9d}{r['mu'][0]:>12.4f}{r['sigma'][0]:>12.4f}"
            f"{r['mu'][1]:>12.4f}{r['sigma'][1]:>12.4f}{r['n_samples']:>10d}"
        )
    for warning in result["warnings"]:
        lines.append(f"  WARNING {warning}")

    tables = [("mahalanobis", None, f"2D ({pair[0]}, {pair[1]}) shift [sigma] -- the headline number")]
    if full:
        tables += [
            ("marginal_sigma", 0, f"marginal {pair[0]} shift [sigma]"),
            ("marginal_sigma", 1, f"marginal {pair[1]} shift [sigma]"),
            ("absolute", 0, f"absolute {pair[0]} shift (width-independent)"),
            ("absolute", 1, f"absolute {pair[1]} shift (width-independent)"),
        ]

    mock_width = max(len(m) for m in mocks) + 2
    for key, index, title in tables:

        def value(run, mock):
            v = run["shifts"][mock][key]
            return float(v) if index is None else float(v[index])

        # The floor applies to the sigma-normalised statistics only. An "absolute" row is in raw
        # parameter units, where a shift of 0.07 means something entirely different.
        normalised = key != "absolute"

        def row(name, values, floor_key):
            # '=' marks a row the runs do not actually separate on: their spread is inside what two
            # runs of the SAME config produce anyway (Q2_FLOOR).
            spread = max(values) - min(values)
            mark = " =" if (normalised and len(values) > 1 and spread < Q2_FLOOR[floor_key]) else ""
            return f"{name:<{mock_width}}" + "".join(f"{v:>{width}.3f}" for v in values) + mark

        lines += ["", f"=== {title}", ""]
        lines.append(f"{'mock':<{mock_width}}" + "".join(f"{r['label']:>{width}}" for r in runs))
        lines.append("-" * (mock_width + width * len(runs)))
        for mock in mocks:
            lines.append(row(mock, [value(r, mock) for r in runs], "mock"))
        lines.append("-" * (mock_width + width * len(runs)))
        for name, reduce_fn in (("MEAN |.|", np.mean), ("MAX |.|", np.max)):
            lines.append(row(name, [reduce_fn([abs(value(r, m)) for m in mocks]) for r in runs], name))
        # Source clustering is the axis that has actually discriminated architectures, so it gets its
        # own line rather than being averaged into the rest.
        gate = [m for m in mocks if "source_clustering" in m and "no_sys" not in m]
        if gate:
            values = [max(abs(value(r, m)) for m in gate) for r in runs]
            lines.append(row("MAX |.| source_clustering", values, "MAX |.|"))

    lines += [
        "",
        "This is posterior LOCATION and is a DIFFERENT observable from the paired FoM, which sees only",
        "WIDTH. A run can win one and lose the other; that is a trade-off to report, not to average.",
        "Rank informativeness with run_comparison -- never with this table, and never the reverse.",
        "",
        f"WASH FLOOR: {Q2_FLOOR['mock']:.2f} sigma per mock, {Q2_FLOOR['MEAN |.|']:.2f} on MEAN, "
        f"{Q2_FLOOR['MAX |.|']:.2f} on MAX. Rows marked '=' are inside it and",
        "carry NO information about which run is more robust. Clearing it is necessary, not sufficient.",
        "",
        f"That is the MEASURED twin-pair scatter ({Q2_SCATTER['mock']:.3f} sigma per mock) times "
        f"{Q2_CONSERVATISM}, for the uncertainty on a",
        "one-pair estimate and for scanning every row of the table at once. It is deliberately",
        "demanding: two runs inside it are a coin flip, and a coin flip is a reason to keep the SIMPLER",
        "architecture, not to carry the more complex one. The scatter is seed-dominated (sigma_seed",
        "0.049 vs sigma_MC 0.015 per run), so longer chains cannot lower it -- only more twin pairs can.",
        "",
        "Training LENGTH moves this statistic by more than the floor: the same config at 250k vs 129k",
        "steps differs by 0.25 sigma RMS (source_clustering_gatti 0.19 -> 0.76). Compare Q2 only between",
        "runs at a comparable step count, or the number is about the budget, not the architecture.",
    ]
    return "\n".join(lines)


def sbc_null_scales(n_mocks):
    """Analytic 1-sigma scales of the two SBC statistics under perfect calibration.

    DERIVED FROM `n_mocks`, NOT HARDCODED. Q1's seed floor and Q2's wash floor both had to be MEASURED
    from a twin pair because their nulls are empirical. Q3's is not: if the posterior is calibrated the
    normalised rank of the truth is exactly U(0, 1), so both scales follow from the mock count in
    closed form and no twin run is needed. Hardcoding the 1000 that today's coverage stage happens to
    write is the same mistake as the hardcoded mock list this module's docstring already calls out.

      bias        SE of the sample mean of U(0,1)     = sqrt((1/12) / n)
      dispersion  SE of the sample variance / (1/12)  = sqrt((mu4 - sigma^4) / n) / (1/12),
                  with the fourth central moment mu4 = 1/80 and sigma^4 = 1/144

    At n = 1000 that is 0.0091 on the bias and 0.028 on the dispersion ratio.
    """
    var_uniform = 1.0 / 12.0
    bias = np.sqrt(var_uniform / n_mocks)
    dispersion = np.sqrt((1.0 / 80.0 - var_uniform**2) / n_mocks) / var_uniform
    return {"bias": float(bias), "dispersion": float(dispersion)}


def flow_prior_selection(inference_dir):
    """`prior_selection` the flow in this directory recorded, or None if it wrote no flow_config.yaml."""
    path = os.path.join(inference_dir, "flow_config.yaml")
    if not os.path.isfile(path):
        return None
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _dig(cfg, ("diagnostics", "prior_selection"))


def assert_wide_prior(inference_dir):
    """Gate: the coverage mocks' truths must come from the wide analysis prior.

    REFUSES rather than warns. A Sobol-grid truth set is narrower than the prior the flow was trained
    against, so the truth lands too near the centre of every posterior, and SBC and HPD both report
    that as overconfidence. There is no way to read past it: the whole table would be measuring the
    prior mismatch. If a run legitimately has no `flow_config.yaml`, that is also a refusal -- an
    unrecorded prior selection is not evidence of the right one.
    """
    selection = flow_prior_selection(inference_dir)
    if selection != PRIOR_SELECTION_REQUIRED:
        raise ValueError(
            f"{inference_dir}: prior_selection is {selection!r}, not {PRIOR_SELECTION_REQUIRED!r}. The "
            f"coverage truths were not drawn from the wide analysis prior, which fakes overconfidence "
            f"in both SBC and HPD -- every number in a Q3 table built on this run would be measuring "
            f"that mismatch rather than the density estimator."
        )


def sbc_ranks(h5, n_params, n_samples):
    """SBC rank of the truth among the posterior samples, per parameter. Yields (index, ranks).

    Restates `msi/utils/diagnostics.py::run_sbc_precomputed`'s rank line -- keep the two in step. Two
    deliberate differences, neither of which changes the statistic:

      * it reads ONE PARAMETER AT A TIME out of the h5 rather than materialising the whole
        (n_samples, n_mocks, n_params) array. That array is ~400 MB at float32 per run and a table
        spans 40 runs; a parameter slice is ~40 MB.
      * it does not draw the data-averaged-posterior samples. Those exist only to feed sbi's c2st,
        which needs torch and is not computed here -- and msi draws them with an UNSEEDED
        `np.random.randint`, so its logged `c2st_dap` is not reproducible run to run anyway.

    A rank is a COMPARISON COUNT, so it is exact in float32: there is no sample-mean reduction over
    theta to accumulate badly, which is what inflated every Q2 shift by ~5-6% when it was done in a
    scratch script. It is also invariant under any monotone reparametrisation of a parameter, so the
    `bary_Mc` raw-vs-log10 convention cannot corrupt it either.
    """
    theta_true = h5["theta_true"][:]
    for j in range(n_params):
        samples = h5["theta_sample"][:, :, j]
        assert samples.shape[0] == n_samples
        yield j, np.sum(samples < theta_true[None, :, j], axis=0)


def hpd_ecp(log_prob_true, log_prob_sample, n_alpha=100):
    """Per-mock HPD coverage indicators and the credibility grid: (alpha, ecp) of shapes (a,), (n, a).

    Restates `msi/utils/diagnostics.py::posterior_hpd_check` (Hermans+ crisis-SBI algorithm 1,
    arXiv:2302.03026) -- keep the two in step -- with one difference: it returns the per-mock indicator
    MATRIX instead of averaging over mocks immediately. That is what lets the caller bootstrap the
    deviation over mocks; msi only ever needed the mean because it went straight to a plot.

    The credibility grid and the array length both come from the same stride, so alpha and ecp stay
    aligned when n_samples is not a multiple of n_alpha, exactly as the msi version does.
    """
    n_samples = log_prob_sample.shape[0]
    step = max(1, n_samples // n_alpha)
    cls_indices = np.arange(0, n_samples, step)
    alpha = cls_indices / n_samples
    # Descending sort per mock, then the log-prob at each credibility level; the truth is inside the
    # level's HPD region when its log-prob is at least that threshold.
    descending = np.sort(log_prob_sample, axis=0)[::-1]
    log_prob_at_cls = descending[cls_indices, :]
    return alpha, (log_prob_at_cls <= log_prob_true[None, :]).T


def coverage(run_dirs, flow_name=None, n_alpha=100, n_boot=4000, seed=0, reference=None):
    """Q3 table: is each run's DENSITY ESTIMATOR calibrated? A gate, never a ranking.

    Coverage is invariant to how informative the compression is -- a lossy summary simply yields a
    BROADER posterior, which a well-trained estimator learns and reports honestly, and in the limit of
    a summary carrying nothing the posterior IS the prior: perfectly calibrated and useless. So this
    cannot rank compressions and must never be read as if it could. What it can do is flag a
    PATHOLOGICAL summary -- heavy-tailed, degenerate, ill-conditioned -- that defeats the flow, which
    is the precondition for Q1 and Q2 meaning anything at all.
    """
    import h5py

    loaded, warnings = _load_runs(run_dirs, flow_name=flow_name, finder=find_flow_dir)
    runs = []
    for meta in loaded:
        inference_dir, params = meta["inference_dir"], meta["params"]
        assert_wide_prior(inference_dir)
        entry = dict(meta)
        with h5py.File(os.path.join(inference_dir, "mcmc_samples.h5"), "r") as h:
            n_samples, n_mocks, n_params = h["theta_sample"].shape
            if n_params != len(params):
                raise ValueError(
                    f"{inference_dir}/mcmc_samples.h5: theta_sample has {n_params} parameter columns "
                    f"but the run declares {len(params)} {params}. The column order cannot be "
                    f"established, so no per-parameter number out of this file would be trustworthy."
                )
            scales = sbc_null_scales(n_mocks)
            entry["sbc"] = {}
            for j, ranks in sbc_ranks(h, n_params, n_samples):
                normalised = ranks.astype(np.float64) / n_samples
                bias = float(normalised.mean() - 0.5)
                dispersion = float(normalised.var(ddof=1) * 12.0)
                entry["sbc"][params[j]] = {
                    "bias_z": bias / scales["bias"],
                    "dispersion_z": (dispersion - 1.0) / scales["dispersion"],
                    "bias": bias,
                    "dispersion": dispersion,
                    "ks_p": _ks_uniform_p(normalised),
                }
            alpha, ecp = hpd_ecp(h["log_prob_true"][:], h["log_prob_sample"][:], n_alpha=n_alpha)
            entry["real_idx"] = h["real_idx"][:]
        # Kept for the paired contrast below: (n_mocks, n_alpha) of bools is ~100 kB, unlike the
        # theta_sample array it came from.
        entry["ecp"] = ecp
        entry["alpha"] = alpha

        # Bootstrap over MOCKS, not over the alpha grid: the curve is strongly autocorrelated in alpha
        # (each level is a nested subset of the next), so an alpha-wise error bar understates nothing
        # and means nothing. Resampling rows of the indicator matrix is the same bootstrap-over-mocks
        # `run_comparison.paired_ratio` uses, and it is nearly free here.
        deviation = ecp.mean(axis=0) - alpha
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, ecp.shape[0], (n_boot, ecp.shape[0]))
        boot = np.array([ecp[rows].mean(axis=0).mean() - alpha.mean() for rows in draws])
        entry["hpd"] = {
            "mean_deviation": float(deviation.mean()),
            "max_abs_deviation": float(np.abs(deviation).max()),
            "ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        }
        entry.update({"n_mocks": n_mocks, "n_samples": n_samples, "scales": scales})
        runs.append(entry)

    # Only parameters every run actually has can be tabulated side by side; the rest are named, not
    # hidden. Normally a no-op because Q3 is read per probe, but the probes genuinely differ (10
    # parameters on combined, 6 on lensing, 7 on clustering) and a cross-probe invocation is legal.
    per_run = [set(r["sbc"]) for r in runs]
    common = [p for p in runs[0]["params"] if all(p in s for s in per_run)] if runs else []
    dropped = sorted(set.union(*per_run) - set(common)) if per_run else []
    if not common:
        raise ValueError("the runs share no parameter in common; there is nothing to tabulate")
    if dropped:
        warnings.append(f"not present in every run, left out of the table: {dropped}")

    # The gate is in units of the analytic sigma, and that sigma depends on the mock count. Runs with
    # different mock counts are on different scales and their z-scores are not comparable.
    counts = sorted({r["n_mocks"] for r in runs})
    if len(counts) > 1:
        warnings.append(
            f"the runs were evaluated on different numbers of coverage mocks ({counts}), so the "
            f"analytic sigma differs between them and the z-scores below are not on one scale"
        )
    if reference is not None:
        paired_hpd(runs, reference, n_boot=n_boot, seed=seed, warnings=warnings)
    return {
        "runs": runs,
        "params": common,
        "warnings": warnings,
        "n_alpha": len(alpha),
        "reference": reference,
    }


def paired_hpd(runs, reference, n_boot=4000, seed=0, warnings=None):
    """Attach each run's PAIRED HPD difference against `reference`, in place.

    WHY PAIRED, and it is the same argument as the Q1 FoM's. Two runs' unpaired HPD CIs are dominated
    by BETWEEN-MOCK variance -- whether the truth of mock i happens to sit in the bulk or the tail of
    its own posterior -- which is a property of the mock, identical for both runs, and pure noise for
    the comparison. The coverage mock sets ARE identical across these runs, so that variance can be
    differenced away instead of tolerated: per mock, take the fraction of credibility levels whose HPD
    region contains the truth, difference it between the runs, and average over mocks. The alpha grid
    cancels exactly in the difference, which is why this needs no reference curve.

    Rows are matched on the FULL (i_sobol, i_signal, i_noise) tuple via `align_to`, never on position
    -- the same rule, for the same reason, as everywhere else in these two modules.

    THE CI IS OVER MOCKS AND IS NOT A REPRODUCIBILITY INTERVAL. Q1 has a measured 1.5% seed scatter and
    Q2 a measured 0.073 sigma; Q3 has NO measured seed scatter, because no v18 twin pair exists. So a
    paired difference excluding zero says the two runs really do differ ON THESE MOCKS -- it does not
    say the difference would survive a reseed. Treat it as a screen, not a verdict.
    """
    ref = next((r for r in runs if r["label"] == reference or r["run_dir"] == reference), None)
    if ref is None:
        raise ValueError(f"reference {reference!r} is not among the runs {[r['label'] for r in runs]}")
    rng = np.random.default_rng(seed)
    n_mocks = ref["ecp"].shape[0]
    draws = rng.integers(0, n_mocks, (n_boot, n_mocks))
    # Fraction of credibility levels containing the truth, per mock.
    ref_frac = ref["ecp"].mean(axis=1)
    for run in runs:
        if run is ref:
            run["paired"] = {"delta": 0.0, "ci": (0.0, 0.0), "win_frac": 0.0}
            continue
        if run["alpha"].shape != ref["alpha"].shape or not np.allclose(run["alpha"], ref["alpha"]):
            raise ValueError(
                f"{run['label']} and {reference} were evaluated on different credibility grids "
                f"({run['alpha'].size} vs {ref['alpha'].size} levels); the paired difference is not "
                f"defined across grids"
            )
        order = align_to(ref["real_idx"], run["real_idx"])
        # `order` maps run rows onto reference rows, so invert it to put the run in reference order.
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        delta = run["ecp"].mean(axis=1)[inverse] - ref_frac
        boot = delta[draws].mean(axis=1)
        run["paired"] = {
            "delta": float(delta.mean()),
            "ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
            "win_frac": float((delta > 0).mean()),
        }
    if warnings is not None:
        warnings.append(
            f"paired HPD is against {reference}; its CI is over MOCKS and is NOT a reproducibility "
            f"interval -- no Q3 seed scatter has ever been measured, so this screens, it does not settle"
        )


def _ks_uniform_p(normalised_ranks):
    """Two-sided KS p-value of the normalised SBC ranks against U(0, 1).

    Reported for continuity with what `msi/utils/coverage.py` logs (via sbi's `check_sbc`), NOT gated
    on. With 1000 mocks the test rejects at deviations far too small to threaten a posterior: on
    v18 combined/bench_v10_mean_std_k20 the logged p-values on (Om, s8, w0) are 1e-05, 1e-04 and 2e-07
    while the accompanying c2st is only 0.55, 0.55, 0.53. A p-value-only table condemns every run and
    separates none of them, which is why the z-scores above lead and this trails.
    """
    from scipy import stats

    return float(stats.kstest(normalised_ranks, "uniform").pvalue)


def format_coverage(result):
    """Render the Q3 table: parameters down the page, runs across it, in units of the analytic sigma."""
    runs, params = result["runs"], result["params"]
    width = max(14, max(len(r["label"]) for r in runs) + 2)
    gate = Q3_CONSERVATISM
    lines = [
        "Q3 estimator validity: is each run's DENSITY ESTIMATOR calibrated? A GATE, NOT A RANKING.",
        "",
        "HPD expected coverage vs credibility, deviation from the diagonal. NEGATIVE = below the",
        "diagonal = OVERCONFIDENT. The CI is bootstrapped over mocks; one excluding 0 is a real offset.",
        f"{'run':<{width}}{'steps':>9}{'mocks':>8}{'samples':>9}{'mean_dev':>11}" f"{'95% CI':>20}{'max_dev':>10}",
    ]
    for r in runs:
        lo, hi = r["hpd"]["ci"]
        lines.append(
            f"{r['label']:<{width}}{r['steps']:>9d}{r['n_mocks']:>8d}{r['n_samples']:>9d}"
            f"{r['hpd']['mean_deviation']:>11.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>20}"
            f"{r['hpd']['max_abs_deviation']:>10.4f}"
        )
    if result.get("reference") is not None:
        lines += [
            "",
            f"PAIRED HPD against {result['reference']}, differenced per mock on the shared mock set.",
            "Positive delta = this run's truth falls inside MORE credibility levels = better covered.",
            "'*' marks a CI excluding 0. Over MOCKS ONLY -- see the footer before calling one real.",
            f"{'run':<{width}}{'steps':>9}{'delta':>11}{'95% CI (mocks)':>22}{'win%':>7}",
        ]
        for r in runs:
            lo, hi = r["paired"]["ci"]
            tag = "  <- reference" if r["label"] == result["reference"] else ("" if lo <= 0.0 <= hi else "  *")
            lines.append(
                f"{r['label']:<{width}}{r['steps']:>9d}{r['paired']['delta']:>11.4f}"
                f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{r['paired']['win_frac'] * 100:>6.0f}%{tag}"
            )
    for warning in result["warnings"]:
        lines.append(f"  WARNING {warning}")

    tables = [
        (
            "bias_z",
            "SBC rank bias [analytic sigma] -- mean(rank/N) - 0.5, so the SIGN is the direction",
            "away from 0 = the truth sits systematically off-centre = BIASED",
        ),
        (
            "dispersion_z",
            "SBC rank dispersion [analytic sigma] -- var(rank/N)/(1/12) - 1",
            "POSITIVE = U-shaped = OVERCONFIDENT; negative = ranks piled at the centre = too broad",
        ),
    ]
    param_width = max(max(len(p) for p in params), len("n beyond gate")) + 2
    for key, title, reading in tables:
        lines += ["", f"=== {title}", f"    {reading}", ""]
        lines.append(f"{'param':<{param_width}}" + "".join(f"{r['label']:>{width}}" for r in runs))
        lines.append("-" * (param_width + width * len(runs)))
        for name in params:
            values = [r["sbc"][name][key] for r in runs]
            # The gate is per CELL against the known null, not a spread across the row: Q3 asks whether
            # THIS estimator is calibrated, which is an absolute question with an exact answer.
            mark = "" if all(abs(v) < gate for v in values) else " *"
            lines.append(f"{name:<{param_width}}" + "".join(f"{v:>{width}.2f}" for v in values) + mark)
        lines.append("-" * (param_width + width * len(runs)))
        peak = [max(abs(r["sbc"][p][key]) for p in params) for r in runs]
        lines.append(f"{'MAX |z|':<{param_width}}" + "".join(f"{v:>{width}.2f}" for v in peak))
        beyond = [sum(abs(r["sbc"][p][key]) >= gate for p in params) for r in runs]
        lines.append(f"{'n beyond gate':<{param_width}}" + "".join(f"{v:>{width}d}" for v in beyond))

    lines += ["", "=== SBC KS p-value vs uniform -- REPORTED, NOT GATED ON (see the footer)", ""]
    lines.append(f"{'param':<{param_width}}" + "".join(f"{r['label']:>{width}}" for r in runs))
    lines.append("-" * (param_width + width * len(runs)))
    for name in params:
        values = [r["sbc"][name]["ks_p"] for r in runs]
        lines.append(f"{name:<{param_width}}" + "".join(f"{v:>{width}.2e}" for v in values))

    lines += [
        "",
        "COVERAGE CANNOT RANK COMPRESSIONS. A lossy summary yields a BROADER posterior, which a",
        "well-trained estimator learns and reports honestly -- in the limit of a summary carrying",
        "nothing the posterior IS the prior: perfectly calibrated and completely useless. What this",
        "table can do is flag a PATHOLOGICAL summary that defeats the flow. Rank informativeness with",
        "run_comparison, robustness with `robustness` -- never with this table, and never the reverse.",
        "",
        f"GATE: |z| < {gate:.1f} analytic sigma. Rows marked '*' hold at least one run beyond it. The",
        "sigma is DERIVED, not measured: calibrated ranks are exactly U(0,1), so `sbc_null_scales` gets",
        f"it from the mock count in closed form ({runs[0]['scales']['bias']:.4f} on the bias, "
        f"{runs[0]['scales']['dispersion']:.3f} on the dispersion at",
        f"n={runs[0]['n_mocks']}). The factor {gate} is the scanning term ONLY -- Q1 and Q2 carry a further",
        "x1.3 for the uncertainty on a measured one-pair scatter, which has no referent when the null",
        "is exact. Raising it to 3.25 for consistency with those tables is always safe.",
        "",
        "A SHARED MILD REJECTION ON THE COSMOLOGICAL PARAMETERS IS EXPECTED AND IS NOT EVIDENCE ABOUT",
        "ANY ARCHITECTURE. Every run measured so far -- including the transformer, whose compression has",
        "nothing in common with the GCNNs' -- rejects mildly on Om/s8/w0 and is clean on every nuisance.",
        "That makes it a property of the MOCK SET, not of a network. Judge an arm by whether it departs",
        "from the cohort's shared pattern, never by whether it rejects at all. This is also why the KS",
        "p-value is reported and not gated on: at 1000 mocks it rejects on deviations far too small to",
        "threaten a posterior, so it would condemn every run and separate none of them.",
        "",
        "PASSING HERE IS A WEAK PASS. The flow's own train split cuts WITHIN permutation 19 (it trains",
        "on perms 16-18 plus perm 19 patches 0,1 while these mocks are perm 19 patches 2,3), so the",
        "holdout is slightly optimistic for every run. Uniform across runs, so a comparison is fair.",
        "",
        "NO TARP NUMBER IS COMPUTED HERE: the `tarp` package is not in this venv, and hand-rolling DRP",
        "is exactly the re-derivation this module exists to prevent. THE NUMBERS ABOVE ARE THE ANCHOR --",
        "and they are the only pristine ones. tarp 0.1.1's bootstrap mutates its caller's arrays in",
        "place (drp.py:174-176, ~100 of 1000 mocks overwritten by duplicates, cumulatively), and msi",
        "runs HPD -> TARP -> TARP marginals -> SBC -> l-C2ST over one shared pair of arrays. So",
        "2_posterior_{tarp,tarp_marginals,sbc}.png, the SBC KS/c2st values in the run logs and the",
        "l-C2ST p-value are all computed AFTER that corruption; only 2_posterior_hpd.png and the h5",
        "this table reads predate it. Verified against sbi in torch_env, 2026-08-19.",
        "",
        "If you do consult the TARP plot, read it QUALITATIVELY and by its SYMMETRY ABOUT (0.5, 0.5):",
        "antisymmetric and crossing at 0.5 = overconfident, a single bow entirely on one side = biased.",
        "Do NOT pattern-match 'above the diagonal' to overconfidence -- that call has been made and",
        "corrected twice.",
    ]
    return "\n".join(lines)


def comparable_to_fiducial(variant):
    """Whether a DES chain variant may be divided by the run's fiducial chain (see PRIOR_QUALIFIERS)."""
    return not any(qualifier in variant for qualifier in PRIOR_QUALIFIERS)


def _chain_fom(chain_dir, label, params, pair):
    """FoM and (Om, S8) summary of one chain, or None if that chain was not sampled.

    `fom_per_mock` reduces over axis 0 of an (n_samples, n_mocks, n_params) array; a single chain is
    one "mock". Going through it rather than repeating det(cov)**-0.5 here is what guarantees this is
    the same FoM the ranking table uses.
    """
    chain = load_chain(chain_dir, label, params)
    if chain is None:
        return None
    plane = _plane(chain, params, pair)
    return {
        "fom": float(fom_per_mock(chain[:, None, :], params, pair=pair)[0]),
        "sigma": plane.std(axis=0, ddof=1),
        "mean": plane.mean(axis=0),
    }


def des_fom(
    run_dirs, variants=("DESy3",), pair=("Om", "S8"), reference=None, fiducial_label=FIDUCIAL_LABEL, flow_name=None
):
    """Q4 diagnostic: FoM on the real DES Y3 data vector, per run, normalised WITHIN each run.

    Same statistic as the grid FoM that ranks runs -- literally the same function -- evaluated on the
    real observation instead of the correctly-specified coverage mocks. That one substitution removes
    its sign: a misspecified data vector maps to a summary the estimator was never trained to
    describe, which can land where the flow is sharp or where it is diffuse with no systematic
    direction, and a narrow posterior can be narrow in the wrong place. Read it for consistency with
    the grid ranking; never as a ranking.

    The headline column is therefore DES/fid, not the raw FoM: the same run's own `fiducial_bench`
    chain, sampled by the same flow in the same directory with the same function, is the only
    like-for-like reference available for a single data vector. That ratio is what "the real data is
    less constraining than a correctly-specified mock" actually means, it is dimensionless, and it is
    comparable across runs in a way the absolute FoM is not -- an absolute FoM mixes the divergence
    with how informative the compression is, which is Q1's question and is answered properly by
    `run_comparison`. This is the same referencing rule as `posterior_shifts`: against the run's own
    fiducial, never across runs.

    Caveat kept in view: `fiducial_bench_mean` is a NOISELESS mean data vector while DESy3 is one noisy
    realization, so the ratio carries a realization scatter that no single number here can separate.
    """
    loaded, warnings = _load_runs(run_dirs, flow_name=flow_name)
    reference_label = None
    runs = []
    for meta in loaded:
        chain_dir, params = meta["inference_dir"], meta["params"]
        entry = {**meta, "variants": {}}
        entry["fiducial"] = _chain_fom(chain_dir, f"{fiducial_label}_mean", params, pair)
        if entry["fiducial"] is None:
            warnings.append(
                f"{meta['label']}: no chain_{fiducial_label}_mean.npy, so its DES FoM can only be read "
                f"in absolute terms -- there is nothing within the run to normalise it against"
            )
        for variant in variants:
            columns = chain_params(params, variant)
            summary = _chain_fom(chain_dir, variant, columns, pair)
            if summary is not None:
                entry["variants"][variant] = summary
        missing = [v for v in variants if v not in entry["variants"]]
        if missing:
            # Named rather than left to be inferred from a run's absence from one table: a run that
            # quietly drops out of a variant reads as agreement between the runs that remain.
            warnings.append(f"{meta['label']}: not sampled for {missing}")
        runs.append(entry)
        if reference is not None and os.path.normpath(meta["run_dir"]) == os.path.normpath(reference):
            reference_label = meta["label"]
    if reference is not None and reference_label is None:
        raise ValueError(f"--reference {reference} is not among the runs")
    return {
        "runs": runs,
        "variants": list(variants),
        "pair": tuple(pair),
        "reference": reference_label,
        "warnings": warnings,
    }


def format_des_fom(result):
    """Render the Q4 table, with the caveat attached to the numbers rather than left to the reader."""
    runs, pair = result["runs"], result["pair"]
    width = max(16, max(len(r["label"]) for r in runs) + 2)
    lines = [
        "Q4 DES Y3 FoM -- UNSIGNED DIAGNOSTIC, NOT A RANKING.",
        "Higher is NOT better here: on a misspecified observation the FoM has no direction, and",
        "ranking on it selects for confident wrongness. Use it only to flag a DES-vs-sim divergence.",
        "",
        "Read the DES/fid column, not FoM: it is this run's DES posterior against this run's OWN",
        f"{FIDUCIAL_LABEL} posterior, same flow and same statistic, so it isolates the divergence from",
        "how informative the compression is (that is Q1 -- use run_comparison). Below 1 means the real",
        "data constrains less than a correctly-specified mock. The absolute FoM column mixes the two",
        "and is comparable across runs only in the loosest sense.",
    ]
    for warning in result["warnings"]:
        lines.append(f"  WARNING {warning}")

    for variant in result["variants"]:
        present = [r for r in runs if variant in r["variants"]]
        if not present:
            continue
        base = None
        if result["reference"] is not None:
            match = [r for r in present if r["label"] == result["reference"]]
            base = match[0]["variants"][variant]["fom"] if match else None
        lines += ["", f"=== chain_{variant}.npy"]
        comparable = comparable_to_fiducial(variant)
        if not comparable:
            lines.append(
                "    DES/fid not formed: this variant restricts the sampled space or the prior while the"
                " fiducial mock was only sampled with the default one, so the ratio would measure that"
                " restriction, not the data-vs-sim divergence."
            )
        lines.append("")
        header = f"{'run':<{width}}{'steps':>9}{'DES/fid':>10}{'FoM':>12}{'FoM_fid':>12}"
        if base is not None:
            header += f"{'/ref':>9}"
        header += f"{'sig_' + pair[0]:>12}{'sig_' + pair[1]:>12}{'mean_' + pair[0]:>12}{'mean_' + pair[1]:>12}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in present:
            v, fid = r["variants"][variant], r["fiducial"]
            show = fid if comparable else None
            ratio = f"{v['fom'] / show['fom']:>10.3f}" if show else f"{'n/a':>10}"
            fid_fom = f"{show['fom']:>12.1f}" if show else f"{'n/a':>12}"
            row = f"{r['label']:<{width}}{r['steps']:>9d}{ratio}{v['fom']:>12.1f}{fid_fom}"
            if base is not None:
                row += f"{v['fom'] / base:>9.3f}"
            row += f"{v['sigma'][0]:>12.4f}{v['sigma'][1]:>12.4f}{v['mean'][0]:>12.4f}{v['mean'][1]:>12.4f}"
            if r["label"] == result["reference"]:
                row += "  <- reference"
            lines.append(row)
    lines += [
        "",
        "DES/fid well below 1 across every run is a data-vs-sim mismatch signal worth investigating.",
        "One run diverging from the others is about that run -- but it is not evidence that it is worse,",
        f"and the ratio carries the scatter of a single noisy realization against a noiseless " f"{FIDUCIAL_LABEL}.",
        "The '/ref' column, if shown, is a raw across-run FoM ratio: NOT the paired per-mock median",
        "ratio that run_comparison reports, and it has no bootstrap CI and no seed floor. Do not read",
        "the two side by side as if they were the same quantity.",
    ]
    return "\n".join(lines)


def _resolve(root, names):
    return [os.path.join(root, n) if root else n for n in names]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Every subcommand takes the run list, the root and the flow name. Only the two chain-reading ones
    # take --pair and --fiducial: `coverage` is per-parameter over the run's whole parameter list and
    # references the truth, not a fiducial mock, so offering it either would imply a knob that is not
    # there.
    for name in ("robustness", "des-fom", "coverage"):
        sub = subparsers.add_parser(name)
        sub.add_argument("runs", nargs="+", help="run directory names under --root (or full paths)")
        sub.add_argument("--root", type=str, default="", help="common parent directory of the runs")
        sub.add_argument(
            "--flow_name",
            type=str,
            default=None,
            help="which flow to read when a run holds more than one (e.g. ensemble_flow)",
        )
        if name == "coverage":
            sub.add_argument(
                "--reference",
                type=str,
                default=None,
                help="run to difference the HPD coverage against, PAIRED on the shared mock set; "
                "needed to compare two runs' calibration, since the unpaired CIs are dominated by "
                "between-mock variance that is common to both",
            )
            continue
        sub.add_argument("--pair", type=str, nargs=2, default=["Om", "S8"], help="parameter pair")
        sub.add_argument(
            "--fiducial", type=str, default=FIDUCIAL_LABEL, help="label of the correctly-specified baseline mock"
        )

    robust = subparsers.choices["robustness"]
    robust.add_argument(
        "--full", action="store_true", help="also print the marginal and absolute shifts per parameter"
    )

    des = subparsers.choices["des-fom"]
    # Comma-separated rather than nargs="+": a variadic option placed before the positional run list
    # swallows it, and "--variants A run1 run2" then fails with an unhelpful argparse error.
    des.add_argument(
        "--variants",
        type=lambda s: [v for v in s.split(",") if v],
        # lambdaCDM and nla are in the default set on purpose: they are the two variants that drop a
        # column, so the handling that exists for them is exercised by the default invocation rather
        # than only when someone remembers to ask. A variant a run was not sampled for is skipped and
        # named in the warnings, so listing one costs nothing.
        default=["DESy3", "DESy3_w0gt-1", "DESy3_lambdaCDM", "DESy3_w0gt-1_nla"],
        help="comma-separated chain variants to tabulate; lambdaCDM drops the w0 column and nla drops "
        "bta, both handled (default: DESy3,DESy3_w0gt-1,DESy3_lambdaCDM,DESy3_w0gt-1_nla)",
    )
    des.add_argument("--reference", type=str, default=None, help="run whose FoM the others are shown against")

    args = parser.parse_args(argv)
    dirs = _resolve(args.root, args.runs)

    if args.command == "coverage":
        reference = args.reference
        if reference is not None:
            resolved = os.path.join(args.root, reference) if args.root else reference
            if resolved not in dirs:
                dirs = [resolved] + dirs
            reference = run_labels(dirs)[dirs.index(resolved)]
        print(format_coverage(coverage(dirs, flow_name=args.flow_name, reference=reference)))
    elif args.command == "robustness":
        result = robustness(dirs, pair=tuple(args.pair), fiducial_label=args.fiducial, flow_name=args.flow_name)
        print(format_robustness(result, full=args.full))
    else:
        reference = os.path.join(args.root, args.reference) if (args.root and args.reference) else args.reference
        if reference is not None and reference not in dirs:
            dirs = [reference] + dirs
        result = des_fom(
            dirs,
            variants=tuple(args.variants),
            pair=tuple(args.pair),
            reference=reference,
            fiducial_label=args.fiducial,
            flow_name=args.flow_name,
        )
        print(format_des_fom(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
