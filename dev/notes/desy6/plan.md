# DES Y6 port of y3-deep-lss: plan

Dev-phase plan, started 2026-11-20 on branch `desy6`. Repo docs must not point here. The upstream
counterpart is msfm `dev/notes/desy6/plan.md`.

## Goal and rules

The goal is to train and evaluate the compression networks on the DES Y6 grid `.tfrecord`s, which
msfm builds from its `configs/desy6/v1/default.yaml`:

- `survey.name: DESy6`;
- 4 source bins and 5 lens bins (`maglim1,3,4,5,6`), with bias parameters `bg1,bg3,bg4,bg5,bg6`;
- files `DESy6_grid_dmb_????.tfrecord`, and the merge writes `cls/grid_cls.h5`;
- no fiducial set yet.

Rules:

- **Y3 stays bit-identical.** Every change defaults to the Y3 behaviour when its key is absent, and
  a Y3 regression proves it (Verification).
- **Nothing from Y3 enters Y6 unless a Y6 file says so.** The Y6 configs sit in survey subdirs and
  are selected from the msfm config's `survey.name`.
- **Every `sbatch` waits for the user's OK.**

## State (2026-11-21)

Items 1-7 of section 2 are done and committed (msfm `d0abff2`; y3-deep-lss `e55676b`..). The Y3
regression checks of section 4 (items 1-3) passed. The Y6 scale cuts are derived, with results in
`data/desy6/README.md`. The Y6 smoke training (section 4, item 4) passed on 2026-11-21 (job 5416191,
5:28):
- 200 steps on the 24 msfm benchmark tfrecords;
- 4 + 5 smoothing channels, with the predicted clustering repetitions [14, 4, 3, 2, 2];
- 11 labels, and the W&B tag `desy6/v1` accepted;
- the eval wrote `obs/preds/DESy6` only (D9) and 24 × 80 grid predictions, all finite.

The run is in `claude/bench/desy6_deep_lss_smoke/`. **Next:** the production tfrecords (msfm), the
Cls path on msfm's `grid_cls.h5`, and the msi port.

## 1. What is Y3-specific today (sweep of 2026-11-20)

The Python package is survey-agnostic already:

- bin counts come from `len(z_bins)`;
- bias names are looked up, not built;
- grid-only training needs no fiducial tfrecords;
- msfm's catalog loader reads the cached maps of `survey.name`.

What is not survey-agnostic:

1. `configs/scales/*.yaml` (4 clustering entries) and `configs/probes/*.yaml` (`bg2`).
2. The observation labels, hardcoded `"DESy3"` in `utils/evaluation.py` and
   `utils/cls_evaluation.py`.
3. Silent failure modes:
   - a short `l_min`/`l_max` list falls back to defaults (`get_cls_bounds_per_pair`);
   - a missing scales key gives `smoothing_kwargs = None` with only a warning;
   - msi selects only `DESy3*` observations, so a Y6 inference stage would produce no chain and no
     error;
   - the `run_comparison` gate has no survey field.
4. `submissions/clariden/common.sh`:
   - the `DESy3_` tfrecord glob;
   - `INPUT` is not overridable;
   - the probes/scales paths are not survey-aware.
5. The Y3 scales values were derived by hand. msfm `notebooks/desy3/scale_cuts.ipynb` was later
   ported to msfm `dev/scripts/noise_models/derive_scale_cut_noise.py`, and the values pasted into
   the YAML. Two Y6 traps:
   - `z_bar = sum(z * nz)` assumes n(z) sums to 1. Y3 sums to 0.994; Y6 sums to 100 (trapz on
     dz = 0.01), which gives z̄ = 41;
   - `bg{i+1}` is a KeyError on Y6.

## 2. Work, in order

1. **msfm: shared products helpers.** Move the marked-line writer, the provenance recorder and the
   clean-commit check from `msfm/apps/products/desy6.py` into `msfm/utils/products.py`. Unchanged
   behaviour, checked by `values --check`, an identical `--dry-run` of every step, and the `nz`
   record reproduced into a scratch file.
2. **Product steps `deep_lss/apps/products/desy6.py`**, the msfm pattern: steps write only their
   outputs; `values` rewrites the lines marked `# <- step values: <dotted.key>`; `--check`;
   `--provenance` records both commits in `data/desy6/provenance.yaml`; `data/desy6/README.md`
   documents each step.
   - `signal` (compute node, `--source` = the projection root):
     - writes the mean signal C_l over the 20 perms of the K grid cosmologies nearest the
       fiducial (prior-normalized distance over `cosmo` + `bary`) to `data/desy6/desy6_signal_cls.h5`;
     - lensing is `kg + A_IA ia`, clustering is the counts at the fiducial bias and n_gal;
     - re-projected files only.
   - `values` writes into `configs/scales/desy6/8wl,32gc.yaml`:
     - `theta_fwhm` (8 / 32 Mpc/h at the normalized mean z), `l_max` (the threshold crossing) and
       `theta_fwhm_base` (just above the smallest clustering FWHM / √2, rounded up to 0.1′);
     - `white_noise_sigma` from rank 0, with the spread over ranks 1..K-1 logged.
   - Cross-check: the beam math on the Y3 v18 config reproduces the committed `8wl,32gc.yaml`
     `theta_fwhm` and `l_max` to the n(z) normalization (~0.6%).
3. **Y6 configs.**
   - `configs/scales/desy6/8wl,32gc.yaml`: `scale_cuts.survey: DESy6`, the inputs
     `transverse_scale` and `threshold`, and the derived keys marked and `null` until `values` runs.
   - `configs/probes/desy6/{lensing,clustering,combined}.yaml`, with `bg1, bg3, bg4, bg5, bg6` in
     msfm order. 2x2pt/3x2pt are not ported: msfm's `lenses_before_sources` works by bin position.
4. **Run** `signal --provenance` on a compute node (debug), then `values --provenance`, and review
   the numbers before committing them.
5. **Python.**
   - The observation label is `survey.name` and `f"{survey}_no_sys"`;
     `evaluate_des_y3` → `evaluate_des`.
   - `configuration.check_scale_cuts`:
     - list lengths equal `len(z_bins)` per active probe;
     - `scale_cuts.survey` (default `DESy3`) equals `survey.name`;
     - called by `run_training`, `run_evaluation` and `run_cls_training+evaluation`.
   - A `survey` field in the `run_comparison` gate.
6. **Submission scripts.**
   - `common.sh`:
     - `SURVEY` from `$MSFM_CONFIG` (`DESy3` if the file is missing);
     - survey subdirs for `SCALES_CONFIG` and a new `PROBES_DIR`;
     - `INPUT` and `TRAIN_TFR` overridable, the glob prefix from `SURVEY`;
     - `require_y3_inference`, called after `STAGES=` in `prod/maps.sh` and `prod/cls.sh`.
   - Use `$PROBES_DIR` wherever `configs/probes/` was spelled out.
7. **Docs.** A DES Y6 paragraph in the README.

## 3. Decisions

- **D1** (2026-11-20): `white_noise_sigma` from the grid cosmology nearest the fiducial, as msfm's
  Y3 `notebooks/metacal_bias.ipynb` aliased the fiducial. σ enters sims and data alike, so a
  stand-in changes the effective scale cut, not a bias. Re-derive on the fiducial projection.
- **D2** (2026-11-20): DES Y6 is evaluated by default, as Y3 (prediction files only; no chains
  while msi is not ported).
- **D3** (2026-11-20): the scales values come from product steps in y3-deep-lss, with the helpers
  shared through msfm.
- **D4** (2026-11-20): Y6 configs in `configs/{probes,scales}/desy6/`. Y3 stays at the top level.
- **D5** (2026-11-20): bias names stay msfm's (`bg1, bg3-bg6`, msfm D3). y3-deep-lss looks names
  up. msi's `plotting.py` is the port that must accept them.
- **D6** (2026-11-21): metacal 4 keeps its 8 Mpc/h beam (10.4′) although its 1% crossing
  (ℓ ≈ 1670) lies above the band limit. Its `l_max` is the band limit 1535 (2.0% of the squared
  beam left), and the step warns. Rejected: capping the beam at the band-limit crossing (11.4′).
  The cause is the deeper Y6 sources (mean z 1.16 against Y3's 0.94).
- **D7** (2026-11-21): `theta_fwhm_base` is the base with the smallest overshoot at no more
  repetitions than the Y3 rule: 35.5′, +8.9% at most. The literal Y3 rule gives +15.4% on Y6, since
  MagLim 4-6 sit between the coarse √n steps. On the Y3 beams the new rule returns Y3's 40.4′.
- **D8** (2026-11-21): σ stays on the nearest grid cosmology although the next four give 7-31%
  more lensing noise (README, Results).
- **D9** (2026-11-21): DES Y6 evaluates only the default observation `DESy6`, the analogue of Y3's
  `DESy3`. The test variants `_no_sys` and `_no_psi_rot` are opt-in (`--des_variants`). Their
  default stays on for DES Y3, whose frozen runs and msi chains carry them. The msi port has to
  follow the same default.

## 4. Verification

1. **Y3 regression.**
   - `DRYRUN=1` of `prod/maps.sh` and `prod/cls.sh` with Y3 defaults, before and after: an empty
     diff.
   - `check_scale_cuts` passes on:
     - every Y3 scales config × the Y3 v16/v17/v18 msfm configs;
     - every frozen v18 run's saved `configs.yaml`.
   - The labels resolve to `DESy3`/`DESy3_no_sys`.
2. **Y6 negative tests.**
   - `check_scale_cuts` raises for Y3 scales on the Y6 config, and for the `null` skeleton.
   - Y3 `combined.yaml` on Y6 raises on `bg2`.
   - `values --check` exits 1 after a hand edit.
3. **Y6 dry run.** `VERSION=desy6/v1 DRYRUN=1 PROBE=combined INPUTS=maps STAGES="train eval"`; with
   the default STAGES it stops at `require_y3_inference`.
4. **Y6 smoke training** (debug, <30 min).
   - Inputs: the 24 tfrecords of the msfm CPU benchmark,
     `claude/bench/desy6_grid_cpu/s24/tfrecords/` (production schema, stand-in bias table).
   - Settings: `PROBE=combined INPUTS=maps TRAIN_EXTRA=--n_steps=200 STAGES="train eval"`, with
     `RUNS_ROOT` under `claude/`.
   - Expected: 9 smoothing channels, 11 labels, `obs/preds/DESy6{,_no_sys}{,_no_psi_rot}`, and a
     W&B tag `desy6/v1` that is accepted.

## 5. Later

- The Cls path and maps+cls on Y6 need msfm's merge `grid_cls.h5` of the production grid.
- `apps/tuning/run_diagnostics.py`: DES labels and the `fiducial_bench` mock, after the msi port.
- The msi port: obs keys, `bg` lists, `n_patches`/`n_noise`, runs configs.
- Where the production tfrecords live (`/ritom`, iopsstor, lp184 store): set by `INPUT` once msfm
  has run.
