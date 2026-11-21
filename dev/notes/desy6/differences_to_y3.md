# Y6 v1 networks vs our Y3 v18: differences

Dev-phase note (2026-11-21). Repo docs must not point here.

y3-deep-lss runs the same code on both surveys. Which survey a run belongs to comes from the msfm
config's `survey.name`. This list records every place where a Y6 training or evaluation differs
from the Y3 v18 production runs, so that no difference is silent. Kinds in the table:

- **upstream**: follows from the msfm forward model;
- **decision**: a stated user decision (D numbers from `plan.md`);
- **method**: the same quantity, derived another way.

Y3 side: the v18 production runs (`maps_gcnn/<probe>/v1`, `maps_gcnn_nocls/<probe>/v1`,
`cls/<probe>/v1`), `configs/scales/8wl,32gc.yaml`, `configs/probes/*.yaml`, arXiv:2511.04681.
Y6 side: `configs/scales/desy6/8wl,32gc.yaml`, `configs/probes/desy6/`, `data/desy6/README.md`,
and `plan.md` (this folder). The forward-model differences are in msfm
`dev/notes/desy6/differences_to_y3.md`, the inference differences in msi
`dev/notes/desy6/differences_to_y3.md`.

## Identical (the networks)

- **Networks.** The production net configs `configs/maps/prod/<arch>/<probe>/{maps,maps+cls}.yaml`
  (GCNN `classic`) and `configs/cls/`, loss `vmim`, and the data split `configs/data/default.yaml`
  (80% of the cosmology realizations for training, all noise realizations on both sides).
- **Training.** Grid only, `n_steps: auto` under the same wall budgets (1 × 12 h for lensing and
  clustering, 2 × 12 h for combined), `mirrored` on one 4-GPU node.
- **Scale-cut definition** (`8wl,32gc`):
  - Gaussian beams of 8 Mpc/h (sources) and 32 Mpc/h (lenses) at each bin's mean z, in msfm's
    fiducial cosmology;
  - `l_max` at the 1% crossing of the squared beam;
  - white noise equal to the smoothed signal power at `l_max`;
  - `l_min` 30 and `n_sigma_support: 3`.
  The smoothing runs on the fly in the same per-probe smoothing layer.
- **Lensing parameters.** `configs/probes/desy6/lensing.yaml` is a copy of Y3's:
  `Om, s8, w0, Aia, n_Aia, bta`.
- **Evaluation.** The held-out grid split, and the default DES observation built from msfm's
  catalog maps.

## Differences

| # | item | Y3 v18 | Y6 v1 | kind | note |
|---|---|---|---|---|---|
| 1 | input maps | msfm `v18`, `DESy3_grid_dmb_????.tfrecord`; 4 source + 4 lens bins | msfm `desy6/v1`, `DESy6_grid_dmb_????.tfrecord`; 4 source + 5 lens bins (MagLim 1, 3-6) | upstream | 9 smoothing channels instead of 8 |
| 2 | bias parameters | `bg1`-`bg4` | `bg1, bg3, bg4, bg5, bg6`, named by survey bin, in msfm order | upstream | `configs/probes/desy6/`. 11 labels for combined (Y3 10), 8 for clustering (Y3 7) |
| 3 | lensing `theta_fwhm` | 29.5, 20.1, 15.0, 12.6′ | 24.3, 19.2, 13.2, 10.4′ | upstream, method | the deeper Y6 sources (mean z 0.41-1.16). The mean z is now normalized by the n(z) sum; Y3's derivation skipped that, so the Y3 beams are 1-3% wider than 8 Mpc/h (documented only, Y3 stays frozen) |
| 4 | lensing `l_max` | 589, 863, 1159, 1382 | 715, 905, 1316, **1535** | decision (D6) | metacal 4 has no 1% crossing below the band limit 3 n_side − 1 = 1535. It keeps its 8 Mpc/h beam and takes `l_max` at the band limit, with 2.0% of the squared beam left; the step warns |
| 5 | clustering `theta_fwhm` | 130.2, 88.9, 68.1, 57.0′ | 128.1, 67.8, 56.5, 50.2, 46.1′ | upstream | the Y6 lens n(z), mean z 0.31-1.01 |
| 6 | clustering `l_max` | 133, 195, 255, 305 | 136, 256, 307, 346, 377 | upstream | |
| 7 | `theta_fwhm_base` | 40.4′, just above min θ / √2; repetitions 11, 5, 3, 2; at most +2.9% | 35.5′, the smallest worst overshoot at no more repetitions than Y3's rule; repetitions 14, 4, 3, 2, 2; at most +8.9% | decision (D7) | Y3's rule gives +15.4% on Y6, because MagLim 4-6 sit between its coarse √n steps. On the Y3 beams the new rule returns 40.4′ |
| 8 | `white_noise_sigma` | from the fiducial signal (msfm `notebooks/desy3/scale_cuts.ipynb`) | from `cosmo_132332`, the grid cosmology nearest the fiducial, mean over 20 perms | decision (D1, D8) | Y6 has no fiducial projection. The next four nearest give 7-31% more lensing noise. Re-derive on the fiducial projection |
| 9 | DES observations | `DESy3`, `DESy3_no_sys`, `DESy3_no_psi_rot`, `DESy3_no_sys_no_psi_rot` | `DESy6` only | decision (D9) | the test variants are opt-in through `--des_variants` (`evaluation.des_variants`). The default stays on for Y3, whose frozen runs and msi chains carry them |
| 10 | mock observations | from `OBS_INPUT`: Buzzard (`buzzard_flock*`), `fiducial_bench*`, and the box-size, particle-count and redshift-resolution mocks | none | upstream | no Y6 fiducial projection and no Buzzard-like Y6 mock; `--include_mocks` finds no files |

## Not there yet

- **Production training.** It waits for msfm's production grid tfrecords (`INPUT`). The smoke
  training passed on the 24 msfm benchmark tfrecords (job 5416191, 2026-11-21).
- **The Cls path and maps+cls.** Both need msfm's merged `grid_cls.h5` of the production grid.
- **Inference.** The infer stage runs msi as on Y3 and needs msi's branch `desy6`. No Y6 inference
  has run yet.
- **Other probes.** 2x2pt and 3x2pt are not ported: msfm's `lenses_before_sources` works by bin
  position, which is wrong for lens labels 1, 3-6. Neither are the `_nla`, `_min`, `_ext`,
  `cross` and `clustering_auto` variants.
- **Fiducial diagnostics.** The DES labels of `run_diagnostics.py` and its `fiducial_bench` mock
  wait for the msi port and a Y6 fiducial mock.

## Plumbing (no effect on the networks)

- **Survey selection.** `submissions/clariden/common.sh` reads `SURVEY` from `MSFM_CONFIG` (`DESy3`
  when the file is missing). From it, the script picks `configs/{probes,scales}/desy6/` and the
  `<survey>_grid_dmb_` glob. `INPUT` and `TRAIN_TFR` can be overridden. Y6 usage:
  `VERSION=desy6/v1 STAGES="train eval" INPUT=...`. Runs land in `runs/desy6/v1/default/`, with the
  W&B tag `desy6/v1`.
- **Guards.** `configuration.check_scale_cuts` (`scale_cuts.survey` must equal `survey.name`, and
  each per-bin list must have one entry per bin) runs in the three entry points. `run_comparison`
  gates on `survey.name`, so a Y3 run never pairs with a Y6 run.
- **Scale-cut values as products.** `deep_lss.apps.products.desy6` (steps `signal` and `values`)
  writes the marked config lines, with `--check`, and with `--provenance` into
  `data/desy6/provenance.yaml` (both commits). The helpers are shared through msfm
  `msfm/utils/products.py`. Y3: a notebook, its values pasted into the YAML.
- **Names.** `evaluate_des_y3` is now `evaluate_des`; the observation keys are
  `obs/preds/<survey>...`.

## Environment

- Y3 v18: Clariden GH200, account `a0158`.
- Y6: daint GH200, account `lp184`, in the same `tensorflow` container with `tf_env`. The products
  run in `torch_env` on the login node.
