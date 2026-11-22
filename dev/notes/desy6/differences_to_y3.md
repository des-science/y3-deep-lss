# Y6 v1 networks vs our Y3 v18: differences

Dev-phase note (2026-11-21). Repo docs must not point here.

y3-deep-lss runs the same code on both surveys. Which survey a run belongs to comes from the msfm
config's `survey.name`. This list records every place where a Y6 training or evaluation differs
from the Y3 v18 production runs, so that no difference is silent. Kinds in the table:

- **upstream**: follows from the msfm forward model, or from the same rule applied to Y6 inputs;
- **decision**: a stated user decision (D numbers from `plan.md`);
- **method**: the same quantity, derived another way;
- **history**: the Y3 run predates today's repo config, which Y6 uses.

Y3 side: the saved `configs.yaml` of the nine v18 production runs on the store
(`maps_gcnn/<probe>/v1`, `maps_gcnn_nocls/<probe>/v1`, `cls/<probe>/v1`),
`configs/scales/8wl,32gc.yaml`, `configs/probes/*.yaml`, arXiv:2511.04681. Y6 side:
`configs/scales/desy6/8wl,32gc.yaml`, `configs/probes/desy6/`, `data/desy6/README.md`, and
`plan.md` (this folder). The forward-model differences are in msfm
`dev/notes/desy6/differences_to_y3.md`, the inference differences in msi
`dev/notes/desy6/differences_to_y3.md`.

## Identical

- **Networks.** The production net configs `configs/maps/prod/deepsphere/<probe>/{maps,maps+cls}.yaml`
  (GCNN `classic`) and `configs/cls/branch/default.yaml`, loss `vmim`, and the data split
  `configs/data/default.yaml` (80% of each cosmology's realizations for training, all noise
  realizations on both sides; the Y6 grid has the same 2500 × 20 × 4 × 5 layout). Checked key for
  key against the nine saved v18 configs on 2026-11-21: eight match, `maps_gcnn/combined/v1`
  differs in its data loader (row 13).
- **Code.** Since the v18 runs trained, the training path changed only in how net configs are
  composed (they resolve to the same keys), a new standalone Cls net, msfm's tf.data RAM cap and
  msfm's survey-aware file lookups. deepsphere is unchanged.
- **Training.** Grid only, `n_steps: auto` under the same wall budgets (1 × 12 h for lensing and
  clustering, 2 × 12 h for combined), `mirrored` on one 4-GPU node. The step count follows each
  survey's throughput, so it is not matched (neither was it across the Y3 runs).
- **Scale-cut definition** (`8wl,32gc`):
  - Gaussian beams of 8 Mpc/h (sources) and 32 Mpc/h (lenses) at each bin's mean z, in msfm's
    fiducial cosmology, which is the same on both surveys;
  - `l_max` at the 1% crossing of the squared beam;
  - white noise equal to the smoothed signal power at `l_max`;
  - `l_min` 30 and `n_sigma_support: 3`.
- **Smoothing rule.** The same on-the-fly layer applies one base kernel ⌈(θ / base)²⌉ times per
  bin. The base is `theta_fwhm_base` for the clustering of the combined runs (per-probe path,
  clustering at nside 256). Lensing in every run takes base = min θ of its channels, and so do Y3's
  clustering-only runs (single-kernel path). Y6's clustering-only runs differ (row 9), and the
  overshoot on the Y6 beams differs too (rows 7-9).
- **Lensing parameters.** `configs/probes/desy6/lensing.yaml` is a copy of Y3's:
  `Om, s8, w0, Aia, n_Aia, bta`.
- **Evaluation.** The held-out grid split with its `grid_(...)` examples, and the default DES
  observation built from msfm's catalog maps.

## Differences

| # | item | Y3 v18 | Y6 v1 | kind | note |
|---|---|---|---|---|---|
| 1 | input maps | msfm `v18`, `DESy3_grid_dmb_????.tfrecord`; 4 source + 4 lens bins | msfm `desy6/v1`, `DESy6_grid_dmb_????.tfrecord`; 4 source + 5 lens bins (MagLim 1, 3-6) | upstream | 9 smoothing channels instead of 8 |
| 2 | bias parameters | `bg1`-`bg4` | `bg1, bg3, bg4, bg5, bg6`, named by survey bin, in msfm order | upstream | `configs/probes/desy6/`. 11 labels for combined (Y3 10), 8 for clustering (Y3 7) |
| 3 | lensing `theta_fwhm` | 29.5, 20.1, 15.0, 12.6′ | 24.3, 19.2, 13.2, 10.4′ | upstream, method | the deeper Y6 sources (mean z 0.41-1.16). The mean z is now normalized by the n(z) sum; Y3's derivation skipped that, so the Y3 beams are 1-3% wider than 8 Mpc/h (documented only, Y3 stays frozen) |
| 4 | lensing `l_max` | 589, 863, 1159, 1382 | 715, 905, 1316, **1535** | decision (D6), method | metacal 4 has no 1% crossing below the band limit 3 n_side − 1 = 1535. It keeps its 8 Mpc/h beam and takes `l_max` at the band limit, with 2.0% of the squared beam left; the step warns. Y6 takes `l_max` and σ from the rounded beam the networks use. Y3 took them from the unrounded one, so its lensing `l_max` are 1-3 off the rounded beams' crossings (589, 864, 1158, 1379) |
| 5 | clustering `theta_fwhm` | 130.2, 88.9, 68.1, 57.0′ | 128.1, 67.8, 56.5, 50.2, 46.1′ | upstream | the Y6 lens n(z), mean z 0.31-1.01 |
| 6 | clustering `l_max` | 133, 195, 255, 305 | 136, 256, 307, 346, 377 | upstream | Y3's values from the unrounded beams equal the rounded beams' crossings |
| 7 | `theta_fwhm_base` (clustering) | 40.4′, just above min θ / √2; repetitions 11, 5, 3, 2; at most +2.9% | 35.5′, the smallest worst overshoot at no more repetitions than Y3's rule; repetitions 14, 4, 3, 2, 2; at most +8.9% | decision (D7) | Y3's rule gives +15.4% on Y6, because MagLim 4-6 sit between its coarse √n steps. On the Y3 beams the new rule returns 40.4′. Y3 uses it in the combined runs only, Y6 in the clustering-only runs too (row 9); lensing ignores it (row 8) |
| 8 | lensing effective FWHM, every run | base 12.6′; repetitions 6, 3, 2, 1; 30.9, 21.8, 17.8, 12.6′; at most +18.8% (metacal 3) | base 10.4′; repetitions 6, 4, 2, 1; 25.5, 20.8, 14.7, 10.4′; at most +11.4% (metacal 3) | upstream | the same rule on the new beams. Y6 lensing sits closer to 8 Mpc/h than Y3 (metacal 3 at 8.9 against 9.5 Mpc/h). The smoke log shows the repetitions 6, 4, 2, 1 |
| 9 | clustering-only base kernel and effective FWHM | base = min θ = 57.0′; repetitions 6, 3, 2, 1; 139.6, 98.7, 80.6, 57.0′; at most +18.4% (MagLim 3) | `theta_fwhm_base` 35.5′ through `theta_fwhm_base_single_probe: True`; repetitions 14, 4, 3, 2, 2; 132.8, 71.0, 61.5, 50.2, 50.2′, as the combined runs' clustering; at most +8.9% | decision (D11) | Y3's base rule would give 46.1′ on Y6 and smooth MagLim 5 at 65.2′ (+29.9%, 41.6 Mpc/h instead of 32). The key is opt-in and absent from every Y3 config, so Y3 clustering-only runs, frozen ones included, keep base = min θ (checked on all 36 Y3 scales × probe × arch combinations) |
| 10 | `white_noise_sigma` | from the fiducial signal (msfm `notebooks/desy3/scale_cuts.ipynb`) | from the benchmark `fiducial_bench`, mean over its 20 perms | decision (D12) | Y6 has no fiducial projection. `fiducial_bench` has the fiducial cosmology, the fiducial set's resolution and the config's fiducial n(z), so only the simulation set differs. The grid cosmology nearest the fiducial (`cosmo_132332`, S8 0.72) would give 10-13% less lensing noise |
| 11 | DES observations | `DESy3`, `DESy3_no_sys`, `DESy3_no_psi_rot`, `DESy3_no_sys_no_psi_rot` | `DESy6` only | decision (D9) | the test variants are opt-in through `--des_variants` (`evaluation.des_variants`). The default stays on for Y3, whose frozen runs and msi chains carry them |
| 12 | mock observations | from `OBS_INPUT`: Buzzard (`buzzard_flock*`), `fiducial_bench*`, and the box-size, particle-count and redshift-resolution mocks | `fiducial_bench`, `box_size`, `particle_count`, `redshift_resolution`, `fiducial_bench_source_clustering_in_place`, `fiducial_bench_source_clustering_no_sys` | upstream | made by msfm on 2026-11-22 from the four benchmark projections (`configs/desy6/v1/mock_observations.yaml`), in `/ritom/.../desy6/v1/default/obs/`. No dark-matter-only, Gatti, IA-shell or Buzzard-like Y6 mock |
| 13 | data loader, `maps_gcnn/combined/v1` | `examples_shuffle_buffer` 256, `n_readers` 32 (eval 64), `n_workers` AUTOTUNE | `maps/shared/dataloader.yaml`: 1024, 16, 128 | history | the run predates the composed prod configs (2026-09-04). Example order and throughput only; the network and loss keys match. The other five map runs already use today's loader |

## Not there yet

- **Production training.** It waits for msfm's production grid tfrecords (`INPUT`). The smoke
  training passed on the 24 msfm benchmark tfrecords (job 5416191, 2026-11-21).
- **The Cls path and maps+cls.** Both need msfm's merged `grid_cls.h5` of the production grid.
  The metacal 4 auto pair will be binned up to ℓ = 1535 (row 4), the band limit of the raw C_ℓ.
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
  `VERSION=desy6/v1 INPUT=... sbatch prod/maps.sh`, combined through `prod/maps_chain.sh`. Runs
  land in `runs/desy6/v1/default/`. W&B logs to the same project `y3-deep-lss`, told apart by the
  tag `desy6/v1`. On msfm's `desy6` checkout the Y3 default `VERSION=v18` is such a missing file
  (the config moved to `configs/desy3/v18/`), so Y3 runs go from a `main` checkout.
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
