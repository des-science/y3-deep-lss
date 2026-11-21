# `data/desy6/`

The products behind the DES Y6 scale cuts, `configs/scales/desy6/8wl,32gc.yaml`: the smoothing
scales, the ℓ cuts and the noise levels the networks are trained with. One script makes them,
`deep_lss/apps/products/desy6.py`, on the DES Y6 forward model of msfm
(`configs/desy6/v1/default.yaml` there). It writes only to this directory and to the config lines
marked `# <- step values: <dotted.key>`. With `--provenance`, it requires committed code here and in
msfm and records each step in `provenance.yaml`: its command, inputs, outputs and both commits.

## Status

| step | output | in git | status |
|---|---|---|---|
| `signal` | `desy6_signal_cls.h5` | yes | done 2026-11-21, 5 grid cosmologies × 20 permutations |
| `values` | the marked lines of `configs/scales/desy6/8wl,32gc.yaml` | yes | done 2026-11-21 |

## Method

The method is that of the DES Y3 `configs/scales/8wl,32gc.yaml`, derived for Y3 in msfm
`notebooks/desy3/scale_cuts.ipynb`:

- **Beam.** For each bin, a Gaussian beam whose FWHM is a fixed transverse comoving scale at the
  bin's mean redshift. That is 8 Mpc/h for the sources and 32 Mpc/h for the lenses
  (`scale_cuts.<probe>.transverse_scale`), in msfm's fiducial cosmology, rounded to 0.1 arcmin.
- **ℓ cut.** `l_max` is the first ℓ where the squared beam falls below `scale_cuts.threshold`
  (0.01). A beam too narrow to get there below the band limit 3 n_side − 1 keeps its width, and
  its `l_max` is the band limit; the step warns.
- **Noise.** `white_noise_sigma` is the flat white noise whose power equals the smoothed signal
  power at `l_max`: `sigma = sqrt(high_pass² · low_pass² · C_ℓ(l_max) / Ω_pix)`. Here
  `high_pass` is the ℓ_min = 30 filter, and the signal is lensing `κ + A_IA I` and clustering
  galaxy counts at the fiducial bias and n_gal.
- **Base kernel.** The per-probe clustering smoothing repeats one kernel of `theta_fwhm_base`
  ⌈(θ / base)²⌉ times, so its effective FWHMs are base·√n. The step takes the base ≤ min θ with the
  smallest worst overshoot among bases that cost no more repetitions than Y3's rule (just above
  min θ / √2). On the Y3 beams this rule gives Y3's 40.4′ exactly.

Three differences to Y3, all deliberate:

- **Fiducial stand-in.** There is no DES Y6 fiducial projection, so the signal is that of the grid
  cosmology nearest the fiducial: prior-normalized distance over the CosmoGrid parameters, averaged
  over all its permutations. The next nearest ones show how much the noise depends on that choice.
  The same noise enters the simulations and the data, so the stand-in moves the effective scale
  cut, not the centre of the posterior. Rerun both steps on the fiducial projection once it exists.
- **Normalized mean redshift.** The mean redshift is normalized by the n(z) sum. The DES Y6 n(z)
  files integrate to 1 on dz = 0.01, so they sum to 100. Applied to the Y3 files (sums
  0.963-0.994 for the sources, 1 for the lenses), the Y3 derivation's unnormalized mean
  reproduces the committed Y3 lensing beams exactly. So the Y3 source mean redshifts are 0.6-3.7%
  low, and the Y3 lensing beams 1-3% wider than 8 Mpc/h. The Y3 clustering beams are exact.
  `l_max` and the noise follow the rounded beam the networks use; Y3 took them from the unrounded
  one, which moves its `l_max` by 1-3.
- **Base kernel by overshoot.** Y3's rule only places the smallest bin. On Y3 the others happened
  to land just below a step (+2.9% at most). On Y6 it would give +15.4%: the extra lens bins
  MagLim 5 and 6 sit close to MagLim 4, between the coarse √2/√3/√4 steps.

## Running

```bash
P=/users/athomsen/dlss/repos/.claude/bin/plot.sh   # torch_env on the login node
M=/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/configs/desy6/v1/default.yaml
SRC=/capstor/store/cscs/userlab/lp184/tomaszk/261109_cosmogridv1_desy6/000_test/cosmogrid/processed/desy6/CosmoGrid/bary
SC=configs/scales/desy6/8wl,32gc.yaml

# signal: anafast of 5 cosmologies x 20 permutations; the pool fills the CPUs of the process
# (48 in a daint login session; 3:46 in one process). taskset narrows the login session's CPU set further
OMP_NUM_THREADS=4 $P python -m deep_lss.apps.products.desy6 signal \
    --msfm-config $M --scales $SC --source $SRC --provenance
# values: seconds
$P python -m deep_lss.apps.products.desy6 values --msfm-config $M --scales $SC --provenance
$P python -m deep_lss.apps.products.desy6 values --msfm-config $M --scales $SC --check   # exit 1 if stale
```

`--dry-run` lists a step's inputs and outputs. `signal --max-perms 1 --n-nearest 1` is the smoke
test; it writes nothing.

## Steps

### `signal`: signal power spectra

- **Inputs:** the msfm config, CosmoGridV1's metainfo (the grid parameters), and the projection's
  `grid/cosmo_*/perm_*/projected_probes_maps_v11dmb.h5`.
- **Output:** `desy6_signal_cls.h5`:
  - `cls` of shape (K, 3 n_side, n_z): source bins, then lens bins;
  - per cosmology, `cosmo_key`, `params` (`param_names`) and `distance`;
  - the fiducial `tomo_Aia`, `tomo_bg` and `ng_bar` the maps were built with.
- **Checks:**
  - every map is one nside-512 map, and the file has every configured bin (`maglim5`, `maglim6`
    exist only in the re-projected files);
  - the spectra are finite and positive.

### `values`: the scale cuts in the config

- **Inputs:** the msfm config and n(z), `desy6_signal_cls.h5`.
- **Output:** `theta_fwhm`, `l_max` and `white_noise_sigma` of both probes, and `theta_fwhm_base`
  of clustering.
- **Checks:**
  - the signal file was made with the config's bins, fiducial IA amplitudes, bias and n_gal;
  - the rewritten config differs from the old one only in the marked values;
  - the noise of the other cosmologies, relative to the nearest, is logged per bin.

## Results (2026-11-21)

The five grid cosmologies nearest the fiducial (Om 0.26, s8 0.84, S8 0.78), in prior-normalized
distance:

| rank | cosmology | distance | Om | s8 | w0 |
|---|---|---|---|---|---|
| 0 | `cosmo_132332` | 0.113 | 0.240 | 0.804 | −0.97 |
| 1 | `cosmo_040100` | 0.171 | 0.274 | 0.822 | −0.97 |
| 2 | `cosmo_143827` | 0.195 | 0.245 | 0.934 | −1.21 |
| 3 | `cosmo_096324` | 0.200 | 0.260 | 0.853 | −1.13 |
| 4 | `cosmo_041044` | 0.217 | 0.297 | 0.834 | −0.84 |

The scale cuts, from rank 0, next to Y3's `configs/scales/8wl,32gc.yaml`:

| | bins | mean z | `theta_fwhm` (′) | `l_max` | Y3 `theta_fwhm` (′) |
|---|---|---|---|---|---|
| lensing | metacal 1-4 | 0.41, 0.54, 0.85, 1.16 | 24.3, 19.2, 13.2, 10.4 | 715, 905, 1316, **1535** | 29.5, 20.1, 15.0, 12.6 |
| clustering | MagLim 1, 3-6 | 0.31, 0.62, 0.78, 0.90, 1.01 | 128.1, 67.8, 56.5, 50.2, 46.1 | 136, 256, 307, 346, 377 | 130.2, 88.9, 68.1, 57.0 |

- **metacal 4** has no 1% crossing below the band limit. Its 10.4′ beam still leaves 2.0% of the
  squared beam at ℓ = 1535, so its `l_max` and noise are taken there.
- **`theta_fwhm_base`** is 35.5′: 25 repetitions (14, 4, 3, 2, 2), effective FWHM 132.8, 71.0, 61.5,
  50.2, 50.2′. That is +8.9% at most, in MagLim 4 and 6.
- **Noise spread.** The noise of ranks 1-4 relative to rank 0, per bin:

| rank | lensing 1-4 | clustering 1, 3-6 |
|---|---|---|
| 1 | 1.20, 1.23, 1.22, 1.21 | 1.02, 1.06, 1.04, 1.01, 1.04 |
| 2 | 1.16, 1.21, 1.19, 1.14 | 1.12, 1.15, 1.13, 1.09, 1.08 |
| 3 | 1.09, 1.13, 1.10, 1.07 | 1.03, 1.08, 1.04, 1.01, 1.00 |
| 4 | 1.31, 1.31, 1.30, 1.29 | 1.04, 1.09, 1.10, 1.08, 1.08 |

  Rank 0 has the lowest S8 of the five (0.72), so the fiducial's lensing noise is likely 10-20%
  higher than the one in the config.
