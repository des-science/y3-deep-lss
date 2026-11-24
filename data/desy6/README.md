# `data/desy6/`

The products behind the DES Y6 scale cuts: the smoothing scales, the ℓ cuts and the noise levels
the networks are trained with. One script makes them, `deep_lss/apps/products/desy6.py`, for one
dataset version of msfm's DES Y6 forward model at a time. The msfm config
`configs/desy6/<version>/<version>.yaml` there selects the version, whose scales config is
`configs/scales/desy6/<version>/8wl,32gc.yaml` here and whose products are in `<version>/` of this
directory. The fiducial IA and bias amplitudes enter the signal, so every version has its own. The
script writes only there and to the config lines marked `# <- step values: <dotted.key>`. With
`--provenance`, it requires committed code here and in msfm and records each step in
`<version>/provenance.yaml`: its command, inputs, outputs and both commits. The submission scripts
take the scales config of the dataset they run on, `VERSION`.

## Status

| version | step | output | in git | status |
|---|---|---|---|---|
| v1 | `signal` | `v1/desy6_signal_cls.h5` | yes | done 2026-11-21 |
| v1 | `values` | the marked lines of `configs/scales/desy6/v1/8wl,32gc.yaml` | yes | done 2026-11-21 |
| v2 | `signal` | `v2/desy6_signal_cls.h5` | yes | done 2026-11-24, provenance not yet recorded |
| v2 | `values` | the marked lines of `configs/scales/desy6/v2/8wl,32gc.yaml` | yes | done 2026-11-24, provenance not yet recorded |

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
- **Base kernel.** The networks smooth each probe by repeating one kernel of FWHM
  `theta_fwhm_base`, `base_repetitions` times per bin, at the probe's `base_nside` (512 for
  lensing, 256 for clustering), in every run. On a sphere sampled this coarsely the kernel deepsphere
  builds is narrower than its nominal Gaussian (2.6% at 10.4′ on nside 512, 7.4% at 8.3′). So the
  step builds that kernel on the footprint, takes its realized FWHM w from the kernel's second
  moment, and repeats it n = ⌈(θ / w)²⌉ times, which never smooths a bin less than its beam: the
  effective FWHM w·√n ≥ θ. Of the bases ≤ min θ whose repetitions total no more than Y3's rule
  (a nominal base just above min θ / √2), it takes the one with the smallest worst overshoot.

Three differences to Y3, all deliberate:

- **Fiducial maps from the benchmarks.** Y3 took the signal from the fiducial set. DES Y6 has no
  fiducial projection, so the signal is the mean over the 20 permutations of the benchmark
  `fiducial_bench`. It has the fiducial cosmology, the fiducial set's resolution (900 Mpc/h,
  576M particles, 69 shells) and the config's fiducial n(z), to 4e-8 in every bin.
- **Normalized mean redshift.** The mean redshift is normalized by the n(z) sum. The DES Y6 n(z)
  files integrate to 1 on dz = 0.01, so they sum to 100. Applied to the Y3 files (sums
  0.963-0.994 for the sources, 1 for the lenses), the Y3 derivation's unnormalized mean
  reproduces the committed Y3 lensing beams exactly. So the Y3 source mean redshifts are 0.6-3.7%
  low, and the Y3 lensing beams 1-3% wider than 8 Mpc/h. The Y3 clustering beams are exact.
  `l_max` and the noise follow the rounded beam the networks use; Y3 took them from the unrounded
  one, which moves its `l_max` by 1-3.
- **One base kernel per probe, in every run.** Y3 sets a base for the clustering of its combined
  runs only, from the nominal width (40.4′, +2.9% at most); everything else takes the smallest
  beam as its base, so Y3's clustering-only runs smooth MagLim 3 at +18.4% and its lensing reaches
  +18.8%. Here each probe has its own base kernel, the same in its single-probe and combined
  runs, so the two differ only in what the network sees.

v1 still followed Y3's nominal rule, for clustering only (35.5′). Its clustering-only runs used that
base too, through a key that has since been removed, so the v1 scales config now smooths them with
the smallest beam as the base, and its sha256 no longer matches `v1/provenance.yaml`.

## Running

```bash
P=/users/athomsen/dlss/repos/.claude/bin/plot.sh   # torch_env on the login node
V=v2
M=/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/configs/desy6/$V/$V.yaml
SRC=/capstor/store/cscs/userlab/lp184/tomaszk/261109_cosmogridv1_desy6/000_test/cosmogrid/processed/desy6/CosmoGrid/bary
SC=configs/scales/desy6/$V/8wl,32gc.yaml

# signal: anafast of 20 permutations, 5 s; the pool fills the CPUs of the process (48 in a daint login
# session). taskset narrows the login session's CPU set further
OMP_NUM_THREADS=4 $P python -m deep_lss.apps.products.desy6 signal \
    --msfm-config $M --scales $SC --source $SRC --provenance
# values: half a minute, most of it building the base kernels
$P python -m deep_lss.apps.products.desy6 values --msfm-config $M --scales $SC --provenance
$P python -m deep_lss.apps.products.desy6 values --msfm-config $M --scales $SC --check   # exit 1 if stale
```

`--dry-run` lists a step's inputs and outputs. `signal --max-perms 1` is the smoke test; it writes
nothing.

## Steps

### `signal`: signal power spectra

- **Inputs:** the msfm config and the projection's
  `benchmarks/fiducial_bench/perm_00{00..19}/projected_probes_maps_v11dmb.h5`.
- **Output:** `desy6_signal_cls.h5`:
  - `cls` of shape (3 n_side, n_z), the mean over the permutations: source bins, then lens bins;
  - the fiducial `tomo_Aia`, `tomo_bg` and `ng_bar` the maps were built with.
- **Checks:**
  - every map is one nside-512 map, and the file has every configured bin (`maglim5`, `maglim6`
    exist only in the re-projected files);
  - the spectra are finite and positive.

### `values`: the scale cuts in the config

- **Inputs:** the msfm config, its n(z) and pixel file, `desy6_signal_cls.h5`, and each probe's
  `base_nside` from the scales config.
- **Output:** `theta_fwhm`, `l_max`, `white_noise_sigma`, `theta_fwhm_base` and
  `base_repetitions` of both probes.
- **Checks:**
  - the signal file was made with the config's bins, fiducial IA amplitudes, bias and n_gal;
  - the signal has one spectrum per configured bin;
  - no base kernel realizes more than its nominal FWHM, which the search for the base relies on;
  - the rewritten config differs from the old one only in the marked values.

## Results

v1 (2026-11-22), next to Y3's `configs/scales/8wl,32gc.yaml`:

| | bins | mean z | `theta_fwhm` (′) | `l_max` | Y3 `theta_fwhm` (′) |
|---|---|---|---|---|---|
| lensing | metacal 1-4 | 0.41, 0.54, 0.85, 1.16 | 24.3, 19.2, 13.2, 10.4 | 715, 905, 1316, **1535** | 29.5, 20.1, 15.0, 12.6 |
| clustering | MagLim 1, 3-6 | 0.31, 0.62, 0.78, 0.90, 1.01 | 128.1, 67.8, 56.5, 50.2, 46.1 | 136, 256, 307, 346, 377 | 130.2, 88.9, 68.1, 57.0 |

- **metacal 4** has no 1% crossing below the band limit. Its 10.4′ beam still leaves 2.0% of the
  squared beam at ℓ = 1535, so its `l_max` and noise are taken there.
- **`theta_fwhm_base`** is 35.5′: 25 repetitions (14, 4, 3, 2, 2), effective FWHM 132.8, 71.0, 61.5,
  50.2, 50.2′. That is +8.9% at most, in MagLim 4 and 6.
- **Noise and cosmology.** `white_noise_sigma` follows the signal power. The grid cosmology
  nearest the fiducial in prior-normalized distance, `cosmo_132332` (S8 0.72 against 0.78), gives
  0.90, 0.87, 0.88, 0.88 of the fiducial lensing noise and 0.97, 0.88, 0.95, 0.93, 0.96 of the
  clustering noise.

v2 (2026-11-24) has v1's beams and `l_max`, which depend only on the cosmology and the n(z). Its base
kernels follow the realized width, for both probes:

| | `base_nside` | `theta_fwhm_base` (′) | realized (′) | `base_repetitions` | effective FWHM (′) | overshoot |
|---|---|---|---|---|---|---|
| lensing | 512 | 8.3 | 7.69 | 10, 7, 3, 2 | 24.31, 20.34, 13.31, 10.87 | +0.0 to +5.9% |
| clustering | 256 | 36.1 | 35.60 | 13, 4, 3, 2, 2 | 128.36, 71.20, 61.66, 50.35, 50.35 | +0.2 to +9.2% |

v1's realized overshoots were -2.7 to +8.5% (lensing, base 10.4′) and -1.9 to +6.8% (clustering,
35.5′), so metacal 4 and MagLim 5 were smoothed slightly less than their beams. Over the footprint,
the realized width of one kernel varies by about 1% between pixels (1st to 99th percentile, lensing),
and the repetitions use its mean.

v2's fiducial IA (A_IA 0.13, η 2.0) and bias (1.54, 1.85, 1.76, 1.93, 1.90) change the
signal, so its noise is 1.11, 1.03, 1.01, 1.01 × v1's in lensing and 1.14, 1.23, 1.12, 1.19, 1.14 ×
v1's in clustering, the latter in proportion to the bias.
