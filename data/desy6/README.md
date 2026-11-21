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
| `signal` | `desy6_signal_cls.h5` | yes | not run |
| `values` | the marked lines of `configs/scales/desy6/8wl,32gc.yaml` | yes | not run |

## Method

The method is that of the DES Y3 `configs/scales/8wl,32gc.yaml`, derived for Y3 in msfm
`notebooks/desy3/scale_cuts.ipynb`:

- **Beam.** For each bin, a Gaussian beam whose FWHM is a fixed transverse comoving scale at the
  bin's mean redshift. That is 8 Mpc/h for the sources and 32 Mpc/h for the lenses
  (`scale_cuts.<probe>.transverse_scale`), in msfm's fiducial cosmology, rounded to 0.1 arcmin.
- **ℓ cut.** `l_max` is the first ℓ where the squared beam falls below `scale_cuts.threshold`
  (0.01).
- **Noise.** `white_noise_sigma` is the flat white noise whose power equals the smoothed signal
  power at `l_max`: `sigma = sqrt(high_pass² · low_pass² · C_ℓ(l_max) / Ω_pix)`. Here
  `high_pass` is the ℓ_min = 30 filter, and the signal is lensing `κ + A_IA I` and clustering
  galaxy counts at the fiducial bias and n_gal.
- **Base kernel.** `theta_fwhm_base` of the clustering kernel is just above the smallest clustering
  FWHM / √2, so that every ratio of the per-probe smoothing stays below 2 under the ceil.

Two differences to Y3, both deliberate:

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

## Running

```bash
P=/users/athomsen/dlss/repos/.claude/bin/plot.sh   # torch_env on the login node
M=/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/configs/desy6/v1/default.yaml
SRC=/capstor/store/cscs/userlab/lp184/tomaszk/261109_cosmogridv1_desy6/000_test/cosmogrid/processed/desy6/CosmoGrid/bary
SC=configs/scales/desy6/8wl,32gc.yaml

# signal: anafast of 5 cosmologies x 20 permutations, about a minute on 16 cores
OMP_NUM_THREADS=4 taskset -c 0-15 $P python -m deep_lss.apps.products.desy6 signal \
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
