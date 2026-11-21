# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen (with Claude)

Derives the DES Y6 scale cuts of configs/scales/desy6/, one step per product (data/desy6/README.md), on the msfm
DES Y6 forward model. Every step writes only to data/desy6/ and to the config lines marked
"# <- step values: <dotted.key>", and stops on a failed check. With --provenance, it requires committed code in this
repository and in msfm and records its inputs, outputs and both commits in data/desy6/provenance.yaml.

The method is the one of the DES Y3 configs/scales/8wl,32gc.yaml: a Gaussian beam per bin of a fixed transverse
comoving scale at the bin's mean redshift, l_max where the squared beam falls below the threshold, and white noise
equal to the smoothed signal power at l_max.

Steps, in order:
    signal   mean signal C_l (lensing kg + A_IA ia, clustering counts at the fiducial bias) of the grid cosmologies
             nearest the fiducial, over all their permutations, from the CosmoGrid projection (--source). The
             nearest is the stand-in for the fiducial cosmology, the next ones show how much the noise depends on it
    values   theta_fwhm, l_max, theta_fwhm_base and white_noise_sigma, written into the marked scales config lines

Usage:
    OMP_NUM_THREADS=4 python -m deep_lss.apps.products.desy6 signal --msfm-config <msfm>/configs/desy6/v1/default.yaml \
        --scales configs/scales/desy6/8wl,32gc.yaml --source <projection> --max-perms 1 --n-nearest 1   # smoke test
    OMP_NUM_THREADS=4 taskset -c 0-15 python -m deep_lss.apps.products.desy6 signal \
        --msfm-config <msfm>/configs/desy6/v1/default.yaml --scales configs/scales/desy6/8wl,32gc.yaml \
        --source <projection>   # about a minute; the pool fills the cores taskset leaves it
    python -m deep_lss.apps.products.desy6 values --msfm-config <msfm>/configs/desy6/v1/default.yaml \
        --scales configs/scales/desy6/8wl,32gc.yaml
    python -m deep_lss.apps.products.desy6 values ... --check   # exit 1 if stale

--dry-run lists a step's inputs and outputs and writes nothing. --max-perms N runs step signal on the first N
permutations of every cosmology and writes nothing, so a partial average can never pass for the product.
"""

import argparse
import multiprocessing
import os
import sys

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM

import msfm
from msfm.utils import clustering, cosmogrid, files, filenames, logger, parameters, products, redshift, scales
from msfm.utils import input_output
from msfm.utils import source_clustering_bias as scb

LOGGER = logger.get_logger(__file__)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MSFM_DIR = os.path.abspath(os.path.join(os.path.dirname(msfm.__file__), ".."))
SURVEY_NAME = "DESy6"
OUT_ROOT = "data/desy6"
SCALES_ROOT = "configs/scales/desy6"
STEP_ORDER = ["signal", "values"]
SIGNAL_FILE = "data/desy6/desy6_signal_cls.h5"
PROVENANCE_FILE = "data/desy6/provenance.yaml"
# the code a step runs, in this repository and in msfm
CODE_PATHS = ("deep_lss", "configs")
MSFM_CODE_PATHS = ("msfm", "configs")
# the probes of the scales config and their msfm galaxy samples, in channel order
PROBE_SAMPLES = {"lensing": "metacal", "clustering": "maglim"}


def _out_path(rel_path):
    """Absolute path of an output, which must lie under data/desy6/."""
    rel_path = os.path.normpath(rel_path)
    if os.path.isabs(rel_path) or not (rel_path + os.sep).startswith(OUT_ROOT + os.sep):
        raise ValueError(f"DES Y6 outputs must lie under {OUT_ROOT}/, got {rel_path}")
    return os.path.join(REPO_DIR, rel_path)


def _rel(path):
    """Repository-relative path for files inside this repository, absolute otherwise."""
    return products.rel_path(path, REPO_DIR)


def _fiducial_setup(conf):
    """The fiducial amplitudes the signal maps are built with: per-bin A_IA, lens bias and lens counts per pixel."""
    fid = conf["analysis"]["fiducial"]
    tomo_z, tomo_nz = files.load_redshift_distributions("metacal", conf)
    tomo_Aia = redshift.get_tomo_amplitudes(fid["Aia"], fid["n_Aia"], tomo_z, tomo_nz, conf["survey"]["metacal"]["z0"])
    tomo_bg = np.array([fid[p] for p in conf["analysis"]["params"]["bg"]["linear"]])
    n_side = conf["analysis"]["n_side"]
    ng_bar = np.array(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
    assert len(tomo_bg) == len(conf["survey"]["maglim"]["z_bins"]), "one linear bias per lens bin"
    return {"tomo_Aia": np.asarray(tomo_Aia, dtype=np.float64), "tomo_bg": tomo_bg, "ng_bar": ng_bar}


def _nearest_cosmologies(conf, n_nearest):
    """The grid cosmologies nearest the fiducial, in prior-normalized distance over the CosmoGrid parameters.

    Returns:
        tuple: (keys, params, distances, param names), nearest first
    """
    names = conf["analysis"]["params"]["cosmo"] + conf["analysis"]["params"]["bary"]
    info = cosmogrid.get_cosmo_params_info(os.path.join(MSFM_DIR, conf["files"]["meta_info"]), simset="grid")
    raw = np.stack([info[name] for name in names], axis=-1).astype(np.float64)
    grid = parameters.raw_to_prior_units(raw, names)
    fiducial = np.array(parameters.get_fiducials(names, conf), dtype=np.float64)
    priors = parameters.get_prior_intervals(names, conf)
    distance = np.sqrt(np.sum(((grid - fiducial) / (priors[:, 1] - priors[:, 0])) ** 2, axis=-1))
    order = np.argsort(distance)[:n_nearest]
    keys = [scb.cosmo_key(info["path_par"][i]) for i in order]
    return keys, grid[order], distance[order], names


def _perm_cls(task):
    """C_l of one permutation's signal maps, lensing bins then lens bins. A module function, for the pool."""
    path, metacal_bins, maglim_bins, n_side, setup, l_max, datapath = task
    with h5py.File(path, "r") as f:
        for key in [f"map/{m}/{b}" for m in ("kg", "ia") for b in metacal_bins] + [f"map/dg/{b}" for b in maglim_bins]:
            if key not in f:
                raise KeyError(f"{path} has no {key}, so it is not a re-projected DES Y6 file")
            if f[key].shape != (hp.nside2npix(n_side),):
                raise ValueError(f"{path}: {key} has shape {f[key].shape}, not one nside-{n_side} map")
        kg = np.stack([f[f"map/kg/{b}"][:] for b in metacal_bins], axis=-1)
        ia = np.stack([f[f"map/ia/{b}"][:] for b in metacal_bins], axis=-1)
        dg = np.stack([f[f"map/dg/{b}"][:] for b in maglim_bins], axis=-1)

    wl = kg + setup["tomo_Aia"] * ia
    mean_dg = np.mean(dg, axis=0)
    gc = clustering.galaxy_density_to_count(setup["ng_bar"], (dg - mean_dg) / mean_dg, setup["tomo_bg"])
    maps = np.concatenate([wl, gc], axis=-1)
    return np.stack(
        [
            hp.anafast(maps[:, i], lmax=l_max, pol=False, datapath=datapath, use_pixel_weights=True)
            for i in range(maps.shape[1])
        ],
        axis=-1,
    )


def step_signal(conf, scales_conf, args):
    """Writes the mean signal C_l of the n_nearest grid cosmologies nearest the fiducial to data/desy6/.

    As configs/scales/8wl,32gc.yaml did for Y3 on the fiducial maps (msfm notebooks/desy3/scale_cuts.ipynb): lensing
    is kg + A_IA ia with the fiducial IA amplitude per bin, clustering the galaxy counts of the fiducial linear bias and
    n_gal. Y6 has no fiducial projection yet, so the nearest grid cosmology stands in for it.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run or --max-perms
    """
    source = os.path.abspath(args.source)
    meta_path = os.path.join(MSFM_DIR, conf["files"]["meta_info"])
    out_path = _out_path(SIGNAL_FILE)
    inputs = [os.path.abspath(args.msfm_config), source, meta_path]
    if args.dry_run:
        return products.dry_run("signal", inputs, [out_path], rel=_rel)

    n_side = conf["analysis"]["n_side"]
    l_max = 3 * n_side - 1
    metacal_bins = conf["survey"]["metacal"]["z_bins"]
    maglim_bins = conf["survey"]["maglim"]["z_bins"]
    setup = _fiducial_setup(conf)
    datapath = os.path.join(MSFM_DIR, conf["files"]["healpy_data"])
    with_bary = conf["analysis"]["modelling"]["baryonified"]

    keys, params, distances, names = _nearest_cosmologies(conf, args.n_nearest)
    cosmo_dirs = scb.get_cosmogrid_dirs(conf, source, with_fiducial=False)
    n_perms = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    if args.max_perms is not None:
        n_perms = min(n_perms, args.max_perms)
    tasks = []
    for key in keys:
        for perm in range(n_perms):
            path = filenames.get_filename_full_maps(os.path.join(cosmo_dirs[key], f"perm_{perm:04d}"), with_bary)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} does not exist")
            tasks.append((path, metacal_bins, maglim_bins, n_side, setup, l_max, datapath))
    for key, p, d in zip(keys, params, distances):
        LOGGER.info(f"{key}: distance {d:.4f}, " + ", ".join(f"{n} {v:.4g}" for n, v in zip(names, p)))

    # healpy threads per task come from OMP_NUM_THREADS, so the pool fills the cores without oversubscribing them
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    n_procs = max(1, min(len(tasks), len(os.sched_getaffinity(0)) // n_threads))
    LOGGER.info(f"anafast of {len(tasks)} maps files in {n_procs} processes x {n_threads} threads")
    with multiprocessing.Pool(n_procs) as pool:
        perm_cls = np.stack(pool.map(_perm_cls, tasks)).reshape(len(keys), n_perms, l_max + 1, -1)
    mean_cls = np.mean(perm_cls, axis=1)
    assert np.all(np.isfinite(mean_cls)) and np.all(mean_cls[:, 2:] > 0), "non-finite or non-positive signal C_l"

    if args.max_perms is not None:
        LOGGER.info(f"--max-perms {args.max_perms}: the C_l passed every check, nothing is written")
        return None

    with h5py.File(out_path, "w") as f:
        f.create_dataset("cls", data=mean_cls)
        f.create_dataset("params", data=params)
        f.create_dataset("distance", data=distances)
        f.create_dataset("cosmo_key", data=np.array(keys, dtype="S"))
        f.create_dataset("tomo_Aia", data=setup["tomo_Aia"])
        f.create_dataset("tomo_bg", data=setup["tomo_bg"])
        f.create_dataset("ng_bar", data=setup["ng_bar"])
        f.attrs["param_names"] = names
        f.attrs["z_bins"] = metacal_bins + maglim_bins
        f.attrs["n_perms"] = n_perms
        f.attrs["n_side"] = n_side
        f.attrs["about"] = (
            "cls[k]: mean signal C_l (ell = 0..3 n_side - 1, axis 1; bins axis 2, lensing then clustering) over the "
            "permutations of the k-th grid cosmology nearest the fiducial"
        )
    return inputs, [out_path]


def _scale_cut(cls, fwhm, l_min, threshold, n_side):
    """l_max where the squared beam falls below threshold, and the white noise of the smoothed signal power there.

    Follows msfm dev/scripts/noise_models/derive_scale_cut_noise.py, the port of msfm notebooks/desy3/scale_cuts.ipynb.
    """
    ell = np.arange(len(cls))
    low_pass = scales.gaussian_low_pass_factor_alm(ell, theta_fwhm=fwhm, arcmin=True) ** 2
    high_pass = scales.gaussian_high_pass_factor_alm(ell, l_min=l_min) ** 2
    below = ell[low_pass < threshold]
    l_max = int(below[0]) if len(below) else int(ell[-1])
    cl_noise = high_pass[l_max] * low_pass[l_max] * cls[l_max]
    return l_max, float(np.sqrt(cl_noise / hp.nside2pixarea(n_side)))


def step_values(conf, scales_conf, args):
    """Writes the derived scale cuts into the scales config lines marked "# <- step values: <dotted.key>".

    theta_fwhm: the beam of scale_cuts.<probe>.transverse_scale (Mpc/h) at the bin's mean redshift, in the fiducial
    cosmology of msfm's config, rounded to 0.1 arcmin. l_max and white_noise_sigma follow from the rounded beam and the
    signal C_l of the nearest grid cosmology (step signal); the next cosmologies only report the spread.
    theta_fwhm_base: just above the smallest clustering beam / sqrt(2), so that every ratio of the per-probe smoothing
    stays below 2 under the ceil.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run, --check or when every value is up to date
    """
    scales_path = os.path.abspath(args.scales)
    signal_path = _out_path(SIGNAL_FILE)
    nz_paths = [
        filenames.get_filename_z_distribution(
            os.path.join(MSFM_DIR, conf["dirs"]["redshift_distributions"]),
            sample,
            files.get_z_bin_number(z_bin),
            prefix=conf["survey"]["name"].lower(),
        )
        for sample in PROBE_SAMPLES.values()
        for z_bin in conf["survey"][sample]["z_bins"]
    ]
    inputs = [os.path.abspath(args.msfm_config), signal_path, *nz_paths]
    if args.dry_run:
        return products.dry_run("values", inputs, [scales_path], rel=_rel)
    if not os.path.isfile(signal_path):
        raise FileNotFoundError(f"{_rel(signal_path)} does not exist, run the step signal first")

    n_side = conf["analysis"]["n_side"]
    setup = _fiducial_setup(conf)
    with h5py.File(signal_path, "r") as f:
        cls = f["cls"][:]
        made_with = {key: f[key][:] for key in ("tomo_Aia", "tomo_bg", "ng_bar")}
        z_bins = list(f.attrs["z_bins"])
        cosmo_keys = f["cosmo_key"][:].astype(str)
    assert z_bins == [z for sample in PROBE_SAMPLES.values() for z in conf["survey"][sample]["z_bins"]], z_bins
    for key, value in made_with.items():
        assert np.allclose(value, setup[key], rtol=1e-12), f"stale {_rel(signal_path)}: {key} {value} != {setup[key]}"

    fid = conf["analysis"]["fiducial"]
    cosmo = FlatLambdaCDM(Om0=fid["Om"], Ob0=fid["Ob"], H0=fid["H0"], Tcmb0=2.7255, Neff=3.046, m_nu=0.02 * u.eV)
    h = fid["H0"] / 100
    threshold = scales_conf["scale_cuts"]["threshold"]

    values, sigmas, i_bin = {}, [], 0
    for probe, sample in PROBE_SAMPLES.items():
        probe_conf = scales_conf["scale_cuts"][probe]
        tomo_z, tomo_nz = files.load_redshift_distributions(sample, conf)
        # the files are normalized differently per survey (Y3 sums to 1, Y6 integrates to 1), so divide by the sum
        z_bar = np.array([np.sum(z * nz) / np.sum(nz) for z, nz in zip(tomo_z, tomo_nz)])
        theta = (probe_conf["transverse_scale"] * u.Mpc / h) / cosmo.comoving_distance(z_bar)
        fwhm = np.round(scales.rad_to_arcmin(theta.value), 1)
        l_min = probe_conf["l_min"]
        assert len(l_min) == len(fwhm), f"scale_cuts.{probe}.l_min has {len(l_min)} entries for {len(fwhm)} bins"

        l_max, sigma = [], []
        for j, f_j in enumerate(fwhm):
            cut = [_scale_cut(cls[k, :, i_bin + j], f_j, l_min[j], threshold, n_side) for k in range(len(cls))]
            l_max.append(cut[0][0])
            sigma.append(float(f"{cut[0][1]:.5g}"))
            sigmas.append([c[1] for c in cut])
        i_bin += len(fwhm)
        LOGGER.info(f"{probe}: mean z {np.round(z_bar, 4).tolist()}, theta_fwhm {fwhm.tolist()}, l_max {l_max}")

        values[f"scale_cuts.{probe}.theta_fwhm"] = [float(x) for x in fwhm]
        values[f"scale_cuts.{probe}.white_noise_sigma"] = sigma
        values[f"scale_cuts.{probe}.l_max"] = l_max
        if probe == "clustering":
            values[f"scale_cuts.{probe}.theta_fwhm_base"] = float(np.floor(10 * np.min(fwhm) / np.sqrt(2)) / 10 + 0.1)

    # how much the noise level depends on the stand-in cosmology
    sigmas = np.array(sigmas)
    for k in range(1, sigmas.shape[1]):
        ratio = sigmas[:, k] / sigmas[:, 0]
        LOGGER.info(f"sigma of {cosmo_keys[k]} / {cosmo_keys[0]}: {np.round(ratio, 3).tolist()}")

    if not products.write_marked_values(scales_conf, scales_path, values, args.check, rel=_rel):
        return None
    return inputs, [scales_path]


STEPS = {"signal": step_signal, "values": step_values}


def _record_provenance(step, commits, inputs, outputs, scales_path):
    record = products.provenance_record(
        command="python -m deep_lss.apps.products.desy6 " + " ".join(sys.argv[1:]),
        config=_rel(scales_path),
        commits=commits,
        inputs=inputs,
        outputs=outputs,
        rel=_rel,
    )
    about = "Written by deep_lss.apps.products.desy6, one record per step: how each DES Y6 scale cut was made."
    products.record_provenance(_out_path(PROVENANCE_FILE), step, record, STEP_ORDER, about)
    LOGGER.info(f"recorded step {step} in {PROVENANCE_FILE}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=list(STEPS), help="the product to derive")
    parser.add_argument(
        "--msfm-config", required=True, help="msfm DES Y6 config, e.g. <msfm>/configs/desy6/v1/default.yaml"
    )
    parser.add_argument(
        "--scales", required=True, help="DES Y6 scales config, e.g. configs/scales/desy6/8wl,32gc.yaml"
    )
    parser.add_argument("--check", action="store_true", help="values only: write nothing, exit 1 if a value is stale")
    parser.add_argument("--dry-run", action="store_true", help="list the inputs and outputs, write nothing")
    parser.add_argument("--source", default=None, help="signal only: the projection root, which holds grid/")
    parser.add_argument("--n-nearest", type=int, default=5, help="signal only: grid cosmologies nearest the fiducial")
    parser.add_argument(
        "--max-perms", type=int, default=None, help="signal only: average the first N permutations, write nothing"
    )
    parser.add_argument(
        "--provenance",
        action="store_true",
        help="for production runs: refuse uncommitted code here or in msfm, record the step in data/desy6/",
    )
    args = parser.parse_args(argv)
    if args.check and args.step != "values":
        parser.error("--check only applies to the step values")
    if (args.source is not None) != (args.step == "signal"):
        parser.error("--source is required by the step signal and applies to no other step")
    if args.max_perms is not None and (args.step != "signal" or args.max_perms < 1):
        parser.error("--max-perms takes a positive number and applies to the step signal")
    if args.n_nearest < 1:
        parser.error("--n-nearest takes a positive number")

    conf = files.load_config(args.msfm_config)
    if conf["survey"]["name"] != SURVEY_NAME:
        raise ValueError(f"{args.msfm_config} is a {conf['survey']['name']} config, these products are {SURVEY_NAME}")
    scales_path = os.path.abspath(args.scales)
    if not (_rel(scales_path) + os.sep).startswith(SCALES_ROOT + os.sep):
        raise ValueError(f"the DES Y6 scales configs lie under {SCALES_ROOT}/, got {_rel(scales_path)}")
    scales_conf = input_output.read_yaml(scales_path)
    if scales_conf["scale_cuts"].get("survey") != SURVEY_NAME:
        raise ValueError(f"{_rel(scales_path)} is not marked scale_cuts.survey: {SURVEY_NAME}")

    # checked before the step runs, so a dry run with --provenance catches uncommitted code too
    commits = None
    if args.provenance:
        commits = {
            "deep_lss_commit": products.clean_commit(REPO_DIR, CODE_PATHS),
            "msfm_commit": products.clean_commit(MSFM_DIR, MSFM_CODE_PATHS),
        }
    result = STEPS[args.step](conf, scales_conf, args)
    if result is not None:
        for path in result[1]:
            LOGGER.info(f"wrote {path}")
        if args.provenance:
            _record_provenance(args.step, commits, *result, scales_path)


if __name__ == "__main__":
    main()
