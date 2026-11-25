# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen (with Claude)

Derives the DES Y6 scale cuts of one msfm dataset version, one step per product (data/desy6/README.md). The msfm
config <msfm>/configs/desy6/<version>/<version>.yaml selects the version, whose scales configs lie in
configs/scales/desy6/<version>/ and whose products in data/desy6/<version>/. Every step writes only there and to the
config lines marked "# <- step values: <dotted.key>", and stops on a failed check. With --provenance, it requires
committed code in this repository and in msfm and records its inputs, outputs and both commits in
data/desy6/<version>/provenance.yaml.

The method is the one of the DES Y3 configs/scales/desy3/v18/8wl,32gc.yaml: a Gaussian beam per bin of a fixed
transverse comoving scale at the bin's mean redshift, l_max where the squared beam falls below the threshold, and white
noise equal to the smoothed signal power at l_max. The networks smooth each probe by repeating one base kernel per bin,
and this step chooses that kernel so that no bin is smoothed less than its beam (_base_kernel).

Steps, in order:
    signal   mean signal C_l (lensing kg + A_IA ia, clustering counts at the fiducial bias) of the fiducial maps, over
             all their permutations, from the benchmark fiducial_bench of the CosmoGrid projection (--source)
    values   theta_fwhm, l_max, white_noise_sigma and each probe's base kernel (theta_fwhm_base, base_repetitions),
             written into the marked scales config lines

Usage:
    OMP_NUM_THREADS=4 python -m deep_lss.apps.products.desy6 signal --msfm-config <msfm>/configs/desy6/v2/v2.yaml \
        --scales configs/scales/desy6/v2/8wl,32gc.yaml --max-perms 1   # smoke test
    OMP_NUM_THREADS=4 python -m deep_lss.apps.products.desy6 signal \
        --msfm-config <msfm>/configs/desy6/v2/v2.yaml --scales configs/scales/desy6/v2/8wl,32gc.yaml
        # the pool fills the CPUs of the process (48 in a daint login session)
    python -m deep_lss.apps.products.desy6 values --msfm-config <msfm>/configs/desy6/v2/v2.yaml \
        --scales configs/scales/desy6/v2/8wl,32gc.yaml
    python -m deep_lss.apps.products.desy6 values ... --check   # exit 1 if stale

signal reads the projection of the msfm config's locations.yaml (msfm.utils.locations) unless --source names another.

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
from sklearn.neighbors import BallTree

import msfm
from msfm.utils import clustering, files, filenames, locations, logger, products, redshift, scales
from msfm.utils import input_output

from deep_lss.utils.configuration import PROBE_SAMPLES, _downsample_mask, get_smooth_nside_indices

LOGGER = logger.get_logger(__file__)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MSFM_DIR = os.path.abspath(os.path.join(os.path.dirname(msfm.__file__), ".."))
SURVEY_NAME = "DESy6"
STEP_ORDER = ["signal", "values"]
# step signal: the benchmark at the fiducial cosmology, resolution and n(z), and its number of permutations
SIGNAL_DIR = os.path.join("benchmarks", "fiducial_bench")
SIGNAL_N_PERMS = 20
# the code a step runs, in this repository and in msfm
CODE_PATHS = ("deep_lss", "configs")
MSFM_CODE_PATHS = ("msfm", "configs")


def dataset_paths(msfm_config):
    """The products of one msfm dataset version, e.g. <msfm>/configs/desy6/v2/v2.yaml gives data/desy6/v2/ and
    configs/scales/desy6/v2/, so the scale cuts of one version are never derived from another's fiducials.

    Returns:
        dict: repository-relative out_root, scales_root, signal and provenance
    """
    dataset = os.path.dirname(os.path.relpath(os.path.abspath(msfm_config), os.path.join(MSFM_DIR, "configs")))
    survey, _, version = dataset.partition(os.sep)
    if survey != SURVEY_NAME.lower() or not version or os.sep in version:
        raise ValueError(f"{msfm_config} is not under <msfm>/configs/{SURVEY_NAME.lower()}/<version>/")
    out_root = os.path.join("data", dataset)
    return {
        "out_root": out_root,
        "scales_root": os.path.join("configs", "scales", dataset),
        "signal": os.path.join(out_root, "desy6_signal_cls.h5"),
        "provenance": os.path.join(out_root, "provenance.yaml"),
    }


def _out_path(rel_path, paths):
    """Absolute path of an output, which must lie under the dataset version's data/desy6/<version>/."""
    rel_path = os.path.normpath(rel_path)
    if os.path.isabs(rel_path) or not (rel_path + os.sep).startswith(paths["out_root"] + os.sep):
        raise ValueError(f"DES Y6 outputs must lie under {paths['out_root']}/, got {rel_path}")
    return os.path.join(REPO_DIR, rel_path)


def _rel(path):
    """Repository-relative path for files inside this repository, absolute otherwise."""
    return products.rel_path(path, REPO_DIR)


def _fiducial_setup(conf):
    """The fiducial amplitudes the signal maps are built with: per-bin A_IA, lens bias, lens counts per pixel and lens
    magnification coefficient C (empty without magnification)."""
    fid = conf["analysis"]["fiducial"]
    tomo_z, tomo_nz = files.load_redshift_distributions("metacal", conf)
    tomo_Aia = redshift.get_tomo_amplitudes(fid["Aia"], fid["n_Aia"], tomo_z, tomo_nz, conf["survey"]["metacal"]["z0"])
    tomo_bg = np.array([fid[p] for p in conf["analysis"]["params"]["bg"]["linear"]])
    n_side = conf["analysis"]["n_side"]
    ng_bar = np.array(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
    assert len(tomo_bg) == len(conf["survey"]["maglim"]["z_bins"]), "one linear bias per lens bin"
    magnification, mag_params = files.get_magnification(conf)
    tomo_cg = clustering.magnification_coefficient([fid[p] for p in mag_params]) if magnification else []
    return {
        "tomo_Aia": np.asarray(tomo_Aia, dtype=np.float64),
        "tomo_bg": tomo_bg,
        "ng_bar": ng_bar,
        "tomo_cg": np.asarray(tomo_cg, dtype=np.float64),
        # the projection's map type of the lens n(z) convergence
        "lens_kappa": files.get_lens_convergence_map_type(conf),
    }


def _perm_cls(task):
    """C_l of one permutation's signal maps, lensing bins then lens bins. A module function, for the pool."""
    path, metacal_bins, maglim_bins, n_side, setup, l_max, datapath = task
    # the convergence of the lens n(z), for the magnification of the lens counts
    magnified = setup["tomo_cg"].size > 0
    keys = [f"map/{m}/{b}" for m in ("kg", "ia") for b in metacal_bins] + [f"map/dg/{b}" for b in maglim_bins]
    keys += [f"map/{setup['lens_kappa']}/{b}" for b in maglim_bins] if magnified else []
    with h5py.File(path, "r") as f:
        for key in keys:
            if key not in f:
                raise KeyError(f"{path} has no {key}, so it is not a re-projected DES Y6 file")
            if f[key].shape != (hp.nside2npix(n_side),):
                raise ValueError(f"{path}: {key} has shape {f[key].shape}, not one nside-{n_side} map")
        kg = np.stack([f[f"map/kg/{b}"][:] for b in metacal_bins], axis=-1)
        ia = np.stack([f[f"map/ia/{b}"][:] for b in metacal_bins], axis=-1)
        dg = np.stack([f[f"map/dg/{b}"][:] for b in maglim_bins], axis=-1)
        kl = np.stack([f[f"map/{setup['lens_kappa']}/{b}"][:] for b in maglim_bins], axis=-1) if magnified else None

    wl = kg + setup["tomo_Aia"] * ia
    mean_dg = np.mean(dg, axis=0)
    gc = clustering.galaxy_density_to_count(
        setup["ng_bar"],
        (dg - mean_dg) / mean_dg,
        setup["tomo_bg"],
        mg=kl - np.mean(kl, axis=0) if magnified else None,
        cg=setup["tomo_cg"] if magnified else None,
    )
    maps = np.concatenate([wl, gc], axis=-1)
    return np.stack(
        [
            hp.anafast(maps[:, i], lmax=l_max, pol=False, datapath=datapath, use_pixel_weights=True)
            for i in range(maps.shape[1])
        ],
        axis=-1,
    )


def step_signal(conf, scales_conf, args):
    """Writes the mean signal C_l of the fiducial maps to data/desy6/<version>/.

    As configs/scales/desy3/v18/8wl,32gc.yaml did for Y3 on the fiducial maps (msfm notebooks/desy3/scale_cuts.ipynb):
    lensing is kg + A_IA ia with the fiducial IA amplitude per bin, clustering the galaxy counts of the fiducial linear
    bias and n_gal, magnified at the fiducial alpha if the config has lens magnification. The maps are the
    SIGNAL_N_PERMS permutations of SIGNAL_DIR, which has the fiducial cosmology, resolution and n(z); Y6 has no
    fiducial projection.

    Returns:
        tuple: (input paths, output paths), or None with --dry-run or --max-perms
    """
    signal_dir = os.path.join(os.path.abspath(args.source), SIGNAL_DIR)
    out_path = _out_path(args.paths["signal"], args.paths)
    inputs = [os.path.abspath(args.msfm_config), signal_dir]
    if args.dry_run:
        return products.dry_run("signal", inputs, [out_path], rel=_rel)

    n_side = conf["analysis"]["n_side"]
    l_max = 3 * n_side - 1
    metacal_bins = conf["survey"]["metacal"]["z_bins"]
    maglim_bins = conf["survey"]["maglim"]["z_bins"]
    setup = _fiducial_setup(conf)
    datapath = os.path.join(MSFM_DIR, conf["files"]["healpy_data"])
    with_bary = conf["analysis"]["modelling"]["baryonified"]

    n_perms = SIGNAL_N_PERMS if args.max_perms is None else min(SIGNAL_N_PERMS, args.max_perms)
    tasks = []
    for perm in range(n_perms):
        path = filenames.get_filename_full_maps(os.path.join(signal_dir, f"perm_{perm:04d}"), with_bary)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} does not exist")
        tasks.append((path, metacal_bins, maglim_bins, n_side, setup, l_max, datapath))

    # healpy threads per task come from OMP_NUM_THREADS, so the pool fills the cores without oversubscribing them
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    n_procs = max(1, min(len(tasks), len(os.sched_getaffinity(0)) // n_threads))
    LOGGER.info(f"anafast of {len(tasks)} maps files in {n_procs} processes x {n_threads} threads")
    with multiprocessing.Pool(n_procs) as pool:
        mean_cls = np.mean(np.stack(pool.map(_perm_cls, tasks)), axis=0)
    assert np.all(np.isfinite(mean_cls)) and np.all(mean_cls[2:] > 0), "non-finite or non-positive signal C_l"

    if args.max_perms is not None:
        LOGGER.info(f"--max-perms {args.max_perms}: the C_l passed every check, nothing is written")
        return None

    with h5py.File(out_path, "w") as f:
        f.create_dataset("cls", data=mean_cls)
        f.create_dataset("tomo_Aia", data=setup["tomo_Aia"])
        f.create_dataset("tomo_bg", data=setup["tomo_bg"])
        f.create_dataset("ng_bar", data=setup["ng_bar"])
        # only with magnification, so that a signal without it is written as before
        if setup["tomo_cg"].size:
            f.create_dataset("tomo_cg", data=setup["tomo_cg"])
        f.attrs["maps"] = SIGNAL_DIR
        f.attrs["z_bins"] = metacal_bins + maglim_bins
        f.attrs["n_perms"] = n_perms
        f.attrs["n_side"] = n_side
        f.attrs["about"] = (
            "cls: mean signal C_l (ell = 0..3 n_side - 1, axis 0; bins axis 1, lensing then clustering) over the "
            "permutations of the fiducial maps"
        )
    return inputs, [out_path]


def _scale_cut(cls, fwhm, l_min, threshold, n_side):
    """l_max where the squared beam falls below threshold, and the white noise of the smoothed signal power there.

    Follows msfm dev/scripts/noise_models/derive_scale_cut_noise.py, the port of msfm notebooks/desy3/scale_cuts.ipynb.
    A beam too narrow to reach the threshold below the band limit 3 n_side - 1 gets l_max there instead.

    Returns:
        tuple: (l_max, sigma, squared beam at l_max), the last above threshold only at the band limit
    """
    ell = np.arange(len(cls))
    low_pass = scales.gaussian_low_pass_factor_alm(ell, theta_fwhm=fwhm, arcmin=True) ** 2
    high_pass = scales.gaussian_high_pass_factor_alm(ell, l_min=l_min) ** 2
    below = ell[low_pass < threshold]
    l_max = int(below[0]) if len(below) else int(ell[-1])
    cl_noise = high_pass[l_max] * low_pass[l_max] * cls[l_max]
    return l_max, float(np.sqrt(cl_noise / hp.nside2pixarea(n_side))), float(low_pass[l_max])


def _kernel_footprint(conf, sample, nside):
    """The pixels a probe's smoothing kernel is built on at nside, as in configuration.get_smoothing_kwargs.

    Returns:
        tuple: (lat, lon) in radians of the data vector's pixels at nside, and whether the sample's mask covers each
    """
    n_side = conf["analysis"]["n_side"]
    data_vec_pix, _, _, _ = files.load_pixel_file(conf)
    mask = files.get_tomo_dv_masks(conf)[sample]
    if nside == n_side:
        indices = data_vec_pix
    elif nside < n_side:
        indices, parent = get_smooth_nside_indices(data_vec_pix, n_side, nside)
        mask = _downsample_mask(mask, parent, len(indices))
    else:
        raise ValueError(f"base_nside {nside} exceeds the maps' n_side {n_side}")
    lon, lat = hp.pix2ang(nside, indices, nest=True, lonlat=True)
    return np.radians(np.stack([lat, lon], axis=1)), np.any(mask > 0, axis=1)


def _realized_fwhm(theta, covered, max_base, n_sigma_support):
    """base -> FWHM (arcmin) that one application of deepsphere's HealpySmoothing kernel of FWHM base realizes.

    The kernel as HealpySmoothing builds it: each pixel takes its k nearest pixels, k being the most any pixel has
    within n_sigma_support sigma, with Gaussian weights; each column is divided by its pixel's row sum. On a sphere
    sampled this coarsely it is narrower than its nominal Gaussian. The realized FWHM is that of the Gaussian with the
    kernel's second moment, averaged over the pixels the mask covers; repetitions add second moments.
    """
    to_sigma = np.radians(1 / 60) / np.sqrt(8 * np.log(2))
    tree = BallTree(theta, metric="haversine")
    k_max = int(np.max(tree.query_radius(theta, r=max_base * to_sigma * n_sigma_support, count_only=True)))
    dist, ind = tree.query(theta, k=k_max, sort_results=True)

    def realized(base):
        sigma = base * to_sigma
        k = int(np.max(np.sum(dist <= sigma * n_sigma_support, axis=1)))
        weight = np.exp(-0.5 * (dist[:, :k] / sigma) ** 2)
        weight /= np.sum(weight, axis=1)[ind[:, :k]]
        second = np.sum(weight * dist[:, :k] ** 2, axis=1) / np.sum(weight, axis=1)
        return float(np.sqrt(np.mean(second[covered]) / 2) / to_sigma)

    return realized


def _base_kernel(fwhm, realized):
    """The base kernel of one probe: a base FWHM, repeated per bin ceil((fwhm / realized(base))^2) times, so that no
    bin is smoothed less than its beam.

    Among bases <= min(fwhm) on a 0.1 arcmin grid whose repetitions total no more than the DES Y3 rule's (a nominal base
    just above min(fwhm) / sqrt(2)), the one with the smallest worst overshoot; a tie goes to the larger base.

    Returns:
        tuple: (base, realized FWHM, repetitions, effective FWHM)
    """
    fwhm = np.asarray(fwhm, dtype=np.float64)
    budget = np.sum(np.ceil((fwhm / (np.floor(10 * np.min(fwhm) / np.sqrt(2)) / 10 + 0.1)) ** 2))
    best = None
    for base in np.arange(int(round(10 * np.min(fwhm))), 0, -1) / 10:
        # a kernel realizes less than its nominal FWHM (asserted below), so once the nominal repetitions exceed the
        # budget, the realized ones of this and every smaller base do too
        if np.sum(np.ceil((fwhm / base) ** 2)) > budget:
            break
        width = realized(base)
        assert width <= base, f"the {base} arcmin kernel realizes {width:.3f} arcmin, more than nominal"
        reps = np.ceil((fwhm / width) ** 2).astype(int)
        overshoot = np.max(width * np.sqrt(reps) / fwhm)
        if np.sum(reps) <= budget and (best is None or overshoot < best[0]):
            best = (overshoot, float(base), width, reps)
    if best is None:
        raise ValueError(f"no base kernel for {fwhm.tolist()} within {budget:.0f} repetitions")
    _, base, width, reps = best
    return base, width, reps, width * np.sqrt(reps)


def step_values(conf, scales_conf, args):
    """Writes the derived scale cuts into the scales config lines marked "# <- step values: <dotted.key>".

    theta_fwhm: the beam of scale_cuts.<probe>.transverse_scale (Mpc/h) at the bin's mean redshift, in the fiducial
    cosmology of msfm's config, rounded to 0.1 arcmin. l_max and white_noise_sigma follow from the rounded beam and the
    fiducial signal C_l (step signal). A bin whose beam does not reach the threshold below the band limit keeps its
    beam and gets l_max there, with a warning.
    theta_fwhm_base and base_repetitions: each probe's base kernel at scale_cuts.<probe>.base_nside (_base_kernel).

    Returns:
        tuple: (input paths, output paths), or None with --dry-run, --check, or when every value is up to date and
            --provenance is not set
    """
    scales_path = os.path.abspath(args.scales)
    signal_path = _out_path(args.paths["signal"], args.paths)
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
    pixel_path = os.path.join(MSFM_DIR, conf["files"]["pixels"])
    inputs = [os.path.abspath(args.msfm_config), signal_path, pixel_path, *nz_paths]
    if args.dry_run:
        return products.dry_run("values", inputs, [scales_path], rel=_rel)
    if not os.path.isfile(signal_path):
        raise FileNotFoundError(f"{_rel(signal_path)} does not exist, run the step signal first")

    n_side = conf["analysis"]["n_side"]
    setup = _fiducial_setup(conf)
    with h5py.File(signal_path, "r") as f:
        cls = f["cls"][:]
        made_with = {key: f[key][:] for key in ("tomo_Aia", "tomo_bg", "ng_bar")}
        # a signal without magnification has no tomo_cg
        made_with["tomo_cg"] = f["tomo_cg"][:] if "tomo_cg" in f else np.zeros(0)
        z_bins = list(f.attrs["z_bins"])
    assert z_bins == [z for sample in PROBE_SAMPLES.values() for z in conf["survey"][sample]["z_bins"]], z_bins
    assert cls.shape == (3 * n_side, len(z_bins)), f"{_rel(signal_path)}: C_l of shape {cls.shape}, rerun step signal"
    for key, value in made_with.items():
        assert value.shape == setup[key].shape and np.allclose(
            value, setup[key], rtol=1e-12
        ), f"stale {_rel(signal_path)}: {key} {value} != {setup[key]}"

    fid = conf["analysis"]["fiducial"]
    cosmo = FlatLambdaCDM(Om0=fid["Om"], Ob0=fid["Ob"], H0=fid["H0"], Tcmb0=2.7255, Neff=3.046, m_nu=0.02 * u.eV)
    h = fid["H0"] / 100
    threshold = scales_conf["scale_cuts"]["threshold"]
    n_sigma_support = scales_conf["scale_cuts"]["n_sigma_support"]

    values, i_bin = {}, 0
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
            l_max_j, sigma_j, beam_j = _scale_cut(cls[:, i_bin + j], f_j, l_min[j], threshold, n_side)
            l_max.append(l_max_j)
            sigma.append(float(f"{sigma_j:.5g}"))
            if beam_j >= threshold:
                LOGGER.warning(
                    f"{conf['survey'][sample]['z_bins'][j]}: the squared {f_j} arcmin beam is still {beam_j:.4f} at "
                    f"the band limit, so l_max is the band limit {l_max_j}, not a {threshold} crossing"
                )
        i_bin += len(fwhm)
        LOGGER.info(f"{probe}: mean z {np.round(z_bar, 4).tolist()}, theta_fwhm {fwhm.tolist()}, l_max {l_max}")

        values[f"scale_cuts.{probe}.theta_fwhm"] = [float(x) for x in fwhm]
        values[f"scale_cuts.{probe}.white_noise_sigma"] = sigma
        values[f"scale_cuts.{probe}.l_max"] = l_max

        if "base_nside" not in probe_conf:
            raise ValueError(
                f"scale_cuts.{probe} has no base_nside, so it predates the stated base kernel (DES Y6 v1); its "
                "values come from the commit its provenance.yaml records"
            )
        base_nside = probe_conf["base_nside"]
        theta_pix, covered = _kernel_footprint(conf, sample, base_nside)
        realized = _realized_fwhm(theta_pix, covered, float(np.min(fwhm)), n_sigma_support)
        base, width, reps, effective = _base_kernel(fwhm, realized)
        values[f"scale_cuts.{probe}.theta_fwhm_base"] = base
        values[f"scale_cuts.{probe}.base_repetitions"] = [int(r) for r in reps]
        LOGGER.info(
            f"{probe}: base kernel {base} arcmin at nside {base_nside} realizes {width:.3f}, repetitions "
            f"{reps.tolist()}, effective FWHM {np.round(effective, 2).tolist()} "
            f"(+{100 * np.min(effective / fwhm - 1):.1f}% to +{100 * np.max(effective / fwhm - 1):.1f}%)"
        )

    written = products.write_marked_values(scales_conf, scales_path, values, args.check, rel=_rel)
    # with --provenance an up-to-date config is recorded too: these inputs at this commit give its values
    if args.check or not (written or args.provenance):
        return None
    return inputs, [scales_path]


STEPS = {"signal": step_signal, "values": step_values}


def _record_provenance(step, commits, inputs, outputs, scales_path, paths):
    record = products.provenance_record(
        command="python -m deep_lss.apps.products.desy6 " + " ".join(sys.argv[1:]),
        config=_rel(scales_path),
        commits=commits,
        inputs=inputs,
        outputs=outputs,
        rel=_rel,
    )
    about = "Written by deep_lss.apps.products.desy6, one record per step: how each DES Y6 scale cut was made."
    products.record_provenance(_out_path(paths["provenance"], paths), step, record, STEP_ORDER, about)
    LOGGER.info(f"recorded step {step} in {paths['provenance']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=list(STEPS), help="the product to derive")
    parser.add_argument(
        "--msfm-config", required=True, help="msfm DES Y6 config, e.g. <msfm>/configs/desy6/v2/v2.yaml"
    )
    parser.add_argument(
        "--scales", required=True, help="scales config of that version, e.g. configs/scales/desy6/v2/8wl,32gc.yaml"
    )
    parser.add_argument("--check", action="store_true", help="values only: write nothing, exit 1 if a value is stale")
    parser.add_argument("--dry-run", action="store_true", help="list the inputs and outputs, write nothing")
    parser.add_argument(
        "--source",
        default=None,
        help="signal only: the projection root, which holds benchmarks/ (default: the projection of the msfm "
        "config's locations.yaml)",
    )
    parser.add_argument(
        "--max-perms", type=int, default=None, help="signal only: average the first N permutations, write nothing"
    )
    parser.add_argument(
        "--provenance",
        action="store_true",
        help="for production runs: refuse uncommitted code here or in msfm, record the step in data/desy6/<version>/",
    )
    args = parser.parse_args(argv)
    if args.check and args.step != "values":
        parser.error("--check only applies to the step values")
    if args.source is not None and args.step != "signal":
        parser.error("--source applies to the step signal only")
    if args.source is None and args.step == "signal":
        args.source = locations.get(args.msfm_config, "projection")
        LOGGER.info(f"--source {args.source}, the projection of the msfm config's locations.yaml")
    if args.max_perms is not None and (args.step != "signal" or args.max_perms < 1):
        parser.error("--max-perms takes a positive number and applies to the step signal")

    conf = files.load_config(args.msfm_config)
    if conf["survey"]["name"] != SURVEY_NAME:
        raise ValueError(f"{args.msfm_config} is a {conf['survey']['name']} config, these products are {SURVEY_NAME}")
    args.paths = dataset_paths(args.msfm_config)
    scales_path = os.path.abspath(args.scales)
    if os.path.dirname(_rel(scales_path)) != args.paths["scales_root"]:
        raise ValueError(
            f"the scales configs of this dataset lie in {args.paths['scales_root']}/, got {_rel(scales_path)}"
        )
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
            _record_provenance(args.step, commits, *result, scales_path, args.paths)


if __name__ == "__main__":
    main()
