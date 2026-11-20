"""Export one-point statistics of the clustering maps a trained network sees, for the paper_2
clustering pixel-PDF figure.

Held-out grid realizations go through the GridPipeline preprocessing (parse, mask, normalize,
nside downsampling) and the run's own smoothing front-end, built with
configuration.get_smoothing_kwargs(mode="eval") as in run_evaluation.py, white noise included.
Per realization and lens bin, the footprint pixels give the four moments and a histogram of the
standardized map. No maps are stored.

The msfm `main` checkout must come first on sys.path (--msfm_root), since a frozen v18 run dir
embeds the flat Y3 data paths of that branch.
"""

import argparse
import datetime
import glob
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

STORE = "/capstor/store/cscs/swissai/a0158/athomsen/deep_lss"
REPOS = "/users/athomsen/dlss/repos"

MOMENTS = ["mean", "var", "gamma1", "kurtosis"]
HIST_LO, HIST_HI, HIST_WIDTH = -6.0, 10.0, 0.1
# near-fiducial realization for the variance budget, distance in prior widths over these labels
BUDGET_PARAMS = ["Om", "s8", "w0", "bg1", "bg2", "bg3", "bg4"]


def setup():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--msfm_root", type=str, required=True, help="msfm main snapshot with a MSFM_COMMIT file")
    parser.add_argument("--run_dir", type=str, default=f"{STORE}/runs/v18/default/maps_gcnn/clustering/v1")
    parser.add_argument("--tfr_dir", type=str, default="/ritom/scratch/cscs/athomsen/v18/default/tfrecords/grid")
    parser.add_argument(
        "--out", type=str, default=f"{REPOS}/deep_lss_paper/paper_2/plotting/cache/clustering_pixel_pdf.h5"
    )
    parser.add_argument("--first_file", type=int, default=0)
    parser.add_argument("--n_files", type=int, default=None, help="default: all files from --first_file on")
    parser.add_argument("--seed", type=int, default=20261120, help="white noise, stateless per realization")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_parallel", type=int, default=16, help="files read concurrently")
    parser.add_argument("--n_poisson", type=int, default=10, help="Poisson draws for the variance budget")
    return parser.parse_args()


args = setup()

# the snapshot must shadow the editable msfm install (on the desy6 branch)
sys.path.insert(0, os.path.abspath(args.msfm_root))
import msfm  # noqa: E402

assert os.path.realpath(msfm.__file__).startswith(os.path.realpath(args.msfm_root)), msfm.__file__

import h5py  # noqa: E402
import tensorflow as tf  # noqa: E402

from msfm.grid_pipeline import GridPipeline  # noqa: E402
from msfm.utils import files, scales, tfrecords  # noqa: E402
from deepsphere import healpy_layers  # noqa: E402
from deep_lss.utils import configuration  # noqa: E402

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


def git_hash(repo_dir):
    """HEAD of a repo, with a -dirty suffix for tracked modifications."""
    try:
        head = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head + ("-dirty" if status else "")
    except (OSError, subprocess.CalledProcessError):
        # no git in the container: resolve HEAD by hand
        git_dir = os.path.join(repo_dir, ".git")
        with open(os.path.join(git_dir, "HEAD")) as f:
            ref = f.read().strip()
        if not ref.startswith("ref: "):
            return ref + "-dirty-unknown"
        ref = ref[5:]
        ref_file = os.path.join(git_dir, ref)
        if os.path.exists(ref_file):
            with open(ref_file) as f:
                return f.read().strip() + "-dirty-unknown"
        with open(os.path.join(git_dir, "packed-refs")) as f:
            for line in f:
                if line.strip().endswith(ref):
                    return line.split()[0] + "-dirty-unknown"
        return "unknown"


def read_records(path, wanted):
    """Yield (record index, serialized example) for the wanted record indices, seeking past the rest."""
    wanted = set(int(i) for i in wanted)
    last = max(wanted)
    with open(path, "rb") as f:
        i = 0
        while i <= last:
            header = f.read(12)
            if len(header) < 12:
                break
            (length,) = struct.unpack("<Q", header[:8])
            if i in wanted:
                data = f.read(length)
                f.seek(4, 1)
                yield i, data
            else:
                f.seek(length + 4, 1)
            i += 1


def build_dataset(grid_pipeline, file_list, signal_indices, parent_output_idx, batch_size, n_parallel):
    """The GridPipeline.get_dset chain for i_noise = 0, reading only the wanted records of each file."""
    gp = grid_pipeline

    def gen(path):
        yield from read_records(path.decode(), signal_indices)

    signature = (tf.TensorSpec((), tf.int64), tf.TensorSpec((), tf.string))
    dset = tf.data.Dataset.from_tensor_slices(file_list)
    dset = dset.interleave(
        lambda f: tf.data.Dataset.from_generator(gen, args=(f,), output_signature=signature),
        cycle_length=n_parallel,
        block_length=1,
        num_parallel_calls=n_parallel,
        deterministic=True,
    )

    def parse(i_record, serialized):
        dv = tfrecords.parse_inverse_grid(
            serialized,
            [0],
            n_pix=gp.n_dv_pix,
            n_z_metacal=gp.n_z_metacal,
            n_z_maglim=gp.n_z_maglim,
            n_z_cross=gp.n_z_cross,
            n_params=gp.n_all_params,
            n_noise=gp.n_noise_total,
            n_cls=gp.n_cls,
            with_lensing=gp.with_lensing,
            with_clustering=gp.with_clustering,
            with_cross=gp.with_cross,
            return_maps=gp.return_maps,
            return_cls=gp.return_cls,
        )
        dv["i_record"] = i_record
        return dv

    dset = dset.map(parse, num_parallel_calls=n_parallel, deterministic=True)
    dset = dset.flat_map(lambda dv: gp._split_noise_realizations(dv, [0]))
    dset = dset.batch(batch_size, drop_remainder=False)

    def augment(dv):
        i_record = dv.pop("i_record")
        maps, _, cosmo, index = gp._augmentations(dv)
        return maps, cosmo, index, i_record

    dset = dset.map(augment, num_parallel_calls=4, deterministic=True)

    # as in GridPipeline.get_dset
    parent_tf = tf.constant(parent_output_idx, dtype=tf.int32)
    n_pix_out = int(parent_output_idx.max()) + 1

    def downsample(maps, *rest):
        maps_t = tf.transpose(maps, perm=[1, 0, 2])
        maps_t = tf.math.unsorted_segment_mean(maps_t, parent_tf, n_pix_out)
        return (tf.transpose(maps_t, perm=[1, 0, 2]), *rest)

    dset = dset.map(downsample, num_parallel_calls=4, deterministic=True)
    return dset.prefetch(2)


def footprint_stats(y, fp_indices, n_hist):
    """Moments (B, n_bins, 4) and standardized-map histograms (B, n_bins, n_hist + 2) over footprint pixels.

    The histogram's first and last entries are the underflow and overflow counts.
    """
    moments, hists = [], []
    for i_bin, fp in enumerate(fp_indices):
        v = tf.cast(tf.gather(y[..., i_bin], fp, axis=1), tf.float64)
        mean = tf.reduce_mean(v, axis=1, keepdims=True)
        d = v - mean
        m2 = tf.reduce_mean(d**2, axis=1)
        m3 = tf.reduce_mean(d**3, axis=1)
        m4 = tf.reduce_mean(d**4, axis=1)
        moments.append(tf.stack([mean[:, 0], m2, m3 / m2**1.5, m4 / m2**2 - 3.0], axis=-1))

        z = d / tf.sqrt(m2)[:, None]
        idx = tf.cast(tf.floor((z - HIST_LO) / HIST_WIDTH), tf.int32) + 1
        idx = tf.clip_by_value(idx, 0, n_hist + 1)
        n_row = tf.shape(idx)[0]
        offsets = tf.range(n_row)[:, None] * (n_hist + 2)
        counts = tf.math.bincount(tf.reshape(idx + offsets, [-1]), minlength=n_row * (n_hist + 2))
        hists.append(tf.reshape(counts[: n_row * (n_hist + 2)], (n_row, n_hist + 2)))
    return tf.stack(moments, axis=1).numpy(), tf.stack(hists, axis=1).numpy()


def main():
    t_start = time.time()
    run_dir = args.run_dir
    conf = configuration.load_run_configs(os.path.join(run_dir, "configs.yaml"))
    net_conf, dlss_conf, data_conf, msfm_conf = conf["net"], conf["dlss"], conf["data"], conf["msfm"]
    loss_func = conf["run"]["loss_func"]
    assert dlss_conf["dset"]["common"]["with_clustering"] and not dlss_conf["dset"]["common"]["with_lensing"]

    # the kernel is loaded from the run dir; refuse to let a missing file make the layer write there
    smoothing_dir = os.path.join(run_dir, "smoothing")
    kernel_files = sorted(glob.glob(os.path.join(smoothing_dir, "*coo-*.npy")))
    assert len(kernel_files) == 2, kernel_files
    smoothing_kwargs = configuration.get_smoothing_kwargs(
        loss_func, msfm_conf, dlss_conf, net_conf, dir_base=run_dir, mode="eval"
    )
    assert "split_probes" not in smoothing_kwargs, "expected the uniform-nside single-kernel path"
    smooth_nside, smooth_indices, parent_output_idx = configuration.resolve_smooth_nside(
        net_conf, dlss_conf, msfm_conf
    )
    assert smooth_nside == smoothing_kwargs["nside"] and parent_output_idx is not None
    spmm_backend = net_conf["network"].get("spmm_backend", "csr")

    # the smoothing layer as the network builds it (resnet.py), white noise included
    front = healpy_layers.HealpySmoothing(**smoothing_kwargs, spmm_backend=spmm_backend)
    front_nn = healpy_layers.HealpySmoothing(
        **{**smoothing_kwargs, "white_noise_sigma": None}, spmm_backend=spmm_backend
    )
    sigma_label = front.file_label
    assert any(sigma_label in f for f in kernel_files), (sigma_label, kernel_files)

    mask = smoothing_kwargs["mask"].astype(np.float32)
    stddev = np.asarray(smoothing_kwargs["white_noise_sigma"], dtype=np.float32)
    n_bins = mask.shape[1]
    fp_indices = [np.flatnonzero(mask[:, i] == 1.0) for i in range(n_bins)]
    hist_edges = np.round(np.arange(HIST_LO, HIST_HI + HIST_WIDTH / 2, HIST_WIDTH), 10)
    n_hist = len(hist_edges) - 1
    reps = front.per_channel_repetitions
    fwhm_eff = front.fwhm_arcmin * np.sqrt(reps)
    print(f"smoothing: nside {smooth_nside}, base fwhm {front.fwhm_arcmin:.2f}', repetitions {reps}")
    print(f"effective fwhm {np.round(fwhm_eff, 2)}', white noise sigma {stddev}")
    print(f"footprint pixels (mask == 1) per bin: {[len(f) for f in fp_indices]}")
    print(
        f"fractional mask pixels per bin: {[int(np.sum((mask[:, i] > 0) & (mask[:, i] < 1))) for i in range(n_bins)]}"
    )

    # the pipeline as run_evaluation builds it, but returning every label
    common = {**dlss_conf["dset"]["common"], **dlss_conf["dset"]["eval"]["grid"]}
    pipe_kwargs = {k: v for k, v in common.items() if k not in {"with_cross_z", "with_cross_probe", "params"}}
    grid_pipeline = GridPipeline(conf=msfm_conf, params=None, **pipe_kwargs, return_maps=True, return_cls=False)
    all_params = list(grid_pipeline.all_params)
    signal_indices = grid_pipeline._parse_indices(
        data_conf["signal_indices"], "signal_indices", grid_pipeline.n_signal_total, is_eval=True
    )
    print(f"labels: {all_params}")
    print(f"held-out signal indices: {signal_indices}")

    file_list = sorted(glob.glob(os.path.join(args.tfr_dir, "DESy3_grid_dmb_????.tfrecord")))
    assert len(file_list) == msfm_conf["analysis"]["grid"]["n_cosmos"], len(file_list)
    stop = None if args.n_files is None else args.first_file + args.n_files
    file_list = file_list[args.first_file : stop]
    print(f"reading {len(file_list)} files from {args.tfr_dir}")

    checks = {}

    # check 1: this reader reproduces GridPipeline.get_dset on the first file
    ref_kwargs = {**net_conf["dset"]["eval"]["common"], **data_conf, **net_conf["dset"]["eval"]["grid"]}
    ref_kwargs.update(noise_indices=[0], local_batch_size=len(signal_indices), drop_remainder=False)
    ref_dset = grid_pipeline.get_dset(
        tfr_pattern=file_list[0],
        **ref_kwargs,
        downsample_nside=smooth_nside,
        parent_output_idx=parent_output_idx,
    )
    ref_maps, _, ref_cosmo, ref_index = next(iter(ref_dset))
    mine = build_dataset(grid_pipeline, file_list[:1], signal_indices, parent_output_idx, len(signal_indices), 1)
    my_maps, my_cosmo, my_index, my_record = next(iter(mine))
    order_ref = np.argsort(ref_index[1].numpy())
    order_mine = np.argsort(my_index[1].numpy())
    checks["pipeline_max_abs_diff"] = float(np.max(np.abs(ref_maps.numpy()[order_ref] - my_maps.numpy()[order_mine])))
    checks["pipeline_labels_equal"] = bool(np.array_equal(ref_cosmo.numpy()[order_ref], my_cosmo.numpy()[order_mine]))
    checks["record_index_is_i_signal"] = bool(np.array_equal(my_record.numpy(), my_index[1].numpy()))
    print(f"check (reader vs GridPipeline.get_dset): {checks}")

    # export; the layer's white noise is stateful, so one global seed and a fixed file order fix it
    tf.random.set_seed(args.seed)
    dset = build_dataset(grid_pipeline, file_list, signal_indices, parent_output_idx, args.batch_size, args.n_parallel)
    out_moments, out_moments_nn, out_hist, out_cosmo, out_index = [], [], [], [], []
    budget_cols = [all_params.index(p) for p in BUDGET_PARAMS]
    prior = msfm_conf["analysis"]["grid"]["priors"]
    fid = msfm_conf["analysis"]["fiducial"]
    fid_vec = np.array([fid[p] for p in BUDGET_PARAMS])
    width = np.array([prior[p][1] - prior[p][0] for p in BUDGET_PARAMS])
    best = {"dist": np.inf}
    n_done, t_loop = 0, time.time()
    for maps, cosmo, index, _ in dset:
        y = front(maps)
        y0 = front_nn(maps)
        mom, hist = footprint_stats(y, fp_indices, n_hist)
        mom_nn, _ = footprint_stats(y0, fp_indices, n_hist)
        out_moments.append(mom)
        out_moments_nn.append(mom_nn)
        out_hist.append(hist)
        out_cosmo.append(cosmo.numpy())
        out_index.append(np.stack([i.numpy() for i in index], axis=-1))

        dist = np.sqrt(np.sum(((cosmo.numpy()[:, budget_cols] - fid_vec) / width) ** 2, axis=1))
        j = int(np.argmin(dist))
        if dist[j] < best["dist"]:
            best = {"dist": float(dist[j]), "i_global": n_done + j}
        n_done += maps.shape[0]
        if (n_done // maps.shape[0]) % 50 == 0:
            rate = n_done / (time.time() - t_loop)
            print(f"{n_done} maps, {rate:.1f} maps/s", flush=True)

    moments = np.concatenate(out_moments)
    moments_nn = np.concatenate(out_moments_nn)
    hist = np.concatenate(out_hist)
    cosmo = np.concatenate(out_cosmo)
    index = np.concatenate(out_index)
    assert np.all(np.isin(index[:, 1], signal_indices)) and np.all(index[:, 2] == 0)
    assert hist.max() < np.iinfo(np.uint16).max
    print(f"exported {len(cosmo)} maps in {time.time() - t_loop:.0f} s")

    # variance budget on the near-fiducial realization: white noise + Poisson + signal
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)
    maglim_mask = files.get_tomo_dv_masks(msfm_conf)["maglim"]
    n_side = msfm_conf["analysis"]["n_side"]
    import healpy as hp

    n_bar = np.array(msfm_conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
    rng = np.random.default_rng(args.seed)
    var_poisson = []
    for _ in range(args.n_poisson):
        pn = (rng.poisson(np.broadcast_to(n_bar, maglim_mask.shape)) - n_bar).astype(np.float32) * maglim_mask
        pn, _ = scales.data_vector_to_smoothed_data_vector(
            pn,
            data_vec_pix=data_vec_pix,
            n_side=n_side,
            l_min=msfm_conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            l_max=msfm_conf["analysis"]["scale_cuts"]["clustering"]["l_max"],
            theta_fwhm=msfm_conf["analysis"]["scale_cuts"]["clustering"]["theta_fwhm"],
            arcmin=True,
            mask=maglim_mask,
            conf=msfm_conf,
            hard_cut=msfm_conf["analysis"]["scale_cuts"]["hard_cut"],
        )
        pn = grid_pipeline.normalize_clustering(tf.constant(pn[None], tf.float32)) * grid_pipeline.masks_maglim
        pn = tf.transpose(
            tf.math.unsorted_segment_mean(
                tf.transpose(pn, [1, 0, 2]), tf.constant(parent_output_idx, tf.int32), len(smooth_indices)
            ),
            [1, 0, 2],
        )
        mom_pn, _ = footprint_stats(front_nn(pn), fp_indices, n_hist)
        var_poisson.append(mom_pn[0, :, 1])
    var_poisson = np.array(var_poisson)
    i_best = best["i_global"]
    budget = {
        "total": moments[i_best, :, 1],
        "no_white_noise": moments_nn[i_best, :, 1],
        "white_noise": stddev.astype(np.float64) ** 2,
        "poisson": var_poisson.mean(axis=0),
        "poisson_std": var_poisson.std(axis=0, ddof=1),
    }
    budget["signal"] = budget["no_white_noise"] - budget["poisson"]
    budget["signal_from_total"] = budget["total"] - budget["white_noise"] - budget["poisson"]
    print(f"variance budget on realization {i_best} (labels {dict(zip(all_params, cosmo[i_best].round(4)))})")
    for k, v in budget.items():
        print(f"  {k:18s} {np.round(v, 5)}")

    # write
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with h5py.File(tmp, "w") as f:
        comp = {"compression": "gzip", "compression_opts": 4}
        f.create_dataset("moments/white_noise", data=moments, **comp)
        f.create_dataset("moments/no_white_noise", data=moments_nn, **comp)
        f["moments"].attrs["columns"] = MOMENTS
        f.create_dataset("hist/counts", data=hist[..., 1:-1].astype(np.uint16), **comp)
        f.create_dataset("hist/underflow", data=hist[..., 0].astype(np.uint32), **comp)
        f.create_dataset("hist/overflow", data=hist[..., -1].astype(np.uint32), **comp)
        f["hist/edges"] = hist_edges
        f.create_dataset("labels/params", data=cosmo.astype(np.float32), **comp)
        f["labels"].attrs["params"] = all_params
        for k, name in enumerate(["i_sobol", "i_signal", "i_noise"]):
            f[f"labels/{name}"] = index[:, k].astype(np.int64)
        f["footprint/n_pix"] = np.array([len(fp) for fp in fp_indices])
        for k, v in budget.items():
            f[f"variance_budget/{k}"] = np.asarray(v, dtype=np.float64)
        f["variance_budget"].attrs["i_realization"] = i_best
        f["variance_budget"].attrs["distance_prior_widths"] = best["dist"]
        f["variance_budget"].attrs["distance_params"] = BUDGET_PARAMS
        f["variance_budget"].attrs["n_poisson"] = args.n_poisson
        s = f.create_group("smoothing")
        s.attrs["nside"] = smooth_nside
        s.attrs["base_fwhm_arcmin"] = front.fwhm_arcmin
        s.attrs["per_channel_repetitions"] = reps
        s.attrs["effective_fwhm_arcmin"] = fwhm_eff
        s.attrs["nominal_fwhm_arcmin"] = dlss_conf["scale_cuts"]["clustering"]["theta_fwhm"]
        s.attrs["white_noise_sigma"] = stddev
        s.attrs["n_sigma_support"] = dlss_conf["scale_cuts"]["n_sigma_support"]
        f.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        f.attrs["run_dir"] = run_dir
        f.attrs["configs_yaml"] = open(os.path.join(run_dir, "configs.yaml")).read()
        f.attrs["tfr_dir"] = args.tfr_dir
        f.attrs["files"] = f"{args.first_file}:{args.first_file + len(file_list)}"
        f.attrs["signal_indices"] = signal_indices
        f.attrs["i_noise"] = 0
        f.attrs["n_maps"] = len(cosmo)
        f.attrs["seed"] = args.seed
        f.attrs["noise_seed_rule"] = "tf.random.set_seed(seed) once, the smoothing layer's own draws in file order"
        f.attrs["git_msfm"] = open(os.path.join(args.msfm_root, "MSFM_COMMIT")).read().strip()
        f.attrs["git_y3_deep_lss"] = git_hash(f"{REPOS}/y3-deep-lss")
        f.attrs["git_deepsphere"] = git_hash(f"{REPOS}/deepsphere-cosmo-tf2")
        f.attrs["checks"] = json.dumps(checks)
        f.attrs["script"] = os.path.abspath(__file__)
    os.replace(tmp, args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
