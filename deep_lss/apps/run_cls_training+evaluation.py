import argparse, h5py, os, random, yaml
from pathlib import Path

import numpy as np
import tensorflow as tf

for gpu in tf.config.list_physical_devices(device_type="GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from msfm.utils import files, logger

LOGGER = logger.get_logger(__file__)

from deep_lss.models.grid_model import GridLossModel
from deep_lss.nets import CLS_NETWORKS, MultiLayerPerceptron
from deep_lss.nets.layers.cls.whitening import AsinhScaleLayer, PCAWhiteningLayer
from deep_lss.utils import cls_evaluation, configuration, evaluation, training_helpers

from msi.utils import dataset


class EarlyStopper:
    def __init__(self, patience, min_delta=0.0, min_steps=0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.best_loss = float("inf")
        self.wait = 0

    def update(self, loss):
        """Returns True if this is a new best (caller should save checkpoint)."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
            return True
        self.wait += 1
        return False

    def should_stop(self, step):
        return step >= self.min_steps and self.wait >= self.patience


class ReduceLROnPlateau:
    def __init__(self, factor, patience, min_delta=0.0, cooldown=0, min_lr=0.0):
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.best_loss = float("inf")
        self.wait = 0
        self.cooldown_counter = 0

    def update(self, loss, current_lr):
        """Return the (possibly reduced) LR given the latest vali loss."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
            return current_lr
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
            return current_lr
        self.wait += 1
        if self.wait >= self.patience:
            new_lr = max(current_lr * self.factor, self.min_lr)
            self.wait = 0
            self.cooldown_counter = self.cooldown
            return new_lr
        return current_lr


def setup():
    parser = argparse.ArgumentParser(
        description="Train an MLP summary network on binned power spectra (Cls) using the mutual information loss."
    )
    parser.add_argument("--msfm_config", required=True)
    parser.add_argument("--probes_config", default=None)
    parser.add_argument("--scales_config", default=None)
    parser.add_argument("--loss_config", required=True)
    # --mlp_config is a deprecated alias kept so existing dev submission scripts keep working.
    parser.add_argument("--net_config", "--mlp_config", dest="net_config", required=True)
    parser.add_argument("--data_config", required=True, help="train/test split config (configs/data/)")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--restore_checkpoint", action="store_true")
    parser.add_argument(
        "--precache_only",
        action="store_true",
        help="Build the hard_rebinned Cls cache then exit (no training). Requires scale_cut=hard_rebinned.",
    )

    # Observation inclusion flags (all default off)
    parser.add_argument("--include_grid", action="store_true")
    parser.add_argument("--n_grid_examples", type=int, default=16)
    parser.add_argument("--include_des", action="store_true")
    parser.add_argument("--include_mocks", action="store_true", help="evaluate mock observations from data_dir/obs/")
    parser.add_argument(
        "--mock_labels",
        nargs="+",
        default=None,
        help="mock labels to evaluate; if omitted, every *_obs_maps.h5 in data_dir/obs/ is evaluated",
    )

    args = parser.parse_args()
    if not args.probes_config:
        parser.error("--probes_config is required")
    return args


def main():
    args = setup()
    msfm_conf = files.load_config(args.msfm_config)
    from msfm.utils import input_output

    dlss_conf = configuration.read_split_configs(args.probes_config, args.scales_config)
    net_conf = input_output.read_yaml(args.net_config)
    cls_n_bins = net_conf.get("cls_n_bins", 16)

    seed = net_conf.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if net_conf.get("deterministic_ops", False):
        tf.config.experimental.enable_op_determinism()
        LOGGER.info("Enabled tf.config.experimental.enable_op_determinism()")
    LOGGER.info(f"seed           = {seed}")

    ema_momentum = net_conf.get("ema_momentum", None)
    LOGGER.info(f"ema_momentum   = {ema_momentum}")

    loss_conf = input_output.read_yaml(args.loss_config)
    data_conf = input_output.read_yaml(args.data_config)
    loss_function = loss_conf.get("loss_function", "mutual_info")
    mi = loss_conf.get("mutual_info_loss", {})

    common = dlss_conf["dset"]["common"]
    with_lensing = common["with_lensing"]
    with_clustering = common["with_clustering"]
    with_cross_z = common["with_cross_z"]
    with_cross_probe = common["with_cross_probe"]
    lenses_before_sources = common.get("lenses_before_sources", common.get("ggl_only", False))

    if with_lensing and not with_clustering:
        probe = "lensing"
    elif with_clustering and not with_lensing:
        probe = "clustering"
    elif with_lensing and with_clustering:
        probe = "combined"
    else:
        probe = "cross"

    params = dlss_conf["dset"]["training"]["params"]
    n_params = len(params)
    # the MSE loss regresses one physical value per parameter; the mutual-info summary is wider
    n_summary = n_params if loss_function == "mse" else mi.get("dim_summary_fac", 2) * n_params

    scale_cut = dlss_conf.get("scale_cut") or net_conf.get("scale_cut", "hard_rebinned")

    scales_name = Path(args.scales_config).stem if args.scales_config else None

    n_steps = net_conf["n_steps"]
    batch_size = net_conf["batch_size"]
    log_every = net_conf["log_every"]
    vali_every = net_conf["vali_every"]
    signal_indices = data_conf["signal_indices"]
    noise_indices = data_conf["noise_indices"]
    # the MSE loss has no z-feature regularization; mutual-info reads it from the loss config's regu block
    regu = {} if loss_function == "mse" else mi.get("regu", {})
    z_weight = regu.get("z_weight", None)
    z_type = regu.get("z_type", None)
    z_layer = regu.get("z_layer", "last")

    # The VICReg invariance term needs per-sample (i_sobol, i_signal) ids in the train dataset.
    # Derive the flag up front (same conditions as GridLossModel.setup_grid_loss_step) because the
    # dataset is built before the model; we assert it matches the model attribute after setup.
    uses_invariance = z_type == "vicreg" and isinstance(z_weight, dict) and z_weight.get("invariance") is not None

    pred_dir = os.path.join(args.out_dir, args.model_name)
    os.makedirs(pred_dir, exist_ok=True)

    pred_file = os.path.join(pred_dir, f"preds_{n_steps}.h5")

    LOGGER.info(f"probe          = {probe}")
    LOGGER.info(f"pred_dir       = {pred_dir}")
    LOGGER.info(f"pred_file      = {pred_file}")
    LOGGER.info(f"params         = {params}")
    LOGGER.info(f"n_steps        = {n_steps}")
    LOGGER.info(f"signal_indices = {signal_indices}")
    LOGGER.info(f"noise_indices  = {noise_indices}")
    LOGGER.info(f"z_type         = {z_type}")
    LOGGER.info(f"z_weight       = {z_weight}")
    LOGGER.info(f"z_layer        = {z_layer}")

    # provenance only: the cls app always re-reads from CLI flags, never reloads this file
    with open(os.path.join(pred_dir, "configs.yaml"), "w") as f:
        yaml.dump(
            {
                "mlp": net_conf,
                "dlss": dlss_conf,
                "loss": loss_conf,
                "data": data_conf,
                "msfm": msfm_conf,
                "run": {"model_name": args.model_name, "scale_cut": scale_cut},
            },
            f,
        )

    # cls_transform selects how the binned Cls are transformed before the network:
    #   "asinh_per_feature":     per-feature asinh(x/s), applied INSIDE the model via an
    #                            AsinhScaleLayer whose scale is fit from the training Cls and
    #                            stored in the checkpoint. The preprocessing feeds raw Cls.
    #   "log1p_fixed":           external sign(x)*log1p(|x|/1e-10), applied in preprocessing.
    #   "none":                  no transform (raw Cls fed to the network).
    # asinh is only wired for hard_rebinned (it fits its scale from the raw Cls, which only that
    # branch provides), so default to it there and to the signed-log otherwise.
    default_transform = "asinh_per_feature" if scale_cut == "hard_rebinned" else "log1p_fixed"
    cls_transform = net_conf.get("cls_transform", default_transform)
    if cls_transform not in ("log1p_fixed", "none", "asinh_per_feature"):
        raise ValueError(f"Unknown cls_transform={cls_transform!r}")
    use_asinh = cls_transform == "asinh_per_feature"
    # Whether the external (preprocessing) signed-log is applied. With asinh the model owns the
    # transform, so the preprocessing must feed raw Cls everywhere (dataset + static eval + obs).
    apply_log = cls_transform == "log1p_fixed"
    ell_weighting = net_conf.get("ell_weighting", None)
    if use_asinh and ell_weighting is not None:
        raise NotImplementedError(
            "cls_transform=asinh_per_feature with ell_weighting is not supported: the per-feature "
            "scale would need to be computed on the ell-weighted Cls. Set ell_weighting: null."
        )

    if scale_cut == "hard_rebinned":
        if scales_name is None:
            raise ValueError("--scales_config is required when scale_cut=hard_rebinned")
        from deep_lss.utils import cls_preprocessing

        if args.precache_only:
            LOGGER.warning(
                f"--precache_only: building hard_rebinned cache for scales_name={scales_name}, then exiting."
            )
            cls_preprocessing.build_rebinned_cls_cache(
                data_dir=args.data_dir,
                msfm_conf=msfm_conf,
                dlss_conf=dlss_conf,
                cls_n_bins=cls_n_bins,
                scales_name=scales_name,
            )
            LOGGER.warning("Cache built. Exiting.")
            return
        cl_dset_train, cl_dset_test, out_dict = cls_preprocessing.get_rebinned_cls_dsets(
            data_dir=args.data_dir,
            msfm_conf=msfm_conf,
            dlss_conf=dlss_conf,
            params=params,
            cls_n_bins=cls_n_bins,
            scales_name=scales_name,
            signal_indices=signal_indices,
            noise_indices=noise_indices,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=with_cross_z,
            with_cross_probe=with_cross_probe,
            lenses_before_sources=lenses_before_sources,
            batch_size=batch_size,
            seed=seed,
            return_pair_ids=uses_invariance,
            apply_log=apply_log,
        )
    else:
        if uses_invariance:
            raise NotImplementedError(
                f"VICReg invariance (z_weight['invariance']) is only wired up for scale_cut=hard_rebinned; "
                f"got scale_cut={scale_cut!r}."
            )
        if use_asinh:
            raise NotImplementedError(
                f"cls_transform=asinh_per_feature is only wired for scale_cut=hard_rebinned (it fits its "
                f"per-feature scale from the raw Cls); got scale_cut={scale_cut!r}."
            )
        cl_dset_train, cl_dset_test, out_dict = dataset.get_binned_power_spectra_dset_for_scale_cut(
            scale_cut,
            base_dir=args.data_dir,
            msfm_conf=msfm_conf,
            dlss_conf=dlss_conf,
            params=params,
            signal_indices=signal_indices,
            noise_indices=noise_indices,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=with_cross_z,
            with_cross_probe=with_cross_probe,
            lenses_before_sources=lenses_before_sources,
            batch_size=batch_size,
            apply_log=apply_log,
            standardize=False,
            ell_weighting=ell_weighting,
        )

    n_cls = out_dict["grid/cls/train"].shape[-1]

    # per-parameter label std for the MSE loss (parameters live on different scales); computed from the
    # physical training labels (grid/cosmos is never standardized in the pipeline).
    label_std = None
    if loss_function == "mse":
        mse_conf = loss_conf.get("mse_loss", {})
        if mse_conf.get("standardize_labels", True):
            _cosmos_train = np.asarray(out_dict["grid/cosmos/train"], dtype=np.float32)
            label_std = _cosmos_train.std(axis=0)
            LOGGER.info(f"MSE label_std = {label_std}")

    # Per-feature asinh transform applied inside the model. Its scale is fit on the raw training
    # Cls (median|x| per feature) and stored as a checkpoint weight, so the same transform is
    # reused at evaluation / inference time. Applied before whitening (see MultiLayerPerceptron).
    if use_asinh:
        input_transform = AsinhScaleLayer()
        input_transform.fit(out_dict["grid/cls_raw/train"])
    else:
        input_transform = None

    n_pca = net_conf.get("pca_components", None)
    if n_pca is not None:
        pca_whiten = net_conf.get("pca_whiten", True)
        whitening_layer = PCAWhiteningLayer(n_components=n_pca, whiten=pca_whiten)
        if apply_log:
            pca_fit_data = out_dict["grid/cls/train"]
        else:
            pca_fit_data = out_dict["grid/cls_raw/train"].copy()
            if ell_weighting is not None and out_dict.get("ell_weights") is not None:
                pca_fit_data = pca_fit_data * out_dict["ell_weights"].astype(pca_fit_data.dtype)
        if use_asinh:
            # Whiten the asinh-transformed features (matches the runtime transform -> whiten order).
            pca_fit_data = input_transform(tf.constant(pca_fit_data, dtype=tf.float32)).numpy()
        whitening_layer.fit(pca_fit_data)
    else:
        whitening_layer = None

    # ---- network architecture switch ----
    # `network: {name, kwargs}` mirrors the map-level selection (see deep_lss/nets/__init__.py
    # CLS_NETWORKS). An absent block => "mlp" reading the legacy top-level keys, so existing
    # configs/mlp/*.yaml keep working unchanged.
    network_block = net_conf.get("network", {})
    net_name = network_block.get("name", "mlp")
    net_kwargs = dict(network_block.get("kwargs", {}))
    if net_name not in CLS_NETWORKS:
        raise ValueError(f"Unknown network.name={net_name!r}; expected one of {sorted(CLS_NETWORKS)}")
    LOGGER.info(f"network        = {net_name}")

    if net_name == "mlp":
        # mlp reads its architecture from network.kwargs if present, else the legacy top-level keys.
        def _mlp_arg(key, default=None):
            if key in net_kwargs:
                return net_kwargs[key]
            return net_conf.get(key, default) if default is not None else net_conf[key]

        summary_net = MultiLayerPerceptron(
            output_size=n_summary,
            num_hidden_units=_mlp_arg("num_hidden_units"),
            num_layers=_mlp_arg("num_layers"),
            dropout_rate=_mlp_arg("dropout_rate"),
            normalization=_mlp_arg("normalization", "layer"),
            activation=_mlp_arg("activation", "relu"),
            whitening=whitening_layer,
            residual=_mlp_arg("residual", False),
            input_transform=input_transform,
        )
    else:
        # Channel nets (cls_cnn / cls_transformer) reshape the flat vector to (bins, pairs) internally.
        # This needs the fixed-cls_n_bins-per-pair layout that only the hard_rebinned cache guarantees.
        if scale_cut != "hard_rebinned":
            raise ValueError(
                f"network.name={net_name!r} requires scale_cut=hard_rebinned (fixed cls_n_bins per "
                f"pair for the (bins, pairs) reshape); got scale_cut={scale_cut!r}."
            )
        # PCA whitening rotates across the flat feature axis and destroys the (bin, pair) channel
        # structure; the per-feature asinh input_transform is structure-preserving and IS supported.
        if whitening_layer is not None:
            raise ValueError(
                f"network.name={net_name!r} is incompatible with PCA whitening (pca_components): it "
                f"rotates across the flat feature axis and destroys the (bin, pair) channel structure. "
                f"Remove pca_components; use cls_transform: asinh_per_feature or none instead."
            )
        # The penultimate z-regularization path in base_model iterates network.layers[:-1] as a
        # sequential stack, which these non-sequential nets do not satisfy.
        if z_layer == "penultimate":
            raise ValueError(
                f"network.name={net_name!r} does not support z_layer='penultimate' (that path assumes "
                f"a sequential layer list); use z_layer='last'."
            )
        n_pairs = n_cls // cls_n_bins
        assert n_pairs * cls_n_bins == n_cls, (
            f"n_cls={n_cls} is not divisible by cls_n_bins={cls_n_bins}; the channel reshape requires "
            f"the hard_rebinned layout (fixed cls_n_bins per pair)."
        )
        common_kwargs = dict(
            output_size=n_summary,
            cls_n_bins=cls_n_bins,
            n_pairs=n_pairs,
            input_transform=input_transform,
        )
        if net_name == "cls_transformer":
            from deep_lss.utils import cls_preprocessing

            pair_zi, pair_zj, pair_ptype, n_z = cls_preprocessing.get_selected_pair_identity(
                msfm_conf=msfm_conf,
                dlss_conf=dlss_conf,
                with_lensing=with_lensing,
                with_clustering=with_clustering,
                with_cross_z=with_cross_z,
                with_cross_probe=with_cross_probe,
                lenses_before_sources=lenses_before_sources,
            )
            assert len(pair_zi) == n_pairs, (
                f"pair identity length {len(pair_zi)} != n_pairs {n_pairs}; the identity ordering is "
                f"misaligned with the flat Cls pair axis (check the probe-selection flags)."
            )
            LOGGER.info(
                f"pair identity  = {n_pairs} tokens, n_z={n_z} (zi={pair_zi}, zj={pair_zj}, ptype={pair_ptype})"
            )
            common_kwargs.update(pair_zi=pair_zi, pair_zj=pair_zj, pair_ptype=pair_ptype, n_z=n_z)
        summary_net = CLS_NETWORKS[net_name](**common_kwargs, **net_kwargs)

    summary_net.build((None, n_cls))
    summary_net.summary()

    lr = float(net_conf["learning_rate"])
    sched = net_conf.get("lr_schedule", "cosine")
    warmup_steps = int(net_conf.get("lr_warmup_steps", 0))
    if sched in ("constant", "plateau"):
        # Pass a scalar so optimizer.learning_rate is an assignable Variable
        # (a schedule object would be immutable, blocking warmup / plateau updates).
        lr_schedule = lr
    else:
        lr_alpha = net_conf.get("lr_alpha", 0.0)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=n_steps,
            alpha=lr_alpha,
            warmup_steps=warmup_steps,
        )
    weight_decay = net_conf.get("weight_decay", None)
    ema_kwargs = {"use_ema": True, "ema_momentum": float(ema_momentum)} if ema_momentum is not None else {}
    if weight_decay is not None:
        optimizer = tf.keras.optimizers.AdamW(lr_schedule, weight_decay=float(weight_decay), **ema_kwargs)
    else:
        optimizer = tf.keras.optimizers.Adam(lr_schedule, **ema_kwargs)

    summary_dir = os.path.join(pred_dir, "network/history")
    model = GridLossModel(
        summary_net,
        n_side=None,
        indices=None,
        optimizer=optimizer,
        checkpoint_dir=os.path.join(pred_dir, "network/checkpoint"),
        summary_dir=summary_dir,
        restore_checkpoint=args.restore_checkpoint,
        xla=net_conf.get("xla", False),
    )

    if loss_function == "mse":
        model.setup_grid_loss_step(
            batch_size=batch_size,
            dim_theta=n_params,
            loss="mse",
            dim_x=n_cls,
            dim_summary=n_summary,
            clip_by_global_norm=net_conf.get("clip_by_global_norm", 1.0),
            label_std=label_std,
        )
    else:
        # standardize theta inside the variational head (head models z = (theta - mean) / std; log_prob stays
        # in physical units via the constant log-Jacobian). The MI-bound optimum is affine-invariant, so this
        # is pure optimization conditioning -- but it matters even for the standard 6-param target (~30x scale
        # spread between Om and n_Aia under-trains the tight directions; measured +30% mock FoM 2026-07-12),
        # and is required for the extended target (H0 ~ 70 NaNs the head in physical units) and for flow heads
        # (raw theta feeds the coupling MLPs directly). Default ON; computed from the physical training labels.
        theta_shift, theta_scale = None, None
        if mi.get("standardize_theta", True):
            theta_shift, theta_scale = training_helpers.theta_standardization_from_samples(
                out_dict["grid/cosmos/train"]
            )
            LOGGER.info(f"VMIM head theta standardization: shift = {theta_shift}, scale = {theta_scale}")

        model.setup_grid_loss_step(
            batch_size=batch_size,
            dim_theta=n_params,
            loss="mutual_info",
            dim_x=n_cls,
            dim_summary=n_summary,
            mutual_info_estimator=mi["estimator"],
            clip_by_global_norm=net_conf.get("clip_by_global_norm", 1.0),
            mutual_info_kwargs={
                "theta_shift": theta_shift,
                "theta_scale": theta_scale,
                "density_estimator": mi["density_estimator"],
                "num_hidden_layers": mi["kwargs"].get("num_hidden_layers", 2),
                "num_hidden_units": mi["kwargs"].get("num_hidden_units", 128),
                "activation": mi["kwargs"].get("activation", "relu"),
                "full_covariance": mi["kwargs"].get("full_covariance", True),
                "num_components": mi["kwargs"].get("num_components", 4),
                "num_layers": mi["kwargs"].get("num_layers", 4),
                "scale_eps": float(mi["kwargs"].get("scale_eps", 1e-5)),
                "log_scale_clip": float(mi["kwargs"].get("log_scale_clip", 5.0)),
                "permute": mi["kwargs"].get("permute", False),
            },
            z_weight=z_weight,
            z_type=z_type,
            z_layer=z_layer,
        )

    assert getattr(model, "grid_train_step_uses_pair_ids", False) == uses_invariance, (
        f"uses_invariance derived from the loss config ({uses_invariance}) disagrees with the model's "
        f"grid_train_step_uses_pair_ids ({getattr(model, 'grid_train_step_uses_pair_ids', False)}); the "
        f"dataset would not match the train-step signature."
    )

    tb_writer = tf.summary.create_file_writer(summary_dir)

    es_conf = net_conf.get("early_stopping", {})
    early_stopper = (
        EarlyStopper(
            patience=es_conf.get("patience", 10),
            min_delta=float(es_conf.get("min_delta", 1e-4)),
            min_steps=es_conf.get("min_steps", 0),
        )
        if es_conf
        else None
    )

    rlrop_conf = net_conf.get("reduce_lr_on_plateau", {})
    reduce_lr = (
        ReduceLROnPlateau(
            factor=float(rlrop_conf.get("factor", 0.5)),
            patience=rlrop_conf.get("patience", 3),
            min_delta=float(rlrop_conf.get("min_delta", 0.0)),
            cooldown=rlrop_conf.get("cooldown", 0),
            min_lr=float(rlrop_conf.get("min_lr", 0.0)),
        )
        if (sched == "plateau" and rlrop_conf)
        else None
    )

    # --- test-set tensors for the vali MSE metric and final test evaluation — must match the training preprocessing ---
    # grid/cls/test is already log-transformed; grid/cls_raw/test is linear.
    if apply_log:
        cls_test_eval = np.array(out_dict["grid/cls/test"], dtype=np.float32)
    else:
        cls_test_eval = np.array(out_dict["grid/cls_raw/test"], dtype=np.float32)
        if ell_weighting is not None and out_dict.get("ell_weights") is not None:
            cls_test_eval = cls_test_eval * out_dict["ell_weights"].astype(np.float32)
    grid_cosmos_eval = np.array(out_dict["grid/cosmos/test"], dtype=np.float32)

    # The train/test split is over REALIZATIONS, so all n_cosmo Sobol points appear in the test set,
    # concatenated cosmology-major. A contiguous head therefore covers only ~n_eval/n_realizations
    # cosmologies (~26 of 2500 at the old n_eval=2048), which is far too few distinct theta to
    # estimate parameter recovery. Draw the subset at random (fixed seed) so it spans the prior.
    _eval_rng = np.random.default_rng(12345)
    _n_eval = min(int(net_conf.get("n_vali_mse_examples", 8192)), len(cls_test_eval))
    _eval_idx = np.sort(_eval_rng.choice(len(cls_test_eval), size=_n_eval, replace=False))
    cls_test_eval_tf = tf.constant(cls_test_eval[_eval_idx])
    grid_cosmos_eval_tf = tf.constant(grid_cosmos_eval[_eval_idx])

    # Per-parameter normalization for the vali MSE. An unweighted mean of squared errors in PHYSICAL
    # units is dominated by the widest priors: for the lensing_nla target (Om, s8, w0, Aia, n_Aia)
    # the label variances are [0.009, 0.053, 0.075, 3.0, 8.3], so Aia+n_Aia carry 98.8% of the metric
    # and Om just 0.08% -- it measures IA nuisance recovery, not the cosmology the FoM is built on.
    # Dividing by the label std puts every parameter on a "fraction of prior variance left
    # unexplained" footing (1.0 = no better than predicting the mean); vali_nmse_cosmo then averages
    # only the cosmological parameters and is the quantity to rank quick screening runs on.
    _cosmos_train_flat = np.asarray(out_dict["grid/cosmos/train"], dtype=np.float32).reshape(-1, n_params)
    mse_scale_tf = tf.constant(np.maximum(_cosmos_train_flat.std(axis=0), 1e-12), dtype=tf.float32)
    fom_param_idx = training_helpers.cosmo_param_indices(params, msfm_conf["analysis"]["params"]["cosmo"])
    LOGGER.info(
        f"vali MSE: {_n_eval} random test examples; vali_nmse_cosmo over "
        f"{[params[i] for i in fom_param_idx]} (of {params})"
    )

    # Full-test-set vali_loss costs ~76% of walltime for these small nets (200k examples run eagerly
    # with a host sync per batch). vali_subsample caps it at a fixed random subset; the MC error on
    # the bound is negligible for early-stopping / LR-plateau decisions. None => full test set.
    vali_subsample = net_conf.get("vali_subsample", None)
    if vali_subsample is not None and int(vali_subsample) < len(cls_test_eval):
        _vs_idx = np.sort(_eval_rng.choice(len(cls_test_eval), size=int(vali_subsample), replace=False))
        cl_dset_vali = (
            tf.data.Dataset.from_tensor_slices((cls_test_eval[_vs_idx], grid_cosmos_eval[_vs_idx]))
            .batch(batch_size)
            .cache()
            .prefetch(2)
        )
        LOGGER.info(f"vali_loss on a fixed random subsample of {int(vali_subsample)}/{len(cls_test_eval)} examples")
    else:
        cl_dset_vali = cl_dset_test

    train_steps, train_losses = [], []
    vali_steps, vali_losses_history, vali_mse_history = [], [], []
    vali_nmse_cosmo_history = []

    for i, batch in LOGGER.progressbar(enumerate(cl_dset_train), at_level="info", total=n_steps + 1, desc="training"):
        if i > n_steps:
            break

        # Linear warmup for the scalar-LR modes; after warmup the plateau reducer owns the LR.
        if sched in ("constant", "plateau") and warmup_steps > 0 and i < warmup_steps:
            optimizer.learning_rate.assign(lr * (i + 1) / warmup_steps)

        if uses_invariance:
            cl_batch, cosmo_batch, i_sobol_batch, i_signal_batch = batch
            loss = model.grid_train_step(cl_batch, cosmo_batch, i_sobol_batch, i_signal_batch)
        else:
            cl_batch, cosmo_batch = batch
            loss = model.grid_train_step(cl_batch, cosmo_batch)

        if i % log_every == 0:
            train_loss_val = float(loss.numpy())
            train_steps.append(i)
            train_losses.append(train_loss_val)
            with tb_writer.as_default():
                tf.summary.scalar("loss/train", train_loss_val, step=i)
            tb_writer.flush()

        if i > 0 and i % vali_every == 0:
            vali_loss_vals = [
                float(model.vali_loss_fn(model(cl_v, training=False), cosmo_v).numpy())
                for cl_v, cosmo_v in cl_dset_vali
            ]
            vali_loss = np.mean(vali_loss_vals)
            vali_steps.append(i)
            vali_losses_history.append(vali_loss)
            vali_preds = tf.concat(
                [model(cls_test_eval_tf[j : j + batch_size], training=False) for j in range(0, _n_eval, batch_size)],
                axis=0,
            )
            if hasattr(model, "vali_posterior_mean_fn"):
                posterior_mean = model.vali_posterior_mean_fn(vali_preds)
                _err = posterior_mean - grid_cosmos_eval_tf
                # vali_mse stays the raw physical-units mean for continuity with older runs; the
                # normalized per-parameter version below is what the metric should actually be read on.
                mse_val = float(tf.reduce_mean(tf.square(_err)).numpy())
                nmse_per_param = tf.reduce_mean(tf.square(_err / mse_scale_tf), axis=0).numpy()
                nmse_cosmo = float(np.mean(nmse_per_param[fom_param_idx])) if fom_param_idx else float("nan")
            else:
                mse_val = float("nan")
                nmse_per_param = np.full(n_params, np.nan, dtype=np.float32)
                nmse_cosmo = float("nan")
            vali_mse_history.append(mse_val)
            vali_nmse_cosmo_history.append(nmse_cosmo)
            with tb_writer.as_default():
                tf.summary.scalar("loss/vali", vali_loss, step=i)
                tf.summary.scalar("loss/vali_mse", mse_val, step=i)
                tf.summary.scalar("loss/vali_nmse_cosmo", nmse_cosmo, step=i)
                for _p, _v in zip(params, nmse_per_param):
                    tf.summary.scalar(f"nmse/{_p}", float(_v), step=i)
                tf.summary.scalar("lr", float(optimizer.learning_rate.numpy()), step=i)
            tb_writer.flush()
            LOGGER.info(
                f"step {i:>7d}  vali_loss = {vali_loss:.4f}  vali_nmse_cosmo = {nmse_cosmo:.4f}  |  "
                + "  ".join(f"{p}={v:.3f}" for p, v in zip(params, nmse_per_param))
            )

            # Reduce the LR on a validation plateau (guarded so warmup never undoes a reduction).
            if reduce_lr is not None and i >= warmup_steps:
                current_lr = float(optimizer.learning_rate.numpy())
                new_lr = reduce_lr.update(vali_loss, current_lr)
                if new_lr < current_lr:
                    optimizer.learning_rate.assign(new_lr)
                    LOGGER.info(f"  -> reduce LR {current_lr:.2e} -> {new_lr:.2e}")

            if early_stopper is not None:
                improved = early_stopper.update(vali_loss)
                if improved:
                    model.save_model()
                    LOGGER.info(f"  -> new best vali_loss={vali_loss:.4f}, saved checkpoint")
                if early_stopper.should_stop(i):
                    LOGGER.info(
                        f"Early stopping at step {i} "
                        f"(best={early_stopper.best_loss:.4f}, wait={early_stopper.wait})"
                    )
                    break

    if ema_momentum is not None:
        # Overwrite live weights with their EMA average (in place), then save/evaluate with those.
        # This supersedes the early-stopping "best vali" restore as the final-weight selector;
        # early stopping still controls *when* the loop stops.
        optimizer.finalize_variable_values(model.trainable_variables)
        LOGGER.info(f"Finalized EMA weights (momentum={ema_momentum})")
        model.save_model()
    elif early_stopper is None:
        model.save_model()
    else:
        model.restore_model()

    cls_evaluation.save_loss_curve(
        pred_dir=pred_dir,
        pred_file=pred_file,
        train_steps=train_steps,
        train_losses=train_losses,
        vali_steps=vali_steps,
        vali_losses=vali_losses_history,
        vali_mse=vali_mse_history,
        vali_nmse_cosmo=vali_nmse_cosmo_history,
        log_every=log_every,
    )

    # --- evaluate on test set (directly from out_dict, matching the notebook) ---
    LOGGER.info("Evaluating on test set...")
    cls_test = cls_test_eval  # already extracted above with correct apply_log/ell_weighting
    grid_cosmos = grid_cosmos_eval

    grid_preds = np.concatenate(
        [
            model(tf.constant(cls_test[i : i + batch_size]), training=False).numpy()
            for i in range(0, len(cls_test), batch_size)
        ],
        axis=0,
    )

    with h5py.File(pred_file, "w") as f:
        f.create_dataset("grid/preds/test", data=grid_preds)
        f.create_dataset("grid/cosmos/test", data=grid_cosmos)
        f.create_dataset("grid/i_sobol/test", data=out_dict["grid/i_sobol/test"])
        f.create_dataset("grid/i_signal/test", data=out_dict["grid/i_signal/test"])
        f.create_dataset("grid/i_noise/test", data=out_dict["grid/i_noise/test"])

    LOGGER.info(f"Saved {len(grid_preds)} test predictions to {pred_file}")

    # --- named grid observations (example 0 per cosmology, labeled by simulation indices) ---
    if args.include_grid:
        obs_i_sobol = out_dict["grid/obs/i_sobol"]
        obs_i_signal = out_dict["grid/obs/i_signal"]
        obs_i_noise = out_dict["grid/obs/i_noise"]
        obs_cls = out_dict["grid/obs/cls"]
        obs_cosmos = out_dict["grid/obs/cosmos"]

        n_grid_obs = min(args.n_grid_examples, len(obs_cls))
        obs_preds = np.concatenate(
            [
                model(tf.constant(obs_cls[i : i + batch_size], dtype=tf.float32), training=False).numpy()
                for i in range(0, n_grid_obs, batch_size)
            ],
            axis=0,
        )[:n_grid_obs]

        for k in range(n_grid_obs):
            label = f"grid_({int(obs_i_sobol[k])},{int(obs_i_signal[k])},{int(obs_i_noise[k])})"
            evaluation.append_obs_to_file(pred_file, f"obs/preds/{label}", obs_preds[k])
            evaluation.append_obs_to_file(pred_file, f"obs/cosmos/{label}", obs_cosmos[k])

        LOGGER.info(f"Saved {n_grid_obs} grid observations")

    # --- mock observations ---
    if args.include_mocks:
        mock_labels = args.mock_labels
        if mock_labels is None:
            mock_labels = evaluation.discover_mock_labels(args.data_dir)
            LOGGER.info(f"Auto-discovered {len(mock_labels)} mock(s) in {args.data_dir}/obs: {mock_labels}")
        for label in mock_labels:
            try:
                cls_evaluation.evaluate_mock_cls(
                    label=label,
                    model=model,
                    pred_file=pred_file,
                    data_dir=args.data_dir,
                    msfm_conf=msfm_conf,
                    dlss_conf=dlss_conf,
                    params=params,
                    batch_size=batch_size,
                    with_lensing=with_lensing,
                    with_clustering=with_clustering,
                    with_cross_z=with_cross_z,
                    with_cross_probe=with_cross_probe,
                    lenses_before_sources=lenses_before_sources,
                    apply_log=apply_log,
                    ell_weighting=ell_weighting,
                    scale_cut=scale_cut,
                    cls_n_bins=cls_n_bins,
                )
            except Exception as e:
                LOGGER.warning(f"mock {label} evaluation failed ({e}), skipping")

    # --- DES Y3 real-data observation ---
    if args.include_des:
        try:
            cls_evaluation.evaluate_des_y3(
                model=model,
                msfm_conf=msfm_conf,
                dlss_conf=dlss_conf,
                data_dir=args.data_dir,
                pred_file=pred_file,
                with_lensing=with_lensing,
                with_clustering=with_clustering,
                with_cross_z=with_cross_z,
                with_cross_probe=with_cross_probe,
                lenses_before_sources=lenses_before_sources,
                apply_log=apply_log,
                ell_weighting=ell_weighting,
                scale_cut=scale_cut,
                cls_n_bins=cls_n_bins,
            )
        except Exception as e:
            LOGGER.warning(f"DES Y3 evaluation failed ({e}), skipping")


if __name__ == "__main__":
    main()
