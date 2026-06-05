import argparse, h5py, os, yaml
import numpy as np
import tensorflow as tf

for gpu in tf.config.list_physical_devices(device_type="GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from tqdm import tqdm

from msfm.utils import files, logger

LOGGER = logger.get_logger(__file__)

from deep_lss.models.grid_model import GridLossModel
from deep_lss.nets.mlp import MultiLayerPerceptron, PCAWhiteningLayer
from deep_lss.utils import cls_evaluation, evaluation
from deep_lss.utils.mutual_info_loss import distance_correlation

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


def setup():
    parser = argparse.ArgumentParser(
        description="Train an MLP summary network on binned power spectra (Cls) using the mutual information loss."
    )
    parser.add_argument("--msfm_config", required=True)
    parser.add_argument("--dlss_config", required=True)
    parser.add_argument("--mlp_config", required=True)
    parser.add_argument("--vmim_config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--restore_checkpoint", action="store_true")

    # Observation inclusion flags (all default off)
    parser.add_argument("--include_grid", action="store_true")
    parser.add_argument("--n_grid_examples", type=int, default=4)
    parser.add_argument("--include_des", action="store_true")
    parser.add_argument("--include_mocks", action="store_true", help="evaluate mock observations from data_dir/obs/")
    parser.add_argument("--mock_labels", nargs="+", default=["fiducial_bench"])

    return parser.parse_args()


def main():
    args = setup()
    msfm_conf = files.load_config(args.msfm_config)
    from msfm.utils import input_output

    dlss_conf = input_output.read_yaml(args.dlss_config)
    mlp_conf = input_output.read_yaml(args.mlp_config)
    vmim_conf = input_output.read_yaml(args.vmim_config)

    common = dlss_conf["dset"]["common"]
    with_lensing = common["with_lensing"]
    with_clustering = common["with_clustering"]
    with_cross_z = common["with_cross_z"]
    with_cross_probe = common["with_cross_probe"]
    ggl_only = common.get("ggl_only", False)

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
    n_summary = 2 * n_params

    scale_cut = mlp_conf.get("scale_cut", "hard")

    n_steps = mlp_conf["n_steps"]
    batch_size = mlp_conf["batch_size"]
    log_every = mlp_conf["log_every"]
    vali_every = mlp_conf["vali_every"]
    signal_indices = mlp_conf.get("signal_indices", None)
    noise_indices = mlp_conf.get("noise_indices", 0.8)

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

    with open(os.path.join(pred_dir, "configs.yaml"), "w") as f:
        yaml.dump_all([mlp_conf, vmim_conf, dlss_conf, msfm_conf], f)

    apply_log = mlp_conf.get("apply_log", True)
    ell_weighting = mlp_conf.get("ell_weighting", None)

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
        ggl_only=ggl_only,
        batch_size=batch_size,
        apply_log=apply_log,
        standardize=False,
        ell_weighting=ell_weighting,
    )

    n_cls = out_dict["grid/cls/train"].shape[-1]

    n_pca = mlp_conf.get("pca_components", None)
    if n_pca is not None:
        pca_whiten = mlp_conf.get("pca_whiten", True)
        whitening_layer = PCAWhiteningLayer(n_components=n_pca, whiten=pca_whiten)
        if apply_log:
            pca_fit_data = out_dict["grid/cls/train"]
        else:
            pca_fit_data = out_dict["grid/cls_raw/train"].copy()
            if ell_weighting is not None and out_dict.get("ell_weights") is not None:
                pca_fit_data = pca_fit_data * out_dict["ell_weights"].astype(pca_fit_data.dtype)
        whitening_layer.fit(pca_fit_data)
    else:
        whitening_layer = None

    summary_net = MultiLayerPerceptron(
        output_size=n_summary,
        num_hidden_units=mlp_conf["num_hidden_units"],
        num_layers=mlp_conf["num_layers"],
        dropout_rate=mlp_conf["dropout_rate"],
        normalization=mlp_conf.get("normalization", "layer"),
        activation=mlp_conf.get("activation", "relu"),
        whitening=whitening_layer,
    )
    summary_net.build((None, n_cls))
    summary_net.summary()

    lr = float(mlp_conf["learning_rate"])
    if mlp_conf.get("lr_schedule", "cosine") == "constant":
        lr_schedule = lr
    else:
        warmup_steps = mlp_conf.get("lr_warmup_steps", 0)
        lr_alpha = mlp_conf.get("lr_alpha", 0.0)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=n_steps,
            alpha=lr_alpha,
            warmup_steps=warmup_steps,
        )
    weight_decay = mlp_conf.get("weight_decay", None)
    if weight_decay is not None:
        optimizer = tf.keras.optimizers.AdamW(lr_schedule, weight_decay=float(weight_decay))
    else:
        optimizer = tf.keras.optimizers.Adam(lr_schedule)

    summary_dir = os.path.join(pred_dir, "network/history")
    model = GridLossModel(
        summary_net,
        n_side=None,
        indices=None,
        optimizer=optimizer,
        checkpoint_dir=os.path.join(pred_dir, "network/checkpoint"),
        summary_dir=summary_dir,
        restore_checkpoint=args.restore_checkpoint,
        xla=mlp_conf.get("xla", False),
    )

    model.setup_grid_loss_step(
        batch_size=batch_size,
        dim_theta=n_params,
        loss="mutual_info",
        dim_x=n_cls,
        dim_summary=n_summary,
        mutual_info_estimator=vmim_conf["estimator"],
        clip_by_global_norm=mlp_conf.get("clip_by_global_norm", 1.0),
        mutual_info_kwargs={
            "density_estimator": vmim_conf["density_estimator"],
            "num_hidden_layers": vmim_conf.get("num_hidden_layers", 2),
            "num_hidden_units": vmim_conf.get("num_hidden_units", 128),
            "activation": vmim_conf.get("activation", "relu"),
            "full_covariance": vmim_conf.get("full_covariance", True),
            "num_components": vmim_conf.get("num_components", 4),
            "num_layers": vmim_conf.get("num_layers", 4),
            "scale_eps": float(vmim_conf.get("scale_eps", 1e-5)),
            "log_scale_clip": float(vmim_conf.get("log_scale_clip", 5.0)),
        },
    )

    tb_writer = tf.summary.create_file_writer(summary_dir)

    es_conf = mlp_conf.get("early_stopping", {})
    early_stopper = (
        EarlyStopper(
            patience=es_conf.get("patience", 10),
            min_delta=float(es_conf.get("min_delta", 1e-4)),
            min_steps=es_conf.get("min_steps", 0),
        )
        if es_conf
        else None
    )

    # --- test-set tensors for DC evaluation — must match the training preprocessing ---
    # grid/cls/test is already log-transformed; grid/cls_raw/test is linear.
    if apply_log:
        cls_test_dc = np.array(out_dict["grid/cls/test"], dtype=np.float32)
    else:
        cls_test_dc = np.array(out_dict["grid/cls_raw/test"], dtype=np.float32)
        if ell_weighting is not None and out_dict.get("ell_weights") is not None:
            cls_test_dc = cls_test_dc * out_dict["ell_weights"].astype(np.float32)
    grid_cosmos_dc = np.array(out_dict["grid/cosmos/test"], dtype=np.float32)
    _n_dc = min(2048, len(cls_test_dc))
    cls_test_dc_tf = tf.constant(cls_test_dc[:_n_dc])
    grid_cosmos_dc_tf = tf.constant(grid_cosmos_dc[:_n_dc])

    train_steps, train_losses = [], []
    vali_steps, vali_losses_history, vali_dc_history = [], [], []

    for i, (cl_batch, cosmo_batch) in tqdm(enumerate(cl_dset_train), total=n_steps + 1):
        if i > n_steps:
            break
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
                for cl_v, cosmo_v in cl_dset_test
            ]
            vali_loss = np.mean(vali_loss_vals)
            vali_steps.append(i)
            vali_losses_history.append(vali_loss)
            dc_preds = tf.concat(
                [model(cls_test_dc_tf[j : j + batch_size], training=False) for j in range(0, _n_dc, batch_size)],
                axis=0,
            )
            dc_val = float(distance_correlation(dc_preds, grid_cosmos_dc_tf, training=False).numpy())
            vali_dc_history.append(dc_val)
            with tb_writer.as_default():
                tf.summary.scalar("loss/vali", vali_loss, step=i)
                tf.summary.scalar("loss/vali_dc", dc_val, step=i)
            tb_writer.flush()
            LOGGER.info(f"step {i:>7d}  vali_loss = {vali_loss:.4f}  vali_dc = {dc_val:.4f}")

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

    if early_stopper is None:
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
        vali_dc=vali_dc_history,
        log_every=log_every,
    )

    # --- evaluate on test set (directly from out_dict, matching the notebook) ---
    LOGGER.info("Evaluating on test set...")
    cls_test = cls_test_dc  # already extracted above with correct apply_log/ell_weighting
    grid_cosmos = grid_cosmos_dc

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
        for label in args.mock_labels:
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
                    ggl_only=ggl_only,
                    apply_log=apply_log,
                    ell_weighting=ell_weighting,
                    scale_cut=scale_cut,
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
                ggl_only=ggl_only,
                apply_log=apply_log,
                ell_weighting=ell_weighting,
                scale_cut=scale_cut,
            )
        except Exception as e:
            LOGGER.warning(f"DES Y3 evaluation failed ({e}), skipping")


if __name__ == "__main__":
    main()
