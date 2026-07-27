"""Parameter counts for the current configs/deepsphere/{lensing,clustering,combined}/maps+cls.yaml.

Builds each ResNetMapsPlusCLSNetwork exactly as run_training.py does (vmim loss, dim_summary_fac=1,
so n_output = n_params of the probes config) and prints total/trainable/non-trainable counts plus a
per-layer summary. Run inside the tensorflow container on a compute node (see the .sh next to this
file).
"""

import os

import numpy as np
import tensorflow as tf

REPOS = "/users/athomsen/dlss/repos"
OUT = "/iopsstor/scratch/cscs/athomsen/deep_lss/claude/tmp/smoke_multires_gcnn"

# Optional override to compare the current raw-concat fusion (map_feature_dim unset) against a
# common bottleneck width for both branches: set MAP_FEATURE_DIM to add a Dense map_projection of
# that width and resize the Cls embedding branch's final layer to match.
_MAP_FEATURE_DIM_OVERRIDE = os.environ.get("MAP_FEATURE_DIM")
MAP_FEATURE_DIM_OVERRIDE = int(_MAP_FEATURE_DIM_OVERRIDE) if _MAP_FEATURE_DIM_OVERRIDE else None

from msfm.utils import input_output
from deep_lss.utils import configuration
from deep_lss.nets import NETWORKS
from deep_lss.nets.encoders.maps.gcnn.resnet_multires import ResNetMultiResEncoder
from deep_lss.nets.composite.resnet_maps_plus_cls import ResNetMapsPlusCLSNetwork
from deep_lss.nets.layers.cls.embedding import get_cls_embedding_layers

CASES = [
    ("lensing", "lensing_nla"),
    ("clustering", "clustering"),
    ("combined", "combined_nla"),
]

msfm_conf = input_output.read_yaml(f"{REPOS}/multiprobe-simulation-forward-model/configs/v17/baseline.yaml")
n_side = msfm_conf["analysis"]["n_side"]

results = {}
for probe, probes_cfg in CASES:
    print(f"\n===== {probe} (probes/{probes_cfg}.yaml) =====", flush=True)
    dlss_conf = configuration.read_split_configs(
        f"{REPOS}/y3-deep-lss/configs/probes/{probes_cfg}.yaml",
        f"{REPOS}/y3-deep-lss/configs/scales/8wl,32gc.yaml",
    )
    net_conf = input_output.read_yaml(f"{REPOS}/y3-deep-lss/configs/deepsphere/{probe}/maps+cls.yaml")

    n_output = len(dlss_conf["dset"]["training"]["params"])  # vmim, dim_summary_fac=1
    n_z_bins = 0
    if dlss_conf["dset"]["common"]["with_lensing"]:
        n_z_bins += len(msfm_conf["survey"]["metacal"]["z_bins"])
    if dlss_conf["dset"]["common"]["with_clustering"]:
        n_z_bins += len(msfm_conf["survey"]["maglim"]["z_bins"])
    batch = net_conf["dset"]["training"]["grid"]["local_batch_size"]

    smoothing_kwargs = configuration.get_smoothing_kwargs("mutual_info", msfm_conf, dlss_conf, net_conf, dir_base=OUT)
    smooth_nside, smooth_indices, _ = configuration.resolve_smooth_nside(net_conf, dlss_conf, msfm_conf)
    is_multires = "split_probes" in smoothing_kwargs
    input_norm = bool(net_conf["network"].get("input_norm", False))
    print(f"n_output={n_output}, n_z_bins={n_z_bins}, nside={smooth_nside}, "
          f"n_pix={len(smooth_indices)}, multires={is_multires}")

    net_spec = NETWORKS["resnet"](
        out_features=n_output,
        smoothing_kwargs=None if is_multires else smoothing_kwargs,
        **({"input_norm": True} if input_norm and not is_multires else {}),
        **({"smoothing_external": True} if is_multires else {}),
        **net_conf["network"]["kwargs"],
    )
    _, l_min_per_pair, l_max_per_pair = configuration.get_cls_bounds_per_pair(msfm_conf, dlss_conf)
    cls_conf = net_conf["network"]["cls"]
    if MAP_FEATURE_DIM_OVERRIDE is not None:
        cls_conf = dict(cls_conf)
        cls_conf["embedding_layers"] = list(cls_conf["embedding_layers"][:-1]) + [MAP_FEATURE_DIM_OVERRIDE]
    map_encoder = None
    if is_multires:
        map_encoder = ResNetMultiResEncoder(
            smoothing_kwargs=smoothing_kwargs,
            layers=net_spec.get_conv_layers(),
            nside=smooth_nside,
            n_neighbors=net_conf["network"]["n_neighbors"],
            max_batch_size=batch,
            input_norm=input_norm,
        )
    network = ResNetMapsPlusCLSNetwork(
        conv_layers=None if is_multires else net_spec.get_conv_layers(),
        cls_embedding_layers=get_cls_embedding_layers(
            cls_conf["embedding_layers"], dropout_rate=cls_conf["embedding_dropout_rate"]
        ),
        regression_head_layers=net_spec.get_head_layers_no_flatten(),
        n_side=None if is_multires else smooth_nside,
        tfr_n_side=n_side,
        indices=None if is_multires else smooth_indices,
        n_neighbors=net_conf["network"]["n_neighbors"],
        max_batch_size=batch,
        initial_Fin=None if is_multires else n_z_bins,
        n_cls_bins=cls_conf["n_bins"],
        l_min_per_pair=l_min_per_pair,
        l_max_per_pair=l_max_per_pair,
        cls_transform=cls_conf["transform"],
        map_feature_dim=MAP_FEATURE_DIM_OVERRIDE
        if MAP_FEATURE_DIM_OVERRIDE is not None
        else net_conf["network"].get("map_feature_dim", None),
        map_encoder=map_encoder,
    )
    if network.gcnn is not None:
        network.gcnn.build((batch, len(smooth_indices), n_z_bins))
    network(
        (tf.zeros((2, len(smooth_indices), n_z_bins)), tf.zeros((2, 3 * n_side, len(l_min_per_pair)))),
        training=False,
    )
    if network.map_projection is not None:
        print(f"map_projection kernel shape: {tuple(network.map_projection.kernel.shape)}")
    total = network.count_params()
    trainable = int(sum(np.prod(v.shape) for v in network.trainable_variables))
    non_trainable = int(sum(np.prod(v.shape) for v in network.non_trainable_variables))
    assert trainable + non_trainable == total
    results[probe] = (total, trainable, non_trainable)
    network.summary(line_length=100)
    print(f"{probe}: total {total:,} | trainable {trainable:,} | non-trainable {non_trainable:,}", flush=True)

print("\n===== SUMMARY =====")
for probe, (total, trainable, non_trainable) in results.items():
    print(f"{probe:>11}: total {total:>12,} | trainable {trainable:>12,} | non-trainable {non_trainable:>10,}")
print("\nDONE", flush=True)
