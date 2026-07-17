"""Smoke test for the multi-resolution DeepSphere/ResNet path (ResNetMultiResEncoder).

Mirrors the run_training.py construction for configs/deepsphere/combined/{maps+cls,maps}.yaml with
smooth_nside: {clustering: 256}, plus a single-resolution regression build (config without
smooth_nside) and a checkpoint save/restore round trip. Run inside the tensorflow container on a
compute node (see smoke_multires_gcnn.sh next to this file).
"""

import copy
import os
import re

import numpy as np
import tensorflow as tf

REPOS = "/users/athomsen/dlss/repos"
OUT = "/iopsstor/scratch/cscs/athomsen/deep_lss/runs/smoke_multires_gcnn"

from msfm.utils import input_output
from deep_lss.utils import configuration
from deep_lss.nets import NETWORKS
from deep_lss.nets.encoders.maps.gcnn.resnet_multires import ResNetMultiResEncoder, split_layers_at_nside
from deep_lss.nets.composite.resnet_maps_plus_cls import ResNetMapsPlusCLSNetwork
from deep_lss.nets.layers.cls.embedding import get_cls_embedding_layers
from deep_lss.nets.layers.maps.input_normalization import compute_input_norm_stats


def stage(msg):
    print(f"\n===== {msg} =====", flush=True)


msfm_conf = input_output.read_yaml(f"{REPOS}/multiprobe-simulation-forward-model/configs/v17/baseline.yaml")
dlss_conf = configuration.read_split_configs(
    f"{REPOS}/y3-deep-lss/configs/probes/combined_nla.yaml",
    f"{REPOS}/y3-deep-lss/configs/scales/8wl,32gc.yaml",
)
net_conf = input_output.read_yaml(f"{REPOS}/y3-deep-lss/configs/deepsphere/combined/maps+cls.yaml")

n_side = msfm_conf["analysis"]["n_side"]
n_z_bins = len(msfm_conf["survey"]["metacal"]["z_bins"]) + len(msfm_conf["survey"]["maglim"]["z_bins"])
n_output = 12  # arbitrary summary dim for the smoke test
effective_local_batch_size = net_conf["dset"]["training"]["grid"]["local_batch_size"]

stage("1. smoothing spec")
smoothing_kwargs = configuration.get_smoothing_kwargs(
    "mutual_info", msfm_conf, dlss_conf, net_conf, dir_base=OUT
)
assert "split_probes" in smoothing_kwargs, "expected a split_probes spec for the combined multi-res config"
specs = smoothing_kwargs["split_probes"]
by_probe = {s["probe"]: s for s in specs}
assert by_probe["lensing"]["smoothing_kwargs"]["nside"] == 512
assert by_probe["lensing"]["parent_output_idx"] is None
assert by_probe["clustering"]["smoothing_kwargs"]["nside"] == 256
assert by_probe["clustering"]["parent_output_idx"] is not None
print("split spec OK:", [(s["probe"], s["smoothing_kwargs"]["nside"], s["n_channels"]) for s in specs])

smooth_nside, smooth_indices, parent_output_idx = configuration.resolve_smooth_nside(net_conf, dlss_conf, msfm_conf)
assert smooth_nside == 512 and parent_output_idx is None, "network must run at native 512, no pipeline downsampling"
fine_indices = np.asarray(by_probe["lensing"]["smoothing_kwargs"]["indices"])
coarse_indices = np.asarray(by_probe["clustering"]["smoothing_kwargs"]["indices"])
expected_coarse, _ = configuration.get_smooth_nside_indices(fine_indices, 512, 256)
assert np.array_equal(expected_coarse, coarse_indices), "coarse footprint != parent set of fine footprint"
print(f"index alignment OK: {len(fine_indices)} px @512 -> {len(coarse_indices)} px @256")

stage("2. splitter unit checks")
probe_spec_kwargs = dict(out_features=n_output, smoothing_kwargs=None, smoothing_external=True)
net_spec = NETWORKS["resnet"](**probe_spec_kwargs, **net_conf["network"]["kwargs"])
pre, post, split_Fin = split_layers_at_nside(net_spec.get_conv_layers(), 512, n_z_bins, 256)
print(f"split: {len(pre)} pre-layers, {len(post)} post-layers, Fin at split = {split_Fin}")
assert len(pre) == 1 and split_Fin == net_conf["network"]["kwargs"]["base_channels"]
try:
    split_layers_at_nside(net_spec.get_conv_layers(), 512, n_z_bins, 24)
    raise AssertionError("expected ValueError for an unreachable injection nside")
except ValueError as e:
    print(f"unreachable-nside error OK: {e}")

stage("3. multi-res encoder build + forward (maps+cls conv branch)")
encoder = ResNetMultiResEncoder(
    smoothing_kwargs=smoothing_kwargs,
    layers=net_spec.get_conv_layers(),
    nside=smooth_nside,
    n_neighbors=net_conf["network"]["n_neighbors"],
    max_batch_size=effective_local_batch_size,
    input_norm=True,
)
maps = tf.random.normal((2, len(smooth_indices), n_z_bins))
feats = encoder(maps, training=False)
print(f"encoder features: {feats.shape}")
assert feats.shape[0] == 2 and len(feats.shape) == 3

stage("4. input-norm measurement path")
fake_dset = tf.data.Dataset.from_tensors((tf.random.normal((4, len(smooth_indices), n_z_bins)),)).repeat(3)
stats = compute_input_norm_stats(encoder.smooth_groups, fake_dset, n_batches=3, masks=encoder.masks)
assert len(stats) == 2, f"expected 2 input-norm groups, got {len(stats)}"
assert stats[0][0].shape == (by_probe["lensing"]["n_channels"],)
assert stats[1][0].shape == (by_probe["clustering"]["n_channels"],)
encoder.load_input_norm_stats(stats)
feats = encoder(maps, training=False)
print(f"stats loaded, forward OK: {feats.shape}")

stage("5. maps+cls composite (multi-res)")
_, l_min_per_pair, l_max_per_pair = configuration.get_cls_bounds_per_pair(msfm_conf, dlss_conf)
cls_conf = net_conf["network"]["cls"]


def build_multires_composite():
    spec = NETWORKS["resnet"](**probe_spec_kwargs, **net_conf["network"]["kwargs"])
    enc = ResNetMultiResEncoder(
        smoothing_kwargs=smoothing_kwargs,
        layers=spec.get_conv_layers(),
        nside=smooth_nside,
        n_neighbors=net_conf["network"]["n_neighbors"],
        max_batch_size=effective_local_batch_size,
        input_norm=True,
    )
    net = ResNetMapsPlusCLSNetwork(
        conv_layers=None,
        cls_embedding_layers=get_cls_embedding_layers(cls_conf["embedding_layers"],
                                                      dropout_rate=cls_conf["embedding_dropout_rate"]),
        regression_head_layers=spec.get_head_layers_no_flatten(),
        n_side=None,
        tfr_n_side=n_side,
        indices=None,
        n_neighbors=net_conf["network"]["n_neighbors"],
        max_batch_size=effective_local_batch_size,
        initial_Fin=None,
        n_cls_bins=cls_conf["n_bins"],
        l_min_per_pair=l_min_per_pair,
        l_max_per_pair=l_max_per_pair,
        cls_transform=cls_conf["transform"],
        map_feature_dim=net_conf["network"].get("map_feature_dim", None),
        map_encoder=enc,
    )
    return net


network = build_multires_composite()
cls_in = tf.random.normal((2, 3 * n_side, len(l_min_per_pair)))
out = network((maps, cls_in), training=False)
print(f"composite output: {out.shape}")
assert out.shape == (2, n_output)
n_params_multires = network.count_params()
print(f"multi-res composite params: {n_params_multires:,}")

stage("6. checkpoint save/restore round trip (multi-res lineage)")
ckpt_dir = os.path.join(OUT, "ckpt_roundtrip")
ckpt = tf.train.Checkpoint(network=network)
path = ckpt.write(os.path.join(ckpt_dir, "ckpt"))
network2 = build_multires_composite()
_ = network2((maps, cls_in), training=False)  # build variables
status = tf.train.Checkpoint(network=network2).read(path)
status.assert_existing_objects_matched()

# weight-level equality — immune to the white-noise layer in the smoothing (survey-noise
# emulation, active also at training=False), and catches untracked variables directly
assert len(network.variables) == len(network2.variables)
n_diff = 0
for v1, v2 in zip(network.variables, network2.variables):
    # Keras uniquifies layer names with a global counter, so only compare names modulo the suffix
    base1, base2 = (re.sub(r"_\d+", "", v.name) for v in (v1, v2))
    assert base1 == base2 and v1.shape == v2.shape, (v1.name, v2.name, v1.shape, v2.shape)
    if not np.array_equal(v1.numpy(), v2.numpy()):
        n_diff += 1
        print(f"MISMATCH after restore: {v1.name} {v1.shape}")
assert n_diff == 0, f"{n_diff} variables not restored"
print(f"all {len(network2.variables)} variables identical after restore")

# output-level sanity check: pin the global RNG so both forward passes draw the same white noise.
# Loose tolerance on purpose — the weight comparison above is the exact check; the forward pass is
# GPU-nondeterministic at the 1e-3 level (atomic-add ordering in unsorted_segment_mean / sparse
# matmuls, amplified through the conv stack).
tf.random.set_seed(1234)
out_seeded = network((maps, cls_in), training=False)
tf.random.set_seed(1234)
out2 = network2((maps, cls_in), training=False)
np.testing.assert_allclose(out_seeded.numpy(), out2.numpy(), rtol=0.05, atol=0.02)
print("checkpoint round trip OK (weights identical, outputs equal to GPU-nondeterminism level)")

stage("7. maps-only multi-res encoder (head inside gcnn_post)")
maps_conf = input_output.read_yaml(f"{REPOS}/y3-deep-lss/configs/deepsphere/combined/maps.yaml")
spec_m = NETWORKS["resnet"](**probe_spec_kwargs, **maps_conf["network"]["kwargs"])
encoder_m = ResNetMultiResEncoder(
    smoothing_kwargs=smoothing_kwargs,
    layers=spec_m.get_layers(),
    nside=smooth_nside,
    n_neighbors=maps_conf["network"]["n_neighbors"],
    max_batch_size=maps_conf["dset"]["training"]["grid"]["local_batch_size"],
    input_norm=True,
)
out_m = encoder_m(maps, training=False)
print(f"maps-only output: {out_m.shape}")
assert out_m.shape == (2, n_output)

stage("8. single-resolution regression build (config without smooth_nside)")
net_conf_sr = copy.deepcopy(net_conf)
del net_conf_sr["network"]["smooth_nside"]
smoothing_kwargs_sr = configuration.get_smoothing_kwargs(
    "mutual_info", msfm_conf, dlss_conf, net_conf_sr, dir_base=OUT
)
assert "split_probes" not in smoothing_kwargs_sr
spec_sr = NETWORKS["resnet"](
    out_features=n_output, smoothing_kwargs=smoothing_kwargs_sr, input_norm=True,
    **net_conf_sr["network"]["kwargs"],
)
network_sr = ResNetMapsPlusCLSNetwork(
    conv_layers=spec_sr.get_conv_layers(),
    cls_embedding_layers=get_cls_embedding_layers(cls_conf["embedding_layers"],
                                                  dropout_rate=cls_conf["embedding_dropout_rate"]),
    regression_head_layers=spec_sr.get_head_layers_no_flatten(),
    n_side=smooth_nside,
    tfr_n_side=n_side,
    indices=smooth_indices,
    n_neighbors=net_conf_sr["network"]["n_neighbors"],
    max_batch_size=effective_local_batch_size,
    initial_Fin=n_z_bins,
    n_cls_bins=cls_conf["n_bins"],
    l_min_per_pair=l_min_per_pair,
    l_max_per_pair=l_max_per_pair,
    cls_transform=cls_conf["transform"],
    map_feature_dim=net_conf_sr["network"].get("map_feature_dim", None),
)
network_sr.gcnn.build((effective_local_batch_size, len(smooth_indices), n_z_bins))
out_sr = network_sr((maps, cls_in), training=False)
assert out_sr.shape == (2, n_output)
n_params_sr = network_sr.count_params()
print(f"single-res composite params: {n_params_sr:,}")

stage("9. param accounting")
base_ch = net_conf["network"]["kwargs"]["base_channels"]
n_lens = by_probe["lensing"]["n_channels"]
n_clust = by_probe["clustering"]["n_channels"]
# expected deltas multi-res vs single-res: first PseudoConv Fin (n_lens vs n_z_bins), the two
# injection Denses, and input-norm variables split (n_lens,)+(n_clust,) vs (n_z_bins,) [same total]
d_pseudoconv = (n_lens - n_z_bins) * 4 * base_ch  # Conv1D kernel 4*Fin*Fout (+bias unchanged)
d_proj = (n_clust + 1) * base_ch
d_fuse = (2 * base_ch + 1) * base_ch
expected_delta = d_pseudoconv + d_proj + d_fuse
actual_delta = n_params_multires - n_params_sr
print(f"expected param delta {expected_delta:+,}, actual {actual_delta:+,}")
assert actual_delta == expected_delta, "unexpected parameter difference between the two lineages"

print("\nALL SMOKE TESTS PASSED", flush=True)
