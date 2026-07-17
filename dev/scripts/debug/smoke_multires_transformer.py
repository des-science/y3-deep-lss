"""Smoke test for the multi-resolution transformer path (HealpixMultiResMapEncoder).

Companion to smoke_multires_gcnn.py, added when the shared MultiResEncoderMixin was factored out
of the two multi-res encoders: builds the combined transformer maps+cls and maps-only networks
exactly as run_training.py does (configs/transformer/combined/{maps+cls,maps}.yaml with
smooth_nside: {clustering: 256}), exercises the input-norm measurement interface, and round-trips
a checkpoint. Runs fp32 with jit_compile_body off — the mixin plumbing is precision-agnostic.
Run inside the tensorflow container on a compute node (see smoke_multires_transformer.sh).
"""

import os
import re

import numpy as np
import tensorflow as tf

REPOS = "/users/athomsen/dlss/repos"
OUT = "/iopsstor/scratch/cscs/athomsen/deep_lss/runs/smoke_multires_gcnn"

from msfm.utils import input_output
from deep_lss.utils import configuration
from deep_lss.nets.composite.transformer_maps_plus_cls import TransformerMapsPlusCLSNetwork
from deep_lss.nets.encoders.maps.transformer.network import (
    HealpixMultiResMapEncoder,
    HealpixTransformerNetwork,
)
from deep_lss.nets.heads.regression_head import get_regression_head
from deep_lss.nets.layers.cls.embedding import get_cls_embedding_layers
from deep_lss.nets.layers.maps.input_normalization import compute_input_norm_stats


def stage(msg):
    print(f"\n===== {msg} =====", flush=True)


msfm_conf = input_output.read_yaml(f"{REPOS}/multiprobe-simulation-forward-model/configs/v17/baseline.yaml")
dlss_conf = configuration.read_split_configs(
    f"{REPOS}/y3-deep-lss/configs/probes/combined_nla.yaml",
    f"{REPOS}/y3-deep-lss/configs/scales/8wl,32gc.yaml",
)
net_conf = input_output.read_yaml(f"{REPOS}/y3-deep-lss/configs/transformer/combined/maps+cls.yaml")

n_side = msfm_conf["analysis"]["n_side"]
n_z_bins = len(msfm_conf["survey"]["metacal"]["z_bins"]) + len(msfm_conf["survey"]["maglim"]["z_bins"])
n_output = 12  # arbitrary summary dim for the smoke test

stage("1. smoothing spec")
smoothing_kwargs = configuration.get_smoothing_kwargs(
    "mutual_info", msfm_conf, dlss_conf, net_conf, dir_base=OUT
)
assert "split_probes" in smoothing_kwargs, "expected a split_probes spec for the combined multi-res config"
by_probe = {s["probe"]: s for s in smoothing_kwargs["split_probes"]}
assert by_probe["lensing"]["smoothing_kwargs"]["nside"] == 512
assert by_probe["clustering"]["smoothing_kwargs"]["nside"] == 256
smooth_nside, smooth_indices, _ = configuration.resolve_smooth_nside(net_conf, dlss_conf, msfm_conf)
print(f"split spec OK, network at nside {smooth_nside} with {len(smooth_indices)} px")

stage("2. maps+cls composite build + forward (run_training construction)")
cls_conf = net_conf["network"]["cls"]
_, l_min_per_pair, l_max_per_pair = configuration.get_cls_bounds_per_pair(msfm_conf, dlss_conf)
head_conf = net_conf["network"].get("head", {}) or {}


def build_composite():
    return TransformerMapsPlusCLSNetwork(
        smoothing_kwargs=smoothing_kwargs,
        smooth_indices=smooth_indices,
        nside=smooth_nside,
        token_nside=net_conf["network"]["token_nside"],
        in_channels=n_z_bins,
        map_feature_dim=net_conf["network"]["map_feature_dim"],
        transformer_kwargs=net_conf["network"]["kwargs"],
        tfr_n_side=n_side,
        n_cls_bins=cls_conf["n_bins"],
        l_min_per_pair=l_min_per_pair,
        l_max_per_pair=l_max_per_pair,
        cls_embedding_layers=get_cls_embedding_layers(
            cls_conf["embedding_layers"], dropout_rate=cls_conf["embedding_dropout_rate"]
        ),
        regression_head_layers=get_regression_head(
            out_features=n_output,
            head_type="dense",
            dense_layers=None,
            dropout_rate=head_conf.get("dropout_rate", None),
        )[1:],
        jit_compile_body=False,  # keep the smoke fast; the mixin plumbing is XLA-agnostic
        cls_transform=cls_conf["transform"],
        input_norm=True,
    )


network = build_composite()
assert isinstance(network.map_encoder, HealpixMultiResMapEncoder)
maps = tf.random.normal((2, len(smooth_indices), n_z_bins))
cls_in = tf.random.normal((2, 3 * n_side, len(l_min_per_pair)))
out = network((maps, cls_in), training=False)
print(f"composite output: {out.shape}, params: {network.count_params():,}")
assert out.shape == (2, n_output)

stage("3. input-norm measurement path (mixin interface)")
fake_dset = tf.data.Dataset.from_tensors((tf.random.normal((4, len(smooth_indices), n_z_bins)),)).repeat(3)
enc = network.map_encoder
stats = compute_input_norm_stats(enc.smooth_groups, fake_dset, n_batches=3, masks=enc.masks)
assert len(stats) == 2, f"expected 2 input-norm groups, got {len(stats)}"
assert stats[0][0].shape == (by_probe["lensing"]["n_channels"],)
assert stats[1][0].shape == (by_probe["clustering"]["n_channels"],)
enc.load_input_norm_stats(stats)
out = network((maps, cls_in), training=False)
print(f"stats loaded, forward OK: {out.shape}")

stage("4. checkpoint save/restore round trip")
ckpt_dir = os.path.join(OUT, "ckpt_roundtrip_transformer")
path = tf.train.Checkpoint(network=network).write(os.path.join(ckpt_dir, "ckpt"))
network2 = build_composite()
_ = network2((maps, cls_in), training=False)  # build variables
status = tf.train.Checkpoint(network=network2).read(path)
status.assert_existing_objects_matched()

# weight-level equality — immune to the white-noise layer in the smoothing (survey-noise
# emulation, active also at training=False); names compared modulo Keras' global uniquifier
assert len(network.variables) == len(network2.variables)
n_diff = 0
for v1, v2 in zip(network.variables, network2.variables):
    base1, base2 = (re.sub(r"_\d+", "", v.name) for v in (v1, v2))
    assert base1 == base2 and v1.shape == v2.shape, (v1.name, v2.name, v1.shape, v2.shape)
    if not np.array_equal(v1.numpy(), v2.numpy()):
        n_diff += 1
        print(f"MISMATCH after restore: {v1.name} {v1.shape}")
assert n_diff == 0, f"{n_diff} variables not restored"
print(f"all {len(network2.variables)} variables identical after restore")

stage("5. maps-only multi-res transformer")
maps_conf = input_output.read_yaml(f"{REPOS}/y3-deep-lss/configs/transformer/combined/maps.yaml")
network_m = HealpixTransformerNetwork(
    smoothing_kwargs=smoothing_kwargs,
    smooth_indices=smooth_indices,
    nside=smooth_nside,
    token_nside=maps_conf["network"]["token_nside"],
    in_channels=n_z_bins,
    num_outputs=n_output,
    transformer_kwargs=maps_conf["network"]["kwargs"],
    jit_compile_body=False,
    head_dropout_rate=(maps_conf["network"].get("head", {}) or {}).get("dropout_rate", None),
    input_norm=True,
)
assert isinstance(network_m.map_encoder, HealpixMultiResMapEncoder)
out_m = network_m(maps, training=False)
print(f"maps-only output: {out_m.shape}")
assert out_m.shape == (2, n_output)

print("\nALL SMOKE TESTS PASSED", flush=True)
