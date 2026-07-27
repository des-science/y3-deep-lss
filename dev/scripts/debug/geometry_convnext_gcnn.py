"""Geometry/build check for the opt-in ConvNeXt residual body of the DeepSphere ResNet map encoder.

Test 2 of the bench_v5 ConvNeXt plan. Run on a compute node inside the tensorflow container
(``srun --environment=tensorflow --gpu-bind=none python geometry_convnext_gcnn.py``). Confirms:

  1. residual_block_type="residual" (default) reproduces the classic body exactly — the conv-layer
     snapshot is byte-identical in type/order to the pre-change code (all Healpy_ResidualLayer,
     none is a Healpy_ConvNeXtLayer);
  2. residual_block_type="convnext" swaps the body to Healpy_ConvNeXtLayer, which is channel-
     preserving (no Fout attribute) and carries the depthwise [C, K] kernel;
  3. both specs build into a real HealpyGCNN and run a forward pass to the summary dim without shape
     errors, and the ConvNeXt body does not change the channel count across its blocks;
  4. the depthwise body kernel grows exactly linearly in the polynomial order K (slope C per block,
     no C^2 term) — the concrete sense in which "large K is cheap" holds for the block.
"""

import numpy as np
import healpy as hp
import tensorflow as tf

from deepsphere import HealpyGCNN
from deep_lss.nets.encoders.maps.gcnn.resnet import ResNetLayers


def stage(msg):
    print(f"\n===== {msg} =====", flush=True)


# geometry matches configs/deepsphere/combined/{maps+cls,bench_v5/convnext}.yaml exactly
KW = dict(base_channels=32, pool_layers=3, conv_layers=2, residual_layers=5, poly_degree=5)
N_OUT = 12
NSIDE_IN = 256  # small enough for a quick forward; body runs at the pooled nside


def body_layers(spec):
    """The residual-body block instances in a spec's conv-layer snapshot."""
    return [
        layer
        for layer in spec.get_conv_layers()
        if type(layer).__name__ in ("Healpy_ResidualLayer", "Healpy_ConvNeXtLayer")
    ]


stage("1. default (residual) body is unchanged")
spec_res = ResNetLayers(out_features=N_OUT, **KW)
res_body = body_layers(spec_res)
assert len(res_body) == KW["residual_layers"], (len(res_body), KW["residual_layers"])
assert all(type(b).__name__ == "Healpy_ResidualLayer" for b in res_body), [type(b).__name__ for b in res_body]
assert not any(type(b).__name__ == "Healpy_ConvNeXtLayer" for b in spec_res.get_conv_layers())
# no ConvNeXt-only kwargs leak into the residual spec's block objects
assert not hasattr(res_body[0], "mlp_ratio")
print(f"residual body: {len(res_body)} x Healpy_ResidualLayer, no ConvNeXt layer present -> unchanged")

stage("2. convnext body swaps to Healpy_ConvNeXtLayer (channel-preserving, depthwise [C, K])")
spec_cnx = ResNetLayers(
    out_features=N_OUT,
    residual_block_type="convnext",
    mlp_ratio=4,
    layer_scale_init=1e-6,
    drop_path_rate=0.1,
    **KW,
)
cnx_body = body_layers(spec_cnx)
assert len(cnx_body) == KW["residual_layers"], (len(cnx_body), KW["residual_layers"])
assert all(type(b).__name__ == "Healpy_ConvNeXtLayer" for b in cnx_body), [type(b).__name__ for b in cnx_body]
# channel-preserving spec: no Fout attribute -> stays None for split_layers_at_nside / Fin bookkeeping
assert all(getattr(b, "Fout", None) is None for b in cnx_body), "ConvNeXt block must not expose Fout"
print(f"convnext body: {len(cnx_body)} x Healpy_ConvNeXtLayer, all Fout=None (channel-preserving)")

# the pooling stages are identical between the two specs (only the body changed)
res_types = [type(l).__name__ for l in spec_res.get_conv_layers() if type(l).__name__ != "Healpy_ResidualLayer"]
cnx_types = [type(l).__name__ for l in spec_cnx.get_conv_layers() if type(l).__name__ != "Healpy_ConvNeXtLayer"]
assert res_types == cnx_types, (res_types, cnx_types)
print(f"pooling/stem stages identical across both bodies: {cnx_types}")


def build_gcnn(spec):
    n_pix = hp.nside2npix(NSIDE_IN)
    indices = np.arange(n_pix)
    tf.random.set_seed(11)
    model = HealpyGCNN(nside=NSIDE_IN, indices=indices, layers=spec.get_layers())
    model.build(input_shape=(2, n_pix, 1))
    return model, n_pix


stage("3. both specs build into a HealpyGCNN and run a forward pass")
np.random.seed(7)
for label, spec in (("residual", spec_res), ("convnext", spec_cnx)):
    model, n_pix = build_gcnn(spec)
    m_in = np.random.normal(size=[2, n_pix, 1]).astype(np.float32)
    out = model(m_in, training=False)
    assert out.shape == (2, N_OUT), (label, out.shape)
    n_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    print(f"{label:9s}: forward OK -> {tuple(out.shape)}, trainable params {n_params:,}")

stage("4. convnext body preserves the channel count across its blocks + depthwise kernel is [C, K]")
model_cnx, _ = build_gcnn(spec_cnx)
built_body = [layer for layer in model_cnx.layers_use if type(layer).__name__ == "GCNN_ConvNeXtLayer"]
assert len(built_body) == KW["residual_layers"], (len(built_body), KW["residual_layers"])
# channel count entering the body = base_channels * channel_multiplier^(#widening stages);
# pool_widen defaults True over pool_layers + conv_layers widening -> whatever it is, every block
# must keep it constant. Read it off the first block's depthwise kernel [C, K] and check all match.
c0, k0 = (int(x) for x in built_body[0].dwconv.kernel.shape)
assert k0 == KW["poly_degree"], (k0, KW["poly_degree"])
for b in built_body:
    c, k = (int(x) for x in b.dwconv.kernel.shape)
    assert (c, k) == (c0, k0), ("channel count not preserved across body", (c, k), (c0, k0))
print(f"all {len(built_body)} ConvNeXt blocks share depthwise kernel shape (C, K) = ({c0}, {k0})")

stage("5. large K is cheap in the DEPTHWISE BODY (bigk sanity)")
# poly_degree drives THREE things: (a) the depthwise body kernels [C, K] -> cheap (C*K per block,
# no C^2), and (b) the conv_layers pooling Chebyshev convs + (c) the graph-conv regression head,
# both of which are full [K*Fin, Fout] kernels and grow with K the expensive way. "Cheap under
# depthwise" is a statement about the BODY only, so measure the body kernels directly rather than
# the whole-network param count (which the pooling convs + head dominate).
def body_dwconv_params(model):
    return int(
        np.sum(
            [
                np.prod(layer.dwconv.kernel.shape)
                for layer in model.layers_use
                if type(layer).__name__ == "GCNN_ConvNeXtLayer"
            ]
        )
    )


spec_bigk = ResNetLayers(
    out_features=N_OUT, residual_block_type="convnext", mlp_ratio=4, layer_scale_init=1e-6, **{**KW, "poly_degree": 16}
)
model_k5, n_pix = build_gcnn(spec_cnx)
model_k16, _ = build_gcnn(spec_bigk)
body_k5 = body_dwconv_params(model_k5)
body_k16 = body_dwconv_params(model_k16)
c_big = int([l for l in model_k16.layers_use if type(l).__name__ == "GCNN_ConvNeXtLayer"][0].dwconv.kernel.shape[0])
expected_body_delta = KW["residual_layers"] * c_big * (16 - 5)  # C*(K2-K1) per block, no C^2 term
p_k5 = int(np.sum([np.prod(v.shape) for v in model_k5.trainable_variables]))
p_k16 = int(np.sum([np.prod(v.shape) for v in model_k16.trainable_variables]))
print(f"depthwise body kernels: K=5 {body_k5:,} -> K=16 {body_k16:,} (delta {body_k16 - body_k5:+,}); "
      f"expected exactly {expected_body_delta:+,}")
print(f"(whole-network params K=5 {p_k5:,} -> K=16 {p_k16:,} (delta {p_k16 - p_k5:+,}) — pooling convs + head, "
      f"NOT the body, dominate this)")
# exact equality proves the body kernel grows linearly in K with slope C per block (no C^2 term);
# the pointwise MLP (the bulk of each block) is untouched by K, so this is the whole body-side cost.
assert body_k16 - body_k5 == expected_body_delta, "depthwise body-kernel growth must be exactly residual_layers*C*(K2-K1)"

print("\nALL GEOMETRY CHECKS PASSED", flush=True)
