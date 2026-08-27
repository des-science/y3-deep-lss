from .encoders.maps.gcnn.resnet import ResNetLayers

# ResNetMultiResEncoder is intentionally NOT in any registry below: it is selected implicitly by
# run_training.py when a per-probe (split_probes) smoothing spec is present, and built from the
# `resnet` spec's layer lists — not chosen by network.name.
from .encoders.maps.gcnn.resnet_multires import ResNetMultiResEncoder
from .encoders.maps.legacy.vit import ViTLayers, GTLayers
from .encoders.maps.legacy.one_d_conv import OneDConvLayers
from .composite.resnet_summary import ResNetSummaryNetwork
from .composite.transformer_summary import TransformerSummaryNetwork
from .encoders.cls.mlp import MultiLayerPerceptron
from .encoders.cls.cnn import ClsConv1D
from .encoders.cls.transformer import ClsTransformer

NETWORKS = {
    "resnet": ResNetLayers,
    "vision_transformer": ViTLayers,
    "graph_transformer": GTLayers,
    "one_d_conv": OneDConvLayers,
}

# Cls (binned power-spectrum) summary encoders, selected via the Cls config's `network.name`
# (see run_cls_training+evaluation.py). All are pre-built tf.keras.Models with the same
# call(inputs, training) -> (B, n_summary) contract on a flat (B, n_cls) input, so they are
# drop-in swappable in GridLossModel (n_side=None). "mlp" is the default / backward-compatible
# path; "cls_cnn" and "cls_transformer" reshape the flat vector to (bins, pairs) internally.
CLS_NETWORKS = {
    "mlp": MultiLayerPerceptron,
    "cls_cnn": ClsConv1D,
    "cls_transformer": ClsTransformer,
}

# Network names built as a pre-assembled tf.keras.Model (smoothing + nested tokenizer +
# transformer) and passed to BaseModel directly with n_side=None, rather than through the
# layer-list / HealpyGCNN path. "resnet" is also pre-assembled now (ResNetSummaryNetwork owns
# the readout on both paths); only the legacy NETWORKS entries still use the layer-list path.
TRANSFORMER_NETWORKS = frozenset({"nested_transformer"})
