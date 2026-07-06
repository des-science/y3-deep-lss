from .resnet import ResNetLayers
from .transformer import ViTLayers, GTLayers
from .one_d_conv import OneDConvLayers
from .maps_plus_cls_network import MapsPlusCLSNetwork
from .transformer_networks import HealpixTransformerNetwork, TransformerMapsPlusCLSNetwork

NETWORKS = {
    "resnet": ResNetLayers,
    "vision_transformer": ViTLayers,
    "graph_transformer": GTLayers,
    "one_d_conv": OneDConvLayers,
}

# Network names that are built as a pre-assembled tf.keras.Model (smoothing + nested
# tokenizer + transformer) and passed to BaseModel directly with n_side=None, rather than
# through the layer-list / HealpyGCNN path used by the entries in NETWORKS above.
TRANSFORMER_NETWORKS = frozenset({"nested_transformer"})
