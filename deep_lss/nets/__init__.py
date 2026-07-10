from .encoders.maps.resnet import ResNetLayers
from .encoders.maps.vit import ViTLayers, GTLayers
from .encoders.maps.one_d_conv import OneDConvLayers
from .encoders.maps.transformer.network import HealpixTransformerNetwork
from .composite.resnet_maps_plus_cls import ResNetMapsPlusCLSNetwork
from .composite.transformer_maps_plus_cls import TransformerMapsPlusCLSNetwork

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
