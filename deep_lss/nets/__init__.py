from .resnet import ResNetLayers
from .transformer import ViTLayers, GTLayers
from .one_d_conv import OneDConvLayers
from .maps_plus_cls_network import MapsPlusCLSNetwork

NETWORKS = {
    "resnet": ResNetLayers,
    "vision_transformer": ViTLayers,
    "graph_transformer": GTLayers,
    "one_d_conv": OneDConvLayers,
}
