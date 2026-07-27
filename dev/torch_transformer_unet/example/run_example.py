"""Minimal example: a forward pass through the HEALPix dense-regression U-Net.

This shows how to drive the network for a real HEALPix application through the
wrapper ``HealpixDeepNestedUNet`` (you never touch the nested U-Net core directly).
The setup is a full-sky ``nside=128`` map compressed to ``nside_down=16`` top-level
tokens. There is no training logic — just a single forward pass — to make explicit
which arguments come from the *data* (geometry) and which are the tunable
*hyperparameters* loaded from ``hyperparameters.yaml``.

Run inside the torch env (torch is not importable on the login node):

    uenv start --view=default pytorch/v2.9.1:v2
    source ~/dlss/torch_env/bin/activate
    python run_example.py
"""

import logging
import pathlib
import sys

import healpy as hp
import torch
import yaml

# Put the torch_transformer package on the path (its parent dir, dev/) so this
# script can be run directly from the example/ folder.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from torch_transformer.healpix_deep_unet import HealpixDeepNestedUNet  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def main():
    # Show the wrapper's INFO line describing the derived HEALPix geometry.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # --- Hyperparameters: loaded from the config, passed straight to the wrapper ---
    with open(HERE / "hyperparameters.yaml") as f:
        hparams = yaml.safe_load(f)

    # --- Geometry: fixed by the data, NOT hyperparameters ---
    nside = 128                        # resolution of the input maps
    nside_down = 16                    # top-level-token resolution (the one geometry knob)
    in_channels = 1                    # number of map channels (e.g. one density field)
    num_pixels = hp.nside2npix(nside)  # full sky = 12 * nside**2 = 196_608 pixels

    # Build the network. Geometry is explicit; every tunable knob comes from the YAML.
    # Internally this derives num_nested_levels = order(128) - order(16) = 7 - 4 = 3, so
    # each nside_down pixel becomes one top-level token holding (128/16)**2 = 64 = 4**3
    # fine pixels (three size-4 nested axes).
    model = HealpixDeepNestedUNet(
        num_pixels=num_pixels,
        nside=nside,
        nside_down=nside_down,
        in_channels=in_channels,
        **hparams,
    )
    model.eval()

    # --- Forward pass on a random batch, shaped exactly like a pipeline batch: (B, P, C) ---
    # P is the number of NESTED-ordered HEALPix pixels; here the full sphere, but any
    # whole number of top-level tokens (a partial footprint) is also accepted.
    batch_size = 2
    x = torch.randn(batch_size, num_pixels, in_channels)
    with torch.no_grad():
        y = model(x)  # dense, residual-corrected map (input + learned correction)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"input  : {tuple(x.shape)}  (B, P, C)")
    print(f"output : {tuple(y.shape)}  (same shape; dense correction at nside={nside})")
    print(f"params : {n_params:,}")


if __name__ == "__main__":
    main()
