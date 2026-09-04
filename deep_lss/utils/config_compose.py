# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Resolve a net config that is composed from shared pieces via an ``extends:`` key.

The twelve prod net configs were ~50% duplication: `dset` and `optimization` were byte-identical
in all of them, `training` had exactly two variants, and the `network` encoder block was written
once in each half of every maps/maps+cls pair. That last one drifted -- the 2026-09-02
`base_embed_dim` fix for transformer/clustering landed on one file and not its partner, under a
header in both asserting the `cls:` block was the only difference.

So a config names its bases instead of repeating them::

    extends:
      - maps/shared/dataloader.yaml
      - maps/shared/optimization.yaml
      - maps/shared/budget/1x12h.yaml
      - maps/shared/encoders/transformer/nested.yaml
      - maps/shared/encoders/transformer/per_probe/clustering.yaml
      - maps/shared/cls_branch.yaml   # present only in the maps+cls half of the pair

The contract:

  * Bases apply left to right, and the including file's own keys win over all of them.
  * Dicts merge recursively. Scalars AND LISTS are replaced wholesale, never concatenated --
    `embedding_layers: [512, 512, 512, 512]` must not become eight entries.
  * A base path resolves against the nearest ancestor directory named ``configs``, falling back to
    the including file's own directory; an absolute path is used as-is. The fallback is what keeps
    ad hoc yamls outside `configs/` working, which `submissions/clariden/shared/benchmark_sweep.sh`
    explicitly supports.
  * A cycle, an over-deep chain, or a missing base is an error naming the files involved.
  * **`extends:` is stripped from the result.** run_training.py dumps the resolved net config into
    the run directory and a restore reads it back from there, so a saved config that still carried
    `extends:` would re-resolve on restore and silently pick up any later edit to a shared file.

A file with no ``extends:`` key resolves to itself, which is what lets the 198 dev configs and
every already-saved run `configs.yaml` keep loading unchanged.

This module deliberately imports only `os` and `yaml`. `configuration.py` pulls in numpy and
`msfm.utils`, and `config_check.py` is login-node tooling that has to stay importable without the
TensorFlow environment; both need to compose configs, so the resolver cannot live in either.
"""

import os

import yaml

EXTENDS_KEY = "extends"

# A chain deeper than this is a mistake, not a design. The prod tree uses one level.
MAX_DEPTH = 8


def _read(path):
    """Parse one YAML file, failing with the file name attached rather than a bare parser error."""
    if not os.path.isfile(path):
        raise ValueError(f"{path}: no such config file")
    with open(path, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: not parseable as YAML -- {exc}")
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: parsed as {type(cfg).__name__}, expected a mapping")
    return cfg


def _configs_root(path):
    """The nearest ancestor directory named ``configs``, or None if the file is outside one."""
    node = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.basename(node) == "configs":
            return node
        parent = os.path.dirname(node)
        if parent == node:
            return None
        node = parent


def _resolve_base(base, including_path):
    """Turn one `extends:` entry into an absolute path, relative to the configs root or the file."""
    if os.path.isabs(base):
        return os.path.normpath(base)
    root = _configs_root(including_path) or os.path.dirname(os.path.abspath(including_path))
    return os.path.normpath(os.path.join(root, base))


def merge(base, override):
    """Merge `override` onto `base`, recursing into dicts and replacing everything else."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _extends_list(cfg, path):
    """The `extends:` entries of one parsed config, accepting a bare string for a single base."""
    bases = cfg.get(EXTENDS_KEY)
    if bases is None:
        return []
    if isinstance(bases, str):
        return [bases]
    if not isinstance(bases, list) or not all(isinstance(b, str) for b in bases):
        raise ValueError(f"{path}: {EXTENDS_KEY} must be a path or a list of paths, got {bases!r}")
    return bases


def _load(path, chain):
    """Resolve one config against its bases; `chain` is the include stack, for cycles and depth."""
    path = os.path.normpath(os.path.abspath(path))
    if path in chain:
        cycle = " -> ".join(chain + [path])
        raise ValueError(f"{EXTENDS_KEY} cycle: {cycle}")
    if len(chain) >= MAX_DEPTH:
        raise ValueError(f"{path}: {EXTENDS_KEY} nested more than {MAX_DEPTH} deep -- {' -> '.join(chain)}")

    cfg = _read(path)
    resolved = {}
    for base in _extends_list(cfg, path):
        base_path = _resolve_base(base, path)
        if not os.path.isfile(base_path):
            raise ValueError(f"{path}: {EXTENDS_KEY} '{base}' resolves to {base_path}, which does not exist")
        resolved = merge(resolved, _load(base_path, chain + [path]))

    cfg.pop(EXTENDS_KEY, None)
    return merge(resolved, cfg)


def load_composed(path):
    """Load a net config, resolving any ``extends:`` chain into a single flat mapping.

    Args:
        path (str): path to the config.

    Returns:
        dict: the resolved config, with no ``extends:`` key. Identical to what ``yaml.safe_load``
        would return for a config that does not use ``extends:``.
    """
    return _load(path, [])
