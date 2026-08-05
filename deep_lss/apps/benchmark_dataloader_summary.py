# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Author: Arne Thomsen

Pretty-print the JSONL produced by benchmark_dataloader.py as one table per probe.

    python benchmark_dataloader_summary.py /path/to/results.jsonl
"""

import sys
import json
from collections import defaultdict

COLS = [
    ("label", "config", 20),
    ("local_batch_size", "batch", 6),
    ("n_readers", "read", 5),
    ("n_prefetch", "pref", 5),
    ("n_workers", "work", 5),
    ("examples_shuffle_buffer", "eshuf", 6),
    ("examples_per_s_mean", "ex/s", 8),
    ("mb_per_s_mean", "MB/s", 8),
    ("max_steps_per_s_4gpu", "step/s", 7),
    ("median_batch_ms", "bt_med", 7),
    ("p95_batch_ms", "bt_p95", 7),
    ("rss_plateau_gb", "rss_GB", 7),
    ("vmhwm_peak_gb", "peak_GB", 8),
]


def main():
    path = sys.argv[1]
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_probe = defaultdict(list)
    for r in rows:
        probe = str(r.get("label", "")).split("/")[0]
        by_probe[probe].append(r)

    for probe in sorted(by_probe):
        rs = by_probe[probe]
        geo = rs[0]
        print(
            f"\n===== {probe}  (n_dv_pix={geo['n_dv_pix']}, n_channels={geo['n_channels']}, "
            f"{geo['mb_per_example']} MB/example, downsample_nside={geo['downsample_nside']}) ====="
        )
        header = "  ".join(f"{short:>{w}}" for _, short, w in COLS)
        print(header)
        print("-" * len(header))
        for r in rs:
            cells = []
            for key, _, w in COLS:
                v = r.get(key, "")
                if key == "label":
                    v = str(v).split("/", 1)[-1]
                cells.append(f"{str(v):>{w}}")
            print("  ".join(cells))


if __name__ == "__main__":
    main()
