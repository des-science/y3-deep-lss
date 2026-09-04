# bench_v5 — deferred (shelved, not deleted)

These ConvNeXt-body variants are kept for later but are **excluded from the submit/benchmark glob**
(`bench_v5/*.yaml` does not recurse into subdirs). They were benchmarked (step_ms recorded in their
headers and in the v5 `benchmark_results.jsonl`) but are **not** in the recommended round-1 submit set.

Why deferred: each re-tests a body-capacity axis that bench_v4 already found flat or bad on the classic
residual block (see `dev/notes/bench_v4_deepsphere_combined_2026-07-23.md`). The bench_v4 diagnosis is
that the combined-probe gap is a **readout / cross-probe-mixing** problem, not a body-capacity one, so
these are lower expected value than `convnext` (the modern-block baseline, DropPath off),
`attn_body` (cross-probe attention), and the `2x/` full-budget rematches. NOTE: the no-DropPath default
(2026-07-24) means these inherit `drop_path_rate: 0.0`; `convnext_droppath.yaml` is the single DropPath-on
config.

| config | axis | bench_v4 precedent for deferring |
|---|---|---|
| `convnext_deep.yaml` (res 10) | depth | v4 `deep_trunk` overfit grid->DES (grid 2238 / DES 1321) |
| `convnext_wide.yaml` (base 64) | width | v4 said capacity is not the lever (`w64` barely moved); step-starved here (239 ms -> 120k) |
| `convnext_bigk.yaml` (poly 8) | spectral reach | v4 `poly8` lost; cheap-to-test != likely-to-help |
| `convnext_poolsplit.yaml` (conv @ nside 256, base 64) | fine-res real conv | v4 `graph_unet_256` was the **worst** variant (grid 1518); slowest here (355 ms -> 80k) |
| `injection_conv.yaml` (1 cheby conv @ nside 256 on the fused stream) | fine-res real conv | same axis as `graph_unet_256` (worst v4 body variant); 351.9 ms -> only 80k steps at 1x12 h = too step-starved to be useful (user call 2026-07-24) |

When to revive: if `convnext` beats the `bench_v5/default` (pool_head) anchor at equal 1x12 h budget,
promote the ConvNeXt block and *then* these capacity axes become worth a second look on the winning
block. n_steps in each is sized for the ORIGINAL 1x12 h benchmark; re-check before submitting.
