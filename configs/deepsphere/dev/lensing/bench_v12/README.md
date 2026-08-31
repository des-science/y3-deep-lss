# bench_v12 — LENSING, GCNN arms

**The round definition, rationale, shared core, launch plan and scoring rules live in
[`../../combined/bench_v12/README.md`](../../combined/bench_v12/README.md).** Read that first.

> ## !! PARKED -- THIS PROBE IS NOT RUN IN bench_v12 !!  (2026-08-31)
>
> Every `.yaml` here has moved to **`_deferred/`**, which the submit glob skips. The round is
> decided on **`combined` only**, and the winning combination is then transplanted here: move the
> matching file back out of `_deferred/` and run it once (single 12 h job,
> `wall_budget_seconds: 39600`).
>
> Nothing in these files is wrong. The trunk match, the shared core and the one-knob headers are
> all still correct, and the clustering pair carries the widened `base_channels: 128` /
> `base_embed_dim: 64` that keeps the trunk at 512 / 1024. They are parked because they are not
> part of the *measurement*, not because they are stale.
>
> Everything below describes the arms as written, and stays valid for the transplant.

This file holds only what is specific to lensing.

## Probe specifics

- **Geometry:** base_channels 64 / pool_layers 3, trunk 512; the map branch runs at nside 512 throughout, no `smooth_nside`. The residual body lands at nside 16
  (~448 footprint pixels), the same as every other probe — that is what pins the readout.
- **Budget:** 39 600 s of training = **ONE 12 h job**, not a chain. `MAX_RUNS=1`, `RUN_NUM=1`.
  A single job has **no second job in which to correct a mis-sized budget**, so if the rate comes
  in far below projection the cosine never anneals and the eval tail never runs.
- **Anchor already on disk:** `bench_v11_simple` **is** `classic.yaml` (`config_check diff`
  reports *identical*): 125 300 steps at 3.172 it/s sustained, same 39 600 s wall. **Do not re-run it.**

## What lensing needs that combined does not

`convnext.yaml` **has never been run on this probe.** Bare ConvNeXt at the `mean` readout exists
only on combined; every ConvNeXt run on lensing (`bench_v7_full`, `v1`, `v2`,
`bench_v10_mean_std_*`) carries DropPath, attention, or both, and `bench_v11_convnext_mean_std`
carries the `mean_std` readout. So this one job is what closes the block × probe square.

## Rate and the step confound

| arm | it/s | steps at 39 600 s |
|---|---|---|
| `classic` (on disk) | 3.172 **measured** | 125 300 |
| `convnext` | ~3.85 **PROJECTED** | ~152 k |

The projection comes from `bench_v11_convnext_mean_std` on this probe, adjusted for `mean` being
~1.4% cheaper than `mean_std` as measured on combined. **It is not measured for this geometry.**
At ~1.21× the steps and the +7.3%-per-2× elasticity, ConvNeXt starts ~2.3% ahead before the block
does anything — so **a ConvNeXt win under ~2.5% here is not evidence for the block**, while a loss
is unambiguous. Read the realised count from `throughput.json` and correct the table above.

## Files

Same five arms and the same one-knob chain as combined: `classic` → `convnext` →
`convnext_nodrop` → `convnext_nodrop_droppath`, with `classic_nodrop` closing the 2 × 2.
Optimizer is Adam in every arm; weight decay is not tested this round.
Per the launch plan in the combined README, **none of them runs in this round**. They sit in
`_deferred/` until combined has resolved both questions, and then exactly one of them -- the
winning (block, regularization) combination -- is moved back out and run once.
