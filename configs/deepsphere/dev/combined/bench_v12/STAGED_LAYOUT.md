# The staged all-ConvNeXt layout

**Layout reference for bench_v12's `staged_nodrop*` arms.** Drafted as a standalone round,
`bench_v13`, and folded into bench_v12 on 2026-08-31 before any job started — see that round's
README under "The bench_v13 fold". This file is the design record: the stage table, the cost, the
smoke test, and what the layout gives up.

**The question it answers:** does the GCNN still perform when its two bare Chebyshev convolutions
are replaced by ConvNeXt blocks, so the whole network is nothing but strided pseudo-convs and
ConvNeXt blocks?

The motivation is the paper figure (`paper_2_tex/figures/networks.tex`) — a GCNN with one block
type throughout is a far simpler thing to draw beside the transformer than one with a residual
body sitting on top of two conv+LayerNorm slots. The expectation is a **wash**, and a wash is a
pass: this is a change adopted for consistency, not for FoM.

The two arms that carry it are `staged_nodrop.yaml` (drop_path 0.0) and
`staged_nodrop_droppath.yaml` (0.1). Both live in this directory.

## The layout

Verified by simulating the layer stack (`nside`/width bookkeeping mirroring `HealpyGCNN`'s):

| | layer | nside | width |
|---|---|---|---|
| stem | PseudoConv | 512 → 256 | n_z → 64 |
| | PseudoConv | 256 → 128 | 64 → 128 |
| | PseudoConv | 128 → 64 | 128 → 256 |
| stage 1 | **ConvNeXt × 1** | 64 | 256 |
| | PseudoConv | 64 → 32 | 256 → **512** |
| stage 2 | **ConvNeXt × 1** | 32 | 512 |
| | PseudoConv | 32 → 16 | 512 |
| trunk | **ConvNeXt × 5** | 16 | 512 |
| readout | mean pool | | 512 |

Against the legacy layout (`_deferred/convnext_nodrop.yaml`): **5 blocks → 7, bare Chebyshev
convs 2 → 0**, and the two interleaved `LayerNorm`s disappear because each block opens with its
own.

**Both invariants hold.** The first graph convolution is still at nside 64, and the trunk is still
nside 16 × 512. The multi-resolution injection split for clustering is also unchanged — the layer
walk still splits after layer 0 with `Fin_at_split=64`, so `ResNetMultiResEncoder` sees exactly
what it saw before.

## Why the widening moves into the PseudoConv

`Healpy_ConvNeXtLayer` is channel-preserving and *deliberately* exposes no `Fout`, so
`HealpyGCNN` keeps the running channel count across it. A block therefore cannot carry the
256 → 512 step. `HealpyPseudoConv` can (it is a `Conv1D(Fout, kernel=4, stride=4)`), and putting
it there is ConvNeXt's own design: downsample layers change the width, blocks never do.

That also fixes a real trap in the legacy loop. `pool_layers` widens **after** appending, so the
three pooling stages leave the tensor at 256 and it is the *first Chebyshev* that widens to 512 —
the config reads as though the pooling stages already reached the trunk width. The staged form
states each width outright, so there is nothing to misread.

## Cost — MEASURED (job 3244154, 2026-08-31)

Both configs built through the real `run_training` path on one GPU, batch 16, synthetic batches
(`benchmark_resnet.py --single`):

| | params | peak GB | step ms | throughput |
|---|---|---|---|---|
| legacy layout (`_deferred/convnext_nodrop`) | 15.952 M | 7.99 | 239.8 | 66.7 |
| `staged` | 16.091 M | 10.8 | 254.0 | 63.0 |
| | **1.01×** | **1.35×** | **1.06×** | |

**Parameters landed within 0.5 % of prediction** (+0.139 M measured against +0.13 M computed).
Step time came in at 1.06× against a 1.04× prediction, which was for graph-conv work alone rather
than the whole step. The two effects nearly cancel as expected: the block at nside 64 runs at
C=256 and is cheaper than the Chebyshev it replaces, while stage 2's block is dearer than its Cheb
because the inverted bottleneck outweighs a single conv.

**Peak memory is the one thing the arithmetic missed: 8.0 -> 10.8 GB, +35 %.** The MLP expansion
to 4C inside the nside-64 block materializes a wide activation at 16x the pixel count of the trunk,
which parameter counting does not see. Harmless here (the GCNN is not memory-bound -- 10.8 of
~120 GB, and the ~85 GB NCCL band that constrains the transformer does not apply), but it is a
real 1.35x and it would bite first if the batch size were ever raised.

Because the budget is `n_steps: auto`, a 6 % slower step buys ~6 % fewer steps, i.e. ~0.4 % of FoM
at the +7.3 %-per-2× elasticity. Well inside the 0.049 floor.

## It is a real architecture change

Same nside, same trunk, near-identical cost — but not a re-drawing:

- stage 1's spatial mixing becomes depthwise + pointwise instead of a full ChebK
- stage 2 gains an inverted bottleneck
- both gain a residual path they did not have

So it is measured, never adopted on the arithmetic — see the scoring note below for what the
fold did to that measurement's attribution.

## What it costs elsewhere

**`conv_type` becomes unreachable in the staged layout.** The ConvNeXt block is Chebyshev-only
(no depthwise Monomial/Bernstein path), and there are no bare conv slots left for a basis choice.
In practice it is already dead — `bernstein` appears only in `bench_v4`, and bench_v8's README
records the stages as settled at `cheby` — and the legacy path is untouched, so every existing
config still builds. But a *new* run cannot select a different polynomial basis.

`poly_degree` still applies: it is the ChebK order of each block's depthwise conv.

## The launch blocker, and how the fold removed it

~~**Sync the regularization keys to bench_v12's winner.**~~ Resolved by the fold. As a standalone
round this file had to wait for bench_v12 to name a regularization, or else move two knobs at once.
Folded in, **both** regularizations run staged — `staged_nodrop` and `staged_nodrop_droppath` are
one knob apart — so there is nothing left to sync and nothing left to wait for.

~~The staged code path has never been executed.~~ **Smoke-tested 2026-08-31, job 3244154: it
builds, status OK.** The log confirms the intended stack directly --

```
Staged GCNN layout: 5 nside halvings, widths [64, 128, 256, 512, 512],
                    blocks [0, 0, 1, 1, 5], trunk width 512, 7 convnext blocks total
```

-- and the multi-resolution seam is unchanged from the parent: both report clustering *"injected at
64 channels, fused to 64"*. The layer count after the fusion drops 13 -> 11, which is the two
`LayerNorm`s the staged layout removes.

The smoke test also flushed out a **pre-existing bug in `benchmark_resnet.py`**, unrelated to this
arm: it called `get_optimizer` without a `WallClockBudget`, so `n_steps: auto` reached the cosine
schedule as the string `"auto"` and raised `TypeError`. That made every `prod/` and bench_v8+
config unbenchmarkable. Fixed in the same commit by substituting a concrete step count (the
benchmark measures step time and memory; nothing there depends on the LR schedule).

## The arms

| file | knob vs | keys | jobs |
|---|---|---|---|
| `staged_nodrop.yaml` | `classic_nodrop.yaml` | 9 — a **package**, block + layout | 2 (`afterany` chain) |
| `staged_nodrop_droppath.yaml` | `staged_nodrop.yaml` | 1 (`drop_path_rate`) | 2 (`afterany` chain) |

`base_channels`/`pool_layers`/`conv_layers`/`residual_layers` out, `stage_widths`/`stage_blocks`
in, plus the ConvNeXt block's inseparable three. `config_check diff` reports `MORE THAN ONE KNOB`
on the first row and that is expected — it is one layout, not six choices, exactly as for the
ConvNeXt block's three-key package.

## Scoring

`compare-runs` paired FoM. Floor **0.049**; a tie is the expected and acceptable outcome, and it
argues for **adopting** the staged arm rather than dropping it — the usual "a tie goes to the
simpler net" rule points here, because this *is* the simpler net (one block type, two fewer knob
families, no hidden widening).

**What the fold cost this measurement.** The intended reference was
`bench_v12_convnext_nodrop` — the same block and regularization in the legacy layout — which would
have isolated the layout to one knob. That run will not exist. The reference is now
`bench_v12_classic_nodrop`, so a margin reads as **block + layout together**. Two mitigations, and
the honest limit of each:

- bench_v11 measured the block alone at **~+0.7%** once its 1.18× step confound is paid back — a
  wash. So most of any margin here is attributable to the layout by elimination, not by design.
- If the staged arm **loses outside the floor**, `_deferred/convnext_nodrop.yaml` is ready to run
  as the tiebreak (2 further jobs) and restores the clean one-knob layout contrast. Only spend it
  then.

**Q3 `coverage` is required**: the summary dimensionality is unchanged at 512, but the readout's
upstream path is not, and a pathological summary defeating the flow is the one failure mode Q1
cannot see.

## Not in this round

**Distributing blocks across resolutions.** Real ConvNeXt puts blocks at every stage (3,3,9,3);
this arm keeps the current 1/1/5 split so it stays one knob from its parent. Rebalancing
`stage_blocks` is now a one-line config change and is the natural bench_v14 if this lands.
