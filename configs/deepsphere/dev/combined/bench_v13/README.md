# bench_v13 — COMBINED, the trunk cleanup

**One arm, one question:** does the GCNN still perform when its two bare Chebyshev convolutions
are replaced by ConvNeXt blocks, so the whole network is nothing but strided pseudo-convs and
ConvNeXt blocks?

The motivation is the paper figure (`paper_2_tex/figures/networks.tex`) — a GCNN with one block
type throughout is a far simpler thing to draw beside the transformer than one with a residual
body sitting on top of two conv+LayerNorm slots. The expectation is a **wash**, and a wash is a
pass: this is a change adopted for consistency, not for FoM.

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

Against `bench_v12/convnext_nodrop`: **5 blocks → 7, bare Chebyshev convs 2 → 0**, and the two
interleaved `LayerNorm`s disappear because each block opens with its own.

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

## Cost — near neutral

Computed from the layer definitions, **not measured**:

| | params | graph-conv work |
|---|---|---|
| `bench_v12/convnext_nodrop` | 14.76 M | 26.2 (arb.) |
| `staged` | 14.89 M | 27.3 |
| | **1.01×** | **1.04×** |

The two effects nearly cancel: the block at nside 64 runs at C=256 (8.4 against the Chebyshev's
10.5, and one depthwise conv instead of a full K·C² one), while stage 2's block costs 8.4 against
5.2 because the inverted bottleneck is heavier than a single Cheb.

Because the budget is `n_steps: auto`, a ~4 % slower step buys ~4 % fewer steps, i.e. ~0.3 % of
FoM at the +7.3 %-per-2× elasticity. Well inside the 0.049 floor.

## It is a real architecture change

Same nside, same trunk, near-identical cost — but not a re-drawing:

- stage 1's spatial mixing becomes depthwise + pointwise instead of a full ChebK
- stage 2 gains an inverted bottleneck
- both gain a residual path they did not have

So it is measured as one knob off `bench_v12_convnext_nodrop`, never adopted on the arithmetic.

## What it costs elsewhere

**`conv_type` becomes unreachable in the staged layout.** The ConvNeXt block is Chebyshev-only
(no depthwise Monomial/Bernstein path), and there are no bare conv slots left for a basis choice.
In practice it is already dead — `bernstein` appears only in `bench_v4`, and bench_v8's README
records the stages as settled at `cheby` — and the legacy path is untouched, so every existing
config still builds. But a *new* run cannot select a different polynomial basis.

`poly_degree` still applies: it is the ChebK order of each block's depthwise conv.

## !! Before launching !!

**Sync the regularization keys to bench_v12's winner.** `staged.yaml` currently mirrors
`convnext_nodrop` (`dropout_rate: null`, `drop_path_rate: 0.0`). If `convnext_nodrop_droppath`
wins that round, set `drop_path_rate: 0.1` here as well — otherwise the contrast moves two knobs
and attributes to neither.

**The staged code path has never been executed.** The layout was verified by simulating the layer
stack in pure Python; the real build needs TensorFlow, which is not on the login node. Run it
through `.claude/bin/run_job.sh` (or give this arm's first job a short `--partition=debug` slot)
before spending a 2 × 12 h chain. `run_training.py` builds and traces the network at startup, so
a mistake surfaces in minutes.

## The arm

| file | one knob vs | keys | jobs |
|---|---|---|---|
| `staged.yaml` | `bench_v12/convnext_nodrop.yaml` | 6 (one change) | 2 (`afterany` chain) |

The control needs no run: it is `bench_v12_convnext_nodrop`, already on disk by then.

Six keys, one knob — `base_channels`/`pool_layers`/`conv_layers`/`residual_layers` out,
`stage_widths`/`stage_blocks` in. They are one layout, not six choices; `config_check diff` will
report `MORE THAN ONE KNOB` and that is expected here, exactly as for the ConvNeXt block's
three-key package in bench_v12.

## Scoring

`compare-runs` paired FoM against `bench_v12_convnext_nodrop`. Floor **0.049**; a tie is the
expected and acceptable outcome, and it argues for **adopting** this arm rather than dropping it —
the usual "a tie goes to the simpler net" rule points here, because this *is* the simpler net
(one block type, two fewer knob families, no hidden widening).

**Q3 `coverage` is required**: the summary dimensionality is unchanged at 512, but the readout's
upstream path is not, and a pathological summary defeating the flow is the one failure mode Q1
cannot see.

## Not in this round

**Distributing blocks across resolutions.** Real ConvNeXt puts blocks at every stage (3,3,9,3);
this arm keeps the current 1/1/5 split so it stays one knob from its parent. Rebalancing
`stage_blocks` is now a one-line config change and is the natural bench_v14 if this lands.
