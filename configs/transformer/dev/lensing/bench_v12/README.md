# bench_v12 — LENSING, transformer arms

**The round definition, the shared core, the launch plan and the scoring rules live in
[`../../../../deepsphere/dev/combined/bench_v12/README.md`](../../../../deepsphere/dev/combined/bench_v12/README.md).**
Read that first. This file holds only the transformer side.

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


## Why the transformer is in a GCNN round at all

Because question **B** — does the shared regression head need dropout — is **one** question. Both
architectures build their head with the same `get_regression_head(...)` call
([`heads/regression_head.py`](../../../../deep_lss/nets/heads/regression_head.py)), so the layer
being removed is *identical*, not analogous. Answering it on one architecture and assuming it
transfers is exactly the mistake DropPath already punished (+5.0% on ConvNeXt, 0.946 on classic).

The block question (**A**) has no transformer counterpart, so the transformer contributes
**three** arms against the GCNN's five — but question B is now answered symmetrically on both
architectures, since DropPath was added to this network for this round (see below).

## The arms

| file | head dropout | body DropPath | one knob vs |
|---|---|---|---|
| `transformer.yaml` | 0.1 | 0.0 | — (the anchor; see below) |
| `transformer_nodrop.yaml` | **null** | 0.0 | `transformer` |
| `transformer_nodrop_droppath.yaml` | **null** | **0.1** | `transformer_nodrop` |

Adam 1e-4 in both; **weight decay is not tested this round** (it has not paid off previously, and
a third regularizer in a round expecting washes buys multiple comparisons, not an answer).

### The anchor is no longer `transformer/prod/lensing/maps+cls.yaml`

Two keys differ, both carried by **every** arm here and neither a knob of this round:

| key | prod → bench_v12 | why |
|---|---|---|
| `network.map_feature_dim` | 512 → **null** | the maps+Cls fusion now has the GCNN's shape: pool → LN → concat → head, no projection `Dense` |

Both are documented in full in the [round README](../../../../deepsphere/dev/combined/bench_v12/README.md).
**If an arm here ships, prod must be updated to match before it is used as a default.**

## `transformer.yaml` needs a run even though it changes nothing

It is the prod recipe unchanged, and prod is already on `n_steps: auto` + the same wall budget as
the GCNN. **But the recipe has never been run at that budget.** Nearest thing on disk:
`bench_v7_transformer`, 180 000 steps at 4.78 it/s = 37.7 ks — close enough to 39 600 s to serve as an **approximate** anchor, but it was sized under fixed `n_steps`.

So `transformer.yaml` is both the round's first genuinely wall-matched transformer anchor and the
reference `transformer_nodrop.yaml` is measured against. Running the ablation without it would
leave the contrast anchored on a differently-sized run.

## !! The readout changed in code on 2026-08-31 — read this before comparing to anything older !!

`NestedHierarchicalLocalWindowTransformer`'s **final pre-pool `LayerNorm` was removed**
([`nested_transformer.py`](../../../../deep_lss/nets/encoders/maps/transformer/nested_transformer.py)),
so the readout is now **pool -> norm**, matching the GCNN twin and ConvNeXt's own classifier
(`global average pool -> LN -> Linear`) instead of the ViT convention (`norm -> pool`). The norm the
head needs is supplied after the pool by `TransformerSummaryNetwork` — `map_norm` with a Cls branch,
the regression head's own leading `LayerNorm` otherwise. An explicit `tf.cast(x, tf.float32)` took
the removed layer's place so every pooling branch sees the dtype it saw before; that cast is load
bearing, not cosmetic, because the pre-norm residual stream is unbounded in depth and the pool would
otherwise average N bf16 values of it.

The two orderings are **not** equivalent: `LayerNorm` acts per token across channels, so norming
first equalizes every token's magnitude before the average and the pool stops being a spatial mean of
the body's features. Amplitude across sky positions is signal for this task.

**Three consequences:**

1. **Every transformer checkpoint written before this is dead.** A variable was removed, and
   `restore_model*` chains `assert_existing_objects_matched()`, so a stale restore hard-errors rather
   than silently loading a partial graph. `bench_v7_transformer` is no longer an anchor for anything.
2. **`transformer/prod/<probe>/maps+cls.yaml` is unchanged as a file but now builds a different
   network** than the runs made with it. The config diff against `transformer.yaml` below still
   reports *identical*; that is a statement about the YAML, not about the architecture on disk.
3. **The contrast was taken as a hard change, not behind a flag**, so there is no measurement of what
   the reordering cost. That was a deliberate call: the goal is a unified readout for the paper
   figure, and both orderings are standard in their own families.

## Budget and probe specifics

- **Budget:** 39 600 s = **ONE 12 h job**. Same wall as every GCNN arm of the round.
- 150 k was this probe's measured optimum and 250 k **regressed on every FoM variant**, so the transformer's budget is not monotone here. 180 k sits between them and is the wall-fill; do not extend it.

## What is shared with the GCNN arms, and what is not

Shared: the mean readout over the same 448 nside-16 tokens (the encoder pools them itself — there
is **no `map_pool` option** on this path), the same head, the same Adam/cosine/warmup/clip, batch
16, per-probe smoothing and `input_norm`.

Deliberately not shared, and out of scope: trunk width (1024 vs the GCNN's 512 — pinned *within* an
architecture, never *across*), `map_feature_dim: 512` (only the transformer's pooled feature needs
crushing to match the Cls embedding), and bf16 + `jit_compile_body`, which is a memory decision.
The paper figure draws all three explicitly, so none of them is a shared component being broken.

## Prior dropout evidence exists and cannot be scored

v17 ran `t0_no_dropout` / `t0_block_dropout` (lensing) and `bench_t2_no_dropout` /
`bench_t2_block_dropout` (clustering, combined). **Those run dirs contain no `configs.yaml`**, so
`run_comparison`'s comparability gate refuses them outright. Do not quote a hand-computed number
from them, and do not reach for `--no_strict` to get past the gate.

## DropPath was added to this network for this round (2026-08-31)

`NestedHierarchicalLocalWindowTransformer` had `LayerScale` on each residual branch but **no
stochastic depth**. It now takes a `drop_path_rate` kwarg, threaded network → local stages →
global blocks, and applies `deepsphere.gnn_layers.DropPath` — **the same layer the GCNN's ConvNeXt
block uses** — to the attention and MLP branches, after `LayerScale`, immediately before each
residual add.

Two deliberate choices, both aimed at making this the *same* knob as the GCNN's:

- **Constant rate across depth**, not timm's linear 0 → max ramp, because `gcnn/resnet.py` passes
  one rate to every block.
- **Same layer, same position** (`x + DropPath(branch)`), so a cross-architecture read of the knob
  is legitimate rather than an analogy.

`DropPath` declares **no trainable variables**, so `drop_path_rate: 0.0` is a graph no-op and
existing checkpoints are unaffected — the same reasoning the file's `block_dropout_rate` comment
already relies on. (Note this is the opposite case to the pre-pool `LayerNorm` removal above,
which *did* break lineage because it removed a variable.)

**It has never been run, or even smoke-tested** — but it needs neither a debug job nor a rate
probe. `run_training.py` builds and traces the network (`_build_trace`, ~line 860) before the
input-norm statistics (~line 1073) and long before the training loop, so a plumbing error is a
crash within minutes, not a wasted 12 h slot. Launch `transformer_nodrop_droppath` **first and
alone** — it exercises all three of the 2026-08-31 changes at once — and release the other arms
once its log shows `Built transformer network nested_transformer`. The chains are `afterany`, so
a build-time crash still releases the follower job; that is the only thing ordering protects.

**The GCNN's DropPath result does not transfer.** Its sign was already block-dependent inside one
architecture (+5.0% on ConvNeXt, **0.946 — a loss — on classic**). A sign that flips between two
blocks of one network is not a sign to assume across two networks.

## What this round still does not test

**Weight decay**, on either architecture. The full ViT/DeiT/Swin recipe is bare head + stochastic
depth + AdamW; this round covers the first two. AdamW has not paid off in previous rounds here,
and a third regularizer in a round that expects washes buys multiple comparisons, not an answer.

## Launching

Via the **`submit` skill**. `MODEL_DIR=bench_v12_<basename>`, `RUN_NUM=1`, own run dir per arm.
Set `NET_CONFIG` and `PROBE` together and pass `ARCH=transformer` (a wandb tag only — keep it
consistent by hand).
