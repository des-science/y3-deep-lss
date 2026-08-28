# bench_v11 — CLUSTERING, THE TRANSPLANT

**Status (2026-08-28): LAUNCHED.** 2 arms, 2 jobs, single 12 h each (no chain). Jobs 3211839 (simple), 3211840 (simple_mean_std).

**The question.** bench_v11 on COMBINED settled that the *readout* is the whole gain and the
machinery is not. This round asks whether that holds on clustering alone — the last thing between here
and a final architecture. If it does, the production net for all three probes becomes the classic
block with a `mean_std` readout, and ConvNeXt, DropPath and interleaved attention are deleted.

**What combined measured** (paired FoM, 1000 mocks, seed floor 0.049), at fixed `mean` readout:

| contrast | ratio |
|---|---|
| the whole ConvNeXt + DropPath + attention stack (`v2` vs `bench_v11_simple`) | 1.074 |
| the readout alone, classic block (`simple_mean_std` vs `simple`) | 1.069 |
| both stacked (`bench_v8_mean_std`) over the readout alone | 1.028 **=** |
| the ConvNeXt block ISOLATED, no DropPath, no attention | 1.025 **=** |

The two levers are **substitutes, not additive**, and the block on its own does nothing.

## The arms

| arm | block | DropPath | attn | readout | budget | jobs |
|---|---|---|---|---|---|---|
| **`simple.yaml`** — the floor | classic | 0.0 | none | mean | 39 600 s | 1 |
| **`simple_mean_std.yaml`** — the candidate | classic | 0.0 | none | **mean_std** | 39 600 s | 1 |

**The anchors are already on disk at the SAME wall, so this round needs none of its own:**

| anchor | block | DropPath | attn | readout | budget |
|---|---|---|---|---|---|
| `bench_v10_mean_std_1x` | convnext | 0.1 | every 2 | mean_std | 39 600 s |
| `v2` (v18 production default) | convnext | 0.1 | every 2 | mean | 39 600 s |
| `bench_v10_mean_std_2x` | convnext | 0.1 | every 2 | mean_std | 79 200 s — **NOT equal wall, do not pair on it** |

## The contrasts

| contrast | knob | clean? |
|---|---|---|
| `simple` → `simple_mean_std` | `map_pool` mean → mean_std | ✅ one knob, verified |
| `simple_mean_std` → `bench_v10_mean_std_1x` | the ConvNeXt+DropPath+attention package, at fixed mean_std | ✅ the decision contrast |
| `simple` → `v2` | the same package, at fixed mean | ✅ the same question from the other side |
| `bench_v7/simple` → `bench_v11/simple` | sizing mode only (3 keys, 1 knob) | ✅ a wall-matched refresh, not an arm |

**Declared confound, and it runs the conservative way.** The classic block is ~1.16× SLOWER than
ConvNeXt, so at equal wall these arms train ~15% fewer steps than the anchors — worth roughly 1.5%
of FoM *against* them. A win or a wash is therefore conservative; only a loss beyond ~2% is
ambiguous between the block and the budget. On combined the local budget elasticity measured **~0**
(`bench_v8_long` at 2× wall scored 1.069 vs `v2`'s 1.074 at 1×, a wash), so it may be smaller still.

## Scoring

```bash
P=/users/athomsen/dlss/repos/y3-deep-lss/.venv/bin/python3
R=/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v18/default/maps/clustering
$P -m deep_lss.apps.tuning.run_comparison --root $R --reference bench_v11_simple \
   bench_v11_simple_mean_std bench_v10_mean_std_1x v2
$P -m deep_lss.apps.tuning.run_diagnostics robustness --root $R bench_v11_simple \
   bench_v11_simple_mean_std bench_v10_mean_std_1x v2
$P -m deep_lss.apps.tuning.run_diagnostics coverage --root $R --reference bench_v10_mean_std_1x \
   bench_v11_simple_mean_std bench_v11_simple
```

**Q3 coverage is REQUIRED here**, not optional: `mean_std` changes the summary dimensionality
512 → 1024, and a pathological summary defeating the flow is the one failure mode Q1 and Q2
structurally cannot see.

**The decision rule, pre-registered.**
`simple_mean_std / bench_v10_mean_std_1x >= 0.951` (a wash or better) → **the combined result
transplants; adopt the classic + mean_std recipe as the production default on all three probes.**
Below 0.951 on *either* probe → the machinery earns its place on that probe and the final
architecture is probe-dependent, which is a materially worse outcome for the paper and should be
reported as such rather than averaged away.
