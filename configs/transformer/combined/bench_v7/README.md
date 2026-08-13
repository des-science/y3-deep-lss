# bench_v7 — transformer reference arms (`transformer`), all three probes

**Status: COMPLETE AND SCORED** (2026-08-11). All three runs finished; results for the whole round —
GCNN, transformer and the Cls baseline together — are in
[`../../../deepsphere/combined/bench_v7/RESULTS.md`](../../../deepsphere/combined/bench_v7/RESULTS.md).

Headline for these arms: a wash against the GCNN on lensing (1.014) and tied on clustering (1.091 vs
1.085), but **combined wins by 12.3% over the best GCNN** (1.123, 82% of mocks) — the round's largest
effect and the premise of `deepsphere/combined/bench_v8/`.

One transformer run per probe, alongside the bench_v7 GCNN arms (`configs/deepsphere/<probe>/bench_v7/`,
owned separately). Configs live in `configs/transformer/<probe>/bench_v7/transformer.yaml`; the probe
defaults are left untouched.

## Were the transformer defaults up to date? Architecturally, yes

A parsed-YAML diff of each probe's `maps+cls.yaml` against the corresponding v17 reference run's own
`configs.yaml` shows only `spmm_backend: csr` (numerically equivalent to `coo` up to fp32 tolerance —
a pure speedup, no lineage change), `checkpoint_every`, and `n_steps`. Nothing architectural is stale.
Only two things change here: **batch 20 → 16** (to match the GCNN arms) on all three, and
**`base_embed_dim` 32 → 64 on clustering** (trunk match, below).

## Trunk width — clustering was half, now matched at 1024

Trunk = `base_embed_dim × 2^num_nested_levels` with `growth: double`, and
`num_nested_levels = log2(nside / token_nside)`. Clustering runs at nside 256, so it gets **four**
levels where lensing and combined get five:

| probe | nside | levels | `channel_dims` | trunk |
|---|---|---|---|---|
| lensing | 512 | 5 | `[32, 64, 128, 256, 512, 1024]` | 1024 |
| combined | 512 | 5 | `[32, 64, 128, 256, 512, 1024]` | 1024 |
| clustering (was) | 256 | 4 | `[32, 64, 128, 256, 512]` | **512** |
| clustering (now) | 256 | 4 | `[64, 128, 256, 512, 1024]` | **1024** |

Read off the runs' own training logs, not derived. Same defect and same 2× factor the GCNN had on this
probe, fixed the same way — widen the base so the shorter hierarchy still ends at the common trunk.

Trunk is pinned **within** an architecture, across probes. It is deliberately not pinned *across*
architectures: the GCNN arms sit at 512, these at 1024, each at its own validated width. GCNN-vs-
transformer comparison goes through paired FoM, not matched widths.

**The clustering change costs more than it looks.** Unlike the GCNN — where `map_pool: mean` makes the
trunk width literally the pooled map vector, so a mismatch silently unbalances the concat — the
transformer's `map_feature_dim: 512` already projects any trunk down to 512 before fusion. Nothing was
silently broken. What the change buys is comparable body capacity, which is what was asked for, but it
means: this arm is no longer the architecture the v17 clustering reference validated; body FLOPs go
~4×; the 250 k step optimum measured at `base_embed_dim: 32` does not carry over; and the ~85 GB NCCL
band has not been re-checked at trunk 1024.

## Wall budget and step counts

Single probes get **one 12 h job** (~10.5 h training = 37.8 ks + the eval/inference tail); combined
gets a **2 × 12 h `afterany` chain** (79.2 ks). Same split as the probe defaults use.

| probe | wall | `n_steps` | assumed rate | confidence |
|---|---|---|---|---|
| lensing | 1 × 12 h | 160 k | 4.43 it/s | **medium** — 3.85 measured, but at batch 20 |
| combined | 2 × 12 h | 310 k | 4.03 it/s | **medium** — ~3.5 measured, but at batch 20 |
| clustering | 1 × 12 h | 90 k | 2.5 it/s | **low — effectively a guess** |

All rounded **down** to a multiple of 10 k. These are interim values: rate probes
(jobs 3021037 / 3021038 / 3021039, `maps/benchmarks/rate_probe.sh`) were submitted 2026-08-06 to
replace them with measurements. **Re-size from `claude/bench/rate_probe/rates.jsonl` before launching.**

**No rate here is measured at this config's actual settings.** All three inherit a batch-20
measurement and apply an assumed **1.15×** for the drop to batch 16. Batch 16/20 is 0.8× the work per
step, so a perfectly compute-bound step would give 1.25×, but per-step fixed costs (allreduce, kernel
launches, smoothing setup) do not shrink with batch — 1.15 is a deliberately conservative middle.

**Clustering stacks two more unmeasured factors on top of that.** `base_embed_dim` 32 → 64 raises body
FLOPs ~4×, but how much of the *step* that is depends on the body's share, which is unknown: a 60%
body share gives 2.8× on the step, an 80% share gives 3.4× — 2.6 vs 2.2 it/s. The 2.5 sits at the
pessimistic end on purpose.

Sanity note on lensing: 150 k × 20 = 3.0 M samples was this probe's measured optimum, and 250 k × 20
**regressed on all FoM variants**. 167 k × 16 = 2.67 M samples, ~89% of it — on the safe side of a
maximum whose far side is known to be bad. Sample-matching exactly would need 187 k ≈ 11.8 h training,
which does not fit one 12 h job.

## Measure before launching — the single-probe jobs cannot be rescued

This is the part that matters most now that the single probes are one job each.

On a 2 × 12 h chain a bad rate is recoverable: read the sustained it/s off job 1 and correct by editing
`<dir_model>/configs.yaml` before job 2 starts. **A 1 × 12 h single job has no second job.** If the
real rate is below the assumption, the cosine never anneals to zero, `run_evaluation.py` and
`run_inference.py` never run, and the job produces **nothing scorable at all**.

So the format that cannot be rescued (lensing, clustering) now carries the least certain numbers — and
clustering combines the round's largest unmeasured architecture change *with* that format.

**Recommended: a few hundred steps per config on `--partition=debug`** (jobs under 30 min belong there),
read the sustained 4-GPU it/s, then set `n_steps = it/s × 37.8 ks` for the single probes and
`× 79.2 ks` for combined. That also settles whether batch 16 at trunk 1024 stays inside the ~85 GB
NCCL band on clustering. Note the synthetic single-GPU benchmark apps are the wrong instrument — they
under-predicted a real 4-GPU cost by ~40% in bench_v6.

Sizing errors are asymmetric: oversizing forfeits the anneal tail *and* the eval tail, undersizing only
wastes wall. Every number above errs low on purpose.

## Launching

Use the **`submit` skill** — do not write a new submission script. Set `PROBE` and `NET_CONFIG`
together, `ARCH=transformer`, `MODEL_DIR=bench_v7_transformer`. Single probes: `MAX_RUNS=1`.
Combined: 2-job `afterany` chain, and **verify job 2's progress-bar total equals
`n_steps − start_step`**.

Outputs land in
`/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v18/default/maps/<probe>/bench_v7_transformer/`.

**`MODEL_DIR` must be set explicitly, and it is NOT the net-config basename here.** It defaults to
that basename — `default` — which would put this arm in a run dir named `default`, indistinguishable
from a promoted probe default and awkward to line up against the GCNN arms (`bench_v7_full`,
`bench_v7_simple`). The round's runs are named `bench_v7_transformer` on all three probes, so pass it
explicitly. The first launch (2026-08-06) did not, and the two completed single-probe runs were
renamed after the fact — including the `dir_model` line inside each run's own `configs.yaml`, which
is the only path reference a rename has to follow.

**The combined arm was missed in that cleanup and stayed as `combined/default` until 2026-08-11.**
It escaped notice because its eval stage died on the msfm `data/` rename (see below) on the same day,
so it had no `preds_*.h5` and did not show up in any output inventory — the wrong name and the
missing outputs hid each other. It was renamed only after its eval+inference was re-run (job
3056404). Two lessons worth keeping: check the run dirs, not just the outputs, when auditing a round;
and a run whose name is wrong is much harder to notice when it is also incomplete.

**DONE 2026-08-11: these files were renamed from `default.yaml` to `transformer.yaml`.** Note what that
does and does not buy, because an earlier version of this paragraph overstated it: `MODEL_DIR` falls
back to the config *basename*, so it is now `transformer` rather than `default` — still **not**
`bench_v7_transformer`, so it must STILL be passed explicitly. The GCNN arms have the identical
property (`full`/`simple` are not `bench_v7_full`/`bench_v7_simple`); nothing in a round whose run dirs
carry a round prefix can rely on the fallback.

What the rename actually fixes is the *failure mode*. `default` was a plausible run-dir name — it
collides with the promoted-probe-default convention, so an unset `MODEL_DIR` produced a directory that
looked legitimate and sat unnoticed for five days. `transformer` is self-evidently a config name, so
the same mistake now reads as a mistake. Fail-loud instead of fail-silent, not fail-never.

v18 is `extended_nla: True`, so the **plain** probe configs (`lensing`, `clustering`, `combined`) are
correct — not the `_nla` variants. Getting that wrong does not raise.

## Scoring

Use the **`compare-runs` skill** — paired FoM(Ωm, S8) over the 1000 `mcmc_samples.h5` mocks, plus
`vali_total`. **Seed floor 1.5%.** Never rank by eye.
