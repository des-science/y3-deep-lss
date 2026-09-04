# bench_v7 — first v18/default GCNN reference round, all three probes

**Status: COMPLETE AND SCORED** (2026-08-11). All nine map-level runs plus a four-probe Cls baseline
finished. **Results: [`RESULTS.md`](RESULTS.md)** — that file is the answer; this one is the round
definition and rationale.

Headline: the GCNN `full`-over-`simple` margin is +2.6% lensing / +8.5% clustering / +4.1% combined,
so it is only clear of the 1.5% seed floor on clustering — but the **combined transformer beats the
best GCNN by 12.3%**, and all nine arms beat the Cls baseline. Follow-up round:
`../bench_v8/`.

This round spans three probe directories — `deepsphere/{lensing,clustering,combined}/bench_v7/`.
This file carries the shared rationale; the other two hold their probe-specific notes and point here.

## What the round is for

Two things at once:

1. **The first GCNN reference set on v18/default**, for lensing, clustering and combined. The old
   probe defaults (`deepsphere/<probe>/maps+cls.yaml`) predate the bench rounds and still carry the
   **flatten readout** at base 32 — the configuration measured at **0.399 against the two-point
   baseline**, i.e. losing to 2pt by 2.5×. They are not a usable starting point. Those defaults are
   left untouched here on purpose; the user updates them once this round has an answer.
2. **Full stack vs simple**, to test whether the bench_v6 margin is real or noise.

## The two configs

`full.yaml` is a faithful port of the bench_v6 champion `bench_v6_convnext_droppath_attn`
(paired **0.967** vs the transformer `t2_cls` on v17). `simple.yaml` is the bench_v6 **anchor**,
`bench_v5_pool_head_w64` (paired **0.918**) — not an arbitrary small config, but the exact recipe
the full stack was built on top of.

| | `simple.yaml` | `full.yaml` |
|---|---|---|
| readout | `map_pool: mean` | `map_pool: mean` |
| trunk | 512 | 512 |
| residual block | classic | **convnext** |
| DropPath | **0.0** | **0.1** |
| interleaved attention | **none** | **every 2** |
| v17 paired vs `t2_cls` | 0.918 | 0.967 |

**The contrast is three knobs, not one, and that is deliberate.** bench_v6 already attributed the
levers individually on v17 (block 1.024, DropPath-on-convnext 1.050, attention 1.005–1.056, against a
**1.5% seed floor**). This round asks the question attribution cannot answer — does the stacked
0.967-vs-0.918 margin survive a dataset change, or is it inside the noise? **Do not quote this pair as
evidence about any single lever.**

Both configs keep the two levers that are unambiguously above the seed floor: the **mean-pool
readout** (+32.1% paired over flatten, 95% of mocks) and **width with pool** (+8.4%, but 0.890
*without* the pooled readout). Neither is a candidate for removal.

`simple.yaml` sets `drop_path_rate: 0.0` **explicitly**. DropPath is +5.0% on the ConvNeXt block but
**0.946 on the classic block — it hurts there**. The explicit zero is there so nobody "improves" the
simple arm by switching it on.

## The transformer reference arm

One transformer run per probe, `bench_v7_transformer`, alongside the six GCNN runs. Configs live under
`configs/transformer/<probe>/bench_v7/transformer.yaml` (renamed from `default.yaml` on 2026-08-11) —
the probe defaults are left untouched, same as on the GCNN side.

**The transformer defaults were already architecturally up to date.** A parsed-YAML diff of each
probe's `maps+cls.yaml` against the corresponding v17 reference run's own `configs.yaml` shows only
`spmm_backend: csr` (numerically equivalent to `coo` up to fp32 tolerance — a pure speedup, no lineage
change), `checkpoint_every`, and `n_steps`. Nothing architectural is stale, so nothing here changes it
except the trunk fix below.

**But the trunk was NOT matched across probes — same defect, same 2× factor, as the GCNN.** Trunk
width = `base_embed_dim × 2^num_nested_levels` with `growth: double`, and
`num_nested_levels = log2(nside / token_nside)`. Clustering runs at nside 256, so it gets **four**
levels where lensing and combined get five:

| probe | nside | levels | `channel_dims` (from the run logs) | trunk |
|---|---|---|---|---|
| lensing | 512 | 5 | `[32, 64, 128, 256, 512, 1024]` | 1024 |
| combined | 512 | 5 | `[32, 64, 128, 256, 512, 1024]` | 1024 |
| clustering | 256 | 4 | `[32, 64, 128, 256, 512]` | **512** |

Those are read off the runs' own training logs, not derived. `base_embed_dim: 64` on clustering gives
`[64, 128, 256, 512, 1024]` → 1024, matching. That is the **only** change to the transformer recipe in
this round.

**The argument is weaker here than on the GCNN, and that is worth knowing before keeping it.** On the
GCNN the mismatch was *mechanical* — with `map_pool: mean` the trunk width *is* the pooled map vector,
so a 256-wide trunk would silently unbalance the concat. The transformer has `map_feature_dim: 512`,
which projects whatever the trunk is down to 512 before the concat, so **the fusion is already
balanced either way and nothing is silently broken today**. What the change buys is comparable *body
capacity* across probes. That is a comparability argument, not a bug fix.

It is also not free, and all of the cost lands on clustering: the arm is no longer the architecture the
v17 clustering reference validated; doubling `base_embed_dim` doubles the width at every level so body
FLOPs go ~4× and the measured 7.34 it/s will drop substantially; the 250 k step optimum measured at
`base_embed_dim: 32` does not carry over; and batch 20 has not been re-checked against the ~85 GB NCCL
band at trunk 1024.

**Transformer sizing — note lensing is deliberately NOT wall-matched to the GCNN arms:**

| probe | `n_steps` | wall | rate |
|---|---|---|---|
| lensing | 150 k | 1 × 12 h | 3.85 it/s, **MEASURED** |
| combined | 250 k | 2 × 12 h | ~3.5 it/s, **MEASURED** |
| clustering | 158 k | 2 × 12 h | **ASSUMED ~2.0 it/s — not measured** |

150 k is lensing's measured **optimum** — **250 k regressed on all FoM variants**. The transformer's
budget on lensing is not monotone, so extending it to fill the GCNN arms' 2 × 12 h wall would actively
degrade the reference. Each arm gets its own best budget; wall-matching across architectures is not a
goal. Clustering's 158 k assumes a deliberately pessimistic ~2.0 it/s to undersize — **measure and
correct on job 1**.

## GCNN trunk width is pinned to 512 on every probe

Trunk is pinned **within** each architecture, across probes — that is what makes the probes mutually
comparable and the combined-vs-single sanity gate meaningful. It is deliberately **not** pinned
*across* architectures: the GCNN arms sit at 512 and the transformer arms at 1024, each at its own
validated width. Comparing a GCNN to a transformer goes through paired FoM, not through matched widths.

Trunk = `base_channels × 2^pool_layers`. This is not cosmetic: with `map_pool: mean` the trunk width
*is* the pooled map vector, which the Cls branch's `embedding_layers: [..., 512]` exists to balance at
the concat. A probe with a different trunk would silently unbalance its own fusion and break
comparability with the others.

| probe | input nside | `pool_layers` | `base_channels` | trunk | body |
|---|---|---|---|---|---|
| lensing | 512 | 3 | 64 | 512 | nside 16 |
| clustering | 256 | 2 | **128** | 512 | nside 16 |
| combined | 512 (+256 inj.) | 3 | 64 | 512 | nside 16 |

**`base_channels: 128` on clustering is the one setting this round changes on its own authority.** The
old clustering default used 64 to match a *base-32* lensing config; that pairing is obsolete now that
lensing moves to a 512 trunk. See `clustering/bench_v7/full.yaml`.

Every probe's residual body lands at **nside 16 (~448 footprint tokens)** — each stage halves nside,
and `pool_layers + conv_layers` is 5/4/5 from 512/256/512. So the attention blocks cost the same ~3%
on all three; token count does not vary by probe.

## Sizing — MEASURED 2026-08-06

**The wall budget differs by probe, and it is not a free choice.** Single probes run as **ONE 12 h
job** (~10.5 h training + the eval+inference tail inside the same job = **37.8 ks**); combined does
not fit one job and runs as a **2 × 12 h `afterany` chain** (~11 h training per job = **79.2 ks**).
That is the established convention — the old lensing default is 170 k / 4.45 it/s = 10.6 h in one
job, clustering 140 k / 3.73 = 10.4 h — and it is what keeps the winning single-probe config directly
adoptable as a 1-job default without re-sizing.

`n_steps = measured it/s × basis`, then **floored to the nearest 10 k**.

| probe | config | it/s used | basis | raw | **`n_steps`** | source |
|---|---|---|---|---|---|---|
| lensing | `full` | 3.53 | 37.8 ks | 133.4 k | **130 k** | production run 3021200 |
| lensing | `simple` | 3.12 | 37.8 ks | 117.9 k | **110 k** | production run 3021201 |
| clustering | `full` | 3.57 | 37.8 ks | 135.0 k | **130 k** | production run 3021202 |
| clustering | `simple` | 3.16 | 37.8 ks | 119.4 k | **110 k** | production run 3021203 |
| combined | `full` | **2.62** | 79.2 ks | 207.5 k | **200 k** | **min** of 3.24 (job 1) / 2.62 (job 2) |
| combined | `simple` | **2.90** | 79.2 ks | 229.7 k | **220 k** | **min** of 2.90 (job 1) / 3.03 (job 2) |

Measured with real 4-GPU runs at batch 16 — never `shared/benchmark_sweep.sh`, which is single-GPU
and synthetic and under-predicts the ConvNeXt block's 4-GPU cost by ~40%, erring toward oversizing.

**The 25-minute `SKIP_EVAL=1` sizing probe is a good instrument — on the single probes it landed
within 1–3% of the full production rate**, and the 10 k floor absorbed the difference entirely: all
four single-probe configs kept the `n_steps` the probe predicted, so those files still describe the
runs on disk exactly.

**Both combined configs were re-sized after the fact and no longer describe their runs on disk** —
see the `!! PROVENANCE !!` blocks in their headers. They were the only two arms sized off a v17 rate
that was never re-measured on v18, and `full`'s 3.25 was 17% optimistic; job 2 was on course to finish
training with ~4 minutes to spare, leaving eval and inference unrun and the chain unscorable. Both
were rescued with a 3rd `afterany` job (3028402 / 3028404), the same fix bench_v6 needed.

### Node variation is the dominant term on combined — size on the MINIMUM

`combined/full` measured **3.24 it/s in job 1 and 2.62 in job 2** on identical work: a **24% swing
between nodes**. That is ~3× the block-type effect it is meant to resolve and an order of magnitude
above the 1.5% seed floor. A chain must complete on the worst node it draws, and the error is
asymmetric, so combined `n_steps` is sized on the **minimum** sustained rate observed, not the mean.
Sizing on job 1's 3.24 is precisely what produced the rescue.

This does not apply to the single probes: they are one job, so there is only one node to draw, and a
mis-draw costs wall rather than scorability.

**Do not re-derive a block-speed conclusion from combined's job-2 numbers.** On job-1 rates ConvNeXt
was faster there too (3.24 vs 2.90), consistent with bench_v6 and with both single probes. The
apparent inversion in job 2 is node noise.

**Every single-probe number is conservative in three independent ways**, which is deliberate:

1. **The windowed rate was still climbing when read.** Warmup bleeds well past step 500 — the
   from-step-500 window gave 3.33/3.34, and only the second-half window reached 3.50/3.59. The
   headline uses the second half, which agrees with the last-25% window to ~1%.
2. **The 10 k floor rounds down again.**
3. Sizing errors are **asymmetric** — oversizing forfeits the anneal tail *and* the eval tail
   (nothing scorable at all), undersizing only wastes wall.

**Do not read the rate off tqdm's cumulative figure.** It reads 3.22–3.26 against sustained
3.50–3.59: a ~9% startup drag that would undersize every arm. Use a windowed
Δsteps/Δelapsed that starts after warmup and still spans validation stalls.

**Two pre-measurement worries, both resolved:**

- **`base_channels: 128` on clustering costs nothing.** At 3.59 it/s it was the *fastest* arm in the
  sweep, marginally quicker than lensing's identical architecture. The warning that this
  never-before-run width might come in expensive is retired.
- **My "lensing must be faster than combined" lower-bound argument was thinner than stated.** It held
  (3.50 > 3.25) but by only 2.5%, not the comfortable margin implied.

**The one number not measured on v18 is combined's.** The sweep covered the four single-probe arms
only. v18 has the same map geometry and probe set as v17, so the step cost should carry, but two
single-probe arms came in *faster* than the v17 combined figure for the same architecture — consistent
either with lensing being cheaper than combined or with v18 being slightly faster overall, which this
sweep cannot separate. Carrying 3.25 is the conservative read. **Confirm against job 1's real rate.**

**To resize a chain already running, edit `<dir_model>/configs.yaml`, not the file in this directory.**
A restored job reads the run-dir snapshot (`run_training.py:359`), never `--net_config`. The failure is
silent — the job starts normally and only the progress-bar total (`n_steps − start_step`) reveals it.
This bit three runs in bench_v6. Applies to the combined chain only; the single probes are one job.

## Wall-matched, not step-matched

**ConvNeXt is faster than the classic block on the single probes** — production runs measured ×1.13
on lensing (3.53 vs 3.12) and ×1.13 on clustering (3.57 vs 3.16), in the same direction as bench_v6's
×1.17 at base 64 though somewhat smaller.

**The resulting budget bias points in OPPOSITE directions on the single probes and on combined.**
This is the single most important thing to carry into scoring:

| probe | `full` | `simple` | budget bias |
|---|---|---|---|
| lensing | 130 k | 110 k | **+18% to `full`** (~+1.5% FoM) |
| clustering | 130 k | 110 k | **+18% to `full`** (~+1.5% FoM) |
| combined | 200 k | 220 k | **+10% to `simple`** (~+1% FoM) |

On the single probes the bias is the honest consequence of ConvNeXt being cheaper. On **combined** it
is an artifact: `full`'s job 2 drew a slow node (2.62 vs job 1's 3.24), and the conservative
size-on-the-minimum rule turned that draw into a permanently smaller budget. On job-1 rates ConvNeXt
was faster on combined too.

**What this means when reading the result.** Budget is worth ~+7% per 2× with the pooled readout, so
these are ~1–1.5% effects against a 1.5% seed floor — small, but the same order as the margin the
round is trying to resolve.

- On **lensing and clustering**, the bias runs *against* the round's hypothesis: if `simple` lands
  within the seed floor of `full`, the case for the simpler architecture is *stronger* than the raw
  numbers show.
- On **combined**, it runs *with* it: a `simple` win of less than ~2% there is **not** evidence of
  anything architectural. Do not report one without this caveat.

## v18 specifics

- v18 is **`extended_nla: True`**, so the **plain** probe configs (`lensing`, `clustering`,
  `combined`) are correct — *not* the `_nla` variants. The "v17+ needs `_nla`" rule tracks
  `extended_nla`, not the version number, and getting it wrong does not raise.
- `training.sh` already defaults to `VERSION=v18 SUBVERSION=default`.
- The Cls cache `data/v18/default/cls/rebinned_nb16_8wl,32gc.h5` is built and matches `n_bins: 16` +
  the default `SCALES=8wl,32gc`.
- **v18 folds source clustering into the shape noise**, so `source_clustering_in_place` is now partly
  *in* the training distribution. The GCNN's headline robustness margin (≤0.27σ vs `t2_cls`'s 0.44σ,
  driven by `source_clustering_fixed` at 1.12σ) was measured on v17 where it was not. Do not read a
  shrunken margin on v18 as the GCNN getting worse.

## Launching

Use the **`submit` skill** — do not write a new submission script. Six GCNN runs, but **they do not
all launch the same way**:

- **lensing and clustering — ONE 12 h job each** (`training.sh`, `RUN_NUM=1`). Not a chain. `n_steps`
  is sized so training and the eval+inference tail both fit inside the single job.
- **combined — a 2-job `afterany` chain each** (`training_chainer.sh`, `MAX_RUNS=2`). `n_steps` is the
  TOTAL across the chain and the cosine spans it; job 1 TIMEOUTs by design. **`afterany`, never
  `afterok`** — the TIMEOUT is the expected exit and `afterok` would never release job 2.

Set `PROBE` and `NET_CONFIG` together, and use `MODEL_DIR=bench_v7_full` / `bench_v7_simple` under the
probe's own run dir. Outputs land in
`/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v18/default/maps/<probe>/`.

Do **not** set `OUTPUT` or `SKIP_EVAL` for these — both exist for throwaway sizing probes and would
send a production run to the wrong place or skip its eval+inference entirely.

**Verify the sizing once combined's job 2 starts:** its progress-bar total must equal
`n_steps − start_step`.

**A fan-out of short jobs must not go to `--partition=debug`:** `debug-qos` is `MaxJobsPU=1`,
`MaxSubmitJobsPU=2`, so it serializes, and the third and later `sbatch` calls in a loop are *rejected
at submit time*. These are 12 h jobs, so they go to `normal` regardless — but the sizing probes hit
this.

## Scoring

Use the **`compare-runs` skill** — paired FoM(Ωm, S8) over the 1000 `mcmc_samples.h5` mocks, plus
`vali_total`. **Seed floor 1.5%**; never the 16-mock `chain_grid_*` route. Never rank by eye.

Three questions this round can answer, and one it cannot:

- **Does the full stack beat simple on v18?** If the gap is inside 1.5%, prefer `simple`.
- **The combined ≥ single-probe sanity gate**, carried unchecked since bench_v4. The pinned 512 trunk
  is what makes this comparison meaningful; this is the first round that can run it.
- **Does attention still earn its place?** Its mechanism on combined was long-range *cross-probe*
  mixing, which does not exist on a single probe. If the single-probe full-vs-simple gaps are smaller
  than combined's, that is the first suspect.
- **It cannot attribute individual levers** — see the three-knob note above.

Informativeness only. Robustness (posterior bias on the systematics mocks), estimator validity
(SBC/TARP/HPD) and real-data behaviour (PPC) are separate questions with separate instruments. DES FoM
is unsigned under misspecification and is not to be ranked on.
