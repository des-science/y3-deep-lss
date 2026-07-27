# bench_v6 — DeepSphere/GCNN combined-probe HPO, round 3

**Status: RUNNING.** Submitted 2026-07-26 as jobs 2903503–2903510 (four 2-job `afterany` chains).

Full reasoning: `dev/notes/bench_v6_deepsphere_combined_2026-07-26.md`.
Evidence it rests on: `dev/notes/bench_v5_paired_reanalysis_2026-07-26.md`.

**Target:** transformer `t2_cls`, paired ratio 1.000 by definition.
**Anchor:** `bench_v5_pool_head_w64` — base 64, mean-pool readout, classic residual body, 210 k steps,
**paired 0.918**, vali −10.600. Every config here is that run plus one thing.

## The four runs

Each is a 2 × 12 h chain sized to spend **~11 h of each job on training** (≈ 79.2 ks), so
`n_steps = it/s × 79.2 ks`.

| config | one knob vs | tests | n_steps (file) | n_steps AS RUN | it/s (measured) | projected |
|---|---|---|---|---|---|---|
| `droppath_classic.yaml` | the anchor | DropPath on the classic block | 225 k | 225 k | **3.00** | 2.85 |
| `convnext.yaml` | the anchor | the ConvNeXt block: quality and speed at base 64 | 280 k | **350 k** | **3.52** | 4.4 |
| `convnext_droppath.yaml` | `convnext.yaml` | DropPath on the ConvNeXt block | 266 k | **330 k** | **3.35** | 4.2 |
| `convnext_droppath_attn.yaml` | `convnext_droppath.yaml` | interleaved global attention | 258 k | **290 k** | **3.25** | 3.7 |

Rates are sustained 4-GPU figures from the runs themselves (2026-07-26, ~5.8 h windows including
validation stalls). **The `n_steps` in the files is what a fresh 2 × 12 h chain should use; it is NOT
what the runs on disk trained to.** The attempted mid-round resize did not take — see below — so the
three ConvNeXt runs kept their original budgets and were finished with a third chained job. When
comparing these runs, the wall is 3 × 12 h for the ConvNeXt arms and 2 × 12 h for `droppath_classic`.

Two independent lines — classic and ConvNeXt — each getting DropPath, plus attention on top of the
ConvNeXt one (it was null-to-negative on the classic body, so it is not repeated there). Nothing here
depends on predicting how the two block types compare; both are run.

`_deferred/` holds the shelved configs, including the 4 × 12 h budget arms. Excluded from the submit glob.

## Sizing: what the projections got wrong, and how the runs were rescued

**The ConvNeXt block is only 1.17× faster than the classic block at base 64 — not the 1.54× projected.**
Pre-launch sizing extrapolated from base-32 rates (classic 4.99, convnext 5.80, classic b64 2.95) with a
fixed-overhead + ~C² body model. It over-predicted the ConvNeXt advantage by ~25 %. **That model does not
survive the width doubling; do not size a ConvNeXt config at a new width by extrapolation again.** Since
the cost refund is most of why the ConvNeXt line exists under the 24 h cap, its expected edge over
`droppath_classic` is correspondingly smaller: ~258–280 k steps against 225 k, not the ~350 k assumed.

Two smaller corrections, both worth reusing: DropPath costs **~0.7 %** on the classic block (3.00 vs the
anchor's 3.02), not the ~4 % carried over from base 32 — but ~4.8 % on the ConvNeXt body, close to its
base-32 figure. And the attention blocks cost only **~3 %** at base 64 versus ~13 % at base 32, because
the projections scale ~C² while the 448×448 attention matrix does not, so attention shrinks as a fraction
of the step as the body widens. `convnext_droppath_attn` also ran without approaching the memory band, so
the OOM / NCCL-hang worry for that geometry is retired.

**The projections nearly killed all three ConvNeXt runs.** At real rates they reach ~292 k / 278 k /
270 k against targets of 350 k / 330 k / 290 k. That is not a truncated cosine — job 2 hits its wall
*inside training*, so `run_evaluation.py` and `run_inference.py` never run and the job produces no
preds, no chains, nothing scorable.

### The failed fix — read this before trying to resize a chain

`n_steps` was edited in these YAMLs before job 2 launched (350 k→280 k etc.). **That silently did
nothing, and the runs went on training the old budgets.** A restored job does not read `--net_config`
at all:

```python
elif args.restore_checkpoint and (args.dir_model is not None):     # run_training.py:344
    conf = configuration.load_run_configs(os.path.join(dir_model, "configs.yaml"))   # :351
```

The config comes from **`<dir_model>/configs.yaml`** — the snapshot job 1 wrote at first launch
(`run_training.py:330`) and which nothing later overwrites. The failure is silent: the job starts
normally and only the progress-bar total (`n_steps − start_step`) gives it away.

**So: to resize a chain in flight, edit `<dir_model>/configs.yaml`, not the file in this directory.**
Editing this directory only affects a run started fresh with `RUN_NUM=1`. The underlying rebuild does
work — `decay_steps = n_steps − warmup_steps` is read at job start (`deep_lss/utils/optimization.py`)
while `optimizer.iterations` restores from the checkpoint, so a restored job rebuilds a *shortened*
cosine that still anneals to zero, at the cost of a one-time LR step-down at the handoff.

**How these three were actually rescued.** Caught mid-job-2 with ~5 h of wall left, by comparing the
progress-bar totals against the intended `n_steps`. Because each run was already 87–96 % through its
cosine, the remaining work at the *original* budget was only ~46 k / 36 k / 12 k steps (~4.2 h / 3.4 h /
1.7 h including the eval tail), so the cheapest fix was a **third `afterany` job with `configs.yaml`
left untouched** — completing each cosine to zero as designed. Jobs 2908306 / 2908307 / 2908308. No
config surgery, and total wall 25.5–28 h against the 24 h target rather than another full 12 h.

Two durable rules. Sizing errors are **asymmetric** — oversizing forfeits the anneal tail *and* the
eval tail, undersizing only wastes wall — so size conservatively and correct upward. And **check the
progress-bar total on job 2**, not just the rate: it is the only place the effective `n_steps` surfaces.
`start_step >= n_steps` is safe (`run_training.py:1315` warns, skips the loop and falls through to
eval), so a rescue job can be sized without knowing exactly where the previous one died.

Note `benchmark_v5.sh` is still the wrong instrument for this: single-GPU and synthetic, it *under*-
predicts the ConvNeXt block's 4-GPU cost by ~40 %, erring in the dangerous direction.

## Launching

`n_steps` is the TOTAL across the chain and the cosine spans it. Job 1 TIMEOUTs by design; job 2 trains
the remainder and runs eval + inference. **`--dependency=afterany`, never `afterok`** — the TIMEOUT is
the expected exit.

Verify the sizing once job 2 starts: its progress-bar total must equal `n_steps − start_step`. If it
does not, the run is on a different budget than you think (see above), and there is a whole job's worth
of wall in which to fix it by editing `<dir_model>/configs.yaml`.

```bash
cd /users/athomsen/dlss/repos/y3-deep-lss

for f in configs/deepsphere/combined/bench_v6/*.yaml; do
  name=$(basename "${f%.yaml}")
  J1=$(ARCH=deepsphere PROBE=combined NET_CONFIG="$PWD/$f" MODEL_DIR="bench_v6_${name}" \
       sbatch --parsable --job-name="bv6_${name}" --export=ALL,RUN_NUM=1 submissions/clariden/training.sh)
  ARCH=deepsphere PROBE=combined NET_CONFIG="$PWD/$f" MODEL_DIR="bench_v6_${name}" \
       sbatch --dependency=afterany:$J1 --job-name="bv6_${name}" --export=ALL,RUN_NUM=2 \
       submissions/clariden/training.sh
done
```

Outputs land in `/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v17/baseline/maps/combined/bench_v6_<name>/`.

## Scoring

```bash
.venv/bin/python3 -m deep_lss.utils.run_comparison \
  --root /iopsstor/scratch/cscs/athomsen/deep_lss/runs/v17/baseline/maps/combined \
  --reference t2_cls bench_v6_droppath_classic bench_v5_pool_head_w64   # ... etc
```

Paired FoM(Ωm, S8) over the 1000 `mcmc_samples.h5` mocks, plus `vali_total`. **Seed floor 1.5 %.** Never
the 16-mock `chain_grid_*` route (~2.4 % floor, blind below ~7 %, and the source of every claim this round
had to overturn).

**Do not diagnose overfitting from the train−vali gap** — 2× budget widened it while validation loss
improved by 0.567 nats. Use `vali_total`.

Informativeness only. Robustness (posterior bias on the systematics mocks), estimator validity
(SBC/TARP/HPD) and real-data behaviour (PPC) are separate questions with separate instruments. DES FoM is
unsigned under misspecification and is not to be ranked on.
