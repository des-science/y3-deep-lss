# bench_v7 — CLUSTERING transformer arm (`transformer`)

**Status: CONFIGS WRITTEN, NOTHING LAUNCHED, SIZING IS A GUESS — MEASURE FIRST** (2026-08-06).

Full rationale, the trunk table and the scoring rules live in
[`../../combined/bench_v7/README.md`](../../combined/bench_v7/README.md). Clustering-specific notes:

- **The only transformer arm whose architecture changed.** `base_embed_dim` 32 → 64. Clustering runs
  at nside 256 → 4 nested levels against lensing's and combined's 5, so its trunk was **512 against
  their 1024**; 64 × 2⁴ = 1024 matches. Plus `local_batch_size` 20 → 16 like the other two.
- **Read the config header before keeping the change.** Unlike the GCNN case it fixes no bug —
  `map_feature_dim: 512` already equalizes what reaches the head — it buys comparable body capacity.
  The costs all land here: no longer the validated v17 clustering architecture, ~4× body FLOPs, and the
  250 k step optimum measured at `base_embed_dim: 32` does not carry over.
- **One 12 h job**, `MAX_RUNS=1`. `n_steps: 94000` from a **guessed** 2.5 it/s × 37.8 ks.

## This is the round's least trustworthy number, in the format that cannot be rescued

The 2.5 it/s stacks two unmeasured factors on one measurement taken at neither of this config's
settings:

- 7.34 it/s **measured** — at `base_embed_dim: 32`, batch 20.
- 32 → 64 raises body FLOPs ~4×, but the effect on the *step* depends on the body's share of it, which
  is unknown: 60% body → 2.8× on the step, 80% → 3.4× (2.6 vs 2.2 it/s).
- batch 20 → 16 gives back ~1.15×.

2.5 sits at the pessimistic end on purpose. But a **single 12 h job has no second job to correct the
sizing in** — if the real rate is lower, the cosine never anneals to zero, `run_evaluation.py` and
`run_inference.py` never run, and the job produces nothing scorable at all.

**Run a few hundred steps on `--partition=debug` first**, then set `n_steps = it/s × 37.8 ks`. That
also settles whether batch 16 at trunk 1024 stays inside the ~85 GB NCCL band, which has not been
checked. The synthetic single-GPU benchmark apps are the wrong instrument — they under-predicted a real
4-GPU cost by ~40% in bench_v6.
