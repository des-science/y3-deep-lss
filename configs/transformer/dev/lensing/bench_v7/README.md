# bench_v7 — LENSING transformer arm (`transformer`)

**Status: CONFIGS WRITTEN, NOTHING LAUNCHED, SIZING NOT MEASURED** (2026-08-06).

Full rationale, the trunk table and the scoring rules live in
[`../../combined/bench_v7/README.md`](../../combined/bench_v7/README.md). Lensing-specific notes only:

- **Architecture unchanged** from the v17 unified recipe except `local_batch_size` 20 → 16 (to match
  the GCNN arms). Trunk is already 1024 (`[32, 64, 128, 256, 512, 1024]`, levels = log2(512/16) = 5),
  so no `base_embed_dim` fix is needed on this probe.
- **One 12 h job**, `MAX_RUNS=1`. `n_steps: 167000` from an assumed 4.43 it/s (3.85 **measured**, but
  at batch 20, × 1.15 for batch 16) × 37.8 ks.
- **Do not raise this to fill a 2 × 12 h wall.** 150 k × 20 = 3.0 M samples was this probe's measured
  optimum and 250 k × 20 **regressed on all FoM variants** — the budget is not monotone here. 167 k ×
  16 = 2.67 M samples sits ~89% of the way to the optimum, on the safe side.
- **Measure the rate before launching.** A single job has no second job in which to correct the
  sizing; if the real rate is below 4.43 the cosine never anneals and the job produces nothing
  scorable. A few hundred steps on `--partition=debug` settles it.
