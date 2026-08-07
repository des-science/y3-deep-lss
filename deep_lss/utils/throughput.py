# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2026
Author: Arne Thomsen

Throughput measurement and wall-clock training budgets.

Two related concerns live here:

* :class:`ThroughputTracker` records how fast a run is actually training, in bins, and writes the
  result next to the model. It exists because the only throughput number a run used to expose was
  tqdm's, and tqdm reports the CUMULATIVE average over the whole progress bar -- which includes XLA
  compilation and the dataloader ramp. On a `jit_compile_body: true` transformer that is a 29-44%
  understatement of the sustained rate, so sizing `n_steps` off it undersizes the next run by about
  a third. Every finished run now leaves a measured rate behind, which makes the run archive itself
  the sizing database.

* :class:`WallClockBudget` lets a run be bounded by elapsed training seconds instead of a fixed step
  count. See the class docstring for why a measured-then-fixed step count is not good enough.
"""

import json
import os
from time import time

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class ThroughputTracker:
    """
    Measure the sustained training rate of a run and persist it.

    Call :meth:`update` once per training step; it is a couple of Python operations and costs
    nothing at the few-it/s rates this code trains at. Call :meth:`write` when training ends.

    The tracker bins steps into fixed-width blocks and stores a rate per block, so the *drift*
    within a job survives into the output rather than being averaged away. That matters: a
    transformer clustering run measured on 2026-08-06 ramped from 5.8 to 8.8 it/s over its first six
    hours while the GCNN runs beside it were flat to within 4%, and a single average hides that
    completely.

    Args:
        start_step (int): step the run resumes from (0 for a fresh run), so a chained job reports
            rates for the steps it actually trained.
        dir_model (str, optional): run directory to write ``throughput.json`` / ``throughput.jsonl``
            into. No files are written when None.
        bin_steps (int, optional): block width. Defaults to 2000, which gives a production run ~90
            bins and still produces several for a short probe — at 10000 a 1300-step validation run
            binned to nothing and reported no drift at all, which is the one thing it must not do.
    """

    def __init__(self, start_step=0, dir_model=None, bin_steps=2000):
        self.start_step = int(start_step)
        self.dir_model = dir_model
        self.bin_steps = int(bin_steps)

        self._t0 = time()
        self._bin_step0 = self.start_step
        self._bin_t0 = self._t0
        self._last_step = self.start_step
        self.bins = []

    def update(self, step):
        """Record `step` as completed and close the current bin if it is full."""
        self._last_step = step
        if step - self._bin_step0 >= self.bin_steps:
            now = time()
            d_step = step - self._bin_step0
            d_t = now - self._bin_t0
            if d_t > 0:
                self.bins.append(
                    {
                        "step": step,
                        "wall_s": round(now - self._t0, 1),
                        "it_per_s": round(d_step / d_t, 4),
                    }
                )
            self._bin_step0 = step
            self._bin_t0 = now

    def summary(self):
        """
        Return the throughput summary as a dict.

        ``sustained_it_per_s`` is the number to size the next run with: it is measured over the
        second half of this job, which is past compilation and past any dataloader ramp.
        ``cumulative_it_per_s`` is the tqdm-equivalent figure and is reported only so the gap
        between the two is visible.
        """
        elapsed = time() - self._t0
        n_trained = self._last_step - self.start_step
        out = {
            "start_step": self.start_step,
            "end_step": self._last_step,
            "steps_trained": n_trained,
            "train_seconds": round(elapsed, 1),
            "cumulative_it_per_s": round(n_trained / elapsed, 4) if elapsed > 0 else None,
            "sustained_it_per_s": None,
            "first_bin_it_per_s": None,
            "last_bin_it_per_s": None,
            "drift_percent": None,
            "n_bins": len(self.bins),
            "bin_steps": self.bin_steps,
        }
        if self.bins:
            rates = [b["it_per_s"] for b in self.bins]
            second_half = rates[len(rates) // 2 :]
            out["sustained_it_per_s"] = round(sum(second_half) / len(second_half), 4)
            out["first_bin_it_per_s"] = rates[0]
            out["last_bin_it_per_s"] = rates[-1]
            if rates[0] > 0:
                out["drift_percent"] = round(100.0 * (rates[-1] / rates[0] - 1.0), 1)
        else:
            # too short to bin (a probe, or a job that died early): fall back to the cumulative rate
            out["sustained_it_per_s"] = out["cumulative_it_per_s"]
        return out

    def write(self):
        """Log the summary and, if `dir_model` was given, persist it. Returns the summary dict."""
        s = self.summary()
        sustained = s["sustained_it_per_s"]
        cumulative = s["cumulative_it_per_s"]
        if sustained is not None and cumulative is not None:
            LOGGER.info(
                f"Throughput: {sustained:.3f} it/s sustained (second half of this job), "
                f"{cumulative:.3f} it/s cumulative including compilation"
            )
            if s["drift_percent"] is not None and abs(s["drift_percent"]) >= 10.0:
                LOGGER.warning(
                    f"Step rate drifted {s['drift_percent']:+.1f}% across this job "
                    f"({s['first_bin_it_per_s']:.3f} -> {s['last_bin_it_per_s']:.3f} it/s). "
                    "A rate measured early in a run does not describe this configuration; size it "
                    "with a wall-clock budget rather than a fixed step count."
                )

        if self.dir_model is not None:
            try:
                with open(os.path.join(self.dir_model, "throughput.json"), "w") as f:
                    json.dump(s, f, indent=2)
                # append so a chained job leaves its own trace instead of overwriting job 1's
                with open(os.path.join(self.dir_model, "throughput.jsonl"), "a") as f:
                    for b in self.bins:
                        f.write(json.dumps(b) + "\n")
            except OSError as err:  # never let bookkeeping kill a finished run
                LOGGER.warning(f"Could not write throughput files to {self.dir_model}: {err}")
        return s


class WallClockBudget:
    """
    Bound a run by elapsed training wall-clock instead of by a fixed step count.

    WHY THIS EXISTS, AND WHY MEASURING FIRST IS NOT ENOUGH. Sizing `n_steps` wrong is asymmetric:
    oversize and the cosine never anneals and `run_evaluation.py` / `run_inference.py` never run, so
    a whole allocation produces nothing scorable; undersize and the leftover wall is simply wasted.
    The obvious fix -- measure the rate in a short probe, then fix `n_steps` -- was tried for
    bench_v7 and is not sound, for a reason visible in that round's own logs:

    * the six single-probe runs finished 1.0-1.7 h before their 12 h wall, because the *budget
      constant* (usable training seconds per job) was a guess rather than a measurement;
    * one arm (transformer / clustering) ramped from 5.8 to 8.8 it/s over its first six hours, so no
      window anywhere in its first 20 k steps described the run: every one of them under-predicted
      the whole-run rate by 21-25%;
    * and a standalone probe of that same configuration did *not* show the ramp, so the drift is not
      a property of the config that any amount of pre-measurement could have pinned down.

    A wall-clock budget sidesteps all of it. The run trains until the budget is spent, whatever rate
    it happens to achieve, so contention, node-to-node variation, restarts and ramps are all
    absorbed; the annealing tail and the eval tail are guaranteed by construction. `n_steps` becomes
    an output of the run rather than an input to it.

    The cost, which is real: two runs of the same config no longer take the same number of steps, so
    this is equal-WALL benchmarking, not equal-SAMPLE. That is already what a fixed-wall round
    measures -- in bench_v7 the ConvNeXt arm got 18% more steps than the classic one at equal wall --
    but it does mean a controlled sample-budget comparison should keep using a fixed `n_steps`.

    Chained jobs share one budget: `consumed_seconds` carries what earlier jobs already spent, so the
    cosine continues along a single global curve instead of restarting.

    Args:
        total_seconds (float): training seconds for the whole run, summed across a chain.
        job_seconds (float, optional): training seconds this job may spend before stopping to hand
            over to the next one. Defaults to `total_seconds` (a single-job run).
        consumed_seconds (float, optional): training seconds already spent by earlier jobs.
    """

    def __init__(self, total_seconds, job_seconds=None, consumed_seconds=0.0):
        self.total_seconds = float(total_seconds)
        self.job_seconds = float(job_seconds) if job_seconds is not None else self.total_seconds
        self.consumed_seconds = float(consumed_seconds)
        if self.total_seconds <= 0:
            raise ValueError(f"wall_budget_seconds must be positive, got {self.total_seconds}")
        self._t0 = None
        # wall-clock offset at which warmup ended; the cosine spans budget MINUS this, so that a
        # step-based warmup and a time-based decay meet without a discontinuity. Restored for a chain.
        self.warmup_end_seconds = None

    def start(self):
        """Mark the beginning of this job's training loop."""
        self._t0 = time()
        return self

    @property
    def job_elapsed(self):
        """Training seconds spent by THIS job."""
        return 0.0 if self._t0 is None else time() - self._t0

    @property
    def elapsed(self):
        """Training seconds spent across the whole run, including earlier jobs in a chain."""
        return self.consumed_seconds + self.job_elapsed

    @property
    def fraction(self):
        """Fraction of the total budget consumed, clipped to [0, 1]."""
        return min(max(self.elapsed / self.total_seconds, 0.0), 1.0)

    @property
    def exhausted(self):
        """True once the whole run's budget is spent -- training is finished."""
        return self.elapsed >= self.total_seconds

    @property
    def job_exhausted(self):
        """True once this job's share is spent -- stop and let the next job in the chain resume."""
        return self.job_elapsed >= self.job_seconds

    def projected_total_steps(self, start_step, step):
        """Extrapolate the final step count from the rate so far, for the progress bar and logs."""
        if self._t0 is None or step <= start_step:
            return None
        rate = (step - start_step) / max(self.job_elapsed, 1e-9)
        return int(start_step + rate * (self.total_seconds - self.consumed_seconds))

    def state(self):
        """Serializable progress, to be written after every checkpoint so a chain can resume."""
        return {
            "consumed_seconds": round(self.elapsed, 1),
            "total_seconds": self.total_seconds,
            "warmup_end_seconds": self.warmup_end_seconds,
        }


def read_budget_state(dir_model):
    """Read the persisted :class:`WallClockBudget` progress of an earlier job, or None."""
    path = os.path.join(dir_model, "budget_progress.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as err:
        LOGGER.warning(f"Could not read {path} ({err}); treating the budget as unconsumed")
        return None


def write_budget_state(dir_model, budget):
    """Persist :class:`WallClockBudget` progress so the next job in the chain continues the curve."""
    path = os.path.join(dir_model, "budget_progress.json")
    try:
        with open(path, "w") as f:
            json.dump(budget.state(), f, indent=2)
    except OSError as err:
        LOGGER.warning(f"Could not write {path}: {err}")
