# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Diff a benchmark config against its parent, and check a whole round for the traps that are silent.

Benchmark rounds are run under a one-knob discipline: every config differs from a named parent in
exactly one field, because a two-knob arm cannot be attributed to either knob afterwards. Verifying
that means flattening two nested YAML trees and diffing them -- which was re-typed as a throwaway
heredoc at least five times across the bench_v7-v10 rounds, each with a slightly different
flattener. This is that flattener, once.

`check` deliberately asserts nothing round-specific. Trunk width, pooling mode and step budget are
different in every round, and a checker that hardcodes this round's values is wrong by the next one.
What it does check is what is silent and generic:

  * `n_steps: auto` without `wall_budget_seconds` -- the contract in run_training.py:643. The run
    raises there, but only after the job has been queued, scheduled and started.
  * `job_budget_seconds > wall_budget_seconds` -- not caught anywhere: the per-job share simply never
    binds, job 1 runs to the full budget and the chain silently does twice the intended work.
  * bare scientific notation. `1e-3` in YAML is a STRING, not a float (it needs `1.0e-3`), and a
    string learning rate fails deep inside the optimizer or, worse, is silently accepted somewhere
    that stringifies.
  * keys present in some files of a round but not others -- nearly always an omission during a
    copy-edit, and it changes what the arm is testing.

and then it prints the key that VARY across the round, which is the one-knob discipline made
visible: one differing key per arm is what the round is supposed to look like.

Typical use::

    python -m deep_lss.utils.config_check diff configs/deepsphere/dev/combined/bench_v8/mean_std.yaml \\
        configs/deepsphere/dev/combined/bench_v9/moments.yaml
    python -m deep_lss.utils.config_check check configs/deepsphere/dev/*/bench_v10/*.yaml
"""

import argparse
import os
import re

import yaml

# A float written without a decimal point or a sign on the exponent is not a float to YAML 1.1 --
# it is a string, and nothing downstream necessarily complains.
_BARE_EXPONENT = re.compile(r"^[-+]?\d+(\.\d*)?[eE][-+]?\d+$")

_ABSENT = "<absent>"


def flatten(node, prefix=""):
    """Flatten a nested config into a single dict of dotted keys, so two trees can be diffed."""
    flat = {}
    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def load(path):
    """Parse one config, failing with the file name attached rather than a bare parser error."""
    with open(path) as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: not parseable as YAML -- {exc}")
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: parsed as {type(cfg).__name__}, expected a mapping")
    return cfg


def diff(parent_path, child_path):
    """Every key on which `child` differs from `parent`, as (key, parent_value, child_value)."""
    parent, child = flatten(load(parent_path)), flatten(load(child_path))
    return [
        (key, parent.get(key, _ABSENT), child.get(key, _ABSENT))
        for key in sorted(set(parent) | set(child))
        if parent.get(key, _ABSENT) != child.get(key, _ABSENT)
    ]


def format_diff(parent_path, child_path, changes):
    """Render a parent->child diff, with the knob count stated rather than left to be counted."""
    lines = [f"{parent_path}", f"  -> {child_path}", ""]
    if not changes:
        lines.append("  identical -- these two configs describe the same run")
        return "\n".join(lines)
    width = max(len(key) for key, _, _ in changes)
    for key, before, after in changes:
        lines.append(f"  {key:<{width}}  {before!r} -> {after!r}")
    lines += ["", f"  {len(changes)} key(s) differ"]
    if len(changes) > 1:
        lines.append(
            "  MORE THAN ONE KNOB. If these are meant to be one contrast, the result cannot be "
            "attributed to either change; state which knob the arm is testing."
        )
    return "\n".join(lines)


def _problems(path, flat):
    """The silent traps, checked per file. Returns a list of human-readable strings."""
    found = []
    n_steps = flat.get("training.n_steps")
    wall = flat.get("training.wall_budget_seconds")
    job = flat.get("training.job_budget_seconds")

    if n_steps == "auto" and wall is None:
        found.append("training.n_steps: auto requires training.wall_budget_seconds (run_training.py:643)")
    if isinstance(n_steps, int) and n_steps <= 0:
        found.append(f"training.n_steps must be a positive cap or 'auto', got {n_steps!r}")
    if isinstance(wall, int) and isinstance(job, int) and job > wall:
        found.append(
            f"training.job_budget_seconds ({job}) exceeds training.wall_budget_seconds ({wall}); the "
            f"per-job share never binds, so job 1 runs the whole budget and a chain doubles it"
        )
    for key, value in flat.items():
        if isinstance(value, str) and _BARE_EXPONENT.match(value):
            found.append(
                f"{key}: {value!r} parsed as a STRING, not a float -- write it as "
                f"{float(value):.1e}".replace("e-0", "e-").replace("e+0", "e+")
            )
    return [f"{os.path.basename(path)}: {p}" for p in found]


def check(paths):
    """Parse a group of configs, collect the silent traps, and report what varies across the group."""
    flats, problems = {}, []
    for path in paths:
        flat = flatten(load(path))
        flats[path] = flat
        problems += _problems(path, flat)

    # An asymmetric key set is a HEURISTIC, not a contract: an optional key with a code-side default
    # is legitimately absent from the arms that do not use it (`conv_widen` on the non-U-net arms).
    # It is reported as a note so a genuine copy-edit omission is visible, but it does not fail the
    # check -- a checker that cries wolf on every round stops being read.
    all_keys = set().union(*flats.values()) if flats else set()
    notes = [
        f"{key}: absent from {sorted(os.path.basename(p) for p in flats if key not in flats[p])} but "
        f"set in the others -- intended (an option those arms do not use) or an omission?"
        for key in sorted(all_keys)
        if any(key not in flat for flat in flats.values())
    ]

    varying = sorted(key for key in all_keys if len({repr(flat.get(key, _ABSENT)) for flat in flats.values()}) > 1)
    return {"flats": flats, "varying": varying, "problems": problems, "notes": notes}


def format_check(result, paths):
    """Render the round summary: what varies, and what is wrong."""
    flats, varying = result["flats"], result["varying"]
    lines = [f"{len(paths)} config(s) parsed", ""]

    if not varying:
        lines.append("every key is identical across these configs")
    else:
        names = [os.path.basename(p) for p in paths]
        name_width = max(max(len(n) for n in names), 12) + 2
        key_width = max(len(k) for k in varying) + 2
        lines.append("keys that VARY across the group (one differing key per arm is the target):")
        lines.append("")
        lines.append(f"{'key':<{key_width}}" + "".join(f"{n:>{name_width}}" for n in names))
        lines.append("-" * (key_width + name_width * len(names)))
        for key in varying:
            cells = "".join(f"{str(flats[p].get(key, _ABSENT)):>{name_width}}" for p in paths)
            lines.append(f"{key:<{key_width}}{cells}")

    if result["notes"]:
        lines += ["", f"NOTES ({len(result['notes'])}) -- check, do not assume:"]
        lines += [f"  {n}" for n in result["notes"]]

    lines.append("")
    if result["problems"]:
        lines.append(f"PROBLEMS ({len(result['problems'])}):")
        lines += [f"  {p}" for p in result["problems"]]
    else:
        lines.append("no problems found (budget contract and numeric literals are consistent)")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    diff_parser = subparsers.add_parser("diff", help="show how a child config differs from its parent")
    diff_parser.add_argument("parent", help="the config the child is one knob away from")
    diff_parser.add_argument("child", help="the config being checked")

    check_parser = subparsers.add_parser("check", help="validate a group of configs and show what varies")
    check_parser.add_argument("configs", nargs="+", help="config paths (a shell glob over a round)")

    args = parser.parse_args(argv)
    if args.command == "diff":
        changes = diff(args.parent, args.child)
        print(format_diff(args.parent, args.child, changes))
        return 0
    paths = sorted(args.configs)
    result = check(paths)
    print(format_check(result, paths))
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
