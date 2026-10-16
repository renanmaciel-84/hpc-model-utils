"""ADR-094/R80/R89: the job-log comparator (ticket-037).

Defines "same log" once, as an executable. Two logs, the telemetry-off
run and the telemetry-on run of the same job, are the same when they are
equal after a closed, named set of masks has rewritten their volatile
tokens, and when the on log carries no line that only telemetry could
have written.

Two modes. ``strict`` (the fake-Slurm K2 proof, R80) masks the
``HPCMU_START ... at=`` timestamp, the CLI asctime prefix and the
workspace root. ``prd`` (the operator gate, R67/R89) adds the named field
masks of ``PRD_MASKS``, grouped by source: ``ids`` (the job id on the
``HPCMU_START`` line), ``slurm`` (the ``srun:`` and ``slurmstepd:`` line
shapes of the ticket-004 probe record, plus the job-level ``CANCELLED
AT`` variant) and ``cobre`` (the format strings of
``crates/cobre-cli/src/progress.rs`` and ``summary.rs`` at the pinned
v0.17.0). A mask rewrites fields inside a line it recognizes and never
drops a line. NEWAVE and DECOMP masks are added here at ticket-059, under
ADR-094's rework trigger.

A signature check runs first in every mode: an on-side line that starts
with ``srun:`` or ``slurmstepd:``, or contains ``hpcmu-telemetry``,
``infra.telemetry`` or ``Traceback``, is rejected whenever the on log
holds that masked line more often than the off log. Three shapes then
decide the rest: ``exact``, ``prefix-tail`` (V4/V5: the common prefix
plus a tail check) and ``order-only`` (V2/V3: order differences are
tolerated).

Stdlib only, so the operator can run it on whatever host holds the two
logs, with no checkout venv: ``python3 -I deploy/modelops/
compare_job_logs.py [--mode {strict,prd}] [--shape {exact,prefix-tail,
order-only}] [--off-root PATH] [--on-root PATH] OFF ON``. Exit 0 for the
same log, 1 for a difference, 2 for a usage or read error.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import zip_longest
from pathlib import Path


class Mode(StrEnum):
    STRICT = "strict"
    PRD = "prd"


class Shape(StrEnum):
    EXACT = "exact"
    PREFIX_TAIL = "prefix-tail"
    ORDER_ONLY = "order-only"


@dataclass(frozen=True)
class Mask:
    name: str
    source: str
    pattern: re.Pattern[str]
    repl: str | Callable[[re.Match[str]], str]


@dataclass(frozen=True)
class Difference:
    reason: str
    line: int
    off: str | None
    on: str | None


_DUR = r"(?:\d+h \d+m|\d+m \d+s|\d+\.\d+s|\d+ms)"
_PCT = r"(?:<1%|\d+%)"
_HMS = r"\d+:\d\d:\d\d"

_HMS_RE = re.compile(_HMS)
_DUR_RE = re.compile(rf"\b{_DUR}")
_PCT_RE = re.compile(_PCT)
_FWD_BWD_RE = re.compile(r"fwd: \d+ms / bwd: \d+ms")
_LP_RE = re.compile(r"LP: \d+\.\dms avg")


def _rewrite(
    *subs: tuple[re.Pattern[str], str],
) -> Callable[[re.Match[str]], str]:
    def repl(match: re.Match[str]) -> str:
        text = match.group(0)
        for pattern, replacement in subs:
            text = pattern.sub(replacement, text)
        return text

    return repl


def _slurm_line(match: re.Match[str]) -> str:
    nodes = ", nodes <nodes>" if match["nodes"] else ""
    return f"  SLURM:     job <jobid>{nodes}{match['cpus'] or ''}"


STRICT_MASKS: tuple[Mask, ...] = (
    Mask(
        "hpcmu-start-at",
        "hpcmu",
        re.compile(
            r"^(HPCMU_START job=\S+ restart=\S+ at=)"
            r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$"
        ),
        r"\g<1><TS>",
    ),
    Mask(
        "cli-asctime",
        "hpcmu",
        re.compile(
            r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} "
            r"(?=(?:DEBUG|INFO|WARNING|ERROR|CRITICAL) )"
        ),
        "<ASCTIME> ",
    ),
)

PRD_MASKS: tuple[Mask, ...] = (
    Mask(
        "hpcmu-start-job",
        "ids",
        re.compile(r"^(HPCMU_START job=)\S+"),
        r"\g<1><JOBID>",
    ),
    Mask(
        "slurm-task-killed",
        "slurm",
        re.compile(r"^srun: error: \S+: task \d+: Killed$"),
        "srun: error: <host>: task <taskid>: Killed",
    ),
    Mask(
        "slurm-step-cancelled",
        "slurm",
        re.compile(
            r"^slurmstepd: error: \*\*\* STEP \d+\.\S+ ON \S+ "
            r"CANCELLED AT \S+( DUE TO TIME LIMIT)? \*\*\*$"
        ),
        "slurmstepd: error: *** STEP <jobid>.<stepid> ON <host> "
        r"CANCELLED AT <ts>\g<1> ***",
    ),
    Mask(
        "slurm-job-cancelled",
        "slurm",
        re.compile(
            r"^slurmstepd: error: \*\*\* JOB \d+ ON \S+ "
            r"CANCELLED AT \S+( DUE TO TIME LIMIT)? \*\*\*$"
        ),
        "slurmstepd: error: *** JOB <jobid> ON <host> "
        r"CANCELLED AT <ts>\g<1> ***",
    ),
    Mask(
        "cobre-training",
        "cobre",
        re.compile(r"^Training +\d+/\d+ iter  .*$"),
        _rewrite((_FWD_BWD_RE, "fwd: <MS> / bwd: <MS>"), (_HMS_RE, "<HMS>")),
    ),
    Mask(
        "cobre-simulation",
        "cobre",
        re.compile(r"^Simulation \d+/\d+ scenarios  .*$"),
        _rewrite((_LP_RE, "LP: <MS> avg"), (_HMS_RE, "<HMS>")),
    ),
    Mask(
        "cobre-setup",
        "cobre",
        re.compile(
            r"^(  (?:Load|Stochastic fit|Production fit|Evaporation fit"
            rf"|Broadcast): +){_DUR}$"
        ),
        r"\g<1><DUR>",
    ),
    Mask(
        "cobre-training-complete",
        "cobre",
        re.compile(rf"^(Training complete in ){_DUR}"),
        r"\g<1><DUR>",
    ),
    Mask(
        "cobre-simulation-complete",
        "cobre",
        re.compile(rf"^(Simulation complete in ){_DUR}"),
        r"\g<1><DUR>",
    ),
    Mask(
        "cobre-avg-iter",
        "cobre",
        re.compile(r"^(  Avg iter:     )\d+ms$"),
        r"\g<1><MS>",
    ),
    Mask(
        "cobre-avg-scenario",
        "cobre",
        re.compile(r"^(  Avg/scenario: )\d+\.\d{3}s$"),
        r"\g<1><DUR>",
    ),
    Mask(
        "cobre-time-split",
        "cobre",
        re.compile(
            r"^(?:  Time split:   (?:Forward |Solver )"
            r"| {16}(?:Backward |Serial   |Other  )).*$"
        ),
        _rewrite((_DUR_RE, "<DUR>"), (_PCT_RE, "<PCT>")),
    ),
    Mask(
        "cobre-output-written",
        "cobre",
        re.compile(r"^(Output written to .*/ )\(\d+\.\ds\)$"),
        r"\g<1>(<DUR>)",
    ),
    Mask(
        "cobre-host",
        "cobre",
        re.compile(r"^(  Host:      )\S+$"),
        r"\g<1><host>",
    ),
    Mask(
        "cobre-layout-host",
        "cobre",
        re.compile(r"^(  Layout:    \d+ ranks? on )\S+$"),
        r"\g<1><host>",
    ),
    Mask(
        "cobre-rank-host",
        "cobre",
        re.compile(r"^    \S+(: ranks )"),
        r"    <host>\g<1>",
    ),
    Mask(
        "cobre-slurm",
        "cobre",
        re.compile(
            r"^  SLURM:     job \d+(?P<nodes>, nodes .+?)?"
            r"(?P<cpus>, \d+ CPUs/task)?$"
        ),
        _slurm_line,
    ),
)

_SIGNATURE_PREFIXES = ("srun:", "slurmstepd:")
_SIGNATURE_MARKERS = ("hpcmu-telemetry", "infra.telemetry", "Traceback")
_ANCHOR_RE = re.compile(
    r"^slurmstepd: error: \*\*\* (?:JOB|STEP) .* CANCELLED AT "
)


def read_lines(path: Path) -> list[str]:
    text = path.read_bytes().decode("utf-8", errors="surrogateescape")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def mask_line(line: str, *, mode: Mode, root: str | None) -> str:
    if root:
        line = line.replace(root, "<WS>")
    masks = STRICT_MASKS + PRD_MASKS if mode is Mode.PRD else STRICT_MASKS
    for mask in masks:
        line = mask.pattern.sub(mask.repl, line)
    return line


def _is_signature(line: str) -> bool:
    return line.startswith(_SIGNATURE_PREFIXES) or any(
        marker in line for marker in _SIGNATURE_MARKERS
    )


def _signature(off: Sequence[str], on: Sequence[str]) -> Difference | None:
    off_count = Counter(off)
    seen: Counter[str] = Counter()
    for n, line in enumerate(on, 1):
        if _is_signature(line):
            seen[line] += 1
            if seen[line] > off_count[line]:
                return Difference("signature", n, None, line)
    return None


def _exact(off: Sequence[str], on: Sequence[str]) -> Difference | None:
    for n, (off_line, on_line) in enumerate(zip_longest(off, on), 1):
        if off_line != on_line:
            both = off_line is not None and on_line is not None
            return Difference(
                "mismatch" if both else "length", n, off_line, on_line
            )
    return None


def _order_only(off: Sequence[str], on: Sequence[str]) -> Difference | None:
    off_count, on_count = Counter(off), Counter(on)
    if off_count == on_count:
        return None
    for n, line in enumerate(on, 1):
        if on_count[line] > off_count[line]:
            return Difference("count", n, None, line)
    # unequal counters with no on-side excess leave an off-side excess
    n, line = next(
        (n, line)
        for n, line in enumerate(off, 1)
        if off_count[line] > on_count[line]
    )
    return Difference("count", n, line, None)


def _last_anchor(lines: Sequence[str]) -> int | None:
    return next(
        (i for i in reversed(range(len(lines))) if _ANCHOR_RE.match(lines[i])),
        None,
    )


def _prefix_tail(off: Sequence[str], on: Sequence[str]) -> Difference | None:
    off_anchor, on_anchor = _last_anchor(off), _last_anchor(on)
    if off_anchor is None or on_anchor is None:
        return Difference(
            "missing anchor",
            0,
            None if off_anchor is None else off[off_anchor],
            None if on_anchor is None else on[on_anchor],
        )
    tail = _exact(off[off_anchor:], on[on_anchor:])
    if tail is not None:
        return replace(tail, reason="tail", line=on_anchor + tail.line)
    shared = min(off_anchor, on_anchor)
    prefix = _exact(off[:shared], on[:shared])
    if prefix is not None:
        return replace(prefix, reason="prefix")
    return None


def compare(
    off: Sequence[str],
    on: Sequence[str],
    *,
    mode: Mode,
    shape: Shape,
    off_root: str | None,
    on_root: str | None,
) -> Difference | None:
    masked_off = [mask_line(x, mode=mode, root=off_root) for x in off]
    masked_on = [mask_line(x, mode=mode, root=on_root) for x in on]
    found = _signature(masked_off, masked_on)
    if found is not None:
        return found
    match shape:
        case Shape.EXACT:
            return _exact(masked_off, masked_on)
        case Shape.ORDER_ONLY:
            return _order_only(masked_off, masked_on)
        case Shape.PREFIX_TAIL:
            return _prefix_tail(masked_off, masked_on)


def _report(difference: Difference) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="backslashreplace")
    off = "<none>" if difference.off is None else difference.off
    on = "<none>" if difference.on is None else difference.on
    sys.stdout.write(
        f"differ: {difference.reason} at line {difference.line}\n"
        f"off: {off}\non: {on}\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare_job_logs",
        description="Compare a telemetry-off and a telemetry-on job log.",
    )
    parser.add_argument(
        "--mode", type=Mode, choices=list(Mode), default=Mode.STRICT
    )
    parser.add_argument(
        "--shape", type=Shape, choices=list(Shape), default=Shape.EXACT
    )
    parser.add_argument("--off-root", metavar="PATH")
    parser.add_argument("--on-root", metavar="PATH")
    parser.add_argument("off", metavar="OFF", type=Path)
    parser.add_argument("on", metavar="ON", type=Path)
    args = parser.parse_args(argv)

    logs: list[list[str]] = []
    for path in (args.off, args.on):
        try:
            logs.append(read_lines(path))
        except OSError as exc:
            sys.stderr.write(
                f"compare_job_logs: cannot read {path}: {exc.strerror}\n"
            )
            return 2
    difference = compare(
        logs[0],
        logs[1],
        mode=args.mode,
        shape=args.shape,
        off_root=args.off_root,
        on_root=args.on_root,
    )
    if difference is None:
        sys.stdout.write("same\n")
        return 0
    _report(difference)
    return 1


if __name__ == "__main__":
    sys.exit(main())
