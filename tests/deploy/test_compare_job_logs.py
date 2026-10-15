"""Tests for deploy.modelops.compare_job_logs (ticket-037, ADR-094).

Acceptance criteria, selected with ``-k <keyword>``: ``strict`` (the
closed strict mask set), ``signature`` (the on-side signature check),
``prd`` (the prd field masks), ``shape`` (exact, prefix-tail and
order-only). The CLI exit codes and the ``python3 -I`` run are the
``cli`` tests. Every log line is a placeholder: hosts ``node-a`` ..
``node-y``, job ids ``1000``/``1234``/``5678`` and synthetic stamps.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

from deploy.modelops.compare_job_logs import (
    PRD_MASKS,
    STRICT_MASKS,
    Difference,
    Mode,
    Shape,
    compare,
    main,
    mask_line,
    read_lines,
)

_SCRIPT = Path(__file__).parents[2] / "deploy/modelops/compare_job_logs.py"


def _compare(
    off: list[str],
    on: list[str],
    *,
    mode: Mode = Mode.PRD,
    shape: Shape = Shape.EXACT,
    off_root: str | None = None,
    on_root: str | None = None,
) -> Difference | None:
    return compare(
        off, on, mode=mode, shape=shape, off_root=off_root, on_root=on_root
    )


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------- strict

_START = "HPCMU_START job=1000 restart=0 at={at}"
_ASCTIME = "{ts} INFO hpc_model_utils.core.lifecycle.finalize: exited 0"
_ROOT_LINE = "model input read from {root}/deck/input.dat"


def _strict_log(at: str, ts: str, root: str) -> list[str]:
    return [
        _START.format(at=at),
        _ASCTIME.format(ts=ts),
        _ROOT_LINE.format(root=root),
        "steady line",
    ]


def test_strict_masks_volatile_tokens_returns_none() -> None:
    off = _strict_log(
        "2000-01-01T00:00:00Z", "2000-01-01 00:00:01,001", "/ws/a"
    )
    on = _strict_log("2000-01-01T00:09:09Z", "2000-01-01 00:09:10,987", "/ws/b")
    assert off != on
    assert (
        _compare(off, on, mode=Mode.STRICT, off_root="/ws/a", on_root="/ws/b")
        is None
    )


@pytest.mark.parametrize(
    ("field", "old", "new"),
    [("job", "job=1000", "job=1001"), ("restart", "restart=0", "restart=1")],
)
def test_strict_job_and_restart_are_not_masked(
    field: str, old: str, new: str
) -> None:
    off = _strict_log(
        "2000-01-01T00:00:00Z", "2000-01-01 00:00:01,001", "/ws/a"
    )
    on = _strict_log("2000-01-01T00:09:09Z", "2000-01-01 00:09:10,987", "/ws/b")
    on[0] = on[0].replace(old, new)
    found = _compare(
        off, on, mode=Mode.STRICT, off_root="/ws/a", on_root="/ws/b"
    )
    assert found is not None
    assert (found.reason, found.line) == ("mismatch", 1)
    assert found.off == "HPCMU_START job=1000 restart=0 at=<TS>"
    assert new in (found.on or ""), field


def test_strict_other_lines_and_unmatched_root_are_kept() -> None:
    line = "2000-01-01 00:00:01,001 TRACE x: y"
    assert mask_line(line, mode=Mode.STRICT, root="/ws/a") == line
    assert mask_line(line, mode=Mode.STRICT, root=None) == line
    asctime = "2000-01-01 00:00:01,001 WARNING x: y"
    assert (
        mask_line(asctime, mode=Mode.STRICT, root=None)
        == "<ASCTIME> WARNING x: y"
    )


def test_strict_mode_does_not_apply_prd_masks() -> None:
    line = "srun: error: node-a: task 0: Killed"
    assert mask_line(line, mode=Mode.STRICT, root=None) == line
    assert mask_line(line, mode=Mode.PRD, root=None) != line


def test_strict_root_is_masked_before_the_field_masks() -> None:
    root = "/ws/10:20:30"
    line = f"[00:00:01] {root}/x"
    training = (
        "Training   1/9 iter  LB: 1.0e0  UB: 2.0e0  gap: 5.0%  "
        f"fwd: 1ms / bwd: 2ms  [00:00:01 < 00:00:09] {root}/x"
    )
    assert mask_line(line, mode=Mode.PRD, root=root) == "[00:00:01] <WS>/x"
    assert mask_line(training, mode=Mode.PRD, root=root).endswith(
        "[<HMS> < <HMS>] <WS>/x"
    )


def test_strict_mask_set_is_closed() -> None:
    assert [m.name for m in STRICT_MASKS] == ["hpcmu-start-at", "cli-asctime"]
    assert {m.source for m in STRICT_MASKS} == {"hpcmu"}


def test_strict_read_lines_keeps_carriage_return_and_bad_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "log"
    path.write_bytes(b"a\r\nb\n\xff\n\n")
    assert read_lines(path) == ["a\r", "b", "\udcff", ""]
    path.write_bytes(b"x")
    assert read_lines(path) == ["x"]
    path.write_bytes(b"")
    assert read_lines(path) == []


# ------------------------------------------------------------- signature

_SIGNATURE_LINES = [
    pytest.param("srun: error: node-a: task 0: Killed", id="srun"),
    pytest.param("slurmstepd: error: x", id="slurmstepd"),
    pytest.param("hpcmu-telemetry: x", id="marker"),
    pytest.param("x infra.telemetry x", id="module"),
    pytest.param("Traceback (most recent call last):", id="traceback"),
]
_SIGNATURE_BASE = [
    "HPCMU_START job=1000 restart=0 at=2000-01-01T00:00:00Z",
    "iteration 1",
    "slurmstepd: error: *** JOB 1000 ON node-a CANCELLED AT 2000-01-01T00:00:00 ***",
]


@pytest.mark.parametrize("shape", list(Shape))
@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize("extra", _SIGNATURE_LINES)
def test_signature_rejects_an_appended_line(
    extra: str, mode: Mode, shape: Shape
) -> None:
    on = [*_SIGNATURE_BASE, extra]
    found = _compare(_SIGNATURE_BASE, on, mode=mode, shape=shape)
    assert found is not None
    assert found.reason == "signature"
    assert (found.line, found.off) == (len(on), None)


@pytest.mark.parametrize("shape", list(Shape))
@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize("extra", _SIGNATURE_LINES)
def test_signature_line_on_both_sides_is_not_a_signature_difference(
    extra: str, mode: Mode, shape: Shape
) -> None:
    both = [*_SIGNATURE_BASE, extra]
    assert _compare(both, list(both), mode=mode, shape=shape) is None


def test_signature_counts_only_the_on_side_excess() -> None:
    srun = "srun: forcing job termination"
    assert _compare(["a", srun], ["a"]) == Difference("length", 2, srun, None)
    found = _compare(["a", srun], [srun, "a", srun])
    assert found == Difference("signature", 3, None, srun)


def test_signature_check_runs_on_the_masked_lines() -> None:
    off = ["srun: error: node-a: task 0: Killed"]
    on = ["srun: error: node-x: task 1: Killed"]
    assert _compare(off, on, mode=Mode.PRD) is None
    found = _compare(off, on, mode=Mode.STRICT)
    assert found is not None
    assert found.reason == "signature"


# ------------------------------------------------------------------- prd


class _MaskCase(NamedTuple):
    name: str
    base: str
    perturbed: str
    expected: str


_TRAIN = (
    "Training   3/50 iter  LB: 1.23450e5  UB: 1.30000e5  gap: 5.0%  "
    "fwd: {fwd} / bwd: {bwd}  {cell}"
)
_SIM = "Simulation 10/100 scenarios  {lp}  {cell}"
_CANCEL = "slurmstepd: error: *** {kind} ON {host} CANCELLED AT {ts}{limit} ***"

_MASK_CASES = (
    _MaskCase(
        "hpcmu-start-job",
        "HPCMU_START job=1234 restart=0 at=2000-01-01T00:00:00Z",
        "HPCMU_START job=5678 restart=0 at=2000-01-01T00:09:09Z",
        "HPCMU_START job=<JOBID> restart=0 at=<TS>",
    ),
    _MaskCase(
        "slurm-task-killed",
        "srun: error: node-a: task 0: Killed",
        "srun: error: node-x: task 1: Killed",
        "srun: error: <host>: task <taskid>: Killed",
    ),
    _MaskCase(
        "slurm-step-cancelled",
        _CANCEL.format(
            kind="STEP 1234.9",
            host="node-a",
            ts="2000-01-01T00:00:00",
            limit="",
        ),
        _CANCEL.format(
            kind="STEP 5678.3",
            host="node-x",
            ts="2000-01-01T00:07:31",
            limit="",
        ),
        "slurmstepd: error: *** STEP <jobid>.<stepid> ON <host> "
        "CANCELLED AT <ts> ***",
    ),
    _MaskCase(
        "slurm-step-cancelled",
        _CANCEL.format(
            kind="STEP 1234.batch",
            host="node-b",
            ts="2000-01-01T00:00:00",
            limit=" DUE TO TIME LIMIT",
        ),
        _CANCEL.format(
            kind="STEP 5678.batch",
            host="node-y",
            ts="2000-01-01T00:07:31",
            limit=" DUE TO TIME LIMIT",
        ),
        "slurmstepd: error: *** STEP <jobid>.<stepid> ON <host> "
        "CANCELLED AT <ts> DUE TO TIME LIMIT ***",
    ),
    _MaskCase(
        "slurm-job-cancelled",
        _CANCEL.format(
            kind="JOB 1234", host="node-a", ts="2000-01-01T00:00:00", limit=""
        ),
        _CANCEL.format(
            kind="JOB 5678", host="node-x", ts="2000-01-01T00:07:31", limit=""
        ),
        "slurmstepd: error: *** JOB <jobid> ON <host> CANCELLED AT <ts> ***",
    ),
    _MaskCase(
        "slurm-job-cancelled",
        _CANCEL.format(
            kind="JOB 1234",
            host="node-b",
            ts="2000-01-01T00:00:00",
            limit=" DUE TO TIME LIMIT",
        ),
        _CANCEL.format(
            kind="JOB 5678",
            host="node-y",
            ts="2000-01-01T00:07:31",
            limit=" DUE TO TIME LIMIT",
        ),
        "slurmstepd: error: *** JOB <jobid> ON <host> "
        "CANCELLED AT <ts> DUE TO TIME LIMIT ***",
    ),
    _MaskCase(
        "cobre-training",
        _TRAIN.format(fwd="120ms", bwd="340ms", cell="[00:00:05 < 00:01:10]"),
        _TRAIN.format(fwd="99ms", bwd="12ms", cell="[100:02:03 < 100:04:05]"),
        _TRAIN.format(fwd="<MS>", bwd="<MS>", cell="[<HMS> < <HMS>]"),
    ),
    _MaskCase(
        "cobre-training",
        _TRAIN.format(fwd="120ms", bwd="340ms", cell="[00:00:05]"),
        _TRAIN.format(fwd="1ms", bwd="2ms", cell="[00:07:09]"),
        _TRAIN.format(fwd="<MS>", bwd="<MS>", cell="[<HMS>]"),
    ),
    _MaskCase(
        "cobre-simulation",
        _SIM.format(lp="LP: 1.2ms avg", cell="[00:00:05 < 00:01:00]"),
        _SIM.format(lp="LP: 9.9ms avg", cell="[10:11:12 < 10:12:13]"),
        _SIM.format(lp="LP: <MS> avg", cell="[<HMS> < <HMS>]"),
    ),
    _MaskCase(
        "cobre-simulation",
        _SIM.format(lp="", cell="[00:00:05]"),
        _SIM.format(lp="", cell="[00:09:59]"),
        _SIM.format(lp="", cell="[<HMS>]"),
    ),
    _MaskCase(
        "cobre-setup",
        "  Load:            12ms",
        "  Load:            987ms",
        "  Load:            <DUR>",
    ),
    _MaskCase(
        "cobre-setup",
        "  Stochastic fit:  1.5s",
        "  Stochastic fit:  2m 5s",
        "  Stochastic fit:  <DUR>",
    ),
    _MaskCase(
        "cobre-setup",
        "  Production fit:  2m 5s",
        "  Production fit:  1h 5m",
        "  Production fit:  <DUR>",
    ),
    _MaskCase(
        "cobre-setup",
        "  Evaporation fit: 1h 5m",
        "  Evaporation fit: 3ms",
        "  Evaporation fit: <DUR>",
    ),
    _MaskCase(
        "cobre-setup",
        "  Broadcast:       3ms",
        "  Broadcast:       4.5s",
        "  Broadcast:       <DUR>",
    ),
    _MaskCase(
        "cobre-training-complete",
        "Training complete in 12.5s (50 iterations, converged)",
        "Training complete in 1h 5m (50 iterations, converged)",
        "Training complete in <DUR> (50 iterations, converged)",
    ),
    _MaskCase(
        "cobre-simulation-complete",
        "Simulation complete in 3m 12s (100 scenarios)",
        "Simulation complete in 4.2s (100 scenarios)",
        "Simulation complete in <DUR> (100 scenarios)",
    ),
    _MaskCase(
        "cobre-avg-iter",
        "  Avg iter:     120ms",
        "  Avg iter:     87ms",
        "  Avg iter:     <MS>",
    ),
    _MaskCase(
        "cobre-avg-scenario",
        "  Avg/scenario: 0.123s",
        "  Avg/scenario: 12.345s",
        "  Avg/scenario: <DUR>",
    ),
    _MaskCase(
        "cobre-time-split",
        "  Time split:   Forward  12.3s (45%)",
        "  Time split:   Forward  1m 5s (<1%)",
        "  Time split:   Forward  <DUR> (<PCT>)",
    ),
    _MaskCase(
        "cobre-time-split",
        "  Time split:   Forward  12.3s (45%)"
        "   solve 10.1s \u00b7 wait 300ms (2% of phase)",
        "  Time split:   Forward  1h 5m (46%)"
        "   solve 2m 5s \u00b7 wait 12.5s (<1% of phase)",
        "  Time split:   Forward  <DUR> (<PCT>)"
        "   solve <DUR> \u00b7 wait <DUR> (<PCT> of phase)",
    ),
    _MaskCase(
        "cobre-time-split",
        "                Backward 1m 5s (<1%)",
        "                Backward 300ms (12%)",
        "                Backward <DUR> (<PCT>)",
    ),
    _MaskCase(
        "cobre-time-split",
        "                Serial   2h 5m (7%)   bound 1.2s · other 300ms",
        "                Serial   9.9s (8%)   bound 1m 5s · other 2h 5m",
        "                Serial   <DUR> (<PCT>)   bound <DUR> · other <DUR>",
    ),
    _MaskCase(
        "cobre-time-split",
        "  Time split:   Solver 4.2s (80%)",
        "  Time split:   Solver 1m 2s (79%)",
        "  Time split:   Solver <DUR> (<PCT>)",
    ),
    _MaskCase(
        "cobre-time-split",
        "                Other  1.0s (20%)",
        "                Other  2.0s (21%)",
        "                Other  <DUR> (<PCT>)",
    ),
    _MaskCase(
        "cobre-output-written",
        "Output written to out/ (1.5s)",
        "Output written to out/ (9.2s)",
        "Output written to out/ (<DUR>)",
    ),
    _MaskCase(
        "cobre-host",
        "  Host:      node-a",
        "  Host:      node-x",
        "  Host:      <host>",
    ),
    _MaskCase(
        "cobre-layout-host",
        "  Layout:    4 ranks on node-a",
        "  Layout:    4 ranks on node-x",
        "  Layout:    4 ranks on <host>",
    ),
    _MaskCase(
        "cobre-layout-host",
        "  Layout:    1 rank on node-a",
        "  Layout:    1 rank on node-x",
        "  Layout:    1 rank on <host>",
    ),
    _MaskCase(
        "cobre-rank-host",
        "    node-a: ranks 0-1  (2 ranks)",
        "    node-x: ranks 0-1  (2 ranks)",
        "    <host>: ranks 0-1  (2 ranks)",
    ),
    _MaskCase(
        "cobre-rank-host",
        "    node-b: ranks 2-3  (2 ranks)",
        "    node-y: ranks 2-3  (2 ranks)",
        "    <host>: ranks 2-3  (2 ranks)",
    ),
    _MaskCase(
        "cobre-slurm",
        "  SLURM:     job 1234, nodes node-[1-2,5], 4 CPUs/task",
        "  SLURM:     job 5678, nodes node-[7-8,9], 4 CPUs/task",
        "  SLURM:     job <jobid>, nodes <nodes>, 4 CPUs/task",
    ),
    _MaskCase(
        "cobre-slurm",
        "  SLURM:     job 1234, nodes node-a",
        "  SLURM:     job 5678, nodes node-x",
        "  SLURM:     job <jobid>, nodes <nodes>",
    ),
    _MaskCase(
        "cobre-slurm",
        "  SLURM:     job 1234, 4 CPUs/task",
        "  SLURM:     job 5678, 4 CPUs/task",
        "  SLURM:     job <jobid>, 4 CPUs/task",
    ),
    _MaskCase(
        "cobre-slurm",
        "  SLURM:     job 1234",
        "  SLURM:     job 5678",
        "  SLURM:     job <jobid>",
    ),
)

_UNMASKED = [
    "srun: forcing job termination",
    "srun: Job step aborted: Waiting up to 32 seconds for job step to finish.",
    "  Layout:    4 ranks across 2 nodes",
    "HPCMU_SHAPE model=cobre",
    "Training   1/9 iter  LB: 1.0e0  UB: 2.0e0  gap: 5.0%",
    "Output written to out/ (1.5s) and more",
    "2000-01-01 00:00:01,001 TRACE x: y",
]


@pytest.mark.parametrize("case", _MASK_CASES, ids=lambda c: c.name)
def test_prd_mask_rewrites_the_field_and_keeps_the_line(
    case: _MaskCase,
) -> None:
    for line in (case.base, case.perturbed):
        assert mask_line(line, mode=Mode.PRD, root=None) == case.expected
    mask = next(m for m in PRD_MASKS if m.name == case.name)
    assert mask.pattern.sub(mask.repl, case.base) != case.base
    assert mask.pattern.sub(mask.repl, case.perturbed) != case.perturbed


def test_prd_time_split_duration_needs_a_word_boundary() -> None:
    line = "                Other  x12ms (20%)"
    masked = mask_line(line, mode=Mode.PRD, root=None)
    assert masked == "                Other  x12ms (<PCT>)"


def test_prd_masks_are_a_closed_named_set() -> None:
    names = [m.name for m in PRD_MASKS]
    assert len(set(names)) == len(names)
    assert {c.name for c in _MASK_CASES} == set(names)
    assert {m.source for m in PRD_MASKS} == {"ids", "slurm", "cobre"}
    assert [m.source for m in PRD_MASKS] == sorted(
        (m.source for m in PRD_MASKS), key=("ids", "slurm", "cobre").index
    )


@pytest.mark.parametrize("line", _UNMASKED)
def test_prd_unnamed_shapes_pass_through_unchanged(line: str) -> None:
    assert mask_line(line, mode=Mode.PRD, root=None) == line


def test_prd_perturbed_log_is_equal_in_prd_and_unequal_in_strict() -> None:
    off = [c.base for c in _MASK_CASES]
    on = [c.perturbed for c in _MASK_CASES]
    assert _compare(off, on, mode=Mode.PRD) is None
    assert _compare(off, on, mode=Mode.STRICT) is not None
    masked = [mask_line(x, mode=Mode.PRD, root=None) for x in off]
    assert len(masked) == len(off)


def test_prd_injected_srun_line_is_a_signature_difference() -> None:
    off = [c.base for c in _MASK_CASES]
    on = [c.perturbed for c in _MASK_CASES]
    on.append("srun: error: node-x: task 1: Killed")
    found = _compare(off, on, mode=Mode.PRD)
    assert found is not None
    assert (found.reason, found.line) == ("signature", len(on))


def test_prd_value_outside_the_masks_is_a_mismatch() -> None:
    off = [c.base for c in _MASK_CASES]
    on = [
        c.perturbed.replace("LB: 1.23450e5", "LB: 1.23451e5")
        for c in _MASK_CASES
    ]
    found = _compare(off, on, mode=Mode.PRD)
    assert found is not None
    assert found.reason == "mismatch"
    assert found.off is not None and "LB: 1.23450e5" in found.off
    assert found.on is not None and "LB: 1.23451e5" in found.on


# ----------------------------------------------------------------- shapes

_HEADER = "HPCMU_START job=1234 restart=0 at=2000-01-01T00:00:00Z"
_TAIL = [
    "srun: forcing job termination",
    "srun: Job step aborted: Waiting up to 32 seconds for job step to finish.",
    "job finished rc=137",
]


def _cancel(ts: str) -> str:
    return f"slurmstepd: error: *** JOB 1234 ON node-a CANCELLED AT {ts} ***"


def _iterations(n: int) -> list[str]:
    return [f"iteration {i}" for i in range(1, n + 1)]


def test_shape_prefix_tail_tolerates_a_longer_prefix_and_other_stamps() -> None:
    off = [_HEADER, *_iterations(4), _cancel("2000-01-01T00:00:00"), *_TAIL]
    on = [_HEADER, *_iterations(2), _cancel("2000-01-01T00:07:31"), *_TAIL]
    assert _compare(off, on, shape=Shape.PREFIX_TAIL) is None
    assert _compare(on, off, shape=Shape.PREFIX_TAIL) is None
    strict = _compare(off, on, mode=Mode.STRICT, shape=Shape.PREFIX_TAIL)
    assert strict is not None
    assert strict.reason == "signature"


def test_shape_prefix_tail_on_side_without_an_anchor_is_missing_anchor() -> (
    None
):
    off = [_HEADER, *_iterations(2), _cancel("2000-01-01T00:00:00")]
    found = _compare(off, off[:-1], shape=Shape.PREFIX_TAIL)
    assert found == Difference(
        "missing anchor",
        0,
        "slurmstepd: error: *** JOB <jobid> ON <host> CANCELLED AT <ts> ***",
        None,
    )


def test_shape_prefix_tail_both_sides_without_an_anchor_is_missing_anchor() -> (
    None
):
    bare = [_HEADER, *_iterations(2)]
    assert _compare(bare, bare, shape=Shape.PREFIX_TAIL) == Difference(
        "missing anchor", 0, None, None
    )


def test_shape_prefix_tail_tails_differing_by_one_line_is_tail() -> None:
    off = [_HEADER, *_iterations(2), _cancel("2000-01-01T00:00:00"), *_TAIL]
    on = [*off[:-1], "job finished rc=0"]
    found = _compare(off, on, shape=Shape.PREFIX_TAIL)
    assert found == Difference("tail", len(on), "job finished rc=137", on[-1])


def test_shape_prefix_tail_tail_is_numbered_on_the_on_side() -> None:
    off = [_HEADER, *_iterations(4), _cancel("2000-01-01T00:00:00"), *_TAIL]
    on = [_HEADER, *_iterations(1), _cancel("2000-01-01T00:00:00"), *_TAIL]
    on[-1] = "job finished rc=0"
    found = _compare(off, on, shape=Shape.PREFIX_TAIL)
    assert found is not None
    assert (found.reason, found.line) == ("tail", len(on))


def test_shape_prefix_tail_tail_of_unequal_length_is_tail() -> None:
    off = [_HEADER, _cancel("2000-01-01T00:00:00"), *_TAIL]
    on = [*off, "job finished again"]
    found = _compare(off, on, shape=Shape.PREFIX_TAIL)
    assert found == Difference("tail", len(on), None, "job finished again")


def test_shape_prefix_tail_prefix_not_shared_is_prefix() -> None:
    off = [
        _HEADER,
        "iteration 1",
        "iteration 2",
        _cancel("2000-01-01T00:00:00"),
    ]
    on = [_HEADER, "iteration 1", "iteration X", _cancel("2000-01-01T00:00:00")]
    found = _compare(off, on, shape=Shape.PREFIX_TAIL)
    assert found == Difference("prefix", 3, "iteration 2", "iteration X")


def test_shape_prefix_tail_uses_the_last_anchor() -> None:
    anchor = _cancel("2000-01-01T00:00:00")
    off = [_HEADER, anchor, "iteration 1", anchor]
    on = [_HEADER, anchor, "iteration X", anchor]
    found = _compare(off, on, shape=Shape.PREFIX_TAIL)
    assert found == Difference("prefix", 3, "iteration 1", "iteration X")


def test_shape_order_only_tolerates_a_permutation() -> None:
    off = [_HEADER, "a", "b", "b", "c"]
    on = ["c", "b", _HEADER, "a", "b"]
    assert _compare(off, on, shape=Shape.ORDER_ONLY) is None
    found = _compare(off, on, shape=Shape.EXACT)
    assert found is not None
    assert (found.reason, found.line) == ("mismatch", 1)


def test_shape_order_only_extra_line_is_count_on_the_on_side() -> None:
    off = ["a", "b", "c"]
    on = ["c", "d", "b", "a"]
    assert _compare(off, on, shape=Shape.ORDER_ONLY) == Difference(
        "count", 2, None, "d"
    )


def test_shape_order_only_repeated_line_counts_its_first_occurrence() -> None:
    off = ["a", "b", "b"]
    on = ["b", "a", "b", "b"]
    assert _compare(off, on, shape=Shape.ORDER_ONLY) == Difference(
        "count", 1, None, "b"
    )


def test_shape_order_only_missing_line_is_count_on_the_off_side() -> None:
    off = ["a", "b", "c"]
    on = ["c", "a"]
    assert _compare(off, on, shape=Shape.ORDER_ONLY) == Difference(
        "count", 2, "b", None
    )


def test_shape_exact_one_side_running_out_is_length() -> None:
    off = ["a", "b"]
    assert _compare(off, [*off, "c"]) == Difference("length", 3, None, "c")
    assert _compare([*off, "c"], off) == Difference("length", 3, "c", None)
    assert _compare([], ["a"]) == Difference("length", 1, None, "a")
    assert _compare([], []) is None


# -------------------------------------------------------------------- cli


def test_cli_same_prints_same_and_returns_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    off = _write(tmp_path / "off.log", ["a", "b"])
    on = _write(tmp_path / "on.log", ["a", "b"])
    assert main([str(off), str(on)]) == 0
    assert capsys.readouterr().out == "same\n"


def test_cli_differ_prints_both_sides_and_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    off = _write(tmp_path / "off.log", ["a", "b"])
    on = _write(tmp_path / "on.log", ["a", "b", "c"])
    assert main([str(off), str(on)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("differ: ")
    assert out == "differ: length at line 3\noff: <none>\non: c\n"


def test_cli_differ_escapes_undecodable_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    off = tmp_path / "off.log"
    on = tmp_path / "on.log"
    off.write_bytes(b"bad \xff byte\n")
    on.write_bytes(b"bad \xfe byte\n")
    assert main([str(off), str(on)]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[1:] == [
        r"off: bad \udcff byte",
        r"on: bad \udcfe byte",
    ]


def test_cli_roots_mode_and_shape_are_passed_through(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    off = _write(
        tmp_path / "off.log",
        ["in /ws/a/x", "b", "srun: error: node-a: task 0: Killed"],
    )
    on = _write(
        tmp_path / "on.log",
        ["srun: error: node-x: task 1: Killed", "b", "in /ws/b/x"],
    )
    args = ["--off-root", "/ws/a", "--on-root", "/ws/b", str(off), str(on)]
    assert main(args) == 1
    assert main(["--mode", "prd", *args]) == 1
    assert main(["--mode", "prd", "--shape", "order-only", *args]) == 0
    assert capsys.readouterr().out.endswith("same\n")


def test_cli_missing_file_returns_2_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    on = _write(tmp_path / "on.log", ["a"])
    missing = tmp_path / "missing.log"
    assert main([str(missing), str(on)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"compare_job_logs: cannot read {missing}: No such file or directory\n"
    )


def test_cli_unreadable_second_file_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    off = _write(tmp_path / "off.log", ["a"])
    assert main([str(off), str(tmp_path)]) == 2
    assert capsys.readouterr().err.startswith(
        f"compare_job_logs: cannot read {tmp_path}: "
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--mode", "loose", "a", "b"],
        ["--shape", "any", "a", "b"],
        ["a"],
        ["--x"],
    ],
)
def test_cli_usage_errors_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_cli_runs_isolated_outside_the_package(tmp_path: Path) -> None:
    off = _write(tmp_path / "off.log", ["a", "b"])
    on = _write(tmp_path / "on.log", ["a", "b"])
    run = subprocess.run(
        [sys.executable, "-I", str(_SCRIPT), str(off), str(on)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert (run.returncode, run.stdout, run.stderr) == (0, "same\n", "")


def test_cli_module_imports_neither_click_nor_the_package() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    assert (
        re.findall(r"^(?:import|from) (?:click|hpc_model_utils)", source, re.M)
        == []
    )
