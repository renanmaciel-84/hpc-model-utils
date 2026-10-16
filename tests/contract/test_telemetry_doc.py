"""ADR-072/R56: docs/telemetry.md carries the render_markdown() block.

The text between the two marker lines must equal `render_markdown()` byte
for byte, so the written contract cannot drift from schema.py, its single
source (ADR-068). With HPCMU_REGEN_GOLDENS=1 the block test rewrites only
the text between the markers, keeps every other byte of the doc, and fails.
The helpers it uses never read the environment, so the in-memory tests
behave the same with or without the variable and never write the doc.

The rendering tests assert that every contract item reaches the block, and
the layout tests hold `SUMMARY_FIELDS` to the summary.json objects of the
export goldens, which this module only reads.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Final

import pytest

from hpc_model_utils.infra.telemetry.schema import (
    HEADER_FIELDS,
    HEADLINE,
    HEADLINE_FORMAT_TERMS,
    HEADLINE_FORMATS,
    HEADLINE_KEY_PREFIX,
    HEADLINE_POOLING_TERMS,
    HEADLINE_POOLINGS,
    MEANING_CONVENTIONS,
    PARQUET_METADATA_KEYS,
    PHASES,
    SAMPLE_COLUMNS,
    SAMPLE_STATS,
    SAMPLES_FILE,
    SUMMARY_FIELDS,
    WARNING_CODES,
    WORKSPACE_COLUMNS,
    WORKSPACE_FILE,
    WORKSPACE_SCOPE,
    WORKSPACE_STATS,
    StatSpec,
    SummaryField,
    Term,
    render_markdown,
)

BEGIN_MARKER: Final = (
    "<!-- BEGIN GENERATED: render_markdown() in "
    "src/hpc_model_utils/infra/telemetry/schema.py; do not edit -->"
)
END_MARKER: Final = "<!-- END GENERATED -->"

_REGEN_ENV: Final = "HPCMU_REGEN_GOLDENS"
_ROOT: Final = Path(__file__).resolve().parents[2]
_DOC_REL: Final = Path("docs") / "telemetry.md"
_DOC_PATH: Final = _ROOT / _DOC_REL
_GOLDENS_PATH: Final = (
    _ROOT / "tests" / "goldens" / "telemetry" / "export_expected.json"
)

_HEADINGS: Final = (
    "### Version and files",
    "### Parquet key-value metadata",
    "### Conventions",
    "### samples.parquet columns",
    "### workspace.parquet columns",
    "### Staging header fields",
    "### summary.json layout",
    "### summary.json statistics",
    "### Headline values and hooks",
    "### Warning codes",
)
_JSON_TYPES: Final = frozenset(
    {"object", "array", "string", "integer", "number", "map"}
)

type _Json = None | bool | int | float | str | list[_Json] | dict[str, _Json]


# The doc and its markers.


def _read_doc() -> str:
    # newline="" keeps every byte: no \r\n is folded into \n.
    with _DOC_PATH.open(encoding="utf-8", newline="") as stream:
        return stream.read()


def _write_doc(text: str) -> None:
    with _DOC_PATH.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)


def _marker_offsets(doc: str, marker: str) -> list[int]:
    """The offsets of the lines of doc that are exactly marker."""
    offsets: list[int] = []
    offset = 0
    for line in doc.split("\n"):
        if line == marker:
            offsets.append(offset)
        offset += len(line) + 1
    return offsets


def _split(doc: str) -> tuple[str, str, str]:
    """Return the head through the BEGIN line, the block, and the tail.

    The head ends with the BEGIN line's newline, and the tail starts at the
    END line. A marker line that is missing, repeated or out of order fails
    the test with a message naming the marker.
    """
    found: list[int] = []
    for marker in (BEGIN_MARKER, END_MARKER):
        offsets = _marker_offsets(doc, marker)
        if len(offsets) != 1:
            pytest.fail(
                f"the marker line {marker!r} occurs {len(offsets)} times in "
                f"{_DOC_REL}; it must occur exactly once"
            )
        found.append(offsets[0])
    begin, end = found
    if end < begin:
        pytest.fail(
            f"the marker line {END_MARKER!r} comes before {BEGIN_MARKER!r} "
            f"in {_DOC_REL}"
        )
    head_end = begin + len(BEGIN_MARKER) + 1
    return doc[:head_end], doc[head_end:end], doc[end:]


def _regenerated(doc: str) -> str:
    """doc with its block replaced by render_markdown(), the rest kept."""
    head, _, tail = _split(doc)
    return head + render_markdown() + tail


def _compare_block(doc: str) -> None:
    """Fail at the first doc line where the block leaves render_markdown()."""
    head, block, _ = _split(doc)
    expected = render_markdown()
    if block == expected:
        return
    got = block.split("\n")
    want = expected.split("\n")
    index = next(
        (i for i, pair in enumerate(zip(got, want)) if pair[0] != pair[1]),
        min(len(got), len(want)),
    )
    got_line = got[index] if index < len(got) else None
    want_line = want[index] if index < len(want) else None
    line = head.count("\n") + index + 1
    pytest.fail(
        f"{_DOC_REL} line {line} differs from render_markdown(): "
        f"{got_line!r} != {want_line!r}; run with {_REGEN_ENV}=1"
    )


def _set_regen(monkeypatch: pytest.MonkeyPatch, regen: bool) -> None:
    if regen:
        monkeypatch.setenv(_REGEN_ENV, "1")
    else:
        monkeypatch.delenv(_REGEN_ENV, raising=False)


def test_block_equals_render_markdown() -> None:
    doc = _read_doc()
    if os.environ.get(_REGEN_ENV) == "1":
        _write_doc(_regenerated(doc))
        pytest.fail(
            f"{_REGEN_ENV}=1: regenerated the block of {_DOC_REL}; "
            "re-run without it to verify"
        )
    _compare_block(doc)


def test_block_markers_occur_once_in_order() -> None:
    doc = _read_doc()
    begins = _marker_offsets(doc, BEGIN_MARKER)
    ends = _marker_offsets(doc, END_MARKER)

    assert len(begins) == 1, BEGIN_MARKER
    assert len(ends) == 1, END_MARKER
    assert begins[0] < ends[0]
    assert "".join(_split(doc)) == doc


def test_block_doc_is_utf8_and_ends_with_newline() -> None:
    text = _DOC_PATH.read_bytes().decode("utf-8")

    assert text.endswith("\n")
    assert _read_doc() == text


@pytest.mark.parametrize("regen", [False, True], ids=["env-unset", "env-1"])
def test_block_compare_fails_on_one_changed_meaning_character(
    monkeypatch: pytest.MonkeyPatch, regen: bool
) -> None:
    _set_regen(monkeypatch, regen)
    before = _DOC_PATH.read_bytes()
    head, block, tail = _split(_read_doc())
    meaning = SAMPLE_COLUMNS[0].meaning
    assert block.count(meaning) == 1
    changed = ("x" if meaning[0] != "x" else "y") + meaning[1:]
    doc = head + block.replace(meaning, changed) + tail

    with pytest.raises(
        pytest.fail.Exception, match=r"line \d+ .*HPCMU_REGEN_GOLDENS=1"
    ):
        _compare_block(doc)
    assert _DOC_PATH.read_bytes() == before


@pytest.mark.parametrize("regen", [False, True], ids=["env-unset", "env-1"])
def test_block_rewrite_keeps_every_byte_outside_markers(
    monkeypatch: pytest.MonkeyPatch, regen: bool
) -> None:
    _set_regen(monkeypatch, regen)
    head = f"# Title Δ ×\n\nprose before − ≤\n\n{BEGIN_MARKER}\n"
    tail = f"{END_MARKER}\n\nprose after\r\nwith a CRLF line\n\n"
    stale = "### Version and files\n\nstale text\n"

    rewritten = _regenerated(head + stale + tail)

    assert rewritten == head + render_markdown() + tail
    assert rewritten.startswith(head)
    assert rewritten.endswith(tail)


@pytest.mark.parametrize(
    ("doc", "marker"),
    [
        (f"prose\n{END_MARKER}\n", BEGIN_MARKER),
        (f"{BEGIN_MARKER}\n{BEGIN_MARKER}\n{END_MARKER}\n", BEGIN_MARKER),
        (f"{BEGIN_MARKER}\nblock\n", END_MARKER),
        (f"{BEGIN_MARKER}\n{END_MARKER}\n{END_MARKER}\n", END_MARKER),
        (f"{END_MARKER}\nblock\n{BEGIN_MARKER}\n", END_MARKER),
        (f"text {BEGIN_MARKER}\n{END_MARKER}\n", BEGIN_MARKER),
    ],
    ids=[
        "no-begin",
        "two-begins",
        "no-end",
        "two-ends",
        "end-first",
        "begin-not-a-line",
    ],
)
def test_block_split_names_a_bad_marker(doc: str, marker: str) -> None:
    with pytest.raises(pytest.fail.Exception, match=re.escape(repr(marker))):
        _split(doc)


# The rendered block (AC 3).


def _sections() -> dict[str, list[str]]:
    """The lines of each ### section of render_markdown(), by heading."""
    sections: dict[str, list[str]] = {}
    lines: list[str] = []
    for line in render_markdown().split("\n"):
        if line.startswith("### "):
            lines = sections.setdefault(line, [])
        else:
            lines.append(line)
    return sections


def _rows(heading: str) -> list[str]:
    """The table rows of a section whose first cell is a code span."""
    return [line for line in _sections()[heading] if line.startswith("| `")]


def _published(spec: StatSpec) -> list[str]:
    """ADR-071: the published names of spec, in aggregate order."""
    names: list[str] = []
    for aggregate in spec.aggregates:
        if aggregate != "total":
            names.append(f"{spec.column}_{aggregate}")
        elif spec.total_name is not None:
            names.append(spec.total_name)
        else:
            raise ValueError(f"{spec.column}: a total without a total_name")
    return names


def _escaped(text: str) -> str:
    """Requirement 4: backslash before each \\, *, < and &."""
    return re.sub(r"([\\*<&])", r"\\\1", text)


def test_rendered_headings_in_order() -> None:
    rendered = render_markdown()
    headings = [line for line in rendered.split("\n") if line.startswith("#")]

    assert headings == list(_HEADINGS)
    assert rendered.startswith("### Version and files\n")
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")


@pytest.mark.parametrize(
    ("heading", "names", "count"),
    [
        (_HEADINGS[3], tuple(c.name for c in SAMPLE_COLUMNS), 64),
        (_HEADINGS[4], tuple(c.name for c in WORKSPACE_COLUMNS), 12),
        (_HEADINGS[5], tuple(f.name for f in HEADER_FIELDS), 22),
        (_HEADINGS[1], tuple(k.name for k in PARQUET_METADATA_KEYS), 6),
        (_HEADINGS[6], tuple(f.path for f in SUMMARY_FIELDS), 33),
        (_HEADINGS[9], tuple(w.code for w in WARNING_CODES), 9),
    ],
    ids=["samples", "workspace", "header", "metadata", "summary", "warnings"],
)
def test_rendered_rows_in_tuple_order(
    heading: str, names: tuple[str, ...], count: int
) -> None:
    rows = _rows(heading)

    assert len(names) == count
    assert len(rows) == count
    for row, name in zip(rows, names, strict=True):
        assert row.startswith(f"| `{name}` |"), row


def test_rendered_term_rows_in_tuple_order() -> None:
    rows = _rows(_HEADINGS[8])
    term_rows = [
        r for r in rows if not r.startswith(f"| `{HEADLINE_KEY_PREFIX}")
    ]
    terms = (*HEADLINE_POOLING_TERMS, *HEADLINE_FORMAT_TERMS)

    assert term_rows == [
        f"| `{t.name}` | {_escaped(t.meaning)} |" for t in terms
    ]


def test_rendered_statistic_names() -> None:
    section = "\n".join(_sections()[_HEADINGS[7]])
    sample_names = [n for s in SAMPLE_STATS for n in _published(s)]
    workspace_names = [n for s in WORKSPACE_STATS for n in _published(s)]
    expected_rows = [
        f"| `{spec.column}` | `{file}` | `{stat_map}` | "
        + ", ".join(f"`{name}`" for name in _published(spec))
        + " |"
        for specs, file, stat_map in (
            (SAMPLE_STATS, SAMPLES_FILE, "jobs[].nodes[].stats"),
            (WORKSPACE_STATS, WORKSPACE_FILE, "workspace.stats"),
        )
        for spec in specs
    ]

    assert len(set(sample_names)) == len(sample_names) == 210
    assert len(set(workspace_names)) == len(workspace_names) == 7
    assert [
        n for n in (*sample_names, *workspace_names) if f"`{n}`" not in section
    ] == []
    assert _rows(_HEADINGS[7]) == expected_rows
    assert len(expected_rows) == 63


def test_rendered_hook_rows() -> None:
    rows = _rows(_HEADINGS[8])
    hook_rows = [r for r in rows if r.startswith(f"| `{HEADLINE_KEY_PREFIX}")]

    assert len(HEADLINE) == len(hook_rows) == 19
    for spec, row in zip(HEADLINE, hook_rows, strict=True):
        assert row.startswith(f"| `telemetry.{spec.key}` |"), row
        assert f"| `{spec.pooling}` |" in row, row
        assert row.endswith(f"| `{spec.fmt}` |"), row


def test_rendered_scopes_and_cap() -> None:
    headline = "\n".join(_sections()[_HEADINGS[8]])
    warnings = _sections()[_HEADINGS[9]]

    assert (*PHASES, WORKSPACE_SCOPE) == ("model", "finalize", "workspace")
    assert "the scopes are `model`, `finalize`, `workspace`." in headline
    assert "`telemetry.<scope>.<statistic>`" in headline
    assert "The list holds at most `500` warnings (`WARNINGS_CAP`)." in warnings


def test_rendered_escapes_paths_with_markdown_specials() -> None:
    samples = {r.split(" | ", 1)[0]: r for r in _rows(_HEADINGS[3])}

    assert r"/sys/block/\<dev>/device" in samples["| `disk_read_bytes_per_s`"]
    assert r"infiniband/\*/ports" in samples["| `efa_rx_bytes_per_s`"]


def test_rendered_every_text_escaped() -> None:
    rendered = render_markdown()
    texts = [
        *(c.meaning for c in (*SAMPLE_COLUMNS, *WORKSPACE_COLUMNS)),
        *(f.meaning for f in HEADER_FIELDS),
        *(k.meaning for k in PARQUET_METADATA_KEYS),
        *(f.meaning for f in SUMMARY_FIELDS),
        *(t.meaning for t in (*HEADLINE_POOLING_TERMS, *HEADLINE_FORMAT_TERMS)),
        *(w.meaning for w in WARNING_CODES),
    ]

    assert [t for t in texts if f"| {_escaped(t)} |" not in rendered] == []
    assert [
        t for t in MEANING_CONVENTIONS if f"\n- {_escaped(t)}\n" not in rendered
    ] == []


def test_rendered_is_deterministic() -> None:
    assert render_markdown() == render_markdown()


# The summary.json layout against the export goldens (AC 4).


def _summaries() -> list[tuple[str, dict[str, _Json]]]:
    """(case, summary) of every case of the export goldens, narrowed."""
    with _GOLDENS_PATH.open(encoding="utf-8") as stream:
        cases: _Json = json.load(stream)
    if not isinstance(cases, dict):
        raise TypeError(f"{_GOLDENS_PATH}: not a JSON object")
    summaries: list[tuple[str, dict[str, _Json]]] = []
    for name, case in cases.items():
        summary = case.get("summary") if isinstance(case, dict) else None
        if not isinstance(summary, dict):
            raise TypeError(f"case {name}: no summary object")
        summaries.append((name, summary))
    return summaries


@dataclasses.dataclass(frozen=True, slots=True)
class _Walk:
    """Every (path, value) and every (path, object) of one summary.

    A path joins keys with "." and marks list elements with "[]". The
    values below a map path are not walked, and an object that is a list
    element is listed as an object only: its path is element-only.
    """

    values: list[tuple[str, _Json]]
    objects: list[tuple[str, dict[str, _Json]]]


_MAP_PATHS: Final = frozenset(
    f.path for f in SUMMARY_FIELDS if f.json_type == "map"
)


def _walk(summary: dict[str, _Json]) -> _Walk:
    walk = _Walk(values=[], objects=[])

    def visit_object(path: str, obj: dict[str, _Json]) -> None:
        walk.objects.append((path, obj))
        for key, child in obj.items():
            child_path = f"{path}.{key}" if path else key
            walk.values.append((child_path, child))
            visit(child_path, child)

    def visit(path: str, value: _Json) -> None:
        if path in _MAP_PATHS:
            return
        if isinstance(value, dict):
            visit_object(path, value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    visit_object(f"{path}[]", item)
                else:
                    walk.values.append((f"{path}[]", item))
                    visit(f"{path}[]", item)

    visit_object("", summary)
    return walk


def _walks() -> list[tuple[str, _Walk]]:
    return [(name, _walk(summary)) for name, summary in _summaries()]


def _fields() -> dict[str, SummaryField]:
    return {f.path: f for f in SUMMARY_FIELDS}


def _matches(json_type: str, value: _Json) -> bool:
    if isinstance(value, bool):
        return False
    if json_type == "integer":
        return isinstance(value, int)
    if json_type == "number":
        return isinstance(value, int | float)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "array":
        return isinstance(value, list)
    return json_type in ("object", "map") and isinstance(value, dict)


def test_summary_fields_match_export_goldens_paths() -> None:
    paths = {path for _, walk in _walks() for path, _ in walk.values}

    assert len(SUMMARY_FIELDS) == 33
    assert len(_fields()) == 33
    assert len(_MAP_PATHS) == 5
    assert paths == set(_fields())


def test_summary_fields_required_present_in_every_parent() -> None:
    missing: list[str] = []
    for case, walk in _walks():
        for parent, obj in walk.objects:
            for field in SUMMARY_FIELDS:
                head, _, key = field.path.rpartition(".")
                if field.required and head == parent and key not in obj:
                    missing.append(f"{case}: {field.path}")

    assert missing == []


def test_summary_fields_json_types() -> None:
    fields = _fields()
    wrong = [
        f"{case}: {path} = {value!r}"
        for case, walk in _walks()
        for path, value in walk.values
        if value is not None and not _matches(fields[path].json_type, value)
    ]

    assert {f.json_type for f in SUMMARY_FIELDS} <= _JSON_TYPES
    assert wrong == []


def test_summary_fields_nulls_only_on_nullable() -> None:
    fields = _fields()
    nulls = {
        path
        for _, walk in _walks()
        for path, value in walk.values
        if value is None
    }

    assert [p for p in sorted(nulls) if not fields[p].nullable] == []


def test_terms_derive_headline_constants() -> None:
    assert [t.name for t in HEADLINE_POOLING_TERMS] == [
        "node_rows",
        "sum_nodes",
        "walks",
    ]
    assert [t.name for t in HEADLINE_FORMAT_TERMS] == [
        "bytes",
        "cores",
        "fraction",
    ]
    assert HEADLINE_POOLINGS == frozenset({"node_rows", "sum_nodes", "walks"})
    assert HEADLINE_FORMATS == frozenset({"bytes", "cores", "fraction"})


def _texts() -> list[str]:
    terms = (*HEADLINE_POOLING_TERMS, *HEADLINE_FORMAT_TERMS)
    return [
        *(text for t in terms for text in (t.name, t.meaning)),
        *(
            text
            for f in SUMMARY_FIELDS
            for text in (f.path, f.json_type, f.meaning)
        ),
        *MEANING_CONVENTIONS,
    ]


@pytest.mark.parametrize("text", _texts())
def test_terms_and_summary_fields_texts_fit_a_table_cell(text: str) -> None:
    assert text
    assert text == text.strip()
    assert not {"\n", "|", "`"} & set(text)


@pytest.mark.parametrize(
    "record",
    [HEADLINE_POOLING_TERMS[0], SUMMARY_FIELDS[0]],
    ids=lambda r: type(r).__name__,
)
def test_terms_and_summary_fields_are_frozen_and_slotted(
    record: Term | SummaryField,
) -> None:
    first = dataclasses.fields(record)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(record, first, "other")
    assert not hasattr(record, "__dict__")
