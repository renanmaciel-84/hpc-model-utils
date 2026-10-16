"""Subprocess tests for deploy/modelops/scripts/ensure-tools.sh (ADR-044).

The script runs against a real ``file://`` bare remote, real git and real
flock. A recording git shim on ``PATH`` and a fake uv passed as ``--uv``
append one JSON line per call: argv, cwd, what stdin is, and the ``UV_*`` /
``GIT_TERMINAL_PROMPT`` environment.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "deploy" / "modelops" / "scripts" / "ensure-tools.sh"
PYTHON = "3.12.7"
TOOL = "demo-tool"
UTILS_TOOL = "hpc-model-utils"
SMOKE = "demo-cli"
ANNOTATED_TAG = "v1.0.0"
LIGHTWEIGHT_TAG = "v1.0.1"
PLATFORM = "linux-x86_64-gnu"
_TIMEOUT = 60
_REQUIRED_TOOLS = (
    "bash",
    "git",
    "flock",
    "readlink",
    "chmod",
    "date",
    "mkdir",
    "mv",
    "rm",
    "ln",
    "sleep",
)

_RECORD = r"""
q() {
  local s=${1//\\/\\\\}
  s=${s//\"/\\\"}
  printf '"%s"' "$s"
}
record() {
  local calls=$1 rec sep="" arg var stdin
  shift
  stdin=$(readlink "/proc/$$/fd/0" || true)
  rec='{"argv":['
  for arg in "$@"; do
    rec+="$sep$(q "$arg")"
    sep=","
  done
  rec+="],\"cwd\":$(q "$PWD"),\"stdin\":$(q "$stdin"),\"env\":{"
  sep=""
  for var in $(compgen -e); do
    case $var in
      UV_* | GIT_TERMINAL_PROMPT)
        rec+="$sep$(q "$var"):$(q "${!var}")"
        sep=","
        ;;
    esac
  done
  printf '%s}}\n' "$rec" >>"$calls"
}
"""

_GIT_SHIM = r"""#!__BASH__
set -euo pipefail
__RECORD__
record "$FAKE_GIT_CALLS" "$@"
if [[ ${1:-} == ls-remote && -n ${FAKE_GIT_LS_REMOTE:-} ]]; then
  printf '%s\n' "$FAKE_GIT_LS_REMOTE"
  exit 0
fi
exec "__GIT__" "$@"
"""

_FAKE_UV = r"""#!__BASH__
set -euo pipefail
__RECORD__
record "$FAKE_UV_CALLS" "$@"
plat=__PLATFORM__
case "${1:-} ${2:-}" in
  "python install")
    ver=$3
    patch="$UV_PYTHON_INSTALL_DIR/cpython-$ver-$plat"
    mkdir -p "$patch/bin"
    printf '%s\n' \
      '#!__BASH__' \
      'printf '\''%s %s\n'\'' "$(readlink "/proc/$$/fd/0" || true)" "$*" >>"$FAKE_SMOKE_CALLS"' \
      'if [[ ${FAKE_PYTHON_IMPORT_FAIL:-0} == 1 ]]; then' \
      '  echo "fake python stderr" >&2' \
      '  exit 1' \
      'fi' >"$patch/bin/python3"
    chmod 0755 "$patch/bin/python3"
    ln -sfn "cpython-$ver-$plat" "$UV_PYTHON_INSTALL_DIR/cpython-${ver%.*}-$plat"
    echo "fake uv: installed python $ver" >&2
    ;;
  "sync "*)
    echo "fake uv sync stdout"
    echo "fake uv sync stderr" >&2
    if [[ -n ${FAKE_UV_SYNC_GATE:-} ]]; then
      for ((i = 0; i < 600; i++)); do
        [[ ! -e $FAKE_UV_SYNC_GATE ]] || break
        sleep 0.05
      done
      [[ -e $FAKE_UV_SYNC_GATE ]] || exit 3
    fi
    mkdir -p .venv/bin
    if [[ ${FAKE_UV_FAIL_SYNC:-0} == 1 ]]; then
      echo "error: fake sync exploded" >&2
      exit 1
    fi
    ver="" prev=""
    for arg in "$@"; do
      [[ $prev != --python ]] || ver=$arg
      prev=$arg
    done
    patch="$UV_PYTHON_INSTALL_DIR/cpython-$ver-$plat"
    minor="$UV_PYTHON_INSTALL_DIR/cpython-${ver%.*}-$plat"
    if [[ ${FAKE_UV_PYTHON_OUTSIDE:-0} == 1 ]]; then
      ln -sfn "__BASH__" .venv/bin/python
    elif [[ ${FAKE_UV_NO_PYTHON:-0} != 1 ]]; then
      ln -sfn "$patch/bin/python3" .venv/bin/python
    fi
    home="$patch/bin"
    case ${FAKE_UV_HOME:-patch} in
      minor) home="$minor/bin" ;;
      dotdot) home="$patch/../${minor##*/}/bin" ;;
    esac
    printf 'home = %s\nimplementation = CPython\nversion_info = %s\n' \
      "$home" "$ver" >.venv/pyvenv.cfg
    printf '%s\n' "$(<"$FAKE_SMOKE_STUB")" >".venv/bin/$FAKE_UV_SMOKE_CMD"
    chmod 0755 ".venv/bin/$FAKE_UV_SMOKE_CMD"
    ;;
  *)
    echo "fake uv: unexpected call $*" >&2
    exit 2
    ;;
esac
"""

_SMOKE_STUB = r"""#!__BASH__
printf '%s %s\n' "$(readlink "/proc/$$/fd/0" || true)" "$*" >>"$FAKE_SMOKE_CALLS"
echo "fake smoke stdout"
echo "fake smoke stderr" >&2
if [[ ${FAKE_UV_SMOKE_FAIL:-0} == 1 ]]; then
  exit 1
fi
"""


@dataclass(frozen=True)
class Remote:
    url: str
    commit: str
    tag_object: str


@dataclass(frozen=True)
class Harness:
    root: Path
    uv: Path
    bash: Path
    path: str
    home: Path
    uv_calls: Path
    git_calls: Path
    smoke_calls: Path
    smoke_stub: Path
    remote: Remote

    def install_dir(self, name: str = TOOL, sha: str | None = None) -> Path:
        return self.root / name / (sha or self.remote.commit)

    def line(self, name: str = TOOL) -> str:
        return f"HPCMU_TOOL {name} {self.install_dir(name)}\n"

    def argv(
        self,
        *,
        tag: str = ANNOTATED_TAG,
        sha: str | None = None,
        python: str = PYTHON,
        extra: Sequence[str] = (),
        name: str = TOOL,
    ) -> list[str]:
        return [
            "--root",
            str(self.root),
            "--uv",
            str(self.uv),
            "--python",
            python,
            *extra,
            "--tool",
            name,
            self.remote.url,
            tag,
            sha or self.remote.commit,
            SMOKE,
        ]

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env = {
            "PATH": self.path,
            "HOME": str(self.home),
            "GIT_CONFIG_NOSYSTEM": "1",
            "FAKE_UV_CALLS": str(self.uv_calls),
            "FAKE_GIT_CALLS": str(self.git_calls),
            "FAKE_SMOKE_CALLS": str(self.smoke_calls),
            "FAKE_SMOKE_STUB": str(self.smoke_stub),
            "FAKE_UV_SMOKE_CMD": SMOKE,
        }
        if extra:
            env.update(extra)
        return env

    def run(
        self,
        argv: Sequence[str],
        *,
        extra_env: Mapping[str, str] | None = None,
        stdin_mode: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if stdin_mode:
            return subprocess.run(
                [str(self.bash), "-s", "--", *argv],
                input=SCRIPT.read_text(),
                env=self.env(extra_env),
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
        return subprocess.run(
            [str(self.bash), str(SCRIPT), *argv],
            stdin=subprocess.DEVNULL,
            env=self.env(extra_env),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )

    def reset_calls(self) -> None:
        for calls in (self.uv_calls, self.git_calls, self.smoke_calls):
            calls.unlink(missing_ok=True)


def _git(cwd: Path, home: Path, *args: str) -> str:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    ).stdout.strip()


def _read_calls(calls_file: Path) -> list[dict[str, Any]]:
    if not calls_file.exists():
        return []
    calls: list[dict[str, Any]] = []
    for line in calls_file.read_text().splitlines():
        record = json.loads(line)
        assert isinstance(record, dict)
        calls.append(record)
    return calls


def _uv_env(call: Mapping[str, Any]) -> dict[str, str]:
    env = call["env"]
    assert isinstance(env, dict)
    return {k: v for k, v in env.items() if k.startswith("UV_")}


def _tree_entries(tree: Path) -> list[Path]:
    entries = [tree]
    for dirpath, dirnames, filenames in os.walk(tree):
        base = Path(dirpath)
        entries.extend(base / name for name in (*dirnames, *filenames))
    return entries


def _snapshot(tree: Path) -> dict[str, tuple[int, int, int, int]]:
    snapshot: dict[str, tuple[int, int, int, int]] = {}
    for path in _tree_entries(tree):
        st = path.lstat()
        snapshot[str(path.relative_to(tree))] = (
            st.st_mode,
            st.st_size,
            st.st_mtime_ns,
            st.st_ctime_ns,
        )
    return snapshot


def _make_writable(tree: Path) -> None:
    if not tree.exists():
        return
    for path in _tree_entries(tree):
        mode = path.lstat().st_mode
        if not stat.S_ISLNK(mode):
            path.chmod(stat.S_IMODE(mode) | stat.S_IWUSR)


def _logs(root: Path) -> list[Path]:
    logs_dir = root / ".logs"
    return sorted(logs_dir.glob("*.log")) if logs_dir.is_dir() else []


def _clear_logs(root: Path) -> None:
    for log in _logs(root):
        log.unlink()


def _named_log(line: str, harness: Harness) -> Path:
    """The log a failure line names: it ends the line and lies in .logs/."""
    match = re.search(r"\(log: (?P<log>/\S+)\)$", line)
    assert match is not None, line
    log = Path(match["log"])
    assert log.parent == harness.root / ".logs"
    assert log.is_file()
    return log


def _assert_runtime_failure(
    result: subprocess.CompletedProcess[str],
    harness: Harness,
    sha: str,
    *,
    name: str = TOOL,
) -> tuple[str, Path]:
    assert result.returncode == 1, result.stderr
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert len(lines) == 1, result.stderr
    assert lines[0].startswith(f"ensure-tools: {name}@{sha}: ")
    log = _named_log(lines[0], harness)
    assert not (harness.install_dir(name, sha=sha) / ".ready").exists()
    return lines[0], log


def _wait_until(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + _TIMEOUT
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.05)


@pytest.fixture
def real_bin_dir(tmp_path: Path) -> Path:
    """PATH entries for the real tools the script and the fakes need."""
    bin_dir = tmp_path / "realbin"
    bin_dir.mkdir()
    for tool in _REQUIRED_TOOLS:
        src = shutil.which(tool)
        assert src is not None, f"{tool} not found on PATH"
        (bin_dir / tool).symlink_to(src)
    return bin_dir


@pytest.fixture(scope="module")
def remote() -> Iterator[Remote]:
    """A bare remote whose commit carries an annotated and a lightweight tag."""
    assert shutil.which("git") is not None, "git not found on PATH"
    with tempfile.TemporaryDirectory(prefix="hpcmu-ensure-") as name:
        base = Path(name).resolve()
        home = base / "home"
        home.mkdir()
        work = base / "work"
        work.mkdir()
        _git(work, home, "-c", "init.defaultBranch=main", "init", "-q")
        (work / "pyproject.toml").write_text('[project]\nname = "demo-tool"\n')
        (work / "uv.lock").write_text("version = 1\n")
        _git(work, home, "add", "pyproject.toml", "uv.lock")
        _git(work, home, "commit", "-q", "-m", "release")
        _git(work, home, "tag", "-a", ANNOTATED_TAG, "-m", "annotated release")
        _git(work, home, "tag", LIGHTWEIGHT_TAG)
        bare = base / "remote.git"
        _git(base, home, "clone", "-q", "--bare", str(work), str(bare))
        commit = _git(work, home, "rev-parse", "HEAD")
        tag_object = _git(work, home, "rev-parse", ANNOTATED_TAG)
        assert tag_object != commit
        yield Remote(url=f"file://{bare}", commit=commit, tag_object=tag_object)


@pytest.fixture
def harness(real_bin_dir: Path, remote: Remote) -> Iterator[Harness]:
    with tempfile.TemporaryDirectory(prefix="hpcmu-ensure-") as name:
        base = Path(name).resolve()
        bash = real_bin_dir / "bash"
        bash_path = str(bash.resolve())
        git_path = str((real_bin_dir / "git").resolve())

        shim_dir = base / "shimbin"
        shim_dir.mkdir()
        git_shim = shim_dir / "git"
        git_shim.write_text(
            _GIT_SHIM.replace("__RECORD__", _RECORD)
            .replace("__BASH__", bash_path)
            .replace("__GIT__", git_path)
        )
        git_shim.chmod(0o755)

        uv = base / "uvbin" / "uv"
        uv.parent.mkdir()
        uv.write_text(
            _FAKE_UV.replace("__RECORD__", _RECORD)
            .replace("__BASH__", bash_path)
            .replace("__PLATFORM__", PLATFORM)
        )
        uv.chmod(0o755)

        smoke_stub = base / "smoke-stub"
        smoke_stub.write_text(_SMOKE_STUB.replace("__BASH__", bash_path))
        home = base / "home"
        home.mkdir()

        h = Harness(
            root=base / "tools",
            uv=uv,
            bash=bash,
            path=f"{shim_dir}{os.pathsep}{real_bin_dir}",
            home=home,
            uv_calls=base / "uv-calls.jsonl",
            git_calls=base / "git-calls.jsonl",
            smoke_calls=base / "smoke-calls.txt",
            smoke_stub=smoke_stub,
            remote=remote,
        )
        assert re.fullmatch(r"/[A-Za-z0-9._/-]+", str(h.root)), h.root
        yield h
        _make_writable(h.root)


_TAG_KINDS = [
    pytest.param(ANNOTATED_TAG, id="annotated"),
    pytest.param(LIGHTWEIGHT_TAG, id="lightweight"),
]


@pytest.mark.parametrize("tag", _TAG_KINDS)
def test_ensure_tools_miss_path_prints_line_and_marks_ready(
    harness: Harness, tag: str
) -> None:
    result = harness.run(harness.argv(tag=tag))

    assert result.returncode == 0, result.stderr
    assert result.stdout == harness.line()
    assert result.stderr == ""

    install = harness.install_dir()
    ready = (install / ".ready").read_text().splitlines()
    assert ready[:4] == [
        f"tool={TOOL}",
        f"tag={tag}",
        f"sha={harness.remote.commit}",
        f"python={PYTHON}",
    ]
    assert len(ready) == 5
    assert re.fullmatch(
        r"installed_at=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ready[4]
    )
    assert not (install / ".ready.tmp").exists()
    assert (install / "pyproject.toml").is_file()
    assert (install / ".git" / "shallow").is_file()
    for sub in (".python", ".locks", ".logs"):
        assert (harness.root / sub).is_dir()

    for path in _tree_entries(install):
        mode = path.lstat().st_mode
        if not stat.S_ISLNK(mode):
            assert mode & 0o222 == 0, path
    with pytest.raises(PermissionError, match="Permission denied"):
        (install / "write-probe").write_text("x")


@pytest.mark.parametrize("tag", _TAG_KINDS)
def test_ensure_tools_miss_path_records_uv_calls(
    harness: Harness, tag: str
) -> None:
    result = harness.run(harness.argv(tag=tag))

    assert result.returncode == 0, result.stderr
    python_dir = str(harness.root / ".python")
    calls = _read_calls(harness.uv_calls)
    assert [call["argv"] for call in calls] == [
        ["python", "install", PYTHON, "--no-bin"],
        ["sync", "--frozen", "--no-dev", "--no-editable", "--python", PYTHON],
    ]
    install_call, sync_call = calls
    assert _uv_env(install_call) == {
        "UV_PYTHON_INSTALL_DIR": python_dir,
        "UV_PYTHON_PREFERENCE": "only-managed",
        "UV_PYTHON_DOWNLOADS": "automatic",
    }
    assert _uv_env(sync_call) == {
        "UV_PYTHON_INSTALL_DIR": python_dir,
        "UV_PYTHON_PREFERENCE": "only-managed",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_COMPILE_BYTECODE": "1",
        "UV_LINK_MODE": "copy",
    }
    assert sync_call["cwd"] == str(harness.install_dir())
    assert all(call["stdin"] == "/dev/null" for call in calls)

    interpreter = harness.install_dir() / ".venv" / "bin" / "python"
    assert str(interpreter.resolve()).startswith(
        f"{python_dir}/cpython-{PYTHON}-"
    )


@pytest.mark.parametrize("tag", _TAG_KINDS)
def test_ensure_tools_miss_path_verifies_pin_and_logs_output(
    harness: Harness, tag: str
) -> None:
    result = harness.run(harness.argv(tag=tag))

    assert result.returncode == 0, result.stderr
    url = harness.remote.url
    install = str(harness.install_dir())
    calls = _read_calls(harness.git_calls)
    assert [call["argv"] for call in calls] == [
        ["ls-remote", url, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
        [
            "-c",
            "advice.detachedHead=false",
            "-c",
            "http.lowSpeedLimit=1000",
            "-c",
            "http.lowSpeedTime=60",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            tag,
            url,
            install,
        ],
        ["-C", install, "rev-parse", "HEAD"],
    ]
    for call in calls:
        assert call["env"].get("GIT_TERMINAL_PROMPT") == "0"
        assert call["stdin"] == "/dev/null"
    assert harness.smoke_calls.read_text().splitlines() == ["/dev/null --help"]

    logs = _logs(harness.root)
    assert len(logs) == 1
    assert re.fullmatch(
        rf"{TOOL}-{harness.remote.commit}-\d{{8}}T\d{{6}}Z-\d+\.log",
        logs[0].name,
    )
    text = logs[0].read_text()
    for marker in (
        "fake uv: installed python",
        "fake uv sync stdout",
        "fake uv sync stderr",
        "fake smoke stdout",
        "fake smoke stderr",
    ):
        assert marker in text


def test_ensure_tools_miss_path_two_tools_print_in_argument_order(
    harness: Harness,
) -> None:
    url = harness.remote.url
    sha = harness.remote.commit
    argv = [
        "--root",
        str(harness.root),
        "--uv",
        str(harness.uv),
        "--python",
        PYTHON,
        "--tool",
        "zeta-tool",
        url,
        LIGHTWEIGHT_TAG,
        sha,
        SMOKE,
        "--tool",
        TOOL,
        url,
        ANNOTATED_TAG,
        sha,
        SMOKE,
    ]

    result = harness.run(argv)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == harness.line("zeta-tool") + harness.line(TOOL)
    syncs = [c for c in _read_calls(harness.uv_calls) if c["argv"][0] == "sync"]
    assert [c["cwd"] for c in syncs] == [
        str(harness.install_dir("zeta-tool")),
        str(harness.install_dir(TOOL)),
    ]


def test_ensure_tools_miss_path_pyarrow_import_runs_after_help(
    harness: Harness,
) -> None:
    # stdin mode: the script's own stdin is a pipe, so a /dev/null record
    # proves the step's </dev/null redirect.
    result = harness.run(harness.argv(name=UTILS_TOOL), stdin_mode=True)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == harness.line(UTILS_TOOL)
    assert (harness.install_dir(UTILS_TOOL) / ".ready").is_file()
    assert harness.smoke_calls.read_text().splitlines() == [
        "/dev/null --help",
        "/dev/null -I -c import pyarrow.parquet",
    ]
    logs = _logs(harness.root)
    assert len(logs) == 1
    assert "== smoke import: pyarrow.parquet" in logs[0].read_text()


def test_ensure_tools_hit_path_makes_no_calls_and_writes_nothing(
    harness: Harness,
) -> None:
    first = harness.run(harness.argv())
    assert first.returncode == 0, first.stderr
    harness.reset_calls()
    before = _snapshot(harness.root)

    second = harness.run(harness.argv())

    assert second.returncode == 0, second.stderr
    assert second.stderr == ""
    assert second.stdout == first.stdout == harness.line()
    assert _read_calls(harness.git_calls) == []
    assert _read_calls(harness.uv_calls) == []
    assert not harness.smoke_calls.exists()
    assert _snapshot(harness.root) == before


def test_ensure_tools_hit_path_pyarrow_import_not_rerun(
    harness: Harness,
) -> None:
    first = harness.run(harness.argv(name=UTILS_TOOL))
    assert first.returncode == 0, first.stderr
    harness.reset_calls()
    _clear_logs(harness.root)
    before = _snapshot(harness.root)

    second = harness.run(harness.argv(name=UTILS_TOOL))

    assert second.returncode == 0, second.stderr
    assert second.stderr == ""
    assert second.stdout == harness.line(UTILS_TOOL)
    assert not harness.smoke_calls.exists()
    assert _logs(harness.root) == []
    assert _snapshot(harness.root) == before


def test_ensure_tools_hit_path_python_mismatch_exits_1(
    harness: Harness,
) -> None:
    assert harness.run(harness.argv()).returncode == 0
    harness.reset_calls()
    _clear_logs(harness.root)
    assert _logs(harness.root) == []
    ready = harness.install_dir() / ".ready"
    recorded = ready.read_text()

    result = harness.run(harness.argv(python="3.12.8"))

    assert result.returncode == 1
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert len(lines) == 1, result.stderr
    assert lines[0].startswith(
        f"ensure-tools: {TOOL}@{harness.remote.commit}: "
        f"installed with python {PYTHON}, not 3.12.8"
    )
    assert f"keep --python {PYTHON} or pin a new tag" in lines[0]
    assert _logs(harness.root) == [_named_log(lines[0], harness)]
    assert _read_calls(harness.git_calls) == []
    assert _read_calls(harness.uv_calls) == []
    assert ready.read_text() == recorded


def test_ensure_tools_hit_path_minor_home_guard_exits_1(
    harness: Harness,
) -> None:
    assert harness.run(harness.argv()).returncode == 0
    harness.reset_calls()
    venv = harness.install_dir() / ".venv"
    cfg = venv / "pyvenv.cfg"
    venv.chmod(0o755)
    cfg.chmod(0o644)
    minor = harness.root / ".python" / f"cpython-3.12-{PLATFORM}"
    cfg.write_text(f"home = {minor}/bin\n")
    _clear_logs(harness.root)
    assert _logs(harness.root) == []

    result = harness.run(harness.argv())

    assert result.returncode == 1
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert len(lines) == 1, result.stderr
    assert "venv guard: pyvenv.cfg home is not" in lines[0]
    assert _logs(harness.root) == [_named_log(lines[0], harness)]
    assert _read_calls(harness.git_calls) == []
    assert _read_calls(harness.uv_calls) == []


def test_ensure_tools_failure_tag_object_sha_is_not_the_commit(
    harness: Harness,
) -> None:
    sha = harness.remote.tag_object

    result = harness.run(harness.argv(tag=ANNOTATED_TAG, sha=sha))

    line, log = _assert_runtime_failure(result, harness, sha)
    assert f"tag {ANNOTATED_TAG} does not resolve to the pinned SHA" in line
    assert [c["argv"][0] for c in _read_calls(harness.git_calls)] == [
        "ls-remote"
    ]
    assert _read_calls(harness.uv_calls) == []
    assert f"resolves to commit {harness.remote.commit}" in log.read_text()


def test_ensure_tools_failure_absent_tag(harness: Harness) -> None:
    result = harness.run(harness.argv(tag="v9.9.9"))

    line, _ = _assert_runtime_failure(result, harness, harness.remote.commit)
    assert "tag v9.9.9 not found on the remote" in line
    assert _read_calls(harness.uv_calls) == []


def test_ensure_tools_failure_moved_tag_head_mismatch(
    harness: Harness,
) -> None:
    moved = "a" * 40
    listing = f"{moved}\trefs/tags/{LIGHTWEIGHT_TAG}"

    result = harness.run(
        harness.argv(tag=LIGHTWEIGHT_TAG, sha=moved),
        extra_env={"FAKE_GIT_LS_REMOTE": listing},
    )

    line, log = _assert_runtime_failure(result, harness, moved)
    assert "cloned HEAD does not match the pinned SHA" in line
    assert f"cloned HEAD is {harness.remote.commit}" in log.read_text()
    assert _read_calls(harness.uv_calls) == []


@pytest.mark.parametrize(
    ("fake_env", "reason", "detail"),
    [
        pytest.param(
            {"FAKE_UV_FAIL_SYNC": "1"},
            "uv sync failed",
            "error: fake sync exploded",
            id="uv_sync",
        ),
        pytest.param(
            {"FAKE_UV_HOME": "minor"},
            "venv guard: pyvenv.cfg home is not",
            f"cpython-3.12-{PLATFORM}/bin",
            id="minor_home",
        ),
        pytest.param(
            {"FAKE_UV_PYTHON_OUTSIDE": "1"},
            "venv guard: .venv/bin/python resolves outside",
            "resolved: /",
            id="python_outside_root",
        ),
        pytest.param(
            {"FAKE_UV_SMOKE_FAIL": "1"},
            f"smoke run '{SMOKE} --help' failed",
            "fake smoke stderr",
            id="smoke_run",
        ),
    ],
)
def test_ensure_tools_failure_leaves_no_ready(
    harness: Harness, fake_env: dict[str, str], reason: str, detail: str
) -> None:
    result = harness.run(harness.argv(), extra_env=fake_env)

    line, log = _assert_runtime_failure(result, harness, harness.remote.commit)
    assert reason in line
    assert detail in log.read_text()
    for raw in ("fake uv sync", "fake smoke", "exploded"):
        assert raw not in result.stderr


def test_ensure_tools_failure_pyarrow_import_fails_leaves_no_ready(
    harness: Harness,
) -> None:
    sha = harness.remote.commit

    result = harness.run(
        harness.argv(name=UTILS_TOOL),
        extra_env={"FAKE_PYTHON_IMPORT_FAIL": "1"},
    )

    line, log = _assert_runtime_failure(result, harness, sha, name=UTILS_TOOL)
    assert line.startswith(
        f"ensure-tools: {UTILS_TOOL}@{sha}: "
        "smoke import 'pyarrow.parquet' failed (log: "
    )
    text = log.read_text()
    assert "ensure-tools: FAILED: smoke import 'pyarrow.parquet' failed" in text
    assert "fake python stderr" in text


def test_ensure_tools_failure_missing_venv_python(harness: Harness) -> None:
    result = harness.run(harness.argv(), extra_env={"FAKE_UV_NO_PYTHON": "1"})

    line, log = _assert_runtime_failure(result, harness, harness.remote.commit)
    venv = harness.install_dir() / ".venv"
    assert (venv / "pyvenv.cfg").is_file()
    interpreter = venv / "bin" / "python"
    assert not interpreter.exists()
    assert not interpreter.is_symlink()
    assert "venv guard: .venv/bin/python is missing" in line
    assert "FAILED: venv guard: .venv/bin/python is missing" in log.read_text()


def test_ensure_tools_failure_dotdot_home(harness: Harness) -> None:
    result = harness.run(harness.argv(), extra_env={"FAKE_UV_HOME": "dotdot"})

    line, log = _assert_runtime_failure(result, harness, harness.remote.commit)
    assert "venv guard: pyvenv.cfg home is not" in line
    prefix = f"{harness.root}/.python/cpython-{PYTHON}-"
    venv = harness.install_dir() / ".venv"
    assert str((venv / "bin" / "python").resolve()).startswith(prefix)
    cfg_lines = (venv / "pyvenv.cfg").read_text().splitlines()
    home = cfg_lines[0].removeprefix("home = ")
    # Only the .. component sets this home apart from a valid one.
    assert home.startswith(prefix)
    assert home.endswith("/bin")
    assert f"/../cpython-3.12-{PLATFORM}/bin" in home
    assert f"home: {home}" in log.read_text()


def test_ensure_tools_failure_lock_timeout(harness: Harness) -> None:
    locks = harness.root / ".locks"
    locks.mkdir(parents=True)
    lock_file = locks / f"{TOOL}-{harness.remote.commit}.lock"

    with lock_file.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        result = harness.run(harness.argv(extra=["--lock-timeout", "1"]))

    line, _ = _assert_runtime_failure(result, harness, harness.remote.commit)
    assert "timed out after 1s waiting for the install lock" in line
    assert _read_calls(harness.git_calls) == []
    assert _read_calls(harness.uv_calls) == []


def test_ensure_tools_failure_missing_flock(
    harness: Harness, tmp_path: Path
) -> None:
    no_flock = tmp_path / "noflockbin"
    no_flock.mkdir()
    for tool in _REQUIRED_TOOLS:
        if tool != "flock":
            src = shutil.which(tool)
            assert src is not None, f"{tool} not found on PATH"
            (no_flock / tool).symlink_to(src)
    shim_dir = harness.path.split(os.pathsep)[0]

    result = harness.run(
        harness.argv(), extra_env={"PATH": f"{shim_dir}{os.pathsep}{no_flock}"}
    )

    line, _ = _assert_runtime_failure(result, harness, harness.remote.commit)
    assert "flock not found on PATH" in line
    assert _read_calls(harness.git_calls) == []


def test_ensure_tools_rerun_after_sync_failure_replaces_partial_dir(
    harness: Harness,
) -> None:
    failed = harness.run(harness.argv(), extra_env={"FAKE_UV_FAIL_SYNC": "1"})
    _assert_runtime_failure(failed, harness, harness.remote.commit)
    install = harness.install_dir()
    assert (install / ".venv").is_dir()
    marker = install / "partial-marker"
    marker.write_text("left by the interrupted install\n")
    harness.reset_calls()

    result = harness.run(harness.argv())

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == harness.line()
    assert not marker.exists()
    assert (install / ".ready").is_file()
    assert "clone" in [
        arg for call in _read_calls(harness.git_calls) for arg in call["argv"]
    ]


def test_ensure_tools_concurrent_runs_sync_once(
    harness: Harness, tmp_path: Path
) -> None:
    gate = tmp_path / "sync-gate"
    env = harness.env({"FAKE_UV_SYNC_GATE": str(gate)})
    argv = [str(harness.bash), str(SCRIPT), *harness.argv()]

    procs = [
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        # One run is inside sync and the other has opened its log, so it
        # passed the .ready check before the install finished.
        _wait_until(
            lambda: (
                harness.uv_calls.exists()
                and '"sync"' in harness.uv_calls.read_text()
                and len(_logs(harness.root)) == 2
            ),
            "one run in sync and both runs on the miss path",
        )
        gate.touch()
        outputs = [proc.communicate(timeout=_TIMEOUT) for proc in procs]
    finally:
        gate.touch()
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=_TIMEOUT)

    for proc, (stdout, stderr) in zip(procs, outputs, strict=True):
        assert proc.returncode == 0, stderr
        assert stderr == ""
        assert stdout == harness.line()
    argvs = [call["argv"][0] for call in _read_calls(harness.uv_calls)]
    assert argvs == ["python", "sync"]
    log_texts = [log.read_text() for log in _logs(harness.root)]
    assert sum("installed by a concurrent run" in t for t in log_texts) == 1


def test_ensure_tools_stdin_mode_prints_same_line(harness: Harness) -> None:
    piped = harness.run(harness.argv(), stdin_mode=True)

    assert piped.returncode == 0, piped.stderr
    assert piped.stderr == ""
    assert piped.stdout == harness.line()
    calls = _read_calls(harness.uv_calls) + _read_calls(harness.git_calls)
    assert len(calls) == 5
    assert all(call["stdin"] == "/dev/null" for call in calls)
    assert harness.smoke_calls.read_text().splitlines() == ["/dev/null --help"]

    from_file = harness.run(harness.argv())
    assert from_file.returncode == 0, from_file.stderr
    assert from_file.stdout == piped.stdout


_USAGE_CASES = (
    "missing_root",
    "relative_root",
    "dotdot_root",
    "trailing_slash_root",
    "symlinked_root",
    "relative_uv",
    "non_executable_uv",
    "minor_only_python",
    "zero_lock_timeout",
    "repeated_root",
    "unknown_option",
    "no_tool",
    "incomplete_tool",
    "duplicate_name",
    "uppercase_name",
    "ssh_url",
    "branch_tag",
    "short_sha",
    "spaced_smoke_command",
)


def _usage_argv(h: Harness, case: str) -> list[str]:
    root = str(h.root)
    url = h.remote.url
    sha = h.remote.commit
    tool = ["--tool", TOOL, url, ANNOTATED_TAG, sha, SMOKE]

    def build(
        *,
        root: str = root,
        uv: str = str(h.uv),
        python: str = PYTHON,
        extra: Sequence[str] = (),
        tools: Sequence[str] = tool,
    ) -> list[str]:
        return ["--root", root, "--uv", uv, "--python", python, *extra, *tools]

    h.root.mkdir(parents=True, exist_ok=True)
    link = h.root.parent / "tools-link"
    link.symlink_to(h.root)
    not_executable = h.root.parent / "not-executable"
    not_executable.write_text("#!/bin/sh\n")
    not_executable.chmod(0o644)

    cases = {
        "missing_root": ["--uv", str(h.uv), "--python", PYTHON, *tool],
        "relative_root": build(root="tools"),
        "dotdot_root": build(root=f"{root}/../tools"),
        "trailing_slash_root": build(root=f"{root}/"),
        "symlinked_root": build(root=str(link)),
        "relative_uv": build(uv="uv"),
        "non_executable_uv": build(uv=str(not_executable)),
        "minor_only_python": build(python="3.12"),
        "zero_lock_timeout": build(extra=["--lock-timeout", "0"]),
        "repeated_root": build(extra=["--root", root]),
        "unknown_option": build(extra=["--force"]),
        "no_tool": build(tools=()),
        "incomplete_tool": build(tools=["--tool", TOOL, url, ANNOTATED_TAG]),
        "duplicate_name": build(tools=[*tool, *tool]),
        "uppercase_name": build(
            tools=["--tool", "Demo", url, ANNOTATED_TAG, sha, SMOKE]
        ),
        "ssh_url": build(
            tools=[
                "--tool",
                TOOL,
                "git@example.com:org/repo.git",
                ANNOTATED_TAG,
                sha,
                SMOKE,
            ]
        ),
        "branch_tag": build(tools=["--tool", TOOL, url, "main", sha, SMOKE]),
        "short_sha": build(
            tools=["--tool", TOOL, url, ANNOTATED_TAG, sha[:12], SMOKE]
        ),
        "spaced_smoke_command": build(
            tools=["--tool", TOOL, url, ANNOTATED_TAG, sha, "demo cli"]
        ),
    }
    return cases[case]


@pytest.mark.parametrize("case", _USAGE_CASES)
def test_ensure_tools_usage_error_exits_2(harness: Harness, case: str) -> None:
    result = harness.run(_usage_argv(harness, case))

    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert len(lines) == 1, result.stderr
    assert lines[0].startswith("ensure-tools: ")
    assert _read_calls(harness.git_calls) == []
    assert _read_calls(harness.uv_calls) == []


def test_ensure_tools_script_is_executable_and_template_safe() -> None:
    assert SCRIPT.stat().st_mode & 0o111 == 0o111
    text = SCRIPT.read_text()
    assert "{{" not in text
    assert "@@" not in text
