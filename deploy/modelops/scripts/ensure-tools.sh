#!/usr/bin/env bash
# Ensure immutable per-SHA tool installs under a shared tools root (ADR-044).
#
#   ensure-tools.sh --root <abs dir> --uv <abs file> --python <X.Y.Z>
#       [--lock-timeout <seconds>]
#       { --tool <name> <repo-url> <tag> <sha40> <smoke-command>
#       | --tool-tag <name> <repo-url> <tag> <smoke-command> } ...
#
# A cache hit (<root>/<name>/<sha40>/.ready) runs neither git nor uv and
# writes nothing (ADR-030). A miss verifies the tag against the pinned SHA,
# clones it, syncs the locked environment on the shared exact-patch
# interpreter, and marks the directory ready and read-only. The
# hpc-model-utils install also imports pyarrow.parquet through its own
# interpreter (ADR-091).
#
# A --tool-tag tool pins no SHA. Its first run resolves the tag with git
# ls-remote (the peeled commit of an annotated tag) under a per-tag lock and
# records tag -> sha40 read-only in <root>/.tags/<name>/<tag>. Later runs
# read the record and run no git, and the install stays keyed by, and checked
# against, the recorded SHA, so a tag that moves later is never picked up
# silently: installing it fails until the record is removed. A tag that is not
# on the remote fails before any install. After install,
# "<smoke-command> --version" must print exactly "<smoke-command> X.Y.Z" for
# tag vX.Y.Z.
#
# stdout carries only "HPCMU_TOOL <name> <dir>" lines. A failure is exactly
# one stderr line (ADR-050); git, uv and smoke-run output goes to the log
# under <root>/.logs/. The Task runs this file as `bash -s -- <args>` with the
# script on stdin, so every external command reads /dev/null.
set -euo pipefail

export GIT_TERMINAL_PROMPT=0

# Explicit sets: bracket ranges such as [a-z] depend on the locale.
readonly lower='abcdefghijklmnopqrstuvwxyz'
readonly upper='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
readonly digit='0123456789'
readonly alnum="${upper}${lower}${digit}"
readonly root_re="^/[${alnum}._/-]+\$"
readonly name_re="^[${lower}][${lower}${digit}-]{0,63}\$"
readonly https_re="^https://[${alnum}.-]+/[${alnum}_./-]+\$"
readonly file_re="^file:///[${alnum}_./-]+\$"
readonly tag_re="^v[${digit}]+\.[${digit}]+\.[${digit}]+\$"
readonly sha_re="^[${digit}abcdef]{40}\$"
readonly python_re="^3\.[${digit}]+\.[${digit}]+\$"
readonly timeout_re="^[1-9][${digit}]*\$"
readonly home_re='^home ?= ?(.*)$'

fail() {
  printf 'ensure-tools: %s\n' "$1" >&2
  exit "${2:-1}"
}

fail_usage() {
  fail "$1" 2
}

root=""
uv=""
python=""
lock_timeout=1800
modes=()
names=()
repos=()
tags=()
shas=()
cmds=()
declare -A seen=()

while (($# > 0)); do
  case $1 in
    --root | --uv | --python | --lock-timeout)
      (($# >= 2)) || fail_usage "$1 needs a value"
      [[ -z ${seen[$1]+x} ]] || fail_usage "$1 given more than once"
      seen[$1]=1
      case $1 in
        --root) root=$2 ;;
        --uv) uv=$2 ;;
        --python) python=$2 ;;
        --lock-timeout) lock_timeout=$2 ;;
      esac
      shift 2
      ;;
    --tool)
      (($# >= 6)) ||
        fail_usage "--tool needs <name> <repo-url> <tag> <sha40> <smoke-command>"
      modes+=(pin)
      names+=("$2")
      repos+=("$3")
      tags+=("$4")
      shas+=("$5")
      cmds+=("$6")
      shift 6
      ;;
    --tool-tag)
      (($# >= 5)) ||
        fail_usage "--tool-tag needs <name> <repo-url> <tag> <smoke-command>"
      modes+=(tag)
      names+=("$2")
      repos+=("$3")
      tags+=("$4")
      shas+=("")
      cmds+=("$5")
      shift 5
      ;;
    *)
      arg=${1:0:64}
      fail_usage "unexpected argument '${arg//[^${alnum}._\/:=-]/?}'"
      ;;
  esac
done

[[ -n ${seen[--root]+x} ]] || fail_usage "missing --root"
[[ -n ${seen[--uv]+x} ]] || fail_usage "missing --uv"
[[ -n ${seen[--python]+x} ]] || fail_usage "missing --python"
[[ $root =~ $root_re && $root != */ && "$root/" != */../* ]] ||
  fail_usage "--root must be an absolute [A-Za-z0-9._/-] path with no .. component and no trailing /"
[[ $uv == /* && -f $uv && -x $uv ]] ||
  fail_usage "--uv must be an absolute path to an executable file"
[[ $python =~ $python_re ]] ||
  fail_usage "--python must be an exact patch version 3.X.Y"
[[ $lock_timeout =~ $timeout_re ]] ||
  fail_usage "--lock-timeout must be a positive integer"
((${#names[@]} > 0)) || fail_usage "at least one --tool or --tool-tag is required"

declare -A seen_names=()
for i in "${!names[@]}"; do
  n=$((i + 1))
  flag=--tool
  [[ ${modes[i]} == pin ]] || flag=--tool-tag
  [[ ${names[i]} =~ $name_re ]] ||
    fail_usage "$flag #$n: name must match [a-z][a-z0-9-]{0,63}"
  [[ ${repos[i]} =~ $https_re || ${repos[i]} =~ $file_re ]] ||
    fail_usage "$flag #$n: repo-url must be https://<host>/<path> or file:///<path>"
  [[ ${tags[i]} =~ $tag_re ]] ||
    fail_usage "$flag #$n: tag must match vX.Y.Z (tags only, never branches)"
  [[ ${modes[i]} == tag || ${shas[i]} =~ $sha_re ]] ||
    fail_usage "$flag #$n: sha must be 40 lowercase hex characters"
  [[ ${cmds[i]} =~ $name_re ]] ||
    fail_usage "$flag #$n: smoke-command must match [a-z][a-z0-9-]{0,63}"
  [[ -z ${seen_names[${names[i]}]+x} ]] ||
    fail_usage "$flag #$n: duplicate name ${names[i]}"
  seen_names[${names[i]}]=1
done

mkdir -p -- "$root" </dev/null >/dev/null 2>&1 ||
  fail_usage "cannot create --root"
canonical=$(readlink -f -- "$root" </dev/null 2>/dev/null) ||
  fail_usage "cannot resolve --root"
[[ $canonical == "$root" ]] ||
  fail_usage "--root is not canonical: readlink -f resolves it elsewhere"
readonly root uv python lock_timeout

# State of the tool being ensured. The log is opened lazily: a successful hit
# writes nothing. A --tool-tag tool's t_sha stays empty until its tag
# resolves, so messages and the log name use the tag until then. t_pin names
# the SHA in messages: "pinned" for --tool, "recorded" for --tool-tag.
t_mode=""
t_pin=""
t_name=""
t_repo=""
t_tag=""
t_sha=""
t_cmd=""
t_dir=""
t_record=""
remote_sha=""
log=""
lock_fd=""
fail_reason=""
fail_detail=""

note() {
  { printf '%s\n' "$1" >>"$log"; } 2>/dev/null || fail_tool "cannot write the log"
}

# Every runtime failure: one stderr line naming the log, details in the log.
fail_tool() {
  [[ -n $log ]] || open_log
  {
    printf 'ensure-tools: FAILED: %s\n' "$1" >>"$log"
    if [[ -n ${2:-} ]]; then
      printf '%s\n' "$2" >>"$log"
    fi
  } 2>/dev/null || true
  fail "$t_name@${t_sha:-$t_tag}: $1 (log: $log)"
}

open_log() {
  local stamp
  stamp=$(date -u +%Y%m%dT%H%M%SZ </dev/null 2>/dev/null) ||
    fail "$t_name@${t_sha:-$t_tag}: cannot read the clock"
  log="$root/.logs/$t_name-${t_sha:-$t_tag}-$stamp-$$.log"
  mkdir -p -- "$root/.logs" </dev/null >/dev/null 2>&1 ||
    fail_tool "cannot create $root/.logs"
  note "ensure-tools: $t_name@${t_sha:-$t_tag} tag=$t_tag repo=$t_repo pid=$$"
}

# Exclusive flock on $1, kept open in lock_fd until the caller closes it.
take_lock() {
  local file=$1 what=$2 fd rc=0
  { exec {fd}>>"$file"; } 2>>"$log" || fail_tool "cannot open the $what lock"
  flock -w "$lock_timeout" "$fd" </dev/null >>"$log" 2>&1 || rc=$?
  if ((rc == 1)); then
    fail_tool "timed out after ${lock_timeout}s waiting for the $what lock"
  elif ((rc != 0)); then
    fail_tool "could not take the $what lock"
  fi
  lock_fd=$fd
}

require_commands() {
  local required
  for required in git flock; do
    command -v "$required" >/dev/null 2>&1 ||
      fail_tool "$required not found on PATH"
  done
}

# ADR-044, R97: the venv interpreter and its pyvenv.cfg home must sit in the
# exact-patch directory under <root>/.python, never the minor-version link.
guard_venv() {
  local prefix="$root/.python/cpython-$python-" bin="$t_dir/.venv/bin/python"
  local cfg="$t_dir/.venv/pyvenv.cfg" target home="" line
  fail_detail=""
  if [[ ! -e $bin ]]; then
    fail_reason="venv guard: .venv/bin/python is missing"
    return 1
  fi
  if ! target=$(readlink -f -- "$bin" </dev/null 2>/dev/null); then
    fail_reason="venv guard: .venv/bin/python does not resolve"
    return 1
  fi
  if [[ $target != "$prefix"* ]]; then
    fail_reason="venv guard: .venv/bin/python resolves outside ${prefix}*"
    fail_detail="resolved: $target"
    return 1
  fi
  if [[ ! -f $cfg || ! -r $cfg ]]; then
    fail_reason="venv guard: .venv/pyvenv.cfg is missing"
    return 1
  fi
  while IFS= read -r line || [[ -n $line ]]; do
    if [[ $line =~ $home_re ]]; then
      home=${BASH_REMATCH[1]}
    fi
  done <"$cfg"
  if [[ $home != "$prefix"*/bin || $home == */../* || $home == */./* ]]; then
    fail_reason="venv guard: pyvenv.cfg home is not ${prefix}*/bin"
    fail_detail="home: $home"
    return 1
  fi
}

# A .ready install is valid when it recorded this interpreter and the venv
# guards pass.
verify_ready() {
  local ready="$t_dir/.ready" line recorded=""
  fail_detail=""
  if [[ ! -f $ready || ! -r $ready ]]; then
    fail_reason=".ready is not a readable file"
    return 1
  fi
  while IFS= read -r line || [[ -n $line ]]; do
    if [[ $line == python=* ]]; then
      recorded=${line#python=}
    fi
  done <"$ready"
  if [[ $recorded != "$python" ]]; then
    if [[ $recorded =~ $python_re ]]; then
      fail_reason="installed with python $recorded, not $python; an installed SHA never changes interpreter, so keep --python $recorded or pin a new tag"
    else
      fail_reason=".ready records no valid python= line; pin a new tag"
    fi
    return 1
  fi
  guard_venv
}

# R131, F10: an annotated tag's plain line is the tag object; the peeled
# ^{} line is the commit and takes precedence. Sets remote_sha. Never call it
# in $( ): fail_tool would exit only the subshell.
remote_commit() {
  local out sha ref peeled="" plain=""
  note "== git ls-remote $t_repo $t_tag"
  out=$(git ls-remote "$t_repo" "refs/tags/$t_tag" "refs/tags/$t_tag^{}" \
    </dev/null 2>>"$log") || fail_tool "git ls-remote failed"
  note "$out"
  while IFS=$'\t' read -r sha ref; do
    if [[ $ref == "refs/tags/$t_tag^{}" ]]; then
      peeled=$sha
    elif [[ $ref == "refs/tags/$t_tag" ]]; then
      plain=$sha
    fi
  done <<<"$out"
  remote_sha=${peeled:-$plain}
  [[ -n $remote_sha ]] || fail_tool "tag $t_tag not found on the remote"
}

# A tag record is valid when it names this tool, repo and tag and holds a
# 40-hex SHA. Sets t_sha.
read_record() {
  local line tool="" repo="" tag="" sha=""
  if [[ ! -f $t_record || ! -r $t_record ]]; then
    fail_reason="tag record $t_record is not a readable file"
    return 1
  fi
  while IFS= read -r line || [[ -n $line ]]; do
    case $line in
      tool=*) tool=${line#tool=} ;;
      repo=*) repo=${line#repo=} ;;
      tag=*) tag=${line#tag=} ;;
      sha=*) sha=${line#sha=} ;;
    esac
  done <"$t_record"
  if [[ $tool != "$t_name" || $repo != "$t_repo" || $tag != "$t_tag" ||
    ! $sha =~ $sha_re ]]; then
    fail_reason="tag record $t_record does not match this tool, repo and tag; remove it to resolve the tag again"
    return 1
  fi
  t_sha=$sha
}

# A --tool-tag tool: read the tag record, or resolve the tag once under the
# per-tag lock and write the record read-only (ADR-030: a hit runs no git and
# opens no log). The record is written to a temporary file, made read-only and
# renamed into place, so a reader without the lock sees no record or a whole
# one. Never call it in $( ): fail_tool would exit only the subshell.
resolve_tag() {
  local tag_fd stamp tmp
  t_record="$root/.tags/$t_name/$t_tag"
  if [[ -e $t_record ]]; then
    read_record || fail_tool "$fail_reason"
    return 0
  fi
  open_log
  require_commands
  mkdir -p -- "$root/.locks" "$root/.tags/$t_name" </dev/null >>"$log" 2>&1 ||
    fail_tool "cannot create $root/.tags/$t_name"
  take_lock "$root/.locks/$t_name-tag-$t_tag.lock" tag
  tag_fd=$lock_fd
  # Double-checked locking: a concurrent run may have recorded the tag.
  if [[ -e $t_record ]]; then
    note "== resolved by a concurrent run"
    read_record || fail_tool "$fail_reason"
  else
    remote_commit
    stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ </dev/null 2>>"$log") ||
      fail_tool "cannot read the clock"
    tmp="$t_record.tmp.$$"
    {
      printf 'tool=%s\nrepo=%s\ntag=%s\nsha=%s\nresolved_at=%s\n' \
        "$t_name" "$t_repo" "$t_tag" "$remote_sha" "$stamp" >"$tmp"
    } 2>>"$log" || fail_tool "cannot write the tag record"
    { chmod a-w -- "$tmp" && mv -- "$tmp" "$t_record"; } </dev/null >>"$log" 2>&1 ||
      fail_tool "cannot write the tag record"
    t_sha=$remote_sha
    note "== recorded $t_tag as $t_sha in $t_record"
  fi
  exec {tag_fd}>&-
}

install_tool() {
  local head py_fd stamp version

  if [[ -L $t_dir ]]; then
    note "== removing the partial install (symlink)"
    rm -f -- "$t_dir" </dev/null >>"$log" 2>&1 ||
      fail_tool "cannot remove the partial install"
  elif [[ -e $t_dir ]]; then
    note "== removing the partial install"
    { chmod -R u+w -- "$t_dir" && rm -rf -- "$t_dir"; } </dev/null >>"$log" 2>&1 ||
      fail_tool "cannot remove the partial install"
  fi

  remote_commit
  if [[ $remote_sha != "$t_sha" ]]; then
    note "tag $t_tag resolves to commit $remote_sha"
    fail_tool "tag $t_tag does not resolve to the $t_pin SHA"
  fi

  note "== git clone $t_tag"
  git -c advice.detachedHead=false -c http.lowSpeedLimit=1000 \
    -c http.lowSpeedTime=60 clone --quiet --depth 1 --branch "$t_tag" \
    "$t_repo" "$t_dir" </dev/null >>"$log" 2>&1 || fail_tool "git clone failed"
  head=$(git -C "$t_dir" rev-parse HEAD </dev/null 2>>"$log") ||
    fail_tool "git rev-parse HEAD failed in the clone"
  if [[ $head != "$t_sha" ]]; then
    note "cloned HEAD is $head"
    fail_tool "cloned HEAD does not match the $t_pin SHA (did the tag move?)"
  fi

  note "== uv python install $python"
  take_lock "$root/.locks/python-$python.lock" python
  py_fd=$lock_fd
  UV_PYTHON_INSTALL_DIR="$root/.python" \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_DOWNLOADS=automatic \
    "$uv" python install "$python" --no-bin </dev/null >>"$log" 2>&1 ||
    fail_tool "uv python install failed"
  exec {py_fd}>&-

  # R97: copy mode keeps every venv file out of the head-node uv cache, and
  # downloads=never forbids sync from fetching a different interpreter.
  note "== uv sync"
  (
    cd -- "$t_dir" &&
      UV_PYTHON_INSTALL_DIR="$root/.python" \
        UV_PYTHON_PREFERENCE=only-managed \
        UV_PYTHON_DOWNLOADS=never \
        UV_COMPILE_BYTECODE=1 \
        UV_LINK_MODE=copy \
        "$uv" sync --frozen --no-dev --no-editable --python "$python"
  ) </dev/null >>"$log" 2>&1 || fail_tool "uv sync failed"

  note "== venv guards"
  guard_venv || fail_tool "$fail_reason" "$fail_detail"

  note "== smoke run: $t_cmd --help"
  "$t_dir/.venv/bin/$t_cmd" --help </dev/null >>"$log" 2>&1 ||
    fail_tool "smoke run '$t_cmd --help' failed"

  # A --tool-tag tool must report its tag's version. Its output is untrusted:
  # it goes only to the log. The subshell keeps the unset local; with stdout
  # not a terminal, rich prints no control codes unless these force them.
  if [[ $t_mode == tag ]]; then
    note "== version check: $t_cmd --version"
    version=$(
      unset FORCE_COLOR TTY_COMPATIBLE TTY_INTERACTIVE
      NO_COLOR=1 TERM=dumb "$t_dir/.venv/bin/$t_cmd" --version </dev/null 2>>"$log"
    ) || fail_tool "version check '$t_cmd --version' failed"
    note "$version"
    [[ $version == "$t_cmd ${t_tag#v}" ]] ||
      fail_tool "version check: '$t_cmd --version' does not report ${t_tag#v}"
  fi

  if [[ $t_name == hpc-model-utils ]]; then
    note "== smoke import: pyarrow.parquet"
    "$t_dir/.venv/bin/python" -I -c 'import pyarrow.parquet' </dev/null >>"$log" 2>&1 ||
      fail_tool "smoke import 'pyarrow.parquet' failed"
  fi

  stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ </dev/null 2>>"$log") ||
    fail_tool "cannot read the clock"
  {
    printf 'tool=%s\ntag=%s\nsha=%s\npython=%s\ninstalled_at=%s\n' \
      "$t_name" "$t_tag" "$t_sha" "$python" "$stamp" >"$t_dir/.ready.tmp"
  } 2>>"$log" || fail_tool "cannot write .ready.tmp"
  mv -- "$t_dir/.ready.tmp" "$t_dir/.ready" </dev/null >>"$log" 2>&1 ||
    fail_tool "cannot write .ready"
  chmod -R a-w -- "$t_dir" </dev/null >>"$log" 2>&1 ||
    fail_tool "cannot make the install read-only"
  note "== installed $t_dir"
}

ensure_one() {
  local sha_fd
  t_mode=$1
  t_name=$2
  t_repo=$3
  t_tag=$4
  t_sha=$5
  t_cmd=$6
  t_pin=pinned
  log=""
  if [[ $t_mode == tag ]]; then
    t_pin=recorded
    resolve_tag
  fi
  t_dir="$root/$t_name/$t_sha"

  if [[ -e $t_dir/.ready ]]; then
    verify_ready || fail_tool "$fail_reason" "$fail_detail"
    return 0
  fi

  [[ -n $log ]] || open_log
  require_commands
  mkdir -p -- "$root/.locks" </dev/null >>"$log" 2>&1 ||
    fail_tool "cannot create $root/.locks"
  take_lock "$root/.locks/$t_name-$t_sha.lock" install
  sha_fd=$lock_fd
  # Double-checked locking: a concurrent run may have finished the install.
  if [[ -e $t_dir/.ready ]]; then
    note "== installed by a concurrent run"
    verify_ready || fail_tool "$fail_reason" "$fail_detail"
  else
    install_tool
  fi
  exec {sha_fd}>&-
}

for i in "${!names[@]}"; do
  ensure_one "${modes[i]}" "${names[i]}" "${repos[i]}" "${tags[i]}" \
    "${shas[i]}" "${cmds[i]}"
  shas[i]=$t_sha
done

for i in "${!names[@]}"; do
  printf 'HPCMU_TOOL %s %s\n' "${names[i]}" "$root/${names[i]}/${shas[i]}"
done
