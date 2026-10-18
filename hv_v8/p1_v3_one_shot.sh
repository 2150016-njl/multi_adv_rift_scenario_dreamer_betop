#!/usr/bin/env bash
# Run a SINGLE preflight + real CARLA single record + gated full 95 on
# an immutable source copy. No changes to hv_v8/p1_native_audit.py.
# No `set -e`, no `exit` of the caller's shell, and no CLI arguments guessed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
RUNNER="$ROOT/hv_v8/p1_frozen_mainline_runner_v3.py"
PRETEST="$ROOT/hv_v8/p1_v3_integration_preflight.py"

printf 'ROOT=%s\nPY=%s\n' "$ROOT" "$PY"
if [[ ! -f "$RUNNER" || ! -f "$PRETEST" ]]; then
  echo 'P1_V3_STOP: bundle was not completely unpacked'
  RC=70
elif ! "$PY" -m py_compile "$RUNNER" "$PRETEST"; then
  echo 'P1_V3_STOP: Python source compile failed'
  RC=71
elif ! "$PY" "$PRETEST"; then
  echo 'P1_V3_STOP: consolidated population/pose contract regression failed'
  RC=72
elif ! "$PY" "$RUNNER" --endpoint-self-test; then
  echo 'P1_V3_STOP: endpoint contract regression failed'
  RC=73
else
  "$PY" "$RUNNER" --full-after-smoke --no-pause
  RC=$?
fi
printf '\nP1_V3_RUNNER_EXIT=%s\n' "$RC"
printf '日志及结果路径请查找上方 RUN_LOG / P1_OUTPUT / P1_FULL95_OUTPUT。\n'
if [[ -t 0 ]]; then
  printf '按 Enter 返回现有终端（不启动新 Shell）：'
  IFS= read -r _
fi
# Deliberately do not exit/exec the user's interactive shell.
true
