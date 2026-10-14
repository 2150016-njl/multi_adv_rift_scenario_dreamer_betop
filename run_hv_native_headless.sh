#!/usr/bin/env bash
# Follow the documented RIFT recipe: full audit under riftuser, no --render.
# Usage from the root terminal: bash run_hv_native_headless.sh [--startup-only]
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
ROOT="${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}"
PY="${PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}"
CARLA_ROOT="${CARLA_ROOT:-$ROOT/carla}"
PAIR="${HV_NATIVE_PAIR_REPORT:-$ROOT/data/hv_init/eval/snapshot_audit_20261013_160509_936449/v2_pairs100.jsonl.audit.json}"
PORT="${HV_NATIVE_PORT:-31100}"
[ "$(id -u)" = 0 ] || { echo 'Run this wrapper from the root terminal; it switches the full audit to riftuser.' >&2; exit 2; }
for file in "$PY" "$ROOT/run_hv_native_diagnostics.py" "$CARLA_ROOT/CarlaUE4.sh" "$PAIR"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 2; }
done
# Root prepares exactly one NEW output directory, not the existing evaluation
# tree. The Python runner accepts it only while empty and owned by its user.
OUT="$("$PY" - "$ROOT" <<'PY'
from pathlib import Path
from datetime import datetime
import os,pwd,sys
user=pwd.getpwnam('riftuser')
if user.pw_uid==0:raise SystemExit('riftuser must not have UID 0')
out=Path(sys.argv[1])/'data/hv_init/eval'/('native_headless_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
out.mkdir(parents=True,exist_ok=False)
os.chown(str(out),user.pw_uid,user.pw_gid)
print(out)
PY
)"
echo "RESULT_DIR=$OUT"
exec runuser -u riftuser -- env ROOT="$ROOT" PY="$PY" OUT="$OUT" CARLA_ROOT="$CARLA_ROOT" \
  PAIR="$PAIR" PORT="$PORT" /bin/bash -lc '
  set -euo pipefail
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
  export CARLA_PYTHONAPI="$CARLA_ROOT/PythonAPI/carla"
  # Reuse the exact egg that imported successfully on this server and is
  # recorded in the existing reproduction documentation. Never pick the
  # alphabetically first egg: that selects py2.7 when both are installed.
  CARLA_EGG="$CARLA_PYTHONAPI/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
  export CARLA_EGG
  [ -f "$CARLA_EGG" ] || { echo "Missing previously verified CARLA egg: $CARLA_EGG" >&2; exit 2; }
  echo "CARLA_EGG=$CARLA_EGG"
  export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  [ "$(id -u)" != 0 ] || { echo "Refusing to run CARLA as root" >&2; exit 2; }
  echo "Audit and CARLA user: $(id)"
  cd "$ROOT/RIFT"
  exec "$PY" -u "$ROOT/run_hv_native_diagnostics.py" \
    --pair-report "$PAIR" --port "$PORT" --auto-port --render-mode nullrhi \
    --pool-records 100 --native-records 20 --probe-ticks 20 \
    --startup-timeout 300 --audit-timeout 300 \
    --prepared-output --output-dir "$OUT" "$@"
  ' hv-native-headless "$@"
