#!/usr/bin/env bash
# Deliberately one-record only. --offline: evidence and source checks, no CARLA.
# --real: runs one actual CARLA audit; must be on an exclusively controlled instance.
set -euo pipefail
ROOT=${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}
PY=${PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}
BASE="$ROOT/data/hv_init/v7"
SCRIPT_DIR="$ROOT/hv_v8"
SNAP="$ROOT/data/hv_init/rift_labeled_phase7_v2_uncovered_routes_seed0_7/worker0_seed0/route27_env0_RouteScenario_24333_idx27_rep0_frame145.json"
OLD="$BASE/p1_frozen_capsule_route140_20261018_141003_2150/hv_v8/p1_native_audit.py"
MODE=${1:---offline}
if [[ "$MODE" != --offline && "$MODE" != --real ]]; then
  echo 'Usage: bash hv_v8/p1_occupancy_record100_run.sh [--offline|--real]' >&2; exit 2
fi
for p in "$PY" "$SNAP" "$OLD" "$SCRIPT_DIR/p1_occupancy_contract_repair.py" "$SCRIPT_DIR/test_p1_occupancy_contract.py" "$SCRIPT_DIR/p1_calibration_ledger.py"; do
  [[ -f "$p" ]] || { echo "P1_PRECHECK_MISSING=$p" >&2; exit 2; }
done
"$PY" "$SCRIPT_DIR/test_p1_occupancy_contract.py" --parent-source "$OLD" --snapshot "$SNAP"
"$PY" "$SCRIPT_DIR/p1_occupancy_contract_repair.py" --root "$ROOT"
if [[ "$MODE" == --offline ]]; then
  echo 'P1_OFFLINE_COMPLETE_NO_CARLA_RUN';exit 0
fi
# A REAL run changes CARLA world and spawns actors. No switching shared worlds.
export PYTHONPATH="$ROOT/carla/PythonAPI/carla:$ROOT/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${PYTHONPATH:-}"
"$PY" - <<'PY'
import carla, hashlib
cl=carla.Client('127.0.0.1',31120);cl.set_timeout(20)
w=cl.get_world();m=w.get_map();a=w.get_actors()
assert m.name.replace('\\','/').split('/')[-1]=='Town15','not Town15; refusing map switch'
assert hashlib.sha256(m.to_opendrive().encode('utf-8')).hexdigest()=='8e9c69cfb3e750ace3e5d6b057e46786952a073c3924f319bbabb1a166dc2c17'
assert len(a.filter('vehicle.*'))==0 and len(a.filter('walker.pedestrian.*'))==0,'nonempty world'
print('P1_REAL_WORLD_PREFLIGHT=Town15_SHA_MATCH_0_VEHICLES_0_WALKERS')
PY
CREATELOG="$BASE/p1_occupancy_capsule_create_$(date +%Y%m%d_%H%M%S)_$$.log"
CREATE_OUTPUT=$("$PY" "$SCRIPT_DIR/p1_occupancy_contract_repair.py" --root "$ROOT" --create)
printf '%s\n' "$CREATE_OUTPUT" | tee "$CREATELOG"
CAP=$(printf '%s\n' "$CREATE_OUTPUT" | sed -n 's/^P1_OCCUPANCY_NEW_CAPSULE=//p' | tail -1)
[[ -n "$CAP" ]] || { echo 'P1_CANDIDATE_CAPSULE_PATH_NOT_RETURNED' >&2; exit 2; }
CAND_SHA=ed27a2b78ffc7c3e6159db85d5756ff2ad9bfef754e79043225d343f5840854f
ACTUAL_SHA=$(sha256sum "$CAP/hv_v8/p1_native_audit.py" | cut -d' ' -f1)
[[ "$ACTUAL_SHA" == "$CAND_SHA" ]] || { echo "P1_WRONG_CANDIDATE_SHA=$ACTUAL_SHA" >&2; exit 2; }
[[ -f "$CAP/hv_v8/p1_native_audit.py" ]] || exit 2
# Validate snapshot and 95 manifest before materializing a one-record copy.
RUNSTAMP="$(date +%Y%m%d_%H%M%S)_$$"
STAGE="$BASE/p1_occupancy_record100_manifest_$RUNSTAMP"
MANIFEST="$STAGE/record_100.jsonl"
"$PY" - "$BASE/manifests/p1_calibration95.jsonl" "$SNAP" "$STAGE" <<'PY'
import sys,hashlib,json
from pathlib import Path
manifest,snap,stage=map(Path,sys.argv[1:])
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56'
assert hashlib.sha256(snap.read_bytes()).hexdigest()=='37bd5c64865372a4627b9a7c9d53f03bbe4b8cb2b3dc7ff8979a3f0d203447c7'
rows=[x for x in manifest.read_text().splitlines() if x.strip()]
assert len(rows)==95
r=json.loads(rows[8]);assert r['record_index']==100 and r['scene_id']=='RouteScenario_24333_idx27_rep0_frame145'
assert Path(r['snapshot_path']).resolve()==snap.resolve()
stage.mkdir(parents=True,exist_ok=False)
(stage/'record_100.jsonl').write_text(rows[8]+'\n')
print('P1_OCCUPANCY_RECORD100_MANIFEST_HASH_VERIFIED')
PY
export PYTHONPATH="$CAP:$ROOT/RIFT:$ROOT/scenario-dreamer:$ROOT/carla/PythonAPI:$ROOT/carla/PythonAPI/carla:$ROOT/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${PYTHONPATH:-}"
OUT="$BASE/p1_occupancy_real_record100_$RUNSTAMP"
LOG="$OUT.log"
ORIGIN=$(cd "$CAP" && "$PY" -c 'import importlib.util;print(importlib.util.find_spec("hv_v8.p1_native_audit").origin)')
[[ "$ORIGIN" == "$CAP/hv_v8/p1_native_audit.py" ]] || { echo "WRONG_MODULE_ORIGIN=$ORIGIN" >&2;exit 2; }
echo "P1_REAL_CANDIDATE_CAPSULE=$CAP"
echo "P1_REAL_RECORD100_OUTPUT=$OUT"
echo "P1_REAL_RECORD100_LOG=$LOG"
set +e
(
 cd "$CAP"
 "$PY" -m hv_v8.p1_native_audit \
  --manifest "$MANIFEST" \
  --rift-root "$ROOT/RIFT" \
  --rift-recognition-config "$ROOT/RIFT/rift/cbv/recognition/config/attention.yaml" \
  --rift-mode eval --carla-pythonapi "$ROOT/carla/PythonAPI/carla" \
  --map-identity-manifest "$BASE/manifests/p1_map_identity_manifest_v2_20261017_154737.json" \
  --host 127.0.0.1 --port 31120 --startup-timeout 600 \
  --startup-rpc-timeout 5 --startup-poll-interval 2 --rpc-timeout 120 \
  --expected-carla-version 0.9.15 --limit 1 --output "$OUT"
) > "$LOG" 2>&1
RC=$?
set -e
echo "P1_REAL_RECORD100_PROCESS_EXIT=$RC"
tail -n 18 "$LOG"
if [[ "$RC" -ne 0 || ! -f "$OUT/record_0100.json" || ! -f "$OUT/summary.json" ]];then
 echo 'P1_REAL_RECORD100_EVIDENCE_INCOMPLETE_DO_NOT_PROCEED';exit 3
fi
LEDGER="$BASE/p1_occupancy_record100_ledger_$RUNSTAMP.json"
"$PY" "$SCRIPT_DIR/p1_calibration_ledger.py" --snapshot "$SNAP" \
 --record "$OUT/record_0100.json" --summary "$OUT/summary.json" \
 --frozen-source "$CAP/hv_v8/p1_native_audit.py" --output "$LEDGER"
"$PY" - "$OUT/record_0100.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));o=r.get('fixed_model3_scene_occupancy',{})
print('P1_REAL_OCCUPANCY_STATUS=',o.get('status'))
print('P1_REAL_OCCUPANCY_REASON=',o.get('reason'))
print('P1_REAL_EXPECTED_BACKGROUND_COUNT=',o.get('expected_background_count'))
print('P1_REAL_BACKGROUND_ATTEMPTS=',len(o.get('background',[])))
print('P1_REAL_EXPECTED_TARGET_COUNT=',o.get('expected_target_count'))
print('P1_REAL_JOINT_MODEL3_SPAWN_ACCEPTED=',o.get('joint_model3_spawn_accepted'))
print('P1_REAL_CLEANUP_VERIFIED=',o.get('cleanup_verified'))
print('P1_REAL_FULL_SCENE_PHYSICAL_FEASIBILITY=',o.get('full_scene_physical_feasibility'))
assert o.get('expected_background_count')==7 and o.get('expected_target_count')==2
assert len(o.get('background',[]))==7
assert o.get('cleanup_verified') is True
assert o.get('full_scene_physical_feasibility')=='unknown'
print('P1_OCCUPANCY_POPULATION_REAL_RECORD100_CHECK_PASS_NOT_P1_PHYSICS_PASS')
PY
