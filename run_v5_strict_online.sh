#!/usr/bin/env bash
# Gate V5 checkpoint and run ONE strict CARLA/RIFT scene through existing launcher.
# This script does NOT loop seeds, retry, snap, clamp or register CBVs.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PY:-python}"
TAG=""; ROUTES=""; SEED=""; GPU="0"; BASE_PORT=31000; BASE_TM_PORT=51000
CHECKPOINT="${V5_CKPT:-$ROOT/data/hv_init/checkpoints/hv_v5_joint_raw_strong_route_20261012/best.ckpt}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2";shift 2;;
    --routes) ROUTES="$2";shift 2;;
    --seed) SEED="$2";shift 2;;
    --gpu) GPU="$2";shift 2;;
    --base-port) BASE_PORT="$2";shift 2;;
    --base-tm-port) BASE_TM_PORT="$2";shift 2;;
    --checkpoint) CHECKPOINT="$2";shift 2;;
    *) echo "Unknown argument: $1" >&2;exit 2;;
  esac
done
[ -n "$TAG" ] && [ -n "$ROUTES" ] && [ -n "$SEED" ] || {
  echo "Usage: bash run_v5_strict_online.sh --tag TAG --routes ID --seed N [--gpu N] [--checkpoint CKPT]" >&2
  exit 2
}
[ -f "$CHECKPOINT" ] || { echo "Missing checkpoint: $CHECKPOINT" >&2;exit 1; }
[ -f "$ROOT/run_strict_hv_eval.sh" ] || { echo "Existing strict launcher not found" >&2;exit 1; }
export V5_CHECKPOINT="$CHECKPOINT"
"$PY" - <<'PY'
import json,os,torch
from pathlib import Path
ckpt=Path(os.environ['V5_CHECKPOINT'])
try: d=torch.load(str(ckpt),map_location='cpu',weights_only=False)
except TypeError:d=torch.load(str(ckpt),map_location='cpu')
assert d.get('schema_version')=='hv_v5_raw_checkpoint_v1','V5 checkpoint provenance mismatch'
assert d.get('config',{}).get('map_mode')=='right_handed','Map-mode mismatch'
ready=ckpt.parent/'training_readiness.json'
assert ready.is_file(),'Missing S1 readiness proof'
r=json.loads(ready.read_text())
assert r.get('ready_for_limited_CARLA_test') is True, 'S1 gate failed: no online test'
assert int(r['best_epoch'])==int(d['epoch']), 'Readiness metadata does not match selected checkpoint epoch'
print('V5 S1 readiness verified:',r.get('best_epoch'),r.get('best_s1',{}).get('joint_plus_route_proxy'))
PY
# Everything else is delegated to the proven strict, one-shot RIFT launcher.
ROOT="$ROOT" PY="$PY" bash "$ROOT/run_strict_hv_eval.sh" \
  --tag "$TAG" --routes "$ROUTES" --seed "$SEED" --gpu "$GPU" \
  --base-port "$BASE_PORT" --base-tm-port "$BASE_TM_PORT" \
  --checkpoint "$CHECKPOINT" \
  --entry high_value_init.scene_generator_v5:ScenarioDreamerHighValueSceneGeneratorV5 \
  --generation-mode full_fov_inpainting --map-coordinate-mode right_handed
