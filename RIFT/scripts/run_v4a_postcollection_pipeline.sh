#!/usr/bin/env bash
# V4-A post-collection pipeline.
#
# This script deliberately stops at the first failed gate.  It never deletes or
# overwrites a dataset, checkpoint, collection source, or report.  Every Python
# command's merged stdout/stderr is emitted to the terminal and stored in one
# valid JSON document at $RUN_DIR/command_output.json.
#
# Required environment variables:
#   ROOT, HV_ROOT, PY, AE_CKPT, LDM_CKPT, V3_BEST
# Optional:
#   V4A_ALL_COLLECT  (the completed all-route V4-A collection directory)
#   V4A_UNION_R4     (existing Phase7 + earlier V4-A union)
#   V4A_RUN_TAG      (default: v4a_postcollection_YYYYmmdd_HHMMSS)
#   V4A_A0_GPU       (default: 1)
#   V4A_EVAL_MAX_RECORDS (default: 1000; set -1 for the whole test split)
#   RUN_A0_ORACLE=1  (also run four-sample offline oracle; never online)

set -Eeuo pipefail

: "${ROOT:?export ROOT first}"
: "${HV_ROOT:?export HV_ROOT first}"
: "${PY:?export PY first}"
: "${AE_CKPT:?export AE_CKPT first}"
: "${LDM_CKPT:?export LDM_CKPT first}"
: "${V3_BEST:?export V3_BEST first}"

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:${PYTHONPATH:-}"

V4A_ALL_COLLECT="${V4A_ALL_COLLECT:-$HV_ROOT/rift_labeled_v4a_non_town12_allroute_30k_16workers}"
V4A_UNION_R4="${V4A_UNION_R4:-$HV_ROOT/rift_labeled_v4a_union_phase7_r4}"
V4A_RUN_TAG="${V4A_RUN_TAG:-v4a_postcollection_$(date +%Y%m%d_%H%M%S)}"
V4A_A0_GPU="${V4A_A0_GPU:-1}"
V4A_EVAL_MAX_RECORDS="${V4A_EVAL_MAX_RECORDS:-1000}"
RUN_A0_ORACLE="${RUN_A0_ORACLE:-0}"

RUN_DIR="$HV_ROOT/eval/$V4A_RUN_TAG"
V4A_COLLECTION_SUMMARY="$RUN_DIR/collection_summary.json"
V4A_TRAIN_UNION="${V4A_TRAIN_UNION:-$HV_ROOT/rift_labeled_v4a_union_phase7_${V4A_RUN_TAG}}"
V4A_MERGE_SUMMARY="$RUN_DIR/merge_summary.json"
V4A_DATA="${V4A_DATA:-$HV_ROOT/datasets/hv_v4a_fullfov_forward_${V4A_RUN_TAG}}"
V4A_BUILD_DIAGNOSTICS="$RUN_DIR/dataset_build_diagnostics.json"
V4A_DATA_AUDIT="$RUN_DIR/dataset_audit.json"
V4A_A0="${V4A_A0:-$HV_ROOT/checkpoints/hv_v4a_a0_fullfov_forward_from_v3_${V4A_RUN_TAG}}"
V4A_A0_EVAL="$RUN_DIR/a0_test_raw_single.json"
V4A_A0_GENERATED="$RUN_DIR/a0_test_raw_single_generated.jsonl"
V4A_A0_ORACLE="$RUN_DIR/a0_test_bestof4_oracle.json"
V4A_A0_ORACLE_GENERATED="$RUN_DIR/a0_test_bestof4_oracle_generated.jsonl"
EVENTS_JSON="$RUN_DIR/command_output.json"
RUN_MANIFEST="$RUN_DIR/run_manifest.json"

if [ -e "$RUN_DIR" ]; then
  printf 'refusing to overwrite existing run directory: %s\nChoose a new V4A_RUN_TAG.\n' "$RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
printf '%s\n' '{"schema_version":"v4a_postcollection_command_output_v1","events":[]}' > "$EVENTS_JSON"

write_manifest() {
  "$PY" - "$RUN_MANIFEST" <<'PY'
import json, os, sys
keys = [
    "ROOT", "HV_ROOT", "V4A_ALL_COLLECT", "V4A_UNION_R4", "V4A_TRAIN_UNION",
    "V4A_DATA", "V4A_A0", "V3_BEST", "AE_CKPT", "LDM_CKPT", "V4A_EVAL_MAX_RECORDS",
]
json.dump({"schema_version": "v4a_postcollection_pipeline_v1",
           "environment": {key: os.environ.get(key) for key in keys},
           "online_policy": "no_clamp_no_lane_snap_no_rerank_no_online_best_of_n"},
          open(sys.argv[1], "w"), indent=2)
PY
}

append_event() {
  local stage="$1" rc="$2" log_file="$3"
  "$PY" - "$EVENTS_JSON" "$stage" "$rc" "$log_file" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[4])
text = path.read_text(encoding="utf-8", errors="replace")
event = {
    "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "stage": sys.argv[2], "return_code": int(sys.argv[3]), "terminal_output": text,
}
output = pathlib.Path(sys.argv[1])
document = json.loads(output.read_text(encoding="utf-8"))
document["events"].append(event)
temporary = output.with_suffix(".tmp")
temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
PY
}

run_stage() {
  local stage="$1"; shift
  local log_file="$RUN_DIR/${stage}.terminal.log"
  local rc=0
  printf '\n===== START %s =====\n' "$stage" | tee "$log_file"
  set +e
  "$@" 2>&1 | tee -a "$log_file"
  rc=${PIPESTATUS[0]}
  set -e
  printf '===== END %s rc=%s =====\n' "$stage" "$rc" | tee -a "$log_file"
  append_event "$stage" "$rc" "$log_file"
  if [ "$rc" -ne 0 ]; then
    printf 'STOP: %s failed. Full output: %s; JSON: %s\n' "$stage" "$log_file" "$EVENTS_JSON" >&2
    return "$rc"
  fi
}

run_python_check() {
  local stage="$1"; shift
  run_stage "$stage" "$PY" -c "$*"
}

export V4A_ALL_COLLECT V4A_UNION_R4 V4A_TRAIN_UNION V4A_DATA V4A_A0 V3_BEST AE_CKPT LDM_CKPT V4A_EVAL_MAX_RECORDS
write_manifest

run_python_check preflight '
import os, sys
required = ("ROOT", "HV_ROOT", "V4A_ALL_COLLECT", "V4A_UNION_R4", "V3_BEST", "AE_CKPT", "LDM_CKPT")
for key in required:
    value = os.environ[key]
    print(key, value, "exists=", os.path.exists(value))
if not os.path.isdir(os.environ["V4A_ALL_COLLECT"]): sys.exit("missing V4A_ALL_COLLECT")
if not os.path.isdir(os.environ["V4A_UNION_R4"]): sys.exit("missing V4A_UNION_R4")
for key in ("V3_BEST", "AE_CKPT", "LDM_CKPT"):
    if not os.path.isfile(os.environ[key]): sys.exit("missing checkpoint: " + key)
'

run_stage collection_summary "$PY" "$ROOT/RIFT/scripts/summarize_high_value_labeled_snapshots.py" \
  --dump_dir "$V4A_ALL_COLLECT" --min_positive_agents 2 --min_forward_positive_agents 2 \
  --fov 64 --forward_min_x 0 --target_fov_margin 2 --output_json "$V4A_COLLECTION_SUMMARY"

run_python_check merge_preflight '
import os, sys
for path in (os.environ["V4A_TRAIN_UNION"],):
    print("must_not_exist", path, "exists=", os.path.exists(path))
    if os.path.exists(path): sys.exit("refusing to overwrite existing merge output: " + path)
'
run_stage merge_sources "$PY" "$ROOT/RIFT/scripts/merge_labeled_snapshot_sources.py" \
  --input_dir "$V4A_UNION_R4" --input_dir "$V4A_ALL_COLLECT" \
  --output_dir "$V4A_TRAIN_UNION" --summary_json "$V4A_MERGE_SUMMARY"

run_python_check dataset_preflight '
import os, sys
path=os.environ["V4A_DATA"]
print("must_not_exist", path, "exists=", os.path.exists(path))
if os.path.exists(path): sys.exit("refusing to overwrite existing dataset output: " + path)
'
run_stage build_strict_v4a_dataset "$PY" "$ROOT/scenario-dreamer/build_high_value_scene_dataset_v4.py" \
  --input_dir "$V4A_TRAIN_UNION" --output_dir "$V4A_DATA" --map_source hd_map \
  --rift_root "$ROOT/RIFT" --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \
  --num_generate_agents 2 --forward_min_x 0 --target_fov_margin 2 \
  --quality_filter_v2 --min_target_speed 0 --max_target_lane_distance 5 \
  --max_town_fraction 0.35 --max_topology_fraction 0.75 --min_accepted_records 30000 \
  --min_records_per_town_per_split 1 --min_records_per_topology_per_split 1 \
  --diagnostics_json "$V4A_BUILD_DIAGNOSTICS" --seed 42

run_stage audit_strict_v4a_dataset "$PY" "$ROOT/scenario-dreamer/audit_high_value_scene_dataset_v4.py" \
  --train_jsonl "$V4A_DATA/train.jsonl" --val_jsonl "$V4A_DATA/val.jsonl" --test_jsonl "$V4A_DATA/test.jsonl" \
  --fov 64 --forward_min_x 0 --target_fov_margin 2 --max_num_agents 61 \
  --max_target_lane_distance 5 --max_town_fraction 0.35 --max_topology_fraction 0.75 \
  --min_records_per_town_per_split 1 --min_records_per_topology_per_split 1 --output_json "$V4A_DATA_AUDIT"

run_python_check a0_preflight '
import os, sys
path=os.environ["V4A_A0"]
print("must_not_exist", path, "exists=", os.path.exists(path))
if os.path.exists(path): sys.exit("refusing to overwrite existing A0 output: " + path)
'
run_stage train_v4a_a0 env "CUDA_VISIBLE_DEVICES=$V4A_A0_GPU" "$PY" "$ROOT/scenario-dreamer/train_high_value_scene_generator_v4a.py" \
  --train_jsonl "$V4A_DATA/train.jsonl" --val_jsonl "$V4A_DATA/val.jsonl" --output_dir "$V4A_A0" \
  --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" --resume_checkpoint "$V3_BEST" \
  --num_generate_agents 2 --batch_size 32 --epochs 30 --lr 1e-5 --seed 0 \
  --decoded_state_loss_weight 0.4 --decoded_position_loss_weight 2.0 --decoded_heading_loss_weight 0.7 \
  --decoded_speed_loss_weight 1.0 --decoded_size_loss_weight 0.5 --physical_bounds_loss_weight 0.3 \
  --lane_corridor_loss_weight 0.10 --lane_heading_loss_weight 0.0 --initial_clearance_loss_weight 0.10 \
  --forward_min_x 0 --target_fov_margin 2 --max_target_lane_distance 5 --best_metric val_decoded_position_l1

run_stage eval_v4a_a0_raw_single env "CUDA_VISIBLE_DEVICES=$V4A_A0_GPU" "$PY" "$ROOT/scenario-dreamer/eval_high_value_scene_generator_v4a.py" \
  --jsonl "$V4A_DATA/test.jsonl" --checkpoint "$V4A_A0/best.ckpt" \
  --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" --num_generate_agents 2 \
  --num_samples_per_condition 1 --max_records "$V4A_EVAL_MAX_RECORDS" --fov 64 --forward_min_x 0 \
  --target_fov_margin 2 --max_target_lane_distance 5 --output_json "$V4A_A0_EVAL" \
  --output_generated_jsonl "$V4A_A0_GENERATED"

if [ "$RUN_A0_ORACLE" = "1" ]; then
  run_stage eval_v4a_a0_bestof4_oracle env "CUDA_VISIBLE_DEVICES=$V4A_A0_GPU" "$PY" "$ROOT/scenario-dreamer/eval_high_value_scene_generator_v4a.py" \
    --jsonl "$V4A_DATA/test.jsonl" --checkpoint "$V4A_A0/best.ckpt" \
    --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" --num_generate_agents 2 \
    --num_samples_per_condition 4 --max_records "$V4A_EVAL_MAX_RECORDS" --fov 64 --forward_min_x 0 \
    --target_fov_margin 2 --max_target_lane_distance 5 --output_json "$V4A_A0_ORACLE" \
    --output_generated_jsonl "$V4A_A0_ORACLE_GENERATED"
fi

run_python_check final_status '
import json, os
paths={"collection": os.environ["V4A_ALL_COLLECT"], "union": os.environ["V4A_TRAIN_UNION"],
       "dataset": os.environ["V4A_DATA"], "a0_best": os.path.join(os.environ["V4A_A0"], "best.ckpt")}
print(json.dumps({key: {"path": value, "exists": os.path.exists(value)} for key, value in paths.items()}, indent=2))
'
printf '\nCOMPLETE: command transcript JSON: %s\n' "$EVENTS_JSON"
printf 'Run manifest: %s\n' "$RUN_MANIFEST"
printf 'A0 is an offline experimental checkpoint only. Do not enable online best-of-N, clamp, lane snapping, or audit registration.\n'
