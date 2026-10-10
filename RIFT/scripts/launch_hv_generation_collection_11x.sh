#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}
CARLA_ROOT=${CARLA_ROOT:-$ROOT/carla}
PY=${PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}
HV_ROOT=${HV_ROOT:-$ROOT/data/hv_init}

SESSION=${SESSION:-hv_collect_s4_s8}
OUT_ROOT=${OUT_ROOT:-$HV_ROOT/rift_labeled_seed4_8_parallel}
ROUTES=${ROUTES:-rift/scenario/route/bench2drive220.xml}
SEEDS=${SEEDS:-"4 5 6 7 8"}

SAMPLE_INTERVAL_S=${SAMPLE_INTERVAL_S:-0.5}
ACTOR_RADIUS=${ACTOR_RADIUS:-80}
MAP_RADIUS=${MAP_RADIUS:-64}
MAX_FRAMES_PER_EPISODE=${MAX_FRAMES_PER_EPISODE:-400}
MAX_SNAPSHOTS_PER_EPISODE=${MAX_SNAPSHOTS_PER_EPISODE:-120}

LAUNCHER=${LAUNCHER:-auto}
if [ "$LAUNCHER" = "auto" ]; then
  if command -v tmux >/dev/null 2>&1; then
    LAUNCHER=tmux
  else
    LAUNCHER=plain
  fi
fi

export ROOT CARLA_ROOT PY HV_ROOT
export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"

mkdir -p "$OUT_ROOT"
cd "$ROOT/RIFT" || exit 1

if [ "$LAUNCHER" = "tmux" ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    exit 1
  fi
else
  echo "tmux is unavailable for this user; using plain bash parallel mode."
  echo "Outputs from 11 collectors will be interleaved in this terminal."
fi

ROUTE_RANGES=(
  "1711-2201"
  "2204-2709"
  "2715-3144"
  "3178-3540"
  "3561-3876"
  "3890-17598"
  "17635-23930"
  "24041-24784"
  "24785-25951"
  "25955-27018"
  "27494-28330"
)

launch_window() {
  local idx="$1"
  local range="$2"
  local gpu=$((idx % 9))
  local base_port=$((30000 + idx * 300))
  local tm_port=$((50000 + idx * 300))
  local window
  window=$(printf "p%02d" "$idx")
  local cmd
  cmd=$(cat <<EOF
set -u
export ROOT="$ROOT"
export CARLA_ROOT="$CARLA_ROOT"
export PY="$PY"
export HV_ROOT="$HV_ROOT"
export PYTHONPATH="$PYTHONPATH"
cd "$ROOT/RIFT" || exit 1
echo "[collector $window] routes=$range gpu=$gpu base_port=$base_port tm_port=$tm_port seeds=$SEEDS"
for seed in $SEEDS; do
  out_dir="$OUT_ROOT/${window}_seed\${seed}"
  mkdir -p "\$out_dir"
  echo "[collector $window] START seed=\$seed out=\$out_dir"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/collect_high_value_init_scenes.py \\
    --routes "$ROUTES" \\
    --routes_subset "$range" \\
    --output_dir "\$out_dir" \\
    --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \\
    --rift_speed_limits_dir "$ROOT/RIFT/data/speed_limits/Speed-Limits" \\
    --sample_interval_s "$SAMPLE_INTERVAL_S" \\
    --actor_radius "$ACTOR_RADIUS" \\
    --map_radius "$MAP_RADIUS" \\
    --max_frames_per_episode "$MAX_FRAMES_PER_EPISODE" \\
    --max_snapshots_per_episode "$MAX_SNAPSHOTS_PER_EPISODE" \\
    --seed "\$seed" \\
    --base_port "$base_port" \\
    --base_tm_port "$tm_port" \\
    --no_resume
  status=\$?
  echo "[collector $window] END seed=\$seed status=\$status"
  if [ "\$status" -ne 0 ]; then
    echo "[collector $window] failed; continuing with next seed after 10s"
    sleep 10
  fi
done
echo "[collector $window] all seeds complete"
EOF
)
  if [ "$LAUNCHER" = "tmux" ]; then
    if [ "$idx" -eq 0 ]; then
      tmux new-session -d -s "$SESSION" -n "$window" "bash -lc '$cmd'"
    else
      tmux new-window -t "$SESSION" -n "$window" "bash -lc '$cmd'"
    fi
  else
    bash -lc "$cmd" &
    PIDS+=("$!")
  fi
}

PIDS=()
for idx in "${!ROUTE_RANGES[@]}"; do
  launch_window "$idx" "${ROUTE_RANGES[$idx]}"
done

if [ "$LAUNCHER" = "tmux" ]; then
  echo "Started 11 collectors in tmux session: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  echo "Switch windows: Ctrl-b then n/p, or Ctrl-b then window number"
else
  echo "Started 11 collectors in plain mode. PIDs: ${PIDS[*]}"
  failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  exit "$failed"
fi
