#!/usr/bin/env bash
set -euo pipefail
: "${CARLA_ROOT:?set CARLA_ROOT}"
: "${CARLA_PORT:=31120}"
: "${CARLA_LOG:=/tmp/carla_hv_v8_${CARLA_PORT}.log}"

exec "$CARLA_ROOT/CarlaUE4.sh" \
  -nosound \
  -nullrhi \
  "-carla-rpc-port=${CARLA_PORT}" \
  >"$CARLA_LOG" 2>&1
