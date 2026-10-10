#!/bin/bash
GPU_ID=$1

if [ "$GPU_ID" -eq 0 ]; then
    CARLA_DEVICE_ID=0
else
    CARLA_DEVICE_ID=$(($GPU_ID + 1))
fi

# Use pgrep with -f to search the full command line for processes that include both
# 'CarlaUE4-Linux-Shipping' and the specified '-graphicsadapter' option.
CARLA_BIN="CarlaUE4-Linux-Shipping"
server_pid=$(pgrep -u "$(whoami)" -f "${CARLA_BIN}.*-graphicsadapter=${CARLA_DEVICE_ID}")

if [[ -z "$server_pid" ]]; then
  exit 0
fi

pgid=$(ps -o pgid= -p "$server_pid" | tr -d ' ')

if kill -SIGKILL -- "-$pgid" 2>/dev/null; then
  echo ">> Senting SIGKILL to process group $pgid for further clean up"
else
  echo ">> Failed to send SIGKILL to process group $pgid, trying to kill individual process..."
  kill -SIGKILL "$server_pid"
fi

sleep 1

if kill -0 "$server_pid" 2>/dev/null; then
  echo ">> Failed to terminate Carla (PID $server_pid) — still alive"
else
  echo ">> Successfully kill process group $pgid"
fi