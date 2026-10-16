#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CARLA_ROOT:?set CARLA_ROOT}"
: "${CARLA_PYTHONAPI:?set CARLA_PYTHONAPI}"
: "${RIFT_ROOT:=$ROOT/RIFT}"
: "${CARLA_PORT:=31120}"
: "${PAIR_MANIFEST:?set PAIR_MANIFEST}"
: "${OUT_ROOT:?set OUT_ROOT}"

mkdir -p "$OUT_ROOT"
SERVER_LOG="$OUT_ROOT/carla_server.log"
"$ROOT/scripts/start_dedicated_carla.sh" &
CARLA_PID=$!
trap 'kill "$CARLA_PID" 2>/dev/null || true; wait "$CARLA_PID" 2>/dev/null || true' EXIT

python - "$CARLA_PORT" <<'PY'
import socket, sys, time
port=int(sys.argv[1])
deadline=time.time()+600
while time.time()<deadline:
    s=socket.socket()
    try:
        s.settimeout(2)
        s.connect(("127.0.0.1",port))
        s.close()
        break
    except OSError:
        s.close()
        time.sleep(2)
else:
    raise SystemExit("CARLA TCP port did not become reachable")
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m hv_v8.p1_native_audit \
  --manifest "$PAIR_MANIFEST" \
  --rift-root "$RIFT_ROOT" \
  --rift-recognition-config "$RIFT_ROOT/rift/cbv/recognition/config/attention.yaml" \
  --rift-mode eval \
  --carla-pythonapi "$CARLA_PYTHONAPI" \
  --port "$CARLA_PORT" \
  --output "$OUT_ROOT/p1_$STAMP"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m hv_v8.p2_support \
  --manifest "$PAIR_MANIFEST" \
  --rift-root "$RIFT_ROOT" \
  --rift-recognition-config "$RIFT_ROOT/rift/cbv/recognition/config/attention.yaml" \
  --rift-mode eval \
  --carla-pythonapi "$CARLA_PYTHONAPI" \
  --port "$CARLA_PORT" \
  --sample-m 1.0 \
  --map-radius-m 140 \
  --output "$OUT_ROOT/p2_$STAMP"
