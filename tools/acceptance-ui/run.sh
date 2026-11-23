#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

background=false
if [[ "${1:-}" == --background ]]; then
    background=true
    shift
fi
port="${1:-8081}"
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo "用法：./run.sh [--background] [端口]" >&2
    exit 2
fi
python="${PYTHON:-python3}"
if ! "$background"; then
    exec "$python" -B server.py "$port"
fi

exec 9>.server-start.lock
flock 9
health() {
    curl --noproxy '*' -fsS --max-time 1 "http://127.0.0.1:$port/api/health" 2>/dev/null |
        "$python" -B -c 'import json,sys; d=json.load(sys.stdin); sys.exit(d.get("service") != "deeplink-acceptance-ui" or d.get("status") != "ok")' 2>/dev/null
}
if ! health; then
    nohup "$python" -B server.py "$port" </dev/null >>server.log 2>&1 9>&- &
    pid=$!
    for _ in {1..25}; do
        if health; then
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -n 12 server.log >&2
            exit 1
        fi
        sleep 0.2
    done
fi
if ! health; then
    tail -n 12 server.log >&2
    exit 1
fi
printf '验收实验台 http://127.0.0.1:%s\n' "$port"
