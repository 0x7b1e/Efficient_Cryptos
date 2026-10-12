#!/bin/bash
set -euo pipefail
ANYCAST_DIR="${ANYCAST_DIR:-/usr/local/lib/shadow-anycast}"
cd "$ANYCAST_DIR"
node="${1:-30}"
# Runs as root via sudo: accept only a short alphanumeric node id.
if [[ ! "$node" =~ ^[A-Za-z0-9_-]{1,16}$ ]]; then
    echo "invalid Anycast node id: $node" >&2; exit 2
fi
# Anycast may print its help after connecting. Keep its output in a log and
# wait for the tunnel to become ready instead of trusting that message.
log=$(mktemp)
trap 'rm -f "$log"' EXIT
./anycast.sh --connect "$node" >"$log" 2>&1 || true
sed '/^[[:space:]]*█████/,$d' "$log"
for _ in $(seq 1 60); do
    if ss -ltn 2>/dev/null | grep -Eq '(^|[[:space:]])(127\.0\.0\.1|0\.0\.0\.0):1087([[:space:]]|$)'; then
        echo "Anycast proxy ready on 127.0.0.1:1087 (node $node)"
        exit 0
    fi
    sleep 1
done
echo "Anycast reconnect timed out: proxy 127.0.0.1:1087 is not listening" >&2
exit 1
