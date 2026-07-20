#!/usr/bin/env bash
# Run every example. They are the point of this repo — if one breaks, the
# interface broke.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

# Examples run against the in-memory mock so they need neither a network nor a
# real key. To run them for real, unset this and set NEXARA_API_KEY (and
# NEXARA_BASE_URL for a local instance).
export NEXARA_USE_MOCK="${NEXARA_USE_MOCK:-1}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

failed=0
for f in examples/*.py; do
    printf '%-34s' "$(basename "$f")"
    if out=$("$PY" "$f" 2>&1); then
        echo "OK"
    else
        echo "FAILED"
        echo "$out" | sed 's/^/    /'
        failed=1
    fi
done
exit $failed
