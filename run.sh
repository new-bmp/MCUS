#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

PORT=8000
SETUP=0
BROWSER=0
NO_MODEL=0
FOREGROUND=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --setup) SETUP=1 ;;
        --browser) BROWSER=1 ;;
        --no-browser) BROWSER=0 ;;
        --no-model) NO_MODEL=1 ;;
        --foreground) FOREGROUND=1 ;;
        --port)
            shift
            PORT=${1:?"--port requires a value"}
            ;;
        -h|--help)
            echo "Usage: sh run.sh [--setup] [--browser] [--no-model] [--foreground] [--port N]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

find_python() {
    if [ -x "$ROOT/.venv/bin/python" ]; then
        printf '%s\n' "$ROOT/.venv/bin/python"
        return
    fi
    for candidate in python3.12 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return
        fi
    done
    return 1
}

PYTHON=$(find_python || true)
if [ -z "$PYTHON" ]; then
    echo "Python 3.12 was not found." >&2
    exit 1
fi

if [ "$SETUP" -eq 1 ] && [ ! -x "$ROOT/.venv/bin/python" ]; then
    "$PYTHON" -m venv "$ROOT/.venv"
    PYTHON="$ROOT/.venv/bin/python"
fi

if [ "$SETUP" -eq 1 ]; then
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install -r "$ROOT/requirements.txt"
fi

if ! "$PYTHON" -c 'import fastapi,uvicorn,cv2,ultralytics,httpx,h5py,pyarrow,imageio_ffmpeg,scipy,torch' >/dev/null 2>&1; then
    echo "Project dependencies are incomplete. Run: sh run.sh --setup" >&2
    exit 1
fi

if [ "$FOREGROUND" -eq 1 ]; then
    set -- serve --port "$PORT"
else
    set -- start --port "$PORT" --max-port "$((PORT + 10))"
    [ "$BROWSER" -eq 0 ] || set -- "$@" --browser
fi
[ "$NO_MODEL" -eq 0 ] || set -- "$@" --no-model
exec "$PYTHON" -m app.cli "$@"
