#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

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
    echo "Python 3 was not found." >&2
    exit 1
fi
if ! "$PYTHON" -c 'import fastapi,uvicorn,cv2,h5py,numpy,scipy,httpx,pyarrow,torch,imageio_ffmpeg,ultralytics' >/dev/null 2>&1; then
    echo "Project dependencies are incomplete. Run: sh run.sh --setup" >&2
    exit 1
fi
if [ "${1:-}" = "--robots" ]; then
    shift
    exec "$PYTHON" -m app.cli robots "$@"
fi
if [ "$#" -eq 0 ]; then
    echo "Usage: sh full.sh DATASET_PATH_OR_ID (--all | --episode ID) [full options]" >&2
    echo "List robot types: sh full.sh --robots" >&2
    echo "Example: sh full.sh /data/insert_usb --all" >&2
    exit 2
fi

if [ -z "${ALICE_GPU_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    ALICE_GPU_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, - || true)
fi
export ALICE_GPU_DEVICES=${ALICE_GPU_DEVICES:-}

if [ -n "$ALICE_GPU_DEVICES" ]; then
    GPU_COUNT=$(printf '%s\n' "$ALICE_GPU_DEVICES" | awk -F, '{print NF}')
else
    GPU_COUNT=0
fi
if [ -z "${ALICE_FULL_WORKERS:-}" ]; then
    if [ "$GPU_COUNT" -gt 0 ]; then ALICE_FULL_WORKERS=$GPU_COUNT; else ALICE_FULL_WORKERS=2; fi
fi
case "$ALICE_FULL_WORKERS" in
    ''|*[!0-9]*) echo "ALICE_FULL_WORKERS must be an integer from 1 to 32." >&2; exit 2 ;;
esac
if [ "$ALICE_FULL_WORKERS" -lt 1 ]; then
    echo "ALICE_FULL_WORKERS must be an integer from 1 to 32." >&2
    exit 2
fi
if [ "$ALICE_FULL_WORKERS" -gt 32 ]; then ALICE_FULL_WORKERS=32; fi
export ALICE_FULL_WORKERS
export ALICE_VIDEO_ENCODER=${ALICE_VIDEO_ENCODER:-auto}
export ALICE_VIDEO_ACCELERATOR=${ALICE_VIDEO_ACCELERATOR:-auto}

if [ "$GPU_COUNT" -gt 0 ] && ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "Warning: $GPU_COUNT NVIDIA GPU(s) found, but this PyTorch build cannot use CUDA; frame processing will fall back to CPU." >&2
fi

CPU_COUNT=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '8')
THREADS_PER_JOB=$((CPU_COUNT / ALICE_FULL_WORKERS))
if [ "$THREADS_PER_JOB" -lt 1 ]; then THREADS_PER_JOB=1; fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$THREADS_PER_JOB}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$THREADS_PER_JOB}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$THREADS_PER_JOB}
export ALICE_OPENCV_THREADS=${ALICE_OPENCV_THREADS:-$THREADS_PER_JOB}

PORT=${ALICE_PORT:-8000}
PARALLEL=${ALICE_FULL_PARALLEL:-$ALICE_FULL_WORKERS}
case "$PARALLEL" in
    ''|*[!0-9]*) echo "ALICE_FULL_PARALLEL must be an integer from 1 to 32." >&2; exit 2 ;;
esac
if [ "$PARALLEL" -lt 1 ] || [ "$PARALLEL" -gt 32 ]; then
    echo "ALICE_FULL_PARALLEL must be an integer from 1 to 32." >&2
    exit 2
fi
if [ "${ALICE_FULL_RESTART:-0}" = "1" ]; then
    "$PYTHON" -m app.cli stop >/dev/null 2>&1 || true
fi
if [ "${ALICE_FULL_LOAD_YOLO:-0}" = "1" ]; then
    "$PYTHON" -m app.cli start --port "$PORT" --max-port "$((PORT + 10))"
else
    "$PYTHON" -m app.cli start --port "$PORT" --max-port "$((PORT + 10))" --no-model
fi

if [ -n "$ALICE_GPU_DEVICES" ]; then
    exec "$PYTHON" -m app.cli full "$@" --url auto --parallel "$PARALLEL" \
        --expect-workers "$ALICE_FULL_WORKERS" --expect-gpus "$ALICE_GPU_DEVICES"
fi
exec "$PYTHON" -m app.cli full "$@" --url auto --parallel "$PARALLEL" \
    --expect-workers "$ALICE_FULL_WORKERS"
