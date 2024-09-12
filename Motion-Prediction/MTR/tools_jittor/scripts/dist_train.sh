#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

python train.py --launcher jittor ${PY_ARGS}

