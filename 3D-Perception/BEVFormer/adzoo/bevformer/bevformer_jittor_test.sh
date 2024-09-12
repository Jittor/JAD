#!/usr/bin/env bash

T=`date +%m%d%H%M`c

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))


WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

export JT_SAVE_MEM=1
# export lazy_execution=0

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mpirun -np ${GPUS_PER_NODE}  python   \
    $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/eval.$T