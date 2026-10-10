#!/bin/bash
set -e

EGO_CFG=$1
CBV_CFG=$2
MODE=$3
REPETITIONS=$4
SEED=$5
GPU_ID=$6
RENDER=$7

CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/run.py \
    --ego_cfg $EGO_CFG \
    --cbv_cfg $CBV_CFG \
    --mode $MODE \
    --repetitions $REPETITIONS \
    --seed $SEED \
    $( [ "$RENDER" = true ] && echo "--render" )




