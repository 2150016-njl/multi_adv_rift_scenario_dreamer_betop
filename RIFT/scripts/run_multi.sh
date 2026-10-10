#!/bin/bash

MAX_RETRIES=5        # max retry count
RETRY_COUNT=0        # initial retry count

EGO_CFG="pdm_lite.yaml"
CBV_CFG="rift_pluto.yaml"
MODE="train_cbv"
REPETITIONS=2
SEED=0
GPU_ID=0
RENDER=false

while getopts "t:e:c:m:r:s:g:v" opt; do
  case ${opt} in
    t) MAX_RETRIES=$OPTARG ;;       # --max_retries
    e) EGO_CFG=$OPTARG ;;          # --ego_cfg
    c) CBV_CFG=$OPTARG ;;          # --cbv_cfg
    m) MODE=$OPTARG ;;               # --mode
    r) REPETITIONS=$OPTARG ;;      # --repetitions
    s) SEED=$OPTARG ;;               # --seed
    g) GPU_ID=$OPTARG ;;           # --gpu
    v) RENDER=true ;;                # --render
    \?) echo "Usage: $0 [-t max_retries] [-e EGO_CFG] [-c CBV_CFG] [-m MODE] [-r REPETITIONS] [-s SEED] [-g GPU_ID] [-v]" 1>&2; exit 1 ;;
  esac
done

# pre cleaning
bash scripts/clean_carla.sh $GPU_ID

# loop until success or max retries
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    # log current retry count
    echo ">> Trying $RETRY_COUNT of $MAX_RETRIES times..."

    # run the script
    bash -e scripts/run_once.sh $EGO_CFG $CBV_CFG $MODE $REPETITIONS $SEED $GPU_ID $RENDER
    EXIT_STATUS=$?
    
    # check if the script executed successfully
    if [ $EXIT_STATUS -eq 0 ]; then
        echo ">> run.py executed successfully!"
        break  # break the loop
    elif [ $EXIT_STATUS -eq 99 ]; then
        echo ">> Retryable Error detected in attempt $RETRY_COUNT, retrying..."

        # further cleaning
        bash scripts/clean_carla.sh $GPU_ID

        sleep 5  # wait for 5 seconds before retrying
    
    else
        echo ">> Unknown error detected in attempt $RETRY_COUNT, Exiting..."

        # further cleaning
        bash scripts/clean_carla.sh $GPU_ID

        break  # break the loop
    fi
done