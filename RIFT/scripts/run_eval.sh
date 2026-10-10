#!/bin/bash

set -e


CBV_CFG_LIST=("rift_pluto.yaml" "grpo_pluto.yaml" "ppo_pluto.yaml" "reinforce_pluto.yaml" "sft_pluto.yaml" "rtr_pluto.yaml" "rs_pluto.yaml")
GPU_LIST=(2 3 4 5 6 7)

# CBV_CFG_LIST=("fppo_rs.yaml" "pluto.yaml")
# GPU_LIST=(0 1)  

run_eval () {
    local CBV_CFG=$1
    local GPU_ID=$2

    for EGO_CFG in "pdm_lite.yaml" "plant.yaml"
    do
        for SEED in 0 1 2
        do
            if python scripts/check_eval.py --ego_name "$EGO_CFG" --cbv_name "$CBV_CFG" --seed "$SEED"; then
                echo "Evaluation finished. Skipping $CBV_CFG | $EGO_CFG | seed=$SEED"
                continue
            fi

            echo "Running $CBV_CFG | $EGO_CFG | seed=$SEED on GPU $GPU_ID"
            CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/run.py \
                --ego_cfg $EGO_CFG \
                --cbv_cfg $CBV_CFG \
                --mode eval \
                --repetitions 1 \
                --seed $SEED \
                --render \
                > /dev/null
            sleep 10
        done
    done
}

# running evaluation parallelly
for i in "${!CBV_CFG_LIST[@]}"
do
    CBV_CFG=${CBV_CFG_LIST[$i]}
    GPU_ID=${GPU_LIST[$i]}

    run_eval "$CBV_CFG" "$GPU_ID" &  # run in background
    sleep 5
done

wait  # wait for all background processes to finish