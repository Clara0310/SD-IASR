#!/bin/bash
# ============================================================
# RQ4: 消融實驗 - 在 Grocery 資料集上跑所有變體
# ============================================================
# 使用方式：
#   cd SD-IASR
#   bash experiments/RQ4/run_all.sh [VARIANT]
#
# 跑單一變體：bash experiments/RQ4/run_all.sh noOrtho
# 跑全部變體：bash experiments/RQ4/run_all.sh all
# ============================================================

DATASET="Grocery_and_Gourmet_Food"

# 與最佳配置一致的超參數
EMB_DIM=128
LR=0.0005
BATCH_SIZE=256
EPOCHS=1000
PATIENCE=75
LOW_K=5
MID_K=5
LAYERS=2
NHEAD=8
LAMBDA_SPEC=2.0
LAMBDA_PROTO=0.0
LAMBDA_ALPHA=0.0
TAU=0.3
DROPOUT=0.4
DECAY_DAYS=0.002
NEG_SAMPLE=200
TEST_FREQ=10
COOC_WINDOW=5
COOC_WEIGHT=1.0
MILESTONES="82,157"
LR_GAMMA=0.5
LAMBDA_CL=0.05
CL_TAU=0.2
ALPHA_CF=0.2

COMMON_ARGS="--dataset ${DATASET} \
  --embedding_dim ${EMB_DIM} --lr ${LR} --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} --patience ${PATIENCE} \
  --low_k ${LOW_K} --mid_k ${MID_K} --num_layers ${LAYERS} --nhead ${NHEAD} \
  --lambda_spec ${LAMBDA_SPEC} --lambda_proto ${LAMBDA_PROTO} --lambda_alpha ${LAMBDA_ALPHA} \
  --tau ${TAU} --dropout ${DROPOUT} --decay_days ${DECAY_DAYS} \
  --num_neg_train ${NEG_SAMPLE} --test_freq ${TEST_FREQ} \
  --cooc_window ${COOC_WINDOW} --cooc_weight ${COOC_WEIGHT} \
  --milestones ${MILESTONES} --lr_gamma ${LR_GAMMA} \
  --lambda_cl ${LAMBDA_CL} --cl_tau ${CL_TAU} \
  --alpha_cf ${ALPHA_CF}"

run_variant() {
    local VARIANT=$1
    echo ""
    echo "============================================================"
    echo "Running ablation: ${VARIANT} on ${DATASET}"
    echo "============================================================"

    LOG_FILE="training_log/ablation_${VARIANT}_${DATASET}.log"

    nohup python experiments/RQ4/run_ablation.py \
        ${COMMON_ARGS} \
        --variant ${VARIANT} \
        >> ${LOG_FILE} 2>&1 &

    echo "PID: $!"
    echo "Log: ${LOG_FILE}"
}

# 決定要跑哪些變體
TARGET=${1:-"all"}

if [ "$TARGET" == "all" ]; then
    echo "Running ALL ablation variants sequentially..."
    for V in noOrtho noSpec shareProj noDual singleGraph; do
        run_variant $V
        echo "Waiting for ${V} to finish..."
        wait
    done
    echo ""
    echo "All ablation experiments completed!"
else
    run_variant $TARGET
fi
