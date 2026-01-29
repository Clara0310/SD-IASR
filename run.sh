#!/bin/bash

# 1. 定義資料集名稱
DATASET="Grocery_and_Gourmet_Food"

# 2. 模型基礎超參數
EMB_DIM=16
LR=0.005
BATCH_SIZE=256
EPOCHS=1000
PATIENCE=50
MAX_SEQ_LEN=50

# 3. SD-IASR 核心解耦參數
# low_k 與 mid_k 控制圖譜濾波器的傳播步數
LOW_K=2
MID_K=2
# lambda_3 控制模型正則化強度
LAMBDA_3=0.01

# 4. 執行訓練指令
# 移除了 ALPHA，並加入了 --resume 續跑與 --max_seq_len 參數
python run.py \
    --dataset $DATASET \
    --embedding_dim $EMB_DIM \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --max_seq_len $MAX_SEQ_LEN \
    --low_k $LOW_K \
    --mid_k $MID_K \
    --lambda_3 $LAMBDA_3 \
    --gpu 0 \
    --resume