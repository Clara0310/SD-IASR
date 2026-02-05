#!/bin/bash

# 1. 定義資料集名稱
DATASET="Grocery_and_Gourmet_Food"

# 2. 模型基礎超參數
EMB_DIM=64 #提升嵌入維度以增強表達能力
BERT_DIM=768
LR=0.001 #0.005 -> 0.001 ，降低學習率以提升訓練穩定性
BATCH_SIZE=256
EPOCHS=1000
PATIENCE=50
MAX_SEQ_LEN=50

# 3. SD-IASR 核心解耦參數
# low_k 與 mid_k 控制圖譜濾波器的傳播步數
LOW_K=5 #2 -> 3 ，增加低頻傳播步數以捕捉更多鄰居資訊
MID_K=5 #2 -> 3 ，增加中頻傳播步數以捕捉更多鄰居資訊

# Transformer 相關參數
LAYERS=2      # 增加 Transformer 深度
NHEAD=8      # 增加注意力頭數以提升模型表達能力

# loss 權重參數
LAMBDA_1=1.0
LAMBDA_2=1.0
LAMBDA_3=0.01

# 4. 執行訓練指令
# 移除了 ALPHA，並加入了 --resume 續跑與 --max_seq_len 參數
python run.py \
    --dataset $DATASET \
    --embedding_dim $EMB_DIM \
    --bert_dim $BERT_DIM \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --max_seq_len $MAX_SEQ_LEN \
    --low_k $LOW_K \
    --mid_k $MID_K \
    --num_layers $LAYERS \
    --nhead $NHEAD \
    --lambda_1 $LAMBDA_1 \
    --lambda_2 $LAMBDA_2 \
    --lambda_3 $LAMBDA_3 \
    --gpu 0 \
    "$@" #彈性接收指令的參數（ex. resume or not）
