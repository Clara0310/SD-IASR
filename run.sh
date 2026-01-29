# 自動化執行所有前處理步驟

#!/bin/bash

# 定義資料集名稱
DATASET="Grocery_and_Gourmet_Food"

# 模型超參數設定
EMB_DIM=16         # 初始嵌入維度
LR=0.005           # 學習率
BATCH_SIZE=256     # 批次大小
EPOCHS=1000        # 最大訓練輪數
PATIENCE=50        # Early Stopping 的耐心值

# SD-IASR 核心超參數 (譜解耦相關)
LOW_K=2            # 低通濾波器階數 (捕捉相似性)
MID_K=2            # 中通濾波器階數 (捕捉互補性)
ALPHA=0.5          # 相似性權重
LAMBDA_3=0.01      # 正則化項權重

# 執行訓練指令 (請注意變數名稱的大小寫一致性)
python run.py \
    --dataset $DATASET \
    --embedding_dim $EMB_DIM \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --low_k $LOW_K \
    --mid_k $MID_K \
    --lambda_3 $LAMBDA_3 \
    --gpu 0
    --resume