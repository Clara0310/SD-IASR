#!/bin/bash

# ====================================================
# 1. 定義資料集名稱 (支援命令列輸入)
# ====================================================
# 預設資料集名稱
DEFAULT_DATASET="Grocery_and_Gourmet_Food"

# 邏輯判斷：如果第一個參數存在且不是旗標(以 - 開頭)，則將其視為資料集名稱
if [[ -n "$1" && "$1" != -* ]]; then
    DATASET="$1"
    shift # 關鍵：移除第一個參數，避免把它重複傳給 python 導致報錯
else
    DATASET="$DEFAULT_DATASET"
fi

echo "Running on dataset: $DATASET"

# 2. 模型基礎超參數
EMB_DIM=128 #提升嵌入維度以增強表達能力
BERT_DIM=768
LR=0.0005 #0.001 調小為 0.0005，稍微調降以穩定訓練
BATCH_SIZE=256
EPOCHS=1000
PATIENCE=100
MAX_SEQ_LEN=50

# 3. SD-IASR 核心解耦參數
# low_k 與 mid_k 控制圖譜濾波器的傳播步數
LOW_K=1 #2  ，增加低頻傳播步數以捕捉更多鄰居資訊
MID_K=1 #2  ，增加中頻傳播步數以捕捉更多鄰居資訊

# Transformer 相關參數
LAYERS=2      # 增加 Transformer 深度
NHEAD=8      # 增加注意力頭數以提升模型表達能力

#lr_scheduler 相關參數
LR_MODE="max"     # 因為指標是 HR@10，所以是越大越好
LR_FACTOR=0.5     # 觸發時將學習率乘以 0.1 
LR_PATIENCE=25     # 這是排程器的耐心值（例如 15 次沒進步就降速）

# loss 權重參數
LAMBDA_1=1.0
LAMBDA_2=1.5
LAMBDA_3=0.001 # 提高 Weight Decay 正則化 (從 0.001 提升至 0.01)

LAMBDA_DIFF=0.03   # 階段十二建議的黃金分割點
GAMMA=0.1          # 稍微回升一點圖信號


# 提高 Dropout (從 0.2 提升至 0.3)
DROPOUT=0.35 

# 4. 執行訓練指令
# 移除了 ALPHA，並加入了 --resume 續跑與 --max_seq_len 參數
python run.py \
    --dataset $DATASET \
    --lambda_diff $LAMBDA_DIFF \
    --gamma $GAMMA \
    --embedding_dim $EMB_DIM \
    --bert_dim $BERT_DIM \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --max_seq_len $MAX_SEQ_LEN \
    --lr_mode $LR_MODE \
    --lr_factor $LR_FACTOR \
    --lr_patience $LR_PATIENCE \
    --low_k $LOW_K \
    --mid_k $MID_K \
    --num_layers $LAYERS \
    --nhead $NHEAD \
    --lambda_1 $LAMBDA_1 \
    --lambda_2 $LAMBDA_2 \
    --lambda_3 $LAMBDA_3 \
    --dropout $DROPOUT \
    --gpu 0 \
    "$@" #彈性接收指令的參數（ex. resume or not）
