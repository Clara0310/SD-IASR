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
PATIENCE=75
MAX_SEQ_LEN=50

# 3. SD-IASR 核心解耦參數
# low_k 與 mid_k 控制圖譜濾波器的傳播步數
LOW_K=5 #2  ，增加低頻傳播步數以捕捉更多鄰居資訊
MID_K=5 #2  ，增加中頻傳播步數以捕捉更多鄰居資訊

# Transformer 相關參數
LAYERS=2      # 增加 Transformer 深度
NHEAD=8      # 增加注意力頭數以提升模型表達能力

#lr_scheduler 相關參數 (Warmup + MultiStep)
WARM_UP_EPOCHS=5          # 前 5 個 epoch 線性增長 LR
MILESTONES="40,80"        # full softmax 收斂快，plateau 預計 Ep35~50，在 Ep40/80 切 LR
LR_GAMMA=0.5              # 每次降為原來的一半

# loss 權重參數
LAMBDA_1=0.0 # 已停用：改用 Sampled Softmax 單一損失，不再需要分支損失
LAMBDA_2=0.0 # 已停用：同上
LAMBDA_REG=0.01 # 提高正則化
LAMBDA_PROTO=0.0  # 關閉：proto loss 與 BPR 目標衝突，移除以專注排名學習
LAMBDA_SPEC=2.0   # 強力推開兩通道（原0.3太弱，Feat_Sim無法降低）
TAU=0.3       # 強去噪溫度
DROPOUT=0.4 # 提高 dropout 以加強正則化（0.3→0.4）

num_prototypes=64 # 全局意圖原型的數量


LAMBDA_DIFF=0.01   # 商品層級解耦損失（Item-level Disentangle Loss）的權重係數
GAMMA=0.05        # 圖信號
DECAY_DAYS=0.002  # 時間衰減率（每天）: 1年前的商品權重≈0.48，5年前≈0.03
LAMBDA_ALPHA=0.0  # 關閉：Feat_Sim=0.000 靠架構自然維持，讓 intent_net 自由學習個人化
TEST_FREQ=10      # 每 N 個 epoch 順帶跑一次 test 評估（0 = 關閉）
NEG_SAMPLE=200    # 恢復 200（NEG=500 對 Grocery 太難，Ep15 後完全卡死）
ALPHA_CF=0.2      # 非參數歷史 CF 分數權重（0=關閉）
COOC_WINDOW=5     # 訓練序列共現圖滑動視窗大小（0=關閉，5=考慮序列中相距5以內的商品對）
COOC_WEIGHT=1.0   # 共現邊相對於 also_view 邊的權重縮放
POP_NEG_ALPHA=0.0   # 關閉：流行度加權對 Grocery 效果不佳（流行度分布太平均）
LAMBDA_CL=0.05    # CL4SRec 對比學習損失權重（0.1→0.05，降低干擾）
CL_TAU=0.2        # InfoNCE 溫度（越小越嚴格）
LABEL_SMOOTHING=0.0  # 已停用：label smoothing 壓低 positive logit，損害 NDCG




# 4. 執行訓練指令
# 移除了 ALPHA，並加入了 --resume 續跑與 --max_seq_len 參數
python run.py \
    --dataset $DATASET \
    --gamma $GAMMA \
    --embedding_dim $EMB_DIM \
    --bert_dim $BERT_DIM \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --max_seq_len $MAX_SEQ_LEN \
    --warm_up_epochs $WARM_UP_EPOCHS \
    --milestones $MILESTONES \
    --lr_gamma $LR_GAMMA \
    --low_k $LOW_K \
    --mid_k $MID_K \
    --num_layers $LAYERS \
    --nhead $NHEAD \
    --lambda_1 $LAMBDA_1 \
    --lambda_2 $LAMBDA_2 \
    --lambda_reg $LAMBDA_REG \
    --lambda_alpha $LAMBDA_ALPHA \
    --lambda_proto $LAMBDA_PROTO \
    --lambda_spec $LAMBDA_SPEC \
    --num_prototypes $num_prototypes \
    --tau $TAU \
    --dropout $DROPOUT \
    --decay_days $DECAY_DAYS \
    --test_freq $TEST_FREQ \
    --num_neg_train $NEG_SAMPLE \
    --alpha_cf $ALPHA_CF \
    --cooc_window $COOC_WINDOW \
    --cooc_weight $COOC_WEIGHT \
    --pop_neg_alpha $POP_NEG_ALPHA \
    --lambda_cl $LAMBDA_CL \
    --cl_tau $CL_TAU \
    --label_smoothing $LABEL_SMOOTHING \
    --use_full_softmax \
    --gpu 0 \
    "$@" #彈性接收指令的參數（ex. resume or not）
