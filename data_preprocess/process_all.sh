#!/bin/bash

# 1. 取得參數與設定預設值
# $1: 資料集名稱 (預設: Grocery_and_Gourmet_Food)
# $2: 訓練負採樣數 (預設: 1)
# $3: 測試負採樣數 (預設: 100, 若輸入 -1 則為全排名)
DATASET=${1:-"Grocery_and_Gourmet_Food"}
TRAIN_NEG=${2:-1}
TEST_NEG=${3:-100}

# 2. 設定以時間命名的子資料夾路徑
# 格式：data_preprocess_log/{資料集名稱}/{時間}/
TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="data_preprocess_log/${DATASET}/${TIMESTAMP}"
mkdir -p "$BASE_DIR"

LOG_FILE="${BASE_DIR}/process.log"
YAML_FILE="${BASE_DIR}/sh_setting.yaml"

# 3. 儲存參數設定到 YAML 檔
cat <<EOF > "$YAML_FILE"
# Preprocessing Parameter Settings
dataset: "$DATASET"
train_neg_num: $TRAIN_NEG
test_neg_num: $TEST_NEG
full_ranking: $( [ "$TEST_NEG" -eq -1 ] && echo "true" || echo "false" )
timestamp: "$(date '+%Y-%m-%d %H:%M:%S')"
EOF

# 4. 使用 exec 將後續輸出導向到新的 log 位置
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================================="
echo "開始執行前處理程序"
echo "資料集: $DATASET"
echo "參數設定已儲存至: $YAML_FILE"
echo "Log 紀錄中: $LOG_FILE"
echo "=========================================================="

# 5. 確保在專案根目錄執行
cd "$(dirname "$0")/.."

# 6. 依序執行 Python 腳本
echo "[Step 1/7] Running 1_feature_filter.py..."
python data_preprocess/1_feature_filter.py $DATASET

echo "[Step 2/7] Running 2_edge_extractor.py..."
python data_preprocess/2_edge_extractor.py $DATASET

echo "[Step 3/7] Running 3_edge_filter.py..."
python data_preprocess/3_edge_filter.py $DATASET

echo "[Step 4/7] Running 4_data_formulator.py..."
python data_preprocess/4_data_formulator.py $DATASET

echo "[Step 5/7] Running 5_embs_generator.py..."
python data_preprocess/5_embs_generator.py $DATASET

echo "[Step 6/7] Running 6_seq_constructor.py..."
python data_preprocess/6_seq_constructor.py $DATASET

echo "[Step 7/7] Running 7_dataset_split.py..."
python data_preprocess/7_dataset_split.py $DATASET $TRAIN_NEG $TEST_NEG

echo "=========================================================="
echo "前處理執行完畢！所有紀錄已儲存於: $BASE_DIR"
echo "完成時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

