#!/bin/bash
DATASET="Grocery_and_Gourmet_Food"

# 確保在根目錄執行（或從腳本位置推算根目錄）
cd "$(dirname "$0")/.."

# 依序執行前處理步驟，路徑對齊根目錄
python data_preprocess/1_feature_filter.py $DATASET
python data_preprocess/2_edge_extractor.py $DATASET
python data_preprocess/3_edge_filter.py $DATASET
python data_preprocess/4_data_formulator.py $DATASET
python data_preprocess/5_embs_generator.py $DATASET
python data_preprocess/6_seq_constructor.py $DATASET
python data_preprocess/7_dataset_split.py $DATASET