#!/bin/bash
DATASET="Grocery_and_Gourmet_Food"

# 依序執行前處理步驟
python 1_feature_filter.py $DATASET
python 2_edge_extractor.py $DATASET
python 3_edge_filter.py $DATASET
python 4_data_formulator.py $DATASET
python 5_embs_generator.py $DATASET
python 6_seq_constructor.py $DATASET
python 7_dataset_split.py $DATASET