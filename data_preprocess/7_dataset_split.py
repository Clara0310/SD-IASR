import json
import numpy as np
import random
import sys
import os

def pair_construct(user_seq_dict, num_items, train_neg_num, test_neg_num, random_seed=1):
    """
    根據使用者行為序列構造訓練、驗證與測試集。
    """
    random.seed(random_seed)
    all_items = set(range(num_items))
    train_set, val_set, test_set = [], [], []

    print(f"Constructing dataset pairs (Train neg: {train_neg_num}, Test neg: {test_neg_num if test_neg_num != -1 else 'FULL_RANKING'})...")

    for user, seq_with_time in user_seq_dict.items():
        item_seq = [x[0] for x in seq_with_time]
        if len(item_seq) < 3:
            continue
        
        # 獲取所有可能的負樣本候選
        candidate_negs = list(all_items - set(item_seq))
        
        # 處理全排名邏輯：若 test_neg_num 為 -1，則取全部候選
        actual_test_neg = len(candidate_negs) if test_neg_num == -1 else min(len(candidate_negs), test_neg_num)

        # 1. 測試集
        test_pos = item_seq[-1]
        test_history = item_seq[:-1]
        test_neg = random.sample(candidate_negs, actual_test_neg)
        test_set.append([test_history, test_pos] + test_neg)

        # 2. 驗證集
        val_pos = item_seq[-2]
        val_history = item_seq[:-2]
        val_neg = random.sample(candidate_negs, actual_test_neg)
        val_set.append([val_history, val_pos] + val_neg)

        # 3. 訓練集
        for i in range(1, len(item_seq) - 2):
            train_pos = item_seq[i]
            train_history = item_seq[:i]
            actual_train_neg = min(len(candidate_negs), train_neg_num)
            train_neg = random.sample(candidate_negs, actual_train_neg)
            train_set.append([train_history, train_pos] + train_neg)

    return np.array(train_set, dtype=object), np.array(val_set, dtype=object), np.array(test_set, dtype=object)

def main():
    if len(sys.argv) < 2:
        print("Usage: python 7_dataset_split.py <dataset_name> [train_neg] [test_neg]")
        print("Note: Set [test_neg] to -1 for Full Ranking.")
        sys.exit(1)

    data_name = sys.argv[1]
    
    # 接收參數，給予預設值
    train_neg = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    test_neg = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    # 定義路徑
    json_path = f"./data_preprocess/data/{data_name}.json"
    seq_path = f"./data_preprocess/tmp/{data_name}_sequences_with_time.json"
    save_path = f"./data_preprocess/processed/{data_name}.npz"
    
    if not os.path.exists("./data_preprocess/processed"):
        os.makedirs("./data_preprocess/processed")

    # 1. 讀取商品資料 (取得 num_items)
    features = []
    sim_edge_index = []
    com_edge_index = []
    
    print(f"Loading product data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            js = json.loads(line.strip())
            features.append(list(js['uint64_feature'].values()))
            uid = js['node_id']
            for vid in js['neighbor']['0'].keys():
                sim_edge_index.append([uid, int(vid)])
            for vid in js['neighbor']['1'].keys():
                com_edge_index.append([uid, int(vid)])

    features = np.array(features).squeeze()
    num_items = features.shape[0]

    # 2. 載入行為序列
    print(f"Loading sequences from {seq_path}...")
    user_seq_dict = {}
    with open(seq_path, 'r', encoding='utf-8') as f:
        try:
            user_seq_dict = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    user_seq_dict.update(json.loads(line))

    # 3. 執行資料集切分
    train_set, val_set, test_set = pair_construct(user_seq_dict, num_items, train_neg, test_neg)

    # 4. 儲存打包後的數據
    np.savez(save_path, 
             features=features, 
             sim_edge_index=np.array(sim_edge_index),
             com_edge_index=np.array(com_edge_index), 
             train_set=train_set,
             val_set=val_set, 
             test_set=test_set)

    print(f"Success! Processed data saved to: {save_path}")
    print(f"Final items: {num_items}, Sequences: {len(user_seq_dict)}")

if __name__ == '__main__':
    main()