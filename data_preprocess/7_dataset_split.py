import json
import numpy as np
import random
import sys
import os

def pair_construct(user_seq_dict, num_items, train_neg_num, test_neg_num, random_seed=1):
    """
    根據使用者行為序列構造訓練、驗證與測試集。
    採用 Leave-one-out 策略。
    """
    random.seed(random_seed)
    all_items = set(range(num_items))
    train_set, val_set, test_set = [], [], []

    print(f"Constructing dataset pairs (Train neg: {train_neg_num}, Test neg: {test_neg_num})...")

    for user, seq_with_time in user_seq_dict.items():
        item_seq = [x[0] for x in seq_with_time]
        if len(item_seq) < 3:
            continue

        # 1. 測試集：最後一個互動商品作為正樣本
        # Leave-one-out 策略
        test_pos = item_seq[-1]
        test_history = item_seq[:-1]
        test_neg = random.sample(list(all_items - set(item_seq)), test_neg_num)
        test_set.append([test_history, test_pos] + test_neg)

        # 2. 驗證集：倒數第二個互動商品作為正樣本
        val_pos = item_seq[-2]
        val_history = item_seq[:-2]
        val_neg = random.sample(list(all_items - set(item_seq)), test_neg_num)
        val_set.append([val_history, val_pos] + val_neg)

        # 3. 訓練集：其餘互動作為訓練樣本
        # 每個步驟都視為一個訓練對：[目前的歷史, 下一個商品]
        for i in range(1, len(item_seq) - 2):
            train_pos = item_seq[i]
            train_history = item_seq[:i]
            train_neg = random.sample(list(all_items - set(item_seq)), train_neg_num)
            train_set.append([train_history, train_pos] + train_neg)

    return np.array(train_set, dtype=object), np.array(val_set, dtype=object), np.array(test_set, dtype=object)

def main():
    if len(sys.argv) < 2:
        print("Usage: python 7_dataset_split.py <dataset_name>")
        sys.exit(1)

    data_name = sys.argv[1]
    
    # 定義路徑
    json_path = f"./data_preprocess/data/{data_name}.json"
    seq_path = f"./data_preprocess/tmp/{data_name}_sequences_with_time.json"
    save_path = f"./data_preprocess/processed/{data_name}.npz"
    
    if not os.path.exists("./data_preprocess/processed"):
        os.makedirs("./data_preprocess/processed")

    # 1. 讀取商品資料 (JSON Lines)
    features = []
    sim_edge_index = [] # 儲存相似圖邊索引
    com_edge_index = [] # 儲存互補圖邊索引
    
    print(f"Loading product data from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            js = json.loads(line)
            # 提取特徵：[CID2_ID, CID3_ID, Price]
            features.append(list(js['uint64_feature'].values()))
            uid = js['node_id']
            # 分別提取雙圖鄰居
            for vid in js['neighbor']['0'].keys():
                sim_edge_index.append([uid, int(vid)])
            for vid in js['neighbor']['1'].keys():
                com_edge_index.append([uid, int(vid)])

    features = np.array(features).squeeze()
    num_items = features.shape[0]

    # 2. 載入行為序列
    # 修正：使用更穩健的方式讀取行為序列
    print(f"Loading sequences from {seq_path}...")
    user_seq_dict = {}
    with open(seq_path, 'r', encoding='utf-8') as f:
        try:
            # 嘗試標準讀取
            user_seq_dict = json.load(f)
        except json.JSONDecodeError:
            # 若失敗，代表是 JSON Lines 格式，改用逐行讀取
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line: continue
                # 假設格式為 {"reviewerID": [...]} 或單純的資料列
                data = json.loads(line)
                user_seq_dict.update(data)

    # 3. 執行資料集切分
    train_set, val_set, test_set = pair_construct(user_seq_dict, num_items, train_neg_num=1, test_neg_num=100)

    # 4. 儲存打包後的數據
    # 包含：特徵、雙圖邊索引、訓練/驗證/測試集
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