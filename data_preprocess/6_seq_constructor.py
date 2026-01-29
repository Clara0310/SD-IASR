import json
import sys
import os
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python 6_seq_constructor.py <dataset_name>")
        sys.exit(1)

    data_name = sys.argv[1]
    # 根據你的路徑設定
    raw_interaction_path = f"./data_preprocess/raw_data/{data_name}.json"
    id_dict_path = f"./data_preprocess/tmp/{data_name}_id_dict.txt"
    output_path = f"./data_preprocess/tmp/{data_name}_sequences_with_time.json"

    if not os.path.exists(raw_interaction_path):
        print(f"Error: Interaction file {raw_interaction_path} not found.")
        sys.exit(1)

    # 1. 載入商品 ASIN 到數字 ID 的映射 (確保序列與圖節點一致)
    asin_to_id = {}
    with open(id_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                asin, num_id = parts
                asin_to_id[asin] = int(num_id)

    # 2. 讀取互動紀錄並按使用者分組
    user_history = defaultdict(list)
    print(f"Reading interactions from {raw_interaction_path}...")
    
    count = 0
    with open(raw_interaction_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                user = data['reviewerID']
                item_asin = data['asin']
                timestamp = int(data['unixReviewTime'])
                
                # 【回應你的點 2】只保留在圖中的商品，確保模型能找到對應的 Spectral Embedding
                if item_asin in asin_to_id:
                    user_history[user].append({
                        "item_id": asin_to_id[item_asin],
                        "timestamp": timestamp
                    })
                count += 1
                if count % 100000 == 0:
                    print(f"Processed {count} lines...")
            except Exception as e:
                continue

    # 3. 排序並生成序列
    final_sequences = {}
    # 【回應你的點 3】放寬長度限制，你可以根據實驗需求調整
    min_seq_len = 3   
    max_seq_len = 50  # 增加長度以保留更多長期行為

    print("Sorting sequences by time and saving timestamps...")
    for user, items in user_history.items():
        # 【回應你的點 1】嚴格按時間戳記升冪排序
        items.sort(key=lambda x: x['timestamp'])
        
        # 過濾掉連續重複點擊同一個商品的行為 (去噪)
        cleaned_items = []
        for i in range(len(items)):
            if i == 0 or items[i]['item_id'] != items[i-1]['item_id']:
                cleaned_items.append(items[i])
        
        # 檢查長度
        if len(cleaned_items) >= min_seq_len:
            # 【回應你的點 4】保留時間戳記，格式為 [ [ID, Time], [ID, Time], ... ]
            seq_with_time = [[x['item_id'], x['timestamp']] for x in cleaned_items]
            # 取得最近的行為
            final_sequences[user] = seq_with_time[-max_seq_len:]

    # 4. 儲存序列結果
    with open(output_path, 'w', encoding='utf-8') as f:
        for user, seq in final_sequences.items():
            # 將每一對 key-value 轉成 json 字串並加上換行符號
            line = json.dumps({user: seq})
            f.write(line + '\n')

    print(f"Finished!")
    print(f"Total users with valid sequences: {len(final_sequences)}")
    print(f"Sequences with timestamps saved to: {output_path}")

if __name__ == '__main__':
    main()