# 過濾特徵不完整的商品

import sys
import json
import os

# 從命令列參數取得資料集名稱
if len(sys.argv) < 2:
    print("Usage: python 1_feature_filter.py <dataset_name>")
    sys.exit(1)

data_name = sys.argv[1]

# 設定檔案路徑
raw_data_path = './data_preprocess/raw_data/meta_{}.json'.format(data_name)
output_dir = './data_preprocess/tmp/'
output_path = os.path.join(output_dir, 'filtered_meta_{}.json'.format(data_name))

# 確保輸出目錄存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(raw_data_path):
    print(f"Error: {raw_data_path} not found. Please put your Amazon meta data in ./raw_data/")
    sys.exit(1)

print(f"Filtering items for {data_name} with incomplete features...")

def is_float(s):
    """檢查字串是否可轉換為浮點數"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

# 開啟輸入的元資料檔案
total_node_num = 0
feature_sets = [set() for _ in range(3)] # 儲存 cid2, cid3, price 的唯一值

with open(raw_data_path, 'r', encoding='utf-8') as f, \
     open(output_path, 'w', encoding='utf-8') as out_file:
    
    for line in f:
        try:
            # 將每一行解析為字典 (Amazon 原始資料通常是 json 格式，但有時是 python dict 格式)
            # 這裡優先使用 json.loads，若失敗則嘗試 eval
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = eval(line)
            
            # 1. 檢查是否有基本的 category 和 price 欄位
            # 2. 論文要求至少 4 級類別 (索引 1, 2, 3 對應 cid1, cid2, cid3)
            if 'category' in data and len(data['category']) >= 4 and 'price' in data:
                cid2 = data['category'][2]
                cid3 = data['category'][3]
                price_str = data['price']
                
                if price_str == "" or price_str is None:
                    continue
                
                # 處理價格：移除貨幣符號並轉為浮點數
                # 通常 Amazon 資料價格格式如 "$10.99"
                raw_price = price_str[1:] if price_str.startswith('$') else price_str
                
                if is_float(raw_price):
                    price_val = float(raw_price)
                    
                    # 收集特徵統計 (用於除錯)
                    feature_sets[0].add(cid2)
                    feature_sets[1].add(cid3)
                    feature_sets[2].add(price_val)
                    
                    # 將有效的商品寫入輸出檔案 (保留原始 JSON 格式)
                    out_file.write(json.dumps(data) + '\n')
                    total_node_num += 1
                    
        except Exception as e:
            continue

# 列印過濾後的統計資訊
print(f'Finished! Total valid nodes: {total_node_num}')
print(f'Unique CID2: {len(feature_sets[0])}, Unique CID3: {len(feature_sets[1])}')
print(f'Filtered data saved to: {output_path}')