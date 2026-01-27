# 建立商品 JSON 與鄰接關係

import json
import sys
import os

# 為每個商品 ASIN 分配唯一的數字 ID
def assign_num_id(u, node_to_num_id): 
    if u not in node_to_num_id:
        node_to_num_id[u] = len(node_to_num_id)
    return node_to_num_id[u]

# 初始化節點資料結構
def init_node(num_id):
    return {
        "node_id": num_id,
        "node_type": 0,
        "node_weight": 1.0,
        # neighbor 中 "0" 代表相似圖邊 (also_view), "1" 代表互補圖邊 (also_buy)
        "neighbor": {"0": [], "1": [], "2": [], "3": []},
        "uint64_feature": {}, # 存放 CID2, CID3, Price
        "edge": []
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python 4_data_formulator.py <dataset_name>")
        sys.exit(1)

    data_name = sys.argv[1]
    
    # 設定輸入與輸出路徑
    sim_edges_path = f"./data_preprocess/tmp/filtered_{data_name}_sim.edges"
    cor_edges_path = f"./data_preprocess/tmp/filtered_{data_name}_cor.edges"
    meta_path = f"./data_preprocess/tmp/filtered_meta_{data_name}.json"
    
    output_dir = "./data_preprocess/data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_file_path = os.path.join(output_dir, f"{data_name}.json")

    # ID 映射與字典檔案
    id_dict_file = open(f'./data_preprocess/tmp/{data_name}_id_dict.txt', 'w')
    cid2_dict_file = open(f'./data_preprocess/tmp/{data_name}_cid2_dict.txt', 'w')
    cid3_dict_file = open(f'./data_preprocess/tmp/{data_name}_cid3_dict.txt', 'w')

    node_to_num_id = {}
    node_data = {}
    valid_node_asin = set()

    print(f"Formulating JSON data for {data_name}...")

    # 1. 處理相似邊 (Type 0)
    with open(sim_edges_path, 'r') as f:
        for line in f:
            u, v, _ = line.strip('\n').split('\t')
            valid_node_asin.add(u)
            valid_node_asin.add(v)
            uid = assign_num_id(u, node_to_num_id)
            vid = assign_num_id(v, node_to_num_id)
            
            if uid not in node_data: node_data[uid] = init_node(uid)
            if vid not in node_data: node_data[vid] = init_node(vid)
            
            node_data[uid]['neighbor']['0'].append(vid)
            node_data[vid]['neighbor']['0'].append(uid)

    # 2. 處理互補邊 (Type 1)
    with open(cor_edges_path, 'r') as f:
        for line in f:
            u, v, _ = line.strip('\n').split('\t')
            valid_node_asin.add(u)
            valid_node_asin.add(v)
            uid = assign_num_id(u, node_to_num_id)
            vid = assign_num_id(v, node_to_num_id)
            
            if uid not in node_data: node_data[uid] = init_node(uid)
            if vid not in node_data: node_data[vid] = init_node(vid)
            
            node_data[uid]['neighbor']['1'].append(vid)
            node_data[vid]['neighbor']['1'].append(uid)

    # 3. 填充節點特徵 (CID2, CID3, Price)
    cid2_dict, cid3_dict = {}, {}
    def get_id(s, d):
        if s not in d: d[s] = len(d)
        return d[s]

    with open(meta_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            asin = data['asin']
            if asin in valid_node_asin:
                # 提取第 2, 3 級類別與價格
                c2, c3 = data['category'][2], data['category'][3]
                price = float(data['price'][1:]) if data['price'].startswith('$') else float(data['price'])
                
                num_id = node_to_num_id[asin]
                cid2_id = get_id(c2, cid2_dict)
                cid3_id = get_id(c3, cid3_dict)
                
                # 格式：{"0": [CID2_ID], "1": [CID3_ID], "2": [Price]}
                node_data[num_id]['uint64_feature'] = {"0": [cid2_id], "1": [cid3_id], "2": [price]}

    # 寫入字典檔案
    for asin, num_id in cid2_dict.items(): cid2_dict_file.write(f"{num_id}\t{asin}\n")
    for asin, num_id in cid3_dict.items(): cid3_dict_file.write(f"{num_id}\t{asin}\n")
    for asin, num_id in node_to_num_id.items(): id_dict_file.write(f"{asin}\t{num_id}\n")

    # 4. 輸出最終 JSON 檔案
    with open(out_file_path, 'w') as out_file:
        for u in sorted(node_data.keys()):
            u_data = node_data[u]
            # 將鄰居列表轉為權重字典 (預設權重 1.0)
            for t in ['0', '1']:
                u_data['neighbor'][t] = {v: 1.0 for v in u_data['neighbor'][t]}
            
            json.dump(u_data, out_file)
            out_file.write('\n')

    print(f"Finished! Total nodes formulated: {len(node_data)}")
    print(f"Data saved to: {out_file_path}")

if __name__ == '__main__':
    main()