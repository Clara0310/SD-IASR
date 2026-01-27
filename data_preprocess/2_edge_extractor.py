# 修改：分別提取 also_view (Sim) 與 also_buy (Rel)

import sys
import json
import os
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python 2_edge_extractor.py <dataset_name>")
        sys.exit(1)

    data_name = sys.argv[1]
    input_path = f"./data_preprocess/tmp/filtered_meta_{data_name}.json"
    
    # 讀取過濾後的元資料
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 儲存邊及其權重 (行為次數)
    # 相似性邊 (Similarity): also_view
    sim_edges = defaultdict(int)
    # 互補性邊 (Complementarity): also_buy
    rel_edges = defaultdict(int)

    print(f"Extracting Dual-View edges for {data_name}...")

    # 收集所有有效的商品 ASIN
    all_asin = set()
    for line in lines:
        data = json.loads(line)
        all_asin.add(data['asin'])

    for line in lines:
        data = json.loads(line)
        asin = data['asin']

        # 1. 處理 also_view (相似商品)
        if 'also_view' in data:
            for rid in data['also_view']:
                if rid in all_asin:
                    u, v = str(asin), str(rid)
                    if u > v: u, v = v, u # 保持無向圖一致性
                    sim_edges[(u, v)] += 1

        # 2. 處理 also_buy (互補商品)
        if 'also_buy' in data:
            for rid in data['also_buy']:
                if rid in all_asin:
                    u, v = str(asin), str(rid)
                    if u > v: u, v = v, u
                    rel_edges[(u, v)] += 1

    # 輸出相似邊檔案
    fout_sim = open(f"./data_preprocess/tmp/{data_name}_sim.edges", 'w')
    for (u, v), w in sim_edges.items():
        fout_sim.write('\t'.join([u, v, str(w)]) + '\n')
    fout_sim.close()

    # 輸出互補邊檔案
    fout_rel = open(f"./data_preprocess/tmp/{data_name}_cor.edges", 'w')
    for (u, v), w in rel_edges.items():
        fout_rel.write('\t'.join([u, v, str(w)]) + '\n')
    fout_rel.close()

    print(f"Extraction finished!")
    print(f"Total Similarity edges (also_view): {len(sim_edges)}")
    print(f"Total Complementarity edges (also_buy): {len(rel_edges)}")

if __name__ == "__main__":
    main()