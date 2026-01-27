# 過濾度數過低的節點

import sys
import os
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python 3_edge_filter.py <dataset_name>")
        sys.exit(1)

    data_name = sys.argv[1]
    
    # 設定邊數最少閾值 (可依需求調整，預設為 1)
    threshold = 1
    print(f"Filtering nodes with edge num threshold = {threshold}")

    # 定義檔案路徑
    sim_path = f"./data_preprocess/tmp/{data_name}_sim.edges"
    cor_path = f"./data_preprocess/tmp/{data_name}_cor.edges"

    if not os.path.exists(sim_path) or not os.path.exists(cor_path):
        print("Error: Similarity or Complementarity edge files not found.")
        sys.exit(1)

    # 讀取原始邊資料
    with open(sim_path, 'r') as f:
        sim_edges = f.readlines()
    with open(cor_path, 'r') as f:
        cor_edges = f.readlines()

    # 儲存將被過濾掉的無效節點集
    invalid_nodes = set()

    # 進行多輪過濾，直到沒有節點再被移除 (確保雙圖一致性)
    filter_round = 0
    while True:
        filter_round += 1
        print(f"Round {filter_round}")

        node_sim_score = defaultdict(int)
        node_cor_score = defaultdict(int)
        valid_nodes = set()

        # 統計相似圖中的節點度數
        for line in sim_edges:
            u, v, _ = line.strip('\n').split('\t')
            if u in invalid_nodes or v in invalid_nodes:
                continue
            node_sim_score[u] += 1
            node_sim_score[v] += 1
            valid_nodes.add(u)
            valid_nodes.add(v)
  
        # 統計互補圖中的節點度數
        for line in cor_edges:
            u, v, _ = line.strip('\n').split('\t')
            if u in invalid_nodes or v in invalid_nodes:
                continue
            node_cor_score[u] += 1
            node_cor_score[v] += 1
            valid_nodes.add(u)
            valid_nodes.add(v)
  
        print(f"Current valid nodes: {len(valid_nodes)}")

        stop_sign = True
        # 檢查是否有節點未達到相似性閾值
        for u in list(node_sim_score.keys()):
            if node_sim_score[u] < threshold:
                invalid_nodes.add(u)
                stop_sign = False

        # 檢查是否有節點未達到互補性閾值
        for u in list(node_cor_score.keys()):
            if node_cor_score[u] < threshold:
                invalid_nodes.add(u)
                stop_sign = False
        
        # 若本輪沒有新節點被加入 invalid_nodes，則停止
        if stop_sign:
            break
        print(f"Nodes filtered so far: {len(invalid_nodes)}")

    # 寫入過濾後的結果
    filtered_sim_path = f"./data_preprocess/tmp/filtered_{data_name}_sim.edges"
    filtered_cor_path = f"./data_preprocess/tmp/filtered_{data_name}_cor.edges"

    with open(filtered_sim_path, 'w') as f_sim:
        for line in sim_edges:
            u, v, _ = line.strip('\n').split('\t')
            if u not in invalid_nodes and v not in invalid_nodes:
                f_sim.write(line)
  
    with open(filtered_cor_path, 'w') as f_cor:
        for line in cor_edges:
            u, v, _ = line.strip('\n').split('\t')
            if u not in invalid_nodes and v not in invalid_nodes:
                f_cor.write(line)

    print(f"Filtering finished! Final valid nodes: {len(valid_nodes)}")
    print(f"Filtered edges saved to tmp/filtered_...")

if __name__ == "__main__":
    main()