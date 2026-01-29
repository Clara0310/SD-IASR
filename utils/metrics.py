# 計算 HR@k, NDCG

import numpy as np
import torch

def get_metrics(real_item_idx, scores, k_list=[10, 20]):
    """
    計算 HR@K 與 NDCG@K 指標
    
    參數:
    - real_item_idx: 正確商品在 scores 中的索引。
      在我們的 7_dataset_split.py 設計中，正樣本總是放在第 0 位。
    - scores: 模型輸出的預測分數，形狀為 [batch_size, 1 + neg_num]。
    - k_list: 要評估的 Top-K 門檻，例如 [10, 20]。
    
    返回:
    - metrics: 包含所有 K 值的 HR 與 NDCG 的字典。
    """
    metrics = {}
    
    # 1. 取得分數最高的前 max(k) 個索引
    # topk_indices shape: [batch_size, max_k]
    _, topk_indices = torch.topk(scores, max(k_list), dim=-1)
    
    # 2. 將 GPU 張量轉為 CPU/Numpy 方便計算
    topk_indices = topk_indices.cpu().numpy()
    
    # 根據 7_dataset_split.py，正樣本在輸入 scores 時位於 index 0
    target_label = 0 
    batch_size = scores.size(0)

    for k in k_list:
        hr_count = 0
        ndcg_sum = 0.0
        
        for i in range(batch_size):
            # 取得單個使用者的前 K 個預測結果
            target_top_k = topk_indices[i, :k]
            
            # 檢查正樣本是否在 Top-K 中 (HR)
            if target_label in target_top_k:
                hr_count += 1
                
                # 計算 NDCG (考慮排名權重)
                # rank 為正樣本在列表中的位置 (0-based)
                rank = np.where(target_top_k == target_label)[0][0]
                # 公式: 1 / log2(rank + 2)
                ndcg_sum += 1.0 / np.log2(rank + 2)
        
        # 計算批次平均值
        metrics[f'HR@{k}'] = hr_count / batch_size
        metrics[f'NDCG@{k}'] = ndcg_sum / batch_size
        
    return metrics

def print_metrics(metrics):
    """格式化輸出指標結果"""
    output_str = " | ".join([f"{key}: {val:.4f}" for key, val in metrics.items()])
    print(f"Results: {output_str}")