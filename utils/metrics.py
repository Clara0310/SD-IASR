import torch
import numpy as np
from sklearn.metrics import ndcg_score

def get_metrics(item_num, scores, k_list=[5, 10, 20]):
    """
    計算 1:100 負採樣下的評估指標
    
    Args:
        item_num: 總商品數量 (用於負採樣，但在此模式下主要參考 scores 結構)
        scores: 模型對正樣本與負樣本的預測得分，Shape 應為 [batch_size, 1 + neg_num]
        k_list: 要計算的 Top-K 列表
    """
    metrics = {}
    batch_size = scores.shape[0]
    
    # 在 1:100 模式中，通常 scores 的第 0 欄是正樣本
    # 我們計算正樣本在所有 100 個候選商品中的排名
    _, indices = torch.sort(scores, descending=True, dim=-1)
    
    # 找出正樣本 (索引為 0) 在排序後的排名位置
    # rank shape: [batch_size], 數值從 0 開始 (0 代表排在第 1 名)
    ranks = (indices == 0).nonzero(as_tuple=True)[1]
    
    # Top-k 評估指標計算
    for k in k_list:
        # Hit Ratio @ K: 正樣本排名在 K 之前的比例
        hit_at_k = (ranks < k).float().mean().item()
        metrics[f'HR@{k}'] = hit_at_k
        
        # NDCG @ K: 考慮排名先後的歸一化折損累計增益
        # 公式: 1 / log2(rank + 2)
        ndcg_at_k = (ranks < k).float() * (1 / torch.log2(ranks.float() + 2))
        metrics[f'NDCG@{k}'] = ndcg_at_k.mean().item()
        
    # ------新增與 SR-Rec 一致的全量 NDCG 計算------
    # 建立標籤矩陣：第 0 欄是正樣本 (1)，其餘是負樣本 (0)
    # scores 的形狀是 [batch_size, 101]
    y_true = np.zeros(scores.shape)
    y_true[:, 0] = 1
    
    # 取得模型預測的分數並轉為 numpy
    y_score = scores.detach().cpu().numpy()
    
    # 計算全量 NDCG (不傳入 k 參數，即為 SR-Rec 採用的方法)
    metrics['NDCG_Full'] = ndcg_score(y_true, y_score)
    #----------------------------------------------
    
    return metrics

def print_metrics(metrics):
    """格式化印出評估結果"""
    output = []
    # 依照鍵值排序確保輸出整齊
    for key in sorted(metrics.keys()):
        output.append(f"{key}: {metrics[key]:.4f}")
    print(" | ".join(output))

# 如果您未來需要進行全量排序，可以保留此函數作為備用
def get_all_item_metrics(scores, targets, k_list=[5, 10, 20]):
    """全量商品排序評估 (原版邏輯)"""
    metrics = {}
    _, topk_indices = torch.topk(scores, max(k_list), dim=-1)
    
    for k in k_list:
        hit = 0
        ndcg = 0
        for i in range(len(targets)):
            target = targets[i]
            topk = topk_indices[i][:k]
            if target in topk:
                hit += 1
                rank = (topk == target).nonzero(as_tuple=True)[0].item()
                ndcg += 1 / np.log2(rank + 2)
        
        metrics[f'HR@{k}'] = hit / len(targets)
        metrics[f'NDCG@{k}'] = ndcg / len(targets)
    return metrics