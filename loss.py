import torch
import torch.nn as nn
import torch.nn.functional as F

class SDIASRLoss(nn.Module):
    """
    SD-IASR 專用損失函數模組
    包含 BPR 推薦損失與權重正則化。
    """
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_reg=0.01):
        super(SDIASRLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_reg = lambda_reg

    def bpr_loss(self, scores):
        """
        貝葉斯個性化排序損失 (BPR Loss)
        目標：最大化正樣本分數與負樣本分數之間的差距。
        scores shape: [batch_size, 1 + neg_num] (index 0 為正樣本)
        """
        # 提取正樣本分數並擴展維度以便廣播
        pos_scores = scores[:, 0].unsqueeze(1)  # [batch_size, 1]
        # 提取其餘負樣本分數
        neg_scores = scores[:, 1:]              # [batch_size, neg_num]
        
        # 公式: -E[log(sigmoid(pos - neg))]
        # 加入 1e-10 防止數值不穩定 (log 0)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        return loss

    def forward(self, scores, sim_scores, rel_scores, model):
        # 主損失
        l_seq = self.bpr_loss(scores)
        
        # 輔助損失：確保相似通道與互補通道各自有排序能力
        l_sim = self.bpr_loss(sim_scores)
        l_rel = self.bpr_loss(rel_scores)
        
        # 正則化
        reg_loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name: reg_loss += torch.norm(param, p=2)
        
        # 最終聯合損失
        total_loss = l_seq + self.lambda_1 * l_sim + self.lambda_2 * l_rel + self.lambda_reg * reg_loss
        return total_loss, l_seq, l_sim, l_rel