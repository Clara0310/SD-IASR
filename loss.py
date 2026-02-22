import torch
import torch.nn as nn
import torch.nn.functional as F

class SDIASRLoss(nn.Module):
    """
    SD-IASR 專用損失函數模組
    包含 BPR 推薦損失與權重正則化。
    """
    def __init__(self, lambda_1=2.0, lambda_2=2.0, lambda_reg=0.01, lambda_cl=0.05, lambda_proto=0.1, tau=0.1):
        super(SDIASRLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_reg = lambda_reg
        self.lambda_cl = lambda_cl # 對比學習權重
        self.lambda_proto = lambda_proto # [新增] 原型損失權重
        self.tau = tau             # 溫度參數

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
    
    # [新增] 跨視角意圖對比損失 (InfoNCE)
    def calculate_cl_loss(self, view1, view2):
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        
        # 同一個使用者的相似與互補意圖應互為正樣本
        pos_score = torch.sum(view1 * view2, dim=-1) / self.tau
        all_score = torch.matmul(view1, view2.t()) / self.tau
        
        exp_all_score = torch.exp(all_score).sum(dim=-1)
        cl_loss = -torch.log(torch.exp(pos_score) / exp_all_score)
        return cl_loss.mean()

    # [核心新增] 意圖-原型對齊損失 (BARec 風格)
    def calculate_proto_loss(self, proto_scores):
        """
        將 User Intent 分配到最接近的原型中心。
        這是一個聚類任務，強迫相似意圖的人聚集在一起。
        """
        # 使用 Cross-Entropy 讓意圖更明確地歸屬於某個中心
        # 這裡我們不使用標籤，而是最大化分配的熵 (Entropy Minimization)
        probs = F.softmax(proto_scores / self.tau, dim=-1)
        log_probs = F.log_softmax(proto_scores / self.tau, dim=-1)
        # 最小化信息熵，使分配更「尖銳」
        loss = -torch.mean(torch.sum(probs * log_probs, dim=-1))
        return loss

    def forward(self, scores, sim_scores, rel_scores,u_sim, u_cor, p_sim_s, p_cor_s, model):
        # 1. 序列推薦 Loss (BPR)
        l_seq = -torch.mean(torch.log(torch.sigmoid(scores[:, 0].unsqueeze(1) - scores[:, 1:]) + 1e-10))
        l_sim = -torch.mean(torch.log(torch.sigmoid(sim_scores[:, 0].unsqueeze(1) - sim_scores[:, 1:]) + 1e-10))
        l_rel = -torch.mean(torch.log(torch.sigmoid(rel_scores[:, 0].unsqueeze(1) - rel_scores[:, 1:]) + 1e-10))
        
        # 2. 正則化
        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.norm(param, p=2)
            
        # 3. 意圖層級對比學習
        l_cl = self.calculate_cl_loss(u_sim, u_cor)
        
        # 4. 原型聚類損失
        l_proto = self.calculate_proto_loss(p_sim_s) + self.calculate_proto_loss(p_cor_s)
        
        # 最終組合：加入 lambda_proto * l_proto
        total_loss = (l_seq + self.lambda_1 * l_sim + self.lambda_2 * l_rel) + \
                     self.lambda_reg * reg_loss + \
                     self.lambda_cl * l_cl + \
                     self.lambda_proto * l_proto
                     
        return total_loss, l_seq, l_sim, l_rel, l_cl, l_proto