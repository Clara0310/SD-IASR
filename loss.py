import torch
import torch.nn as nn
import torch.nn.functional as F

class SDIASRLoss(nn.Module):
    """
    SD-IASR 專用損失函數模組
    包含 BPR 推薦損失與權重正則化。
    """
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_reg=0.01, lambda_proto=0.1, lambda_spec=0.5, lambda_cl=0.005, lambda_alpha=0.5, tau=0.3):
        super(SDIASRLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_reg = lambda_reg
        self.lambda_proto = lambda_proto # 原型損失權重
        self.lambda_spec = lambda_spec # 譜圖層解耦權重
        self.lambda_cl = lambda_cl
        self.lambda_alpha = lambda_alpha  # Alpha 熵正則化權重，防止雙通道崩塌成單通道
        self.tau = tau

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

    def forward(self, scores, sim_scores, rel_scores, weights, u_sim, u_cor, p_sim_s, p_cor_s, x_sim, x_cor, model):
        # 1. BPR 推薦損失
        l_seq = -torch.mean(torch.log(torch.sigmoid(scores[:, 0].unsqueeze(1) - scores[:, 1:]) + 1e-10))
        l_sim = -torch.mean(torch.log(torch.sigmoid(sim_scores[:, 0].unsqueeze(1) - sim_scores[:, 1:]) + 1e-10))
        l_rel = -torch.mean(torch.log(torch.sigmoid(rel_scores[:, 0].unsqueeze(1) - rel_scores[:, 1:]) + 1e-10))

        # 2. 原型聚類損失
        l_proto = self.calculate_proto_loss(p_sim_s) + self.calculate_proto_loss(p_cor_s)

        # 3. 通道正交解耦損失（作用在最終表示 x_sim/x_cor 上，而非 raw）
        # 直接懲罰兩個通道的最終輸出相似度，強制 proj_sim 和 proj_cor 學出不同的語義空間
        cos_sim_spec = F.cosine_similarity(x_sim, x_cor, dim=-1)
        l_spec = torch.mean(cos_sim_spec**2)

        # 4. 四維權重熵正則化：防止權重崩塌至 one-hot（四通道退化成單通道）
        # H(weights) = -Σ w_i * log(w_i)
        # 在均勻分佈 [0.25, 0.25, 0.25, 0.25] 時最大 (log4≈1.386)
        # weights shape: [batch, 4]
        H_weights = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)  # [batch]
        l_alpha = -H_weights.mean()  # 負熵，最小化此項 = 最大化熵

        # 5. 正則化：只對 weight 矩陣 (非 bias、非 LayerNorm、非 Embedding)
        reg_loss = sum(
            torch.norm(param, p=2)
            for name, param in model.named_parameters()
            if param.requires_grad and param.dim() >= 2
            and 'embedding' not in name and 'norm' not in name
        )

        total_loss = (l_seq + self.lambda_1 * l_sim + self.lambda_2 * l_rel) + \
                     self.lambda_reg * reg_loss + \
                     self.lambda_proto * l_proto + \
                     self.lambda_spec * l_spec + \
                     self.lambda_alpha * l_alpha

        return total_loss, l_seq, l_proto, l_spec, l_alpha