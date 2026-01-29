# 新增：差異化計分函數與動態權重 α

import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentPredictor(nn.Module):
    def __init__(self, emb_dim):
        super(IntentPredictor, self).__init__()
        self.emb_dim = emb_dim

        # 動態權重分配網路 (用於計算 alpha)
        self.alpha_net = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )

        # 雙線性轉換矩陣 (用於計算不同空間的互動分數)
        self.w_sim = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        self.w_rel = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        
        nn.init.xavier_uniform_(self.w_sim)
        nn.init.xavier_uniform_(self.w_rel)

    def forward(self, sim_intents, rel_intents, target_embs):
        """
        sim_intents: (u_sim_last, u_sim_att)
        rel_intents: (u_rel_last, u_rel_att)
        target_embs: 候選商品的嵌入 [batch, neg_num + 1, emb_dim]
        """
        u_sim_last, u_sim_att = sim_intents
        u_rel_last, u_rel_att = rel_intents

        # 1. 意圖融合：結合近期與全局資訊
        # 這裡採用加和或拼接後的線性轉換，確保與候選商品維度一致
        u_sim = u_sim_last + u_sim_att
        u_rel = u_rel_last + u_rel_att

        # 2. 計算動態權重 alpha
        # 融合四個意圖特徵來判斷當前使用者受哪種關係影響較大
        combined_context = torch.cat([u_sim_last, u_sim_att, u_rel_last, u_rel_att], dim=-1)
        alpha = self.alpha_net(combined_context) # [batch, 1]

        # 3. 計算雙視角得分
        # target_embs shape: [batch, N, emb_dim]
        # 使用雙線性轉換計算使用者意圖與候選商品的匹配度
        
        # 相似性得分 (Similarity Score)
        # score = u_sim * W * target_item
        sim_score = torch.matmul(u_sim, self.w_sim) # [batch, emb_dim]
        sim_score = torch.bmm(target_embs, sim_score.unsqueeze(2)).squeeze(2) # [batch, N]

        # 互補性得分 (Complementarity Score)
        rel_score = torch.matmul(u_rel, self.w_rel)
        rel_score = torch.bmm(target_embs, rel_score.unsqueeze(2)).squeeze(2) # [batch, N]

        # 4. 最終預測值：由 alpha 進行自適應融合
        # Final Score = alpha * Sim_Score + (1 - alpha) * Rel_Score
        final_score = alpha * sim_score + (1 - alpha) * rel_score

        return final_score, alpha