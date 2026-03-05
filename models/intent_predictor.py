# 四分支意圖預測器：{短期, 長期} × {相似, 互補}
# 使用 concat+project 可學習融合：將 4 個 user 向量與 2 個 item 向量分別拼接後投影成單一向量
# 允許跨通道交互，梯度集中在單一評分路徑，訓練信號更強

import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentPredictor(nn.Module):
    def __init__(self, emb_dim, dropout=0.0):
        super(IntentPredictor, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)

        # 使用者意圖融合投影：將 4 個通道向量 (SimS, SimL, CorS, CorL) 拼接後投影
        self.user_proj = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        # 商品特徵融合投影：將 sim + cor 兩個通道向量拼接後投影
        self.item_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def _get_user_repr(self, sim_intents, rel_intents):
        u_sim_last, u_sim_att = sim_intents
        u_rel_last, u_rel_att = rel_intents
        u_cat = torch.cat([
            self.dropout(u_sim_last),
            self.dropout(u_sim_att),
            self.dropout(u_rel_last),
            self.dropout(u_rel_att)
        ], dim=-1)  # [B, 4D]
        return self.user_proj(u_cat)  # [B, D]

    def forward(self, sim_intents, rel_intents, target_sim_embs, target_cor_embs, user_spec_sim, user_spec_cor):
        """
        sim_intents: (u_sim_last, u_sim_att)
        rel_intents: (u_rel_last, u_rel_att)
        target_sim_embs: [batch, neg_num + 1, emb_dim]
        target_cor_embs: [batch, neg_num + 1, emb_dim]
        """
        u_final = self._get_user_repr(sim_intents, rel_intents)  # [B, D]

        # 拼接 sim + cor 商品向量後投影成單一鍵向量
        i_cat = torch.cat([target_sim_embs, target_cor_embs], dim=-1)  # [B, 1+neg, 2D]
        i_final = self.item_proj(i_cat)  # [B, 1+neg, D]

        # 單一點積評分（梯度集中，訓練信號更強）
        # 除以 sqrt(D) 穩定 softmax：LayerNorm 後 dot product std ≈ sqrt(D)，不縮放會讓初始 loss >> ln(K)
        scale = self.emb_dim ** 0.5
        final_score = torch.bmm(i_final, u_final.unsqueeze(2)).squeeze(2) / scale  # [B, 1+neg]

        # 均勻權重供 logging 使用（不影響訓練）
        weights = torch.full((u_final.size(0), 4), 0.25, device=u_final.device)

        return final_score, weights, final_score, final_score

    def forward_full(self, sim_intents, rel_intents, all_sim_embs, all_cor_embs, user_spec_sim, user_spec_cor):
        """全矩陣加速運算（驗證/測試時用）"""
        u_final = self._get_user_repr(sim_intents, rel_intents)  # [B, D]

        # 拼接所有商品的 sim + cor 向量後投影
        all_cat = torch.cat([all_sim_embs, all_cor_embs], dim=-1)  # [N, 2D]
        all_i_final = self.item_proj(all_cat)  # [N, D]

        scale = self.emb_dim ** 0.5
        return torch.matmul(u_final, all_i_final.t()) / scale  # [B, N]
