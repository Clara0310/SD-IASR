# 四分支意圖預測器：{短期, 長期} × {相似, 互補}

import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentPredictor(nn.Module):
    def __init__(self, emb_dim, dropout=0.0):
        super(IntentPredictor, self).__init__()
        self.emb_dim = emb_dim

        self.dropout = nn.Dropout(dropout)

        # 四維意圖權重網路：輸出 4 個分支的權重
        # 輸入 = [u_sim_last, u_sim_att, u_cor_last, u_cor_att, user_spec_sim, user_spec_cor]
        # 輸出 = [w_sim_short, w_sim_long, w_cor_short, w_cor_long]
        self.intent_net = nn.Sequential(
            nn.Linear(emb_dim * 6, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 4),
            # 不加 Softmax，在 forward 中用 F.softmax 以便控制 temperature
        )

        # 雙線性轉換矩陣（sim 和 cor 各一組，short/long 共用同一組 W）
        self.w_sim = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        self.w_rel = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))

        nn.init.xavier_uniform_(self.w_sim)
        nn.init.xavier_uniform_(self.w_rel)

    def forward(self, sim_intents, rel_intents, target_sim_embs, target_cor_embs, user_spec_sim, user_spec_cor):
        """
        sim_intents: (u_sim_last, u_sim_att)
        rel_intents: (u_rel_last, u_rel_att)
        target_sim_embs: 候選商品的相似性嵌入 [batch, neg_num + 1, emb_dim]
        target_cor_embs: 候選商品的互補性嵌入 [batch, neg_num + 1, emb_dim]
        user_spec_sim: 用戶的相似性譜特徵簽名 [batch, emb_dim]
        user_spec_cor: 用戶的互補性譜特徵簽名 [batch, emb_dim]
        """
        u_sim_last, u_sim_att = sim_intents
        u_rel_last, u_rel_att = rel_intents

        # 1. 計算四維意圖權重
        combined_context = torch.cat([u_sim_last, u_sim_att, u_rel_last, u_rel_att,
                                      user_spec_sim, user_spec_cor], dim=-1)
        weights = F.softmax(self.intent_net(combined_context), dim=-1)  # [batch, 4]

        # 2. 四分支評分（不再融合 last+att，保留時間維度）
        # ① 短期相似：u_sim_last · W_sim · target_sim
        s_sim_short = torch.matmul(self.dropout(u_sim_last), self.w_sim)
        s_sim_short = torch.bmm(target_sim_embs, s_sim_short.unsqueeze(2)).squeeze(2)

        # ② 長期相似：u_sim_att · W_sim · target_sim
        s_sim_long = torch.matmul(self.dropout(u_sim_att), self.w_sim)
        s_sim_long = torch.bmm(target_sim_embs, s_sim_long.unsqueeze(2)).squeeze(2)

        # ③ 短期互補：u_cor_last · W_rel · target_cor
        s_cor_short = torch.matmul(self.dropout(u_rel_last), self.w_rel)
        s_cor_short = torch.bmm(target_cor_embs, s_cor_short.unsqueeze(2)).squeeze(2)

        # ④ 長期互補：u_cor_att · W_rel · target_cor
        s_cor_long = torch.matmul(self.dropout(u_rel_att), self.w_rel)
        s_cor_long = torch.bmm(target_cor_embs, s_cor_long.unsqueeze(2)).squeeze(2)

        # 3. 加權融合：score = Σ w_i * score_i
        # weights[:, i:i+1] → [batch, 1] 自動廣播到 [batch, neg_num+1]
        final_score = (weights[:, 0:1] * s_sim_short +
                       weights[:, 1:2] * s_sim_long +
                       weights[:, 2:3] * s_cor_short +
                       weights[:, 3:4] * s_cor_long)

        # 4. 計算 sim 和 cor 的合併分數（供 BPR 輔助損失使用）
        sim_score = s_sim_short + s_sim_long
        rel_score = s_cor_short + s_cor_long

        return final_score, weights, sim_score, rel_score

    def forward_full(self, sim_intents, rel_intents, all_sim_embs, all_cor_embs, user_spec_sim, user_spec_cor):
        """
        全矩陣加速運算（驗證/測試時用）
        """
        u_sim_last, u_sim_att = sim_intents
        u_rel_last, u_rel_att = rel_intents

        # 1. 計算四維意圖權重
        combined_context = torch.cat([u_sim_last, u_sim_att, u_rel_last, u_rel_att,
                                      user_spec_sim, user_spec_cor], dim=-1)
        weights = F.softmax(self.intent_net(combined_context), dim=-1)  # [Batch, 4]

        # 2. 四分支全矩陣評分
        # [Batch, Dim] @ [Dim, Dim] -> [Batch, Dim]
        # [Batch, Dim] @ [Dim, Num_Items] -> [Batch, Num_Items]

        s_sim_short = torch.matmul(torch.matmul(u_sim_last, self.w_sim), all_sim_embs.t())
        s_sim_long  = torch.matmul(torch.matmul(u_sim_att,  self.w_sim), all_sim_embs.t())
        s_cor_short = torch.matmul(torch.matmul(u_rel_last, self.w_rel), all_cor_embs.t())
        s_cor_long  = torch.matmul(torch.matmul(u_rel_att,  self.w_rel), all_cor_embs.t())

        # 3. 加權融合
        scores = (weights[:, 0:1] * s_sim_short +
                  weights[:, 1:2] * s_sim_long +
                  weights[:, 2:3] * s_cor_short +
                  weights[:, 3:4] * s_cor_long)

        return scores
