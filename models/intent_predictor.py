# 新增：差異化計分函數與動態權重 α

import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentPredictor(nn.Module):
    def __init__(self, emb_dim,dropout=0.0):
        super(IntentPredictor, self).__init__()
        self.emb_dim = emb_dim
        
        # 定義 Dropout 層
        self.dropout = nn.Dropout(dropout)

        # 動態權重分配網路 (用於計算 alpha)
        self.alpha_net = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )

        # 雙線性轉換矩陣 (用於計算不同空間的互動分數)
        self.w_sim = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        self.w_rel = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        
        # [新增] 定義融合層，將拼接後的 2*emb_dim 壓縮回 emb_dim
        self.fusion_sim = nn.Linear(emb_dim * 2, emb_dim)
        self.fusion_rel = nn.Linear(emb_dim * 2, emb_dim)
        
        nn.init.xavier_uniform_(self.w_sim)
        nn.init.xavier_uniform_(self.w_rel)

        # 初始化權重
        nn.init.xavier_uniform_(self.fusion_sim.weight)
        nn.init.xavier_uniform_(self.fusion_rel.weight)
        
    def forward(self, sim_intents, rel_intents, target_sim_embs, target_cor_embs):
        """
        sim_intents: (u_sim_last, u_sim_att)
        rel_intents: (u_rel_last, u_rel_att)
        target_sim_embs: 候選商品的相似性嵌入 [batch, neg_num + 1, emb_dim]
        target_cor_embs: 候選商品的互補性嵌入 [batch, neg_num + 1, emb_dim]
        """
        u_sim_last, u_sim_att = sim_intents
        u_rel_last, u_rel_att = rel_intents


        # --- [修改] 意圖融合：從「加法」改為「拼接 + 線性層」 ---
        # 1. 意圖融合：結合近期與全局資訊
        # 這裡採用加和或拼接後的線性轉換，確保與候選商品維度一致
        #u_sim = self.dropout(u_sim_last + u_sim_att)
        #u_rel = self.dropout(u_rel_last + u_rel_att)
        
        # 相似性意圖融合
        u_sim = self.fusion_sim(torch.cat([u_sim_last, u_sim_att], dim=-1))
        u_sim = self.dropout(F.relu(u_sim))
        
        # 互補性意圖融合
        u_rel = self.fusion_rel(torch.cat([u_rel_last, u_rel_att], dim=-1))
        u_rel = self.dropout(F.relu(u_rel))
        # ---------------------------------------------------
        
        
        

        # 2. 計算動態權重 alpha
        # 融合四個意圖特徵來判斷當前使用者受哪種關係影響較大
        combined_context = torch.cat([u_sim_last, u_sim_att, u_rel_last, u_rel_att], dim=-1)
        alpha = self.alpha_net(combined_context) # [batch, 1]

        # 3. 計算雙視角得分
        # target_embs shape: [batch, N, emb_dim]
        # 使用雙線性轉換計算使用者意圖與候選商品的匹配度
        
        # 相似性得分 (Similarity Score)
        # score = u_sim * W * target_item
        sim_score = torch.matmul(u_sim, self.w_sim) 
        sim_score = torch.bmm(target_sim_embs, sim_score.unsqueeze(2)).squeeze(2)

        # 互補性得分 (Complementarity Score)
        rel_score = torch.matmul(u_rel, self.w_rel)
        rel_score = torch.bmm(target_cor_embs, rel_score.unsqueeze(2)).squeeze(2)
        
        # 4. 最終預測值：由 alpha 進行自適應融合
        # Final Score = alpha * Sim_Score + (1 - alpha) * Rel_Score
        final_score = alpha * sim_score + (1 - alpha) * rel_score

        return final_score, alpha, sim_score, rel_score
    


    # [新增這個方法]
    # [請將此方法加入 models/intent_predictor.py 的 IntentPredictor 類別中]
    
    def forward_full(self, sim_intents, rel_intents, all_sim_embs, all_cor_embs):
        """
        全矩陣加速運算 (配合你的 AlphaNet 和 Bilinear Layer)
        """
        # 1. 解包 Intent (因為你的模型回傳的是 tuple)
        u_sim_last, u_sim_att = sim_intents
        u_rel_last, u_rel_att = rel_intents

        # 2. 計算 Alpha (使用你的 self.alpha_net)
        # 你的模型需要把這四個接起來才能算 alpha
        combined_context = torch.cat([u_sim_last, u_sim_att, u_rel_last, u_rel_att], dim=-1)
        alpha = self.alpha_net(combined_context) # [Batch, 1]

        # 3. 意圖融合 (對應原本 forward 的邏輯)
        u_sim = self.dropout(u_sim_last + u_sim_att)
        u_rel = self.dropout(u_rel_last + u_rel_att)

        # 4. 全矩陣加速運算 (Matrix Multiplication)
        # 你的模型有 w_sim 和 w_rel，所以要先乘上這個權重矩陣
        
        # Step A: 使用者向量變換 [Batch, Dim] @ [Dim, Dim] -> [Batch, Dim]
        u_sim_trans = torch.matmul(u_sim, self.w_sim) 
        u_rel_trans = torch.matmul(u_rel, self.w_rel) 
        
        # Step B: 與所有商品做內積 [Batch, Dim] @ [Dim, Num_Items] -> [Batch, Num_Items]
        # all_sim_embs.t() 會把 [Num, Dim] 轉成 [Dim, Num]
        sim_scores = torch.matmul(u_sim_trans, all_sim_embs.t())
        rel_scores = torch.matmul(u_rel_trans, all_cor_embs.t())
        
        # 5. 加權融合
        # alpha 自動廣播: [Batch, 1] * [Batch, Num_Items]
        scores = alpha * sim_scores + (1 - alpha) * rel_scores
        
        return scores