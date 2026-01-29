# 整體模型封裝 (SD-IASR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spectral_layers import SpectralDisentangler
from models.sequential_encoder import SequentialEncoder
from models.intent_predictor import IntentPredictor

class SDIASR(nn.Module):
    def __init__(self, item_num, emb_dim, low_k, mid_k, max_seq_len):
        super(SDIASR, self).__init__()
        self.item_num = item_num
        self.emb_dim = emb_dim
        
        # 1. 商品初始嵌入層 (可載入預訓練的 BERT 嵌入)
        self.item_embedding = nn.Embedding(item_num, emb_dim)
        
        # 2. 譜關係解耦模組 (Spectral Disentangling Module)
        self.spectral_disentangler = SpectralDisentangler(item_num, emb_dim, low_k, mid_k)
        
        # 3. 序列編碼與意圖捕捉模組 (Dual-Channel Transformer)
        self.sequential_encoder = SequentialEncoder(emb_dim, max_seq_len)
        
        # 4. 使用者意圖預測模組 (Intent-Aware Prediction)
        self.predictor = IntentPredictor(emb_dim)

    def forward(self, seq_indices, target_indices, sim_laplacian, com_laplacian):
        """
        seq_indices: 使用者歷史行為序列 [batch, seq_len]
        target_indices: 正樣本與負樣本商品 ID [batch, 1 + neg_num]
        sim_laplacian & com_laplacian: 相似性與互補性圖拉普拉斯矩陣
        """
        # A. 取得所有商品的基礎嵌入
        all_item_indices = torch.arange(self.item_num).to(seq_indices.device)
        initial_embs = self.item_embedding(all_item_indices)

        # B. 執行譜解耦：生成相似性特徵 X_sim 與 互補性特徵 X_cor
        # 這對應論文 3.3 節的解耦過程
        x_sim, x_cor = self.spectral_disentangler(initial_embs, sim_laplacian, com_laplacian)

        # C. 提取序列中各商品的解耦特徵
        # seq_indices 形狀為 [batch, seq_len]
        seq_sim_embs = F.embedding(seq_indices, x_sim) # [batch, seq_len, emb_dim]
        seq_cor_embs = F.embedding(seq_indices, x_cor) # [batch, seq_len, emb_dim]
        
        # 建立 Padding Mask (若序列值為 0 則屏蔽)
        mask = (seq_indices == 0)

        # D. 雙通道序列編碼：捕捉近期與全局意圖
        # 這對應論文 3.4 節
        sim_intents, cor_intents = self.sequential_encoder(seq_sim_embs, seq_cor_embs, mask)

        # E. 取得候選商品的特徵 (同樣經過譜解耦)
        # 為了計算分數，目標商品的特徵需要同時考慮兩個空間
        target_sim_embs = F.embedding(target_indices, x_sim) # [batch, 1+neg, emb_dim]
        target_cor_embs = F.embedding(target_indices, x_cor)
        # 融合候選商品特徵作為評分基準
        target_embs = (target_sim_embs + target_cor_embs) / 2

        # F. 意圖預測與自適應融合
        # 產出最終推薦得分與動態權重 alpha (對應論文 3.5 節)
        scores, alpha = self.predictor(sim_intents, cor_intents, target_embs)

        return scores, alpha

    def load_pretrain_embedding(self, embedding_matrix):
        """將 BERT 產出的語義嵌入載入模型"""
        self.item_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # 凍結或微調可依需求決定，建議先微調
        self.item_embedding.weight.requires_grad = True