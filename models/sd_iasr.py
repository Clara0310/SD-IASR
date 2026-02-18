import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spectral_layers import SpectralDisentangler
from models.sequential_encoder import SequentialEncoder
from models.intent_predictor import IntentPredictor

class SDIASR(nn.Module):
    def __init__(self, item_num, bert_dim, emb_dim, low_k, mid_k, max_seq_len, num_layers, nhead,dropout=0.1,gamma=0.1):
        super(SDIASR, self).__init__()
        self.item_num = item_num
        self.emb_dim = emb_dim
        self.gamma = gamma
        
        
        # === 1. 修改：多特徵融合初始嵌入層 (Multi-View Feature Fusion) ===
        # A. 定義價格 Embedding
        self.price_emb_dim = 64  # 設定價格向量維度
        self.price_embedding = nn.Embedding(20, self.price_emb_dim) # 20 bins
        
        # B. 定義融合後的輸入總維度
        # 輸入 = BERT(CID2) + BERT(CID3) + Price
        # 例如: 768 + 768 + 64 = 1600
        self.fusion_input_dim = bert_dim + bert_dim + self.price_emb_dim
        
        # C. 特徵融合投影層 (MLP)
        # 負責將拼接後的巨大向量 (1600維) 壓縮回 emb_dim (如 128維)
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.fusion_input_dim, emb_dim),
            nn.LayerNorm(emb_dim), # [新增] 標準化高維特徵輸入
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)  # [新增] 確保輸出給 item_embedding 的特徵穩定
        )
        
        # D. 最終的商品嵌入層
        # 注意：這裡存放的是「融合並降維後」的結果，所以維度直接是 emb_dim
        self.item_embedding = nn.Embedding(item_num, emb_dim)
        
        # (原本的 self.proj 已被 feature_fusion 取代，故移除)
        # 1. 商品初始嵌入層 (Item Embedding)
        # 修改：增加投影層，將 768 維 BERT 降維至 emb_dim (如 64)
        #self.item_embedding = nn.Embedding(item_num, bert_dim)
        #self.proj = nn.Linear(bert_dim, emb_dim)
        
        # === 2. 譜關係解耦模組 === (Spectral Disentangling Module)
        self.spectral_disentangler = SpectralDisentangler(
                    item_num, 
                    emb_dim, 
                    low_k, 
                    mid_k, 
                    dropout=dropout,  # <--- 加上這行
                    gamma=gamma
                )        
        
        # === 3. 序列編碼與意圖捕捉模組 === (Dual-Channel Transformer)
        self.sequential_encoder = SequentialEncoder(
            emb_dim, 
            max_seq_len, 
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout
        )
        
        # --- [新增這一行] ---
        self.layer_norm = nn.LayerNorm(emb_dim)  # 用於穩定譜解耦後的特徵
        
        # === 4. 使用者意圖預測模組 === (Intent-Aware Prediction)
        self.predictor = IntentPredictor(emb_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout) # <--- [新增] 定義一個全域 dropout 層
        
        

    def forward(self, seq_indices, time_indices, target_indices, adj_self, adj_dele):
        """
        seq_indices: 使用者歷史行為序列 [batch, seq_len]
        target_indices: 正樣本與負樣本商品 ID [batch, 1 + neg_num]
        adj_self & adj_dele: 相似性與互補性圖鄰接矩陣
        """
        # A. 取得所有商品的基礎嵌入
        all_item_indices = torch.arange(self.item_num).to(seq_indices.device)
        #initial_embs = self.proj(self.item_embedding(all_item_indices)) # [num_items, emb_dim]
        # 修改：因為 item_embedding 已經在 load_pretrain_embedding 時被初始化為 emb_dim
        # 所以這裡不需要再做 self.proj，直接取用即可
        initial_embs = self.item_embedding(all_item_indices) # [num_items, emb_dim]
        
        # [新增] 在初始 Embedding 後加 Dropout
        initial_embs = self.dropout(initial_embs)

        
        # B. 執行譜解耦：生成相似性特徵 X_sim 與 互補性特徵 X_cor
        #x_sim, x_cor = self.spectral_disentangler(initial_embs, sim_laplacian, com_laplacian)
        raw_sim, raw_cor = self.spectral_disentangler(initial_embs, adj_self, adj_dele)
        
        # [關鍵] 在這裡加回殘差，讓模型保留 BERT 語義
        # 這樣譜特徵就不會被 identity 淹沒到坍縮，但 BERT 特徵又能被保護
        x_sim = initial_embs + self.gamma * raw_sim
        x_cor = initial_embs + self.gamma * raw_cor
        
        # 執行原本的 LayerNorm
        x_sim = self.layer_norm(x_sim)
        x_cor = self.layer_norm(x_cor)
        
        # === [新增] 計算兩個空間的特徵相似度 (診斷點 2) ===
        # 我們想知道譜解耦後，兩個矩陣是否分得很開
        with torch.no_grad():
            # 計算所有商品在兩個空間的餘弦相似度均值
            feat_sim = F.cosine_similarity(x_sim, x_cor, dim=-1).mean()
        # ===============================================
    
        # C. 提取序列中各商品的解耦特徵
        seq_sim_embs = F.embedding(seq_indices, x_sim) # [batch, seq_len, emb_dim]
        seq_cor_embs = F.embedding(seq_indices, x_cor) # [batch, seq_len, emb_dim]
        
        # D. 修正：建立 Padding Mask (True 代表該位置是 0，需要屏蔽)
        # 這會與 SequentialEncoder 內的 nn.TransformerEncoder 完美對齊
        mask = (seq_indices == 0)

        # E. 雙通道序列編碼：捕捉近期與全局意圖
        sim_intents, cor_intents = self.sequential_encoder(seq_sim_embs, seq_cor_embs, time_indices, mask)

        # F. 取得候選商品的特徵 (同樣經過譜解耦)
        target_sim_embs = F.embedding(target_indices, x_sim) # [batch, 1+neg, emb_dim]
        target_cor_embs = F.embedding(target_indices, x_cor)
        
        # 融合候選商品特徵作為評分基準
        #target_embs = (target_sim_embs + target_cor_embs) / 2

        # G. 意圖預測與自適應融合
        scores, alpha, sim_scores, rel_scores = self.predictor(sim_intents, cor_intents, target_sim_embs, target_cor_embs)

        # [修改回傳值] 加入兩個全局意圖向量 (u_sim_att, u_cor_att) 以供診斷與 Loss 使用
        u_sim_att = sim_intents[1]
        u_cor_att = cor_intents[1]
        
        return scores, alpha, sim_scores, rel_scores, feat_sim, u_sim_att, u_cor_att,x_sim, x_cor
    
    def load_pretrain_embedding(self, cid2_emb, cid3_emb, item_to_cid, item_to_price):
        
        # self.item_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # # 允許在訓練過程中微調嵌入權重
        # self.item_embedding.weight.requires_grad = True
        
        """
        將 BERT 分類嵌入與價格特徵融合，並初始化商品嵌入
        Args:
            cid2_emb, cid3_emb: [num_categories, 768] 的 numpy 陣列
            item_to_cid: {item_num_id: (cid2_id, cid3_id)}
            item_to_price: {item_num_id: price_bin_id} (0~19)
        """
        import numpy as np
        
        # 建立一個新的商品嵌入矩陣 [item_num, 768]
        # 注意：這會改變 self.item_embedding 的維度，run.py 的 embedding_dim 需設為 768
        #pretrained_weight = np.zeros((self.item_num, self.emb_dim))
        # 修正：這裡必須固定為 768，因為這是接收原始 BERT 向量的地方
        print("Initializing item embeddings with Multi-View Features (CID2 + CID3 + Price)...")
        
        fusion_weights = []
        
        # [新增] 取得模型當前所在的裝置 (CPU 或 CUDA)
        device = self.price_embedding.weight.device
        
        # 使用 torch.no_grad 避免在初始化階段計算梯度
        with torch.no_grad():
            for i in range(self.item_num):
                # 1. 取得 BERT 類別特徵 (768維)
                if i in item_to_cid:
                    c2_id, c3_id = item_to_cid[i]
                    vec_c2 = torch.from_numpy(cid2_emb[c2_id]).float().to(device) # [修正] 移至 device
                    vec_c3 = torch.from_numpy(cid3_emb[c3_id]).float().to(device) # [修正] 移至 device
                else:
                    vec_c2 = torch.randn(768).to(device) # [修正] 移至 device
                    vec_c3 = torch.randn(768).to(device) # [修正] 移至 device
                
                # 2. 取得價格特徵 (64維)
                if i in item_to_price:
                    p_id = item_to_price[i]
                    # [修正] 建立 Tensor 後立刻移至 device
                    p_tensor = torch.tensor(p_id).to(device)
                    vec_price = self.price_embedding(p_tensor)
                else:
                    # 若無價格資訊，使用第 0 號 bin 或隨機
                    p_tensor = torch.tensor(0).to(device)
                    vec_price = self.price_embedding(p_tensor)
                
                # 3. 特徵拼接 (Concatenate)
                # 形狀: [768] + [768] + [64] -> [1600]
                concat_vec = torch.cat([vec_c2, vec_c3, vec_price], dim=0)
                fusion_weights.append(concat_vec.unsqueeze(0))

            # 將列表轉為 Tensor: [item_num, 1600]
            all_features = torch.cat(fusion_weights, dim=0)
            
            # 4. 通過融合層降維: [item_num, 1600] -> [item_num, emb_dim]
            # 這一步相當於 SR-Rec 中的 projection
            final_embs = self.feature_fusion(all_features)
            
            # 5. 將結果複製到 item_embedding
            self.item_embedding.weight.data.copy_(final_embs)
            
            # 確保之後訓練時可以更新
            self.item_embedding.weight.requires_grad = True
            
        print(f"Successfully fused features and mapped to {self.item_num} items with dim {self.emb_dim}.")
        
        

    # [新增這個方法]

    def predict_full(self, seq_indices, time_indices, adj_self, adj_dele):
        # 1. 取得 "所有" 商品的 Embedding (0 ~ item_num-1)
        all_item_indices = torch.arange(self.item_num).to(seq_indices.device)
        initial_embs = self.item_embedding(all_item_indices)
        initial_embs = self.dropout(initial_embs)
        
        
        # 2. 執行譜解耦 (算出所有商品的 Sim/Cor 特徵)
        # [Num_Items, Dim]
        #x_sim, x_cor = self.spectral_disentangler(initial_embs, sim_laplacian, com_laplacian)
        
        #  變數名要對應新傳入的參數 (adj_self, adj_dele)
        raw_sim, raw_cor = self.spectral_disentangler(initial_embs, adj_self, adj_dele)

        # 2. [關鍵遺漏] 這裡也要加回 BERT 殘差！否則測試時會完全失去 BERT 資訊
        x_sim = initial_embs + self.gamma * raw_sim
        x_cor = initial_embs + self.gamma * raw_cor
    
        # --- [新增以下兩行] ---
        x_sim = self.layer_norm(x_sim)
        x_cor = self.layer_norm(x_cor)
        
        # 3. 取得序列特徵 & User Intent
        # 這裡會從 x_sim 查表拿出序列裡的商品特徵
        seq_sim_embs = F.embedding(seq_indices, x_sim)
        seq_cor_embs = F.embedding(seq_indices, x_cor)
        
        mask = (seq_indices == 0)
        
        # 取得意圖 Tuple: (last, att)
        sim_intents, cor_intents = self.sequential_encoder(seq_sim_embs, seq_cor_embs, time_indices, mask)
        
        # 4. 呼叫加速版的 Predictor
        # [修正] 直接傳入 tuple，讓 IntentPredictor 自己去解包
        scores = self.predictor.forward_full(sim_intents, cor_intents, x_sim, x_cor)
        
        return scores # 回傳 [Batch, Num_Items]
    
    
    # 新增：一次性取得所有商品特徵
    def get_all_item_features(self, adj_self, adj_dele):
        all_item_indices = torch.arange(self.item_num).to(adj_self.device)
        initial_embs = self.item_embedding(all_item_indices)
        initial_embs = self.dropout(initial_embs)
        raw_sim, raw_cor = self.spectral_disentangler(initial_embs, adj_self, adj_dele)
        x_sim = self.layer_norm(initial_embs + self.gamma * raw_sim)
        x_cor = self.layer_norm(initial_embs + self.gamma * raw_cor)
        return x_sim, x_cor

    # 新增：接收算好的特徵進行預測
    def predict_full_fast(self, seq_indices, time_indices, x_sim, x_cor):
        seq_sim_embs = F.embedding(seq_indices, x_sim)
        seq_cor_embs = F.embedding(seq_indices, x_cor)
        mask = (seq_indices == 0)
        sim_intents, cor_intents = self.sequential_encoder(seq_sim_embs, seq_cor_embs, time_indices, mask)
        return self.predictor.forward_full(sim_intents, cor_intents, x_sim, x_cor)