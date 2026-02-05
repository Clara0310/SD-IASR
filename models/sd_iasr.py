import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spectral_layers import SpectralDisentangler
from models.sequential_encoder import SequentialEncoder
from models.intent_predictor import IntentPredictor

class SDIASR(nn.Module):
    def __init__(self, item_num, bert_dim, emb_dim, low_k, mid_k, max_seq_len):
        super(SDIASR, self).__init__()
        self.item_num = item_num
        self.emb_dim = emb_dim
        
        # 1. 商品初始嵌入層 (Item Embedding)
        # 修改：增加投影層，將 768 維 BERT 降維至 emb_dim (如 64)
        self.item_embedding = nn.Embedding(item_num, bert_dim)
        self.proj = nn.Linear(bert_dim, emb_dim)
        
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
        initial_embs = self.proj(self.item_embedding(all_item_indices)) # [num_items, emb_dim]

        # B. 執行譜解耦：生成相似性特徵 X_sim 與 互補性特徵 X_cor
        x_sim, x_cor = self.spectral_disentangler(initial_embs, sim_laplacian, com_laplacian)

        # C. 提取序列中各商品的解耦特徵
        seq_sim_embs = F.embedding(seq_indices, x_sim) # [batch, seq_len, emb_dim]
        seq_cor_embs = F.embedding(seq_indices, x_cor) # [batch, seq_len, emb_dim]
        
        # D. 修正：建立 Padding Mask (True 代表該位置是 0，需要屏蔽)
        # 這會與 SequentialEncoder 內的 nn.TransformerEncoder 完美對齊
        mask = (seq_indices == 0)

        # E. 雙通道序列編碼：捕捉近期與全局意圖
        sim_intents, cor_intents = self.sequential_encoder(seq_sim_embs, seq_cor_embs, mask)

        # F. 取得候選商品的特徵 (同樣經過譜解耦)
        target_sim_embs = F.embedding(target_indices, x_sim) # [batch, 1+neg, emb_dim]
        target_cor_embs = F.embedding(target_indices, x_cor)
        
        # 融合候選商品特徵作為評分基準
        #target_embs = (target_sim_embs + target_cor_embs) / 2

        # G. 意圖預測與自適應融合
        scores, alpha, sim_scores, rel_scores = self.predictor(sim_intents, cor_intents, target_sim_embs, target_cor_embs)

        return scores, alpha, sim_scores, rel_scores
    
    def load_pretrain_embedding(self, cid2_emb, cid3_emb, item_to_cid):
        # self.item_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # # 允許在訓練過程中微調嵌入權重
        # self.item_embedding.weight.requires_grad = True
        
        """
        將 BERT 分類嵌入映射到商品嵌入
        cid2_emb, cid3_emb: [num_categories, 768] 的 numpy 陣列
        item_to_cid: 字典或陣列，格式為 {item_num_id: (cid2_id, cid3_id)}
        """
        import numpy as np
        
        # 建立一個新的商品嵌入矩陣 [item_num, 768]
        # 注意：這會改變 self.item_embedding 的維度，run.py 的 embedding_dim 需設為 768
        #pretrained_weight = np.zeros((self.item_num, self.emb_dim))
        # 修正：這裡必須固定為 768，因為這是接收原始 BERT 向量的地方
        pretrained_weight = np.zeros((self.item_num, 768))
        
        for i in range(self.item_num):
            if i in item_to_cid:
                c2_id, c3_id = item_to_cid[i]
                # 融合第 2 級與第 3 級類別的 BERT 向量 (768維)
                item_vec = (cid2_emb[c2_id] + cid3_emb[c3_id]) / 2.0
                pretrained_weight[i] = item_vec
            else:
                # 若無類別資訊則保持隨機
                pretrained_weight[i] = np.random.normal(size=(768,))

        # 確保 item_embedding 的權重資料能對應 768 維度
        self.item_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight).float())
        self.item_embedding.weight.requires_grad = True
        print(f"Successfully mapped category BERT embeddings to {self.item_num} items.")
        
        # for i in range(self.item_num):
        #     if i in item_to_cid:
        #         c2_id, c3_id = item_to_cid[i]
        #         # 融合第 2 級與第 3 級類別的 BERT 向量作為商品初始表徵
        #         item_vec = (cid2_emb[c2_id] + cid3_emb[c3_id]) / 2.0
        #         pretrained_weight[i] = item_vec
        #     else:
        #         # 若無類別資訊則保持隨機（或全 0）
        #         pretrained_weight[i] = np.random.normal(size=(self.emb_dim,))

        # self.item_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight).float())
        # self.item_embedding.weight.requires_grad = True
        # print(f"Successfully mapped category BERT embeddings to {self.item_num} items.")