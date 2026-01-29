# 新增：雙通道 Transformer 與 Fusion Pooling

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        # 實作位置編碼以捕捉序列時序資訊
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1), :]

class MultiChannelTransformer(nn.Module):
    def __init__(self, emb_dim, nhead=2, num_layers=1, dropout=0.1):
        super(MultiChannelTransformer, self).__init__()
        # 使用 PyTorch 內建的 Transformer Encoder 層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=nhead, 
            dim_feedforward=emb_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(emb_dim)

    def forward(self, seq_embs, mask=None):
        # 加上位置編碼後送入 Transformer
        x = self.pos_encoder(seq_embs)
        output = self.transformer(x, src_key_padding_mask=mask)
        return output

class IntentCapture(nn.Module):
    def __init__(self, emb_dim):
        super(IntentCapture, self).__init__()
        # 用於計算注意力分數的權重矩陣
        self.attention_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1, bias=False)
        )

    def forward(self, seq_output, mask=None):
        """
        實作 Fusion Pooling 策略
        """
        # 1. 取序列最後一個位置作為近期意圖 (Recent Intent: U_last)
        # 假設 seq_output 是 [batch, seq_len, emb_dim]
        u_last = seq_output[:, -1, :]

        # 2. 計算注意力池化作為全局意圖 (Global Intent: U_att)
        attn_weights = self.attention_net(seq_output).squeeze(-1) # [batch, seq_len]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        u_att = torch.bmm(attn_weights.unsqueeze(1), seq_output).squeeze(1) # [batch, emb_dim]

        # 3. 融合意圖 (這裡使用拼接或加權，論文建議融合後輸出)
        # 為了保持維度一致，這裡採用相加或後續由 Predictor 處理
        return u_last, u_att

class SequentialEncoder(nn.Module):
    def __init__(self, emb_dim, max_seq_len=50):
        super(SequentialEncoder, self).__init__()
        # 雙通道 Transformer
        self.sim_transformer = MultiChannelTransformer(emb_dim)
        self.rel_transformer = MultiChannelTransformer(emb_dim)
        
        self.intent_capture = IntentCapture(emb_dim)

    def forward(self, sim_seq_embs, rel_seq_embs, mask=None):
        """
        sim_seq_embs: 相似性特徵序列 [batch, seq_len, emb_dim]
        rel_seq_embs: 互補性特徵序列 [batch, seq_len, emb_dim]
        """
        # 通道 1: 處理相似性行為路徑
        sim_out = self.sim_transformer(sim_seq_embs, mask)
        u_sim_last, u_sim_att = self.intent_capture(sim_out, mask)
        
        # 通道 2: 處理互補性行為路徑
        rel_out = self.rel_transformer(rel_seq_embs, mask)
        u_rel_last, u_rel_att = self.intent_capture(rel_out, mask)
        
        return (u_sim_last, u_sim_att), (u_rel_last, u_rel_att)