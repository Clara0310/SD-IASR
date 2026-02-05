# models/sequential_encoder.py
# 修正維度對齊與遮罩邏輯後的完整版本

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
        # 注意：mask 格式需為 [batch, seq_len]，True 表示該位置被遮蔽
        x = self.pos_encoder(seq_embs)
        output = self.transformer(x, src_key_padding_mask=mask)
        return output

class IntentCapture(nn.Module):
    def __init__(self, emb_dim):
        super(IntentCapture, self).__init__()
        # 用於計算注意力分數的權重網路
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
        u_last = seq_output[:, -1, :]

        # 2. 計算注意力池化作為全局意圖 (Global Intent: U_att)
        attn_weights = self.attention_net(seq_output).squeeze(-1) # [batch, seq_len]
        
        # 修正：確保遮罩維度與 attn_weights 嚴格對齊 (皆為 50)
        if mask is not None:
            # 使用 float('-inf') 確保 Padding 位置在 Softmax 後權重為 0
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 數值穩定性：防止序列全為 Padding 時產生的 NaN
        attn_weights = torch.nan_to_num(attn_weights)
        
        u_att = torch.bmm(attn_weights.unsqueeze(1), seq_output).squeeze(1) # [batch, emb_dim]

        return u_last, u_att

class SequentialEncoder(nn.Module):
    def __init__(self, emb_dim, max_seq_len, num_layers, nhead):
        super(SequentialEncoder, self).__init__()
        
        # 新增：初始化位置編碼
        self.pos_encoder = PositionalEncoding(emb_dim, max_len=max_seq_len)
        
        # 使用 nn.TransformerEncoderLayer 配合 num_layers 增加深度
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        
        # 雙通道 Transformer 架構
        # 相似性通道與互補性通道皆增加深度
        self.sim_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.intent_capture = IntentCapture(emb_dim)

    def forward(self, sim_seq_embs, cor_seq_embs, mask):
        """
        sim_seq_embs: 相似性特徵序列 [batch, seq_len, emb_dim]
        cor_seq_embs: 互補性特徵序列 [batch, seq_len, emb_dim]
        mask: Padding Mask [batch, seq_len]
        """
        # 核心修正：加上位置編碼 (Positional Encoding)
        sim_seq_embs = self.pos_encoder(sim_seq_embs)
        cor_seq_embs = self.pos_encoder(cor_seq_embs)
        
        # 之後再送入 Transformer
        
        # 通道 1: 處理相似性行為路徑
        sim_out = self.sim_transformer(sim_seq_embs, src_key_padding_mask=mask)
        u_sim_last, u_sim_att = self.intent_capture(sim_out, mask)
        
        # 通道 2: 處理互補性行為路徑
        cor_out = self.cor_transformer(cor_seq_embs, src_key_padding_mask=mask)
        u_cor_last, u_cor_att = self.intent_capture(cor_out, mask)
        
        return (u_sim_last, u_sim_att), (u_cor_last, u_cor_att)