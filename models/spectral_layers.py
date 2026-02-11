# 參考 SR-Rec 的 GCN_Low/Mid，實作譜解耦

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class SpectralConv(nn.Module):
    def __init__(self, c_in, c_out, prop_step,dropout=0.0):
        super(SpectralConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.prop_step = prop_step
        self.dropout = dropout  # 儲存 dropout 率
        # 定義譜卷積的權重矩陣
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        nn.init.xavier_uniform_(self.weight)

    def get_laplacian(self, edge_index, num_nodes):
        """計算正規化圖拉普拉斯矩陣: L = I - D^(-1/2) A D^(-1/2)"""
        row, col = edge_index[0], edge_index[1]
        # 建立鄰接矩陣 (無向圖)
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        # 加入自環並計算度數矩陣 D
        adj_loop = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_loop.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # 計算正規化鄰接矩陣 A_hat = D^-1/2 * A * D^-1/2
        a_hat = d_mat_inv_sqrt.dot(adj_loop).dot(d_mat_inv_sqrt).tocoo()
        
        # 轉為 PyTorch 稀疏張量
        indices = torch.from_numpy(np.vstack((a_hat.row, a_hat.col)).astype(np.int64))
        values = torch.from_numpy(a_hat.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, torch.Size(a_hat.shape))

    def forward(self, x, laplacian, filter_type='low'):
        """
        實作譜濾波器
        filter_type: 'low' 為低通濾波器 (相似性), 'mid' 為中通濾波器 (互補性)
        """
        """
        實作帶有殘差連接的譜濾波器
        """
        # 1. 紀錄原始輸入作為殘差項 (Identity)
        # 這能確保高品質的 BERT 特徵被完整保留
        identity = x
        
        # 2. 執行線性轉換與 Dropout
        # 注意：這裡會將特徵從 x 變換為卷積分支的特徵
        x_transformed = torch.matmul(x, self.weight)
        x_transformed = F.dropout(x_transformed, p=self.dropout, training=self.training)
        
        # training=self.training 確保只有在訓練時會隨機丟棄，測試時會保留
        #x = F.dropout(x, p=self.dropout, training=self.training)
        
        if filter_type == 'low':
            # --- 低通濾波分支 ---
            out = x_transformed
            for _ in range(self.prop_step):
                out = torch.spmm(laplacian, out)
                
            # [關鍵修改] 加上殘差項
            return identity + out
            
        elif filter_type == 'mid':
            # [恢復並強化] 中通濾波邏輯
            low_component = x_transformed
            for _ in range(self.prop_step):
                low_component = torch.spmm(laplacian, low_component)
            
            # 中通信號 = (低頻 k 階) - (低頻 k+1 階)
            mid_signal = low_component - torch.spmm(laplacian, low_component)
            
            # 給予中通特徵 2.0 倍的縮放係數，強迫模型注意到互補關係的微調
            return identity + 2.0 * mid_signal
        
        

class SpectralDisentangler(nn.Module):
    def __init__(self, item_num, emb_dim, low_k, mid_k,dropout=0.0):
        super(SpectralDisentangler, self).__init__()
        self.low_conv = SpectralConv(emb_dim, emb_dim, low_k, dropout=dropout)
        self.mid_conv = SpectralConv(emb_dim, emb_dim, mid_k, dropout=dropout)
        
    def forward(self, item_embs, sim_laplacian, com_laplacian):
        """
        雙通道譜解耦過程
        item_embs: 初始商品嵌入 (來自 BERT)
        """
        # 通道 1: 相似性解耦 (低通濾波)
        sim_features = self.low_conv(item_embs, sim_laplacian, filter_type='low')
        
        # 通道 2: 互補性解耦 (中通濾波)
        com_features = self.mid_conv(item_embs, com_laplacian, filter_type='mid')
        
        return sim_features, com_features