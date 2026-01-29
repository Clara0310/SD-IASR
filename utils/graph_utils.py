# 產生拉普拉斯矩陣與鄰接矩陣

import torch
import numpy as np
import scipy.sparse as sp

def create_laplacian(edge_index, num_nodes):
    """
    將邊索引轉換為正規化拉普拉斯矩陣: L = I - D^-1/2 * A * D^-1/2
    """
    # 1. 建立對稱鄰接矩陣
    edges = edge_index.T
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    
    # 轉換為無向圖 (A = A + A^T)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # 2. 加入自環 A_hat = A + I
    adj_at = adj + sp.eye(adj.shape[0])
    
    # 3. 計算度數矩陣 D
    rowsum = np.array(adj_at.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # 4. 正規化: D^-1/2 * A_hat * D^-1/2
    norm_adj = d_mat_inv_sqrt.dot(adj_at).dot(d_mat_inv_sqrt).tocoo()
    
    # 5. 轉為 PyTorch 稀疏張量
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape))