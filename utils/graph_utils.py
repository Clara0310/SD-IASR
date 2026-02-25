# 產生拉普拉斯矩陣與鄰接矩陣

import torch
import numpy as np
import scipy.sparse as sp


# utils/graph_utils.py 

def create_sr_matrices(edge_index, num_nodes):
    row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
    # 對稱化
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # [核心修正] 列歸一化：安全處理度數為 0 的商品
    rowsum = np.array(adj.sum(1)).flatten()
    r_inv = np.zeros_like(rowsum)
    mask = rowsum > 0 # 只對有邊的節點計算倒數
    r_inv[mask] = np.power(rowsum[mask], -1)
    
    r_mat_inv = sp.diags(r_inv)
    adj_norm = r_mat_inv.dot(adj)
    
    # 生成 A+I (Self) 與 A-I (Dele)
    adj_norm_self = adj_norm + sp.eye(num_nodes)
    adj_norm_dele = adj_norm - sp.eye(num_nodes)
    
    def to_torch_sparse(mx):
        mx = mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
        values = torch.from_numpy(mx.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(mx.shape))

    return to_torch_sparse(adj_norm_self), to_torch_sparse(adj_norm_dele)


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