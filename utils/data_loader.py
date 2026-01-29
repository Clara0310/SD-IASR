# 載入序列資料與圖矩陣

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequentialDataset(Dataset):
    def __init__(self, data, max_len=50):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # data 格式: [history_seq, pos_item, neg_items...]
        line = self.data[index]
        history = line[0] # 這是 [[id, time], [id, time]...]
        
        # 只提取商品 ID
        item_history = [x[0] for x in history]
        
        # 進行 Padding (補 0) 或截斷
        seq = np.zeros(self.max_len, dtype=np.int64)
        idx = min(len(item_history), self.max_len)
        seq[:idx] = item_history[-idx:]
        
        # 正樣本與負樣本
        target_items = np.array(line[1:], dtype=np.int64)
        
        return torch.LongTensor(seq), torch.LongTensor(target_items)

def get_loader(dataset_path, batch_size, max_len):
    # 載入 7_dataset_split.py 產出的 npz
    data = np.load(dataset_path, allow_pickle=True)
    
    train_loader = DataLoader(SequentialDataset(data['train_set'], max_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequentialDataset(data['val_set'], max_len), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SequentialDataset(data['test_set'], max_len), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, data