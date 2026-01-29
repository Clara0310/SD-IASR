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
        # data 格式來自 7_dataset_split.py 的輸出
        line = self.data[index]
        history = line[0] 
        
        # 1. 提取商品 ID 並處理可能的格式差異
        item_history = []
        for x in history:
            if isinstance(x, (list, np.ndarray, tuple)):
                # 如果格式是 [item_id, timestamp]，提取第一個元素
                item_history.append(x[0])
            else:
                # 如果格式已經是單純的 item_id (int)
                item_history.append(x)
        
        # 2. 修正：確保 seq 長度永遠等於 self.max_len，解決維度不匹配問題
        seq = np.zeros(self.max_len, dtype=np.int64)
        idx = min(len(item_history), self.max_len)
        if idx > 0:
            # 採用「靠右對齊」：將最近的行為放在末端，前方補 0
            # 這樣產生的 mask (seq == 0) 才能與 Transformer 期待的維度一致
            seq[-idx:] = item_history[-idx:]
        
        # 3. 取得正樣本與負樣本 (其餘元素)
        target_items = np.array(line[1:], dtype=np.int64)
        
        return torch.LongTensor(seq), torch.LongTensor(target_items)

def get_loader(dataset_path, batch_size, max_len):
    """載入 7_dataset_split.py 產出的 .npz 打包數據"""
    # 使用 allow_pickle=True 讀取非結構化資料
    data = np.load(dataset_path, allow_pickle=True)
    
    # 建立 PyTorch DataLoader
    train_loader = DataLoader(
        SequentialDataset(data['train_set'], max_len), 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        SequentialDataset(data['val_set'], max_len), 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        SequentialDataset(data['test_set'], max_len), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader, data