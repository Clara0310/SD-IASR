import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequentialDataset(Dataset):
    def __init__(self, data, max_len=50, time_span=256):
        self.data = data
        self.max_len = max_len
        self.time_span = time_span

    def __len__(self):
        return len(self.data)
    
    # [新增函數] 計算時間間隔並離散化
    def compute_time_intervals(self, timestamps):
        if len(timestamps) < 2:
            return np.zeros(len(timestamps), dtype=np.int64)
        
        # 計算時間差 (單位: 秒)
        timestamps = np.array(timestamps)
        # 間隔 = [0, t1-t0, t2-t1, ...]
        intervals = np.zeros_like(timestamps)
        intervals[1:] = timestamps[1:] - timestamps[:-1]
        
        # 離散化 (Log Bucketing)
        # 公式: bucket = round(log(interval + 1))
        # 這樣短時間的差異會分很細，長時間的差異會分很粗
        # 加上 1 是避免 log(0)
        time_buckets = np.log(intervals + 1) / np.log(2) # log2
        time_buckets = np.round(time_buckets).astype(np.int64)
        
        # 截斷超過 time_span 的值
        time_buckets = np.clip(time_buckets, 0, self.time_span - 1)
        return time_buckets

    def __getitem__(self, index):
        # data 格式來自 7_dataset_split.py 的輸出
        line = self.data[index]
        
        history = line[0] 
        history_times = line[1] # 取出時間
        
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
            
        #  Time 處理
        time_buckets = self.compute_time_intervals(history_times)
        time_seq = np.zeros(self.max_len, dtype=np.int64)
        if idx > 0:
            time_seq[-idx:] = time_buckets[-idx:]
        
        # 3. 取得正樣本與負樣本 (其餘元素)
        target_items = np.array(line[2:], dtype=np.int64)
        
        return torch.LongTensor(seq), torch.LongTensor(time_seq), torch.LongTensor(target_items)
    
def get_loader(dataset_path, batch_size, max_len):
    """載入 7_dataset_split.py 產出的 .npz 打包數據"""
    # 使用 allow_pickle=True 讀取非結構化資料
    data = np.load(dataset_path, allow_pickle=True)
    
    # 建立 PyTorch DataLoader
    train_loader = DataLoader(
        SequentialDataset(data['train_set'], max_len), 
        batch_size=1, 
        shuffle=True
    )
    val_loader = DataLoader(
        SequentialDataset(data['val_set'], max_len), 
        batch_size=1, 
        shuffle=False
    )
    test_loader = DataLoader(
        SequentialDataset(data['test_set'], max_len), 
        batch_size=1, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader, data