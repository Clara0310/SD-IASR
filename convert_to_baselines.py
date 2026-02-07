import json
import os
import argparse
import sys

def main():
    # 1. 設定參數解析，讓腳本可以接收 --dataset 參數
    parser = argparse.ArgumentParser(description="Convert SD-IASR data for Baselines (SASRec, BARec, STIRec)")
    parser.add_argument('--dataset', type=str, required=True, help="The name of the dataset (e.g., Grocery_and_Gourmet_Food)")
    
    args = parser.parse_args()
    dataset_name = args.dataset

    # 定義輸入檔案路徑 (這是 process_all.sh 產生的中間檔)
    input_file = f"./data_preprocess/tmp/{dataset_name}_sequences_with_time.json"
    output_base = "./data_for_baselines"

    # 檢查輸入檔案是否存在
    if not os.path.exists(input_file):
        print(f"Error: 找不到檔案 {input_file}")
        print(f"請確認你是否已經針對 {dataset_name} 執行過 'process_all.sh'？")
        sys.exit(1)

    print(f"[{dataset_name}] Loading raw sequences from {input_file}...")
    
    # 讀取資料
    user_seq_dict = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            # 嘗試讀取整個 JSON
            user_seq_dict = json.load(f)
        except json.JSONDecodeError:
            # 如果失敗，嘗試逐行讀取 (相容不同的 JSON 格式)
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        user_seq_dict.update(data)
                    except:
                        pass

    print(f"Original users: {len(user_seq_dict)}")

    # 2. 資料過濾與排序
    # 邏輯：長度 < 3 丟棄，並確保按時間排序
    valid_data = []
    # 為了讓 User ID 在所有 Baseline 中一致，我們將其轉為 int 並排序
    sorted_user_ids = sorted(user_seq_dict.keys(), key=lambda x: int(x))

    for uid_str in sorted_user_ids:
        seq_data = user_seq_dict[uid_str]
        # seq_data 格式: [[item_id, time], [item_id, time], ...]
        
        if len(seq_data) < 3:
            continue
        
        # 按時間排序
        seq_data.sort(key=lambda x: x[1])
        
        item_seq = [x[0] for x in seq_data]
        timestamps = [x[1] for x in seq_data]
        
        valid_data.append({
            'u': int(uid_str),
            'items': item_seq,
            'times': timestamps
        })

    print(f"Valid users (len >= 3): {len(valid_data)}")
    
    # 建立輸出資料夾
    sasrec_dir = os.path.join(output_base, "SASRec")
    barec_dir = os.path.join(output_base, "BARec", dataset_name)
    stirec_dir = os.path.join(output_base, "STIRec")
    
    os.makedirs(sasrec_dir, exist_ok=True)
    os.makedirs(barec_dir, exist_ok=True)
    os.makedirs(stirec_dir, exist_ok=True)

    # ==========================================
    # 3. 輸出給 SASRec
    # 格式：User_ID Item_ID (一行一個互動)
    # 調整：Item ID 全部 +1 (因為 SASRec 預設 0 是 padding)
    # ==========================================
    sasrec_file = os.path.join(sasrec_dir, f"{dataset_name}.txt")
    print(f"Exporting SASRec -> {sasrec_file}")
    
    with open(sasrec_file, 'w') as f:
        for user_row in valid_data:
            # User ID + 1 (避免 0 的問題，雖然 SASRec 內部會重編，但這樣比較安全)
            u_id = user_row['u'] + 1 
            for i_id in user_row['items']:
                f.write(f"{u_id} {i_id + 1}\n")

    # ==========================================
    # 4. 輸出給 BARec
    # 格式：User \t Item \t Time (分為 train/valid/test)
    # 調整：Item ID 全部 +1
    # ==========================================
    print(f"Exporting BARec -> {barec_dir}/[train|valid|test].txt")
    
    with open(os.path.join(barec_dir, "train.txt"), 'w') as f_train, \
         open(os.path.join(barec_dir, "valid.txt"), 'w') as f_val, \
         open(os.path.join(barec_dir, "test.txt"), 'w') as f_test:
        
        for user_row in valid_data:
            # BARec 的 util.py 讀檔後會建立 Map，這裡給它純數字即可
            # 為了統一，我們一樣給它 +1 的 ID
            u = user_row['u'] + 1
            items = [i + 1 for i in user_row['items']]
            times = user_row['times']
            
            # Split (最後一個是 Test, 倒數第二個是 Valid, 其餘 Train)
            # Train
            for i in range(len(items) - 2):
                f_train.write(f"{u}\t{items[i]}\t{times[i]}\n")
            
            # Valid
            f_val.write(f"{u}\t{items[-2]}\t{times[-2]}\n")
            
            # Test
            f_test.write(f"{u}\t{items[-1]}\t{times[-1]}\n")

    # ==========================================
    # 5. 輸出給 STIRec
    # 格式：User_ID Item_1 Item_2 ... (一行一個 User)
    # 調整：Item ID 全部 +1
    # ==========================================
    stirec_file = os.path.join(stirec_dir, f"{dataset_name}.txt")
    print(f"Exporting STIRec -> {stirec_file}")
    
    with open(stirec_file, 'w') as f:
        for user_row in valid_data:
            u_id = user_row['u'] + 1
            # 轉成字串列表，記得 +1
            item_strs = [str(i + 1) for i in user_row['items']]
            f.write(f"{u_id} {' '.join(item_strs)}\n")

    print(f"[{dataset_name}] Conversion Complete!")
    print("-" * 30)

if __name__ == "__main__":
    main()