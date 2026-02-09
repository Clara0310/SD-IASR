import json
import os
import argparse
import sys

def main():
    # 1. 設定參數解析
    parser = argparse.ArgumentParser(description="Convert SD-IASR data for Baselines (SASRec, BARec, STIRec)")
    parser.add_argument('--dataset', type=str, required=True, help="The name of the dataset (e.g., Grocery_and_Gourmet_Food)")
    
    args = parser.parse_args()
    dataset_name = args.dataset

    # 定義輸入檔案路徑
    input_file = f"./data_preprocess/tmp/{dataset_name}_sequences_with_time.json"
    output_base = "./data_for_baselines"

    if not os.path.exists(input_file):
        print(f"Error: 找不到檔案 {input_file}")
        print(f"請確認你是否已經針對 {dataset_name} 執行過 'process_all.sh'？")
        sys.exit(1)

    print(f"[{dataset_name}] Loading raw sequences from {input_file}...")
    
    # 讀取資料
    user_seq_dict = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            user_seq_dict = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        user_seq_dict.update(data)
                    except:
                        pass

    print(f"Original users: {len(user_seq_dict)}")

    # ==========================================
    # 2. 資料過濾與 ID 映射 (關鍵修正)
    # ==========================================
    valid_data = []
    
    # [修正 1] User ID 是字串，我們按字母順序排序
    sorted_user_ids = sorted(user_seq_dict.keys())
    
    # [修正 2] 建立 User String -> Int 的映射 (從 0 開始編號)
    user_str_to_int = {u_str: i for i, u_str in enumerate(sorted_user_ids)}

    print(f"Mapping {len(sorted_user_ids)} users to integers (0 to {len(sorted_user_ids)-1})...")

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
            'u': user_str_to_int[uid_str], # [修正 3] 使用映射後的整數 ID
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
    # ==========================================
    sasrec_file = os.path.join(sasrec_dir, f"{dataset_name}.txt")
    print(f"Exporting SASRec -> {sasrec_file}")
    
    with open(sasrec_file, 'w') as f:
        for user_row in valid_data:
            # User ID + 1 (SASRec 習慣從 1 開始)
            u_id = user_row['u'] + 1 
            for i_id in user_row['items']:
                # Item ID 保持 +1
                f.write(f"{u_id} {i_id + 1}\n")

    # ==========================================
    # 4. 輸出給 BARec
    # ==========================================
    print(f"Exporting BARec -> {barec_dir}/[train|valid|test].txt")
    
    with open(os.path.join(barec_dir, "train.txt"), 'w') as f_train, \
         open(os.path.join(barec_dir, "valid.txt"), 'w') as f_val, \
         open(os.path.join(barec_dir, "test.txt"), 'w') as f_test:
        
        for user_row in valid_data:
            u = user_row['u'] + 1
            items = [i + 1 for i in user_row['items']]
            times = user_row['times']
            
            # Split
            for i in range(len(items) - 2):
                f_train.write(f"{u}\t{items[i]}\t{times[i]}\n")
            
            f_val.write(f"{u}\t{items[-2]}\t{times[-2]}\n")
            f_test.write(f"{u}\t{items[-1]}\t{times[-1]}\n")

    # ==========================================
    # 5. 輸出給 STIRec
    # ==========================================
    stirec_file = os.path.join(stirec_dir, f"{dataset_name}.txt")
    print(f"Exporting STIRec -> {stirec_file}")
    
    with open(stirec_file, 'w') as f:
        for user_row in valid_data:
            u_id = user_row['u'] + 1
            item_strs = [str(i + 1) for i in user_row['items']]
            f.write(f"{u_id} {' '.join(item_strs)}\n")

    print(f"[{dataset_name}] Conversion Complete!")
    print("-" * 30)

if __name__ == "__main__":
    main()