# 使用 BERT 產生類別嵌入

import sys
import os
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel

def main():
    if len(sys.argv) < 2:
        print("Usage: python 5_embs_generator.py <dataset_name>")
        sys.exit(1)

    data_name = sys.argv[1]
    print(f"Generating category embeddings with BERT for {data_name}...")

    # 定義分類字典路徑 (由 4_data_formulator.py 產出)
    data_path1 = f"./data_preprocess/tmp/{data_name}_cid2_dict.txt"
    data_path2 = f"./data_preprocess/tmp/{data_name}_cid3_dict.txt"
    
    # 確保輸出目錄存在
    emb_dir = "./data_preprocess/embs"
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)
    
    # 檢查設備：優先使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 載入預訓練的 BERT 模型和 tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    
    embedding_matrix_list = []
    
    # 為兩個分類級別分別生成嵌入
    for i, data_path in enumerate([data_path1, data_path2]):
        if not os.path.exists(data_path):
            print(f"Error: {data_path} not found.")
            continue
            
        # 讀取分類 ID 和對應的文本
        id_pd = pd.read_csv(data_path, sep='\t', header=None, names=['id', 'name'])
        # 按分類 ID 排序，確保矩陣索引正確
        id_pd = id_pd.sort_values(by='id')
        categories = id_pd['name'].values

        embeddings_matrix = []
        print(f"Processing category level {i+2}, total unique names: {len(categories)}")

        # 為每個分類文本生成嵌入
        for text in categories:
            # 處理可能的缺失值
            text = str(text) if pd.notnull(text) else "unknown"
            
            # 使用 tokenizer 將文本轉換為 token
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

            # 使用 BERT 生成嵌入 (不計算梯度)
            with torch.no_grad():
                outputs = model(**inputs)

            # 獲得最後隱藏層輸出並對序列維度取平均
            # last_hidden_states shape: [1, seq_len, 768]
            embeddings = torch.mean(outputs.last_hidden_state, dim=1) 
            embeddings_matrix.append(embeddings.cpu())
        
        # 連接為矩陣並儲存
        embeddings_matrix = torch.cat(embeddings_matrix, dim=0).numpy()
        embedding_matrix_list.append(embeddings_matrix)
        print(f"Category {i+2} embedding shape: {embeddings_matrix.shape}")
    
    # 儲存為 NPZ 檔案
    save_path = f"./data_preprocess/embs/{data_name}_embeddings.npz"
    np.savez(save_path, cid2_emb=embedding_matrix_list[0], cid3_emb=embedding_matrix_list[1])
    print(f"Embeddings saved to: {save_path}")

if __name__ == '__main__':
    main()