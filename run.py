import datetime
from xml.parsers.expat import model
import torch
import torch.optim as optim
import argparse
import os
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import yaml
from tqdm import tqdm

# 匯入自定義模組
from models import SDIASR
from utils.data_loader import get_loader
from utils.graph_utils import create_laplacian
from utils.metrics import get_metrics, print_metrics
from loss import SDIASRLoss

def main():
    parser = argparse.ArgumentParser(description="Run SD-IASR Model")
    # 基礎設定
    parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=64) 

    parser.add_argument('--bert_dim', type=int, default=768, help='Dimension of pre-trained BERT embeddings')
    parser.add_argument('--lr', type=float, default=0.001) #0.005
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    
    # SD-IASR 核心超參數
    parser.add_argument('--low_k', type=int, default=5, help='Prop steps for similarity graph')
    parser.add_argument('--mid_k', type=int, default=5, help='Prop steps for complementarity graph')
    
    # Transformer 相關參數
    # 新增：控制 Transformer 層數，建議設為 2 或 3
    parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer layers')
    # 新增：接收 Transformer 多頭注意力數量
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    
    #loss 權重參數
    parser.add_argument('--lambda_1', type=float, default=1.0, help='Weight for similarity loss')
    parser.add_argument('--lambda_2', type=float, default=1.0, help='Weight for complementarity loss')
    parser.add_argument('--lambda_3', type=float, default=0.01, help='Regularization weight')
    
    # 新增Dropout 參數
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    parser.add_argument('--max_seq_len', type=int, default=50)
    
    #lr_scheduler 相關參數
    parser.add_argument('--lr_mode', type=str, default='max', help='Scheduler mode (min or max)')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Learning rate reduction factor')
    parser.add_argument('--lr_patience', type=int, default=5, help='Scheduler patience (epochs to wait before reduction)')
    
    # 續跑功能開關
    parser.add_argument('--resume', action='store_true', help='是否從上次的最佳權重續跑')
    
    args = parser.parse_args()
    
    # 建立時間標記字串
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # 1. 建立 Checkpoints 目錄與儲存 Config
    checkpoint_dir = f"./checkpoints/{args.dataset}/{timestamp}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f)
    print(f"Hyperparameters saved to {config_path}")

    # 2. 裝置設定
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 載入資料與預處理圖結構
    data_path = f"./data_preprocess/processed/{args.dataset}.npz"
    train_loader, val_loader, test_loader, raw_data = get_loader(data_path, args.batch_size, args.max_seq_len)
    
    num_items = raw_data['features'].shape[0]
    
    # === 新增：價格特徵處理 (參考 SR-Rec) ===
    # 原始 features 結構: [cid2, cid3, price]
    raw_features = raw_data['features']
    prices = raw_features[:, 2].reshape(-1, 1) # 取出價格欄位
    
    # 使用 KBinsDiscretizer 將價格分為 20 個區間
    # encode='ordinal' 會輸出 0, 1, 2... 的整數 ID
    est = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
    price_ids = est.fit_transform(prices).astype(int).flatten()
    
    # 建立 item_to_price 字典，稍後傳給模型
    item_to_price = {i: price_ids[i] for i in range(num_items)}
    print("Price discretization finished.")
    # ======================================
    
    
    
    sim_laplacian = create_laplacian(raw_data['sim_edge_index'], num_items).to(device)
    com_laplacian = create_laplacian(raw_data['com_edge_index'], num_items).to(device)

    # 4. 初始化模型與 Loss
    model = SDIASR(
        item_num=num_items, 
        bert_dim=args.bert_dim,     # 使用新接收的參數
        emb_dim=args.embedding_dim, 
        low_k=args.low_k, 
        mid_k=args.mid_k, 
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers, # 傳入層數
        nhead=args.nhead ,          # 傳入頭數
        dropout=args.dropout        # 傳入 dropout
    ).to(device)
    
    # --- 新增：載入預訓練 BERT 嵌入的邏輯 ---
    # 從 raw_data 提取商品與類別映射關係
    item_to_cid = {}
    for i in range(num_items):
        # 根據 4_data_formulator.py，features 索引 0 為 cid2, 索引 1 為 cid3
        item_to_cid[i] = (int(raw_data['features'][i][0]), int(raw_data['features'][i][1]))

    emb_path = f"./data_preprocess/embs/{args.dataset}_embeddings.npz"
    if os.path.exists(emb_path):
        print(f"Loading BERT embeddings from {emb_path}...")
        emb_data = np.load(emb_path)
        # 呼叫模型內部的載入函數（需搭配修改後的 models/sd_iasr.py）
        model.load_pretrain_embedding(
            cid2_emb=emb_data['cid2_emb'], 
            cid3_emb=emb_data['cid3_emb'], 
            item_to_cid=item_to_cid,
            item_to_price=item_to_price
        )
    else:
        print("Warning: BERT embedding file not found. Using random initialization.")
    # ---------------------------------------

    # 這裡手動設定 lambda_1 和 lambda_2
    #criterion = SDIASRLoss(lambda_reg=args.lambda_3)
    criterion = SDIASRLoss(
        lambda_1=args.lambda_1, 
        lambda_2=args.lambda_2,
        lambda_reg=args.lambda_3
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 加入學習率排程器
    # 當 Val HR@10 超過 5 個 Epoch 沒有進步時，將學習率縮小為一半 (0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=args.lr_mode, 
        factor=args.lr_factor, 
        patience=args.lr_patience, 
        verbose=True
    )

    
    best_hr = 0
    start_epoch = 0
    model_save_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    # --- 續跑邏輯 (Resume Logic) ---
    if args.resume and os.path.exists(model_save_path):
        print(f"找到現有權重，正在從 {model_save_path} 載入並續跑...")
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint)
        print("權重載入成功！")
    # ----------------------------------

    # 5. 訓練與驗證循環
    early_stop_count = 0
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, total_l_seq, total_l_sim, total_l_rel = 0, 0, 0, 0 # 新增各項累計
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        for seqs, times, targets in pbar:
            seqs, times, targets = seqs.to(device), times.to(device), targets.to(device)
            
            optimizer.zero_grad()
            # 取得分支分數
            scores, alpha, sim_scores, rel_scores = model(seqs, times, targets, sim_laplacian, com_laplacian)
            
            # 計算聯合損失
            loss, l_seq, l_sim, l_rel = criterion(scores, sim_scores, rel_scores, model)
            
            loss.backward()
            optimizer.step()
            
            # 累計損失與各項分數
            total_loss += loss.item()
            total_l_seq += l_seq.item()
            total_l_sim += l_sim.item()
            total_l_rel += l_rel.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 計算平均值
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_l_seq = total_l_seq / num_batches
        avg_l_sim = total_l_sim / num_batches
        avg_l_rel = total_l_rel / num_batches

        # 驗證階段
        model.eval()
        val_hr = []
        with torch.no_grad():
            # [修改] 使用 tqdm 包裝 val_loader，顯示 "Epoch {epoch} Validating"
            for seqs, times, targets in tqdm(val_loader, desc=f"Epoch {epoch} Validating"):
                seqs, times, targets = seqs.to(device), times.to(device), targets.to(device)
                
                # scores, _ = model(seqs, targets, sim_laplacian, com_laplacian)
                # 修改為接收四個值（後三個在驗證時通常用不到，可以用 _ 忽略）
                scores, _, _, _ = model(seqs, times, targets, sim_laplacian, com_laplacian)
                metrics = get_metrics(0, scores, k_list=[10])
                val_hr.append(metrics['HR@10'])
        
        avg_hr = np.mean(val_hr)
        
        
        # 取得當前學習率以便觀察
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- 完整印出所有指標 ---
        print(f"Epoch {epoch} | TotalLoss: {avg_loss:.4f} | L_seq: {avg_l_seq:.4f} | L_sim: {avg_l_sim:.4f} | L_rel: {avg_l_rel:.4f}")
        print(f"Val HR@10: {avg_hr:.4f} | Current LR: {current_lr}")        
        
        # 執行學習率調整：根據目前的 avg_hr 判斷是否需要降速
        scheduler.step(avg_hr)
        
        # Early Stopping 與權重儲存
        if avg_hr > best_hr:
            best_hr = avg_hr
            early_stop_count = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")
        else:
            early_stop_count += 1
            if early_stop_count >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs.")
                break

    # 6. 最終測試
    print("\n" + "="*20 + " Final Testing " + "="*20)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_scores = []
    with torch.no_grad():
        for seqs, times, targets in tqdm(test_loader, desc="Testing"):
            seqs, times, targets = seqs.to(device), times.to(device), targets.to(device)
            
            #scores, _ = model(seqs, targets, sim_laplacian, com_laplacian)
            # 同樣改為接收四個值
            scores, _, _, _ = model(seqs, times, targets, sim_laplacian, com_laplacian)
            
            test_scores.append(scores)
        
        all_test_scores = torch.cat(test_scores, dim=0)
        final_results = get_metrics(0, all_test_scores, k_list=[5 , 10, 20])
        print_metrics(final_results)
    
if __name__ == "__main__":
    main()