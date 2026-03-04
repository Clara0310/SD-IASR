import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F  # [新增或確認這行]
import argparse
import os
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import yaml
from tqdm import tqdm

# 匯入自定義模組
from models import SDIASR
from utils.data_loader import get_loader
from utils.graph_utils import create_laplacian,create_sr_matrices
from loss import SDIASRLoss

def main():
    parser = argparse.ArgumentParser(description="Run SD-IASR Model")
    # 基礎設定
    parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=128) # 從 64 調大

    parser.add_argument('--bert_dim', type=int, default=768, help='Dimension of pre-trained BERT embeddings')
    parser.add_argument('--lr', type=float, default=0.0005) #0.001 調小為 0.0005，#稍微調降以穩定訓練
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
    parser.add_argument('--lambda_reg', type=float, default=0.01, help='Regularization weight')
    parser.add_argument('--lambda_proto', type=float, default=0.01, help='Weight for Prototype loss')
    parser.add_argument('--lambda_spec', type=float, default=0.05, help='Weight for Spectral Orthogonality loss')
    parser.add_argument('--lambda_alpha', type=float, default=0.5, help='Weight for alpha entropy regularization (prevents channel collapse)')
    parser.add_argument('--tau', type=float, default=0.3, help='Temperature for CL')
   
    
    parser.add_argument('--gamma', type=float, default=0.1, help='Spectral signal ratio')
    parser.add_argument('--decay_days', type=float, default=0.002, help='Temporal decay rate per day for edge weights (0=no decay)')
    
    # 新增Dropout 參數
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    parser.add_argument('--max_seq_len', type=int, default=50)
    
    #lr_scheduler 相關參數 (Warmup + MultiStep)
    parser.add_argument('--warm_up_epochs', type=int, default=5, help='Linear warmup epochs')
    parser.add_argument('--milestones', type=str, default='50,100', help='Comma-separated epoch milestones for LR decay')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='LR decay factor at each milestone')
    
    # 續跑功能開關
    parser.add_argument('--resume', action='store_true', help='是否從上次的最佳權重續跑')
    # [新增] 續跑模式專用的模型路徑
    parser.add_argument('--resume_path', type=str, default=None, help='續跑模式下，指定要載入的模型路徑 (.pth)')
    
    # [新增] 測試模式專用參數
    parser.add_argument('--test_only', action='store_true', help='只執行測試，跳過訓練')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='測試模式下，指定要載入的模型路徑 (.pth)')
    
    
    parser.add_argument('--num_prototypes', type=int, default=64, help='Number of global intent prototypes')
    
    args = parser.parse_args()
    
    # 建立時間標記字串
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # 1. 建立 Checkpoints 目錄與儲存 Config
    if not args.test_only:
        # === 情況 A: 訓練模式 ===
        checkpoint_dir = f"./checkpoints/{args.dataset}/{timestamp}"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(vars(args), f)
        print(f"Hyperparameters saved to {config_path}")

        # [修正] 在這裡定義 model_save_path
        model_save_path = os.path.join(checkpoint_dir, "best_model.pth")
        
    else:
        # === 情況 B: 測試模式 ===
        # 檢查是否有提供 checkpoint 路徑
        if args.checkpoint_path is None:
            raise ValueError("Error: --test_only requires --checkpoint_path")
            
        # [修正] 直接使用使用者提供的路徑作為 model_save_path
        model_save_path = args.checkpoint_path
        print(f"Test Mode: Loading model from {model_save_path}")

    # 2. 裝置設定
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 載入資料與預處理圖結構
    data_path = f"./data_preprocess/processed/{args.dataset}.npz"
    train_loader, val_loader, test_loader, raw_data = get_loader(data_path, args.batch_size, args.max_seq_len)
    num_items = raw_data['features'].shape[0]
    
    
    
    #==================================================================================
    # 1. 預先將驗證集與測試集的全歷史轉為矩陣 (加上 Padding)
    def prepare_history_matrix(data_list, max_len=2000): # max_len 視資料集歷史長度而定
        matrix = []
        for item in data_list:
            hist = list(item[0]) # 取得該使用者的全歷史
            if len(hist) > max_len:
                hist = hist[-max_len:]
            else:
                hist = hist + [0] * (max_len - len(hist)) # Padding
            matrix.append(hist)
        return torch.LongTensor(matrix)

    print("Preparing Full History Masking Matrices...")
    val_history_matrix = prepare_history_matrix(raw_data['val_set']).to(device)
    test_history_matrix = prepare_history_matrix(raw_data['test_set']).to(device)
    #==================================================================================
    
    
    
    
  
    
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
    
    
    
    
    #==================================================================================
    # 1. 取得兩組邊 (不再合併)
    sim_edges = torch.tensor(raw_data['sim_edge_index']).t().to(device) # [2, E1]
    com_edges = torch.tensor(raw_data['com_edge_index']).t().to(device) # [2, E2]

    # 2. 時間衰減邊權重：根據商品在訓練序列中的最近互動時間加權
    sim_weights = None
    com_weights = None
    if args.decay_days > 0:
        print("Computing temporal decay edge weights...")
        # 計算每個商品最近一次被互動的時間
        item_latest_time = np.zeros(num_items)
        global_max_time = 0
        for entry in raw_data['train_set']:
            seq, times = list(entry[0]), list(entry[1])
            for item_id, t in zip(seq, times):
                item_latest_time[item_id] = max(item_latest_time[item_id], t)
                global_max_time = max(global_max_time, t)

        # 計算每個商品的新近度分數
        days_since = (global_max_time - item_latest_time) / 86400.0  # 轉換為天數
        item_recency = np.exp(-args.decay_days * days_since)
        item_recency[item_latest_time == 0] = 0.01  # 從未出現在序列中的商品給最低權重

        # 對每條邊：權重 = 兩端商品新近度的幾何平均
        sim_rows = sim_edges[0].cpu().numpy()
        sim_cols = sim_edges[1].cpu().numpy()
        sim_weights = np.sqrt(item_recency[sim_rows] * item_recency[sim_cols])

        com_rows = com_edges[0].cpu().numpy()
        com_cols = com_edges[1].cpu().numpy()
        com_weights = np.sqrt(item_recency[com_rows] * item_recency[com_cols])

        print(f"Edge weight stats - Sim: mean={sim_weights.mean():.4f}, min={sim_weights.min():.4f}, max={sim_weights.max():.4f}")
        print(f"Edge weight stats - Com: mean={com_weights.mean():.4f}, min={com_weights.min():.4f}, max={com_weights.max():.4f}")

    # 3. 物理隔離：分別產生對應通道的矩陣（帶時間衰減邊權）
    adj_sim, adj_sim_dele = create_sr_matrices(sim_edges, num_items, edge_weights=sim_weights)
    adj_cor, adj_cor_dele = create_sr_matrices(com_edges, num_items, edge_weights=com_weights)

    adj_sim     = adj_sim.to(device)
    adj_sim_dele = adj_sim_dele.to(device)
    adj_cor     = adj_cor.to(device)
    adj_cor_dele = adj_cor_dele.to(device)
    print(f"Physical Isolation: Sim_Edges({sim_edges.shape[1]}), Com_Edges({com_edges.shape[1]})")




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
        dropout=args.dropout ,       # 傳入 dropout
        gamma=args.gamma , 
        num_prototypes=args.num_prototypes
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

    # 初始化 Criterion
    criterion = SDIASRLoss(
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_reg=args.lambda_reg,
        lambda_proto=args.lambda_proto,
        lambda_spec=args.lambda_spec,
        lambda_alpha=args.lambda_alpha,
        tau=args.tau
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 學習率排程器：Warmup + MultiStep（參考 STIRec）
    # Warmup 階段線性增長 LR，之後在指定 milestone 降速
    milestones = [int(m) for m in args.milestones.split(',')]
    warm_up_epochs = args.warm_up_epochs
    lr_gamma = args.lr_gamma

    def lr_lambda(epoch):
        if epoch < warm_up_epochs:
            return (epoch + 1) / warm_up_epochs  # 線性 warmup
        factor = 1.0
        for m in milestones:
            if epoch >= m:
                factor *= lr_gamma
        return factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"LR Scheduler: Warmup {warm_up_epochs} epochs → MultiStep at {milestones} with gamma={lr_gamma}")

    
    best_hr = 0
    start_epoch = 0
    
    # --- 續跑邏輯 (Resume Logic) ---
    # 優先檢查是否有提供手動路徑，如果沒有再找當前資料夾 (雖然當前資料夾通常是空的)
    load_resume_path = args.resume_path if args.resume_path else model_save_path

    if args.resume and load_resume_path and os.path.exists(load_resume_path):
        print(f"找到現有權重，正在從 {load_resume_path} 載入並續跑...")
        checkpoint = torch.load(load_resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_hr = checkpoint['best_hr']
        start_epoch = checkpoint['epoch'] + 1
        print(f"續跑成功！從 Epoch {start_epoch} 繼續，目前最佳 HR@10={best_hr:.4f}")
    elif args.resume:
        print(f"警告：設定了 --resume 但找不到模型檔案 {load_resume_path}，將從頭開始訓練。")
    # ----------------------------------

    
    # 5. 訓練與驗證循環(只有訓練模式才跑)
    if not args.test_only:
        early_stop_count = 0
        
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss, total_l_seq, total_l_proto, total_l_spec, total_l_alpha = 0, 0, 0, 0, 0
            total_alpha = 0
            total_feat_sim = 0
            
            
            
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
            for batch_idx, (seqs, times, targets, _) in enumerate(pbar):
                seqs, times, targets = seqs.to(device), times.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(seqs, times, targets, adj_sim, adj_sim_dele, adj_cor, adj_cor_dele)
                scores, alpha, sim_scores, rel_scores, feat_sim, u_sim, u_cor, p_sim_s, p_cor_s, r_sim, r_cor = outputs

                loss, l_seq, l_proto, l_spec, l_alpha = criterion(
                    scores, sim_scores, rel_scores, alpha, u_sim, u_cor, p_sim_s, p_cor_s, r_sim, r_cor, model
                )
                loss.backward()
                
                # [新增] 梯度裁剪：將所有參數的梯度範數限制在 5.0 以內
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                total_l_seq += l_seq.item()
                total_l_proto += l_proto.item()
                total_l_spec += l_spec.item()
                total_l_alpha += l_alpha.item()
                total_alpha += alpha.mean().item()
                total_feat_sim += feat_sim.item()

                pbar.set_postfix({
                    "L_seq": f"{l_seq.item():.4f}",
                    "α_avg": f"{alpha.mean().item():.3f}",
                    "Spec": f"{l_spec.item():.3f}",
                    "Sim": f"{feat_sim.item():.2f}"
                })
                
                # 定期記錄 Alpha 分佈 (每 200 個 Batch 印出一次細節)
                if (batch_idx % 200 == 0): # 使用 batch_idx 才能每 200 步印一次
                    a_val = alpha.detach()
                    print(f"\n[Alpha Dist] Min: {a_val.min():.4f} | Max: {a_val.max():.4f} | Std: {a_val.std():.4f}")
            
            num_batches = len(train_loader)
            avg_loss = total_loss / num_batches
            avg_l_seq = total_l_seq / num_batches
            avg_proto_loss = total_l_proto / num_batches
            avg_spec_loss = total_l_spec / num_batches
            avg_l_alpha = total_l_alpha / num_batches
            avg_alpha = total_alpha / num_batches
            avg_feat_sim = total_feat_sim / num_batches

            # 驗證階段
            model.eval()
            val_hr_10 = []
            val_ndcg_10 = []
            with torch.no_grad():
                # [關鍵優化] 進入 Batch 迴圈前先算好一次就好！
                x_sim_all, x_cor_all, raw_sim_all, raw_cor_all = model.get_all_item_features(adj_sim, adj_sim_dele, adj_cor, adj_cor_dele)

                for seqs, times, targets, batch_indices in tqdm(val_loader, desc=f"Epoch {epoch} Validating"):
                    seqs, times, targets = seqs.to(device), times.to(device), targets.to(device)

                    # targets 現在只有 [Batch, 1]，就是正確答案
                    target_pos = targets.squeeze() # [Batch]

                    # 1. 算出所有商品的分數 [Batch, Num_Items]
                    # 改呼叫 fast 版本，傳入緩存的特徵
                    scores = model.predict_full_fast(seqs, times, x_sim_all, x_cor_all, raw_sim_all, raw_cor_all)
                    
                    
                    # 2. 取得正確答案的分數
                    # gather 需要 index 維度一致，所以 unsqueeze
                    pos_scores = scores.gather(1, target_pos.unsqueeze(1)) # [Batch, 1]
                    
                    # 3. Masking (屏蔽歷史購買過的商品)
                    # --- [核心優化：全歷史 Masking 一行搞定] ---
                    # 取得這一個 batch 對應的全歷史張量
                    batch_hist = val_history_matrix[batch_indices] 
                    # GPU 平行寫入 -inf
                    scores.scatter_(1, batch_hist, -float('inf')) 
                                       
                    # 如果正確答案在 seqs 裡，它剛剛被誤殺了，現在我們把它還原
                    scores.scatter_(1, target_pos.unsqueeze(1), pos_scores)
                    
                    # 4. 計算排名 (GPU 平行運算)
                    # 排名 = (有多少個商品的分數 > 正確答案的分數) + 1
                    # 這是最快的排名算法，完全不需要 sort
                    rank = (scores > pos_scores).sum(dim=1) + 1 # [Batch]
                    
                    # 5. 計算指標
                    # HR@10
                    hr_10 = (rank <= 10).float().mean()
                    val_hr_10.append(hr_10.item())
                    
                    # NDCG@10
                    ndcg_10 = (1.0 / torch.log2(rank.float() + 1.0)) * (rank <= 10).float()
                    val_ndcg_10.append(ndcg_10.mean().item())

            avg_hr = np.mean(val_hr_10)
            avg_ndcg = np.mean(val_ndcg_10)
            
            
            # 取得當前學習率以便觀察
            current_lr = optimizer.param_groups[0]['lr']
            
            # --- 完整印出所有指標 ---
            print(f"Epoch {epoch} | TotalLoss: {avg_loss:.4f} | L_seq: {avg_l_seq:.4f} | L_proto: {avg_proto_loss:.4f} | L_spec: {avg_spec_loss:.4f} | L_alpha: {avg_l_alpha:.4f} | Alpha: {avg_alpha:.4f} | Feat_Sim: {avg_feat_sim:.4f}")
            print(f"Val HR@10: {avg_hr:.4f} | Val NDCG@10: {avg_ndcg:.4f} | Current LR: {current_lr}")        
            
            # 執行學習率調整：每個 epoch 結束後 step
            scheduler.step()
            
            # Early Stopping 與權重儲存
            if avg_hr > best_hr:
                best_hr = avg_hr
                early_stop_count = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_hr': best_hr,
                    'epoch': epoch,
                }, model_save_path)
                print(f"New best model saved to {model_save_path}")
            else:
                early_stop_count += 1
                if early_stop_count >= args.patience:
                    print(f"Early stopping triggered after {args.patience} epochs.")
                    break

    # 6. 最終測試
    # 6. 最終測試 (修正版：支援 Full Ranking)
    print("\n" + "="*20 + " Final Testing (Full Ranking) " + "="*20)
    
    # 載入最佳權重
    # [修改] 根據模式決定載入哪個權重
    if args.test_only:
        # 測試模式：載入指定的檔案
        load_path = args.checkpoint_path
    else:
        # 訓練模式：載入剛剛存好的 best_model
        load_path = model_save_path

    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)  # 相容舊格式
        print(f"Loaded best model from {load_path}")
    else:
        print(f"Error: Model file not found at {load_path}")
        return # 找不到模型就沒必要測了
    
    model.eval()
    
    # 用來儲存所有 batch 的結果
    test_hr_5, test_ndcg_5 = [], []
    test_hr_10, test_ndcg_10 = [], []
    test_hr_20, test_ndcg_20 = [], []
    
    with torch.no_grad():
        # [新增] 迴圈外先預計算一次
        x_sim_all, x_cor_all, raw_sim_all, raw_cor_all = model.get_all_item_features(adj_sim, adj_sim_dele, adj_cor, adj_cor_dele)

        for seqs, times, targets, batch_indices in tqdm(test_loader, desc="Testing"):
            seqs, times, targets = seqs.to(device), times.to(device), targets.to(device)

            # targets 在全排名模式下只有 [Batch, 1]，就是正確答案
            target_pos = targets.squeeze()
            # 處理 batch_size=1 的邊緣情況
            if target_pos.dim() == 0:
                target_pos = target_pos.unsqueeze(0)

            # 1. [關鍵] 呼叫 predict_full 算出所有商品的分數 [Batch, Num_Items]
            scores = model.predict_full_fast(seqs, times, x_sim_all, x_cor_all, raw_sim_all, raw_cor_all)
            
            
            # 2. 取得正確答案的分數
            # gather 需要 index 維度一致，所以 unsqueeze
            pos_scores = scores.gather(1, target_pos.unsqueeze(1)) # [Batch, 1]
            
            
            # 3. Masking (屏蔽歷史購買過的商品)
            # --- [全歷史 Masking 優化] ---
            batch_hist = test_history_matrix[batch_indices]
            scores.scatter_(1, batch_hist, -float('inf'))           
            # 如果正確答案在 seqs 裡，它剛剛被誤殺了，現在我們把它還原
            scores.scatter_(1, target_pos.unsqueeze(1), pos_scores)
            
            # 4. 計算排名 (GPU 平行運算，極速！)
            # 排名 = (有多少個商品的分數 > 正確答案的分數) + 1
            rank = (scores > pos_scores).sum(dim=1) + 1 # [Batch]
            
            # 5. 計算指標
            # HR@K
            test_hr_5.append((rank <= 5).float().mean().item())
            test_hr_10.append((rank <= 10).float().mean().item())
            test_hr_20.append((rank <= 20).float().mean().item())
            
            # NDCG@K
            test_ndcg_5.append(((1.0 / torch.log2(rank.float() + 1.0)) * (rank <= 5).float()).mean().item())
            test_ndcg_10.append(((1.0 / torch.log2(rank.float() + 1.0)) * (rank <= 10).float()).mean().item())
            test_ndcg_20.append(((1.0 / torch.log2(rank.float() + 1.0)) * (rank <= 20).float()).mean().item())

    # 輸出最終平均結果
    print("-" * 30)
    print(f"Test HR@5:   {np.mean(test_hr_5):.4f} | NDCG@5:  {np.mean(test_ndcg_5):.4f}")
    print(f"Test HR@10:  {np.mean(test_hr_10):.4f} | NDCG@10: {np.mean(test_ndcg_10):.4f}")
    print(f"Test HR@20:  {np.mean(test_hr_20):.4f} | NDCG@20: {np.mean(test_ndcg_20):.4f}")
    print("-" * 30)
    
if __name__ == "__main__":
    main()