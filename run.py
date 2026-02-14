import datetime
from xml.parsers.expat import model
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
from utils.graph_utils import create_laplacian
from utils.metrics import get_metrics, print_metrics
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
    
    # [新增] 測試模式專用參數
    parser.add_argument('--test_only', action='store_true', help='只執行測試，跳過訓練')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='測試模式下，指定要載入的模型路徑 (.pth)')
    
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
    
    
    
    #sim_laplacian = create_laplacian(raw_data['sim_edge_index'], num_items).to(device)
    #com_laplacian = create_laplacian(raw_data['com_edge_index'], num_items).to(device)
    # --- [替換為以下程式碼] ---
    # 1. 取得兩組邊的聯集
    sim_edges = torch.tensor(raw_data['sim_edge_index'])
    com_edges = torch.tensor(raw_data['com_edge_index'])
    combined_edges = torch.cat([sim_edges, com_edges], dim=1) # 合併邊

    # 2. 移除重複的邊並保持無向圖一致性
    combined_edges, _ = torch.sort(combined_edges, dim=0)
    combined_edges = torch.unique(combined_edges, dim=1)

    # 3. 建立唯一的合併拉普拉斯矩陣 (取代原本的兩個)
    combined_laplacian = create_laplacian(combined_edges, num_items).to(device)
    print(f"Graph merged: Total Unique Edges({combined_edges.shape[1]})")
    
    

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
    #model_save_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    # --- 續跑邏輯 (Resume Logic) ---
    if args.resume and os.path.exists(model_save_path):
        print(f"找到現有權重，正在從 {model_save_path} 載入並續跑...")
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint)
        print("權重載入成功！")
    # ----------------------------------

    
    # 5. 訓練與驗證循環(只有訓練模式才跑)
    if not args.test_only:
        early_stop_count = 0
        
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss, total_l_seq, total_l_sim, total_l_rel = 0, 0, 0, 0 # 新增各項累計
            total_alpha = 0     # [新增] 初始化 alpha 累加器
            total_feat_sim = 0  # [新增] 初始化特徵相似度累加器
            total_item_diff_loss = 0  # [新增] 初始化意圖差異損失累加器
            
            
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
            for seqs, times, targets in pbar:
                seqs, times, targets = seqs.to(device), times.to(device), targets.to(device)
                
                optimizer.zero_grad()
                # 1. 取得模型輸出
                # 配合階段四的 sd_iasr.py，這裡要接收 9 個回傳值
                
                #outputs = model(seqs, times, targets, sim_laplacian, com_laplacian)
                # --- [修改這一行] ---
                # 將原本餵入 sim_laplacian, com_laplacian 改為兩次都餵入同一個 combined_laplacian
                outputs = model(seqs, times, targets, combined_laplacian, combined_laplacian)
                
                scores, alpha, sim_scores, rel_scores, feat_sim, u_sim_att, u_cor_att, x_sim, x_cor = outputs

                # 2. 計算原始的聯合損失 (BPR + 正則化)
                loss, l_seq, l_sim, l_rel = criterion(scores, sim_scores, rel_scores, model)

                # 3. [關鍵新增] 商品層級解耦損失 (Item-level Disentangle Loss)
                # 我們希望所有商品的相似嵌入與互補嵌入越不一樣越好
                # 使用餘弦相似度的絕對值均值作為懲罰
                item_diff_loss = torch.mean(torch.abs(F.cosine_similarity(x_sim, x_cor, dim=-1)))

                # 4. 融合最終損失
                # 將解耦權重從 0.1 降至 0.01 (降一個數量級)
                #total_final_loss = loss + 0.15 * item_diff_loss
                # --- [修改這一行] ---
                # 將解耦權重從 0.15 降至 0.05
                total_final_loss = loss + 0.05 * item_diff_loss

                # 5. 執行反向傳播與優化
                total_final_loss.backward()
                optimizer.step()
                
                # 累計損失與各項分數
                total_loss += loss.item()
                total_l_seq += l_seq.item()
                total_l_sim += l_sim.item()
                total_l_rel += l_rel.item()
                # [新增] 累計監控數值
                total_alpha += alpha.mean().item()  # 紀錄 Alpha 均值
                total_feat_sim += feat_sim.item()   # 紀錄特徵相似度
                total_item_diff_loss += item_diff_loss.item() # 紀錄意圖差異損失
                
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "alpha": f"{alpha.mean().item():.3f}"})
                
            # 計算平均值
            num_batches = len(train_loader)
            avg_loss = total_loss / num_batches
            avg_l_seq = total_l_seq / num_batches
            avg_l_sim = total_l_sim / num_batches
            avg_l_rel = total_l_rel / num_batches
            
            avg_alpha = total_alpha / len(train_loader)
            avg_feat_sim = total_feat_sim / len(train_loader)
            
            avg_item_diff_loss = total_item_diff_loss / len(train_loader)

            # 驗證階段
            model.eval()
            val_hr_10 = []
            val_ndcg_10 = []
            with torch.no_grad():
                for seqs, times, targets in tqdm(val_loader, desc=f"Epoch {epoch} Validating"):
                    seqs, times, targets = seqs.to(device), times.to(device), targets.to(device)
                    
                    # targets 現在只有 [Batch, 1]，就是正確答案
                    target_pos = targets.squeeze() # [Batch]
                    
                    # 1. 算出所有商品的分數 [Batch, Num_Items]
                    #scores = model.predict_full(seqs, times, sim_laplacian, com_laplacian)
                    scores = model.predict_full(seqs, times, combined_laplacian, combined_laplacian)
                    # 2. 取得正確答案的分數
                    # gather 需要 index 維度一致，所以 unsqueeze
                    pos_scores = scores.gather(1, target_pos.unsqueeze(1)) # [Batch, 1]
                    
                    # 3. Masking (屏蔽歷史購買過的商品)
                    # 這些商品的分數設為 -inf，讓它們排在最後面，不影響排名
                    scores.scatter_(1, seqs, -float('inf'))
                    
                    # 3.1 [修正] 把正確答案的分數「救回來」！
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
            print(f"Epoch {epoch} | TotalLoss: {avg_loss:.4f} | L_seq: {avg_l_seq:.4f} | L_sim: {avg_l_sim:.4f} | L_rel: {avg_l_rel:.4f}| Item_Diff_Loss: {avg_item_diff_loss:.4f} | Alpha: {avg_alpha:.4f} | Feat_Sim: {avg_feat_sim:.4f}")
            print(f"Val HR@10: {avg_hr:.4f} | Val NDCG@10: {avg_ndcg:.4f} | Current LR: {current_lr}")        
            
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
        model.load_state_dict(torch.load(load_path))
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
        for seqs, times, targets in tqdm(test_loader, desc="Testing"):
            seqs, times, targets = seqs.to(device), times.to(device), targets.to(device)
            
            # targets 在全排名模式下只有 [Batch, 1]，就是正確答案
            target_pos = targets.squeeze() 
            # 處理 batch_size=1 的邊緣情況
            if target_pos.dim() == 0:
                target_pos = target_pos.unsqueeze(0)
            
            # 1. [關鍵] 呼叫 predict_full 算出所有商品的分數 [Batch, Num_Items]
            # 確保你在 models/sd_iasr.py 裡已經加入了 predict_full 方法
            #scores = model.predict_full(seqs, times, sim_laplacian, com_laplacian)
            scores = model.predict_full(seqs, times, combined_laplacian, combined_laplacian)
            
            # 2. 取得正確答案的分數
            # gather 需要 index 維度一致，所以 unsqueeze
            pos_scores = scores.gather(1, target_pos.unsqueeze(1)) # [Batch, 1]
            
            # 3. Masking (屏蔽歷史購買過的商品)
            # 將歷史商品的 index 設為負無限大，讓它們排在最後面
            scores.scatter_(1, seqs, -float('inf'))
            
            # 3.1. [修正] 把正確答案的分數「救回來」！
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