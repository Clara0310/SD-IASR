import torch
import torch.optim as optim
import argparse
import os
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
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    
    # SD-IASR 核心超參數
    parser.add_argument('--low_k', type=int, default=2, help='Prop steps for similarity graph')
    parser.add_argument('--mid_k', type=int, default=2, help='Prop steps for complementarity graph')
    parser.add_argument('--lambda_3', type=float, default=0.01, help='Regularization weight')
    parser.add_argument('--max_seq_len', type=int, default=50)
    
    # 新增：續跑功能開關
    parser.add_argument('--resume', action='store_true', help='是否從上次的最佳權重續跑')
    
    args = parser.parse_args()

    # 1. 建立 Checkpoints 目錄與儲存 Config
    checkpoint_dir = f"./checkpoints/{args.dataset}"
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
    sim_laplacian = create_laplacian(raw_data['sim_edge_index'], num_items).to(device)
    com_laplacian = create_laplacian(raw_data['com_edge_index'], num_items).to(device)

    # 4. 初始化模型與 Loss
    model = SDIASR(num_items, args.embedding_dim, args.low_k, args.mid_k, args.max_seq_len).to(device)
    
    criterion = SDIASRLoss(lambda_reg=args.lambda_3)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 新增：續跑邏輯 (Resume Logic) ---
    best_hr = 0
    start_epoch = 0
    model_save_path = os.path.join(checkpoint_dir, "best_model.pth")

    if args.resume and os.path.exists(model_save_path):
        print(f"找到現有權重，正在從 {model_save_path} 載入並續跑...")
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint)
        print("權重載入成功！")
        # 由於我們沒有額外儲存 epoch 資訊，這裡會從第 0 輪開始重新優化，但權重是基礎好的
    # ----------------------------------

    # 5. 訓練與驗證循環
    early_stop_count = 0
    
    #暫時註解掉訓練
    
    # for epoch in range(start_epoch, args.epochs):
    #     model.train()
    #     total_loss = 0
    #     pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    #     for seqs, targets in pbar:
    #         seqs, targets = seqs.to(device), targets.to(device)
            
    #         optimizer.zero_grad()
    #         scores, _ = model(seqs, targets, sim_laplacian, com_laplacian)
            
    #         loss, _, _ = criterion(scores, model)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    #     # 驗證階段
    #     model.eval()
    #     val_hr = []
    #     with torch.no_grad():
    #         for seqs, targets in val_loader:
    #             seqs, targets = seqs.to(device), targets.to(device)
    #             scores, _ = model(seqs, targets, sim_laplacian, com_laplacian)
    #             metrics = get_metrics(0, scores, k_list=[10])
    #             val_hr.append(metrics['HR@10'])
        
    #     avg_hr = np.mean(val_hr)
    #     print(f"Epoch {epoch} | Avg Loss: {total_loss/len(train_loader):.4f} | Val HR@10: {avg_hr:.4f}")

    #     # Early Stopping 與權重儲存
    #     if avg_hr > best_hr:
    #         best_hr = avg_hr
    #         early_stop_count = 0
    #         torch.save(model.state_dict(), model_save_path)
    #         print(f"New best model saved to {model_save_path}")
    #     else:
    #         early_stop_count += 1
    #         if early_stop_count >= args.patience:
    #             print(f"Early stopping triggered after {args.patience} epochs.")
    #             break

    #暫時註解掉訓練

    # 6. 最終測試
    # print("\n" + "="*20 + " Final Testing " + "="*20)
    # model.load_state_dict(torch.load(model_save_path))
    # model.eval()
    # test_scores = []
    # with torch.no_grad():
    #     for seqs, targets in tqdm(test_loader, desc="Testing"):
    #         seqs, targets = seqs.to(device), targets.to(device)
    #         scores, _ = model(seqs, targets, sim_laplacian, com_laplacian)
    #         test_scores.append(scores)
        
    #     all_test_scores = torch.cat(test_scores, dim=0)
    #     final_results = get_metrics(0, all_test_scores, k_list=[10, 20])
    #     print_metrics(final_results)
    
    # 修改 run.py 的測試區塊，以符合 1:99 負採樣評估
    print("\n" + "="*20 + " Final Testing (1:99 Negative Sampling) " + "="*20)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    test_metrics = {'HR@5': [], 'HR@10': [], 'NDCG@10': []}

    with torch.no_grad():
        for seqs, targets in tqdm(test_loader, desc="Testing"):
            seqs, targets = seqs.to(device), targets.to(device)
            
            # --- 核心修改：負採樣產生 ---
            # 為每個 batch 隨機產生 99 個負樣本 (排除掉 targets)
                                
            # 修改後的負採樣邏輯
            sampled_indices = []
            for target in targets:
                # 使用 .flatten()[0] 確保只取得一個元素，不論維度如何
                pos = target.flatten()[0].item() 
                neg = []
                while len(neg) < 99:
                    n = np.random.randint(1, num_items)
                    if n != pos:
                        neg.append(n)
                sampled_indices.append([pos] + neg)    
            
            neg_samples = torch.LongTensor(sampled_indices).to(device) # [batch, 99]
            
            # 組合正負樣本: [batch, 100] (第 0 欄是正樣本)
            test_items = torch.cat([targets.unsqueeze(1), neg_samples], dim=1)
            
            # 只對這 100 個商品計算得分
            # 注意：模型需要支援只計算特定 items 的得分，若無則需從全量 scores 中 index 出來
            all_scores, _ = model(seqs, targets, sim_laplacian, com_laplacian)
            # 從全量得分中提取出這 100 個候選者的分數
            sampled_scores = torch.gather(all_scores, 1, test_items) 
            
            # 使用您修改後的 metrics.py 計算
            res = get_metrics(num_items, sampled_scores, k_list=[5, 10])
            test_metrics['HR@5'].append(res['HR@5'])
            test_metrics['HR@10'].append(res['HR@10'])
            test_metrics['NDCG@10'].append(res['NDCG@10'])

    # 印出平均結果
    print(f"Final HR@5: {np.mean(test_metrics['HR@5']):.4f}")
    print(f"Final HR@10: {np.mean(test_metrics['HR@10']):.4f}")
    print(f"Final NDCG@10: {np.mean(test_metrics['NDCG@10']):.4f}")

if __name__ == "__main__":
    main()