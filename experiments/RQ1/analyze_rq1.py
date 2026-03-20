"""
RQ1: 譜解耦語意分離效果分析 (Spectral Disentanglement Effectiveness)
=====================================================================
分析項目：
  1. feat_sim_raw / feat_sim_proj 指標計算
  2. t-SNE 視覺化：raw_sim vs raw_cor（以 CID3 類別著色）
  3. x_sim vs x_cor 空間 t-SNE 比較
  4. 訓練曲線：feat_sim 隨 epoch 變化
  5. 頻譜能量分布：低頻 vs 中頻信號在不同通道的分布

使用方式：
  cd SD-IASR
  python experiments/RQ1/analyze_rq1.py --dataset Grocery_and_Gourmet_Food \
      --checkpoint checkpoints/Grocery_and_Gourmet_Food/20260309_1945/best_model.pth \
      --training_log training_log/train_20260309_1945.log
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import re
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.preprocessing import KBinsDiscretizer
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 全域字體放大
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
})

from models import SDIASR
from utils.graph_utils import create_sr_matrices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--training_log', type=str, default=None,
                        help='訓練 log 路徑，用於繪製 feat_sim 訓練曲線')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出目錄（預設 experiments/RQ1/output/<dataset>）')
    parser.add_argument('--top_k_cid3', type=int, default=15,
                        help='t-SNE 只顯示前 K 大類別（避免太雜亂）')
    parser.add_argument('--tsne_sample', type=int, default=5000,
                        help='t-SNE 最大取樣商品數（太多會很慢）')
    parser.add_argument('--cooc_window', type=int, default=5,
                        help='共現圖滑動視窗大小（須與訓練一致）')
    parser.add_argument('--decay_days', type=float, default=0.002,
                        help='時間衰減率（須與訓練一致）')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def load_model_and_data(args):
    """載入模型 checkpoint 與資料，重建圖結構"""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 載入處理過的資料
    data_path = f"./data_preprocess/processed/{args.dataset}.npz"
    raw_data = np.load(data_path, allow_pickle=True)
    num_items = raw_data['features'].shape[0]
    features = raw_data['features']

    # 價格離散化
    prices = features[:, 2].reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
    price_ids = est.fit_transform(prices).astype(int).flatten()
    item_to_price = {i: price_ids[i] for i in range(num_items)}
    item_to_cid = {i: (int(features[i, 0]), int(features[i, 1])) for i in range(num_items)}

    # 重建圖結構（含共現圖 + 時間衰減）
    sim_edges = torch.tensor(raw_data['sim_edge_index']).t().to(device)
    com_edges = torch.tensor(raw_data['com_edge_index']).t().to(device)

    # 時間衰減
    sim_weights, com_weights = None, None
    if args.decay_days > 0:
        item_latest_time = np.zeros(num_items)
        global_max_time = 0
        for entry in raw_data['train_set']:
            seq, times = list(entry[0]), list(entry[1])
            for item_id, t in zip(seq, times):
                item_latest_time[item_id] = max(item_latest_time[item_id], t)
                global_max_time = max(global_max_time, t)
        days_since = (global_max_time - item_latest_time) / 86400.0
        item_recency = np.exp(-args.decay_days * days_since)
        item_recency[item_latest_time == 0] = 0.01

        sim_rows = sim_edges[0].cpu().numpy()
        sim_cols = sim_edges[1].cpu().numpy()
        sim_weights = np.sqrt(item_recency[sim_rows] * item_recency[sim_cols])
        com_rows = com_edges[0].cpu().numpy()
        com_cols = com_edges[1].cpu().numpy()
        com_weights = np.sqrt(item_recency[com_rows] * item_recency[com_cols])

    # 共現圖
    if args.cooc_window > 0:
        cooc_counts = defaultdict(float)
        for entry in raw_data['train_set']:
            seq = [x for x in list(entry[0]) if x != 0]
            for i in range(len(seq)):
                for j in range(i + 1, min(i + args.cooc_window + 1, len(seq))):
                    u, v = seq[i], seq[j]
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    cooc_counts[(u, v)] += 1.0 / (j - i)
        if cooc_counts:
            cooc_src = np.array([k[0] for k in cooc_counts], dtype=np.int64)
            cooc_dst = np.array([k[1] for k in cooc_counts], dtype=np.int64)
            cooc_w = np.array([cooc_counts[k] for k in cooc_counts], dtype=np.float32)
            cooc_w = cooc_w / cooc_w.max()
            if args.decay_days > 0:
                cooc_w = cooc_w * np.sqrt(item_recency[cooc_src] * item_recency[cooc_dst])
            both_src = np.concatenate([cooc_src, cooc_dst])
            both_dst = np.concatenate([cooc_dst, cooc_src])
            both_w = np.concatenate([cooc_w, cooc_w])
            cooc_edge_tensor = torch.tensor(np.stack([both_src, both_dst]), dtype=torch.long).to(device)
            n_orig = sim_edges.shape[1]
            sim_edges = torch.cat([sim_edges, cooc_edge_tensor], dim=1)
            orig_w = sim_weights if sim_weights is not None else np.ones(n_orig, dtype=np.float32)
            sim_weights = np.concatenate([orig_w, both_w])

    adj_sim, adj_sim_dele = create_sr_matrices(sim_edges, num_items, edge_weights=sim_weights)
    adj_cor, adj_cor_dele = create_sr_matrices(com_edges, num_items, edge_weights=com_weights)
    adj_sim, adj_sim_dele = adj_sim.to(device), adj_sim_dele.to(device)
    adj_cor, adj_cor_dele = adj_cor.to(device), adj_cor_dele.to(device)

    # 從 checkpoint 的 config 讀取模型參數
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, 'config.yaml')
    import yaml
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        # 使用預設值
        cfg = {'embedding_dim': 128, 'bert_dim': 768, 'low_k': 5, 'mid_k': 5,
               'max_seq_len': 50, 'num_layers': 2, 'nhead': 8, 'dropout': 0.3,
               'gamma': 0.1, 'num_prototypes': 64}

    model = SDIASR(
        item_num=num_items, bert_dim=cfg['bert_dim'], emb_dim=cfg['embedding_dim'],
        low_k=cfg['low_k'], mid_k=cfg['mid_k'], max_seq_len=cfg['max_seq_len'],
        num_layers=cfg['num_layers'], nhead=cfg['nhead'], dropout=cfg['dropout'],
        gamma=cfg['gamma'], num_prototypes=cfg['num_prototypes']
    ).to(device)

    # 載入 BERT 嵌入（初始化 item_embedding）
    emb_path = f"./data_preprocess/embs/{args.dataset}_embeddings.npz"
    if os.path.exists(emb_path):
        emb_data = np.load(emb_path)
        model.load_pretrain_embedding(emb_data['cid2_emb'], emb_data['cid3_emb'],
                                      item_to_cid, item_to_price)

    # 載入 checkpoint 權重
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {args.checkpoint}")
    epoch = checkpoint.get('epoch', '?')
    val_hr = checkpoint.get('best_val_hr', '?')
    print(f"  Epoch: {epoch}, Val HR@10: {val_hr}")

    return model, device, adj_sim, adj_sim_dele, adj_cor, adj_cor_dele, features, num_items


def extract_embeddings(model, device, adj_sim, adj_sim_dele, adj_cor, adj_cor_dele):
    """提取所有中間表示"""
    with torch.no_grad():
        all_idx = torch.arange(model.item_num).to(device)
        initial_embs = model.item_embedding(all_idx)
        # 注意：eval mode 下 dropout 不生效，所以不用特別處理

        raw_sim, _ = model.spectral_disentangler(initial_embs, adj_sim, adj_sim_dele)
        _, raw_cor = model.spectral_disentangler(initial_embs, adj_cor, adj_cor_dele)
        raw_sim_normed = model.layer_norm(raw_sim)
        raw_cor_normed = model.layer_norm(raw_cor)

        id_residual = model.id_emb(all_idx)
        x_sim = model.proj_sim(torch.cat([initial_embs, raw_sim_normed], dim=-1)) + id_residual
        x_cor = model.proj_cor(torch.cat([initial_embs, raw_cor_normed], dim=-1)) + id_residual

    return {
        'initial_embs': initial_embs.cpu().numpy(),
        'raw_sim': raw_sim.cpu().numpy(),
        'raw_cor': raw_cor.cpu().numpy(),
        'raw_sim_normed': raw_sim_normed.cpu().numpy(),
        'raw_cor_normed': raw_cor_normed.cpu().numpy(),
        'x_sim': x_sim.cpu().numpy(),
        'x_cor': x_cor.cpu().numpy(),
    }


def compute_metrics(embs):
    """計算 RQ1 核心指標"""
    print("\n" + "=" * 60)
    print("RQ1 指標分析")
    print("=" * 60)

    # 1. feat_sim_raw：LayerNorm 後的 raw_sim vs raw_cor
    raw_sim_t = torch.from_numpy(embs['raw_sim_normed'])
    raw_cor_t = torch.from_numpy(embs['raw_cor_normed'])
    feat_sim_raw = F.cosine_similarity(raw_sim_t, raw_cor_t, dim=-1)
    print(f"\nfeat_sim_raw (raw_sim vs raw_cor, after LayerNorm):")
    print(f"  Mean: {feat_sim_raw.mean():.6f}")
    print(f"  Std:  {feat_sim_raw.std():.6f}")
    print(f"  Min:  {feat_sim_raw.min():.6f}")
    print(f"  Max:  {feat_sim_raw.max():.6f}")

    # 2. feat_sim_proj：投影後的 x_sim vs x_cor
    x_sim_t = torch.from_numpy(embs['x_sim'])
    x_cor_t = torch.from_numpy(embs['x_cor'])
    feat_sim_proj = F.cosine_similarity(x_sim_t, x_cor_t, dim=-1)
    print(f"\nfeat_sim_proj (x_sim vs x_cor, after projection):")
    print(f"  Mean: {feat_sim_proj.mean():.6f}")
    print(f"  Std:  {feat_sim_proj.std():.6f}")
    print(f"  Min:  {feat_sim_proj.min():.6f}")
    print(f"  Max:  {feat_sim_proj.max():.6f}")

    # 3. 原始（未 LayerNorm）raw_sim vs raw_cor
    raw_sim_orig = torch.from_numpy(embs['raw_sim'])
    raw_cor_orig = torch.from_numpy(embs['raw_cor'])
    feat_sim_raw_orig = F.cosine_similarity(raw_sim_orig, raw_cor_orig, dim=-1)
    print(f"\nfeat_sim_raw_original (before LayerNorm):")
    print(f"  Mean: {feat_sim_raw_orig.mean():.6f}")
    print(f"  |raw_sim| mean norm: {raw_sim_orig.norm(dim=-1).mean():.4f}")
    print(f"  |raw_cor| mean norm: {raw_cor_orig.norm(dim=-1).mean():.4f}")
    print(f"  amplitude ratio: {raw_sim_orig.norm(dim=-1).mean() / raw_cor_orig.norm(dim=-1).mean():.2f}x")

    return {
        'feat_sim_raw': feat_sim_raw.numpy(),
        'feat_sim_proj': feat_sim_proj.numpy(),
    }


def plot_tsne_comparison(embs, features, output_dir, top_k_cid3=15, max_sample=5000):
    """t-SNE 視覺化：比較 raw_sim vs raw_cor 與 x_sim vs x_cor"""
    cid3_labels = features[:, 1].astype(int)
    cnt = Counter(cid3_labels)

    # 選擇 top_k 大類別
    top_cids = [cid for cid, _ in cnt.most_common(top_k_cid3)]
    mask = np.isin(cid3_labels, top_cids)
    indices = np.where(mask)[0]

    # 取樣
    if len(indices) > max_sample:
        indices = np.random.choice(indices, max_sample, replace=False)
    indices = np.sort(indices)
    labels = cid3_labels[indices]

    # 為每個類別分配顏色
    unique_labels = sorted(set(labels))
    cmap = cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {l: cmap(i) for i, l in enumerate(unique_labels)}
    colors = [label_to_color[l] for l in labels]

    # 繪製四個子圖：raw_sim, raw_cor, x_sim, x_cor
    emb_pairs = [
        ('raw_sim_normed', 'Raw Similarity Channel (Low-pass)'),
        ('raw_cor_normed', 'Raw Complementarity Channel (Mid-pass)'),
        ('x_sim', 'Projected Similarity Channel'),
        ('x_cor', 'Projected Complementarity Channel'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f't-SNE Visualization of Item Embeddings (Top-{top_k_cid3} CID3 Categories)',
                 fontsize=20, fontweight='bold')

    for ax, (key, title) in zip(axes.flatten(), emb_pairs):
        data = embs[key][indices]
        print(f"  Running t-SNE for {key} ({len(indices)} items)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        coords = tsne.fit_transform(data)

        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=3, alpha=0.6)
        ax.set_title(title, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    # 圖例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=label_to_color[l], markersize=6,
                                   label=f'CID3={l} ({cnt[l]})')
                       for l in unique_labels]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.12, 0.5),
               fontsize=12, ncol=1)

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    save_path = os.path.join(output_dir, 'tsne_4channel_comparison.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_sim_vs_cor_tsne(embs, features, output_dir, top_k_cid3=15, max_sample=3000):
    """
    單張 t-SNE：同時繪製 x_sim 和 x_cor，以顏色區分通道，以形狀區分類別
    展示兩個空間是否有效分離
    """
    cid3_labels = features[:, 1].astype(int)
    cnt = Counter(cid3_labels)
    top_cids = [cid for cid, _ in cnt.most_common(top_k_cid3)]
    mask = np.isin(cid3_labels, top_cids)
    indices = np.where(mask)[0]
    if len(indices) > max_sample:
        indices = np.random.choice(indices, max_sample, replace=False)
    indices = np.sort(indices)

    x_sim_sub = embs['x_sim'][indices]
    x_cor_sub = embs['x_cor'][indices]

    # 合併做 t-SNE
    combined = np.vstack([x_sim_sub, x_cor_sub])
    print(f"  Running combined t-SNE ({combined.shape[0]} points)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(combined)

    n = len(indices)
    coords_sim = coords[:n]
    coords_cor = coords[n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords_sim[:, 0], coords_sim[:, 1], c='#2196F3', s=3, alpha=0.4, label='x_sim (Similarity)')
    ax.scatter(coords_cor[:, 0], coords_cor[:, 1], c='#FF5722', s=3, alpha=0.4, label='x_cor (Complementarity)')
    ax.legend(fontsize=14, markerscale=5)
    ax.set_title('t-SNE: Similarity vs Complementarity Space Separation', fontsize=18, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    save_path = os.path.join(output_dir, 'tsne_sim_vs_cor_separation.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feat_sim_distribution(metrics, output_dir):
    """feat_sim 分布直方圖"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, key, title in [
        (axes[0], 'feat_sim_raw', 'feat_sim_raw (After Spectral Filtering)'),
        (axes[1], 'feat_sim_proj', 'feat_sim_proj (After Projection)')
    ]:
        data = metrics[key]
        ax.hist(data, bins=100, color='#42A5F5', edgecolor='white', alpha=0.8)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean={data.mean():.4f}')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('# Items')
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feat_sim_distribution.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curve(log_path, output_dir):
    """從訓練 log 解析 Feat_Sim 與 L_spec 隨 epoch 變化"""
    if log_path is None or not os.path.exists(log_path):
        print("  跳過訓練曲線（未提供 training_log）")
        return

    epochs, feat_sims, l_specs, l_seqs = [], [], [], []
    pattern = re.compile(
        r'Epoch (\d+).*L_seq: ([\d.]+).*L_spec: ([\d.]+).*Feat_Sim: ([-\d.]+)'
    )

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                l_seqs.append(float(m.group(2)))
                l_specs.append(float(m.group(3)))
                feat_sims.append(float(m.group(4)))

    if not epochs:
        print("  訓練 log 中未找到可解析的 Feat_Sim 記錄")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, feat_sims, color='#2196F3', linewidth=1)
    axes[0].set_title('Feat_Sim (x_sim ⊥ x_cor) over Training', fontsize=16)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].axhline(0, color='gray', linestyle=':', linewidth=0.5)

    axes[1].plot(epochs, l_specs, color='#FF5722', linewidth=1)
    axes[1].set_title('L_spec (Orthogonality Loss) over Training', fontsize=16)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')

    axes[2].plot(epochs, l_seqs, color='#4CAF50', linewidth=1)
    axes[2].set_title('L_seq (Recommendation Loss) over Training', fontsize=16)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def compute_per_category_analysis(embs, features, output_dir, top_k=20):
    """每個 CID3 類別內的通道分析"""
    cid3 = features[:, 1].astype(int)
    cnt = Counter(cid3)
    top_cids = [c for c, _ in cnt.most_common(top_k)]

    x_sim_t = torch.from_numpy(embs['x_sim'])
    x_cor_t = torch.from_numpy(embs['x_cor'])

    results = []
    for c in top_cids:
        mask = cid3 == c
        sim_sub = x_sim_t[mask]
        cor_sub = x_cor_t[mask]
        n = mask.sum()

        # 類別內 x_sim 的平均 pairwise cosine sim（取樣避免 O(n^2)）
        if n > 200:
            idx = np.random.choice(n, 200, replace=False)
            sim_sub_sample = sim_sub[idx]
            cor_sub_sample = cor_sub[idx]
        else:
            sim_sub_sample = sim_sub
            cor_sub_sample = cor_sub

        sim_normed = F.normalize(sim_sub_sample, dim=-1)
        cor_normed = F.normalize(cor_sub_sample, dim=-1)

        # 類別內聚集度（intra-class similarity）
        sim_intra = (sim_normed @ sim_normed.t()).fill_diagonal_(0).sum() / (len(sim_normed) * (len(sim_normed) - 1))
        cor_intra = (cor_normed @ cor_normed.t()).fill_diagonal_(0).sum() / (len(cor_normed) * (len(cor_normed) - 1))

        # 通道間正交度
        cross_sim = F.cosine_similarity(sim_sub, cor_sub, dim=-1).mean()

        results.append({
            'cid3': c, 'count': int(n),
            'sim_intra': sim_intra.item(),
            'cor_intra': cor_intra.item(),
            'cross_channel': cross_sim.item(),
        })

    print(f"\n{'CID3':>6} | {'Count':>6} | {'Sim Intra':>10} | {'Cor Intra':>10} | {'Cross':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['cid3']:>6} | {r['count']:>6} | {r['sim_intra']:>10.4f} | {r['cor_intra']:>10.4f} | {r['cross_channel']:>8.4f}")

    # 繪製柱狀圖
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(results))
    width = 0.3
    ax.bar(x - width, [r['sim_intra'] for r in results], width, label='Sim Intra-class', color='#2196F3')
    ax.bar(x, [r['cor_intra'] for r in results], width, label='Cor Intra-class', color='#FF5722')
    ax.bar(x + width, [r['cross_channel'] for r in results], width, label='Cross-channel', color='#9E9E9E')
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{r['cid3']}" for r in results], rotation=45, fontsize=12)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Per-Category Intra-class Similarity & Cross-channel Orthogonality', fontsize=18)
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'per_category_analysis.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    args = parse_args()

    # 設定輸出目錄
    if args.output_dir is None:
        args.output_dir = os.path.join('experiments', 'RQ1', 'output', args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # 1. 載入模型
    print("\n[Step 1] 載入模型與資料...")
    model, device, adj_sim, adj_sim_dele, adj_cor, adj_cor_dele, features, num_items = \
        load_model_and_data(args)

    # 2. 提取嵌入
    print("\n[Step 2] 提取嵌入向量...")
    embs = extract_embeddings(model, device, adj_sim, adj_sim_dele, adj_cor, adj_cor_dele)

    # 3. 計算指標
    print("\n[Step 3] 計算分離指標...")
    metrics = compute_metrics(embs)

    # 4. feat_sim 分布圖
    print("\n[Step 4] 繪製 feat_sim 分布圖...")
    plot_feat_sim_distribution(metrics, args.output_dir)

    # 5. t-SNE 視覺化
    print("\n[Step 5] t-SNE 視覺化（四通道比較）...")
    plot_tsne_comparison(embs, features, args.output_dir,
                         top_k_cid3=args.top_k_cid3, max_sample=args.tsne_sample)

    # 6. sim vs cor 分離 t-SNE
    print("\n[Step 6] t-SNE：Sim vs Cor 空間分離...")
    plot_sim_vs_cor_tsne(embs, features, args.output_dir,
                         top_k_cid3=args.top_k_cid3, max_sample=3000)

    # 7. 訓練曲線
    print("\n[Step 7] 繪製訓練曲線...")
    plot_training_curve(args.training_log, args.output_dir)

    # 8. 每類別分析
    print("\n[Step 8] 每類別通道分析...")
    compute_per_category_analysis(embs, features, args.output_dir)

    print(f"\n所有圖表已輸出至: {args.output_dir}")


if __name__ == '__main__':
    main()
