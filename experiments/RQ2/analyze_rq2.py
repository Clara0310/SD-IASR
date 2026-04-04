"""
RQ2: 雙通道獨立投影的語意分工效果 (Dual-Channel Independent Projection)
=====================================================================
分析項目：
  1. proj_weight_sim：proj_sim vs proj_cor 權重矩陣的 cosine similarity
  2. 使用者分群：依「重複購買比例」分高/中/低三組
  3. 箱型圖：各組使用者在 x_sim vs x_cor 空間的聚集度比較
  4. 統計檢定：三組使用者的 intra-group cosine similarity 分布 + t-test
  5. 投影層權重視覺化：proj_sim vs proj_cor 的 weight heatmap

使用方式：
  cd SD-IASR
  python experiments/RQ2/analyze_rq2.py --dataset Grocery_and_Gourmet_Food \
      --checkpoint checkpoints/Grocery_and_Gourmet_Food/20260309_1945/best_model.pth
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--cooc_window', type=int, default=5)
    parser.add_argument('--decay_days', type=float, default=0.002)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_users', type=int, default=-1,
                        help='分析最大使用者數（-1 表示全部）')
    return parser.parse_args()


def load_model_and_data(args):
    """載入模型 checkpoint 與資料（與 RQ1 共用邏輯）"""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

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

    # 重建圖結構
    sim_edges = torch.tensor(raw_data['sim_edge_index']).t().to(device)
    com_edges = torch.tensor(raw_data['com_edge_index']).t().to(device)

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

    # 載入模型
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, 'config.yaml')
    import yaml
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {'embedding_dim': 128, 'bert_dim': 768, 'low_k': 5, 'mid_k': 5,
               'max_seq_len': 50, 'num_layers': 2, 'nhead': 8, 'dropout': 0.3,
               'gamma': 0.1, 'num_prototypes': 64}

    model = SDIASR(
        item_num=num_items, bert_dim=cfg['bert_dim'], emb_dim=cfg['embedding_dim'],
        low_k=cfg['low_k'], mid_k=cfg['mid_k'], max_seq_len=cfg['max_seq_len'],
        num_layers=cfg['num_layers'], nhead=cfg['nhead'], dropout=cfg['dropout'],
        gamma=cfg['gamma'], num_prototypes=cfg['num_prototypes']
    ).to(device)

    emb_path = f"./data_preprocess/embs/{args.dataset}_embeddings.npz"
    if os.path.exists(emb_path):
        emb_data = np.load(emb_path)
        model.load_pretrain_embedding(emb_data['cid2_emb'], emb_data['cid3_emb'],
                                      item_to_cid, item_to_price)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    return model, device, adj_sim, adj_sim_dele, adj_cor, adj_cor_dele, raw_data, features, num_items


# ============================================================
# 1. 投影層權重分析
# ============================================================
def analyze_projection_weights(model, output_dir):
    """比較 proj_sim 和 proj_cor 的權重矩陣"""
    print("\n" + "=" * 60)
    print("RQ2-1: 投影層權重分析 (proj_weight_sim)")
    print("=" * 60)

    layers = [(0, 'Linear1 (256→128)'), (3, 'Linear2 (128→128)')]
    results = []

    for idx, name in layers:
        w_sim = model.proj_sim[idx].weight.data.flatten()
        w_cor = model.proj_cor[idx].weight.data.flatten()
        cos_sim = F.cosine_similarity(w_sim.unsqueeze(0), w_cor.unsqueeze(0)).item()

        b_sim = model.proj_sim[idx].bias.data
        b_cor = model.proj_cor[idx].bias.data
        cos_sim_bias = F.cosine_similarity(b_sim.unsqueeze(0), b_cor.unsqueeze(0)).item()

        # L2 distance
        l2_dist = (w_sim - w_cor).norm().item()

        results.append({
            'layer': name, 'cos_sim_weight': cos_sim,
            'cos_sim_bias': cos_sim_bias, 'l2_dist': l2_dist
        })

        print(f"\n  {name}:")
        print(f"    Weight cosine similarity: {cos_sim:.6f}")
        print(f"    Bias cosine similarity:   {cos_sim_bias:.6f}")
        print(f"    Weight L2 distance:       {l2_dist:.4f}")

    # 權重矩陣視覺化
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(r'Projection Layer Weight Comparison: $proj_{sim}$ vs $proj_{cor}$', fontsize=20, fontweight='bold')

    for row, (idx, name) in enumerate(layers):
        w_sim = model.proj_sim[idx].weight.data.cpu().numpy()
        w_cor = model.proj_cor[idx].weight.data.cpu().numpy()
        diff = w_sim - w_cor

        vmax = max(abs(w_sim).max(), abs(w_cor).max())
        for ax, data, title in [
            (axes[row, 0], w_sim, f'$proj_{{sim}}$ {name}'),
            (axes[row, 1], w_cor, f'$proj_{{cor}}$ {name}'),
            (axes[row, 2], diff, f'Difference (sim - cor)'),
        ]:
            if 'Difference' in title:
                im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                               vmin=-vmax * 0.3, vmax=vmax * 0.3)
            else:
                im = ax.imshow(data, aspect='auto', cmap='viridis', vmin=-vmax, vmax=vmax)
            ax.set_title(title, fontsize=16)
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, 'proj_weight_comparison.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")

    return results


# ============================================================
# 2. 使用者分群
# ============================================================
def extract_unique_user_sequences(train_set):
    """
    train_set 為增強訓練集（每位使用者的所有前綴子序列）。
    每位使用者的條目以遞增序列長度排列，長度重置為 1 代表新使用者開始。
    回傳每位唯一使用者的完整訓練序列（最長前綴 + 目標物品 entry[2]）。
    格式：List of (full_seq: List[int], entry)
    """
    user_data = []
    last_entry = None
    for entry in train_set:
        seq = [x for x in list(entry[0]) if x != 0]
        if len(seq) == 1 and last_entry is not None:
            prefix = [x for x in list(last_entry[0]) if x != 0]
            full_seq = prefix + [int(last_entry[2])]
            user_data.append((full_seq, last_entry))
        last_entry = entry
    if last_entry is not None:
        prefix = [x for x in list(last_entry[0]) if x != 0]
        full_seq = prefix + [int(last_entry[2])]
        user_data.append((full_seq, last_entry))
    return user_data


def compute_user_repeat_ratio(raw_data, features):
    """計算每位使用者的重複購買比例（基於 CID3 類別）"""
    train_set = raw_data['train_set']
    user_data = extract_unique_user_sequences(train_set)
    print(f"  唯一使用者數: {len(user_data)}")
    user_repeat_ratios = []

    for full_seq, entry in user_data:
        seq = full_seq
        if len(seq) < 2:
            user_repeat_ratios.append(0.0)
            continue

        # 取得序列中每個商品的 CID3
        cid3_seq = [int(features[item_id, 1]) for item_id in seq]
        cid3_counter = Counter(cid3_seq)

        # 重複比例 = 出現 >1 次的類別佔比
        repeated = sum(1 for item_cid3 in cid3_seq if cid3_counter[item_cid3] > 1)
        ratio = repeated / len(cid3_seq)
        user_repeat_ratios.append(ratio)

    ratios = np.array(user_repeat_ratios)
    print(f"\n  使用者重複購買比例統計:")
    print(f"    Mean: {ratios.mean():.4f}, Median: {np.median(ratios):.4f}")
    print(f"    Min: {ratios.min():.4f}, Max: {ratios.max():.4f}")

    # 分三組：使用 33/67 百分位數
    p33, p67 = np.percentile(ratios, [33, 67])
    groups = np.zeros(len(ratios), dtype=int)
    groups[ratios <= p33] = 0  # low
    groups[(ratios > p33) & (ratios <= p67)] = 1  # mid
    groups[ratios > p67] = 2  # high

    print(f"    分群閾值: Low ≤ {p33:.4f}, Mid ≤ {p67:.4f}, High > {p67:.4f}")
    print(f"    Low: {(groups == 0).sum()}, Mid: {(groups == 1).sum()}, High: {(groups == 2).sum()}")

    return ratios, groups, p33, p67


# ============================================================
# 3. 使用者序列聚集度分析
# ============================================================
def analyze_user_clustering(model, device, adj_sim, adj_sim_dele, adj_cor, adj_cor_dele,
                            raw_data, features, groups, output_dir, max_users=10000):
    """各組使用者在 x_sim vs x_cor 空間的聚集度"""
    print("\n" + "=" * 60)
    print("RQ2-2: 使用者序列聚集度分析")
    print("=" * 60)

    # 提取所有商品的嵌入
    with torch.no_grad():
        x_sim, x_cor, _, _ = model.get_all_item_features(adj_sim, adj_sim_dele, adj_cor, adj_cor_dele)
        x_sim = x_sim.cpu()
        x_cor = x_cor.cpu()

    user_data = extract_unique_user_sequences(raw_data['train_set'])
    group_names = ['Low Repeat', 'Mid Repeat', 'High Repeat']

    # 對每個使用者計算其歷史序列在 x_sim / x_cor 空間的平均 pairwise cosine similarity
    sim_clustering = {0: [], 1: [], 2: []}
    cor_clustering = {0: [], 1: [], 2: []}

    # 取樣使用者
    np.random.seed(42)
    n_total = len(user_data)
    if max_users > 0 and n_total > max_users:
        sample_idx = np.random.choice(n_total, max_users, replace=False)
    else:
        sample_idx = np.arange(n_total)

    print(f"  分析 {len(sample_idx)} 位使用者...")

    for uid in sample_idx:
        seq, entry = user_data[uid]
        if len(seq) < 3:
            continue

        g = groups[uid]
        items = torch.LongTensor(seq)

        # 取出該使用者歷史商品的嵌入
        sim_embs = F.normalize(x_sim[items], dim=-1)
        cor_embs = F.normalize(x_cor[items], dim=-1)

        # 計算平均 pairwise cosine similarity
        n = len(seq)
        sim_pairwise = (sim_embs @ sim_embs.t()).fill_diagonal_(0).sum() / (n * (n - 1))
        cor_pairwise = (cor_embs @ cor_embs.t()).fill_diagonal_(0).sum() / (n * (n - 1))

        sim_clustering[g].append(sim_pairwise.item())
        cor_clustering[g].append(cor_pairwise.item())

    # 列印統計
    for g in range(3):
        sim_arr = np.array(sim_clustering[g])
        cor_arr = np.array(cor_clustering[g])
        print(f"\n  {group_names[g]} ({len(sim_arr)} users):")
        print(f"    x_sim clustering: mean={sim_arr.mean():.4f}, std={sim_arr.std():.4f}")
        print(f"    x_cor clustering: mean={cor_arr.mean():.4f}, std={cor_arr.std():.4f}")
        print(f"    差值 (sim - cor):  {(sim_arr - cor_arr).mean():.4f}")

    # ---- 視覺化 1：箱型圖 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(r'User Sequence Clustering: $\mathbf{x_{sim}}$ vs $\mathbf{x_{cor}}$ by Repeat Purchase Group',
                 fontsize=20, fontweight='bold')

    for g, ax in enumerate(axes):
        data = [sim_clustering[g], cor_clustering[g]]
        bp = ax.boxplot(data, labels=['$x_{sim}$', '$x_{cor}$'], patch_artist=True,
                        widths=0.5, showfliers=False)
        bp['boxes'][0].set_facecolor('#2196F3')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#FF5722')
        bp['boxes'][1].set_alpha(0.7)

        # 中位數線改紅色
        for median_line in bp['medians']:
            median_line.set_color('red')
            median_line.set_linewidth(2)

        # 標注中位數、Q1、Q3 數字
        for i, arr in enumerate(data):
            x_pos = i + 1
            q1 = np.percentile(arr, 25)
            median = np.percentile(arr, 50)
            q3 = np.percentile(arr, 75)
            ax.text(x_pos + 0.27, median, f'{median:.3f}', va='center', ha='left',
                    fontsize=11, color='red', fontweight='bold')
            ax.text(x_pos + 0.27, q1, f'{q1:.3f}', va='center', ha='left',
                    fontsize=10, color='#444444')
            ax.text(x_pos + 0.27, q3, f'{q3:.3f}', va='center', ha='left',
                    fontsize=10, color='#444444')

        ax.set_title(f'{group_names[g]}', fontsize=16)
        ax.set_ylabel('Intra-sequence Cosine Similarity')
        ax.tick_params(labelleft=True)

        # 加上 t-test 結果
        t_stat, p_val = stats.ttest_ind(sim_clustering[g], cor_clustering[g])
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.text(0.5, 0.95, f'p={p_val:.2e} {sig}', transform=ax.transAxes,
                ha='center', va='top', fontsize=13, style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(output_dir, 'user_clustering_boxplot.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")

    # ---- 視覺化 2：分布對比 + 統計檢定 ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Intra-sequence Similarity Distribution by User Group',
                 fontsize=20, fontweight='bold')

    for g in range(3):
        # x_sim 分布
        sim_mean = np.mean(sim_clustering[g])
        cor_mean = np.mean(cor_clustering[g])
        axes[0, g].hist(sim_clustering[g], bins=50, alpha=0.7, color='#2196F3',
                        edgecolor='white', label='$x_{sim}$')
        axes[0, g].hist(cor_clustering[g], bins=50, alpha=0.7, color='#FF5722',
                        edgecolor='white', label='$x_{cor}$')
        axes[0, g].axvline(sim_mean, color='#1565C0', linestyle='--', linewidth=1.5,
                           label=f'$\\bar{{x}}_{{sim}}$={sim_mean:.3f}')
        axes[0, g].axvline(cor_mean, color='#BF360C', linestyle='--', linewidth=1.5,
                           label=f'$\\bar{{x}}_{{cor}}$={cor_mean:.3f}')
        axes[0, g].set_title(f'{group_names[g]}', fontsize=16)
        axes[0, g].set_xlabel('Intra-sequence Cosine Similarity', fontsize=11)
        axes[0, g].legend(fontsize=11)
        axes[0, g].set_ylabel('# Users', fontsize=11)

        # 差值分布 (sim - cor)
        diff = np.array(sim_clustering[g]) - np.array(cor_clustering[g])
        axes[1, g].hist(diff, bins=50, color='#9C27B0', edgecolor='white', alpha=0.7)
        axes[1, g].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[1, g].axvline(diff.mean(), color='red', linestyle='-', linewidth=1.5,
                           label=f'Mean={diff.mean():.4f}')
        axes[1, g].set_title(f'Difference (sim - cor)', fontsize=16)
        axes[1, g].set_xlabel('$x_{sim}$ − $x_{cor}$ Cosine Similarity Difference', fontsize=11)
        axes[1, g].legend(fontsize=13)
        axes[1, g].set_ylabel('# Users', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(output_dir, 'user_clustering_distribution.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return sim_clustering, cor_clustering


# ============================================================
# 4. 跨組統計檢定
# ============================================================
def statistical_tests(sim_clustering, cor_clustering, output_dir):
    """跨組統計顯著性分析"""
    print("\n" + "=" * 60)
    print("RQ2-3: 統計顯著性檢定")
    print("=" * 60)

    group_names = ['Low', 'Mid', 'High']

    # 組內：x_sim vs x_cor 的 paired t-test
    print("\n  [組內] x_sim vs x_cor (Independent t-test):")
    for g in range(3):
        t_stat, p_val = stats.ttest_ind(sim_clustering[g], cor_clustering[g])
        print(f"    {group_names[g]}: t={t_stat:.4f}, p={p_val:.2e}")

    # 跨組：High 的 sim_clustering 是否 > Low 的 sim_clustering
    print("\n  [跨組] x_sim 聚集度 (High vs Low):")
    t_stat, p_val = stats.ttest_ind(sim_clustering[2], sim_clustering[0])
    print(f"    t={t_stat:.4f}, p={p_val:.2e}")
    print(f"    High mean={np.mean(sim_clustering[2]):.4f}, Low mean={np.mean(sim_clustering[0]):.4f}")

    print("\n  [跨組] x_cor 聚集度 (High vs Low):")
    t_stat, p_val = stats.ttest_ind(cor_clustering[2], cor_clustering[0])
    print(f"    t={t_stat:.4f}, p={p_val:.2e}")
    print(f"    High mean={np.mean(cor_clustering[2]):.4f}, Low mean={np.mean(cor_clustering[0]):.4f}")

    # Effect size (Cohen's d)
    def cohens_d(a, b):
        na, nb = len(a), len(b)
        pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
        return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0

    print("\n  [Effect Size] Cohen's d:")
    for g in range(3):
        d = cohens_d(sim_clustering[g], cor_clustering[g])
        print(f"    {group_names[g]} (sim vs cor): d={d:.4f}")


# ============================================================
# 5. 重複購買比例分布圖
# ============================================================
def plot_repeat_ratio_distribution(ratios, groups, p33, p67, output_dir):
    """使用者重複購買比例分布"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ['#4CAF50', '#FFC107', '#F44336']
    group_names = ['Low', 'Mid', 'High']

    for g in range(3):
        mask = groups == g
        ax.hist(ratios[mask], bins=40, alpha=0.6, color=colors[g],
                label=f'{group_names[g]} (n={mask.sum()})', edgecolor='white')

    ax.axvline(p33, color='gray', linestyle='--', linewidth=1, label=f'P33={p33:.3f}')
    ax.axvline(p67, color='gray', linestyle=':', linewidth=1, label=f'P67={p67:.3f}')
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('Repeat Purchase Ratio (by CID3)')
    ax.set_ylabel('# Users')
    ax.set_title('User Repeat Purchase Ratio Distribution', fontsize=18, fontweight='bold')
    ax.legend(fontsize=13)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'repeat_ratio_distribution.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 6. 綜合對比柱狀圖
# ============================================================
def plot_summary_bar(sim_clustering, cor_clustering, output_dir):
    """三組使用者在兩個空間的平均聚集度對比"""
    group_names = ['Low Repeat', 'Mid Repeat', 'High Repeat']
    sim_means = [np.mean(sim_clustering[g]) for g in range(3)]
    cor_means = [np.mean(cor_clustering[g]) for g in range(3)]
    sim_stds = [np.std(sim_clustering[g]) for g in range(3)]
    cor_stds = [np.std(cor_clustering[g]) for g in range(3)]

    x = np.arange(3)
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, sim_means, width, yerr=sim_stds, label='$x_{sim}$',
           color='#2196F3', alpha=0.8, capsize=3)
    ax.bar(x + width / 2, cor_means, width, yerr=cor_stds, label='$x_{cor}$',
           color='#FF5722', alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_ylabel('Mean Intra-sequence Cosine Similarity')
    ax.set_title(r'Semantic Division of Labor: $x_{sim}$ vs $x_{cor}$ by User Group',
                 fontsize=18, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'summary_bar_chart.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join('experiments', 'RQ2', 'output', args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # 1. 載入模型
    print("\n[Step 1] 載入模型與資料...")
    model, device, adj_sim, adj_sim_dele, adj_cor, adj_cor_dele, raw_data, features, num_items = \
        load_model_and_data(args)

    # 2. 投影層權重分析
    print("\n[Step 2] 投影層權重分析...")
    weight_results = analyze_projection_weights(model, args.output_dir)

    # 3. 使用者分群
    print("\n[Step 3] 使用者分群...")
    ratios, groups, p33, p67 = compute_user_repeat_ratio(raw_data, features)

    # 4. 重複購買比例分布圖
    print("\n[Step 4] 繪製重複購買比例分布圖...")
    plot_repeat_ratio_distribution(ratios, groups, p33, p67, args.output_dir)

    # 5. 使用者聚集度分析
    print("\n[Step 5] 使用者序列聚集度分析...")
    sim_clustering, cor_clustering = analyze_user_clustering(
        model, device, adj_sim, adj_sim_dele, adj_cor, adj_cor_dele,
        raw_data, features, groups, args.output_dir, max_users=args.max_users
    )

    # 6. 統計檢定
    print("\n[Step 6] 統計檢定...")
    statistical_tests(sim_clustering, cor_clustering, args.output_dir)

    # 7. 綜合對比柱狀圖
    print("\n[Step 7] 綜合對比圖...")
    plot_summary_bar(sim_clustering, cor_clustering, args.output_dir)

    print(f"\n所有圖表已輸出至: {args.output_dir}")
    print("\n⚠️  SD-IASR-shareProj 對照實驗需要重新訓練模型（共用 proj 參數），無法從現有 checkpoint 分析。")


if __name__ == '__main__':
    main()
