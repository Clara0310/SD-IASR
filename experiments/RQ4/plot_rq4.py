import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# ── 資料 ──────────────────────────────────────────────────────────────
models = ['SD-IASR\n(Full)', 'w/o\nOrtho', 'w/o\nSpec', 'Share\nProj', 'w/o\nDual', 'Single\nGraph']
model_labels_table = ['SD-IASR (Full)', 'w/o Ortho', 'w/o Spec', 'Share Proj', 'w/o Dual', 'Single Graph']

metrics = ['HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20']

data = {
    'SD-IASR (Full)': [0.1016, 0.1247, 0.1547, 0.0766, 0.0845, 0.0914],
    'w/o Ortho':      [0.0956, 0.1156, 0.1412, 0.0669, 0.0733, 0.0797],
    'w/o Spec':       [0.0903, 0.1064, 0.1267, 0.0701, 0.0752, 0.0803],
    'Share Proj':     [0.0906, 0.1087, 0.1303, 0.0655, 0.0714, 0.0768],
    'w/o Dual':       [0.0985, 0.1164, 0.1401, 0.0746, 0.0803, 0.0863],
    'Single Graph':   [0.0937, 0.1135, 0.1382, 0.0688, 0.0752, 0.0814],
}

colors = ['#2C6FAC', '#E87D4C', '#5BAD72', '#A569BD', '#E8B84B', '#7FB3D3']

n_models = len(models)
n_metrics = len(metrics)
x = np.arange(n_metrics)
total_width = 0.75
bar_width = total_width / n_models

# ── 版面 ──────────────────────────────────────────────────────────────
# 左邊留 0.22 給 Variant 名稱欄，右邊到 0.97
fig = plt.figure(figsize=(13, 9))
ax = fig.add_axes([0.22, 0.38, 0.75, 0.55])
ax_table = fig.add_axes([0.22, 0.00, 0.75, 0.38])

# ── 長條圖 ────────────────────────────────────────────────────────────
for i, (label, vals) in enumerate(data.items()):
    offset = (i - n_models / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, vals, bar_width * 0.9,
                  color=colors[i], label=label,
                  edgecolor='white', linewidth=0.5,
                  zorder=3)
    # SD-IASR (Full) 加邊框強調
    if i == 0:
        for bar in bars:
            bar.set_edgecolor('#1A3F6F')
            bar.set_linewidth(1.2)

ax.set_xlim(-0.5, n_metrics - 0.5)
ax.set_ylim(0, 0.185)
ax.set_xticks([])
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Ablation Study on Grocery_and_Gourmet_Food', fontsize=15, fontweight='bold', pad=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 圖例
legend_patches = [mpatches.Patch(color=colors[i], label=model_labels_table[i]) for i in range(n_models)]
ax.legend(handles=legend_patches, ncol=3, loc='upper right',
          framealpha=0.9, fontsize=11, handlelength=1.2)

# ── 下方表格 ──────────────────────────────────────────────────────────
ax_table.set_xlim(-0.5, n_metrics - 0.5)
ax_table.set_ylim(0, 1)
ax_table.axis('off')

n_rows = n_models + 1  # header + models
row_h = 1.0 / n_rows
col_positions = x  # 與 bar chart 對齊

# 名稱欄用 figure coordinates（ax_table 左邊界 = 0.22）
# fig_table_bottom / top 對應 ax_table 在 figure 中的位置
fig_left = 0.22   # ax_table 左邊界（figure coords）
fig_bottom = 0.00
fig_top = 0.38
name_fig_x_sq = 0.01    # ■ 的 figure x
name_fig_x_txt = 0.035  # 文字的 figure x

def row_fig_y(row_idx_center):
    """把 ax_table 的 axes-y 轉成 figure-y"""
    axes_y = 1.0 - row_h * row_idx_center
    return fig_bottom + axes_y * (fig_top - fig_bottom)

# header row — 名稱
fig.text(name_fig_x_txt, row_fig_y(0.5), 'Variant',
         ha='left', va='center', fontsize=12, fontweight='bold', color='black')

# header row — metrics（用 ax_table data coords）
header_y = 1.0 - row_h * 0.5
for j, m in enumerate(metrics):
    ax_table.text(col_positions[j], header_y, m, ha='center', va='center',
                  fontsize=12, fontweight='bold', color='black')

# header 下方分隔線（橫跨全寬，用 figure coords）
line_y_fig = row_fig_y(1.0)
fig.add_artist(plt.Line2D([0.01, 0.97], [line_y_fig, line_y_fig],
                           color='black', linewidth=1.0,
                           transform=fig.transFigure, clip_on=False))

# 各 model row
for i, (label, vals) in enumerate(data.items()):
    row_y_ax = 1.0 - row_h * (i + 1.5)   # axes coords
    row_y_fig = row_fig_y(i + 1.5)         # figure coords
    is_full = (i == 0)

    # ■（figure coords）
    fig.text(name_fig_x_sq, row_y_fig, '■',
             ha='left', va='center', fontsize=13, color=colors[i])
    # 名稱（figure coords）
    fig.text(name_fig_x_txt, row_y_fig, model_labels_table[i],
             ha='left', va='center', fontsize=12, color='black',
             fontweight='bold' if is_full else 'normal')

    # 數值（ax_table data coords）
    for j, v in enumerate(vals):
        ax_table.text(col_positions[j], row_y_ax, f'{v:.4f}',
                      ha='center', va='center', fontsize=11,
                      fontweight='bold' if is_full else 'normal',
                      color='black')

    # 行分隔線
    if i < n_models - 1:
        sep_y_fig = row_fig_y(i + 2)
        fig.add_artist(plt.Line2D([0.01, 0.97], [sep_y_fig, sep_y_fig],
                                   color='#CCCCCC', linewidth=0.5,
                                   transform=fig.transFigure, clip_on=False))

# 底部線
bot_y_fig = row_fig_y(n_rows)
fig.add_artist(plt.Line2D([0.01, 0.97], [bot_y_fig, bot_y_fig],
                           color='black', linewidth=1.0,
                           transform=fig.transFigure, clip_on=False))

# ── 輸出 ──────────────────────────────────────────────────────────────
out_dir = 'experiments/RQ4/output'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'rq4_ablation.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
