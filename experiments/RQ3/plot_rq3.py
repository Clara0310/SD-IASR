"""
RQ3: 整體效能比較 - grouped bar chart（仿學姊風格）
=====================================================
輸出三張圖，各對應一個資料集：
  Grocery_and_Gourmet_Food, Sports_and_Outdoors, Pet_Supplies

使用方式：
  cd SD-IASR
  python experiments/RQ3/plot_rq3.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 全域字體放大
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})

# ============================================================
# 實驗數據（來自表 4-8）
# ============================================================
METRICS = ['HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20']

DATA = {
    'Grocery_and_Gourmet_Food': {
        'SASRec':  [0.0644, 0.0877, 0.1053, 0.0471, 0.0547, 0.0591],
        'BARec':   [0.0630, 0.0790, 0.0996, 0.0477, 0.0528, 0.0580],
        'STIRec':  [0.1073, 0.1226, 0.1424, 0.0895, 0.0943, 0.0994],
        'SD-IASR': [0.1086, 0.1247, 0.1457, 0.0902, 0.0954, 0.1007],
    },
    'Sports_and_Outdoors': {
        'SASRec':  [0.0195, 0.0253, 0.0337, 0.0152, 0.0171, 0.0192],
        'BARec':   [0.0231, 0.0369, 0.0572, 0.0151, 0.0196, 0.0246],
        'STIRec':  [0.0424, 0.0520, 0.0661, 0.0352, 0.0383, 0.0417],
        'SD-IASR': [0.0436, 0.0564, 0.0741, 0.0354, 0.0395, 0.0439],
    },
    'Pet_Supplies': {
        'SASRec':  [0.0688, 0.0826, 0.1006, 0.0596, 0.0640, 0.0686],
        'BARec':   [0.0499, 0.0635, 0.0835, 0.0297, 0.0341, 0.0391],
        'STIRec':  [0.1054, 0.1233, 0.1479, 0.0926, 0.0984, 0.1045],
        'SD-IASR': [0.1060, 0.1247, 0.1531, 0.0929, 0.0989, 0.1060],
    },
}

DATASET_TITLES = {
    'Grocery_and_Gourmet_Food': 'Grocery & Gourmet Food',
    'Sports_and_Outdoors':      'Sports & Outdoors',
    'Pet_Supplies':             'Pet Supplies',
}

# 顏色（仿學姊深淺藍風格，四個模型）
COLORS = {
    'SASRec':  '#4472C4',   # 深藍
    'BARec':   '#70AD47',   # 綠
    'STIRec':  '#ED7D31',   # 橘
    'SD-IASR': '#FF0000',   # 紅（最重要，最突出）
}

MODELS = ['SASRec', 'BARec', 'STIRec', 'SD-IASR']


def plot_dataset(dataset, output_dir):
    data = DATA[dataset]
    title = DATASET_TITLES[dataset]

    n_metrics = len(METRICS)
    n_models = len(MODELS)
    x = np.arange(n_metrics)
    width = 0.18
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    # 上方 bar chart + 下方 table
    fig, (ax, ax_table) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    for i, model in enumerate(MODELS):
        values = data[model]
        has_data = any(v is not None for v in values)
        plot_vals = [v if v is not None else 0 for v in values]

        if has_data:
            bars = ax.bar(x + offsets[i], plot_vals, width * 0.92,
                          label=model, color=COLORS[model], alpha=0.85,
                          edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, values):
                if val is not None and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.001,
                            f'{val:.4f}', ha='center', va='bottom',
                            fontsize=7.5, rotation=90)
        else:
            ax.bar(x + offsets[i], plot_vals, width * 0.92,
                   label=f'{model} (N/A)', color=COLORS[model], alpha=0.3,
                   edgecolor=COLORS[model], linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylabel('Score')
    ax.set_title(f'{title}', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # ---- 下方數據表格（手動繪製，對齊上方 bar 的 x 位置）----
    # ax_table 用跟 ax 完全相同的 xlim，確保欄位對齊
    ax_table.set_xlim(ax.get_xlim())
    ax_table.set_ylim(0, len(MODELS) + 1)
    ax_table.axis('off')

    n_rows = len(MODELS)
    x_left = ax.get_xlim()[0]
    x_right = ax.get_xlim()[1]
    name_x = x_left - 0.7   # Model 名稱放在 axes 左邊外側

    # 表頭背景（移除，全白）

    # 計算 Improv.（SD-IASR vs STIRec）
    sdiasr_vals = data['SD-IASR']
    stirec_vals = data['STIRec']
    improv_vals = [(s - t) / t * 100 if t else None
                   for s, t in zip(sdiasr_vals, stirec_vals)]

    # 表格總列數 = MODELS + 1 列 Improv.
    total_rows = n_rows + 1
    ax_table.set_ylim(0, total_rows + 1)

    # 重新畫分隔線（多一列）
    for row_y in range(total_rows + 2):
        ax_table.plot([name_x, x_right], [row_y, row_y],
                      color='#AAAAAA', linewidth=0.6, clip_on=False)

    # 表頭：指標名稱對齊 bar 的 x 位置
    for j, metric in enumerate(METRICS):
        ax_table.text(x[j], total_rows + 0.5, metric,
                      ha='center', va='center', fontsize=11, fontweight='bold')

    # 各 model row
    for i, model in enumerate(MODELS):
        color = COLORS[model]
        row_y = total_rows - 1 - i + 0.5

        ax_table.text(name_x, row_y, f'\u25A0  {model}',
                      ha='left', va='center', fontsize=11,
                      fontweight='bold', color=color, clip_on=False)

        values = data[model]
        is_sdiasr = (model == 'SD-IASR')
        is_stirec = (model == 'STIRec')
        for j, val in enumerate(values):
            text = f'{val:.4f}' if val is not None else '—'
            t = ax_table.text(x[j], row_y, text,
                              ha='center', va='center', fontsize=11, color='black',
                              fontweight='bold' if is_sdiasr else 'normal')
            if is_stirec:
                t.set_text(text)
                t.set_usetex(False)
                # 底線用 annotate 實現
                ax_table.annotate('', xy=(x[j] + 0.18, row_y - 0.38),
                                  xytext=(x[j] - 0.18, row_y - 0.38),
                                  arrowprops=dict(arrowstyle='-', color='black', lw=0.8))

    # Improv. 列（最底列）
    improv_y = 0.5
    ax_table.text(name_x, improv_y, 'Improv. (%)',
                  ha='left', va='center', fontsize=11,
                  fontweight='bold', color='black', clip_on=False)
    for j, val in enumerate(improv_vals):
        text = f'+{val:.2f}%' if val is not None else '—'
        ax_table.text(x[j], improv_y, text,
                      ha='center', va='center', fontsize=11,
                      color='black', fontweight='bold')

    plt.tight_layout(h_pad=0.5)
    fname = dataset.replace(' ', '_') + '_performance.png'
    save_path = os.path.join(output_dir, fname)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    output_dir = os.path.join('experiments', 'RQ3', 'output')
    os.makedirs(output_dir, exist_ok=True)

    for dataset in DATA:
        plot_dataset(dataset, output_dir)

    print("\n所有圖表已輸出至: experiments/RQ3/output/")


if __name__ == '__main__':
    main()
