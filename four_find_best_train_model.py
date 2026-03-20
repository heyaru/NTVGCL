import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# 1. 忽略特定的 FutureWarning 警告
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# ==================== 全局学术样式配置 ====================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# 配置路径
datasets_name = "Car-Hacking"
root = f"{datasets_name}_graphs_W40S10"
base_results_path = f"{root}_results_ALL"


def plot_slice_with_best_highlight_final(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 解析参数列
    if 'batch_size' not in df.columns:
        df['batch_size'] = df['save_tag'].apply(lambda x: int(re.search(r'bs(\d+)', x).group(1)))
        df['lr'] = df['save_tag'].apply(lambda x: float(re.search(r'lr([\d\.]+)', x).group(1)))
        df['hidden_dim'] = df['save_tag'].apply(lambda x: int(re.search(r'h(\d+)', x).group(1)))
        df['projection_dim'] = df['save_tag'].apply(lambda x: int(re.search(r'p(\d+)', x).group(1)))

    target_metric = 'AUC'
    params = ['batch_size', 'lr', 'hidden_dim', 'projection_dim']

    param_labels = {
        'batch_size': 'Batch Size, B',
        'lr': 'Learning Rate, L',
        'hidden_dim': 'Hidden Dimension, H',
        'projection_dim': 'Projection Dimension, P'
    }

    # 提取最优解并处理 Tag 中的缩写
    best_idx = df[target_metric].idxmax()
    best_row = df.loc[[best_idx]]
    # 保存最优结果 CSV
    output_dir = os.path.dirname(csv_path)
    best_row.to_csv(os.path.join(output_dir, "best_hyperparameter_config.csv"), index=False)

    raw_tag = str(best_row['save_tag'].values[0])
    best_config_name = raw_tag.replace('bs', 'B').replace('lr', 'L').replace('h', 'H').replace('p', 'P').upper()

    # ==================== 2. 绘图部分 ====================
    # 修改 figsize: 高度从 8 减小到 4 (实现高度压缩一半)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)

    for i, param in enumerate(params):
        # 背景散点
        sns.stripplot(x=param, y=target_metric, data=df, ax=axes[i],
                      color='skyblue', alpha=0.4, jitter=True, s=10, zorder=1)

        # 趋势线
        sns.pointplot(x=param, y=target_metric, data=df, ax=axes[i],
                      color='gray', capsize=.1, linestyles='--', markers='d',
                      scale=1.2, errwidth=2)

        # 最优高亮
        sns.stripplot(x=param, y=target_metric, data=best_row, ax=axes[i],
                      color='red', marker='*', s=12, linewidth=1.5,
                      edgecolor='black', zorder=10, label=best_config_name)

        # 调整刻度
        axes[i].tick_params(axis='both', which='major', labelsize=14, width=2, length=4)
        for spine in axes[i].spines.values():
            spine.set_linewidth(2)

        # 标题与标签
        display_name = param_labels[param]
        axes[i].set_title(f'Impact of {display_name[:1]} on {datasets_name}',
                          fontsize=15, fontweight='bold', pad=12)
        axes[i].set_xlabel(display_name, fontsize=14, fontweight='bold')
        axes[i].grid(axis='y', linestyle=':', alpha=0.6, linewidth=1)

        # 图例字号相应缩小
        legend_prop = {'size': 11, 'weight': 'bold'}
        axes[i].legend(loc='lower right', prop=legend_prop, frameon=True, edgecolor='black', framealpha=0.9)

        if i == 0:
            axes[i].set_ylabel(target_metric, fontsize=18, fontweight='bold')

    plt.tight_layout(w_pad=1.5)

    # 保存图片
    save_filename = f"Hyperparameter Sensitivity Analysis (Target - {target_metric}) on {datasets_name}.png"
    output_dir = os.path.dirname(csv_path)
    plot_path = os.path.join(output_dir, save_filename)

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 绘制完成！高度已压缩。\n📊 保存路径: {plot_path}")
    plt.show()


if __name__ == "__main__":
    input_file = os.path.join(base_results_path, "all_models_metrics_comparison.csv")
    plot_slice_with_best_highlight_final(input_file)
