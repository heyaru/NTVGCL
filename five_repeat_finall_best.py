import pandas as pd
import glob
import os
import re


def aggregate_metrics(save_dir):
    # 1. 查找所有匹配 metrics-*.csv 的文件
    file_pattern = os.path.join(save_dir, "metrics-*.csv")
    files = glob.glob(file_pattern)

    if len(files) == 0:
        print(f"❌ 未在路径 {save_dir} 下找到任何 metrics-*.csv 文件")
        return

    print(f"🔍 找到 {len(files)} 个实验文件，正在汇总...")

    all_data = []

    for f in files:
        # 读取 CSV，跳过空行，提取前两列
        df = pd.read_csv(f, header=None, names=['Metric', 'Value'])
        # 只保留有数值的部分（过滤掉混淆矩阵的表头文字）
        df = df[pd.to_numeric(df['Value'], errors='coerce').notnull()]
        df['Value'] = df['Value'].astype(float)
        all_data.append(df)

    # 2. 合并并计算统计量
    combined = pd.concat(all_data)

    # 按指标名称分组计算
    stats = combined.groupby('Metric')['Value'].agg(['mean', 'std']).reset_index()

    # 3. 格式化输出 (均值 ± 标准差)
    stats['Combined'] = stats.apply(lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)

    # 按照常见的指标顺序排序（可选）
    metric_order = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'Latency(ms)']
    stats['Metric'] = pd.Categorical(stats['Metric'], categories=metric_order, ordered=True)
    stats = stats.sort_values('Metric')

    # 4. 保存为最终结果
    output_path = os.path.join(save_dir, "final_aggregated_metrics.csv")
    stats.to_csv(output_path, index=False)

    print("-" * 30)
    print(stats[['Metric', 'Combined']])
    print("-" * 30)
    print(f"✅ 汇总完成！最终结果已保存至: {output_path}")


if __name__ == "__main__":
    # 请填入你保存 metrics 文件的文件夹路径
    # 例如：target_dir = r"survival-Sonata_graphs_W40S10_results\AE-test_MIX_CRG-bs64_lr0.1_h16_p16"
    target_dir = "./"  # 当前目录下测试
    aggregate_metrics(target_dir)