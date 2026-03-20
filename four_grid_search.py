import os
import pandas as pd
import re
from tqdm import tqdm

# ==================== 配置 ====================
datasets_name = "Car-Hacking"
W, S = 40, 10
root = f"{datasets_name}_graphs_W40S10"
base_results_path = f"{root}_results_ALL"
# 匹配训练文件夹的前缀，例如 train_100000_CRG_bs64_lr0.01...
folder_prefix = "train_CRG_"


def run_batch_evaluation():
    # 1. 搜集所有包含 save_tag 的训练文件夹
    all_folders = [f for f in os.listdir(base_results_path) if f.startswith(folder_prefix)]
    print(f"🚀 找到 {len(all_folders)} 个模型等待评估...")

    summary_list = []

    for folder_name in tqdm(all_folders, desc="Overall Progress"):
        # 从文件夹名中提取 save_tag
        # 例如从 "train_100000_CRG_bs64_lr0.01" 提取 "bs64_lr0.01"
        save_tag = folder_name.replace(folder_prefix, "")

        print(f"\n🔍 Testing Model: {save_tag}")

        # 2. 调用你的推理脚本
        # 使用 os.system 或 subprocess 运行，并传入当前的 save_tag
        cmd = f"python four_infer_simsiam_unsuper-Offset=0.py --save_tag {save_tag}"
        exit_code = os.system(cmd)

        if exit_code == 0:
            # 3. 读取刚刚生成的 metrics.csv
            # 这里的路径要对应你推理脚本里保存 metrics 的位置
            # 推理脚本中 save_dir = fr"{root}_results\AE-test_MIX_per50000_CRG-{save_tag}"
            metrics_path = os.path.join(base_results_path, f"AE-test_MIX_CRG-{save_tag}", "metrics.csv")

            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                # 将 Metric 列转置为行，方便合并
                # 原始格式: [["AUC", 0.99], ["F1", 0.93]] -> 目标格式: {"AUC": 0.99, "F1": 0.93}
                metrics_dict = dict(zip(df['Metric'], df['Value']))
                metrics_dict['save_tag'] = save_tag

                summary_list.append(metrics_dict)
            else:
                print(f"⚠️ Warning: {metrics_path} not found.")
        else:
            print(f"❌ Error occurred while testing {save_tag}")

    # 4. 合并所有结果并保存
    if summary_list:
        final_df = pd.DataFrame(summary_list)
        # 调整列顺序，让 save_tag 在最前面
        cols = ['save_tag'] + [c for c in final_df.columns if c != 'save_tag']
        final_df = final_df[cols]

        output_file = fr"{base_results_path}\all_models_metrics_comparison.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\n✨ 所有模型测试完成！结果已汇总至: {output_file}")
    else:
        print("😭 未能搜集到任何有效指标。")


if __name__ == "__main__":
    run_batch_evaluation()
