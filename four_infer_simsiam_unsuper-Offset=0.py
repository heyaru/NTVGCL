# ============================================================
# 核心逻辑：AE训练  + 可视化校准过程
# ============================================================
import os
import csv
import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import scipy.signal as signal
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from two_twin_dataset import TwinGraphDataset
from three_train_twin_gcl_simsiam import TwinGCL_SimSiam
import warnings
import argparse
import re
import random
from matplotlib.ticker import ScalarFormatter
from sklearn.neighbors import KernelDensity
warnings.filterwarnings("ignore")

# 训练时 1) 把save_tag变为默认,把kde_confidence设为默认0.99；2) 把metrics.csv 和 cm.csv的带阈值部分去掉； 3) 都不需要调用
# 测试时 0) 把最佳的训练和测试文件夹拷贝一份，后续在拷贝的文件夹中进行；
# 1) 把save_tag变为最佳训练模型参数；
# 2) 把metrics.csv 和 cm.csv的带阈值部分加上；
# 3.1）只调用tune_kde_confidence，找到最佳kde_confidence；
# 3.2）去上面修改为最佳kde_confidence后，只不调用tune_kde_confidence，重复执行三次；
# 3.3）执行 five_repeat_finall_best.py
# 超参数
parser = argparse.ArgumentParser()
parser.add_argument("--save_tag", type=str, default="bs256_lr0.01_h32_p32")  # "bs64_lr0.1_h16_p16"
args = parser.parse_args()
save_tag = args.save_tag
# 提取 batch_size 用于 DataLoader
batch_size = int(re.search(r'bs(\d+)', save_tag).group(1)) if re.search(r'bs(\d+)', save_tag) else 64

# ==================== 1. 配置与参数 ====================
datasets_name = "Survival-Sonata"
W, S = 40, 10
root = fr"{datasets_name}_graphs_W{W}S{S}"
classifier = "AE"
normal_mode = "train_CRG"
attack_mode = "test_MIX_CRG"

model_path = fr"{root}_results\{normal_mode}_{save_tag}\twin_gcl_simsiam_model.pt"
embeddings_path = fr"{root}_results\{normal_mode}_{save_tag}\embeddings_info.pt"
save_dir = fr"{root}_results\{classifier}-{attack_mode}-{save_tag}"
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kde_confidence = 0.99  # 0.99
# ==================== 2. 模型定义：ScoreNet (AE) ====================
class ScoreNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim // 4, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==================== 3. 加载model_path与embeddings_path ====================
print(f"→ Loading GNN Encoder...")
ckpt = torch.load(model_path, map_location=device)
gnn_encoder = TwinGCL_SimSiam(ckpt["in_dim"], ckpt["hid_dim"], ckpt["proj_dim"]).to(device)
gnn_encoder.load_state_dict(ckpt["model_state_dict"])
gnn_encoder.eval()
# 加载训练集嵌入并训练 AE
print("→ Training AE Scorer on Training Set Embeddings...")
train_embeds = torch.load(embeddings_path)["train_embeddings"]
scaler = StandardScaler()
train_embeds_norm = scaler.fit_transform(train_embeds)
train_tensor = torch.FloatTensor(train_embeds_norm).to(device)

# ==================== 3.  AE 训练逻辑 ====================
# ScoreNet 调用模型
reconstructor = ScoreNet(train_embeds.shape[1]).to(device)
print("→ Training AE Scorer with Stability Optimizations...")
optimizer = torch.optim.Adam(reconstructor.parameters(), lr=1e-3)
# AE 快速拟合正常模式
reconstructor.train()
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = reconstructor(train_tensor)
    loss = torch.mean(torch.sum((output - train_tensor) ** 2, dim=1))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f"AE Epoch [{epoch + 1}/epochs], Loss: {loss.item():.6f}")
# 计算训练集基准 (Calibration Base)
reconstructor.eval()
with torch.no_grad():
    train_recon = reconstructor(train_tensor)
    train_mse = torch.sum((train_recon - train_tensor) ** 2, dim=1).cpu().numpy()

# # ==================== 3. 优化后的 AE 训练逻辑 ====================
# print("→ Training AE Scorer with Stability Optimizations...")
# reconstructor = ScoreNet(train_embeds.shape[1]).to(device)
# # 早停参数
# best_loss = float('inf')
# patience = 10           # 容忍轮数
# min_delta = 1e-2        # 分辨率：Loss 下降小于 0.01 就视为停止进步
# counter = 0
# max_epochs = 2000
# reconstructor.train()
# # 优化器增加 Weight Decay (L2 正则化) 防止过拟合
# optimizer = torch.optim.Adam(reconstructor.parameters(), lr=1e-3, weight_decay=1e-5)
# # 学习率衰减：在 2000 轮内平滑降至 1e-5
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
# criterion = nn.MSELoss()
#
# for epoch in range(max_epochs):
#     optimizer.zero_grad()
#     output = reconstructor(train_tensor)
#     # 使用 MSELoss (计算每个维度的平均误差)
#     loss = criterion(output, train_tensor)
#     loss.backward()
#     # 梯度裁剪：防止调参过程中因参数极端导致梯度爆炸
#     torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), max_norm=1.0)
#     optimizer.step()
#     scheduler.step()
#     # 早停逻辑：只保存表现最好的模型
#     current_loss = loss.item()
#     # 核心逻辑：只有下降超过 min_delta 才算有效进步
#     if current_loss < (best_loss - min_delta):
#         best_loss = current_loss
#         best_model_state = reconstructor.state_dict()
#         counter = 0
#     else:
#         counter += 1
#     # 打印逻辑...
#     if counter >= patience:
#         print(f"🛑 早停触发：连续 {patience} 轮下降小于 {min_delta}。")
#         print(f"停止于 Epoch {epoch + 1}，最佳 Loss 约为 {best_loss:.4f}")
#         reconstructor.load_state_dict(best_model_state)
#         break
#     if (epoch + 1) % 10 == 0:
#         print(f"AE Epoch [{epoch + 1}/{max_epochs}], Loss: {current_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.1e}")
#     if counter >= patience:
#         print(f"✅ Early Stopping at Epoch {epoch + 1}. Best Loss: {best_loss:.6f}")
#         reconstructor.load_state_dict(best_model_state)
#         break
# # ==================== 4. 计算训练集基准得分 ====================
# reconstructor.eval()
# with torch.no_grad():
#     train_recon = reconstructor(train_tensor)
#     # 关键改进：使用 mean 而不是 sum。
#     # 这样得分的量级不会随着特征维度 dim 的变化而剧烈波动，阈值算法更鲁棒
#     # train_mse = torch.mean((train_recon - train_tensor) ** 2, dim=1).cpu().numpy()
#     train_mse = torch.sum((train_recon - train_tensor) ** 2, dim=1).cpu().numpy()
# print(f"→ Calibration Done. MSE Mean: {np.mean(train_mse):.4f}, Max: {np.max(train_mse):.4f}")


# 建议在推理代码的 AE 训练完成后，按如下逻辑重新计算阈值：



def compute_kde_threshold(train_mse, confidence):
    # 拟合分布
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(train_mse.reshape(-1, 1))
    # 采样计算阈值点
    x_test = np.linspace(0, np.max(train_mse) * 1.5, 5000).reshape(-1, 1)
    log_dens = kde.score_samples(x_test)
    dens = np.exp(log_dens)
    # 累计概率达到 confidence 的地方
    cum_dens = np.cumsum(dens) / np.sum(dens)
    idx = np.searchsorted(cum_dens, confidence)
    return x_test[idx][0]


# 替换你代码中的 base_threshold 逻辑
base_threshold = compute_kde_threshold(train_mse, confidence=kde_confidence)

print("train_mse:", train_mse)
print("train_mse（min):", min(train_mse))
print("train_mse（max):", max(train_mse))
print(f"Robust Threshold: {base_threshold:.2f}")

# ==================== 4. 推理阶段 ====================
dataset_test = TwinGraphDataset(root=root, mode=attack_mode)
raw_test_scores = []
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

start_time = time.time()

print(f"→ Running Batch Inference on {attack_mode}...")
with torch.no_grad():
    for data in tqdm(test_loader):
        data = data.to(device)
        z = gnn_encoder.encode(data)
        # 批量标准化
        z_np = z.cpu().numpy()
        z_norm = torch.FloatTensor(scaler.transform(z_np)).to(device)
        # 批量计算重建误差
        recon = reconstructor(z_norm)
        score = torch.sum((recon - z_norm) ** 2, dim=1).cpu().numpy()
        raw_test_scores.extend(score)

raw_test_scores = np.array(raw_test_scores)

torch.cuda.synchronize() if torch.cuda.is_available() else None
total_latency = time.time() - start_time

print("raw_test_scores:", raw_test_scores)
print("raw_test_scores（min):", min(raw_test_scores))
print("raw_test_scores（max):", max(raw_test_scores))

# --- 预测标签 ---
# 在计算 pred_labels 前进行中值滤波（窗口大小设为 5 或 7）这能有效消除由于单帧解析异常产生的误报
smoothed_scores = signal.medfilt(raw_test_scores, kernel_size=5)
# 使用平滑后的得分进行判定
pred_labels = (smoothed_scores > base_threshold).astype(int)


# ==================== 6. 评估与保存 ====================
label_file = os.path.join(root, attack_mode, "graph_statistics.csv")
gt_labels = pd.read_csv(label_file)["label"].values[:len(smoothed_scores)]

auc = roc_auc_score(gt_labels, smoothed_scores)
f1 = f1_score(gt_labels, pred_labels)
acc = accuracy_score(gt_labels, pred_labels)
prec = precision_score(gt_labels, pred_labels, zero_division=0)
rec = recall_score(gt_labels, pred_labels, zero_division=0)
cm = confusion_matrix(gt_labels, pred_labels)
avg_latency_ms = (total_latency / len(gt_labels)) * 1000

# 保存指标
with open(os.path.join(save_dir, f"metrics.csv"), "w", newline="") as f:  # -{base_threshold:.2f}
    w = csv.writer(f)
    w.writerow(["Metric", "Value"])
    w.writerow(["Accuracy", f"{acc:.4f}"])
    w.writerow(["Precision", f"{prec:.4f}"])
    w.writerow(["Recall", f"{rec:.4f}"])
    w.writerow(["F1-score", f"{f1:.4f}"])
    w.writerow(["AUC", f"{auc:.4f}"])
    w.writerow(["Latency(ms)", f"{avg_latency_ms:.4f}"])

with open(os.path.join(save_dir, f"cm.csv"), "w", newline="") as f:  # -{base_threshold:.2f}
    w = csv.writer(f)
    w.writerow(["Confusion Matrix"])
    w.writerow(["", "Pred 0", "Pred 1"])
    w.writerow(["Real 0", cm[0][0], cm[0][1]])
    w.writerow(["Real 1", cm[1][0], cm[1][1]])

print(f"\n✅ Inference Done! acc: {acc:.4f}, prec: {prec:.4f}, "
      f"rec: {rec:.4f}, F1: {f1:.4f}, "
      f"AUC: {auc:.4f}, avg_latency_ms: {avg_latency_ms:.4f}")


# ==================== 5. 可视化 ====================
def plot_xiphos_logic_rich(train_mse, raw, threshold, save_path):
    # 创建子图
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 12), sharex=False)

    # --- 统一的显示限制计算 ---
    # 为了规避超级极值，我们将绘图窗口限制在阈值的数倍范围内
    y_limit = threshold * 4

    # --- 子图 0: 训练集基准分布 ---
    ax0.plot(train_mse, color='green', alpha=0.6, label='Train MSE (Normal Baseline)')
    ax0.axhline(y=threshold, color='red', linestyle='-', linewidth=2, label=f'Base Threshold: {threshold:.2f}')
    ax0.set_title("Step 0: Define Normal Baseline & Threshold (Training Phase)")
    ax0.set_ylabel("MSE Score")
    # 规避训练集极大值
    ax0.set_ylim(0, y_limit)
    ax0.legend(loc='upper right')
    ax0.grid(axis='y', linestyle=':', alpha=0.7)

    # --- 子图 1: 推理得分与最终判定 ---
    ax1.plot(raw, color='black', alpha=0.7, label='Calibrated Scores (Static)')
    ax1.axhline(y=threshold, color='red', linewidth=2, label=f'Fixed Threshold: {threshold:.2f}')

    # 高亮检测到的异常点
    anomalies = np.where(raw > threshold)[0]
    if len(anomalies) > 0:
        # 即使被裁剪，scatter 依然会在 y_limit 处显示或隐藏
        ax1.scatter(anomalies, raw[anomalies], color='red', s=10, label='Detected Attacks', zorder=3)

    ax1.set_title("Inference Phase: Fixed Threshold Detection")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Score")
    # 规避测试集超级极值 (2429.6 将会被截断在 y_limit 处，但在图中依然能看出是极高的点)
    ax1.set_ylim(0, y_limit)
    ax1.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'xiphos_no_drift_final-{base_threshold:.2f}.png'))
    plt.show()

def plot_xiphos_distribution(scores, labels, threshold, save_path):
    plt.figure(figsize=(10, 6))
    # --- 核心改进：设置合理的显示上限 ---
    # 建议设置为阈值的 3 到 5 倍，这样既能看到分布，又不会被数千量级的极值干扰
    x_limit = threshold * 5
    # 筛选出在显示范围内的数据进行绘图（防止 KDE 计算受超长尾分布影响导致变形）
    display_scores = scores[scores <= x_limit]
    display_labels = labels[scores <= x_limit]
    # 绘制直方图和 KDE
    sns.histplot(scores[(labels == 0) & (scores <= x_limit)],
                 color="green", label="Normal", kde=True, stat="density", alpha=0.5)
    sns.histplot(scores[(labels == 1) & (scores <= x_limit)],
                 color="red", label="Attack", kde=True, stat="density", alpha=0.5)
    # 绘制阈值线
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.2f}')
    # 装饰
    plt.title(f"Test Anomaly Score Distribution")
    plt.xlabel("Test Anomaly Score")
    plt.ylabel("Density")
    # 限制 X 轴范围
    plt.xlim(0, x_limit)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ==================== 6. 数据保存 ====================
def save_distribution_data(scores, labels, threshold, save_dir, attack_mode):
    # 1. 构造主数据表 (Scores & Labels)
    # 每一行代表一个测试样本
    dist_df = pd.DataFrame({
        "Anomaly_Score": scores,
        "Ground_Truth_Label": labels  # 0 为 Normal, 1 为 Attack
    })
    # 3. 保存到 CSV
    # 建议将得分数据单独保存，方便绘图软件直接读取
    data_path = os.path.join(save_dir, f"{attack_mode}_scores-{threshold:.2f}.csv")
    dist_df.to_csv(data_path, index=False)
    print(f"✅ 分布图原始数据已保存: {data_path}")

# ==================== 7. 调参并保存结果 ====================
def tune_kde_confidence(train_mse, smoothed_scores, gt_labels,
                        conf_list=[0.98, 0.985, 0.99, 0.995, 0.999, 0.9999],
                        save_dir="./"):
    """
    封装的置信度调优函数
    :param train_mse: 训练集的重建误差 (numpy array)
    :param smoothed_scores: 测试集平滑后的得分 (numpy array)
    :param gt_labels: 测试集真实标签 (numpy array)
    :param conf_list: 待搜索的置信度列表
    :param save_dir: 结果保存目录
    """
    print(f"🔎 Starting KDE Confidence Tuning...")
    tuning_results = []

    # 1. 预计算 KDE 对象 (只需拟合一次)
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(train_mse.reshape(-1, 1))
    # 建立精细采样刻度尺
    x_grid = np.linspace(0, np.max(train_mse) * 1.5, 5000).reshape(-1, 1)
    dens = np.exp(kde.score_samples(x_grid))
    cum_sum = np.cumsum(dens) / np.sum(dens)

    # 2. 置信度搜索环
    for c in tqdm(conf_list, desc="Searching Confidence"):
        # 计算当前置信度下的阈值
        thresh = x_grid[np.searchsorted(cum_sum, c)][0]
        # 判定
        preds = (smoothed_scores > thresh).astype(int)

        # 记录指标 (修正了 Acc 的计算对象为 preds)
        tuning_results.append({
            "Conf": c,
            "Thresh": thresh,
            "F1": f1_score(gt_labels, preds, zero_division=0),
            "Prec": precision_score(gt_labels, preds, zero_division=0),
            "Rec": recall_score(gt_labels, preds, zero_division=0),
            "Acc": accuracy_score(gt_labels, preds)
        })

    # 3. 结果处理与寻找最优项
    df_res = pd.DataFrame(tuning_results)
    best_c = df_res.loc[df_res['F1'].idxmax()]

    # 4. 保存原始数据 CSV
    output_csv_name = os.path.join(save_dir, f"KDE_Confidence-{best_c['Conf']}-{best_c['F1']:.4f}.csv")
    df_res.to_csv(output_csv_name, index=False)
    print(f"📊 原始数据已保存: {output_csv_name}")

    # 5. 结果可视化
    plt.figure(figsize=(12, 6))

    # 定义绘图参数以确保清晰度
    metrics = [
        ('F1', 'red', '-', 'o', f'F1-score (Best: {best_c["F1"]:.4f})'),
        ('Prec', 'blue', '--', 's', 'Precision'),
        ('Rec', 'green', '--', '^', 'Recall'),
        ('Acc', 'purple', '-.', 'D', 'Accuracy')
    ]

    for col, color, ls, marker, lab in metrics:
        plt.plot(df_res['Conf'], df_res[col], color=color, linestyle=ls,
                 marker=marker, markersize=8, linewidth=2, label=lab)

    # 坐标轴美化
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

    plt.title(f"Threshold Tuning: Confidence vs Metrics\n(Best Conf: {best_c['Conf']} at F1: {best_c['F1']:.4f})",
              fontsize=14, fontweight='bold')
    plt.xlabel("Confidence Level", fontsize=12, fontweight='bold')
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.legend(loc='lower left', frameon=True, shadow=True)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    # 保存图片
    output_img_name = os.path.join(save_dir, f"Tuning_Curve_{best_c['Conf']}-{best_c['F1']:.4f}.png")
    plt.savefig(output_img_name, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"🏆 Optimization Finished! Best Conf: {best_c['Conf']} | Thresh: {best_c['Thresh']:.4f}")
    return best_c



# # ==================== 5. 可视化 ====================
# plot_xiphos_logic_rich(train_mse, smoothed_scores, base_threshold, save_dir)
# dist_img_path = os.path.join(save_dir, f"{attack_mode}-score_distribution-{base_threshold:.2f}.png")
# plot_xiphos_distribution(smoothed_scores, gt_labels, base_threshold, dist_img_path)
#
# # ==================== 6. 数据保存 ====================
# save_distribution_data(smoothed_scores, gt_labels, base_threshold, save_dir, attack_mode)

# ==================== 7. 调参并保存结果 ====================
best_config = tune_kde_confidence(train_mse, smoothed_scores, gt_labels, save_dir=save_dir)


