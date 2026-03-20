# ============================================================
# three_train_twin_gcl_simsiam.py
# Twin-GCL (SimSiam) —— 无负样本自监督图对比学习
# ============================================================

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from two_twin_dataset import TwinGraphDataset
import argparse
from sklearn.manifold import TSNE

# 调超参数
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--hidden_dim", type=int, default=16)
parser.add_argument("--projection_dim", type=int, default=16)
parser.add_argument("--save_tag", type=str, default="bs64_lr0.1_h16_p16")
args = parser.parse_args()

batch_size = args.batch_size
lr = args.lr
hidden_dim = args.hidden_dim
projection_dim = args.projection_dim
save_tag = args.save_tag

# 固定参数
epochs = 8

# ==================== 配置 ====================#
W, S = 40, 10
datasets_name = "Car-Hacking"
root = fr"{datasets_name}_graphs_W{W}S{S}"

normal_mode = "train_CRG"
save_dir = fr"{root}_results\{normal_mode}_{save_tag}"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================
# 模型
# ============================================================


class TwinGCL_SimSiam(nn.Module):
    def __init__(self, in_dim, hid_dim, proj_dim):
        super().__init__()
        # -------- Encoder (GNN) --------
        self.gnn1 = GCNConv(in_dim, hid_dim)
        self.gnn2 = GCNConv(hid_dim, hid_dim)
        self.gnn3 = GCNConv(hid_dim, hid_dim)
        # -------- Projector MLP --------
        self.projector = nn.Sequential(
            nn.Linear(hid_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )

        # -------- Predictor MLP --------
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Linear(proj_dim // 2, proj_dim)
        )

    # GNN 编码器
    def encode(self, batch):
        x, edge, b = batch.x, batch.edge_index, batch.batch
        h = F.relu(self.gnn1(x, edge))
        h = F.relu(self.gnn2(h, edge)) + h
        h = F.relu(self.gnn3(h, edge)) + h
        g = global_mean_pool(h, b)
        return g

    # 补充
    def encode_with_predictor(self, g):
        h = self.encode(g)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        p = self.predictor(z)
        p = F.normalize(p, dim=1)
        return z, p

    def forward_backbone(self, batch):
        h = self.encode(batch)
        z = self.projector(h)
        return z

    def forward(self, g1, g2):
        # 1. 提取 hidden feature
        h1 = self.encode(g1)
        h2 = self.encode(g2)
        # 2. 投影到 projection space (z)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        # 3. 预测 (p)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        # 4. 在计算相似度前进行标准化 (重要：z 不在这里做 normalize，而是在 loss 函数里做)
        # 或者在 forward 里统一返回标准化的结果，让 loss 函数更简洁
        return p1, p2, z1, z2

# ============================================================
# 损失函数
# ============================================================
def simsiam_loss(p1, p2, z1, z2):
    """
    SimSiam Loss 函数：计算 (p1, z2) 和 (p2, z1) 的负余弦相似度均值的平均值
    .detach()是对 z（投影器的输出）进行 stop-gradient 操作
    """
    # 计算前先标准化，确保计算的是余弦相似度
    p1 = F.normalize(p1, dim=-1)
    p2 = F.normalize(p2, dim=-1)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    # 计算第一组的负余弦相似度
    loss1 = -F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
    # 计算第二组的负余弦相似度
    loss2 = -F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
    return (loss1 + loss2) / 2


# ============================================================
# 数据集加载
# ============================================================
def collate_pairs(batch):
    g1_list = [item[0] for item in batch]
    g2_list = [item[1] for item in batch]
    y_list = [item[2] for item in batch]
    batch1 = Batch.from_data_list(g1_list)
    batch2 = Batch.from_data_list(g2_list)
    labels = torch.stack(y_list).squeeze(1)
    return batch1, batch2, labels


train_set = TwinGraphDataset(root, mode=normal_mode)  # 调用数据
loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_pairs, drop_last=True)
print(f"Train pairs: {len(train_set)}  batches: {len(loader)}")
input_dim = train_set[0][0].num_node_features

# ============================================================
# 开始训练
# ============================================================
def train():

    model = TwinGCL_SimSiam(input_dim, hidden_dim, projection_dim).to(device)  # 加载模型
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # 学习率调度器（LR Scheduler）：这是对比学习的标配。在固定轮次下，随着训练进行逐渐减小学习率，可以帮助模型在后期更稳定地收敛。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_curve = []

    all_samples_count = len(train_set)
    total_training_time = 0.0

    for epoch in range(0, epochs):
        epoch_loss = 0
        model.train()

        for g1, g2, _ in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            g1, g2 = g1.to(device), g2.to(device)
            # START: 计时开始（涵盖前向传播、损失计算及反向传播）
            start_time = time.time()
            # 前向传播
            p1, p2, z1, z2 = model(g1, g2)
            loss = simsiam_loss(p1, p2, z1, z2)
            # 反向优化
            optim.zero_grad()
            loss.backward()
            optim.step()
            # END: 计时结束
            torch.cuda.synchronize()  # 如果使用 GPU，建议同步以获得准确时间
            total_training_time += (time.time() - start_time)

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        loss_curve.append(avg_loss)
        print(f"Epoch {epoch+1} | Loss={avg_loss:.4f}")
        # 在 epoch 循环结束时调用
        scheduler.step()

    # 1. 保存时间指标：Training Latency (per sample) 包含：总训练时间、总样本吞吐量、单样本延迟、以及当前的超参数
    total_samples_seen = all_samples_count * epochs  # 注意：all_samples_count 是训练集总数，epochs 是总轮数
    avg_latency_ms = (total_training_time / total_samples_seen) * 1000
    training_metrics_df = pd.DataFrame({
        "Metric": ["Total Training Time (s)", "Total Samples Seen", "Epochs", "Training Latency (ms/sample)"],
        "Value": [f"{total_training_time:.2f}", f"{total_samples_seen}", f"{epochs}", f"{avg_latency_ms:.4f}"]
    })
    metrics_csv_path = os.path.join(save_dir, "training_metrics.csv")
    training_metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✅ Training metrics saved to → {metrics_csv_path}")
    print(f"⏱️ Avg Latency: {avg_latency_ms:.4f} ms/sample")

    # 2. 保存模型
    save_model_path = os.path.join(save_dir, "twin_gcl_simsiam_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim": input_dim,
        "hid_dim": hidden_dim,
        "proj_dim": projection_dim
    }, save_model_path)
    print(f"Model saved → {save_model_path}")

    # 3. 保存训练损失值
    loss_df = pd.DataFrame({
        "Epoch": range(1, len(loss_curve) + 1),
        "Loss": loss_curve
    })
    csv_loss_path = os.path.join(save_dir, "loss_data_simsiam.csv")
    loss_df.to_csv(csv_loss_path, index=False)
    print(f"Loss data saved → {csv_loss_path}")
    return model

# ============================================================
# 训练结束，保存图嵌入
# ============================================================


def get_embedding(model):
    # ============================================================
    #  提取训练集图的embedding Z
    # ============================================================
    print(f"→ Starting extracting the embedding of the training set")
    model.eval()
    Z = []
    # 使用较简单的 DataLoader 批量读取
    embed_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)
    with torch.no_grad():
        for batch1, _, _ in tqdm(embed_loader, desc="Extracting embeddings"):
            batch1 = batch1.to(device)
            h = model.encode(batch1)
            Z.append(h.cpu().numpy())
    Z = np.vstack(Z)
    print(f"training set embedding shape={Z.shape}\n")

    # 保存嵌入
    save_path = os.path.join(save_dir, "embeddings_info.pt")
    torch.save({
        "train_embeddings": Z  # (训练集嵌入)：所有正常样本的特征向量。
    }, save_path)
    print(f"✓ Saved embeddings → {save_path}\n")
    return Z


def Visualization(Z):
    # ============================================================
    # 对embedding Z 进行T-SNE 降维, 得到Z_2d，
    # ============================================================
    # 如果数据量太大，采样可视化
    if Z.shape[0] > 5000:
        np.random.seed(42)  # 保证采样一致性
        idx = np.random.choice(Z.shape[0], 5000, replace=False)
        Z = Z[idx]
    print("→ Running T-SNE ...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="random",
        random_state=42,
        n_iter=1500
    )
    Z_2d = tsne.fit_transform(Z)
    print("✓ T-SNE completed\n")

    # ============================================================
    # T-SNE 可视化
    # ============================================================
    plt.figure(figsize=(8, 7))
    plt.scatter(
        Z_2d[:, 0], Z_2d[:, 1],
        s=12,
        alpha=0.85
    )
    plt.title(f"T-SNE_embeddings")
    plt.grid(True)
    plt.tight_layout()
    # 保存图片
    tsne_out_path = os.path.join(save_dir, f"T-SNE_embeddings.png")
    plt.savefig(tsne_out_path, dpi=300)
    plt.close()
    print(f"✓ Saved T-SNE KMeans clustering figure → {tsne_out_path}\n")
    print("\n========== All Done ==========\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    model = train()
    Z = get_embedding(model)
    # Visualization(Z)
