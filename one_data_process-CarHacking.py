# ===============================================================
# Data Loader + Graph Builder
# ===============================================================
import os
import torch
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm

# 全局参数
datasets_name = "Car-Hacking"
W, S = 40, 10
# 正常数据
normal_mode = "train_CRG"
train_file = rf"HCRL\{datasets_name}\normal_run_data.csv"
train_df = pd.read_csv(train_file, on_bad_lines="skip")  # .iloc[0:100000]
# 混合攻击数据
attack_mode = "test_MIX_CRG"
test_file = rf"HCRL\{datasets_name}\process\test_DoS+Fuzzy+gear+RPM.csv"
test_df = pd.read_csv(test_file, on_bad_lines="skip")
# 保存路径
out_dir = rf"{datasets_name}_graphs_W{W}S{S}"
os.makedirs(out_dir, exist_ok=True)

# ===============================================================
# 1) 数据加载与清洗
# ===============================================================
def safe_hex(x):
    """Convert hex string to int safely."""
    if x is None:
        return 0
    x = str(x).strip()
    if x.lower() in ["", "nan", "t", "r"]:
        return 0
    try:
        return int(x, 16)
    except:
        try:
            return int(float(x))
        except:
            return 0


def fix_label(df):
    """
    修复测试集中的 label 错位问题：
    - label 正常在最后一列
    - 若 label 混入 data1~data8，则将其挤回 label 列
    """
    for i in range(len(df)):
        row = df.loc[i]
        label = row.get("label", None)
        # 若 label 已经正常，则不处理
        if label in ["T", "R"]:
            continue
        clean = []
        for j in range(1, 9):
            v = str(row[f"data{j}"]).strip()
            if v in ["T", "R"]:
                label = v
                clean.append("nan")  # data 字段复原
            else:
                clean.append(v)
        # 更新 label
        if label is not None:
            df.loc[i, "label"] = label
        # 更新 data 字段
        for j in range(1, 9):
            df.loc[i, f"data{j}"] = clean[j-1]
    return df


def load_and_clean_csv(df, out_csv_path, with_label):
    """
    加载 CSV → 清洗 → safe_hex 十进制 → 保存处理后的 CSV
    """
    # 分配列名
    cols = ["Timestamp", "ID", "DLC"] + [f"data{i}" for i in range(1, 9)]
    if with_label:
        cols += ["label"]
    df.columns = cols
    # 调用fix_label函数，测试集需要修复 label 错位
    if with_label:
        df = fix_label(df)
    # 调用safe_hex函数，十六进制 → 十进制
    df["ID"] = df["ID"].apply(safe_hex)
    for i in range(1, 9):
        df[f"data{i}"] = df[f"data{i}"].apply(safe_hex)
    # 保存处理好的 CSV
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    print(f"✓ Cleaned CSV saved → {out_csv_path}")
    return df


# ===============================================================
# 2) 构图
# ===============================================================
def build_CRG_graphs(df, out_path, W, S, is_train):
    """
    XIPHOS CRG 构图函数
    节点：窗口内的每一条 CAN 消息 (共 W 个节点)
    边：1. 相邻消息连接；2. 相同 ID 消息连接
    """
    os.makedirs(out_path, exist_ok=True)
    stats = []
    windows = range(0, len(df) - W + 1, S)

    for idx, start in enumerate(tqdm(windows)):
        win = df.iloc[start:start + W].copy().reset_index(drop=True)

        # 1. 标签处理
        if is_train:
            y = 0
            ratio = 0.0
        else:
            # 只要窗口内有一条 T (Attack)，整个图就是异常
            flags = win["label"].apply(lambda x: 1 if str(x).upper() == "T" else 0)
            y = 1 if flags.sum() > 0 else 0
            ratio = flags.mean()

        # 2. 节点特征 (Node Features)
        # 注意：这里需要对 ID 进行归一化或 LabelEncoding
        X = []
        for i in range(W):
            msg = win.iloc[i]
            # 特征向量：[ID, DLC, Data1...Data8] (根据你的数据列调整)
            node_feats = [msg["ID"]]
            for d_idx in range(1, 9):
                node_feats.append(msg[f"data{d_idx}"])
            X.append(node_feats)
        x = torch.tensor(X, dtype=torch.float)

        # 3. 构建边索引 (Edge Index)
        edge_sources = []
        edge_targets = []

        # 类型 A：连接相邻的消息 (Temporal Causality)
        # 0->1, 1->2, ..., (W-2)->(W-1)
        for i in range(W - 1):
            edge_sources.append(i)
            edge_targets.append(i + 1)

        # 类型 B：连接相同 ID 的消息 (Identity Correlation)
        # 寻找窗口内 ID 相同的消息并连边
        id_groups = win.groupby("ID").indices
        for cid, indices in id_groups.items():
            if len(indices) > 1:
                # 在相同 ID 的节点之间建立全连接或链式连接
                # 这里推荐链式连接（按出现顺序），以保持因果序
                for k in range(len(indices) - 1):
                    u, v = indices[k], indices[k + 1]
                    edge_sources.append(u)
                    edge_targets.append(v)
                    # # 如果需要无向图，可以加上反向边
                    # edge_sources.append(v)
                    # edge_targets.append(u)

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

        # 4. 封装 Data 对象
        y_tensor = torch.tensor([y], dtype=torch.long)
        g = Data(x=x, edge_index=edge_index, y=y_tensor)

        # 记录该图中哪些节点是攻击节点 (用于后期可视化分析)
        if not is_train:
            g.attack_node_indices = torch.tensor(
                win[win["label"].apply(lambda x: str(x).upper() == "T")].index.tolist())

        # 5. 保存图
        fname = f"crg_{idx}_{start}_{start + W}_y{int(y)}.pt"
        torch.save(g, os.path.join(out_path, fname))

        stats.append({
            "idx": idx,
            "start": start,
            "end": start + W,
            "label": int(y),
            "attack_ratio": ratio
        })

    # 保存统计信息
    pd.DataFrame(stats).to_csv(os.path.join(out_path, "graph_statistics.csv"), index=False)
    print(f"✓ CRG Construction Complete. Total graphs: {len(stats)}")


# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":

    # ---------------- Train ----------------
    print("\n=== Load Train CSV ===")
    train_csv_path = os.path.join(out_dir, normal_mode, f"raw_{normal_mode}.csv")
    train_df = load_and_clean_csv(train_df, train_csv_path, with_label=False)  # 无标签
    print("\n=== Building Train Graphs ===")
    build_CRG_graphs(train_df, os.path.join(out_dir, normal_mode), W, S, is_train=True)
    print("\n✓ All Train graphs generated successfully.")

    # ---------------- Test ----------------
    print("\n=== Load Test CSV ===")
    test_csv_path = os.path.join(out_dir, attack_mode, f"raw_{attack_mode}.csv")
    test_df = load_and_clean_csv(test_df, test_csv_path, with_label=True)  # 有标签
    print("\n=== Building Test Graphs ===")
    build_CRG_graphs(test_df, os.path.join(out_dir, attack_mode), W, S, is_train=False)
    print("\n✓ All Test graphs generated successfully.")
