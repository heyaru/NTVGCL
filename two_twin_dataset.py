import os
import torch
import re
from torch_geometric.data import InMemoryDataset, Data

# ---------------------- Configuration --------------------------
datasets_name = "Car-Hacking"
normal_mode = "train_CRG"
# attack_mode = "test_DoS_50000_CRG"
W, S = 40, 10

root = fr"{datasets_name}_graphs_W{W}S{S}"


class TwinGraphDataset(InMemoryDataset):
    """
    Twin-GCL 数据集封装：
    - 读取由 1_data_process.py 生成的 .pt 图文件
    - 训练阶段生成时序孪生图对
    - 测试阶段直接返回单张图
    """
    def __init__(self, root, mode=f'{normal_mode}', transform=None, pre_transform=None):
        self.mode = mode
        super().__init__(root, transform, pre_transform)
        self.data_list = self.load_graphs()

    def load_graphs(self):
        folder = os.path.join(self.root, self.mode)
        graph_files = sorted(
            [f for f in os.listdir(folder) if f.endswith('.pt')],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )
        # # ---- 新增：打印前 5 个图文件名 ----
        # print("📄 前 5 个图文件：")
        # for name in graph_files[:5]:
        #     print("   •", name)
        graphs = [torch.load(os.path.join(folder, f)) for f in graph_files]
        print("📁 使用的图目录：", folder)
        print(f"✅ 已按时间顺序加载 {len(graphs)} 张图")
        return graphs

    def __len__(self):
        if self.mode == f'{normal_mode}':
            return len(self.data_list) - 1  # 训练阶段需要成对
        else:
            return len(self.data_list)      # 测试阶段使用全部图

    def __getitem__(self, idx):
        if self.mode == f'{normal_mode}':
            g1 = self.data_list[idx]
            g2 = self.data_list[idx + 1]
            pair_label = torch.tensor([1], dtype=torch.long)  # 正对标签
            return g1, g2, pair_label
        else:  # 测试阶段
            g = self.data_list[idx]
            return g  # 只返回单张图


if __name__ == "__main__":
    print("TwinGraphDataset Successful!")
    # 训练集
    # dataset_train = TwinGraphDataset(root=root, mode=f'{normal_mode}')  # 正常数据
    # print(f"训练集长度: {len(dataset_train)}")
    #
    # i = 1
    # print(f"\n🔍 训练集取出的第{i}个训练孪生对图信息检查：")
    # g1, g2, lab = dataset_train[i]
    # print(f"\n" + "=" * 40)
    # print(f"--- Pair Index {i} ---")
    # # 1. 孪生对标签信息
    # print(f"Pair Label: {lab.item()}")
    # # 2. 图的拓扑结构和标签信息
    # print(f"Nodes: {g1.num_nodes}")
    # print(f"Edges: {g1.num_edges}")
    # print(f"Label (y): {g1.y.item()}")
    # # 3. 节点属性 (Features)
    # print(f"Node Feature Dim: {g1.num_node_features}")
    # print("Example Node Attributes (First 3 nodes):")
    # # 打印前 3 个节点的特征向量
    # print(g1.x[:3])
    # # 4. 额外属性 (如你自定义的 attack_ids)
    # print(f"Attack IDs: {getattr(g1, 'attack_node_indices', 'None')}")
    # print("=" * 40)
    #
    # # 测试集
    # dataset_test = TwinGraphDataset(root=root, mode=f'{attack_mode}')    # 测试数据
    # print(f"测试集长度: {len(dataset_test)}")
    #
    # i = 171
    # print(f"\n🔍 测试集取出的第{i}个测试图信息检查：")
    # g = dataset_test[i]
    # print(f"\n" + "=" * 40)
    # print(f"--- Pair Index {i} ---")
    # # 2. 图的拓扑结构和标签信息
    # print(f"Nodes: {g.num_nodes}")
    # print(f"Edges: {g.num_edges}")
    # print(f"Label (y): {g.y.item()}")
    # # 3. 节点属性 (Features)
    # print(f"Node Feature Dim: {g.num_node_features}")
    # print("Example Node Attributes (First 3 nodes):")
    # # 打印前 3 个节点的特征向量
    # print(g.x[:10])
    # # 4. 额外属性 (如你自定义的 attack_ids)
    # print(f"Attack IDs: {getattr(g, 'attack_node_indices', 'None')}")
    # print("=" * 40)
    #
