import os
import itertools

# ========== 1. 根据你的图片设定的参数范围 ==========
search_space = {
    "batch_size": [64, 128, 256],
    "lr": [1e-1, 1e-2],
    "hidden_dim": [16, 32, 64],
    "projection_dim": [16, 32, 64]
}


def run_full_grid_search():
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"🚀 启动网格搜索 | 总计组合数: {len(combinations)}")

    for i, params in enumerate(combinations):
        # 构造保存标签，方便后续数据提取
        tag = f"bs{params['batch_size']}_lr{params['lr']}_h{params['hidden_dim']}_p{params['projection_dim']}"

        cmd = (
            f"python three_train_twin_gcl_simsiam.py "
            f"--batch_size {params['batch_size']} "
            f"--lr {params['lr']} "
            f"--hidden_dim {params['hidden_dim']} "
            f"--projection_dim {params['projection_dim']} "
            f"--save_tag {tag}"
        )

        print(f"\n[{i + 1}/{len(combinations)}] 正在运行: {tag}")
        os.system(cmd)


if __name__ == "__main__":
    run_full_grid_search()
