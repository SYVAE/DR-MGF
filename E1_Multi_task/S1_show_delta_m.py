import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as st

STATS = [
    "semantic loss", "mean iou", "pix acc", "depth loss",
    "abs err", "rel err", "normal loss", "mean",
    "median", "<11.25", "<22.5", "<30"
]

DELTA_STATS = [
    "mean iou", "pix acc", "abs err", "rel err",
    "mean", "median", "<11.25", "<22.5", "<30"
]

# 基准值和符号（用于delta计算）
BASE = np.array([0.3830, 0.6376, 0.6754, 0.2780, 25.01, 19.21, 0.3014, 0.5720, 0.6915])

SIGN = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])
KK = np.ones(9) * -1

STATS_IDX_MAP = [4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17]
TIME_IDX = 34
SEEDS = [0]


# ==================== 数据加载和解析 ====================
def load_logs(file_path):
    """从日志文件加载训练和测试数据"""
    logs = {"train": [{} for _ in SEEDS], "test": [{} for _ in SEEDS]}
    min_epoch = float('inf')

    for seed in SEEDS:
        logs["train"][seed] = {stat: [] for stat in STATS}
        logs["test"][seed] = {stat: [] for stat in STATS}
        logs["train"][seed]["time"] = []

    for seed in SEEDS:
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("Epoch:"):
                    parts = line.split()
                    # 解析统计指标
                    for i, stat in enumerate(STATS):
                        logs["train"][seed][stat].append(float(parts[STATS_IDX_MAP[i]]))
                        logs["test"][seed][stat].append(float(parts[STATS_IDX_MAP[i] + 15]))
                    # 解析时间
                    logs["train"][seed]["time"].append(float(parts[TIME_IDX]))

        # 更新最小epoch数
        min_epoch = min(min_epoch,
                        len(logs["train"][seed]["semantic loss"]),
                        len(logs["test"][seed]["semantic loss"]))

    return logs, min_epoch


def calculate_statistics(logs, min_epoch, mode="test"):
    """计算统计指标的均值和标准差"""
    stats_dict = {}
    print(" " * 25 + " | ".join([f"{s:5s}" for s in STATS]))

    if mode == "test":
        print(mode)
        string = f"{'DRMGF':30s} "
        for stat in STATS:
            values = []
            for seed in SEEDS:
                values.append(np.array(logs[mode][seed][stat][min_epoch - 10:min_epoch]).mean())
            values = np.array(values)
            stats_dict[stat] = values.copy()
            mu = values.mean()
            string += f" | {mu:5.4f}"
        print(string)

    return stats_dict


def calculate_delta(test_stats, method_name):
    """计算delta指标"""

    def delta_fn(a):
        return (KK ** SIGN * (a - BASE) / BASE).mean() * 100.0

    tmp = np.zeros(len(DELTA_STATS))
    for i, stat in enumerate(DELTA_STATS):
        tmp[i] = test_stats[stat].mean()
    delta = delta_fn(tmp)
    print(f"{method_name:30s} delta: {delta:4.3f}")
    return delta



# ==================== 主执行流程 ====================
def main():
    # 加载数据
    file_path = "/home/sunyi/workspace/DR-MGF-MTL/Baseline_res/20221129/nyuv2/Metalr0.1_ema_0.5_aux_0.42022-11-16-11-29-11/metaGF-equal-sd0-metalr0.1-ema0.5-auxlr0.4.log"
    logs, min_epoch = load_logs(file_path)

    # 计算统计信息
    test_stats = calculate_statistics(logs, min_epoch, mode="test")
    # 计算delta指标
    deltas = {}
    method = 'drmgf'
    deltas[method] = calculate_delta(test_stats, method)


if __name__ == "__main__":
    main()