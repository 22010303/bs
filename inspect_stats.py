import argparse
import json
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


def parse_args():
    p = argparse.ArgumentParser(description='Inspect LeRobot Dataset Statistics')
    p.add_argument('--dataset_root', type=str, default='./demo_data_pih',
                   help='Path to the dataset root directory')
    p.add_argument('--repo_name', type=str, default='ur5e_pih_language',
                   help='Dataset repo_id (folder name inside dataset_root)')
    return p.parse_args()


def convert_to_serializable(obj):
    """将 numpy 类型转换为 Python 原生类型以便 JSON 打印"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def main():
    args = parse_args()

    print(f"--- Loading metadata from: {args.dataset_root}/{args.repo_name} ---")

    try:
        # 加载元数据
        dataset_metadata = LeRobotDatasetMetadata(
            args.repo_name,
            root=args.dataset_root
        )

        # 提取统计信息
        stats = dataset_metadata.stats

        if not stats:
            print("Warning: No statistics found in this dataset.")
            return

        # 格式化打印
        # stats 通常包含 'action', 'observation.state', 'observation.image' 等键
        # 每个键下有 'min', 'max', 'mean', 'std' 等
        printable_stats = convert_to_serializable(stats)

        print("\n[Dataset Statistics]")
        print(json.dumps(printable_stats, indent=4))

        # 额外打印一些特征信息以便确认
        print("\n[Feature Schema]")
        for key, feat in dataset_metadata.features.items():
            print(f"- {key}: {feat}")

    except Exception as e:
        print(f"Error: Could not load metadata. {e}")


if __name__ == '__main__':
    main()