#!/usr/bin/env python3
"""
YOLO11-Pose 训练脚本 - 针对小目标(缝衣针)检测优化
包含训练过程监控和小目标检测参数优化
"""
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import os
import yaml
import torch
import time
from pathlib import Path


def print_section(title):
    """打印分隔标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_environment():
    """检查运行环境"""
    print_section("环境检查")

    # GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] GPU: {gpu_name}")
        print(f"[OK] GPU Memory: {gpu_memory:.1f} GB")
        device = 'cuda:0'
    else:
        print("[WARN] CUDA not available, using CPU")
        device = 'cpu'

    # PyTorch版本
    print(f"[OK] PyTorch: {torch.__version__}")

    return device


def validate_dataset(data_yaml):
    """验证数据集配置"""
    print_section("数据集验证")

    with open(data_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_path = Path(config['path'])
    train_path = base_path / config['train']
    val_path = base_path / config['val']

    # 检查路径
    if not train_path.exists():
        raise FileNotFoundError(f"Train path not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val path not found: {val_path}")

    # 统计图片数量
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    train_images = [f for f in train_path.iterdir() if f.suffix.lower() in img_extensions]
    val_images = [f for f in val_path.iterdir() if f.suffix.lower() in img_extensions]

    print(f"[OK] Train images: {len(train_images)}")
    print(f"[OK] Val images: {len(val_images)}")
    print(f"[OK] Classes: {config.get('names', {})}")
    print(f"[OK] Keypoints: {config.get('kpt_shape', 'N/A')}")

    # 检查标签文件
    train_labels = base_path / 'labels' / 'train'
    if train_labels.exists():
        label_count = len(list(train_labels.glob('*.txt')))
        print(f"[OK] Train labels: {label_count}")

    return len(train_images), len(val_images)


def get_small_object_config():
    """
    小目标检测优化参数

    针对缝衣针这类小目标的优化策略：
    1. 提高输入分辨率 - 小目标在高分辨率下更容易检测，已在model.train中配置
    2. 降低mosaic比例 - mosaic会缩小目标，对小目标不利
    3. 减少scale增强范围 - 避免目标被缩得太小
    """
    config = {
        # === 小目标优化的数据增强 ===
        'mosaic': 0.5,          # 降低mosaic比例，原图1.0会让小目标更小
        'mixup': 0.0,           # 关闭mixup，避免小目标被混淆
        'scale': 0.3,           # 缩小scale范围(默认0.5)，防止目标过小
        'copy_paste': 0.0,      # 小数据集关闭copy_paste

        # === 常规增强保持 ===
        'degrees': 180.0,       # 针可以任意角度
        'translate': 0.15,       # 第二次训练增加平移参数，不影响小目标识别
        'fliplr': 0.5,
        'flipud': 0.5,          # 启用垂直翻转
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    }
    return config


class TrainingMonitor:
    """训练过程监控器"""

    def __init__(self):
        self.start_time = None
        self.epoch_times = []

    def on_train_start(self):
        self.start_time = time.time()
        print_section("训练开始")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def on_train_end(self, results):
        total_time = time.time() - self.start_time
        print_section("训练完成")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best model: {results.save_dir}/weights/best.pt")
        print(f"Last model: {results.save_dir}/weights/last.pt")
        print(f"Results saved to: {results.save_dir}")

# 此处为函数默认值，没有传入参数时使用，后面使用CONFIG覆盖
def train_needle_pose(
    model_name='yolo11s-pose.pt',
    data_yaml='data.yaml',
    epochs=100,
    imgsz=1280,
    batch=8,
    device=None,
    project='runs/pose',
    name='needle'
):
    """
    训练缝衣针姿态检测模型

    Args:
        model_name: 预训练模型 (yolo11n-pose.pt / yolo11s-pose.pt / ...)
        data_yaml: 数据集配置文件
        epochs: 训练轮数
        imgsz: 输入图像尺寸 (小目标建议1280)
        batch: 批次大小 (高分辨率时需降低)
        device: 训练设备
        project: 项目保存目录
        name: 实验名称
    """

    # 1. 环境检查
    if device is None:
        device = check_environment()

    # 2. 验证数据集
    train_count, val_count = validate_dataset(data_yaml)

    # 3. 加载模型
    print_section("加载预训练模型")
    model = YOLO(model_name)
    print(f"[OK] Model: {model_name}")
    print(f"[OK] Task: pose estimation")

    # 4. 获取小目标优化参数
    small_obj_config = get_small_object_config()

    # 5. 打印训练配置
    print_section("训练配置")
    print(f"  Input size: {imgsz}x{imgsz} (高分辨率利于小目标)")
    print(f"  Batch size: {batch}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")
    print(f"  Mosaic: {small_obj_config['mosaic']} (降低以保护小目标)")
    print(f"  Scale: {small_obj_config['scale']} (限制缩放范围)")

    # 显存估算
    if 'cuda' in str(device):
        mem_per_img = (imgsz / 640) ** 2 * 0.5  # 粗略估算 GB
        estimated_mem = mem_per_img * batch
        print(f"  Estimated GPU memory: ~{estimated_mem:.1f} GB")

    # 6. 初始化监控器
    monitor = TrainingMonitor()
    monitor.on_train_start()

    # 7. 开始训练
    print_section("Training Progress")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,

        # 优化器
        optimizer='AdamW',
        lr0=0.0001, # val_loss波动大，从0.001降低
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        nbs = 64, # 梯度累积步数 = 64 / 8 = 8，即每8个batch更新一次梯度

        # 小目标优化的数据增强
        **small_obj_config,

        # 保存设置
        save=True,
        save_period=10,         # 每10轮保存checkpoint

        # 验证和可视化
        val=True,
        plots=True,             # 生成训练曲线、混淆矩阵等

        # 提前停止
        patience=50,

        # 项目设置
        project=project,
        name=name,
        exist_ok=True,

        # 详细输出
        verbose=True,
    )

    # 8. 训练完成
    monitor.on_train_end(results)

    # 9. 验证最佳模型
    print_section("验证最佳模型")
    best_model_path = f'{results.save_dir}/weights/best.pt'
    best_model = YOLO(best_model_path)
    metrics = best_model.val(data=data_yaml)

    # 打印关键指标
    print("\n--- Detection Metrics ---")
    print(f"  Box mAP50:    {metrics.box.map50:.4f}")
    print(f"  Box mAP50-95: {metrics.box.map:.4f}")

    print("\n--- Pose Metrics ---")
    print(f"  Pose mAP50:    {metrics.pose.map50:.4f}")
    print(f"  Pose mAP50-95: {metrics.pose.map:.4f}")

    return best_model, results


def print_resolution_guide():
    """打印分辨率选择指南"""
    print_section("输入分辨率选择指南")
    print("""
    | 分辨率 | 显存(batch=8) | 适用场景           |
    |--------|---------------|-------------------|
    | 640    | ~4 GB         | 中大目标，快速训练  |
    | 1280   | ~12 GB        | 小目标，推荐       |
    | 1920   | ~24 GB        | 极小目标，需高端GPU |

    缝衣针属于小目标，建议使用 1280 或更高分辨率。
    如果显存不足，可以降低 batch size。
    """)


if __name__ == '__main__':
    # 打印分辨率指南
    print_resolution_guide()

    # 训练配置
    CONFIG = {
        'model_name': 'yolo11s-pose.pt',  # 预训练的模型
        'data_yaml': 'data.yaml',          # 数据集配置
        'epochs': 200,                     # 训练轮数
        'imgsz': 1280,                     # 输入分辨率 (小目标用1280)
        'batch': 8,                        # 批次大小 (显存不足则降低)
        'project': 'runs/pose',            # 保存目录
        'name': 'needle_pose_0114',             # 实验名称
    }

    print_section("训练参数")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    # 开始训练
    best_model, results = train_needle_pose(**CONFIG)

    # 导出ONNX
    print_section("导出ONNX")
    best_model.export(format='onnx', imgsz=CONFIG['imgsz'], simplify=True)

    print("\n" + "=" * 60)
    print(" ALL DONE!")
    print("=" * 60)
