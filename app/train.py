"""
模型训练脚本
============
使用 MobileNetV2 迁移学习训练舌质分类和舌苔分类模型。

数据集目录结构要求:
    data/
    ├── tongue_body/          # 舌质分类数据
    │   ├── pale_red/         # 淡红舌
    │   ├── pale/             # 淡白舌
    │   ├── red/              # 红舌
    │   ├── crimson/          # 绛舌
    │   └── purple/           # 青紫舌
    └── coating/              # 舌苔分类数据
        ├── thin_white/       # 薄白苔
        ├── thick_white/      # 厚白苔
        ├── yellow/           # 黄苔
        ├── greasy_yellow/    # 黄腻苔
        ├── peeled/           # 剥苔
        └── gray_black/       # 灰黑苔

使用方法:
    uv run python -m app.train --data_dir data/ --epochs 20 --batch_size 16

推荐数据集:
    - ZhongJing-OMNI: https://github.com/pariskang/ZhongJing-OMNI
    - 自采小样本 + 数据增强（旋转、翻转、色彩抖动、CutMix）
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .model import build_mobilenetv2, TONGUE_BODY_LABELS, COATING_LABELS, MODEL_DIR


def get_train_transforms():
    """训练数据增强：旋转、翻转、色彩抖动等"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """验证数据预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_model(
    data_dir: str,
    labels: list,
    model_path: str,
    task_name: str,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
):
    """
    训练单个分类模型。

    参数:
        data_dir: 数据目录路径（含子文件夹，每个子文件夹为一个类别）
        labels: 类别标签列表
        model_path: 模型保存路径
        task_name: 任务名称（用于日志显示）
        epochs: 训练轮数
        batch_size: 批量大小
        learning_rate: 学习率
    """
    print(f"\n{'='*60}")
    print(f"  开始训练: {task_name}")
    print(f"  数据目录: {data_dir}")
    print(f"  类别数: {len(labels)}")
    print(f"  类别: {labels}")
    print(f"  训练轮数: {epochs}")
    print(f"  批量大小: {batch_size}")
    print(f"{'='*60}\n")

    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"[错误] 数据目录不存在: {data_dir}")
        print("请先准备数据集，参考 README.md 中的说明。")
        return False

    # 加载数据集
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    try:
        full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    except Exception as e:
        print(f"[错误] 加载数据集失败: {e}")
        return False

    if len(full_dataset) == 0:
        print(f"[错误] 数据目录中没有图像: {data_dir}")
        return False

    # 检查类别是否匹配
    found_classes = full_dataset.classes
    print(f"  发现的类别文件夹: {found_classes}")
    print(f"  数据集总样本数: {len(full_dataset)}")

    # 按 80/20 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 验证集使用不同的预处理
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备: {device}")

    model = build_mobilenetv2(num_classes=len(found_classes))
    model = model.to(device)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练循环
    best_val_acc = 0.0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels_idx in train_loader:
            images, labels_idx = images.to(device), labels_idx.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_idx)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels_idx.size(0)
            correct += (predicted == labels_idx).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels_idx in val_loader:
                images, labels_idx = images.to(device), labels_idx.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels_idx)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels_idx.size(0)
                correct += (predicted == labels_idx).sum().item()

        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * correct / total if total > 0 else 0

        scheduler.step()

        print(f"  Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"    → 保存最佳模型 (验证准确率: {val_acc:.1f}%)")

    print(f"\n  训练完成！最佳验证准确率: {best_val_acc:.1f}%")
    print(f"  模型已保存至: {model_path}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="中医舌诊AI模型训练脚本")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据根目录路径（包含 tongue_body/ 和 coating/ 子目录）")
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮数（默认20）")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批量大小（默认16）")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率（默认0.001）")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "body", "coating"],
                        help="训练任务: all(全部) / body(仅舌质) / coating(仅舌苔)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  中医AI舌诊 - MobileNetV2 迁移学习训练脚本")
    print("=" * 60)
    print(f"  数据目录: {args.data_dir}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批量大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  训练任务: {args.task}")

    from .model import TONGUE_BODY_MODEL_PATH, COATING_MODEL_PATH

    success_count = 0
    total_tasks = 0

    if args.task in ("all", "body"):
        total_tasks += 1
        body_data_dir = os.path.join(args.data_dir, "tongue_body")
        if train_model(body_data_dir, TONGUE_BODY_LABELS, TONGUE_BODY_MODEL_PATH,
                       "舌质分类", args.epochs, args.batch_size, args.lr):
            success_count += 1

    if args.task in ("all", "coating"):
        total_tasks += 1
        coating_data_dir = os.path.join(args.data_dir, "coating")
        if train_model(coating_data_dir, COATING_LABELS, COATING_MODEL_PATH,
                       "舌苔分类", args.epochs, args.batch_size, args.lr):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"  训练总结: {success_count}/{total_tasks} 个任务成功完成")
    if success_count == total_tasks:
        print("  ✅ 全部训练完成！现在可以运行 app.main 启动应用使用深度学习模型。")
    else:
        print("  ⚠️ 部分任务未完成，应用将使用规则推理模式作为后备。")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
