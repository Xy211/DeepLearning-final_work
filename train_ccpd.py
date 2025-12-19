"""
CCPD2020 车牌识别训练与验证脚本（基于 ccpd_green 划分）。

特点：
- 自动解析 CCPD 文件名中的车牌标签与车牌框，裁剪后进行多头分类（每个字符一头）。
- 支持 7/8 位车牌（新能源为 8 位），缺省字符用 -1 做 mask。
- 仅依赖 PyTorch / torchvision / opencv-python。

用法示例：
python train_ccpd.py --data_root CCPD2020/CCPD2020/ccpd_green --epochs 3 --batch_size 64 --device cuda:0
python train_ccpd.py --data_root CCPD2020/CCPD2020/ccpd_green --epochs 1 --limit_train 500 --limit_val 200
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models


# CCPD 标签映射表（官方定义）
PROVINCES = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "警",
    "学",
    "O",
]
ALPHAS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "O",
]
ADS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "O",
]


@dataclass
class Sample:
    image_path: Path
    labels: List[int]  # 长度 <= 8，缺省用 -1 占位


def parse_plate_from_name(filename: str) -> List[int]:
    """从 CCPD 文件名解析出车牌索引序列。"""
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"文件名格式异常: {filename}")
    label_part = parts[4]
    indices = [int(x) for x in label_part.split("_")]
    if len(indices) not in (7, 8):
        raise ValueError(f"车牌长度异常: {filename}")
    return indices


def parse_bbox_from_name(filename: str) -> Optional[Tuple[int, int, int, int]]:
    """从文件名解析出车牌框 (x1, y1, x2, y2)。"""
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    bbox_part = parts[2]
    try:
        p1, p2 = bbox_part.split("_")
        x1, y1 = map(int, p1.split("&"))
        x2, y2 = map(int, p2.split("&"))
        return x1, y1, x2, y2
    except Exception:
        return None


class CCPDDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: T.Compose,
        limit: Optional[int] = None,
        max_len: int = 8,
    ):
        self.root = Path(root)
        self.transform = transform
        self.max_len = max_len
        self.samples: List[Sample] = []

        all_imgs = sorted([p for p in self.root.glob("**/*.jpg")])
        if limit:
            random.seed(42)
            all_imgs = random.sample(all_imgs, min(limit, len(all_imgs)))

        for img_path in all_imgs:
            try:
                plate_indices = parse_plate_from_name(img_path.name)
            except Exception:
                continue
            labels = plate_indices + [-1] * (max_len - len(plate_indices))
            self.samples.append(Sample(image_path=img_path, labels=labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = cv2.imread(str(sample.image_path))
        if img is None:
            raise FileNotFoundError(sample.image_path)
        # 按 bbox 裁剪车牌
        bbox = parse_bbox_from_name(sample.image_path.name)
        if bbox:
            x1, y1, x2, y2 = bbox
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                img = img[y1:y2, x1:x2]
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        labels = torch.tensor(sample.labels, dtype=torch.long)
        return img, labels


class PlateNet(nn.Module):
    def __init__(self, max_len: int = 8):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.heads = nn.ModuleList()
        for i in range(max_len):
            if i == 0:
                num_classes = len(PROVINCES)
            elif i == 1:
                num_classes = len(ALPHAS)
            else:
                num_classes = len(ADS)
            self.heads.append(nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        feat = self.backbone(x)
        logits = [head(feat) for head in self.heads]
        return logits


def compute_loss(
    logits: List[torch.Tensor],
    targets: torch.Tensor,
    criterion: nn.Module,
    head_weights: Optional[List[float]] = None,
):
    total_loss = torch.tensor(0.0, device=targets.device)
    total_chars = 0
    total_correct = 0

    for i, logit in enumerate(logits):
        t = targets[:, i]
        mask = t != -1
        if mask.sum() == 0:
            continue
        weight = head_weights[i] if head_weights and i < len(head_weights) else 1.0
        loss_i = criterion(logit[mask], t[mask]) * weight
        total_loss = total_loss + loss_i
        preds = logit[mask].argmax(dim=1)
        total_correct += (preds == t[mask]).sum().item()
        total_chars += mask.sum().item()

    return total_loss, total_correct, total_chars


def train_one_epoch(model, loader, optimizer, device, criterion, head_weights=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss, corr, chars = compute_loss(logits, labels, criterion, head_weights=head_weights)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += corr
        total += chars

    avg_loss = running_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device, criterion, head_weights=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss, corr, chars = compute_loss(logits, labels, criterion, head_weights=head_weights)
        running_loss += loss.item()
        correct += corr
        total += chars

    avg_loss = running_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc


def decode_indices(indices: Sequence[int]) -> str:
    chars = []
    for i, idx in enumerate(indices):
        if idx == -1:
            continue
        if i == 0:
            table = PROVINCES
        elif i == 1:
            table = ALPHAS
        else:
            table = ADS
        if 0 <= idx < len(table):
            chars.append(table[idx])
    return "".join(chars)


def parse_args():
    parser = argparse.ArgumentParser(description="CCPD2020 车牌识别训练与验证")
    parser.add_argument("--data_root", required=True, help="ccpd_green 根目录，包含 train/val/test")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=64, help="batch 大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--device", default="cuda:0", help="训练设备，如 cuda:0 或 cpu")
    parser.add_argument("--limit_train", type=int, default=None, help="限制训练集样本数量（调试用）")
    parser.add_argument("--limit_val", type=int, default=None, help="限制验证集样本数量（调试用）")
    parser.add_argument("--save_path", default="ccpd_recognition.pth", help="模型保存路径")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((96, 240)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CCPDDataset(
        root=os.path.join(args.data_root, "train"),
        transform=transform,
        limit=args.limit_train,
    )
    val_ds = CCPDDataset(
        root=os.path.join(args.data_root, "val"),
        transform=transform,
        limit=args.limit_val,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PlateNet(max_len=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 加权 province 位，提升汉字首位学习
    head_weights = [2.0] + [1.0] * 7

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, criterion, head_weights=head_weights
        )
        val_loss, val_acc = evaluate(
            model, val_loader, device, criterion, head_weights=head_weights
        )
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"Saved best model to {args.save_path} (val_acc={val_acc:.4f})")

    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()
