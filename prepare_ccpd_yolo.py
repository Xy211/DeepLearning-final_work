"""
将 CCPD2020（ccpd_green 划分）的文件名标注转换为 YOLO 所需的 label txt，并在 out_dir 下建立 images/labels 目录结构。

使用方式（在 DLFinal 目录）：
python prepare_ccpd_yolo.py --ccpd_root CCPD2020/CCPD2020/ccpd_green --out_dir ccpd_yolo

生成内容：
- ccpd_yolo/labels/train|val|test/*.txt 与图片一一对应（class 0，单类车牌）。
- ccpd_yolo/images/train|val|test/ 硬链接指向原图（无额外占用）。
- ccpd_yolo/dataset.yaml 供 Ultralytics YOLO 训练使用。

随后可用 YOLO 训练（示例，需安装 ultralytics）：
    yolo train model=yolov8n.pt data=ccpd_yolo/dataset.yaml epochs=100 imgsz=640 device=0 batch=32
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple


def parse_bbox_from_name(filename: str) -> Optional[Tuple[int, int, int, int]]:
    """
    从 CCPD 文件名解析车牌 bbox (x1, y1, x2, y2)。
    文件名格式例：01-90_265-231&522_405&574-...-xxxx.jpg
    bbox 在第三段（索引 2），形如 231&522_405&574
    """
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


def convert_split(split_dir: Path, labels_dir: Path, images_out_dir: Path):
    """
    为一个划分（train/val/test）生成 YOLO 标签文件。
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(split_dir.glob("*.jpg"))
    for img_path in images:
        bbox = parse_bbox_from_name(img_path.name)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        # YOLO 标注：cx, cy, w, h（归一化）
        # 需要用图像尺寸进行归一化
        import cv2  # 局部导入以避免全局依赖

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / float(w)
        bh = (y2 - y1) / float(h)
        label_path = labels_dir / (img_path.stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            # 单类车牌，class id = 0
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        # 在 out_dir/images 下建立硬链接，避免占用额外空间
        out_img = images_out_dir / img_path.name
        if not out_img.exists():
            try:
                os.link(img_path, out_img)
            except OSError:
                # Windows 某些情况下硬链接受限，退回复制
                shutil.copy2(img_path, out_img)


def write_dataset_yaml(out_dir: Path):
    yaml_path = out_dir / "dataset.yaml"
    content = f"""path: {out_dir.resolve()}
train: images/train
val: images/val
test: images/test
names:
  0: plate
"""
    yaml_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="将 CCPD 生成 YOLO 标注和数据集配置")
    parser.add_argument("--ccpd_root", required=True, help="ccpd_green 根目录（含 train/val/test）")
    parser.add_argument("--out_dir", default="ccpd_yolo", help="输出目录，用于存放 labels 和 dataset.yaml")
    args = parser.parse_args()

    ccpd_root = Path(args.ccpd_root)
    out_dir = Path(args.out_dir)
    labels_root = out_dir / "labels"
    images_root = out_dir / "images"

    for split in ["train", "val", "test"]:
        split_dir = ccpd_root / split
        labels_dir = labels_root / split
        images_dir = images_root / split
        convert_split(split_dir, labels_dir, images_dir)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    write_dataset_yaml(out_dir)
    print(f"转换完成，标签与配置位于: {out_dir.resolve()}")
    print("可使用 ultralytics 训练示例:")
    print(f"yolo train model=yolov8n.pt data={out_dir/'dataset.yaml'} epochs=100 imgsz=640 device=0 batch=32")


if __name__ == "__main__":
    main()
