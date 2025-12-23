"""
本地网页版车牌检测+识别演示，支持三种检测方案：
- simple：轮廓法（默认，无需额外权重）
- enhanced：Sobel+形态学，召回更高
- yolo：使用 YOLO 权重进行车牌检测（需提供 --det_weights）

识别使用 train_ccpd.py 训练的 PlateNet，上传图片即可返回标注结果。
示例：
python web_app.py --weights ccpd_recognition_new.pth --device cuda:0 --port 8000 --detector yolo --det_weights runs/detect/train*/weights/best.pt
"""

from __future__ import annotations

import argparse
import base64
import io
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from flask import Flask, render_template_string, request

from train_ccpd import ADS, ALPHAS, PROVINCES, PlateNet, decode_indices


def simple_plate_detection(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """使用简单轮廓法返回若干疑似车牌框 (x1, y1, x2, y2)。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 2.0 < aspect_ratio < 6.0:
            boxes.append((x, y, x + w, y + h))
    return boxes


def enhanced_plate_detection(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Sobel+形态学，召回更高。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 800:
            continue
        aspect_ratio = w / float(h)
        if 2.0 < aspect_ratio < 6.5:
            boxes.append((x, y, x + w, y + h))
    return boxes


def load_yolo_model(weights: str, device: str = "cpu"):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("需要安装 ultralytics 才能使用 YOLO 检测：pip install ultralytics") from exc
    if not weights:
        raise ValueError("缺少 YOLO 权重路径 --det_weights")
    model = YOLO(weights)
    model.to(device)
    return model


def yolo_plate_detection(model, image: np.ndarray, device: str) -> List[Tuple[int, int, int, int]]:
    """使用 YOLO 返回车牌框。"""
    results = model.predict(image, verbose=False, device=device)[0]
    boxes = results.boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    dets = []
    for (x1, y1, x2, y2), score in zip(xyxy, confs):
        # 可按需要过滤置信度
        dets.append((x1, y1, x2, y2))
    return dets


def crop_plate(image: np.ndarray, box: Tuple[int, int, int, int], pad: int = 4) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2]


def load_model(weights: str, device: torch.device, max_len: int = 8) -> PlateNet:
    model = PlateNet(max_len=max_len).to(device)
    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_plate_text(model: PlateNet, image_rgb: np.ndarray, device: torch.device) -> Tuple[str, float]:
    """对车牌 RGB 图像进行识别，返回字符字符串与得分（log 概率和）。"""
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((96, 240)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_t = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_t)
        preds = []
        score = 0.0
        for logit in logits:
            logp = F.log_softmax(logit, dim=1)
            pred_idx = logp.argmax(dim=1).item()
            preds.append(pred_idx)
            score += logp[0, pred_idx].item()
    return decode_indices(preds), score


def encode_image_to_base64(image_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def create_app(model: PlateNet, device: torch.device, detector: str = "simple", det_model=None, det_device: str = "cpu"):
    app = Flask(__name__)

    TEMPLATE = """
    <!doctype html>
    <html lang="zh">
    <head>
        <meta charset="utf-8">
        <title>本地车牌识别演示</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 900px; margin: auto; }
            .preview { margin-top: 20px; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; padding: 6px; border-radius: 4px; }
            .result { font-size: 18px; margin-top: 12px; }
        </style>
    </head>
    <body>
    <div class="container">
        <h2>本地车牌识别演示</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">上传并识别</button>
        </form>
        {% if plate_text %}
        <div class="result">
            <b>识别结果：</b> {{ plate_text }}<br>
            <b>检测框数量：</b> {{ num_boxes }}
        </div>
        {% endif %}
        {% if image_b64 %}
        <div class="preview">
            <img src="data:image/jpeg;base64,{{ image_b64 }}" alt="result">
        </div>
        {% endif %}
    </div>
    </body>
    </html>
    """

    @app.route("/", methods=["GET", "POST"])
    def index():
        plate_text = None
        image_b64 = None
        num_boxes = 0

        if request.method == "POST":
            file = request.files.get("image")
            if file:
                data = file.read()
                np_data = np.frombuffer(data, np.uint8)
                bgr = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                if bgr is None:
                    plate_text = "无法读取图像"
                else:
                    # 检测方式选择
                    if detector == "yolo" and det_model is not None:
                        boxes = yolo_plate_detection(det_model, bgr, det_device)
                    elif detector == "enhanced":
                        boxes = enhanced_plate_detection(bgr)
                    else:
                        boxes = simple_plate_detection(bgr)
                    num_boxes = len(boxes)
                    if not boxes:
                        # 无检测框时，使用整图中心裁剪作为兜底，但标记为 0 个检测框
                        h, w = bgr.shape[:2]
                        boxes = [(int(0.1 * w), int(0.3 * h), int(0.9 * w), int(0.7 * h))]
                        num_boxes = 0
                    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                    x1, y1, x2, y2 = boxes[0]
                    crop_bgr = crop_plate(bgr, boxes[0])
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    plate_text, _ = predict_plate_text(model, crop_rgb, device)

                    annotated = bgr.copy()
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        plate_text,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    image_b64 = encode_image_to_base64(annotated)

        return render_template_string(
            TEMPLATE,
            plate_text=plate_text,
            image_b64=image_b64,
            num_boxes=num_boxes,
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="本地网页车牌识别演示")
    parser.add_argument("--weights", default="ccpd_recognition.pth", help="训练好的模型权重路径")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="端口")
    parser.add_argument("--device", default="cpu", help="推理设备，如 cpu 或 cuda:0")
    parser.add_argument(
        "--detector",
        choices=["simple", "enhanced", "yolo"],
        default="simple",
        help="车牌检测方式：simple/增强/YOLO（推荐 yolo，需提供 --det_weights）",
    )
    parser.add_argument("--det_weights", default=None, help="YOLO 车牌检测权重路径（detector=yolo 时必填）")
    parser.add_argument("--det_device", default=None, help="YOLO 推理设备，默认与 --device 一致")
    return parser.parse_args()


def main():
    args = parse_args()
    # 如果提供了 YOLO 权重而未显式指定检测器，自动切换到 yolo
    if args.det_weights and args.detector != "yolo":
        args.detector = "yolo"
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.weights, device=device)
    det_model = None
    det_device = args.det_device or args.device
    if args.detector == "yolo":
        det_model = load_yolo_model(args.det_weights, device=det_device)
    app = create_app(model, device, detector=args.detector, det_model=det_model, det_device=det_device)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
