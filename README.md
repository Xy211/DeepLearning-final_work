# 深度学习车牌识别系统（DLFinal）

本项目实现了一个本地可运行的车牌检测与识别系统，支持训练、推理与网页演示。

## 目录结构

- `train_ccpd.py`：车牌识别模型训练脚本（PlateNet，多头 ResNet18）。
- `prepare_ccpd_yolo.py`：将 CCPD 标签转换为 YOLO 检测训练格式。
- `web_app.py`：本地网页演示（上传图片，显示检测框与识别结果）。
- `课程测试报告.docx`：课程测试报告（含可视化与截图）。

## 环境依赖

- Python 3.8+（建议 3.10+）
- 主要依赖：`torch`、`torchvision`、`opencv-python`、`flask`、`numpy`
- 使用 YOLO 检测时需要：`ultralytics`

## 数据准备

- 数据集：CCPD2020（使用 `ccpd_green` 划分）。
- 目录结构参考：`CCPD2020/CCPD2020/ccpd_green/{train,val,test}`。

## 车牌识别训练（`train_ccpd.py`）

- 在 CCPD2020 的 `ccpd_green` 划分上训练多头 ResNet18 分类器（支持 8 位新能源车牌），从文件名解析标签与车牌框，裁剪后逐位交叉熵训练。
- CPU 调试示例：`python train_ccpd.py --data_root CCPD2020/CCPD2020/ccpd_green --epochs 1 --limit_train 200 --limit_val 50 --device cpu`
- GPU 全量示例：`python train_ccpd.py --data_root CCPD2020/CCPD2020/ccpd_green --epochs 15 --batch_size 128 --device cuda:0`
- 最优模型自动保存到 `ccpd_recognition.pth`（可用 `--save_path` 指定）。

## 车牌检测训练（YOLO）

1) 生成 YOLO 数据：
`python prepare_ccpd_yolo.py --ccpd_root CCPD2020/CCPD2020/ccpd_green --out_dir ccpd_yolo`

2) 训练 YOLO：
`yolo train model=yolov8n.pt data=ccpd_yolo/dataset.yaml epochs=100 imgsz=640 device=0 batch=32`

## 本地网页演示（`web_app.py`）

- 使用已训练的 CCPD 模型，可选三种检测：`simple`（默认轮廓法）、`enhanced`（Sobel+形态学）、`yolo`（需提供车牌 YOLO 权重，推荐更精准）。
- 若提供 `--det_weights` 则自动切到 YOLO 检测。启动示例：
`python web_app.py --weights ccpd_recognition.pth --device cuda:0 --det_weights runs/detect/train4/weights/best.pt --det_device cuda:0 --port 5000`
- 仅用简单检测示例：
`python web_app.py --weights ccpd_recognition.pth --device cpu --detector simple --port 5000`
- 打开浏览器访问 `http://127.0.0.1:5000`，上传图片即可查看检测框与识别文字。

## 课程报告

- Word：`课程测试报告.docx`（含系统截图与可视化）

