# 深度学习软件系统设计

本目录包含若干可在本地运行的小示例（按需安装依赖）。

数据集采用[CCPD2020](https://github.com/detectRecog/CCPD)

## CCPD 车牌识别训练（`train_ccpd.py`）

- 在 CCPD2020 的 `ccpd_green` 划分上训练多头 ResNet18 分类器（支持 7/8 位新能源车牌），从文件名解析标签与车牌框，裁剪后逐位交叉熵训练。
- CPU 调试示例：`python train_ccpd.py --data_root CCPD2020/CCPD2020/ccpd_green --epochs 1 --limit_train 200 --limit_val 50 --device cpu`
- GPU 全量示例：`python train_ccpd.py --data_root CCPD2020/CCPD2020/ccpd_green --epochs 15 --batch_size 128 --device cuda:0`
- 最优模型自动保存到 `ccpd_recognition.pth`（可用 `--save_path` 指定）。

## 本地网页程序（`web_app.py`）

- 使用已训练的 CCPD 模型与可选的轮廓检测器，提供本地 Flask 页面上传图片并显示标注结果。
- 启动示例：`python web_app.py --weights ccpd_recognition.pth --device cpu --port 5000`
- 打开浏览器访问 `http://127.0.0.1:5000`，上传图片即可查看检测框与识别文字。
