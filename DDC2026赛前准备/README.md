# DDC2026 赛前准备实证

报名前（2026-11-09）在自己电脑上跑通的链路，非现场速成。48 小时现场将在此基础上新增机械臂集成、判断链路可视化与失败回放，见报名表"现场新增"部分。

## 1. YOLO 训练

ultralytics 训练 yolov8n（10 epochs，coco8 数据集，CPU 训练 22 秒）。

- `训练曲线-results.png` — loss 与 mAP 曲线
- `混淆矩阵.png`
- `验证集预测对照.jpg` — 标注 vs 预测
- 训练日志截图（Epoch / box_loss / cls_loss 表格）

## 2. ONNX 导出

best.pt → best.onnx（12.3 MB，opset 18），`ONNX: export success`。

见 `ONNX导出与算子统计.png`（导出成功日志 + 结构解析命令输出同屏）。

## 3. ONNX 结构核验

用 onnx 库程序化解析 best.onnx：

- opset: 18
- 输入: `images [1, 3, 640, 640]`（BCHW）
- 输出: `output0`
- 算子分布：Conv×64、Sigmoid×58、Mul×58、Concat×17、Reshape×8、Split×8、Add×8 等，共 15 类算子

Netron 可视化核验（局部高清）：

- `Netron结构-输入层.png` — images 输入经 Conv + Sigmoid + Mul 的 backbone 起始段
- `Netron结构-中间层.png` — Split / Conv / Sigmoid / Mul / Add 分支段
- `Netron结构-Concat层.png` — Concat 汇聚段，张量形状 1×64×80×80

## 待补（12/5 前）

地平线 OpenExplorer PTQ 量化 → 编译 `.bin` → RDK X5 板端推理。进场第一天就能接活，不占用现场时间补基础。

## 环境

Python 3.11.9 · torch 2.14.1+cpu · ultralytics 8.4.208 · onnx 1.23.2 · AMD Ryzen AI 9 365（CPU 训练）
