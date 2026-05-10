# CLAUDE.md — CardioFit 项目上下文

> **启动时先读 `AI_CONTEXT.md`** — 它是三端共享的完整项目上下文。本文件是 Claude Code 专用补充。

## 项目简介

基于 PPG + ECG + SCG 多模态信号，使用 ResNet18-SE-LSTM 预测 CO 和 VO₂max。参考论文: 董雪学位论文 (codongxue)。

## 与原始 CardioFit 脚手架的关键变更

- **框架**: PyTorch → TensorFlow/Keras (复用论文代码)
- **架构**: 三独立编码器 + 融合 → 论文单编码器 ResNet18-SE-LSTM
- **输入**: 三模态分别处理 → 统一重采样 125Hz 堆叠为 (125,3)
- **工程**: OmegaConf/Hydra → 简单 YAML + pathlib

## 项目结构

```
中期CO/
├── AI_CONTEXT.md                 ← 三端共享上下文 (先读这个!)
├── .tasks.yaml                   ← 任务认领表
├── configs/default.yaml          ← 唯一配置文件
├── src/cardiofit/
│   ├── preprocessing/            # ECG/PPG/SCG 处理器
│   ├── dataset/                  # HDF5 加载 + 数据切分
│   ├── models/                   # SE Block + ResNet18-SE-LSTM
│   ├── training/                 # 损失函数 + 回调
│   └── evaluation/               # 指标 + Bland-Altman
├── scripts/                      # CLI 入口
├── outputs/                      # checkpoints, logs, onnx
├── requirements-mac.txt
└── requirements-wsl.txt
```

## 编码规范

同 AI_CONTEXT.md。特别强调: pathlib, 类型注解, YAML配置, logging 非 print, 按受试者切分防泄露。

## 常用命令

```bash
# 预处理
python scripts/build_preprocessed_dataset.py --config configs/default.yaml

# 训练
python scripts/train_multimodal.py --config configs/default.yaml

# 评估
python scripts/evaluate_multimodal.py --config configs/default.yaml

# 导出 ONNX
python scripts/export_onnx.py --config configs/default.yaml
```

## 参考代码

- 模型核心: `~/code/codongxue/src/training/train_svco_model.py`
- 预处理: `~/code/codongxue/src/preprocessing/data_step1.py`
- ONNX推理: `~/code/codongxue/src/inference/infer_svco_onnx.py`

## 三端协作

- `.tasks.yaml` 做任务认领 — 开始前先读，改状态为 in_progress
- `AI_CONTEXT.md` 做上下文传递 — 启动时先读
- 按 Phase 切分支 — 一个分支一个 Phase，不绑定工具
