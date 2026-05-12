# CLAUDE.md — CardioFit 项目上下文

> **启动时先读 `AI_CONTEXT.md`** — 它是三端共享的完整项目上下文。本文件是 Claude Code 专用补充。

## 项目简介

基于 PPG + ECG + SCG 多模态信号，使用 TensorFlow/Keras 实现 ResNet18-SE-LSTM 预测 CO 和 VO₂max。

## 框架

> ⚠️ **TensorFlow/Keras** — 禁止引入 PyTorch。

## 项目结构

```
cardiofit/
├── AI_CONTEXT.md          ← 三端共享上下文 (先读这个!)
├── AGENTS.md              ← Codex 指引
├── CLAUDE.md              ← Claude Code 指引 (本文件)
├── .tasks.yaml            ← 任务调度表 (权威进度来源)
├── TECHNICAL_DOC.md       ← 技术规格文档
├── configs/default.yaml   ← 唯一配置文件
├── src/cardiofit/
│   ├── preprocessing/     # ECG/PPG/SCG 处理器
│   ├── dataset/           # HDF5 加载 + 数据切分
│   ├── models/            # SE Block + ResNet18-SE-LSTM
│   ├── training/          # 损失函数 + 回调
│   ├── evaluation/        # 指标 + Bland-Altman
│   ├── features/          # 手工特征工程
│   └── utils/             # 工具函数
├── scripts/               # CLI 入口
├── tests/                 # 测试
├── requirements-mac.txt   # Mac 依赖
└── requirements-wsl.txt   # Windows WSL2 依赖
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

# 测试
pytest tests/ -v
```

## 三端协作

- `.tasks.yaml` 是任务调度的**权威来源** — 由人类手动分配任务
- `AI_CONTEXT.md` 做上下文传递 — 启动时先读
- 不要自行认领 `free` 任务，等人类分配并 `git push` 后再执行

## 参考资源

- 参考论文: 董雪学位论文 — ResNet18-SE-LSTM 架构
- 项目技术规格: `TECHNICAL_DOC.md`
