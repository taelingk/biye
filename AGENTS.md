# AGENTS.md — Codex 行为指引

> **启动时必须先读 `AI_CONTEXT.md`** — 它是三端共享的完整项目上下文。本文件是 Codex 专用补充。

## 项目简介

基于 PPG + ECG + SCG 多模态信号，使用 TensorFlow/Keras 实现 ResNet18-SE-LSTM 预测 CO 和 VO₂max。

## 工作规则

1. 开始任何工作前，先 `git pull` 获取最新代码
2. 读取 `.tasks.yaml`，只处理人类已分配给你（`handler: codex`）且状态为 `in_progress` 的任务
3. **不要**自行认领 `status: free` 的任务 — 由人类分配
4. 完成后告知人类，由人类将 `status` 改为 `done`
5. **不要**修改其他工具正在处理的文件（检查 `.tasks.yaml` 中的 `files_touched`）

## 禁止事项

- 禁止修改 `.tasks.yaml` 中 `status: in_progress` 且 `handler` 不是 `codex` 的任务涉及的文件
- 禁止硬编码绝对路径，使用 `configs/default.yaml` 中的相对路径
- 禁止使用 `print`，使用 `logging` 模块
- 禁止引入 PyTorch 依赖 — 本项目使用 TensorFlow/Keras

## 编码规范

- **路径**: `pathlib.Path` + YAML 配置
- **类型**: 所有公共函数必须有类型注解
- **文档**: Google 风格 docstring
- **数据泄露**: 按受试者切分，同一受试者不能跨 train/val/test

详见 `AI_CONTEXT.md` 的"编码规范"章节。

## 常用命令

```bash
# 预处理
python scripts/build_preprocessed_dataset.py --config configs/default.yaml

# 训练
python scripts/train_multimodal.py --config configs/default.yaml

# 评估
python scripts/evaluate_multimodal.py --config configs/default.yaml

# 测试
pytest tests/ -v
```

## 项目结构

```
cardiofit/
├── AI_CONTEXT.md          ← 三端共享上下文 (先读这个!)
├── AGENTS.md              ← Codex 指引 (本文件)
├── CLAUDE.md              ← Claude Code 指引
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
└── tests/                 # 测试
```
