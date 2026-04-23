# CardioFit

基于穿戴式多模态生理信号（ECG + PPG + SCG）的动态心肺功能评测系统，用于无创估计：

- `VO2max`（最大摄氧量）
- `CO`（心输出量）

项目核心目标、信号规范、模型设计与开发阶段定义见 `TECHNICAL_DOC.md`。

## 当前仓库状态

本仓库当前以技术方案与研发规范为主，已沉淀：

- 项目上下文与开发约束：`CLAUDE.md`
- 完整技术规格：`TECHNICAL_DOC.md`
- 依赖清单（初版）：`requirements.txt`
- 协作规范：`CONTRIBUTING.md`
- 开发里程碑拆分：`ROADMAP.md`

## 技术栈（规划）

- Python 3.10+
- PyTorch + PyTorch Lightning
- Hydra + OmegaConf
- HDF5（h5py）
- 信号处理：scipy / neurokit2 / biosppy
- 实验管理：WandB

## 建议目录结构

```text
cardiofit/
├── configs/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── src/
└── tests/
```

## 快速开始（初始化环境）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Git 仓库规范

- 已包含标准仓库文件：`.gitignore`、`.editorconfig`、`.gitattributes`、`LICENSE`
- 已包含协作模板：`.github/ISSUE_TEMPLATE/`、`.github/pull_request_template.md`
- 已包含工程质量门禁：`.github/workflows/ci.yml`、`.pre-commit-config.yaml`
- 推荐首次提交前检查：

```bash
git status
pytest tests/ -v
```

### 本地启用 pre-commit

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 推荐开发顺序

1. 预处理管道（ECG/PPG/SCG）
2. 特征工程与跨模态特征
3. Dataset + DataLoader
4. 基线模型（XGBoost/传统方程）
5. 深度模型与融合策略
6. 评估、消融、可视化

## 关键约束

- 严格按受试者划分 train/val/test，避免数据泄露
- 仅在训练集执行增强与标准化拟合
- 输出范围校验：
  - VO2max: `[10, 80]`
  - CO: `[2, 30]`

## 常用命令（规划）

```bash
# 预处理
python scripts/preprocess_all.py --config configs/default.yaml

# 训练
python scripts/train.py --config configs/experiment/vo2max_multimodal.yaml

# 评估
python scripts/evaluate.py --checkpoint runs/best.ckpt

# 测试
pytest tests/ -v
```

## 文档

- 技术规格：`TECHNICAL_DOC.md`
- 项目上下文：`CLAUDE.md`
- 协作规范：`CONTRIBUTING.md`
- 路线图：`ROADMAP.md`
