# CLAUDE.md — CardioFit 项目上下文

> 这是 Claude Code 的项目上下文文件。当你在本项目中工作时，请始终参考此文件。

## 项目简介

**CardioFit** 是一个基于穿戴式多模态生理信号（ECG + PPG + SCG）与深度学习的心肺功能评估系统。目标是通过贴片式可穿戴传感器采集的信号，无创估计 VO₂max（最大摄氧量）和 CO（心输出量）。

完整技术规格请阅读 `TECHNICAL_DOC.md`。

## 核心技术决策

- **语言**: Python 3.10+
- **深度学习框架**: PyTorch + PyTorch Lightning
- **配置管理**: OmegaConf + Hydra
- **数据存储**: HDF5 (h5py)
- **实验跟踪**: WandB
- **信号处理**: neurokit2, scipy, biosppy
- **机器学习基线**: scikit-learn, XGBoost, LightGBM
- **测试**: pytest
- **类型注解**: 所有公共函数必须有类型注解

## 项目结构速查

```
cardiofit/
├── configs/          # YAML 配置文件 (Hydra)
├── data/             # 数据目录 (不入 Git)
│   ├── raw/          # 原始采集数据
│   ├── processed/    # 预处理后 HDF5
│   └── splits/       # 训练/验证/测试划分
├── src/              # 主代码
│   ├── preprocessing/ # 信号预处理 (滤波/R峰/分段/同步)
│   ├── features/      # 特征工程
│   ├── dataset/       # PyTorch Dataset & DataLoader
│   ├── models/        # 深度学习模型
│   │   ├── backbone/  # 单模态编码器 (1D-CNN + BiLSTM/Transformer)
│   │   ├── fusion/    # 融合策略 (early/mid/late)
│   │   ├── heads/     # 回归头 (VO₂max, CO)
│   │   └── baselines/ # 基线模型 (FRIEND, SVM, XGBoost)
│   ├── training/      # 训练循环、损失、调度器
│   ├── evaluation/    # 指标、Bland-Altman、消融
│   └── utils/         # 通用工具
├── scripts/          # CLI 入口脚本
├── notebooks/        # Jupyter 探索性分析
└── tests/            # 单元测试
```

## 编码规范

1. **类型注解**: 所有函数签名必须有 type hints
2. **Docstring**: 使用 Google 风格 docstring
3. **日志**: 使用 `logging` 模块，禁止 `print`
4. **路径**: 使用 `pathlib.Path`，禁止字符串拼接路径
5. **配置**: 所有超参数通过 YAML 传入，禁止硬编码魔数
6. **测试**: 核心函数必须有单元测试
7. **数据泄露防范**: train/val/test 严格按受试者划分，同一受试者的数据只能在一个集合中

## 关键信号参数

| 信号 | 采样率 | 通道数 | 滤波范围 | 备注 |
|------|--------|--------|----------|------|
| ECG | 500 Hz | 1 | 0.5-40 Hz | 50Hz 陷波 |
| PPG | 100 Hz | 2 (红光+红外) | 0.5-8 Hz | 运动伪迹敏感 |
| SCG | 800 Hz | 3 (xyz) | 1-40 Hz | z轴为主轴 |

## 模型架构要点

- 三个独立的 1D-CNN + BiLSTM/Transformer **编码器** 分别处理 ECG/PPG/SCG
- 一个 MLP **编码器** 处理人口学特征 (age, sex, height, weight, BMI)
- **融合层** 合并多模态嵌入 (默认用中期注意力融合)
- 共享 FC 层后分两个**回归头**: VO₂max 和 CO
- 损失 = Huber(VO₂max) + Huber(CO)

## 常用命令

```bash
# 预处理
python scripts/preprocess_all.py --config configs/default.yaml

# 训练
python scripts/train.py --config configs/experiment/vo2max_multimodal.yaml

# 评估
python scripts/evaluate.py --checkpoint runs/best.ckpt

# 消融
python scripts/ablation_study.py

# 测试
pytest tests/ -v
```

## 开发优先级

1. `src/preprocessing/` — 信号处理管道 (最基础)
2. `src/features/` — 特征工程
3. `src/dataset/` — Dataset 和 DataLoader
4. `src/models/baselines/` — XGBoost 等基线 (验证管道)
5. `src/models/backbone/` + `fusion/` + `heads/` — 深度学习模型
6. `src/training/` — 训练循环
7. `src/evaluation/` — 评估与可视化

## 重要提醒

- 在真实数据到位前，请用 `generate_synthetic_data()` 合成数据进行开发测试
- SCG 信号在高运动负荷下质量很差，需要做好质量评估和降级处理
- 80 名受试者数据量有限，注意防过拟合 (Dropout 0.2-0.3, 早停, 数据增强)
- VO₂max 和 CO 存在物理约束关系 (Fick 原理)，可考虑加入物理约束损失
- 输出范围校验: VO₂max ∈ [10, 80], CO ∈ [2, 30]
