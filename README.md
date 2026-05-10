# CardioFit

基于穿戴式多模态生理信号（PPG + ECG + SCG）与深度学习的无创心肺功能预测系统。

**输入**: ECG (500Hz) + PPG (100Hz) + SCG (800Hz) 三模态信号 + 临床特征 (10维)
**输出**: 心输出量 CO (L/min) + 最大摄氧量 VO2max (mL/kg/min)
**模型**: ResNet18-SE-LSTM (基于董雪2025学位论文，改编为多模态+双输出)

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements-mac.txt   # Mac开发
pip install -r requirements-wsl.txt   # Windows GPU训练

# 2. 预处理数据
python scripts/build_preprocessed_dataset.py

# 3. 训练
python scripts/train_multimodal.py

# 4. 评估
python scripts/evaluate_multimodal.py --checkpoint outputs/checkpoints/best_model.keras

# 5. 导出ONNX
python scripts/export_onnx.py --checkpoint outputs/checkpoints/best_model.keras
```

## 项目结构

```
中期CO/
├── AI_CONTEXT.md              # 三端共享上下文
├── .tasks.yaml                # 任务认领表
├── configs/default.yaml       # 统一配置
├── src/cardiofit/
│   ├── preprocessing/         # ECG/PPG/SCG信号处理
│   ├── dataset/               # HDF5数据加载
│   ├── models/                # ResNet18-SE-LSTM
│   ├── training/              # 损失/回调
│   └── evaluation/            # 指标/Bland-Altman
├── scripts/                   # CLI入口
├── outputs/                   # checkpoints/logs/onnx
└── data/                      # raw/processed/splits (不入Git)
```

## 技术栈

- **框架**: TensorFlow/Keras 2.21 (复用论文代码)
- **信号处理**: scipy / neurokit2
- **数据格式**: HDF5 (每受试者一个文件)
- **配置管理**: YAML + pathlib (零硬编码路径)
- **跨平台**: Mac Apple Silicon 开发 + Windows WSL2 GPU 训练

## 参考

- 论文模型: [taelingk/codongxue](https://github.com/taelingk/codongxue)
- 技术规格: `TECHNICAL_DOC.md`
- 三端协作: `AI_CONTEXT.md`, `.tasks.yaml`
