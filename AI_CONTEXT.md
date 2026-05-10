# CardioFit — AI 工具共享上下文

> 三端启动时先读此文件。Claude Code / Codex / Antigravity 通用。

## 项目目标

基于 PPG + ECG + SCG 多模态穿戴式信号，使用深度学习方法（ResNet18-SE-LSTM）无创预测 **心输出量 (CO)** 和 **最大摄氧量 (VO₂max)**。

## 技术路线

```
原始信号 (ECG 500Hz, PPG 100Hz, SCG 800Hz)
  → 各自滤波去噪
  → 统一重采样至 125Hz
  → ECG R 峰检测 + 窗口提取 (125, 3)
  → ResNet18-SE-LSTM + 临床特征 → 双输出头
  → CO + VO₂max
```

## 当前进度

| Phase | 状态 | 负责人 | 分支 |
|-------|------|--------|------|
| 1. 环境搭建 | in_progress | — | — |
| 2. 预处理 | pending | — | — |
| 3. 数据加载 | pending | — | — |
| 4. 模型定义 | pending | — | — |
| 5. 训练脚本 | pending | — | — |
| 6. 评估管道 | pending | — | — |
| 7. ONNX导出 | pending | — | — |

## 架构决策记录

1. **框架选 TensorFlow/Keras** — 复用董雪论文的 ResNet18-SE-LSTM 实现
2. **统一采样率 125Hz** — 匹配论文架构，三种信号重采样到同频
3. **ECG R 峰对齐** — 论文用 PPG 波谷对齐，改为通用性更好的 R 峰
4. **三通道输入** (125, 3) = [ECG, PPG(IR), SCG_z] — 替代论文 [PPG, PPG', PPG'']
5. **双输出头** — 共享主干，分别输出 CO 和 VO₂
6. **HDF5 每受试者一个文件** — 内存友好，支持按受试者切分
7. **零硬编码路径** — 全部从 `configs/default.yaml` 读取

## 信号参数

| 信号 | 原始采样率 | 通道 | 滤波范围 | 备注 |
|------|-----------|------|----------|------|
| ECG | 500 Hz | 1 | 0.5-40 Hz + 50Hz陷波 | R峰对齐锚点 |
| PPG | 100 Hz | 2 (红+红外) | 0.5-8 Hz | 选择红外通道 |
| SCG | 800 Hz | 3 (xyz) | 1-40 Hz | 选择 z 轴 |

所有信号最终重采样到 125Hz，窗口大小 = 125 样本 (1秒)，R 峰前 40 + 后 85。

## 临床特征 (10维)

`[年龄, 性别(男1女0), 体重kg, 身高cm, BSA_m², BMI, HR_bpm, SBP_mmHg, DBP_mmHg, PP_mmHg]`

## 编码规范

- **路径**: `pathlib.Path` + YAML配置，禁止硬编码绝对路径
- **类型**: 所有公共函数必须有类型注解
- **文档**: Google 风格 docstring
- **日志**: `logging` 模块，禁止 `print`
- **配置**: 超参数通过 YAML 传入
- **数据泄露**: 按受试者切分，同一受试者不能跨 train/val/test

## 模块接口契约

### preprocessing/

```python
# ecg_processor.py
def process_ecg(raw: np.ndarray, fs: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """→ (signal_125hz, r_peaks_125hz)"""

# ppg_processor.py
def process_ppg(raw: np.ndarray, fs: int = 100, ch: int = 1) -> np.ndarray:
    """→ signal_125hz"""

# scg_processor.py
def process_scg(raw: np.ndarray, fs: int = 800, axis: int = 2) -> np.ndarray:
    """→ signal_125hz"""

# sync_and_segment.py
def extract_windows(ecg: np.ndarray, ppg: np.ndarray, scg: np.ndarray, r_peaks: np.ndarray, window_size: int = 125) -> np.ndarray:
    """→ (N_windows, 125, 3)"""

# build_hdf5.py
def build_subject_hdf5(raw_dir: Path, output_dir: Path, sid: str, clinical: dict) -> None:
    """处理单个受试者，写入 subject_{sid}.h5"""
```

### dataset/

```python
# hdf5_loader.py
def load_subject(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """→ (signals (N,125,3), clinical (N,10), co (N,1), vo2 (N,1))"""

# data_split.py
def split_subjects(subject_ids: list, ratios: tuple = (0.7, 0.15, 0.15), seed: int = 42) -> dict:
    """→ {'train': [...], 'val': [...], 'test': [...]}"""
```

### models/

```python
# resnet_se_lstm.py
def build_multimodal_resnet_se_lstm(
    input_shape: tuple = (125, 3),
    n_clinical: int = 10,
    l2_reg: float = 4.11e-05,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """→ compiled Keras Model with dual outputs ('co_output', 'vo2_output')"""
```

## 跨平台说明

| 平台 | 用途 | 环境 |
|------|------|------|
| MacBook M1 Pro | 开发、预处理调试、小批量验证 | tensorflow-metal |
| Windows WSL2 | GPU 正式训练 | tensorflow[and-cuda] |

依赖文件: `requirements-mac.txt` / `requirements-wsl.txt`

## 参考资源

- 论文模型: `~/code/codongxue/src/training/train_svco_model.py`
- 论文预处理: `~/code/codongxue/src/preprocessing/`
- 项目技术规格: `TECHNICAL_DOC.md`
