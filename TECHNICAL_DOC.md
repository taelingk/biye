# 基于穿戴式多模态生理信号的动态心肺功能评测系统 — 技术文档

> **项目代号**: CardioFit
> **版本**: v2.0
> **框架**: TensorFlow / Keras（禁止 PyTorch）
> **最后更新**: 2026-05
> **作者**: 郭凌宇 / 北航生物与医学工程学院

---

## 1. 项目概述

### 1.1 目标

构建一个基于穿戴式多模态生理信号（ECG + PPG + SCG）与深度学习的心肺功能动态评估系统，实现对 **峰值摄氧量（VO₂max）** 与 **心输出量（CO）** 的准确、无创、连续估计。

### 1.2 核心输入/输出

| 维度 | 说明 |
|------|------|
| **输入信号** | 单导联 ECG、双波长 PPG（红光+红外）、三轴 SCG（MEMS 加速度计） |
| **辅助输入** | 人口统计学特征：年龄、性别、身高、体重、BMI |
| **输出目标 1** | VO₂max (mL·kg⁻¹·min⁻¹) — 心肺耐力核心指标 |
| **输出目标 2** | CO (L·min⁻¹) — 心输出量 |
| **金标准** | VO₂max → CPET 呼吸代谢仪；CO → CNAP 无创心输出量监测 |

### 1.3 性能目标

- VO₂max 预测：相关系数 r ≥ 0.9，MAPE ≤ 10%
- CO 预测：相关系数 r ≥ 0.9，误差 < ±10%
- Bland-Altman 一致性分析通过

---

## 2. 系统架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    CardioFit System Architecture                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │  Wearable    │    │  Data        │    │  Deep Learning     │  │
│  │  Hardware    │───▶│  Pipeline    │───▶│  Model Engine      │  │
│  │  (Patch)     │    │              │    │                    │  │
│  └─────────────┘    └──────────────┘    └────────┬───────────┘  │
│        │                   │                      │              │
│        │                   │                      ▼              │
│  ┌─────▼─────┐    ┌───────▼──────┐    ┌────────────────────┐  │
│  │ BLE/SD    │    │  Dataset     │    │  Evaluation &      │  │
│  │ Transfer  │    │  Manager     │    │  Visualization     │  │
│  └───────────┘    └──────────────┘    └────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 项目目录结构

```
cardiofit/
├── README.md
├── pyproject.toml                  # 项目配置 (使用 uv/poetry)
├── requirements.txt
├── configs/
│   ├── default.yaml                # 默认超参数
│   ├── experiment/                 # 不同实验配置
│   │   ├── vo2max_ecg_only.yaml
│   │   ├── vo2max_multimodal.yaml
│   │   ├── co_estimation.yaml
│   │   └── ablation_study.yaml
│   └── hardware/
│       └── sensor_spec.yaml        # 传感器参数定义
│
├── data/
│   ├── raw/                        # 原始采集数据 (不入版本控制)
│   │   ├── subject_001/
│   │   │   ├── ecg.csv
│   │   │   ├── ppg.csv
│   │   │   ├── scg.csv
│   │   │   ├── cpet_vo2.csv
│   │   │   ├── cnap_co.csv
│   │   │   └── metadata.json       # 人口学信息
│   │   └── ...
│   ├── processed/                  # 预处理后数据
│   ├── splits/                     # 训练/验证/测试划分
│   │   ├── train_subjects.txt
│   │   ├── val_subjects.txt
│   │   └── test_subjects.txt
│   └── README.md                   # 数据字典
│
├── src/cardiofit/
│   ├── __init__.py
│   │
│   ├── preprocessing/              # 信号预处理模块
│   │   ├── __init__.py
│   │   ├── ecg_processor.py        # 滤波、R峰检测、重采样至125Hz
│   │   ├── ppg_processor.py        # 滤波、重采样至125Hz
│   │   ├── scg_processor.py        # 滤波、选z轴、重采样至125Hz
│   │   ├── sync_and_segment.py     # R峰对齐 + 窗口提取 → (N,125,3)
│   │   └── build_hdf5.py           # 单受试者 → HDF5 完整管道
│   │
│   ├── features/                   # 特征工程模块 (基线模型用)
│   │   ├── __init__.py
│   │   ├── ecg_features.py         # HR, HRV, QRS duration
│   │   ├── ppg_features.py         # PAT, 脉搏幅度, SDPPG
│   │   ├── scg_features.py         # STI, Tei, 幅值, 频域
│   │   └── cross_modal.py          # PTT, EMD 跨模态特征
│   │
│   ├── dataset/                    # 数据集管理模块
│   │   ├── __init__.py
│   │   ├── hdf5_loader.py          # HDF5 加载 → (signals, clinical, co, vo2)
│   │   ├── data_split.py           # 按受试者切分 train/val/test
│   │   └── augmentation.py         # 数据增强 (时移、加噪)
│   │
│   ├── models/                     # 深度学习模型 (TensorFlow/Keras)
│   │   ├── __init__.py
│   │   ├── resnet_se_lstm.py       # 主模型构建函数 + 编译函数
│   │   ├── se_block.py             # SEBlock + ResidualSEBlock
│   │   └── standardize_layers.py   # 内置标准化层 (ONNX兼容)
│   │
│   ├── training/                   # 训练管理模块
│   │   ├── __init__.py
│   │   ├── losses.py               # Huber 双任务损失
│   │   └── callbacks.py            # 早停、模型保存回调
│   │
│   ├── evaluation/                 # 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py              # R², RMSE, MAE, MAPE, Pearson r
│   │   ├── bland_altman.py         # Bland-Altman 一致性分析
│   │   └── visualization.py        # 结果可视化
│   │
│   └── utils/
│       └── __init__.py             # 种子管理等工具
│
├── scripts/
│   ├── build_preprocessed_dataset.py  # 批量预处理入口
│   ├── train_multimodal.py            # 训练入口
│   ├── evaluate_multimodal.py         # 评估入口
│   ├── export_onnx.py                 # ONNX 导出
│   └── verify_pipeline.py             # 全管线验证
│
├── tests/                          # 单元测试
│
├── configs/
│   └── default.yaml                # 唯一配置文件
│
└── docs/                           # 分析文档
```

---

## 4. 数据规范

### 4.1 原始信号格式

每位受试者一个目录 `subject_XXX/`，包含以下文件：

#### `metadata.json`
```json
{
  "subject_id": "S001",
  "age": 25,
  "sex": "M",
  "height_cm": 175.0,
  "weight_kg": 70.0,
  "bmi": 22.86,
  "resting_hr": 68,
  "smoking": false,
  "activity_level": "moderate",
  "medical_history": [],
  "test_date": "2026-03-15",
  "test_protocol": "ramp_cycle_ergometer"
}
```

#### 信号文件 CSV 格式

| 文件 | 采样率 | 列名 | 单位 |
|------|--------|------|------|
| `ecg.csv` | 500 Hz | `timestamp_ms`, `ecg_mv` | 毫秒, 毫伏 |
| `ppg.csv` | 100 Hz | `timestamp_ms`, `ppg_red`, `ppg_ir` | 毫秒, 任意单位 |
| `scg.csv` | 800 Hz | `timestamp_ms`, `acc_x`, `acc_y`, `acc_z` | 毫秒, mg |
| `cpet_vo2.csv` | 逐呼吸 | `timestamp_ms`, `vo2_ml_kg_min`, `vco2`, `ve`, `rr`, `hr` | — |
| `cnap_co.csv` | 逐搏 | `timestamp_ms`, `co_l_min`, `sv_ml`, `hr`, `map_mmhg` | — |

### 4.2 预处理后数据格式

预处理管道输出 HDF5 文件，每位受试者一个：

```
subject_001.h5
├── raw/
│   ├── ecg          # shape: (N_samples,)
│   ├── ppg_red      # shape: (N_samples,)
│   ├── ppg_ir       # shape: (N_samples,)
│   ├── scg_xyz      # shape: (N_samples, 3)
│   └── timestamps   # shape: (N_samples,) 统一时间戳
├── beats/
│   ├── ecg_beats    # shape: (N_beats, beat_len_ecg)
│   ├── ppg_beats    # shape: (N_beats, beat_len_ppg, 2)
│   ├── scg_beats    # shape: (N_beats, beat_len_scg, 3)
│   └── beat_times   # shape: (N_beats,) 每个心搏的起始时间
├── features/
│   ├── ecg_feat     # shape: (N_beats, D_ecg)
│   ├── ppg_feat     # shape: (N_beats, D_ppg)
│   ├── scg_feat     # shape: (N_beats, D_scg)
│   └── cross_feat   # shape: (N_beats, D_cross)
├── labels/
│   ├── vo2max       # scalar: mL·kg⁻¹·min⁻¹
│   ├── vo2_timeseries  # shape: (N_timepoints, 2) [time, vo2]
│   ├── co_timeseries   # shape: (N_timepoints, 2) [time, co]
│   └── exercise_stage  # shape: (N_beats,) 0=rest/1=warmup/2=load/3=peak/4=recovery
└── metadata/           # 人口学信息 (attrs)
```

### 4.3 数据划分策略

- **留出法**: 70% 训练 / 15% 验证 / 15% 测试（按受试者划分，避免数据泄露）
- **交叉验证**: 5-fold CV（按受试者分组）
- **LOSO**: Leave-One-Subject-Out 用于泛化评估
- 训练/验证/测试集需性别、年龄、BMI 分布均衡

---

## 5. 信号预处理管道

### 5.1 总体流程

```
原始信号 → 滤波去噪 → 信号质量评估 → 多模态同步 → R峰检测 → 心搏分段 → 特征提取
```

### 5.2 各信号处理细节

#### ECG 处理 (`ecg_processor.py`)

```python
class ECGProcessor:
    """ECG信号预处理与特征提取"""

    def __init__(self, fs: int = 500):
        self.fs = fs

    def preprocess(self, ecg_raw: np.ndarray) -> np.ndarray:
        """
        预处理流程:
        1. 带通滤波: 0.5-40 Hz (4阶 Butterworth)
        2. 陷波滤波: 50 Hz 工频干扰
        3. 基线漂移校正: 中值滤波法 (200ms + 600ms 窗)
        """
        pass

    def detect_r_peaks(self, ecg_filtered: np.ndarray) -> np.ndarray:
        """
        R峰检测算法:
        - 主选: Pan-Tompkins 算法
        - 备选: Hamilton, neurokit2 内置检测器
        返回: R峰位置索引数组
        """
        pass

    def extract_hrv(self, r_peaks: np.ndarray) -> dict:
        """
        HRV 特征提取:
        - 时域: SDNN, RMSSD, pNN50, mean_RR
        - 频域: LF, HF, LF/HF ratio
        - 非线性: SD1, SD2 (Poincaré)
        """
        pass

    def segment_beats(self, ecg: np.ndarray, r_peaks: np.ndarray,
                      pre_ms: int = 200, post_ms: int = 400) -> np.ndarray:
        """按R峰分割单个心搏，固定窗长"""
        pass
```

#### PPG 处理 (`ppg_processor.py`)

```python
class PPGProcessor:
    """PPG信号预处理与特征提取"""

    def __init__(self, fs: int = 100):
        self.fs = fs

    def preprocess(self, ppg_raw: np.ndarray) -> np.ndarray:
        """
        1. 带通滤波: 0.5-8 Hz
        2. 运动伪迹去除: 自适应滤波 (加速度参考)
        3. 归一化
        """
        pass

    def detect_peaks_and_onsets(self, ppg: np.ndarray) -> tuple:
        """检测 PPG 波峰和波谷（起始点）"""
        pass

    def extract_features(self, ppg: np.ndarray, peaks, onsets) -> dict:
        """
        PPG 特征:
        - 脉搏上升时间 (Pulse Arrival Time, PAT)
        - 脉搏幅度及其变化率
        - 峰-谷比
        - 二次导数特征 (SDPPG): a, b, c, d, e 波 → 血管硬度指标
        - SpO2 估算 (红光/红外比值)
        """
        pass
```

#### SCG 处理 (`scg_processor.py`)

```python
class SCGProcessor:
    """SCG信号预处理与心脏机械事件识别"""

    def __init__(self, fs: int = 800):
        self.fs = fs

    def preprocess(self, scg_xyz: np.ndarray) -> np.ndarray:
        """
        1. 选择主轴 (通常为 z 轴 / 背腹方向)，或合成三轴
        2. 带通滤波: 1-40 Hz (心脏振动频段)
        3. 呼吸去除: 低频分离 (< 0.5 Hz)
        4. 模板匹配集合平均 (ensemble averaging) 提升 SNR
        """
        pass

    def detect_ao_ac(self, scg: np.ndarray, r_peaks: np.ndarray) -> dict:
        """
        识别关键心脏机械事件特征点:
        - AO: 主动脉瓣开启 (收缩期最大正加速)
        - AC: 主动脉瓣关闭
        - MC: 二尖瓣关闭
        - MO: 二尖瓣开启

        方法: 基于R峰的窗口搜索 + 模板匹配
        """
        pass

    def compute_sti(self, r_peaks, ao_points, ac_points, mc_points) -> dict:
        """
        计算收缩时间间隔 (STI):
        - IVCT: 等容收缩时间 (MC → AO)
        - LVET: 左室射血时间 (AO → AC)
        - IVRT: 等容舒张时间 (AC → MO)
        - QS2: 电机械收缩期 (Q波 → AC)
        - PEP: 射血前期 (Q波 → AO)
        - Tei指数: (IVCT + IVRT) / LVET
        """
        pass

    def extract_amplitude_features(self, scg_beats: np.ndarray) -> dict:
        """
        幅值特征:
        - AO峰幅值, AC峰幅值
        - 峰-峰值 (peak-to-peak)
        - 充盈期振幅 (diastolic amplitude)
        - RMS 能量
        """
        pass

    def extract_frequency_features(self, scg_beats: np.ndarray) -> dict:
        """
        频域特征:
        - 功率谱密度 (PSD) 在不同频段的分布
        - 主频率
        - 频谱熵
        """
        pass
```

#### 多模态同步 (`sync.py`)

```python
class MultiModalSync:
    """多模态信号时间对齐"""

    def align_signals(self, ecg, ppg, scg,
                      ecg_fs, ppg_fs, scg_fs,
                      target_fs: int = 500) -> dict:
        """
        统一时间轴与采样率:
        1. 以ECG的R峰为锚点
        2. 将PPG/SCG重采样或插值到统一采样率
        3. 基于心搏级别对齐 (beat-level alignment)
        """
        pass
```

### 5.3 信号质量评估 (`quality.py`)

```python
class SignalQualityAssessor:
    """信号质量评估，剔除低质量片段"""

    def assess_ecg_quality(self, ecg_beat: np.ndarray) -> float:
        """
        ECG质量评分 (0-1):
        - 模板相关性 (与集合平均模板的相关系数)
        - SNR 估计
        - 异常RR间期检测
        阈值: < 0.6 判定为低质量，丢弃
        """
        pass

    def assess_scg_quality(self, scg_beat: np.ndarray) -> float:
        """
        SCG质量评分:
        - 波形形态学评估
        - 呼吸伪迹强度
        - 运动伪迹检测 (加速度突变)
        """
        pass

    def assess_ppg_quality(self, ppg_beat: np.ndarray) -> float:
        """PPG 信号质量（饱和、脱落、运动等）"""
        pass
```

---

## 6. 特征工程

### 6.1 完整特征表

| 类别 | 特征名 | 来源 | 维度 | 生理意义 |
|------|--------|------|------|----------|
| **ECG 时域** | HR | ECG | 1 | 即时心率 |
| | SDNN | ECG | 1 | 心率变异性 (自主神经) |
| | RMSSD | ECG | 1 | 副交感活性 |
| | pNN50 | ECG | 1 | 副交感张力 |
| | QRS_duration | ECG | 1 | 传导速度 |
| **ECG 频域** | LF_power | ECG | 1 | 交感+副交感 |
| | HF_power | ECG | 1 | 副交感 |
| | LF_HF_ratio | ECG | 1 | 自主神经平衡 |
| **PPG** | PAT | PPG | 1 | 脉搏到达时间 |
| | pulse_amplitude | PPG | 1 | 脉搏幅度 |
| | rise_time | PPG | 1 | 上升时间 |
| | SDPPG_ratio | PPG | 1 | 血管硬度 |
| | SpO2 | PPG | 1 | 血氧饱和度 |
| **SCG 时域** | IVCT | SCG | 1 | 等容收缩时间 |
| | LVET | SCG | 1 | 射血时间 |
| | IVRT | SCG | 1 | 等容舒张时间 |
| | PEP | SCG | 1 | 射血前期 |
| | Tei_index | SCG | 1 | 全局心功能指标 |
| **SCG 幅值** | AO_amplitude | SCG | 1 | 主动脉瓣开启力度 |
| | AC_amplitude | SCG | 1 | 主动脉瓣关闭 |
| | diastolic_amp | SCG | 1 | 舒张期充盈振幅 |
| | peak_to_peak | SCG | 1 | 峰峰值 |
| | rms_energy | SCG | 1 | 均方根能量 |
| **SCG 频域** | dominant_freq | SCG | 1 | 主频率 |
| | spectral_entropy | SCG | 1 | 频谱复杂度 |
| | band_power_5_15 | SCG | 1 | 5-15Hz 功率 |
| | band_power_15_25 | SCG | 1 | 15-25Hz 功率 |
| **跨模态** | PTT | ECG+PPG | 1 | 脉搏传导时间 (R-PPG峰) |
| | EMD | ECG+SCG | 1 | 电机械延迟 |
| **人口学** | age | — | 1 | 年龄 |
| | sex | — | 1 | 性别 (0/1) |
| | height | — | 1 | 身高 |
| | weight | — | 1 | 体重 |
| | bmi | — | 1 | BMI |

**手工特征总维度**: 约 30-35 维

---

## 7. 深度学习模型架构

> ⚠️ 框架: **TensorFlow / Keras**。以下代码与 `src/cardiofit/models/` 中的实际实现一致。

### 7.1 主模型: ResNet18-SE-LSTM（双输出）

```
Signal (None, 125, 3)                Clinical (None, 10)
  [ECG, PPG_IR, SCG_z]              [age,sex,weight,height,BSA,BMI,HR,SBP,DBP,PP]
        │                                    │
        ▼                                    ▼
  StandardizeSignalFlat              StandardizeClinical
        │                                    │
        ▼                                    │
  Conv1D(64, 7, s=2) → BN → ReLU            │
        │                                    │
        ▼                                    │
  ResidualSEBlock × 2 (64 filters)           │
        │                                    │
        ▼                                    │
  ResidualSEBlock × 2 (128 filters, s=2)     │
        │                                    │
        ▼                                    │
  ResidualSEBlock × 2 (256 filters, s=2)     │
        │                                    │
        ▼                                    │
  ResidualSEBlock × 1 (512 filters)          │
        │                                    │
        ▼                                    │
  LSTM(64)                                   │
        │                                    │
        ▼                                    │
  Dense(128, relu) + Dropout(0.3)            │
        │                                    │
        └──────── Concatenate ───────────────┘
                       │
                       ▼
                Dense(64, relu) + Dropout(0.2)
                       │
              ┌────────┼────────┐
              ▼                 ▼
        Dense(1)          Dense(1)
        co_output         vo2_output
```

### 7.2 模块详细设计

#### 7.2.1 SE 通道注意力模块 (`se_block.py`)

```python
class SEBlock(Layer):
    """Squeeze-and-Excitation 通道注意力 (参考董雪论文)

    GlobalAveragePooling → Dense(C//r, relu) → Dense(C, sigmoid) → channel-wise multiply
    """
    def __init__(self, reduction_ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = GlobalAveragePooling1D()
        self.excitation1 = Dense(channels // self.reduction_ratio, activation="relu")
        self.excitation2 = Dense(channels, activation="sigmoid")

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.excitation1(x)
        x = self.excitation2(x)
        x = Reshape((1, -1))(x)
        return Multiply()([inputs, x])
```

#### 7.2.2 残差 SE 块 (`se_block.py`)

```python
class ResidualSEBlock(Layer):
    """ResNet 残差块 + SE 注意力

    Conv1D(3,s) → BN → ReLU → Conv1D(3,1) → BN → SE → Add → ReLU
    可选 1×1 conv shortcut 用于维度匹配
    """
    def __init__(self, filters, stride=1, use_1x1_conv=False,
                 kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv1D(filters, 3, padding="same", strides=stride,
                           kernel_regularizer=kernel_regularizer)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv1D(filters, 3, padding="same",
                           kernel_regularizer=kernel_regularizer)
        self.bn2 = BatchNormalization()
        self.se = SEBlock()

        self.shortcut_conv = None
        if use_1x1_conv:
            self.shortcut_conv = Conv1D(filters, 1, strides=stride,
                                       kernel_regularizer=kernel_regularizer)
            self.shortcut_bn = BatchNormalization()

    def call(self, inputs):
        x = Activation("relu")(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_bn(self.shortcut_conv(inputs))
        else:
            shortcut = inputs
        return Activation("relu")(x + shortcut)
```

#### 7.2.3 主模型构建函数 (`resnet_se_lstm.py`)

```python
def build_multimodal_resnet_se_lstm(
    input_shape: tuple[int, int] = (125, 3),
    n_clinical: int = 10,
    l2_reg: float = 4.11e-05,
    dropout: float = 0.3,
    dropout_shared: float = 0.2,
    dense_units: int = 128,
    lstm_units: int = 64,
    se_reduction: int = 8,
    signal_mean=None, signal_scale=None,
    clinical_mean=None, clinical_scale=None,
) -> tf.keras.Model:
    """构建 ResNet18-SE-LSTM 双输出模型（内置标准化层）

    输入:
        signal_input: (None, 125, 3)  — [ECG, PPG_IR, SCG_z] 三通道堆叠
        clinical_input: (None, 10)    — 10维临床参数

    输出:
        co_output: (None, 1)   — 心输出量预测
        vo2_output: (None, 1)  — VO₂max 预测
    """
    signal_input = Input(shape=input_shape, name="signal_input")
    clinical_input = Input(shape=(n_clinical,), name="clinical_input")

    # --- 内置标准化 (ONNX 导出时自包含) ---
    x = StandardizeSignalFlat(...)(signal_input)
    clin_std = StandardizeClinical(...)(clinical_input)

    # --- ResNet18 骨干 ---
    x = Conv1D(64, 7, strides=2, padding="same", name="conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    for i in range(2):
        x = ResidualSEBlock(64, name=f"res_se_64_{i}")(x)

    x = ResidualSEBlock(128, stride=2, use_1x1_conv=True)(x)
    x = ResidualSEBlock(128)(x)

    x = ResidualSEBlock(256, stride=2, use_1x1_conv=True)(x)
    x = ResidualSEBlock(256)(x)

    x = ResidualSEBlock(512, stride=1, use_1x1_conv=True)(x)

    # --- LSTM 时序建模 ---
    x = LSTM(lstm_units, return_sequences=False)(x)

    # --- 特征层 ---
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)

    # --- 拼接临床特征 ---
    x = Concatenate()([x, clin_std])  # (None, 138)

    # --- 共享层 ---
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout_shared)(x)

    # --- 双输出头 ---
    co_output = Dense(1, name="co_output")(x)
    vo2_output = Dense(1, name="vo2_output")(x)

    return Model(inputs=[signal_input, clinical_input],
                 outputs=[co_output, vo2_output],
                 name="resnet18_se_lstm_multimodal")
```

#### 7.2.4 模型编译 (`resnet_se_lstm.py`)

```python
def compile_model(model, learning_rate=0.001, clipvalue=1.0):
    """Huber 双损失 + Adam 梯度裁剪"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipvalue=clipvalue),
        loss={
            "co_output": tf.keras.losses.Huber(delta=1.0),
            "vo2_output": tf.keras.losses.Huber(delta=1.0),
        },
        loss_weights={"co_output": 1.0, "vo2_output": 1.0},
        metrics={"co_output": ["mae"], "vo2_output": ["mae"]},
    )
    return model
```

---

## 8. 训练配置

### 8.1 默认超参数 (`configs/default.yaml`)

> 以下为实际使用的配置文件结构，完整内容见 `configs/default.yaml`。

```yaml
project:
  name: CardioFit
  seed: 42

paths:
  raw_data: data/raw
  processed_data: data/processed
  checkpoints: outputs/checkpoints
  logs: outputs/logs
  onnx: outputs/onnx

signal:
  sampling_rate_target: 125   # 统一重采样目标 (Hz)
  window_size: 125            # 窗口样本数 (1秒)
  window_before_r: 40
  window_after_r: 85
  channels: 3                 # [ECG, PPG_IR, SCG_z]

model:
  input_shape: [125, 3]
  n_clinical: 10
  lstm_units: 64
  dense_units: 128
  dropout: 0.3
  dropout_shared: 0.2
  l2_reg: 4.11e-05
  se_reduction: 8

training:
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 20
  optimizer: adam
  loss: huber
  huber_delta: 1.0
  loss_weights:
    co_output: 1.0
    vo2_output: 1.0
  clipvalue: 1.0

data_split:
  ratios: [0.7, 0.15, 0.15]
  split_method: per_subject
```

### 8.2 损失函数

损失函数通过 Keras 内置的 `model.compile()` 配置，而非自定义类：

```python
# 在 compile_model() 中配置
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001, clipvalue=1.0),
    loss={
        "co_output": tf.keras.losses.Huber(delta=1.0),
        "vo2_output": tf.keras.losses.Huber(delta=1.0),
    },
    loss_weights={"co_output": 1.0, "vo2_output": 1.0},
    metrics={"co_output": ["mae"], "vo2_output": ["mae"]},
)
```

---

## 9. 评估框架

### 9.1 指标定义 (`metrics.py`)

```python
def compute_metrics(y_true, y_pred) -> dict:
    """计算全套回归评估指标"""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "pearson_r": np.corrcoef(y_true, y_pred)[0, 1],
        "bias": np.mean(y_pred - y_true),          # 系统偏差
        "loa_upper": ...,                            # 一致性上限
        "loa_lower": ...,                            # 一致性下限
    }
```

### 9.2 Bland-Altman 分析 (`bland_altman.py`)

```python
def bland_altman_analysis(measured, predicted, title="VO2max"):
    """
    绘制 Bland-Altman 一致性图:
    - X轴: (measured + predicted) / 2
    - Y轴: predicted - measured
    - 标注: mean bias, ±1.96 SD limits of agreement
    - 输出: 图像 + 统计量字典
    """
    pass
```

### 9.3 消融实验 (`ablation.py`)

```python
ABLATION_CONFIGS = [
    # 单模态实验
    {"name": "ECG_only",     "use_ecg": True,  "use_ppg": False, "use_scg": False},
    {"name": "PPG_only",     "use_ecg": False, "use_ppg": True,  "use_scg": False},
    {"name": "SCG_only",     "use_ecg": False, "use_ppg": False, "use_scg": True},
    # 双模态实验
    {"name": "ECG+PPG",      "use_ecg": True,  "use_ppg": True,  "use_scg": False},
    {"name": "ECG+SCG",      "use_ecg": True,  "use_ppg": False, "use_scg": True},
    {"name": "PPG+SCG",      "use_ecg": False, "use_ppg": True,  "use_scg": True},
    # 三模态 (完整模型)
    {"name": "ECG+PPG+SCG",  "use_ecg": True,  "use_ppg": True,  "use_scg": True},
    # 融合策略对比
    {"name": "early_fusion",  "fusion": "early"},
    {"name": "mid_fusion",    "fusion": "mid"},
    {"name": "late_fusion",   "fusion": "late"},
    # 人口学特征贡献
    {"name": "no_demographics", "use_demo": False},
    {"name": "with_demographics", "use_demo": True},
]
```

---

## 10. 技术栈与依赖

### 10.1 核心依赖

```
# 深度学习框架 (⚠️ 禁止 PyTorch)
tensorflow>=2.16          # Mac: tensorflow + tensorflow-metal
                          # WSL: tensorflow[and-cuda]
tf2onnx>=1.16             # ONNX 导出
onnxruntime>=1.18         # ONNX 推理验证

# 信号处理
scipy>=1.10
neurokit2>=0.2.10         # ECG/PPG/HRV 处理

# 数据处理
numpy>=1.24
pandas>=2.0
h5py>=3.9                 # HDF5 数据存储
scikit-learn>=1.3

# 可视化
matplotlib>=3.7
seaborn>=0.12

# 配置管理
pyyaml>=6.0               # 简单 YAML 配置

# 工具
tqdm>=4.65
pytest>=7.4
```

> ⚠️ 完整依赖见 `requirements-mac.txt`（Mac）和 `requirements-wsl.txt`（Windows WSL2 GPU）。

### 10.2 硬件参考规格

| 组件 | 规格 |
|------|------|
| MCU | 低功耗 ARM Cortex-M4/M33 |
| ECG ADC | 24-bit, ≥500 sps |
| SCG 加速度计 | ±2g, 灵敏度 < 1mg, ≥800 Hz |
| PPG 光源 | 红光 (660nm) + 红外 (940nm) |
| 通信 | BLE 5.0 + SD 卡 |
| 电池 | ≥ 500mAh (支持 2h 连续采集) |

---

## 11. 开发阶段与里程碑

### Phase 1-8: 基础代码框架 (已完成 ✅)

- [x] 项目结构、Git、CI、依赖管理
- [x] 配置系统 (`configs/default.yaml` + `pyyaml`)
- [x] 预处理管道 (`ecg_processor`, `ppg_processor`, `scg_processor`)
- [x] 信号同步与窗口提取 (`sync_and_segment.py`)
- [x] HDF5 构建管道 (`build_hdf5.py`)
- [x] 数据加载与切分 (`hdf5_loader`, `data_split`, `augmentation`)
- [x] ResNet18-SE-LSTM 模型 (TensorFlow/Keras)
- [x] SE 通道注意力 + 残差块
- [x] 内置标准化层 (ONNX 兼容)
- [x] 训练脚本 + 回调 (早停、模型保存)
- [x] 评估管道 (指标 + Bland-Altman + 可视化)
- [x] ONNX 导出
- [x] 合成数据全管线验证通过
### Phase 9: 真实数据训练 (待进行)

- [ ] 导入真实受试者数据到 `data/raw/`
- [ ] 运行预处理管道生成 HDF5
- [ ] 使用真实数据训练 ResNet18-SE-LSTM
- [ ] 初步评估模型性能

### Phase 10: 模型优化与论文 (待进行)

- [ ] 超参数搜索 (learning rate, dropout, l2_reg)
- [ ] 消融实验 (模态贡献、临床特征贡献)
- [ ] 全指标评估 (R², RMSE, MAE, MAPE)
- [ ] Bland-Altman 图绘制
- [ ] 论文图表生成

---

## 12. 关键设计决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| 深度学习框架 | PyTorch / TensorFlow | TensorFlow/Keras | 复用董雪论文 ResNet18-SE-LSTM 实现，减少重写工作量 |
| 数据存储格式 | CSV / HDF5 / LMDB | HDF5 | 支持大规模、多维数组、元数据，IO 高效 |
| 输入架构 | 三编码器融合 / 统一堆叠 | 统一重采样到125Hz堆叠为(125,3) | 复用论文架构，简化实现 |
| R峰检测 | Pan-Tompkins / neurokit2 | neurokit2 | 集成度高，有质量评估 |
| 模型标准化 | 外部scaler / 内置层 | 内置 StandardizeSignalFlat层 | ONNX导出自包含，推理时无需额外处理 |
| 多任务学习 | 共享 / 独立 | 共享 backbone + 独立 head | VO₂max 与 CO 生理关联（Fick 原理），共享特征有益 |
| 配置管理 | OmegaConf+Hydra / 简单YAML | pyyaml + default.yaml | 轻量级够用，减少依赖 |
| 交叉验证 | K-fold / LOSO | 5-fold + LOSO | 平衡计算成本与泛化评估 |

---

## 13. 注意事项与风险

### 13.1 数据泄露防范

- **严格按受试者划分** train/val/test，同一受试者的所有心搏只能出现在同一集合
- 数据增强只在训练集上进行
- 特征标准化的 scaler 仅在训练集上 fit

### 13.2 信号质量

- SCG 对运动伪迹极其敏感，高负荷运动阶段信号质量可能急剧下降
- 需设计自适应的质量评估+降级策略：质量差时增加集合平均窗口或仅使用 ECG+PPG

### 13.3 样本量考量

- 80名受试者属于中等规模，需注意过拟合
- 正则化策略：Dropout (0.2-0.3)、权重衰减、早停、数据增强
- 考虑使用预训练（如在公开 ECG/PPG 数据集上预训练 encoder）

### 13.4 生理合理性

- 模型输出应经过合理性校验：VO₂max ∈ [10, 80] mL·kg⁻¹·min⁻¹, CO ∈ [2, 30] L·min⁻¹
- 可在损失函数中加入物理约束项（Fick 原理: VO₂ = CO × (CaO₂ - CvO₂)）

---

## 14. CLI 接口设计

```bash
# 预处理
python scripts/build_preprocessed_dataset.py --config configs/default.yaml

# 训练
python scripts/train_multimodal.py --config configs/default.yaml

# 评估
python scripts/evaluate_multimodal.py --config configs/default.yaml

# 导出 ONNX
python scripts/export_onnx.py --config configs/default.yaml

# 全管线验证 (合成数据)
python scripts/verify_pipeline.py

# 测试
pytest tests/ -v
```

---

## 15. 三端协作指南

### 15.1 AI 工具与指令文件

| 工具 | 指令文件 | 共享上下文 |
|------|----------|----------|
| Claude Code | `CLAUDE.md` | `AI_CONTEXT.md` |
| OpenAI Codex | `AGENTS.md` | `AI_CONTEXT.md` |
| Antigravity | `AI_CONTEXT.md` | `AI_CONTEXT.md` |

### 15.2 调度规则

- **任务调度由人类手动执行**，AI 工具不自行认领任务
- **`.tasks.yaml`** 是任务进度的唯一权威来源
- 每次切换电脑/工具前：`git pull`；完成后：`git push`
- 不要同时编辑同一文件

### 15.3 开发顺序建议

1. **导入真实数据**: 将受试者数据放入 `data/raw/`
2. **运行预处理**: `python scripts/build_preprocessed_dataset.py`
3. **训练模型**: `python scripts/train_multimodal.py`
4. **评估结果**: `python scripts/evaluate_multimodal.py`
5. **调参与消融**: 修改 `configs/default.yaml` 重复步骤 3-4

### 15.4 模拟数据开发

在真实数据到位前，可用合成数据开发和测试：

```bash
# 全管线验证 (自动生成合成数据 + 训练 + 评估)
python scripts/verify_pipeline.py
```

---

*文档结束。此文档应作为项目开发的核心参考，随项目进展持续更新。*
