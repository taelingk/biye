# 基于穿戴式多模态生理信号的动态心肺功能评测系统 — 技术文档

> **项目代号**: CardioFit  
> **版本**: v1.0  
> **最后更新**: 2025-11  
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
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/              # 信号预处理模块
│   │   ├── __init__.py
│   │   ├── filters.py              # 带通/陷波/自适应滤波
│   │   ├── ecg_processor.py        # QRS检测、R峰定位、HRV
│   │   ├── ppg_processor.py        # PPG特征提取
│   │   ├── scg_processor.py        # SCG特征点识别 (AO/AC/MC/MO)
│   │   ├── sync.py                 # 多模态信号时间同步
│   │   ├── segmentation.py         # 心搏分段
│   │   ├── quality.py              # 信号质量评估与伪迹剔除
│   │   └── augmentation.py         # 数据增强
│   │
│   ├── features/                   # 特征工程模块
│   │   ├── __init__.py
│   │   ├── ecg_features.py         # HR, HRV, QRS duration, ST变异
│   │   ├── ppg_features.py         # PAT, 脉搏幅度, 血管顺应性
│   │   ├── scg_features.py         # STI (IVCT/LVET/IVRT), Tei, 幅值, 频域
│   │   ├── cross_modal.py          # 跨模态特征 (PTT, PEP等)
│   │   └── demographic.py          # 人口学特征编码
│   │
│   ├── dataset/                    # 数据集管理模块
│   │   ├── __init__.py
│   │   ├── cardiofit_dataset.py    # PyTorch Dataset 实现
│   │   ├── dataloader.py           # DataLoader 工厂函数
│   │   └── transforms.py           # 数据变换/增强 pipeline
│   │
│   ├── models/                     # 深度学习模型模块
│   │   ├── __init__.py
│   │   ├── backbone/
│   │   │   ├── __init__.py
│   │   │   ├── ecg_encoder.py      # ECG 1D-CNN 编码器
│   │   │   ├── ppg_encoder.py      # PPG 1D-CNN 编码器
│   │   │   ├── scg_encoder.py      # SCG 1D-CNN 编码器
│   │   │   └── transformer.py      # Transformer/Bi-LSTM 时序编码器
│   │   ├── fusion/
│   │   │   ├── __init__.py
│   │   │   ├── early_fusion.py     # 早期融合 (通道拼接)
│   │   │   ├── mid_fusion.py       # 中期融合 (注意力加权)
│   │   │   └── late_fusion.py      # 晚期融合 (决策层加权)
│   │   ├── heads/
│   │   │   ├── __init__.py
│   │   │   ├── vo2max_head.py      # VO₂max 回归头
│   │   │   └── co_head.py          # CO 回归头
│   │   ├── cardiofit_net.py        # 主模型 (组装 backbone+fusion+head)
│   │   └── baselines/
│   │       ├── __init__.py
│   │       ├── friend_equation.py  # FRIEND 传统方程基线
│   │       ├── svm_model.py        # SVM 回归基线
│   │       ├── xgboost_model.py    # XGBoost 基线
│   │       └── rf_model.py         # 随机森林基线
│   │
│   ├── training/                   # 训练管理模块
│   │   ├── __init__.py
│   │   ├── trainer.py              # 训练循环
│   │   ├── losses.py               # 损失函数 (MSE, Huber, 混合损失)
│   │   ├── optimizer.py            # 优化器工厂
│   │   ├── scheduler.py            # 学习率调度
│   │   ├── callbacks.py            # 早停、模型保存等回调
│   │   └── cross_validation.py     # K-fold / LOSO 交叉验证
│   │
│   ├── evaluation/                 # 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py              # R², RMSE, MAE, MAPE, Pearson r
│   │   ├── bland_altman.py         # Bland-Altman 一致性分析
│   │   ├── ablation.py             # 消融实验框架
│   │   ├── visualization.py        # 结果可视化
│   │   └── statistical_tests.py    # 统计显著性检验
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py                   # 数据读写工具
│       ├── logging.py              # 日志配置
│       ├── reproducibility.py      # 随机种子管理
│       └── config.py               # 配置文件解析 (OmegaConf)
│
├── scripts/
│   ├── preprocess_all.py           # 批量预处理脚本
│   ├── extract_features.py         # 特征提取脚本
│   ├── train.py                    # 训练入口
│   ├── evaluate.py                 # 评估入口
│   ├── ablation_study.py           # 消融实验入口
│   ├── export_model.py             # 模型导出 (ONNX)
│   └── visualize_results.py        # 生成论文图表
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_quality.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_results_visualization.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── firmware/                       # 嵌入式固件 (可选，参考)
│   ├── README.md
│   └── sensor_config.h
│
└── docs/
    ├── data_dictionary.md          # 数据字段说明
    ├── model_card.md               # 模型卡片
    └── experiment_log.md           # 实验记录
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

### 7.1 主模型: CardioFitNet

```
                         ┌─────────────────────┐
                         │    Demographics      │
                         │  (age,sex,BMI,...)   │
                         │    MLP Encoder       │
                         └──────────┬──────────┘
                                    │
     ┌────────────┐  ┌────────────┐ │ ┌────────────┐
     │ ECG Input  │  │ PPG Input  │ │ │ SCG Input  │
     │ (1, L_ecg) │  │ (2, L_ppg) │ │ │ (3, L_scg) │
     └─────┬──────┘  └─────┬──────┘ │ └─────┬──────┘
           │               │        │       │
     ┌─────▼──────┐  ┌─────▼──────┐ │ ┌─────▼──────┐
     │ 1D-CNN     │  │ 1D-CNN     │ │ │ 1D-CNN     │
     │ Encoder    │  │ Encoder    │ │ │ Encoder    │
     │ (ResBlock) │  │ (ResBlock) │ │ │ (ResBlock) │
     └─────┬──────┘  └─────┬──────┘ │ └─────┬──────┘
           │               │        │       │
     ┌─────▼──────┐  ┌─────▼──────┐ │ ┌─────▼──────┐
     │ Bi-LSTM /  │  │ Bi-LSTM /  │ │ │ Bi-LSTM /  │
     │ Transformer│  │ Transformer│ │ │ Transformer│
     └─────┬──────┘  └─────┬──────┘ │ └─────┬──────┘
           │               │        │       │
           └───────┬───────┘        │       │
                   │                │       │
           ┌───────▼────────────────▼───────▼──┐
           │      Multi-Modal Fusion Layer      │
           │   (Attention / Concat / Weighted)  │
           └───────────────┬───────────────────┘
                           │
                    ┌──────▼──────┐
                    │   Shared    │
                    │   FC Layers │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │                         │
        ┌─────▼─────┐            ┌─────▼─────┐
        │VO₂max Head│            │  CO Head   │
        │ (Regress) │            │ (Regress)  │
        └───────────┘            └────────────┘
```

### 7.2 模块详细设计

#### 7.2.1 单模态编码器 (1D-CNN + Sequential)

```python
class SignalEncoder(nn.Module):
    """通用的单模态信号编码器"""
    
    def __init__(self, 
                 in_channels: int,        # ECG=1, PPG=2, SCG=3
                 base_filters: int = 64,
                 n_res_blocks: int = 4,
                 seq_model: str = "bilstm",  # "bilstm" | "transformer"
                 hidden_dim: int = 128,
                 output_dim: int = 128):
        super().__init__()
        
        # Stage 1: 1D-CNN 局部特征提取
        self.cnn = nn.Sequential(
            ResBlock1D(in_channels, base_filters, kernel_size=15, stride=2),
            ResBlock1D(base_filters, base_filters*2, kernel_size=9, stride=2),
            ResBlock1D(base_filters*2, base_filters*4, kernel_size=5, stride=2),
            ResBlock1D(base_filters*4, hidden_dim, kernel_size=3, stride=2),
        )
        
        # Stage 2: 时序建模
        if seq_model == "bilstm":
            self.seq = nn.LSTM(hidden_dim, hidden_dim//2, 
                              num_layers=2, bidirectional=True, 
                              batch_first=True, dropout=0.3)
        elif seq_model == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, 
                dim_feedforward=256, dropout=0.1
            )
            self.seq = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Stage 3: 全局池化 → 嵌入
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (B, C, L)
        h = self.cnn(x)           # (B, hidden_dim, L')
        h = h.permute(0, 2, 1)    # (B, L', hidden_dim)
        h, _ = self.seq(h)        # (B, L', hidden_dim)
        h = h.permute(0, 2, 1)    # (B, hidden_dim, L')
        h = self.pool(h).squeeze(-1)  # (B, hidden_dim)
        return self.proj(h)       # (B, output_dim)
```

#### 7.2.2 ResBlock1D

```python
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride),
            nn.BatchNorm1d(out_ch)
        ) if in_ch != out_ch or stride != 1 else nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)
```

#### 7.2.3 融合策略

```python
class AttentionFusion(nn.Module):
    """中期融合: 多头注意力加权融合"""
    
    def __init__(self, embed_dim: int = 128, n_modalities: int = 3):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim * n_modalities, embed_dim)
    
    def forward(self, ecg_emb, ppg_emb, scg_emb):
        # 堆叠为序列: (3, B, D)
        tokens = torch.stack([ecg_emb, ppg_emb, scg_emb], dim=0)
        attn_out, weights = self.attention(tokens, tokens, tokens)
        attn_out = self.norm(attn_out + tokens)
        # 拼接融合
        fused = attn_out.permute(1, 0, 2).reshape(-1, ecg_emb.size(1) * 3)
        return self.fc(fused), weights


class EarlyFusion(nn.Module):
    """早期融合: 输入层通道拼接"""
    
    def __init__(self, ecg_len, ppg_len, scg_len, target_len=2000):
        super().__init__()
        # 将不同长度信号重采样到统一长度后按通道拼接
        # ECG: 1ch, PPG: 2ch, SCG: 3ch → 总共 6 通道
        self.resample_target = target_len
    
    def forward(self, ecg, ppg, scg):
        # 重采样到统一长度
        ecg_r = F.interpolate(ecg, self.resample_target)
        ppg_r = F.interpolate(ppg, self.resample_target)
        scg_r = F.interpolate(scg, self.resample_target)
        return torch.cat([ecg_r, ppg_r, scg_r], dim=1)  # (B, 6, L)


class LateFusion(nn.Module):
    """晚期融合: 决策层自适应加权"""
    
    def __init__(self, embed_dim: int = 128, n_modalities: int = 3):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, 64),
            nn.ReLU(),
            nn.Linear(64, n_modalities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, ecg_emb, ppg_emb, scg_emb):
        combined = torch.cat([ecg_emb, ppg_emb, scg_emb], dim=-1)
        weights = self.weight_net(combined)  # (B, 3)
        stacked = torch.stack([ecg_emb, ppg_emb, scg_emb], dim=1)  # (B, 3, D)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return fused, weights
```

#### 7.2.4 主模型组装

```python
class CardioFitNet(nn.Module):
    """多模态心肺功能评估主模型"""
    
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim  # 128
        
        # 单模态编码器
        self.ecg_encoder = SignalEncoder(in_channels=1, output_dim=embed_dim)
        self.ppg_encoder = SignalEncoder(in_channels=2, output_dim=embed_dim)
        self.scg_encoder = SignalEncoder(in_channels=3, output_dim=embed_dim)
        
        # 人口学特征编码
        self.demo_encoder = nn.Sequential(
            nn.Linear(5, 32),  # age, sex, height, weight, bmi
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        
        # 融合层 (可切换策略)
        fusion_type = config.fusion_type  # "early" | "mid" | "late"
        if fusion_type == "mid":
            self.fusion = AttentionFusion(embed_dim, n_modalities=3)
        elif fusion_type == "late":
            self.fusion = LateFusion(embed_dim, n_modalities=3)
        elif fusion_type == "early":
            self.fusion = EarlyFusion(...)
            self.unified_encoder = SignalEncoder(in_channels=6, output_dim=embed_dim)
        
        # 回归头
        fused_dim = embed_dim + embed_dim  # signal fusion + demographics
        self.shared_fc = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.vo2max_head = nn.Linear(128, 1)
        self.co_head = nn.Linear(128, 1)
    
    def forward(self, ecg, ppg, scg, demographics):
        """
        Args:
            ecg: (B, 1, L_ecg) 
            ppg: (B, 2, L_ppg)
            scg: (B, 3, L_scg)
            demographics: (B, 5) [age, sex, height, weight, bmi]
        Returns:
            vo2max_pred: (B, 1)
            co_pred: (B, 1)
        """
        ecg_emb = self.ecg_encoder(ecg)
        ppg_emb = self.ppg_encoder(ppg)
        scg_emb = self.scg_encoder(scg)
        demo_emb = self.demo_encoder(demographics)
        
        signal_fused, attn_weights = self.fusion(ecg_emb, ppg_emb, scg_emb)
        combined = torch.cat([signal_fused, demo_emb], dim=-1)
        
        shared = self.shared_fc(combined)
        vo2max_pred = self.vo2max_head(shared)
        co_pred = self.co_head(shared)
        
        return {
            "vo2max": vo2max_pred,
            "co": co_pred,
            "attention_weights": attn_weights
        }
```

---

## 8. 训练配置

### 8.1 默认超参数 (`configs/default.yaml`)

```yaml
# ===== 数据 =====
data:
  root_dir: "data/processed"
  beat_window_ecg: 300     # 采样点 (600ms @ 500Hz)
  beat_window_ppg: 120     # 采样点 (1200ms @ 100Hz)
  beat_window_scg: 640     # 采样点 (800ms @ 800Hz)
  n_beats_per_sample: 10   # 每个样本取连续10个心搏
  augmentation:
    enabled: true
    noise_std: 0.01
    scale_range: [0.95, 1.05]
    time_shift_ms: 20

# ===== 模型 =====
model:
  name: "CardioFitNet"
  embed_dim: 128
  fusion_type: "mid"            # "early" | "mid" | "late"
  seq_model: "bilstm"           # "bilstm" | "transformer"
  n_res_blocks: 4
  dropout: 0.3

# ===== 训练 =====
training:
  epochs: 200
  batch_size: 32
  optimizer:
    type: "AdamW"
    lr: 1e-3
    weight_decay: 1e-4
  scheduler:
    type: "CosineAnnealingWarmRestarts"
    T_0: 20
    T_mult: 2
  loss:
    vo2max_weight: 1.0
    co_weight: 1.0
    type: "huber"               # "mse" | "huber" | "smooth_l1"
  early_stopping:
    patience: 20
    metric: "val_rmse"
    mode: "min"
  gradient_clip: 1.0

# ===== 评估 =====
evaluation:
  cv_folds: 5
  metrics: ["r2", "rmse", "mae", "mape", "pearson_r"]
  bland_altman: true

# ===== 系统 =====
system:
  seed: 42
  num_workers: 4
  device: "cuda"
  mixed_precision: true
  wandb_project: "cardiofit"
```

### 8.2 损失函数

```python
class CardioFitLoss(nn.Module):
    """多任务联合损失"""
    
    def __init__(self, vo2_weight=1.0, co_weight=1.0, loss_type="huber"):
        super().__init__()
        self.vo2_weight = vo2_weight
        self.co_weight = co_weight
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "huber":
            self.criterion = nn.HuberLoss(delta=1.0)
        elif loss_type == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
    
    def forward(self, pred, target):
        loss_vo2 = self.criterion(pred["vo2max"], target["vo2max"])
        loss_co = self.criterion(pred["co"], target["co"])
        total = self.vo2_weight * loss_vo2 + self.co_weight * loss_co
        return {
            "total": total,
            "vo2max_loss": loss_vo2,
            "co_loss": loss_co
        }
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
# 深度学习
torch>=2.0
pytorch-lightning>=2.0
torchmetrics

# 信号处理
scipy
neurokit2            # ECG/PPG/HRV 处理
biosppy              # 生物信号处理
heartpy              # HRV 分析
wfdb                 # 心电数据工具

# 数据处理
numpy
pandas
h5py                 # HDF5 数据存储
scikit-learn

# 机器学习基线
xgboost
lightgbm

# 可视化
matplotlib
seaborn
plotly

# 实验管理
wandb                # 实验跟踪
omegaconf            # 配置管理
hydra-core           # 配置框架

# 工具
tqdm
click                # CLI
pytest               # 测试
```

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

### Phase 1: 基础设施 (第1-2周)

- [ ] 初始化项目结构，配置 Git、CI、依赖管理
- [ ] 实现 `configs/` 配置系统 (OmegaConf + Hydra)
- [ ] 实现 `utils/` 工具模块 (logging, seed, IO)
- [ ] 编写数据字典文档

### Phase 2: 预处理管道 (第3-5周)

- [ ] 实现 `ECGProcessor`（滤波、R峰检测、HRV）
- [ ] 实现 `PPGProcessor`（滤波、波峰检测、特征）
- [ ] 实现 `SCGProcessor`（滤波、AO/AC检测、STI计算）
- [ ] 实现 `MultiModalSync` 多信号同步
- [ ] 实现 `SignalQualityAssessor`
- [ ] 实现 `segmentation.py` 心搏分段
- [ ] 单元测试全覆盖

### Phase 3: 特征工程 (第6-7周)

- [ ] 实现各模态特征提取函数
- [ ] 实现跨模态特征（PTT, EMD）
- [ ] 构建 HDF5 预处理数据写入管道
- [ ] 数据探索性分析 Notebook

### Phase 4: 数据集与DataLoader (第8周)

- [ ] 实现 `CardioFitDataset` (PyTorch Dataset)
- [ ] 实现数据增强 transforms
- [ ] 实现 DataLoader 工厂 (train/val/test)
- [ ] 数据划分脚本

### Phase 5: 模型开发 (第9-12周)

- [ ] 实现 `ResBlock1D`
- [ ] 实现三个 `SignalEncoder`
- [ ] 实现三种融合策略
- [ ] 组装 `CardioFitNet` 主模型
- [ ] 实现基线模型 (FRIEND方程, SVM, XGBoost, RF)
- [ ] 实现损失函数与训练循环

### Phase 6: 训练与调优 (第13-16周)

- [ ] 基线模型训练与评估
- [ ] CardioFitNet 训练 (各融合策略)
- [ ] 超参数搜索 (learning rate, dropout, architecture)
- [ ] 消融实验执行
- [ ] WandB 实验跟踪

### Phase 7: 评估与可视化 (第17-18周)

- [ ] 全指标评估 (R², RMSE, MAE, MAPE)
- [ ] Bland-Altman 图绘制
- [ ] 消融结果汇总
- [ ] 注意力权重可视化
- [ ] 论文图表生成

---

## 12. 关键设计决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| 数据存储格式 | CSV / HDF5 / LMDB | HDF5 | 支持大规模、多维数组、元数据，IO 高效 |
| R峰检测 | Pan-Tompkins / neurokit2 | neurokit2 | 集成度高，有质量评估 |
| SCG 主轴 | z轴 / 合成向量 | z轴为主 + 三轴输入模型 | z轴信噪比最佳，三轴保留空间信息 |
| 序列模型 | LSTM / Transformer | 均实现，实验选优 | 文献中两者均有成功案例 |
| 融合策略 | 早/中/晚 | 中期融合优先 | 文献表明注意力融合效果好，但需消融验证 |
| 多任务学习 | 共享 / 独立 | 共享backbone + 独立head | VO₂max 与 CO 生理关联（Fick 原理），共享特征有益 |
| 交叉验证 | K-fold / LOSO | 5-fold + LOSO | 平衡计算成本与泛化评估 |
| 实验管理 | MLflow / WandB | WandB | 云端协作、可视化友好 |

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
python scripts/preprocess_all.py --input data/raw --output data/processed --config configs/default.yaml

# 特征提取
python scripts/extract_features.py --input data/processed --output data/features

# 训练
python scripts/train.py --config configs/experiment/vo2max_multimodal.yaml --gpus 1

# 评估
python scripts/evaluate.py --checkpoint runs/best_model.ckpt --test-data data/splits/test_subjects.txt

# 消融实验
python scripts/ablation_study.py --config configs/experiment/ablation_study.yaml

# 可视化
python scripts/visualize_results.py --results runs/experiment_001/results.json --output figures/
```

---

## 15. 与 Claude Code 协作指南

### 15.1 开发顺序建议

1. **从预处理开始**: 先实现可靠的信号处理管道，这是整个系统的基础
2. **构建 Dataset**: 确保 DataLoader 正确加载和对齐多模态数据
3. **先跑基线**: 用 XGBoost + 手工特征验证数据管道，建立性能基准
4. **再搭深度模型**: 从单模态开始，逐步加入融合
5. **最后做消融**: 在完整模型可用后系统性消融

### 15.2 每个模块的开发原则

- 每个 `.py` 文件应包含清晰的 docstring、类型注解
- 每个核心函数需配套单元测试
- 使用 `logging` 而非 `print`
- 配置通过 YAML 驱动，避免硬编码
- 数据路径使用 `pathlib.Path`

### 15.3 模拟数据开发

在真实数据到位前，可用合成数据开发和测试：

```python
def generate_synthetic_data(n_subjects=20, n_beats=100):
    """生成模拟的多模态心肺信号数据用于开发测试"""
    for i in range(n_subjects):
        ecg = simulate_ecg(duration=300, fs=500)
        ppg = simulate_ppg(duration=300, fs=100)
        scg = simulate_scg(duration=300, fs=800)
        vo2max = np.random.normal(40, 10)  # 模拟VO2max分布
        co = np.random.normal(5, 1.5)       # 模拟CO分布
        yield {"ecg": ecg, "ppg": ppg, "scg": scg, 
               "vo2max": vo2max, "co": co}
```

---

*文档结束。此文档应作为 Claude Code 项目开发的核心参考，随项目进展持续更新。*
