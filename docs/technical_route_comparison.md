# 技术路线对比分析：是否需要修改 TECHNICAL_DOC.md？

## 总体结论：**不需要大改，但需要 3 处关键增补**

你现有的技术路线设计得**非常好**，整体架构合理且比董雪论文更先进（多模态 vs 单模态）。但董雪论文中有几个被验证有效的具体技术手段，你的文档中**缺失或不够明确**，建议补充。

---

## 一、逐项对比

| 维度 | 董雪论文 | 你的 TECHNICAL_DOC | 差异 |
|------|---------|-------------------|------|
| **信号模态** | 仅 PPG（单模态） | ECG + PPG + SCG（三模态） | ✅ 你的更强 |
| **预测目标** | 仅 CO | VO₂max + CO（双任务） | ✅ 你的更全面 |
| **预处理滤波** | 带通滤波 + 三次样条插值 | 带通/陷波/自适应滤波 + 中值滤波基线校正 | ✅ 相当，你的更丰富 |
| **异常信号剔除** | 基于特征的异常脉搏波剔除 | quality.py 信号质量评估 | ✅ 一致 |
| **多阶导数增强** | ✅ 核心创新：原始+一阶+二阶导数拼接 | ❌ **仅在 PPG 特征提取中提及 SDPPG** | ⚠️ **需补充** |
| **模型骨干** | ResNet18 | ResBlock1D (自定义) | ✅ 相当 |
| **SE 通道注意力** | ✅ SE 模块，核心创新 | ❌ 未在编码器中使用 | ⚠️ **建议补充** |
| **时序建模** | LSTM | Bi-LSTM / Transformer（可切换） | ✅ 你的更灵活 |
| **融合策略** | 单模态，无融合 | 早/中/晚期融合（三种策略可切换） | ✅ 你的更先进 |
| **人口学特征** | 年龄、性别、身高、体重 | 年龄、性别、身高、体重、BMI | ✅ 一致 |
| **消融实验** | 不同模型对比 + 不同输入特征对比 | 模态消融 + 融合策略 + 人口学特征 | ⚠️ **缺少输入特征消融** |

---

## 二、需要修改的 3 处关键内容

### 修改 1：在预处理管道中加入「多阶导数特征增强」

> [!IMPORTANT]
> 这是董雪论文的**核心创新之一**，已被实验验证可以显著提升模型性能。
> 你的文档中 PPG 处理器已提到 SDPPG（二阶导数）作为手工特征，但**未将其作为深度学习模型的输入增强手段**。

**建议修改位置**：

1. **预处理总体流程**（第 282 行）修改为：
```
原始信号 → 滤波去噪 → 信号质量评估 → 多模态同步 → R峰检测 → 心搏分段 → 导数增强 → 特征提取
```

2. **在 `transforms.py` 中新增导数增强 transform**：
```python
class DerivativeEnhancement:
    """多阶导数特征增强（参考董雪论文）
    
    将原始信号与其一阶、二阶导数拼接，构建多通道时序特征矩阵。
    - PPG: (1ch → 3ch) 或 (2ch → 6ch，红光+红外各自求导)
    - ECG: (1ch → 3ch)
    - SCG: (3ch → 9ch，各轴各自求导)
    """
```

3. **修改 CardioFitNet 的输入通道数**：
```python
# 启用导数增强后：
self.ecg_encoder = SignalEncoder(in_channels=3, ...)   # ECG原始+一阶+二阶
self.ppg_encoder = SignalEncoder(in_channels=6, ...)   # PPG 2ch × 3
self.scg_encoder = SignalEncoder(in_channels=9, ...)   # SCG 3ch × 3
```

> [!TIP]
> 导数增强应设计为**可配置选项**，通过 config 文件控制是否启用，方便消融实验对比。

---

### 修改 2：在 ResBlock1D 或编码器中加入「SE 通道注意力模块」

> [!IMPORTANT]
> 董雪论文使用 Squeeze-and-Excitation (SE) 通道注意力，这在其模型中是关键组件。
> 你的编码器仅有 ResBlock → BiLSTM → Pool，**缺少通道注意力机制**。

**建议修改位置**：在 `SignalEncoder` 中的 CNN 后或 ResBlock 内添加 SE 模块

```python
class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation 通道注意力（参考董雪论文 ResNet18-SE-LSTM）"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, C, L)
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w
```

**在 ResBlock1D 中集成**：
```python
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, use_se=True):
        ...
        self.se = SEBlock1D(out_ch) if use_se else nn.Identity()
    
    def forward(self, x):
        ...
        out = self.bn2(self.conv2(out))
        out = self.se(out)              # ← 新增 SE 注意力
        return self.relu(out + residual)
```

---

### 修改 3：在消融实验中加入「输入特征消融」

> [!IMPORTANT]
> 董雪论文专门做了**不同输入特征的预测结果对比**（原始信号 vs 原始+导数 vs 原始+导数+临床参数），这对论文写作非常有价值。

**建议在消融配置中新增**：

```python
ABLATION_CONFIGS = [
    # ... 现有配置 ...
    
    # ===== 新增：输入特征消融（参考董雪论文） =====
    # 导数增强消融
    {"name": "raw_signal_only",        "derivative_enhance": False},
    {"name": "with_derivative",        "derivative_enhance": True},
    
    # SE注意力消融  
    {"name": "without_se",             "use_se": False},
    {"name": "with_se",                "use_se": True},
    
    # 序列模型消融
    {"name": "cnn_only",               "seq_model": None},
    {"name": "cnn_bilstm",             "seq_model": "bilstm"},
    {"name": "cnn_transformer",        "seq_model": "transformer"},
    {"name": "cnn_se_bilstm",          "seq_model": "bilstm", "use_se": True},
]
```

---

## 三、不需要修改的部分

以下设计你的文档**已经做得很好**，不需要改动：

| 设计 | 理由 |
|------|------|
| 多模态架构 (ECG+PPG+SCG) | 比董雪论文的单模态 PPG 更丰富，信息量更大 |
| 三种融合策略可切换 | 灵活且完整，董雪论文无融合设计 |
| Bi-LSTM / Transformer 可切换 | 比董雪论文仅有 LSTM 更灵活 |
| 手工特征模块 (`features/`) | 基线模型必需，消融实验有价值 |
| 信号质量评估 (`quality.py`) | 与董雪论文的异常信号剔除思路一致 |
| HDF5 数据存储 | 合理的工程选择 |
| 跨模态特征 (PTT, PEP) | 这是你的项目独有优势 |
| 双任务学习 (VO₂max + CO) | 合理的多任务设计，Fick 原理支撑 |

---

## 四、修改影响评估

| 修改项 | 代码量 | 影响范围 | 优先级 |
|--------|--------|----------|--------|
| 多阶导数增强 | 小 (~50行) | `transforms.py` + config + 模型输入通道数 | 🔴 高 |
| SE 通道注意力 | 小 (~30行) | `ResBlock1D` 或 `SignalEncoder` | 🟡 中 |
| 输入特征消融 | 小 (配置项) | `ablation.py` + config | 🟢 低（后期做） |

> [!NOTE]
> 这三处修改都是**增量性**的，不会破坏现有架构设计。核心思想是：
> - 你的多模态架构是**框架优势**
> - 董雪论文的导数增强和 SE 注意力是**技术细节优势**
> - 两者结合 = 更强的系统
