# CardioFit Roadmap

该路线图根据 `TECHNICAL_DOC.md` 开发阶段整理，便于执行与跟踪。

## Phase 1 - 基础设施

- [ ] 初始化目录结构与依赖管理
- [ ] 建立配置系统（Hydra + OmegaConf）
- [ ] 完成 `utils` 基础模块（logging/seed/io）
- [ ] 补齐数据字典文档

## Phase 2 - 预处理管道

- [ ] ECG：滤波、R 峰检测、HRV
- [ ] PPG：滤波、峰谷检测、特征初提
- [ ] SCG：滤波、AO/AC 识别、STI
- [ ] 多模态同步（统一时间轴）
- [ ] 信号质量评估与低质片段剔除

## Phase 3 - 特征工程

- [ ] 三模态特征提取函数
- [ ] 跨模态特征（PTT/EMD）
- [ ] HDF5 特征写入管道
- [ ] 数据探索 Notebook

## Phase 4 - Dataset 与 DataLoader

- [ ] 实现 `CardioFitDataset`
- [ ] 训练/验证/测试 DataLoader
- [ ] 数据增强与变换 pipeline
- [ ] 按受试者划分脚本

## Phase 5 - 模型开发

- [ ] 单模态编码器
- [ ] 融合层（early/mid/late）
- [ ] 双任务回归头（VO2max + CO）
- [ ] 基线模型（FRIEND/SVM/XGBoost/RF）

## Phase 6 - 训练与调优

- [ ] 基线训练与对比
- [ ] 深度模型训练与调参
- [ ] 消融实验（模态、融合策略、人口学特征）
- [ ] WandB 实验跟踪

## Phase 7 - 评估与可视化

- [ ] 指标评估（R2/RMSE/MAE/MAPE/r）
- [ ] Bland-Altman 一致性分析
- [ ] 注意力权重可视化
- [ ] 论文级图表导出

## 风险看板（持续跟踪）

- [ ] 数据泄露风险（划分/标准化/增强边界）
- [ ] SCG 高运动阶段信号退化
- [ ] 小样本过拟合
- [ ] 输出生理约束违反
