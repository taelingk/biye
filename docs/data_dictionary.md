# Data Dictionary (Draft)

本文件用于维护数据字段定义，建议与 `TECHNICAL_DOC.md` 同步更新。

## Subject-level metadata

- `subject_id`: 受试者编号
- `age`: 年龄
- `sex`: 性别（建议编码：`M/F` 或 `0/1`）
- `height_cm`: 身高（cm）
- `weight_kg`: 体重（kg）
- `bmi`: 体重指数

## Signal files

- `ecg.csv`: `timestamp_ms`, `ecg_mv`
- `ppg.csv`: `timestamp_ms`, `ppg_red`, `ppg_ir`
- `scg.csv`: `timestamp_ms`, `acc_x`, `acc_y`, `acc_z`
- `cpet_vo2.csv`: `timestamp_ms`, `vo2_ml_kg_min`, ...
- `cnap_co.csv`: `timestamp_ms`, `co_l_min`, ...
