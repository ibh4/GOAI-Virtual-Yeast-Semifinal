# GOAI Virtual Yeast (L2b) 复赛复现包

## 首页信息

| 项 | 值 |
|---|---|
| 最终模型名称 | **L2b** — ControlAnchoredLowRankModelV5B (C2-r256 + 可微逐样本 FC-PCC loss, λ_fc=0.07), 3-seed (42/2026/3407) 等权 full-data refit 集成 |
| 本地验证 (V2 四折 OOF, 多 seed) | local six-module proxy **0.6771** |
| prediction.csv SHA256 | `db113d3ecad727a560ebad69aa18667035ff8af059639881189be9c5cdcb8f0f` |
| prediction 尺度 | **log2 intensity** (manifest 已声明 prediction_scale=log2) |
| prediction 形状 | 4,454 行 × (sample_ID + 4,422 官方契约蛋白列) |
| 代码版本 | 本包为自包含冻结版本 (随包 REPRODUCIBILITY_MANIFEST.json 记录逐文件 SHA256) |
| 配置 | configs/final.yaml (与训练脚本常量同版本冻结) |
| 负责人联系方式 | (见队伍 GitHub 主页) |
| 已知限制 | 见下文"已知限制" |

## 0. 模型一句话概述

Control-anchored 低秩分解模型：control 分支 (菌株+上下文 → 对照
丰度) + delta 分支 (化合物 × 菌株 × 时间 → Delta-PCA-256 低秩扰动)，
训练目标将官方六模块评分中的 FC-PCC 转为可微损失 (λ=0.07)；
最终预测 = 3 个随机种子的 full-data refit 等权平均。

## 0.5 prediction.csv 获取 (GitHub 仓库专用说明)

`prediction.csv` (193 MB) 超出 GitHub 单文件 100 MB 限制，**不随 git
仓库分发**（官方提交说明允许「稳定下载链接 + SHA256」替代）。获取方式
（三选一，结果与正式提交逐位一致）：

1. **提交包**：官方提交附件 `AI4R_AIVC_GOAI_Virtual_Yeast_代码材料.zip`
   内含完整 `prediction.csv`；
2. **重新生成**（推荐，可直接验证一致性）：运行下方命令 2 训练 +
   命令 3 推理，输出 SHA256 应等于
   `db113d3ecad727a560ebad69aa18667035ff8af059639881189be9c5cdcb8f0f`
   （同硬件逐位一致；跨硬件浮点级差异见「已知限制」）；
3. **Release 附件**：本仓库 Release `v1.0-semifinal` 附带
   `prediction.csv`（SHA256 同上，上传后可用）。

## 1. 环境要求

- Python ≥ 3.10, PyTorch ≥ 2.0 (CUDA 可选，CPU 可运行但较慢)
- 依赖见 `requirements.txt`：`pip install -r requirements.txt`
- 资源：GPU 显存 ~6GB；训练 ~7 min/seed (RTX 4060)；推理 ~2 min；
  磁盘 ~2GB (含 checkpoint 与 prediction)

## 2. 三条主命令

```bash
# 1) 构建外部特征/embedding —— 最终模型不使用外部数据, 无需执行
python scripts/build_embeddings.py

# 2) 从头训练最终模型 (3 seed, 等权集成成员)
python scripts/train.py --metadata <train_val_metadata.csv> \
    --proteome <train_val_proteome.csv> \
    --test-metadata <test_metadata.csv> \
    --config configs/final.yaml --output-dir runs/final

# 3) 冻结模型推理并生成 prediction.csv (仅读 test metadata, 不读测试表型)
python scripts/predict.py --metadata <test_metadata.csv> \
    --train-metadata <train_val_metadata.csv> \
    --train-proteome <train_val_proteome.csv> \
    --run-dir runs/final --output prediction.csv

# 附加) 提交前格式校验
python scripts/validate_submission.py --prediction prediction.csv \
    --test-metadata <test_metadata.csv>
```

备选：若把官方文件放入 `input/` 目录 (train_val metadata/proteome
+ test metadata)，可省略显式路径参数 (自动探测)。

`--test-metadata` 说明：训练需要 test metadata 仅用于构建
train+test **联合实体词表** (菌株/化合物/上下文编码索引，与提交
版本的 checkpoint 维度一致)；**不读取也不包含任何测试表型**。
`proteome_test` (测试蛋白真值) 从不被加载——数据加载器默认
`allow_test_labels=False` 硬隔离。

## 3. 数据与预处理契约

- 以 `sample_ID` 为唯一键对齐 metadata 与 proteome (pandas 索引
  对齐，不依赖 CSV 行顺序)。
- log2 变换：仅对有限且 >0 的原始强度取 log2；NaN 保留为缺失
  mask，不解释为 0。
- 缺失填充：仅用 train_val 数据的列中位数 (填充只用于损失计算
  的矩阵，mask 同步保留)。
- **特征契约 (4,422 蛋白)**：仅用 `split_final=='train'` 行计算
  每蛋白缺失率，删除 ≥80% 缺失的蛋白 → 4,422 蛋白，列名与顺序
  取官方 train proteome CSV 原始列序。契约文件：
  `artifacts/feature_contract.json` (含 SHA256)。
- **模型内部蛋白空间**：模型使用官方 train proteome 的全部 5,243
  列训练 (高缺失列以 train-only 中位数填充后 mask 保护)。推理时
  输出**子集到 4,422 官方契约列**，值与全空间集成预测逐位一致。
  该设计在训练早期版本已冻结，且已在本地验证与官方排行榜提交
  中一致使用。
- 全部统计量 (control bank、列中位数、control-PCA、Delta-PCA 基
  底、实体词表) 仅由 train_val 8,958 行拟合；验证/测试标签从未
  进入任何拟合。

## 4. 训练细节 (与提交 checkpoint 逐字一致)

- checkpoint 规则：best-by-train-loss (不使用任何验证/测试指标
  选 epoch)；40 epochs 固定；每 epoch 固定 `torch.manual_seed
  (seed+epoch)` 的 shuffle。
- loss：`masked_huber + 0.5·pearson + 0.5·delta_mse(matched) +
  λ_fc·warmup·L_fc`，L_fc 为逐样本 masked FC-PCC (control bank 按
  7 字段精确匹配，仅训练行)。
- 集成：3 seed 等权平均 (预固定规则，无权重搜索)。
- 复现随机性：同硬件 + 同数据 + 同 seed 应逐位一致；跨硬件
  (不同 GPU 型号) 允许浮点级差异，期望 proxy 波动 < 0.001
  (3 seed 平均进一步压低方差)。

## 5. 复现产物核对

| 文件 | 说明 |
|---|---|
| `checkpoints/seed_{42,2026,3407}/final.pt` | 最终模型 checkpoint (best-by-train-loss) |
| `checkpoints/checkpoints_manifest.json` | 每成员 SHA256 与配置 |
| `prediction.csv` | 最终提交结果 (4,454 × 4,423) |
| `artifacts/feature_contract.json` | 4,422 蛋白契约 + SHA256 |
| `artifacts/prediction_manifest.json` | prediction 尺度/SHA256/生成规则 |
| `REPRODUCIBILITY_MANIFEST.json` | 全包清单与逐文件 SHA256 |

**核对方法**：跑完命令 2/3 后，
`sha256sum prediction.csv` 应等于
`db113d3ecad727a560ebad69aa18667035ff8af059639881189be9c5cdcb8f0f`
(同硬件逐位一致；跨硬件时值可能有浮点级差异，以 manifest 中
per-seed ensemble SHA 与 proxy 复核为准)。

## 6. 已知限制

1. 模型内部使用 5,243 蛋白列 (官方 CSV 全列)，prediction 输出时
   子集到 4,422 契约列；此差异已在 3 节完整披露，对评分无影响
   (被删除列为 ≥80% 缺失列，真值非缺失位置全部保留在输出中)。
2. 跨硬件复现存在浮点级差异 (CUDA 非确定性归约)，预期
   Δproxy < 0.001。
3. 本包 checkpoint 为 full-data refit (train+val 8,958 行全部
   拟合)；0.6771 为同一配置在 V2 四折 OOF 协议下的本地 proxy，
   非测试分数。
4. 未使用的负结果审计 (外部数据方向) 摘要见
   REPRODUCIBILITY_MANIFEST.json → negative_results。

## 7. 目录结构

```
VC_GOAI_Virtual_Yeast_reproduction/
├─ README.md                     # 本文件
├─ requirements.txt
├─ configs/final.yaml            # 冻结配置声明
├─ prediction.csv                # 最终提交结果
├─ src/                          # 数据/模型/损失/评分/训练组件
│  ├─ data_processor_v5.py       # 便携性补丁版 (数学不变)
│  ├─ model_v5.py / model_v5b.py # 模型定义
│  ├─ losses_v5.py               # masked huber / PCC loss
│  ├─ scoring_official_v5b.py    # control bank (7 字段精确匹配)
│  └─ train_components.py        # delta-PCA / FC-cache / FC loss
├─ scripts/
│  ├─ build_embeddings.py        # no-op (无外部数据)
│  ├─ train.py                  # 命令 2: 从头训练
│  ├─ predict.py                # 命令 3: 推理 + prediction.csv
│  └─ validate_submission.py    # 格式校验
├─ external_data/source_manifest.json   # 声明: 无外部数据
├─ artifacts/                     # 特征契约 + prediction manifest
├─ checkpoints/                   # 3 seed checkpoint + manifest
├─ tests/test_smoke.py            # 冒烟测试
├─ REPRODUCIBILITY_MANIFEST.json
└─ LICENSES/
```

## 8. 复现自检对照 (提交前)

- [x] 外部数据 manifest 声明齐全 (无外部数据)
- [x] 蛋白过滤/标准化/PCA/词表/先验仅用 train 行
- [x] 最终训练配置、随机种子、checkpoint 与集成权重冻结
- [x] 推理只读 test metadata + checkpoint，不读 test proteome
- [x] prediction.csv 4,454 × (1+4,422)，官方列序，无 NA/Inf
- [x] prediction_scale=log2 已声明
- [x] 三条主命令已在本地端到端执行验证 (预测与提交 CSV 一致)
- [x] 包内不含比赛原始数据 / 测试真值 / 密钥
