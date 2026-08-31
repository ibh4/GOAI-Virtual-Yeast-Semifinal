# GOAI Virtual Yeast — 酵母扰动蛋白组响应预测

<p align="center">
  <b>GOAI 虚拟细胞方向复赛作品 · Control-Anchored 低秩分解 × 可微评分对齐 × 系统性外部数据审计</b><br>
  把官方六模块评分变成可微损失，用 oracle 审计证明剩余缺口属于科学问题而非工程。
</p>

<p align="center">
  <img alt="Track" src="https://img.shields.io/badge/Track-虚拟细胞%20AIVC-0F4C81?style=for-the-badge">
  <img alt="Model" src="https://img.shields.io/badge/Model-L2b%20(C2--r256%20%2B%20FC--PCC)-0E7C7B?style=for-the-badge">
  <img alt="Score" src="https://img.shields.io/badge/OOF%20Proxy-0.6771-22C55E?style=for-the-badge">
  <img alt="Seeds" src="https://img.shields.io/badge/Ensemble-3%20seed%20等权-8B5CF6?style=for-the-badge">
  <img alt="External Data" src="https://img.shields.io/badge/外部数据依赖-零-F59E0B?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-64748B?style=for-the-badge">
</p>

***

## 0. 项目一句话

**GOAI Virtual Yeast** 是面向虚拟细胞赛道的酵母扰动蛋白组响应预测项目：给定扰动条件（化合物 × 菌株 × 时间），预测 4,454 个测试样本的 4,422 个蛋白 log2 丰度。核心模型 **L2b** 将预测分解为「对照丰度 + 低秩扰动残差」，并把官方评分中的逐样本 FC-PCC 指标直接转化为可微损失（λ=0.07），本地 OOF proxy 从 0.6698 提升至 **0.6771**（12/12 seed×fold 配对全部正向）。

这个项目的目标不是堆外部特征，而是回答三个科学问题：**评分对齐能带来多少增益、交叉验证的索引完整性如何保证、公开外部数据的真实信息上限在哪里**。我们对 8 个外部数据方向执行了带严格负对照的统一审计，全部冻结为负结果，并用 oracle 上限审计给出机制解释——decoder 子空间可支撑 0.81，瓶颈在「化合物 → 扰动响应」的映射本身。

> 仓库定位：**复赛代码材料 + 端到端复现包 + 8 方向负结果审计链**。正式提交版本将冻结为 Release tag。

***

## 1. 关键信息速查

| 项              | 值                                                                             |
| -------------- | ----------------------------------------------------------------------------- |
| 最终模型           | **L2b** — ControlAnchoredLowRankModelV5B (C2-r256 + 可微逐样本 FC-PCC, λ\_fc=0.07) |
| 集成方式           | 3 seed (42/2026/3407) 等权 full-data refit（预固定规则，无权重搜索）                         |
| 本地验证           | V2 四折 LOSO × 3-seed OOF **0.6771**（C2 基线 0.6698，Δ+0.0073，12/12 配对正向）          |
| V3 外推协议        | 化合物分组五折 0.6171（S1 未见化合物 = 0.474，定位瓶颈）                                         |
| Oracle 上限      | coefficient oracle 0.8074（+0.19，架构非瓶颈的机制证明）                                   |
| prediction.csv | 4,454 × (sample\_ID + 4,422)，log2，SHA256 `db113d3e…`（完整值见 §6）                 |
| 外部数据           | **零依赖**（8 方向审计后全部弃用，审计链保留）                                                    |
| 许可证            | MIT（`LICENSES/`）                                                              |

***

## 2. 总体架构：Control-Anchored 低秩分解

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, Microsoft YaHei, sans-serif", "background": "#FFFFFF", "primaryColor": "#EFF6FF", "primaryTextColor": "#1E293B", "primaryBorderColor": "#93C5FD", "lineColor": "#94A3B8"}}}%%
flowchart LR
    C(["输入条件<br/>化合物 · 菌株 · 时间 · 上下文"]):::input

    subgraph CTRL["对照分支 Control"]
        SE["菌株 + 上下文<br/>Embedding"]:::ctrl
        CP["control_head<br/>→ 192 维系数"]:::ctrl
        CPCA["Control-PCA 基底<br/>仅在对照样本上拟合"]:::pca
    end

    subgraph DELTA["扰动分支 Delta"]
        DE["化合物 × 菌株 × t<br/>条件编码"]:::delta
        DP["delta_head<br/>→ 256 维系数 z"]:::delta
        DPCA["Delta-PCA-256 基底<br/>treatment − 精确匹配 control"]:::pca
    end

    W["共享 Decoder W<br/>5,243 蛋白空间"]:::decoder
    OUT(["y_pred = 对照丰度 + 低秩扰动残差<br/>子集到 4,422 契约列"]):::output

    C --> SE --> CP
    C --> DE --> DP
    CP --> W
    DP --> W
    CPCA -.-> CP
    DPCA -.-> DP
    W --> OUT

    classDef input fill:#FDBA74,stroke:#FFEDD5,color:#7C2D12,stroke-width:2px;
    classDef ctrl fill:#93C5FD,stroke:#DBEAFE,color:#1E3A8A,stroke-width:2px;
    classDef delta fill:#5EEAD4,stroke:#CCFBF1,color:#134E4A,stroke-width:2px;
    classDef pca fill:#FDE68A,stroke:#FEF3C7,color:#78350F,stroke-width:2px;
    classDef decoder fill:#C4B5FD,stroke:#EDE9FE,color:#4C1D95,stroke-width:3px;
    classDef output fill:#86EFAC,stroke:#DCFCE7,color:#14532D,stroke-width:2px;
```

模型的核心思想：**绝对丰度由对照分支解释，扰动响应由低秩分支解释**。Delta-PCA 基底在「treatment − 精确匹配 control」的 Δ 矩阵上拟合，天然聚焦评分所关心的相对响应；全链路 NaN-aware（observed mask 参与所有损失）。

***

## 3. 核心创新：可微评分对齐（L2b）

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, Microsoft YaHei, sans-serif", "background": "#FFFFFF", "primaryColor": "#F8FAFC", "primaryTextColor": "#1E293B", "primaryBorderColor": "#CBD5E1", "lineColor": "#94A3B8"}}}%%
flowchart LR
    A["官方 FC 模块<br/>corr(ŷ−c, y−c)<br/>非平滑 · 不可反传"]:::problem
    B["常见做法<br/>训后校准 / 代理 loss<br/>与评分目标仍错位"]:::problem
    C["L2b: masked 逐样本<br/>FC-PCC 可微损失<br/>λ=0.07 + 5ep warmup"]:::solution
    D["7 字段精确匹配<br/>control bank<br/>仅训练行"]:::control
    E(["OOF 0.6698 → 0.6771<br/>12/12 配对正向<br/>FC 模块 +0.0105"]):::result

    A --> B
    B -.被替换.- C
    D -.监督信号.-> C
    C --> E

    classDef problem fill:#FCA5A5,stroke:#FEE2E2,color:#7F1D1D,stroke-width:2px;
    classDef solution fill:#7DD3FC,stroke:#E0F2FE,color:#0C4A6E,stroke-width:2px;
    classDef control fill:#FDE68A,stroke:#FEF3C7,color:#78350F,stroke-width:2px;
    classDef result fill:#86EFAC,stroke:#DCFCE7,color:#14532D,stroke-width:3px;
```

**相对已有方法新在哪里**：不是训后校准，也不是代理 loss 近似，而是把评分器的非平滑统计量直接嵌入训练循环，并证明其对多 seed 稳健（三 seed Δproxy 分别 +0.0081 / +0.0080 / +0.0067）。单一可微 FC-PCC 项贡献的 +0.0073 超过所有外部特征方向之和。

***

## 4. 验证与审计体系

### 4.1 双验证协议

| 体系                  | 设计                   | 检验目标     | L2b              |
| ------------------- | -------------------- | -------- | ---------------- |
| V2 四折 LOSO × 3 seed | query 菌株 seen-化合物进训练 | 菌株/总体稳健性 | **0.6771**       |
| V3 五折化合物分组          | query 菌株处理样本零进训练     | 未见化合物外推  | 0.6171（S1=0.474） |

### 4.2 Oracle 上限审计：瓶颈定位

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, Microsoft YaHei, sans-serif", "background": "#FFFFFF", "primaryColor": "#F8FAFC", "primaryTextColor": "#1E293B", "primaryBorderColor": "#CBD5E1", "lineColor": "#94A3B8"}}}%%
flowchart LR
    A(["L2b 现状<br/>V3 五折 0.6171"]):::now
    B(["Coefficient Oracle<br/>z = z* 解析解<br/>0.8074"]):::oracle
    C(["上限差距 +0.19<br/>架构不是瓶颈<br/>缺的是条件→z* 映射"]):::gap

    A ==>|"固定 decoder<br/>逐样本解析求 z*"| B
    B ==> C

    classDef now fill:#FDBA74,stroke:#FFEDD5,color:#7C2D12,stroke-width:2px;
    classDef oracle fill:#7DD3FC,stroke:#E0F2FE,color:#0C4A6E,stroke-width:3px;
    classDef gap fill:#F9A8D4,stroke:#FBCFE8,color:#831843,stroke-width:2px;
```

后续蒸馏实验（L9A/B/C）证明 z\* 中可从条件预测的成分已被主干捕获，剩余为样本特异——**距 0.70 的 +0.023 缺口属于科学问题（需要任务贴近的新扰动数据），不属于调参**。这统一解释了下面全部 8 个负结果。

### 4.3 外部数据负结果链（8 方向统一范式审计）

| 方向                          | 判定证据（全部冻结）                                 |
| --------------------------- | ------------------------------------------ |
| ChEMBL 靶点 + Alliance 同源（L5） | 增益 100% 来自截距（intercept-only 消融差 1e-8）      |
| Dryad HIP/HOP kNN（L6）       | 信号真实（shuffle 归零）但 16 化合物池天花板：+0.0005       |
| 分子结构 → HIP（L7/L10）          | scaffold 严格复核后真实信号 +0.05（p=0.0001），下游净效应为负 |
| STRING 网络 / 序列嵌入（L4）        | Δproxy +0.0006，远低于门槛                       |
| latent teacher 蒸馏（L9）       | 增益为初始化伪影（独立重初始化后消失，−0.010）                 |
| Mean-Delta / 场景校准           | 折间不稳定 / cross-fitted α 折间 0.31–1.17        |

**核心结论：可预测性 ≠ 可用性**——外部信号可以统计真实但「太稀薄」，无法克服旁路过拟合。完整决策文档 + JSON 存档随包分发，供社区复用。

### 4.4 数据完整性工程

* **INDEX MISALIGNMENT 审计**：发现早期 split 生成的索引语义错位（每折 \~1,500 行验证数据混入训练），完成根因定位 → 调用链逐文件判定 → 受污染实验全量重训 → 双验证协议重建；

* **评分器独立审计**：从官方公式独立重写六模块评分器，与主评分器在全部 OOF 预测上逐位一致（max |Δ| = 1.11e-16）；

* **防覆盖机制**：每折 run\_manifest（config SHA + sample\_id SHA）一致才复用、不一致立即停止。

***

## 5. 复现流程：三条主命令

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, Microsoft YaHei, sans-serif", "background": "#FFFFFF", "primaryColor": "#F8FAFC", "primaryTextColor": "#1E293B", "primaryBorderColor": "#CBD5E1", "lineColor": "#94A3B8"}}}%%
flowchart LR
    A["1 build_embeddings<br/>无需执行<br/>零外部数据"]:::step
    B["2 train.py<br/>3 seed 从头训练<br/>~7 min/seed"]:::step
    C["3 predict.py<br/>冻结推理<br/>仅读 test metadata"]:::step
    D["validate_submission.py<br/>格式校验"]:::check
    E(["prediction.csv<br/>4,454 × 4,422<br/>与提交逐位一致"]):::result

    A -.跳过.- B --> C --> D --> E

    classDef step fill:#7DD3FC,stroke:#E0F2FE,color:#0C4A6E,stroke-width:2px;
    classDef check fill:#FDE68A,stroke:#FEF3C7,color:#78350F,stroke-width:2px;
    classDef result fill:#86EFAC,stroke:#DCFCE7,color:#14532D,stroke-width:3px;
```

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

备选：把官方文件放入 `input/` 目录（train\_val metadata/proteome + test metadata）可省略显式路径参数。训练需要 test metadata 仅用于构建 train+test **联合实体词表**（编码索引与 checkpoint 维度一致），**不读取任何测试表型**——数据加载器默认 `allow_test_labels=False` 硬隔离。

<details>
<summary><b>环境要求</b>（点击展开）</summary>

* Python ≥ 3.10, PyTorch ≥ 2.0（CUDA 可选，CPU 可运行但较慢）

* `pip install -r requirements.txt`

* 资源：GPU 显存 \~6GB；训练 \~7 min/seed（RTX 4060）；推理 \~2 min；磁盘 \~2GB

</details>

<details>
<summary><b>数据与预处理契约</b>（点击展开）</summary>

* 以 `sample_ID` 为唯一键对齐 metadata 与 proteome（不依赖 CSV 行顺序）

* log2 仅对有限且 >0 的原始强度；NaN 保留为 mask，不解释为 0

* 缺失填充仅用 train 列中位数；**特征契约 4,422 蛋白**由 `split_final=='train'` 行缺失率 <80% 决定（`artifacts/feature_contract.json` 含 SHA256）

* 模型内部用全 5,243 列训练，输出子集到契约列（值与全空间预测逐位一致）

* 全部统计量（control bank / 中位数 / PCA 基底 / 词表）仅由 train 行拟合

</details>

***

## 6. prediction.csv 获取

`prediction.csv`（193 MB）超出 GitHub 单文件 100 MB 限制，**不随 git 仓库分发**（官方提交说明允许「稳定下载链接 + SHA256」）。获取方式三选一，结果与正式提交逐位一致：

| 方式             | 说明                                                    |
| -------------- | ----------------------------------------------------- |
| **提交包**        | 官方提交附件 `AI4R_AIVC_GOAI_Virtual_Yeast_代码材料.zip` 内含完整文件 |
| **重新生成**（推荐）   | 运行 §5 命令 2 + 3，输出 SHA256 应等于下表值（同硬件逐位一致）              |
| **Release 附件** | 本仓库 Release `v1.0-semifinal` 附带（上传后可用）                |

**核对值**：`db113d3ecad727a560ebad69aa18667035ff8af059639881189be9c5cdcb8f0f`

跨硬件时存在浮点级差异（CUDA 非确定性归约），以 manifest 中 per-seed ensemble SHA 与 proxy 复核为准。

***

## 7. 目录结构

```text
GOAI-Virtual-Yeast-Semifinal/
├── README.md                          # 本文件
└── VC_GOAI_Virtual_Yeast_reproduction/
    ├── README.md                      # 复现包主说明（首页信息表 + 训练细节）
    ├── requirements.txt
    ├── configs/final.yaml             # 冻结配置
    ├── prediction.csv                 # ⚠️ 不入 git（见 §6 获取方式）
    ├── REPRODUCIBILITY_MANIFEST.json  # 全包清单 + 逐文件 SHA256
    ├── src/                           # 数据 / 模型 / 损失 / 评分 / 训练组件
    │   ├── data_processor_v5.py       # 便携性补丁版（数学不变）
    │   ├── model_v5.py / model_v5b.py # 模型定义
    │   ├── losses_v5.py               # masked huber / PCC loss
    │   ├── scoring_official_v5b.py    # control bank（7 字段精确匹配）
    │   └── train_components.py        # delta-PCA / FC-cache / FC loss
    ├── scripts/
    │   ├── build_embeddings.py        # no-op（零外部数据）
    │   ├── train.py                   # 命令 2：从头训练
    │   ├── predict.py                 # 命令 3：推理 + prediction.csv
    │   └── validate_submission.py     # 格式校验
    ├── checkpoints/                   # 3 seed checkpoint + manifest
    ├── artifacts/                     # 特征契约 + prediction manifest
    ├── external_data/                 # 声明：无外部数据
    ├── tests/test_smoke.py            # 冒烟测试
    └── LICENSES/
```

***

## 8. 已知限制

1. 模型内部使用 5,243 蛋白列，prediction 输出时子集到 4,422 契约列（对评分无影响，完整披露见复现包 README §3）
2. 跨硬件复现存在浮点级差异，预期 Δproxy < 0.001（3 seed 平均进一步压低方差）
3. checkpoint 为 full-data refit（8,958 行全部拟合）；0.6771 为 V2 四折 OOF 本地 proxy，非测试分数
4. 提交冻结后的追加审计（checkpoint soup +0.0013 / 分组均衡训练 −0.0036 / 组级 latent teacher 差距不足）均按同范式冻结，未切换提交

***

## 9. 可复用资产

* 可微 FC-PCC 评分对齐实现（`src/losses_v5.py` + `train_components.py`）

* NaN-aware 低秩分解架构（`src/model_v5.py`）

* 双验证 split 协议（V2 LOSO / V3 化合物分组）

* Oracle 上限审计思路（固定 decoder 解析求 z\*）

* 8 方向负结果决策文档 + JSON（REPRODUCIBILITY\_MANIFEST.json → negative\_results）

* 对照实验设计三层方法论：shuffle 保留增益 ≠ 信号有效 / 对照须独立重初始化 / 高相关空间须报告 real−shuffle 增量

***

## 10. License

MIT License. See `VC_GOAI_Virtual_Yeast_reproduction/LICENSES/` for details.
