# 基于知识网络与蚁群优化的专利技术机会发现系统

## 一、研究背景与动机

在技术创新日益加速的时代，如何从海量专利文献中识别出有价值的技术机会，是技术管理与创新战略研究的核心问题。传统方法依赖领域专家的主观判断，不仅耗时费力，还容易遗漏跨领域的新兴技术组合。

本项目以 Ren & Zhao (2021) 发表于 *Technovation* 的研究为理论基础，实现了一套完整的、数据驱动的专利技术机会发现系统。该方法的核心思想是：**将专利中的技术术语及其共现关系建模为知识网络（Domain Knowledge Network, DKN），通过网络结构特征量化技术机会的新颖性与价值，再利用蚁群优化算法在网络中搜索最优的技术组合子网**。

本项目选择**具身智能（Embodied Intelligence）**作为应用领域，数据来源于 Derwent Innovation 专利数据库，涵盖 2012–2026 年间的国际专利。

---

## 二、完整 Pipeline 概览

系统采用六步流水线设计，每一步独立可运行，产物通过文件系统持久化，下游步骤自动检测上游产物是否需要重建。

```
Step0: 数据导入与清洗
  │  将 Derwent 原始 CSV（8809 条、44 列中文列头）解析为标准化英文专利记录
  │  产物: patents.csv (8620 条有效记录)
  ▼
Step1: NLP 处理 + 知识网络构建
  │  对每篇专利进行句法分析和依存关系提取，构建两种不同时间范围的知识网络
  │  产物: hdkn.pkl.gz (28,672 节点), pdkn.pkl.gz (55,806 节点)
  ▼
Step2: 回归分析工作流（4-Run）
  │  以专利被引次数为因变量，子网特征为自变量，
  │  通过负二项回归识别影响专利价值的关键网络特征，提取显著系数
  │  产物: objective_coefficients.json (New_e=+2.17, Eigen=-170.99, Constraint=-111.86)
  ▼
Step3: 蚁群优化（ACO）搜索
  │  以回归系数构建目标函数 Z，在 PDKN 上用蚁群算法搜索使 Z 最大的 15 节点子网
  │  产物: aco_candidates.json (每组 80 个候选子网)
  ▼
Step3.5: 候选合并与多样性过滤 + 富化
  │  合并所有 ACO 运行的候选子网，去重 + 80% 节点重叠过滤，
  │  选出 Top-30 多样化子网，对每个子网进行节点/边分类和专利信息富化
  │  产物: aco_merged_top30_enriched.json (30 个富化子网)
  ▼
Step4: RAG 报告生成
  │  将富化子网的结构数据和专利证据注入 LLM prompt，
  │  按原文方法论生成结构化的技术机会解读报告
  │  产物: rank_N.md (Markdown 格式分析报告)
```

---

## 三、Step0 — 数据导入与清洗

**脚本**：`scripts/step0_import_patents.py`

**做什么**：从 Derwent Innovation 导出的原始 CSV 中解析、清洗、标准化专利数据，生成下游流水线统一使用的 `patents.csv`。

**具体流程**：

1. **字段映射**：将 44 列中文列头映射到标准英文字段（patent_id、title、abstract、app_year 等），优先选取 DWPI 规范化版本的标题和摘要
2. **年份过滤**：剔除申请年早于 2012 年的专利
3. **多层语言核验**：
   - 非英文段落剥离：检测并剥离 WIPO 多语种专利摘要中附加的非英文翻译段落
   - 非拉丁字符比例检测：CJK、西里尔文、阿拉伯文等字符占比超过 5% 则排除
   - 重音拉丁字符检测：识别德语（ä, ö, ü, ß）、法语（é, è, ê, ç）等高频重音字符
   - OCR 乱码过滤：剥离专利摘要中混入的 OCR 图标签文本
   - Lemma 级非英语功能词检测：检测残留的非英语功能词作为安全网
4. **去重**：同一公开号保留最新版本

**清洗结果**：保留 **8620** 条有效英文专利记录，时间跨度 **2012–2026 年**。

| 申请年 | 专利数 | 申请年 | 专利数 |
|--------|--------|--------|--------|
| 2012 | 5 | 2020 | 690 |
| 2013 | 35 | 2021 | 733 |
| 2014 | 30 | 2022 | 1016 |
| 2015 | 27 | 2023 | 1204 |
| 2016 | 59 | 2024 | 1566 |
| 2017 | 185 | 2025 | 2126 |
| 2018 | 337 | 2026 | 67 |
| 2019 | 540 | **合计** | **8620** |

2023–2025 年占比超过 56%，体现了具身智能领域近年来的研究热潮。

---

## 四、Step1 — NLP 处理与知识网络构建

**脚本**：`scripts/step1_build_networks.py`

**做什么**：对每篇专利进行 NLP 处理，提取技术术语和术语间的共现关系，合并为两种不同时间范围的知识网络（DKN）。

**NLP 处理流程**（针对每篇专利的标题+摘要）：

1. **分词与词形还原**：使用 spaCy（en_core_web_sm）进行分词、POS 标注和词形还原
2. **连字符与复合词处理**：将 "machine-learning"、"machine learning" 统一为 "machine-learning" 节点
3. **停用词过滤**：spaCy 默认停用词 + 专利法律套话（wherein、thereof）+ 泛化功能词（unit、step）
4. **依存关系提取**：从句法依存树中提取名词性依存对作为边（代表两个技术术语的语义关联）
5. **补边机制**：对孤立节点通过共现窗口补充边，确保子网连通性

**网络合并与加权**：

将所有单专利图合并为一张大图。同一术语对若在多篇专利中出现，其年份列表叠加，通过时间衰减公式计算边权重：

> **weight = Σ α^(T − year)**

其中 α 为衰减因子（由 Step2 的 Run 1 数据驱动选择），T 为参考年份。

**构建两种网络**：

| 网络 | 纳入的专利 | 参考年份 T | 节点数 | 边数 | 用途 |
|------|----------|-----------|--------|------|------|
| **HDKN** | app_year ≤ 2022 | 2022 | 28,672 | 133,350 | 特征提取基准、回归训练 |
| **PDKN** | 全时段 | 2026 | 55,806 | 295,792 | ACO 搜索空间 |

HDKN 仅包含历史期数据，确保回归训练不包含未来信息（防止信息泄漏）。PDKN 相比 HDKN 节点增长 94.6%、边增长 121.8%，反映了 2023–2026 年具身智能领域的大量新技术术语和技术组合。

---

## 五、Step2 — 回归分析工作流

**脚本**：`scripts/regression_workflow.py`

**做什么**：以 HDKN 时间段内的专利为样本，用被引次数（forward citations）作为因变量衡量专利影响力，用子网结构特征作为自变量，通过负二项回归（NB）和零膨胀负二项回归（ZINB）模型，识别哪些网络特征显著影响专利价值，提取显著系数供 ACO 搜索使用。

### 5.1 子网特征体系

对 HDKN 中每篇专利，提取其相关术语构成的 15 节点子网，计算 7 种特征：

**新颖性特征**（衡量子网包含多少"新"知识元素）：

| 特征 | 定义 | 直觉含义 |
|------|------|----------|
| New_n | 子网是否包含年份 ≥ 90% 分位数的节点 | 是否引入了新出现的技术术语 |
| New_e | 子网是否包含年份 ≥ 90% 分位数的边 | 是否形成了新的技术组合 |
| Min_pn | 覆盖子网所有节点和边所需的最少专利数 | 子网整合了多少不同来源的知识 |

**常规性特征**（衡量子网依赖多少"主流"知识元素）：

| 特征 | 定义 | 直觉含义 |
|------|------|----------|
| Con_n | 子网节点 Strength（加权度）的中位数 | 术语在已有文献中的活跃程度 |
| Con_e | 子网边 Weight 的中位数 | 术语组合在已有文献中的常见程度 |

**结构特征**（衡量子网在网络拓扑中的位置）：

| 特征 | 定义 | 直觉含义 |
|------|------|----------|
| Eigen | 子网节点特征向量中心性的平均值 | 术语是否处于知识网络的核心位置 |
| Constraint | 子网节点 Burt 网络约束的最小值 | 是否存在结构洞（跨越不同知识集群的桥梁位置） |

### 5.2 4-Run 工作流

回归分析采用系统化的 4 步流程，逐步从模型选择收敛到 ACO 目标函数的构建：

#### Run 1 — 衰减因子选择（Alpha Selection）

**做什么**：对候选 α ∈ {0.0, 0.1, …, 1.0}，逐一重构 HDKN 权重、提取特征、拟合 NB/ZINB，通过 Log-Likelihood 选取最优衰减因子。

**结果**：NB 最优 α = 0.5（LL = -1894.94），后续 Run 采用此值。

#### Run 2 — 全模型 + 共线性检验

**做什么**：使用 α = 0.5，以全部 7 个子网特征 + 3 个控制变量（Back_cite、Assignee、Total_pat）拟合 NB 和 ZINB，输出 VIF 共线性检验报告。

**关键结果**：New_e（p=0.005）、Eigen（p<0.001）、Constraint（p=0.025）三个变量在 NB 中达到 5% 显著性水平；New_n、Min_pn、Con_n 不显著。Vuong 检验 V = 5.63，ZINB 显著优于 NB。

#### Run 3 — 选定变量 + 控制变量

**做什么**：仅保留 Run 2 中显著的 New_e、Eigen、Constraint 加控制变量拟合，验证变量选择的稳健性。

**结果**：AIC = 3814.08，优于 Run 2 全模型（AIC = 3818.36），精简模型更优。

#### Run 4 — 仅选定变量（ACO 目标系数来源）

**做什么**：不含控制变量，仅用 New_e、Eigen、Constraint 拟合，获取纯子网特征对影响力的边际效应，作为 ACO 搜索的目标函数系数。

**结果**：

| 变量 | 系数 | p 值 | 含义 |
|------|------|------|------|
| **New_e** | **+2.1729** | 0.001 | 新技术组合显著提升专利影响力 |
| **Eigen** | **-170.9885** | <0.001 | 网络边缘的术语比核心热门术语更有突破价值 |
| **Constraint** | **-111.8608** | 0.012 | 结构洞位置（跨集群桥梁）更易产生高影响力专利 |

**核心发现**：高价值技术机会存在于知识网络的边缘与结构洞处——包含新颖技术组合、远离网络中心、跨越不同知识集群的子网，最有可能产生高影响力专利。

---

## 六、Step3 — 蚁群优化（ACO）搜索

**脚本**：`scripts/step3_pdkn_aco.py`

**做什么**：基于 Run 4 的回归系数构建线性目标函数，在 PDKN（包含 2012–2026 全时段专利的知识网络）上用蚁群优化算法搜索使目标函数值最大的 15 节点子网——即最有可能成为高价值技术机会的技术术语组合。

### 6.1 目标函数

> **Z = 2.1729 × New_e − 170.9885 × Eigen − 111.8608 × Constraint**

ACO 搜索使 Z 最大化的子网——同时具备高新颖性（New_e 大，包含新的技术组合）、低核心性（Eigen 小，远离网络核心）和低约束性（Constraint 小，处于结构洞位置）。

### 6.2 新颖度引导机制

算法基于经典 MAX-MIN Ant System（MMAS），并引入三项新颖度引导机制，提高发现新颖技术组合的概率：

1. **节点新颖度增强**：每个节点的启发式信息 η 根据其关联新边的比例增强——η = strength × (1 + γ × novelty_ratio)，γ = 2.0
2. **新颖蚂蚁群体**：30% 的蚂蚁从高新颖度节点出发，更倾向于探索新技术组合
3. **新边 bonus**：新颖蚂蚁经过新边（PDKN 有而 HDKN 无）时，选择概率乘以 6.0 的 bonus

### 6.3 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| num_ants | 200–500 | 蚂蚁数量（多组对比） |
| num_generations | 400 | 最大迭代代数 |
| early_stop_patience | 100 | 连续无改善代数阈值 |
| top_k_per_gen | 5 | 每代用于信息素更新的最优解数量 |
| pheromone.alpha | 1.5 | 信息素因子 |
| pheromone.rho | 0.85 | 信息素保留率 |
| subnetwork_size | 15 | 子网节点数 |
| novelty.weight (γ) | 2.0 | 新颖度启发增强系数 |
| novelty.novel_ant_ratio | 0.3 | 新颖蚂蚁占比 |
| novelty.new_edge_bonus | 6.0 | 新边 bonus 乘数 |

### 6.4 搜索结果

500 只蚂蚁取得最高 Z 值（+0.6343），在最终代的种群平均/中位数也明显优于其他配置。

---

## 七、Step3.5 — 候选合并与多样性过滤 + 子网富化

**脚本**：`scripts/merge_aco_candidates.py`

**做什么**：将所有 ACO 运行（12 个配置 × 80 个候选）的结果统一合并、去重、按节点重叠率进行多样性过滤，选出 Top-30 多样化子网，然后对每个子网进行结构化富化——为每个节点/边标注类型（new/marginal/special/conventional）、匹配代表性专利信息、计算新颖来源和可行性锚点。

### 7.1 合并与过滤流程

| 步骤 | 数量 |
|------|------|
| 原始候选子网（12 组 × 80 个） | 960 |
| 去重后（相同节点集 + 相同分数） | 846 |
| **80% 节点重叠过滤后 Top-30** | **30** |

最终 30 个子网的 Z 分数范围为 [+0.5406, +0.6343]，全部为正值。

### 7.2 子网富化（aco_to_rag.py）

对每个 Top-30 子网，执行以下富化操作：

**节点分类**（对齐原文 Table 3）：

| 类型 | 判定条件 | 含义 |
|------|---------|------|
| **new** | 不在 HDKN 中，或首现年份 ≥ 90% 分位数 | 全新技术术语 |
| **marginal** | 在 HDKN 中但特征向量中心性极低（< 1e-4） | 网络边缘的冷门术语，具有突破潜力 |
| **conventional** | 其余 | 成熟的主流技术术语 |

**边分类**：

| 类型 | 判定条件 | 含义 |
|------|---------|------|
| **new** | 不在 HDKN 中，或首现年份 ≥ 90% 分位数 | 全新的技术关联 |
| **special** | 仅出现在 1 篇专利中 | 罕见但可能有突破性的关联 |
| **conventional** | 其余 | 成熟的技术关联 |

**附加富化信息**：

- 每个节点注入 `eigen`（特征向量中心性）和 `is_marginal`（是否低于 PDKN 正 Eigen 值的第 70 百分位）
- 所有节点匹配 `representative_patent`（按引用数排序选取最佳代表专利）
- special 边匹配 `sole_patent`（唯一支撑专利，含标题和摘要）
- new 边匹配 `representative_patent`
- conventional 边匹配 `top_patents`（引用数最高的 3 篇）
- 全局标注 `novelty_sources`（新颖来源节点列表）和 `feasibility_anchors`（可行性锚点列表）
- 计算 `z_score` 和 `feature_scores`（各特征分项得分）

**产物**：默认写入 `outputs/runs/<run_id>/03_merged_rag/aco_merged_top30_enriched.json`——结构化 JSON，每个子网包含完整的分类信息、专利证据和评分，供 Step4 RAG 报告生成直接使用。`data/processed/rag/` 仅作为兼容历史流程的显式导出位置。

---

## 八、Step4 — RAG 报告生成

**脚本**：`scripts/step4_rag_report.py`

**做什么**：读取 Step3.5 产出的富化子网 JSON，为每个技术机会子网构建包含结构数据和专利证据的 prompt，调用 DeepSeek API（OpenAI 兼容接口）按原文方法论生成结构化的 Markdown 分析报告。

### 8.1 数据注入 prompt 的流程

对每个子网，执行以下数据提取和 prompt 构建：

1. **Z 值分解注入**：从 `z_score` 和 `feature_scores` 构建得分摘要（如 Z=0.6343, New_e=1.0, Eigen=0.0029），让 LLM 了解该子网的优势来源

2. **novelty_sources 子簇注入**：将新颖来源节点列表以原形词形式展示，标注"这些节点形成紧密互连的新兴技术单元，应作为整体分析"

3. **节点标注**：
   - new 节点标记 ★（新兴技术术语）
   - marginal 节点标记 ◇(Eigen=...)（网络边缘节点）
   - 通过 `is_marginal` 字段或 `eigen=0.0` 兜底判定

4. **边选取（分层配额策略）**：
   - special 边全选（它们是 novel combination 的核心线索）
   - 剩余配额在 new 和 conventional 之间大致均分
   - 确保每种边类型都有代表进入 prompt（top_n_edges=10）

5. **边描述增强**：
   - special 边附加 `△ 唯一支撑专利: [专利号] 标题`
   - new 边附加 `★ 该关联在 HDKN 中不存在` + 代表专利
   - conventional 边附加 `□ 成熟技术关联`

6. **专利证据按优先级收集**（top_k_patents=15）：
   - P1: special 边的 sole_patent（消歧关键）
   - P2: new 边的代表专利
   - P3: new 节点的代表专利
   - P4: marginal 节点的代表专利
   - P5: conventional 边的 top_patents

### 8.2 Prompt 设计（对齐原文方法论）

**System Prompt** 包含：
- 方法论背景：向 LLM 解释 ★/◇/△/□ 各分类的含义，以及 Novel Combination 的定义
- 纪律要求：绝对忠于证据、强制引证格式、信息缺失处理

**User Prompt** 引导 LLM 按四段式结构输出：

| 段落 | 内容 | 对齐原文步骤 |
|------|------|-------------|
| **1. 技术机会命名** | 从新颖组合中提炼 ≤15 字的命名 | 综合概括 |
| **2. 重要节点与边识别** | 按 ★/◇/△/□ 分类列出并解读 | 原文步骤 1：分类 |
| **3. 新颖组合分析** | 提取 2-3 个 novel combinations 并解读技术含义 | 原文步骤 2-3：提取+推断 |
| **4. 技术机会综合解读** | 归类为"技术手段+应用功能"二维描述 | 原文步骤 4：综合 |

**新颖组合分析的优先级**：
1. 最高：涉及特殊边(△)的组合（唯一支撑专利提供最具体的技术语义线索）
2. 次高：新兴节点子簇（多个 ★ 节点通过 new edges 互连形成的技术单元）
3. 第三：新兴边连接的节点对

### 8.3 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| model | deepseek-chat | LLM 模型 |
| temperature | 0.1 | 低温度降低幻觉 |
| max_tokens | 4096 | 最大生成长度 |
| top_n_edges | 10 | 边选取数量（分层配额） |
| top_k_patents | 15 | 最多提供的专利证据数量 |
| evidence_max_chars | 60000 | 证据库字符串上限 |

**产物**：`outputs/runs/<run_id>/04_rag_reports/rank_N.md`——每个子网一份结构化 Markdown 报告。

---

## 九、系统实现

### 9.1 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.8+ |
| NLP | spaCy（en_core_web_sm） |
| 图计算 | NetworkX |
| 统计建模 | statsmodels（NB、ZINB） |
| 数据处理 | pandas、numpy |
| 配置管理 | YAML + Python config |
| 日志系统 | loguru（脚本）、logging（库） |
| RAG 报告 | DeepSeek API（OpenAI 兼容接口） |

### 9.2 关键设计特性

- **可恢复执行**：每步产物带有 metadata（含上游哈希值），重跑时自动检测是否需要重建
- **Run ID 隔离**：每次运行通过唯一 run_id 隔离产物，便于对比不同参数的结果
- **防信息泄漏**：严格验证 HDKN 的参考年份等于 hist_end_year，确保训练数据不包含未来信息
- **NLP 缓存**：文件级 NLP 缓存机制避免重复处理，支持增量更新
- **分层配额选边**：确保 special / new / conventional 三种边类型在 RAG prompt 中均有代表
- **优先级专利收集**：按解读重要性排序专利，防止关键证据被 token 截断

---

## 十、目录结构

```
.
├── configs/                             # 配置文件
│   ├── aco_config.yaml                 # ACO 算法完整配置（含参数优化注释）
│   ├── rag_config.yaml                 # RAG 报告生成配置（模型/参数/证据上限）
│   └── features.yaml                   # 回归特征选择
├── data/                                # 数据目录
│   ├── raw/
│   │   ├── csv2026-03-18-15-53-59.csv  # Derwent 原始导出（8809 条）
│   │   ├── patents.csv                 # 清洗后标准数据（8620 条）
│   │   └── patents_excluded.csv        # 被排除记录（附原因）
│   └── processed/
│       └── rag/                        # 历史兼容导出目录（默认不写）
│           ├── aco_merged_top30_enriched.json   # 富化子网 JSON
│           ├── aco_merged_top30_candidates.json  # 精简候选列表
│           └── merge_summary.json               # 合并摘要
├── outputs/runs/<run_id>/               # 运行产物（按 run_id 隔离）
│   ├── 01_networks/                    # Step1: HDKN + PDKN
│   ├── 02_regression/                  # Step2: 回归结果与报告
│   ├── 03_aco*/                        # Step3: 各配置 ACO 搜索结果
│   ├── 03_merged_rag/                  # Step3.5: 当前 run 的合并/富化结果
│   └── 04_rag_reports/                 # Step4: RAG 分析报告
│       ├── rank1_*.md                  # 各 rank 的 Markdown 报告
│       └── rag_meta.json              # 报告生成元数据
├── scripts/                             # 可执行脚本
│   ├── step0_import_patents.py         # Step0: 数据清洗
│   ├── step1_build_networks.py         # Step1: 网络构建
│   ├── step2_hdkn_regression.py        # Step2: 单步回归（简易版）
│   ├── regression_workflow.py          # Step2: 4-Run 回归工作流（推荐）
│   ├── step3_pdkn_aco.py              # Step3: ACO 搜索
│   ├── merge_aco_candidates.py         # Step3.5: 候选合并 + 富化
│   ├── step4_rag_report.py            # Step4: RAG 报告生成
│   ├── run_all.py                      # Pipeline 编排器（Step1→4）
│   └── alpha_selection.py              # 独立 α 选择
├── src/patent_opportunity_analysis/     # 核心代码库
│   ├── nlp_utils.py                    # NLP 文本处理
│   ├── patent_graph.py                 # 单专利图构建
│   ├── dkn_builder.py                  # DKN 构建与合并
│   ├── feature_extraction.py           # 子网特征计算
│   ├── feature_registry.py             # 特征注册表
│   ├── regression_model.py             # NB / ZINB 模型拟合
│   ├── aco_search.py                   # ACO 算法（含新颖度引导）
│   ├── aco_to_rag.py                   # ACO 子网 → 富化 JSON（节点/边分类 + 专利匹配）
│   ├── rag_report_generator.py         # RAG 报告生成器（数据提取 + LLM 调用）
│   ├── rag_prompts.py                  # LLM Prompt 模板（System + User）
│   ├── rag_patents.py                  # 专利证据检索（CSV 查找 + 内嵌回退）
│   └── utils/                          # 工具模块
├── docs/                                # 技术文档
│   ├── methodology.md                  # 变量定义、权重计算、特征说明、RAG 方法论
│   ├── field_mapping.md                # CSV 字段映射
│   └── nlp_notes.md                    # NLP 分词与停用词说明
├── tests/                               # 单元测试与集成测试
├── requirements.txt
├── setup.py
├── readme.md                            # 快速入门
└── PROJECT_OVERVIEW.md                  # 本文档
```

---

## 十一、使用指南

### 11.1 环境搭建

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 11.2 完整运行流程

```bash
# Step 0: 数据清洗
python scripts/step0_import_patents.py

# Step 1: 构建知识网络
python scripts/step1_build_networks.py --hist-end-year 2022

# Step 2: 回归工作流（4 个 Run 依次执行）
python scripts/regression_workflow.py --run-id <ID> --run 1 --alphas "0.0,0.1,...,1.0"
python scripts/regression_workflow.py --run-id <ID> --run 2 --alpha 0.5
python scripts/regression_workflow.py --run-id <ID> --run 3 --vars "New_e,Eigen,Constraint"
python scripts/regression_workflow.py --run-id <ID> --run 4 --vars "New_e,Eigen,Constraint"

# Step 3: ACO 搜索（可多次运行不同蚂蚁数）
python scripts/step3_pdkn_aco.py --run-id <ID> --test-num-ants 500 --force

# Step 3.5: 合并候选并生成富化 JSON
python scripts/merge_aco_candidates.py --run-id <ID> --top-n 30 --overlap 0.8

# Step 4: 生成 RAG 分析报告（需设置 DEEPSEEK_API_KEY 环境变量）
export DEEPSEEK_API_KEY=your_api_key
.venv/bin/python scripts/step4_rag_report.py --run-id <ID>

# 仅生成指定 rank 的报告
.venv/bin/python scripts/step4_rag_report.py --run-id <ID> --rank 1 --force
```

### 11.3 快速测试

```bash
python scripts/run_all.py --limit 500
```

使用前 500 条专利快速验证 Step1→Step3 的整个流程。

---

## 十二、参考文献

Ren, H., & Zhao, Y. (2021). Technology opportunity discovery based on constructing, evaluating, and searching knowledge networks. *Technovation*, *101*, 102196. https://doi.org/10.1016/j.technovation.2020.102196

---

*基于 Run ID: 20260331_195849_d009849ba_h2022 的数据撰写*
*最后更新：2026-04-03*
