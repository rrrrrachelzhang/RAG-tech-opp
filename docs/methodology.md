# 方法论与变量定义参考文档

> 基于 Ren, H., & Zhao, Y. (2021) 论文，整理关键定义并核对代码实现状态。

---

## 1. 变量定义（对照原文）

### 1.1 因变量（Dependent Variable）
- **Cited**: 样本专利的 forward citations count（被引用次数）

### 1.2 解释变量（Explanatory Variables）

| 特征 | 原文定义 | 代码实现 | 状态 |
|------|---------|---------|------|
| **New_n** | 子网是否包含 Year_n ≥ 90th percentile 的节点（二元 0/1） | `feature_registry._compute_new_n`，使用 p90 阈值 | ✅ 已对齐 |
| **New_e** | 子网是否包含 Year_e ≥ 90th percentile 的边（二元 0/1） | `feature_registry._compute_new_e`，使用 p90 阈值 | ✅ 已对齐 |
| **Min_pn** | 覆盖子网所有节点和边的最小专利数（set cover） | `feature_registry._compute_min_pn`，贪心 set cover | ✅ 已对齐 |
| **Con_n** | Si 中所有节点 Strength 的中位数 | `feature_registry._compute_con_n`，`np.median()` | ✅ 已对齐 |
| **Con_e** | Si 中所有边 Weight 的中位数 | `feature_registry._compute_con_e`，`np.median()` | ✅ 已对齐 |
| **Eigen** | HDKN 上 eigenvector centrality，取子网节点平均 | `feature_extraction.compute_eigen_centrality` | ✅ 已对齐 |
| **Constraint** | HDKN 上 Burt's constraint，取子网节点最小值 | `feature_registry._compute_constraint` | ✅ 已对齐 |

### 1.3 控制变量（Control Variables）
- **Back_cite**: 后向引用数（backward_cites）
- **Assignee**: 专利权人类型（组织=1 / 个人=0）
- **Total_pat**: HDKN 时间段最后 5 年内，focal patent 的所有 IPC 类别的专利总数

### 1.4 Year_n / Year_e 定义
- **Year_n**：节点在 HDKN 中首次出现的年份（`year_min` 属性）
- **Year_e**：边在 HDKN 中首次出现的年份（`year_min` 属性）

---

## 2. 权重与衰减因子计算

### 2.1 时间衰减权重公式

```
weight = Σ α^(T - year)
```

- `α`：衰减因子（decay factor），通过 Run 1 Alpha Selection 选取最优值
- `T`：参考年份（reference year）
- `year`：专利申请年（application year）

**HDKN 的 T** = `hist_end_year`（历史截止年）
**PDKN 的 T** = `max_year`（数据最大申请年，仅用于展示，不用于特征/回归）

### 2.2 权重计算流程

```
专利数据 → build_patent_graph() → 单专利图（收集 years）
    ↓
merge_patent_graphs() → 合并图（合并 years 列表）
    ↓
compute_time_decay_weights() → 计算权重（基于 years）
    ↓
HDKN/PDKN（包含 strength/weight/year_min）
```

### 2.3 边权重叠加

- 同一对词在不同专利出现时，`years` 列表合并，权重按各年份累加
- 同一专利中同一依存边出现多次，只计一次（unique per patent）

### 2.4 节点 Strength

支持两种模式（`NODE_STRENGTH_MODE` 配置）：
- `"time_decay"`：`strength = Σ α^(T - year) for year in node.years`
- `"weighted_degree"`：`strength = Σ(incident edge weights)`

### 2.5 years 存储

`years` 使用 `list` 存储（而非 set），保留同一年多次出现的计数，确保边权累加正确。

---

## 3. 特征计算说明

### 3.1 New_n / New_e

- **阈值**：HDKN 中所有节点/边 Year_n/Year_e 的 90% 分位数（`p90_year_n`、`p90_year_e`）
- **判断**：子网中存在 `year_min > p90` 的节点/边 → 值为 1
- **不在 HDKN 中的节点**：视为首现于 `current_year`，若 `current_year > p90` 则视为新
- **配置**：`NOVELTY_THRESHOLD_MODE = "quantile_90"` 或 `"hist_end_year"`

### 3.2 Min_pn

- **定义**：覆盖子网所有**节点和边**的最小专利数（set cover problem）
- **算法**：贪心 set cover（`feature_registry._compute_min_pn`）
- **语义**：Min_pn 越大，Si 整合的来自不同专利的新知识越多，衡量新颖性

### 3.3 Con_n / Con_e

- **Con_n**：子网 Si 中**所有**节点 Strength 的中位数（`np.median()`）
- **Con_e**：子网 Si 中**所有**边 Weight 的中位数（`np.median()`）
- **数据来源**：strength/weight 必须来自 HDKN
- **语义**：值越大，Si 包含的主流知识元素越多，衡量常规性

### 3.4 Eigen

- 在 HDKN 上计算所有节点的 eigenvector centrality（加权图）
- 取子网节点的平均值（`np.mean()`）
- 大图使用 power iteration，小图使用 numpy 精确计算

### 3.5 Constraint

- 在 HDKN 上计算所有节点的 Burt's constraint（结构洞约束）
- 取子网节点 constraint 的**最小值**（min）
- 实现：`networkx.algorithms.structuralholes.constraint()`

### 3.6 特征计算数据来源原则

- **所有特征必须基于 HDKN**：避免未来信息泄漏（data leakage）
- **回归训练样本**：使用 `app_year <= HIST_END_YEAR` 的专利
- **子网提取**：从 HDKN 中提取

---

## 4. 回归模型与系数选取

### 4.1 模型类型

固定使用 **NB + ZINB 双模型**：
- 负二项回归（NB）：处理过度离散的计数数据
- 零膨胀负二项回归（ZINB）：额外处理零膨胀
- 通过 Vuong 非嵌套模型检验比较两者

### 4.2 回归工作流（4-Run Process）

```
Run 1: Alpha Selection — 候选 α 逐一拟合 NB/ZINB，选 ZINB LL 最大的 α
Run 2: 全模型 — 用最优 α 拟合全部特征 + 控制变量，输出共线性报告（VIF）
Run 3: 选定变量 + 控制变量 — 审查 Run 2 后选定显著变量，加控制变量拟合
Run 4: 仅选定变量 — 同 Run 3 但不含控制变量，系数用于 ACO 目标函数
Combined: 合并 Run 2/3/4 对比表
```

### 4.3 回归公式

Run 2 全模型：
```
Cited ~ New_n + New_e + Min_pn + Con_n + Con_e + Eigen + Constraint + Back_cite + Assignee + Total_pat
```

Run 3/4 选定变量（示例）：
```
Cited ~ New_e + Eigen + Constraint [+ Back_cite + Assignee + Total_pat]
```

### 4.4 ACO 目标系数

Run 4 完成后生成 `objective_coefficients.json`，包含 NB 模型中 p<0.05 的显著子网特征系数（原始尺度），供 Step3 ACO 搜索使用。

目标函数 Z = β₁ × New_e + β₂ × Eigen + β₃ × Constraint，ACO 最大化此值以发现高影响力技术机会子网。

---

## 5. Alpha 选取流程

### 5.1 流程概述

对候选 α 列表（默认 `[0.50, 0.60, 0.70, 0.80, 0.90, 1.00]`），逐一：
1. 加载 HDKN
2. 以该 α 重算时间衰减权重
3. 提取特征 + 拟合 NB 与 ZINB
4. 记录 ZINB 的 Log-Likelihood

选出 LL 最大的 α 作为最优衰减因子。

### 5.2 关键设计

- 所有 α 统一走相同计算路径，避免 α=1 复用 Step2 导致逻辑不一致
- `force_rebuild_hdkn_stats=True`：每个 α 强制重建 HDKN 统计缓存
- Constraint 缓存键包含边权采样，不同 α 对应不同缓存

### 5.3 与 Step2 的关系

- Step2 可自动复用 Alpha Selection 的最优 α 和模型结果
- `regression_meta.json` 中的 `decay_factor` 记录实际使用的 α
- 特征集完全匹配时可复用模型，否则仅复用 α 值重算权重

---

## 6. 数据清洗（Step0）

### 6.1 数据来源

Derwent Innovation 导出的原始 CSV（44 列中文列头），涵盖具身智能领域的国际专利。

### 6.2 清洗流程

1. **字段映射**：中文列头 → 标准字段（patent_id, title, abstract, app_year 等），优先取 DWPI 英文版
2. **年份过滤**：剔除 `app_year < 2012` 的专利
3. **语言核验**：
   - 剥离 WIPO 专利中附加的非英文翻译段落（按 `|` 分段）
   - 标题和摘要分别检测非拉丁字符占比（阈值 5%）
   - 重音拉丁字符检测（德语、法语等）
   - OCR 乱码过滤（FIG 标签等）
   - Lemma 级非英语功能词检测
4. **去重**：同一公开号保留最新版本
5. **输出**：`patents.csv`（通过）+ `patents_excluded.csv`（排除）

### 6.3 实现

脚本：`scripts/step0_import_patents.py`，详见 `docs/field_mapping.md`。

---

## 7. ACO 新颖度引导机制

### 7.1 设计目标

在 PDKN 上搜索时引导蚂蚁探索更多新边（PDKN 有而 HDKN 无的边），提高发现新颖技术机会的概率。

### 7.2 三项机制

1. **节点新颖度比率 (node novelty ratio)**：
   - 每个节点计算其关联边中新边的比例
   - 启发式信息 η = strength × (1 + γ × novelty_ratio)，γ 为 novelty weight（默认 2.0）

2. **新颖蚂蚁群体 (novel ant ratio)**：
   - 30% 的蚂蚁从高新颖度节点出发（按 novelty_ratio 排序）
   - 这些蚂蚁经过新边时获得权重乘数 bonus

3. **新边 bonus (new edge bonus)**：
   - 新颖蚂蚁走新边时，选择概率乘以 bonus 系数（默认 6.0）

### 7.3 配置

```yaml
# configs/aco_config.yaml
novelty:
  weight: 2.0
  novel_ant_ratio: 0.3
  new_edge_bonus: 6.0
```

---

## 8. 子网富化（ACO → RAG）

### 8.1 富化脚本

模块：`src/patent_opportunity_analysis/aco_to_rag.py`
调用入口：`scripts/merge_aco_candidates.py`

### 8.2 节点分类规则（对齐原文 Table 3）

| 类型 | 判定条件 | 对应特征变量 |
|------|---------|-------------|
| **new** | 不在 HDKN 中，或 Year_n ≥ p90_year_n | New_n |
| **marginal** | 在 HDKN 中但 eigenvector centrality < 1e-4 | Eigen（负系数） |
| **conventional** | 其余 | Con_n |

### 8.3 边分类规则

| 类型 | 判定条件 | 对应特征变量 |
|------|---------|-------------|
| **new** | 不在 HDKN 中，或 Year_e ≥ p90_year_e | New_e |
| **special** | PDKN 中仅出现在 1 篇专利中 | Min_pn（整合多来源知识） |
| **conventional** | 其余 | Con_e |

### 8.4 is_marginal 判定

除了 `_classify_node` 的硬阈值判定（`eigen < 1e-4`）外，`enrich_opportunities` 还计算 PDKN 正 Eigen 值的**第 70 百分位**作为动态阈值：

```python
all_eigen_vals = [v for v in eigen_dict.values() if v > 0]
eigen_threshold = np.percentile(all_eigen_vals, 70)
node["is_marginal"] = (ntype == "marginal") or (ntype == "conventional" and eigen < eigen_threshold)
```

这确保了即使 `_classify_node` 的硬阈值漏判，`is_marginal` 字段仍能标记处于网络边缘的 conventional 节点。

### 8.5 富化产物字段

每个子网的 JSON 结构：

```json
{
  "opportunity_rank": 1,
  "z_score": 0.6343,
  "feature_scores": {"New_e": 1.0, "Eigen": 0.0029, "Constraint": 0.0093},
  "nodes": [
    {
      "stem": "beam-form",
      "original_forms": ["beam-form"],
      "type": "new",
      "strength": 1.39,
      "year_first": 2022,
      "eigen": 0.0,
      "is_marginal": false,
      "representative_patent": {"id": "CN115865157A", "title": "...", "abstract": "..."}
    }
  ],
  "edges": [
    {
      "node_pair": ["correspond", "improv"],
      "type": "special",
      "year_e": 2020,
      "sole_patent": {"id": "CN113299382A", "title": "...", "abstract": "..."}
    }
  ],
  "novelty_sources": ["commun-predict", "predict-beam", "beam-form"],
  "feasibility_anchors": ["optim", "data", "model"]
}
```

---

## 9. RAG 报告生成方法论

### 9.1 对齐原文解读流程

原文的技术机会解读流程（Section 4.3）：
1. 根据目标函数 Z 中各变量的系数符号，将子网的节点/边分为六类
2. 从"边缘节点 × 特殊边"和"新兴节点 × 特殊边"中提取**新颖组合（Novel Combinations）**
3. 查找新颖组合涉及的专利，推断其具体技术含义
4. 将节点/边归类为技术手段和应用功能，综合形成可理解的机会描述

### 9.2 Prompt 中的节点/边分类标记

| 标记 | 类型 | 含义 |
|------|------|------|
| ★ | 新兴节点 | HDKN 中不存在的技术词（New_n 贡献者） |
| ◇(Eigen=...) | 边缘节点 | 特征向量中心性低，处于网络边缘 |
| △ | 特殊边 | 仅 1 篇专利支撑的稀有关联（Min_pn 贡献者） |
| □ | 常规边 | 历史连接强度高的成熟关联（Con_e 贡献者） |

### 9.3 边选取策略（分层配额）

`_sort_edges()` 使用分层配额而非简单排序，确保每种边类型都有代表：

1. **special 边全选**（通常 0-2 条，它们是 novel combination 的核心线索）
2. **剩余配额** 在 new 和 conventional 之间大致均分
3. 某类型数量不足时，配额让给另一类型
4. new 边按 year_e 降序，conventional 边按 con_e 降序

### 9.4 专利证据收集优先级

按解读重要性排序，确保关键证据不被 token 截断：

| 优先级 | 来源 | 理由 |
|--------|------|------|
| P1 | special 边的 sole_patent | 消歧关键——唯一支撑专利定义了该稀有关联的具体技术含义 |
| P2 | new 边的代表专利 | 新兴技术关联的核心证据 |
| P3 | new 节点的代表专利 | 新兴技术术语的定义来源 |
| P4 | marginal 节点的代表专利 | 网络边缘节点的技术背景 |
| P5 | conventional 边的 top_patents | 成熟技术基础的参考 |

### 9.5 报告四段式结构

| 段落 | 对齐原文步骤 | 分析要求 |
|------|-------------|---------|
| 1. 技术机会命名 | 综合概括 | ≤15 字，[技术手段]+[应用对象]+[目标效果] |
| 2. 重要节点与边识别 | 步骤 1: 分类 | 按 ★/◇/△/□ 逐类列出并解读角色 |
| 3. 新颖组合分析 | 步骤 2-3: 提取+推断 | 2-3 个 novel combinations，优先分析 △ 边 |
| 4. 技术机会综合解读 | 步骤 4: 综合 | "技术手段"+"应用功能"二维结构，每句带引证 |

---

*合并自：original_definition_checklist.md、weight_audit.md、decay_weight_audit.md、feature_audit.md、new_n_new_e_audit.md、regression_variable_audit.md、step2_alpha_selection_flow_audit.md*
*最后更新：2026-04-03*
