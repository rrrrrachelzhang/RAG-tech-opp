# NLP 分词与停用词说明

---

## 1. 分词与规范化

### 1.1 连字符处理

**单个 token 内含连字符**（如 `machine-learning`）：
- `normalize_token()` 按 `-` 分割，对每段分别做 lemmatization，再用 `-` 拼回
- `is_valid_token()` 中含 `-` 的 token 可能被过滤（`isalnum()` 为 False）

**空格分隔的连字符序列**（如 `machine - learning`）：
- `_merge_hyphenated_tokens()` 识别并合并为 `machine-learning`

**compound 关系**（如 `machine learning`）：
- `_merge_compound_tokens()` 识别 spaCy 的 compound 依存关系
- 合并为 `machine-learning` 格式，与连字符节点统一

### 1.2 数字处理

| 类型 | 示例 | 处理结果 |
|------|------|----------|
| 纯数字 | `2024`, `3` | 过滤 |
| 字母数字混合 | `sim2real`, `B2B` | 保留 |
| 含数字的连字符词 | `GPT-3` | 单 token 时通常被过滤 |

### 1.3 缩写处理

- 纯字母缩写（`CNN`、`VLA`）：保留（除非是停用词）
- 字母数字缩写（`B2B`、`C2C`）：保留

### 1.4 下游节点验证

`patent_graph.is_valid_node()` 补充规则：
- 允许节点名中包含 `-` 和 `_`
- `cleaned = node.replace('-', '').replace('_', '')` 后需为 `isalnum()`

---

## 2. 停用词

基于专利文本统计分析的停用词配置。

### 2.1 当前采用方案（方案 C）

```
wherein, thereof, say, mean, unit, step, end, second
```

### 2.2 各类别说明

**专利法律套话（强建议）**：
- `wherein`（962次）、`thereof`（491次）、`say`/`said`（251次）、`mean`/`means`（288次）

**泛化功能词（中建议）**：
- `unit`（2006次，如 "processing unit"）
- `step`（1357次，如 "method step"）
- `end`（2300次，多为泛化用法）

**序数词（弱建议）**：
- `second`（1445次，如 "second embodiment"）

### 2.3 不作为停用词的技术词

| 词汇 | 说明 |
|------|------|
| determine | ACO 中高频，有技术含义 |
| generate | ACO 中高频，有技术含义 |
| control | 核心技术词（机器人控制） |
| robot, arm | 领域核心词 |

### 2.4 Lemma 自动覆盖

以下词形因已有停用词的 lemma 被自动过滤：
- `according` → lemma `accord`
- `comprising`/`comprises` → lemma `comprise`
- `obtaining` → lemma `obtain`
- `based` → lemma `base`

### 2.5 注意事项

- 加入 `say` 可过滤 `said`，加入 `mean` 可过滤 `means`
- 修改停用词后需清空 NLP 缓存并重新运行 Step1→Step2→Step3

---

*合并自：nlp_token_handling.md、stopwords_recommendation.md*
