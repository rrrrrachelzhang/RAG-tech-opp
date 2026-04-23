# Data Directory

## 目录结构

```
data/
├── raw/                                    # 原始数据与清洗产物
│   ├── csv2026-03-18-15-53-59.csv         # Derwent Innovation 原始导出（44 列，中文列头，~8800 条）
│   ├── patents.csv                        # 清洗后标准化数据（8620 条，由 step0 生成）
│   ├── patents_excluded.csv               # 被排除记录（含排除原因）
│   └── patents_test.csv                   # 测试用小数据集
└── processed/
    └── rag/                               # Step3.5 产物（供 Step4 RAG 报告生成使用）
        ├── aco_merged_top30_enriched.json  # 富化子网 JSON（含节点/边分类、专利信息、评分）
        ├── aco_merged_top30_candidates.json # 精简候选列表（仅含 nodes/score/size）
        └── merge_summary.json             # 合并过程摘要（候选数量、过滤参数等）
```

## raw/ — 原始数据

### 数据清洗

使用 `step0_import_patents.py` 从原始 CSV 生成标准化数据：

```bash
python scripts/step0_import_patents.py
```

清洗步骤：
1. 解析 Derwent CSV 中文列头，映射到标准字段（patent_id, title, abstract, app_year 等）
2. 优先取 DWPI 版英文标题/摘要，缺失时回退到原始字段
3. 剔除 `app_year < 2012` 的专利
4. 多层语言核验（非拉丁字符占比检测、OCR 乱码过滤、重音字符检测等）
5. 同一公开号去重（保留最新版本）

### patents.csv 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| patent_id | str | 专利公开号（唯一标识） |
| title | str | 英文标题 |
| abstract | str | 英文摘要 |
| app_year | int | 申请年份 |
| forward_cites | int | 被引用次数（因变量来源） |
| backward_cites | int | 引用专利数（控制变量来源） |
| ipc_classes | str | IPC 分类号（分号分隔） |
| assignee_type | int | 专利权人类型（1=组织，0=个人） |

## processed/rag/ — 富化子网数据

由 `scripts/merge_aco_candidates.py` 生成，流程：
1. 合并所有 ACO 运行的候选子网（去重 + 80% 节点重叠多样性过滤）
2. 调用 `aco_to_rag.enrich_opportunities()` 对 Top-30 子网进行富化
3. 每个子网的节点按 new/marginal/conventional 分类，边按 new/special/conventional 分类
4. 匹配代表性专利信息（标题、摘要、引用数）
5. 注入 eigen 值、is_marginal 标记、novelty_sources 和 feasibility_anchors

### 生成命令

```bash
python scripts/merge_aco_candidates.py --run-id <run_id> --top-n 30 --overlap 0.8
```

## 编码

CSV 文件统一使用 UTF-8 编码。

---

*最后更新：2026-04-03*
