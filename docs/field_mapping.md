# 数据字段映射与核对报告

## 概述

本文档详细记录了从Derwent Innovation导出数据到标准化专利记录的字段映射过程，基于Ren & Zhao (2021)的研究方法。

## 原始数据结构

Derwent导出的CSV文件包含41列中文字段名（跳过前3行标题）：

| 列索引 | 中文字段名 | 英文对应 | 说明 |
|--------|-----------|----------|------|
| 0 | 公开号 | Publication Number | 专利公开号 |
| 1 | 标题 | Title | 专利标题 |
| 2 | 标题 - DWPI | Title - DWPI | DWPI规范化的标题 |
| 3 | 优先权号 | Priority Number | 优先权信息 |
| 4 | 优先权号/优先权日 - DWPI | Priority Info - DWPI | 优先权详细信息 |
| 5 | 优先权日 | Priority Date | 优先权日期 |
| 6 | 优先权日 - DWPI | Priority Date - DWPI | DWPI优先权日期 |
| 7 | 申请号 | Application Number | 专利申请号 |
| 8 | 申请日期 | Application Date | 申请日期 |
| 9 | 公开专利文献类型识别代码 | Publication Type | 公开类型代码 |
| 10 | 公开日期 | Publication Date | 公开日期 |
| 11 | 发明人 - 带有地址 | Inventors | 发明人及其地址 |
| 12 | 专利权人/申请人 | Assignees/Applicants | 专利权人/申请人 |
| 13 | 当前专利权人 - 美国 | Current US Assignee | 当前美国专利权人 |
| 14 | DWPI 分类 | DWPI Classification | DWPI分类 |
| 15 | DWPI 手工代码 | DWPI Manual Codes | DWPI手工代码 |
| 16 | IPC - 现版 | IPC - Current | 当前IPC分类 |
| 17 | IPC - 现版 - DWPI | IPC - Current - DWPI | DWPI IPC分类 |
| 18 | CPC - 现版 | CPC - Current | 当前CPC分类 |
| 19 | CPC - 现版 - DWPI | CPC - Current - DWPI | DWPI CPC分类 |
| 20 | 美国分类 | US Classification | 美国专利分类 |
| 21 | ECLA | ECLA | 欧洲分类 |
| 22 | 摘要 - DWPI | Abstract - DWPI | DWPI规范化的摘要 |
| 23 | 摘要 | Abstract | 原始摘要 |
| 24 | 申请月 | Application Month | 申请月份 |
| 25 | 公开月 | Publication Month | 公开月份 |
| 26 | 申请年 | Application Year | 申请年份 |
| 27 | 公开年 | Publication Year | 公开年份 |
| 28 | 公开国家/地区代码 | Publication Country | 公开国家代码 |
| 29 | 第一发明人 | First Inventor | 第一发明人 |
| 30 | 发明人计数 | Inventor Count | 发明人数量 |
| 31 | 专利权人/申请人 (首位) | First Assignee/Applicant | 首位专利权人/申请人 |
| 32 | 专利权人计数 | Assignee Count | 专利权人数量 |
| 33 | 引用的参考文献 - 专利 | Cited Patents | 引用的专利文献 |
| 34 | 引用的专利第一专利权人 | First Cited Patent Assignee | 引用的第一专利权人 |
| 35 | 引用的参考文献数 - 专利 | Cited Patent Count | 引用的专利文献数量 |
| 36 | 引用的参考文献 - 非专利 | Cited Non-Patents | 引用的非专利文献 |
| 37 | 引用的非专利 - DOI | Cited Non-Patent DOI | 非专利文献DOI |
| 38 | 引用的参考文献计数 - 非专利 | Cited Non-Patent Count | 引用的非专利文献数量 |
| 39 | 施引专利计数 | Citing Patent Count | 被引用次数（前向引用） |
| 40 | 引用的专利详细信息 - DPCI | Cited Patent Details - DPCI | 引用的专利详细信息 |

## 当前实现

### 数据导入脚本（`scripts/step0_import_patents.py`）

上述字段映射已在 `step0_import_patents.py` 中完整实现。该脚本取代了早期的 `extract_patents.py`，修正了所有已知的列索引映射错误。

### 字段映射表

| 标准字段 | 映射的 Derwent 字段 | 列索引 | 优先级 | 说明 |
|---------|------------------|--------|--------|------|
| `patent_id` | 公开号 | 0 | **必须** | 唯一标识符 |
| `title` | 标题 - DWPI → 标题 | 2 → 1 | **优先 → 回退** | DWPI 规范化标题优先 |
| `abstract` | 摘要 - DWPI → 摘要 | 22 → 23 | **优先 → 回退** | DWPI 规范化摘要优先 |
| `app_year` | 申请年 / 公开年 | 26 / 27 | **必须** | 申请年优先，公开年回退 |
| `forward_cites` | 施引专利计数 | 39 | **必须** | 被引用次数 |
| `backward_cites` | 引用的参考文献数 - 专利 | 35 | **必须** | 引用的专利数 |
| `ipc_classes` | IPC - 现版 → IPC - 现版 - DWPI | 16 → 17 | **优先 → 回退** | IPC 分类号 |
| `assignee_type` | 专利权人/申请人 (首位) | 31 | **必须** | 组织=1 / 个人=0 |

### 数据清洗流程

`step0_import_patents.py` 执行以下清洗步骤：

1. **解析原始 CSV**：跳过前 3 行标题，按列索引映射到标准字段
2. **年份过滤**：剔除 `app_year < min_year`（默认 2012）的专利
3. **语言核验**（标题和摘要分别检测）：
   - 剥离 WIPO 专利中附加的非英文翻译段落（按 `|` 分段，保留拉丁字母主导段落）
   - 检测非拉丁字符（CJK / Cyrillic / Arabic 等）占所有字母字符的比例
   - 比例 > 5% 的标题或摘要判定为非英语，排除该记录
4. **去重**：同一公开号保留最新版本
5. **输出**：`patents.csv`（通过核验）+ `patents_excluded.csv`（排除记录及原因）

### 字段选择理由

1. **patent_id 使用公开号**：全球唯一标识符，一个申请可对应多次公开
2. **优先使用 DWPI 版本**：DWPI 提供规范化的英文标题和摘要，更适合 NLP 处理
3. **引用计数**：forward_cites = 施引专利计数，backward_cites = 引用的参考文献数 - 专利

---

*最后更新：2026-03-31* | *实现脚本：`scripts/step0_import_patents.py`*