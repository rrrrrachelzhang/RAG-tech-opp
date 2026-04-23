#!/usr/bin/env python3
"""
Step 0: 从 Derwent Innovation 原始导出 CSV 导入专利数据到 patents.csv

功能：
1. 解析原始 CSV（含中文列头），映射到标准字段
2. 剔除 app_year < min_year 的专利
3. 语言核验：确保 title 和 abstract 为英语（检测 CJK/Cyrillic/Arabic 字符占比）
4. 去重（同一公开号保留最新版本）
5. 输出 patents.csv（通过核验）和 patents_excluded.csv（未通过核验）
"""

import argparse
import csv
import logging
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)

# ── 原始 CSV 列索引 ──
COL_PATENT_ID = 0        # 公开号
COL_TITLE_ORIG = 1       # 标题（原始语言）
COL_ABSTRACT_ORIG = 4    # 摘要（原始语言）
COL_APP_YEAR = 26        # 申请年
COL_ASSIGNEE = 14        # 专利权人/申请人
COL_FWD_CITES = 35       # 施引专利计数
COL_BWD_CITES = 36       # 引用的参考文献数 - 专利
COL_IPC = 18             # IPC - 现版
COL_TITLE_ENG = 42       # 标题 (英语)
COL_ABSTRACT_ENG = 43    # 摘要 (英语)

HEADER_ROW_INDEX = 2     # 0-indexed: 第三行是列头
OUTPUT_COLUMNS = [
    "patent_id", "title", "abstract", "app_year",
    "forward_cites", "backward_cites", "ipc_classes", "assignee",
]

# ── 语言检测用的 Unicode 范围正则 ──
# CJK 统一汉字 + 日文假名 + 韩文音节 + 西里尔字母 + 阿拉伯字母
_NON_LATIN_RE = re.compile(
    r'[\u4e00-\u9fff'       # CJK Unified Ideographs
    r'\u3040-\u309f'        # Hiragana
    r'\u30a0-\u30ff'        # Katakana
    r'\uac00-\ud7af'        # Hangul Syllables
    r'\u0400-\u04ff'        # Cyrillic
    r'\u0600-\u06ff'        # Arabic
    r'\u0980-\u09ff'        # Bengali
    r'\u0900-\u097f'        # Devanagari
    r'\u0e00-\u0e7f'        # Thai
    r']'
)

# 重音拉丁字符正则（德语 ä/ö/ü/ß，法语 é/è/ê，西班牙语 ñ/á 等）
# 不包含希腊字母（π/θ/σ 等数学符号在英文专利中合法出现）
_ACCENTED_LATIN_RE = re.compile(
    r'[\u00c0-\u00d6'       # À-Ö
    r'\u00d8-\u00f6'        # Ø-ö
    r'\u00f8-\u00ff'        # ø-ÿ
    r'\u0100-\u017f'        # Latin Extended-A
    r'\u0180-\u024f'        # Latin Extended-B
    r'\u1e00-\u1eff'        # Latin Extended Additional (ḿ, ṁ 等)
    r']'
)

# OCR 图标签模式：页码 "1/21" 后跟图编号/标签文本
_OCR_PAGE_NUM_RE = re.compile(r'\s+\d{1,3}\s*/\s*\d{1,3}\s+[A-Z]')


def is_english_text(
    text: str,
    max_non_latin_ratio: float = 0.05,
    max_accented_ratio: float = 0.008,
    min_length: int = 10,
) -> Tuple[bool, str]:
    """
    判断文本是否为英语。

    策略：
    1. 非拉丁字符（CJK/Cyrillic/Arabic 等）占比 > max_non_latin_ratio → 非英语
    2. 重音拉丁字符（ä/ö/ü/ñ/é 等）占比 > max_accented_ratio → 非英语
       （可检测德语/法语/西班牙语等拉丁字母语言；不影响希腊字母数学符号）
    3. 文本过短时放行

    Returns:
        (is_english, reason): True 表示文本为英语，reason 为判定理由
    """
    if not text:
        return True, ""

    alpha_chars = [c for c in text if unicodedata.category(c).startswith('L')]
    if len(alpha_chars) < min_length:
        return True, ""

    n_alpha = len(alpha_chars)

    non_latin_count = len(_NON_LATIN_RE.findall(text))
    if non_latin_count / n_alpha > max_non_latin_ratio:
        return False, f"non_latin={non_latin_count}/{n_alpha} ({non_latin_count/n_alpha:.3f})"

    accented_count = len(_ACCENTED_LATIN_RE.findall(text))
    if accented_count >= 2 and accented_count / n_alpha > max_accented_ratio:
        return False, f"accented_latin={accented_count}/{n_alpha} ({accented_count/n_alpha:.3f})"

    return True, ""


def strip_ocr_figure_text(text: str) -> str:
    """
    剥离摘要末尾附带的 OCR 图标签文本。

    部分专利的摘要被 OCR 时会把图表中的标签文字也拼入末尾，
    典型模式：正文 + "1/21 TA1 | A1 LA1 Layer 1..." + "FIG. 1A"
    这些内容生成大量无意义的乱码 token，需要剥离。
    """
    if not text:
        return text

    match = _OCR_PAGE_NUM_RE.search(text)
    if not match:
        return text

    before = text[:match.start()].rstrip()
    if not before:
        return text

    if before[-1] in '.;:)':
        return before

    last_period = before.rfind('.')
    if last_period > len(before) * 0.5:
        return before[:last_period + 1]

    return text


def strip_non_english_segments(text: str) -> str:
    """
    剥离文本中的非英文翻译段落。

    WIPO 专利摘要常有格式: "English abstract | French title | 日本語タイトル English abstract..."
    策略：按 ' | ' 分段，保留英文段落（同时检测 CJK/Cyrillic 和重音拉丁字符）。
    """
    if not text:
        return text

    segments = re.split(r'\s*\|\s*', text)
    if len(segments) <= 1:
        return text

    english_segments = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        alpha_chars = [c for c in seg if unicodedata.category(c).startswith('L')]
        if not alpha_chars:
            english_segments.append(seg)
            continue
        n = len(alpha_chars)
        non_latin = len(_NON_LATIN_RE.findall(seg))
        if non_latin / n > 0.3:
            continue
        accented = len(_ACCENTED_LATIN_RE.findall(seg))
        if accented >= 2 and accented / n > 0.01:
            continue
        english_segments.append(seg)

    return ' | '.join(english_segments) if english_segments else text


def safe_int(val: str, default: int = 0) -> int:
    """安全地将字符串转为整数"""
    val = val.strip()
    if not val or val in ('-', 'nan', 'None'):
        return default
    try:
        return int(val)
    except ValueError:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default


def parse_raw_csv(raw_path: Path) -> List[Dict]:
    """
    解析 Derwent Innovation 导出的原始 CSV。

    Returns:
        解析后的记录列表，每条记录为 dict（键为 OUTPUT_COLUMNS）
    """
    records = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i in range(HEADER_ROW_INDEX):
            next(reader)
        header = next(reader)
        logger.info(f"原始 CSV 列数: {len(header)}")

        for row_idx, row in enumerate(reader):
            if len(row) < COL_ABSTRACT_ENG + 1:
                logger.warning(f"行 {row_idx + HEADER_ROW_INDEX + 2} 列数不足 ({len(row)})，跳过")
                continue

            patent_id = row[COL_PATENT_ID].strip()
            if not patent_id:
                continue

            title_eng = row[COL_TITLE_ENG].strip()
            title_orig = row[COL_TITLE_ORIG].strip()
            abstract_eng = row[COL_ABSTRACT_ENG].strip()
            abstract_orig = row[COL_ABSTRACT_ORIG].strip()

            title = title_eng if title_eng else title_orig
            abstract = abstract_eng if abstract_eng else abstract_orig

            # 剥离附加的非英文翻译段落（常见于 WIPO 专利）
            title = strip_non_english_segments(title)
            abstract = strip_non_english_segments(abstract)

            # 剥离 OCR 图标签文本（附在摘要末尾的 FIG 标签乱码）
            abstract = strip_ocr_figure_text(abstract)

            app_year = safe_int(row[COL_APP_YEAR])
            forward_cites = safe_int(row[COL_FWD_CITES])
            backward_cites = safe_int(row[COL_BWD_CITES])
            ipc_raw = row[COL_IPC].strip()
            assignee = row[COL_ASSIGNEE].strip()

            records.append({
                "patent_id": patent_id,
                "title": title,
                "abstract": abstract,
                "app_year": app_year,
                "forward_cites": forward_cites,
                "backward_cites": backward_cites,
                "ipc_classes": ipc_raw,
                "assignee": assignee,
            })

    logger.info(f"原始 CSV 解析完成: {len(records)} 条记录")
    return records


def deduplicate(records: List[Dict]) -> List[Dict]:
    """
    去重：同一 patent_id 保留最后出现的那条（通常是最新版本）。
    """
    seen: Dict[str, int] = {}
    for i, rec in enumerate(records):
        seen[rec["patent_id"]] = i

    deduped = [records[i] for i in sorted(seen.values())]
    removed = len(records) - len(deduped)
    if removed > 0:
        logger.info(f"去重: 移除 {removed} 条重复 patent_id")
    return deduped


def filter_and_validate(
    records: List[Dict],
    min_year: int,
    max_non_latin_ratio: float,
) -> Tuple[List[Dict], List[Dict]]:
    """
    过滤与语言核验。

    Returns:
        (通过记录列表, 被排除记录列表)
    """
    passed = []
    excluded = []
    reasons: Counter = Counter()

    for rec in records:
        pid = rec["patent_id"]

        # 1) 年份过滤
        if rec["app_year"] < min_year:
            rec["_exclude_reason"] = f"app_year={rec['app_year']} < {min_year}"
            excluded.append(rec)
            reasons["year_too_old"] += 1
            continue

        # 2) 标题 + 摘要不能同时为空
        if not rec["title"] and not rec["abstract"]:
            rec["_exclude_reason"] = "title and abstract both empty"
            excluded.append(rec)
            reasons["empty_text"] += 1
            continue

        # 3) 语言核验：标题和摘要分别检测
        title_text = rec["title"] or ""
        abstract_text = rec["abstract"] or ""
        title_ok, title_reason = is_english_text(
            title_text, max_non_latin_ratio=max_non_latin_ratio,
        )
        if not title_ok:
            rec["_exclude_reason"] = f"non-English title ({title_reason})"
            excluded.append(rec)
            reasons["non_english_title"] += 1
            continue
        abstract_ok, abstract_reason = is_english_text(
            abstract_text, max_non_latin_ratio=max_non_latin_ratio,
        )
        if not abstract_ok:
            rec["_exclude_reason"] = f"non-English abstract ({abstract_reason})"
            excluded.append(rec)
            reasons["non_english_abstract"] += 1
            continue

        passed.append(rec)

    logger.info(f"过滤结果: 通过={len(passed)}, 排除={len(excluded)}")
    for reason, count in reasons.most_common():
        logger.info(f"  排除原因 [{reason}]: {count}")

    return passed, excluded


def write_csv(records: List[Dict], output_path: Path, columns: List[str]) -> None:
    """写入 CSV 文件"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"已写入 {len(records)} 条到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 0: 导入并清洗专利数据")
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=Path("data/raw/csv2026-03-18-15-53-59.csv"),
        help="Derwent Innovation 导出的原始 CSV 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/patents.csv"),
        help="输出的 patents.csv 路径",
    )
    parser.add_argument(
        "--excluded",
        type=Path,
        default=Path("data/raw/patents_excluded.csv"),
        help="被排除记录的输出路径",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2012,
        help="最小申请年份（< min_year 的记录被剔除，默认 2012）",
    )
    parser.add_argument(
        "--max-non-latin-ratio",
        type=float,
        default=0.05,
        help="非拉丁字符占比上限（超过则判定为非英语，默认 0.05）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行：只统计不写文件",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step 0: 专利数据导入与清洗")
    logger.info("=" * 60)
    logger.info(f"原始 CSV: {args.raw_csv}")
    logger.info(f"最小年份: {args.min_year}")
    logger.info(f"非拉丁字符占比上限: {args.max_non_latin_ratio}")

    # 1) 解析原始 CSV
    records = parse_raw_csv(args.raw_csv)

    # 2) 去重
    records = deduplicate(records)

    # 3) 过滤 + 语言核验
    passed, excluded = filter_and_validate(
        records,
        min_year=args.min_year,
        max_non_latin_ratio=args.max_non_latin_ratio,
    )

    # 4) 统计
    if passed:
        years = Counter(r["app_year"] for r in passed)
        logger.info(f"\n通过记录年份分布（前 5 年）:")
        for y in sorted(years)[:5]:
            logger.info(f"  {y}: {years[y]}")
        logger.info(f"  ...")
        for y in sorted(years)[-5:]:
            logger.info(f"  {y}: {years[y]}")

    if excluded:
        logger.info(f"\n被排除样本（前 10 条）:")
        for rec in excluded[:10]:
            reason = rec.get("_exclude_reason", "unknown")
            logger.info(f"  {rec['patent_id']} (y={rec['app_year']}): {reason} | title={rec['title'][:50]}")

    if args.dry_run:
        logger.info("\n[试运行] 不写文件，仅展示统计。")
        return

    # 5) 写入
    write_csv(passed, args.output, OUTPUT_COLUMNS)
    write_csv(excluded, args.excluded, OUTPUT_COLUMNS)

    logger.info(f"\n✅ 完成！通过 {len(passed)} 条，排除 {len(excluded)} 条")


if __name__ == "__main__":
    main()
