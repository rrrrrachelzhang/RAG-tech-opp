"""
RAG 专利数据获取模块：从 patents.csv 中按专利号批量查询专利文本（标题 + 摘要）。
"""
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .utils.paths import RAW_PATENT_FILE

logger = logging.getLogger(__name__)

_patent_cache: Optional[Dict[str, str]] = None
_cached_path: Optional[Path] = None


def _load_patent_index(csv_path: Path) -> Dict[str, str]:
    """
    从 CSV 加载专利索引，key 为 patent_id，value 为拼接后的文本（标题 + 摘要）。
    """
    global _patent_cache, _cached_path
    if (
        _patent_cache is not None
        and _cached_path is not None
        and _cached_path.resolve() == csv_path.resolve()
    ):
        return _patent_cache

    if not csv_path.exists():
        logger.warning("专利 CSV 文件不存在: %s", csv_path)
        _patent_cache = {}
        return _patent_cache

    index: Dict[str, str] = {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("patent_id", "").strip()
                title = row.get("title", "").strip()
                abstract = row.get("abstract", "").strip()
                text = f"{title}\n{abstract}" if abstract else title
                if pid:
                    index[pid] = text
        logger.info("已加载 %d 条专利记录", len(index))
    except (csv.Error, OSError) as e:
        logger.exception("读取专利 CSV 失败: %s", e)
        raise

    _patent_cache = index
    _cached_path = csv_path.resolve()
    return index


def fetch_patent_texts(
    patent_ids: List[str],
    csv_path: Optional[Path] = None,
) -> Dict[str, str]:
    """
    从本地专利 CSV 中批量查询专利文本。

    Args:
        patent_ids: 专利号列表，例如 ['US12485556B1', 'EP4008501B1']
        csv_path: 可选，指定 CSV 路径；默认使用 data/raw/patents.csv

    Returns:
        字典 {patent_id: 文本}，文本为「标题 + 摘要」。
        若某专利不存在，则值为 "暂无摘要"。
    """
    path = csv_path or RAW_PATENT_FILE
    index = _load_patent_index(path)

    result: Dict[str, str] = {}
    for pid in patent_ids:
        pid = pid.strip()
        if not pid:
            continue
        result[pid] = index.get(pid, "暂无摘要")
        if result[pid] == "暂无摘要":
            logger.debug("专利 %s 未在 CSV 中找到", pid)

    return result


def clear_cache() -> None:
    """清空专利缓存，用于测试或切换数据源时强制重新加载。"""
    global _patent_cache, _cached_path
    _patent_cache = None
    _cached_path = None
