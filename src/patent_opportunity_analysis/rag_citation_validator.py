"""
RAG 报告引用验证模块：提取报告中的专利号引用，与证据库交叉验证，
识别幻觉引用（LLM 编造或超出证据库范围的专利号）。

验证逻辑：
1. 从报告文本中用正则提取所有形如 [CNxxxxxx]、[USxxxxxxx]、[WOxxxxxxx] 的引用
2. 与注入 prompt 的 core_patent_ids 列表做交叉验证
3. 分类为：valid（在证据库中）/ hallucinated（不在证据库中）/ malformed（格式异常）
"""
import re
import logging
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

# 专利号正则：匹配常见专利局前缀 + 字母数字组合
# 支持 CN/US/WO/EP/JP/KR 等主流专利局
_PATENT_ID_PATTERN = re.compile(
    r'\[('
    r'(?:CN|US|WO|EP|JP|KR|DE|FR|GB|CA|AU|IN|RU|BR)'  # 专利局前缀
    r'\d{6,12}'                                          # 数字部分（6-12位）
    r'[A-Z]?\d?'                                         # 可选的类型后缀，如 A1, B1, A
    r')\]'
)


def extract_citations(report_text: str) -> List[str]:
    """
    从报告文本中提取所有专利号引用（去重保序）。

    Args:
        report_text: 报告 Markdown 文本

    Returns:
        专利号列表，保持首次出现顺序
    """
    matches = _PATENT_ID_PATTERN.findall(report_text)
    seen: Set[str] = set()
    ordered: List[str] = []
    for pid in matches:
        if pid not in seen:
            ordered.append(pid)
            seen.add(pid)
    return ordered


def validate_citations(
    report_text: str,
    evidence_patent_ids: List[str],
) -> Dict[str, Any]:
    """
    验证报告中的所有引用是否在证据库中。

    Args:
        report_text: LLM 生成的报告文本
        evidence_patent_ids: 注入 prompt 的证据库专利 ID 列表

    Returns:
        验证结果字典，包含：
        - total_citations: 引用总次数（含重复）
        - unique_citations: 去重后的引用数
        - valid_citations: 在证据库中的引用列表
        - hallucinated_citations: 不在证据库中的引用列表（幻觉）
        - faithfulness_rate: 忠实度比率 = valid / unique
        - validation_passed: 是否无幻觉（bool）
    """
    evidence_set = set(evidence_patent_ids)
    extracted = extract_citations(report_text)
    all_occurrences = _PATENT_ID_PATTERN.findall(report_text)

    valid: List[str] = []
    hallucinated: List[str] = []
    for pid in extracted:
        if pid in evidence_set:
            valid.append(pid)
        else:
            hallucinated.append(pid)

    total = len(all_occurrences)
    unique = len(extracted)
    faithfulness_rate = (len(valid) / unique) if unique > 0 else 1.0

    result: Dict[str, Any] = {
        "total_citations": total,
        "unique_citations": unique,
        "valid_citations": valid,
        "hallucinated_citations": hallucinated,
        "faithfulness_rate": round(faithfulness_rate, 4),
        "validation_passed": len(hallucinated) == 0,
    }

    if hallucinated:
        logger.warning(
            "发现 %d 个幻觉引用: %s",
            len(hallucinated), hallucinated,
        )
    else:
        logger.info(
            "引用验证通过: %d 个唯一引用全部在证据库中",
            unique,
        )

    return result


def clean_hallucinated_citations(
    report_text: str,
    hallucinated_ids: List[str],
    strategy: str = "mark",
) -> str:
    """
    清理报告中的幻觉引用。

    Args:
        report_text: 原始报告
        hallucinated_ids: 幻觉引用列表
        strategy:
            - "mark": 在幻觉引用后添加警告标记 [CN123A][⚠未验证]
            - "remove": 直接移除幻觉引用 [CN123A]
            - "keep": 保留原样（仅记录日志）

    Returns:
        处理后的报告文本
    """
    if not hallucinated_ids or strategy == "keep":
        return report_text

    cleaned = report_text
    for pid in hallucinated_ids:
        target = f"[{pid}]"
        if strategy == "mark":
            replacement = f"[{pid}][⚠未验证]"
        elif strategy == "remove":
            replacement = ""
        else:
            raise ValueError(f"未知的清理策略: {strategy}")
        cleaned = cleaned.replace(target, replacement)

    return cleaned


def generate_validation_report(
    validation_result: Dict[str, Any],
    rank: int = None,
) -> str:
    """
    生成可读的验证报告文本（追加到主报告末尾或独立保存）。

    Args:
        validation_result: validate_citations 的返回结果
        rank: 机会的 rank 编号（可选）

    Returns:
        Markdown 格式的验证报告
    """
    lines = []
    header = "### 引用验证报告"
    if rank is not None:
        header += f"（rank={rank}）"
    lines.append(header)
    lines.append("")
    lines.append(f"- **引用总次数**: {validation_result['total_citations']}")
    lines.append(f"- **唯一引用数**: {validation_result['unique_citations']}")
    lines.append(f"- **有效引用数**: {len(validation_result['valid_citations'])}")
    lines.append(f"- **幻觉引用数**: {len(validation_result['hallucinated_citations'])}")
    lines.append(f"- **忠实度**: {validation_result['faithfulness_rate']:.2%}")
    lines.append(f"- **验证状态**: {'✅ 通过' if validation_result['validation_passed'] else '⚠️ 未通过'}")

    if validation_result["hallucinated_citations"]:
        lines.append("")
        lines.append("**幻觉引用列表**（不在证据库中）:")
        for pid in validation_result["hallucinated_citations"]:
            lines.append(f"- `[{pid}]`")

    return "\n".join(lines)
