"""
RAG 技术机会报告生成工作流

对齐 Ren & Zhao (2021) 方法论：
  1. 从 enriched JSON 中提取节点/边分类（new/marginal/special/conventional）
  2. 按解读重要性排序：special → new → marginal → conventional
  3. 构建 Z 值分解、novelty_sources、节点标记（★/◇）、边描述（含专利信息）
  4. 调用 DeepSeek API 生成结构化 Markdown 报告
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .rag_patents import fetch_patent_texts
from .rag_prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 格式检测
# ---------------------------------------------------------------------------

def _detect_format(data: Any) -> str:
    """自动检测输入 JSON 的格式: "enriched" | "candidates"。"""
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if "opportunity_rank" in first and "edges" in first:
                return "enriched"
            if "rank" in first and "nodes" in first:
                return "candidates"
    if isinstance(data, dict) and "opportunities" in data:
        return "candidates"
    raise ValueError(
        "无法识别的 JSON 格式。期望 enriched (含 opportunity_rank/edges) "
        "或 candidates (含 rank/nodes) 格式"
    )


# ---------------------------------------------------------------------------
# 边排序（分层配额策略）
# ---------------------------------------------------------------------------

def _sort_edges(edges: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """
    分层配额选边，确保 special / new / conventional 三种类型都有代表。

    配额规则：
      1. special 边全选（它们是 novel combination 的核心线索）
      2. 剩余配额在 new 和 conventional 之间大致均分
      3. 某类型不够时，配额让给另一类型
    排序规则：
      - special: 按 year_e 降序
      - new: 按 year_e 降序
      - conventional: 按 con_e 降序
    """
    by_type: Dict[str, List[Dict[str, Any]]] = {
        "special": [], "new": [], "conventional": [],
    }
    for e in edges:
        etype = e.get("type", "conventional")
        by_type.setdefault(etype, []).append(e)

    special_sorted = sorted(
        by_type.get("special", []),
        key=lambda e: -(e.get("year_e", 0) or 0),
    )
    selected = special_sorted[:]

    remaining = top_n - len(selected)
    if remaining <= 0:
        return selected[:top_n]

    new_sorted = sorted(
        by_type.get("new", []),
        key=lambda e: -(e.get("year_e", 0) or 0),
    )
    conv_sorted = sorted(
        by_type.get("conventional", []),
        key=lambda e: -(e.get("con_e", 0.0) or 0.0),
    )

    n_new_target = (remaining + 1) // 2
    n_conv_target = remaining - n_new_target

    n_new = min(len(new_sorted), n_new_target)
    n_conv = min(len(conv_sorted), n_conv_target)

    if n_new < n_new_target:
        n_conv = min(len(conv_sorted), remaining - n_new)
    elif n_conv < n_conv_target:
        n_new = min(len(new_sorted), remaining - n_conv)

    selected += new_sorted[:n_new]
    selected += conv_sorted[:n_conv]
    return selected


# ---------------------------------------------------------------------------
# 专利 ID 收集（按解读重要性排序）
# ---------------------------------------------------------------------------

def _collect_priority_patent_ids(
    top_edges: List[Dict[str, Any]],
    nodes_raw: List[Dict[str, Any]],
) -> List[str]:
    """
    按解读重要性收集专利 ID，确保关键证据不被截断。

    优先级：
      1. special 边的 sole_patent（消歧关键）
      2. new 边的代表专利
      3. 新兴节点(★)的代表专利
      4. 边缘节点(◇)的代表专利
      5. conventional 边的 top_patents 及其余节点专利
    """
    ids: List[str] = []
    seen: Set[str] = set()

    def _add(pid: str) -> None:
        pid = pid.strip()
        if pid and pid not in seen:
            ids.append(pid)
            seen.add(pid)

    # P1: special 边的 sole_patent
    for e in top_edges:
        if e.get("type") == "special":
            for key in ("sole_patent", "representative_patent"):
                pat = e.get(key)
                if pat and isinstance(pat, dict):
                    _add(pat.get("id", ""))

    # P2: new 边的代表专利
    for e in top_edges:
        if e.get("type") == "new":
            rep = e.get("representative_patent")
            if rep and isinstance(rep, dict):
                _add(rep.get("id", ""))

    # P3: 新兴节点的代表专利
    for n in nodes_raw:
        if isinstance(n, dict) and n.get("type") == "new":
            rep = n.get("representative_patent")
            if rep and isinstance(rep, dict):
                _add(rep.get("id", ""))

    # P4: 边缘节点的代表专利（is_marginal 或 type==marginal 或 eigen 极低的 conventional 节点）
    for n in nodes_raw:
        if not isinstance(n, dict):
            continue
        is_marginal_node = (
            n.get("is_marginal", False)
            or n.get("type") == "marginal"
            or (n.get("type") == "conventional"
                and n.get("eigen") is not None
                and n.get("eigen", 1.0) == 0.0)
        )
        if is_marginal_node:
            rep = n.get("representative_patent")
            if rep and isinstance(rep, dict):
                _add(rep.get("id", ""))

    # P5: conventional 边的 top_patents
    for e in top_edges:
        if e.get("type") == "conventional":
            for pat in e.get("top_patents", []):
                _add(pat.get("id", ""))

    # P6: 其余所有边/节点中的专利
    for e in top_edges:
        for key in ("representative_patent", "sole_patent"):
            pat = e.get(key)
            if pat and isinstance(pat, dict):
                _add(pat.get("id", ""))
        for pat in e.get("top_patents", []):
            _add(pat.get("id", ""))
    for n in nodes_raw:
        if isinstance(n, dict):
            rep = n.get("representative_patent")
            if rep and isinstance(rep, dict):
                _add(rep.get("id", ""))

    return ids


# ---------------------------------------------------------------------------
# 内嵌专利文本备用
# ---------------------------------------------------------------------------

def _build_embedded_patent_fallback(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> Dict[str, str]:
    """从 enriched JSON 内嵌数据构建专利文本备用字典（CSV 查不到时使用）。"""
    fallback: Dict[str, str] = {}

    for n in nodes:
        if not isinstance(n, dict):
            continue
        rep = n.get("representative_patent")
        if rep and isinstance(rep, dict):
            pid = rep.get("id", "")
            title = rep.get("title", "")
            abstract = rep.get("abstract", "")
            if pid and (title or abstract):
                fallback[pid] = f"{title}\n{abstract}" if abstract else title

    for e in edges:
        for key in ("representative_patent", "sole_patent"):
            pat = e.get(key)
            if pat and isinstance(pat, dict):
                pid = pat.get("id", "")
                title = pat.get("title", "")
                abstract = pat.get("abstract", "")
                if pid and (title or abstract):
                    fallback[pid] = f"{title}\n{abstract}" if abstract else title
        for pat in e.get("top_patents", []):
            pid = pat.get("id", "")
            title = pat.get("title", "")
            if pid and title and pid not in fallback:
                fallback[pid] = title

    return fallback


# ---------------------------------------------------------------------------
# enriched 格式提取
# ---------------------------------------------------------------------------

def _extract_from_enriched(
    opportunity: Dict[str, Any],
    top_n_edges: int,
    top_k_patents: int,
    patents_csv_path: Optional[Path],
    evidence_max_chars: int,
) -> Tuple[List[str], List[str], str, str, str, List[str]]:
    """
    从 enriched 格式提取：节点列表、边描述、证据字符串、Z 值分解、novelty_sources。

    返回 6 元组 (nodes_display, edges_desc, evidence_str, score_block, novelty_block, core_patent_ids)。
    """
    nodes_raw: List[Dict[str, Any]] = opportunity.get("nodes", [])
    novelty_sources: List[str] = opportunity.get("novelty_sources", [])
    novelty_set = set(novelty_sources)

    # ---- Z 值分解 (score_block) ----
    z_score = opportunity.get("z_score")
    feature_scores = opportunity.get("feature_scores", {})
    score_block = ""
    if z_score is not None:
        score_block = f"- **目标函数 Z = {z_score:.4f}**"
        if feature_scores:
            components = []
            for var_name in ["New_e", "Eigen", "Constraint", "New_n", "Min_pn", "Con_e"]:
                val = feature_scores.get(var_name)
                if val is not None:
                    components.append(f"{var_name}={val}")
            for k, v in feature_scores.items():
                if k not in ["New_e", "Eigen", "Constraint", "New_n", "Min_pn", "Con_e"]:
                    components.append(f"{k}={v}")
            if components:
                score_block += f"\n- 分项得分: {', '.join(components)}"

    # ---- novelty_sources (novelty_block) ----
    novelty_block = ""
    if novelty_sources:
        src_display = []
        for ns in novelty_sources:
            for n in nodes_raw:
                if isinstance(n, dict) and n.get("stem") == ns:
                    originals = n.get("original_forms", [])
                    src_display.append(originals[0] if originals else ns)
                    break
            else:
                src_display.append(ns)
        novelty_block = (
            f"- **新颖性来源节点子簇**: {', '.join(src_display)}\n"
            f"  （这些节点在 PDKN 中形成紧密互连的新兴技术单元，应作为整体分析）"
        )

    # ---- 节点标注 ★/◇ (nodes_display) ----
    nodes_display: List[str] = []
    for n in nodes_raw:
        if isinstance(n, dict):
            originals = n.get("original_forms", [])
            stem = n.get("stem", "")
            label = originals[0] if originals else stem
            ntype = n.get("type", "conventional")
            eigen_val = n.get("eigen")

            if ntype == "new" or stem in novelty_set:
                label += "★"
            else:
                is_marginal = (
                    n.get("is_marginal", False)
                    or ntype == "marginal"
                )
                if is_marginal and eigen_val is not None:
                    label += f"◇(Eigen={eigen_val:.6f})"
                elif is_marginal:
                    label += "◇"
                elif eigen_val is not None and eigen_val == 0.0:
                    label += "◇(Eigen=0.0)"

            nodes_display.append(label)
        else:
            nodes_display.append(str(n))

    # ---- 排序并选取 top_n 边 ----
    edges_raw: List[Dict[str, Any]] = opportunity.get("edges", [])
    top_edges = _sort_edges(edges_raw, top_n_edges)

    # ---- 边描述（含专利信息和类型标识）----
    edges_desc: List[str] = []
    for e in top_edges:
        pair = e.get("node_pair", [])
        src = pair[0] if len(pair) > 0 else "?"
        tgt = pair[1] if len(pair) > 1 else "?"
        etype = e.get("type", "unknown")
        year_e = e.get("year_e")
        con_e = e.get("con_e")

        parts = [f"类型: {etype}"]
        if year_e is not None:
            parts.append(f"首现年份: {year_e}")
        if con_e is not None:
            parts.append(f"历史连接强度: {con_e:.2f}")

        if etype == "special":
            sole = e.get("sole_patent") or e.get("representative_patent")
            if sole and isinstance(sole, dict):
                pid = sole.get("id", "")
                title = sole.get("title", "")
                parts.append(f"△ 唯一支撑专利: [{pid}] {title}")
        elif etype == "new":
            parts.append("★ 该关联在 HDKN 中不存在，为新兴技术组合")
            rep = e.get("representative_patent")
            if rep and isinstance(rep, dict):
                parts.append(
                    f"代表专利: [{rep.get('id', '')}] {rep.get('title', '')}"
                )
        elif etype == "conventional":
            parts.append("□ 成熟技术关联")

        edges_desc.append(f"[{src}] 与 [{tgt}] ({', '.join(parts)})")

    # ---- 按解读重要性收集专利 ID ----
    priority_patent_ids = _collect_priority_patent_ids(top_edges, nodes_raw)
    core_patent_ids = priority_patent_ids[:top_k_patents]

    # 从 CSV 查找专利全文
    patent_texts = fetch_patent_texts(core_patent_ids, csv_path=patents_csv_path)

    # CSV 查不到的，用 enriched JSON 内嵌数据补充
    embedded_fallback = _build_embedded_patent_fallback(nodes_raw, top_edges)
    for pid in core_patent_ids:
        if patent_texts.get(pid) == "暂无摘要" and pid in embedded_fallback:
            patent_texts[pid] = embedded_fallback[pid]

    # 组装证据字符串
    evidence_str = ""
    for pid, text in patent_texts.items():
        evidence_str += f"[{pid}]：{text}\n"

    if len(evidence_str) > evidence_max_chars:
        evidence_str = evidence_str[:evidence_max_chars] + "\n[证据库已截断]"
        logger.warning("证据库已截断至 %d 字符", evidence_max_chars)

    return nodes_display, edges_desc, evidence_str, score_block, novelty_block, core_patent_ids


# ---------------------------------------------------------------------------
# candidates 格式提取
# ---------------------------------------------------------------------------

def _extract_from_candidates(
    opportunity: Dict[str, Any],
    top_k_patents: int,
    patents_csv_path: Optional[Path],
    evidence_max_chars: int,
) -> Tuple[List[str], List[str], str, str, str, List[str]]:
    """从 candidates 格式（仅含 nodes/score/size）提取信息。"""
    nodes: List[str] = opportunity.get("nodes", [])
    edges_desc: List[str] = ["（候选格式无边信息，请基于节点组合进行分析）"]
    evidence_str = "（候选格式无内嵌专利证据）"
    score = opportunity.get("score")
    score_block = f"- **目标函数 Z = {score:.4f}**" if score is not None else ""
    return nodes, edges_desc, evidence_str, score_block, "", []


# ---------------------------------------------------------------------------
# 报告生成主入口
# ---------------------------------------------------------------------------

def generate_opportunity_report(
    opportunity_data: Dict[str, Any],
    *,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model_name: str = "deepseek-chat",
    max_tokens: int = 4096,
    temperature: float = 0.1,
    top_n_edges: int = 10,
    top_k_patents: int = 15,
    evidence_max_chars: int = 60000,
    patents_csv_path: Optional[Path] = None,
    data_format: str = "enriched",
    enable_citation_validation: bool = True,
    citation_cleanup_strategy: str = "mark",
    append_validation_report: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    根据技术机会子网数据生成专利分析报告。

    Args:
        opportunity_data: 单个机会的字典数据（enriched 或 candidates 格式）
        api_key: DeepSeek API Key
        base_url: API base URL
        model_name: 模型名称
        max_tokens: 最大生成 token 数
        temperature: 生成温度
        top_n_edges: 取前 N 条边（仅 enriched 格式有效）
        top_k_patents: 最多提供的核心专利数量
        evidence_max_chars: 证据库字符串总长度上限
        patents_csv_path: 专利 CSV 路径（None 则使用默认 data/raw/patents.csv）
        data_format: "enriched" 或 "candidates"
        enable_citation_validation: 是否启用引用验证（仅 enriched 格式有效）
        citation_cleanup_strategy: 幻觉引用处理策略 "mark" | "remove" | "keep"
        append_validation_report: 是否在报告末尾追加验证报告

    Returns:
        (report_text, validation_result) 二元组
        - report_text: 可能经过清理的报告文本
        - validation_result: 验证结果字典（enable_citation_validation=False 时为空字典）
    """
    if not api_key:
        raise ValueError(
            "未配置 DEEPSEEK_API_KEY，请设置环境变量或修改 configs/rag_config.yaml"
        )

    try:
        from openai import OpenAI
        import openai as _openai_module
    except ImportError as exc:
        raise ImportError("请安装 openai: pip install openai") from exc

    client = OpenAI(api_key=api_key, base_url=base_url)

    if data_format == "enriched":
        nodes, edges_desc, evidence_str, score_block, novelty_block, core_patent_ids = (
            _extract_from_enriched(
                opportunity_data, top_n_edges, top_k_patents,
                patents_csv_path, evidence_max_chars,
            )
        )
    else:
        nodes, edges_desc, evidence_str, score_block, novelty_block, core_patent_ids = (
            _extract_from_candidates(
                opportunity_data, top_k_patents,
                patents_csv_path, evidence_max_chars,
            )
        )

    user_prompt = build_user_prompt(
        nodes, edges_desc, evidence_str, score_block, novelty_block,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except _openai_module.APIError as e:
        logger.exception("DeepSeek API 调用失败: %s", e)
        raise

    choice = response.choices[0]
    report_text = choice.message.content or ""
    logger.info("报告生成成功，长度 %d 字符", len(report_text))

    # === 引用验证 ===
    validation_result: Dict[str, Any] = {}
    if enable_citation_validation and data_format == "enriched":
        from .rag_citation_validator import (
            validate_citations,
            clean_hallucinated_citations,
            generate_validation_report,
        )

        validation_result = validate_citations(report_text, core_patent_ids)

        if validation_result["hallucinated_citations"]:
            report_text = clean_hallucinated_citations(
                report_text,
                validation_result["hallucinated_citations"],
                strategy=citation_cleanup_strategy,
            )

        if append_validation_report:
            opp_rank = opportunity_data.get("opportunity_rank") or opportunity_data.get("rank")
            validation_md = generate_validation_report(validation_result, rank=opp_rank)
            report_text = f"{report_text}\n\n---\n\n{validation_md}"

    return report_text, validation_result


# ---------------------------------------------------------------------------
# JSON 加载
# ---------------------------------------------------------------------------

def load_opportunities(json_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """
    从 JSON 文件加载 opportunities 列表，自动检测格式。

    Returns:
        (opportunities_list, format_name) 二元组
    """
    if not json_path.exists():
        raise FileNotFoundError(f"opportunities 文件不存在: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fmt = _detect_format(data)

    if fmt == "enriched":
        return (data if isinstance(data, list) else []), fmt
    elif isinstance(data, list):
        return data, fmt
    else:
        return data.get("opportunities", []), fmt
