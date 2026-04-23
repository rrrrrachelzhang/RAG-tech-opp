# src/patent_opportunity_analysis/aco_to_rag.py
"""
ACO 子网 → RAG 富化 JSON 转换模块

将蚁群算法输出的机会子网转换为结构化 JSON，包含：
- 节点/边按论文定义分类（new / marginal / conventional / special）
- 代表性专利信息（标题、摘要、引用数）
- 新颖来源与可行性锚点
- 领域上下文元数据

下游 RAG 模块可直接使用此 JSON 进行证据检索和逻辑验证。
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from loguru import logger
from nltk.stem import PorterStemmer

from .hdkn_feature_cache import (
    build_hdkn_subnetwork_feature_cache,
    compute_subnetwork_features_for_aco,
)
from .feature_registry import _normalize_edge_key
from .patent_graph import PatentRecord

_porter = PorterStemmer()


# ---------------------------------------------------------------------------
# 1. 辅助：构建专利元数据索引 & 词干→原形映射
# ---------------------------------------------------------------------------

def build_patent_lookup(
    patents: List[PatentRecord],
) -> Dict[str, Dict[str, Any]]:
    """构建 patent_id → {title, abstract, forward_cites, app_year} 查找表"""
    lookup: Dict[str, Dict[str, Any]] = {}
    for p in patents:
        lookup[p.patent_id] = {
            "id": p.patent_id,
            "title": p.title or "",
            "abstract": p.abstract or "",
            "forward_cites": p.forward_cites,
            "app_year": p.app_year,
        }
    return lookup


def build_stem_to_originals(
    patents: List[PatentRecord],
) -> Dict[str, Set[str]]:
    """扫描所有专利的标题+摘要，构建 stem → {原形词1, 原形词2, …} 映射

    仅对字母词执行词干提取，不处理数字或标点。
    """
    mapping: Dict[str, Set[str]] = defaultdict(set)
    _word_re = re.compile(r"[a-zA-Z]+(?:-[a-zA-Z]+)*")
    for p in patents:
        text = (p.title or "") + " " + (p.abstract or "")
        for word in _word_re.findall(text):
            lower = word.lower()
            stem = _porter.stem(lower)
            mapping[stem].add(lower)
    return dict(mapping)


# ---------------------------------------------------------------------------
# 2. 节点/边分类
# ---------------------------------------------------------------------------

def _classify_node(
    node: str,
    pdkn_graph: nx.Graph,
    hdkn_graph: nx.Graph,
    p90_year_n: float,
    strength_dict: Dict[str, float],
    eigen_dict: Dict[str, float],
    pdkn_ref_year: int,
    strength_p50: float,
) -> str:
    """按论文 Table 3 分类节点

    - new: Year_n > p90 或不在 HDKN 中
    - marginal: 特征向量中心性极低（< 全网中位数的 1/10）且不是 new
    - conventional: 其余（通常是高 strength 的成熟节点）
    """
    # --- 新节点判定 ---
    hdkn_fy = None
    if node in hdkn_graph:
        hdkn_fy = hdkn_graph.nodes[node].get("first_year") or hdkn_graph.nodes[node].get("year_min")
    if hdkn_fy is None:
        return "new"
    if hdkn_fy >= p90_year_n:
        return "new"

    # --- 边缘节点判定：Eigen 极低 ---
    eigen = eigen_dict.get(node)
    if eigen is not None and eigen < 1e-4:
        return "marginal"

    return "conventional"


def _classify_edge(
    u: str, v: str,
    pdkn_graph: nx.Graph,
    hdkn_graph: nx.Graph,
    p90_year_e: float,
) -> str:
    """按论文 Table 3 分类边

    - new: Year_e > p90 或不在 HDKN 中
    - special: 仅出现在 1 篇专利中
    - conventional: 其余
    """
    # --- 新边判定 ---
    hdkn_fy = None
    if hdkn_graph.has_edge(u, v):
        ed = hdkn_graph.edges[u, v]
        hdkn_fy = ed.get("first_year") or ed.get("year_min")
    if hdkn_fy is None:
        return "new"
    if hdkn_fy >= p90_year_e:
        return "new"

    # --- 特殊边判定：仅 1 篇专利 ---
    if pdkn_graph.has_edge(u, v):
        patents_on_edge = pdkn_graph.edges[u, v].get("patents", set())
        if len(patents_on_edge) == 1:
            return "special"

    return "conventional"


# ---------------------------------------------------------------------------
# 3. 专利选取辅助
# ---------------------------------------------------------------------------

def _pick_representative_patent(
    patent_ids: Set[str],
    patent_lookup: Dict[str, Dict[str, Any]],
    prefer_recent: bool = True,
) -> Optional[Dict[str, str]]:
    """从候选专利中选出一篇代表性专利（引用最多或最新）"""
    if not patent_ids:
        return None
    candidates = []
    for pid in patent_ids:
        meta = patent_lookup.get(pid)
        if meta:
            candidates.append(meta)
    if not candidates:
        pid0 = next(iter(patent_ids))
        return {"id": pid0, "title": "", "abstract": ""}

    if prefer_recent:
        candidates.sort(key=lambda m: (m.get("forward_cites", 0), m.get("app_year", 0)), reverse=True)
    else:
        candidates.sort(key=lambda m: m.get("forward_cites", 0), reverse=True)

    best = candidates[0]
    return {
        "id": best["id"],
        "title": best["title"],
        "abstract": best["abstract"][:300],
    }


def _pick_top_patents(
    patent_ids: Set[str],
    patent_lookup: Dict[str, Dict[str, Any]],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """从候选专利中选出 top_n 篇（按引用降序）"""
    if not patent_ids:
        return []
    candidates = []
    for pid in patent_ids:
        meta = patent_lookup.get(pid)
        if meta:
            candidates.append(meta)
    candidates.sort(key=lambda m: m.get("forward_cites", 0), reverse=True)
    results = []
    for c in candidates[:top_n]:
        results.append({
            "id": c["id"],
            "title": c["title"],
            "cited_by": c.get("forward_cites", 0),
        })
    return results


# ---------------------------------------------------------------------------
# 4. 核心富化函数
# ---------------------------------------------------------------------------

def enrich_opportunities(
    opportunities: List[Dict[str, Any]],
    PDKN: Any,
    HDKN: Any,
    hdkn_cache: Dict[str, Any],
    feature_names: List[str],
    linear_objective: Any,
    pdkn_ref_year: int,
    patent_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    stem_to_originals: Optional[Dict[str, Set[str]]] = None,
    domain_field: str = "embodied intelligence",
) -> List[Dict[str, Any]]:
    """将 ACO 输出的粗粒度机会列表转换为模版格式的富化 JSON

    Args:
        opportunities: aco_search_opportunities 返回的列表，每项含 nodes/score/size
        PDKN: DKNNetwork (PDKN)
        HDKN: DKNNetwork (HDKN)
        hdkn_cache: build_hdkn_subnetwork_feature_cache 的产物
        feature_names: 参与目标函数的特征名列表
        linear_objective: LinearObjectiveFunction 实例
        pdkn_ref_year: PDKN 的参考年份
        patent_lookup: patent_id → {id, title, abstract, forward_cites, app_year}
        stem_to_originals: stem → {original_form1, …}
        domain_field: 领域名称

    Returns:
        富化后的机会 JSON 列表
    """
    if patent_lookup is None:
        patent_lookup = {}
    if stem_to_originals is None:
        stem_to_originals = {}

    pdkn_graph = PDKN.graph if hasattr(PDKN, "graph") else PDKN
    hdkn_graph = HDKN.graph if hasattr(HDKN, "graph") else HDKN

    p90_year_n = hdkn_cache.get("p90_year_n", 9999)
    p90_year_e = hdkn_cache.get("p90_year_e", 9999)
    strength_dict = hdkn_cache.get("_strength_dict", {})
    eigen_dict = hdkn_cache.get("_eigen_dict", {})

    # 全网 strength 中位数（用于 marginal 判定的参照）
    all_strengths = [v for v in strength_dict.values() if v and not np.isnan(v)]
    strength_p50 = float(np.median(all_strengths)) if all_strengths else 1.0

    # 第 70 百分位 Eigen 阈值（Eigen 低于此值的节点标记为 is_marginal）
    all_eigen_vals = [v for v in eigen_dict.values()
                      if v is not None and np.isfinite(v) and v > 0]
    eigen_threshold = (
        float(np.percentile(all_eigen_vals, 70)) if all_eigen_vals else 1e-4
    )

    # PDKN 节点数 & 年份范围
    pdkn_years = set()
    for _, data in pdkn_graph.nodes(data=True):
        fy = data.get("first_year") or data.get("year_min")
        if fy:
            pdkn_years.add(int(fy))
    pdkn_year_range = [min(pdkn_years), max(pdkn_years)] if pdkn_years else [0, 0]

    enriched: List[Dict[str, Any]] = []

    for rank, opp in enumerate(opportunities, 1):
        node_names: List[str] = opp.get("nodes", [])
        z_score = opp.get("score", 0.0)

        if len(node_names) < 2:
            continue

        # ---- 子图 & 特征 ----
        valid_nodes = [n for n in node_names if pdkn_graph.has_node(n)]
        if len(valid_nodes) < 2:
            continue
        subg = pdkn_graph.subgraph(valid_nodes).copy()

        feats = compute_subnetwork_features_for_aco(
            subg, HDKN,
            ref_year=pdkn_ref_year,
            hdkn_cache=hdkn_cache,
            selected_features=feature_names,
        )
        feature_scores = {k: round(float(v), 4) if np.isfinite(float(v)) else 0.0
                         for k, v in feats.items()}

        # ---- 节点富化 ----
        nodes_enriched: List[Dict[str, Any]] = []
        novelty_sources: List[str] = []
        feasibility_anchors: List[str] = []

        for node in valid_nodes:
            ntype = _classify_node(
                node, pdkn_graph, hdkn_graph,
                p90_year_n, strength_dict, eigen_dict,
                pdkn_ref_year, strength_p50,
            )
            nd = pdkn_graph.nodes[node]
            fy = nd.get("first_year") or nd.get("year_min")
            node_strength = nd.get("strength", strength_dict.get(node, 0.0))
            node_eigen = eigen_dict.get(node, 0.0)
            node_patents: Set[str] = nd.get("patents", set())

            originals = sorted(stem_to_originals.get(node, {node}))

            eigen_rounded = round(float(node_eigen), 6) if node_eigen else 0.0
            node_is_marginal = (
                ntype == "marginal"
                or (ntype == "conventional" and eigen_rounded < eigen_threshold)
            )

            node_entry: Dict[str, Any] = {
                "stem": node,
                "original_forms": originals,
                "type": ntype,
                "strength": round(float(node_strength), 2) if node_strength else 0.0,
                "year_first": int(fy) if fy else None,
                "eigen": eigen_rounded,
                "is_marginal": node_is_marginal,
            }

            rep = _pick_representative_patent(node_patents, patent_lookup)
            if rep:
                node_entry["representative_patent"] = rep

            if ntype in ("new", "marginal"):
                novelty_sources.append(node)
            else:
                feasibility_anchors.append(node)

            nodes_enriched.append(node_entry)

        # 按 strength 降序排列可行性锚点
        feasibility_anchors.sort(
            key=lambda n: strength_dict.get(n, 0.0), reverse=True
        )

        # ---- 边富化 ----
        edges_enriched: List[Dict[str, Any]] = []
        for u, v in subg.edges():
            etype = _classify_edge(u, v, pdkn_graph, hdkn_graph, p90_year_e)
            ed = pdkn_graph.edges[u, v] if pdkn_graph.has_edge(u, v) else {}
            edge_fy = ed.get("first_year") or ed.get("year_min")
            edge_patents: Set[str] = ed.get("patents", set())
            ek = _normalize_edge_key(u, v)
            edge_weight = ed.get("weight", 0.0)
            con_e_val = hdkn_cache.get("_weight_dict", {}).get(ek, edge_weight)

            edge_entry: Dict[str, Any] = {
                "node_pair": list(ek),
                "type": etype,
                "year_e": int(edge_fy) if edge_fy else None,
            }

            if etype == "special":
                sole = _pick_representative_patent(edge_patents, patent_lookup)
                if sole:
                    edge_entry["sole_patent"] = sole
            elif etype == "conventional":
                edge_entry["con_e"] = round(float(con_e_val), 2) if con_e_val else 0.0
                top = _pick_top_patents(edge_patents, patent_lookup, top_n=3)
                if top:
                    edge_entry["top_patents"] = top
            else:
                rep = _pick_representative_patent(edge_patents, patent_lookup)
                if rep:
                    edge_entry["representative_patent"] = rep

            edges_enriched.append(edge_entry)

        # ---- 组装 ----
        enriched.append({
            "opportunity_rank": rank,
            "z_score": round(float(z_score), 4),
            "feature_scores": feature_scores,
            "nodes": nodes_enriched,
            "edges": edges_enriched,
            "novelty_sources": novelty_sources,
            "feasibility_anchors": feasibility_anchors,
            "domain_context": {
                "field": domain_field,
                "pdkn_node_count": pdkn_graph.number_of_nodes(),
                "pdkn_year_range": pdkn_year_range,
            },
        })

    logger.success(f"富化完成: {len(enriched)}/{len(opportunities)} 个机会子网")
    return enriched
