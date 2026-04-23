# src/patent_opportunity_analysis/feature_registry.py
"""
特征注册表与统一特征计算接口

实现论文定义的8个特征：
- New_n, New_e: 新颖性（二值 0/1）——若 Si 中存在至少一个 Year_n(Year_e) > HDKN 中 90% 分位数的节点(边)，则为 1
- Min_pn: 最小专利覆盖数（set cover），覆盖 G_current 所有节点+边
- Con_n, Con_e: 常规性（中位数）——仅对 first_year<current_year 的历史节点/边取 strength/weight 中位数
- Eigen: 特征向量中心性（平均值）
- Constraint: Burt网络约束（最小值）
- Betweenness: 中介中心度（平均值，预期系数为正）
- Avg_Weight: 语义连贯性/技术紧密度 - 子网连边平均权重（预期系数为正）
"""

from typing import Dict, Optional, List, Set, Tuple, Any, Callable
import networkx as nx
import numpy as np
from loguru import logger
from math import inf

# 特征注册表：特征名 -> 计算函数
FEATURE_REGISTRY: Dict[str, Callable] = {
    "New_n": None,  # 将在 build_hdkn_feature_cache 后注册
    "New_e": None,
    "Min_pn": None,
    "Con_n": None,
    "Con_e": None,
    "Eigen": None,
    "Constraint": None,
    "Betweenness": None,
    "Avg_Weight": None,
}

def _normalize_edge_key(u: str, v: str) -> Tuple[str, str]:
    """规范化无向图边键"""
    return tuple(sorted([u, v]))

def compute_features_for_subnetwork(
    G_current: nx.Graph,
    hdkn_cache: Dict[str, Any],
    selected_features: Optional[List[str]] = None,
    current_patent_id: Optional[str] = None,
    current_year: Optional[int] = None,
    extra_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    基于当前专利独立构建的子图 G_current 与 HDKN 对比计算特征。

    Args:
        G_current: 当前专利的子图（由 build_patent_graph + 句法依存 独立构建）
        hdkn_cache: HDKN 特征缓存
        selected_features: 要计算的特征列表，None 表示计算全部
        current_patent_id: 当前专利 ID
        current_year: 当前专利所属年份
        extra_ctx: 额外上下文（如 title_stems, title_pairs），供特征函数使用

    Returns:
        特征字典 {feature_name: value}
    """
    if selected_features is None:
        selected_features = list(FEATURE_REGISTRY.keys())

    invalid_features = [f for f in selected_features if f not in FEATURE_REGISTRY]
    if invalid_features:
        raise ValueError(
            f"无效的特征名: {invalid_features}. 可用: {list(FEATURE_REGISTRY.keys())}"
        )

    if len(selected_features) == 0:
        raise ValueError("至少需要一个特征")

    hdkn_graph = hdkn_cache.get("hdkn_graph")
    if hdkn_graph is None:
        raise ValueError("hdkn_cache 必须包含 'hdkn_graph'")

    ctx = {
        **hdkn_cache,
        "current_patent_id": current_patent_id,
        "current_year": current_year,
    }
    if extra_ctx:
        ctx.update(extra_ctx)
    results = {}
    for feature_name in selected_features:
        fn = FEATURE_REGISTRY.get(feature_name)
        if fn is None:
            raise ValueError(f"特征 {feature_name} 的计算函数未注册")
        try:
            value = fn(G_current, ctx)
            results[feature_name] = value
        except Exception as e:
            logger.error(f"计算特征 {feature_name} 失败: {e}")
            results[feature_name] = np.nan

    return results


# ==================== 各特征计算函数（使用缓存） ====================

def _get_node_first_year(node: str, ctx: Dict[str, Any]) -> Optional[int]:
    """O(1) 获取节点在 HDKN 中的首现年份"""
    d = ctx.get("_year_min_node_dict") or ctx.get("_node_first_year_dict")
    if d is not None:
        return d.get(node)
    hdkn_graph = ctx.get("hdkn_graph")
    if hdkn_graph and node in hdkn_graph:
        return hdkn_graph.nodes[node].get("first_year") or hdkn_graph.nodes[node].get("year_min")
    return None


def _get_edge_first_year(u: str, v: str, ctx: Dict[str, Any]) -> Optional[int]:
    """O(1) 获取边在 HDKN 中的首现年份"""
    ek = _normalize_edge_key(u, v)
    d = ctx.get("_year_min_edge_dict") or ctx.get("_edge_first_year_dict")
    if d is not None:
        return d.get(ek)
    hdkn_graph = ctx.get("hdkn_graph")
    if hdkn_graph and hdkn_graph.has_edge(u, v):
        ed = hdkn_graph.edges[u, v]
        return ed.get("first_year") or ed.get("year_min")
    return None


def _is_node_historical(node: str, current_year: int, ctx: Dict[str, Any]) -> bool:
    """节点在 current_year 之前已存在于 HDKN（first_year < current_year）"""
    fy = _get_node_first_year(node, ctx)
    if fy is None:
        return False  # 不在 HDKN 中
    return fy < current_year


def _is_edge_historical(u: str, v: str, current_year: int, ctx: Dict[str, Any]) -> bool:
    """边在 current_year 之前已存在于 HDKN（first_year < current_year）"""
    fy = _get_edge_first_year(u, v, ctx)
    if fy is None:
        return False  # 不在 HDKN 中
    return fy < current_year


def _get_novelty_thresholds(ctx: Dict[str, Any]) -> Tuple[float, float]:
    """获取 New_n/New_e 的阈值（论文：90% 分位数；或 hist_end_year 工程版）"""
    from . import config as _config
    mode = getattr(_config, "NOVELTY_THRESHOLD_MODE", "quantile_90")
    hist_end_year = ctx.get("hist_end_year", getattr(_config, "HIST_END_YEAR", 2022))
    if mode == "hist_end_year":
        return float(hist_end_year), float(hist_end_year)
    p90_n = ctx.get("p90_year_n")
    p90_e = ctx.get("p90_year_e")
    if p90_n is None:
        p90_n = hist_end_year
    if p90_e is None:
        p90_e = hist_end_year
    return float(p90_n), float(p90_e)


def _compute_new_n(G_current: nx.Graph, ctx: Dict[str, Any]) -> float:
    """
    New_n: Si 中是否存在"新"节点（二值 0/1）。

    回归语境（有 title_stems）：仅检查标题词干是否在 HDKN 中。
    ACO 语境（无 title_stems）：检查子网所有节点，
        若节点不在 HDKN 中、或其 Year_n > p90 阈值，视为新节点。
    """
    hdkn_graph = ctx["hdkn_graph"]
    title_stems: Optional[set] = ctx.get("title_stems")

    if title_stems:
        return float(any(s not in hdkn_graph.nodes for s in title_stems))

    # ACO 语境：逐节点检查
    thresh_n, _ = _get_novelty_thresholds(ctx)
    year_dict = ctx.get("_year_min_node_dict", {})
    for n in G_current.nodes():
        if n not in hdkn_graph.nodes:
            return 1.0
        fy = year_dict.get(n)
        if fy is not None and fy > thresh_n:
            return 1.0
    return 0.0


def _compute_new_e(G_current: nx.Graph, ctx: Dict[str, Any]) -> float:
    """
    New_e: Si 中是否存在"新"边（二值 0/1）。

    回归语境（有 title_pairs）：仅检查标题词对是否在 HDKN 中。
    ACO 语境（无 title_pairs）：检查子网所有边，
        若边不在 HDKN 中、或其 Year_e > p90 阈值，视为新边。
    """
    hdkn_graph = ctx["hdkn_graph"]
    title_pairs: Optional[set] = ctx.get("title_pairs")

    if title_pairs:
        return float(any(
            not hdkn_graph.has_edge(u, v) and not hdkn_graph.has_edge(v, u)
            for u, v in title_pairs
        ))

    # ACO 语境：逐边检查
    _, thresh_e = _get_novelty_thresholds(ctx)
    year_dict = ctx.get("_year_min_edge_dict", {})
    for u, v in G_current.edges():
        if not hdkn_graph.has_edge(u, v):
            return 1.0
        ek = _normalize_edge_key(u, v)
        fy = year_dict.get(ek)
        if fy is not None and fy > thresh_e:
            return 1.0
    return 0.0

def _compute_min_pn(G_current: nx.Graph, ctx: Dict[str, Any]) -> float:
    """Min_pn: 覆盖 G_current 所有节点+边的最小专利数（set cover）。新节点/边由当前专利覆盖"""
    from .feature_extraction import (
        compute_min_pn_setcover_exact,
        compute_min_pn_setcover_greedy
    )

    hdkn_graph = ctx["hdkn_graph"]
    cached_node_patents = ctx.get("_node_patents")
    cached_edge_patents = ctx.get("_edge_patents")
    current_patent_id = ctx.get("current_patent_id")

    node_patents = {}
    edge_patents = {}
    if cached_node_patents is not None and cached_edge_patents is not None:
        for n in G_current.nodes():
            ps = cached_node_patents.get(n)
            if ps:
                node_patents[n] = ps
            elif current_patent_id:
                node_patents[n] = {current_patent_id}
        for u, v in G_current.edges():
            ek = _normalize_edge_key(u, v)
            ps = cached_edge_patents.get(ek)
            if ps:
                edge_patents[ek] = ps
            elif current_patent_id:
                edge_patents[ek] = {current_patent_id}
    else:
        for n in G_current.nodes():
            if n in hdkn_graph:
                ps = hdkn_graph.nodes[n].get("patents", set())
                if ps:
                    node_patents[n] = ps
            elif current_patent_id:
                node_patents[n] = {current_patent_id}
        for u, v in G_current.edges():
            ek = _normalize_edge_key(u, v)
            if hdkn_graph.has_edge(u, v):
                ps = hdkn_graph.edges[u, v].get("patents", set())
                if ps:
                    edge_patents[ek] = ps
            elif current_patent_id:
                edge_patents[ek] = {current_patent_id}

    target_elements = set()
    for node in G_current.nodes():
        target_elements.add(("n", node))
    for u, v in G_current.edges():
        target_elements.add(("e", _normalize_edge_key(u, v)))

    if not target_elements:
        return 0.0

    patent_to_elements: Dict[str, Set[Tuple[str, Any]]] = {}
    for element_type, element_id in target_elements:
        patents_set = node_patents.get(element_id, set()) if element_type == "n" else edge_patents.get(element_id, set())
        if not patents_set and current_patent_id:
            patents_set = {current_patent_id}
        for patent in patents_set:
            if patent not in patent_to_elements:
                patent_to_elements[patent] = set()
            patent_to_elements[patent].add((element_type, element_id))

    covered = set().union(*patent_to_elements.values())
    if covered != target_elements:
        logger.warning(f"Min_pn: {len(target_elements - covered)} 个元素无法覆盖，返回NaN")
        return np.nan

    min_pn_mode = ctx.get("min_pn_mode", "greedy")
    use_exact = min_pn_mode == "exact" or (len(patent_to_elements) <= 20 and len(target_elements) <= 20)
    if use_exact:
        try:
            return float(compute_min_pn_setcover_exact(target_elements, patent_to_elements))
        except Exception as e:
            logger.warning(f"Exact set cover 失败，回退到 greedy: {e}")
    return float(compute_min_pn_setcover_greedy(target_elements, patent_to_elements))

def _compute_con_n(G_current: nx.Graph, ctx: Dict[str, Any]) -> float:
    """Con_n: Si 中所有节点 Strength 的中位数（论文 Section 3.3(c)）"""
    strength_dict = ctx.get("_strength_dict") or ctx.get("node_strength", {})
    strengths = []
    for n in G_current.nodes():
        v = strength_dict.get(n)
        if v is not None and not np.isnan(v):
            strengths.append(float(v))
    if not strengths:
        return 0.0
    return float(np.median(strengths))


def _compute_con_e(G_current: nx.Graph, ctx: Dict[str, Any]) -> float:
    """Con_e: Si 中所有边 Weight 的中位数（论文 Section 3.3(c)）"""
    weight_dict = ctx.get("_weight_dict") or ctx.get("edge_weight", {})
    weights = []
    for u, v in G_current.edges():
        ek = _normalize_edge_key(u, v)
        val = weight_dict.get(ek)
        if val is not None and not np.isnan(val):
            weights.append(float(val))
    if not weights:
        return 0.0
    return float(np.median(weights))


def _compute_eigen(G_current: nx.Graph, ctx: Dict[str, Any]) -> float:
    """Eigen: Si 中所有节点 eigenvector centrality 的平均值（论文 Section 3.3(d)）"""
    eigen_dict = ctx.get("_eigen_dict") or ctx.get("node_eigen", {})
    vals = []
    for n in G_current.nodes():
        v = eigen_dict.get(n)
        if v is not None and not np.isnan(v):
            vals.append(float(v))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _compute_constraint(G_current: nx.Graph, ctx: Dict[str, Any]) -> float:
    """Constraint: Si 中所有节点 Burt's constraint 的最小值（论文 Section 3.3(e)）"""
    constraint_dict = ctx.get("_constraint_dict") or ctx.get("node_constraint", {})
    vals = []
    for n in G_current.nodes():
        v = constraint_dict.get(n)
        if v is not None and not np.isnan(v):
            vals.append(float(v))
    if not vals:
        return 1.0
    return float(min(vals))


def _compute_betweenness(G_current: nx.Graph, ctx: Dict[str, Any]) -> float:
    """Betweenness: Si 中所有节点 betweenness centrality 的平均值（扩展特征）"""
    betweenness_dict = ctx.get("_betweenness_dict") or ctx.get("node_betweenness", {})
    vals = []
    for n in G_current.nodes():
        v = betweenness_dict.get(n)
        if v is not None and not np.isnan(v):
            vals.append(float(v))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _compute_avg_weight(G_current: nx.Graph, ctx: Dict[str, Any]) -> float:
    """Avg_Weight: Si 中所有边权重的平均值（扩展特征）"""
    if G_current.number_of_edges() == 0:
        return 0.0
    weight_dict = ctx.get("_weight_dict") or ctx.get("edge_weight", {})
    weights = []
    for u, v in G_current.edges():
        ek = _normalize_edge_key(u, v)
        val = weight_dict.get(ek)
        if val is not None and not np.isnan(val):
            weights.append(float(val))
    if not weights:
        return 0.0
    return float(np.mean(weights))


# 注册特征计算函数（在 build_hdkn_feature_cache 后调用）
def _register_feature_functions():
    """注册特征计算函数到注册表"""
    FEATURE_REGISTRY["New_n"] = _compute_new_n
    FEATURE_REGISTRY["New_e"] = _compute_new_e
    FEATURE_REGISTRY["Min_pn"] = _compute_min_pn
    FEATURE_REGISTRY["Con_n"] = _compute_con_n
    FEATURE_REGISTRY["Con_e"] = _compute_con_e
    FEATURE_REGISTRY["Eigen"] = _compute_eigen
    FEATURE_REGISTRY["Constraint"] = _compute_constraint
    FEATURE_REGISTRY["Betweenness"] = _compute_betweenness
    FEATURE_REGISTRY["Avg_Weight"] = _compute_avg_weight
