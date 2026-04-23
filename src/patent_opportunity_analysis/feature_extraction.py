# src/patent_opportunity_analysis/feature_extraction.py

from typing import Set, Dict, Optional, List, Callable, Tuple, Any
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
import hashlib
import pickle
import gzip
import json
from pathlib import Path
from tqdm import tqdm
from math import inf

# 导入配置
from . import config as _config
from .utils.paths import CACHE_DIR
MIN_PN_MODE = getattr(_config, 'MIN_PN_MODE', 'setcover_greedy')
CENTRALITY_SOURCE = getattr(_config, 'CENTRALITY_SOURCE', 'HDKN')
HIST_END_YEAR = getattr(_config, 'HIST_END_YEAR', 2022)

def compute_new_flags(hdkn_graph: nx.Graph, subg: nx.Graph, thr_node: float, thr_edge: float):
    """
    计算新节点/边标志（使用HDKN）
    
    Args:
        hdkn_graph: HDKN图对象（历史网络）
        subg: 子图
        thr_node: 节点阈值
        thr_edge: 边阈值
    """
    new_n = 0
    new_e = 0

    for _, data in subg.nodes(data=True):
        if data.get("year_min", 0) >= thr_node:
            new_n = 1

    for _, _, data in subg.edges(data=True):
        if data.get("year_min", 0) >= thr_edge:
            new_e = 1

    return new_n, new_e

def compute_min_pn_union(subg: nx.Graph) -> int:
    """
    工程版：计算节点和边专利集合的并集大小
    
    Args:
        subg: 子图
    
    Returns:
        并集大小
    """
    patent_sets = []

    for _, d in subg.nodes(data=True):
        patent_sets.append(d.get("patents", set()))

    for _, _, d in subg.edges(data=True):
        patent_sets.append(d.get("patents", set()))

    if not patent_sets:
        return 0

    return len(set().union(*patent_sets))


def compute_min_pn_setcover_exact(
    target_elements: Set[Tuple[str, Any]],
    patent_to_elements: Dict[str, Set[Tuple[str, Any]]]
) -> int:
    """
    精确计算最小集合覆盖（使用bitmask + BFS/迭代加深）
    
    Args:
        target_elements: 目标元素集合，每个元素是 (type, id) 元组
        patent_to_elements: 专利到元素集合的映射
    
    Returns:
        最小专利数
    """
    if not target_elements:
        return 0
    
    if not patent_to_elements:
        return inf  # 无法覆盖
    
    # 将元素映射到索引（用于bitmask）
    element_list = list(target_elements)
    element_to_idx = {elem: idx for idx, elem in enumerate(element_list)}
    num_elements = len(element_list)
    
    # 将专利转换为bitmask
    patent_masks = {}
    for patent, elements in patent_to_elements.items():
        mask = 0
        for elem in elements:
            if elem in element_to_idx:
                mask |= (1 << element_to_idx[elem])
        patent_masks[patent] = mask
    
    # 目标mask：所有位都为1
    target_mask = (1 << num_elements) - 1
    
    # BFS搜索最小覆盖
    from collections import deque
    queue = deque([(0, 0)])  # (current_mask, num_patents)
    visited = {0}
    
    while queue:
        current_mask, num_patents = queue.popleft()
        
        if current_mask == target_mask:
            return num_patents
        
        # 尝试添加每个专利
        for patent, mask in patent_masks.items():
            new_mask = current_mask | mask
            if new_mask not in visited:
                visited.add(new_mask)
                queue.append((new_mask, num_patents + 1))
    
    return inf  # 无法完全覆盖

def compute_min_pn_setcover_greedy(
    target_elements: Set[Tuple[str, Any]],
    patent_to_elements: Dict[str, Set[Tuple[str, Any]]]
) -> int:
    """
    贪心算法计算最小集合覆盖
    
    Args:
        target_elements: 目标元素集合
        patent_to_elements: 专利到元素集合的映射
    
    Returns:
        最小专利数（贪心近似）
    """
    if not target_elements:
        return 0
    
    uncovered = target_elements.copy()
    selected_patents = set()
    
    while uncovered:
        # 选择覆盖未覆盖元素最多的专利
        best_patent = None
        best_coverage = 0
        
        for patent, elements in patent_to_elements.items():
            if patent in selected_patents:
                continue
            coverage = len(elements & uncovered)
            if coverage > best_coverage:
                best_coverage = coverage
                best_patent = patent
        
        if best_patent is None:
            break  # 无法继续覆盖
        
        selected_patents.add(best_patent)
        uncovered -= patent_to_elements[best_patent]
    
    if uncovered:
        # 无法完全覆盖
        return inf
    
    return len(selected_patents)


def compute_min_pn_nodes_edges(
    subg: nx.Graph,
    graph_with_patents: nx.Graph,
    fallback_patent_id: Optional[str] = None,
    mode: str = "greedy",
) -> float:
    """
    计算覆盖子网所有节点和边的最小专利数（符合论文定义）。
    
    与 feature_registry._compute_min_pn 逻辑一致，供 ACO 等模块使用。
    
    Args:
        subg: 子图 Si
        graph_with_patents: 包含 patents 属性的图（HDKN 或 PDKN），用于获取节点/边的专利归属
        fallback_patent_id: 对图中不存在的节点/边使用的回退专利 ID（如当前专利）
        mode: "greedy" 或 "exact"
    
    Returns:
        最小专利数，无法覆盖时返回 np.nan
    """
    def _ek(u: str, v: str) -> Tuple[str, str]:
        return tuple(sorted([u, v]))

    target_elements: Set[Tuple[str, Any]] = set()
    for n in subg.nodes():
        target_elements.add(("n", n))
    for u, v in subg.edges():
        target_elements.add(("e", _ek(u, v)))

    if not target_elements:
        return 0.0

    node_patents: Dict[Any, Set[str]] = {}
    edge_patents: Dict[Tuple[str, str], Set[str]] = {}
    for n in subg.nodes():
        ps = set()
        if n in graph_with_patents:
            ps = graph_with_patents.nodes[n].get("patents", set()) or set()
        if not ps and fallback_patent_id:
            ps = {fallback_patent_id}
        if ps:
            node_patents[n] = ps
    for u, v in subg.edges():
        ek = _ek(u, v)
        ps = set()
        if graph_with_patents.has_edge(u, v):
            ps = graph_with_patents.edges[u, v].get("patents", set()) or set()
        if not ps and fallback_patent_id:
            ps = {fallback_patent_id}
        if ps:
            edge_patents[ek] = ps

    patent_to_elements: Dict[str, Set[Tuple[str, Any]]] = {}
    for elem_type, elem_id in target_elements:
        ps = node_patents.get(elem_id, set()) if elem_type == "n" else edge_patents.get(elem_id, set())
        if not ps and fallback_patent_id:
            ps = {fallback_patent_id}
        for p in ps:
            if p not in patent_to_elements:
                patent_to_elements[p] = set()
            patent_to_elements[p].add((elem_type, elem_id))

    covered = set().union(*patent_to_elements.values()) if patent_to_elements else set()
    if covered != target_elements:
        logger.warning(f"Min_pn: {len(target_elements - covered)} 个元素无法覆盖，返回NaN")
        return np.nan

    use_exact = mode == "exact" and len(patent_to_elements) <= 20 and len(target_elements) <= 20
    if use_exact:
        try:
            return float(compute_min_pn_setcover_exact(target_elements, patent_to_elements))
        except Exception as e:
            logger.warning(f"Min_pn exact set cover 失败，回退 greedy: {e}")
    return float(compute_min_pn_setcover_greedy(target_elements, patent_to_elements))


def compute_min_pn_setcover(subg: nx.Graph) -> int:
    """
    旧版：仅覆盖子网所有 **边** 的最小专利数。

    ⚠️ 论文定义是覆盖所有 **节点和边**，请使用 feature_registry._compute_min_pn
    或 compute_min_pn_nodes_edges 代替。

    Args:
        subg: 子图

    Returns:
        最小专利数
    """
    if subg.number_of_edges() == 0:
        return 0
    
    # 收集每条边的专利集合
    edge_patents = {}
    for u, v, data in subg.edges(data=True):
        edge_key = tuple(sorted([u, v]))
        patents_set = data.get("patents", set())
        if patents_set:  # 只处理有专利的边
            edge_patents[edge_key] = patents_set
    
    if not edge_patents:
        return 0
    
    # 构建专利到边的反向索引（优化：避免每次迭代重新计算）
    patent_to_edges: Dict[str, Set] = {}
    for edge_key, patents_set in edge_patents.items():
        for patent in patents_set:
            if patent not in patent_to_edges:
                patent_to_edges[patent] = set()
            patent_to_edges[patent].add(edge_key)
    
    # 贪心 set cover（优化版）
    covered_edges = set()
    selected_patents = set()
    uncovered_edges = set(edge_patents.keys())  # 未覆盖的边集合
    
    while uncovered_edges:
        # 找到覆盖未覆盖边最多的专利（使用反向索引加速）
        best_patent = None
        best_coverage = 0
        
        # 只遍历有未覆盖边的专利（使用反向索引）
        candidates = set()
        for edge_key in uncovered_edges:
            candidates.update(edge_patents[edge_key])
        
        for patent in candidates:
            if patent in selected_patents:
                continue
            
            # 使用反向索引快速计算覆盖数
            patent_edges = patent_to_edges.get(patent, set())
            coverage = len(patent_edges & uncovered_edges)  # 交集大小
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_patent = patent
        
        if best_patent is None:
            break
        
        # 选择该专利，标记覆盖的边
        selected_patents.add(best_patent)
        newly_covered = patent_to_edges.get(best_patent, set()) & uncovered_edges
        uncovered_edges -= newly_covered
        covered_edges.update(newly_covered)
    
    return len(selected_patents)


def compute_min_pn(subg: nx.Graph, mode: str = None) -> int:
    """
    计算最小专利数
    
    Args:
        subg: 子图
        mode: 计算模式 ("setcover_greedy" 或 "union")，如果为 None 则使用配置
    
    Returns:
        最小专利数
    """
    if mode is None:
        mode = MIN_PN_MODE
    
    if mode == "setcover_greedy":
        return compute_min_pn_setcover(subg)
    else:
        return compute_min_pn_union(subg)

def compute_conventionality(
    subg: nx.Graph,
    node_stats_df: Optional[pd.DataFrame] = None,
    edge_stats_df: Optional[pd.DataFrame] = None
):
    """
    计算常规性特征（从缓存聚合）
    
    Args:
        subg: 子图
        node_stats_df: 节点统计DataFrame（从缓存）
        edge_stats_df: 边统计DataFrame（从缓存）
    
    Returns:
        (con_n, con_e)
    """
    # 如果提供了缓存数据，从缓存聚合
    if node_stats_df is not None and edge_stats_df is not None:
        # Con_n: 从node_stats_df查strength，取median
        node_strengths = []
        for node in subg.nodes():
            if node in node_stats_df.index:
                strength = node_stats_df.at[node, 'strength']
                if not np.isnan(strength):
                    node_strengths.append(float(strength))
        con_n = float(np.median(node_strengths)) if node_strengths else np.nan
        
        # Con_e: 从edge_stats_df查weight，取median（subg 可能含 PDKN 独有边，仅 HDKN 边可查）
        # 使用 .loc[[edge_key]] 避免 tuple 被解析为 (row,col)，且避免重复索引时的 AssertionError
        edge_weights = []
        for u, v in subg.edges():
            edge_key = tuple(sorted([u, v]))
            if edge_key in edge_stats_df.index:
                vals = edge_stats_df.loc[[edge_key], 'weight'].values
                weight = float(vals[0]) if len(vals) > 0 else np.nan
                if not np.isnan(weight):
                    edge_weights.append(weight)
        con_e = float(np.median(edge_weights)) if edge_weights else np.nan
        
        return con_n, con_e
    else:
        # 回退到旧实现（从子图直接读取）
        edge_weights = [d.get("weight", 0.0) for _, _, d in subg.edges(data=True)]
        node_strength = [d.get("strength", 0.0) for _, d in subg.nodes(data=True)]

        con_e = float(np.median(edge_weights)) if edge_weights else 0.0
        con_n = float(np.median(node_strength)) if node_strength else 0.0
        return con_n, con_e

# 全局缓存，避免重复计算相同图的eigenvector
_eigen_cache: Dict[int, Dict[str, float]] = {}
# 全局缓存，避免对每条专利重复计算整图 constraint（Step2 主要瓶颈）
_constraint_cache: Dict[int, Dict[str, float]] = {}
# HDKN constraint map 预计算缓存（内存）
_hdkn_constraint_map_cache: Dict[str, Dict[str, float]] = {}

def _get_graph_hash(graph: nx.Graph) -> int:
    """生成图的哈希值用于缓存"""
    # 使用节点和边的数量作为简单哈希
    return hash((graph.number_of_nodes(), graph.number_of_edges(), tuple(sorted(graph.nodes()))))

def _get_graph_fingerprint(graph: nx.Graph, hist_end_year: int, weight: Optional[str] = None) -> str:
    """生成图的稳定指纹用于缓存文件名

    包含边权采样，确保不同 α 衰减下的图使用不同缓存（constraint 依赖边权）。
    
    Args:
        graph: 图对象
        hist_end_year: 历史截止年份
        weight: 权重字段名
    
    Returns:
        指纹字符串（用于缓存文件名）
    """
    # 使用节点数、边数、节点ID集合的哈希值生成稳定指纹
    nodes_sorted = tuple(sorted(graph.nodes()))
    edges_sorted = tuple(sorted(graph.edges()))
    # 边权采样：不同 α 导致边权不同，需区分缓存
    weight_key = weight or "weight"
    edges_with_data = list(graph.edges(data=True))
    edges_with_data.sort(key=lambda x: (x[0], x[1]))
    step = max(1, len(edges_with_data) // 500)
    weight_sample = tuple(
        round(edges_with_data[i][2].get(weight_key, 0.0), 6)
        for i in range(0, min(len(edges_with_data), 500), step)
    )
    fingerprint_data = (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        nodes_sorted[:100] if len(nodes_sorted) > 100 else nodes_sorted,
        edges_sorted[:100] if len(edges_sorted) > 100 else edges_sorted,
        hist_end_year,
        weight_key,
        weight_sample,
    )
    fingerprint_str = str(fingerprint_data)
    return hashlib.md5(fingerprint_str.encode('utf-8')).hexdigest()[:16]

def get_hdkn_constraint_map(
    hdkn_dkn,
    nodes: Optional[List[str]] = None,
    weight: Optional[str] = "weight",
    use_cache: bool = True,
    hist_end_year: Optional[int] = None
) -> Dict[str, float]:
    """
    在 HDKN 上预计算节点 constraint 并缓存
    
    核心优化：只计算一次全图 constraint，然后对每个子网 Si 仅做 O(|Si|) 的 min 聚合。
    
    Args:
        hdkn_dkn: HDKN 对象（DKNNetwork 或 nx.Graph），必须是 HDKN
        nodes: 可选的节点集合；若提供，只计算这些节点（用于加速）
        weight: 若图有边权则沿用现有权重字段名；否则 None
        use_cache: 是否启用落盘缓存
        hist_end_year: 历史截止年份（用于缓存一致性校验）
    
    Returns:
        dict[node_id] = constraint_value (float)
    
    Raises:
        ValueError: 如果传入的不是 HDKN
    """
    # 断言：确保传入的是 HDKN
    if hasattr(hdkn_dkn, 'assert_kind'):
        hdkn_dkn.assert_kind("HDKN")
        hdkn_graph = hdkn_dkn.graph
        if hist_end_year is None:
            hist_end_year = hdkn_dkn.hist_end_year
    else:
        hdkn_graph = hdkn_dkn
        if hist_end_year is None:
            hist_end_year = HIST_END_YEAR
    
    if hist_end_year is None:
        hist_end_year = HIST_END_YEAR
    
    # 生成缓存键
    graph_fingerprint = _get_graph_fingerprint(hdkn_graph, hist_end_year, weight)
    cache_key = f"{graph_fingerprint}_{hist_end_year}_{weight or 'none'}"
    
    # 检查内存缓存（挂在 hdkn_dkn 对象上）
    if hasattr(hdkn_dkn, 'node_metrics') and isinstance(hdkn_dkn.node_metrics, dict):
        if 'constraint' in hdkn_dkn.node_metrics:
            constraint_map = hdkn_dkn.node_metrics['constraint']
            logger.info(f"✅ 从内存缓存（hdkn_dkn.node_metrics）读取 constraint map（{len(constraint_map)} 个节点）")
            # 如果指定了 nodes，过滤结果
            if nodes is not None:
                return {n: constraint_map.get(n, 1.0) for n in nodes if n in constraint_map}
            return constraint_map
    
    # 检查全局内存缓存
    if cache_key in _hdkn_constraint_map_cache:
        constraint_map = _hdkn_constraint_map_cache[cache_key]
        logger.info(f"✅ 从全局内存缓存读取 constraint map（{len(constraint_map)} 个节点）")
        # 如果指定了 nodes，过滤结果
        if nodes is not None:
            return {n: constraint_map.get(n, 1.0) for n in nodes if n in constraint_map}
        return constraint_map
    
    # 检查落盘缓存
    if use_cache:
        cache_file = CACHE_DIR / f"hdkn_constraint_{cache_key}.pkl.gz"
        if cache_file.exists():
            try:
                with gzip.open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # 一致性校验
                    if (cached_data.get('graph_fingerprint') == graph_fingerprint and
                        cached_data.get('hist_end_year') == hist_end_year and
                        cached_data.get('weight') == weight):
                        constraint_map = cached_data['constraint_map']
                        logger.info(f"✅ 从磁盘缓存读取 constraint map（{len(constraint_map)} 个节点），跳过计算")
                        logger.info(f"   缓存文件: {cache_file.name}")
                        # 存入内存缓存
                        _hdkn_constraint_map_cache[cache_key] = constraint_map
                        # 存入 hdkn_dkn 对象
                        if hasattr(hdkn_dkn, 'node_metrics'):
                            if not isinstance(hdkn_dkn.node_metrics, dict):
                                hdkn_dkn.node_metrics = {}
                            hdkn_dkn.node_metrics['constraint'] = constraint_map
                        # 如果指定了 nodes，过滤结果
                        if nodes is not None:
                            return {n: constraint_map.get(n, 1.0) for n in nodes if n in constraint_map}
                        return constraint_map
                    else:
                        logger.warning(f"缓存文件不一致，将重新计算（fingerprint/hist_end_year/weight 不匹配）")
            except Exception as e:
                logger.warning(f"读取缓存文件失败: {e}，将重新计算")
    
    # 计算 constraint
    num_nodes_to_compute = len(nodes) if nodes else hdkn_graph.number_of_nodes()
    num_nodes_total = hdkn_graph.number_of_nodes()
    num_edges_total = hdkn_graph.number_of_edges()
    
    # 检查是否跳过大图的 Constraint 计算（性能优化）
    skip_for_large = getattr(_config, 'SKIP_CONSTRAINT_FOR_LARGE_GRAPH', False)
    large_graph_threshold = getattr(_config, 'LARGE_GRAPH_THRESHOLD', 5000)
    
    if skip_for_large and num_nodes_total > large_graph_threshold:
        logger.warning(
            f"⚠️  跳过 Constraint 计算（图大小 {num_nodes_total} > {large_graph_threshold}，计算太慢）"
        )
        logger.warning("   提示：Constraint 是可选变量，不影响回归模型训练")
        logger.warning("   如需启用，请设置 config.SKIP_CONSTRAINT_FOR_LARGE_GRAPH = False")
        return {}
    
    logger.info(f"📊 计算 HDKN constraint map（图大小: {num_nodes_total}节点, {num_edges_total}边）...")
    logger.info(f"   将计算 {num_nodes_to_compute} 个节点的 constraint（候选节点: {len(nodes) if nodes else '全部'}）")
    
    try:
        from scipy import sparse as sp

        # 使用排序保证节点顺序可复现（多次运行 LL 一致）
        all_nodes = sorted(hdkn_graph.nodes())
        n = len(all_nodes)
        node_to_idx = {nd: i for i, nd in enumerate(all_nodes)}

        logger.info("   [1/3] 构建稀疏邻接矩阵...")
        try:
            A = nx.to_scipy_sparse_array(hdkn_graph, nodelist=all_nodes, weight=weight, format='csr')
        except AttributeError:
            A = nx.to_scipy_sparse_matrix(hdkn_graph, nodelist=all_nodes, weight=weight, format='csr')
        A = A.astype(np.float64)
        A.setdiag(0)
        A.eliminate_zeros()

        logger.info("   [2/3] 计算比例矩阵 P 和 P²...")
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        D_inv = sp.diags(1.0 / row_sums, format='csr')
        P = D_inv @ A
        PP = P @ P

        logger.info("   [3/3] 聚合 Burt's constraint...")
        M = P + PP
        mask = (A != 0).astype(np.float64)
        M_masked = M.multiply(mask)
        M_sq = M_masked.multiply(M_masked)
        constraints_arr = np.array(M_sq.sum(axis=1)).flatten()

        constraint_dict = {all_nodes[i]: float(constraints_arr[i]) for i in range(n)}

        logger.info(f"   ✅ Constraint 计算完成（{len(constraint_dict)} 个节点）")
        
        # 处理 NaN 值（低度节点或孤立节点可能返回 NaN）
        # 将 NaN 替换为保守默认值 1.0
        nan_count = 0
        for node, value in constraint_dict.items():
            if np.isnan(value) or value is None:
                constraint_dict[node] = 1.0
                nan_count += 1
        if nan_count > 0:
            logger.info(f"   处理了 {nan_count} 个 NaN/None 值（低度节点或孤立节点），使用默认值 1.0")
        
        # 存入内存缓存
        _hdkn_constraint_map_cache[cache_key] = constraint_dict
        
        # 存入 hdkn_dkn 对象
        if hasattr(hdkn_dkn, 'node_metrics'):
            if not isinstance(hdkn_dkn.node_metrics, dict):
                hdkn_dkn.node_metrics = {}
            hdkn_dkn.node_metrics['constraint'] = constraint_dict
        
        # 存入落盘缓存
        if use_cache:
            try:
                cache_file = CACHE_DIR / f"hdkn_constraint_{cache_key}.pkl.gz"
                cache_data = {
                    'graph_fingerprint': graph_fingerprint,
                    'hist_end_year': hist_end_year,
                    'weight': weight,
                    'constraint_map': constraint_dict
                }
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.debug(f"Constraint map 已保存到缓存: {cache_file}")
            except Exception as e:
                logger.warning(f"保存缓存文件失败: {e}")
        
        # 如果指定了 nodes，过滤结果
        if nodes is not None:
            return {n: constraint_dict.get(n, 1.0) for n in nodes if n in constraint_dict}
        
        return constraint_dict
        
    except Exception as e:
        logger.error(f"计算 constraint 失败: {e}")
        raise


# 中介中心度缓存（用于 ACO 等场景）
_hdkn_betweenness_map_cache: Dict[str, Dict[str, float]] = {}


def get_hdkn_betweenness_map(
    hdkn_dkn,
    nodes: Optional[List[str]] = None,
    weight: Optional[str] = "weight",
    hist_end_year: Optional[int] = None
) -> Dict[str, float]:
    """
    在 HDKN 上预计算节点 Betweenness Centrality（中介中心度）。

    大图使用 k 采样近似以加速。预期该特征系数为正（中介性高的节点更易被引用）。

    Args:
        hdkn_dkn: HDKN 对象（DKNNetwork 或 nx.Graph）
        nodes: 可选的节点集合；若提供，只返回这些节点
        weight: 边权字段名
        hist_end_year: 历史截止年份（用于缓存键）

    Returns:
        dict[node_id] = betweenness_value (float, normalized)
    """
    if hasattr(hdkn_dkn, 'assert_kind'):
        hdkn_dkn.assert_kind("HDKN")
        hdkn_graph = hdkn_dkn.graph
        if hist_end_year is None:
            hist_end_year = getattr(hdkn_dkn, 'hist_end_year', HIST_END_YEAR)
    else:
        hdkn_graph = hdkn_dkn
        hist_end_year = hist_end_year or HIST_END_YEAR

    cache_key = _get_graph_fingerprint(hdkn_graph, hist_end_year, weight)

    if cache_key in _hdkn_betweenness_map_cache:
        bc_dict = _hdkn_betweenness_map_cache[cache_key]
        if nodes is not None:
            return {n: bc_dict.get(n, 0.0) for n in nodes if n in bc_dict}
        return bc_dict

    n_nodes = hdkn_graph.number_of_nodes()
    k_sample = min(5000, n_nodes) if n_nodes > 2000 else None
    if k_sample:
        logger.info(f"计算 HDKN Betweenness（k={k_sample} 采样近似，{n_nodes} 节点）...")
    else:
        logger.info(f"计算 HDKN Betweenness（{n_nodes} 节点）...")

    bc_dict = nx.betweenness_centrality(
        hdkn_graph,
        k=k_sample,
        weight=weight,
        normalized=True,
        seed=42
    )
    _hdkn_betweenness_map_cache[cache_key] = bc_dict

    if nodes is not None:
        return {n: bc_dict.get(n, 0.0) for n in nodes if n in bc_dict}
    return bc_dict


def compute_eigen_centrality_power_iteration(
    graph: nx.Graph,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    weight: str = "weight"
) -> Dict[str, float]:
    """
    使用 Power Iteration + scipy 稀疏矩阵计算特征向量中心性。
    相比纯 Python 循环版本，大规模稀疏图上快 ~50-100x。
    """
    if graph.number_of_nodes() == 0:
        return {}

    nodes = sorted(graph.nodes())  # 保证节点顺序可复现
    n = len(nodes)

    try:
        A = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight=weight, format='csr')
    except AttributeError:
        A = nx.to_scipy_sparse_matrix(graph, nodelist=nodes, weight=weight, format='csr')

    x = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        x_new = A @ x
        norm = np.linalg.norm(x_new, 1)
        if norm == 0:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new
            break
        x = x_new

    return {nodes[i]: float(x[i]) for i in range(n)}

def compute_eigen_centrality(
    hdkn_graph: nx.Graph,
    subg: nx.Graph,
    use_cache: bool = True,
    node_stats_df: Optional[pd.DataFrame] = None
):
    """
    计算特征向量中心性（从缓存聚合）
    
    Args:
        hdkn_graph: HDKN图对象（历史网络）
        subg: 子图
        use_cache: 是否使用缓存（向后兼容）
        node_stats_df: 节点统计DataFrame（从缓存）
    
    Returns:
        子网节点eigen的平均值
    """
    if len(subg.nodes()) == 0:
        return 0.0
    
    # 如果提供了缓存数据，从缓存聚合
    if node_stats_df is not None:
        eigen_values = []
        for node in subg.nodes():
            if node in node_stats_df.index:
                eigen = node_stats_df.loc[node, 'eigen']
                if not np.isnan(eigen):
                    eigen_values.append(float(eigen))
        return float(np.mean(eigen_values)) if eigen_values else np.nan
    
    # 回退到旧实现（计算全图）
    if len(hdkn_graph.nodes) == 0:
        return 0.0
    
    # 检查缓存
    cache_key = _get_graph_hash(hdkn_graph)
    if use_cache and cache_key in _eigen_cache:
        ec = _eigen_cache[cache_key]
    else:
        try:
            logger.info(f"📊 计算 Eigen centrality（首次计算，图大小: {hdkn_graph.number_of_nodes()}节点, {hdkn_graph.number_of_edges()}边）...")
            # 优先使用power iteration（更快）
            if hdkn_graph.number_of_nodes() > 1000:
                # 对于大图，使用power iteration
                logger.info("   检查图的连通性...")
                is_conn = nx.is_connected(hdkn_graph)
                logger.info(f"   图连通性: {is_conn}")
                if is_conn:
                    logger.info("   使用 power iteration 计算 eigen centrality（可能需要1-2分钟）...")
                    ec = compute_eigen_centrality_power_iteration(hdkn_graph, weight="weight")
                    logger.info("   ✅ Eigen centrality 计算完成")
                else:
                    # 对于不连通图，使用最大连通分量
                    logger.info("   查找最大连通分量...")
                    largest_cc = max(nx.connected_components(hdkn_graph), key=len)
                    logger.info(f"   最大连通分量大小: {len(largest_cc)}")
                    if len(largest_cc) > 1:
                        hdkn_sub = hdkn_graph.subgraph(largest_cc).copy()
                        logger.info("   使用 power iteration 计算最大连通分量的 eigen centrality（可能需要1-2分钟）...")
                        ec = compute_eigen_centrality_power_iteration(hdkn_sub, weight="weight")
                        # 为不在最大连通分量中的节点设置0
                        ec = {n: ec.get(n, 0.0) for n in hdkn_graph.nodes()}
                        logger.info("   ✅ Eigen centrality 计算完成")
                    else:
                        return 0.0
            else:
                # 对于小图，使用numpy版本（更精确）
                if nx.is_connected(hdkn_graph):
                    ec = nx.eigenvector_centrality_numpy(hdkn_graph, weight="weight", max_iter=1000)
                else:
                    largest_cc = max(nx.connected_components(hdkn_graph), key=len)
                    if len(largest_cc) > 1:
                        hdkn_sub = hdkn_graph.subgraph(largest_cc).copy()
                        ec = nx.eigenvector_centrality_numpy(hdkn_sub, weight="weight", max_iter=1000)
                        ec = {n: ec.get(n, 0.0) for n in hdkn_graph.nodes()}
                    else:
                        return 0.0
            
            # 存入缓存
            if use_cache:
                _eigen_cache[cache_key] = ec
        except (nx.NetworkXError, ValueError, np.linalg.LinAlgError) as e:
            # 如果计算失败，使用度中心性作为替代
            logger.warning(f"Eigenvector计算失败，使用度中心性替代: {e}")
            try:
                ec = nx.degree_centrality(hdkn_graph)
            except Exception:
                return 0.0
    
    values = [ec.get(n, 0.0) for n in subg.nodes()]
    return float(np.mean(values)) if values else 0.0


def compute_constraint_feature(
    hdkn_graph: nx.Graph,
    subg: nx.Graph,
    constraint_map: Optional[Dict[str, float]] = None,
    use_cache: bool = True,
    allow_fallback: bool = True,
    node_stats_df: Optional[pd.DataFrame] = None
) -> float:
    """
    计算 Burt's constraint 特征（重构版：使用预计算的 constraint_map）
    
    核心优化：
    1. 优先使用预计算的 constraint_map（通过 get_hdkn_constraint_map 获得）
    2. 对每个子网 Si 仅做 O(|Si|) 的 min 聚合，不再重复计算
    3. 如果没有提供 constraint_map，会尝试从 hdkn_graph 对象获取或回退到旧实现
    
    Args:
        hdkn_graph: HDKN 图对象（DKNNetwork 或 nx.Graph）
        subg: 子图
        constraint_map: 预计算的 constraint map（dict[node_id] = constraint_value）
                        如果为 None，会尝试从 hdkn_graph.node_metrics 获取
        use_cache: 是否使用缓存（仅用于回退到旧实现时）
    
    Returns:
        子网节点 constraint 的最小值
    
    注意：
        - 强烈建议在特征提取入口预计算 constraint_map，然后传递给此函数
        - 如果未提供 constraint_map，函数会回退到旧实现（低效，会警告）
    """
    if len(subg.nodes()) == 0:
        return 0.0
    
    subg_nodes = list(subg.nodes())
    
    # 如果提供了缓存数据，从缓存聚合
    if node_stats_df is not None:
        constraint_values = []
        for node in subg_nodes:
            if node in node_stats_df.index:
                constraint = node_stats_df.loc[node, 'constraint']
                if not np.isnan(constraint):
                    constraint_values.append(float(constraint))
        if constraint_values:
            return float(min(constraint_values))
        else:
            logger.warning("Constraint: 所有节点constraint缺失，返回保守默认值1.0")
            return 1.0
    
    # 尝试获取预计算的 constraint_map
    if constraint_map is None:
        # 尝试从 hdkn_graph 对象获取
        if hasattr(hdkn_graph, 'node_metrics') and isinstance(hdkn_graph.node_metrics, dict):
            if 'constraint' in hdkn_graph.node_metrics:
                constraint_map = hdkn_graph.node_metrics['constraint']
                logger.debug(f"从 hdkn_graph.node_metrics 读取 constraint map")
        
        # 如果 hdkn_graph 是 DKNNetwork，也尝试从它获取
        if constraint_map is None and hasattr(hdkn_graph, 'graph'):
            # hdkn_graph 可能是 DKNNetwork，尝试获取底层 graph
            underlying_graph = hdkn_graph.graph if hasattr(hdkn_graph, 'graph') else hdkn_graph
            if hasattr(hdkn_graph, 'node_metrics') and isinstance(hdkn_graph.node_metrics, dict):
                if 'constraint' in hdkn_graph.node_metrics:
                    constraint_map = hdkn_graph.node_metrics['constraint']
                    logger.debug(f"从 DKNNetwork.node_metrics 读取 constraint map")
    
    # 如果找到了预计算的 constraint_map，直接做 min 聚合
    if constraint_map is not None:
        # O(|Si|) 的 min 聚合
        subg_constraints = []
        for node in subg_nodes:
            value = constraint_map.get(node, 1.0)  # 缺失节点使用保守默认值 1.0
            # 处理 NaN 值（低度节点可能返回 NaN）
            if np.isnan(value) or value is None:
                value = 1.0
            subg_constraints.append(float(value))
        
        if not subg_constraints:
            logger.warning(f"子网 Si 中所有节点都缺失 constraint 值，返回默认值 1.0")
            return 1.0
        
        result = float(min(subg_constraints))
        logger.debug(f"Constraint(Si) = {result:.6f} (从预计算 map 聚合，|Si|={len(subg_nodes)})")
        return result
    
    # 回退到旧实现（低效，会警告）
    if not allow_fallback:
        raise RuntimeError(
            "❌ 未提供预计算的 constraint_map，且 allow_fallback=False（测试模式禁止回退）。"
            "请确保在特征提取入口调用 get_hdkn_constraint_map 预计算一次。"
        )
    
    logger.warning(
        "⚠️  未提供预计算的 constraint_map，回退到旧实现（低效）。"
        "建议在特征提取入口调用 get_hdkn_constraint_map 预计算一次。"
    )
    
    # 检查是否跳过大图的 Constraint 计算（性能优化）
    skip_for_large = getattr(_config, 'SKIP_CONSTRAINT_FOR_LARGE_GRAPH', False)
    large_graph_threshold = getattr(_config, 'LARGE_GRAPH_THRESHOLD', 5000)
    
    # 获取底层 graph（如果是 DKNNetwork）
    actual_hdkn_graph = hdkn_graph.graph if hasattr(hdkn_graph, 'graph') else hdkn_graph
    
    if skip_for_large and actual_hdkn_graph.number_of_nodes() > large_graph_threshold:
        logger.debug(f"跳过 Constraint 计算（图大小 {actual_hdkn_graph.number_of_nodes()} > {large_graph_threshold}）")
        return 0.0
    
    try:
        # 旧实现：使用子网节点集合作为缓存键的一部分
        if use_cache:
            subg_nodes_key = tuple(sorted(subg_nodes))
            cache_key = (_get_graph_hash(actual_hdkn_graph), subg_nodes_key)
            
            if cache_key in _constraint_cache:
                constraint_dict = _constraint_cache[cache_key]
                subg_constraints = [constraint_dict.get(node, float('inf')) for node in subg_nodes]
                return float(min(subg_constraints)) if subg_constraints else 0.0
        
        # 旧实现：计算 constraint（低效）
        optimization_mode = getattr(_config, 'CONSTRAINT_OPTIMIZATION_MODE', 'subgraph_only')
        
        if optimization_mode == "subgraph_only":
            logger.debug(f"计算 Constraint（旧实现：只计算 {len(subg_nodes)} 个子网节点，图大小: {actual_hdkn_graph.number_of_nodes()}节点）...")
            constraint_dict = nx.algorithms.structuralholes.constraint(
                actual_hdkn_graph,
                nodes=subg_nodes,
                weight="weight"
            )
        else:
            logger.debug(f"计算 Constraint（旧实现：计算整个图，{actual_hdkn_graph.number_of_nodes()}节点）...")
            constraint_dict = nx.algorithms.structuralholes.constraint(
                actual_hdkn_graph,
                weight="weight"
            )
        
        # 存入缓存
        if use_cache:
            _constraint_cache[cache_key] = constraint_dict
        
        # 取最小值
        subg_constraints = [
            constraint_dict.get(node, float('inf'))
            for node in subg_nodes
        ]
        
        return float(min(subg_constraints)) if subg_constraints else 0.0
    except Exception as e:
        logger.warning(f"计算 constraint 失败: {e}")
        return 0.0


# ==================== HDKN特征缓存构建 ====================

def build_hdkn_feature_cache(
    hdkn_dkn,
    weight_edge_attr: str = "weight",
    weight_node_attr: str = "strength",
    use_cache: bool = True,
    min_pn_mode: str = "greedy"
) -> Dict[str, Any]:
    """
    构建HDKN特征缓存（全局预计算，性能关键）
    
    缓存包含：
    - p90_year_n: HDKN nodes的Year_n的90分位阈值
    - p90_year_e: HDKN edges的Year_e的90分位阈值
    - node_strength: dict[node] = HDKN.Strength
    - edge_weight: dict[(u,v)] = HDKN.Weight（无向图边规范化key）
    - node_eigen: dict[node] = eigenvector centrality (computed on HDKN)
    - node_constraint: dict[node] = Burt constraint (computed on HDKN)
    - node_patents: dict[node] = set of patent IDs
    - edge_patents: dict[edge_key] = set of patent IDs
    
    Args:
        hdkn_dkn: HDKN对象（DKNNetwork或nx.Graph），必须是HDKN
        weight_edge_attr: 边权重属性名（默认"weight"）
        weight_node_attr: 节点强度属性名（默认"strength"）
        use_cache: 是否启用磁盘缓存
        min_pn_mode: Min_pn计算模式（"exact"或"greedy"）
    
    Returns:
        特征缓存字典
    """
    # 断言：确保传入的是HDKN
    if hasattr(hdkn_dkn, 'assert_kind'):
        hdkn_dkn.assert_kind("HDKN")
        hdkn_graph = hdkn_dkn.graph
        hist_end_year = hdkn_dkn.hist_end_year
    else:
        hdkn_graph = hdkn_dkn
        hist_end_year = HIST_END_YEAR
    
    # 生成缓存键
    graph_fingerprint = _get_graph_fingerprint(hdkn_graph, hist_end_year, weight_edge_attr)
    cache_key = f"hdkn_feature_cache_{graph_fingerprint}_{hist_end_year}_{weight_edge_attr}_{weight_node_attr}"
    
    # 检查磁盘缓存
    if use_cache:
        cache_file = CACHE_DIR / f"{cache_key}.pkl.gz"
        meta_file = CACHE_DIR / f"{cache_key}_meta.json"
        
        if cache_file.exists() and meta_file.exists():
            try:
                # 读取meta文件验证一致性
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                if (meta.get('graph_fingerprint') == graph_fingerprint and
                    meta.get('hist_end_year') == hist_end_year and
                    meta.get('weight_edge_attr') == weight_edge_attr and
                    meta.get('weight_node_attr') == weight_node_attr):
                    
                    # 读取缓存数据
                    with gzip.open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    logger.info(f"✅ 从磁盘缓存读取HDKN特征缓存（{len(cache_data.get('node_strength', {}))}个节点）")
                    logger.info(f"   缓存文件: {cache_file.name}")
                    
                    # 添加hdkn_graph引用（不缓存，因为图对象太大）
                    cache_data['hdkn_graph'] = hdkn_graph
                    cache_data['min_pn_mode'] = min_pn_mode
                    
                    return cache_data
                else:
                    logger.warning("缓存文件不一致，将重新计算")
            except Exception as e:
                logger.warning(f"读取缓存文件失败: {e}，将重新计算")
    
    # 计算缓存
    logger.info(f"📊 构建HDKN特征缓存（图大小: {hdkn_graph.number_of_nodes()}节点, {hdkn_graph.number_of_edges()}边）...")
    
    cache = {
        'hdkn_graph': hdkn_graph,
        'hist_end_year': hist_end_year,
        'graph_fingerprint': graph_fingerprint,
        'weight_edge_attr': weight_edge_attr,
        'weight_node_attr': weight_node_attr,
        'min_pn_mode': min_pn_mode,
    }
    
    # 1. 计算P90阈值（Year_n和Year_e）
    logger.info("   计算P90阈值...")
    node_years = []
    for node, data in hdkn_graph.nodes(data=True):
        year_n = data.get("year_min")  # 映射：year_min -> Year_n
        if year_n is not None:
            node_years.append(year_n)
    
    edge_years = []
    for u, v, data in hdkn_graph.edges(data=True):
        year_e = data.get("year_min")  # 映射：year_min -> Year_e
        if year_e is not None:
            edge_years.append(year_e)
    
    cache['p90_year_n'] = float(np.percentile(node_years, 90)) if node_years else hist_end_year
    cache['p90_year_e'] = float(np.percentile(edge_years, 90)) if edge_years else hist_end_year
    logger.info(f"   P90 Year_n: {cache['p90_year_n']:.2f}, P90 Year_e: {cache['p90_year_e']:.2f}")
    
    # 2. 提取node_strength和edge_weight
    logger.info("   提取节点strength和边weight...")
    node_strength = {}
    node_patents = {}
    for node, data in hdkn_graph.nodes(data=True):
        strength = data.get(weight_node_attr)
        if strength is not None:
            node_strength[node] = float(strength)
        patents_set = data.get("patents", set())
        if patents_set:
            node_patents[node] = patents_set
    
    edge_weight = {}
    edge_patents = {}
    for u, v, data in hdkn_graph.edges(data=True):
        edge_key = tuple(sorted([u, v]))  # 规范化无向图边
        weight = data.get(weight_edge_attr)
        if weight is not None:
            edge_weight[edge_key] = float(weight)
        patents_set = data.get("patents", set())
        if patents_set:
            edge_patents[edge_key] = patents_set
    
    cache['node_strength'] = node_strength
    cache['edge_weight'] = edge_weight
    cache['node_patents'] = node_patents
    cache['edge_patents'] = edge_patents
    logger.info(f"   提取完成: {len(node_strength)}个节点strength, {len(edge_weight)}条边weight")
    
    # 3. 计算node_eigen（直接计算全图，不使用旧接口）
    logger.info("   计算Eigen centrality...")
    try:
        # 直接计算全图的eigenvector centrality字典
        if hdkn_graph.number_of_nodes() > 1000:
            eigen_dict = compute_eigen_centrality_power_iteration(hdkn_graph, weight=weight_edge_attr)
        else:
            if nx.is_connected(hdkn_graph):
                eigen_dict = nx.eigenvector_centrality_numpy(hdkn_graph, weight=weight_edge_attr, max_iter=1000)
            else:
                largest_cc = max(nx.connected_components(hdkn_graph), key=len)
                if len(largest_cc) > 1:
                    hdkn_sub = hdkn_graph.subgraph(largest_cc).copy()
                    eigen_dict = nx.eigenvector_centrality_numpy(hdkn_sub, weight=weight_edge_attr, max_iter=1000)
                    eigen_dict = {n: eigen_dict.get(n, 0.0) for n in hdkn_graph.nodes()}
                else:
                    eigen_dict = {n: 0.0 for n in hdkn_graph.nodes()}
        cache['node_eigen'] = eigen_dict
        logger.info(f"   ✅ Eigen centrality计算完成（{len(eigen_dict)}个节点）")
    except Exception as e:
        logger.error(f"计算Eigen centrality失败: {e}")
        cache['node_eigen'] = {}
    
    # 4. 计算node_constraint（使用现有函数）
    logger.info("   计算Constraint...")
    try:
        constraint_dict = get_hdkn_constraint_map(
            hdkn_dkn,
            nodes=None,  # 计算所有节点
            weight=weight_edge_attr,
            use_cache=use_cache,
            hist_end_year=hist_end_year
        )
        cache['node_constraint'] = constraint_dict
        logger.info(f"   ✅ Constraint计算完成（{len(constraint_dict)}个节点）")
    except Exception as e:
        logger.error(f"计算Constraint失败: {e}")
        cache['node_constraint'] = {}
    
    # 保存到磁盘缓存
    if use_cache:
        try:
            # 保存数据（不包含hdkn_graph，因为太大）
            cache_data_to_save = {k: v for k, v in cache.items() if k != 'hdkn_graph'}
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(cache_data_to_save, f)
            
            # 保存meta文件
            meta = {
                'graph_fingerprint': graph_fingerprint,
                'hist_end_year': hist_end_year,
                'weight_edge_attr': weight_edge_attr,
                'weight_node_attr': weight_node_attr,
            }
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"HDKN特征缓存已保存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存文件失败: {e}")
    
    logger.success(f"✅ HDKN特征缓存构建完成")
    return cache


def extract_title_subnetwork(
    patent_title: str,
    hdkn: nx.Graph,
    nlp_processor=None,
    patent_id: str = None,
) -> Tuple[nx.Graph, Set[str], Set[Tuple[str, str]]]:
    """
    根据论文 Ren & Zhao (2021) Section 3.2(4) 构建专利标题子网 S_i，
    同时返回原始词干集合和词对集合（用于 New_n/New_e 新颖性判断）。

    Returns:
        (S_i, title_stems, title_pairs)
        - S_i: HDKN 上的 induced subgraph
        - title_stems: 标题中所有词干（含 HDKN 中没有的）
        - title_pairs: 标题中所有词对 (u,v)，key 已排序（含 HDKN 中没有的边）
    """
    if nlp_processor is None:
        from .nlp_utils import NLPProcessor
        nlp_processor = NLPProcessor()

    deps = nlp_processor.extract_dependencies(patent_title, patent_id=patent_id)

    title_stems: Set[str] = set()
    title_pairs: Set[Tuple[str, str]] = set()
    for edge in deps:
        title_stems.add(edge.head)
        title_stems.add(edge.dependent)
        title_pairs.add(tuple(sorted([edge.head, edge.dependent])))

    if not title_stems:
        logger.debug(f"专利 {patent_id}: 标题解析无有效词干节点，返回空图")
        return nx.Graph(), set(), set()

    hdkn_nodes = [stem for stem in title_stems if stem in hdkn]

    if not hdkn_nodes:
        logger.debug(
            f"专利 {patent_id}: 标题词干 {title_stems} 均不在 HDKN 中，返回空图"
        )
        return nx.Graph(), title_stems, title_pairs

    matched_ratio = len(hdkn_nodes) / len(title_stems)
    if matched_ratio < 0.5:
        logger.debug(
            f"专利 {patent_id}: 标题词干匹配率较低 "
            f"({len(hdkn_nodes)}/{len(title_stems)} = {matched_ratio:.1%})"
        )

    subgraph = hdkn.subgraph(hdkn_nodes).copy()

    if subgraph.number_of_nodes() == 0:
        return subgraph, title_stems, title_pairs

    if not nx.is_connected(subgraph):
        components = list(nx.connected_components(subgraph))
        largest_cc = max(components, key=len)
        logger.debug(
            f"专利 {patent_id}: 标题子网不连通（{len(components)} 个分量），"
            f"取最大连通分量（{len(largest_cc)}/{subgraph.number_of_nodes()} 节点）"
        )
        subgraph = subgraph.subgraph(largest_cc).copy()

    return subgraph, title_stems, title_pairs
