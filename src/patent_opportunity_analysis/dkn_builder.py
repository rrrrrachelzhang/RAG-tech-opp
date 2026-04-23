# src/dkn_builder.py

from typing import List, Tuple
import networkx as nx
from tqdm import tqdm
from loguru import logger

# 导入相关模块
from . import nlp_utils as _nlp_utils
from . import patent_graph as _patent_graph
from . import config as _config
from .utils import dkn_wrapper as _dkn_wrapper

NLPProcessor = _nlp_utils.NLPProcessor
PatentRecord = _patent_graph.PatentRecord
build_patent_graph = _patent_graph.build_patent_graph
DECAY_FACTOR = _config.DECAY_FACTOR
NODE_STRENGTH_MODE = getattr(_config, 'NODE_STRENGTH_MODE', 'weighted_degree')
DKNNetwork = _dkn_wrapper.DKNNetwork

def merge_patent_graphs(graphs: List[nx.Graph], batch_size: int = 100) -> nx.Graph:
    """
    批量合并专利图，提高大规模数据处理的效率
    优化边属性聚合，使用更高效的数据结构
    
    Args:
        graphs: 要合并的图列表
        batch_size: 批处理大小
    """
    if not graphs:
        return nx.Graph()
    
    logger.debug(f"开始合并 {len(graphs)} 个图，批处理大小: {batch_size}")
    merged_graph = nx.Graph()
    
    # 使用进度条显示合并进度
    total_batches = (len(graphs) + batch_size - 1) // batch_size
    
    # 批量处理
    for batch_idx in tqdm(range(0, len(graphs), batch_size), desc="合并图", total=total_batches):
        batch = graphs[batch_idx:batch_idx + batch_size]
        
        # 先收集本批次的节点和边数据
        # 注意：years 使用 list 保留多重计数，避免同一年多次贡献被去重导致边权少算
        batch_nodes = {}
        batch_edges = {}
        
        for g in batch:
            # 收集节点数据
            for node, data in g.nodes(data=True):
                if node not in batch_nodes:
                    batch_nodes[node] = {
                        'patents': set(),
                        'years': []
                    }
                batch_nodes[node]['patents'].update(data.get("patents", set()))
                batch_nodes[node]['years'].extend(list(data.get("years", set())))

            # 收集边数据
            for u, v, data in g.edges(data=True):
                edge_key = tuple(sorted([u, v]))
                if edge_key not in batch_edges:
                    batch_edges[edge_key] = {
                        'patents': set(),
                        'years': [],
                        'relations': set()
                    }
                batch_edges[edge_key]['patents'].update(data.get("patents", set()))
                batch_edges[edge_key]['years'].extend(list(data.get("years", set())))
                batch_edges[edge_key]['relations'].update(data.get("relations", set()))
        
        # 批量更新合并图
        for node, node_data in batch_nodes.items():
            if not merged_graph.has_node(node):
                merged_graph.add_node(node, patents=set(), years=[])
            merged_graph.nodes[node]["patents"].update(node_data['patents'])
            merged_graph.nodes[node]["years"].extend(node_data['years'])

        for (u, v), edge_data in batch_edges.items():
            if not merged_graph.has_edge(u, v):
                merged_graph.add_edge(u, v, patents=set(), years=[], relations=set())
            merged_graph.edges[u, v]["patents"].update(edge_data['patents'])
            merged_graph.edges[u, v]["years"].extend(edge_data['years'])
            merged_graph.edges[u, v]["relations"].update(edge_data['relations'])

    # 为节点和边统一设置 first_year（首现年份），HDKN/PDKN 均使用，O(1) 查询
    for n, data in merged_graph.nodes(data=True):
        years = data.get("years") or []
        if years:
            fy = min(years)
            merged_graph.nodes[n]["first_year"] = fy
            merged_graph.nodes[n]["year_min"] = fy
    for u, v, data in merged_graph.edges(data=True):
        years = data.get("years") or []
        if years:
            fy = min(years)
            merged_graph.edges[u, v]["first_year"] = fy
            merged_graph.edges[u, v]["year_min"] = fy

    logger.debug(f"合并完成: {merged_graph.number_of_nodes()} 个节点, {merged_graph.number_of_edges()} 条边")
    return merged_graph

def compute_time_decay_weights(dkn_graph: nx.Graph, total_year: int, alpha: float = DECAY_FACTOR, expected_ref_year: int = None):
    """
    计算时间衰减权重
    
    Args:
        dkn_graph: DKN图对象（可以是HDKN或PDKN）
        total_year: 参考年份（用于计算时间差）
        alpha: 衰减因子
        expected_ref_year: 期望的参考年份（用于验证，如果提供且不匹配则抛出异常）
    
    Raises:
        ValueError: 如果提供了 expected_ref_year 且与 total_year 不匹配
    """
    # 验证参考年份（如果提供了期望值）
    if expected_ref_year is not None and total_year != expected_ref_year:
        raise ValueError(
            f"参考年份不匹配: 期望 {expected_ref_year}, 实际 {total_year}. "
            f"这可能导致衰减权重计算错误。请检查调用处是否传入了正确的参考年份。"
        )
    
    # 先计算边权重（节点 strength 可能需要用到）
    # years 可为 list（保留多重计数）或 set（兼容旧缓存），均支持迭代
    for u, v, data in dkn_graph.edges(data=True):
        years = data.get("years") or []
        weight = sum(alpha ** (total_year - y) for y in years)
        dkn_graph.edges[u, v]["weight"] = weight
        first_year = min(years) if years else total_year
        dkn_graph.edges[u, v]["year_min"] = first_year
        dkn_graph.edges[u, v]["first_year"] = first_year  # 首现年份，用于时间因果过滤 O(1)

    # 节点权重（根据配置选择计算方式）
    for n, data in dkn_graph.nodes(data=True):
        years = data.get("years") or []
        first_year = min(years) if years else total_year
        dkn_graph.nodes[n]["year_min"] = first_year
        dkn_graph.nodes[n]["first_year"] = first_year  # 首现年份，用于时间因果过滤 O(1)
        
        if NODE_STRENGTH_MODE == "weighted_degree":
            # 严格版：加权度 = sum(incident edge weights)
            strength = sum(
                dkn_graph.edges[n, neighbor].get("weight", 0.0)
                for neighbor in dkn_graph.neighbors(n)
            )
        else:
            # 工程版：时间衰减权重之和 = sum(α^(T-year))
            strength = sum(alpha ** (total_year - y) for y in years)
        
        dkn_graph.nodes[n]["strength"] = strength

def build_dkns(patents: List[PatentRecord], hist_end_year: int) -> Tuple[DKNNetwork, DKNNetwork]:
    """
    构建：
    - HDKN（历史）：包含 app_year <= hist_end_year 的专利，ref_year = hist_end_year
    - PDKN（全时期）：包含所有专利，ref_year = max_year
    
    使用缓存和批量处理优化性能
    
    Returns:
        Tuple[DKNNetwork, DKNNetwork]: (HDKN, PDKN) 封装对象
    """
    nlp = NLPProcessor(use_cache=True)

    all_graphs = []
    hist_graphs = []

    max_year = max(p.app_year for p in patents) if patents else hist_end_year

    logger.info(f"开始构建DKN，处理 {len(patents)} 条专利...")
    
    # 使用进度条处理专利
    for p in tqdm(patents, desc="构建专利图"):
        try:
            text = (p.title or "") + ". " + (p.abstract or "")
            # 使用patent_id作为缓存键
            deps = nlp.extract_dependencies(text, patent_id=p.patent_id)
            g = build_patent_graph(p, deps)
            all_graphs.append(g)
            if p.app_year <= hist_end_year:
                hist_graphs.append(g)
        except Exception as e:
            logger.warning(f"处理专利 {p.patent_id} 时出错: {e}, 跳过")
            continue

    logger.info(f"开始合并图: HDKN({len(hist_graphs)}个图), PDKN({len(all_graphs)}个图)")
    
    # 批量合并图
    hdkn_graph = merge_patent_graphs(hist_graphs, batch_size=100)
    pdkn_graph = merge_patent_graphs(all_graphs, batch_size=100)

    # 检查HDKN是否为空
    if hdkn_graph.number_of_nodes() == 0:
        logger.warning(f"⚠️  HDKN为空！所有专利的年份都大于 {hist_end_year}")
        logger.warning(f"   这可能导致特征提取中的某些计算不准确。")
        logger.warning(f"   建议检查数据年份分布或调整 HIST_END_YEAR 配置。")
    else:
        logger.info("计算HDKN时间衰减权重...")
    # 验证 HDKN 使用正确的参考年份（HIST_END_YEAR）
    compute_time_decay_weights(hdkn_graph, total_year=hist_end_year, expected_ref_year=hist_end_year)
    
    logger.info("计算PDKN时间衰减权重...")
    compute_time_decay_weights(pdkn_graph, total_year=max_year)

    # 创建封装对象并验证不变量
    hdkn_network = DKNNetwork(
        kind="HDKN",
        graph=hdkn_graph,
        ref_year=hist_end_year,
        hist_end_year=hist_end_year
    )
    hdkn_network.assert_invariants()
    
    pdkn_network = DKNNetwork(
        kind="PDKN",
        graph=pdkn_graph,
        ref_year=max_year,
        hist_end_year=hist_end_year
    )
    pdkn_network.assert_invariants()
    
    logger.info(f"HDKN: {hdkn_network}")
    logger.info(f"PDKN: {pdkn_network}")

    return hdkn_network, pdkn_network
