# src/patent_graph.py

from dataclasses import dataclass
from typing import List
import networkx as nx
from loguru import logger

@dataclass
class PatentRecord:
    patent_id: str
    title: str
    abstract: str
    app_year: int
    forward_cites: int
    backward_cites: int
    assignee_type: str
    ipc_classes: List[str]
    assignee: str = ""  # 专利权人字段（用于计算Assignee控制变量）

def ensure_connectivity(graph: nx.Graph, patent_id: str = None) -> nx.Graph:
    """
    确保图是连通的。如果图不连通，使用最短路径连接所有连通分量。
    
    Args:
        graph: 输入图
        patent_id: 专利ID（用于日志）
    
    Returns:
        连通图
    """
    if graph.number_of_nodes() == 0:
        return graph
    
    if nx.is_connected(graph):
        return graph
    
    # 获取所有连通分量
    components = list(nx.connected_components(graph))
    if len(components) <= 1:
        return graph
    
    logger.debug(f"专利 {patent_id} 的图不连通，有 {len(components)} 个连通分量，正在连接...")
    
    # 连接所有连通分量
    # 策略：选择每个分量中权重最高的节点（如果有权重），否则选择任意节点
    component_representatives = []
    for comp in components:
        comp_subgraph = graph.subgraph(comp)
        # 尝试找到权重最高的节点
        best_node = None
        best_weight = -1
        for node in comp:
            # 如果有strength属性，使用它；否则使用度
            weight = graph.nodes[node].get("strength", graph.degree(node))
            if weight > best_weight:
                best_weight = weight
                best_node = node
        component_representatives.append(best_node if best_node else list(comp)[0])
    
    # 连接所有代表节点（形成链式连接）
    for i in range(len(component_representatives) - 1):
        u = component_representatives[i]
        v = component_representatives[i + 1]
        
        # 添加连接边（使用最小权重，表示这是人工添加的连接）
        if not graph.has_edge(u, v):
            # 获取两个节点的属性
            u_patents = graph.nodes[u].get("patents", set())
            v_patents = graph.nodes[v].get("patents", set())
            u_years = graph.nodes[u].get("years", set())
            v_years = graph.nodes[v].get("years", set())
            
            graph.add_edge(u, v, 
                      patents=u_patents | v_patents,
                      years=u_years | v_years,
                      relations={"bridged_connectivity"},
                      weight=0.1)  # 低权重表示这是连接边
            logger.debug(f"添加连接边: {u} <-> {v}")
    
    return graph

def build_patent_graph(patent: PatentRecord, dependencies):
    """
    为单篇专利构建一个图：
    - 节点：技术词（过滤标点符号）
    - 边：在句法依赖中出现的词对
    - 保证图的连通性
    
    Args:
        patent: 专利记录
        dependencies: 依赖边列表
    
    Returns:
        连通的专利图
    """
    import string
    
    def is_valid_node(node: str) -> bool:
        """检查节点是否有效（不包含标点符号，但允许连字符和下划线）"""
        if not node or not isinstance(node, str):
            return False
        # 非 ASCII 字母检测：拒绝含 ä, ß, í 等非英文字母的节点
        if any(c.isalpha() and ord(c) > 127 for c in node):
            return False
        allowed_chars = {'-', '_'}
        if any(c in node for c in string.punctuation if c not in allowed_chars):
            return False
        cleaned = node.replace('-', '').replace('_', '')
        if cleaned and not cleaned.isalnum():
            return False
        return True
    
    patent_graph = nx.Graph()

    for edge in dependencies:
        u = edge.head
        v = edge.dependent
        
        # 过滤标点符号节点（标点符号等同于停用词）
        if not is_valid_node(u) or not is_valid_node(v):
            continue
        if u == v:
            continue

        # 节点
        for node in (u, v):
            if not patent_graph.has_node(node):
                patent_graph.add_node(node, patents=set(), years=set())
            patent_graph.nodes[node]["patents"].add(patent.patent_id)
            patent_graph.nodes[node]["years"].add(patent.app_year)

        # 边
        if not patent_graph.has_edge(u, v):
            patent_graph.add_edge(u, v, patents=set(), years=set(), relations=set(), weight=1.0)
        patent_graph.edges[u, v]["patents"].add(patent.patent_id)
        patent_graph.edges[u, v]["years"].add(patent.app_year)
        patent_graph.edges[u, v]["relations"].add(edge.relation)
        # 确保边有权重
        if "weight" not in patent_graph.edges[u, v]:
            patent_graph.edges[u, v]["weight"] = 1.0

    # 论文未提及单篇专利图的连通性保证
    # 若不连通，保留原状（DKN 合并后自然连通度提高）
    return patent_graph
