# src/utils/dkn_wrapper.py

from dataclasses import dataclass
from typing import Literal, Set, Dict, Tuple, Any
import networkx as nx

@dataclass
class DKNNetwork:
    """DKN网络封装，包含身份标签和不变量
    
    用于彻底消除 HDKN 与 PDKN 在代码中的混淆与误传。
    
    Attributes:
        kind: 网络类型，"HDKN" 或 "PDKN"
        graph: 底层的 NetworkX 图对象
        ref_year: 参考年份（用于时间衰减权重计算）
        hist_end_year: 历史截止年份（仅用于 HDKN，PDKN 中设为 max_year）
    """
    kind: Literal["HDKN", "PDKN"]
    graph: nx.Graph
    ref_year: int
    hist_end_year: int
    
    def assert_kind(self, expected: Literal["HDKN", "PDKN"]):
        """断言网络类型
        
        Args:
            expected: 期望的网络类型
            
        Raises:
            ValueError: 如果网络类型不匹配
        """
        if self.kind != expected:
            raise ValueError(
                f"网络类型不匹配: 期望 {expected}, 实际 {self.kind}. "
                f"这可能导致逻辑错误。请检查函数调用处是否传入了错误的网络类型。"
            )
    
    def assert_invariants(self):
        """断言不变量
        
        - HDKN: ref_year == hist_end_year
        - PDKN: ref_year == max_year（此验证在构建时完成）
        
        Raises:
            ValueError: 如果不变量违反
        """
        if self.kind == "HDKN":
            if self.ref_year != self.hist_end_year:
                raise ValueError(
                    f"HDKN 不变量违反: ref_year ({self.ref_year}) != hist_end_year ({self.hist_end_year}). "
                    f"HDKN 的 ref_year 必须等于 hist_end_year。"
                )
        elif self.kind == "PDKN":
            # PDKN 的 ref_year 应该是 max_year，此验证在构建时完成
            # 这里只检查 ref_year 是否合理（大于等于 hist_end_year）
            if self.ref_year < self.hist_end_year:
                raise ValueError(
                    f"PDKN 不变量违反: ref_year ({self.ref_year}) < hist_end_year ({self.hist_end_year}). "
                    f"PDKN 的 ref_year 应该大于等于 hist_end_year。"
                )
    
    # 代理方法：使 DKNNetwork 可以像 nx.Graph 一样使用
    def number_of_nodes(self):
        """返回节点数"""
        return self.graph.number_of_nodes()
    
    def number_of_edges(self):
        """返回边数"""
        return self.graph.number_of_edges()
    
    def subgraph(self, nodes):
        """返回子图，返回新的 DKNNetwork 对象
        
        Args:
            nodes: 节点列表或集合
            
        Returns:
            DKNNetwork: 新的 DKNNetwork 对象
        """
        subg = self.graph.subgraph(nodes).copy()
        return DKNNetwork(
            kind=self.kind,
            graph=subg,
            ref_year=self.ref_year,
            hist_end_year=self.hist_end_year
        )
    
    def nodes(self, data=False):
        """返回节点迭代器"""
        return self.graph.nodes(data=data)
    
    def edges(self, data=False):
        """返回边迭代器"""
        return self.graph.edges(data=data)
    
    def neighbors(self, node):
        """返回节点的邻居"""
        return self.graph.neighbors(node)
    
    def has_node(self, node):
        """检查节点是否存在"""
        return self.graph.has_node(node)
    
    def has_edge(self, u, v):
        """检查边是否存在"""
        return self.graph.has_edge(u, v)
    
    def __getitem__(self, key):
        """支持 graph[node] 访问节点属性"""
        return self.graph[key]
    
    def __contains__(self, node):
        """支持 'node in graph' 语法"""
        return node in self.graph
    
    def __str__(self):
        """字符串表示"""
        return f"DKNNetwork(kind={self.kind}, ref_year={self.ref_year}, |V|={self.number_of_nodes()}, |E|={self.number_of_edges()})"
    
    def __repr__(self):
        """详细表示"""
        return self.__str__()
