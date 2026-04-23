#!/usr/bin/env python3
"""
测试 HDKN 衰减逻辑：验证参考年份正确性

运行方式：
    python test_hdkn_decay.py
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx

# 导入模块
from src.patent_opportunity_analysis import config as _config
from src.patent_opportunity_analysis import dkn_builder as _dkn_builder

HIST_END_YEAR = _config.HIST_END_YEAR
DECAY_FACTOR = _config.DECAY_FACTOR
compute_time_decay_weights = _dkn_builder.compute_time_decay_weights

def test_hdkn_decay_reference_year():
    """测试 HDKN 衰减使用正确的参考年份"""
    
    # 设置测试参数
    test_hist_end_year = 2020  # 测试用的 HIST_END_YEAR
    test_patent_year = 2010    # 测试专利年份
    expected_delta = 10        # 期望的时间差：2020 - 2010 = 10
    alpha = DECAY_FACTOR       # 0.9
    
    # 创建测试图
    test_graph = nx.Graph()
    test_graph.add_node("test_node", patents=set(), years={test_patent_year})
    test_graph.add_node("other_node", patents=set(), years={test_patent_year})
    test_graph.add_edge("test_node", "other_node", patents=set(), years={test_patent_year})
    
    # 计算衰减权重（使用正确的参考年份）
    compute_time_decay_weights(test_graph, total_year=test_hist_end_year, alpha=alpha)
    
    # 验证结果
    node_strength = test_graph.nodes["test_node"]["strength"]
    edge_weight = test_graph.edges[("test_node", "other_node")]["weight"]
    
    expected_strength = alpha ** expected_delta  # 0.9^10 ≈ 0.349
    expected_weight = alpha ** expected_delta
    
    print(f"测试参数:")
    print(f"  HIST_END_YEAR (参考年份): {test_hist_end_year}")
    print(f"  专利年份: {test_patent_year}")
    print(f"  时间差 (Δt): {expected_delta}")
    print(f"  衰减因子 (α): {alpha}")
    print(f"\n计算结果:")
    print(f"  节点强度: {node_strength:.6f}")
    print(f"  边权重: {edge_weight:.6f}")
    print(f"\n期望结果:")
    print(f"  节点强度: {expected_strength:.6f}")
    print(f"  边权重: {expected_weight:.6f}")
    
    # 验证（允许浮点误差）
    tolerance = 1e-6
    assert abs(node_strength - expected_strength) < tolerance, \
        f"节点强度不匹配: 期望 {expected_strength:.6f}, 实际 {node_strength:.6f}"
    assert abs(edge_weight - expected_weight) < tolerance, \
        f"边权重不匹配: 期望 {expected_weight:.6f}, 实际 {edge_weight:.6f}"
    
    print("\n✅ 测试通过：HDKN 衰减使用正确的参考年份")
    
    # 验证如果使用错误的参考年份会得到不同的结果
    wrong_ref_year = 2026  # 错误的参考年份（当前年份）
    wrong_graph = nx.Graph()
    wrong_graph.add_node("test_node", patents=set(), years={test_patent_year})
    wrong_graph.add_node("other_node", patents=set(), years={test_patent_year})
    wrong_graph.add_edge("test_node", "other_node", patents=set(), years={test_patent_year})
    
    compute_time_decay_weights(wrong_graph, total_year=wrong_ref_year, alpha=alpha)
    wrong_strength = wrong_graph.nodes["test_node"]["strength"]
    wrong_delta = wrong_ref_year - test_patent_year  # 16
    wrong_expected = alpha ** wrong_delta  # 0.9^16 ≈ 0.185
    
    print(f"\n对比测试（使用错误的参考年份 {wrong_ref_year}）:")
    print(f"  时间差 (Δt): {wrong_delta}")
    print(f"  节点强度: {wrong_strength:.6f}")
    print(f"  期望结果: {wrong_expected:.6f}")
    print(f"  差异: {abs(wrong_strength - expected_strength):.6f}")
    
    assert abs(wrong_strength - expected_strength) > tolerance, \
        "错误参考年份应该产生不同的结果"
    
    print("✅ 验证通过：错误的参考年份会产生不同的衰减结果")
    
    # 测试验证功能（expected_ref_year）
    print(f"\n测试验证功能（expected_ref_year）:")
    try:
        verify_graph = nx.Graph()
        verify_graph.add_node("test_node", patents=set(), years={test_patent_year})
        verify_graph.add_node("other_node", patents=set(), years={test_patent_year})
        verify_graph.add_edge("test_node", "other_node", patents=set(), years={test_patent_year})
        compute_time_decay_weights(
            verify_graph, 
            total_year=wrong_ref_year, 
            expected_ref_year=test_hist_end_year,
            alpha=alpha
        )
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"  ✅ 正确捕获错误: {e}")
    
    print("\n✅ 验证功能测试通过")

if __name__ == "__main__":
    print("=" * 60)
    print("HDKN 衰减逻辑测试")
    print("=" * 60)
    print(f"\n当前配置:")
    print(f"  HIST_END_YEAR = {HIST_END_YEAR}")
    print(f"  DECAY_FACTOR = {DECAY_FACTOR}")
    print()
    
    test_hdkn_decay_reference_year()
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)
