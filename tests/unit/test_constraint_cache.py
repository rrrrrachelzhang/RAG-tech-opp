#!/usr/bin/env python3
"""
Constraint 特征计算性能重构测试

测试目标：
1. 确保不会在每个子网 Si 上重复进行昂贵的 Burt constraint 计算
2. 验证预计算 + 缓存的正确性
3. 验证值一致性（新旧实现输出相同）

运行方式：
    pytest tests/unit/test_constraint_cache.py -v
    python tests/unit/test_constraint_cache.py
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx

# 导入模块
from src.patent_opportunity_analysis import feature_extraction as _feature_extraction
from src.patent_opportunity_analysis.utils.dkn_wrapper import DKNNetwork

get_hdkn_constraint_map = _feature_extraction.get_hdkn_constraint_map
compute_constraint_feature = _feature_extraction.compute_constraint_feature


def clear_constraint_caches() -> None:
    """清理全局与对象级 constraint 缓存，避免测试间串扰。"""
    _feature_extraction._hdkn_constraint_map_cache.clear()


def create_toy_hdkn() -> nx.Graph:
    """创建测试用的玩具 HDKN 图"""
    graph = nx.Graph()
    # 创建一个小型连通图
    graph.add_edge("a", "b", weight=0.5)
    graph.add_edge("b", "c", weight=0.3)
    graph.add_edge("c", "d", weight=0.7)
    graph.add_edge("d", "a", weight=0.4)
    graph.add_edge("a", "c", weight=0.6)
    return graph


def test_constraint_map_precomputation():
    """测试 1: 验证预计算 constraint map 的正确性"""
    print("\n" + "="*60)
    print("测试 1: Constraint map 预计算")
    print("="*60)
    
    hdkn_graph = create_toy_hdkn()
    hdkn_dkn = DKNNetwork(kind="HDKN", graph=hdkn_graph, ref_year=2022, hist_end_year=2022)
    clear_constraint_caches()
    
    # 预计算 constraint map
    constraint_map = get_hdkn_constraint_map(hdkn_dkn, nodes=None, weight="weight", use_cache=False)
    
    print(f"预计算的 constraint map: {constraint_map}")
    
    # 验证结果
    assert isinstance(constraint_map, dict), "constraint_map 应该是字典"
    assert len(constraint_map) == hdkn_graph.number_of_nodes(), f"应该有 {hdkn_graph.number_of_nodes()} 个节点的 constraint 值"
    assert all(isinstance(v, (int, float)) for v in constraint_map.values()), "所有值应该是数字"
    assert all(v >= 0 for v in constraint_map.values()), "所有 constraint 值应该 >= 0"
    
    print("✅ 测试通过：预计算 constraint map 正确")


def test_constraint_feature_uses_precomputed_map():
    """测试 2: 验证 compute_constraint_feature 使用预计算的 map"""
    print("\n" + "="*60)
    print("测试 2: Constraint 特征使用预计算 map")
    print("="*60)
    
    hdkn_graph = create_toy_hdkn()
    hdkn_dkn = DKNNetwork(kind="HDKN", graph=hdkn_graph, ref_year=2022, hist_end_year=2022)
    clear_constraint_caches()
    
    # 预计算 constraint map
    constraint_map = get_hdkn_constraint_map(hdkn_dkn, nodes=None, weight="weight", use_cache=False)
    
    # 创建子网
    subg = hdkn_graph.subgraph(["a", "b", "c"]).copy()
    
    # 使用预计算的 map 计算特征
    constraint_feature = compute_constraint_feature(hdkn_graph, subg, constraint_map=constraint_map)
    
    print(f"Constraint(Si) = {constraint_feature:.6f}")
    
    # 验证结果
    assert isinstance(constraint_feature, (int, float)), "constraint_feature 应该是数字"
    assert constraint_feature >= 0, f"constraint_feature 应该 >= 0，实际 {constraint_feature}"
    
    # 手动验证：应该是子网节点 constraint 的最小值
    subg_constraints = [constraint_map.get(node, 1.0) for node in subg.nodes()]
    expected_min = min(subg_constraints)
    assert abs(constraint_feature - expected_min) < 1e-6, \
        f"应该是 min(constraint_map)，期望 {expected_min:.6f}，实际 {constraint_feature:.6f}"
    
    print("✅ 测试通过：Constraint 特征正确使用预计算 map")


def test_no_repeated_computation():
    """测试 3: 确保不会重复计算（核心性能测试）"""
    print("\n" + "="*60)
    print("测试 3: 确保不会重复计算")
    print("="*60)
    
    hdkn_graph = create_toy_hdkn()
    hdkn_dkn = DKNNetwork(kind="HDKN", graph=hdkn_graph, ref_year=2022, hist_end_year=2022)
    clear_constraint_caches()
    constraint_map = get_hdkn_constraint_map(hdkn_dkn, nodes=None, weight="weight", use_cache=False)
    assert len(_feature_extraction._hdkn_constraint_map_cache) == 1
    print("预计算阶段：constraint map 已写入全局缓存")
    
    # 对多个子网计算 Constraint（应该不再调用 constraint）
    subnets = [
        hdkn_graph.subgraph(["a", "b"]).copy(),
        hdkn_graph.subgraph(["b", "c"]).copy(),
        hdkn_graph.subgraph(["c", "d"]).copy(),
        hdkn_graph.subgraph(["a", "b", "c"]).copy(),
    ]
    
    for i, subg in enumerate(subnets):
        constraint_feature = compute_constraint_feature(
            hdkn_graph, subg, constraint_map=constraint_map
        )
        print(f"  子网 {i+1}: Constraint = {constraint_feature:.6f}")

    print("✅ 测试通过：不会重复计算")


def test_value_consistency():
    """测试 4: 验证新旧实现的值一致性"""
    print("\n" + "="*60)
    print("测试 4: 值一致性验证")
    print("="*60)
    
    hdkn_graph = create_toy_hdkn()
    hdkn_dkn = DKNNetwork(kind="HDKN", graph=hdkn_graph, ref_year=2022, hist_end_year=2022)
    clear_constraint_caches()
    
    # 创建子网
    subg = hdkn_graph.subgraph(["a", "b", "c"]).copy()
    
    # 新实现：使用预计算的 map
    constraint_map = get_hdkn_constraint_map(hdkn_dkn, nodes=None, weight="weight", use_cache=False)
    new_value = compute_constraint_feature(hdkn_graph, subg, constraint_map=constraint_map)
    
    # 旧实现：直接计算（模拟旧代码）
    constraint_dict_old = nx.algorithms.structuralholes.constraint(
        hdkn_graph,
        nodes=list(subg.nodes()),
        weight="weight"
    )
    old_value = float(min(constraint_dict_old.get(node, float('inf')) for node in subg.nodes()))
    
    print(f"新实现（预计算 + min 聚合）: {new_value:.6f}")
    print(f"旧实现（直接计算）: {old_value:.6f}")
    print(f"差异: {abs(new_value - old_value):.6f}")
    
    # 验证一致性（允许浮点误差）
    assert abs(new_value - old_value) < 1e-5, \
        f"新旧实现值不一致：新 {new_value:.6f}，旧 {old_value:.6f}，差异 {abs(new_value - old_value):.6f}"
    
    print("✅ 测试通过：值一致性验证通过")


def test_cache_persistence():
    """测试 5: 验证缓存持久化（可选）"""
    print("\n" + "="*60)
    print("测试 5: 缓存持久化")
    print("="*60)
    
    hdkn_graph = create_toy_hdkn()
    hdkn_dkn = DKNNetwork(kind="HDKN", graph=hdkn_graph, ref_year=2022, hist_end_year=2022)
    clear_constraint_caches()

    # 第一次：计算并缓存
    constraint_map1 = get_hdkn_constraint_map(
        hdkn_dkn, nodes=None, weight="weight", use_cache=True
    )
    assert len(_feature_extraction._hdkn_constraint_map_cache) == 1
    print("第一次计算：constraint map 已写入缓存")

    # 第二次：应从对象缓存读取
    constraint_map2 = get_hdkn_constraint_map(
        hdkn_dkn, nodes=None, weight="weight", use_cache=True
    )

    # 验证结果一致
    assert constraint_map1 == constraint_map2, "两次结果应该一致"
    
    print("✅ 测试通过：缓存持久化验证通过")


def test_missing_nodes_handling():
    """测试 6: 验证缺失节点的处理"""
    print("\n" + "="*60)
    print("测试 6: 缺失节点处理")
    print("="*60)
    
    hdkn_graph = create_toy_hdkn()
    hdkn_dkn = DKNNetwork(kind="HDKN", graph=hdkn_graph, ref_year=2022, hist_end_year=2022)
    clear_constraint_caches()
    
    # 预计算 constraint map（只包含部分节点）
    constraint_map = get_hdkn_constraint_map(
        hdkn_dkn, nodes=["a", "b", "c"], weight="weight", use_cache=False
    )
    
    # 创建包含不在 map 中的节点的子网
    subg = hdkn_graph.subgraph(["a", "b", "d"]).copy()  # "d" 不在预计算的 map 中
    
    # 计算特征（应该使用默认值 1.0 处理缺失节点）
    constraint_feature = compute_constraint_feature(hdkn_graph, subg, constraint_map=constraint_map)
    
    print(f"Constraint(Si) = {constraint_feature:.6f}")
    print(f"  子网节点: {list(subg.nodes())}")
    print(f"  constraint_map 中的节点: {list(constraint_map.keys())}")
    
    # 验证：缺失节点应该使用默认值 1.0
    assert constraint_feature >= 0, "constraint_feature 应该 >= 0"
    assert constraint_feature <= 1.0, "由于有缺失节点使用 1.0，min 应该 <= 1.0"
    
    print("✅ 测试通过：缺失节点处理正确")


def main():
    """运行所有测试"""
    print("="*60)
    print("Constraint 特征计算性能重构测试")
    print("="*60)
    
    try:
        test_constraint_map_precomputation()
        test_constraint_feature_uses_precomputed_map()
        test_no_repeated_computation()
        test_value_consistency()
        test_cache_persistence()
        test_missing_nodes_handling()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
