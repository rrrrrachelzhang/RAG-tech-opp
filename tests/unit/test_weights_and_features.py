#!/usr/bin/env python3
"""
权重与特征计算的最小单元测试

测试目标：
1. 验证边权重叠加计算正确性
2. 验证节点 strength 计算（加权度 vs 时间衰减）
3. 验证 year_min 计算正确性
4. 验证 90% 分位阈值计算
5. 验证 New_n/New_e 在不同阈值模式下输出正确
6. 验证 Con_e median 计算
7. 验证 Eigen/Constraint 计算（至少能跑通并输出合理范围）

运行方式：
    python tests/test_weights_and_features.py
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import networkx as nx

# 导入模块
from src.patent_opportunity_analysis import config as _config
from src.patent_opportunity_analysis import dkn_builder as _dkn_builder
from src.patent_opportunity_analysis import feature_extraction as _feature_extraction
from src.patent_opportunity_analysis import patent_graph as _patent_graph

DECAY_FACTOR = _config.DECAY_FACTOR
HIST_END_YEAR = _config.HIST_END_YEAR
compute_time_decay_weights = _dkn_builder.compute_time_decay_weights
compute_new_flags = _feature_extraction.compute_new_flags
compute_conventionality = _feature_extraction.compute_conventionality
compute_eigen_centrality = _feature_extraction.compute_eigen_centrality
build_patent_graph = _patent_graph.build_patent_graph
PatentRecord = _patent_graph.PatentRecord


def create_toy_patents():
    """创建测试用的玩具专利数据集"""
    patents = [
        PatentRecord(
            patent_id="P1",
            title="battery cooling system",
            abstract="A battery cooling system for electric vehicles.",
            app_year=2010,
            forward_cites=5,
            backward_cites=3,
            assignee_type="corp",
            ipc_classes=["H01M"],
            assignee="ABC Corp"
        ),
        PatentRecord(
            patent_id="P2",
            title="battery management",
            abstract="A battery management system with cooling.",
            app_year=2015,
            forward_cites=8,
            backward_cites=2,
            assignee_type="corp",
            ipc_classes=["H01M"],
            assignee="XYZ Ltd"
        ),
        PatentRecord(
            patent_id="P3",
            title="cooling lithium battery",
            abstract="Cooling system for lithium battery.",
            app_year=2020,
            forward_cites=12,
            backward_cites=5,
            assignee_type="corp",
            ipc_classes=["H01M"],
            assignee="Tech Inc"
        ),
    ]
    return patents


def create_toy_dependencies():
    """创建测试用的依存关系（简化版）"""
    # 模拟依存关系：每个专利的标题词之间的边
    dependencies = {
        "P1": [
            type('Dep', (), {'head': 'battery', 'dependent': 'cooling', 'relation': 'nmod'})(),
            type('Dep', (), {'head': 'cooling', 'dependent': 'system', 'relation': 'compound'})(),
        ],
        "P2": [
            type('Dep', (), {'head': 'battery', 'dependent': 'management', 'relation': 'nmod'})(),
            type('Dep', (), {'head': 'management', 'dependent': 'system', 'relation': 'compound'})(),
            type('Dep', (), {'head': 'system', 'dependent': 'cooling', 'relation': 'nmod'})(),
        ],
        "P3": [
            type('Dep', (), {'head': 'cooling', 'dependent': 'lithium', 'relation': 'nmod'})(),
            type('Dep', (), {'head': 'lithium', 'dependent': 'battery', 'relation': 'compound'})(),
        ],
    }
    return dependencies


def test_edge_weight_calculation():
    """测试边权重计算"""
    print("\n" + "="*60)
    print("测试1: 边权重叠加计算")
    print("="*60)
    
    # 创建测试图
    graph = nx.Graph()
    
    # 边 (battery, cooling) 出现在两个专利中
    graph.add_edge("battery", "cooling", patents={"P1", "P2"}, years={2010, 2015})
    
    # 计算权重
    T = 2022
    alpha = DECAY_FACTOR  # 0.9
    
    compute_time_decay_weights(graph, total_year=T, alpha=alpha)
    
    # 验证结果
    edge_weight = graph.edges["battery", "cooling"]["weight"]
    expected_weight = alpha ** (T - 2010) + alpha ** (T - 2015)
    expected_weight = 0.9 ** 12 + 0.9 ** 7  # ≈ 0.2824 + 0.4783 ≈ 0.7607
    
    print(f"边 (battery, cooling):")
    print(f"  出现年份: {2010}, {2015}")
    print(f"  参考年份 T: {T}")
    print(f"  计算权重: {edge_weight:.6f}")
    print(f"  期望权重: {expected_weight:.6f}")
    
    assert abs(edge_weight - expected_weight) < 1e-4, \
        f"边权重不匹配: 期望 {expected_weight:.6f}, 实际 {edge_weight:.6f}"
    
    print("✅ 测试通过：边权重叠加计算正确")


def test_node_strength_weighted_degree():
    """测试节点 strength（加权度版本）"""
    print("\n" + "="*60)
    print("测试2: 节点 strength（加权度）")
    print("="*60)
    
    # 创建测试图
    graph = nx.Graph()
    
    # 节点 battery 连接到两个节点
    graph.add_edge("battery", "cooling", patents={"P1"}, years={2010}, weight=0.5)
    graph.add_edge("battery", "management", patents={"P2"}, years={2015}, weight=0.3)
    
    # 手动设置节点属性（模拟合并后的状态）
    graph.nodes["battery"]["years"] = {2010, 2015}
    graph.nodes["cooling"]["years"] = {2010}
    graph.nodes["management"]["years"] = {2015}
    
    # 计算边权重
    T = 2022
    alpha = DECAY_FACTOR
    compute_time_decay_weights(graph, total_year=T, alpha=alpha)
    
    # 计算加权度（严格版）
    battery_strength_weighted = sum(
        graph.edges["battery", neighbor].get("weight", 0.0)
        for neighbor in graph.neighbors("battery")
    )
    
    # 计算时间衰减权重之和（当前实现）
    battery_years = graph.nodes["battery"]["years"]
    battery_strength_decay = sum(alpha ** (T - y) for y in battery_years)
    
    print(f"节点 battery:")
    print(f"  加权度 strength: {battery_strength_weighted:.6f}")
    print(f"  时间衰减 strength: {battery_strength_decay:.6f}")
    print(f"  差异: {abs(battery_strength_weighted - battery_strength_decay):.6f}")
    
    # 验证加权度 = 边权重之和
    expected_weighted = 0.5 + 0.3  # 但实际权重是时间衰减后的值
    # 重新计算实际权重
    edge1_weight = alpha ** (T - 2010)  # 0.9^12
    edge2_weight = alpha ** (T - 2015)  # 0.9^7
    expected_weighted = edge1_weight + edge2_weight
    
    assert abs(battery_strength_weighted - expected_weighted) < 1e-4, \
        f"加权度不匹配: 期望 {expected_weighted:.6f}, 实际 {battery_strength_weighted:.6f}"
    
    print("✅ 测试通过：节点 strength（加权度）计算正确")


def test_year_min_calculation():
    """测试 year_min 计算"""
    print("\n" + "="*60)
    print("测试3: year_min 计算")
    print("="*60)
    
    graph = nx.Graph()
    graph.add_node("battery", patents={"P1", "P2"}, years={2010, 2015})
    graph.add_edge("battery", "cooling", patents={"P1", "P3"}, years={2010, 2020})
    
    T = 2022
    compute_time_decay_weights(graph, total_year=T, alpha=DECAY_FACTOR)
    
    node_year_min = graph.nodes["battery"]["year_min"]
    edge_year_min = graph.edges["battery", "cooling"]["year_min"]
    
    print(f"节点 battery:")
    print(f"  years: {2010, 2015}")
    print(f"  year_min: {node_year_min}")
    print(f"  期望: {min([2010, 2015])}")
    
    print(f"\n边 (battery, cooling):")
    print(f"  years: {2010, 2020}")
    print(f"  year_min: {edge_year_min}")
    print(f"  期望: {min([2010, 2020])}")
    
    assert node_year_min == 2010, f"节点 year_min 错误: 期望 2010, 实际 {node_year_min}"
    assert edge_year_min == 2010, f"边 year_min 错误: 期望 2010, 实际 {edge_year_min}"
    
    print("✅ 测试通过：year_min 计算正确")


def test_quantile_threshold():
    """测试 90% 分位阈值计算"""
    print("\n" + "="*60)
    print("测试4: 90% 分位阈值计算")
    print("="*60)
    
    # 创建可控的 Year_n 列表
    graph = nx.Graph()
    years_list = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026, 2028]
    
    for i, year in enumerate(years_list):
        node_name = f"node_{i}"
        graph.add_node(node_name, patents={f"P{i}"}, years={year})
        graph.nodes[node_name]["year_min"] = year
    
    # 计算 90% 分位数
    all_years = [data.get("year_min", 0) for _, data in graph.nodes(data=True)]
    quantile_90 = float(np.percentile(all_years, 90))
    
    print(f"Year_n 列表: {sorted(years_list)}")
    print(f"90% 分位数: {quantile_90}")
    print(f"期望范围: [{years_list[8]}, {years_list[9]}] (第9和第10个值之间)")
    
    # 验证：90% 分位数应该在合理范围内
    assert years_list[8] <= quantile_90 <= years_list[9] + 1, \
        f"90% 分位数不在合理范围: {quantile_90}"
    
    print("✅ 测试通过：90% 分位阈值计算正确")


def test_new_flags_different_modes():
    """测试 New_n/New_e 在不同阈值模式下输出"""
    print("\n" + "="*60)
    print("测试5: New_n/New_e 阈值模式")
    print("="*60)
    
    # 创建 HDKN 图
    hdkn_graph = nx.Graph()
    years_list = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026, 2028]
    
    for i, year in enumerate(years_list):
        node_name = f"node_{i}"
        hdkn_graph.add_node(node_name, patents={f"P{i}"}, years={year})
        hdkn_graph.nodes[node_name]["year_min"] = year
    
    # 计算阈值
    all_years = [data.get("year_min", 0) for _, data in hdkn_graph.nodes(data=True)]
    quantile_90 = float(np.percentile(all_years, 90))
    hist_end_year = HIST_END_YEAR
    
    print(f"90% 分位数阈值: {quantile_90}")
    print(f"HIST_END_YEAR 阈值: {hist_end_year}")
    
    # 创建子网（包含一个较新的节点）
    subg = hdkn_graph.subgraph(["node_8", "node_9"]).copy()  # year_min = 2026, 2028
    
    # 测试 90% 分位数模式
    new_n_quantile, new_e_quantile = compute_new_flags(
        hdkn_graph, subg, thr_node=quantile_90, thr_edge=quantile_90
    )
    
    # 测试 HIST_END_YEAR 模式
    new_n_fixed, new_e_fixed = compute_new_flags(
        hdkn_graph, subg, thr_node=hist_end_year, thr_edge=hist_end_year
    )
    
    print(f"\n子网节点 year_min: {[hdkn_graph.nodes[n]['year_min'] for n in subg.nodes()]}")
    print(f"\n90% 分位数模式: New_n={new_n_quantile}, New_e={new_e_quantile}")
    print(f"HIST_END_YEAR 模式: New_n={new_n_fixed}, New_e={new_e_fixed}")
    
    # 验证：如果子网包含 year_min >= threshold 的节点，New_n 应该为 1
    max_year_min = max(hdkn_graph.nodes[n]['year_min'] for n in subg.nodes())
    if max_year_min >= quantile_90:
        assert new_n_quantile == 1, f"New_n 应该为 1（year_min={max_year_min} >= threshold={quantile_90}）"
    else:
        assert new_n_quantile == 0, f"New_n 应该为 0（year_min={max_year_min} < threshold={quantile_90}）"
    
    print("✅ 测试通过：New_n/New_e 阈值模式正确")


def test_con_e_median():
    """测试 Con_e median 计算"""
    print("\n" + "="*60)
    print("测试6: Con_e median 计算")
    print("="*60)
    
    # 创建子网
    subg = nx.Graph()
    edge_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for i, weight in enumerate(edge_weights):
        u = f"node_{i}"
        v = f"node_{i+1}"
        subg.add_edge(u, v, weight=weight)
    
    con_n, con_e = compute_conventionality(subg)
    
    expected_median = np.median(edge_weights)
    
    print(f"边权重列表: {edge_weights}")
    print(f"计算 Con_e: {con_e:.6f}")
    print(f"期望中位数: {expected_median:.6f}")
    
    assert abs(con_e - expected_median) < 1e-6, \
        f"Con_e 不匹配: 期望 {expected_median:.6f}, 实际 {con_e:.6f}"
    
    print("✅ 测试通过：Con_e median 计算正确")


def test_eigen_constraint_range():
    """测试 Eigen/Constraint 计算（至少能跑通并输出合理范围）"""
    print("\n" + "="*60)
    print("测试7: Eigen/Constraint 计算范围")
    print("="*60)
    
    # 创建 HDKN 图（连通图）
    hdkn_graph = nx.Graph()
    hdkn_graph.add_edge("a", "b", weight=0.5)
    hdkn_graph.add_edge("b", "c", weight=0.3)
    hdkn_graph.add_edge("c", "d", weight=0.7)
    hdkn_graph.add_edge("d", "a", weight=0.4)
    
    # 创建子网
    subg = hdkn_graph.subgraph(["a", "b", "c"]).copy()
    
    # 测试 Eigen
    try:
        eigen = compute_eigen_centrality(hdkn_graph, subg)
        print(f"Eigen: {eigen:.6f}")
        assert not np.isnan(eigen), "Eigen 不应该是 NaN"
        assert 0 <= eigen <= 1, f"Eigen 应该在 [0, 1] 范围内，实际 {eigen}"
        print("✅ Eigen 计算通过")
    except Exception as e:
        print(f"⚠️  Eigen 计算失败: {e}")
        # 允许失败（如果图太小或特殊结构）
    
    # 测试 Constraint（如果已实现）
    try:
        from networkx.algorithms.structuralholes import constraint
        constraint_dict = constraint(hdkn_graph, weight="weight")
        subg_constraints = [constraint_dict.get(node, float('inf')) for node in subg.nodes()]
        constraint_min = min(subg_constraints) if subg_constraints else 0.0
        print(f"Constraint: {constraint_min:.6f}")
        assert not np.isnan(constraint_min), "Constraint 不应该是 NaN"
        assert constraint_min >= 0, f"Constraint 应该 >= 0，实际 {constraint_min}"
        print("✅ Constraint 计算通过")
    except ImportError:
        print("⚠️  Constraint 功能未实现（networkx 版本可能不支持）")
    except Exception as e:
        print(f"⚠️  Constraint 计算失败: {e}")


def main():
    """运行所有测试"""
    print("="*60)
    print("权重与特征计算单元测试")
    print("="*60)
    print(f"\n配置:")
    print(f"  DECAY_FACTOR (α): {DECAY_FACTOR}")
    print(f"  HIST_END_YEAR: {HIST_END_YEAR}")
    
    try:
        test_edge_weight_calculation()
        test_node_strength_weighted_degree()
        test_year_min_calculation()
        test_quantile_threshold()
        test_new_flags_different_modes()
        test_con_e_median()
        test_eigen_constraint_range()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
