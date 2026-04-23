#!/usr/bin/env python3
"""
测试 HDKN/PDKN 全链路语义一致性

运行方式：
    python test_hdkn_pdkn_consistency.py
"""
import sys
from pathlib import Path
import importlib
import networkx as nx

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入模块
from src.patent_opportunity_analysis import config as _config
from src.patent_opportunity_analysis import dkn_builder as _dkn_builder
from src.patent_opportunity_analysis import patent_graph as _patent_graph

HIST_END_YEAR = _config.HIST_END_YEAR
build_dkns = _dkn_builder.build_dkns
PatentRecord = _patent_graph.PatentRecord

def test_hdkn_data_truncation():
    """测试 HDKN 数据截断正确性"""
    print("\n" + "=" * 60)
    print("测试 1: HDKN 数据截断正确性")
    print("=" * 60)
    
    # 创建测试数据
    patents = [
        PatentRecord("P1", "Title 1", "Abstract 1", 2020, 0, 0, "", []),
        PatentRecord("P2", "Title 2", "Abstract 2", 2021, 0, 0, "", []),
        PatentRecord("P3", "Title 3", "Abstract 3", 2022, 0, 0, "", []),
        PatentRecord("P4", "Title 4", "Abstract 4", 2023, 0, 0, "", []),  # 超过 HIST_END_YEAR
        PatentRecord("P5", "Title 5", "Abstract 5", 2024, 0, 0, "", []),  # 超过 HIST_END_YEAR
    ]
    
    test_hist_end_year = 2022
    
    # 构建 DKN
    hdkn, pdkn = build_dkns(patents, test_hist_end_year)
    
    # 验证 HDKN 只包含 <= HIST_END_YEAR 的专利
    hdkn_graph = hdkn.graph if hasattr(hdkn, 'graph') else hdkn
    pdkn_graph = pdkn.graph if hasattr(pdkn, 'graph') else pdkn
    
    # 检查节点中的年份
    hdkn_years = set()
    for node, data in hdkn_graph.nodes(data=True):
        hdkn_years.update(data.get("years", set()))
    
    pdkn_years = set()
    for node, data in pdkn_graph.nodes(data=True):
        pdkn_years.update(data.get("years", set()))
    
    print(f"\n测试参数:")
    print(f"  HIST_END_YEAR: {test_hist_end_year}")
    print(f"  专利年份: {[p.app_year for p in patents]}")
    
    print(f"\nHDKN 中的年份: {sorted(hdkn_years)}")
    print(f"PDKN 中的年份: {sorted(pdkn_years)}")
    
    # 验证
    assert all(y <= test_hist_end_year for y in hdkn_years), \
        f"HDKN 包含超过 {test_hist_end_year} 的年份: {[y for y in hdkn_years if y > test_hist_end_year]}"
    
    assert hdkn.ref_year == test_hist_end_year, \
        f"HDKN ref_year 不匹配: 期望 {test_hist_end_year}, 实际 {hdkn.ref_year}"
    
    assert pdkn.ref_year == max(p.app_year for p in patents), \
        f"PDKN ref_year 不匹配: 期望 {max(p.app_year for p in patents)}, 实际 {pdkn.ref_year}"
    
    print("\n✅ 测试通过：HDKN 数据截断正确")

def test_hdkn_decay_reference_year():
    """测试 HDKN 衰减参考年份"""
    print("\n" + "=" * 60)
    print("测试 2: HDKN 衰减参考年份")
    print("=" * 60)
    
    # 创建测试数据（确保 max_year >= hist_end_year）
    patents = [
        PatentRecord("P1", "Test", "Test", 2020, 0, 0, "", []),
        PatentRecord("P2", "Test", "Test", 2023, 0, 0, "", []),  # 确保 max_year >= hist_end_year
    ]
    
    test_hist_end_year = 2022
    
    hdkn, pdkn = build_dkns(patents, test_hist_end_year)
    
    # 验证 ref_year
    assert hdkn.ref_year == test_hist_end_year, \
        f"HDKN ref_year 不匹配: 期望 {test_hist_end_year}, 实际 {hdkn.ref_year}"
    
    # 验证不变量
    hdkn.assert_invariants()
    pdkn.assert_invariants()
    
    print(f"\nHDKN ref_year: {hdkn.ref_year}")
    print(f"HIST_END_YEAR: {test_hist_end_year}")
    print("\n✅ 测试通过：HDKN 衰减参考年份正确")

def test_hdkn_only_function_rejects_pdkn():
    """测试 HDKN-only 函数拒绝 PDKN"""
    print("\n" + "=" * 60)
    print("测试 3: HDKN-only 函数拒绝 PDKN")
    print("=" * 60)
    
    # 创建测试数据（确保 max_year >= hist_end_year）
    patents = [
        PatentRecord("P1", "Test", "Test", 2020, 0, 0, "", []),
        PatentRecord("P2", "Test", "Test", 2023, 0, 0, "", []),  # 确保 max_year >= hist_end_year
    ]
    
    test_hist_end_year = 2022
    hdkn, pdkn = build_dkns(patents, test_hist_end_year)
    
    # 测试 assert_kind
    try:
        hdkn.assert_kind("HDKN")
        print("\n✅ HDKN.assert_kind('HDKN') 通过")
    except ValueError as e:
        assert False, f"HDKN 应该通过 assert_kind('HDKN'): {e}"
    
    try:
        pdkn.assert_kind("HDKN")
        assert False, "PDKN 应该拒绝 assert_kind('HDKN')"
    except ValueError as e:
        print(f"✅ PDKN 正确拒绝 assert_kind('HDKN'): {e}")
    
    print("\n✅ 测试通过：HDKN-only 函数正确拒绝 PDKN")

if __name__ == "__main__":
    print("=" * 60)
    print("HDKN/PDKN 全链路语义一致性测试")
    print("=" * 60)
    print(f"\n当前配置:")
    print(f"  HIST_END_YEAR = {HIST_END_YEAR}")
    
    try:
        test_hdkn_data_truncation()
        test_hdkn_decay_reference_year()
        test_hdkn_only_function_rejects_pdkn()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
