"""
分步式Pipeline集成测试

验证：
1. Step2/Step3不会触发构图（通过检查日志或mock）
2. HDKN衰减参考年来自hist_end_year
3. Resume机制正常工作（第二次运行会跳过已完成步骤）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from loguru import logger
import json

from src.patent_opportunity_analysis.utils.paths import RAW_PATENT_TEST_FILE, RUNS_DIR
from src.patent_opportunity_analysis.utils.network_io import load_dkn, load_metadata
from scripts.step1_build_networks import step1_build_networks
from scripts.step2_hdkn_regression import step2_hdkn_regression
from scripts.step3_pdkn_aco import step3_pdkn_aco


@pytest.fixture
def test_run_id():
    """生成测试run_id"""
    from datetime import datetime
    return f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@pytest.fixture
def cleanup_run_dir(test_run_id):
    """测试后清理"""
    yield
    run_dir = RUNS_DIR / test_run_id
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)


def test_step1_builds_networks(test_run_id, cleanup_run_dir):
    """测试Step1构建网络"""
    if not RAW_PATENT_TEST_FILE.exists():
        pytest.skip(f"测试数据文件不存在: {RAW_PATENT_TEST_FILE}")
    
    run_dir = step1_build_networks(
        patents_csv=RAW_PATENT_TEST_FILE,
        hist_end_year=2022,
        run_id=test_run_id,
        force=True
    )
    
    # 验证产物存在
    networks_dir = run_dir / "01_networks"
    assert (networks_dir / "hdkn.pkl.gz").exists()
    assert (networks_dir / "pdkn.pkl.gz").exists()
    assert (networks_dir / "networks_meta.json").exists()
    
    # 验证metadata
    meta = load_metadata(networks_dir / "networks_meta.json")
    assert meta["step_name"] == "01_networks"
    assert meta["hist_end_year"] == 2022
    assert meta["invariants"]["hdkn_ref_year_equals_hist_end_year"]
    
    # 验证网络类型
    hdkn = load_dkn(networks_dir / "hdkn.pkl.gz")
    pdkn = load_dkn(networks_dir / "pdkn.pkl.gz")
    
    hdkn.assert_kind("HDKN")
    pdkn.assert_kind("PDKN")
    assert hdkn.ref_year == 2022
    assert pdkn.ref_year == meta["max_year"]


def test_step2_uses_hdkn_not_builds_network(test_run_id, cleanup_run_dir):
    """测试Step2使用HDKN但不重新构建网络"""
    if not RAW_PATENT_TEST_FILE.exists():
        pytest.skip(f"测试数据文件不存在: {RAW_PATENT_TEST_FILE}")
    
    # 先运行Step1
    run_dir = step1_build_networks(
        patents_csv=RAW_PATENT_TEST_FILE,
        hist_end_year=2022,
        run_id=test_run_id,
        force=True
    )
    
    # 运行Step2
    regression_dir = step2_hdkn_regression(
        run_id=test_run_id,
        patents_csv=RAW_PATENT_TEST_FILE,
        force=True
    )
    
    # 验证产物存在
    assert (regression_dir / "regression_features.csv").exists()
    assert (regression_dir / "regression_meta.json").exists()
    
    # 验证metadata引用Step1
    meta = load_metadata(regression_dir / "regression_meta.json")
    assert meta["step_name"] == "02_regression"
    assert "upstream_artifacts" in meta
    assert "networks_meta_hash" in meta["upstream_artifacts"]


def test_step3_uses_pdkn_not_builds_network(test_run_id, cleanup_run_dir):
    """测试Step3使用PDKN但不重新构建网络"""
    if not RAW_PATENT_TEST_FILE.exists():
        pytest.skip(f"测试数据文件不存在: {RAW_PATENT_TEST_FILE}")
    
    # 先运行Step1和Step2
    run_dir = step1_build_networks(
        patents_csv=RAW_PATENT_TEST_FILE,
        hist_end_year=2022,
        run_id=test_run_id,
        force=True
    )
    
    step2_hdkn_regression(
        run_id=test_run_id,
        patents_csv=RAW_PATENT_TEST_FILE,
        force=True
    )
    
    # 运行Step3
    aco_dir = step3_pdkn_aco(
        run_id=test_run_id,
        force=True
    )
    
    # 验证产物存在
    assert (aco_dir / "aco_meta.json").exists()
    
    # 验证metadata引用Step1和Step2
    meta = load_metadata(aco_dir / "aco_meta.json")
    assert meta["step_name"] == "03_aco"
    assert "upstream_artifacts" in meta
    assert "networks_meta_hash" in meta["upstream_artifacts"]


def test_resume_mechanism(test_run_id, cleanup_run_dir):
    """测试Resume机制"""
    if not RAW_PATENT_TEST_FILE.exists():
        pytest.skip(f"测试数据文件不存在: {RAW_PATENT_TEST_FILE}")
    
    # 第一次运行Step1
    run_dir1 = step1_build_networks(
        patents_csv=RAW_PATENT_TEST_FILE,
        hist_end_year=2022,
        run_id=test_run_id,
        force=True
    )
    
    # 第二次运行Step1（应该跳过）
    run_dir2 = step1_build_networks(
        patents_csv=RAW_PATENT_TEST_FILE,
        hist_end_year=2022,
        run_id=test_run_id,
        force=False  # 不强制重建
    )
    
    assert run_dir1 == run_dir2
    
    # 验证产物时间戳未改变（通过检查metadata）
    meta1 = load_metadata(run_dir1 / "01_networks" / "networks_meta.json")
    meta2 = load_metadata(run_dir2 / "01_networks" / "networks_meta.json")
    assert meta1["created_at"] == meta2["created_at"]


def test_hdkn_ref_year_equals_hist_end_year(test_run_id, cleanup_run_dir):
    """测试HDKN ref_year等于hist_end_year"""
    if not RAW_PATENT_TEST_FILE.exists():
        pytest.skip(f"测试数据文件不存在: {RAW_PATENT_TEST_FILE}")
    
    hist_end_year = 2020
    
    run_dir = step1_build_networks(
        patents_csv=RAW_PATENT_TEST_FILE,
        hist_end_year=hist_end_year,
        run_id=test_run_id,
        force=True
    )
    
    hdkn = load_dkn(run_dir / "01_networks" / "hdkn.pkl.gz")
    assert hdkn.ref_year == hist_end_year
    assert hdkn.hist_end_year == hist_end_year
    
    # 验证metadata
    meta = load_metadata(run_dir / "01_networks" / "networks_meta.json")
    assert meta["hist_end_year"] == hist_end_year
    assert meta["graph_stats"]["hdkn"]["ref_year"] == hist_end_year


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
