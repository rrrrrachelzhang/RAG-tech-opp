#!/usr/bin/env python3
"""集成测试 - 验证特征提取（基于 G_current 独立构建）"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.patent_opportunity_analysis import pipeline as _main_pipeline
from src.patent_opportunity_analysis.utils.paths import RAW_PATENT_FILE
from src.patent_opportunity_analysis.utils.network_io import load_dkn, load_metadata
from src.patent_opportunity_analysis.utils.run_utils import get_run_dir

load_patents_from_csv = _main_pipeline.load_patents_from_csv
extract_features_for_regression = _main_pipeline.extract_features_for_regression

# 加载专利
patents = load_patents_from_csv(RAW_PATENT_FILE, limit=20, smart_select=False)
logger.info(f"加载了 {len(patents)} 条专利")

# 使用已有 run 的 HDKN
run_dir = get_run_dir("full")
hdkn_path = run_dir / "01_networks" / "hdkn.pkl.gz"
meta_path = run_dir / "01_networks" / "networks_meta.json"
if not hdkn_path.exists():
    raise FileNotFoundError(f"请先运行 Step1 构建网络: {hdkn_path}")
HDKN = load_dkn(hdkn_path)
meta = load_metadata(meta_path)
logger.info(f"HDKN: {HDKN.number_of_nodes()} 节点, {HDKN.number_of_edges()} 边")

# 使用新流程：独立构建 G_current + 与 HDKN 对比计算特征
features = extract_features_for_regression(
    HDKN, None, patents,
    selected_features=['New_n', 'New_e', 'Min_pn', 'Con_n', 'Con_e', 'Eigen', 'Constraint'],
    target_patents=patents[:3],
)

logger.info(f"提取特征数: {len(features)}")
if features:
    f = features[0]
    logger.info(f"首条专利 {f['patent_id']}:")
    logger.info(
        f"  New_n: {f['New_n']}, New_e: {f['New_e']}, Min_pn: {f['Min_pn']}, "
        f"Con_n: {f['Con_n']:.4f}, Con_e: {f['Con_e']:.4f}, "
        f"Eigen: {f['Eigen']:.4f}, Constraint: {f['Constraint']:.4f}, "
        f"Back_cite: {f['Back_cite']}, Assignee: {f['Assignee']}, Total_pat: {f['Total_pat']}"
    )
logger.success("OK")
