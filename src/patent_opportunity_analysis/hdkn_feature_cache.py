# src/patent_opportunity_analysis/hdkn_feature_cache.py
"""
HDKN 子网特征缓存构建与 ACO/回归共用接口。

回归侧（pipeline.extract_features_for_regression）与 ACO 侧必须使用同一套
compute_features_for_subnetwork + hdkn_cache，以保证特征命名与尺度一致。
NB 回归在原始（未标准化）特征上拟合，见 regression_model.fit_regression。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx
from loguru import logger

from . import config as _config
from .feature_registry import (
    FEATURE_REGISTRY,
    _register_feature_functions,
    compute_features_for_subnetwork,
)
from .hdkn_stats import build_or_load_hdkn_stats


def build_hdkn_subnetwork_feature_cache(
    HDKN: Any,
    *,
    hist_end_year: Optional[int] = None,
    decay_factor: Optional[float] = None,
    selected_features: Optional[List[str]] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    构建与 pipeline 回归特征提取一致的 hdkn_cache（原始尺度统计）。

    Args:
        HDKN: DKNNetwork 或 nx.Graph（HDKN）
        hist_end_year: 历史截止年；默认取自 HDKN.hist_end_year 或 config
        decay_factor: 时间衰减 α；默认 config.DECAY_FACTOR
        selected_features: 需要参与子网特征计算的变量名；决定 skip_betweenness 等
        force_rebuild: 是否强制重建 HDKN 统计缓存

    Returns:
        可直接传入 compute_features_for_subnetwork 的 hdkn_cache 字典
    """
    _register_feature_functions()

    hdkn_graph = HDKN.graph if hasattr(HDKN, "graph") else HDKN
    if hist_end_year is None:
        if hasattr(HDKN, "hist_end_year"):
            hist_end_year = int(HDKN.hist_end_year)
        else:
            hist_end_year = int(getattr(_config, "HIST_END_YEAR", 2022))

    if selected_features is None:
        selected_features = list(FEATURE_REGISTRY.keys())

    _need_betweenness = "Betweenness" in selected_features
    _need_eigen = "Eigen" in selected_features
    _need_constraint = "Constraint" in selected_features

    _decay = decay_factor if decay_factor is not None else getattr(_config, "DECAY_FACTOR", 0.9)
    ref_year = hist_end_year
    config_dict = {
        "hdkn_end_year": hist_end_year,
        "decay_ref_year": ref_year,
        "decay_factor": _decay,
        "node_strength_mode": getattr(_config, "NODE_STRENGTH_MODE", "weighted_degree"),
        "skip_betweenness": not _need_betweenness,
        "skip_eigen": not _need_eigen,
        "skip_constraint": not _need_constraint,
    }

    logger.debug(
        f"build_hdkn_subnetwork_feature_cache: hist_end_year={hist_end_year}, "
        f"decay_factor={_decay}, selected_features={selected_features}"
    )
    node_stats_df, edge_stats_df, auxiliary_dicts = build_or_load_hdkn_stats(
        HDKN,
        config=config_dict,
        force_rebuild=force_rebuild,
    )

    hdkn_cache: Dict[str, Any] = {
        "hdkn_graph": hdkn_graph,
        "hist_end_year": hist_end_year,
        "node_stats_df": node_stats_df,
        "edge_stats_df": edge_stats_df,
        "p90_year_n": auxiliary_dicts["p90_year_n"],
        "p90_year_e": auxiliary_dicts["p90_year_e"],
        "min_pn_mode": getattr(_config, "MIN_PN_MODE", "greedy"),
        "_strength_dict": node_stats_df["strength"].to_dict(),
        "_eigen_dict": node_stats_df["eigen"].to_dict(),
        "_constraint_dict": node_stats_df["constraint"].to_dict(),
        "_betweenness_dict": node_stats_df["betweenness"].to_dict(),
        "_year_min_node_dict": node_stats_df["year_min_node"].to_dict(),
        "_weight_dict": edge_stats_df["weight"].to_dict(),
        "_year_min_edge_dict": edge_stats_df["year_min_edge"].to_dict(),
        "_node_patents": {
            node: hdkn_graph.nodes[node].get("patents", set()) for node in hdkn_graph.nodes()
        },
        "_edge_patents": {
            tuple(sorted([u, v])): data.get("patents", set())
            for u, v, data in hdkn_graph.edges(data=True)
        },
    }
    return hdkn_cache


def compute_subnetwork_features_for_aco(
    subg: nx.Graph,
    HDKN: Any,
    *,
    ref_year: int,
    hdkn_cache: Dict[str, Any],
    selected_features: Optional[List[str]] = None,
    current_patent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    ACO 侧子网特征：与回归侧相同入口 compute_features_for_subnetwork。

    ref_year 应使用 PDKN.ref_year（机会观测年），与回归样本专利 app_year 语义对齐。

    Args:
        subg: PDKN 上诱导子图（节点/边属性与 HDKN 对齐）
        HDKN: 仅用于类型校验时可由调用方 assert；本函数不强制使用
        ref_year: 当前观测年（PDKN.ref_year）
        hdkn_cache: build_hdkn_subnetwork_feature_cache 的产物
        selected_features: 要计算的特征名列表；None 表示注册表全部
        current_patent_id: 与回归单专利样本对齐时可传入；ACO 通常为 None

    Returns:
        特征名 -> 原始尺度数值
    """
    _register_feature_functions()
    if hasattr(HDKN, "assert_kind"):
        HDKN.assert_kind("HDKN")

    if selected_features is None:
        selected_features = list(FEATURE_REGISTRY.keys())

    return compute_features_for_subnetwork(
        G_current=subg,
        hdkn_cache=hdkn_cache,
        selected_features=selected_features,
        current_patent_id=current_patent_id,
        current_year=ref_year,
    )
