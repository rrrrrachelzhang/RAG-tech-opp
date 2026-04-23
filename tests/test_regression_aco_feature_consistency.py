#!/usr/bin/env python3
"""
回归子网特征路径与 ACO 包装路径一致性：二者须调用同一 compute_features_for_subnetwork。

使用手工构造的微型 hdkn_cache（不依赖磁盘 HDKN 统计缓存），断言
「直接回归接口」与 compute_subnetwork_features_for_aco 在相同参数下数值一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.patent_opportunity_analysis.feature_registry import (  # noqa: E402
    _register_feature_functions,
    compute_features_for_subnetwork,
)
from src.patent_opportunity_analysis.hdkn_feature_cache import (  # noqa: E402
    compute_subnetwork_features_for_aco,
)
from src.patent_opportunity_analysis.utils.dkn_wrapper import DKNNetwork  # noqa: E402


def _minimal_hdkn_cache(hist_end_year: int = 2020) -> tuple[nx.Graph, dict, DKNNetwork]:
    """与 feature_registry 各 _compute_* 兼容的最小图 + cache。"""
    g = nx.Graph()
    g.add_node("A", patents={"pa"}, year_min=2000)
    g.add_node("B", patents={"pb"}, year_min=2000)
    g.add_node("C", patents={"pc_new"}, year_min=2022)
    g.add_edge("A", "B", patents={"eab"}, year_min=2000, weight=0.8)
    g.add_edge("B", "C", patents={"ebc"}, year_min=2022, weight=0.5)

    ek_ab = tuple(sorted(("A", "B")))
    ek_bc = tuple(sorted(("B", "C")))

    cache = {
        "hdkn_graph": g,
        "hist_end_year": hist_end_year,
        "p90_year_n": 2015.0,
        "p90_year_e": 2015.0,
        "min_pn_mode": "greedy",
        "_strength_dict": {"A": 1.0, "B": 2.0, "C": 0.5},
        "_eigen_dict": {"A": 0.1, "B": 0.2, "C": 0.05},
        "_constraint_dict": {"A": 0.4, "B": 0.5, "C": 0.6},
        "_betweenness_dict": {"A": 0.0, "B": 0.01, "C": 0.0},
        "_year_min_node_dict": {"A": 2000, "B": 2000, "C": 2022},
        "_weight_dict": {ek_ab: 0.8, ek_bc: 0.5},
        "_year_min_edge_dict": {ek_ab: 2000, ek_bc: 2022},
        "_node_patents": {n: set(g.nodes[n].get("patents", set())) for n in g.nodes()},
        "_edge_patents": {
            ek_ab: set(g.edges["A", "B"].get("patents", set())),
            ek_bc: set(g.edges["B", "C"].get("patents", set())),
        },
    }
    hdkn = DKNNetwork(kind="HDKN", graph=g, ref_year=hist_end_year, hist_end_year=hist_end_year)
    return g, cache, hdkn


@pytest.mark.parametrize("current_patent_id", [None, "debug_patent_x"])
@pytest.mark.parametrize("ref_year", [2021, 2023])
def test_regression_path_matches_aco_wrapper(current_patent_id: str | None, ref_year: int) -> None:
    _register_feature_functions()
    g, cache, hdkn = _minimal_hdkn_cache(hist_end_year=2020)
    subg = g.subgraph(["A", "B", "C"]).copy()
    feats = [
        "New_n",
        "New_e",
        "Min_pn",
        "Con_n",
        "Con_e",
        "Eigen",
        "Constraint",
        "Betweenness",
        "Avg_Weight",
    ]

    direct = compute_features_for_subnetwork(
        G_current=subg,
        hdkn_cache=cache,
        selected_features=feats,
        current_patent_id=current_patent_id,
        current_year=ref_year,
    )
    via_aco = compute_subnetwork_features_for_aco(
        subg,
        hdkn,
        ref_year=ref_year,
        hdkn_cache=cache,
        selected_features=feats,
        current_patent_id=current_patent_id,
    )

    assert set(direct.keys()) == set(via_aco.keys())
    for k in feats:
        a, b = direct[k], via_aco[k]
        if isinstance(a, float) or isinstance(b, float):
            assert np.isnan(a) and np.isnan(b) or float(a) == pytest.approx(float(b), rel=1e-9, abs=1e-9)
        else:
            assert a == b
