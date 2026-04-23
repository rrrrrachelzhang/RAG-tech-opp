"""
共享回归流程：特征提取 + NB/ZINB 双模型拟合

供 Alpha Selection 与 Step2 共用，保证 Log-Likelihood 一致。
"""

from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from loguru import logger

from . import pipeline as _pipeline
from . import regression_model as _regression_model

extract_features_for_regression = _pipeline.extract_features_for_regression
build_reg_df = _regression_model.build_reg_df
fit_both_nb_zinb = _regression_model.fit_both_nb_zinb


def run_regression_flow(
    HDKN: Any,
    empty_pdkn: Any,
    patents: List[Any],
    selected_features: List[str],
    include_control_vars: bool = True,
    decay_factor: Optional[float] = None,
    force_rebuild_hdkn_stats: bool = False,
) -> Tuple[pd.DataFrame, Any, Any, Optional[dict], List[str]]:
    """
    共享特征提取与回归流程：提取特征、构建 df、拟合 NB 与 ZINB。

    Args:
        HDKN: 已应用时间衰减权重的 HDKN（调用方需先 compute_time_decay_weights）
        empty_pdkn: 空 PDKN 占位符
        patents: 专利列表
        selected_features: 自变量列表
        include_control_vars: 是否包含控制变量
        decay_factor: 衰减因子 α（用于缓存键）
        force_rebuild_hdkn_stats: 是否强制重建 HDKN 统计缓存

    Returns:
        (df, nb_result, zinb_result, vuong_result, used_features)
    """
    # 固定随机种子，保证特征提取与回归的可复现性（同一 α 多次运行 LL 一致）
    np.random.seed(42)
    features = extract_features_for_regression(
        HDKN,
        empty_pdkn,
        patents,
        selected_features=selected_features,
        decay_factor=decay_factor,
        force_rebuild_hdkn_stats=force_rebuild_hdkn_stats,
    )
    df = build_reg_df(features)
    nb_result, zinb_result, vuong_result, used_features = fit_both_nb_zinb(
        df,
        selected_features=selected_features,
        include_control_vars=include_control_vars,
    )
    return df, nb_result, zinb_result, vuong_result, used_features
