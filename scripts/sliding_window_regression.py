#!/usr/bin/env python3
"""
具身智能（Embodied AI）领域专利数据滑动窗口切分 & 批处理回归脚本

功能：
1. 读取 patents.csv / patents_test.csv（支持多文件合并）
2. 按年份滑动窗口切分，为多段回归分析做准备
3. 每个窗口分为"历史网络期（前 L-1 年）"和"回归目标年份（第 L 年）"
4. 自动检测样本量，低于阈值时跳过并警告
5. 对每个窗口执行负二项/泊松回归，汇总系数动态变化
6. 输出统计摘要表格与回归结果

运行方式：
    python scripts/sliding_window_regression.py
    python scripts/sliding_window_regression.py --csv data/raw/patents_test.csv
    python scripts/sliding_window_regression.py --csv data/raw/patents.csv data/raw/patents_test.csv
    python scripts/sliding_window_regression.py --year-start 2015 --year-end 2024
    python scripts/sliding_window_regression.py --window-size 5 --step-size 2
    python scripts/sliding_window_regression.py --run-regression --save
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from scripts.common import init_script
init_script()

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

from src.patent_opportunity_analysis.utils.paths import DATA_RAW_DIR, OUTPUTS_DIR

# ---------------------------------------------------------------------------
# 列名映射：真实 CSV 列名 → 统一内部列名
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "patent_id": "ID",
    "app_year": "Year",
    "forward_cites": "Cited",
    "backward_cites": "Back_cite",
    "ipc_classes": "IPC",
    "assignee": "Assignee",
    "title": "Title",
    "abstract": "Abstract",
}

CORE_COLUMNS = ["ID", "Year", "Cited"]

# 回归特征列（可能由上游 pipeline 计算后追加，此处非必需）
FEATURE_COLUMNS = [
    "New_n", "New_e", "Min_pn",
    "Con_n", "Con_e", "Eigen", "Constraint",
]

# ---------------------------------------------------------------------------
# 数据加载与校验
# ---------------------------------------------------------------------------


def load_and_merge(csv_paths: List[Path]) -> pd.DataFrame:
    """加载一个或多个 CSV 并合并，自动映射列名

    Args:
        csv_paths: CSV 文件路径列表

    Returns:
        合并后的 DataFrame（列名已统一）
    """
    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")
        df = pd.read_csv(p)
        logger.info(f"  读取 {p.name}：{len(df)} 条记录")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # 去重（同一 patent_id 可能同时出现在两个文件中）
    id_col = "patent_id" if "patent_id" in merged.columns else "ID"
    before = len(merged)
    merged = merged.drop_duplicates(subset=[id_col], keep="first")
    dupes = before - len(merged)
    if dupes:
        logger.info(f"  去除重复专利 {dupes} 条")

    # 映射列名
    merged = merged.rename(columns=COLUMN_MAP)

    return merged


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """校验核心字段、清洗异常值

    Returns:
        清洗后的 DataFrame

    Raises:
        ValueError: 缺少必需字段
    """
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必需字段: {missing}")

    before = len(df)
    df = df.dropna(subset=["Year", "Cited"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Cited"] = df["Cited"].astype(int)

    # 过滤明显异常的年份（如 app_year=1）
    df = df[df["Year"] >= 1970]

    after = len(df)
    if before != after:
        logger.warning(f"清洗掉 {before - after} 行（NaN / 异常年份）")

    # 检测可用的特征列
    available_feats = [c for c in FEATURE_COLUMNS if c in df.columns]
    if available_feats:
        logger.info(f"检测到回归特征列: {available_feats}")
    else:
        logger.info("未检测到预计算特征列（New_n 等），窗口仅包含原始专利字段")

    year_range = df["Year"].agg(["min", "max"])
    logger.info(
        f"数据概览：{after} 条记录，年份跨度 {year_range['min']}–{year_range['max']}"
    )
    return df


# ---------------------------------------------------------------------------
# 滑动窗口切分
# ---------------------------------------------------------------------------


def build_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 4,
    step_size: int = 1,
    min_samples: int = 30,
    year_start: int = None,
    year_end: int = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """按年份滑动窗口切分数据

    每个窗口的前 (window_size - 1) 年为历史网络构建期，
    最后 1 年为回归目标年份。

    Args:
        df: 经 validate_dataframe 处理后的 DataFrame
        window_size: 窗口长度（年），默认 4
        step_size: 滑动步长（年），默认 1
        min_samples: 最小样本量阈值，低于则跳过该窗口
        year_start: 窗口起始年下限（None 则从数据最小年份开始）
        year_end: 窗口结束年上限（None 则到数据最大年份）

    Returns:
        (windowed_data, summary_df)
    """
    data_year_min = int(df["Year"].min())
    data_year_max = int(df["Year"].max())

    first_start = year_start if year_start is not None else data_year_min
    last_end = year_end if year_end is not None else data_year_max

    if last_end - first_start + 1 < window_size:
        raise ValueError(
            f"指定年份范围 ({first_start}–{last_end}, 共 {last_end - first_start + 1} 年) "
            f"不足以构造长度为 {window_size} 的窗口"
        )

    windowed_data: Dict[str, pd.DataFrame] = {}
    summary_rows: List[dict] = []

    start = first_start
    while start + window_size - 1 <= last_end:
        end = start + window_size - 1
        label = f"{start}-{end}"

        window_df = df[(df["Year"] >= start) & (df["Year"] <= end)].copy()
        n_total = len(window_df)

        hist_years = list(range(start, end))
        target_year = end

        n_hist = int((window_df["Year"] < target_year).sum())
        n_target = int((window_df["Year"] == target_year).sum())
        avg_cited = round(float(window_df["Cited"].mean()), 2) if n_total > 0 else 0.0

        row = {
            "窗口": label,
            "起始年": start,
            "结束年": end,
            "历史期": f"{hist_years[0]}-{hist_years[-1]}" if hist_years else "",
            "目标年": target_year,
            "总样本": n_total,
            "历史期样本": n_hist,
            "目标年样本": n_target,
            "平均被引": avg_cited,
            "状态": "",
        }

        if n_total < min_samples:
            row["状态"] = f"⚠ 跳过（< {min_samples}）"
            logger.warning(f"窗口 [{label}] 样本量 {n_total} < {min_samples}，跳过")
        elif n_target == 0:
            row["状态"] = "⚠ 跳过（目标年无数据）"
            logger.warning(f"窗口 [{label}] 目标年 {target_year} 无数据，跳过")
        else:
            window_df["_window_label"] = label
            window_df["_period"] = window_df["Year"].apply(
                lambda y, t=target_year: "history" if y < t else "target"
            )
            windowed_data[label] = window_df
            row["状态"] = "✓ 已纳入"

        summary_rows.append(row)
        start += step_size

    summary_df = pd.DataFrame(summary_rows)
    return windowed_data, summary_df


# ---------------------------------------------------------------------------
# 摘要输出
# ---------------------------------------------------------------------------


def print_summary(summary_df: pd.DataFrame, windowed_data: Dict[str, pd.DataFrame]):
    """打印格式化的摘要表格"""
    total_windows = len(summary_df)
    active_windows = len(windowed_data)
    skipped = total_windows - active_windows

    logger.info("=" * 90)
    logger.info("滑动窗口切分摘要")
    logger.info("=" * 90)

    col_widths = {
        "窗口": 12, "历史期": 12, "目标年": 6,
        "总样本": 6, "历史期样本": 8, "目标年样本": 8,
        "平均被引": 8, "状态": 20,
    }
    display_cols = list(col_widths.keys())

    header = " | ".join(col.center(col_widths[col]) for col in display_cols)
    separator = "-+-".join("-" * col_widths[col] for col in display_cols)

    print(f"\n{header}")
    print(separator)

    for _, row in summary_df.iterrows():
        cells = []
        for col in display_cols:
            val = str(row.get(col, ""))
            w = col_widths[col]
            cells.append(val.center(w) if col == "状态" else val.rjust(w))
        print(" | ".join(cells))

    print(separator)
    print(
        f"  合计: {total_windows} 个窗口 | "
        f"有效: {active_windows} | 跳过: {skipped}"
    )

    if windowed_data:
        all_target = pd.concat(
            [w[w["_period"] == "target"] for w in windowed_data.values()]
        )
        print(
            f"  有效窗口目标年总样本: {len(all_target)} | "
            f"总体平均被引: {all_target['Cited'].mean():.2f}"
        )
    print()


def save_windows(
    windowed_data: Dict[str, pd.DataFrame],
    output_dir: Path,
):
    """将每个窗口的数据保存为独立 CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, wdf in windowed_data.items():
        path = output_dir / f"window_{label}.csv"
        wdf.to_csv(path, index=False)

    logger.success(f"已保存 {len(windowed_data)} 个窗口 CSV → {output_dir}")


# ---------------------------------------------------------------------------
# 批处理回归分析
# ---------------------------------------------------------------------------

DEFAULT_REGRESSORS = [
    "New_n", "New_e", "Min_pn",
    "Con_n", "Con_e", "Eigen", "Constraint",
]


def _fit_single_window(
    window_label: str,
    df: pd.DataFrame,
    regressors: List[str],
    use_zinb: bool = False,
) -> Dict:
    """对单个窗口执行回归分析

    回退链（use_zinb=True）: ZINB → NB → Poisson
    回退链（use_zinb=False）: NB → Poisson

    Returns:
        包含系数、标准误、p 值、模型类型等字段的字典；
        若回归失败则所有统计量为 NaN。
    """
    from statsmodels.discrete.count_model import (
        ZeroInflatedNegativeBinomialP,
    )

    result_row: Dict = {"window": window_label, "N": len(df), "model_type": None}

    # 初始化所有统计列为 NaN
    for var in regressors:
        result_row[f"{var}_beta"] = np.nan
        result_row[f"{var}_pvalue"] = np.nan
        result_row[f"{var}_stderr"] = np.nan
    result_row["const_beta"] = np.nan
    result_row["const_pvalue"] = np.nan
    result_row["const_stderr"] = np.nan
    result_row["pseudo_r2"] = np.nan
    result_row["zero_ratio"] = np.nan

    # 检查自变量是否存在
    missing_vars = [v for v in regressors if v not in df.columns]
    if missing_vars:
        logger.warning(
            f"[{window_label}] 缺少自变量 {missing_vars}，跳过回归"
        )
        result_row["model_type"] = "SKIP_MISSING_VARS"
        return result_row

    # 样本量预检：至少需要 > 自变量数 + 截距
    min_n = len(regressors) + 1
    if use_zinb:
        min_n = 2 * (len(regressors) + 1)  # ZINB 有两组参数
    if len(df) <= min_n:
        logger.warning(
            f"[{window_label}] 样本量 {len(df)} ≤ 参数数 {min_n}，跳过回归"
        )
        result_row["model_type"] = "SKIP_LOW_N"
        return result_row

    # 因变量预检
    y = df["Cited"].astype(float)
    if y.var() == 0:
        logger.warning(f"[{window_label}] Cited 方差为 0，跳过回归")
        result_row["model_type"] = "SKIP_ZERO_VAR_Y"
        return result_row

    zero_ratio = float((y == 0).sum()) / len(y)
    result_row["zero_ratio"] = round(zero_ratio, 4)

    # 构建设计矩阵
    X = df[regressors].astype(float)

    # 剔除零方差列
    zero_var = [c for c in regressors if X[c].var() == 0]
    if zero_var:
        logger.warning(f"[{window_label}] 零方差自变量 {zero_var}，已从本窗口剔除")
        X = X.drop(columns=zero_var)
    if X.empty:
        logger.warning(f"[{window_label}] 剔除后无可用自变量，跳过回归")
        result_row["model_type"] = "SKIP_NO_VARS"
        return result_row

    X_const = sm.add_constant(X)
    fit_result = None

    # ---------- 阶段 1: 零膨胀负二项 (ZINB) ----------
    if use_zinb and zero_ratio > 0.1:
        try:
            zinb_model = ZeroInflatedNegativeBinomialP(
                y, X_const,
                exog_infl=X_const,  # inflate 部分使用相同自变量
                p=2,
            )
            zinb_result = zinb_model.fit(
                maxiter=200, disp=False, method="bfgs",
            )
            if not zinb_result.mle_retvals["converged"]:
                raise RuntimeError("ZINB 未收敛")

            logger.info(
                f"[{window_label}] ZINB 收敛 | "
                f"LL={zinb_result.llf:.2f}, AIC={zinb_result.aic:.2f}, "
                f"零比例={zero_ratio:.1%}"
            )
            fit_result = zinb_result
            result_row["model_type"] = "ZINB"
            result_row["aic"] = round(zinb_result.aic, 4)
            result_row["bic"] = round(zinb_result.bic, 4)

        except Exception as zinb_err:
            logger.warning(
                f"[{window_label}] ZINB 失败（{zinb_err}），回退到 NB"
            )

    # ---------- 阶段 2: 负二项 (NB) ----------
    if fit_result is None:
        try:
            nb_model = sm.GLM(y, X_const, family=sm.families.NegativeBinomial())
            nb_result = nb_model.fit(maxiter=100, disp=False)

            if not nb_result.converged:
                raise RuntimeError("NB 回归未收敛")

            logger.info(
                f"[{window_label}] NB 回归收敛 | "
                f"Deviance={nb_result.deviance:.2f}, df_resid={nb_result.df_resid}"
            )
            fit_result = nb_result
            result_row["model_type"] = "NegativeBinomial"

        except Exception as nb_err:
            logger.warning(
                f"[{window_label}] NB 回归失败（{nb_err}），回退到 Poisson"
            )

    # ---------- 阶段 3: 泊松 (Poisson) ----------
    if fit_result is None:
        try:
            poi_model = sm.GLM(y, X_const, family=sm.families.Poisson())
            poi_result = poi_model.fit(maxiter=100, disp=False)

            if not poi_result.converged:
                raise RuntimeError("Poisson 回归未收敛")

            logger.info(
                f"[{window_label}] Poisson 回归收敛 | "
                f"Deviance={poi_result.deviance:.2f}, df_resid={poi_result.df_resid}"
            )
            fit_result = poi_result
            result_row["model_type"] = "Poisson"

        except Exception as poi_err:
            logger.error(
                f"[{window_label}] Poisson 也失败（{poi_err}），标记为 NaN"
            )
            result_row["model_type"] = "FAILED"
            return result_row

    # ---------- 提取统计量 ----------
    params = fit_result.params
    pvalues = fit_result.pvalues
    bse = fit_result.bse

    # ZINB 的 params 包含 inflate_ 前缀的膨胀模型参数，只提取计数模型部分
    for var in params.index:
        if str(var).startswith("inflate_"):
            continue
        col_key = var if var != "const" else "const"
        if col_key in regressors or col_key == "const":
            result_row[f"{col_key}_beta"] = float(params[var])
            result_row[f"{col_key}_pvalue"] = float(pvalues[var])
            result_row[f"{col_key}_stderr"] = float(bse[var])

    # Pseudo R²
    if result_row["model_type"] == "ZINB":
        llf = fit_result.llf
        llnull = fit_result.llnull
        if llnull and llnull != 0:
            result_row["pseudo_r2"] = round(1.0 - llf / llnull, 6)
    else:
        if fit_result.null_deviance and fit_result.null_deviance > 0:
            result_row["pseudo_r2"] = round(
                1.0 - fit_result.deviance / fit_result.null_deviance, 6
            )

    return result_row


_VALID_MODEL_TYPES = {"NegativeBinomial", "Poisson", "ZINB"}


def run_batch_regression(
    window_data_dict: Dict[str, pd.DataFrame],
    regressors: Optional[List[str]] = None,
    use_zinb: bool = False,
) -> pd.DataFrame:
    """对所有滑动窗口执行批处理回归分析

    Args:
        window_data_dict: {窗口标签: DataFrame}，由 build_sliding_windows 产出
        regressors: 自变量列表，默认为 7 个方法变量
        use_zinb: 是否启用零膨胀负二项回归（ZINB → NB → Poisson）

    Returns:
        汇总 DataFrame，索引为窗口标签，包含每个自变量的 beta / pvalue / stderr
    """
    if regressors is None:
        regressors = DEFAULT_REGRESSORS

    model_chain = "ZINB → NB → Poisson" if use_zinb else "NB → Poisson"
    logger.info("=" * 90)
    logger.info(
        f"批处理回归分析 | 窗口数={len(window_data_dict)} | "
        f"自变量={regressors} | 模型链={model_chain}"
    )
    logger.info("=" * 90)

    rows: List[Dict] = []
    for label, wdf in window_data_dict.items():
        row = _fit_single_window(label, wdf, regressors, use_zinb=use_zinb)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("window")

    # ---------- 打印汇总 ----------
    logger.info("=" * 90)
    logger.info("批处理回归汇总")
    logger.info("=" * 90)

    model_counts = summary["model_type"].value_counts()
    logger.info(f"模型分布: {model_counts.to_dict()}")

    fitted = summary[summary["model_type"].isin(_VALID_MODEL_TYPES)]
    if not fitted.empty:
        beta_cols = [c for c in fitted.columns if c.endswith("_beta") and c != "const_beta"]
        pval_cols = [c for c in fitted.columns if c.endswith("_pvalue") and c != "const_pvalue"]
        extra_cols = [c for c in ["zero_ratio", "aic", "bic"] if c in fitted.columns]

        print("\n--- 系数 (β) ---")
        print(fitted[["N", "model_type"] + extra_cols + beta_cols].to_string())
        print("\n--- P 值 ---")
        print(fitted[["N", "model_type"] + pval_cols].to_string())

        print("\n--- 显著性标记（p < 0.05 为 *, p < 0.01 为 **, p < 0.001 为 ***）---")
        sig_df = fitted[pval_cols].copy()
        for col in pval_cols:
            sig_df[col] = sig_df[col].apply(_significance_star)
        sig_df.insert(0, "model_type", fitted["model_type"])
        print(sig_df.to_string())
    else:
        logger.warning("所有窗口回归均失败，无汇总结果")

    print()
    return summary


def _significance_star(p: float) -> str:
    """将 p 值转为显著性标记"""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# 可视化：系数趋势图 & 概念漂移分析
# ---------------------------------------------------------------------------

# 特征中文标签（用于图表标题）
_FEATURE_LABELS: Dict[str, str] = {
    "New_n":      "New_n（新节点数）",
    "New_e":      "New_e（新边数）",
    "Min_pn":     "Min_pn（最小专利覆盖）",
    "Con_n":      "Con_n（节点强度中位数）",
    "Con_e":      "Con_e（边权重中位数）",
    "Eigen":      "Eigen（特征向量中心性）",
    "Constraint": "Constraint（结构洞约束）",
}

# 每个子图的配色
_PALETTE = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
    "#9C27B0", "#009688", "#795548",
]


def _parse_end_year(window_label: str) -> int:
    """从窗口标签（如 '2015-2018'）提取结束年份"""
    return int(window_label.split("-")[-1])


def smooth_betas(
    regression_df: pd.DataFrame,
    regressors: Optional[List[str]] = None,
    method: str = "ewma",
    span: int = 3,
) -> pd.DataFrame:
    """对各特征的 beta 序列进行平滑处理

    Args:
        regression_df: run_batch_regression 的返回值
        regressors: 特征列表
        method: 平滑方法，"ewma" 或 "ma"（移动平均）
        span: EWMA span / MA 窗口宽度

    Returns:
        追加了 *_beta_smoothed 列的 DataFrame 副本
    """
    if regressors is None:
        regressors = DEFAULT_REGRESSORS

    df = regression_df.copy()
    for var in regressors:
        col = f"{var}_beta"
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        if method == "ewma":
            df[f"{var}_beta_smoothed"] = series.ewm(span=span, min_periods=1).mean()
        else:
            df[f"{var}_beta_smoothed"] = series.rolling(window=span, min_periods=1).mean()
    return df


def plot_beta_trends(
    regression_df: pd.DataFrame,
    regressors: Optional[List[str]] = None,
    smooth_method: str = "ewma",
    smooth_span: int = 3,
    save_path: Optional[Path] = None,
    figsize: Tuple = None,
) -> None:
    """生成多子图 β 系数趋势图

    Args:
        regression_df: run_batch_regression 的返回值
        regressors: 要绘制的特征列表
        smooth_method: "ewma" 或 "ma"
        smooth_span: 平滑窗口
        save_path: 图片保存路径（None 则显示）
        figsize: 画布大小
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if regressors is None:
        regressors = DEFAULT_REGRESSORS

    # 只保留成功拟合的窗口
    fitted = regression_df[
        regression_df["model_type"].isin(_VALID_MODEL_TYPES)
    ].copy()
    if fitted.empty:
        logger.warning("无成功拟合的窗口，跳过绘图")
        return

    # 检测实际可绘制的特征
    available = [v for v in regressors if f"{v}_beta" in fitted.columns]
    if not available:
        logger.warning("回归结果中无可绘制的特征列，跳过")
        return

    # 平滑
    fitted = smooth_betas(fitted, available, method=smooth_method, span=smooth_span)

    # X 轴：结束年份
    end_years = [_parse_end_year(lbl) for lbl in fitted.index]

    n_features = len(available)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (7 * n_cols, 3.6 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(
        "滑动窗口回归系数 (β) 趋势图 — 概念漂移检测",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for idx, var in enumerate(available):
        ax = axes[idx // n_cols][idx % n_cols]
        color = _PALETTE[idx % len(_PALETTE)]

        beta_raw = fitted[f"{var}_beta"].values
        beta_sm = fitted[f"{var}_beta_smoothed"].values
        pvals = fitted[f"{var}_pvalue"].values

        # 零线
        ax.axhline(0, color="black", linewidth=0.8, zorder=1)

        # 显著区间高亮（p < 0.05）
        for i, (yr, pv) in enumerate(zip(end_years, pvals)):
            if not pd.isna(pv) and pv < 0.05:
                ax.axvspan(
                    yr - 0.4, yr + 0.4,
                    color=color, alpha=0.12, zorder=0,
                )

        # 原始轨迹
        ax.plot(
            end_years, beta_raw,
            color=color, alpha=0.35, linewidth=1, linestyle="--",
            marker="o", markersize=3, label="原始 β",
        )
        # 平滑趋势
        ax.plot(
            end_years, beta_sm,
            color=color, alpha=0.95, linewidth=2.4,
            marker="s", markersize=4, label=f"平滑 β（{smooth_method.upper()}, span={smooth_span}）",
        )

        # 显著点标星
        for yr, bv, pv in zip(end_years, beta_sm, pvals):
            if not pd.isna(pv) and pv < 0.05:
                ax.annotate(
                    _significance_star(pv),
                    xy=(yr, bv), xytext=(0, 6),
                    textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color=color,
                )

        label_text = _FEATURE_LABELS.get(var, var)
        ax.set_title(label_text, fontsize=11, fontweight="bold")
        ax.set_xlabel("窗口结束年份")
        ax.set_ylabel("β 系数")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7, loc="best")
        ax.grid(axis="y", alpha=0.3)

    # 隐藏多余子图
    for idx in range(n_features, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.success(f"趋势图已保存 → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def analyze_concept_drift(
    regression_df: pd.DataFrame,
    regressors: Optional[List[str]] = None,
    smooth_method: str = "ewma",
    smooth_span: int = 3,
    tail_n: int = 3,
) -> Dict[str, List[str]]:
    """语义分析：检测升温 / 饱和因子

    检测逻辑：
    - 取最近 tail_n 个窗口的平滑 β
    - 计算最近 tail_n 个窗口的线性斜率
    - 斜率 > 0 且末端 p < 0.1 → 升温因子
    - 斜率 < 0 或末端 p > 0.3 → 过时/饱和因子
    - 其余 → 稳定因子

    Returns:
        {"rising": [...], "declining": [...], "stable": [...]}
    """
    if regressors is None:
        regressors = DEFAULT_REGRESSORS

    fitted = regression_df[
        regression_df["model_type"].isin(_VALID_MODEL_TYPES)
    ].copy()
    if fitted.empty:
        logger.warning("无成功拟合的窗口，无法进行漂移分析")
        return {"rising": [], "declining": [], "stable": []}

    fitted = smooth_betas(fitted, regressors, method=smooth_method, span=smooth_span)

    tail = fitted.tail(tail_n)
    result: Dict[str, List[str]] = {"rising": [], "declining": [], "stable": []}

    logger.info("=" * 90)
    logger.info("概念漂移分析（基于最近 {} 个窗口）".format(tail_n))
    logger.info("=" * 90)

    for var in regressors:
        beta_col = f"{var}_beta_smoothed"
        pval_col = f"{var}_pvalue"
        if beta_col not in tail.columns:
            continue

        betas = tail[beta_col].dropna()
        if len(betas) < 2:
            result["stable"].append(var)
            continue

        # 线性斜率（最小二乘）
        x = np.arange(len(betas), dtype=float)
        slope = np.polyfit(x, betas.values, 1)[0]

        last_p = tail[pval_col].iloc[-1] if pval_col in tail.columns else np.nan
        last_beta = betas.iloc[-1]

        if slope > 0 and (not pd.isna(last_p) and last_p < 0.1):
            category = "rising"
            tag = "🔺 升温"
        elif slope < 0 or (not pd.isna(last_p) and last_p > 0.3):
            category = "declining"
            tag = "🔻 饱和/过时"
        else:
            category = "stable"
            tag = "➡️  稳定"

        result[category].append(var)
        logger.info(
            f"  {tag} {var:>12s} | 斜率={slope:+.4f} | "
            f"末端β={last_beta:+.4f} | 末端p={last_p:.4f}"
        )

    # 汇总
    print()
    if result["rising"]:
        logger.success(f"升温技术驱动因子: {', '.join(result['rising'])}")
    if result["declining"]:
        logger.warning(f"过时/饱和因子:     {', '.join(result['declining'])}")
    if result["stable"]:
        logger.info(f"稳定因子:          {', '.join(result['stable'])}")
    print()

    return result


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="具身智能领域专利数据滑动窗口切分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/sliding_window_regression.py
  python scripts/sliding_window_regression.py --csv data/raw/patents_test.csv
  python scripts/sliding_window_regression.py --csv data/raw/patents.csv data/raw/patents_test.csv
  python scripts/sliding_window_regression.py --year-start 2015 --year-end 2024
  python scripts/sliding_window_regression.py --window-size 5 --step-size 2 --save
        """,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=None,
        help="专利数据 CSV 路径（可指定多个，默认: patents.csv + patents_test.csv）",
    )
    parser.add_argument(
        "--window-size", "-L",
        type=int,
        default=4,
        help="窗口长度（年），默认 4",
    )
    parser.add_argument(
        "--step-size", "-S",
        type=int,
        default=1,
        help="滑动步长（年），默认 1",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="窗口最小样本量阈值，低于则跳过（默认 30）",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2015,
        help="窗口起始年下限（默认 2015，具身智能领域建议）",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2024,
        help="窗口结束年上限（默认 2024）",
    )
    parser.add_argument(
        "--run-regression",
        action="store_true",
        help="对每个窗口执行批处理回归分析（需要特征列 New_n 等）",
    )
    parser.add_argument(
        "--zinb",
        action="store_true",
        help="启用零膨胀负二项回归（ZINB → NB → Poisson 回退链）",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="生成 β 系数趋势图并进行概念漂移分析（需配合 --run-regression）",
    )
    parser.add_argument(
        "--smooth-method",
        choices=["ewma", "ma"],
        default="ewma",
        help="β 平滑方法：ewma（指数加权）或 ma（简单移动平均），默认 ewma",
    )
    parser.add_argument(
        "--smooth-span",
        type=int,
        default=3,
        help="平滑窗口宽度 / EWMA span（默认 3）",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="将窗口数据、回归结果及趋势图保存到磁盘",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="保存目录（默认: outputs/sliding_windows）",
    )

    args = parser.parse_args()

    # ---- 确定数据来源 ----
    csv_paths: List[Path] = args.csv or [
        DATA_RAW_DIR / "patents.csv",
        DATA_RAW_DIR / "patents_test.csv",
    ]

    logger.info(f"加载数据文件（共 {len(csv_paths)} 个）：")
    df = load_and_merge(csv_paths)
    df = validate_dataframe(df)

    # ---- 按年份范围过滤 ----
    before_filter = len(df)
    df = df[(df["Year"] >= args.year_start) & (df["Year"] <= args.year_end)]
    logger.info(
        f"年份过滤 [{args.year_start}, {args.year_end}]：{before_filter} → {len(df)} 条"
    )

    # ---- 滑动窗口切分 ----
    logger.info(
        f"窗口参数: L={args.window_size}, S={args.step_size}, "
        f"min_samples={args.min_samples}"
    )
    windowed_data, summary_df = build_sliding_windows(
        df,
        window_size=args.window_size,
        step_size=args.step_size,
        min_samples=args.min_samples,
        year_start=args.year_start,
        year_end=args.year_end,
    )

    # ---- 打印窗口摘要 ----
    print_summary(summary_df, windowed_data)

    # ---- 批处理回归 ----
    regression_df = None
    if args.run_regression:
        regression_df = run_batch_regression(
            windowed_data, use_zinb=args.zinb,
        )

    # ---- 趋势图 & 漂移分析 ----
    drift_result = None
    if args.plot and regression_df is not None:
        out_dir = args.output_dir or (OUTPUTS_DIR / "sliding_windows")
        plot_path = out_dir / "beta_trends.png" if args.save else None

        plot_beta_trends(
            regression_df,
            smooth_method=args.smooth_method,
            smooth_span=args.smooth_span,
            save_path=plot_path,
        )
        drift_result = analyze_concept_drift(
            regression_df,
            smooth_method=args.smooth_method,
            smooth_span=args.smooth_span,
        )
    elif args.plot and regression_df is None:
        logger.warning("--plot 需配合 --run-regression 使用")

    # ---- 可选：保存 ----
    if args.save:
        out_dir = args.output_dir or (OUTPUTS_DIR / "sliding_windows")
        save_windows(windowed_data, out_dir)

        summary_path = out_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.success(f"窗口摘要已保存 → {summary_path}")

        if regression_df is not None:
            reg_path = out_dir / "regression_results.csv"
            regression_df.to_csv(reg_path)
            logger.success(f"回归结果已保存 → {reg_path}")

    return windowed_data, summary_df, regression_df


if __name__ == "__main__":
    windowed_data, summary_df, regression_df = main()
