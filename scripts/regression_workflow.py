#!/usr/bin/env python3
"""
回归分析工作流 (4-Run Process)

标准工作流：
  Run 1: Alpha Selection — 包含所有特征的 NB/ZINB，对候选 α 逐一拟合
  Run 2: 全模型 — 取最优 α，包含所有特征的 NB/ZINB，并输出共线性报告
  (人工审查共线性报告，选定 Run 3/4 使用的变量)
  Run 3: 选定变量 + 控制变量 — NB/ZINB
  Run 4: 仅选定变量 — NB/ZINB（无控制变量）
  combined: 合并 Run 2/3/4 的结果为统一模型比较表

所有数值输出保留至少 4 位小数。

使用方式：
    python scripts/regression_workflow.py --run-id <ID> --run 1
    python scripts/regression_workflow.py --run-id <ID> --run 2 [--alpha 0.95]
    python scripts/regression_workflow.py --run-id <ID> --run 3 --vars "New_e,Con_e,Constraint"
    python scripts/regression_workflow.py --run-id <ID> --run 4 --vars "New_e,Con_e,Constraint"
    python scripts/regression_workflow.py --run-id <ID> --run combined --vars "New_e,Con_e,Constraint"
"""

import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from loguru import logger

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from scripts.common import init_script
init_script()

from src.patent_opportunity_analysis import config as _config
from src.patent_opportunity_analysis import pipeline as _pipeline
from src.patent_opportunity_analysis import regression_flow as _regression_flow
from src.patent_opportunity_analysis import regression_model as _regression_model
from src.patent_opportunity_analysis import dkn_builder as _dkn_builder
from src.patent_opportunity_analysis.utils.paths import RAW_PATENT_FILE
from src.patent_opportunity_analysis.utils.network_io import (
    load_dkn, load_metadata, save_metadata,
)
from src.patent_opportunity_analysis.utils.run_utils import (
    get_run_dir, ensure_run_dirs,
)
from src.patent_opportunity_analysis.utils.regression_analysis import (
    compute_vuong_test,
    generate_combined_regression_report,
)
import statsmodels.api as sm

# ============================================================
# 常量
# ============================================================

ALL_FEATURES: List[str] = [
    "New_n", "New_e", "Min_pn", "Con_n", "Con_e", "Eigen", "Constraint",
]
CONTROL_VARS: List[str] = ["Back_cite", "Assignee", "Total_pat"]
DEFAULT_ALPHAS: List[float] = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
DP = 4  # 最小小数位数


# ============================================================
# 格式化工具
# ============================================================

def fmt(value: Any, decimals: int = DP) -> str:
    """格式化数值，保留至少 decimals 位小数"""
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(v) or np.isinf(v):
        return ""
    return f"{v:.{decimals}f}"


def fmt_sci(value: Any) -> str:
    """紧凑科学计数法: 4.60e-2, 1.21e2"""
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(v) or np.isinf(v):
        return ""
    if v == 0:
        return "0"
    s = f"{v:.2e}"
    mantissa, exp_part = s.split("e")
    exp_int = int(exp_part)
    return f"{mantissa}e{exp_int}"


def fmt_coef(coef: Any, pvalue: Any, decimals: int = DP) -> str:
    """系数 + 显著性星号: -182.8400***"""
    if coef is None:
        return ""
    try:
        c = float(coef)
    except (TypeError, ValueError):
        return ""
    if np.isnan(c):
        return ""
    stars = ""
    try:
        pv = float(pvalue) if pvalue is not None else None
    except (TypeError, ValueError):
        pv = None
    if pv is not None and np.isfinite(pv):
        if pv < 0.001:
            stars = "***"
        elif pv < 0.01:
            stars = "**"
        elif pv < 0.05:
            stars = "*"
    return f"{c:.{decimals}f}{stars}"


def _pseudo_r2(result: Any) -> Optional[float]:
    """McFadden's pseudo R² = 1 − LL_model / LL_null"""
    llf = getattr(result, "llf", None)
    llnull = getattr(result, "llnull", None)
    if llf is not None and llnull is not None and llnull != 0:
        return 1.0 - llf / llnull
    pr = getattr(result, "prsquared", None)
    return float(pr) if pr is not None else None


def _converged(result: Any) -> bool:
    """检查模型是否收敛"""
    retvals = getattr(result, "mle_retvals", None)
    if retvals and isinstance(retvals, dict):
        return retvals.get("converged", True)
    return getattr(result, "converged", True)


def _get_coefs(
    result: Any, var_names: List[str]
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """从拟合结果中提取指定变量的系数和 p 值"""
    params = getattr(result, "params", None)
    if params is None:
        return {v: (None, None) for v in var_names}
    try:
        pvalues = result.pvalues
    except (ValueError, np.linalg.LinAlgError, AttributeError):
        pvalues = None
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for var in var_names:
        coef: Optional[float] = None
        pval: Optional[float] = None
        if hasattr(params, "index") and var in params.index:
            coef = float(params[var])
            if pvalues is not None and hasattr(pvalues, "index") and var in pvalues.index:
                raw = pvalues[var]
                pval = float(raw) if raw is not None and np.isfinite(float(raw)) else None
        out[var] = (coef, pval)
    return out


# ============================================================
# 输出表格
# ============================================================

def _write_and_print(text: str, path: Optional[Path], title: str) -> str:
    """打印表格到控制台并（可选）保存文件"""
    banner = "=" * 80
    print(f"\n{banner}")
    print(f"  {title}")
    print(f"{banner}\n")
    print(text)
    print()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.info(f"💾 {title} 已保存: {path}")
    return text


def table_alpha_selection(
    results: List[Dict], alphas: List[float], output_file: Optional[Path] = None
) -> str:
    """
    生成 Alpha Selection 表格（Tab 分隔，可直接粘贴到 Word/Excel）。

    格式:
        模型  指标          α=0.8   α=0.85  ...
        NB    Log-Likelihood  ...
              是否收敛        ...
        ZINB  Log-Likelihood  ...
              是否收敛        ...
    """
    alpha_headers = [f"α={a}" for a in alphas]
    header = "模型\t指标\t" + "\t".join(alpha_headers)

    def _lookup(alpha: float, key: str, fallback: str = "—") -> str:
        r = next((x for x in results if abs(x["alpha"] - alpha) < 1e-9), None)
        if r is None or r.get(key) is None:
            return fallback
        return str(r[key])

    nb_ll = [fmt(next((r.get("ll_nb") for r in results if abs(r["alpha"] - a) < 1e-9), None)) or "—" for a in alphas]
    nb_cv = ["✓" if _lookup(a, "conv_nb") == "True" else "✗" for a in alphas]
    zinb_ll = [fmt(next((r.get("ll_zinb") for r in results if abs(r["alpha"] - a) < 1e-9), None)) or "—" for a in alphas]
    zinb_cv = ["✓" if _lookup(a, "conv_zinb") == "True" else "✗" for a in alphas]

    lines = [
        header,
        "NB\tLog-Likelihood\t" + "\t".join(nb_ll),
        "\t是否收敛\t" + "\t".join(nb_cv),
        "ZINB\tLog-Likelihood\t" + "\t".join(zinb_ll),
        "\t是否收敛\t" + "\t".join(zinb_cv),
    ]
    return _write_and_print("\n".join(lines), output_file, "Alpha Selection 结果")


def table_collinearity(
    df: pd.DataFrame, all_vars: List[str], output_file: Optional[Path] = None
) -> str:
    """
    生成共线性报告表格：描述性统计 + 皮尔逊相关矩阵（下三角）+ VIF。
    """
    lines: List[str] = []
    lines.append("\t" + "\t".join(all_vars))

    for label, method in [("最小值", "min"), ("最大值", "max"), ("均值", "mean"), ("方差", "var")]:
        vals: List[str] = []
        for var in all_vars:
            v = getattr(df[var], method)()
            vals.append(fmt_sci(v) if label == "方差" else fmt(v))
        lines.append(f"{label}\t" + "\t".join(vals))

    corr = df[all_vars].corr(method="pearson")
    for i, var_i in enumerate(all_vars):
        row: List[str] = []
        for j in range(len(all_vars)):
            if j < i:
                row.append(fmt(corr.iloc[i, j]))
            elif j == i:
                row.append("1.0000")
            else:
                row.append("")
        lines.append(f"{var_i}\t" + "\t".join(row))

    X = df[all_vars].values.astype(float)
    vif_vals: List[float] = []
    for i in range(X.shape[1]):
        y_col = X[:, i]
        X_other = np.delete(X, i, axis=1)
        X_other = sm.add_constant(X_other)
        try:
            r_sq = sm.OLS(y_col, X_other).fit().rsquared
            vif_vals.append(1.0 / (1.0 - r_sq) if r_sq < 1.0 else np.inf)
        except (np.linalg.LinAlgError, ValueError):
            vif_vals.append(np.nan)
    lines.append("VIF\t" + "\t".join(fmt(v) for v in vif_vals))

    return _write_and_print("\n".join(lines), output_file, "共线性报告")


def table_model_results(
    model_pairs: List[Tuple[str, Any, Any, Optional[dict]]],
    all_row_vars: List[str],
    output_file: Optional[Path] = None,
) -> str:
    """
    生成模型比较表格。

    model_pairs: [(label, nb_result, zinb_result, vuong_result), ...]
    all_row_vars: 所有要展示的变量名（自变量 + 控制变量）

    输出格式:
            模型1-NB  模型2-ZINB  模型3-NB  模型4-ZINB  ...
        New_n  0.1600    0.1600
        ...
        Log-Likelihood  ...
        Pseudo R2       ...
        AIC             ...
        Vuong非嵌套模型检验  ...
    """
    n_pairs = len(model_pairs)
    col_headers: List[str] = []
    for idx in range(n_pairs):
        col_headers.append(f"模型{idx * 2 + 1}-NB")
        col_headers.append(f"模型{idx * 2 + 2}-ZINB")
    header = "\t" + "\t".join(col_headers)

    lines: List[str] = [header]

    for var in all_row_vars:
        row_vals: List[str] = []
        for _, nb_res, zinb_res, _ in model_pairs:
            nb_c = _get_coefs(nb_res, [var])
            c, p = nb_c.get(var, (None, None))
            row_vals.append(fmt_coef(c, p) if c is not None else "")
            zinb_c = _get_coefs(zinb_res, [var])
            c, p = zinb_c.get(var, (None, None))
            row_vals.append(fmt_coef(c, p) if c is not None else "")
        lines.append(f"{var}\t" + "\t".join(row_vals))

    lines.append("\t" + "\t".join([""] * len(col_headers)))

    # Log-Likelihood
    ll_vals: List[str] = []
    for _, nb_res, zinb_res, _ in model_pairs:
        ll_vals.append(fmt(getattr(nb_res, "llf", None)))
        ll_vals.append(fmt(getattr(zinb_res, "llf", None)))
    lines.append("Log-Likelihood\t" + "\t".join(ll_vals))

    # Pseudo R²
    r2_vals: List[str] = []
    for _, nb_res, zinb_res, _ in model_pairs:
        r2_vals.append(fmt(_pseudo_r2(nb_res)))
        r2_vals.append(fmt(_pseudo_r2(zinb_res)))
    lines.append("Pseudo R2\t" + "\t".join(r2_vals))

    # AIC
    aic_vals: List[str] = []
    for _, nb_res, zinb_res, _ in model_pairs:
        aic_vals.append(fmt(getattr(nb_res, "aic", None)))
        aic_vals.append(fmt(getattr(zinb_res, "aic", None)))
    lines.append("AIC\t" + "\t".join(aic_vals))

    # Vuong 检验（每对模型跨两列）
    vuong_cells: List[str] = []
    for _, nb_res, zinb_res, vuong_r in model_pairs:
        if vuong_r:
            v_stat = vuong_r.get("vuong_statistic", 0.0)
            p_one = vuong_r.get("pvalue_one_sided", 1.0)
            p_two = vuong_r.get("pvalue_two_sided", 1.0)
            # 使用单侧 p 值判断; V > 0 表示 ZINB 优于 NB（代码计算 m=llf_ZINB-llf_NB）
            if p_one < 0.05 and v_stat > 0:
                prefer = "ZINB显著优于NB"
            elif p_one < 0.05 and v_stat < 0:
                prefer = "NB显著优于ZINB"
            else:
                prefer = "无显著差异"
            p_str = "p<0.001" if p_two < 0.001 else f"p={p_two:.4f}"
            vuong_text = f"{v_stat:.4f}，且{p_str}，{prefer}"
        else:
            vuong_text = "—"
        vuong_cells.append(vuong_text)
        vuong_cells.append("")
    lines.append("Vuong非嵌套模型检验\t" + "\t".join(vuong_cells))

    return _write_and_print("\n".join(lines), output_file, "回归结果")


# ============================================================
# 数据加载与特征提取
# ============================================================

def _load_hdkn_and_patents(
    run_id: str,
    networks_dir: Optional[Path] = None,
    patents_csv: Optional[Path] = None,
) -> Tuple[Path, dict, int, list]:
    """加载 HDKN 路径、网络 metadata、hist_end_year 和专利列表"""
    run_dir = get_run_dir(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"运行目录不存在: {run_dir}，请先运行 Step1")

    dirs = ensure_run_dirs(run_dir)
    networks_dir = networks_dir or dirs["networks_dir"]

    hdkn_path = networks_dir / "hdkn.pkl.gz"
    networks_meta_path = networks_dir / "networks_meta.json"
    if not hdkn_path.exists():
        raise FileNotFoundError(f"HDKN 不存在: {hdkn_path}，请先运行 Step1")
    if not networks_meta_path.exists():
        raise FileNotFoundError(f"网络 metadata 不存在: {networks_meta_path}")

    networks_meta = load_metadata(networks_meta_path)
    hist_end_year = networks_meta["hist_end_year"]
    _config.HIST_END_YEAR = hist_end_year

    patents_csv = patents_csv or RAW_PATENT_FILE
    if not Path(patents_csv).exists():
        raise FileNotFoundError(f"专利数据不存在: {patents_csv}")

    patents = _pipeline.load_patents_from_csv(Path(patents_csv), limit=None, smart_select=False)
    logger.info(f"✅ 加载 {len(patents)} 条专利（hist_end_year={hist_end_year}）")
    return hdkn_path, networks_meta, hist_end_year, patents


def _extract_and_fit(
    hdkn_path: Path,
    hist_end_year: int,
    alpha: float,
    patents: list,
    selected_features: List[str],
    networks_meta: dict,
    include_control_vars: bool = True,
) -> Tuple[pd.DataFrame, Any, Any, Optional[dict], List[str]]:
    """
    对给定 α 重算 HDKN 权重 → 提取特征 → 拟合 NB + ZINB。

    Returns:
        (df, nb_result, zinb_result, vuong_result, used_features)
    """
    from src.patent_opportunity_analysis.utils.dkn_wrapper import DKNNetwork
    import networkx as nx

    HDKN = load_dkn(hdkn_path)
    HDKN.assert_kind("HDKN")
    ref_year = hist_end_year
    _dkn_builder.compute_time_decay_weights(
        HDKN.graph, total_year=ref_year, alpha=alpha, expected_ref_year=ref_year,
    )

    empty_pdkn = DKNNetwork(
        kind="PDKN", graph=nx.Graph(),
        ref_year=networks_meta.get("max_year", hist_end_year),
        hist_end_year=hist_end_year,
    )

    df, nb_result, zinb_result, vuong_result, used_features = (
        _regression_flow.run_regression_flow(
            HDKN, empty_pdkn, patents,
            selected_features=selected_features,
            include_control_vars=include_control_vars,
            decay_factor=alpha,
            force_rebuild_hdkn_stats=True,
        )
    )
    return df, nb_result, zinb_result, vuong_result, used_features


def _fit_from_csv(
    df: pd.DataFrame,
    selected_features: List[str],
    include_control_vars: bool = True,
) -> Tuple[Any, Any, Optional[dict], List[str]]:
    """从已有特征 DataFrame 拟合 NB + ZINB"""
    np.random.seed(42)
    nb_result, zinb_result, vuong_result, used_features = (
        _regression_model.fit_both_nb_zinb(
            df,
            selected_features=selected_features,
            include_control_vars=include_control_vars,
        )
    )
    return nb_result, zinb_result, vuong_result, used_features


def _resolve_output_dir(run_id: str, output_dir: Optional[Path] = None) -> Path:
    """确定输出目录"""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    run_dir = get_run_dir(run_id)
    d = run_dir / "02_regression_workflow"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_alpha(output_dir: Path, cli_alpha: Optional[float] = None) -> float:
    """确定 α 值：CLI 参数 > Run 1 最优 > 默认"""
    if cli_alpha is not None:
        return cli_alpha
    run1_path = output_dir / "run1_results.json"
    if run1_path.exists():
        with open(run1_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        best = meta.get("best_alpha")
        if best is not None:
            logger.info(f"📋 从 Run 1 读取最优 α = {best}")
            return float(best)
    default = getattr(_config, "DECAY_FACTOR", 0.9)
    logger.info(f"⚠️  使用默认 α = {default}")
    return default


def _save_pkl(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Run 1: Alpha Selection
# ============================================================

def run1_alpha_selection(
    run_id: str,
    alphas: Optional[List[float]] = None,
    networks_dir: Optional[Path] = None,
    patents_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[float, Path]:
    """
    Run 1: 对候选 α 逐一拟合 NB + ZINB，选出 ZINB Log-Likelihood 最大的 α。

    Returns:
        (best_alpha, output_dir)
    """
    alphas = alphas or DEFAULT_ALPHAS
    out = _resolve_output_dir(run_id, output_dir)

    logger.info("=" * 80)
    logger.info("📐 Run 1: Alpha Selection")
    logger.info(f"   候选 α: {alphas}")
    logger.info(f"   特征:   {ALL_FEATURES}")
    logger.info(f"   控制:   {CONTROL_VARS}")
    logger.info("=" * 80)

    hdkn_path, networks_meta, hist_end_year, patents = _load_hdkn_and_patents(
        run_id, networks_dir, patents_csv,
    )

    results: List[Dict[str, Any]] = []
    best_alpha: Optional[float] = None
    best_ll: float = -np.inf
    best_df: Optional[pd.DataFrame] = None

    from tqdm import tqdm

    for alpha in tqdm(alphas, desc="Alpha 循环", unit="α"):
        logger.info(f"\n{'─' * 40} α = {alpha} {'─' * 40}")
        try:
            df, nb_res, zinb_res, vuong_res, _ = _extract_and_fit(
                hdkn_path, hist_end_year, alpha, patents, ALL_FEATURES, networks_meta,
            )
            ll_nb = float(nb_res.llf)
            ll_zinb = float(zinb_res.llf)
            conv_nb = _converged(nb_res)
            conv_zinb = _converged(zinb_res)

            rec: Dict[str, Any] = {
                "alpha": alpha,
                "ll_nb": ll_nb,
                "ll_zinb": ll_zinb,
                "conv_nb": conv_nb,
                "conv_zinb": conv_zinb,
                "n_samples": len(df),
            }
            results.append(rec)

            if ll_zinb > best_ll:
                best_ll = ll_zinb
                best_alpha = alpha
                best_df = df
                _save_pkl(nb_res, out / "best_nb.pkl")
                _save_pkl(zinb_res, out / "best_zinb.pkl")
                if vuong_res:
                    _save_json(vuong_res, out / "best_vuong.json")

            logger.info(f"  NB LL = {ll_nb:.4f}  |  ZINB LL = {ll_zinb:.4f}")

        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            logger.warning(f"α = {alpha} 拟合失败: {e}")
            results.append({
                "alpha": alpha,
                "ll_nb": None,
                "ll_zinb": None,
                "conv_nb": False,
                "conv_zinb": False,
                "n_samples": 0,
                "error": str(e),
            })

    if best_alpha is None:
        raise RuntimeError("所有候选 α 均拟合失败，无法选择")

    features_path = out / "regression_features.csv"
    best_df.to_csv(features_path, index=False)
    logger.info(f"💾 最优 α = {best_alpha} 的特征数据已保存: {features_path}")

    meta = {
        "run": 1,
        "best_alpha": best_alpha,
        "best_ll_zinb": best_ll,
        "alphas": alphas,
        "results": results,
        "features": ALL_FEATURES,
        "control_vars": CONTROL_VARS,
        "created_at": datetime.now().isoformat(),
    }
    _save_json(meta, out / "run1_results.json")

    table_alpha_selection(results, alphas, output_file=out / "run1_alpha_selection.txt")

    logger.success(f"\n✅ Run 1 完成！最优 α = {best_alpha}（ZINB LL = {best_ll:.4f}）")
    return best_alpha, out


# ============================================================
# Run 2: 全模型 + 共线性报告
# ============================================================

def run2_full_model(
    run_id: str,
    alpha: Optional[float] = None,
    networks_dir: Optional[Path] = None,
    patents_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Run 2: 使用最优 α，全部特征 + 控制变量拟合 NB/ZINB，输出共线性报告。
    """
    out = _resolve_output_dir(run_id, output_dir)
    alpha = _resolve_alpha(out, alpha)

    logger.info("=" * 80)
    logger.info(f"📊 Run 2: 全模型回归 + 共线性报告（α = {alpha}）")
    logger.info(f"   特征: {ALL_FEATURES}")
    logger.info(f"   控制: {CONTROL_VARS}")
    logger.info("=" * 80)

    features_path = out / "regression_features.csv"
    if features_path.exists():
        logger.info(f"📂 加载已有特征数据: {features_path}")
        df = pd.read_csv(features_path)
    else:
        logger.info("📊 特征数据不存在，重新提取...")
        hdkn_path, networks_meta, hist_end_year, patents = _load_hdkn_and_patents(
            run_id, networks_dir, patents_csv,
        )
        df, _, _, _, _ = _extract_and_fit(
            hdkn_path, hist_end_year, alpha, patents, ALL_FEATURES, networks_meta,
        )
        df.to_csv(features_path, index=False)
        logger.info(f"💾 特征数据已保存: {features_path}")

    nb_res, zinb_res, vuong_res, used = _fit_from_csv(df, ALL_FEATURES, include_control_vars=True)

    _save_pkl(nb_res, out / "run2_nb.pkl")
    _save_pkl(zinb_res, out / "run2_zinb.pkl")
    _save_json(vuong_res, out / "run2_vuong.json")

    available_vars = [v for v in ALL_FEATURES + CONTROL_VARS if v in df.columns]
    table_collinearity(df, available_vars, output_file=out / "run2_collinearity.txt")

    table_model_results(
        [("全模型", nb_res, zinb_res, vuong_res)],
        ALL_FEATURES + CONTROL_VARS,
        output_file=out / "run2_model_results.txt",
    )

    _save_json({
        "run": 2, "alpha": alpha, "features": ALL_FEATURES,
        "control_vars": CONTROL_VARS, "n_samples": len(df),
        "ll_nb": float(nb_res.llf), "ll_zinb": float(zinb_res.llf),
        "created_at": datetime.now().isoformat(),
    }, out / "run2_results.json")

    reports_dir = out / "reports"
    generate_combined_regression_report(
        nb_res, zinb_res, vuong_res, reports_dir, model_name="run2_full_model",
    )

    logger.success(f"\n✅ Run 2 完成！NB LL = {nb_res.llf:.4f}  |  ZINB LL = {zinb_res.llf:.4f}")
    logger.info("请审查共线性报告后，通过 --vars 指定 Run 3/4 使用的变量。")
    return out


# ============================================================
# Run 3: 选定变量 + 控制变量
# ============================================================

def run3_selected_control(
    run_id: str,
    selected_vars: List[str],
    alpha: Optional[float] = None,
    output_dir: Optional[Path] = None,
    suffix: str = "",
) -> Path:
    """
    Run 3: 选定的自变量 + 控制变量拟合 NB/ZINB。
    suffix: 用于区分不同变量组的文件名后缀，如 '_newn' / '_newe'
    """
    out = _resolve_output_dir(run_id, output_dir)
    alpha = _resolve_alpha(out, alpha)
    tag = f"run3{suffix}"

    logger.info("=" * 80)
    logger.info(f"📊 Run 3{suffix}: 选定变量 + 控制变量（α = {alpha}）")
    logger.info(f"   选定: {selected_vars}")
    logger.info(f"   控制: {CONTROL_VARS}")
    logger.info("=" * 80)

    features_path = out / "regression_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"特征数据不存在: {features_path}。请先运行 Run 1 或 Run 2。")
    df = pd.read_csv(features_path)

    invalid = [v for v in selected_vars if v not in df.columns]
    if invalid:
        avail = [c for c in df.columns if c not in ["patent_id", "Cited"] + CONTROL_VARS]
        raise ValueError(f"无效变量: {invalid}。可用特征: {avail}")

    nb_res, zinb_res, vuong_res, used = _fit_from_csv(df, selected_vars, include_control_vars=True)

    _save_pkl(nb_res, out / f"{tag}_nb.pkl")
    _save_pkl(zinb_res, out / f"{tag}_zinb.pkl")
    _save_json(vuong_res, out / f"{tag}_vuong.json")

    row_vars = selected_vars + [v for v in CONTROL_VARS if v in df.columns]
    table_model_results(
        [("选定+控制", nb_res, zinb_res, vuong_res)],
        row_vars,
        output_file=out / f"{tag}_model_results.txt",
    )

    _save_json({
        "run": 3, "alpha": alpha, "selected_vars": selected_vars,
        "control_vars": CONTROL_VARS, "n_samples": len(df),
        "ll_nb": float(nb_res.llf), "ll_zinb": float(zinb_res.llf),
        "created_at": datetime.now().isoformat(),
    }, out / f"{tag}_results.json")

    reports_dir = out / "reports"
    generate_combined_regression_report(
        nb_res, zinb_res, vuong_res, reports_dir, model_name=f"{tag}_selected_control",
    )

    logger.success(f"\n✅ Run 3{suffix} 完成！NB LL = {nb_res.llf:.4f}  |  ZINB LL = {zinb_res.llf:.4f}")
    return out


# ============================================================
# Run 4: 仅选定变量
# ============================================================

def run4_selected_only(
    run_id: str,
    selected_vars: List[str],
    alpha: Optional[float] = None,
    output_dir: Optional[Path] = None,
    suffix: str = "",
) -> Path:
    """
    Run 4: 仅选定的自变量（无控制变量）拟合 NB/ZINB。
    suffix: 用于区分不同变量组的文件名后缀，如 '_newn' / '_newe'
    """
    out = _resolve_output_dir(run_id, output_dir)
    alpha = _resolve_alpha(out, alpha)
    tag = f"run4{suffix}"

    logger.info("=" * 80)
    logger.info(f"📊 Run 4{suffix}: 仅选定变量（α = {alpha}）")
    logger.info(f"   选定: {selected_vars}")
    logger.info("   控制: （无）")
    logger.info("=" * 80)

    features_path = out / "regression_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"特征数据不存在: {features_path}。请先运行 Run 1 或 Run 2。")
    df = pd.read_csv(features_path)

    invalid = [v for v in selected_vars if v not in df.columns]
    if invalid:
        raise ValueError(f"无效变量: {invalid}")

    nb_res, zinb_res, vuong_res, used = _fit_from_csv(df, selected_vars, include_control_vars=False)

    _save_pkl(nb_res, out / f"{tag}_nb.pkl")
    _save_pkl(zinb_res, out / f"{tag}_zinb.pkl")
    _save_json(vuong_res, out / f"{tag}_vuong.json")

    table_model_results(
        [("仅选定", nb_res, zinb_res, vuong_res)],
        selected_vars,
        output_file=out / f"{tag}_model_results.txt",
    )

    _save_json({
        "run": 4, "alpha": alpha, "selected_vars": selected_vars,
        "n_samples": len(df),
        "ll_nb": float(nb_res.llf), "ll_zinb": float(zinb_res.llf),
        "created_at": datetime.now().isoformat(),
    }, out / f"{tag}_results.json")

    reports_dir = out / "reports"
    generate_combined_regression_report(
        nb_res, zinb_res, vuong_res, reports_dir, model_name=f"{tag}_selected_only",
    )

    logger.success(f"\n✅ Run 4{suffix} 完成！NB LL = {nb_res.llf:.4f}  |  ZINB LL = {zinb_res.llf:.4f}")
    return out


# ============================================================
# Combined: 合并 Run 2/3/4 的模型比较表
# ============================================================

def run_combined_table(
    run_id: str,
    selected_vars: List[str],
    output_dir: Optional[Path] = None,
) -> str:
    """
    合并 Run 2/3/4 的结果为 6 列模型比较表：
      模型1-NB  模型2-ZINB  模型3-NB  模型4-ZINB  模型5-NB  模型6-ZINB
    """
    out = _resolve_output_dir(run_id, output_dir)
    model_pairs: List[Tuple[str, Any, Any, Optional[dict]]] = []

    for run_label, run_tag in [("全模型", "run2"), ("选定+控制", "run3"), ("仅选定", "run4")]:
        nb_path = out / f"{run_tag}_nb.pkl"
        zinb_path = out / f"{run_tag}_zinb.pkl"
        vuong_path = out / f"{run_tag}_vuong.json"
        if not nb_path.exists() or not zinb_path.exists():
            logger.warning(f"⚠️  {run_label} 结果不存在（{nb_path}），跳过")
            continue
        nb_res = _load_pkl(nb_path)
        zinb_res = _load_pkl(zinb_path)
        vuong_res = _load_json(vuong_path) if vuong_path.exists() else None
        model_pairs.append((run_label, nb_res, zinb_res, vuong_res))

    if not model_pairs:
        raise RuntimeError("未找到任何运行结果。请先执行 Run 2/3/4。")

    all_row_vars = list(dict.fromkeys(ALL_FEATURES + CONTROL_VARS))

    return table_model_results(
        model_pairs, all_row_vars, output_file=out / "combined_results.txt",
    )


# ============================================================
# CLI 入口
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="回归分析工作流 (4-Run Process)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行流程：
  Run 1  ─ Alpha Selection（所有特征的 NB/ZINB，对候选 α 逐一拟合）
  Run 2  ─ 全模型回归 + 共线性报告（取最优 α，包含所有特征）
  （人工审查共线性报告，选定 Run 3/4 使用的变量）
  Run 3  ─ 选定变量 + 控制变量（NB/ZINB）
  Run 4  ─ 仅选定变量（NB/ZINB，无控制变量）
  combined ─ 合并 Run 2/3/4 结果为统一比较表

示例：
  python scripts/regression_workflow.py --run-id full --run 1
  python scripts/regression_workflow.py --run-id full --run 2 --alpha 0.95
  python scripts/regression_workflow.py --run-id full --run 3 --vars "New_e,Con_e,Constraint"
  python scripts/regression_workflow.py --run-id full --run 4 --vars "New_e,Con_e,Constraint"
  python scripts/regression_workflow.py --run-id full --run combined --vars "New_e,Con_e,Constraint"
        """,
    )
    parser.add_argument("--run-id", type=str, required=True, help="运行 ID（来自 Step1）")
    parser.add_argument(
        "--run", type=str, required=True,
        choices=["1", "2", "3", "4", "combined"],
        help="运行阶段: 1=Alpha Selection, 2=全模型+共线性, 3=选定+控制, 4=仅选定, combined=合并表格",
    )
    parser.add_argument("--alpha", type=float, default=None, help="衰减因子 α（Run 2 可选，默认使用 Run 1 最优值）")
    parser.add_argument("--vars", type=str, default=None, help="选定变量（Run 3/4/combined 必须，逗号分隔）")
    parser.add_argument("--alphas", type=str, default=None, help="候选 α 列表（Run 1，逗号分隔）")
    parser.add_argument("--networks-dir", type=Path, default=None, help="网络目录路径")
    parser.add_argument("--patents-csv", type=Path, default=None, help="专利 CSV 路径")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录")
    parser.add_argument("--suffix", type=str, default="", help="输出文件名后缀（区分不同变量组，如 _newn / _newe）")

    args = parser.parse_args()

    try:
        if args.run == "1":
            alphas = None
            if args.alphas:
                alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
            run1_alpha_selection(
                args.run_id, alphas=alphas,
                networks_dir=args.networks_dir, patents_csv=args.patents_csv,
                output_dir=args.output_dir,
            )

        elif args.run == "2":
            run2_full_model(
                args.run_id, alpha=args.alpha,
                networks_dir=args.networks_dir, patents_csv=args.patents_csv,
                output_dir=args.output_dir,
            )

        elif args.run == "3":
            if not args.vars:
                parser.error("Run 3 需要通过 --vars 指定选定变量（逗号分隔）")
            selected = [v.strip() for v in args.vars.split(",") if v.strip()]
            run3_selected_control(
                args.run_id, selected_vars=selected,
                alpha=args.alpha, output_dir=args.output_dir,
                suffix=args.suffix,
            )

        elif args.run == "4":
            if not args.vars:
                parser.error("Run 4 需要通过 --vars 指定选定变量（逗号分隔）")
            selected = [v.strip() for v in args.vars.split(",") if v.strip()]
            run4_selected_only(
                args.run_id, selected_vars=selected,
                alpha=args.alpha, output_dir=args.output_dir,
                suffix=args.suffix,
            )

        elif args.run == "combined":
            if not args.vars:
                parser.error("combined 需要通过 --vars 指定选定变量（逗号分隔）")
            selected = [v.strip() for v in args.vars.split(",") if v.strip()]
            run_combined_table(
                args.run_id, selected_vars=selected,
                output_dir=args.output_dir,
            )

        return 0

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(f"回归工作流执行失败: {exc}")
        return 1
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130
    except BaseException as exc:
        logger.error(f"未预期的错误: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
