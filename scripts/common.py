"""
scripts 目录公共工具 + α 衰减因子选择公共逻辑

公共工具:
- init_script(): 脚本入口初始化（project_root、sys.path、logging）
- parse_comma_list(): 解析逗号分隔字符串为列表

α 选择:
- run_alpha_selection(): 对候选 α 重构网络权重、提取特征、拟合回归，选出最优 α
- write_alpha_report(): 生成 α 选择 Markdown 报告
"""

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

# ============================================================================
# 脚本初始化工具
# ============================================================================

def _lock_random_seeds(seed: int = 42) -> None:
    """锁死所有随机种子，保证 Log-Likelihood 可复现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def init_script(log_level: str = "INFO", seed: int = 42) -> Path:
    """
    脚本入口初始化：锁死随机种子、设置 project_root、sys.path、logging

    Returns:
        project_root 路径
    """
    _lock_random_seeds(seed)
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.patent_opportunity_analysis.utils.logging_config import setup_project_logging
    setup_project_logging(log_level=log_level)
    return project_root


def parse_comma_list(s: Optional[str]) -> Optional[List[str]]:
    """
    解析逗号分隔字符串为列表，空则返回 None

    Args:
        s: 逗号分隔的字符串，如 "a, b, c"

    Returns:
        去除空白后的列表，空字符串或 None 返回 None
    """
    if not s or not str(s).strip():
        return None
    result = [x.strip() for x in str(s).split(",") if x.strip()]
    return result if result else None


# ============================================================================
# α 衰减因子选择
# ============================================================================

C_VARS = ["Back_cite", "Assignee", "Total_pat"]
DEFAULT_ALPHAS = [0.80, 0.85, 0.90, 0.95, 0.99, 1.00]


def run_alpha_selection(
    run_id: str,
    step_name: str,
    step_label: str,
    output_subdir: str,
    x_vars: List[str],
    report_title: Optional[str] = None,
    alphas: Optional[List[float]] = None,
    patents_csv: Optional[Path] = None,
    networks_dir: Optional[Path] = None,
    force: bool = False,
    min_app_year: Optional[int] = None,
    include_control_vars: bool = True,
) -> Path:
    """
    对每个候选 α 重构网络权重、提取特征、拟合 ZINB，选出 LL 最大的 α。

    Args:
        run_id: 运行 ID
        step_name: 步骤名称（如 "02_1_alpha_selection"）
        step_label: 日志显示标签（如 "Step2.1"）
        output_subdir: 输出子目录名（如 "02_1_alpha_selection"）
        x_vars: 自变量列表
        alphas: 候选 α 列表
        patents_csv: 专利 CSV 路径
        networks_dir: 网络目录路径
        force: 是否强制重算

    Returns:
        输出目录路径
    """
    from src.patent_opportunity_analysis import config as _config
    from src.patent_opportunity_analysis import regression_flow as _regression_flow
    from src.patent_opportunity_analysis import dkn_builder as _dkn_builder
    from src.patent_opportunity_analysis.utils.paths import RAW_PATENT_FILE
    from src.patent_opportunity_analysis.utils.network_io import load_dkn, load_metadata
    from src.patent_opportunity_analysis.utils.run_utils import get_run_dir, ensure_run_dirs
    from src.patent_opportunity_analysis.utils.dkn_wrapper import DKNNetwork
    from src.patent_opportunity_analysis import pipeline as _pipeline
    import networkx as nx

    alphas = alphas or DEFAULT_ALPHAS
    logger.info("=" * 80)
    logger.info(f"📐 {step_label}: α 衰减因子选择")
    logger.info("=" * 80)
    c_vars = C_VARS if include_control_vars else []
    logger.info("固定模型: NB + ZINB（共享流程，α 选择以 ZINB LL 为准）")
    logger.info(f"自变量 X: {x_vars}")
    logger.info(f"控制变量 C: {c_vars if c_vars else '（无）'}")
    logger.info(f"候选 α: {alphas}")

    run_dir = get_run_dir(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"运行目录不存在: {run_dir}，请先运行 Step1")

    dirs = ensure_run_dirs(run_dir)
    networks_dir = networks_dir or dirs["networks_dir"]
    alpha_dir = run_dir / output_subdir
    alpha_dir.mkdir(parents=True, exist_ok=True)

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
    if not patents_csv.exists():
        raise FileNotFoundError(f"专利数据不存在: {patents_csv}")
    patents = _pipeline.load_patents_from_csv(
        patents_csv, limit=None, smart_select=False, min_app_year=min_app_year
    )
    logger.info(f"✅ 加载 {len(patents)} 条专利")

    empty_pdkn = DKNNetwork(
        kind="PDKN",
        graph=nx.Graph(),
        ref_year=networks_meta.get("max_year", hist_end_year),
        hist_end_year=hist_end_year,
    )

    # 所有 α 统一走相同计算路径，避免 α=1 复用 Step2 导致与其他 α 计算逻辑不一致
    results: List[Dict[str, Any]] = []
    pbar = tqdm(alphas, desc="α 循环", unit="α")

    for alpha in pbar:
        pbar.set_postfix(alpha=alpha)
        logger.info(f"\n--- α = {alpha} ---")

        HDKN_alpha = load_dkn(hdkn_path)
        HDKN_alpha.assert_kind("HDKN")
        # 衰减参考年 = 网络快照截断年（hist_end_year），与原文定义一致
        ref_year = hist_end_year
        _dkn_builder.compute_time_decay_weights(
            HDKN_alpha.graph,
            total_year=ref_year,
            alpha=alpha,
            expected_ref_year=ref_year,
        )

        try:
            df, nb_result, zinb_result, vuong_result, _ = _regression_flow.run_regression_flow(
                HDKN_alpha,
                empty_pdkn,
                patents,
                selected_features=x_vars,
                include_control_vars=include_control_vars,
                decay_factor=alpha,
                force_rebuild_hdkn_stats=True,
            )
        except Exception as e:
            logger.warning(f"α={alpha} 回归失败: {e}")
            results.append({
                "alpha": alpha,
                "log_likelihood_nb": None,
                "log_likelihood_zinb": None,
                "converged": False,
                "error": str(e),
                "coefficients": {},
                "pvalues": {},
                "n_samples": 0,
                "summary_nb": None,
                "summary_zinb": None,
            })
            continue

        # 使用 ZINB 的 LL 做 α 选择（与原有逻辑一致）
        llf_zinb = float(zinb_result.llf) if hasattr(zinb_result, "llf") else None
        llf_nb = float(nb_result.llf) if hasattr(nb_result, "llf") else None
        converged = (
            getattr(zinb_result, "mle_retvals", {}) or {}
        ).get("converged", True)

        coef_signs = {}
        coef_sig = {}
        pvalues = getattr(zinb_result, "pvalues", None)
        if hasattr(zinb_result, "params") and zinb_result.params is not None:
            params = zinb_result.params
            for var in x_vars + c_vars:
                if hasattr(params, "index") and var in params.index:
                    v = float(params[var])
                    coef_signs[var] = "+" if v > 0 else ("-" if v < 0 else "0")
                    pv = None
                    if pvalues is not None:
                        if hasattr(pvalues, "index") and var in getattr(pvalues, "index", []):
                            pv = float(pvalues[var])
                        elif isinstance(pvalues, dict) and var in pvalues:
                            pv = float(pvalues[var])
                    coef_sig[var] = pv
                elif isinstance(params, dict) and var in params:
                    v = float(params[var])
                    coef_signs[var] = "+" if v > 0 else ("-" if v < 0 else "0")
                    pv = float(pvalues[var]) if pvalues is not None and isinstance(pvalues, dict) and var in pvalues else None
                    coef_sig[var] = pv

        results.append({
            "alpha": alpha,
            "log_likelihood": llf_zinb,
            "log_likelihood_nb": llf_nb,
            "log_likelihood_zinb": llf_zinb,
            "converged": converged,
            "coefficients": coef_signs,
            "pvalues": coef_sig,
            "n_samples": len(df),
            "summary_nb": str(nb_result.summary()) if hasattr(nb_result, "summary") else None,
            "summary_zinb": str(zinb_result.summary()) if hasattr(zinb_result, "summary") else None,
            "vuong": vuong_result,
        })
        logger.info(f"  NB LL = {llf_nb:.2f}, ZINB LL = {llf_zinb:.2f}, 收敛 = {converged}")

        # 保存模型与特征，供 Step2 复用
        import pickle
        alpha_key = str(alpha)
        nb_pkl = alpha_dir / f"alpha_{alpha_key}_nb.pkl"
        zinb_pkl = alpha_dir / f"alpha_{alpha_key}_zinb.pkl"
        feat_csv = alpha_dir / f"alpha_{alpha_key}_features.csv"
        try:
            with open(nb_pkl, "wb") as f:
                pickle.dump(nb_result, f)
            with open(zinb_pkl, "wb") as f:
                pickle.dump(zinb_result, f)
            df.to_csv(feat_csv, index=False)
        except Exception as e:
            logger.warning(f"保存 α={alpha} 模型失败: {e}")

    valid = [r for r in results if r["log_likelihood"] is not None]
    if not valid:
        raise RuntimeError("所有 α 均回归失败，无法选择")

    best = max(valid, key=lambda r: r["log_likelihood"])
    best_alpha = best["alpha"]
    logger.success(f"\n✅ 最优 α = {best_alpha}（Log-Likelihood = {best['log_likelihood']:.2f}）")

    # 复制最优 α 的模型到标准路径，供 Step2 复用
    import shutil
    best_key = str(best_alpha)
    best_nb = alpha_dir / f"alpha_{best_key}_nb.pkl"
    best_zinb = alpha_dir / f"alpha_{best_key}_zinb.pkl"
    best_feat = alpha_dir / f"alpha_{best_key}_features.csv"
    step2_nb = alpha_dir / "regression_model_NegativeBinomial.pkl"
    step2_zinb = alpha_dir / "regression_model_ZINB.pkl"
    step2_feat = alpha_dir / "regression_features.csv"
    if best_nb.exists() and best_zinb.exists():
        shutil.copy2(best_nb, step2_nb)
        shutil.copy2(best_zinb, step2_zinb)
        if best_feat.exists():
            shutil.copy2(best_feat, step2_feat)
        logger.info(f"💾 最优 α={best_alpha} 的模型已复制到标准路径，供 Step2 复用")

    out_path = alpha_dir / "alpha_selection_results.json"
    meta = {
        "step_name": step_name,
        "created_at": datetime.now().isoformat(),
        "run_id": run_id,
        "hist_end_year": hist_end_year,
        "model_type": "NB+ZINB",
        "x_vars": x_vars,
        "c_vars": c_vars,
        "candidate_alphas": alphas,
        "best_alpha": best_alpha,
        "best_log_likelihood": best["log_likelihood"],
        "results": results,
        "step2_reuse": {
            "model_nb": str(step2_nb),
            "model_zinb": str(step2_zinb),
            "features": str(step2_feat),
        } if (best_nb.exists() and best_zinb.exists()) else None,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 结果已保存: {out_path}")

    # 加载 Step2 的 regression_meta 用于 LL 对比
    step2_meta_path = ensure_run_dirs(run_dir)["regression_dir"] / "regression_meta.json"
    step2_meta = None
    if step2_meta_path.exists():
        try:
            step2_meta = load_metadata(step2_meta_path)
            logger.info(f"📋 已加载 Step2 metadata，用于 LL 对比（α={step2_meta.get('decay_factor')}）")
        except Exception as e:
            logger.warning(f"加载 Step2 metadata 失败: {e}")

    report_path = alpha_dir / "alpha_selection_report.md"
    write_alpha_report(meta, report_path, report_title or f"α 衰减因子选择报告 ({step_label})", step2_meta=step2_meta)
    logger.info(f"📄 报告已保存: {report_path}")

    return alpha_dir


def write_alpha_report(
    meta: Dict,
    path: Path,
    report_title: str = "α 衰减因子选择报告",
    step2_meta: Optional[Dict] = None,
) -> None:
    """生成 α 选择 Markdown 报告"""
    lines = [
        f"# {report_title}",
        "",
        f"**生成时间**: {meta.get('created_at', '')}",
        f"**Run ID**: {meta.get('run_id', '')}",
        f"**hist_end_year**: {meta.get('hist_end_year', '')}",
        "",
        "## 1. 固定模型配置",
        "",
        f"- **模型**: NB + ZINB（双模型，α 选择以 ZINB LL 为准）",
        f"- **自变量 X**: {', '.join(meta.get('x_vars', []))}",
        f"- **控制变量 C**: {', '.join(meta.get('c_vars', []))}",
        f"- **候选 α**: {meta.get('candidate_alphas', [])}",
        "",
        "## 2. 决策结果",
        "",
        f"**最优 α = {meta.get('best_alpha', 'N/A')}**",
        f"**对应 ZINB Log-Likelihood = {meta.get('best_log_likelihood', 'N/A')}**",
        "",
    ]

    # 与 Step2 的 LL 对比
    if step2_meta:
        step2_alpha = step2_meta.get("decay_factor")
        step2_ll_nb = step2_meta.get("log_likelihood_nb")
        step2_ll_zinb = step2_meta.get("log_likelihood_zinb")
        def _fmt_ll(v):
            if v is None:
                return "—"
            try:
                return f"{float(v):.4f}"
            except (TypeError, ValueError):
                return "—"

        lines.extend([
            "## 2.1 与 Step2 的 LL 对比",
            "",
            f"Step2 使用 **α = {step2_alpha}**（decay_factor），回归结果：",
            "",
            "| 模型 | Log-Likelihood |",
            "|------|----------------|",
            f"| NB | {_fmt_ll(step2_ll_nb)} |",
            f"| ZINB | {_fmt_ll(step2_ll_zinb)} |",
            "",
        ])
        step2_features = set(step2_meta.get("selected_features", []))
        as_features = set(meta.get("x_vars", []))
        if step2_features != as_features:
            lines.extend([
                "> **说明**：Step2 与 Alpha Selection 使用不同特征集时，LL 会有差异。",
                "",
            ])
        # 若 Alpha Selection 中有相同 α，对比一致性
        def _alpha_eq(a, b):
            if a is None or b is None:
                return False
            return abs(float(a) - float(b)) < 1e-9

        as_same = next((r for r in meta.get("results", []) if _alpha_eq(r.get("alpha"), step2_alpha)), None)
        if as_same:
            as_ll_nb = as_same.get("log_likelihood_nb")
            as_ll_zinb = as_same.get("log_likelihood_zinb")
            if as_ll_nb is not None and as_ll_zinb is not None:
                diff_nb = as_ll_nb - (step2_ll_nb or 0)
                diff_zinb = as_ll_zinb - (step2_ll_zinb or 0)
                lines.extend([
                    f"**α = {step2_alpha} 时 Alpha Selection 结果**：NB LL = {as_ll_nb:.4f}，ZINB LL = {as_ll_zinb:.4f}",
                    f"（与 Step2 差异：NB Δ={diff_nb:+.4f}，ZINB Δ={diff_zinb:+.4f}，应接近 0）",
                    "",
                ])
        lines.append("")

    lines.extend([
        "## 3. 各 α 结果明细",
        "",
        "| α | NB LL | ZINB LL | 收敛 | 样本量 | 系数符号 (X) | 显著 (p<0.05) |",
        "|---|-------|---------|------|--------|--------------|---------------|",
    ])
    for r in meta.get("results", []):
        ll_nb = r.get("log_likelihood_nb") or r.get("log_likelihood")
        ll_zinb = r.get("log_likelihood_zinb") or r.get("log_likelihood")
        ll_nb_str = f"{ll_nb:.2f}" if ll_nb is not None else "—"
        ll_zinb_str = f"{ll_zinb:.2f}" if ll_zinb is not None else "—"
        conv = "✓" if r.get("converged") else "✗"
        n = r.get("n_samples", "—")
        signs = r.get("coefficients", {})
        pvals = r.get("pvalues", {})
        x_parts = []
        for v in meta.get("x_vars", []):
            s = signs.get(v, "?")
            pv = pvals.get(v)
            sig = "*" if (pv is not None and pv < 0.05) else ""
            x_parts.append(f"{v}:{s}{sig}")
        x_signs = ", ".join(x_parts)
        sig_vars = [v for v in meta.get("x_vars", []) + meta.get("c_vars", [])
                    if pvals.get(v) is not None and pvals[v] < 0.05]
        sig_str = ", ".join(sig_vars) if sig_vars else "—"
        lines.append(f"| {r.get('alpha')} | {ll_nb_str} | {ll_zinb_str} | {conv} | {n} | {x_signs} | {sig_str} |")
    lines.extend(["", "---", ""])

    lines.extend(["", "## 4. 各 α 回归系数全表（NB + ZINB）", ""])
    for r in meta.get("results", []):
        alpha_val = r.get("alpha")
        summary_nb = r.get("summary_nb")
        summary_zinb = r.get("summary_zinb")
        if summary_nb:
            lines.extend([f"### α = {alpha_val} — NB", "", "```", summary_nb.strip(), "```", ""])
        if summary_zinb:
            lines.extend([f"### α = {alpha_val} — ZINB", "", "```", summary_zinb.strip(), "```", ""])
        if not summary_nb and not summary_zinb:
            lines.extend([f"### α = {alpha_val}", "", "（无回归结果）", ""])
    lines.append("---")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
