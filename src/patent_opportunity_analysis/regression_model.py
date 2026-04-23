# src/regression_model.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import pickle
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set
from loguru import logger

# 导入回归分析工具
from .utils import regression_analysis as _regression_analysis
from .utils import errors as _utils_errors
generate_full_regression_analysis = _regression_analysis.generate_full_regression_analysis
compute_vuong_test = _regression_analysis.compute_vuong_test
RegressionModelError = _utils_errors.RegressionModelError

def build_reg_df(records):
    return pd.DataFrame(records)

def fit_regression(
    df: pd.DataFrame,
    model_type: str = "NegativeBinomial",
    zinb_inflate_vars: Optional[List[str]] = None,
    generate_analysis: bool = True,
    output_dir: Path = None,
    selected_features: Optional[List[str]] = None,
    include_control_vars: bool = True,
    compute_vuong: bool = True,  # ZINB 时是否计算 Vuong（fit_both_nb_zinb 中设为 False）
):
    """
    拟合回归模型，支持多种模型类型和特征选择
    
    Args:
        df: 数据框
        model_type: 模型类型 ("NegativeBinomial", "Poisson", "OLS", "ZINB")
        zinb_inflate_vars: ZINB 零膨胀部分使用的变量（None 则用截距）
        generate_analysis: 是否生成分析报告和诊断图
        output_dir: 输出目录
        selected_features: 要使用的特征列表，None表示使用所有可用特征
        include_control_vars: 是否包含控制变量（Back_cite, Assignee, Total_pat）
    
    Returns:
        回归结果对象和使用的特征列表
    """
    logger.info(f"开始拟合{model_type}回归模型...")
    
    # 检查必需变量（Cited必须存在）
    if 'Cited' not in df.columns:
        raise ValueError("缺少必需变量: Cited")
    
    # 确定要使用的特征
    if selected_features is None:
        # 默认使用所有可用特征（排除控制变量）
        control_vars = ['Back_cite', 'Assignee', 'Total_pat', 'Cited']
        available_features = [col for col in df.columns if col not in control_vars]
        selected_features = available_features
        logger.info(f"未指定特征选择，使用所有可用特征: {selected_features}")
    else:
        # 验证特征名
        invalid_features = [f for f in selected_features if f not in df.columns]
        if invalid_features:
            available = [col for col in df.columns if col != 'Cited']
            raise ValueError(
                f"无效的特征名: {invalid_features}. "
                f"数据中可用特征: {available}"
            )
        logger.info(f"使用指定的特征: {selected_features}")
    
    if len(selected_features) == 0:
        raise ValueError("至少需要一个特征")
    
    # 控制变量（可选）
    control_vars = ['Back_cite', 'Assignee', 'Total_pat']
    available_control_vars = (
        [var for var in control_vars if var in df.columns]
        if include_control_vars
        else []
    )

    # 构建回归公式
    formula_vars = selected_features + available_control_vars
    formula = "Cited ~ " + " + ".join(formula_vars)
    logger.info(f"回归公式: {formula}")
    logger.info(f"使用的特征: {selected_features}")
    logger.info(f"使用的控制变量: {available_control_vars}")
    
    try:
        # 数据预检查：移除 inf/NaN（statsmodels 无法处理）
        reg_cols = ['Cited'] + [c for c in formula_vars if c in df.columns]
        df_clean = df[reg_cols].copy()
        n_before = len(df_clean)
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna(subset=reg_cols)
        n_dropped = n_before - len(df_clean)
        if n_dropped > 0:
            logger.warning(f"⚠️  移除 {n_dropped} 行含 inf/NaN 的样本，剩余 {len(df_clean)} 行")
        df = df_clean
        
        # 数据预检查
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"因变量 Cited 统计: 均值={df['Cited'].mean():.2f}, 方差={df['Cited'].var():.2f}")

        # 检查自变量方差
        zero_var_cols = []
        for col in formula_vars:
            if col not in df.columns:
                logger.warning(f"⚠️  变量 {col} 不在数据中，将从公式中移除")
                continue
            if df[col].var() == 0:
                zero_var_cols.append(col)
                logger.warning(f"⚠️  变量 {col} 方差为0，可能导致模型问题")

        if zero_var_cols:
            logger.warning(f"发现 {len(zero_var_cols)} 个零方差变量: {zero_var_cols}，自动从公式中移除")
            formula_vars = [v for v in formula_vars if v not in zero_var_cols]
            selected_features = [f for f in selected_features if f not in zero_var_cols]
            if not formula_vars:
                raise ValueError("移除零方差变量后无剩余自变量，无法拟合模型")
            formula = "Cited ~ " + " + ".join(formula_vars)
            logger.info(f"更新后回归公式: {formula}")

        # 对于小样本，尝试更简单的模型
        if len(df) < 30:
            logger.info("样本量小，尝试使用OLS回归（更稳定）")
            model = smf.ols(formula=formula, data=df)
        else:
            if model_type == "NegativeBinomial":
                # 原文与表6：NB 在原始尺度自变量上拟合，系数可直接用于 Z=ΣβX（不做标准化）
                model = smf.glm(
                    formula=formula,
                    data=df,
                    family=sm.families.NegativeBinomial()
                )
            elif model_type == "Poisson":
                model = smf.glm(
                    formula=formula,
                    data=df,
                    family=sm.families.Poisson()
                )
            elif model_type == "OLS":
                model = smf.ols(formula=formula, data=df)
            elif model_type == "ZINB":
                pass  # handled below
            else:
                raise ValueError(
                    f"不支持的模型类型: {model_type}，"
                    f"支持的类型: NegativeBinomial, Poisson, OLS, ZINB"
                )

        vuong_result = None
        if model_type == "ZINB":
            result = _fit_zinb(df, formula_vars, zinb_inflate_vars)
            if compute_vuong:
                vuong_result = _compute_vuong_for_zinb(df, formula_vars, result, available_control_vars)
        else:
            # 提高优化精度和最大迭代次数，保证 LL 可复现
            result = model.fit(maxiter=2000, tol=1e-10, atol=1e-10)
        logger.info("模型拟合完成")

        # 检查模型收敛性
        if hasattr(result, 'converged') and not result.converged:
            logger.warning("⚠️  模型可能未完全收敛")

        try:
            logger.info(f"\n{result.summary()}")
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning(f"summary() 输出失败 ({e})，继续后续流程")
        
        # 生成分析报告和诊断图
        if generate_analysis and output_dir:
            try:
                analysis_results = generate_full_regression_analysis(
                    result, output_dir, model_name=f"regression_model_{model_type.lower()}",
                    vuong_result=vuong_result,
                )
                logger.success("回归分析报告和诊断图已生成")
            except Exception as e:
                logger.warning(f"生成回归分析时出错: {e}")
        
        # 保存特征选择到meta文件
        if output_dir:
            try:
                meta_file = output_dir / "regression_meta.json"
                meta_data = {
                    'selected_features': selected_features,
                    'control_vars': available_control_vars,
                    'formula': formula,
                    'model_type': model_type,
                    'n_samples': len(df),
                    'n_features': len(selected_features)
                }
                with open(meta_file, 'w') as f:
                    json.dump(meta_data, f, indent=2)
                logger.info(f"回归元数据已保存: {meta_file}")
            except Exception as e:
                logger.warning(f"保存回归元数据失败: {e}")
        
        return result, selected_features
    except Exception as e:
        logger.error(f"模型拟合失败: {e}")
        raise RegressionModelError(f"拟合{model_type}模型时出错: {e}") from e

def _compute_vuong_for_zinb(
    df: pd.DataFrame,
    formula_vars: List[str],
    zinb_result,
    available_control_vars: List[str],
) -> Optional[dict]:
    """拟合 NB 并计算 Vuong 检验（ZINB vs NB）"""
    try:
        formula = "Cited ~ " + " + ".join(formula_vars)
        model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
        nb_result = model.fit(maxiter=2000, tol=1e-10, atol=1e-10)
        return compute_vuong_test(zinb_result, nb_result)
    except Exception as e:
        logger.warning(f"Vuong 检验计算失败: {e}")
        return None


def _fit_zinb(
    df: pd.DataFrame,
    formula_vars: List[str],
    inflate_vars: Optional[List[str]] = None,
):
    """
    拟合零膨胀负二项回归 (ZINB)。

    零膨胀模型将数据分为两个过程：
      1. 二元膨胀过程 (logit): 是否是"结构性零"
      2. 计数过程 (NegBin): 对于非结构性零的样本，其计数值的分布

    Args:
        df: 包含 Cited 列和自变量的 DataFrame
        formula_vars: 计数部分使用的自变量列名列表
        inflate_vars: 膨胀 (logit) 部分使用的变量，None 则仅使用截距项
    """
    endog = df["Cited"].astype(float)

    X = df[formula_vars].astype(float)
    exog_df = sm.add_constant(X)

    if inflate_vars:
        invalid = [v for v in inflate_vars if v not in df.columns]
        if invalid:
            raise ValueError(f"ZINB inflate_vars 中不存在的列: {invalid}")
        exog_infl_df = sm.add_constant(df[inflate_vars].astype(float))
    else:
        exog_infl_df = pd.DataFrame(
            np.ones((len(df), 1)), columns=["const"], index=df.index
        )

    n_zeros = int((endog == 0).sum())
    pct_zeros = n_zeros / len(endog) * 100
    logger.info(
        f"ZINB: 样本量={len(endog)}, 零值计数={n_zeros} ({pct_zeros:.1f}%), "
        f"均值={endog.mean():.2f}, 方差={endog.var():.2f}"
    )

    model = ZeroInflatedNegativeBinomialP(
        endog,
        exog_df,
        exog_infl=exog_infl_df,
        p=2,
    )

    n_inflate = exog_infl_df.shape[1]
    n_count = exog_df.shape[1]
    start_params = np.zeros(n_inflate + n_count + 1)

    pct_zero = (endog == 0).mean()
    pos_mean = float(endog[endog > 0].mean()) if (endog > 0).any() else 1.0
    start_params[0] = np.log(max(pct_zero, 0.01) / max(1 - pct_zero, 0.01))
    start_params[n_inflate] = np.log(max(pos_mean, 1.0))
    start_params[-1] = 2.0

    # 多优化器回退策略：bfgs → nm → lbfgs → powell
    # 第一轮用默认协方差（避免 HC0 导致 Hessian 奇异），收敛后再补 HC0
    ZINB_METHODS = ["bfgs", "nm", "lbfgs", "powell"]
    best_result = None
    best_ll = -np.inf
    best_method = None
    found_converged = False

    for method in ZINB_METHODS:
        try:
            fit_result = model.fit(
                method=method,
                start_params=start_params,
                maxiter=5000,
                disp=False,
            )
            if np.isnan(fit_result.llf):
                logger.warning(f"ZINB {method} 产生 LLF=NaN，尝试下一个优化器")
                continue

            ll = fit_result.llf
            converged = (
                getattr(fit_result, "mle_retvals", None) or {}
            ).get("converged", True)

            if converged:
                best_result, best_ll, best_method = fit_result, ll, method
                found_converged = True
                logger.info(f"ZINB 使用 {method} 优化器收敛，LL={ll:.4f}")
                break

            if ll > best_ll:
                best_result, best_ll, best_method = fit_result, ll, method
            logger.info(f"ZINB {method} 未完全收敛（LL={ll:.4f}），继续尝试")
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            logger.warning(f"ZINB {method} 失败: {e}，尝试下一个优化器")
            continue

    if best_result is None or np.isnan(best_result.llf):
        raise RuntimeError(
            f"ZINB 所有优化器均失败（尝试: {ZINB_METHODS}），LLF=NaN。"
            f"请检查数据或减少自变量数量。"
        )

    result = best_result
    if not found_converged:
        logger.warning(f"⚠️  ZINB 未完全收敛，采用 {best_method} 的最佳结果（LL={best_ll:.4f}）")
    logger.info(f"ZINB 最终: method={best_method}, LL={result.llf:.4f}")

    # 收敛后尝试补计 HC0 稳健标准误（不改变 result 除非成功）
    try:
        hc0_result = None
        if hasattr(result, "get_robustcov_results"):
            hc0_result = result.get_robustcov_results("HC0")
        elif hasattr(result, "_get_robustcov_results"):
            hc0_result = result._get_robustcov_results("HC0")
        if hc0_result is not None and hasattr(hc0_result, "llf"):
            result = hc0_result
            logger.info("已补计 HC0 稳健标准误")
        else:
            logger.warning("HC0 返回无效结果，使用默认标准误")
    except (np.linalg.LinAlgError, ValueError, AttributeError) as e:
        logger.warning(f"HC0 稳健标准误计算失败: {e}，使用默认标准误")

    try:
        logger.info(f"\n{result.summary()}")
    except (ValueError, np.linalg.LinAlgError, AttributeError) as e:
        logger.warning(f"ZINB summary() 输出失败 ({e})，系数仍可用")
    return result


def fit_both_nb_zinb(
    df: pd.DataFrame,
    selected_features: Optional[List[str]] = None,
    include_control_vars: bool = True,
    zinb_inflate_vars: Optional[List[str]] = None,
) -> Tuple[Any, Any, Optional[dict], List[str]]:
    """
    拟合 NB 与 ZINB，并计算 Vuong 检验。供 Alpha Selection 与 Step2 共用。

    Returns:
        (nb_result, zinb_result, vuong_result, used_features)
    """
    # 固定随机种子，保证同一输入下 ZINB 拟合结果可复现（Alpha Selection 多次运行 LL 一致）
    np.random.seed(42)
    nb_result, used_features = fit_regression(
        df,
        model_type="NegativeBinomial",
        generate_analysis=False,
        output_dir=None,
        selected_features=selected_features,
        include_control_vars=include_control_vars,
    )
    zinb_result, _ = fit_regression(
        df,
        model_type="ZINB",
        zinb_inflate_vars=zinb_inflate_vars,
        generate_analysis=False,
        output_dir=None,
        selected_features=selected_features,
        include_control_vars=include_control_vars,
        compute_vuong=False,  # 由本函数统一计算
    )
    try:
        vuong_result = compute_vuong_test(zinb_result, nb_result)
    except Exception as e:
        logger.warning(f"Vuong 检验计算失败: {e}")
        vuong_result = None
    return nb_result, zinb_result, vuong_result, used_features


def extract_nb_significant_coefficients(
    nb_result: Any,
    p_threshold: float = 0.05,
    exclude_names: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    从 NB（GLM NegativeBinomial）结果中提取用于 ACO 线性目标 Z 的系数。

    - 仅保留 p < p_threshold 的项；排除截距与控制变量（控制变量在 ACO 中恒为 0，不进入 Z）。
    - 系数对应 Step2/regression_features.csv 中的原始尺度特征，与 compute_features_for_subnetwork 一致。

    Args:
        nb_result: statsmodels 拟合结果（GLM Results）
        p_threshold: 显著性水平
        exclude_names: 额外排除的参数名（默认排除 Intercept/const 与 Back_cite, Assignee, Total_pat）

    Returns:
        特征名 -> 系数（浮点）
    """
    default_exclude = {"Intercept", "const", "Back_cite", "Assignee", "Total_pat"}
    if exclude_names:
        default_exclude = default_exclude | exclude_names

    if not hasattr(nb_result, "params"):
        logger.warning("nb_result 无 params，返回空 objective 系数")
        return {}

    params = nb_result.params
    pvals = getattr(nb_result, "pvalues", None)
    if pvals is None:
        logger.warning("NB 结果无 pvalues，objective 系数未做 p 值筛选（仅排除截距与控制变量）")

    out: Dict[str, float] = {}
    for name in params.index:
        nm = str(name)
        if nm in default_exclude:
            continue
        try:
            coef = float(params[name])
        except (TypeError, ValueError):
            continue
        if pvals is not None:
            if name not in pvals.index:
                continue
            try:
                pv = float(pvals[name])
            except (TypeError, ValueError):
                continue
            if pv >= p_threshold or not np.isfinite(pv):
                continue
        out[nm] = coef

    return out


def fit_nb(df: pd.DataFrame, generate_analysis: bool = True, output_dir: Path = None):
    """
    拟合负二项回归模型（向后兼容）
    """
    return fit_regression(df, model_type="NegativeBinomial", generate_analysis=generate_analysis, output_dir=output_dir)

def save_model(result, path: Path):
    """保存模型"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    logger.success(f"模型已保存到: {path}")
