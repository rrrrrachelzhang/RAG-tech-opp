# src/utils/regression_analysis.py
"""
回归模型分析工具
生成回归结果分析报告和诊断图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import statsmodels.api as sm
from loguru import logger

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_pearson_correlation_matrix(X: np.ndarray, columns: list) -> pd.DataFrame:
    """
    计算皮尔逊相关系数矩阵

    Args:
        X: 自变量矩阵 (n_samples, n_features)，不含常数项
        columns: 变量名列表

    Returns:
        相关系数矩阵 DataFrame
    """
    df = pd.DataFrame(X, columns=columns)
    return df.corr(method="pearson")


def compute_vif(X: np.ndarray, columns: list) -> pd.DataFrame:
    """
    计算方差膨胀因子 (VIF)
    VIF_j = 1 / (1 - R²_j)，其中 R²_j 为第 j 个自变量对其余自变量的 OLS 回归决定系数

    Args:
        X: 自变量矩阵 (n_samples, n_features)，不含常数项
        columns: 变量名列表

    Returns:
        含 VIF 的 DataFrame
    """
    n = X.shape[1]
    vif_list = []
    for i in range(n):
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)
        X_other = sm.add_constant(X_other)
        try:
            ols = sm.OLS(y, X_other).fit()
            r_sq = ols.rsquared
            vif = 1.0 / (1.0 - r_sq) if r_sq < 1.0 else np.inf
        except Exception:
            vif = np.nan
        vif_list.append({"variable": columns[i], "VIF": vif})
    return pd.DataFrame(vif_list)


def _extract_predictor_matrix(model_result):
    """
    从模型结果中提取自变量矩阵（排除常数项），用于相关性与 VIF 计算
    ZINB 的 exog 仅含计数模型，exog_names 可能含 inflate/alpha，需按列数对齐
    """
    model = model_result.model
    exog = getattr(model, "exog", None)
    exog_names = getattr(model, "exog_names", None) or []
    if exog is None or exog.size == 0:
        return None, None
    exog = np.asarray(exog)
    n_cols = exog.shape[1]
    # 筛选计数模型变量名（排除 inflate_*, alpha），顺序与 exog 列对应
    # ZINB: exog 为计数部分，exog_names 中 inflate_const 在前、alpha 在后
    count_names = [
        n for n in exog_names
        if n not in ("alpha",) and not n.startswith("inflate_")
    ]
    # 取与 exog 列数匹配的名称（exog 第 0 列通常为 const）
    names_for_exog = count_names[:n_cols] if len(count_names) >= n_cols else [
        count_names[i] if i < len(count_names) else f"X{i}" for i in range(n_cols)
    ]
    pred_names = []
    pred_cols = []
    for i, name in enumerate(names_for_exog):
        if name in ("const", "Intercept"):
            continue
        pred_names.append(name)
        pred_cols.append(exog[:, i])
    if not pred_cols:
        return None, None
    X = np.column_stack(pred_cols)
    return X, pred_names


def _is_zinb(model_result) -> bool:
    """判断模型结果是否来自零膨胀负二项回归"""
    cls_name = type(model_result).__name__
    return "ZeroInflated" in cls_name


def compute_vuong_test(zinb_result, nb_result) -> dict:
    """
    计算 Vuong 非嵌套模型检验统计量，比较 ZINB 与 NB。

    Vuong (1989): V = (√n × m̄) / s_m，其中 m_i = llf_ZINB_i - llf_NB_i。
    V 渐近服从 N(0,1)，用于检验两模型是否等价。

    Args:
        zinb_result: ZINB 拟合结果（statsmodels）
        nb_result: NB 拟合结果（statsmodels GLM）

    Returns:
        dict: {"vuong_statistic": V, "pvalue_one_sided": p, "pvalue_two_sided": p2}
    """
    from scipy import stats as scipy_stats

    # ZINB 逐观测对数似然
    llf_zinb = zinb_result.model.loglikeobs(zinb_result.params)

    # NB 逐观测对数似然（GLM family.loglike_obs）
    fam = nb_result.model.family
    scale = getattr(nb_result, "scale", 1.0)
    llf_nb = fam.loglike_obs(
        nb_result.model.endog,
        nb_result.mu,
        var_weights=getattr(nb_result.model, "var_weights", 1.0),
        scale=scale,
    )
    if np.ndim(llf_nb) == 0:
        llf_nb = np.full_like(llf_zinb, llf_nb)
    llf_nb = np.asarray(llf_nb).ravel()

    m = llf_zinb - llf_nb
    n = len(m)
    m_bar = np.mean(m)
    s_m = np.std(m, ddof=0)
    if s_m < 1e-15:
        s_m = 1e-15
    V = np.sqrt(n) * m_bar / s_m

    # 双侧 p 值：H0 两模型等价
    p_two = 2 * (1 - scipy_stats.norm.cdf(abs(V)))
    # 单侧 p 值：H1 ZINB 优于 NB（V>0）
    p_one = 1 - scipy_stats.norm.cdf(V)

    return {
        "vuong_statistic": float(V),
        "pvalue_one_sided": float(p_one),
        "pvalue_two_sided": float(p_two),
    }


def _get_residuals(model_result):
    """统一获取残差，兼容 GLM / OLS / ZINB 等多种模型结果"""
    if hasattr(model_result, "resid_response"):
        return model_result.resid_response
    if hasattr(model_result, "resid"):
        return model_result.resid
    try:
        return model_result.model.endog - model_result.predict()
    except Exception:
        return None


def _write_param_table(f, params_dict: dict, pvalues):
    """写入 Markdown 系数表格"""
    f.write("| 变量 | 系数 | P值 | 显著性 | 解释 |\n")
    f.write("|------|------|-----|--------|------|\n")
    for param_name, param_value in params_dict.items():
        pval = pvalues[param_name] if param_name in pvalues else float("nan")
        pval_valid = pval is not None and not np.isnan(pval)
        sig = "***" if pval_valid and pval < 0.001 else "**" if pval_valid and pval < 0.01 else "*" if pval_valid and pval < 0.05 else ""
        sig_mark = "是" if pval_valid and pval < 0.05 else "否"
        pval_str = f"{pval:.4f}" if pval_valid else "—"
        coef_valid = param_value is not None and not np.isnan(param_value)
        coef_str = f"{param_value:.4f}" if coef_valid else "—"

        if "Intercept" in param_name or param_name == "const":
            interpretation = "截距项"
        elif param_name == "alpha":
            interpretation = "过散布参数（>0 表示存在过散布）"
        elif param_name.startswith("inflate_"):
            interpretation = "零膨胀 logit 系数"
        else:
            interpretation = (
                f"{param_name}每增加1个单位，被引用数的期望值变化 {coef_str}"
            )
        f.write(
            f"| {param_name} | {coef_str} | {pval_str} | "
            f"{sig} ({sig_mark}) | {interpretation} |\n"
        )
    f.write("\n")


def generate_regression_report(
    model_result,
    output_dir: Path,
    model_name: str = "regression_model",
    vuong_result: Optional[dict] = None,
) -> Path:
    """
    生成回归模型分析报告（Markdown格式）
    
    Args:
        model_result: statsmodels回归结果对象
        output_dir: 输出目录
        model_name: 模型名称
    
    Returns:
        报告文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{model_name}_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        from datetime import datetime
        f.write("# 回归模型分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 模型摘要
        f.write("## 1. 模型摘要\n\n")
        if _is_zinb(model_result):
            f.write("> **注意**: ZINB 模型的自变量已标准化（z-score），"
                    "系数反映标准化后每变化1个标准差对应的效应。\n\n")
        f.write("```\n")
        try:
            summary_str = str(model_result.summary())
            summary_str = summary_str.replace("nan", "—")
        except (ValueError, np.linalg.LinAlgError):
            summary_str = f"LL={model_result.llf:.4f}  (Hessian 奇异，无法生成完整摘要)"
        f.write(summary_str)
        f.write("\n```\n\n")
        
        # 系数解释表格
        f.write("## 2. 系数显著性表格\n\n")
        params = model_result.params
        try:
            pvalues = model_result.pvalues
        except (ValueError, np.linalg.LinAlgError):
            pvalues = pd.Series(np.nan, index=params.index)

        if _is_zinb(model_result):
            count_params = {
                k: v for k, v in params.items()
                if not k.startswith("inflate_") and k != "alpha"
            }
            inflate_params = {k: v for k, v in params.items() if k.startswith("inflate_")}
            alpha_params = {k: v for k, v in params.items() if k == "alpha"}

            f.write("### 2.1 计数模型部分 (Negative Binomial)\n\n")
            _write_param_table(f, count_params, pvalues)

            if inflate_params:
                f.write("### 2.2 零膨胀部分 (Logit)\n\n")
                f.write("零膨胀部分建模「结构性零」的概率。正系数意味着该变量增大时，"
                        "产生结构性零的概率更高。\n\n")
                _write_param_table(f, inflate_params, pvalues)

            if alpha_params:
                f.write("### 2.3 过散布参数\n\n")
                _write_param_table(f, alpha_params, pvalues)
        else:
            _write_param_table(f, dict(params), pvalues)

        f.write("\n")
        
        # 显著性标记说明
        f.write("**显著性标记说明**: *** p<0.001, ** p<0.01, * p<0.05\n\n")
        f.write("---\n\n")
        
        # 模型评估指标
        f.write("## 3. 模型评估指标\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|------|------|\n")
        if hasattr(model_result, 'llf'):
            f.write(f"| Log-Likelihood | {model_result.llf:.4f} |\n")
        f.write(f"| AIC | {model_result.aic:.4f} |\n")
        f.write(f"| BIC | {model_result.bic:.4f} |\n")
        if hasattr(model_result, 'prsquared'):
            f.write(f"| Pseudo R-squared | {model_result.prsquared:.4f} |\n")
        f.write("\n")

        # Vuong 检验（ZINB vs NB）
        if vuong_result and _is_zinb(model_result):
            f.write("### 3.1 Vuong 非嵌套模型检验 (ZINB vs NB)\n\n")
            f.write("| 统计量 | 数值 |\n")
            f.write("|--------|------|\n")
            f.write(f"| Vuong's V | {vuong_result['vuong_statistic']:.4f} |\n")
            f.write(f"| 单侧 p 值 (ZINB>NB) | {vuong_result['pvalue_one_sided']:.4f} |\n")
            f.write(f"| 双侧 p 值 | {vuong_result['pvalue_two_sided']:.4f} |\n")
            f.write("\n> V > 0 且 p < 0.05 表示 ZINB 显著优于 NB；V < 0 表示 NB 更优。\n\n")
        
        # 残差统计
        f.write("## 4. 残差统计\n\n")
        residuals = _get_residuals(model_result)
        if residuals is not None:
            f.write("| 统计量 | 数值 |\n")
            f.write("|--------|------|\n")
            f.write(f"| 残差均值 | {np.mean(residuals):.4f} |\n")
            f.write(f"| 残差标准差 | {np.std(residuals):.4f} |\n")
            f.write(f"| 残差最小值 | {np.min(residuals):.4f} |\n")
            f.write(f"| 残差最大值 | {np.max(residuals):.4f} |\n")
        else:
            f.write("（该模型类型不支持自动残差统计）\n")
        f.write("\n")

        # 皮尔逊相关系数矩阵与方差膨胀因子
        X, pred_names = _extract_predictor_matrix(model_result)
        if X is not None and pred_names and len(pred_names) > 1:
            f.write("## 5. 多重共线性诊断\n\n")
            try:
                corr = compute_pearson_correlation_matrix(X, pred_names)
                f.write("### 5.1 皮尔逊相关系数矩阵\n\n")
                f.write("|  | " + " | ".join(pred_names) + " |\n")
                f.write("|" + "---|" * (len(pred_names) + 1) + "\n")
                for i, row_name in enumerate(pred_names):
                    row_vals = [f"{corr.loc[row_name, c]:.3f}" for c in pred_names]
                    f.write(f"| {row_name} | " + " | ".join(row_vals) + " |\n")
                f.write("\n")
                vif_df = compute_vif(X, pred_names)
                f.write("### 5.2 方差膨胀因子 (VIF)\n\n")
                f.write("| 变量 | VIF | 说明 |\n")
                f.write("|------|-----|------|\n")
                for _, r in vif_df.iterrows():
                    v = r["VIF"]
                    if np.isnan(v):
                        note = "—"
                    elif v >= 10:
                        note = "⚠️ 严重共线性"
                    elif v >= 5:
                        note = "注意"
                    else:
                        note = "可接受"
                    v_str = f"{v:.2f}" if not np.isnan(v) else "—"
                    f.write(f"| {r['variable']} | {v_str} | {note} |\n")
                f.write("\n> VIF > 10 表示严重多重共线性，VIF > 5 需关注。\n\n")
            except Exception as e:
                logger.warning(f"多重共线性诊断计算失败: {e}")
                f.write("（计算失败）\n\n")
        else:
            f.write("## 5. 多重共线性诊断\n\n")
            f.write("（自变量不足或无法提取，跳过）\n\n")

        # 变量含义说明
        f.write("## 6. 变量含义说明\n\n")
        f.write("| 变量 | 含义 |\n")
        f.write("|------|------|\n")
        f.write("| New_n | 新节点标志（0或1）- 子网是否包含新技术节点 |\n")
        f.write("| New_e | 新边标志（0或1）- 子网是否包含新技术关联 |\n")
        f.write("| Min_pn | 最小专利数量 - 覆盖子网所有节点和边的最小专利数（set cover），衡量新颖性 |\n")
        f.write("| Con_n | 节点常规性 - 子网节点 Strength 的中位数 |\n")
        f.write("| Con_e | 边常规性 - 子网边 Weight 的中位数 |\n")
        f.write("| Eigen | 特征向量中心性 - 子网节点在HDKN中的平均 eigenvector centrality |\n")
        f.write("| Constraint | Burt's constraint - 子网节点 constraint 的最小值 |\n")
        f.write("| Betweenness | 中介中心度 - 子网节点在HDKN中的平均 betweenness centrality（预期系数为正） |\n")
        f.write("| Avg_Weight | 语义连贯性/技术紧密度 - 子网连边平均权重，高权重代表成熟「黄金组合」（预期系数为正） |\n")
        f.write("| Back_cite | 后向引用数（控制变量） |\n")
        f.write("| Assignee | 专利权人类型（组织=1，个人=0，控制变量） |\n")
        f.write("| Total_pat | HDKN中相同IPC类别的专利总数（控制变量） |\n")
        f.write("| Cited | 被引用数量（因变量） |\n")
        f.write("\n")
    
    logger.success(f"回归分析报告已保存到: {report_path}")
    return report_path

def plot_regression_diagnostics(
    model_result,
    output_dir: Path,
    model_name: str = "regression_model"
):
    """
    绘制回归诊断图
    
    Args:
        model_result: statsmodels回归结果对象
        output_dir: 输出目录
        model_name: 模型名称
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取数据
    fitted_values = model_result.fittedvalues
    residuals = _get_residuals(model_result)
    actual_values = model_result.model.endog

    if residuals is None:
        logger.warning("无法获取残差，跳过诊断图生成")
        return None
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('回归模型诊断图', fontsize=16, fontweight='bold')
    
    # 1. 残差分布直方图
    ax1 = axes[0, 0]
    ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('残差', fontsize=11)
    ax1.set_ylabel('频数', fontsize=11)
    ax1.set_title('残差分布直方图', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 预测值 vs 实际值散点图
    ax2 = axes[0, 1]
    ax2.scatter(actual_values, fitted_values, alpha=0.6, s=50, color='steelblue')
    # 添加对角线
    min_val = min(min(actual_values), min(fitted_values))
    max_val = max(max(actual_values), max(fitted_values))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线 (y=x)')
    ax2.set_xlabel('实际值 (Cited)', fontsize=11)
    ax2.set_ylabel('预测值', fontsize=11)
    ax2.set_title('预测值 vs 实际值散点图', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 残差 vs 预测值
    ax3 = axes[1, 0]
    ax3.scatter(fitted_values, residuals, alpha=0.6, s=50, color='coral')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, label='零线')
    ax3.set_xlabel('预测值', fontsize=11)
    ax3.set_ylabel('残差', fontsize=11)
    ax3.set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q图（正态性检验）
    ax4 = axes[1, 1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('残差Q-Q图（正态性检验）', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片到reports目录
    reports_dir = output_dir.parent / "reports" if output_dir.name == "models" else output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    plot_path = reports_dir / f"regression_diagnostics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success(f"回归诊断图已保存到: {plot_path}")
    return plot_path

def generate_combined_regression_report(
    nb_result,
    zinb_result,
    vuong_result: Optional[dict],
    output_dir: Path,
    model_name: str = "regression_model",
) -> Path:
    """
    生成 NB 与 ZINB 合并报告（单文件，两种模型放在一起）。

    Returns:
        报告文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{model_name}_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        from datetime import datetime
        f.write("# 回归模型分析报告（NB + ZINB）\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 1. 模型比较摘要
        f.write("## 1. 模型比较摘要\n\n")
        f.write("| 指标 | Negative Binomial | ZINB |\n")
        f.write("|------|-------------------|------|\n")
        f.write(f"| Log-Likelihood | {nb_result.llf:.4f} | {zinb_result.llf:.4f} |\n")
        f.write(f"| AIC | {nb_result.aic:.4f} | {zinb_result.aic:.4f} |\n")
        f.write(f"| BIC | {nb_result.bic:.4f} | {zinb_result.bic:.4f} |\n")
        if hasattr(nb_result, "prsquared") and hasattr(zinb_result, "prsquared"):
            f.write(f"| Pseudo R² | {getattr(nb_result, 'prsquared', 0):.4f} | {zinb_result.prsquared:.4f} |\n")
        f.write("\n")

        # 2. Vuong 检验
        if vuong_result:
            f.write("## 2. Vuong 非嵌套模型检验 (ZINB vs NB)\n\n")
            f.write("| 统计量 | 数值 |\n")
            f.write("|--------|------|\n")
            f.write(f"| Vuong's V | {vuong_result['vuong_statistic']:.4f} |\n")
            f.write(f"| 单侧 p 值 (ZINB>NB) | {vuong_result['pvalue_one_sided']:.4f} |\n")
            f.write(f"| 双侧 p 值 | {vuong_result['pvalue_two_sided']:.4f} |\n")
            f.write("\n> V > 0 且 p < 0.05 表示 ZINB 显著优于 NB；V < 0 表示 NB 更优。\n\n")
        f.write("---\n\n")

        # 3. NB 模型详情
        f.write("## 3. Negative Binomial 模型\n\n")
        f.write("```\n")
        f.write(str(nb_result.summary()).replace("nan", "—"))
        f.write("\n```\n\n")

        # 4. ZINB 模型详情
        f.write("## 4. ZINB 模型\n\n")
        f.write("> **注意**: ZINB 自变量已标准化（z-score），系数反映每变化1个标准差对应的效应。\n\n")
        f.write("```\n")
        try:
            f.write(str(zinb_result.summary()).replace("nan", "—"))
        except (ValueError, np.linalg.LinAlgError):
            f.write(f"LL={zinb_result.llf:.4f}  (Hessian 奇异，无法生成完整摘要)")
        f.write("\n```\n\n")

        # 5. 变量含义说明
        f.write("## 5. 变量含义说明\n\n")
        f.write("| 变量 | 含义 |\n")
        f.write("|------|------|\n")
        f.write("| New_n | 新节点标志（0或1） |\n")
        f.write("| New_e | 新边标志（0或1） |\n")
        f.write("| Min_pn | 最小专利覆盖数 |\n")
        f.write("| Con_n | 节点常规性 |\n")
        f.write("| Con_e | 边常规性 |\n")
        f.write("| Eigen | 特征向量中心性 |\n")
        f.write("| Constraint | Burt's constraint |\n")
        f.write("| Back_cite | 后向引用数（控制变量） |\n")
        f.write("| Assignee | 专利权人类型（控制变量） |\n")
        f.write("| Total_pat | 同类专利总数（控制变量） |\n")
        f.write("| Cited | 被引用数量（因变量） |\n")
        f.write("\n")

    logger.success(f"合并回归报告已保存到: {report_path}")
    return report_path


def generate_full_regression_analysis(
    model_result,
    output_dir: Path,
    model_name: str = "regression_model",
    vuong_result: Optional[dict] = None,
) -> dict:
    """
    生成完整的回归分析（报告+诊断图）
    
    Returns:
        包含报告和图片路径的字典
    """
    logger.info("生成回归模型分析...")
    
    # 报告保存到reports目录
    reports_dir = output_dir.parent / "reports" if output_dir.name == "models" else output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    report_path = generate_regression_report(
        model_result, reports_dir, model_name, vuong_result=vuong_result
    )
    plot_path = plot_regression_diagnostics(model_result, output_dir, model_name)
    
    return {
        'report_path': report_path,
        'diagnostics_plot_path': plot_path
    }

