#!/usr/bin/env python3
"""
Step2: 基于 HDKN 做回归分析

功能：
1. 读取Step1保存的HDKN网络
2. 加载专利数据（仅用于特征提取，不重新构建网络）
3. 基于HDKN提取回归特征
4. 训练回归模型
5. 保存模型、系数、报告和metadata

运行方式：
    python scripts/step2_hdkn_regression.py --run-id <run_id> [--patents-csv PATH] [--model-type TYPE] [--force]

可复现性：为保证与 Alpha Selection 的 LL 一致，建议使用 PYTHONHASHSEED=42。
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from scripts.common import init_script, parse_comma_list
init_script()

from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

from src.patent_opportunity_analysis import config as _config
from src.patent_opportunity_analysis import pipeline as _pipeline
from src.patent_opportunity_analysis import regression_flow as _regression_flow
from src.patent_opportunity_analysis.utils.paths import RAW_PATENT_FILE
from src.patent_opportunity_analysis.utils.network_io import (
    load_dkn, load_metadata, save_metadata, compute_file_hash, compute_data_hash
)
from src.patent_opportunity_analysis.utils.run_utils import (
    get_run_dir, get_step_dir, ensure_run_dirs
)
from src.patent_opportunity_analysis.utils import regression_analysis as _regression_analysis
from src.patent_opportunity_analysis import dkn_builder as _dkn_builder
from src.patent_opportunity_analysis.regression_model import extract_nb_significant_coefficients

load_patents_from_csv = _pipeline.load_patents_from_csv
compute_time_decay_weights = _dkn_builder.compute_time_decay_weights
generate_combined_regression_report = _regression_analysis.generate_combined_regression_report


def step2_hdkn_regression(
    run_id: str,
    patents_csv: Path = None,
    force: bool = False,
    networks_dir: Path = None,
    allow_rebuild_network: bool = False,  # Guard: 分步模式下不允许重建网络
    selected_features: Optional[List[str]] = None,
    features_config_file: Optional[Path] = None,
    include_control_vars: Optional[bool] = None,
    decay_factor: Optional[float] = None,  # α 衰减因子，指定时将重算 HDKN 权重并强制重建统计缓存
    limit: Optional[int] = None,
) -> Path:
    """
    执行Step2: 基于HDKN做回归分析
    
    Args:
        run_id: 运行ID（必传）
        patents_csv: 专利数据CSV文件路径（None则从metadata推断或使用默认）
        （已固定为 NB+ZINB 双模型）
        force: 是否强制重算
        networks_dir: 网络目录路径（None则自动推断）
        
    Returns:
        回归目录路径
    """
    logger.info("=" * 80)
    logger.info("📈 Step2: 基于 HDKN 做回归分析")
    logger.info("=" * 80)
    
    # Guard: 分步模式下不允许重建网络
    if not allow_rebuild_network and patents_csv is not None:
        logger.warning("⚠️  分步模式下，patents_csv参数仅用于特征提取，不会重建网络")
        logger.warning("   如果确实需要重建网络，请先运行Step1")
    
    # 确定特征选择（优先级：CLI参数 > YAML配置 > 默认全部）
    features_config_file = features_config_file or project_root / "configs" / "features.yaml"
    if features_config_file.exists():
        try:
            with open(features_config_file, 'r') as f:
                config = yaml.safe_load(f)
            if selected_features is None:
                selected_features = config.get('regression_features')
                if selected_features:
                    logger.info(f"从配置文件读取特征选择: {selected_features}")
            if include_control_vars is None and 'include_control_vars' in config:
                include_control_vars = config.get('include_control_vars', True)
        except Exception as e:
            logger.warning(f"读取特征配置文件失败: {e}")
            if selected_features is None:
                selected_features = None
    
    if selected_features:
        logger.info(f"将使用以下特征: {selected_features}")
    else:
        logger.info("将使用所有可用特征")

    if include_control_vars is None:
        include_control_vars = True
    logger.info(f"包含控制变量: {include_control_vars}")
    
    # 1. 确定运行目录和步骤目录
    run_dir = get_run_dir(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"运行目录不存在: {run_dir}，请先运行Step1")
    
    dirs = ensure_run_dirs(run_dir)
    networks_dir = networks_dir or dirs["networks_dir"]
    regression_dir = dirs["regression_dir"]
    
    # 2. 检查Step1产物
    hdkn_path = networks_dir / "hdkn.pkl.gz"
    networks_meta_path = networks_dir / "networks_meta.json"
    
    if not hdkn_path.exists():
        raise FileNotFoundError(f"HDKN文件不存在: {hdkn_path}，请先运行Step1")
    if not networks_meta_path.exists():
        raise FileNotFoundError(f"网络metadata不存在: {networks_meta_path}，请先运行Step1")
    
    # 3. 加载Step1的metadata
    networks_meta = load_metadata(networks_meta_path)
    hist_end_year = networks_meta["hist_end_year"]
    logger.info(f"📋 Step1 metadata: hist_end_year={hist_end_year}, run_id={networks_meta['run_id']}")
    
    # 临时设置HIST_END_YEAR（用于extract_features_for_regression函数）
    # 注意：extract_features_for_regression内部使用全局HIST_END_YEAR
    original_hist_end_year = getattr(_config, 'HIST_END_YEAR', None)
    _config.HIST_END_YEAR = hist_end_year
    logger.info(f"⚙️  临时设置HIST_END_YEAR={hist_end_year}（从Step1 metadata读取）")
    
    pbar = None
    try:
        # 4. 加载HDKN并验证
        logger.info("\n📂 加载HDKN网络...")
        HDKN = load_dkn(hdkn_path)
        HDKN.assert_kind("HDKN")
        HDKN.assert_invariants()
        
        # 验证ref_year
        if HDKN.ref_year != hist_end_year:
            raise ValueError(
                f"HDKN ref_year ({HDKN.ref_year}) != hist_end_year ({hist_end_year})。"
                f"这可能导致回归分析使用错误的参考年份。"
            )
        
        logger.success(f"✅ HDKN加载成功: {HDKN}")

        # 4.5 确定 α：优先复用 Alpha Selection 的 best_alpha
        # 注意：α 用于 HDKN 权重重算，与 Alpha Selection 一致可减少 LL 波动。
        # best_alpha 优先用于权重重算（即使特征集不匹配）；模型复用仅当特征完全匹配时进行。
        ref_year = hist_end_year
        alpha = decay_factor if decay_factor is not None else getattr(_config, 'DECAY_FACTOR', 0.9)
        alpha_dir = run_dir / "02_1_alpha_selection"
        as_meta_path = alpha_dir / "alpha_selection_results.json"
        as_reuse = None
        as_meta = None
        if as_meta_path.exists() and decay_factor is None:
            try:
                as_meta = load_metadata(as_meta_path)
                best_alpha = as_meta.get("best_alpha")
                as_feats = set(as_meta.get("x_vars", []))
                feat_match = as_feats == set(selected_features or [])
                # 权重重算：有 best_alpha 即使用，减少与 Alpha Selection 的 LL 波动
                if best_alpha is not None:
                    alpha = best_alpha
                    logger.info(f"📋 复用 Alpha Selection 最优 α = {alpha}")
                # 模型复用：仅当特征完全匹配时
                if best_alpha is not None and feat_match:
                    step2_reuse = as_meta.get("step2_reuse")
                    if step2_reuse:
                        nb_src = Path(step2_reuse.get("model_nb", ""))
                        zinb_src = Path(step2_reuse.get("model_zinb", ""))
                        feat_src = Path(step2_reuse.get("features", ""))
                        if nb_src.exists() and zinb_src.exists():
                            as_reuse = {"nb": nb_src, "zinb": zinb_src, "feat": feat_src}
                elif best_alpha is not None and not feat_match:
                    logger.info(f"   特征集不匹配（AS: {sorted(as_feats)}, Step2: {sorted(selected_features or [])}），不复用模型，但已用 best_alpha 重算权重")
            except Exception as e:
                logger.warning(f"读取 Alpha Selection 结果失败: {e}")

        logger.info(f"⚙️  重算 HDKN 时间衰减权重（参考年={ref_year}，α={alpha}）...")
        hdkn_graph = HDKN.graph if hasattr(HDKN, 'graph') else HDKN
        compute_time_decay_weights(hdkn_graph, total_year=ref_year, alpha=alpha, expected_ref_year=ref_year)
        logger.success("✅ 权重重算完成")

        # 5. 检查是否已存在产物（resume检查）
        regression_meta_path = regression_dir / "regression_meta.json"
        if not force and regression_meta_path.exists():
            logger.info("📋 检测到已存在的回归产物，检查metadata一致性...")
            try:
                existing_meta = load_metadata(regression_meta_path)
                if existing_meta.get("upstream_artifacts", {}).get("networks_meta_hash") == compute_data_hash(networks_meta):
                    logger.success("✅ 产物已存在且metadata一致，跳过回归分析（使用 --force 强制重算）")
                    logger.info(f"📁 回归目录: {regression_dir}")
                    return regression_dir
            except Exception as e:
                logger.warning(f"⚠️  读取现有metadata失败: {e}")
                if not force:
                    raise
        
        # 6. 初始化进度条（覆盖核心耗时步骤）
        pbar = tqdm(total=4, desc="Step2进度", unit="步")
        
        # 7. 加载专利数据（仅用于特征提取，不重新构建网络）
        if patents_csv is None:
            # 尝试从metadata推断
            input_data_hash = networks_meta.get("input_data_hash", "")
            # 默认使用标准路径
            patents_csv = RAW_PATENT_FILE
            if not patents_csv.exists():
                raise FileNotFoundError(f"无法找到专利数据文件，请使用 --patents-csv 指定")
        
        logger.info(f"\n📂 加载专利数据（仅用于特征提取）: {patents_csv}")
        logger.info("⚠️  注意：不会重新构建网络，仅使用数据提取特征")
        
        patents = load_patents_from_csv(patents_csv, limit=limit, smart_select=False)
        logger.success(f"✅ 成功加载 {len(patents)} 条专利")
        pbar.update(1)
        
        # 8. 共享流程：提取特征 + 拟合 NB 与 ZINB（或复用 Alpha Selection 结果）
        if as_reuse:
            logger.info("\n📋 复用 Alpha Selection 的回归结果...")
            import shutil
            features_path = regression_dir / "regression_features.csv"
            nb_path = regression_dir / "regression_model_NegativeBinomial.pkl"
            zinb_path = regression_dir / "regression_model_ZINB.pkl"
            shutil.copy2(as_reuse["feat"], features_path)
            shutil.copy2(as_reuse["nb"], nb_path)
            shutil.copy2(as_reuse["zinb"], zinb_path)
            with open(nb_path, "rb") as f:
                nb_result = pickle.load(f)
            with open(zinb_path, "rb") as f:
                zinb_result = pickle.load(f)
            df = pd.read_csv(features_path)
            used_features = selected_features or list(as_meta.get("x_vars", []))
            vuong_result = None
            logger.success("✅ 已从 Alpha Selection 加载模型与特征")
        else:
            logger.info("\n📊 提取回归特征并拟合 NB + ZINB（共享流程）...")
            from src.patent_opportunity_analysis.utils.dkn_wrapper import DKNNetwork
            import networkx as nx
            empty_pdkn = DKNNetwork(
                kind="PDKN",
                graph=nx.Graph(),
                ref_year=networks_meta["max_year"],
                hist_end_year=hist_end_year
            )

            # 图权重已重算，必须强制重建 HDKN 统计，否则会误用旧缓存导致与 Alpha Selection 的 LL 不一致
            df, nb_result, zinb_result, vuong_result, used_features = _regression_flow.run_regression_flow(
                HDKN, empty_pdkn, patents,
                selected_features=selected_features,
                include_control_vars=include_control_vars,
                decay_factor=alpha,
                force_rebuild_hdkn_stats=True,
            )
            features_path = regression_dir / "regression_features.csv"
            df.to_csv(features_path, index=False)
        pbar.update(1)

        features_path = regression_dir / "regression_features.csv"
        nb_path = regression_dir / "regression_model_NegativeBinomial.pkl"
        zinb_path = regression_dir / "regression_model_ZINB.pkl"
        logger.success(f"💾 特征数据已保存: {features_path}")

        if not as_reuse:
            with open(nb_path, 'wb') as f:
                pickle.dump(nb_result, f)
            with open(zinb_path, 'wb') as f:
                pickle.dump(zinb_result, f)
        logger.success(f"💾 模型已保存: {nb_path}, {zinb_path}")
        pbar.update(1)

        coeff_path = None
        objective_coef_path = regression_dir / "objective_coefficients.json"
        obj_coef = extract_nb_significant_coefficients(nb_result)
        with open(objective_coef_path, "w", encoding="utf-8") as f:
            json.dump(obj_coef, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.success(f"💾 ACO 线性目标系数（NB 显著项，原始尺度）: {objective_coef_path}")
        if not obj_coef:
            logger.warning("⚠️  objective_coefficients 为空（无 p<0.05 的子网特征或样本问题），Step3 ACO 的 Z 将恒为 0")

        if hasattr(zinb_result, 'params'):
            params = zinb_result.params
            try:
                pvals = zinb_result.pvalues
            except (ValueError, np.linalg.LinAlgError):
                pvals = None
            if hasattr(params, 'index'):
                var_names = list(params.index)
                coef_values = list(params.values)
            else:
                var_names = (
                    getattr(zinb_result.model, 'exog_names', [])
                    + getattr(zinb_result.model, 'exog_infl_names', [])
                    + ["alpha"]
                )[:len(params)]
                coef_values = list(params)
            pval_values = list(pvals) if pvals is not None else [None] * len(var_names)
            coeff_df = pd.DataFrame({
                'variable': var_names,
                'coefficient': coef_values,
                'pvalue': pval_values[:len(var_names)]
            })
            coeff_path = regression_dir / "regression_coefficients.csv"
            coeff_df.to_csv(coeff_path, index=False)
            logger.success(f"💾 系数已保存: {coeff_path}")

        reports_dir = regression_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        report_path = reports_dir / "regression_model_report.md"
        try:
            generate_combined_regression_report(
                nb_result, zinb_result, vuong_result,
                reports_dir,
                model_name="regression_model",
            )
            logger.success(f"💾 合并回归报告已生成: {report_path}")
        except Exception as e:
            logger.warning(f"生成回归报告时出错: {e}")
        
        llf_nb = float(getattr(nb_result, "llf", float("nan")))
        llf_zinb = float(getattr(zinb_result, "llf", float("nan")))
        regression_meta = {
            "step_name": "02_regression",
            "created_at": datetime.now().isoformat(),
            "run_id": run_id,
            "model_type": "NB+ZINB",
            "hist_end_year": hist_end_year,
            "decay_factor": alpha,
            "log_likelihood_nb": llf_nb,
            "log_likelihood_zinb": llf_zinb,
            "sample_count": len(df),
            "features_count": len(df.columns),
            "selected_features": used_features,
            "upstream_artifacts": {
                "networks_dir": str(networks_dir),
                "networks_meta_path": str(networks_meta_path),
                "networks_meta_hash": compute_data_hash(networks_meta)
            },
            "output_files": {
                "features": str(features_path),
                "model": str(zinb_path),  # Step3 默认使用 ZINB
                "model_nb": str(nb_path),
                "model_zinb": str(zinb_path),
                "coefficients": str(coeff_path) if coeff_path else None,
                "objective_coefficients": str(objective_coef_path),
                "report": str(report_path)
            }
        }
        save_metadata(regression_meta, regression_meta_path)
        pbar.update(1)

        logger.success(f"\n✅ Step2 完成！")
        logger.info(f"📁 回归目录: {regression_dir}")
        logger.info(f"   - 特征数据: {features_path}")
        logger.info(f"   - 模型: {nb_path}, {zinb_path}")
        logger.info(f"   - 系数: {coeff_path}")
        logger.info(f"   - 报告: {report_path}")
        logger.info(f"   - Metadata: {regression_meta_path}")
        
        return regression_dir
    finally:
        if pbar is not None:
            pbar.close()
        # 恢复原始HIST_END_YEAR
        if original_hist_end_year is not None:
            _config.HIST_END_YEAR = original_hist_end_year


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Step2: 基于 HDKN 做回归分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/step2_hdkn_regression.py --run-id 20240203_120000
  python scripts/step2_hdkn_regression.py --run-id 20240203_120000 --model-type Poisson
  python scripts/step2_hdkn_regression.py --run-id 20240203_120000 --features New_n,Min_pn,Con_n,Eigen
  python scripts/step2_hdkn_regression.py --run-id 20240203_120000 --force
        """
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        required=True,
        help='运行ID（必传，来自Step1）'
    )
    parser.add_argument(
        '--patents-csv',
        type=Path,
        default=None,
        help='专利数据CSV文件路径（默认: 从Step1 metadata推断或使用默认路径）'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重算，即使产物已存在'
    )
    parser.add_argument(
        '--networks-dir',
        type=Path,
        default=None,
        help='网络目录路径（默认: outputs/runs/<run_id>/01_networks）'
    )
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help='要使用的特征列表（逗号分隔，如: New_n,Min_pn,Con_n,Eigen）。优先级高于YAML配置'
    )
    parser.add_argument(
        '--features-config',
        type=Path,
        default=None,
        help='特征配置文件路径（默认: configs/features.yaml）'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='衰减因子 α（如 0.9）。指定时将重算 HDKN 权重并强制重建统计缓存'
    )
    parser.add_argument(
        '--no-control-vars',
        action='store_true',
        help='不包含控制变量（Back_cite, Assignee, Total_pat）'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='只使用前N条专利（用于快速测试，默认不限制）'
    )

    args = parser.parse_args()

    selected_features = parse_comma_list(args.features)
    if selected_features:
        logger.info(f"从CLI参数读取特征选择: {selected_features}")

    include_control_vars = not args.no_control_vars
    if args.no_control_vars:
        logger.info("不包含控制变量（--no-control-vars）")

    try:
        regression_dir = step2_hdkn_regression(
            run_id=args.run_id,
            patents_csv=args.patents_csv,
            force=args.force,
            networks_dir=args.networks_dir,
            selected_features=selected_features,
            features_config_file=args.features_config,
            decay_factor=args.alpha,
            include_control_vars=include_control_vars,
            limit=args.limit,
        )
        logger.success(f"\n🎉 Step2 成功完成！回归目录: {regression_dir}")
        return 0
    except Exception as e:
        logger.exception("Step2 执行失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
