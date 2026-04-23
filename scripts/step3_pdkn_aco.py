#!/usr/bin/env python3
"""
Step3: 基于 PDKN 做 ACO 搜索

功能：
1. 读取Step1保存的PDKN网络
2. 读取Step2保存的 objective_coefficients.json（必须）与 regression_meta（可选，用于 α）
3. 执行ACO搜索
4. 保存结果和metadata

运行方式：
    python scripts/step3_pdkn_aco.py --run-id <run_id> [--regression-dir PATH] [--aco-config PATH] [--force]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from scripts.common import init_script
init_script()

from loguru import logger
import pandas as pd

from src.patent_opportunity_analysis import config as _config
from src.patent_opportunity_analysis import aco_search as _aco_search
from src.patent_opportunity_analysis.feature_registry import FEATURE_REGISTRY
from src.patent_opportunity_analysis.utils.aco_utils import load_aco_config
from src.patent_opportunity_analysis.utils.paths import ACO_CONFIG_FILE, RAW_PATENT_FILE
from src.patent_opportunity_analysis.utils.network_io import (
    load_dkn, load_metadata, save_metadata, compute_data_hash, compute_file_hash
)
from src.patent_opportunity_analysis.utils.run_utils import (
    get_run_dir, get_step_dir, ensure_run_dirs
)
from src.patent_opportunity_analysis.aco_to_rag import (
    enrich_opportunities, build_patent_lookup, build_stem_to_originals,
)
from src.patent_opportunity_analysis.pipeline import load_patents_from_csv
from src.patent_opportunity_analysis.hdkn_feature_cache import build_hdkn_subnetwork_feature_cache

aco_search_opportunities = _aco_search.aco_search_opportunities
HIST_END_YEAR = _config.HIST_END_YEAR
SUBNETWORK_SIZE = _config.SUBNETWORK_SIZE
TOP_K_OPPORTUNITIES = _config.TOP_K_OPPORTUNITIES

def step3_pdkn_aco(
    run_id: str,
    networks_dir: Path = None,
    regression_dir: Path = None,
    aco_config_file: Path = None,
    patents_csv: Path = None,
    domain_field: str = "embodied intelligence",
    force: bool = False,
    # 测试模式参数覆盖（仅用于测试，不改变算法逻辑）
    test_num_ants: int = None,
    test_num_generations: int = None,
    test_top_k: int = None,
    test_subnetwork_size: int = None
) -> Path:
    """
    执行Step3: 基于PDKN做ACO搜索

    Args:
        run_id: 运行ID（必传）
        networks_dir: 网络目录路径（None则自动推断）
        regression_dir: 回归目录路径（None则自动推断）
        aco_config_file: ACO配置文件路径（None则使用默认）
        patents_csv: 专利 CSV 路径（用于富化输出中的专利标题/摘要；None 则使用默认路径）
        domain_field: 领域名称（写入富化输出的 domain_context.field）
        force: 是否强制重算

    Returns:
        ACO目录路径
    """
    logger.info("=" * 80)
    logger.info("🐜 Step3: 基于 PDKN 做 ACO 搜索")
    logger.info("=" * 80)
    
    # 1. 确定运行目录和步骤目录
    run_dir = get_run_dir(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"运行目录不存在: {run_dir}，请先运行Step1")
    
    dirs = ensure_run_dirs(run_dir)
    networks_dir = networks_dir or dirs["networks_dir"]
    regression_dir = regression_dir or dirs["regression_dir"]
    aco_dir = dirs["aco_dir"]
    
    # 2. 检查Step1产物
    pdkn_path = networks_dir / "pdkn.pkl.gz"
    networks_meta_path = networks_dir / "networks_meta.json"
    
    if not pdkn_path.exists():
        raise FileNotFoundError(f"PDKN文件不存在: {pdkn_path}，请先运行Step1")
    if not networks_meta_path.exists():
        raise FileNotFoundError(f"网络metadata不存在: {networks_meta_path}，请先运行Step1")
    
    # 3. 加载Step1的metadata
    networks_meta = load_metadata(networks_meta_path)
    hist_end_year = networks_meta["hist_end_year"]
    max_year = networks_meta["max_year"]
    logger.info(f"📋 Step1 metadata: hist_end_year={hist_end_year}, max_year={max_year}")
    
    # 4. 加载PDKN并验证
    logger.info("\n📂 加载PDKN网络...")
    PDKN = load_dkn(pdkn_path)
    PDKN.assert_kind("PDKN")
    PDKN.assert_invariants()
    
    # 验证ref_year
    if PDKN.ref_year != max_year:
        raise ValueError(
            f"PDKN ref_year ({PDKN.ref_year}) != max_year ({max_year})。"
            f"这可能导致ACO搜索使用错误的参考年份。"
        )
    
    logger.success(f"✅ PDKN加载成功: {PDKN}")
    
    # 5. 加载 Step2 线性目标系数（必须）与回归 metadata（衰减 α，可选）
    aco_config_file = aco_config_file or ACO_CONFIG_FILE
    aco_config_pre = load_aco_config(aco_config_file)

    if not regression_dir.exists():
        raise FileNotFoundError(
            f"回归目录不存在: {regression_dir}。请先运行 Step2 生成 objective_coefficients.json。"
        )
    objective_path = regression_dir / "objective_coefficients.json"
    if not objective_path.exists():
        raise FileNotFoundError(
            f"缺少 objective_coefficients.json: {objective_path}。"
            f"请使用当前版本 Step2 重新运行回归（会写出 NB 显著项系数）。"
        )
    with open(objective_path, "r", encoding="utf-8") as f:
        objective_coefficients = json.load(f)
    if not isinstance(objective_coefficients, dict):
        raise ValueError(f"objective_coefficients.json 应为对象/dict，实际: {type(objective_coefficients)}")
    objective_coefficients = {str(k): float(v) for k, v in objective_coefficients.items()}
    logger.info(f"📂 已加载线性目标系数: {objective_path}（{len(objective_coefficients)} 项）")

    regression_meta = None
    regression_meta_path = regression_dir / "regression_meta.json"
    if regression_meta_path.exists():
        regression_meta = load_metadata(regression_meta_path)
        logger.info(f"📋 Step2 metadata: model_type={regression_meta.get('model_type')}")

    # 6. 加载ACO配置
    aco_config = aco_config_pre
    subnetwork_size = aco_config.get("subnetwork_size", SUBNETWORK_SIZE)
    top_k = aco_config.get("top_k_opportunities", TOP_K_OPPORTUNITIES)
    
    override_coefficients = aco_config.get("override_coefficients")
    if override_coefficients is not None and isinstance(override_coefficients, dict) and override_coefficients:
        logger.info(f"📋 调试 override_coefficients 已启用，将覆盖/增补 JSON 系数: {list(override_coefficients.keys())}")
        override_coefficients = {str(k): float(v) for k, v in override_coefficients.items()}
    else:
        override_coefficients = None

    decay_factor = None
    if regression_meta is not None:
        decay_factor = regression_meta.get("decay_factor")
    if decay_factor is not None:
        logger.info(f"📋 使用 Step2 decay_factor={decay_factor} 构建 HDKN 统计缓存（与回归对齐）")
    
    # 测试模式参数覆盖（优先使用测试参数）
    if test_subnetwork_size is not None:
        subnetwork_size = test_subnetwork_size
        logger.info(f"🧪 测试模式: 覆盖subnetwork_size={subnetwork_size}")
    if test_top_k is not None:
        top_k = test_top_k
        logger.info(f"🧪 测试模式: 覆盖top_k={top_k}")
    
    logger.info(f"⚙️  ACO配置: subnetwork_size={subnetwork_size}, top_k={top_k}")
    
    # 测试模式下的ACO参数（用于覆盖全局变量）
    test_aco_params = {}
    if test_num_ants is not None:
        test_aco_params['num_ants'] = test_num_ants
        logger.info(f"🧪 测试模式: 覆盖num_ants={test_num_ants}")
    if test_num_generations is not None:
        test_aco_params['num_generations'] = test_num_generations
        logger.info(f"🧪 测试模式: 覆盖num_generations={test_num_generations}")
    
    # 7. 检查是否已存在产物（resume检查）
    aco_meta_path = aco_dir / "aco_meta.json"
    if not force and aco_meta_path.exists():
        logger.info("📋 检测到已存在的ACO产物，检查metadata一致性...")
        try:
            existing_meta = load_metadata(aco_meta_path)
            upstream_hash = existing_meta.get("upstream_artifacts", {}).get("networks_meta_hash")
            if upstream_hash == compute_data_hash(networks_meta):
                logger.success("✅ 产物已存在且metadata一致，跳过ACO搜索（使用 --force 强制重算）")
                logger.info(f"📁 ACO目录: {aco_dir}")
                return aco_dir
        except Exception as e:
            logger.warning(f"⚠️  读取现有metadata失败: {e}")
            if not force:
                raise
    
    # 8. 加载HDKN（ACO evaluate_solution 需要 HDKN 计算 Constraint 特征）
    hdkn_path = networks_dir / "hdkn.pkl.gz"
    if hdkn_path.exists():
        logger.info("\n📂 加载HDKN网络（用于 ACO Constraint 特征计算）...")
        HDKN = load_dkn(hdkn_path)
        HDKN.assert_kind("HDKN")
        logger.success(f"✅ HDKN加载成功: {HDKN}")
    else:
        logger.warning("⚠️  HDKN文件不存在，Constraint 特征将不可用")
        from src.patent_opportunity_analysis.utils.dkn_wrapper import DKNNetwork
        import networkx as nx
        HDKN = DKNNetwork(
            kind="HDKN", graph=nx.Graph(),
            ref_year=hist_end_year, hist_end_year=hist_end_year
        )
    
    # 9. 执行ACO搜索
    logger.info("\n🐜 执行ACO搜索...")
    logger.info("⚠️  关键：使用PDKN进行搜索，HDKN用于Constraint特征计算")
    
    # 临时覆盖ACO全局参数（仅用于测试模式）
    original_aco_params = {}
    if test_aco_params:
        import src.patent_opportunity_analysis.aco_search as aco_module
        if 'num_ants' in test_aco_params:
            original_aco_params['ACO_NUM_ANTS'] = aco_module.ACO_NUM_ANTS
            aco_module.ACO_NUM_ANTS = test_aco_params['num_ants']
        if 'num_generations' in test_aco_params:
            original_aco_params['ACO_NUM_GENERATIONS'] = aco_module.ACO_NUM_GENERATIONS
            aco_module.ACO_NUM_GENERATIONS = test_aco_params['num_generations']
    
    try:
        opportunities = aco_search_opportunities(
            HDKN=HDKN,
            PDKN=PDKN,
            subnetwork_size=subnetwork_size,
            top_k=top_k,
            output_dir=aco_dir,
            objective_coefficients=objective_coefficients,
            override_coefficients=override_coefficients,
            hist_end_year=hist_end_year,
            pdkn_ref_year=PDKN.ref_year,
            decay_factor=decay_factor,
        )
    finally:
        # 恢复原始ACO参数
        if original_aco_params:
            import src.patent_opportunity_analysis.aco_search as aco_module
            for key, value in original_aco_params.items():
                setattr(aco_module, key, value)
    
    logger.success(f"✅ 找到 {len(opportunities)} 个技术机会")

    # ---------- 输出路径 ----------
    opp_csv_path = None
    opp_json_path = None
    enriched_json_path = None

    # 10. 保存原始结果（简略格式，向后兼容）
    if opportunities:
        opp_df = pd.DataFrame([
            {
                'rank': i + 1,
                'nodes': ', '.join(opp['nodes']),
                'score': opp['score'],
                'size': opp['size']
            }
            for i, opp in enumerate(opportunities)
        ])
        opp_csv_path = aco_dir / "aco_topk.csv"
        opp_df.to_csv(opp_csv_path, index=False)
        logger.success(f"💾 机会列表已保存: {opp_csv_path}")

        opp_json_path = aco_dir / "aco_topk.json"
        with open(opp_json_path, 'w', encoding='utf-8') as f:
            json.dump(opportunities, f, indent=2, ensure_ascii=False, default=str)
        logger.success(f"💾 机会详情已保存: {opp_json_path}")

    # 11. 富化输出（模版格式，含节点/边分类、专利信息、新颖源/可行锚点）
    if opportunities:
        logger.info("\n📝 生成富化子网 JSON ...")

        # 构建 HDKN 特征缓存（与 ACO evaluate 同口径）
        feature_names = list(objective_coefficients.keys()) or list(FEATURE_REGISTRY.keys())
        hdkn_cache = build_hdkn_subnetwork_feature_cache(
            HDKN,
            hist_end_year=hist_end_year,
            decay_factor=decay_factor,
            selected_features=feature_names,
        )

        linear_obj = _aco_search.LinearObjectiveFunction(objective_coefficients)

        # 加载专利元数据（标题/摘要/引用数）
        _csv = patents_csv or RAW_PATENT_FILE
        patent_lookup = {}
        stem_to_originals = {}
        if Path(_csv).exists():
            try:
                patents_list = load_patents_from_csv(Path(_csv), smart_select=False)
                patent_lookup = build_patent_lookup(patents_list)
                stem_to_originals = build_stem_to_originals(patents_list)
                logger.info(f"📂 已加载 {len(patent_lookup)} 条专利元数据用于富化")
            except Exception as e:
                logger.warning(f"⚠️  加载专利 CSV 失败，富化输出将缺少标题/摘要: {e}")
        else:
            logger.warning(f"⚠️  专利 CSV 不存在: {_csv}，富化输出将缺少标题/摘要")

        enriched = enrich_opportunities(
            opportunities,
            PDKN=PDKN,
            HDKN=HDKN,
            hdkn_cache=hdkn_cache,
            feature_names=feature_names,
            linear_objective=linear_obj,
            pdkn_ref_year=PDKN.ref_year,
            patent_lookup=patent_lookup,
            stem_to_originals=stem_to_originals,
            domain_field=domain_field,
        )

        enriched_json_path = aco_dir / "aco_topk_enriched.json"
        with open(enriched_json_path, 'w', encoding='utf-8') as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False, default=str)
        logger.success(f"💾 富化子网 JSON 已保存: {enriched_json_path}")

    # 12. 保存运行日志
    run_log = {
        "run_id": run_id,
        "subnetwork_size": subnetwork_size,
        "top_k": top_k,
        "opportunities_count": len(opportunities),
        "used_objective_coefficients": True,
        "used_override_coefficients": override_coefficients is not None,
        "objective_coefficients_path": str(objective_path),
        "config_file": str(aco_config_file),
        "config_hash": compute_file_hash(aco_config_file) if aco_config_file.exists() else ""
    }
    run_log_path = aco_dir / "aco_run_log.json"
    with open(run_log_path, 'w', encoding='utf-8') as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)
    logger.success(f"💾 运行日志已保存: {run_log_path}")

    # 13. 创建并保存 metadata
    aco_meta = {
        "step_name": "03_aco",
        "created_at": datetime.now().isoformat(),
        "run_id": run_id,
        "subnetwork_size": subnetwork_size,
        "top_k": top_k,
        "opportunities_count": len(opportunities),
        "used_objective_coefficients": True,
        "used_override_coefficients": override_coefficients is not None,
        "objective_coefficients_path": str(objective_path),
        "upstream_artifacts": {
            "networks_dir": str(networks_dir),
            "networks_meta_path": str(networks_meta_path),
            "networks_meta_hash": compute_data_hash(networks_meta),
            "regression_dir": str(regression_dir) if regression_dir.exists() else None,
            "regression_meta_hash": compute_data_hash(regression_meta) if regression_meta else None
        },
        "aco_config": {
            "file": str(aco_config_file),
            "hash": compute_file_hash(aco_config_file) if aco_config_file.exists() else "",
            "subnetwork_size": subnetwork_size,
            "top_k": top_k,
            "override_coefficients": override_coefficients
        },
        "output_files": {
            "topk_csv": str(opp_csv_path) if opp_csv_path else None,
            "topk_json": str(opp_json_path) if opp_json_path else None,
            "topk_enriched_json": str(enriched_json_path) if enriched_json_path else None,
            "run_log": str(run_log_path)
        }
    }
    save_metadata(aco_meta, aco_meta_path)

    logger.success(f"\n✅ Step3 完成！")
    logger.info(f"📁 ACO目录: {aco_dir}")
    logger.info(f"   - 机会列表: {opp_csv_path or 'N/A'}")
    logger.info(f"   - 富化 JSON: {enriched_json_path or 'N/A'}")
    logger.info(f"   - 运行日志: {run_log_path}")
    logger.info(f"   - Metadata: {aco_meta_path}")

    return aco_dir


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Step3: 基于 PDKN 做 ACO 搜索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/step3_pdkn_aco.py --run-id 20240203_120000
  python scripts/step3_pdkn_aco.py --run-id 20240203_120000 --aco-config configs/my_aco_config.yaml
  python scripts/step3_pdkn_aco.py --run-id 20240203_120000 --force
        """
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        required=True,
        help='运行ID（必传，来自Step1）'
    )
    parser.add_argument(
        '--networks-dir',
        type=Path,
        default=None,
        help='网络目录路径（默认: outputs/runs/<run_id>/01_networks）'
    )
    parser.add_argument(
        '--regression-dir',
        type=Path,
        default=None,
        help='回归目录路径（默认: outputs/runs/<run_id>/02_regression）'
    )
    parser.add_argument(
        '--aco-config',
        type=Path,
        default=None,
        help='ACO配置文件路径（默认: configs/aco_config.yaml）'
    )
    parser.add_argument(
        '--patents-csv',
        type=Path,
        default=None,
        help=f'专利CSV路径（用于富化输出中的标题/摘要，默认: {RAW_PATENT_FILE}）'
    )
    parser.add_argument(
        '--domain-field',
        type=str,
        default='embodied intelligence',
        help='领域名称（写入富化输出的 domain_context.field，默认: embodied intelligence）'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重算，即使产物已存在'
    )
    parser.add_argument(
        '--test-num-ants',
        type=int,
        default=None,
        help='测试模式：覆盖蚂蚁数量（仅用于测试）'
    )
    parser.add_argument(
        '--test-num-generations',
        type=int,
        default=None,
        help='测试模式：覆盖迭代代数（仅用于测试）'
    )
    parser.add_argument(
        '--test-top-k',
        type=int,
        default=None,
        help='测试模式：覆盖top_k（仅用于测试）'
    )
    parser.add_argument(
        '--test-subnetwork-size',
        type=int,
        default=None,
        help='测试模式：覆盖子网络大小（仅用于测试）'
    )
    
    args = parser.parse_args()
    
    try:
        aco_dir = step3_pdkn_aco(
            run_id=args.run_id,
            networks_dir=args.networks_dir,
            regression_dir=args.regression_dir,
            aco_config_file=args.aco_config,
            patents_csv=args.patents_csv,
            domain_field=args.domain_field,
            force=args.force,
            test_num_ants=args.test_num_ants,
            test_num_generations=args.test_num_generations,
            test_top_k=args.test_top_k,
            test_subnetwork_size=args.test_subnetwork_size
        )
        logger.success(f"\n🎉 Step3 成功完成！ACO目录: {aco_dir}")
        return 0
    except Exception as e:
        logger.exception("Step3 执行失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
