#!/usr/bin/env python3
"""
Pipeline编排器：按顺序执行Step1->Step2->Step3->Step4

功能：
- 按顺序调用四个步骤的内部函数
- 支持resume机制（跳过已完成的步骤）
- 支持强制重算特定步骤

运行方式：
    python scripts/run_all.py [--run-id ID] [--limit N] [--resume] [--force-step1] [--force-step2] [--force-step3] [--force-step4]

    快速测试（替代原test500系列脚本）：
    python scripts/run_all.py --limit 500
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from scripts.common import init_script
init_script()

from loguru import logger
from src.patent_opportunity_analysis.utils.paths import RAW_PATENT_FILE, ACO_CONFIG_FILE
from src.patent_opportunity_analysis.utils.run_utils import get_run_dir, ensure_run_dirs

# 导入步骤函数
from scripts.step1_build_networks import step1_build_networks
from scripts.step2_hdkn_regression import step2_hdkn_regression
from scripts.step3_pdkn_aco import step3_pdkn_aco
from scripts.step4_rag_report import step4_rag_report

def run_all_pipeline(
    patents_csv: Path = None,
    hist_end_year: int = None,
    max_year: int = None,
    run_id: str = None,
    resume: bool = True,
    force_step1: bool = False,
    force_step2: bool = False,
    force_step3: bool = False,
    force_step4: bool = False,
    aco_config_file: Path = None,
    rag_config_file: Path = None,
    limit: int = None,
    skip_step4: bool = False,
) -> dict:
    """
    执行完整Pipeline（Step1->Step2->Step3->Step4）
    
    Args:
        patents_csv: 专利数据CSV文件路径
        hist_end_year: 历史截止年份
        max_year: 最大年份
        run_id: 运行ID（None则自动生成；指定limit但未指定run_id时默认"test{limit}"）
        resume: 是否启用resume机制（默认True）
        force_step1: 强制重算Step1
        force_step2: 强制重算Step2
        force_step3: 强制重算Step3
        force_step4: 强制重新生成Step4报告
        aco_config_file: ACO配置文件路径
        rag_config_file: RAG配置文件路径
        limit: 只使用前N条专利（用于快速测试，默认不限制）
        skip_step4: 跳过Step4（需要DEEPSEEK_API_KEY，可选跳过）
        
    Returns:
        包含各步骤输出目录的字典
    """
    if limit is not None and run_id is None:
        run_id = f"test{limit}"
        logger.info(f"📋 测试模式: limit={limit}，自动设置 run_id={run_id}")

    logger.info("=" * 80)
    logger.info("🚀 Pipeline编排器：执行完整流程（Step1->Step2->Step3）")
    logger.info("=" * 80)
    
    results = {}
    
    # Step1: 构建网络
    logger.info("\n" + "=" * 80)
    logger.info("🔗 Step1: 构建并保存 HDKN + PDKN 网络")
    logger.info("=" * 80)
    
    try:
        run_dir = step1_build_networks(
            patents_csv=patents_csv or RAW_PATENT_FILE,
            hist_end_year=hist_end_year,
            max_year=max_year,
            run_id=run_id,
            force=force_step1,
            limit=limit,
        )
        results["run_dir"] = run_dir
        results["networks_dir"] = run_dir / "01_networks"
        run_id = run_dir.name  # 获取实际使用的run_id
        logger.success("✅ Step1 完成")
    except Exception as e:
        if resume and "已存在且metadata一致" in str(e):
            logger.info("ℹ️  Step1产物已存在，跳过")
            if run_id is None:
                raise ValueError("Step1跳过但run_id未指定，无法继续后续步骤")
            run_dir = get_run_dir(run_id)
            results["run_dir"] = run_dir
            results["networks_dir"] = run_dir / "01_networks"
        else:
            logger.exception("Step1 执行失败")
            raise
    
    # Step2: 回归分析
    logger.info("\n" + "=" * 80)
    logger.info("📈 Step2: 基于 HDKN 做回归分析")
    logger.info("=" * 80)
    
    try:
        regression_dir = step2_hdkn_regression(
            run_id=run_id,
            patents_csv=patents_csv or RAW_PATENT_FILE,
            force=force_step2,
            limit=limit,
        )
        results["regression_dir"] = regression_dir
        logger.success("✅ Step2 完成")
    except Exception as e:
        if resume and "已存在且metadata一致" in str(e):
            logger.info("ℹ️  Step2产物已存在，跳过")
            results["regression_dir"] = ensure_run_dirs(run_dir)["regression_dir"]
        else:
            logger.exception("Step2 执行失败")
            raise
    
    # Step3: ACO搜索
    logger.info("\n" + "=" * 80)
    logger.info("🐜 Step3: 基于 PDKN 做 ACO 搜索")
    logger.info("=" * 80)
    
    try:
        aco_dir = step3_pdkn_aco(
            run_id=run_id,
            regression_dir=results.get("regression_dir"),
            aco_config_file=aco_config_file or ACO_CONFIG_FILE,
            patents_csv=patents_csv,
            force=force_step3
        )
        results["aco_dir"] = aco_dir
        logger.success("✅ Step3 完成")
    except Exception as e:
        if resume and "已存在且metadata一致" in str(e):
            logger.info("ℹ️  Step3产物已存在，跳过")
            results["aco_dir"] = run_dir / "03_aco"
        else:
            logger.exception("Step3 执行失败")
            raise
    
    # Step4: RAG 报告生成
    if skip_step4:
        logger.info("\nℹ️  跳过 Step4（--skip-step4）")
        results["rag_dir"] = run_dir / "04_rag_reports"
    else:
        import os
        if not os.getenv("DEEPSEEK_API_KEY"):
            logger.warning(
                "\n⚠️  跳过 Step4: 未设置 DEEPSEEK_API_KEY 环境变量。"
                "设置后可单独运行: python scripts/step4_rag_report.py --run-id %s",
                run_id,
            )
            results["rag_dir"] = run_dir / "04_rag_reports"
        else:
            logger.info("\n" + "=" * 80)
            logger.info("📝 Step4: 基于 RAG 生成技术机会分析报告")
            logger.info("=" * 80)

            try:
                rag_dir = step4_rag_report(
                    run_id=run_id,
                    rag_config_file=rag_config_file,
                    patents_csv=patents_csv,
                    force=force_step4,
                )
                results["rag_dir"] = rag_dir
                logger.success("✅ Step4 完成")
            except Exception as e:
                if resume and "已存在且完整" in str(e):
                    logger.info("ℹ️  Step4 报告已存在，跳过")
                    results["rag_dir"] = run_dir / "04_rag_reports"
                else:
                    logger.exception("Step4 执行失败")
                    raise

    # 汇总输出
    logger.info("\n" + "=" * 80)
    logger.success("🎉 完整Pipeline执行完成！")
    logger.info("=" * 80)
    logger.info(f"📁 运行目录: {results['run_dir']}")
    logger.info(f"📁 网络目录: {results['networks_dir']}")
    logger.info(f"📁 回归目录: {results['regression_dir']}")
    logger.info(f"📁 ACO目录: {results['aco_dir']}")
    logger.info(f"📁 RAG报告目录: {results.get('rag_dir', 'N/A')}")
    
    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Pipeline编排器：按顺序执行Step1->Step2->Step3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_all.py
  python scripts/run_all.py --run-id my_run_001
  python scripts/run_all.py --resume --force-step2
  python scripts/run_all.py --hist-end-year 2020
  python scripts/run_all.py --limit 500                    # 快速测试，等价于原test500脚本
  python scripts/run_all.py --limit 500 --run-id my_test   # 自定义test run_id
        """
    )
    
    parser.add_argument(
        '--patents-csv',
        type=Path,
        default=RAW_PATENT_FILE,
        help=f'专利数据CSV文件路径（默认: {RAW_PATENT_FILE}）'
    )
    parser.add_argument(
        '--hist-end-year',
        type=int,
        default=None,
        help='历史截止年份（默认: 从config读取）'
    )
    parser.add_argument(
        '--max-year',
        type=int,
        default=None,
        help='最大年份（默认: 自动从数据计算）'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='运行ID（默认: 自动生成）'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='启用resume机制，跳过已完成的步骤（默认: True）'
    )
    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help='禁用resume机制，强制重算所有步骤'
    )
    parser.add_argument(
        '--force-step1',
        action='store_true',
        help='强制重算Step1'
    )
    parser.add_argument(
        '--force-step2',
        action='store_true',
        help='强制重算Step2'
    )
    parser.add_argument(
        '--force-step3',
        action='store_true',
        help='强制重算Step3'
    )
    parser.add_argument(
        '--force-step4',
        action='store_true',
        help='强制重新生成Step4报告'
    )
    parser.add_argument(
        '--skip-step4',
        action='store_true',
        help='跳过Step4（RAG报告生成，需要DEEPSEEK_API_KEY）'
    )
    parser.add_argument(
        '--aco-config',
        type=Path,
        default=None,
        help='ACO配置文件路径（默认: configs/aco_config.yaml）'
    )
    parser.add_argument(
        '--rag-config',
        type=Path,
        default=None,
        help='RAG配置文件路径（默认: configs/rag_config.yaml）'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='只使用前N条专利（用于快速测试，默认不限制；指定时若未设--run-id则自动用"testN"）'
    )
    
    args = parser.parse_args()
    
    # 从config读取hist_end_year（如果未指定）
    if args.hist_end_year is None:
        from src.patent_opportunity_analysis import config as _config
        args.hist_end_year = _config.HIST_END_YEAR
    
    try:
        results = run_all_pipeline(
            patents_csv=args.patents_csv,
            hist_end_year=args.hist_end_year,
            max_year=args.max_year,
            run_id=args.run_id,
            resume=args.resume,
            force_step1=args.force_step1,
            force_step2=args.force_step2,
            force_step3=args.force_step3,
            force_step4=args.force_step4,
            aco_config_file=args.aco_config,
            rag_config_file=args.rag_config,
            limit=args.limit,
            skip_step4=args.skip_step4,
        )
        logger.success(f"\n🎉 Pipeline执行成功！运行目录: {results['run_dir']}")
        return 0
    except Exception as e:
        logger.exception("Pipeline执行失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
