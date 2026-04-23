#!/usr/bin/env python3
"""
Step1: 构建并保存 HDKN + PDKN 网络

功能：
1. 加载专利数据
2. 构建HDKN（历史网络，ref_year=hist_end_year）
3. 构建PDKN（完整网络，ref_year=max_year）
4. 验证不变量
5. 保存网络和metadata

运行方式：
    python scripts/step1_build_networks.py [--patents-csv PATH] [--hist-end-year YEAR] [--max-year YEAR] [--run-id ID] [--force]
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from scripts.common import init_script
init_script()

from loguru import logger
import yaml
import pandas as pd
from src.patent_opportunity_analysis import config as _config
from src.patent_opportunity_analysis import pipeline as _pipeline
from src.patent_opportunity_analysis import dkn_builder as _dkn_builder
from src.patent_opportunity_analysis.utils.paths import RAW_PATENT_FILE, CONFIG_DIR
from src.patent_opportunity_analysis.utils.network_io import (
    save_dkn, load_dkn, create_networks_metadata, save_metadata, load_metadata,
    compute_file_hash, compute_data_hash, validate_metadata_consistency
)
from src.patent_opportunity_analysis.utils.run_utils import (
    generate_run_id, get_run_dir, get_step_dir, ensure_run_dirs
)

HIST_END_YEAR = _config.HIST_END_YEAR
load_patents_from_csv = _pipeline.load_patents_from_csv
build_dkns = _dkn_builder.build_dkns


def load_config_hash(config_file: Path) -> str:
    """加载配置文件并计算哈希"""
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        return compute_data_hash(config_data)
    return ""


def step1_build_networks(
    patents_csv: Path,
    hist_end_year: int,
    max_year: int = None,
    run_id: str = None,
    force: bool = False,
    config_file: Path = None,
    min_app_year: int = None,
    limit: int = None,
) -> Path:
    """
    执行Step1: 构建并保存HDKN+PDKN
    
    Args:
        patents_csv: 专利数据CSV文件路径
        hist_end_year: 历史截止年份
        max_year: 最大年份（None则自动从数据计算）
        run_id: 运行ID（None则自动生成）
        force: 是否强制重建（即使产物已存在）
        config_file: 配置文件路径（可选）
        limit: 只使用前N条专利（用于快速测试，默认不限制）
        
    Returns:
        运行目录路径
    """
    logger.info("=" * 80)
    logger.info("🔗 Step1: 构建并保存 HDKN + PDKN 网络")
    logger.info("=" * 80)
    
    # 1. 计算输入数据哈希和行数
    if not patents_csv.exists():
        raise FileNotFoundError(f"专利数据文件不存在: {patents_csv}")
    
    input_data_hash = compute_file_hash(patents_csv)
    # 计算输入数据行数（用于测试模式校验）
    input_data_rows = len(pd.read_csv(patents_csv))
    logger.info(f"📂 输入数据: {patents_csv}")
    logger.info(f"📊 数据哈希: {input_data_hash}")
    logger.info(f"📊 数据行数: {input_data_rows}")
    
    # 2. 计算配置哈希（可选，用于metadata一致性校验）
    # 注意：Step1构建网络不需要ACO配置，这里仅用于metadata记录
    config_hash = ""
    if config_file is not None and config_file.exists():
        config_hash = load_config_hash(config_file)
        logger.info(f"⚙️  配置哈希: {config_hash}")
    else:
        logger.debug("⚙️  未指定配置文件，跳过配置哈希计算")
    
    # 3. 生成或使用run_id
    if run_id is None:
        run_id = generate_run_id(
            data_hash=input_data_hash,
            hist_end_year=hist_end_year,
            max_year=max_year
        )
    logger.info(f"🆔 运行ID: {run_id}")
    
    # 4. 创建运行目录结构
    run_dir = get_run_dir(run_id)
    dirs = ensure_run_dirs(run_dir)
    networks_dir = dirs["networks_dir"]
    
    # 5. 检查是否已存在产物（resume检查）
    hdkn_path = networks_dir / "hdkn.pkl.gz"
    pdkn_path = networks_dir / "pdkn.pkl.gz"
    meta_path = networks_dir / "networks_meta.json"
    
    if not force and hdkn_path.exists() and pdkn_path.exists() and meta_path.exists():
        logger.info("📋 检测到已存在的网络产物，检查metadata一致性...")
        
        try:
            existing_meta = load_metadata(meta_path)
            
            # 验证一致性
            expected_meta = {
                "hist_end_year": hist_end_year,
                "max_year": max_year,
                "input_data_hash": input_data_hash,
                "config_hash": config_hash
            }
            
            is_consistent, error_msg = validate_metadata_consistency(
                existing_meta, expected_meta
            )
            
            if is_consistent:
                logger.success(f"✅ 产物已存在且metadata一致，跳过构建（使用 --force 强制重建）")
                logger.info(f"📁 运行目录: {run_dir}")
                return run_dir
            else:
                logger.warning(f"⚠️  Metadata不一致: {error_msg}")
                logger.warning("   使用 --force 强制重建")
                if not force:
                    raise ValueError(f"Metadata不一致，请使用 --force 强制重建: {error_msg}")
        except Exception as e:
            logger.warning(f"⚠️  读取现有metadata失败: {e}")
            if not force:
                logger.warning("   使用 --force 强制重建")
                raise
    
    # 6. 加载专利数据
    logger.info("\n📂 加载专利数据...")
    patents = load_patents_from_csv(
        patents_csv, limit=limit, smart_select=False, min_app_year=min_app_year
    )
    logger.success(f"✅ 成功加载 {len(patents)} 条专利")
    
    # 7. 计算max_year（如果未指定）
    if max_year is None:
        max_year = max(p.app_year for p in patents) if patents else hist_end_year
        logger.info(f"📅 自动计算max_year: {max_year}")
    else:
        logger.info(f"📅 使用指定的max_year: {max_year}")
    
    logger.info(f"📅 HIST_END_YEAR: {hist_end_year}")
    
    # 8. 构建HDKN和PDKN
    logger.info("\n🔗 构建DKN网络...")
    HDKN, PDKN = build_dkns(patents, hist_end_year)
    
    # 验证不变量
    HDKN.assert_kind("HDKN")
    HDKN.assert_invariants()
    PDKN.assert_kind("PDKN")
    PDKN.assert_invariants()
    
    # 验证ref_year
    if HDKN.ref_year != hist_end_year:
        raise ValueError(f"HDKN ref_year ({HDKN.ref_year}) != hist_end_year ({hist_end_year})")
    if PDKN.ref_year != max_year:
        raise ValueError(f"PDKN ref_year ({PDKN.ref_year}) != max_year ({max_year})")
    
    logger.success(f"✅ HDKN: {HDKN.number_of_nodes()} 节点, {HDKN.number_of_edges()} 边, ref_year={HDKN.ref_year}")
    logger.success(f"✅ PDKN: {PDKN.number_of_nodes()} 节点, {PDKN.number_of_edges()} 边, ref_year={PDKN.ref_year}")
    
    # 9. 保存网络
    logger.info("\n💾 保存网络...")
    save_dkn(HDKN, hdkn_path)
    save_dkn(PDKN, pdkn_path)
    
    # 10. 创建并保存metadata
    metadata = create_networks_metadata(
        hdkn=HDKN,
        pdkn=PDKN,
        input_data_hash=input_data_hash,
        config_hash=config_hash,
        hist_end_year=hist_end_year,
        max_year=max_year,
        run_id=run_id,
        patents_count=len(patents),
        input_data_rows=input_data_rows
    )
    save_metadata(metadata, meta_path)
    
    logger.success(f"\n✅ Step1 完成！")
    logger.info(f"📁 运行目录: {run_dir}")
    logger.info(f"📁 网络目录: {networks_dir}")
    logger.info(f"   - HDKN: {hdkn_path}")
    logger.info(f"   - PDKN: {pdkn_path}")
    logger.info(f"   - Metadata: {meta_path}")
    
    return run_dir


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Step1: 构建并保存 HDKN + PDKN 网络",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/step1_build_networks.py
  python scripts/step1_build_networks.py --patents-csv data/raw/patents.csv --hist-end-year 2020
  python scripts/step1_build_networks.py --run-id my_run_001 --force
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
        default=HIST_END_YEAR,
        help=f'历史截止年份（默认: {HIST_END_YEAR}）'
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
        '--force',
        action='store_true',
        help='强制重建，即使产物已存在'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='配置文件路径（可选，用于metadata记录，Step1构建网络不需要ACO配置）'
    )
    parser.add_argument(
        '--min-app-year',
        type=int,
        default=None,
        help='最小申请年份，过滤掉 app_year < 该值的专利（如 2012 表示仅保留 2012 年及以后）'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='只使用前N条专利（用于快速测试，默认不限制）'
    )
    
    args = parser.parse_args()
    
    try:
        run_dir = step1_build_networks(
            patents_csv=args.patents_csv,
            hist_end_year=args.hist_end_year,
            max_year=args.max_year,
            run_id=args.run_id,
            force=args.force,
            config_file=args.config,
            min_app_year=args.min_app_year,
            limit=args.limit,
        )
        logger.success(f"\n🎉 Step1 成功完成！运行目录: {run_dir}")
        return 0
    except Exception as e:
        logger.exception("Step1 执行失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
