#!/usr/bin/env python3
"""
Step4: 基于 RAG 生成技术机会分析报告

功能：
1. 读取合并后的富化子网 JSON（默认 data/processed/rag/aco_merged_top30_enriched.json）
2. 调用 DeepSeek API 为每个技术机会生成结构化 Markdown 报告
3. 保存报告和 metadata

输入查找优先级：
  --input-json > data/processed/rag/aco_merged_top30_enriched.json > outputs/runs/<run_id>/03_aco/

运行方式：
    # 使用默认输入（data/processed/rag/），输出到 run 目录
    python scripts/step4_rag_report.py --run-id <run_id>

    # 直接指定输入 JSON
    python scripts/step4_rag_report.py --run-id <run_id> --input-json path/to/enriched.json

    # 仅生成指定 rank
    python scripts/step4_rag_report.py --run-id <run_id> --rank 1

    # 指定输出目录（不依赖 run-id）
    python scripts/step4_rag_report.py --output-dir outputs/rag_reports --input-json path/to/enriched.json

环境变量：
    DEEPSEEK_API_KEY  必填，DeepSeek API 密钥
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from scripts.common import init_script
init_script()

from loguru import logger

from src.patent_opportunity_analysis.rag_report_generator import (
    generate_opportunity_report,
    load_opportunities,
)
from src.patent_opportunity_analysis.utils.paths import (
    RAW_PATENT_FILE, CONFIG_DIR, RAG_ENRICHED_JSON,
)
from src.patent_opportunity_analysis.utils.network_io import (
    save_metadata, load_metadata,
)
from src.patent_opportunity_analysis.utils.run_utils import (
    get_run_dir, ensure_run_dirs,
)

RAG_CONFIG_FILE = CONFIG_DIR / "rag_config.yaml"


def load_rag_config(config_file: Path) -> dict:
    """加载 RAG 配置文件"""
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_input_json(
    input_json: Path = None,
    run_id: str = None,
) -> Path:
    """
    按优先级查找输入 JSON 文件。

    优先级：
      1. 显式 --input-json 参数
      2. data/processed/rag/aco_merged_top30_enriched.json（合并后的默认路径）
      3. outputs/runs/<run_id>/03_aco/aco_topk_enriched.json（单次运行产物）
    """
    if input_json is not None:
        if input_json.exists():
            return input_json
        raise FileNotFoundError(f"指定的输入文件不存在: {input_json}")

    if RAG_ENRICHED_JSON.exists():
        return RAG_ENRICHED_JSON

    if run_id:
        run_dir = get_run_dir(run_id)
        candidates = [
            run_dir / "03_aco" / "aco_topk_enriched.json",
            run_dir / "03_aco" / "aco_topk.json",
        ]
        for p in candidates:
            if p.exists():
                return p

    hint_paths = [str(RAG_ENRICHED_JSON)]
    if run_id:
        hint_paths.append(str(get_run_dir(run_id) / "03_aco" / "aco_topk_enriched.json"))
    raise FileNotFoundError(
        "未找到输入 JSON。已查找路径:\n  - "
        + "\n  - ".join(hint_paths)
        + "\n请使用 --input-json 显式指定，或先运行合并脚本 / Step3。"
    )


def _resolve_output_dir(
    output_dir: Path = None,
    run_id: str = None,
) -> Path:
    """确定输出目录。"""
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    if run_id:
        run_dir = get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        dirs = ensure_run_dirs(run_dir)
        rag_dir = dirs["rag_dir"]
        rag_dir.mkdir(parents=True, exist_ok=True)
        return rag_dir
    raise ValueError("必须指定 --run-id 或 --output-dir 之一来确定输出目录")


def step4_rag_report(
    run_id: str = None,
    input_json: Path = None,
    output_dir: Path = None,
    rag_config_file: Path = None,
    patents_csv: Path = None,
    rank: int = None,
    force: bool = False,
) -> Path:
    """
    执行 Step4: 基于 RAG 生成技术机会分析报告

    Args:
        run_id: 运行 ID（用于确定输出目录；与 output_dir 二选一）
        input_json: 输入 JSON 路径（None 则按优先级自动查找）
        output_dir: 输出目录（None 则基于 run_id 推断）
        rag_config_file: RAG 配置文件路径（None 则使用默认）
        patents_csv: 专利 CSV 路径（None 则使用默认）
        rank: 仅生成指定 rank 的报告（None 则生成全部）
        force: 是否强制重新生成

    Returns:
        RAG 报告目录路径
    """
    logger.info("=" * 80)
    logger.info("📝 Step4: 基于 RAG 生成技术机会分析报告")
    logger.info("=" * 80)

    # --- 1. 确定输入 / 输出路径 ---
    resolved_input = _resolve_input_json(input_json, run_id)
    rag_dir = _resolve_output_dir(output_dir, run_id)

    logger.info(f"📂 输入文件: {resolved_input}")
    logger.info(f"📁 输出目录: {rag_dir}")

    # --- 2. 加载 RAG 配置 ---
    rag_config_file = rag_config_file or RAG_CONFIG_FILE
    rag_config = load_rag_config(rag_config_file)

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError(
            "未配置 DEEPSEEK_API_KEY 环境变量。"
            "请执行: export DEEPSEEK_API_KEY=sk-xxx"
        )

    base_url = rag_config.get("deepseek_base_url", "https://api.deepseek.com")
    model_name = rag_config.get("model_name", "deepseek-chat")
    max_tokens = int(rag_config.get("max_tokens", 4096))
    temperature = float(rag_config.get("temperature", 0.1))
    top_n_edges = int(rag_config.get("top_n_edges", 5))
    top_k_patents = int(rag_config.get("top_k_patents", 10))
    evidence_max_chars = int(rag_config.get("evidence_max_chars", 60000))

    logger.info(f"⚙️  RAG 配置: model={model_name}, max_tokens={max_tokens}, "
                f"temperature={temperature}")
    logger.info(f"⚙️  报告参数: top_n_edges={top_n_edges}, top_k_patents={top_k_patents}")

    # --- 3. 加载机会数据 ---
    opportunities, data_format = load_opportunities(resolved_input)
    if not opportunities:
        logger.error("未找到任何技术机会数据")
        sys.exit(1)

    logger.info(f"📋 已加载 {len(opportunities)} 个技术机会 (格式: {data_format})")

    # --- 4. 筛选目标 ---
    if rank is not None:
        if data_format == "enriched":
            targets = [o for o in opportunities if o.get("opportunity_rank") == rank]
        else:
            targets = [o for o in opportunities if o.get("rank") == rank]
        if not targets:
            logger.error(f"未找到 rank={rank} 的机会")
            sys.exit(1)
    else:
        targets = opportunities

    # --- 5. 检查是否已存在（resume） ---
    rag_meta_path = rag_dir / "rag_meta.json"
    if not force and rag_meta_path.exists():
        logger.info("📋 检测到已存在的 RAG 报告，检查一致性...")
        try:
            existing_meta = load_metadata(rag_meta_path)
            existing_input = existing_meta.get("input_json")
            if existing_input == str(resolved_input):
                existing_count = existing_meta.get("reports_count", 0)
                if rank is None and existing_count >= len(targets):
                    logger.success(
                        "✅ 报告已存在且完整 (%d 份)，跳过（使用 --force 强制重新生成）",
                        existing_count,
                    )
                    return rag_dir
        except Exception as e:
            logger.warning(f"⚠️  读取现有 metadata 失败: {e}")

    # --- 2b. 读取验证相关配置 ---
    enable_validation = bool(rag_config.get("enable_citation_validation", True))
    cleanup_strategy = rag_config.get("citation_cleanup_strategy", "mark")
    append_validation = bool(rag_config.get("append_validation_report", True))

    # --- 6. 逐个生成报告 ---
    patents_csv = patents_csv or RAW_PATENT_FILE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_files: list = []
    validation_summary: list = []

    for i, target in enumerate(targets, start=1):
        if data_format == "enriched":
            opp_rank = target.get("opportunity_rank", i)
        else:
            opp_rank = target.get("rank", i)

        logger.info(
            f"正在为 rank={opp_rank} 的技术机会生成报告 ({i}/{len(targets)})..."
        )

        try:
            report, validation = generate_opportunity_report(
                target,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_n_edges=top_n_edges,
                top_k_patents=top_k_patents,
                evidence_max_chars=evidence_max_chars,
                patents_csv_path=patents_csv,
                data_format=data_format,
                enable_citation_validation=enable_validation,
                citation_cleanup_strategy=cleanup_strategy,
                append_validation_report=append_validation,
            )
        except Exception as e:
            logger.exception(f"rank={opp_rank} 报告生成失败: {e}")
            continue

        out_path = rag_dir / f"rank{opp_rank}_{timestamp}.md"
        out_path.write_text(report, encoding="utf-8")
        generated_files.append(str(out_path))
        logger.info(f"📄 报告已保存: {out_path}")

        if validation:
            validation_summary.append({
                "rank": opp_rank,
                "file": str(out_path),
                "faithfulness_rate": validation.get("faithfulness_rate"),
                "total_citations": validation.get("total_citations"),
                "hallucinated_count": len(validation.get("hallucinated_citations", [])),
                "validation_passed": validation.get("validation_passed"),
            })
            if not validation.get("validation_passed"):
                logger.warning(
                    f"⚠️ rank={opp_rank} 发现 {len(validation['hallucinated_citations'])} 个幻觉引用"
                )

    logger.info(f"全部完成，共生成 {len(generated_files)} 份报告")

    # --- 7. 保存 metadata ---
    rag_meta = {
        "step_name": "04_rag_reports",
        "created_at": datetime.now().isoformat(),
        "run_id": run_id or "standalone",
        "input_json": str(resolved_input),
        "data_format": data_format,
        "reports_count": len(generated_files),
        "config": {
            "model_name": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_n_edges": top_n_edges,
            "top_k_patents": top_k_patents,
            "evidence_max_chars": evidence_max_chars,
            "config_file": str(rag_config_file),
        },
        "output_files": generated_files,
        "validation_summary": validation_summary,
        "overall_faithfulness": (
            sum(v["faithfulness_rate"] for v in validation_summary) / len(validation_summary)
            if validation_summary else None
        ),
    }
    save_metadata(rag_meta, rag_meta_path)

    logger.success("\n✅ Step4 完成！")
    logger.info(f"📁 RAG 报告目录: {rag_dir}")
    for f in generated_files:
        logger.info(f"   - {f}")
    logger.info(f"   - Metadata: {rag_meta_path}")

    return rag_dir


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Step4: 基于 RAG 生成技术机会分析报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # 使用默认输入（data/processed/rag/），输出到 run 目录
  python scripts/step4_rag_report.py --run-id 20260331_195849_d009849ba_h2022

  # 仅生成 rank=1 的报告
  python scripts/step4_rag_report.py --run-id 20260331_195849_d009849ba_h2022 --rank 1

  # 直接指定输入和输出
  python scripts/step4_rag_report.py --input-json data/processed/rag/aco_merged_top30_enriched.json \\
      --output-dir outputs/rag_reports

  # 强制重新生成
  python scripts/step4_rag_report.py --run-id 20260331_195849_d009849ba_h2022 --force

默认输入: {RAG_ENRICHED_JSON}

环境变量:
  DEEPSEEK_API_KEY    DeepSeek API 密钥（必填）
        """
    )

    parser.add_argument(
        "--run-id", type=str, default=None,
        help="运行 ID（用于确定输出目录，与 --output-dir 二选一）",
    )
    parser.add_argument(
        "--input-json", type=Path, default=None,
        help=f"输入 JSON 路径（默认: {RAG_ENRICHED_JSON}）",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="输出目录（与 --run-id 二选一；指定时不依赖 run 目录结构）",
    )
    parser.add_argument(
        "--rag-config", type=Path, default=None,
        help=f"RAG 配置文件路径（默认: {RAG_CONFIG_FILE}）",
    )
    parser.add_argument(
        "--patents-csv", type=Path, default=None,
        help=f"专利 CSV 路径（默认: {RAW_PATENT_FILE}）",
    )
    parser.add_argument(
        "--rank", type=int, default=None,
        help="仅生成指定 rank 的报告（默认: 生成全部）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新生成，即使报告已存在",
    )

    args = parser.parse_args()

    if not args.run_id and not args.output_dir:
        parser.error("请指定 --run-id 或 --output-dir 之一")

    try:
        rag_dir = step4_rag_report(
            run_id=args.run_id,
            input_json=args.input_json,
            output_dir=args.output_dir,
            rag_config_file=args.rag_config,
            patents_csv=args.patents_csv,
            rank=args.rank,
            force=args.force,
        )
        logger.success(f"\n🎉 Step4 成功完成！RAG 报告目录: {rag_dir}")
        return 0
    except Exception as e:
        logger.exception("Step4 执行失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
