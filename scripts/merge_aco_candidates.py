#!/usr/bin/env python3
"""
合并所有 ACO 搜索结果，按 Z 分数排序，以 80% 节点重叠上限进行多样性过滤，
选出 Top-30 子网并生成富化 JSON，默认写入当前 run 的 03_merged_rag，供 RAG 使用。

运行方式：
    python scripts/merge_aco_candidates.py --run-id <run_id> [--top-n 30] [--overlap 0.8]
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from scripts.common import init_script
init_script()

from loguru import logger

from src.patent_opportunity_analysis.utils.paths import (
    RAW_PATENT_FILE,
    RAG_DATA_DIR,
    get_run_merged_rag_candidates_json,
    get_run_merged_rag_dir,
    get_run_merged_rag_enriched_json,
    get_run_merged_rag_summary_json,
)
from src.patent_opportunity_analysis.utils.network_io import (
    load_dkn, load_metadata,
)
from src.patent_opportunity_analysis.utils.run_utils import get_run_dir, ensure_run_dirs
from src.patent_opportunity_analysis.aco_to_rag import (
    enrich_opportunities, build_patent_lookup, build_stem_to_originals,
)
from src.patent_opportunity_analysis.pipeline import load_patents_from_csv
from src.patent_opportunity_analysis.hdkn_feature_cache import build_hdkn_subnetwork_feature_cache
from src.patent_opportunity_analysis.feature_registry import FEATURE_REGISTRY
from src.patent_opportunity_analysis import aco_search as _aco_search


def _resolve_feature_names(objective_coefficients: Dict[str, float]) -> List[str]:
    """解析富化所需特征；当目标系数为空时回退到默认特征注册表。"""
    return list(objective_coefficients.keys()) or list(FEATURE_REGISTRY.keys())


def load_all_candidates(run_dir: Path) -> List[Dict[str, Any]]:
    """从 run_dir 下所有 03_aco* 目录加载 aco_candidates.json 并合并。"""
    all_candidates: List[Dict[str, Any]] = []
    aco_dirs = sorted(run_dir.glob("03_aco*"))

    for aco_dir in aco_dirs:
        cand_file = aco_dir / "aco_candidates.json"
        if not cand_file.exists():
            logger.debug(f"跳过（无 candidates）: {aco_dir.name}")
            continue
        with open(cand_file, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        for c in candidates:
            c["_source_dir"] = aco_dir.name
        all_candidates.extend(candidates)
        logger.info(f"  {aco_dir.name}: {len(candidates)} 条")

    logger.info(f"合计加载 {len(all_candidates)} 条候选子网")
    return all_candidates


def node_overlap_ratio(set_a: Set[str], set_b: Set[str]) -> float:
    """计算两个节点集的 Jaccard 风格重叠率：交集 / min(|A|, |B|)。"""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def diversity_filter(
    candidates: List[Dict[str, Any]],
    top_n: int = 30,
    max_overlap: float = 0.8,
) -> List[Dict[str, Any]]:
    """按分数降序贪心选取，跳过与已选子网节点重叠超过阈值的候选。"""
    sorted_cands = sorted(candidates, key=lambda c: c["score"], reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_node_sets: List[Set[str]] = []

    for c in sorted_cands:
        if len(selected) >= top_n:
            break
        nodes = set(c["nodes"])
        too_similar = any(
            node_overlap_ratio(nodes, existing) > max_overlap
            for existing in selected_node_sets
        )
        if too_similar:
            continue
        selected.append(c)
        selected_node_sets.append(nodes)

    logger.info(
        f"多样性过滤完成: {len(sorted_cands)} → {len(selected)} "
        f"(top_n={top_n}, max_overlap={max_overlap})"
    )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="合并 ACO candidates 并生成富化 JSON（默认写入当前 run 的 03_merged_rag）")
    parser.add_argument("--run-id", required=True, help="运行 ID")
    parser.add_argument("--top-n", type=int, default=30, help="选取前 N 名（默认 30）")
    parser.add_argument("--overlap", type=float, default=0.8, help="节点重叠上限（默认 0.8）")
    parser.add_argument("--patents-csv", type=Path, default=None)
    parser.add_argument(
        "--export-global-rag",
        action="store_true",
        help="额外导出一份到 data/processed/rag/（兼容历史流程，默认关闭）",
    )
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_id)
    if not run_dir.exists():
        logger.error(f"运行目录不存在: {run_dir}")
        return 1

    # --- 1. 加载并合并 ---
    logger.info("=" * 70)
    logger.info("📦 Step A: 加载所有 ACO candidates")
    logger.info("=" * 70)
    all_cands = load_all_candidates(run_dir)
    if not all_cands:
        logger.error("未找到任何候选子网")
        return 1

    # --- 2. 去重（完全相同节点集 + 相同分数视为重复，保留来源最优的） ---
    seen_keys: set = set()
    unique_cands: List[Dict[str, Any]] = []
    for c in sorted(all_cands, key=lambda x: x["score"], reverse=True):
        key = (tuple(sorted(c["nodes"])), round(c["score"], 8))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_cands.append(c)
    logger.info(f"去重: {len(all_cands)} → {len(unique_cands)} 条唯一候选")

    # --- 3. 多样性过滤 ---
    logger.info("=" * 70)
    logger.info("🔍 Step B: 多样性过滤 (节点重叠 ≤ {:.0%})".format(args.overlap))
    logger.info("=" * 70)
    top_candidates = diversity_filter(unique_cands, top_n=args.top_n, max_overlap=args.overlap)

    # --- 4. 富化 ---
    logger.info("=" * 70)
    logger.info("✨ Step C: 富化子网 (enrich_opportunities)")
    logger.info("=" * 70)

    networks_dir = run_dir / "01_networks"
    regression_dir = ensure_run_dirs(run_dir)["regression_dir"]

    pdkn_path = networks_dir / "pdkn.pkl.gz"
    hdkn_path = networks_dir / "hdkn.pkl.gz"
    networks_meta_path = networks_dir / "networks_meta.json"

    if not pdkn_path.exists() or not hdkn_path.exists():
        logger.error("PDKN/HDKN 文件缺失")
        return 1

    networks_meta = load_metadata(networks_meta_path)
    hist_end_year = networks_meta["hist_end_year"]

    logger.info("加载 PDKN ...")
    PDKN = load_dkn(pdkn_path)
    logger.info("加载 HDKN ...")
    HDKN = load_dkn(hdkn_path)

    objective_path = regression_dir / "objective_coefficients.json"
    with open(objective_path, "r", encoding="utf-8") as f:
        objective_coefficients = {str(k): float(v) for k, v in json.load(f).items()}

    regression_meta_path = regression_dir / "regression_meta.json"
    decay_factor = None
    if regression_meta_path.exists():
        regression_meta = load_metadata(regression_meta_path)
        decay_factor = regression_meta.get("decay_factor")

    feature_names = _resolve_feature_names(objective_coefficients)
    hdkn_cache = build_hdkn_subnetwork_feature_cache(
        HDKN,
        hist_end_year=hist_end_year,
        decay_factor=decay_factor,
        selected_features=feature_names,
    )
    linear_obj = _aco_search.LinearObjectiveFunction(objective_coefficients)

    _csv = args.patents_csv or RAW_PATENT_FILE
    patent_lookup = {}
    stem_to_originals = {}
    if Path(_csv).exists():
        patents_list = load_patents_from_csv(Path(_csv), smart_select=False)
        patent_lookup = build_patent_lookup(patents_list)
        stem_to_originals = build_stem_to_originals(patents_list)
        logger.info(f"已加载 {len(patent_lookup)} 条专利元数据")

    enriched = enrich_opportunities(
        top_candidates,
        PDKN=PDKN,
        HDKN=HDKN,
        hdkn_cache=hdkn_cache,
        feature_names=feature_names,
        linear_objective=linear_obj,
        pdkn_ref_year=PDKN.ref_year,
        patent_lookup=patent_lookup,
        stem_to_originals=stem_to_originals,
        domain_field="embodied intelligence",
    )

    # --- 5. 保存 ---
    logger.info("=" * 70)
    logger.info("💾 Step D: 保存结果")
    logger.info("=" * 70)

    out_dir = get_run_merged_rag_dir(args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_path = get_run_merged_rag_enriched_json(args.run_id)
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False, default=str)
    logger.success(f"富化 JSON 已保存: {enriched_path}")

    candidates_path = get_run_merged_rag_candidates_json(args.run_id)
    slim = []
    for i, c in enumerate(top_candidates, 1):
        slim.append({
            "rank": i,
            "nodes": c["nodes"],
            "score": c["score"],
            "size": c["size"],
            "source": c.get("_source_dir", "unknown"),
        })
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    logger.success(f"候选列表已保存: {candidates_path}")

    summary_path = get_run_merged_rag_summary_json(args.run_id)
    summary = {
        "run_id": args.run_id,
        "total_candidates_loaded": len(all_cands),
        "unique_candidates": len(unique_cands),
        "after_diversity_filter": len(top_candidates),
        "enriched_count": len(enriched),
        "top_n": args.top_n,
        "max_overlap": args.overlap,
        "score_range": {
            "best": round(top_candidates[0]["score"], 6) if top_candidates else None,
            "worst": round(top_candidates[-1]["score"], 6) if top_candidates else None,
        },
        "output_files": {
            "enriched_json": str(enriched_path),
            "candidates_json": str(candidates_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.success(f"合并摘要已保存: {summary_path}")

    if args.export_global_rag:
        RAG_DATA_DIR.mkdir(parents=True, exist_ok=True)
        global_enriched_path = RAG_DATA_DIR / enriched_path.name
        global_candidates_path = RAG_DATA_DIR / candidates_path.name
        global_summary_path = RAG_DATA_DIR / summary_path.name
        for src_path, dst_path in [
            (enriched_path, global_enriched_path),
            (candidates_path, global_candidates_path),
            (summary_path, global_summary_path),
        ]:
            dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.success(f"已额外导出到全局目录: {RAG_DATA_DIR}")

    logger.info("\n📊 Top-30 概览:")
    for i, c in enumerate(top_candidates, 1):
        logger.info(f"  #{i:2d}  Z={c['score']:+.4f}  来源={c.get('_source_dir','?'):20s}  "
                     f"节点={','.join(c['nodes'][:3])}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
