# src/patent_opportunity_analysis/aco_search.py

import json
import random
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from tqdm import tqdm
from loguru import logger

# 导入工具模块
from .utils import aco_utils as _aco_utils
load_aco_config = _aco_utils.load_aco_config
plot_aco_convergence = _aco_utils.plot_aco_convergence
visualize_subnetwork = _aco_utils.visualize_subnetwork
get_diverse_top_k = _aco_utils.get_diverse_top_k

# 导入配置
from . import config as _config

from .feature_registry import FEATURE_REGISTRY

from .hdkn_feature_cache import (
    build_hdkn_subnetwork_feature_cache,
    compute_subnetwork_features_for_aco,
)

# 默认使用配置文件，如果不存在则使用代码中的配置
_aco_config = load_aco_config()
ACO_NUM_ANTS = _aco_config.get('num_ants', _config.ACO_NUM_ANTS)
ACO_NUM_GENERATIONS = _aco_config.get('num_generations', _config.ACO_NUM_GENERATIONS)
ACO_TOP_K_PER_GEN = _aco_config.get('top_k_per_gen', _config.ACO_TOP_K_PER_GEN)
ACO_PHEROMONE_INIT = _aco_config.get('pheromone', {}).get('init', _config.ACO_PHEROMONE_INIT)
ACO_PHEROMONE_ALPHA = _aco_config.get('pheromone', {}).get('alpha', _config.ACO_PHEROMONE_ALPHA)
ACO_HEURISTIC_BETA = _aco_config.get('heuristic', {}).get('beta', _config.ACO_HEURISTIC_BETA)
ACO_RHO = _aco_config.get('pheromone', {}).get('rho',
          _aco_config.get('pheromone', {}).get('evaporation',  # 兼容旧配置键
                                              getattr(_config, 'ACO_RHO', 0.95)))
ACO_TAU_MIN = _aco_config.get('pheromone', {}).get('tau_min', _config.ACO_TAU_MIN)
ACO_TAU_MAX = _aco_config.get('pheromone', {}).get('tau_max', _config.ACO_TAU_MAX)
ACO_MAX_OVERLAP_RATIO = _aco_config.get('max_overlap_ratio', 0.5)
ACO_NOVELTY_WEIGHT = _aco_config.get('novelty', {}).get('weight', 2.0)
ACO_NOVEL_ANT_RATIO = _aco_config.get('novelty', {}).get('novel_ant_ratio', 0.3)
ACO_NEW_EDGE_BONUS = _aco_config.get('novelty', {}).get('new_edge_bonus', None)

HIST_END_YEAR = _config.HIST_END_YEAR


class LinearObjectiveFunction:
    """
    Ren & Zhao (2021) 线性目标 Z = Σ β_k X_k（仅含回归中显著的子网特征；原始尺度）。
    与 NB（GLM）系数及 compute_features_for_subnetwork 输出一致。
    """

    def __init__(self, coefficients: Dict[str, float]) -> None:
        self.coefficients: Dict[str, float] = dict(coefficients)

    def compute_z(self, subnetwork_features: Dict[str, Any]) -> float:
        total = 0.0
        for var, coef in self.coefficients.items():
            raw = subnetwork_features.get(var, 0)
            try:
                x = float(raw)
            except (TypeError, ValueError):
                x = 0.0
            if not np.isfinite(x):
                x = 0.0
            total += float(coef) * x
        return total


def _make_edge_key(a: str, b: str) -> Tuple[str, str]:
    """无向边规范键：避免 sorted() 开销"""
    return (a, b) if a <= b else (b, a)


def construct_solution(
    adj: Dict[str, List[str]],
    start_node: str,
    target_size: int,
    pheromone: Dict[Tuple[str, str], float],
    node_strength: Dict[str, float],
    alpha: float,
    beta: float,
    tau_min: float,
    new_edge_set: Optional[frozenset] = None,
    new_edge_bonus: float = 1.0,
) -> List[str]:
    """
    构建一个解（子网络）。

    遇到死胡同时回溯到 solution 中有未访问邻居的节点继续扩展，
    确保在连通分量足够大时能达到 target_size。

    Args:
        new_edge_set: 新边集合（规范化键）。若非 None，当 current→candidate
            经过新边时，权重乘以 (1 + new_edge_bonus)，引导蚂蚁走新边。
        new_edge_bonus: 新边权重乘数（仅在 new_edge_set 非 None 时生效）。
    """
    solution = [start_node]
    visited = {start_node}
    current = start_node

    for _ in range(target_size - 1):
        neighbors = adj.get(current)
        candidates = [n for n in neighbors if n not in visited] if neighbors else []

        if not candidates:
            found_backtrack = False
            for prev in reversed(solution):
                prev_nb = adj.get(prev)
                if prev_nb:
                    cands = [n for n in prev_nb if n not in visited]
                    if cands:
                        current = prev
                        candidates = cands
                        found_backtrack = True
                        break
            if not found_backtrack:
                break

        weights = []
        for node in candidates:
            ek = (current, node) if current <= node else (node, current)
            tau = pheromone.get(ek, tau_min)
            eta = node_strength.get(node, 1e-10)
            w = (tau ** alpha) * (eta ** beta)
            if new_edge_set is not None and ek in new_edge_set:
                w *= (1.0 + new_edge_bonus)
            weights.append(w)

        total = sum(weights)
        if total == 0.0:
            nxt = random.choice(candidates)
        else:
            r = random.random() * total
            cumsum = 0.0
            nxt = candidates[-1]
            for i, w in enumerate(weights):
                cumsum += w
                if cumsum >= r:
                    nxt = candidates[i]
                    break

        solution.append(nxt)
        visited.add(nxt)
        current = nxt

    return solution

def evaluate_solution(
    solution: List[str],
    pdkn_graph: nx.Graph,
    HDKN,
    linear_objective: LinearObjectiveFunction,
    hdkn_cache: Dict[str, Any],
    pdkn_ref_year: int,
    feature_names: List[str],
) -> float:
    """
    用 NB 显著项线性系数 Z=ΣβX 评估子网。

    使用 subgraph view（不复制），特征均为 O(|V|) 字典查找。
    """
    if len(solution) < 3:
        return float('-inf')

    subg = pdkn_graph.subgraph(solution)
    if subg.number_of_nodes() == 0:
        return 0.0

    feats = compute_subnetwork_features_for_aco(
        subg,
        HDKN,
        ref_year=pdkn_ref_year,
        hdkn_cache=hdkn_cache,
        selected_features=feature_names,
        current_patent_id=None,
    )
    return linear_objective.compute_z(feats)

def update_pheromone(
    pheromone: Dict[Tuple[str, str], float],
    solutions: List[Tuple[List[str], float]],
    rho: float,
    tau_min: float,
    tau_max: float,
    best_z_global: float = float('-inf'),
):
    """
    更新信息素，基于 rank-based 精英策略。

    使用排名倒数作为沉积量，与 Z 值的绝对尺度/正负无关，保证对任意目标函数值域通用。
    论文 Section 3.4: "the smaller the difference [between Z and best_z],
    the more pheromone increase these nodes and edges receive"

    Args:
        pheromone: 边 -> 信息素浓度
        solutions: (节点列表, Z 值) 按 Z 降序排列的 Top-K 解
        rho: 信息素保留率（论文中 ρ=0.95，即 5% 挥发）
        tau_min: MIN-MAX 下限
        tau_max: MIN-MAX 上限
        best_z_global: 历史全局最优 Z 值（用于计算精英奖励增量）
    """
    # 第一步：挥发（论文 ρ 为保留率，如 0.95 表示保留 95%）
    for edge in pheromone:
        pheromone[edge] *= rho

    # 第二步：rank-based 精英奖励
    # 按 Z 降序排列后，rank=1 的最优解沉积最多，rank 越大沉积越少
    if solutions:
        sorted_sols = sorted(solutions, key=lambda x: x[1], reverse=True)
        for rank, (solution, z_val) in enumerate(sorted_sols, start=1):
            # 方式一：与全局最优差距越小，沉积越多（论文语义）
            if np.isfinite(best_z_global) and best_z_global > float('-inf'):
                diff = abs(best_z_global - z_val)
                increment = 1.0 / (1.0 + diff)
            else:
                # 全局最优尚未建立时，退化为 rank 倒数
                increment = 1.0 / rank
            for i in range(len(solution) - 1):
                a, b = solution[i], solution[i + 1]
                ek = (a, b) if a <= b else (b, a)
                if ek not in pheromone:
                    pheromone[ek] = tau_min
                pheromone[ek] += increment

    # 第三步：MIN-MAX 裁剪
    for edge in pheromone:
        pheromone[edge] = max(tau_min, min(tau_max, pheromone[edge]))

def aco_search_opportunities(
    HDKN,
    PDKN,
    subnetwork_size: int,
    top_k: int,
    output_dir: Optional[Path] = None,
    objective_coefficients: Optional[Dict[str, float]] = None,
    override_coefficients: Optional[Dict[str, float]] = None,
    hist_end_year: Optional[int] = None,
    pdkn_ref_year: Optional[int] = None,
    decay_factor: Optional[float] = None,
) -> List[Dict]:
    """
    使用 ACO 搜索技术机会；子网得分 Z 为 NB 显著项线性组合（原始尺度），信息素按 Z 更新。

    节点选择仍用 τ^α·η^β，η 为节点 strength（工程近似，与原文一致：Z 仅用于信息素更新）。

    Args:
        objective_coefficients: Step2 objective_coefficients.json 解析结果（特征名 -> β）
        override_coefficients: 可选，调试时覆盖/增补系数（同名键覆盖 JSON）
        hist_end_year: HDKN 截止年（构建统计缓存，默认 HDKN.hist_end_year / config）
        pdkn_ref_year: PDKN 观测年，传入 compute_features 的 current_year（默认 PDKN.ref_year）
        decay_factor: HDKN 统计缓存用 α（应与 Step2 一致；默认 config）

    Returns:
        机会列表，每个机会包含节点和得分（Z）
    """
    # 断言：确保传入的网络类型正确
    if hasattr(HDKN, 'assert_kind'):
        HDKN.assert_kind("HDKN")
    if hasattr(PDKN, 'assert_kind'):
        PDKN.assert_kind("PDKN")

    if objective_coefficients is None:
        raise ValueError(
            "未提供 objective_coefficients（请先运行 Step2 生成 objective_coefficients.json，"
            "或由调用方传入系数字典）。"
        )

    merged_coefs: Dict[str, float] = {k: float(v) for k, v in objective_coefficients.items()}
    if override_coefficients:
        for k, v in override_coefficients.items():
            merged_coefs[str(k)] = float(v)

    linear_objective = LinearObjectiveFunction(merged_coefs)
    _hist = hist_end_year if hist_end_year is not None else (
        int(HDKN.hist_end_year) if hasattr(HDKN, "hist_end_year") else int(HIST_END_YEAR)
    )
    _ref = pdkn_ref_year if pdkn_ref_year is not None else (
        int(PDKN.ref_year) if hasattr(PDKN, "ref_year") else _hist
    )

    stat_feature_keys = sorted(merged_coefs.keys())
    if not stat_feature_keys:
        stat_feature_keys = sorted(FEATURE_REGISTRY.keys())
        logger.warning(
            "objective 系数为空，HDKN 缓存将按全量特征预计算；Z 恒为 0（请检查 Step2 显著项）"
        )
    logger.info(f"构建 HDKN 子网特征缓存（用于 Z），特征键: {stat_feature_keys}")
    hdkn_cache = build_hdkn_subnetwork_feature_cache(
        HDKN,
        hist_end_year=_hist,
        decay_factor=decay_factor,
        selected_features=stat_feature_keys,
        force_rebuild=False,
    )
    feature_names = stat_feature_keys

    # 获取底层的 graph 对象（如果 PDKN 是 DKNNetwork）
    pdkn_graph = PDKN.graph if hasattr(PDKN, 'graph') else PDKN
    
    if PDKN.number_of_nodes() == 0:
        logger.warning("PDKN为空，无法进行ACO搜索")
        return []
    
    num_nodes = PDKN.number_of_nodes()
    num_edges = PDKN.number_of_edges()

    # 仅在极小图时适度降参；中/大图全部使用用户配置值，避免隐式截断
    if num_nodes < 100:
        adjusted_ants = min(ACO_NUM_ANTS, max(num_nodes, 30))
        adjusted_generations = min(ACO_NUM_GENERATIONS, 50)
        logger.info(
            f"极小规模图（{num_nodes} 节点，{num_edges} 边），"
            f"自动调整: ants={adjusted_ants}, gens={adjusted_generations}"
        )
    else:
        adjusted_ants = ACO_NUM_ANTS
        adjusted_generations = ACO_NUM_GENERATIONS
        logger.info(
            f"图规模: {num_nodes} 节点，{num_edges} 边 — "
            f"使用配置参数: ants={adjusted_ants}, gens={adjusted_generations}"
        )
    
    logger.info(f"开始ACO搜索，参数: 蚂蚁数={adjusted_ants}, 代数={adjusted_generations}, 子网络大小={subnetwork_size}")
    logger.info(f"PDKN ref_year={_ref}（子网特征 current_year），HDKN hist_end_year={_hist}")

    # ── 预计算：邻接表 + 节点加权度 + 新颖度 ──
    hdkn_graph = hdkn_cache["hdkn_graph"]
    logger.info("预计算邻接表、节点 strength 与新颖度...")
    adj: Dict[str, List[str]] = {n: list(pdkn_graph.neighbors(n)) for n in pdkn_graph.nodes()}
    node_strength: Dict[str, float] = {}
    node_novelty_ratio: Dict[str, float] = {}
    for n in pdkn_graph.nodes():
        s = pdkn_graph.degree(n, weight='weight')
        node_strength[n] = max(float(s), 1e-10)
        nbs = adj[n]
        if nbs:
            new_cnt = sum(1 for nb in nbs if not hdkn_graph.has_edge(n, nb))
            node_novelty_ratio[n] = new_cnt / len(nbs)
        else:
            node_novelty_ratio[n] = 0.0

    # η = strength × (1 + γ × novelty_ratio)，让参与新边的节点获得竞争力
    _novelty_weight = ACO_NOVELTY_WEIGHT
    node_eta: Dict[str, float] = {}
    for n, s in node_strength.items():
        nr = node_novelty_ratio.get(n, 0.0)
        node_eta[n] = s * (1.0 + _novelty_weight * nr)

    # 预计算新边集合（PDKN 有而 HDKN 没有的边，规范化键）
    new_edge_set_raw: set = set()
    for u, v in pdkn_graph.edges():
        if not hdkn_graph.has_edge(u, v):
            ek = (u, v) if u <= v else (v, u)
            new_edge_set_raw.add(ek)
    new_edge_set: frozenset = frozenset(new_edge_set_raw)
    logger.info(f"新边集合: {len(new_edge_set)}/{pdkn_graph.number_of_edges()} "
                f"({len(new_edge_set)/pdkn_graph.number_of_edges()*100:.1f}%)")

    # 按新颖度排序，选出 top 候选用于"新颖蚂蚁"起点
    novel_candidates = sorted(
        node_novelty_ratio.items(), key=lambda x: x[1], reverse=True
    )
    novel_start_pool = [n for n, r in novel_candidates if r > 0.1][:max(1000, num_nodes // 10)]
    _novel_ant_count = int(adjusted_ants * ACO_NOVEL_ANT_RATIO)
    _regular_ant_count = adjusted_ants - _novel_ant_count
    _new_edge_bonus = ACO_NEW_EDGE_BONUS if ACO_NEW_EDGE_BONUS is not None else (_novelty_weight * 3)

    logger.info(
        f"新颖度统计: 平均={np.mean(list(node_novelty_ratio.values())):.3f}, "
        f"高新颖节点池={len(novel_start_pool)}, "
        f"蚂蚁分配: 常规={_regular_ant_count}, 新颖={_novel_ant_count}, "
        f"新边bonus={_new_edge_bonus:.1f}"
    )

    # 初始化信息素（论文 Section 4.4: "the initial pheromone concentration to 1"）
    pheromone: Dict[Tuple[str, str], float] = {}
    for u, v in pdkn_graph.edges():
        ek = (u, v) if u <= v else (v, u)
        pheromone[ek] = ACO_PHEROMONE_INIT

    all_pdkn_nodes = list(pdkn_graph.nodes())
    
    best_solutions: List[Tuple[List[str], float]] = []
    best_scores_per_gen: List[float] = []
    avg_scores_per_gen: List[float] = []
    gen_stats_list: List[Dict] = []
    best_z_global = float('-inf')

    early_stop_patience = max(50, adjusted_generations // 4)
    stagnation_count = 0

    _tau_min = ACO_TAU_MIN
    _alpha = ACO_PHEROMONE_ALPHA
    _beta = ACO_HEURISTIC_BETA

    pbar = tqdm(range(adjusted_generations), desc="ACO搜索")

    for generation in pbar:
        solutions = []

        # 常规蚂蚁：随机起点
        for _ in range(_regular_ant_count):
            start = random.choice(all_pdkn_nodes)
            solution = construct_solution(
                adj, start, subnetwork_size,
                pheromone, node_eta, _alpha, _beta, _tau_min,
            )
            if len(solution) >= subnetwork_size:
                score = evaluate_solution(
                    solution, pdkn_graph, HDKN,
                    linear_objective=linear_objective,
                    hdkn_cache=hdkn_cache,
                    pdkn_ref_year=_ref,
                    feature_names=feature_names,
                )
                solutions.append((solution, score))

        # 新颖蚂蚁：从高新颖度节点出发，走新边有 bonus
        if novel_start_pool:
            for _ in range(_novel_ant_count):
                start = random.choice(novel_start_pool)
                solution = construct_solution(
                    adj, start, subnetwork_size,
                    pheromone, node_eta, _alpha, _beta, _tau_min,
                    new_edge_set=new_edge_set,
                    new_edge_bonus=_new_edge_bonus,
                )
                if len(solution) >= subnetwork_size:
                    score = evaluate_solution(
                        solution, pdkn_graph, HDKN,
                        linear_objective=linear_objective,
                        hdkn_cache=hdkn_cache,
                        pdkn_ref_year=_ref,
                        feature_names=feature_names,
                    )
                    solutions.append((solution, score))

        # 选择本代最佳解
        if solutions:
            solutions.sort(key=lambda x: x[1], reverse=True)
            best_solutions.extend(solutions[:ACO_TOP_K_PER_GEN])
            best_solutions.sort(key=lambda x: x[1], reverse=True)
            best_solutions = best_solutions[:top_k * 8]  # 保留更多候选（供 Jaccard 多样性过滤）
            
            # 记录得分
            scores = [s[1] for s in solutions]
            best_score = scores[0]
            if best_score > best_z_global:
                best_z_global = best_score
                stagnation_count = 0
            else:
                stagnation_count += 1
            avg_score = float(np.mean(scores))
            best_scores_per_gen.append(best_score)
            avg_scores_per_gen.append(avg_score)
            gen_stats_list.append({
                'generation': generation + 1,
                'best': best_score,
                'avg': avg_score,
                'min': float(np.min(scores)),
                'std': float(np.std(scores)) if len(scores) > 1 else 0.0,
                'median': float(np.median(scores)),
                'num_solutions': len(solutions),
            })
            
            # 更新进度条
            pbar.set_postfix({
                'best': f'{best_score:.4f}',
                'global': f'{best_z_global:.4f}',
                'stag': stagnation_count,
            })
        else:
            best_scores_per_gen.append(0.0)
            avg_scores_per_gen.append(0.0)
            gen_stats_list.append({
                'generation': generation + 1,
                'best': 0.0, 'avg': 0.0, 'min': 0.0,
                'std': 0.0, 'median': 0.0, 'num_solutions': 0,
            })
        
        # 更新信息素（ρ 为保留率，论文 Section 3.4）
        update_pheromone(
            pheromone, solutions[:ACO_TOP_K_PER_GEN] if solutions else [],
            ACO_RHO, ACO_TAU_MIN, ACO_TAU_MAX,
            best_z_global=best_z_global,
        )

        # 早停：连续多代全局最优 Z 无改善
        if stagnation_count >= early_stop_patience:
            logger.info(
                f"连续 {stagnation_count} 代无改善，触发早停 "
                f"(generation={generation+1}, best_z_global={best_z_global:.6f})"
            )
            break
    
    pbar.close()
    
    # 保存收敛曲线到reports目录
    if output_dir and best_scores_per_gen:
        reports_dir = output_dir.parent / "reports" if output_dir.name == "models" else output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        convergence_path = reports_dir / "aco_progress.png"
        plot_aco_convergence(best_scores_per_gen, avg_scores_per_gen, convergence_path)
        
        # 保存每代统计 CSV（供后续分析）
        if gen_stats_list:
            gen_df = pd.DataFrame(gen_stats_list)
            gen_csv_path = reports_dir / "aco_generations.csv"
            gen_df.to_csv(gen_csv_path, index=False)
            logger.info(f"💾 每代统计已保存: {gen_csv_path}")
    
    # 去重
    seen = set()
    deduped = []
    for solution, score in best_solutions:
        solution_tuple = tuple(sorted(solution))
        if solution_tuple not in seen:
            seen.add(solution_tuple)
            deduped.append({
                'nodes': solution,
                'score': score,
                'size': len(solution)
            })

    # 保存候选池（供后续按不同 max_overlap_ratio 重新筛选，无需重跑 ACO）
    if output_dir:
        candidates_path = output_dir / "aco_candidates.json"
        with open(candidates_path, 'w', encoding='utf-8') as f:
            json.dump(deduped, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 候选池已保存: {candidates_path} ({len(deduped)} 个解)")

    # Jaccard 相似度过滤：筛选出多样性强的 Top-K
    opportunities = get_diverse_top_k(
        deduped,
        top_k=top_k,
        max_overlap_ratio=ACO_MAX_OVERLAP_RATIO,
    )
    
    # 可视化前3个机会的子网络（保存到reports目录）
    if output_dir and opportunities:
        reports_dir = output_dir.parent / "reports" if output_dir.name == "models" else output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        for i, opp in enumerate(opportunities[:min(3, len(opportunities))], 1):
            viz_path = reports_dir / f"opportunity_{i}.png"
            visualize_subnetwork(
                pdkn_graph, opp['nodes'],
                viz_path,
                title=f"技术机会 #{i} (得分: {opp['score']:.4f})"
            )
    
    logger.success(f"ACO搜索完成，找到 {len(opportunities)} 个技术机会")
    return opportunities
